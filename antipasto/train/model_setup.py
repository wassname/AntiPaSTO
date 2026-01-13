"""Model initialization and setup utilities for AntiPaSTO training."""


import torch
from typing import TYPE_CHECKING
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

if TYPE_CHECKING:
    from antipasto.peft_utils.layer_selection import SubspaceCache

from peft import PeftModel

from antipasto.config import TrainingConfig
from antipasto.peft_utils.antipasto_adapter import AntiPaSTOConfig

DEFAULT_CHAT_TEMPLATE = """
{% for message in messages %}
    {% set content = message['content'] %}

    {% if message['role'] == 'user' %}
        {{ '[INST] ' + content | trim + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' ' + content | trim + ' '  }}
    {% endif %}
{% endfor %}
"""

def load_model(model_id, quantization_type="none"):
    """Load base model with optional quantization.
    
    For VLMs (e.g., Gemma 3 4B+), loads the text-only CausalLM class directly
    to avoid VLM wrapper and get standard layer paths.
    """
    model_kwargs = {}
    if quantization_type == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs['quantization_config'] = quantization_config
    elif quantization_type == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs['quantization_config'] = quantization_config

    # Check if this is a VLM config (has text_config nested)
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, 'text_config'):
        logger.info("Detected VLM config, loading text-only model with text_config")
        config=config.text_config

    logger.info(f"Loading model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="cuda:0",
        config=config,
        **model_kwargs
    )

    if 'quantization_config' in model_kwargs:
        base_model.enable_input_require_grads()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    if not tokenizer.chat_template:
        tokenizer.chat_template=DEFAULT_CHAT_TEMPLATE

    return base_model, tokenizer


def setup_adapter(base_model, config: TrainingConfig, target_modules: str, precomputed_indices=None, svd_bases=None):
    """Setup AntiPaSTO adapter on base model.
    
    Args:
        base_model: Base model to add adapter to
        config: Training configuration
        target_modules: PEFT target_modules regex (from LayerSelection)
        precomputed_indices: Optional dict of {layer_name: indices_tensor} for dim selection
        svd_bases: Optional dict of {layer_name.U/V/S: tensor} for loading saved SVD bases
    """
    logger.debug(f"Target modules regex: {target_modules}")

    adapter_config = AntiPaSTOConfig(
        r=config.r,
        rotate_u=config.rot_u,
        rotate_v=config.rot_v,
        max_rotation_angle=config.max_rotation_angle,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        precomputed_indices=precomputed_indices,
        svd_bases=svd_bases,
    )

    # Create PeftModel - AntiPaSTO handles bidirectional steering internally via alpha coefficient
    model = PeftModel(base_model, adapter_config, adapter_name=config.dataset_name)
    
    # Clear precomputed_indices and svd_bases from config after adapter creation (only needed for init)
    if adapter_config.precomputed_indices is not None:
        adapter_config.precomputed_indices = None
    if adapter_config.svd_bases is not None:
        adapter_config.svd_bases = None
    
    logger.info(
        f"Adapter configured: rank={config.r}, target_modules={target_modules}"
    )
    
    # Verify regexp matched expected layers
    adapter_layers = [n for n, m in model.named_modules() if hasattr(m, 'antipasto_u')]
    logger.info(f"Adapter layers matched: {len(adapter_layers)}")

    return model


def compute_loss_subspace_basis(
    config: TrainingConfig,
    subspaces: "SubspaceCache" = None,
    model: torch.nn.Module = None,
    tokenizer = None,
    dataset_pt = None,
) -> torch.Tensor:
    """Select or compute the basis for loss subspace projection.
    
    Returns a frozen, detached basis L_subspace: [d_model, k] that projects
    activations to the loss subspace. The choice of subspace is controlled
    by config.loss_subspace.
    
    Args:
        config: Training config with loss_subspace setting
        subspaces: SubspaceCache from layer selection (required)
        model: Not used (kept for API compat)
        tokenizer: Not used (kept for API compat)
        dataset_pt: Not used (kept for API compat)
        
    Returns:
        L_subspace: [d_model, top_k] tensor
    
    Note: Many loss_subspace options were removed in Jan 2026 cleanup.
    Only taskdiff_x_suppressed_x_write, write, taskdiff are supported.
    See git history for steer*, null, notlogits, weight_svd implementations.
    """
    top_k = config.loss_subspace_rank  # None means auto by energy
    
    # Get from cache - SubspaceCache now stores Subspace objects
    L_cached_sub = subspaces.get(config.loss_subspace)
    if L_cached_sub is None:
        raise ValueError(
            f"Subspace '{config.loss_subspace}' not found in cache. "
            f"Available: {list(subspaces._subspaces.keys())}. "
            f"Valid options: taskdiff_x_suppressed_x_write, write, taskdiff"
        )
    
    L_cached = L_cached_sub.V
    S_cached = L_cached_sub.S  # May be None
    
    # Auto rank selection via energy thresholding (MSRS-style)
    if top_k is None:
        if S_cached is not None and len(S_cached) > 0:
            # Find smallest k such that cumulative energy >= threshold
            energy_frac = config.loss_subspace_energy_frac
            S_float = S_cached.float()
            cumsum = torch.cumsum(S_float, dim=0)
            total = S_float.sum() + 1e-8
            frac_cumsum = cumsum / total
            # Find first index where cumsum >= threshold
            above_threshold = frac_cumsum >= energy_frac
            if above_threshold.any():
                top_k = above_threshold.nonzero(as_tuple=True)[0][0].item() + 1
            else:
                top_k = len(S_cached)  # Use all if threshold never reached
            logger.info(f"Auto loss_subspace_rank: k={top_k} for {frac_cumsum[top_k-1]:.1%} energy (target={energy_frac:.0%})")
        else:
            # Fallback: no S available, use adapter rank
            top_k = config.r
            logger.warning(f"No singular values for '{config.loss_subspace}', falling back to loss_subspace_rank={top_k}")
    
    logger.info(f"Using precomputed L_{config.loss_subspace} from gradient selection")
    L_subspace = L_cached[:, :top_k].detach().requires_grad_(False)
    
    logger.info(f"Loss subspace ({config.loss_subspace}): shape={L_subspace.shape}")
    
    return L_subspace

