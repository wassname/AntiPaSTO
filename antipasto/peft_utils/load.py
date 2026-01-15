
from peft import PeftModel
from pathlib import Path
import safetensors.torch
import torch
import json
from loguru import logger
from typing import Optional, Tuple, Union

from antipasto.peft_utils.layer_selection import LayerSelection


def resolve_adapter_path(adapter_folder: Union[str, Path]) -> Path:
    """Resolve adapter path, downloading from HuggingFace Hub if needed.
    
    Args:
        adapter_folder: Local path or HuggingFace repo ID (e.g., 'wassname/antipasto-gemma-3-1b-honesty')
        
    Returns:
        Local Path to adapter folder
    """
    adapter_folder = str(adapter_folder)
    
    # Check if it's a local path
    local_path = Path(adapter_folder)
    if local_path.exists():
        return local_path
    
    # Try as HuggingFace repo ID
    if "/" in adapter_folder and not adapter_folder.startswith("/"):
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading adapter from HuggingFace: {adapter_folder}")
        local_dir = snapshot_download(
            repo_id=adapter_folder,
            allow_patterns=["*.json", "*.safetensors", "*.pt"],
        )
        return Path(local_dir)
    
    raise FileNotFoundError(f"Adapter not found locally or on HuggingFace: {adapter_folder}")


def add_adapter_name_to_sd(sd, adapter_name="default", prefix="antipasto_"):
    new_sd = {}
    for k, v in sd.items():
        if prefix in k:
            new_k = f"{k}.{adapter_name}"
        new_sd[new_k] = v
    return new_sd


def remove_adapter_name(key, adapter_name="default"):
    if "." not in key:
        return key
    if key.endswith(f".{adapter_name}"):
        return key.removesuffix(f".{adapter_name}")
    return key  # .replace(f".{adapter_name}.", ".")




def save_adapter(
    model: PeftModel, 
    save_folder: Path, 
    adapter_name: str,
    model_id: str = None,
    layer_selection: Optional[LayerSelection] = None,
    precomputed_indices: Optional[dict] = None,
    save_svd_bases: bool = True,
):
    """Save adapter weights, config, and metadata needed for reloading.
    
    Args:
        model: PeftModel with trained adapter
        save_folder: Directory to save to
        adapter_name: Name of the adapter in PeftModel
        model_id: HuggingFace model ID (stored in adapter_config.json for reload)
        layer_selection: Optional LayerSelection for loss computation (saves 0_layer_selection.json)
        precomputed_indices: Optional {layer_name: indices} for dimension selection (saves 0_precomputed_indices.pt)
        save_svd_bases: If True, save U, V, S buffers for exact reload (avoids SVD recomputation issues)
    """
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING

    save_folder.mkdir(parents=True, exist_ok=True)

    config = model.peft_config[adapter_name]
    
    # Set base_model_name_or_path for standard PEFT compatibility
    if model_id is not None:
        config.base_model_name_or_path = model_id
    
    state_dict = model.state_dict()

    prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
    to_return = {k: state_dict[k] for k in state_dict if prefix in k}

    to_return = {remove_adapter_name(k, adapter_name): v for k, v in to_return.items()}
    
    # Optionally include SVD bases (U, V, S) for exact reload
    # These are BufferDict entries, not parameters, so we need to extract them separately
    if save_svd_bases:
        svd_buffers = {}
        for name, module in model.named_modules():
            if hasattr(module, 'antipasto_u') and adapter_name in module.antipasto_u:
                # Extract SVD bases for this adapter
                layer_key = name.replace('.', '_')  # Safe key for safetensors
                svd_buffers[f"svd_bases.{layer_key}.u"] = module.antipasto_u[adapter_name].clone()
                svd_buffers[f"svd_bases.{layer_key}.v"] = module.antipasto_v[adapter_name].clone()
                svd_buffers[f"svd_bases.{layer_key}.s"] = module.antipasto_s[adapter_name].clone()
        if svd_buffers:
            safetensors.torch.save_file(svd_buffers, save_folder / "0_svd_bases.safetensors")
            logger.info(f"Saved SVD bases for {len(svd_buffers) // 3} layers")

    safetensors.torch.save_file(to_return, save_folder / "adapter_model.safetensors")
    config.save_pretrained(save_folder)
    
    # Save layer selection metadata (enables clean reloading)
    if layer_selection is not None:
        with open(save_folder / "0_layer_selection.json", "w") as f:
            json.dump(layer_selection.to_dict(), f, indent=2)
    
    # Save precomputed indices if dimension selection was used
    if precomputed_indices is not None:
        torch.save(precomputed_indices, save_folder / "0_precomputed_indices.pt")

    logger.info(f"Saved adapter to {save_folder}")


def load_adapter(
    adapter_folder: Union[str, Path],
    base_model=None,
    model_id: str = None,
    quantization_type: str = None,
    adapter_name: str = "default",
) -> Tuple[PeftModel, Optional[LayerSelection]]:
    """Load a saved AntiPaSTO adapter with all metadata.
    
    Either provide base_model directly, OR model_id + quantization_type to load it.
    Model ID can also be read from adapter_config.json (base_model_name_or_path).
    
    Supports loading from:
    - Local path: load_adapter("outputs/adapters/my_run")
    - HuggingFace Hub: load_adapter("wassname/antipasto-gemma-3-1b-honesty")
    
    Args:
        adapter_folder: Path to saved adapter or HuggingFace repo ID
        base_model: Pre-loaded base model (optional, provide this OR model_id)
        model_id: HuggingFace model ID to load (optional, auto-detected from adapter_config.json)
        quantization_type: Quantization type for loading model (e.g., "4bit", "8bit", None)
        adapter_name: Name to assign to the loaded adapter
        
    Returns:
        Tuple of (PeftModel with loaded adapter, tokenizer, LayerSelection if saved else None)
    
    Example:
        # Load from HuggingFace:
        model, tokenizer, layer_selection = load_adapter("wassname/antipasto-gemma-3-1b-honesty")
        
        # Load from local path:
        model, tokenizer, layer_selection = load_adapter(Path("outputs/adapters/my_run"))
        
        # For inference:
        from antipasto.gen import ScaleAdapter
        with ScaleAdapter(model, coeff=1.0):
            output = model.generate(...)
    """
    from antipasto.peft_utils.antipasto_adapter import register_antipasto_peft, AntiPaSTOConfig
    from antipasto.train.model_setup import load_model
    
    # Resolve path (downloads from HuggingFace if needed)
    adapter_folder = resolve_adapter_path(adapter_folder)
    
    # Register AntiPaSTO adapter type
    register_antipasto_peft()
    
    # Load adapter_config.json (standard PEFT config)
    adapter_config_path = adapter_folder / "adapter_config.json"
    if not adapter_config_path.exists():
        raise ValueError(f"adapter_config.json not found in {adapter_folder}")
    
    with open(adapter_config_path) as f:
        adapter_config_dict = json.load(f)
    
    # Determine model_id from config if not provided
    if base_model is None and model_id is None:
        model_id = adapter_config_dict.get("base_model_name_or_path")
        if model_id is None:
            raise ValueError(
                "Must provide base_model or model_id, or have base_model_name_or_path in adapter_config.json"
            )
    
    # Load base model if not provided
    if base_model is None:
        base_model, tokenizer = load_model(model_id, quantization_type=quantization_type)
    else:
        tokenizer = None
    
    # Load SVD bases (required for correct reload)
    svd_bases_path = adapter_folder / "0_svd_bases.safetensors"
    if not svd_bases_path.exists():
        raise ValueError(f"0_svd_bases.safetensors not found in {adapter_folder}")
    svd_bases = safetensors.torch.load_file(svd_bases_path)
    logger.info(f"Loaded SVD bases for {len(svd_bases) // 3} layers")
    
    # Build AntiPaSTOConfig from adapter_config.json - fail fast on missing required fields
    adapter_config = AntiPaSTOConfig(
        r=adapter_config_dict["r"],
        rotate_u=adapter_config_dict["rotate_u"],
        rotate_v=adapter_config_dict["rotate_v"],
        rotation_method=adapter_config_dict["rotation_method"],
        max_rotation_angle=adapter_config_dict["max_rotation_angle"],
        alpha=adapter_config_dict["alpha"],
        task_type=adapter_config_dict["task_type"],
        target_modules=adapter_config_dict["target_modules"],
        svd_bases=svd_bases,
    )
    
    # Create PeftModel with adapter
    model = PeftModel(base_model, adapter_config, adapter_name=adapter_name)
    
    # Clear svd_bases from config after creation
    adapter_config.svd_bases = None
    
    logger.info(f"Adapter configured: rank={adapter_config.r}, layers={len([n for n, m in model.named_modules() if hasattr(m, 'antipasto_u')])}")
    
    # Load weights
    sd = safetensors.torch.load_file(adapter_folder / "adapter_model.safetensors")
    sd = add_adapter_name_to_sd(sd, adapter_name=adapter_name, prefix="antipasto_")
    
    result = model.load_state_dict(sd, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected keys in state_dict: {result.unexpected_keys[:5]}")
    
    # Load layer_selection if saved (optional, for evaluation)
    layer_selection = None
    layer_selection_path = adapter_folder / "0_layer_selection.json"
    if layer_selection_path.exists():
        with open(layer_selection_path) as f:
            layer_selection = LayerSelection.from_dict(json.load(f))
    
    logger.info(f"Loaded adapter from {adapter_folder}")
    
    # Load training_config.json if saved (optional, for display/inference metadata)
    training_config = None
    training_config_path = adapter_folder / "training_config.json"
    if training_config_path.exists():
        with open(training_config_path) as f:
            training_config = json.load(f)
        # Attach to model for easy access
        model.antipasto_training_config = training_config
    
    return model, tokenizer, layer_selection
