
from peft import PeftModel
from pathlib import Path
import safetensors.torch
import torch
import json
from loguru import logger
from typing import Optional, Tuple, Union

from antipasto.peft_utils.layer_selection import LayerSelection


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
    bake_centering: bool = True,
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
        bake_centering: If True and using lrelu/LRelu scaling, bake EMA centering into lora_B.bias
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
    adapter_folder: Path,
    base_model=None,
    model_id: str = None,
    quantization_type: str = None,
    adapter_name: str = "default",
) -> Tuple[PeftModel, Optional[LayerSelection]]:
    """Load a saved AntiPaSTO adapter with all metadata.
    
    Either provide base_model directly, OR model_id + quantization_type to load it.
    
    Args:
        adapter_folder: Path to saved adapter (contains adapter_model.safetensors, etc.)
        base_model: Pre-loaded base model (optional, provide this OR model_id)
        model_id: HuggingFace model ID to load (optional, provide this OR base_model)
        quantization_type: Quantization type for loading model (e.g., "nf4", "int8", None)
        adapter_name: Name to assign to the loaded adapter
        
    Returns:
        Tuple of (PeftModel with loaded adapter, LayerSelection if saved else None)
    
    Example:
        model, layer_selection = load_adapter(
            Path("outputs/adapters/my_run"),
            model_id="Qwen/Qwen2.5-3B-Instruct",
        )
        # For inference:
        with ScaleAdapter(model, coeff=1.0):
            output = model.generate(...)
    """
    from antipasto.peft_utils.antipasto_adapter import register_antipasto_peft
    from antipasto.train.model_setup import load_model, setup_adapter
    
    adapter_folder = Path(adapter_folder)
    
    # Register AntiPaSTO adapter type
    register_antipasto_peft()
    
    # Load base model if not provided
    if base_model is None:
        if model_id is None:
            # Try to get model_id from training_config.json
            config_path = adapter_folder / "training_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    training_config = json.load(f)
                model_id = training_config.get("model_name")
                quantization_type = quantization_type or training_config.get("quantization_type")
            else:
                raise ValueError("Must provide base_model or model_id, or have training_config.json in adapter_folder")
        
        base_model, tokenizer = load_model(model_id, quantization_type=quantization_type)
    else:
        tokenizer = None
    
    # Load layer_selection if saved
    layer_selection = None
    layer_selection_path = adapter_folder / "0_layer_selection.json"
    if layer_selection_path.exists():
        with open(layer_selection_path) as f:
            layer_selection = LayerSelection.from_dict(json.load(f))
        target_modules = layer_selection.adapter_regex
    else:
        raise ValueError(f"Missing 0_layer_selection.json in {adapter_folder}")
    
    # Load precomputed_indices if saved (for dimension selection)
    precomputed_indices = None
    indices_path = adapter_folder / "0_precomputed_indices.pt"
    if indices_path.exists():
        precomputed_indices = torch.load(indices_path, weights_only=True)
    else:
        logger.warning(f"No precomputed indices found in {adapter_folder}, proceeding without dimension selection.")
    
    # Load SVD bases if saved (for exact reload without recomputation)
    svd_bases = None
    svd_bases_st = adapter_folder / "0_svd_bases.safetensors"
    if svd_bases_st.exists():
        svd_bases = safetensors.torch.load_file(svd_bases_st)
        logger.info(f"Loaded SVD bases for {len(svd_bases) // 3} layers")
    else:
        logger.warning(f"No SVD bases found in {adapter_folder}, will recompute from base model.")
    
    # Load training config to get adapter settings
    config_path = adapter_folder / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            training_config = json.load(f)
        
        # Create minimal config for setup_adapter
        from antipasto.config import TrainingConfig
        import cattrs
        config = cattrs.structure(training_config, TrainingConfig)
        config.dataset_name = adapter_name  # Use provided adapter name
    else:
        raise ValueError(f"training_config.json not found in {adapter_folder}")
    
    # Setup adapter structure
    model = setup_adapter(
        base_model, 
        config, 
        target_modules=target_modules,
        precomputed_indices=precomputed_indices,
        svd_bases=svd_bases,
    )
    
    # Load weights
    sd = safetensors.torch.load_file(adapter_folder / "adapter_model.safetensors")
    sd = add_adapter_name_to_sd(sd, adapter_name=adapter_name, prefix="antipasto_")
    # FIXME do we use this with lora,dora,road,vera,ia3 too?
    
    result = model.load_state_dict(sd, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected keys in state_dict: {result.unexpected_keys[:5]}")
    
    logger.info(f"Loaded adapter from {adapter_folder}")
    
    return model, tokenizer, layer_selection
