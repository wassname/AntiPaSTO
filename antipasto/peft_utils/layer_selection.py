#!/usr/bin/env python3
"""Centralized layer selection logic for AntiPaSTO training.

FLOW:
1. compute_simple_layer_selection():
   - Computes SVD for all linear layers (needed for adapter init)
   - Selects layers uniformly across valid depth range
   - Computes subspaces: write, read, taskdiff, and combinations

Supported loss_subspace types:
- taskdiff_x_suppressed_x_write (default): empirical stenographic + writable
- taskdiff_x_logits_read: task signal that affects lm_head output
- taskdiff_x_write_not_read: task signal in static write-not-read space
- taskdiff_x_write_x_notlogits: task ∩ write ∩ (lm_head^⊥)
- write, taskdiff: basic building blocks

Key functions:
- compute_simple_layer_selection(): Main entry point for layer/subspace selection
- find_linear_layers(): Discover all linear modules in model
- resolve_target_modules(): Expand "residual-writers" etc. to concrete module lists

Subspace operations are in antipasto/peft_utils/subspaces.py
"""
import re
import pandas as pd
from dataclasses import dataclass
import hashlib
from typing import Dict, List, NamedTuple, Optional, Union
from loguru import logger
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F # no not remove
from baukit.nethook import TraceDict
from torch.utils.data import DataLoader, Subset
from transformers import DataCollatorWithPadding
import gc
from antipasto.peft_utils.subspaces import (
    compute_lm_head_svd,
    compute_lm_head_subspace,
    compute_suppressed_from_hidden_states,
    compute_task_diff_from_hidden_states,
    compute_module_subspace_from_svds,
    compute_stenographic_subspace,
    compute_write_not_read_subspace,
    compute_write_x_notlogits_subspace,
    compute_task_lm_head_subspace,
    compute_task_wnr_subspace,
    find_write_modules,
    find_read_modules,
    approx_intersection,
    Subspace,
    orthonormalize,
)
# model_layer_list imported inside functions to avoid circular import at module load


class AdapterComponents(NamedTuple):
    """Components of an AntiPaSTO adapter for a single layer.
    
    U: [d_out, r] - base output projection (frozen SVD)
    V: [d_in, r] - base input projection (frozen SVD)
    R: [r, r] - learned rotation matrix for U (Cayley transform of params)
    S: [r] - singular values (frozen base)
    S_scaled: [r] - singular values after learned scaling (S * exp(α*λ) or S + α*δ)
    W_res: [d_out, d_in] - residual weight (frozen, for stripping from outputs)
    
    Usage:
        # Output-side projection (standard):
        y_adapter = y - x @ W_res.T  # strip residual
        s = y_adapter @ (U @ R)      # project to S-space via rotated U
        
        # Input-side projection (frozen, for loss_frozen_S mode):
        s_frozen = x @ V * S         # project via frozen V and S (bypasses learned rotation/scaling)
    """
    U: torch.Tensor      # [d_out, r]
    V: torch.Tensor      # [d_in, r]
    R: torch.Tensor      # [r, r] 
    S: torch.Tensor      # [r] - frozen base
    S_scaled: torch.Tensor  # [r] - after learned scaling
    W_res: torch.Tensor  # [d_out, d_in]


def get_adapter_components(model, layer_name: str, coef: float, adapter_name: str, dtype=torch.bfloat16) -> AdapterComponents:
    """Extract adapter components. For LoRA/DoRA, returns identity U/V and ones S."""
    adapter_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            adapter_module = module
            break
    
    if adapter_module is None:
        raise ValueError(f"No module found at {layer_name}")
    
    # Check if this is an AntiPaSTO adapter
    if hasattr(adapter_module, 'antipasto_u') and adapter_name in adapter_module.antipasto_u:
        # AntiPaSTO: use actual SVD components
        U = adapter_module.antipasto_u[adapter_name].detach()  # [d_out, r]
        V = adapter_module.antipasto_v[adapter_name].detach()  # [d_in, r]
        S = adapter_module.antipasto_s[adapter_name].detach()  # [r] - frozen base
        W_res = adapter_module.antipasto_w_res[adapter_name].detach()  # [d_out, d_in]
        
        # Get rotation matrix R (identity if no rotation params)
        if adapter_name in adapter_module.antipasto_rotation_params_u:
            params_u = adapter_module.antipasto_rotation_params_u[adapter_name]
            rotation_method = adapter_module.antipasto_rotation_method[adapter_name]
            max_angle = adapter_module.antipasto_max_rotation_angle[adapter_name]
            R = adapter_module._get_rotation(params_u, alpha=coef, rotation_method=rotation_method, max_angle=max_angle).detach()
        else:
            R = torch.eye(U.shape[1], device=U.device, dtype=dtype)
        
        # Compute S_scaled: S + coef * delta_s
        delta_s = adapter_module.antipasto_delta_s[adapter_name]
        S_scaled = (S + coef * delta_s).detach()
        
        # Cast all to requested dtype for consistent matmuls
        return AdapterComponents(
            U=U.to(dtype), V=V.to(dtype), R=R.to(dtype), 
            S=S.to(dtype), S_scaled=S_scaled.to(dtype), W_res=W_res.to(dtype)
        )
    else:
        # LoRA/DoRA: identity projections (work in activation space)
        # Get dimensions from the module's weight
        if hasattr(adapter_module, 'weight'):
            d_out, d_in = adapter_module.weight.shape
            device = adapter_module.weight.device
        elif hasattr(adapter_module, 'base_layer') and hasattr(adapter_module.base_layer, 'weight'):
            d_out, d_in = adapter_module.base_layer.weight.shape
            device = adapter_module.base_layer.weight.device
        else:
            raise ValueError(f"Cannot determine dimensions for {layer_name}")
        
        # Use min dim as rank to keep projections square-ish
        r = min(d_in, d_out)
        
        # Identity projections: U and V are identity-like, S is ones
        # This makes x @ V * S @ U.T ≈ x (passes through unchanged)
        U = torch.eye(d_out, r, device=device, dtype=dtype)  # [d_out, r]
        V = torch.eye(d_in, r, device=device, dtype=dtype)   # [d_in, r]
        R = torch.eye(r, device=device, dtype=dtype)          # [r, r]
        S = torch.ones(r, device=device, dtype=dtype)         # [r]
        S_scaled = S.clone()
        W_res = torch.zeros(d_out, d_in, device=device, dtype=dtype)  # [d_out, d_in]

        return AdapterComponents(U=U.to(dtype), V=V.to(dtype), R=R.to(dtype), S=S.to(dtype), S_scaled=S_scaled.to(dtype), W_res=W_res.to(dtype))





def build_regexp(layer_indices: List[int], module_suffixes: List[str]) -> str:
    """Build PEFT target_modules regex from layer indices and module suffixes (Cartesian product)."""
    layer_nums = "|".join(str(L) for L in sorted(set(layer_indices)))
    module_names = "|".join(sorted(set(module_suffixes)))
    return f".*\\.({layer_nums})\\..*({module_names})"


def build_regexp_from_paths(layer_paths: List[str]) -> str:
    """Build PEFT target_modules regex from specific layer paths (no Cartesian product).
    
    Given paths like ['model.layers.0.mlp.gate_proj', 'model.layers.2.self_attn.o_proj'],
    builds regex that matches ONLY those specific layer×module combinations.
    """
    if not layer_paths:
        raise ValueError("No layer paths provided")
    
    # Extract layer_idx and module_name from each path
    patterns = []
    for path in layer_paths:
        layer_idx = path_to_layer(path)
        if layer_idx == -1:
            continue
        module_name = path_to_module_name(path)
        # Match this specific layer.module combo
        patterns.append(f"\\.{layer_idx}\\..*{module_name}")
    
    if not patterns:
        raise ValueError(f"Could not parse any layer paths: {layer_paths[:3]}")
    
    # Join with | to match any of the specific paths
    return ".*(" + "|".join(patterns) + ")"


@dataclass
class LayerSelection:
    """Layer selection result: which layers get adapters vs loss computation."""
    adapter_layer_indices: List[int]
    loss_layer_indices: List[int]
    adapter_layer_names: List[str]
    loss_layer_names: List[str]
    n_candidates: int = 0  # Total available candidates (for logging)
    
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "adapter_layer_indices": self.adapter_layer_indices,
            "loss_layer_indices": self.loss_layer_indices,
            "adapter_layer_names": self.adapter_layer_names,
            "loss_layer_names": self.loss_layer_names,
            "n_candidates": self.n_candidates,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'LayerSelection':
        """Deserialize from dict."""
        return cls(
            adapter_layer_indices=d["adapter_layer_indices"],
            loss_layer_indices=d["loss_layer_indices"],
            adapter_layer_names=d["adapter_layer_names"],
            loss_layer_names=d["loss_layer_names"],
            n_candidates=d.get("n_candidates", 0),
        )
    
    @property
    def adapter_regex(self) -> str:
        """Build PEFT target_modules regex from specific adapter layer paths (sparse, not Cartesian)."""
        return build_regexp_from_paths(self.adapter_layer_names)
    
    def translate_to_peft_model(self, model) -> 'LayerSelection':
        """Translate layer names for PeftModel (adds base_model.model prefix).
        
        After wrapping with PeftModel, layer paths change:
        - Before: 'model.layers.9.mlp.down_proj'  
        - After:  'base_model.model.model.layers.9.mlp.down_proj'
        
        This finds the correct paths by checking what actually exists in the PeftModel.
        """
        def translate_name(old_name: str) -> str:
            model_modules = {name for name, _ in model.named_modules()}

            # Most common cases
            candidates = [
                old_name,
                f"base_model.model.{old_name}",
            ]
            for candidate in candidates:
                if candidate in model_modules:
                    return candidate

            # More robust: if the wrapper prefixes are unknown, try a unique suffix match.
            # This handles cases like base_model.model.model.<old_name> or other wrappers.
            suffix_matches = [name for name in model_modules if name.endswith(old_name)]
            if len(suffix_matches) == 1:
                return suffix_matches[0]

            raise KeyError(
                "Could not translate module path for PeftModel. "
                f"old_name={old_name!r}. "
                "Expected either an exact match or a prefixed match like 'base_model.model.<old_name>', "
                "or a unique suffix match. "
                f"suffix_matches={suffix_matches[:5]}" + ("..." if len(suffix_matches) > 5 else "")
            )
        
        return LayerSelection(
            adapter_layer_indices=self.adapter_layer_indices,
            loss_layer_indices=self.loss_layer_indices,
            adapter_layer_names=[translate_name(n) for n in self.adapter_layer_names],
            loss_layer_names=[translate_name(n) for n in self.loss_layer_names],
        )


def path_to_layer(path: str) -> int:
    """Extract layer index from module path.
    
    Args:
        path: Module path like 'model.layers.5.mlp.down_proj'
        
    Returns:
        Layer index (e.g., 5), or -1 if not found
    """
    patterns = [
        r"\.layers\.(\d+)\.",
        r"\.h\.(\d+)\.",
        r"\.blocks\.(\d+)\.",
        r"^layers\.(\d+)\.",
        r"^model\.layers\.(\d+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
    return -1


def path_to_module_name(path: str) -> str:
    """Extract module name (last component) from module path.
    
    Args:
        path: Module path like 'model.layers.5.mlp.down_proj'
        
    Returns:
        Module name (e.g., 'down_proj')
    """
    return path.split('.')[-1]


def build_layer_info(layer_paths: List[str]) -> Dict[str, dict]:
    """Build layer_info dict from layer paths.
    
    Args:
        layer_paths: List of module paths like 'model.layers.5.mlp.down_proj'
        
    Returns:
        Dict mapping path -> {layer_idx: int, module_name: str}
    """
    return {
        path: {
            'layer_idx': path_to_layer(path),
            'module_name': path_to_module_name(path),
        }
        for path in layer_paths
    }


def normalize_layer_spec(layer_spec: List[float | int], total_layers: int) -> List[int]:
    """Convert layer specs (fractions or offsets) to absolute layer numbers."""
    normalized = []
    for x in layer_spec:
        if (x >= 0) and (x < 1):
            x = int(x * total_layers)
        layer_num = int(x) % total_layers
        normalized.append(layer_num)
    return normalized


def find_linear_layers(
    model: nn.Module,
    layer_indices: Optional[List[int]]=None,
    module_suffixes: Optional[List[str]]=None,
    blocklist: List[str] = ['vision']
) -> List[str]:
    """Find Linear modules at specified layer depths with given suffixes.
    
    Returns:
        List of layer paths, sorted by (layer_idx, module_name)
    """
    selected = []
    for name, module in model.named_modules():
        if any(block in name for block in blocklist):
            continue
        if name.endswith('.base_layer'):
            continue
        if not isinstance(module, nn.Linear):
            continue
        
        layer_idx = path_to_layer(name)
        if layer_idx == -1:
            continue
        
        if layer_indices and (layer_idx not in layer_indices):
            continue

        if module_suffixes and not any(name.endswith(suffix) for suffix in module_suffixes):
            continue
    
        selected.append(name)
    
    # Sort by (layer_idx, module_name) for consistent ordering
    selected = sorted(set(selected), key=lambda p: (path_to_layer(p), p))
    return selected


def find_residual_connected_modules(
    model: nn.Module,
    blocklist: List[str] = ['vision', 'embed', 'lm_head', 'norm']
) -> List[str]:
    """Auto-detect modules that read from or write to residual stream.
    
    A module is residual-connected if its input OR output dimension matches
    the model's hidden_size. This generalizes across architectures.
    
    Returns:
        List of module name suffixes, sorted alphabetically (e.g., ['down_proj', 'o_proj', 'q_proj', ...])
    """
    hidden_size = model.config.hidden_size
    
    residual_modules = set()
    
    for name, module in model.named_modules():
        if any(block in name for block in blocklist):
            continue
        if not isinstance(module, nn.Linear):
            continue
        
        # Check if in a transformer layer
        if path_to_layer(name) == -1:
            continue
        
        in_features = module.in_features
        out_features = module.out_features
        
        # Residual-connected: input or output matches hidden_size
        if in_features == hidden_size or out_features == hidden_size:
            suffix = name.split('.')[-1]
            residual_modules.add(suffix)
    
    result = sorted(residual_modules)
    logger.debug(f"Auto-detected residual-connected modules: {result}")
    return result


def resolve_target_modules(
    model: nn.Module,
    target_modules_spec: List[str],
) -> List[str]:
    """Resolve target_modules config to concrete module suffix list.
    
    Args:
        model: Model to inspect for auto-detection
        target_modules_spec: List of module suffixes, or single-element list with special value:
            - ["residual-writers"]: auto-detect modules that write to residual (o_proj, down_proj, ...)
            - ["residual-readers"]: auto-detect modules that read from residual (q_proj, k_proj, ...)
            - ["residual-all"]: all residual-connected modules
            - ["down_proj", "o_proj"]: explicit list of module suffixes
    
    Returns:
        List of module name suffixes (e.g., ["o_proj", "down_proj"])
    """
    SPECIAL_VALUES = {"residual-writers", "residual-readers", "residual-all"}
    
    # Check for single-element special value
    if len(target_modules_spec) == 1 and target_modules_spec[0] in SPECIAL_VALUES:
        spec = target_modules_spec[0]
        if spec == "residual-writers":
            result = find_write_modules(model)
            logger.info(f"Auto-detected residual-writers: {result}")
        elif spec == "residual-readers":
            result = find_read_modules(model)
            logger.info(f"Auto-detected residual-readers: {result}")
        elif spec == "residual-all":
            result = find_residual_connected_modules(model)
            logger.info(f"Auto-detected residual-all: {result}")
        
        if not result:
            raise ValueError(
                f"Auto-detection returned empty list for {spec!r}. "
                f"Model may lack hidden_size config or have non-standard architecture."
            )
        return result
    
    # Explicit list of module suffixes
    logger.info(f"Using explicit target_modules: {target_modules_spec}")
    return target_modules_spec


def compute_task_relevance(
    hsS_cho: torch.Tensor,
    hsS_rej: torch.Tensor,
    S: torch.Tensor,
) -> torch.Tensor:
    """Compute per-dimension task relevance for loss weighting.
    
    Returns weights in [0, 1] indicating how much each S-dimension
    contributes to cho/rej differentiation. Uses mean|cho-rej| * S
    to emphasize dims that both separate classes AND have high singular values.
    
    Args:
        hsS_cho: Chosen activations in S-space [n_samples, r]
        hsS_rej: Rejected activations in S-space [n_samples, r]
        S: Singular values [r]
    
    Returns:
        relevance: [r] weights normalized to [0, 1] range
    """
    # Diff-based relevance: dims where cho/rej actually differ, weighted by S
    mean_diff = (hsS_cho - hsS_rej).mean(dim=0).abs()  # [r]
    relevance = mean_diff * S
    
    # Normalize to [0, 1] range
    relevance = relevance / relevance.max().clamp(min=1e-8)
    
    return relevance


@dataclass
class SubspaceCache:
    """Cache of computed subspace bases, extensible for new subspace types.
    
    Subspaces are computed "for free" during gradient selection (same forward pass).
    Keys: 'suppressed', 'write', 'churn', etc. Values: Subspace objects with V and optional S.
    
    Use get() to retrieve subspaces, returns None if not computed.
    Use get_basis() to get just V (for backward compat).
    """
    _subspaces: Dict[str, "Subspace"] = None
    
    def __post_init__(self):
        if self._subspaces is None:
            self._subspaces = {}
    
    def get(self, name: str) -> Optional["Subspace"]:
        """Get Subspace by name, returns None if not computed."""
        return self._subspaces.get(name)
    
    def get_basis(self, name: str) -> Optional[torch.Tensor]:
        """Get just the V basis tensor (backward compat)."""
        sub = self._subspaces.get(name)
        return sub.V if sub is not None else None
    
    def set(self, name: str, subspace: Union["Subspace", torch.Tensor, None]):
        """Set subspace. Accepts Subspace object or raw V tensor (wrapped automatically)."""
        if subspace is None:
            return
        if isinstance(subspace, torch.Tensor):
            # Backward compat: wrap raw tensor in Subspace
            from antipasto.peft_utils.subspaces import Subspace
            subspace = Subspace(subspace, name=name)
        self._subspaces[name] = subspace
    
    def __contains__(self, name: str) -> bool:
        return name in self._subspaces
    
    def keys(self):
        return self._subspaces.keys()
    
    @property
    def suppressed(self) -> Optional[torch.Tensor]:
        """Shorthand for common subspaces (returns V for backward compat)."""
        return self.get_basis('suppressed')
    
    @property
    def write(self) -> Optional[torch.Tensor]:
        return self.get_basis('write')


@dataclass
class GradientSelection:
    """Full gradient-based selection result: layers, modules, AND dimensions.
    
    Contains everything needed to set up adapter + loss layers in one pass,
    computed from gradient importance rather than hardcoded rules.
    """
    layer_selection: LayerSelection           # Which layers+modules to use
    precomputed_indices: Dict[str, torch.Tensor]  # Which S-dims per layer {layer_name: [r]}
    subspaces: SubspaceCache = None           # Computed subspaces (suppressed, write, etc.)
    
    def __post_init__(self):
        if self.subspaces is None:
            self.subspaces = SubspaceCache()
    
    # Backward compat properties
    @property
    def P_write(self) -> Optional[torch.Tensor]:
        return self.subspaces.write
    
    @property
    def P_suppressed(self) -> Optional[torch.Tensor]:
        return self.subspaces.suppressed


def compute_simple_layer_selection(
    model: nn.Module,
    r: int,
    n_modules: int = 42,
    loss_layer_frac: float = 0.8,
    min_adapter_layer_frac: float = 0.1,
    candidate_modules_filter: Optional[List[str]] = None,
    dim_select_method: str = "top_s",
    loss_subspace: str = "write",
    top_k: int = 256,
    tokenizer = None,
    dataset_pt = None,
    n_samples: int = 512,
    bs: int = 8,
    seed: int = 42,
) -> GradientSelection:
    """Simple layer selection WITHOUT gradient computation (no backward pass).
    
    For large models (12B+) where gradient collection OOMs. Uses:
    - Uniform layer selection across valid depth range
    - top_s dim selection (just top-r singular values)
    - Weight-only subspaces by default; task-diff subspaces if dataset provided
    
    If loss_subspace requires task_diff (e.g. task_intersect_*, stenographic),
    pass tokenizer and dataset_pt to enable forward pass for hidden states.
    
    Args:
        model: Base model (no adapter yet)
        r: Target adapter rank (dims per layer)
        n_modules: Total layer×module combinations to select for adapters
        loss_layer_frac: Depth fraction (0-1) for loss layer (default 0.8 = 80% depth)
        min_adapter_layer_frac: Minimum depth fraction for adapter placement (default 0.1)
        candidate_modules_filter: Optional list of module suffixes to consider
        dim_select_method: "top_s" (recommended) or "random"
        loss_subspace: Subspace for loss projection
        top_k: Rank for subspace computation
        tokenizer: Required if loss_subspace needs task_diff (e.g. task_intersect_*)
        dataset_pt: Required if loss_subspace needs task_diff
        n_samples: Samples for task_diff computation (default 64)
        bs: Batch size for forward pass (default 8)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_layers = model.config.num_hidden_layers

    def _stable_u32(key: str) -> int:
        h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little") % (2**32)
    
    # Find all candidate layer paths
    all_layer_paths_raw = find_linear_layers(model)
    n_total = len(all_layer_paths_raw)

    if candidate_modules_filter is not None:
        all_layer_paths = [
            p for p in all_layer_paths_raw
            if path_to_module_name(p) in set(candidate_modules_filter)
        ]
    else:
        all_layer_paths = all_layer_paths_raw
    n_after_filter = len(all_layer_paths)
    
    filter_desc = f"target_modules={candidate_modules_filter}" if candidate_modules_filter else "all modules"
    logger.info(f"Simple layer selection: {n_total} total → {n_after_filter} after {filter_desc} → requesting {n_modules}")
    
    # Compute SVD for adapter candidate layers
    logger.info(f"Computing SVD for {len(all_layer_paths)} adapter candidate layers...")
    layer_svd_cpu = {}
    for path in tqdm(all_layer_paths, desc="SVD (adapter)"):
        module = model.get_submodule(path)
        W = module.weight.detach().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        layer_svd_cpu[path] = (U.cpu(), S.cpu(), Vh.cpu())
    
    # Also compute SVD for read/write modules not in adapter filter (needed for subspace computation)
    # Find read + write modules that may not be in adapter candidates
    read_modules = find_read_modules(model)
    write_modules = find_write_modules(model)
    subspace_modules = set(read_modules) | set(write_modules)
    adapter_modules = set(path_to_module_name(p) for p in all_layer_paths)
    missing_modules = subspace_modules - adapter_modules
    
    if missing_modules:
        extra_paths = [p for p in all_layer_paths_raw if path_to_module_name(p) in missing_modules]
        logger.info(f"Computing SVD for {len(extra_paths)} extra modules for subspaces: {missing_modules}")
        for path in tqdm(extra_paths, desc="SVD (subspace)"):
            if path not in layer_svd_cpu:
                module = model.get_submodule(path)
                W = module.weight.detach().float()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                layer_svd_cpu[path] = (U.cpu(), S.cpu(), Vh.cpu())
    
    layer_info = build_layer_info(list(layer_svd_cpu.keys()))
    
    # Compute adapter layer range: [min_adapter_layer_frac, loss_layer_frac)
    min_adapter_layer_idx = int(min_adapter_layer_frac * num_layers)
    min_adapter_layer_idx = max(0, min(min_adapter_layer_idx, num_layers - 2))
    
    loss_layer_idx = int(loss_layer_frac * num_layers)
    loss_layer_idx = max(min_adapter_layer_idx + 1, min(loss_layer_idx, num_layers - 1))
    loss_layer_indices = [loss_layer_idx]
    
    # Filter to valid range and sort by layer
    valid_paths = [
        p for p in all_layer_paths
        if min_adapter_layer_idx <= layer_info[p]['layer_idx'] < loss_layer_idx
    ]
    valid_paths = sorted(valid_paths, key=lambda p: (layer_info[p]['layer_idx'], p))
    
    if not valid_paths:
        raise ValueError(
            f"No adapter layers in range [{min_adapter_layer_idx}, {loss_layer_idx}). "
            f"Try adjusting min_adapter_layer_frac={min_adapter_layer_frac} or loss_layer_frac={loss_layer_frac}"
        )
    
    # Select uniformly across available layers (instead of gradient ranking)
    if len(valid_paths) <= n_modules:
        selected_layer_names = valid_paths
    else:
        # Uniform sampling: pick evenly spaced indices
        step = len(valid_paths) / n_modules
        indices = [int(i * step) for i in range(n_modules)]
        selected_layer_names = [valid_paths[i] for i in indices]
    
    selected_layer_indices = sorted(set(layer_info[p]['layer_idx'] for p in selected_layer_names))
    
    logger.info(
        f"Adapter layer range: [{min_adapter_layer_idx}, {loss_layer_idx}) of {num_layers} layers, "
        f"selected {len(selected_layer_names)} adapters uniformly"
    )
    
    # Find loss layer anchor module
    candidates_at_depth = [
        p for p in all_layer_paths
        if layer_info[p]['layer_idx'] == loss_layer_idx
    ]
    if not candidates_at_depth:
        # Fall back to closest
        closest_path = min(all_layer_paths, key=lambda p: abs(layer_info[p]['layer_idx'] - loss_layer_idx))
        closest_idx = layer_info[closest_path]['layer_idx']
        candidates_at_depth = [p for p in all_layer_paths if layer_info[p]['layer_idx'] == closest_idx]
        loss_layer_indices = [closest_idx]
    
    preferred_order = ['q_proj', 'o_proj', 'mlp', 'gate_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj']
    loss_layer_names = []
    for pref in preferred_order:
        matching = [p for p in candidates_at_depth if p.endswith(f'.{pref}')]
        if matching:
            loss_layer_names = [matching[0]]
            break
    if not loss_layer_names:
        loss_layer_names = [candidates_at_depth[0]]
    
    logger.info(f"Loss layer: idx={loss_layer_indices[0]} (frac={loss_layer_frac}), basis module: {loss_layer_names[0].split('.')[-1]}")

    # hidden_states[i] = hidden state AFTER layer i-1 (0 = embeddings)
    # loss_layer_indices are in layer-index space, so +1 to align with hidden_states indexing.
    loss_hs_frac_for_task = (loss_layer_indices[0] + 1) / num_layers
    
    # =========================================================================
    # COLLECT HIDDEN STATES (for all activation-based subspaces and dim selection)
    # =========================================================================
    # Simplified: if dataset_pt is provided, always collect hidden states and compute
    # all subspaces. This makes testing easier and the cost is negligible (~128MB for 7B).
    wanda_methods = ("wanda_svd", "wanda_svd_l1", "wanda_svd_l1_split", "wanda_svd_l1_trip", "wanda_svd_l1_overlap", "wanda_svd_l1_diff", "wanda_svd_balanced", "wanda_svd_fisher", "wanda_svd_fisher_task", "wanda_svd_fisher_task_split", "wanda_svd_task", "wanda_svd_task_balanced", "wanda_svd_triple", "wanda_svd_orthogonal")
    
    hs_stacked = None
    task_diffs = None  # Per-layer task diff vectors for wanda_svd_task variants
    
    if tokenizer is not None and dataset_pt is not None:
        # Always collect hidden states when dataset available - enables all subspaces
        logger.info(f"Collecting hidden states for subspace computation...")
        
        n_samples_use = min(n_samples, len(dataset_pt))
        n_samples_use = n_samples_use - (n_samples_use % 2)  # Ensure even for cho/rej pairs
        subset = Subset(dataset_pt, list(range(n_samples_use)))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", max_length=128)
        dataloader = DataLoader(subset, batch_size=min(bs, n_samples_use), collate_fn=data_collator)
        
        all_hidden_states = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting hidden states"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch, output_hidden_states=True, use_cache=False)
                # Take last token hidden states: [batch, n_layers+1, d_model]
                last_pos = batch["attention_mask"].sum(dim=1) - 1
                batch_indices = torch.arange(outputs.hidden_states[0].shape[0], device=device)
                # Stack all layer outputs for last token
                hs_batch = torch.stack([
                    h[batch_indices, last_pos, :] for h in outputs.hidden_states
                ], dim=1)  # [batch, n_layers+1, d_model]
                all_hidden_states.append(hs_batch.cpu())
        
        hs_stacked = torch.cat(all_hidden_states, dim=0).to(device)  # [n_samples, n_layers+1, d_model]
        logger.info(f"Collected hidden_states: {hs_stacked.shape}")
        
        # Precompute per-layer task diffs (useful for multiple dim_select methods)
        hs_cho = hs_stacked[::2]
        hs_rej = hs_stacked[1::2]
        task_diffs = (hs_cho - hs_rej).float().mean(dim=0)  # [n_layers+1, d_model]
        logger.debug(f"Precomputed task diffs: shape={task_diffs.shape}")
    else:
        # No dataset - only weight-based subspaces available
        if dim_select_method in wanda_methods:
            raise ValueError(
                f"dim_select_method='{dim_select_method}' requires hidden states. "
                f"Pass tokenizer and dataset_pt to compute_simple_layer_selection()."
            )
        # Activation-based loss_subspace will error later with more context
    
    # =========================================================================
    # DIMENSION SELECTION
    # =========================================================================
    # For Wanda-style methods, we project hidden states onto SVD basis.
    # 
    # hs_stacked[:, layer_idx, :] is the RESIDUAL STREAM at layer l.
    # 
    # For W = U @ S @ Vh (where W: [out, in]):
    #   - Input space basis: Vh.T: [d_in, r]
    #   - Output space basis: U: [d_out, r]
    #
    # READ modules (q_proj, k_proj, v_proj, gate_proj, up_proj):
    #   - Input FROM residual (d_in = d_model)
    #   - Use Vh.T to project hs_layer (both are d_model) ✓
    #
    # WRITE modules (o_proj, down_proj):
    #   - Output TO residual (d_out = d_model), but input is internal
    #   - d_in = head_dim × n_heads (o_proj) or intermediate_dim (down_proj)
    #   - Vh.T: [d_in, r] — DIMENSION MISMATCH, can't multiply with hs_layer!
    #   - U: [d_model, r] — correct dimension
    #   - Interpretation: "which output directions of this writer are aligned
    #     with the current residual stream?" This tells us which singular
    #     dimensions would be activated if the residual were fed through.
    #     Not a perfect semantic match, but dimensionally necessary and
    #     empirically useful for dimension selection.
    # =========================================================================
    
    # Precompute write module suffixes for basis selection
    write_suffixes = set(find_write_modules(model))
    
    def is_write_module(path: str) -> bool:
        """Check if module writes to residual stream (use U basis) vs reads (use Vh.T)."""
        suffix = path.split('.')[-1]
        return suffix in write_suffixes
    
    precomputed_indices = {}
    for path in selected_layer_names:
        U_layer, S_full, Vh = layer_svd_cpu[path]
        max_rank = min(r, S_full.shape[0])
        
        if dim_select_method == "top_s":
            # Top-r by singular value (index 0..r-1 since SVD returns sorted)
            indices = torch.arange(max_rank)
        elif dim_select_method == "random":
            # Deterministic per-layer randomness
            gen = torch.Generator()
            gen.manual_seed(_stable_u32(f"{seed}:dim_select:random:{path}"))
            perm = torch.randperm(S_full.shape[0], generator=gen)
            indices = perm[:max_rank].sort().values
        elif dim_select_method == "wanda_svd_l1_trip":
            # Three-way split: r/3 cho + r/3 rej + r/3 diff (task direction)
            # Explicitly includes dims aligned with steering direction
            layer_idx = layer_info[path]['layer_idx']
            
            hs_layer = hs_stacked[:, layer_idx, :].float()
            
            if is_write_module(path):
                basis = U_layer.to(hs_layer.device).float()
            else:
                basis = Vh.T.to(hs_layer.device).float()
            
            activations_S = hs_layer @ basis  # [n_samples, r_full]
            
            # Split cho/rej
            act_cho = activations_S[::2]
            act_rej = activations_S[1::2]
            
            # L1 mean for each direction
            l1_cho = act_cho.abs().mean(dim=0)
            l1_rej = act_rej.abs().mean(dim=0)
            
            # Task alignment from task_diffs
            diff = task_diffs[layer_idx].to(hs_layer.device).float()
            task_alignment = (diff @ basis).abs()  # [r_full]
            
            S_dev = S_full.to(hs_layer.device).float()
            scores_cho = S_dev * l1_cho
            scores_rej = S_dev * l1_rej
            scores_diff = S_dev * task_alignment
            
            # Take 1/3 from each ranking
            third = max_rank // 3
            top_cho = scores_cho.argsort(descending=True)[:third]
            top_rej = scores_rej.argsort(descending=True)[:third]
            top_diff = scores_diff.argsort(descending=True)[:max_rank - 2*third]
            
            # Union
            combined = torch.unique(torch.cat([top_cho, top_rej, top_diff]))
            
            # Pad if needed
            if len(combined) < max_rank:
                scores_union = torch.maximum(torch.maximum(scores_cho, scores_rej), scores_diff)
                scores_union[combined] = -float('inf')
                extra = scores_union.argsort(descending=True)[:max_rank - len(combined)]
                combined = torch.cat([combined, extra])
            
            indices = combined.sort().values[:max_rank].cpu()
        else:
            raise ValueError(f"Unknown dim_select_method: {dim_select_method}. Valid: top_s, random, wanda_svd_l1_trip")
        
        precomputed_indices[path] = indices
    
    logger.info(f"Dimension selection ({dim_select_method}): {sum(len(v) for v in precomputed_indices.values())} total dims")
    
    # Compute weight-only subspaces (no hidden states needed)
    subspaces = SubspaceCache()
    subspaces._subspaces = {}  # Initialize internal dict
    
    # Build layer_info for all paths (needed for subspace computation)
    layer_info_full = build_layer_info(list(layer_svd_cpu.keys()))
    
    # CRITICAL: Use full intermediate rank for subspace geometry computations.
    # Only truncate to top_k (loss_subspace_rank) at final storage.
    # This prevents premature cropping that loses geometric structure:
    # e.g., write(rank=256) - lm_head(rank=256) -> hfl(~128D) -> top_k=1
    # vs buggy: write(rank=1) - lm_head(rank=1) -> garbage
    INTERMEDIATE_SUBSPACE_RANK = 256
    
    # Write subspace from o_proj, down_proj column spaces
    # Use INTERMEDIATE rank for full geometry, store full Subspace for energy-based selection
    write_modules = find_write_modules(model)
    write_subspace = compute_module_subspace_from_svds(
        layer_svds=layer_svd_cpu,
        layer_info=layer_info_full,
        module_filter=write_modules,
        use_column_space=True,
        top_k=INTERMEDIATE_SUBSPACE_RANK,
        device=device,
        dtype=dtype,
        name="write",
    )
    if write_subspace is not None:
        subspaces.set('write', write_subspace)  # Full Subspace with S for energy thresholding
        logger.info(f"write subspace: rank={write_subspace.V.shape[1]}")
    
    # Read subspace from q_proj, k_proj, v_proj, up_proj, gate_proj row spaces
    read_modules = find_read_modules(model)
    read_subspace = compute_module_subspace_from_svds(
        layer_svds=layer_svd_cpu,
        layer_info=layer_info_full,
        module_filter=read_modules,
        use_column_space=False,  # row space for readers
        top_k=INTERMEDIATE_SUBSPACE_RANK,
        device=device,
        dtype=dtype,
        name="read",
    )
    if read_subspace is not None:
        subspaces.set('read', read_subspace)
    
    # attention_out and mlp_out: subsets of write space
    attn_out_modules = [m for m in write_modules if 'o_proj' in m]
    attn_out_subspace = compute_module_subspace_from_svds(
        layer_svds=layer_svd_cpu,
        layer_info=layer_info_full,
        module_filter=attn_out_modules,
        use_column_space=True,
        top_k=INTERMEDIATE_SUBSPACE_RANK,
        device=device,
        dtype=dtype,
        name="attention_out",
    )
    if attn_out_subspace is not None:
        subspaces.set('attention_out', attn_out_subspace)  # Full Subspace
    
    mlp_out_modules = [m for m in write_modules if 'down_proj' in m]
    mlp_out_subspace = compute_module_subspace_from_svds(
        layer_svds=layer_svd_cpu,
        layer_info=layer_info_full,
        module_filter=mlp_out_modules,
        use_column_space=True,
        top_k=INTERMEDIATE_SUBSPACE_RANK,
        device=device,
        dtype=dtype,
        name="mlp_out",
    )
    if mlp_out_subspace is not None:
        subspaces.set('mlp_out', mlp_out_subspace)  # Full Subspace
    
    # logits_read subspace (lm_head read directions) - use INTERMEDIATE rank
    # We also keep full (S,Vh) for null-space computations that need the tail singular vectors.
    lm_head_S_full, lm_head_Vh_full = compute_lm_head_svd(model)
    lm_head_sub = Subspace(
        lm_head_Vh_full[:INTERMEDIATE_SUBSPACE_RANK, :].T.to(device=device, dtype=dtype).detach(),
        name="logits_read",
        S=lm_head_S_full[:INTERMEDIATE_SUBSPACE_RANK].to(device=device, dtype=dtype).detach(),
    )
    if lm_head_sub is not None:
        subspaces.set('logits_read', lm_head_sub)
    
    # =========================================================================
    # ACTIVATION-BASED SUBSPACES (simplified for publication)
    # Only computes subspaces needed for 3 supported loss_subspace types:
    # - taskdiff, taskdiff_x_suppressed_x_write (default)
    # =========================================================================
    if hs_stacked is not None:
        logger.info(f"Computing activation-based subspaces (taskdiff, suppressed)...")
        
        # Compute taskdiff subspace - PCA on cho-rej difference
        task_diff_subspace = compute_task_diff_from_hidden_states(
            hidden_states=hs_stacked,
            top_k=INTERMEDIATE_SUBSPACE_RANK,
            layer_frac=loss_hs_frac_for_task,
        )
        subspaces.set('taskdiff', task_diff_subspace)
        
        # Compute suppressed subspace (written but erased by later layers)
        suppressed_subspace = compute_suppressed_from_hidden_states(
            hidden_states=hs_stacked,
            lm_head_subspace=lm_head_sub,
            top_k=INTERMEDIATE_SUBSPACE_RANK,
        )
        subspaces.set('suppressed', suppressed_subspace)
        
        # taskdiff_x_suppressed = taskdiff ∩ suppressed (stenographic signal)
        steno_subspace = compute_stenographic_subspace(
            task_diff_subspace=task_diff_subspace,
            suppressed_subspace=suppressed_subspace,
            top_k=INTERMEDIATE_SUBSPACE_RANK,
        )
        subspaces.set('taskdiff_x_suppressed', steno_subspace)
        
        # taskdiff_x_suppressed_x_write = steno ∩ write (default loss subspace)
        if write_subspace is not None:
            steno_intersect_write = approx_intersection(steno_subspace, write_subspace, top_k=INTERMEDIATE_SUBSPACE_RANK)
            subspaces.set('taskdiff_x_suppressed_x_write', steno_intersect_write)
        
        # =====================================================================
        # STATIC (model-intrinsic) subspaces - alternative to empirical suppressed
        # =====================================================================
        
        # taskdiff_x_logits_read = task signal that AFFECTS output (opposite of suppressed)
        taskdiff_logits_read = compute_task_lm_head_subspace(
            task_diff_subspace=task_diff_subspace,
            lm_head_subspace=lm_head_sub,
            top_k=INTERMEDIATE_SUBSPACE_RANK,
        )
        subspaces.set('taskdiff_x_logits_read', taskdiff_logits_read)
        
        # Static write-not-read and its task intersection
        if write_subspace is not None and read_subspace is not None:
            # write_not_read = static: write^⊥_read^⊥_lmhead
            write_not_read = compute_write_not_read_subspace(
                write_subspace=write_subspace,
                read_subspace=read_subspace,
                lm_head_subspace=lm_head_sub,
                top_k=INTERMEDIATE_SUBSPACE_RANK,
            )
            subspaces.set('write_not_read', write_not_read)
            
            # taskdiff_x_write_not_read = task in static hidden channels
            taskdiff_wnr = compute_task_wnr_subspace(
                task_diff_subspace=task_diff_subspace,
                write_not_read_subspace=write_not_read,
                top_k=INTERMEDIATE_SUBSPACE_RANK,
            )
            subspaces.set('taskdiff_x_write_not_read', taskdiff_wnr)
        
        # write_x_notlogits = static: write ∩ (lm_head^⊥)
        if write_subspace is not None:
            write_x_notlogits = compute_write_x_notlogits_subspace(
                write_subspace=write_subspace,
                lm_head_subspace=lm_head_sub,
                top_k=INTERMEDIATE_SUBSPACE_RANK,
            )
            subspaces.set('write_x_notlogits', write_x_notlogits)
            
            # taskdiff_x_write_x_notlogits = task ∩ write ∩ (lm_head^⊥)
            taskdiff_write_notlogits = approx_intersection(
                task_diff_subspace, write_x_notlogits, top_k=INTERMEDIATE_SUBSPACE_RANK
            )
            taskdiff_write_notlogits.name = 'taskdiff_x_write_x_notlogits'
            subspaces.set('taskdiff_x_write_x_notlogits', taskdiff_write_notlogits)
        
        computed_subs = list(subspaces._subspaces.keys())
        logger.info(f"Subspaces computed: {computed_subs}")
    
    # Cleanup hidden states if collected
    if hs_stacked is not None:
        del hs_stacked
        gc.collect()
        torch.cuda.empty_cache()
    
    layer_selection = LayerSelection(
        adapter_layer_indices=selected_layer_indices,
        loss_layer_indices=loss_layer_indices,
        adapter_layer_names=sorted(selected_layer_names),
        loss_layer_names=sorted(loss_layer_names),
        n_candidates=len(valid_paths),
    )
    
    # Cleanup
    del layer_svd_cpu
    gc.collect()
    
    return GradientSelection(
        layer_selection=layer_selection,
        precomputed_indices=precomputed_indices,
        subspaces=subspaces,
    )
