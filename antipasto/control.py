import dataclasses
import functools
import re
import typing
from typing import Dict, List, Optional, Iterable, Tuple, Union, Callable, Any, TYPE_CHECKING
from jaxtyping import Float
import warnings
from collections import OrderedDict
from baukit import TraceDict
import torch
from torch import Tensor, nn
from einops import einsum

import contextlib
from transformers import PretrainedConfig, PreTrainedModel


def noop_edit(output, layer, inputs):
    return output


def model_layer_list(model: PreTrainedModel) -> torch.nn.ModuleList:

    target_suffixes = [
        "repeng_layers",  # override
        "model.layers",  # llama, mistral, gemma, qwen, ...
        "transformer.h",  # gpt-2
    ]
    for suffix in target_suffixes:
        candidates = [
            v
            for k, v in model.named_modules()
            if k.endswith(suffix) and isinstance(v, torch.nn.ModuleList)
        ]
        if len(candidates) == 1:
            return candidates[0]

    raise ValueError(
        f"don't know how to get layer list for {type(model)}! try assigning `model.repeng_layers = ...` to override this search."
    )


def get_available_layers(model, regex_filter: Optional[str] = None, layer_range: Optional[Tuple[int, int]] = None) -> Tuple[List[str], List[str]]:
    """Find available layers in a model using named_parameters style paths

    Usage:
        ```
        # all blocks and layers with weights
        get_available_layers(model, layer_range=(0.1, 0.9))
        # get hidden states from layer 10% to 90%
        get_available_layers(model, regex_filter="\d+$", layer_range=(0.1, 0.9))
        # ['model.layers.10', 'model.layers.11',...]
        # get k projections from layer 10 to 20
        get_available_layers(model, regex_filter="k_proj$", layer_range=(10, 20))
        ```

    Outputs:
        - short names with layer numbers replaced by {N}, e.g. `['model.layers.{N}.k_proj', ...]`
        - full names with layer numbers, e.g. `['model.layers.10.k_proj', 'model.layers.11',...]`
    
    """

    # linear layers
    available_layers = [k.replace(".weight", "") for k, v in model.named_parameters()]

    # parents/blocks
    for l in available_layers:
        while len(l) > 0:
            l = ".".join(l.split(".")[:-1])
            if l not in available_layers and l != "":
                available_layers.append(l)

    # filter by range
    n_layers = len(model_layer_list(model))
    if layer_range is not None:
        # handle fractions
        if all(isinstance(x, float) for x in layer_range):
            layer_range = (int(layer_range[0] * n_layers), int(layer_range[1] * n_layers))

        # handle negative
        for i, n in enumerate(layer_range):
            if n < 0:
                layer_range[i] = n_layers + n

        # filter to range
        layer_range = list(range(*layer_range))
        available_layers = [
            s for s in available_layers if any(f".{i}." in s or s.endswith(f".{i}") for i in layer_range)
        ]

    if regex_filter is not None:
        available_layers = [s for s in available_layers if re.search(regex_filter, s)]

    # remove layer numbers
    short_available_layers = sorted(
        set(re.sub(r"\d+", "{N}", s) for s in available_layers)
    )
    return short_available_layers, available_layers


@torch.no_grad()
def baukit_dir_add_hook(
    output: Float[Tensor, "... d_out"],
    layer: str,
    inputs,
    directions: Dict[str, Any],  # dict with {U, delta_s, V} or Tensor
    coeff: float = 1.0,
):
    """
    Edit layer output by applying weight perturbation or activation bias.
    
    Two modes:
    1. S-weighted SVD steering: direction is dict with {'U', 'delta_s', 'V'}
       - U: [d_out, r] = U_svd * sqrt(S), V: [d_in, r] = V_svd * sqrt(S)
       - delta_s: [r] full-rank direction (S-weighted difference, no PCA compression)
       - Reconstructs: delta_W = U @ diag(delta_s) @ V.T
       - Applied: hs_new = hs + coeff * delta_W @ x (input-dependent steering)
       - Like PiSSA initialization: matrices pre-scaled by sqrt(S) for proper weighting
       - Works for varying dimensions (e.g., q_proj d_out=2048, k/v_proj d_out=1024)
       
    2. Activation-space bias (legacy PCA): direction is tensor [d_out]
       - Applies constant bias: hs_new = hs + coeff * delta
       - Same steering for all inputs (input-independent)
       - Requires delta.shape[-1] == output.shape[-1]
    
    Why mode 1 (S-weighted):
    - Singular values (S) encode importance of each SVD component
    - Projecting with U*sqrt(S) weights dimensions by their transformation magnitude
    - Full-rank (no PCA) preserves all preference information across r dimensions
    - Matches PiSSA's V@sqrt(S) and sqrt(S)@U decomposition
    - Reconstruction via scaled U, V gives correct magnitudes automatically
    """
    if isinstance(output, tuple):
        y = output[0]
    else:
        y = output
    
    direction = directions[layer]
    
    # Mode 1: S-weighted SVD steering (full-rank with singular value weighting)
    if isinstance(direction, dict):
        # PiSSA-style: U_scaled and V_scaled = original @ sqrt(S) for proper importance weighting
        # delta_W = U_scaled @ diag(delta_s) @ V_scaled.T
        U_scaled = direction['U_scaled']#.to(y.device, y.dtype)  # [d_out, r] = U * sqrt(S)
        delta_s = direction['delta_s']#.to(y.device, y.dtype)  # [r] full-rank direction
        V_scaled = direction['V_scaled']#.to(y.device, y.dtype)  # [d_in, r] = V * sqrt(S)
        
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        
        # Compute delta_W @ x = U_scaled @ diag(delta_s) @ V_scaled.T @ x
        # Efficient: (U_scaled @ diag(delta_s)) @ (V_scaled.T @ x)
        # x: [b s d_in], V_scaled: [d_in r], delta_s: [r], U_scaled: [d_out r]
        Vt_x = einsum(x, V_scaled, '... d_in, d_in r -> ... r')  # V_scaled.T @ x
        scaled = delta_s * Vt_x  # [r] * [... r] -> [... r], scale by steering direction
        delta_hs = einsum(scaled, U_scaled, '... r, d_out r -> ... d_out')  # U_scaled @ scaled
        
        y = y + coeff * delta_hs
    
    # Mode 2: Activation bias (legacy PCA steering)
    else:
        # Sum k directions to single vector (simple linear combination)
        if direction.dim() == 2:
            delta = direction.sum(dim=0)  # (k, d) -> (d,)
        else:
            delta = direction  # Already (d,) for k=1
        
        delta = delta.to(y.dtype).to(y.device)
        
        # Verify dimension match
        if delta.shape[-1] != y.shape[-1]:
            raise RuntimeError(
                f"Steering vector dimension mismatch at layer {layer}: "
                f"delta.shape={delta.shape}, y.shape={y.shape}. "
                f"Expected delta dim {y.shape[-1]}, got {delta.shape[-1]}"
            )
        
        y = y + coeff * delta
    
    if isinstance(output, tuple):
        output = (y,) + output[1:]
    else:
        output = y
    return output



@contextlib.contextmanager
def steer(model: 'PreTrainedModel', vector: "ControlVector", coeff: float, retain_output=False, retain_grad=False, detach=True, **kwargs):
    """
    Apply steering vector(s) to model modules during forward pass via baukit hooks.
    
    Hooks ANY module path in vector.directions.keys(). Common patterns:
    - Layer blocks: "model.layers.12" → edits residual stream after full layer
    - Sub-modules: "model.layers.12.self_attn.o_proj" → edits that projection's output
    - Any named module works (baukit hooks nn.Module instances)
    
    The directions dict maps {module_path: steering_tensor}, allowing:
    - Same vector to all layers: {f"model.layers.{i}": vec for i in range(N)}
    - Different vector per layer: {f"model.layers.{i}": vecs[i] for i in range(N)}
    - Mixed granularity: hook some layers, some sub-modules
    
    Args:
        model: HuggingFace model
        vector: ControlVector with directions dict {module_path: tensor}
        coeff: Steering coefficient (scales the intervention)
        retain_output: Keep hooked outputs in TraceDict
        retain_grad: Keep gradients through hooks
        detach: Detach hooked tensors from graph
    
    Yields:
        TraceDict with hook results (if retain_output=True)
    
    Example:
        # Steer at layer outputs (residual stream)
        cvec = ControlVector(model_type, {"model.layers.10": v, "model.layers.15": v})
        with steer(model, cvec, coeff=2.0):
            out = model.generate(inputs)
        
        # Steer at specific projection outputs
        cvec = ControlVector(model_type, {"model.layers.10.mlp.down_proj": v_mlp})
        with steer(model, cvec, coeff=1.0):
            out = model(inputs)
    """
    layers=list(vector.directions.keys())
    model.directions = vector.directions
    if coeff is None:
        edit_fn = noop_edit
    else:
        edit_fn = functools.partial(
            baukit_dir_add_hook, directions=vector.directions, coeff=coeff
        )
    with TraceDict(
        model, 
        layers=layers,
        retain_output=retain_output,
        retain_grad=retain_grad,
        detach=detach,
        edit_output=edit_fn,
        **kwargs
    ) as td:
        yield td

