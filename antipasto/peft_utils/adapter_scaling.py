"""
Adapter steering for contrastive training with proper gradient flow.

For AntiPaSTO: Sets `antipasto_alpha` directly on each layer. The Cayley rotation
transform satisfies R(-α) = R(α)^(-1), so a single adapter handles both steering directions.

Key insight: PyTorch's autograd tracks tensor references in the computation graph,
not module attributes. So we can:
1. Replace `module.weight` with `weight * coeff` (graph stores ref to original param)
2. Run forward pass
3. Restore original Parameter
4. Call backward() - gradients flow through scaled tensor to original param

Confirmed in nbs/scratch_adapter_scaling_gradients.ipynb
"""
import torch.nn as nn
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Tuple

from antipasto.peft_utils.antipasto_adapter import AntiPaSTOLayer


def _effective_coeff(coeff: float, even_frac: float) -> float:
    """Compute effective coefficient with even component.
    
    even_frac=0.0: pure odd (full sign flip), effective_coeff = coeff
    even_frac=0.5: half even/half odd, effective_coeff in {+1.0, 0.0}
    even_frac=1.0: pure even (no sign flip), effective_coeff = 1.0
    
    Formula: effective = even_frac + (1 - even_frac) * coeff
    """
    return even_frac + (1.0 - even_frac) * coeff


def scale_antipasto_params(
    module: AntiPaSTOLayer,
    adapter_name: str,
    coeff: float,
    even_frac: float,
    originals: List[Tuple]
) -> None:
    """Set antipasto_alpha to coeff for bidirectional steering."""
    if hasattr(module, 'antipasto_alpha') and adapter_name in module.antipasto_alpha:
        originals.append((module, 'antipasto_alpha', module.antipasto_alpha))
        eff_coeff = _effective_coeff(coeff, even_frac)
        object.__setattr__(module, 'antipasto_alpha', {
            k: eff_coeff if k == adapter_name else v
            for k, v in module.antipasto_alpha.items()
        })


@contextmanager
def ScaleAdapter(
    model: nn.Module,
    coeff: float = 1.0,
    adapter_name: Optional[str] = None,
    even_frac: float = 0.1,
):
    """Temporarily scale adapter params by coeff for bidirectional steering.
    
    Usage:
        with ScaleAdapter(model, coeff=1.0):   # normal direction
            loss_pos = model(x).sum()
        with ScaleAdapter(model, coeff=-1.0):  # inverted direction  
            loss_neg = model(x).sum()
        (loss_pos - loss_neg).backward()  # grads flow correctly
    
    coeff=None disables adapter entirely.
    
    even_frac: Fraction of the scaling that is even (sign-symmetric).
        0.0 = pure odd (full sign flip): coeff=-1 -> -1.0
        0.5 = half even: coeff=-1 -> 0.0, coeff=+1 -> 1.0
        1.0 = pure even (no steering): coeff=-1 -> 1.0
    """
    if adapter_name is None:
        adapter_name = model.active_adapter
    
    if coeff is None:
        with model.disable_adapter():
            yield
        return
    
    originals = []
    
    try:
        for name, module in model.named_modules():
            if isinstance(module, AntiPaSTOLayer) and adapter_name in module.active_adapters:
                scale_antipasto_params(
                    module=module, adapter_name=adapter_name,
                    coeff=coeff, even_frac=even_frac, originals=originals
                )
        yield
        
    finally:
        for module, attr_name, original_param_dict in originals:
            setattr(module, attr_name, original_param_dict)


def get_scale_adapter_fn(
    model: nn.Module,
    adapter_name: Optional[str] = None,
    even_frac: float = 0.1,
) -> Callable[[float], Any]:
    """Get scaling context manager factory. Returns fn(coeff) -> context_manager.
    
    For AntiPaSTO, sets the alpha coefficient directly in each layer.
    The rotation matrices R(alpha) and R(-alpha) are exact inverses via Cayley transform.
    """
    if adapter_name is None:
        adapter_name = model.active_adapter
    
    return lambda coeff: ScaleAdapter(model, coeff=coeff, adapter_name=adapter_name, even_frac=even_frac)
