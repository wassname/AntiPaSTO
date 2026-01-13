"""
AntiPaSTO adapter - combines SVFT (Singular Value Fine-Tuning) with changes
https://github.com/wassname/AntiPaSTO
(c) 2026 Michael J Clark, MIT License

SVFT decomposes weights via SVD: W = U @ S @ V^T
- U, V are frozen singular vectors (orthonormal bases)
- S is diagonal singular values (frozen as s0)
- dS is sparse learnable delta to S (controlled by gate)

Changes are
- Only diagonal
- Add a tail instead of discarding tail of singular vector
- learnable decoder U via delta parameterization (U_eff = U_init + U_delta) which allows the model to modify learned direction which increase expressivity
- SVFT modes: replace_add, replace_mul, adapter_add, **adapter_mult**
- bounded singular values for stability, as negative singular values cause issues
- modified SVD equation to stay in low rank space. Instead of `(U @ S @ V^T) @ x`, we do `(x @ V.T) @ S @ U.T`

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from jaxtyping import Float
from einops import repeat, rearrange, reduce
from peft.tuners.tuners_utils import BaseTunerLayer, BaseTuner
from peft.config import PeftConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils import PeftType
from peft.utils.other import get_pattern_key
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit, Int8Params
from typing import Any, Optional, Union, List
import enum
from loguru import logger
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
)

@dataclass
class AntiPaSTOConfig(PeftConfig):
    """
    Configuration for AntiPaSTO adapter with SVDSteering rotations.
    
    SVD-based steering with PiSSA decomposition: W = U @ S @ V^T + W_res
    - Top-r SVD components (U, S, V) for principal directions
    - Residual W_res captures remaining variance
    - SSVD rotations (selective rotation of U/V singular vectors)
    - Learnable singular value scaling (add/mult)
    - OFT block-diagonal structure (parameter efficiency for rotations)
    - but it's a symmetric intervention
    """
    # AntiPaSTO-specific parameters
    r: int = field(default=16, metadata={"help": "SVD rank for principal components"})
    precomputed_indices: Optional[Dict[str, torch.Tensor]] = field(
        default=None,
        repr=False,  # Don't print in __repr__
        metadata={"help": "Dict of {layer_name: indices_tensor} for data-aware dim selection."}
    )
    svd_bases: Optional[Dict[str, torch.Tensor]] = field(
        default=None,
        repr=False,  # Don't print in __repr__
        metadata={"help": "Dict of {layer_name.U/V/S: tensor} for loading saved SVD bases."}
    )
    
    def __post_init__(self):
        self.peft_type = 'APASTOADAPTER'
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
    
    def to_dict(self):
        """Override to exclude non-serializable fields."""
        d = super().to_dict()
        # Remove precomputed_indices from serialization (only for init)
        d.pop('precomputed_indices', None)
        d.pop('svd_bases', None)
        return d
    rotate_u: bool = field(
        default=False,
        metadata={"help": "Learn rotation on U singular vectors (SVDSteering-style)"}
    )
    rotate_v: bool = field(
        default=True,
        metadata={"help": "Learn rotation on V singular vectors (SVDSteering-style)"}
    )
    rotation_method: Literal["matrix_exp", "cayley"] = field(
        default="cayley",
        metadata={"help": "Rotation parameterization: 'cayley' (recommended, exact reversibility) or 'matrix_exp' (exact but slower)"}
    )
    svd_aligned_init: bool = field(
        default=False,
        metadata={"help": "Initialize delta_s proportional to S (normalized). Gives very stable init (std=0.26 across seeds)."}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Steering coefficient for rotations (1.0 = forward, -1.0 = reverse, 0.0 = disabled)"}
    )
    max_rotation_angle: float = field(
        default=torch.pi/3,
        metadata={"help": "Max rotation angle (radians, soft-clamped). Small angles (≤0.3) ensure R(α)@S ≈ -R(-α)@S for output symmetry at α=±1. Set to inf to disable."}
    )
    # steer_s: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to apply steering to singular value scaling"}
    # )
    
    # Standard PEFT parameters
    target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of module names to apply adapter to"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of modules to save (not adapt)"}
    )


class AntiPaSTOLayer(BaseTunerLayer):
    """
    AntiPaSTO layer with SVDSteering-style decomposition.
    
    W = U @ S @ V^T + W_res where:
    - U, V: Top-r singular vectors (can be rotated)
    - S: Top-r singular values (can be scaled via dS)
    - W_res: Residual matrix (frozen)
    """

    adapter_layer_names = ("antipasto_delta_s", "antipasto_rotation_params_u", "antipasto_rotation_params_v")
    other_param_names = ("antipasto_u", "antipasto_v", "antipasto_s", "antipasto_w_res", "antipasto_alpha", "antipasto_r", "antipasto_rotate_u", "antipasto_rotate_v", "antipasto_rotation_method", "antipasto_max_rotation_angle")

    peft_type = "APASTOADAPTER"

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        self.antipasto_r = {}
        self.antipasto_rotate_u = {}
        self.antipasto_rotate_v = {}
        self.antipasto_rotation_method = {}
        self.antipasto_alpha = {}
        self.antipasto_max_rotation_angle = {}
        self.antipasto_svd_aligned_init = {}
        
        # SVD components (per adapter) - simplified naming like SVDSteering
        self.antipasto_u = BufferDict({})  # U: [d_out, r]
        self.antipasto_v = BufferDict({})  # V: [d_in, r]
        self.antipasto_s = BufferDict({})  # S: [r]
        self.antipasto_w_res = BufferDict({})  # W_res: [d_out, d_in]
        
        # Learnable S scaling (DeLoRA-style)
        self.antipasto_delta_s = nn.ParameterDict({})  # add: S + delta_s
        # loglambda_s removed - only add2 mode supported
        
        # Rotation parameters (SVDSteering-style)
        self.antipasto_rotation_params_u = nn.ParameterDict({})
        self.antipasto_rotation_params_v = nn.ParameterDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False

        # Marker for Coconut to find Bi layers
        self._recursion_cache = None

        self._active_adapter = None

    def update_layer(
        self,
        adapter_name: str,
        alpha,
        r,
        rotate_u,
        rotate_v,
        rotation_method,
        max_rotation_angle,
        svd_aligned_init: bool = False,
        precomputed_indices: Optional[Dict[str, torch.Tensor]] = None,
        svd_bases: Optional[Dict[str, torch.Tensor]] = None,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initialize adapter with simple top-r SVD + residual (PiSSA-style).
        
        If svd_bases provided, uses those U/V/S directly (for exact reload).
        Elif precomputed_indices provided, uses those dim indices.
        Otherwise falls back to naive top-r by singular value (default PiSSA).
        """
        if adapter_name in self.antipasto_u:
            return  # Already initialized

        self.layer_name = layer_name or "unknown_layer"

        self.antipasto_alpha[adapter_name] = float(alpha)
        self.antipasto_r[adapter_name] = r
        self.antipasto_rotate_u[adapter_name] = rotate_u
        self.antipasto_rotate_v[adapter_name] = rotate_v
        self.antipasto_rotation_method[adapter_name] = rotation_method
        self.antipasto_max_rotation_angle[adapter_name] = max_rotation_angle
        self.antipasto_svd_aligned_init[adapter_name] = svd_aligned_init

        # Get base weight
        base_weight = self.get_base_layer().weight
        
        # Dequantize if needed
        if isinstance(base_weight, Params4bit):
            base_weight = bnb.functional.dequantize_4bit(base_weight.data, base_weight.quant_state)
        elif isinstance(base_weight, Int8Params):
            base_weight = bnb.functional.dequantize_8bit(base_weight.data, base_weight.quant_state)
        
        base_weight = base_weight.float()  # [out, in]
        device = base_weight.device

        # Check for saved SVD bases (exact reload)
        # Key format in saved file: svd_bases.{module_path_with_underscores}.u
        # layer_name format: model.layers.1.self_attn.o_proj
        # We need to find matching key by converting layer_name to underscored format
        svd_key_base = None
        if svd_bases is not None:
            layer_key = layer_name.replace('.', '_')
            # Try to find matching key (handle different prefix possibilities)
            for candidate_prefix in ['base_model_model_', 'base_model_', '']:
                candidate_key = f"svd_bases.{candidate_prefix}{layer_key}.u"
                if candidate_key in svd_bases:
                    svd_key_base = f"svd_bases.{candidate_prefix}{layer_key}"
                    break
        
        if svd_key_base is not None:
            U = svd_bases[f"{svd_key_base}.u"].to(device)
            V = svd_bases[f"{svd_key_base}.v"].to(device)
            S = svd_bases[f"{svd_key_base}.s"].to(device)
            r_actual = S.shape[0]
            
            # Compute residual from saved bases
            Vh = V.T
            W_principal = U @ torch.diag(S) @ Vh
            W_res = base_weight - W_principal
            
            logger.debug(f"Loaded SVD bases: layer={layer_name}, r={r_actual}")
        else:
            # Compute SVD from base weight
            U_full, S_full, Vh_full = torch.linalg.svd(base_weight, full_matrices=False)
            max_rank = min(U_full.shape[1], S_full.shape[0])
            r_actual = min(r, max_rank)
            
            # Dimension selection: precomputed_indices (data-aware) or top-r (default PiSSA)
            if precomputed_indices is not None and layer_name in precomputed_indices:
                indices = precomputed_indices[layer_name].to(device)
                r_actual = min(len(indices), r_actual)
                indices = indices[:r_actual]
                
                U = U_full[:, indices]
                Vh = Vh_full[indices, :]
                V = Vh.T
                S = S_full[indices]
                
                logger.debug(f"Precomputed indices init: layer={layer_name}, {len(indices)} dims")
            else:
                # Naive top-r by singular values (original PiSSA)
                U = U_full[:, :r_actual]
                S = S_full[:r_actual]
                Vh = Vh_full[:r_actual, :]
                V = Vh.T
            
            # Compute residual (PiSSA-style)
            W_principal = U @ torch.diag(S) @ Vh
            W_res = base_weight - W_principal

        logger.debug(f"AntiPaSTO Layer Init: {layer_name}, r={r_actual}, norms W={base_weight.norm():.1f}, Wres={W_res.norm():.1f}, Wrank={W_principal.norm():.1f}")
        
        # Store frozen components
        self.antipasto_u[adapter_name] = U.clone().detach().contiguous()
        self.antipasto_v[adapter_name] = V.clone().detach().contiguous()
        self.antipasto_s[adapter_name] = S.clone().detach().contiguous()
        self.antipasto_w_res[adapter_name] = W_res.clone().detach().contiguous()
        
        # Learnable S scaling: S_scaled = S + alpha * delta_s
        self.antipasto_delta_s[adapter_name] = nn.Parameter(
            torch.zeros(r_actual, device=device), 
            requires_grad=True
        )
        if self.antipasto_svd_aligned_init.get(adapter_name, False):
            # SVD-aligned init: delta_s ∝ S (normalized). Very stable across seeds (std=0.26).
            s_normalized = S / S.max()
            self.antipasto_delta_s[adapter_name].data = s_normalized * 4e-4 + 4e-4
        else:
            # Default: small random noise
            nn.init.trunc_normal_(self.antipasto_delta_s[adapter_name], std=4e-4, mean=4e-4)



        def initialize_skew_symmetric_matrix(*args, **kwargs):
            """With contrastive steering coeff=+1 and coeff=-1 produce identical outputs initially, so gradients are zero. Small random init is important for learning as it breaks symmetry."""
            x = torch.zeros(*args, **kwargs)
            # Option B: Draw from skew-symmetric distribution directly
            nn.init.trunc_normal_(x, std=0.003)
            x = x - x.T
            return x
        
        # Initialize rotation parameters (reversible OFT,SSVD-style)
        if rotate_u:
            self.antipasto_rotation_params_u[adapter_name] = nn.Parameter(
                initialize_skew_symmetric_matrix(r_actual, r_actual, device=device)
            )
        
        if rotate_v:
            self.antipasto_rotation_params_v[adapter_name] = nn.Parameter(
                initialize_skew_symmetric_matrix(r_actual, r_actual, device=device)
            )
    def _get_rotation(
        self, 
        params: Float[Tensor, "r r"],
        alpha: float,
        rotation_method: str,
        max_angle: float = 1.0,
    ) -> Float[Tensor, "r r"]:
        """Compute rotation matrix from learnable parameters (SVDSteering-style).
        
        Args:
            params: Rotation parameters (skew-symmetric matrix)
            alpha: Steering coefficient (1.0 = forward, -1.0 = reverse)
            rotation_method: Rotation parameterization method ('cayley' or 'matrix_exp')
            max_angle: Maximum rotation angle in radians (soft constraint)
        
        Returns:
            Orthogonal rotation matrix R ∈ SO(r)
        """
        A = params - params.T  # skew-symmetric projection
        return self._rotation_from_skew(A, alpha, rotation_method, max_angle)
    
    def _rotation_from_skew(
        self,
        A: Float[Tensor, "r r"],
        alpha: float,
        rotation_method: str,
        max_angle: float = 1.0,
    ) -> Float[Tensor, "r r"]:
        """Compute rotation from skew-symmetric matrix with soft angle constraint.
        
        Args:
            A: Skew-symmetric matrix (A = -A.T)
            alpha: Steering coefficient
            rotation_method: 'cayley' (recommended) or 'matrix_exp'
            max_angle: Maximum rotation angle in radians (soft constraint via tanh)
        
        Returns:
            Orthogonal rotation matrix with bounded angle
            
        Rotation methods:
            - cayley: RECOMMENDED. Exact orthogonality, exact reversibility (R(-α) = R(α)^-1),
                     preserves output symmetry Δy(+1) = -Δy(-1). Faster than matrix_exp.
            - matrix_exp: Exact orthogonality and reversibility, but ~3x slower than cayley.
        """
        # Soft clamp rotation angle: small θ ensures R(θ)@S ≈ -R(-θ)@S (first-order approx)
        # This gives additive output symmetry: Δy(+1) ≈ -Δy(-1) around base model

        # if max_angle is not None and max_angle < float('inf'):
        #     A_clamped = max_angle * torch.tanh(A / max_angle)
        # else:
        #     A_clamped = A
        
        if max_angle is not None and max_angle < (torch.pi - 1e-6):
            # Convert desired max rotation angle to A-space limit
            # Inverts: θ = 2 * arctan(limit / 2)
            a_limit = 2 * math.tan(max_angle / 2)
            A_clamped = a_limit * torch.tanh(A / a_limit)
        else:
            A_clamped = A

        assert torch.isfinite(A_clamped).all(), f"Non-finite values in rotation matrix computation on layer {self.layer_name}, from angle {A} and max_angle={max_angle} a_limit={a_limit}"

        if rotation_method == "matrix_exp":
            # Matrix exponential: exp(αA)
            # Exact orthogonality, exact reversibility, always numerically stable
            R = torch.matrix_exp(alpha * A_clamped)
        elif rotation_method == "cayley":
            # Cayley transform: (I - αA/2)^{-1} (I + αA/2)
            # Exact orthogonality, exact reversibility: R(-α) = R(α)^(-1)
            # More efficient than matrix_exp, but can be singular for extreme A
            I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            X = alpha * A_clamped / 2
            try:
                R = torch.linalg.solve(I - X, I + X)
            except (torch._C._LinAlgError, RuntimeError):
                # Fallback to matrix_exp when Cayley is singular
                # This happens with extreme gradients pushing eigenvalues near 1
                R =  torch.matrix_exp(alpha * A_clamped)
        else:
            raise ValueError(f"Unknown rotation method: {rotation_method} (use 'cayley' or 'matrix_exp')")

        assert torch.isfinite(R).all(), "Non-finite values in rotation matrix output"
        return R

    def get_adapted_output(self, x, adapter: str) -> torch.Tensor:
        """
        Compute adapter output (SVDSteering-style).
        
        W_adapted = U_rot @ diag(S_scaled) @ V_rot^T + W_res
        Forward: x @ V_rot @ diag(S_scaled) @ U_rot^T + x @ W_res^T
        
        Note: alpha scales rotations only (steering strength), not S
        """
        alpha = self.antipasto_alpha[adapter]
        
        # BYPASS: When alpha=0, use base_layer to avoid precision drift.
        # The decomposed path (x @ V * S) @ U^T + x @ W_res^T breaks matmul
        # associativity, giving ~0.04-0.08 mean error vs x @ W^T even in float32.
        # This matters for eval baselines; for training at alpha≠0 it's fine.
        if alpha == 0.0:
            return self.base_layer(x)
        # steer_s = self.antipasto_steer_s[adapter]
        
        # Get frozen bases
        U = self.antipasto_u[adapter]  # [d_out, r]
        V = self.antipasto_v[adapter]  # [d_in, r]
        S = self.antipasto_s[adapter]  # [r]
        W_res = self.antipasto_w_res[adapter]  # [d_out, d_in]
        
        # Apply rotations (alpha scales rotation strength, not magnitude)
        max_angle = self.antipasto_max_rotation_angle[adapter]
        
        if self.antipasto_rotate_v[adapter] and adapter in self.antipasto_rotation_params_v:
            R_v = self._get_rotation(
                self.antipasto_rotation_params_v[adapter], 
                alpha=alpha,
                rotation_method=self.antipasto_rotation_method[adapter],
                max_angle=max_angle
            )
            V_rot = V @ R_v  # [d_in, r]
        else:
            V_rot = V
        
        if self.antipasto_rotate_u[adapter] and adapter in self.antipasto_rotation_params_u:
            R_u = self._get_rotation(
                self.antipasto_rotation_params_u[adapter],
                alpha=alpha,
                rotation_method=self.antipasto_rotation_method[adapter],
                max_angle=max_angle
            )
            U_rot = U @ R_u  # [d_out, r]
        else:
            U_rot = U
        
        # Scale S: S_scaled = S + alpha * delta_s
        delta_s = self.antipasto_delta_s[adapter]
        S_scaled = S + alpha * delta_s

        # Match matmul dtypes to the input.
        # Buffers are stored as float32 for numerical stability, but most models run bf16/fp16 on GPU.
        # Casting here avoids dtype mismatch errors and keeps outputs consistent with the base layer.
        compute_dtype = x.dtype
        V_rot = V_rot.to(dtype=compute_dtype)
        U_rot = U_rot.to(dtype=compute_dtype)
        S_scaled = S_scaled.to(dtype=compute_dtype)
        W_res = W_res.to(dtype=compute_dtype)
        
        # Check for NaNs in intermediate tensors
        if not torch.isfinite(V_rot).all():
            raise ValueError(f"NaNs in V_rot for adapter {adapter}. alpha={alpha}, max_angle={max_angle}")
        if not torch.isfinite(U_rot).all():
            raise ValueError(f"NaNs in U_rot for adapter {adapter}. alpha={alpha}, max_angle={max_angle}")
        if not torch.isfinite(S_scaled).all():
            raise ValueError(f"NaNs in S_scaled for adapter {adapter}. scale_mode={scale_mode}")

        # Efficient forward: x @ V_rot @ diag(S_scaled) @ U_rot^T
        x_projected = x @ V_rot  # [..., r]
        x_scaled = x_projected * S_scaled  # [..., r] - broadcast multiply
        x_transformed = x_scaled @ U_rot.T  # [..., d_out]
        
        # Add residual contribution
        x_residual = x @ W_res.T  # [..., d_out]
        
        return x_transformed + x_residual

    def forward(self, x: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
        previous_dtype = x.dtype
        
        assert len(self.active_adapters) <= 1, "AntiPaSTO currently supports only one active adapter at a time."

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if not self.active_adapters:
                return self.base_layer(x, *args, **kwargs).to(previous_dtype)

            # Always compute full adapted weight (no mode switching)
            result = None
            for adapter in self.active_adapters:
                if adapter not in self.antipasto_u:
                    continue

                h = self.get_adapted_output(x, adapter)
                
                if result is None:
                    result = h
                else:
                    result += h  # Multiple adapters (unlikely)
            
            if result is None:
                result = self.base_layer(x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("Merge not implemented for AntiPaSTO yet")

    def unmerge(self) -> None:
        raise NotImplementedError("Unmerge not implemented for AntiPaSTO yet")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "antipasto." + rep


class AntiPaSTOLinear(nn.Module, AntiPaSTOLayer):
    """AntiPaSTO implemented in a dense layer"""
    
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        **kwargs,
    ) -> None:
        super().__init__()
        AntiPaSTOLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, **kwargs)

    def forward(self, hidden_states: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
        return AntiPaSTOLayer.forward(self, hidden_states, *args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "antipasto." + rep


class AntiPaSTOModel(BaseTuner): 
    """
    AntiPaSTO Model - handles adapter injection into base model.
    Inherits from BaseTuner to integrate with PEFT infrastructure.
    """
    prefix: str = "antipasto_"
    tuner_layer_cls = AntiPaSTOLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


    def _create_and_replace(
        self,
        antipasto_config: AntiPaSTOConfig,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key 
        kwargs = {
            "r": antipasto_config.r,
            "task_type": antipasto_config.task_type,
            "target_modules": antipasto_config.target_modules,
            "rotate_u": antipasto_config.rotate_u,
            "rotate_v": antipasto_config.rotate_v,
            "rotation_method": antipasto_config.rotation_method,
            # "block_size": antipasto_config.block_size,
            "alpha": antipasto_config.alpha,
            "max_rotation_angle": antipasto_config.max_rotation_angle,
            "svd_aligned_init": antipasto_config.svd_aligned_init,
            "precomputed_indices": antipasto_config.precomputed_indices,
            "svd_bases": antipasto_config.svd_bases,
            "layer_name": current_key,  # Pass layer name for dim index lookup
            # "data_aware_init_use_magnitudes": antipasto_config.data_aware_init_use_magnitudes,
            # "steer_s": antipasto_config.steer_s,
            **optional_kwargs,
        }

        if isinstance(target, AntiPaSTOLinear):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
    
    @staticmethod
    def _create_new_module(adapter_name, target, **kwargs):
        """Create AntiPaSTOLinear for Linear layers."""
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = AntiPaSTOLinear(
                target, 
                adapter_name, 
                **kwargs
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported for AntiPaSTO. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        return new_module




def register_antipasto_peft():
    """Register custom AntiPaSTO adapter with PEFT (idempotent)."""
    import peft.utils.peft_types
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
    from peft.utils import register_peft_method

    # Check if already registered
    if hasattr(peft.utils.peft_types.PeftType, 'APASTOADAPTER'):
        return  # Already registered

    class PeftType2(str, enum.Enum):
        APASTOADAPTER = "APASTOADAPTER"

    peft.utils.peft_types.PeftType = PeftType2
    PEFT_TYPE_TO_PREFIX_MAPPING[AntiPaSTOConfig.peft_type] = "APASTOADAPTER"
    register_peft_method(
        name="apastoadapter",
        model_cls=AntiPaSTOModel,
        config_cls=AntiPaSTOConfig,
        prefix="antipasto_",
    )
