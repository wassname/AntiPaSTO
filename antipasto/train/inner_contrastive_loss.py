"""

"""

from __future__ import annotations

import os
from jaxtyping import Float, Int
from torch import Tensor
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from typing import Literal, Optional
from loguru import logger

HS2 = Float[Tensor, "b h"]
HS = Float[Tensor, "b t h"]
Mask = Int[Tensor, "b t 1"]


def mask_agg_tokens(
    x: Float[Tensor, "b t"], attn_mask: Float[Tensor, "b t"],
) -> Float[Tensor, "b"]:
    """Weighted mean of per-token scalars over token dimension."""
    if attn_mask.dim() == 3:
        attn_mask = attn_mask.squeeze(-1)
    weighted = reduce(x * attn_mask, "b t -> b", "sum")
    count = reduce(attn_mask, "b t -> b", "sum").clamp(min=1)
    return weighted / count


def mask_agg_tokens_dim(
    x: Float[Tensor, "b t h"], attn_mask: Float[Tensor, "b t"],
) -> Float[Tensor, "b h"]:
    """Weighted mean of per-token vectors over token dimension."""
    if attn_mask.dim() == 2:
        mask = attn_mask.unsqueeze(-1)
    else:
        mask = attn_mask
    weighted = reduce(x * mask, "b t h -> b h", "sum")
    count = reduce(mask, "b t 1 -> b 1", "sum").clamp(min=1)
    return weighted / count


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * log(1 + |x|).
    
    Compresses large values to log-scale while preserving sign and smoothness at zero.
    Commonly used for signed values that span many orders of magnitude.
    """
    return torch.sign(x) * torch.log1p(x.abs())


def compute_fisher_t(
    diff: Float[Tensor, "b r"], 
    eps: float = 1e-6,
    var_floor_frac: float = 0.1,
    abs_std_floor: float = 0.05,
    detach_std: bool = False,
) -> tuple[Float[Tensor, "r"], dict]:
    """
    Compute signed t-statistic per dimension: mu / sqrt(var).
    
    High |t| = large, consistent separation in that dimension.
    Sign indicates direction of separation (cho > rej or cho < rej).
    
    This is the core of Fisher-based loss: dimensions with high variance
    (inconsistent across samples) get downweighted automatically.
    
    The variance floor prevents t-explosion when variance collapses:
    - var_floor = var_floor_frac * mean(var) ensures relative scaling
    - abs_std_floor provides absolute minimum (for few samples where variance is noisy)
    - Together these cap max |t| to prevent gradient explosion
    
    Args:
        diff: cho-rej difference in projeciton-space [b, r]
        eps: numerical stability for variance
        var_floor_frac: variance floor as fraction of median std (0.1 = 10%)
        abs_std_floor: absolute minimum std (prevents t-explosion with <10 samples)
        detach_std: if True, detach std to prevent zero-variance hacking (legacy)
        
    Returns:
        t: signed t-statistic per dimension [r]
        info: dict with floor_activation_rate (fraction of dims hitting floor)
    """
    # Check for NaNs in input immediately - fail fast to find root cause
    if not torch.isfinite(diff).all():
        n_nan = torch.isnan(diff).sum()
        n_inf = torch.isinf(diff).sum()
        raise ValueError(f"compute_fisher_t received non-finite inputs: {n_nan} NaNs, {n_inf} Infs. "
                         f"Range: [{diff.min():.2e}, {diff.max():.2e}]. "
                         "Likely causes: learning rate too high (exploding grads), or SVD projection issues.")

    # Clamp input only to prevent float32 overflow during squaring, not to hide NaNs
    diff = diff.clamp(-1e4, 1e4)
    
    mu = reduce(diff, 'b r -> r', 'mean')
    
    # Compute standard deviation: std = sqrt(var + eps)
    # CRITICAL: eps INSIDE sqrt to bound gradient at 0. d/dx sqrt(x) = 1/(2*sqrt(x)) → ∞ as x→0
    # sqrt(x).clamp() still has infinite gradient at 0; (x + eps).sqrt() doesn't
    var_raw = reduce((diff - mu.unsqueeze(0)).pow(2), 'b r -> r', 'mean')
    std_raw = (var_raw + eps).sqrt()  # eps inside sqrt, not clamp after
    
    # Std floor: fraction of median std across dims
    # This prevents division by tiny numbers in dimensions that haven't learned anything yet
    std_median = std_raw.median()
    std_floor = max(var_floor_frac * std_median + eps, abs_std_floor)
    std = std_raw.clamp(min=std_floor)
    
    # Track how many dims are hitting the floor (diagnostic for tuning floor params)
    floor_activation_rate = (std_raw < std_floor).float().mean().item()
    
    # Optionally detach std to prevent zero-variance hacking (legacy behavior)
    # With floors in place, detach is less necessary but still an option
    if detach_std:
        std = std.detach()
    
    t = mu / std  # [r]
    
    info = {
        "floor_activation_rate": floor_activation_rate,
        "std_floor": std_floor,
        "std_min": std_raw.min().item(),
        "std_median": std_median.item(),
    }
    return t, info


def compute_fisher_scale(
    diff: Float[Tensor, "b r"],
    eps: float = 1e-6,
    var_floor_frac: float = 0.1,
    abs_std_floor: float = 0.05,
    std: Tensor | None = None,
    detach_std: bool = False,
    return_scaled: bool = False,
) -> tuple[Float[Tensor, "r"], Float[Tensor, "r"], dict]:
    """Compute (mu, std) over batch with the same flooring rules as compute_fisher_t.

    This is used when we want a *shared* per-dimension scale (e.g., std from ref)
    but still want gradients through the *means* of other tensors.

    If std is provided, we use it as the denominator (no recomputation/flooring here)
    and return either (mu, std, info) or (mu/std, std, info) depending on return_scaled.

    Returns:
        mu: mean over batch per dimension [r]
        std: floored std per dimension [r]
        info: diagnostics dict
    """
    if not torch.isfinite(diff).all():
        n_nan = torch.isnan(diff).sum()
        n_inf = torch.isinf(diff).sum()
        raise ValueError(
            "compute_fisher_scale received non-finite inputs: "
            f"{n_nan} NaNs, {n_inf} Infs. Range: [{diff.min():.2e}, {diff.max():.2e}]."
        )

    diff = diff.clamp(-1e4, 1e4)
    mu = reduce(diff, "b r -> r", "mean")

    if std is None:
        var_raw = reduce((diff - mu.unsqueeze(0)).pow(2), "b r -> r", "mean")
        std_raw = (var_raw + eps).sqrt()

        std_median = std_raw.median()
        std_floor = max(var_floor_frac * std_median + eps, abs_std_floor)
        std = std_raw.clamp(min=std_floor)

        floor_activation_rate = (std_raw < std_floor).float().mean().item()
        
        if std_raw.numel() > 0:
            std_min = std_raw.min().item()
            std_median_val = std_median.item()
        else:
            std_min = 0.0
            std_median_val = 0.0

        info = {
            "floor_activation_rate": floor_activation_rate,
            "std_floor": std_floor,
            "std_min": std_min,
            "std_median": std_median_val,
        }
    else:
        info = {}

    if detach_std:
        std = std.detach()

    if return_scaled:
        return mu / std, std, info
    return mu, std, info

# =============================================================================
# COHERENCE LOSS COMPONENTS
# =============================================================================

def _barrier_penalty(
    violation: Float[Tensor, "b t"],
    scale: float,
    v_max: Float[Tensor, "b t"] | float = 0.65,
) -> Float[Tensor, "b t"]:
    """Apply log1p_squared barrier penalty to violation magnitudes.
    
    log1p_squared: log1p(scale * v)² + v — smooth, bounded gradients
    """
    return torch.log1p(scale * violation) ** 2 + violation


def compute_tv_coherence(
    ref_logits: Float[Tensor, "b t v"],
    pi_logits: Float[Tensor, "b t v"],
    mask: Mask,
    threshold_frac: float = 0.3,
    threshold_floor: float = 0.1,
    scale: float = 50.0,
    agg_mode: Literal["mean", "lse", "max"] = "lse",
    lse_temperature: float = 5.0,
) -> tuple[Float[Tensor, "b"], dict]:
    """Total Variation coherence: penalize probability mass redistribution.
    
    TV = 0.5 × Σ|p_ref - p_pi| ∈ [0,1] - fraction of mass moved.
    Threshold = α×√H + β: tight on confident tokens, loose on uncertain.
    
    Why TV over KL:
    - Bounded [0,1], no explosion possible
    - Interpretable: "at most X% of mass can move"
    - Bounds KL, entropy change, any event's prob change
    - Can't be gamed by rare token tricks (linear cost for any mass movement)
    
    Aggregation modes (to prevent reward hacking):
    - lse (default): LogSumExp soft-max. Worst tokens dominate but all get gradients.
    - max: Hard max. Sparse gradients (only worst token).
    - mean: Average. Vulnerable to "one bad token hidden by many good".
    
    Uses log1p_squared barrier: log1p(scale×v)² + v — smooth, bounded gradients
    
    Args:
        threshold_frac: α in threshold = α×√H + β (default 0.3)
        threshold_floor: β in threshold = α×√H + β (default 0.02).
                         Units: probability mass fraction ∈ [0,1], NOT nats.
                         This is the minimum TV allowed before penalty kicks in.
        scale: Penalty multiplier
        agg_mode: How to aggregate per-token penalties. lse recommended.
        lse_temperature: τ for LSE. Lower = closer to max.
    """
    ref_p = ref_logits.softmax(-1)
    pi_p = pi_logits.softmax(-1)
    
    # Total Variation: half L1 distance = fraction of mass moved
    tv_per_token = 0.5 * (ref_p - pi_p).abs().sum(-1)  # [b, t], ∈ [0,1]
    
    # Threshold = α×√H + β: sublinear in entropy (MiLe γ=0.5)
    ref_logp = ref_logits.log_softmax(-1)
    H_ref = -(ref_p * ref_logp).sum(-1) + threshold_floor  # [b, t]
    tv_threshold = threshold_frac * H_ref.detach().sqrt()

    violation = F.relu(tv_per_token - tv_threshold)
    # Per-token max violation since TV ∈ [0,1]: v = max(0, TV - thresh) ≤ 1 - thresh
    v_max = (1.0 - tv_threshold).clamp(min=1e-6)
    penalty = _barrier_penalty(violation, scale, v_max=v_max)
    
    # Aggregate per-token penalties to per-sample loss
    mask_flat = mask.squeeze(-1)  # [b, t]
    if agg_mode == "mean":
        loss = mask_agg_tokens(penalty, mask)
    elif agg_mode == "max":
        # Set masked positions to -inf before max
        loss = (penalty - (~mask_flat.bool()) * 1e9).max(dim=1).values
    elif agg_mode == "lse":
        # LogSumExp: τ × log(mean(exp(penalty/τ)))
        # = τ × (logsumexp(penalty/τ) - log(n_tokens))
        τ = lse_temperature
        n_tokens = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
        # Mask out padding with -inf
        penalty_masked = penalty - (~mask_flat.bool()) * 1e9
        loss = τ * (torch.logsumexp(penalty_masked / τ, dim=1) - torch.log(n_tokens.squeeze()))
    else:
        raise ValueError(f"Unknown agg_mode: {agg_mode}")
    
    # Metrics (TV-based coherence): no KL anywhere.
    metrics = {
        "tv": mask_agg_tokens(tv_per_token, mask).mean().detach(),
        "tv_max": tv_per_token.max(dim=1).values.mean().detach(),
        "tv_thresh_frac": float(threshold_frac),
        "tv_thresh": mask_agg_tokens(tv_threshold, mask).mean().detach(),
        "tv_util_mean": mask_agg_tokens((tv_per_token / tv_threshold.clamp(min=1e-9)), mask).mean().detach(),
        "tv_util_max": ((tv_per_token / tv_threshold.clamp(min=1e-9)) * mask.squeeze(-1)).max(dim=1).values.mean().detach(),
    }
    return loss, metrics


def compute_coherence_loss(
    ref_label_logp: Float[Tensor, "b t"],
    pi_label_logp: Float[Tensor, "b t"],
    mask: Mask,
    scale: float = 50.0,
    ref_logits: Float[Tensor, "b t v"] | None = None,
    pi_logits: Float[Tensor, "b t v"] | None = None,
    coh_thresh_frac: float = 0.3,
    thresh_floor: float = 0.02,
    agg_mode: Literal["mean", "lse", "max"] = "lse",
    lse_temperature: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """TV-based coherence loss with log1p_squared barrier and LSE aggregation.
    
    Args:
        ref_label_logp: Reference log prob of true next token [b, t] (for degradation metric)
        pi_label_logp: Policy log prob of true next token [b, t] (for degradation metric)
        mask: Attention mask [b, t, 1]
        scale: Penalty scaling
        ref_logits: Full reference logits [b, t, vocab]
        pi_logits: Full policy logits [b, t, vocab]
        coh_thresh_frac: TV threshold = α×√H + β (fraction of √entropy)
        thresh_floor: β in TV threshold = α×√H + β (default 0.02)
        agg_mode: Token aggregation (lse recommended: worst tokens dominate)
        lse_temperature: LSE temperature τ (default 5.0). Lower = closer to max.
    
    Returns:
        loss: Per-sample coherence loss [b]
        degradation: Per-token NLL degradation [b, t] (for diagnostics)
        metrics: Dict of diagnostic metrics for logging
    """
    if ref_logits is None or pi_logits is None:
        raise ValueError("compute_coherence_loss requires ref_logits and pi_logits")
    
    degradation = ref_label_logp - pi_label_logp  # For diagnostics
    
    loss, tv_metrics = compute_tv_coherence(
        ref_logits,
        pi_logits,
        mask,
        threshold_frac=coh_thresh_frac,
        threshold_floor=thresh_floor,
        scale=scale,
        agg_mode=agg_mode,
        lse_temperature=lse_temperature,
    )
    
    return loss, degradation, tv_metrics


def compute_delta_logp_change(
    pi_cho_label_logp: Float[Tensor, "b t"],
    pi_rej_label_logp: Float[Tensor, "b t"],
    ref_cho_label_logp: Float[Tensor, "b t"],
    ref_rej_label_logp: Float[Tensor, "b t"],
    mask: Mask,
) -> Float[Tensor, "b"]:
    """Compute preference gap change for monotonic ordering constraint.
    
    delta_logp_change = (logp_pi_cho - logp_pi_rej) - (logp_ref_cho - logp_ref_rej)
                      = how much the preference gap changed from baseline
    
    At c=0 (pi=ref), this is zero by construction.
    
    Args:
        pi_cho_label_logp: Policy chosen next-token log probabilities
        pi_rej_label_logp: Policy rejected next-token log probabilities
        ref_cho_label_logp: Reference chosen next-token log probabilities
        ref_rej_label_logp: Reference rejected next-token log probabilities
        mask: Attention mask
    
    Returns:
        delta_logp_change: Per-sample preference gap change (b,)
    """
    pi_gap = mask_agg_tokens(pi_cho_label_logp - pi_rej_label_logp, mask)
    ref_gap = mask_agg_tokens(ref_cho_label_logp - ref_rej_label_logp, mask).detach()
    return pi_gap - ref_gap


def contrastive_steering_loss_with_ref(
    s_ref_cho: HS,
    s_ref_rej: HS,
    s_pos_cho: HS,
    s_pos_rej: HS,
    s_neg_cho: HS,
    s_neg_rej: HS,
    cho_mask: Mask,
    eps=1e-3,
    last_n_tokens: int = None,
    orth_weight: float = 0.01,
    antisym_margin: float = 0.0001,
    focus_softness: float = 0.0,  # How much to soften subspace concentration weighting
    delta_pos_norm_full: Optional[Float[Tensor, "b"]] = None,
    delta_neg_norm_full: Optional[Float[Tensor, "b"]] = None,
    # Fisher normalization params
    fisher_var_floor_frac: float = 0.1,
    fisher_abs_std_floor: float = 0.05,
    fisher_detach_std: bool = False,
    fisher_stats: dict | None = None,
    fisher_stats_key: str | None = None,
    fisher_std_ema_beta: float = 0.1,
):
    """
    Bidirectional antisymmetric separation loss for reversible SVD steering adapters.

    Naming: s_<pass>_<pair> where:
        - s_ = projection-space (projected via U, scaled by 1/sqrt(S))
        - pass = ref (α=0) | pos (α=+1) | neg (α=-1)  
        - pair = cho (chosen) | rej (rejected)

    The adapter forward pass: y = x @ V @ diag(S) @ U.T + x @ W_residual
    
    Before calling this, inputs are projected to projection-space:
        y_adapter = y - x @ W_residual
        s = y_adapter @ U.detach() / diag(sqrt(S.detach()))
    
    We measure antisymmetry as delta_pos · delta_neg < 0 where deltas are from
    reference. This enforces ref is BETWEEN pos and neg in activation space.
    
    Uses standard geometric dot product: dot = ||pos|| × ||neg|| × cos(θ)
    
    Coherence constraint is computed separately via compute_coherence_loss().
    
    Args:
        s_ref_cho, s_ref_rej: Reference (α=0) in projection-space [b, t, r]
        s_pos_cho, s_pos_rej: Policy at α=+1 in projection-space [b, t, r]
        s_neg_cho, s_neg_rej: Policy at α=-1 in projection-space [b, t, r]
        cho_mask: Attention mask (b, t)
        last_n_tokens: Focus loss on final N tokens (where steering signal concentrates)
        orth_weight: Scaling factor for orthogonal penalty (0.0 = disabled)
        
    Returns:
        dict: {loss_proj, dot_delta, dot_ref, cos_delta, separation_norm, loss_orth (if enabled), fisher_mean}
    """

    hs_mask = cho_mask.clone()
    
    # Focus on last N tokens where steering signal is strongest
    if last_n_tokens is not None:
        seq_lengths = hs_mask.sum(dim=1)  # (b,)
        for i in range(hs_mask.shape[0]):
            if seq_lengths[i] > last_n_tokens:
                hs_mask[i, :-last_n_tokens] = 0
    
    # Compute separation vectors (all in projection-space)
    diff_ref = s_ref_cho - s_ref_rej      # [b, t, r] - baseline separation
    diff_pos = s_pos_cho - s_pos_rej      # [b, t, r] - separation at α=+1
    diff_neg = s_neg_cho - s_neg_rej      # [b, t, r] - separation at α=-1
    
    # Aggregate over tokens (attention-weighted mean)
    diff_ref_agg = mask_agg_tokens_dim(diff_ref, hs_mask)  # [b, r]
    diff_pos_agg = mask_agg_tokens_dim(diff_pos, hs_mask)  # [b, r]
    diff_neg_agg = mask_agg_tokens_dim(diff_neg, hs_mask)  # [b, r]
    
    # Compute deltas from reference: how much did each coefficient move from baseline
    delta_pos = diff_pos - diff_ref   # [b, t, r] - change from baseline at α=+1
    delta_neg = diff_neg - diff_ref   # [b, t, r] - change from baseline at α=-1
    antisym_pos_agg = mask_agg_tokens_dim(delta_pos, hs_mask)  # [b, r]
    antisym_neg_agg = mask_agg_tokens_dim(delta_neg, hs_mask)  # [b, r]
    
    # === Fisher t-space: normalize by std to focus on reliable dimensions ===
    # Use std computed from *reference* for pos/neg/ref, so we live in one geometry.
    # EMA on std_ref reduces noise when batch is small.
    fisher_info = {}
    b = diff_pos_agg.shape[0]
    
    if fisher_stats is not None and fisher_stats_key is None:
        raise ValueError("fisher_stats_key must be provided when fisher_stats is not None")

    fisher_scale_kwargs = dict(
        var_floor_frac=fisher_var_floor_frac,
        abs_std_floor=fisher_abs_std_floor,
    )

    # Compute batch std from ref (with floors), then optionally EMA it.
    _mu_ref, std_ref_batch, info_ref = compute_fisher_scale(diff_ref_agg, **fisher_scale_kwargs)
    std_ref = std_ref_batch

    if fisher_stats is not None:
        ema_key = f"fisher_std_ema/{fisher_stats_key}"
        std_ref_detached = std_ref.detach()
        if ema_key in fisher_stats:
            fisher_stats[ema_key] = (1 - fisher_std_ema_beta) * fisher_stats[ema_key] + fisher_std_ema_beta * std_ref_detached
        else:
            fisher_stats[ema_key] = std_ref_detached
        std_ref = fisher_stats[ema_key].to(device=std_ref.device, dtype=std_ref.dtype)

    # Build Fisher-like vectors using a shared std_ref
    v_ref, _, _ = compute_fisher_scale(diff_ref_agg, std=std_ref, detach_std=fisher_detach_std, return_scaled=True)
    v_pos, _, _ = compute_fisher_scale(antisym_pos_agg, std=std_ref, detach_std=fisher_detach_std, return_scaled=True)
    v_neg, _, _ = compute_fisher_scale(antisym_neg_agg, std=std_ref, detach_std=fisher_detach_std, return_scaled=True)

    fisher_info = {
        "fisher_floor_rate": info_ref["floor_activation_rate"],
        "fisher_std_floor": info_ref["std_floor"],
        "fisher_std_min": info_ref["std_min"],
    }

    # Compute raw dot product and cosine of delta vectors
    dot_delta = (v_pos * v_neg).sum().expand(b)  # δ+ · δ-, want negative (antisymmetric)
    dot_ref = (v_ref * v_ref).sum().expand(b)
    cos_delta = F.cosine_similarity(v_pos, v_neg, dim=-1).expand(b)  # cos(δ+, δ-), want -1
    mag_pos = v_pos.norm(p=2).expand(b)
    mag_neg = v_neg.norm(p=2).expand(b)
    separation_norm = v_pos.norm(p=2)
    
    # Orthogonal penalty: penalize energy not in shared antiparallel axis
    # Uses v_pos/v_neg (already in Fisher t-space)
    # Normalized by dot_ref to be dimensionless and scale-invariant with rank r.
    if orth_weight > 0:
        mag_sq_pos = mag_pos * mag_pos
        mag_sq_neg = mag_neg * mag_neg

        orth_waste_sq = ((mag_sq_pos + mag_sq_neg) - 2 * dot_delta.abs()).clamp(min=0)
        
        # Normalize by dot_ref to make dimensionless (comparable to symlog proj_diff)
        # dot_ref = ||t_ref||² which scales with rank, so this removes rank dependence
        orth_ratio = orth_waste_sq / dot_ref.clamp(min=1.0)
        
        # sqrt(ratio) gives scale-free penalty; eps inside sqrt for gradient stability at 0
        loss_orth = (orth_ratio + 1e-6).sqrt() * orth_weight
    else:
        loss_orth = torch.zeros_like(dot_delta)
        orth_waste_sq = None
    
    # === Unified self-calibrating antisymmetry loss ===
    # Dimensionless: (δ+ · δ-) / ||d_ref||² 
    #   - Negative = straddling (good): δ+ and δ- point opposite from ref
    #   - Positive = same-side (bad): both coefs moved same direction
    # Self-calibrating: normalized by ||ref||² so comparable across model/rank/layer
    #
    # IMPORTANT: Use TOTAL ref norm, not per-dim. Per-dim normalization amplifies
    # noise in dims where ref is small (e.g., δ+*δ-=+5, ref²=0.01 → per_dim=+500).
    # At init, adapter is near-identity so both deltas are small noise in same direction.
    # Total norm keeps loss proportional to raw dot (which correctly shows straddling).
    #
    # Linear + quadratic loss with symlog compression:
    #   shifted = per_dim + margin → shifted < 0 is good (past margin)
    #   proj_raw = shifted + relu(shifted)² → linear push, quadratic penalty on bad
    #   loss = symlog(proj_raw) → O(1/x) gradient decay, prevents runaway
    
    # === Normalization for antisymmetry (delta_full mode) ===
    # Cosine-like normalization: numerator in subspace, denominator in full space
    # This naturally penalizes energy outside subspace: it increases denominator
    # without contributing to numerator, diluting the antisymmetry signal.
    ref_norm_sq = (diff_ref_agg.pow(2)).sum(dim=-1, keepdim=True).clamp(min=eps)  # [b, 1] - for diagnostics
    if delta_pos_norm_full is not None and delta_neg_norm_full is not None:
        norm_product = (delta_pos_norm_full * delta_neg_norm_full).unsqueeze(-1).clamp(min=eps)  # [b, 1]
        antisym_norm_sq = norm_product
    else:
        # Fallback: normalize by projected ref norm
        antisym_norm_sq = ref_norm_sq
    
    # === Antisymmetry formulation: ALIGN mode ===
    # cos(delta_pos, ref) × cos(delta_neg, ref) < 0 means one aligns, one anti-aligns
    # with the reference direction. This constrains steering to the ref axis.
    
    # Compute vector-level cosines using Fisher-weighted vectors (t-statistics)
    # This normalizes by std_ref, making dimensions with high variance less influential
    cos_pos_ref = F.cosine_similarity(v_pos, v_ref, dim=-1)  # [b] - Fisher-weighted
    cos_neg_ref = F.cosine_similarity(v_neg, v_ref, dim=-1)  # [b] - Fisher-weighted

    # Make alignment concentration-aware: weight each cosine by how much of the
    # full-space delta energy lies in the loss subspace.
    # This yields: (axis alignment) × (subspace concentration)
    cos_pos_ref_used = cos_pos_ref
    cos_neg_ref_used = cos_neg_ref
    focus_pos = None
    focus_neg = None
    focus_pos_raw = None
    focus_neg_raw = None
    if delta_pos_norm_full is not None and delta_neg_norm_full is not None:
        proj_norm_pos = antisym_pos_agg.norm(dim=-1)  # [b]
        proj_norm_neg = antisym_neg_agg.norm(dim=-1)  # [b]
        focus_pos_raw = proj_norm_pos / delta_pos_norm_full.clamp(min=eps)
        focus_neg_raw = proj_norm_neg / delta_neg_norm_full.clamp(min=eps)
        # Soften: focus^(1-softness). softness=0→raw, 0.5→sqrt, 1→ignore.
        if focus_softness > 0:
            focus_pos = focus_pos_raw.pow(1.0 - focus_softness)
            focus_neg = focus_neg_raw.pow(1.0 - focus_softness)
        else:
            focus_pos = focus_pos_raw
            focus_neg = focus_neg_raw
        cos_pos_ref_used = cos_pos_ref * focus_pos
        cos_neg_ref_used = cos_neg_ref * focus_neg

    # Products in [-1, 1]: negative = one aligns, one anti-aligns (good)
    cos_product = cos_pos_ref * cos_neg_ref  # [b] (raw, projected)
    cos_product_used = cos_pos_ref_used * cos_neg_ref_used  # [b] (used in loss)
    
    # Scale to bounded range [-30, 30] regardless of rank
    # cos_product_used ∈ [-1, 1], scaled gives consistent gradient magnitude
    PROJ_SCALE = 30.0
    r = antisym_pos_agg.shape[-1]  # Keep for diagnostics
    shifted = cos_product_used * PROJ_SCALE + antisym_margin  # [b], ∈ [-30, 30]

    # Linear + quadratic: linear keeps pushing, quadratic penalizes positive (bad)
    # symlog compresses to prevent runaway: proj_raw ∈ [-30, ~90] → loss ∈ [-3.4, 4.5]
    proj_raw = shifted + F.relu(shifted).pow(2)  # [b]
    loss_proj = symlog(proj_raw) + loss_orth  # [b]
    
    # For diagnostics: fake per-dim tensor to keep logging API consistent
    per_dim_antisym = cos_product_used.unsqueeze(-1).expand(-1, r)  # [b, r]

    assert torch.isfinite(loss_proj).all(), f"Non-finite projection loss {loss_proj}"

    result = {
        "loss_proj": loss_proj,
        "dot_delta": dot_delta.mean(),  # δ+ · δ-, want large negative
        "dot_ref": dot_ref.mean(),
        "cos_delta": cos_delta.mean(),  # cos(δ+, δ-), want -1
        # separation_norm should respect the same token masking as the loss.
        # We report the norm of the aggregated separation vector.
        "separation_norm": separation_norm,
        "mag_plus": mag_pos.mean(),  # Magnitude at α=+1
        "mag_minus": mag_neg.mean(),  # Magnitude at α=-1
        "mag_ratio": (torch.minimum(mag_pos, mag_neg) / (torch.maximum(mag_pos, mag_neg) + eps)).mean(),  # min/max, want close to 1
    }

    # Alignment diagnostics
    result["cos_pos_ref_mean"] = cos_pos_ref.mean()
    result["cos_neg_ref_mean"] = cos_neg_ref.mean()
    result["cos_product_mean"] = cos_product.mean()

    # Subspace focus weighting diagnostics (how much delta energy is in loss subspace)
    if delta_pos_norm_full is not None and delta_neg_norm_full is not None:
        assert focus_pos is not None and focus_neg is not None and focus_pos_raw is not None
        result["focus_pos_mean"] = focus_pos.mean()  # Softened if focus_softness > 0
        result["focus_neg_mean"] = focus_neg.mean()
        if focus_softness > 0:
            result["focus_pos_raw_mean"] = focus_pos_raw.mean()
            result["focus_neg_raw_mean"] = focus_neg_raw.mean()
        result["cos_pos_ref_used_mean"] = cos_pos_ref_used.mean()
        result["cos_neg_ref_used_mean"] = cos_neg_ref_used.mean()
        result["cos_product_used_mean"] = cos_product_used.mean()
    
    if orth_weight > 0:
        result["loss_orth"] = loss_orth.mean()
        result["orth_waste_sq"] = orth_waste_sq.mean()
        result["orth_ratio"] = orth_ratio.mean()  # Normalized metric for comparison
    
    result["antisym_separation_ratio"] = (-dot_delta / dot_ref.abs().clamp(min=0.1)).mean()
    result.update(fisher_info)  # Add floor diagnostics
    
    # Loss component metrics (shifted is now [b], shifted_diag is [b, r] for legacy diagnostics)
    past_margin = (shifted < 0).float().mean()  # Fraction of batch past margin (good)
    quad_penalty = F.relu(shifted).pow(2)  # [b] - quadratic penalty on bad samples
    result["straddle_frac"] = past_margin.item()  # Want high (all samples past margin)
    result["antisym_mean"] = per_dim_antisym.mean().item()  # Avg antisymmetry (want << 0)
    result["shifted_mean"] = shifted.mean().item()  # Mean shifted value (want negative)
    result["proj_raw"] = proj_raw.mean().item()  # Pre-symlog loss (want negative)
    result["quad_penalty"] = quad_penalty.mean().item()  # Quadratic penalty on bad dims (want ~0)
    result["antisym_margin"] = antisym_margin  # The margin used
    result["ref_norm_sq_mean"] = ref_norm_sq.mean().item()  # Mean ||ref||² (for margin calibration)
    
    # Subspace concentration diagnostics
    if delta_pos_norm_full is not None and delta_neg_norm_full is not None:
        # Ratio of projected energy to full-space energy
        proj_norm_pos = antisym_pos_agg.norm(dim=-1)  # [b]
        proj_norm_neg = antisym_neg_agg.norm(dim=-1)  # [b]
        subspace_ratio_pos = (proj_norm_pos / delta_pos_norm_full.clamp(min=eps)).mean()
        subspace_ratio_neg = (proj_norm_neg / delta_neg_norm_full.clamp(min=eps)).mean()
        result["subspace_ratio_pos"] = subspace_ratio_pos.item()  # Want close to 1
        result["subspace_ratio_neg"] = subspace_ratio_neg.item()  # Want close to 1

        # If this is non-zero, we're living in the eps clamp regime and gradients can get sharp.
        norm_prod = delta_pos_norm_full * delta_neg_norm_full  # [b]
        result["delta_full_norm_prod_min"] = norm_prod.min().item()
        result["delta_full_norm_prod_mean"] = norm_prod.mean().item()
        result["delta_full_norm_prod_clamp_frac"] = (norm_prod < eps).float().mean().item()
        result["delta_pos_norm_full_min"] = delta_pos_norm_full.min().item()
        result["delta_neg_norm_full_min"] = delta_neg_norm_full.min().item()
    
    return result


def monotonic_ordering_loss(
    delta_logp_neg: Float[Tensor, "b"],  # Change in preference gap at c=-1
    delta_logp_pos: Float[Tensor, "b"],  # Change at c=+1
    H_ref: Float[Tensor, "b"],  # Reference entropy per sample (for stable normalization)
    threshold_frac: float = 0.2,
    threshold_floor: float = 0.02,
    scale: float = 10.0,
):
    """
    Enforce monotonic ordering across coefficient sweep.
    
    Takes min(violation_forward, violation_backward) at batch level so all samples
    in a batch use the same direction. Network naturally converges to one direction
    because that minimizes loss.
    
    Entropy-based threshold: threshold = threshold_frac × √H_ref + threshold_floor.
    This is the MINIMUM separation required from zero.
    Self-calibrating across tasks (like coherence TV threshold).
    
    Constraint: delta_neg < -threshold < 0 < +threshold < delta_pos (or reversed)
    
    Where delta_logp = (logp_pi_cho - logp_pi_rej) - (logp_ref_cho - logp_ref_rej)
                     = how much the preference gap changed from baseline
    
    At c=0 (no steering), delta_logp=0 by construction (implicit, not passed).
    
    Args:
        delta_logp_neg: Preference gap change at c=-1 (b,)
        delta_logp_pos: Preference gap change at c=+1 (b,)
        H_ref: Reference entropy per sample [b] in nats (mean of per-token entropies).
               Note: coherence uses per-token H [b,t]; here we use per-sample since
               delta_logp is already aggregated per-sample.
        threshold_frac: Fraction of √H_ref for threshold (default 0.2, gives ~0.4 nats at H=4)
        threshold_floor: Minimum threshold in nats (default 0.02, prevents div-by-zero on H→0)
        scale: Multiplier for loss magnitude
    
    Returns:
        loss: Scaled barrier loss
        info: Dict with violation fraction
    """
    # Entropy-based threshold: minimum separation required from zero
    # threshold = frac × √H + floor. With H=4, frac=0.2, floor=0.02: threshold ≈ 0.42 nats
    threshold_per_sample = threshold_frac * H_ref.detach().sqrt().abs() + threshold_floor
    
    # Compute per-direction violation components (compute once, reuse)
    # Forward: neg < -threshold < 0 < +threshold < pos
    viol_neg_fwd = F.relu(delta_logp_neg + threshold_per_sample)  # neg should be < -threshold
    viol_pos_fwd = F.relu(threshold_per_sample - delta_logp_pos)  # pos should be > +threshold
    # Backward: pos < -threshold < 0 < +threshold < neg  
    viol_neg_bwd = F.relu(threshold_per_sample - delta_logp_neg)  # neg should be > +threshold
    viol_pos_bwd = F.relu(delta_logp_pos + threshold_per_sample)  # pos should be < -threshold
    
    violation_forward = viol_neg_fwd + viol_pos_fwd
    violation_backward = viol_neg_bwd + viol_pos_bwd
    
    # Linear barrier penalty
    penalty_forward = scale * violation_forward
    penalty_backward = scale * violation_backward
    
    penalty_fwd_mean = penalty_forward.mean()
    penalty_bwd_mean = penalty_backward.mean()
    
    # Pick whichever direction has lower violation for the whole batch
    # Network will naturally converge to one direction as that minimizes loss
    use_forward = penalty_fwd_mean < penalty_bwd_mean
    
    # Reuse precomputed components
    if use_forward:
        loss = penalty_fwd_mean
        violation_neg = viol_neg_fwd
        violation_pos = viol_pos_fwd
    else:
        loss = penalty_bwd_mean
        violation_neg = viol_neg_bwd
        violation_pos = viol_pos_bwd
    
    # Symmetry penalty: penalize |delta_pos| << |delta_neg| or vice versa
    # Prevents one-sided steering (e.g., strong at -1, weak at +1)
    # ratio = min/max in [0,1], asymmetry = 1 - ratio in [0,1]
    mag_pos = delta_logp_pos.abs()
    mag_neg = delta_logp_neg.abs()
    mag_min = torch.minimum(mag_pos, mag_neg)
    mag_max = torch.maximum(mag_pos, mag_neg)
    asymmetry = 1.0 - mag_min / (mag_max + 1e-6)  # 0 = symmetric, 1 = one-sided
    loss = loss 
    
    # Diagnostics (report raw violations, not penalties)
    total_violation = violation_neg + violation_pos
    util_ratio = total_violation / threshold_per_sample.clamp(min=1e-9)
    
    # Report raw violation means for diagnostics (not penalized)
    viol_fwd_raw = violation_forward.mean().item()
    viol_bwd_raw = violation_backward.mean().item()
    
    info = {
        "frac_violated": ((violation_neg > 0) | (violation_pos > 0)).float().mean().item(),
        "violation_pos": violation_pos.mean().item(),
        "violation_neg": violation_neg.mean().item(),
        "util_mean": util_ratio.mean().item(),
        "util_max": util_ratio.max().item(),
        "monotonic_direction": 1 if use_forward else -1,
        "viol_fwd": viol_fwd_raw,
        "viol_bwd": viol_bwd_raw,
        "threshold_frac": float(threshold_frac),
        "threshold_floor": float(threshold_floor),
        "threshold_mean": threshold_per_sample.mean().item(),
        "threshold_median": threshold_per_sample.median().item(),
        "threshold_min": threshold_per_sample.min().item(),
        "threshold_max": threshold_per_sample.max().item(),
        "H_ref_mean": H_ref.mean().item(),
        "H_ref_median": H_ref.median().item(),
        "delta_logp_pos_mean": delta_logp_pos.mean().item(),
        "delta_logp_neg_mean": delta_logp_neg.mean().item(),
        "delta_logp_pos_median": delta_logp_pos.median().item(),
        "delta_logp_neg_median": delta_logp_neg.median().item(),
        "asymmetry_mean": asymmetry.mean().item(),  # 0 = symmetric, 1 = one-sided
        "mag_ratio": (mag_min / (mag_max + 1e-6)).mean().item(),  # min/max, want ~1
    }
    
    return loss, info


def combine_dual_coef_losses(
    loss_pos: dict,
    loss_neg: dict,
    H_ref: torch.Tensor,
    mono_threshold_frac: float = 0.2,
    mono_threshold_floor: float = 0.02,
    monotonic_scaling: float = 10.0,
    enable_monotonic: bool = True,
    enable_coherence: bool = True,
):
    """Combine losses from both coefficient directions (+1 and -1).
    
    Applies:
    1. Projection loss from both coefficients (already flipped per-layer in train_adapter.py)
    2. Coherence losses (if enabled) - prevents NLL gaming per coefficient
    3. Monotonic ordering constraint (if enabled) - enforces reversibility
    
    Note: Per-layer anti-alignment flipping is handled upstream in compute_batch_loss().
    This function just combines the already-flipped losses.
    
    Monotonic ordering (if enabled):
    - Enforces: delta_logp(c=-1) < 0 < delta_logp(c=+1)
    - delta_logp = preference_gap(policy) - preference_gap(reference)
    - At c=0 (no steering), delta_logp=0 by construction
    - This constraint prevents both coefficients from becoming saddle points (both degrading outputs)
    - See monotonic_ordering_loss() for details on hinge penalty structure
    
    Args:
        loss_pos: Loss dict from coef=+1 (contains loss_proj, loss_coh, delta_logp_change)
        loss_neg: Loss dict from coef=-1 (contains loss_proj, loss_coh, delta_logp_change)
        monotonic_margin: Hinge margin for ordering constraint (nats)
        monotonic_scaling: Scale factor for monotonic loss
        enable_monotonic: Whether to apply monotonic ordering constraint
        enable_coherence: Whether to include coherence losses
        
    Returns:
        total_loss: Combined scalar loss for backprop
        losses: Dict with individual loss components
        meta_pos: Dict with metrics for coef=+1 (mono_violation)
        meta_neg: Dict with metrics for coef=-1 (mono_violation)
        meta_shared: Dict with global metrics (loss_monotonic, mono_frac_violated)
    """
    # Per-layer flipping already handled in train_adapter.py:
    # During forward pass, we flip per-layer based on pref_dir alignment.
    # This function just combines the already-flipped losses - no global flip needed.
    #
    # loss_proj is SHARED (antisymmetric loss already combines both coefs).
    # Don't double-count - use loss_proj from either coef (they're identical).
    # Coherence losses ARE separate per coef (each needs own NLL stability guarantee).
    loss_proj_bidirectional = loss_pos["loss_proj"]  # Antisymmetric: same for both coefs
    
    # Combine projection + coherence (no adaptive weighting):
    # - loss_proj_bidirectional: antisymmetric loss (shared, computed once)
    # - loss_pos["loss_coh"], loss_neg["loss_coh"]: per-coef coherence barriers
    #   (both must satisfy coherence; each prevents its own NLL gaming pathway)
    if enable_coherence:
        total = (
            loss_proj_bidirectional + 
            loss_pos["loss_coh"] +      # Prevent coef=+1 from gaming coherence
            loss_neg["loss_coh"]        # Prevent coef=-1 from gaming coherence
        ).mean()
    else:
        total = loss_proj_bidirectional.mean()
    
    # Build metadata dicts
    meta_pos = {}
    meta_neg = {}
    meta_shared = {}
        
    # Optional: Add monotonic ordering constraint (enforces reversibility):
    # - delta_logp_change = policy_preference_gap - reference_preference_gap
    # - At c=0 (no adapter), delta_logp_change=0 by definition
    # - At c=-1, want delta_logp_change < 0 (gap shrinks or reverses)
    # - At c=+1, want delta_logp_change > 0 (gap widens in same direction)
    # - This prevents both from becoming bad (e.g., both increasing NLL via different mechanisms)
    if enable_monotonic:
        delta_logp_neg = loss_neg["delta_logp_change"]
        delta_logp_pos = loss_pos["delta_logp_change"]
        
        loss_monotonic, mono_info = monotonic_ordering_loss(
            delta_logp_neg, delta_logp_pos,
            H_ref=H_ref, threshold_frac=mono_threshold_frac, threshold_floor=mono_threshold_floor,
            scale=monotonic_scaling,
        )
        
        total = total + loss_monotonic
        
        # Monotonic metrics: shared loss value, per-direction violations
        meta_shared["loss_monotonic"] = loss_monotonic.item()
        meta_shared["mono_frac_violated"] = mono_info["frac_violated"]
        meta_shared["mono_direction"] = mono_info["monotonic_direction"]
        meta_shared["mono_util_mean"] = mono_info["util_mean"]  # budget utilization
        meta_shared["mono_util_max"] = mono_info["util_max"]    # worst-case util
        meta_shared["mono_viol_fwd"] = mono_info["viol_fwd"]
        meta_shared["mono_viol_bwd"] = mono_info["viol_bwd"]
        for k, v in mono_info.items():
            if k in {"frac_violated", "violation_pos", "violation_neg", "monotonic_direction", "util_mean", "util_max", "viol_fwd", "viol_bwd"}:
                continue
            meta_shared[f"mono_{k}"] = v
        meta_pos["mono_violation"] = mono_info["violation_pos"]
        meta_neg["mono_violation"] = mono_info["violation_neg"]
    else:
        # Set to 0 instead of None to prevent NaN in aggregation
        meta_shared["loss_monotonic"] = 0.0
        meta_shared["mono_frac_violated"] = 0.0
        meta_shared["mono_direction"] = 0
        meta_pos["mono_violation"] = 0.0
        meta_neg["mono_violation"] = 0.0
    
    meta_shared['loss_total'] = total.item()

    losses = {
        'proj_pos': loss_pos["loss_proj"],
        'proj_neg': loss_neg["loss_proj"],
        'coh_pos': loss_pos["loss_coh"],
        'coh_neg': loss_neg["loss_coh"],
        'mono': loss_monotonic if enable_monotonic else torch.tensor(0.0),        
    }
    
    return total, losses, meta_pos, meta_neg, meta_shared


