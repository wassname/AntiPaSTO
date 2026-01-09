"""Subspace operations for loss projection and layer selection.

Core abstraction: Subspace wraps an orthonormal basis V: [d_model, rank].

Important operator semantics (code meaning, not set theory):

- Projection: x -> x @ V.
- Approx intersection: A & B returns principal-angle shared directions (a symmetric
    "bisector" basis). It is not a strict set-theoretic intersection.
- Orthogonal-complement projection: A - B means A projected into B^perp.
    In symbols: A_perp_B := Π_{B^⊥}(A).

Naming conventions for subspace functions:
- `_x_` means intersection (∩), e.g., `write_x_notlogits` = write ∩ (logits^⊥)
- `_not_` or `not` prefix means complement (^⊥), e.g., `notlogits` = logits^⊥
- Operator precedence: `taskdiff_x_write_x_notlogits` = taskdiff ∩ write ∩ (logits^⊥)

For geometric intuition and taxonomy of named subspaces, see docs/steering_methods.qmd.

All bases are detached (frozen) to prevent gradient hacking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
from loguru import logger
from tqdm import tqdm


def get_hidden_size(model: nn.Module) -> int:
    """Get hidden_size from model config."""
    return model.config.hidden_size


@dataclass
class Subspace:
    """Orthonormal subspace basis with optional importance weights.
    
    V: [d_model, rank] - columns are orthonormal basis vectors.
    S: [rank] - optional importance weights (e.g., singular values, explained variance).
        Used for importance-weighted intersection. If None, all directions treated equally.
    All operations preserve orthonormality and return detached tensors.
    """
    V: Float[Tensor, "d_model rank"]
    name: str = ""
    S: Float[Tensor, "rank"] | None = None  # importance weights (optional)
    
    @property
    def rank(self) -> int:
        return self.V.shape[1]
    
    @property
    def d_model(self) -> int:
        return self.V.shape[0]
    
    def project(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... rank"]:
        """Project x onto subspace: x @ V -> [..., rank]."""
        return einsum(x, self.V, "... d, d r -> ... r")
    
    def project_back(self, x_proj: Float[Tensor, "... rank"]) -> Float[Tensor, "... d_model"]:
        """Lift from subspace back to d_model: x_proj @ V.T -> [..., d_model]."""
        return einsum(x_proj, self.V, "... r, d r -> ... d")
    
    def projector(self) -> Float[Tensor, "d_model d_model"]:
        """Return projection matrix P = V @ V.T: [d_model, d_model]."""
        return einsum(self.V, self.V, "d r, e r -> d e")
    
    def __and__(self, other: Subspace) -> Subspace:
        """Intersection: shared directions between two subspaces.
        
        Uses SVD of P_self @ P_other to find shared subspace.
        Keeps singular vectors with singular value > 0.5 (shared = close to 1).
        """
        return approx_intersection(self, other)
    
    def __sub__(self, other: Subspace) -> Subspace:
        """Orthogonal-complement projection: self projected into other^perp.

        This is not set subtraction. With orthonormal bases, we remove the component
        of span(self) that lies in span(other), then re-orthonormalize.

        Notation: A - B corresponds to A_perp_B := Π_{B^⊥}(A).
        """
        return project_subspace_into_perp(self, other)
    
    def __repr__(self) -> str:
        return f"Subspace({self.name}, rank={self.rank}, d={self.d_model})"


def orthonormalize(V: Float[Tensor, "d k"]) -> Float[Tensor, "d k"]:
    """Ensure V is orthonormal via QR decomposition."""
    Q, _ = torch.linalg.qr(V.float())
    return Q.to(V.dtype).detach()


def normalize_rows(
    X: Tensor,
    *,
    eps_frac: float = 0.01,
    eps_abs: float = 1e-8,
) -> Tensor:
    """Row-wise normalization with a robust floor.

    Plain unit-normalization `x / (||x|| + eps)` can overweight near-zero rows:
    they become effectively pure noise directions with equal vote.

    This uses a median-scaled floor:
        denom = ||x|| + eps_frac * median(||x||)

    So tiny rows get downweighted rather than amplified.
    """
    norms = X.norm(dim=-1, keepdim=True)
    scale = norms.median().clamp(min=eps_abs)
    return X / (norms + eps_frac * scale)


def log_topk_explained_variance(S: Tensor, name: str, ks: tuple[int, ...] = (1, 10, 50)) -> None:
    """Log cumulative explained variance for different k values.
    
    Helps diagnose if k=1 is stable (high explained var) or unstable (low explained var).
    """
    total = S.sum() + 1e-8
    parts = []
    for k in ks:
        if k <= len(S):
            var_k = (S[:k].sum() / total).item()
            parts.append(f"k={k}:{var_k:.1%}")
    if parts:
        logger.info(f"{name} explained_var spectrum: {', '.join(parts)}")


# ============================================================================
# Core tensor operations (no Subspace wrapper)
# ============================================================================

def pca_subspace(
    X: Float[Tensor, "n d"],
    top_k: int = 256,
    normalize_samples: bool = False,
    name: str = "pca",
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Subspace:
    """PCA of sample matrix X, returning orthonormal subspace.
    
    Centralizes the common pattern: center → (optional normalize) → SVD → truncate.
    
    Args:
        X: [n_samples, d_model] data matrix (rows are samples)
        top_k: Number of principal components to keep (None = full rank)
        normalize_samples: If True, normalize each row to unit norm before PCA.
            Use this when you want angular consistency (each sample votes equally)
            rather than magnitude-weighted (outliers dominate). Recommended for
            cho-rej differences where some pairs have larger magnitude.
        name: Name for the returned Subspace
        device: Output device (default: X.device)
        dtype: Output dtype (default: X.dtype)
        
    Returns:
        Subspace with V: [d_model, k] and S: [k] (singular values)
    """
    if device is None:
        device = X.device
    if dtype is None:
        dtype = X.dtype
    
    X_f = X.float()
    
    if normalize_samples:
        # Unit-normalize each sample: equal vote regardless of magnitude
        X_f = normalize_rows(X_f)
    
    # Center
    X_centered = X_f - X_f.mean(dim=0, keepdim=True)
    
    # SVD (thin)
    _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    # Truncate
    if top_k is None:
        k = Vh.shape[0]
    else:
        k = min(top_k, Vh.shape[0])
    
    V = Vh[:k, :].T.to(dtype).to(device).detach()  # [d, k]
    S_out = S[:k].to(dtype).to(device).detach()    # [k]
    
    explained_var = S[:k].sum() / (S.sum() + 1e-8)
    logger.info(f"{name} subspace: rank={k}, explained_var={explained_var:.1%}")
    log_topk_explained_variance(S, name)
    
    return Subspace(V, name=name, S=S_out)


def approx_intersection_bases(
    V_a: Float[Tensor, "d r_a"],
    V_b: Float[Tensor, "d r_b"],
    top_k: int = 256,
    min_overlap: float = 0.1,
) -> tuple[Float[Tensor, "d k"], Float[Tensor, "k"]]:
    """Intersection of two subspaces via principal angles.
    
    Math: SVD of V_a.T @ V_b = U @ S @ Vh gives:
    - S = cos(principal angles) between subspaces
    - S[i] = 1 means direction i is in BOTH subspaces (true intersection)
    - S[i] ≈ 0 means orthogonal (no intersection in that direction)
    
    Returns top-k directions ordered by cos(θ) (most shared first),
    filtered to only include directions with S > min_overlap.
    Symmetric: intersect(A,B) ≈ intersect(B,A) (same subspace, maybe diff basis).
    
    Args:
        V_a, V_b: Orthonormal bases [d_model, rank]
        top_k: Maximum number of intersection directions to return
        min_overlap: Minimum cos(principal_angle) to include (default 0.1).
            S=1 means perfect overlap, S=0 means orthogonal.
            Directions with S < min_overlap are excluded as "not truly shared".
        
    Returns:
        V_shared: [d_model, k] orthonormal basis of shared directions
        S_shared: [k] cos(principal_angles) - overlap quality for each direction
    """
    d = V_a.shape[0]
    original_dtype = V_a.dtype
    V_a_f = V_a.float()
    V_b_f = V_b.float().to(V_a.device)
    
    # Principal angles via SVD
    # S[i] = cos(θ_i) where θ_i is i-th principal angle
    overlap = V_a_f.T @ V_b_f  # [r_a, r_b]
    U_a, S, Vh_b = torch.linalg.svd(overlap, full_matrices=False)
    
    # Filter by overlap quality: only keep directions with cos(θ) > min_overlap
    # S is already sorted descending, so find first index below threshold
    high_overlap_mask = S > min_overlap
    n_high_overlap = high_overlap_mask.sum().item()
    k = min(top_k, n_high_overlap, S.shape[0])
    
    if k == 0:
        logger.warning(f"intersect_bases: no directions with overlap > {min_overlap} (max S={S[0]:.3f}). Returning top-1 anyway.")
        k = 1
    
    # Directions in original space: 
    #   dir_a = V_a @ U_a[:, i]  (direction from A that aligns with B)
    #   dir_b = V_b @ Vh_b.T[:, i]  (direction from B that aligns with A)
    # For true intersection (S=1), dir_a = dir_b
    # For approximate intersection, average to get symmetric result
    dirs_a = V_a_f @ U_a[:, :k]  # [d, k]
    dirs_b = V_b_f @ Vh_b.T[:, :k]  # [d, k]
    
    # Average (bisector) - symmetric and lies between both subspaces
    V_shared = dirs_a + dirs_b
    
    V_out = orthonormalize(V_shared).to(original_dtype).to(V_a.device).detach()
    S_out = S[:k].to(original_dtype).to(V_a.device).detach()
    return V_out, S_out


def importance_weighted_intersection_bases(
    V_a: Float[Tensor, "d r_a"],
    S_a: Float[Tensor, "r_a"],
    V_b: Float[Tensor, "d r_b"],
    S_b: Float[Tensor, "r_b"],
    top_k: int = 256,
) -> tuple[Float[Tensor, "d k"], Float[Tensor, "k"]]:
    """Importance-weighted intersection of two subspaces.
    
    Like approx_intersection_bases, but weights by importance in both subspaces.
    
    Math: G = (V_a @ diag(S_a/sum)).T @ (V_b @ diag(S_b/sum)) gives Gram matrix where
    G[i,j] = (S_a[i]/sum_a) * (S_b[j]/sum_b) * cos(angle_ij). SVD of G finds 
    directions that are both aligned AND important in their original contexts.
    
    For random 256-dim subspaces, typical S_shared values are ~0.001-0.01.
    For aligned subspaces with concentrated importance, S_shared can reach ~0.1.
    
    Args:
        V_a, V_b: Orthonormal bases [d_model, rank]
        S_a, S_b: Importance weights (e.g., singular values) for each direction
        top_k: Number of intersection directions to return (always returns this many)
        
    Returns:
        V_shared: [d_model, top_k] orthonormal basis of shared directions
        S_shared: [top_k] importance scores for each shared direction
    """
    d = V_a.shape[0]
    original_dtype = V_a.dtype
    device = V_a.device
    
    V_a_f = V_a.float()
    V_b_f = V_b.float().to(device)
    S_a_f = S_a.float().to(device)
    S_b_f = S_b.float().to(device)
    
    # Normalize importance weights to sum to 1 (relative importance)
    S_a_norm = S_a_f / (S_a_f.sum() + 1e-8)
    S_b_norm = S_b_f / (S_b_f.sum() + 1e-8)
    
    # Scale bases by importance: V_scaled[:, i] = V[:, i] * S[i]
    V_a_scaled = V_a_f * S_a_norm.unsqueeze(0)  # [d, r_a]
    V_b_scaled = V_b_f * S_b_norm.unsqueeze(0)  # [d, r_b]
    
    # Gram matrix: G[i,j] = S_a[i] * S_b[j] * cos(angle_ij)
    G = V_a_scaled.T @ V_b_scaled  # [r_a, r_b]
    
    # SVD to find principal importance-weighted directions
    U_a, S_shared, Vh_b = torch.linalg.svd(G, full_matrices=False)
    
    # Always return top_k (or as many as available)
    k = min(top_k, S_shared.shape[0])
    
    # Reconstruct shared directions in original space
    # dir_a = V_a @ U_a[:, i] (but we used scaled, so unscale conceptually)
    dirs_a = V_a_f @ U_a[:, :k]  # [d, k]
    dirs_b = V_b_f @ Vh_b.T[:, :k]  # [d, k]
    
    # Bisector (average)
    V_shared = dirs_a + dirs_b
    V_shared = orthonormalize(V_shared).to(original_dtype).to(device).detach()
    S_out = S_shared[:k].to(original_dtype).to(device).detach()
    
    return V_shared, S_out


def approx_intersection(a: Subspace, b: Subspace, top_k: int = 256) -> Subspace:
    """Approximate intersection wrapper for Subspace objects.

    Important: this is not a strict set-theoretic intersection.
    We compute shared directions using principal angles and return a symmetric
    "bisector" basis that lies between the two spans.
    
    If both subspaces have importance weights (S), uses importance-weighted
    intersection that prioritizes directions important in BOTH subspaces.

    Use this when you want directions that are simultaneously supported by both
    mechanisms (e.g., task_diff AND suppressed), not when you want an orthogonal
    complement.
    """
    if a.V.shape[1] < 2 or b.V.shape[1] < 2:
        raise ValueError(
            f"Cannot intersect subspaces with rank < 2: {a.name} has rank {a.V.shape[1]}, "
            f"{b.name} has rank {b.V.shape[1]}. Pass full-rank subspaces before cropping."
        )
    
    # If both have importance weights, use weighted intersection
    if a.S is not None and b.S is not None:
        V_shared, S_shared = importance_weighted_intersection_bases(a.V, a.S, b.V, b.S, top_k=top_k)
        return Subspace(V_shared, name=f"({a.name} & {b.name})", S=S_shared)
    
    # Unweighted intersection also returns S (cos of principal angles)
    V_shared, S_shared = approx_intersection_bases(a.V, b.V, top_k=top_k)
    return Subspace(V_shared, name=f"({a.name} & {b.name})", S=S_shared)


def project_bases_into_perp(
    V_a: Float[Tensor, "d r_a"],
    V_b: Float[Tensor, "d r_b"],
) -> Float[Tensor, "d k"]:
    """Project a into the orthogonal complement of b. Returns tensor.

    This is NOT a set difference.
    With orthonormal bases, we compute:

        V_residual = V_a - V_b @ (V_b^T @ V_a)

    i.e. remove the component of span(a) that lies in span(b), then re-orthonormalize.
    This is the operation you want for "hidden-from-X" constructions like:
    write_x_notlogits = write projected into (logits)^perp.

    Complexity: O(d * r_a * r_b).
    
    Args:
        V_a: Base subspace [d_model, r_a]
        V_b: Subspace to remove [d_model, r_b]
        
    Returns:
        V_result: [d_model, k] orthonormal basis of remaining directions
    """
    d = V_a.shape[0]
    device = V_a.device
    original_dtype = V_a.dtype
    
    V_a_f = V_a.float()
    V_b_f = V_b.float().to(device)
    
    overlap = V_b_f.T @ V_a_f  # [r_b, r_a]
    V_residual = V_a_f - V_b_f @ overlap  # [d, r_a]
    
    norms = V_residual.norm(dim=0)
    keep = norms > 1e-6
    
    if keep.sum() == 0:
        logger.error("Subtraction removed all directions from subspace.")
        return torch.zeros(d, 1, dtype=original_dtype, device=device)
    
    return orthonormalize(V_residual[:, keep]).to(original_dtype).to(device)


def project_subspace_into_perp(a: Subspace, b: Subspace) -> Subspace:
    """Project span(a) into span(b)^perp (wrapper for Subspace objects).

    Notation: A_perp_B := Π_{B^⊥}(A).
    """
    V_result = project_bases_into_perp(a.V, b.V)
    return Subspace(V_result, name=f"({a.name} - {b.name})")


def union_bases(
    bases: List[Float[Tensor, "d r"]],
    top_k: Optional[int] = 4096,
) -> Float[Tensor, "d k"]:
    """Union of subspaces via PCA on concatenated bases. Returns tensor.
    
    Concatenates all bases, then takes top-k SVD components.
    
    Args:
        bases: List of orthonormal bases [d_model, rank_i]
        top_k: Number of output dimensions (defaults to sum of ranks)
        
    Returns:
        V_combined: [d_model, k] orthonormal basis spanning the union
    """
    V_cat = torch.cat(bases, dim=1).float()
    
    if top_k is None:
        top_k = V_cat.shape[1]
    top_k = min(top_k, V_cat.shape[1])
    
    U, S, _ = torch.linalg.svd(V_cat, full_matrices=False)
    return U[:, :top_k].to(bases[0].dtype).to(bases[0].device).detach()


def combine_subspaces(subspaces: List[Subspace], top_k: Optional[int] = 4096) -> Subspace:
    """Union wrapper for Subspace objects."""
    V_combined = union_bases([s.V for s in subspaces], top_k)
    names = "+".join(s.name for s in subspaces)
    return Subspace(V_combined, name=f"({names})")


# ============================================================================
# Module classification (auto-detect read/write modules)
# ============================================================================

# Naming-based disambiguation is only needed for *square* residual-connected linears
# where shape alone can't tell "read" vs "write".
#
# For the model families used in this repo (Qwen3 / Llama / Gemma3 / OLMo-3), these
# suffixes are stable and match HF modeling code:
# - readers: read residual stream -> project to attention/MLP internal dims
# - writers: write back to residual stream
_SQUARE_RESIDUAL_READ_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv_proj",
    "gate_proj",
    "up_proj",
}

_SQUARE_RESIDUAL_WRITE_SUFFIXES = {
    "o_proj",
    "down_proj",
    "out_proj",
}


def _classify_residual_linear_suffixes(
    model: nn.Module,
    blocklist: Optional[List[str]] = None,
) -> tuple[set[str], set[str]]:
    """Return (read_suffixes, write_suffixes) inferred from module shapes.

    Rule:
    - If a Linear is not residual-connected (in!=d_model and out!=d_model): ignore.
    - If in==d_model and out!=d_model: reader (rectangular, unambiguous).
    - If out==d_model and in!=d_model: writer (rectangular, unambiguous).
    - If in==d_model and out==d_model (square residual-connected): require suffix in one
      of the hardcoded square sets above, else FAIL FAST.
    """
    if blocklist is None:
        blocklist = ["vision", "embed", "lm_head", "norm"]

    hidden_size = get_hidden_size(model)
    read_suffixes: set[str] = set()
    write_suffixes: set[str] = set()
    unknown_square: set[str] = set()

    for name, module in model.named_modules():
        if any(block in name for block in blocklist):
            continue
        if not isinstance(module, nn.Linear):
            continue

        in_is_resid = module.in_features == hidden_size
        out_is_resid = module.out_features == hidden_size
        if not (in_is_resid or out_is_resid):
            continue

        suffix = name.split(".")[-1]
        
        # Skip PEFT wrapper layers (base_layer, lora_A, lora_B, etc.)
        if suffix in {"base_layer", "lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"}:
            continue

        if in_is_resid and not out_is_resid:
            read_suffixes.add(suffix)
            continue

        if out_is_resid and not in_is_resid:
            write_suffixes.add(suffix)
            continue

        # Ambiguous: square residual-connected
        if suffix in _SQUARE_RESIDUAL_READ_SUFFIXES:
            read_suffixes.add(suffix)
        elif suffix in _SQUARE_RESIDUAL_WRITE_SUFFIXES:
            write_suffixes.add(suffix)
        else:
            unknown_square.add(suffix)

    if unknown_square:
        raise ValueError(
            "Found square residual-connected Linear modules with unknown suffixes. "
            "Shape alone cannot classify these as residual-readers vs residual-writers. "
            f"unknown_square_suffixes={sorted(unknown_square)}; "
            f"known_square_read={sorted(_SQUARE_RESIDUAL_READ_SUFFIXES)}; "
            f"known_square_write={sorted(_SQUARE_RESIDUAL_WRITE_SUFFIXES)}."
        )

    # Sanity: a suffix should not be in both.
    overlap = read_suffixes & write_suffixes
    if overlap:
        raise ValueError(
            f"Internal error: suffixes classified as both read and write: {sorted(overlap)}"
        )

    return read_suffixes, write_suffixes

def find_write_modules(model: nn.Module, blocklist: List[str] = None) -> List[str]:
    """Find module suffixes that WRITE to the residual stream.

    Uses shape-first classification; for square residual-connected linears we rely on
    a small hardcoded suffix set and FAIL FAST on unknowns.
    """
    _, write_suffixes = _classify_residual_linear_suffixes(model, blocklist=blocklist)
    result = sorted(write_suffixes)
    logger.debug(f"Auto-detected write modules: {result}")
    return result


def find_read_modules(model: nn.Module, blocklist: List[str] = None) -> List[str]:
    """Find module suffixes that READ from the residual stream.

    Uses shape-first classification; for square residual-connected linears we rely on
    a small hardcoded suffix set and FAIL FAST on unknowns.
    """
    read_suffixes, _ = _classify_residual_linear_suffixes(model, blocklist=blocklist)
    result = sorted(read_suffixes)
    logger.debug(f"Auto-detected read modules: {result}")
    return result


# ============================================================================
# Suppressed subspace from layer diffs (model-agnostic)
# ============================================================================

def compute_suppressed_from_hidden_states(
    hidden_states: Float[Tensor, "batch n_layers_plus1 d_model"],
    lm_head_subspace: Subspace,
    top_k: int =256,
    exclude_early_frac: float = 0.1,
) -> Subspace:
    """Compute suppressed subspace from layer hidden state diffs.
    
    This is the model-agnostic approach: no hardcoded layer fractions.
    
    Suppressed = directions that are:
    1. Written (positive diff between layers)
    2. NOT read (consumed) by later layers  
    3. NOT readable by lm_head
    
    Formula:
        layer_diff = h[l+1] - h[l]  for each layer (excluding early layers)
        written = sum(relu(layer_diff))  # positive = energy added
        read = sum(relu(-layer_diff))    # negative = energy consumed
        suppressed = written - read - logitsable
    
    Args:
        hidden_states: [batch, n_layers+1, d_model] - all layer outputs
        lm_head_subspace: Subspace readable by lm_head
        top_k: Number of components to keep
        exclude_early_frac: Fraction of early layers to exclude (default 0.1 = first 10%).
            Early layers process embeddings and contain less steering-relevant signal.
        
    Returns:
        Subspace of suppressed directions
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    batch, n_layers_plus1, d = hidden_states.shape
    n_layers = n_layers_plus1 - 1
    
    # Compute layer-to-layer diffs: [batch, n_layers, d_model]
    layer_diffs: Float[Tensor, "batch n_layers d"] = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    
    # Exclude early layers (they process embeddings, not planning/reasoning)
    start_layer = int(exclude_early_frac * n_layers)
    if start_layer > 0:
        layer_diffs = layer_diffs[:, start_layer:, :]
        logger.debug(f"Suppressed subspace: excluding first {start_layer} layers (of {n_layers})")
    
    # Written = positive diffs (energy added to residual)
    # Read = negative diffs (energy consumed from residual)
    written: Float[Tensor, "batch d"] = F.relu(layer_diffs).sum(dim=1)   # sum over layers
    read: Float[Tensor, "batch d"] = F.relu(-layer_diffs).sum(dim=1)     # sum over layers
    
    # Net written but not read
    net_written: Float[Tensor, "batch d"] = written - read
    
    # Project out lm_head readable directions: (I - P_lmhead) @ x = x - V @ V.T @ x
    # Avoids materializing d×d matrix (OOM for d_model > 8192)
    V_lm: Float[Tensor, "d r_lm"] = lm_head_subspace.V.float().to(device)
    net_written_f = net_written.float()
    proj_onto_lm: Float[Tensor, "batch r_lm"] = einsum(net_written_f, V_lm, "b d, d r -> b r")
    proj_back: Float[Tensor, "batch d"] = einsum(proj_onto_lm, V_lm, "b r, d r -> b d")
    suppressed: Float[Tensor, "batch d"] = net_written_f - proj_back
    
    # PCA on suppressed directions
    suppressed_centered = suppressed - suppressed.mean(dim=0)
    cov = suppressed_centered.T @ suppressed_centered / len(suppressed_centered)
    U, S, _ = torch.linalg.svd(cov)
    
    V_supp = U[:, :top_k].to(dtype).to(device).detach()
    S_supp = S[:top_k].to(dtype).to(device).detach()
    
    explained_var = S[:top_k].sum() / (S.sum() + 1e-8)
    logger.info(f"Suppressed subspace (from layer diffs): rank={V_supp.shape[1]}, explained_var={explained_var:.1%}")
    log_topk_explained_variance(S, "suppressed")
    
    return Subspace(V_supp, name="suppressed", S=S_supp)


# ============================================================================
# Legacy subspace computation (kept for compatibility)
# ============================================================================

def compute_lm_head_subspace(model: nn.Module, top_k: int = 256) -> Subspace:
    """Compute subspace read by lm_head (directions that affect output logits).
    
    Uses right singular vectors (V) of lm_head.weight since it reads from residual.
    lm_head computes logits = h @ W.T, so it reads directions in row-space of W.
    
    Args:
        model: Model with lm_head
        top_k: Number of components
        
    Returns:
        Subspace of directions lm_head reads
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # SVD: W = U @ S @ Vh, row-space = span of Vh rows = right singular vectors
    S, Vh = compute_lm_head_svd(model)
    V_read: Float[Tensor, "d_model top_k"] = Vh[:top_k, :].T  # transpose: [d_model, top_k]
    S_read = S[:top_k].to(dtype).to(device).detach()
    
    V_read = V_read.to(dtype).to(device).detach()
    logger.debug(f"logits_read subspace: rank={V_read.shape[1]}")

    return Subspace(V_read, name="logits_read", S=S_read)


def compute_lm_head_svd(model: nn.Module) -> tuple[Tensor, Tensor]:
    """Return (S, Vh) for lm_head.weight SVD.

    This is the canonical source for lm_head singular values/vectors used by
    activation-weighted null-space constructions.

    Returns:
        S: [rank] singular values (descending)
        Vh: [rank, d_model] right singular vectors (rows)
    """
    # lm_head.weight: [vocab_size, d_model]
    W: Float[Tensor, "vocab d_model"] = model.lm_head.weight.data
    _, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)
    return S, Vh


def compute_embed_subspace(model: nn.Module, top_k: int = 256) -> Subspace:
    """Compute subspace written by embedding layer.
    
    Uses left singular vectors (U) of embed_tokens.weight since it writes to residual.
    
    Args:
        model: Model with embed_tokens
        top_k: Number of components
        
    Returns:
        Subspace of directions embedding writes
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # embed_tokens.weight: [vocab_size, d_model], output is row-indexed
    # Column space = write directions
    W = model.model.embed_tokens.weight.data  # [vocab, d_model]
    
    # Column space via transpose
    U, S, _ = torch.linalg.svd(W.T.float().cpu(), full_matrices=False)
    V_write = U[:, :top_k]  # [d_model, top_k]
    
    V_write = V_write.to(dtype).to(device).detach()
    logger.info(f"Embed write subspace: rank={V_write.shape[1]}")
    
    return Subspace(V_write, name="embed_write")


def compute_write_not_read_subspace(
    write_subspace: Subspace,
    read_subspace: Subspace,
    lm_head_subspace: Optional[Subspace] = None,
    top_k: int =256,
) -> Subspace:
    """Compute Write-Not-Read subspace: directions written but not read.

    Notation: WnR = Write_perp_Read = Π_{Read^⊥}(Write).

    If `lm_head_subspace` is provided, also subtract directions readable by
    the lm_head (since those are "read" at the output interface).
    
    Args:
        write_subspace: Subspace of write directions
        read_subspace: Subspace of read directions
        top_k: Number of components
        
    Returns:
        Subspace of directions written but ignored by reading layers
    """
    wnr = project_subspace_into_perp(write_subspace, read_subspace)
    if lm_head_subspace is not None:
        wnr = project_subspace_into_perp(wnr, lm_head_subspace)
    
    if wnr.rank > top_k:
        wnr = Subspace(wnr.V[:, :top_k], name="write_not_read")
    else:
        wnr.name = "write_not_read"
        
    return wnr


def compute_stenographic_subspace(
    task_diff_subspace: Subspace,
    suppressed_subspace: Subspace,
    top_k: int = 16,
) -> Subspace:
    """Compute Stenographic subspace: task signal hidden in suppressed space.
    
    Steno = TaskDiff ∩ Suppressed (symmetric intersection via mutual projection)
    
    Args:
        task_diff_subspace: Subspace of task differences
        suppressed_subspace: Subspace of suppressed directions
        top_k: Number of components (passed to intersect_subspaces)
        
    Returns:
        Subspace of hidden task signals
    """
    steno = approx_intersection(task_diff_subspace, suppressed_subspace, top_k=top_k)
    steno.name = "taskdiff_x_suppressed"
    return steno


def compute_write_x_notlogits_subspace(
    write_subspace: Subspace,
    lm_head_subspace: Subspace,
    top_k: int =256,
) -> Subspace:
    """Compute write_x_notlogits: write projected into (logits_read)^perp.

    Notation: write_x_notlogits = Write_perp_logits_read = Π_{(logits_read)^⊥}(Write).

    In code this uses project_subspace_into_perp(write, logits), which performs an
    orthogonal-complement projection (see project_bases_into_perp docstring), not a set
    difference.
    
    These directions are written to residual by model layers but don't affect
    output logits (lm_head can't read them). Simpler than write_not_read since
    it ignores layer-to-layer reads.

    Note it includes write to avoid token embeddings that prepopulate the residual stream
    
    Args:
        write_subspace: Subspace of write directions
        lm_head_subspace: Subspace readable by lm_head
        top_k: Number of components
        
    Returns:
        Subspace of directions hidden from final output
    """
    hfl = project_subspace_into_perp(write_subspace, lm_head_subspace)
    
    if hfl.rank > top_k:
        hfl = Subspace(hfl.V[:, :top_k], name="write_x_notlogits")
    else:
        hfl.name = "write_x_notlogits"
        
    return hfl


def compute_logits_tail_subspace(
    hidden_states: Float[Tensor, "batch n_layers_plus1 d_model"],
    lm_head_S: Float[Tensor, "rank"],
    lm_head_Vh: Float[Tensor, "rank d_model"],
    top_k: int = 64,
    layer_range: Optional[tuple] = None,
    null_frac: float = 0.5,
) -> Subspace:
    """Compute wanda_x_notlogits subspace: tail lm_head singular dirs weighted by activation.

    Like write_x_notlogits but empirical: uses actual activations to weight directions.
    
    Method (WANDA-inspired):
    1. Take bottom `null_frac` of lm_head singular directions (low S = low output gain)
    2. Project hidden states into this tail subspace
    3. Weight each direction by activation magnitude (WANDA: ||X||_2 per direction)
    4. PCA on weighted projections to find most-used directions within tail space
    
    This differs from write_x_notlogits (static weight subtraction) by incorporating
    which directions are actually used, not just which could theoretically be hidden.
    
    Args:
        hidden_states: [batch, n_layers+1, d_model] from model output
        lm_head_S: [rank] singular values of lm_head (descending order from SVD)
        lm_head_Vh: [rank, d_model] right singular vectors (rows are basis vectors)
        top_k: Number of components to return
        layer_range: (start_frac, end_frac) for which layers to use (default 0.3-0.8)
        null_frac: Fraction of bottom singular directions to use (default 0.5)
        
    Returns:
        Subspace of actively-used low-gain directions
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    d_model = hidden_states.shape[-1]
    n_layers_plus1 = hidden_states.shape[1]
    n_layers = n_layers_plus1 - 1
    
    if layer_range is None:
        layer_range = (0.3, 0.8)
    
    start_idx = max(1, int(layer_range[0] * n_layers))
    end_idx = min(n_layers, int(layer_range[1] * n_layers))
    
    # Get relevant hidden states [batch, selected_layers, d_model]
    hs_selected = hidden_states[:, start_idx:end_idx, :]
    
    # Take bottom null_frac of singular directions (low S = lm_head ignores)
    rank = lm_head_Vh.shape[0]
    null_start = int((1 - null_frac) * rank)
    null_rank = rank - null_start
    
    if null_rank < top_k:
        logger.warning(f"null_frac={null_frac} gives {null_rank} dims < top_k={top_k}. Expanding.")
        null_start = max(0, rank - top_k * 2)
        null_rank = rank - null_start
    
    # V_tail: [d_model, null_rank] - bottom singular vectors
    V_tail = lm_head_Vh[null_start:, :].T.to(device).to(dtype)  # [d_model, null_rank]
    S_tail = lm_head_S[null_start:].to(device).float()  # [null_rank]
    
    # Project hidden states into tail subspace
    hs_flat = hs_selected.reshape(-1, d_model).float()
    z = hs_flat @ V_tail.float()  # [n, null_rank]
    
    # WANDA-style: weight by activation magnitude (L2 norm per direction)
    # ||X_j||_2 = sqrt(sum_i x_ij^2), captures total energy in each direction
    activation_norm = z.norm(dim=0)  # [null_rank] - L2 norm across samples
    
    # Weight projections by activation norm
    z_weighted = z * activation_norm  # [n, null_rank]
    
    # PCA on weighted projections
    z_centered = z_weighted - z_weighted.mean(dim=0, keepdim=True)
    _, S_pca, Vh_pca = torch.linalg.svd(z_centered, full_matrices=False)
    
    # Top-k directions in tail basis
    k = min(top_k, Vh_pca.shape[0])
    U_top = Vh_pca[:k, :]  # [k, null_rank]
    
    # Map back to residual basis: [k, null_rank] @ [null_rank, d_model] -> [k, d_model]
    V_result = (U_top @ V_tail.T.float()).T  # [d_model, k]
    
    # Orthonormalize
    V_result, _ = torch.linalg.qr(V_result)
    V_result = V_result[:, :k].to(dtype).to(device).detach()
    
    explained_var = (S_pca[:k] ** 2).sum() / ((S_pca ** 2).sum() + 1e-8)
    act_range = f"{activation_norm.min():.2f}-{activation_norm.max():.2f}"
    s_range = f"{S_tail.min():.2e}-{S_tail.max():.2e}"
    logger.info(f"wanda_x_notlogits subspace: rank={V_result.shape[1]}, null_dims={null_rank}, "
                f"explained_var={explained_var:.1%}, activation_range={act_range}, S_range={s_range}")
    log_topk_explained_variance(S_pca ** 2, "wanda_x_notlogits")  # squared because we used variance formula

    return Subspace(V_result, name="wanda_x_notlogits")


def compute_taskdiff_x_write_x_notlogits_subspace(
    hidden_states: Float[Tensor, "batch n_layers_plus1 d_model"],
    write_subspace: "Subspace",
    lm_head_S: Float[Tensor, "rank"],
    lm_head_Vh: Float[Tensor, "rank d_model"],
    top_k: int = 64,
    layer_frac: float = 0.7,
    null_frac: float = 0.5,
) -> "Subspace":
    """Task-discriminative directions in write ∩ lm_head_null.
    
    Finds directions that are:
    1. Writable (in column space of o_proj/down_proj)
    2. Hidden from lm_head (in bottom singular vectors of lm_head)
    3. Task-discriminative (high cho-rej difference magnitude)
    
    Unlike `write_x_notlogits` (weight-only), this uses cho-rej activations
    to find WHICH hidden directions carry task-relevant signal.
    
    Unlike `logits_tail` (sample-specific), this weights by cho-rej
    DIFFERENCE, not total activation magnitude.
    
    Args:
        hidden_states: [batch, n_layers+1, d_model] from contrastive pairs
            Assumes batch dimension alternates cho/rej: [cho_0, rej_0, cho_1, rej_1, ...]
        write_subspace: Subspace of write directions (from compute_write_subspace)
        lm_head_S: [rank] singular values of lm_head (descending)
        lm_head_Vh: [rank, d_model] right singular vectors
        top_k: Number of components to return
        layer_frac: Which layer to use (fraction of total layers)
        null_frac: Fraction of bottom singular vectors to use as "null" (default 0.5)
        
    Returns:
        Subspace of task-discriminative write-lm_null directions
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    batch, n_layers_plus1, d_model = hidden_states.shape
    n_layers = n_layers_plus1 - 1
    
    # Get layer hidden states
    layer_idx = int(layer_frac * n_layers)
    hs = hidden_states[:, layer_idx, :].float()  # [batch, d]
    
    # Split cho/rej (assumes alternating)
    hs_cho = hs[0::2]  # [n_pairs, d]
    hs_rej = hs[1::2]  # [n_pairs, d]
    diff = hs_cho - hs_rej  # [n_pairs, d]
    
    # Step 1: Get lm_head null space (bottom singular vectors = low output gain)
    rank = lm_head_Vh.shape[0]
    null_start = int((1 - null_frac) * rank)
    V_lm_null = lm_head_Vh[null_start:, :].T.to(device).float()  # [d, null_rank]
    
    # Step 2: Intersect with write space
    V_write = write_subspace.V.to(device).float()  # [d, write_rank]
    V_write_lmnull, _ = approx_intersection_bases(V_write, V_lm_null, top_k=256)  # [d, intersect_rank]
    
    if V_write_lmnull.shape[1] < 2:
        logger.warning(f"taskdiff_x_write_x_notlogits: write ∩ lm_null intersection too small ({V_write_lmnull.shape[1]}), using write only")
        V_write_lmnull = V_write
    
    # Step 3: Project differences into write ∩ lm_null
    z_diff = diff @ V_write_lmnull  # [n_pairs, intersect_rank]
    
    # Step 4: Weight by task-discriminative magnitude (mean absolute difference)
    task_weight = z_diff.abs().mean(dim=0)  # [intersect_rank]
    
    # Step 5: PCA on weighted projections to find most task-discriminative directions
    z_weighted = z_diff * task_weight
    z_centered = z_weighted - z_weighted.mean(dim=0, keepdim=True)
    _, S_pca, Vh_pca = torch.linalg.svd(z_centered, full_matrices=False)
    
    # Top-k directions in intersection basis
    k = min(top_k, Vh_pca.shape[0])
    U_top = Vh_pca[:k, :]  # [k, intersect_rank]
    
    # Map back to residual basis: [k, intersect_rank] @ [intersect_rank, d] -> [k, d]
    V_result = (U_top @ V_write_lmnull.T).T  # [d, k]
    
    # Orthonormalize
    V_result = orthonormalize(V_result).to(dtype).to(device).detach()
    
    explained_var = (S_pca[:k] ** 2).sum() / ((S_pca ** 2).sum() + 1e-8)
    intersect_rank = V_write_lmnull.shape[1]
    weight_range = f"{task_weight.min():.2f}-{task_weight.max():.2f}"
    logger.info(f"taskdiff_write_x_notlogits subspace: rank={V_result.shape[1]}, intersect_rank={intersect_rank}, "
                f"explained_var={explained_var:.1%}, task_weight_range={weight_range}")
    log_topk_explained_variance(S_pca ** 2, "taskdiff_write_x_notlogits")
    
    return Subspace(V_result, name="taskdiff_write_x_notlogits")


# ============================================================================
# Subspace computation from precomputed SVDs (used by layer_selection.py)
# ============================================================================

def compute_churn_from_hidden_states(
    hidden_states: Float[Tensor, "batch n_layers_plus1 d_model"],
    top_k: int =256,
) -> Subspace:
    """Compute churn subspace: PCA of layer-to-layer changes.
    
    Churn captures "active computation lanes" - directions where layers
    add and remove energy during processing.
    
    Args:
        hidden_states: [batch, n_layers+1, d_model] - all layer outputs
        top_k: Number of components to keep
        
    Returns:
        Subspace of high-churn directions
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    d_model = hidden_states.shape[-1]
    
    # Layer diffs: [batch, n_layers, d_model]
    layer_diffs: Float[Tensor, "batch n_layers d"] = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    layer_diffs_flat: Float[Tensor, "n d"] = layer_diffs.reshape(-1, d_model).float()
    
    # PCA of layer diffs. normalize_samples=False: layer diffs are already comparable
    # (same scale within a model), and we want magnitude-weighted to capture where
    # most computation happens.
    sub = pca_subspace(
        layer_diffs_flat,
        top_k=top_k,
        normalize_samples=False,
        name="churn",
        device=device,
        dtype=dtype,
    )
    return sub


def compute_churn_constructive_from_hidden_states(
    hidden_states: Float[Tensor, "batch n_layers_plus1 d_model"],
    top_k: int =256,
    layer_range: Optional[tuple] = None,
) -> Subspace:
    """Compute constructive churn: directions where magnitude INCREASES across layers.
    
    Standard churn is unsigned (PCA of layer diffs). This variant filters to directions
    where the residual stream is actively BUILDING signal (amplifying), not erasing it.
    
    Method: For each churn PC, compute whether ||h @ v||^2 increases from early to late layers.
    Keep only PCs where slope > 0 (magnitude growing).
    
    Args:
        hidden_states: [batch, n_layers+1, d_model] - all layer outputs
        top_k: Number of components to keep
        layer_range: Optional (start_frac, end_frac) for slope computation (default 0.2-0.8)
        
    Returns:
        Subspace of constructive (amplifying) churn directions
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    d_model = hidden_states.shape[-1]
    n_layers_plus1 = hidden_states.shape[1]
    n_layers = n_layers_plus1 - 1
    
    if layer_range is None:
        layer_range = (0.2, 0.8)
    
    start_idx = max(1, int(layer_range[0] * n_layers))
    end_idx = min(n_layers, int(layer_range[1] * n_layers))
    
    # First compute regular churn PCs
    layer_diffs: Float[Tensor, "batch n_layers d"] = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    layer_diffs_flat: Float[Tensor, "n d"] = layer_diffs.reshape(-1, d_model).float()
    layer_diffs_centered = layer_diffs_flat - layer_diffs_flat.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(layer_diffs_centered, full_matrices=False)
    
    # Get more PCs than we need to filter
    n_candidates = min(top_k * 3, Vh.shape[0])
    V_candidates: Float[Tensor, "d k"] = Vh[:n_candidates, :].T  # [d_model, n_candidates]
    
    # For each PC, compute magnitude trend across layers
    # Project hidden states onto each PC: [batch, n_layers+1, n_candidates]
    proj_mag_sq = (hidden_states.float() @ V_candidates) ** 2  # [batch, n_layers+1, n_candidates]
    
    # Compute slope via early vs late layer magnitude.
    # We average over a 3-layer window at each endpoint for noise reduction.
    # The "constructive" signal is (late_mag - early_mag) > 0, meaning
    # magnitude in this PC direction is INCREASING through the network.
    # Window size 3 is a tradeoff: smaller = more sensitive but noisier.
    early_mag = proj_mag_sq[:, start_idx:start_idx+3, :].mean(dim=(0, 1))  # [n_candidates]
    late_mag = proj_mag_sq[:, end_idx-3:end_idx, :].mean(dim=(0, 1))  # [n_candidates]
    
    # Constructive = late > early (magnitude increasing)
    mag_slope = late_mag - early_mag  # positive = constructive
    
    # Select top-k by constructiveness (positive slope), sorted by magnitude
    constructive_mask = mag_slope > 0
    if constructive_mask.sum() < top_k:
        # Fallback: take all with positive slope, fill with least negative
        logger.warning(f"Only {constructive_mask.sum()} constructive PCs found, taking {top_k} least suppressive")
        sorted_indices = torch.argsort(mag_slope, descending=True)[:top_k]
    else:
        # Among constructive, sort by explained variance (S) and take top-k
        constructive_indices = torch.where(constructive_mask)[0]
        # Weight by both constructiveness and variance explained
        scores = mag_slope[constructive_indices] * S[constructive_indices]
        sorted_by_score = torch.argsort(scores, descending=True)[:top_k]
        sorted_indices = constructive_indices[sorted_by_score]
    
    V_constructive: Float[Tensor, "d k"] = V_candidates[:, sorted_indices].to(dtype).to(device).detach()
    
    n_positive = (mag_slope[sorted_indices] > 0).sum().item()
    logger.info(f"Churn_constructive subspace: rank={V_constructive.shape[1]}, {n_positive}/{top_k} strictly constructive")
    
    return Subspace(V_constructive, name="churn_constructive")


def compute_churn_suppressive_from_hidden_states(
    hidden_states: Float[Tensor, "batch n_layers_plus1 d_model"],
    top_k: int =256,
    layer_range: Optional[tuple] = None,
) -> Subspace:
    """Compute suppressive churn: directions where magnitude DECREASES across layers.
    
    Complement to constructive churn. These are directions the model is actively
    ERASING or damping during processing. Steering these could fight the model's flow.
    
    Args:
        hidden_states: [batch, n_layers+1, d_model] - all layer outputs
        top_k: Number of components to keep
        layer_range: Optional (start_frac, end_frac) for slope computation (default 0.2-0.8)
        
    Returns:
        Subspace of suppressive (erasing) churn directions
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    d_model = hidden_states.shape[-1]
    n_layers_plus1 = hidden_states.shape[1]
    n_layers = n_layers_plus1 - 1
    
    if layer_range is None:
        layer_range = (0.2, 0.8)
    
    start_idx = max(1, int(layer_range[0] * n_layers))
    end_idx = min(n_layers, int(layer_range[1] * n_layers))
    
    # First compute regular churn PCs
    layer_diffs: Float[Tensor, "batch n_layers d"] = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    layer_diffs_flat: Float[Tensor, "n d"] = layer_diffs.reshape(-1, d_model).float()
    layer_diffs_centered = layer_diffs_flat - layer_diffs_flat.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(layer_diffs_centered, full_matrices=False)
    
    n_candidates = min(top_k * 3, Vh.shape[0])
    V_candidates: Float[Tensor, "d k"] = Vh[:n_candidates, :].T
    
    proj_mag_sq = (hidden_states.float() @ V_candidates) ** 2
    early_mag = proj_mag_sq[:, start_idx:start_idx+3, :].mean(dim=(0, 1))
    late_mag = proj_mag_sq[:, end_idx-3:end_idx, :].mean(dim=(0, 1))
    mag_slope = late_mag - early_mag  # negative = suppressive
    
    # Select top-k by suppressiveness (negative slope)
    suppressive_mask = mag_slope < 0
    if suppressive_mask.sum() < top_k:
        logger.warning(f"Only {suppressive_mask.sum()} suppressive PCs found, taking {top_k} most suppressive")
        sorted_indices = torch.argsort(mag_slope, descending=False)[:top_k]  # Most negative first
    else:
        suppressive_indices = torch.where(suppressive_mask)[0]
        scores = -mag_slope[suppressive_indices] * S[suppressive_indices]  # Higher = more suppressive
        sorted_by_score = torch.argsort(scores, descending=True)[:top_k]
        sorted_indices = suppressive_indices[sorted_by_score]
    
    V_suppressive: Float[Tensor, "d k"] = V_candidates[:, sorted_indices].to(dtype).to(device).detach()
    
    n_negative = (mag_slope[sorted_indices] < 0).sum().item()
    logger.info(f"Churn_suppressive subspace: rank={V_suppressive.shape[1]}, {n_negative}/{top_k} strictly suppressive")
    
    return Subspace(V_suppressive, name="churn_suppressive")


def compute_task_diff_constructive_from_hidden_states(
    hidden_states: Float[Tensor, "batch n_layers_plus1 d_model"],
    top_k: int =256,
    layer_range: Optional[tuple] = None,
) -> Subspace:
    """Compute constructive task_diff: task-discriminative directions being AMPLIFIED.
    
    Standard task_diff is unsigned PCA of (h_cho - h_rej). This variant filters to
    directions where the cho/rej separation is INCREASING across layers - i.e., the
    model is actively building this distinction, not inheriting it from embeddings.
    
    Method: For each task_diff PC, compute slope of |h_cho @ v| - |h_rej @ v| across layers.
    Keep only PCs where separation is growing (constructive discrimination).
    
    Args:
        hidden_states: [batch, n_layers+1, d_model] with interleaved cho/rej pairs
        top_k: Number of components
        layer_range: Optional (start_frac, end_frac) for slope (default 0.3-0.8)
        
    Returns:
        Subspace of constructively-discriminating task directions
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    d_model = hidden_states.shape[-1]
    n_layers_plus1 = hidden_states.shape[1]
    n_layers = n_layers_plus1 - 1
    
    if layer_range is None:
        layer_range = (0.3, 0.8)
    
    start_idx = max(1, int(layer_range[0] * n_layers))
    end_idx = min(n_layers, int(layer_range[1] * n_layers))
    
    # Extract cho and rej (interleaved)
    hs_cho: Float[Tensor, "n_pairs layers d"] = hidden_states[::2]
    hs_rej: Float[Tensor, "n_pairs layers d"] = hidden_states[1::2]
    
    # First compute regular task_diff PCs (on mean diff across layers)
    task_diffs: Float[Tensor, "n_pairs d"] = (
        hs_cho[:, start_idx:end_idx+1, :] - hs_rej[:, start_idx:end_idx+1, :]
    ).mean(dim=1).float()
    
    # Per-sample normalize: each pair votes equally regardless of cho-rej magnitude.
    # Without this, pairs with large ||cho - rej|| dominate PCA.
    task_diffs_norm = normalize_rows(task_diffs)
    task_diffs_centered = task_diffs_norm - task_diffs_norm.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(task_diffs_centered, full_matrices=False)
    
    n_candidates = min(top_k * 3, Vh.shape[0])
    V_candidates: Float[Tensor, "d k"] = Vh[:n_candidates, :].T  # [d_model, n_candidates]
    
    # For each PC, compute magnitude separation trend across layers
    # |h_cho @ v| - |h_rej @ v| should increase for constructive directions
    proj_cho = (hs_cho.float() @ V_candidates).abs()  # [n_pairs, n_layers+1, n_candidates]
    proj_rej = (hs_rej.float() @ V_candidates).abs()
    separation = proj_cho - proj_rej  # positive = cho more aligned
    
    # Compute slope: early vs late separation
    early_sep = separation[:, start_idx:start_idx+3, :].mean(dim=(0, 1))  # [n_candidates]
    late_sep = separation[:, end_idx-3:end_idx, :].mean(dim=(0, 1))
    sep_slope = late_sep - early_sep  # positive = constructive (separation growing)
    
    # Also check that the direction is actually discriminative (|late_sep| > threshold)
    discriminative = late_sep.abs() > 0.01  # Nonzero separation
    
    # Select: constructive AND discriminative
    valid_mask = (sep_slope > 0) & discriminative
    if valid_mask.sum() < top_k:
        logger.warning(f"Only {valid_mask.sum()} constructive+discriminative PCs, taking {top_k} best")
        scores = sep_slope * late_sep.abs()  # Favor growing + large separation
        sorted_indices = torch.argsort(scores, descending=True)[:top_k]
    else:
        valid_indices = torch.where(valid_mask)[0]
        scores = sep_slope[valid_indices] * S[valid_indices]
        sorted_by_score = torch.argsort(scores, descending=True)[:top_k]
        sorted_indices = valid_indices[sorted_by_score]
    
    V_constructive: Float[Tensor, "d k"] = V_candidates[:, sorted_indices].to(dtype).to(device).detach()
    
    n_valid = ((sep_slope[sorted_indices] > 0) & (late_sep[sorted_indices].abs() > 0.01)).sum().item()
    logger.info(f"Task_diff_constructive subspace: rank={V_constructive.shape[1]}, {n_valid}/{top_k} constructive+discriminative")
    
    return Subspace(V_constructive, name="taskdiff_constructive")


def compute_task_diff_from_hidden_states(
    hidden_states: Float[Tensor, "batch n_layers_plus1 d_model"],
    top_k: int =256,
    layer_frac: float = 0.7,
    layer_range: Optional[tuple] = None,
    use_layer_diffs: bool = False,
) -> Subspace:
    """Compute task_diff subspace: PCA of task-discriminative directions.
    
    Expects hidden_states to have interleaved cho/rej pairs:
    [cho_0, rej_0, cho_1, rej_1, ...]
    
    Two modes:
    - use_layer_diffs=True (default): PCA of per-layer contributions that differ between cho/rej.
      Computes (delta_cho - delta_rej) where delta = h[l+1] - h[l]. This captures what each
      layer *adds* to the residual stream that distinguishes cho from rej. More specific.
      
    - use_layer_diffs=False: PCA of raw residual stream difference (h_cho - h_rej).
      The residual stream accumulates contributions, so this captures cumulative signal
      but is less layer-specific.
    
    Args:
        hidden_states: [batch, n_layers+1, d_model] with interleaved pairs
        top_k: Number of components
        layer_frac: Which layer fraction to use if layer_range is None (default 0.7)
        layer_range: Optional (start_frac, end_frac) tuple. Default (0.1, 0.95) covers
            most layers except embeddings. E.g. (0.4, 0.8) for planning layers only.
        use_layer_diffs: If True, use per-layer contributions (h[l+1]-h[l]). If False,
            use raw residual stream states.
        
    Returns:
        Subspace of task-discriminative directions
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    d_model = hidden_states.shape[-1]
    n_layers_plus1 = hidden_states.shape[1]
    n_layers = n_layers_plus1 - 1
    
    # Extract cho and rej (interleaved)
    hs_cho: Float[Tensor, "n_pairs layers d"] = hidden_states[::2]    # [n_pairs, n_layers+1, d_model]
    hs_rej: Float[Tensor, "n_pairs layers d"] = hidden_states[1::2]   # [n_pairs, n_layers+1, d_model]
    
    # Indexing convention:
    # - hidden_states[i] is AFTER layer i-1 (0 = embeddings)
    # - indices in this function (start_idx/end_idx) are hidden_states indices in [1, n_layers]
    #
    # If layer_range is provided, interpret it as (start_frac, end_frac) over the n_layers axis.
    # If layer_range is None, select a single layer based on layer_frac.
    if layer_range is None:
        layer_idx = int(layer_frac * n_layers)
        start_idx = max(1, min(layer_idx, n_layers))
        end_idx = start_idx
    else:
        start_idx = max(1, int(layer_range[0] * n_layers))  # At least layer 1 (skip embeddings)
        end_idx = min(n_layers, int(layer_range[1] * n_layers))
        # Make sure we never select an empty slice.
        end_idx = max(start_idx, end_idx)
    
    if use_layer_diffs:
        # Per-layer contributions: what each layer ADDS that differs between cho/rej
        # delta_cho[l] = h_cho[l+1] - h_cho[l], same for rej
        # task_contribution[l] = delta_cho[l] - delta_rej[l]
        delta_cho = hs_cho[:, 1:, :] - hs_cho[:, :-1, :]  # [n_pairs, n_layers, d]
        delta_rej = hs_rej[:, 1:, :] - hs_rej[:, :-1, :]  # [n_pairs, n_layers, d]
        
        # Slice to layer range (note: delta indices are off-by-one from hidden_states)
        task_contributions = (delta_cho - delta_rej)[:, start_idx-1:end_idx, :]  # [n_pairs, n_layers_range, d]
        
        # Average across layers: residual stream is cumulative, we want directions
        # that *consistently* separate cho/rej, not layer-specific noise.
        # [n_pairs, n_layers_range, d] -> [n_pairs, d]
        task_diffs = task_contributions.mean(dim=1).float()
        layer_desc = f"layer_diffs {start_idx}-{end_idx}"
    else:
        # Raw residual stream difference (cumulative)
        # Average across layers: residual stream accumulates, so layer-stacking just
        # gives PCA the same direction repeated with slight noise. We care about
        # directions that consistently separate, not trajectory evolution.
        # [n_pairs, n_layers_range, d] -> [n_pairs, d]
        task_diffs = (hs_cho[:, start_idx:end_idx+1, :] - hs_rej[:, start_idx:end_idx+1, :]).mean(dim=1).float()
        layer_desc = f"residual {start_idx}-{end_idx}"
    
    # PCA of task diffs with per-sample normalization
    # Each pair votes equally regardless of cho-rej magnitude.
    sub = pca_subspace(
        task_diffs,
        top_k=top_k,
        normalize_samples=True,
        name=f"task_diff({layer_desc})",
        device=device,
        dtype=dtype,
    )
    # Rename to canonical "taskdiff" for downstream code
    return Subspace(sub.V, name="taskdiff", S=sub.S)


def compute_task_read_subspace(
    task_diff_subspace: Subspace,
    read_subspace: Subspace,
    top_k: int = 256,
) -> Subspace:
    """Compute task_read subspace: task signal readable by transformer blocks.

    task_read = task_diff ∩ read

    These are task-discriminative directions that are read by attention/MLP inputs
    (q/k/v projections, up/gate projections, etc.).

    Args:
        task_diff_subspace: Subspace of task differences
        read_subspace: Subspace readable by residual readers
        top_k: Number of components

    Returns:
        Subspace of task signal that read-modules can read
    """
    task_read = approx_intersection(task_diff_subspace, read_subspace)

    if task_read.rank > top_k:
        task_read = Subspace(task_read.V[:, :top_k], name="taskdiff_read")
    else:
        task_read.name = "taskdiff_read"

    return task_read


def compute_task_lm_head_subspace(
    task_diff_subspace: Subspace,
    lm_head_subspace: Subspace,
    top_k: int = 256,
) -> Subspace:
    """Compute taskdiff_x_logits_read subspace: task signal readable by lm_head.

    taskdiff_x_logits_read = taskdiff ∩ logits_read

    These are task-discriminative directions that lm_head can read,
    i.e. they affect output logits.
    """
    taskdiff_logits_read = approx_intersection(task_diff_subspace, lm_head_subspace)

    if taskdiff_logits_read.rank > top_k:
        taskdiff_logits_read = Subspace(taskdiff_logits_read.V[:, :top_k], name="taskdiff_x_logits_read")
    else:
        taskdiff_logits_read.name = "taskdiff_x_logits_read"

    return taskdiff_logits_read


def compute_task_wnr_subspace(
    task_diff_subspace: Subspace,
    write_not_read_subspace: Subspace,
    top_k: int =256,
) -> Subspace:
    """Compute task_wnr subspace: task signal written but not read.
    
    task_wnr = task_diff ∩ write_not_read
    
    These are task-discriminative directions that are written to residual
    but not read by later layers or lm_head.
    
    Args:
        task_diff_subspace: Subspace of task differences
        write_not_read_subspace: Subspace of write-not-read directions
        top_k: Number of components
        
    Returns:
        Subspace of task signal that's written but ignored
    """
    taskdiff_write_not_read = approx_intersection(task_diff_subspace, write_not_read_subspace)

    if taskdiff_write_not_read.rank > top_k:
        taskdiff_write_not_read = Subspace(taskdiff_write_not_read.V[:, :top_k], name="taskdiff_x_write_not_read")
    else:
        taskdiff_write_not_read.name = "taskdiff_x_write_not_read"

    return taskdiff_write_not_read


def compute_module_subspace_from_svds(
    layer_svds: Dict[str, tuple],
    layer_info: Dict[str, dict],
    module_filter: List[str],
    use_column_space: bool = True,
    top_k: int =256,
    device: torch.device = None,
    dtype: torch.dtype = None,
    name: str = "module",
    weight_by_singular_values: bool = False,
) -> Optional[Subspace]:
    """Generic function to compute subspace from specific module types.
    
    Args:
        layer_svds: Dict mapping layer names to (U, S, Vh) tuples
        layer_info: Dict with module metadata
        module_filter: List of module names to include (e.g., ['o_proj'])
        use_column_space: If True, use U (write/output). If False, use V (read/input).
        top_k: Number of components
        device, dtype: Output tensor properties
        name: Subspace name
        weight_by_singular_values: If True, weight basis vectors by sqrt(S/sum(S))
            so high-energy directions dominate. If False (default), all modules
            contribute equally (democratic). Use True for "what IS written",
            False for "what CAN be written".
        
    Returns:
        Subspace from specified modules, or None if no modules matched
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    
    # First pass: find d_model from matching modules
    d_model = None
    for path, (U, S, Vh) in layer_svds.items():
        module_name = layer_info[path]['module_name']
        if module_name not in module_filter:
            continue
        # For write (column space), d_model = U.shape[0] (output dim)
        # For read (row space), d_model = Vh.shape[1] (input dim)
        if use_column_space:
            this_dim = U.shape[0]
        else:
            this_dim = Vh.shape[1]
        if d_model is None:
            d_model = this_dim
        # Only use modules with consistent dimension
        if this_dim != d_model:
            logger.debug(f"Skipping {path} with dim {this_dim} != d_model {d_model}")
            continue
    
    if d_model is None:
        logger.info(f"No modules matched filter {module_filter} in layer_svds - skipping {name} subspace")
        return None
    
    P_sum: Float[Tensor, "d d"] = torch.zeros(d_model, d_model, dtype=torch.float32, device='cpu')

    for path, (U, S, Vh) in layer_svds.items():
        module_name = layer_info[path]['module_name']
        if module_name not in module_filter:
            continue
        
        if use_column_space:
            # Column space from U - skip if wrong dimension
            if U.shape[0] != d_model:
                continue
            k = U.shape[1] if top_k is None else min(int(top_k), U.shape[1])
            basis: Float[Tensor, "d r"] = U[:, :k].float().cpu()
        else:
            # Row space from Vh.T - skip if wrong dimension  
            if Vh.shape[1] != d_model:
                continue
            k = Vh.shape[0] if top_k is None else min(int(top_k), Vh.shape[0])
            basis: Float[Tensor, "d r"] = Vh.T[:, :k].float().cpu()
        
        P_sum += basis @ basis.T
    
    U_result, _, _ = torch.linalg.svd(P_sum)
    k_final = d_model if top_k is None else min(int(top_k), d_model)
    V_result: Float[Tensor, "d k"] = U_result[:, :k_final].to(dtype).to(device).detach()
    
    logger.info(f"{name} subspace (from {module_filter}): rank={V_result.shape[1]}")
    
    return Subspace(V_result, name=name)
