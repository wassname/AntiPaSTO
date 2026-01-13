from attrs import define
from pathlib import Path
from math import pi
from typing import List, Literal, Optional
proj_root = Path(__file__).parent.parent.resolve()

PROMPT= "Pretend you're {persona} person. "
PERSONAS = [[
    "an honest",
], [
    "a dishonest",
]]

# Models to evaluate for baselines (prompting, repeng, etc.)
# Keep in sync with justfile run-models + sweep-train-stages
EVAL_BASELINE_MODELS = [
    # run-models (Table 1: Cross-model)
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    
    # sweep-train-stages (Table 2: OLMo training stages)
    "allenai/Olmo-3-1025-7B",      # Base
    "allenai/Olmo-3-7B-Instruct-SFT",  # SFT
    "allenai/Olmo-3-7B-Instruct-DPO",  # DPO
    "allenai/Olmo-3-7B-Instruct",      # RLVR
    "allenai/Olmo-3-7B-Think-SFT",     # Think SFT
    "allenai/Olmo-3-7B-Think-DPO",     # Think DPO
    "allenai/Olmo-3-7B-Think",         # Think RLVR
]

@define(slots=False)
class TrainingConfig:
    """Configuration for training contrastive AntiPaSTO adapter."""

    seed: int = 42
    """Random seed for reproducibility (layer selection, dim selection, training dynamics)."""
    
    init_n_samples: int = 1000
    """Number of samples for WANDA-style dimension selection and subspace computation.
    
    Higher = more stable activation statistics, but slower init.
    CORDA recommends: hidden_dim / tokens_per_sample * 128 (e.g., 1152 for 12B).
    2000 is safe for models up to ~12B with 512-token sequences.
    Uses first N samples (deterministic with data_seed).
    """

    data_seed: int = 42
    """Fixed seed for data selection (which suffixes are used)."""

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    quantization_type: Literal["4bit", "8bit", "none"] = "none"

    n_modules: int = 512
    """Total number of layer×module combinations to select (by gradient importance).
    
    Examples with n_modules=5:
    - layers.2.q_proj, layers.5.q_proj, layers.3.down_proj, layers.5.down_proj, layers.1.o_proj
    
    The selection is sparse: each layer×module is ranked by gradient, top-k are selected.
    Not Cartesian product - can have multiple modules at one layer, none at another.
    
    Default 42 (≈14 layers × 3 modules for typical 36-layer model).
    """

    target_modules: List[str] = ["residual-writers"]
    """Module names to consider for adapter placement (gradient-based selection).
    
    Special values (single-element list, auto-detected from model architecture):
    - ["residual-writers"]: Modules that write TO residual stream (output_dim = hidden_size, 
        input_dim != hidden_size). E.g., o_proj, down_proj. Default and recommended.
    - ["residual-readers"]: Modules that read FROM residual stream (input_dim = hidden_size).
        E.g., q_proj, k_proj, v_proj, gate_proj, up_proj.
    - ["residual-all"]: All residual-connected modules (input OR output = hidden_size).
    
    Explicit list: ["down_proj", "o_proj"] - only these module suffixes are candidates.
    """

    bs: int = 8
    """Batch size"""

    n_epochs: int = 20

    lr: float = 5e-4
    """Learning rate. Sweep findings: 1e-4 too low (-18 F1), 3e-5 too low (-17 F1).
    For Cayley: 4e-4 to 6e-4. For LoRA/DoRA: ~10x lower (3e-5 to 6e-5).
    """

    wd: float = 1e-8
    """Weight decay"""

    n_logs: int = 10
    """Log this many times per training"""

    val_every_n_samples: int = 512
    """Validate every N training samples (independent of logging)."""

    effective_bs: int = 32
    """Effective batch size via gradient accumulation"""

    quick: bool = False
    """Quick mode for debugging"""

    val_split: float = 0.15
    """Fraction of data for validation"""

    early_stop_patience: int = 11
    """Stop if val loss doesn't improve for N validation checks. 0 = disabled (recommended with one-cycle scheduler)."""

    early_stop_min_delta: float = 0.00001
    """Min relative improvement to count as 'better' (0.001 = 0.1%). Filters noise without being too strict."""

    warmup_pct: float = 0.1
    """Fraction of training for warmup. Early stopping is disabled during warmup."""

    r: int = 64
    """Adapter rank (ideally should be proportional to hidden dim)"""

    rot_u: bool = False
    """Rotate U (output space). Less stable, diverges from loss space"""

    rot_v: bool = True
    """Rotate V (input space). Quite stable and expressive"""

    dim_select_method: Literal["top_s", "random", "wanda_svd_l1_trip"] = "wanda_svd_l1_trip"
    """Method for selecting which SVD dimensions to include in ADAPTER.
    
    Controls which r dimensions (out of full rank) are selected for each layer.
    This is DECOUPLED from loss_subspace - adapter needs reconstruction capacity,
    while loss only guides toward task direction.
    
    - wanda_svd_l1_trip (default): Three-way split: cho + rej + diff.
        Takes r/3 from cho ranking, r/3 from rej ranking, r/3 from |diff| ranking.
        Explicitly includes dims aligned with the steering direction.
        Requires tokenizer/dataset for forward pass.
    
    - top_s: Top-r by singular value magnitude (like PiSSA).
        Preserves maximum reconstruction quality. No forward pass needed.
    
    - random: Random selection of r dims. Sanity check baseline.
    """
    
    max_rotation_angle: float = pi/4.
    """Max rotation angle (rad). Keeps subspace ~70% overlap with original."""

    loss_subspace: Literal[
        # Recommended (default)
        "taskdiff_x_suppressed_x_write",  # Task-discriminative ∩ suppressed ∩ write
        # Simpler alternatives
        "write",  # Write space only (o_proj, down_proj column space)
        "taskdiff",  # Task-discriminative PCA only
    ] = "taskdiff_x_suppressed_x_write"
    """Subspace for loss projection. Projects hidden-state deltas to this subspace.
    
    Default: taskdiff_x_suppressed_x_write = intersection of:
    - taskdiff: PCA on cho-rej difference (task-discriminative directions)
    - suppressed: Written to residual but erased by later layers  
    - write: Column space of o_proj and down_proj (writable by model)
    """

    loss_subspace_rank: Optional[int] = 8
    """Rank (top-k) for loss subspace.

    If None (default), select rank automatically via `loss_subspace_energy_frac`
    using the cached subspace singular values (MSRS-style energy thresholding).
    
    If explicit int, use that rank directly (low rank 2-8 works best empirically).
    """

    loss_subspace_energy_frac: float = 0.6
    """Energy fraction for automatic loss subspace rank selection.

    Only used when `loss_subspace_rank is None`. The selected rank is the smallest
    k such that cumulative energy >= this fraction. 60% was used in MSRS paper
    """

    loss_layer_frac: float = 0.9
    """Depth fraction (0-1) at which to apply representation loss.
    
    The loss is computed at a single layer: int(loss_layer_frac * num_hidden_layers).
    
    Default 0.8 (80% depth) is in the "planning zone" where Fisher ratio and
    cross-sample consistency peak across tested architectures (Qwen, Gemma). Also suppurted by supported by e.g 2024-Gurnee-Universal-Neurons-in-GPT2-Language-Models.md
    
    Rationale: gradient-based layer selection was circular (gradients flow FROM
    the loss, so using them to pick where to PUT the loss is self-fulfilling).
    Simple fixed depth is more robust and generalizes across models.
    """

    min_adapter_layer_frac: float = 0.1
    """Minimum depth fraction (0-1) for adapter placement.
    
    Adapters will only be placed on layers >= int(min_adapter_layer_frac * num_hidden_layers).
    
    Default 0.2 (20% depth) excludes early layers which:
    1. Process raw embeddings and can be unstable for gradient-based selection
    2. Could allow "loss hacking" by modifying very early representations
    3. Have less task-relevant signal (mostly syntactic/shallow processing)
    
    For large models with limited capacity:
    - Set min_adapter_layer_frac=0.4, loss_layer_frac=0.8 to focus on middle-to-late layers
    - This concentrates adapters in the "planning zone" (40-80% depth)
    - Heuristic based on Fisher ratio peaks at 50-70% depth across architectures
    
    Combined with loss_layer_frac, adapters are placed in range [min_adapter_layer_frac, loss_layer_frac).
    """

    dataset_name: str = "honest"

    max_samples: Optional[int] = 800
    """Max training samples (None = all)"""

    n_last_tokens: int = 3
    """Extract from last N tokens of sequence"""

    coh: bool = True
    """Enable coherence constraint.
    
    WARNING (2026-01-01 sweep): coh hurts when threshold is too tight (2.4 vs 31.6 without).
    If enabling, use loose threshold (0.8-2.0 not 0.4). Currently disabled by default.
    """

    coh_weight: float = 10.0
    """Coherence loss scaling.
    
    With log_barrier: scale=50 gives penalty=18 at TV=0.55, → ∞ at TV=1.0
    Hard wall ensures coherence is never violated at extreme TV.
    """

    coh_thresh: float = 0.9
    """TV threshold = α × √(H + floor). This is α.
    
    With α=0.3, floor=0.02, H≈4 nats: threshold ≈ 0.3 × √4.02 ≈ 0.6 (60% mass can move).
    Sublinear in entropy: tight on confident tokens, loose on uncertain.
    
    IMPORTANT: Since TV ∈ [0, 1], values > 1.0 effectively disable coherence
    (threshold exceeds max possible violation). Values > 1.0 crash.
    
    Sweep findings (2026-01-06):
    - gemma4b: coh_thresh=0.9 → 2.3 F1 (best), manually verified coherent
    - gemma1b: coh_thresh=0.8 → 16.9 F1, no_coh → 9.3 F1 (+7.6 improvement)
    - Pattern: moderate-loose (0.8-0.9) > tight (0.5) > too_loose (no_coh) > too_tight (0.0)
    - Recommend: coh_thresh=0.9 (default)
    """

    coh_barrier_mode: Literal["log1p_squared"] = "log1p_squared"
    """Barrier function for coherence violations:
    - log1p_squared: log(1+v)². Grows slowly, doesn't fight projection loss.
    """

    coh_lse_temperature: float = 3.0
    """LSE temperature τ for coh_agg_mode='lse'.

    τ→0 behaves like max (spiky). Larger τ spreads gradients across multiple bad tokens.
    """

    mono: bool = True
    """Enable monotonicity constraint.
    
    2026-01-01 sweep: +22 points with warmup (53.6 vs 31.6 without). Warmup critical.
    Projection loss naturally creates ordering; mono is a safety rail, not driver.
    """

    mono_margin: float = 0.4
    """Monotonic threshold_frac: fraction of √H_ref for minimum separation.
    
    Threshold = threshold_frac × √H_ref + threshold_floor.
    With H_ref=4 nats (typical), threshold_frac=0.4, floor=0.04: threshold ≈ 0.84 nats.
    
    Sweep findings (2026-01-07, gemma1b):
    | margin | F1    |
    | 0.4    | 23.6  | ← current default
    | 0.2    | 17.2  |
    | 0.25   | 0.0   | (collapsed)
    
    This is the MINIMUM separation required from zero (deadzone). Constraint:
    delta_neg < -threshold < 0 < +threshold < delta_pos (or reversed).
    
    Note: Mono is a soft constraint - may not be fully satisfied but keeps
    endpoints on opposite sides of baseline.
    """

    mono_threshold_floor: float = 0.04
    """Absolute minimum threshold in nats (prevents explosion on very confident tokens).
    
    With floor=0.02, even when H_ref→0 (very confident), threshold ≥ 0.02 nats.
    Prevents division issues and provides small stable deadzone.
    """
    
    mono_weight: float = 20.0
    """Monotonicity loss scaling.
    
    WARNING: Values ≥100 trap adapters in bad init - can't learn "no change" at c=0.
    Keep mono_weight < coh_weight to let projection loss dominate early training.
    
    Dec 2024 variance analysis: mono_weight=100 gave 50% CV on flip%, mono_weight=10-20 is stable.
    """
    
    mono_warmup_frac: float = 0.5
    """Fraction of training before mono loss kicks in. During warmup, mono_weight=0.
    
    - -1.0: Follow LR warmup (use warmup_pct)
    - 0.0: No warmup, mono active from start
    - 0.5 (default): Mono kicks in after 50% of training
    
    At symmetric init, mono is satisfied (both endpoints at baseline) → zero gradient
    OR mono fights projection before direction established → saddle trap.
    Long warmup (50%) lets projection find direction first, then mono enforces.
    """

    coh_warmup_frac: float = -1
    """Fraction of training before coherence loss kicks in. During warmup, coh disabled.
    
    - -1.0: Follow LR warmup (use warmup_pct)
    - 0.0: No warmup, coh active from start
    - 0.2: Coh kicks in after 20% of training
    
    2026-01-05 sweep finding: coh=False outperforms coh=True by +5-14 F1.
    Likely the same great-wall problem as mono: coherence fights projection early
    before the adapter has found its steering direction. Warmup lets projection
    establish antisymmetry first, then coherence provides soft guardrails.
    """

    orth_weight: float = 0
    """Orthogonal penalty weight (0.0 = disabled).
    
    Penalizes energy not aligned with shared antiparallel axis.
    
    Sweep findings (2026-01-05):
    - 0.03+: Kills training (-48 F1)
    - 0.01: Borderline (+15 F1)
    - 0.001: Safer (+27 F1)
    - 0.0: Default, works when using antisym_mode=align or antisym_norm=delta_full
    
    Interactions: Redundant with antisym_mode=align (both constrain direction).
    Fully redundant with antisym_norm=delta_full (implicit concentration).
    For LoRA, prefer orth_weight for regularization.
    """
    

    # Loss uses Fisher t-statistic normalization (batch-level t = mu/sqrt(var) per dim)
    # with align mode (constrains steering to reference axis) and delta_full normalization
    # (penalizes energy outside loss subspace). See docs/loss_intuition2.md.

    focus_softness: float = 0.25
    """Softening exponent for subspace FOCUS weighting in delta_full align mode.
    
    When using delta_full normalization, we weight each cosine by how much
    of the delta energy lies in the loss subspace: focus = ||δ_proj|| / ||δ_full||.
    This penalizes energy outside the loss subspace.
    
    This parameter raises focus to power (1 - focus_softness):
        focus_used = focus^(1 - focus_softness)
    
    Values:
    - 0.0: Raw focus ratio. focus=0.1 → 0.1. Strict subspace focus.
    - 0.5: sqrt(focus). focus=0.1 → 0.32. Moderate penalty for out-of-subspace.
    - 1.0: Ignore focus entirely.
    
    Recommended: 0.25 for AntiPaSTO (Cayley).
    """

    antisym_margin: float = 0.0
    """Margin for antisymmetry loss (dimensionless, relative to ref² baseline)."""

    fisher_var_floor_frac: float = 0.1
    """Variance floor as fraction of median std across dims (prevents t-explosion).
    
    When some dimensions have near-zero variance, t = mu/std can explode.
    This floor caps |t| by ensuring std >= floor_frac * median(std).
    Lower = more sensitive to low-variance dims, higher = more conservative.
    """

    fisher_abs_std_floor: float = 0.05
    """Absolute minimum std floor (prevents t-explosion with small batches).
    
    With batch=8, variance estimates are noisy. This absolute floor ensures
    |t| <= mu/0.05 = 20*mu max per dimension regardless of batch noise.
    """

    fisher_detach_std: bool = True
    """Detach std in t = mu/std to prevent zero-variance hacking.
    
    If True (legacy): gradients only flow through mu, model can't learn to
    reduce variance. Simpler but can't reward consistent separation.
    
    If False (default): gradients flow through both mu and std. Model can
    learn to reduce variance in useful dimensions. Variance floors prevent
    gaming by capping max |t|.
    """

    eval_max_dilemmas: Optional[int] = None
    """Max eval dilemmas (None = all)"""

    eval_max_tokens: int = 288
    """Max tokens for eval sample (cropped above this)"""

    output_dir: Path = proj_root / "outputs/adapters"
    
    experiment_name: Optional[str] = None
    """Custom name (auto-generated if None)"""

    use_wandb: bool = True
    wandb_project: str = "AntiPaSTO"
    wandb_tags: Optional[List[str]] = None
    """Tags for organizing WandB runs"""

    verbose: int = 1
    """Logging verbosity: 0=warning, 1=info (default), 2=debug"""


    PROMPT: str = PROMPT
    PERSONAS: List[List[str]] = PERSONAS

    def __attrs_post_init__(self):
        """Validate config constraints after initialization."""
        # Validate layer fractions are in valid range
        if not 0.0 <= self.min_adapter_layer_frac < 1.0:
            raise ValueError(
                f"min_adapter_layer_frac={self.min_adapter_layer_frac} must be in [0, 1). "
            )
        if not 0.0 < self.loss_layer_frac < 1.0:
            raise ValueError(
                f"loss_layer_frac={self.loss_layer_frac} must be in (0, 1). "
                f"Represents depth fraction for loss layer."
            )
        
        # Validate coh_thresh: values > 1.0 make no sense (TV is in [0, 1])
        if self.coh and self.coh_thresh > 1.0:
            raise ValueError(
                f"coh_thresh={self.coh_thresh} > 1.0 is invalid. "
                f"TV is in [0, 1], so threshold > 1 disables coherence entirely. "
                f"Use --no_coh instead, or set coh_thresh <= 0.5 for meaningful constraint."
            )

    @property
    def eval_batch_size(self):
        return self.bs // 2
    
    def get_experiment_name(self) -> str:
        """Generate experiment name: {model_short}-antisym-r{rank}[-{variations}].
        
        Examples: qwen34b-antisym-r24, qwen06b-antisym-r48-urot, gemma12b-antisym-r24-noV
        """
        if self.experiment_name:
            return self.experiment_name
        
        # Shorten model name (critical - shows in truncated view)
        model_map = {
            # Qwen
            'Qwen3-0.6B': 'q06b',
            'Qwen3-4B': 'q4bv1',
            'Qwen3-4B-Base': 'q4bbase',
            'Qwen3-4B-Instruct-2507': 'q4b',
            'Qwen3-14B': 'q14b',
            'Qwen3-32B': 'q32b',
            'qwen3-5lyr-tiny-random': 'rnd',
            # Llama
            'Llama-3.1-8B-Instruct': 'l8b',
            'Llama-3.3-70B-Instruct': 'l70b',
            # Gemma
            'gemma-3-270m-it': 'g270m',
            'gemma-3-1b-it': 'g1b',
            'gemma-3-4b-it': 'g4b',
            'gemma-3-12b-it': 'g12b',
            'gemma-3-27b-it': 'g27b',
            # OLMo
            'Olmo-3-1025-7B': 'olmo7b',
            'Olmo-3-7B-Instruct-SFT': 'olmo7b-sft',
            'Olmo-3-7B-Instruct-DPO': 'olmo7b-dpo',
            'Olmo-3-7B-Instruct': 'olmo7b-i',
            'Olmo-3-7B-Think-SFT': 'olmo7bt-sft',
            'Olmo-3-7B-Think-DPO': 'olmo7bt-dpo',
            'Olmo-3-7B-Think': 'olmo7bt',
            'Olmo-3-7B-RL-Zero-General': 'olmo7b-rl0',
            # Other
            'gpt-oss-20b': 'oss20b',
        }
        model_part = self.model_name.split('/')[-1]
        model_short = model_map.get(model_part, model_part[:8].replace('-', '').lower())
        
        # Loss is now always antisymmetric (no pref_dir)
        loss_short = "antisym"
        
        # Start with critical info
        parts = [model_short, loss_short, f"r{self.r}"]
        
        # Fields already encoded in base name or to skip
        skip_fields = {
            'model_name', 'r', 'experiment_name', 'output_dir', 'use_wandb', 
            'wandb_project', 'wandb_tags', 'save_checkpoints', 'verbose',
            'PROMPT', 'PERSONAS', 'quick', 'n_logs', 'val_every_n_samples',
            'eval_max_dilemmas', 'eval_max_tokens', 'bs', 'effective_bs',
            'n_epochs', 'val_split', 'early_stop_patience', 'max_samples',
            'wd', 'quantization_type',
        }
        
        # Short names for variation keys
        key_short = {
            'loss_mode': 'lm', 'rot_u': 'urot', 'rot_v': 'vrot',
            'n_modules': 'M', 'lr': 'lr', 'coh': 'coh', 'mono': 'mono',
            'orth_weight': 'orth',

            'dataset_name': 'ds', 'n_last_tokens': 'tok', 
            'coh_weight': 'cohW', 'coh_thresh': 'cohK',
            'mono_margin': 'monoM', 'mono_weight': 'monoW',
            'antisym_margin': 'amrg',
            'max_rotation_angle': 'maxR',
            'loss_subspace': 'lsub', 'loss_subspace_rank': 'lsubR',
            'loss_layer_frac': 'lf',
            'min_adapter_layer_frac': 'malf',
            'target_modules': 'tgt',
        }
        
        import attrs
        defaults = TrainingConfig()
        variations = []
        
        for field in attrs.fields(TrainingConfig):
            k = field.name
            if k in skip_fields:
                continue
            v = getattr(self, k)
            dv = getattr(defaults, k)
            if v != dv:
                short = key_short.get(k, k[:4])
                # Format value compactly
                if isinstance(v, bool):
                    variations.append(short if v else f"no{short}")
                elif isinstance(v, float):
                    if v == int(v):
                        variations.append(f"{short}{int(v)}")
                    elif abs(v) < 0.01 or abs(v) >= 100:
                        variations.append(f"{short}{v:.0e}".replace('e-0', 'e-'))
                    else:
                        variations.append(f"{short}{v:.2g}")
                elif isinstance(v, str):
                    variations.append(f"{short}_{v[:4]}" if short else v)
                elif isinstance(v, (list, tuple)):
                    variations.append(f"{short}{len(v)}")
                else:
                    variations.append(f"{short}{v}")
        
        if variations:
            parts.append('-'.join(variations))
        
        return '-'.join(parts)

    @property
    def grad_accum_steps(self):
        return max(1, self.effective_bs // self.bs)


# Preset configs for different hardware/model combinations https://brentyi.github.io/tyro/examples/hierarchical_structures/
default_configs = {
    ".": ("default", TrainingConfig()),

    # These models are too small for reliable results
    "rnd": (
        "Tiny random model 2 layers (debugging/CI)",
        TrainingConfig(
            # google/gemma-3-270m-it
            model_name="wassname/qwen3-5lyr-tiny-random",
            quick=True,
        ),
    ),
    "tiny": (
        "Tiny 18 layers (500mb)",
        TrainingConfig(
            model_name="google/gemma-3-270m-it",
            quick=True,
        ),
    ),
    "q06b-24gb": (
        "Qwen 0.6B on 24GB GPU (fast iteration)",
        TrainingConfig(
            model_name="Qwen/Qwen3-0.6B",
            bs=24,
        ),
    ),


    # larger models
    "q4b-24gb": (
        "Qwen 4B on 24GB GPU (balanced quality/speed)",
        TrainingConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            bs=6,
        ),
    ),

    "q4b-80gb": (
        "Qwen 4B on 80GB GPU (large batch training)",
        TrainingConfig(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            bs=32,
        ),
    ),


    # google/gemma-3-270m-it
    "gemma270m-80gb": (
        "Gemma 3 270m on 80GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-270m-it",
            bs=64,
        ),
    ),
    "gemma1b-80gb": (
        "Gemma 3 1B on 80GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-1b-it",
            bs=64,
        ),
    ),
    "gemma1b-24gb": (
        "Gemma 3 1B on 24GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-1b-it",
            bs=24,
        ),
    ),
    "gemma4b-80gb": (
        "Gemma 3 4B on 80GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-4b-it",
            bs=64,
        ),
    ),
    # add gemma4b
    "gemma12b-80gb": (
        "Gemma 3 12B on 80GB GPU",
        TrainingConfig(
            model_name="google/gemma-3-12b-it",
            bs=4,
        ),
    ),

    # google/gemma-3-27b-it


}
