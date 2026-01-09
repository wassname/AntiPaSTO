# AntiPaSTO: Deep-Dish Inner Alignment

*Serving up parameter-efficient inner alignment, one rotation at a time.*

**Anti-Pa**rallel **S**ubspace **T**raining for **O**rdered steering.

## Quick Start

```sh
uv sync --all-groups
uv run python nbs/train.py tiny --quick  # taste test
uv run python nbs/train.py               # full course (gemma-3-1b-it)
```

## The Recipe

RLHF seasons the outputs but leaves the internals bland. AntiPaSTO marinates the model's hidden states directly, installing a reversible behavioral mode via gradient-based optimization in SVD space.

![Incomplete contrast pairs](docs/img/incomplete_contrast_pairs_v2.svg)

**Ingredients**:
- Incomplete contrast pairs (self-supervised, no preference labels)
- Cayley rotations on V (the secret sauce)
- Projection loss + TV coherence + monotonicity constraints
- 800 synthetic pairs, ~1hr on A100

**What you get**:
- Single adapter, flip the coefficient sign to reverse the effect
- Train on honesty, transfers to 1,360 moral dilemmas (9 value dimensions)
- 3.1x Steering F1 vs prompting on gemma-3-1b-it
- Suppression bypass: steers when prompting triggers refusal

## Architecture


```python
# Adapter: rotate in SVD space
def forward(h, alpha):
    R_v = cayley(theta_v, alpha)  # coefficient-scaled rotation
    S_scaled = S + alpha * delta_S
    return h @ W_res.T + h @ V @ R_v @ diag(S_scaled) @ U.T

# Loss: antiparallel separation + coherence + ordering
def loss(model, x_cho, x_rej):
    delta_pos = model(x_cho, +1) - model(x_rej, +1) - d_ref
    delta_neg = model(x_cho, -1) - model(x_rej, -1) - d_ref
    
    L_proj = symlog(delta_pos @ delta_neg)        # want < 0 (antiparallel)
    B_coh = tv_barrier(p_ref, p_pi, entropy)      # TV trust region
    B_mono = hinge(Delta_neg < 0 < Delta_pos)     # ordered control
    
    return L_proj + B_coh + B_mono
```
![Adapter architecture](docs/img/apastoadapter_architecture.svg)


![Loss geometry](docs/img/loss.svg)

## Project Layout

```
antipasto/           # the kitchen
  config.py          # canonical defaults
  metrics.py         # how we measure flavor
  train/             # cooking instructions
  peft_utils/        # adapter mechanics
docs/                # diagrams, notes
nbs/                 # experiments
outputs/adapters/    # trained models
```

## Status

Full research history (experiments, ablations, dead ends) available on request.

## Citation

```bibtex
@misc{clark2025AntiPaSTO,
  title = {AntiPaSTO: Self-Supervised Steering of Moral Reasoning},
  author = {Clark, Michael J},
  year = {2025},
  url = {https://github.com/wassname/AntiPaSTO/}
}
```
