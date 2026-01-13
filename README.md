# 🍝 AntiPaSTO: Self-Supervised Steering of Moral Reasoning

[PAPER](https://arxiv.org/search/?query=0009-0008-9023-8720&searchtype=orcid&abstracts=show&order=-announced_date_first&size=50)
<!-- TODO update with arxiv link -->

**Anti-Pa**rallel **S**ubspace **T**raining for **O**rdered steering.

*Serving up data-efficient inner alignment, one satisfying rotation at a time.*

Gradient-based steering in SVD transformation space, trained on internal representations without preference labels. Human input: two contrasting words ("honest" vs "dishonest"). Transfers out-of-distribution to moral dilemmas where prompting fails.

## Quick Start

```sh
uv sync --all-groups
uv run python nbs/train.py tiny --quick  # al dente check
# Training complete. Final loss: -2.9062

uv run python nbs/train.py               # full course (Gemma-3-1B)
``` 

## The Recipe

RLHF seasons the outputs but leaves the internals bland. AntiPaSTO marinates the model's hidden states directly—no preference labels required, just two contrasting words simmered into 800 synthetic pairs.

![Incomplete contrast pairs](docs/img/incomplete_contrast_pairs_v2.svg)

**Ingredients**:
- Incomplete contrast pairs (self-supervised, no labels to garnish)
- Cayley rotations on V (the secret sauce—keeps everything orthogonal)
- Projection loss + TV coherence + monotonicity constraints
- 800 synthetic pairs, ~1hr on A100 (low simmer)

**What you get**:
- Single adapter—flip α from +1 to -1 to reverse the flavor
- Train on honesty, transfers to 1,360 moral dilemmas (9 value dimensions)
- Beats prompting on small models (≤4B); complements arithmetic steering methods
- Suppression bypass: steers when prompting triggers refusal or meta-commentary

## Architecture

*The pasta machine: SVD decomposition + Cayley rotations*


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

![Bidirectional control](docs/img/fig_bidirectional_demo.svg)

![Loss geometry](docs/img/loss.svg)

## Project Layout

```
antipasto/           # the kitchen
  config.py          # canonical recipe
  metrics.py         # taste testing
  train/             # cooking instructions
  peft_utils/        # pasta machine internals
docs/                # diagrams, plating notes
nbs/                 # experimental dishes
outputs/adapters/    # trained models (ready to serve)
```

## Status

*Still simmering.* Full research history (experiments, ablations, burnt batches) available on request.

## Acknowledgments

Built on the shoulders of:
- [RepEng](https://github.com/vgel/repeng) — arithmetic steering that inspired this gradient-based approach
- [PiSSA](https://github.com/GraphPKU/PiSSA) — SVD-based adapter initialization
- [SSVD](https://arxiv.org/abs/2409.07268) — rotating V for domain generalization
- [PEFT](https://github.com/huggingface/peft) — the adapter ecosystem
- [DailyDilemmas](https://github.com/chrischiu/dailydilemmas) — the evaluation benchmark

## Citation

```bibtex
@misc{clark2025antipasto,
  title = {AntiPaSTO: Self-Supervised Steering of Moral Reasoning},
  author = {Clark, Michael J.},
  year = {2025},
  eprint = {2501.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2501.XXXXX}
}
```

*arXiv ID pending (submitted, awaiting publication)*
