# scale-bn-demo — how a BatchNorm buffer silently breaks a 3D retrieval model

A self-contained, CPU-runnable reproduction of a subtle failure mode: in a multi-modal
point-cloud encoder, the `BatchNorm` after the first conv layer can have its `running_var`
buffer poisoned by a handful of extreme-scale training samples — causing ~30-point
eval-metric swings between adjacent checkpoints while every training-loss curve stays
perfectly healthy.

This repo generates synthetic CAD-like data, reproduces the failure inside a faithful
mini-encoder, and shows **four** configurations — the bug, and three fixes — so you can see
exactly *why* it happens and *why* each fix works.

```bash
pip install -r requirements.txt
python run_demo.py            # ~a few minutes on CPU
python run_demo.py --quick    # fast smoke run
```

Outputs the figures in `figures/`. Three companion scripts add the rest:

```bash
python symptom_selfcontained.py # smooth loss / spiky grad / see-sawing MRR (grad-accum)
python ablation_anchor.py       # the leak ablation: remove the color+bias anchor
python investigation_story.py   # the checkpoint-diff "breakthrough" figure
python dissect_outlier_step.py  # per-micro-batch receipts for one poisoned step
```

---

## The bug in one paragraph

The point clouds are normalized by a **single dataset-wide divisor** (so that a 3 mm part and
a 4 mm part stay distinguishable — absolute scale is a feature). But that divisor has **no
per-sample clamp**, so a giant outlier (a meter-scale building, or a unit-error upload) keeps
enormous coordinates after "normalization". The first conv layer turns that into enormous
activations, and `BatchNorm`'s `running_var` — an EMA of per-batch variance, **not** gradient-driven —
absorbs the spike. The loss never moves (BN renormalizes within the batch), but at eval time
`BN(x) = (x - running_mean)/sqrt(running_var)` now divides clean inputs by a hugely inflated
variance → the embedding collapses → retrieval tanks. Whichever checkpoint you happen to save
right after an outlier-heavy batch looks broken; the next one looks fine.

## The four variants

| variant | normalization | norm layer | scale info | expected |
|---|---|---|---|---|
| `global+BN` | shared divisor | BatchNorm | in coordinates | **the bug**: running_var explodes, eval volatile |
| `global+GN` | shared divisor | GroupNorm | in coordinates | **stable** — GN has no running stats (train≡eval) |
| `unitbb+GN` | per-sample unit box | GroupNorm | *erased* | stable, but **can't tell 3 mm from 4 mm** |
| `unitbb+GN+FiLM` | per-sample unit box | GroupNorm | explicit conditioning | **stable AND scale-aware** |

The first three isolate the mechanism; the fourth is the full proposed fix — decouple **shape**
(scale-invariant geometry) from **scale** (an explicit `log(size) → Fourier → MLP → FiLM`
conditioning path), so nothing rides on raw activation magnitudes.

### Measured results (1600 steps, CPU; your numbers will vary slightly)

| variant | retrieval MRR | 3mm-vs-4mm margin | running_var | read |
|---|---|---|---|---|
| `global+BN` | **swings 0.03 ↔ 0.34** | 0.29 | **oscillates 0.006 ↔ 2.7** | learns a good baseline (rides with GroupNorm most checkpoints), but collapses to random whenever an outlier batch just poisoned the buffer — several sharp drops over the run |
| `global+GN` | 0.34 (stable) | 0.46 | — (no buffer) | **stable**; keeps usable scale — GroupNorm alone already helps |
| `unitbb+GN` | 0.10 (stable) | **0.00** | — | stable but **scale-blind**, and retrieval suffers because the teacher rewards size |
| `unitbb+GN+FiLM` | **0.38** (stable) | **0.91** | — | **best on both** — stable, and far better size discrimination |

Two takeaways the demo makes concrete:
1. `global+BN` reaches the *same* clean-checkpoint MRR as GroupNorm (~0.34) — the model trains fine —
   but its eval metric is **anti-correlated with `running_var`**: adjacent checkpoints swing
   ~30 points (0.34 → 0.03) purely on whether a scale-outlier batch recently poisoned the buffer.
   That is the "metric trends up, then a massive drop" lottery, with the loss curve healthy
   throughout. (`MRR-vol ≈ 0.13` for `global+BN` vs `≈ 0.01` for the stable variants.)
2. GroupNorm alone fixes *stability* but unit-box normalization *erases* scale — so the explicit
   FiLM conditioning is what restores (and improves) size discrimination, 0.00 → **0.91**.

## What the figures show

`figures/data_diagnostic.png`
- **left**: histogram of per-sample post-global-norm extent — a long tail of outliers far above the "healthy ≈ 1" bulk.
- **right**: first-conv max-channel variance vs extent — variance grows ~ extent², so the tail samples are the poisoners.

`figures/bn_anticorrelation.png`
- `global+BN` only: `running_var` (red, log, left axis) and eval MRR (blue, right axis) on one plot.
  Every running_var spike coincides with an MRR collapse and vice-versa — Pearson
  `corr(log running_var, MRR) ≈ −0.85`. The clearest single view of the bug.

`figures/four_variants.png` (4 panels)
- **running_var** (BN only): `global+BN` explodes/oscillates over orders of magnitude; nothing else has the buffer.
- **eval MRR**: `global+BN` swings between checkpoints; the fixes are flat.
- **PC-PC cosine** (collapse indicator): spikes toward 1.0 for `global+BN` exactly when `running_var` is poisoned.
- **3 mm-vs-4 mm separation**: `unitbb+GN` drops (scale erased); `global+*` and `unitbb+GN+FiLM` retain it.

## Faithful to a real point encoder

The mini-encoder mirrors a standard patch-based point encoder: FPS-sampled centers + kNN
patches, per-patch recentering, `Conv1d(6,128,1) → Norm → ReLU → Conv1d(128,256,1)`,
max-pool/concat, a second conv block, group pooling, and an MLP head to an L2-normalized
embedding. The only substitutions are scale (fewer points/groups for CPU speed) and synthetic
data. The `running_var` poisoning is the identical mechanism.

### A note on the demo's BN momentum

The demo uses PyTorch's **default** BatchNorm `momentum = 0.1` (see `src/model.py`) — no
hand-tuning. The "spike then recover" checkpoint lottery is reproduced through the **data
regime**, not the hyperparameter: rare, catastrophic scale outliers arrive with variable gaps
(batches are sampled *with replacement*, so outlier spacing is Poisson-like) over a long-enough
run (1600 steps, eval every 20). Some checkpoints land in long clean windows where `running_var`
has decayed back to its true value (good eval); others land right after an outlier (poisoned →
sharp drop). That mirrors the real run, where checkpoints are thousands of steps apart and the
same rare-outlier-plus-slow-EMA dynamic produces the see-saw.

### Why the training loss stays smooth (grad accumulation)

An obvious objection: if an outlier is in a training batch, shouldn't the *loss* spike too?
`symptom_selfcontained.py` shows why it doesn't, via two mechanisms working together:

1. **The poisoned micro-batch's loss is capped.** In train mode BatchNorm normalizes the batch
   by its own (huge) variance, and embeddings are L2-normalized before the contrastive loss —
   so a collapsed micro-batch scores roughly *chance level* (`ln(batch) ≈ 3.6`), never
   something astronomical.
2. **Gradient accumulation dilutes what's left.** With `accum=12`, micro-batch `36` (the
   small-scale analog of the real run's large effective batch), the capped bump is averaged
   `1/accum` into the optimizer step.

BatchNorm's buffer gets neither protection: statistics are computed per micro-batch and
absorbed into `running_var` at full strength. `dissect_outlier_step.py` prints the receipts
for one poisoned step: outlier micro-loss 3.73 (≈ `ln 36`), logged window mean 1.46 vs run
median 1.23 (below the clean-step noise max of 1.68), grad-norm ×1.3 — while `running_var`
jumps 0.031 → 34.3 (**×1,100 in one update**: `0.9·0.031 + 0.1·343`) and eval MRR lands at
chance (0.024). Across the whole run the logged loss has **0 spikes** while
`corr(log running_var, MRR) ≈ −0.84`. The loss path is capped then diluted; the buffer path
is neither — that asymmetry is why the bug is so easy to miss.

## Layout

```
src/data.py    synthetic shapes × sizes, outlier injection, precomputed "teacher" embeddings
src/model.py   MiniPointEncoder: Group(FPS,kNN), BN|GN swap, global|unit_bb norm, Fourier+FiLM
src/train.py   contrastive training; logs running_var / MRR / pcpc / size-margin each eval
src/plots.py         the figures
run_demo.py          trains all four variants and renders everything
ablation_anchor.py   removes the color+bias anchor → global+GN's scale margin collapses to 0
investigation_story.py  diffs adjacent checkpoints → the running_var weight-delta figure
symptom_selfcontained.py  grad-accum run → smooth loss + spiky grad + see-sawing MRR
dissect_outlier_step.py   instruments one poisoned optimizer step: micro-batch losses,
                          window mean, running_var jump, eval MRR — the numeric receipts
```

## Caveats

Single training run per variant (no error bars), synthetic data, and a deliberately small
encoder. It is a faithful *mechanism* reproduction and teaching tool, not a benchmark — for
load-bearing claims, repeat across seeds and scales.
