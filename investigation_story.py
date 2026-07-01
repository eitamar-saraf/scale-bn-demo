#!/usr/bin/env python
"""
Generates the "breakthrough" figure for the global+BN run: diff two adjacent
checkpoints tensor-by-tensor and show that first_conv's BatchNorm `running_var`
buffer dwarfs every learnable parameter's relative change.

  figure_weightdelta.png   per-tensor ‖ΔW‖/‖W‖ at a clean->poisoned step

(The symptom — smooth loss / spiky grad / see-sawing eval — is shown in the blog
from the REAL run's wandb curves, not the toy; see tools/pull_real_signals.py.)

    python investigation_story.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.train import train_variant

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = dict(norm_mode="global", norm_layer="bn", film=False, num_group=48)
print(f"device={device} — training global+BN with weight snapshots ...")
hist, model, snaps = train_variant(cfg, n_items=8000, steps=1600, seed=0, device=device,
                                   eval_every=20, snapshot=True)

# ---------- the breakthrough: per-tensor relative weight delta ----------
# find the consecutive snapshot pair with the biggest jump in first_conv BN running_var
RV = "first_conv.1.running_var"
steps_s = sorted(snaps)
def rvmean(sd):
    k = [k for k in sd if k.endswith(RV)][0]; return sd[k].float().mean().item()
jumps = [(steps_s[i], steps_s[i + 1], rvmean(snaps[steps_s[i + 1]]) / max(rvmean(snaps[steps_s[i]]), 1e-9))
         for i in range(len(steps_s) - 1)]
a, b, _ = max(jumps, key=lambda t: t[2])           # clean -> poisoned transition
sdA, sdB = snaps[a], snaps[b]
rows = []
for k in sdA:
    if not torch.is_floating_point(sdA[k]) or sdA[k].numel() < 2:
        continue
    denom = sdA[k].float().norm().item()
    rel = (sdB[k].float() - sdA[k].float()).norm().item() / (denom + 1e-12)
    short = k.replace("point_encoder.", "").replace("encoder.", "")
    rows.append((short, rel))
rows.sort(key=lambda r: r[1], reverse=True)
top = rows[:10][::-1]
names = [r[0] for r in top]; vals = [r[1] for r in top]
colors = ["#c0392b" if "running_var" in n else "#90a4ae" for n in names]
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.barh(range(len(top)), vals, color=colors)
ax.set_yticks(range(len(top))); ax.set_yticklabels(names, fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("relative weight change  ‖ΔW‖ / ‖W‖  between adjacent checkpoints (log)")
ax.set_title(f"The breakthrough: diff the checkpoints tensor-by-tensor.\n"
             f"first_conv running_var jumped ~{vals[-1]/max(vals[-2],1e-12):.0f}× more than anything else "
             f"(steps {a}→{b}) — a buffer the loss never touches")
ax.grid(alpha=0.25, axis="x")
fig.tight_layout(); fig.savefig("figures/figure_weightdelta.png", dpi=140, facecolor="white")
print(f"saved figures/figure_weightdelta.png  (transition {a}->{b}; top tensor: {names[-1]} = {vals[-1]:.1f})")
