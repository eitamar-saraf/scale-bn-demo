#!/usr/bin/env python
"""
Generates the three "investigation story" figures for the global+BN run — the order
in which the bug actually revealed itself:

  1) figure_symptom.png       eval MRR only: it crashes between checkpoints, no visible cause
  2) figure_lookedfine.png    loss + grad-norm, EMA-smoothed: everything you'd watch looks healthy
  3) figure_weightdelta.png   per-tensor relative weight change at a clean->poisoned step:
                              first_conv running_var dwarfs every other tensor (the breakthrough)

    python investigation_story.py
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.train import train_variant

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = dict(norm_mode="global", norm_layer="bn", film=False, num_group=48)
print(f"device={device} — training global+BN with grad-norm + weight snapshots ...")
hist, model, snaps = train_variant(cfg, steps=700, seed=0, device=device,
                                   eval_every=max(8, 700 // 50), snapshot=True)

def ema(x, beta=0.9):
    x = np.asarray(x, float); out = np.empty_like(x); m = x[0]
    for i, v in enumerate(x):
        m = beta * m + (1 - beta) * v; out[i] = m
    return out

step = np.array(hist["step"]); mrr = np.array(hist["mrr"])

# ---------- 1) the symptom: MRR crashes, unexplained ----------
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(step, mrr, color="#1f4e79", lw=2, marker="s", ms=4)
ax.set_ylim(0, max(0.4, mrr.max() * 1.15)); ax.set_xlabel("training step"); ax.set_ylabel("eval retrieval MRR")
ax.set_title("The symptom: the eval score collapses between checkpoints — and nobody could say why\n"
             "(same model, adjacent checkpoints: 0.34 one save, 0.02 the next)")
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("figures/figure_symptom.png", dpi=140, facecolor="white"); print("saved figures/figure_symptom.png")

# ---------- 2) everything you'd normally watch looked fine ----------
ls = np.array(hist["loss_step"]); gs = np.array(hist["grad_step"]); xs = np.arange(1, len(ls) + 1)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(xs, ls, color="#9aa6b2", lw=1, alpha=0.45)
ax[0].plot(xs, ema(ls), color="#2e7d32", lw=2.4)
ax[0].set_title("Training loss (EMA)"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=0.25)
ax[1].plot(xs, gs, color="#9aa6b2", lw=1, alpha=0.45)
ax[1].plot(xs, ema(gs), color="#6a1b9a", lw=2.4)
ax[1].set_title("Gradient norm (EMA)"); ax[1].set_xlabel("step"); ax[1].set_ylabel("‖grad‖"); ax[1].grid(alpha=0.25)
fig.suptitle("Everything you'd normally watch looked perfectly healthy — smooth and converging "
             "(EMA-smoothed; even the raw spikes are minor)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("figures/figure_lookedfine.png", dpi=140, facecolor="white")
print("saved figures/figure_lookedfine.png")

# ---------- 3) the breakthrough: per-tensor relative weight delta ----------
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
