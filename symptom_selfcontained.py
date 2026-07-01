#!/usr/bin/env python
"""The opening "mystery" figure, produced entirely by the toy — no real data.

Trains `global+BN` with gradient accumulation (default accum=12, micro-batch=36), the
small-scale analog of the real run's large effective batch (grad-accum / multi-GPU).
Each micro-batch does its own BatchNorm-train forward, so an outlier micro-batch still
poisons `running_var` via the EMA — but its gradient is averaged 1/accum into the step,
so the *logged* (window-mean) training loss barely moves. Result: the three signals you'd
watch look healthy (smooth loss, dismissable grad spikes) while the eval metric see-saws.

    python symptom_selfcontained.py
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.train import train_variant

ap = argparse.ArgumentParser()
ap.add_argument("--accum", type=int, default=12)
ap.add_argument("--batch", type=int, default=36)
ap.add_argument("--frac", type=float, default=3e-4)
ap.add_argument("--mhi", type=float, default=6000.0)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--evry", type=int, default=20)
ap.add_argument("--n_items", type=int, default=8000)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = dict(norm_mode="global", norm_layer="bn", film=False, num_group=48)
print(f"device={dev}  accum={a.accum} micro_batch={a.batch} (eff={a.accum*a.batch})  frac={a.frac}")

hist, _ = train_variant(cfg, n_items=a.n_items, steps=a.steps, batch=a.batch,
                        eval_every=a.evry, seed=0, device=dev, accum=a.accum,
                        outlier_frac=a.frac, outlier_mult=(30.0, a.mhi))

L = np.array(hist["loss_step"]); step_L = np.arange(1, len(L) + 1)
G = np.array(hist["grad_step"])
s = np.array(hist["step"]); rv = np.array(hist["bn_running_var"]); mrr = np.array(hist["mrr"])

# ---- gates (printed) ----
warmL = L[len(L)//5:]; med = np.median(warmL)
n_spikes = int((warmL > 1.5 * med).sum())
warm = s > 60
r = np.corrcoef(np.log10(rv[warm] + 1e-9), mrr[warm])[0, 1]
print(f"train loss: spikes(>1.5x median)={n_spikes}/{len(warmL)}  max/median={warmL.max()/med:.2f}x  (gate 0)")
print(f"running_var: {rv[warm].min():.2e}..{rv[warm].max():.2e}  ratio={rv[warm].max()/rv[warm].min():.1e}")
print(f"corr(log rv, MRR)={r:+.2f} (gate <-0.5)  MRR {mrr[warm].min():.3f}..{mrr[warm].max():.3f}")

# ---- figure: same 3-panel language as the real one, but 100% toy ----
c_loss, c_grad, c_mrr = "#2e7d32", "#6a1b9a", "#c0392b"
fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
ax[0].plot(step_L, L, color=c_loss, lw=1.2); ax[0].set_ylabel("training loss")
ax[0].set_title("The reproduction: the signals you'd watch look healthy…", fontweight="bold", loc="left")
ax[0].text(0.99, 0.9, "smooth, converging — no spikes", transform=ax[0].transAxes, ha="right", color=c_loss)
ax[0].grid(alpha=0.25)
ax[1].plot(step_L, G, color=c_grad, lw=0.9); ax[1].set_ylabel("gradient norm")
ax[1].text(0.99, 0.9, "occasional spikes — easy to dismiss as normal", transform=ax[1].transAxes, ha="right", color=c_grad)
ax[1].grid(alpha=0.25)
ax[2].plot(s, mrr, color=c_mrr, lw=1.6, marker="o", ms=3); ax[2].set_ylabel("validation score\n(retrieval MRR)")
ax[2].set_xlabel("optimizer step")
ax[2].set_title("…but the evaluation see-saws between checkpoints — the actual symptom (higher = better)",
                fontweight="bold", loc="left", color=c_mrr)
ax[2].grid(alpha=0.25)
fig.tight_layout(); fig.savefig("figures/toy_symptom.png", dpi=140, facecolor="white")
print("saved figures/toy_symptom.png")
