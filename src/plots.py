"""Figures for the demo. Each variant -> a color; key panels tell the story."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "global+BN":        "#c0392b",
    "global+GN":        "#e08e0b",
    "unitbb+GN":        "#2e86c1",
    "unitbb+GN+FiLM":   "#1e8e4e",
}


def plot_all(results, data_stats, out="figures/four_variants.png"):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BatchNorm running_var poisoning & the fixes", fontsize=14, fontweight="bold")

    # (0,0) running_var trajectory (log) — BN variants only
    a = ax[0, 0]
    for name, h in results.items():
        rv = np.array(h["bn_running_var"])
        if np.isfinite(rv).any():
            a.plot(h["step"], rv, label=name, color=COLORS[name], lw=1.8)
    a.set_yscale("log"); a.set_title("first_conv norm running_var (BN only)\nexplodes/oscillates = the bug")
    a.set_xlabel("step"); a.set_ylabel("mean running_var"); a.legend(); a.grid(alpha=0.3)

    # (0,1) eval retrieval MRR — volatility
    a = ax[0, 1]
    for name, h in results.items():
        a.plot(h["step"], h["mrr"], label=name, color=COLORS[name], lw=1.8)
    a.set_title("eval retrieval MRR (clean held-out set)\nvolatile for global+BN, stable for the fixes")
    a.set_xlabel("step"); a.set_ylabel("MRR"); a.set_ylim(0, 1); a.legend(); a.grid(alpha=0.3)

    # (1,0) pcpc collapse indicator
    a = ax[1, 0]
    for name, h in results.items():
        a.plot(h["step"], h["pcpc"], label=name, color=COLORS[name], lw=1.8)
    a.set_title("PC-PC top-1 cosine (embedding-collapse indicator)\nspikes toward 1.0 when running_var is poisoned")
    a.set_xlabel("step"); a.set_ylabel("mean top-1 PC-PC cos"); a.legend(); a.grid(alpha=0.3)

    # (1,1) size discrimination (final): mean 1-cos for 3mm vs 4mm same shape
    a = ax[1, 1]
    names = list(results.keys())
    finals = [np.mean(results[n]["size_margin"][-3:]) for n in names]
    bars = a.bar(range(len(names)), finals, color=[COLORS[n] for n in names])
    a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=20, ha="right")
    a.set_title("3mm-vs-4mm separation (same shape)\nhigher = encoder can tell sizes apart")
    a.set_ylabel("mean (1 - cos) over size pairs"); a.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, finals):
        a.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")


def plot_bn_anticorrelation(hist, out="figures/bn_anticorrelation.png"):
    """global+BN only: overlay running_var (log, left), eval MRR (right) and training
    loss (far right) on one plot. The running_var and MRR see-saw in lock-step, while
    the loss glides smoothly down — oblivious to the whole thing."""
    step = np.array(hist["step"]); rv = np.array(hist["bn_running_var"]); mrr = np.array(hist["mrr"])
    loss = np.array(hist["loss"]) if "loss" in hist else None
    c_rv, c_mrr, c_loss = "#c0392b", "#1f4e79", "#6b7280"
    fig, ax1 = plt.subplots(figsize=(13.5, 5.8))
    fig.subplots_adjust(right=0.84)   # room for two right-hand axes

    # shade the poisoned checkpoints (running_var well above the clean floor)
    thr = 0.1
    for s, r in zip(step, rv):
        if r > thr:
            ax1.axvspan(s - 3.5, s + 3.5, color=c_rv, alpha=0.07, zorder=0)

    ax1.plot(step, rv, color=c_rv, lw=2, marker="o", ms=3.5, zorder=3, label="first_conv running_var")
    ax1.set_yscale("log"); ax1.set_xlabel("training step")
    ax1.set_ylabel("running_var  (log scale)", color=c_rv); ax1.tick_params(axis="y", labelcolor=c_rv)
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(step, mrr, color=c_mrr, lw=2, marker="s", ms=3.5, zorder=3, label="eval retrieval MRR")
    ax2.set_ylabel("eval retrieval MRR", color=c_mrr); ax2.tick_params(axis="y", labelcolor=c_mrr)
    ax2.set_ylim(0, max(0.4, float(mrr.max()) * 1.15))

    if loss is not None:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 58))
        ax3.plot(step, loss, color=c_loss, lw=2.2, ls="--", marker="^", ms=3, zorder=2,
                 label="training loss")
        ax3.set_ylabel("training loss", color=c_loss); ax3.tick_params(axis="y", labelcolor=c_loss)
        ax3.set_ylim(0, float(loss.max()) * 1.1)

    r = float(np.corrcoef(np.log10(rv), mrr)[0, 1])      # anti-correlation, quantified
    ax1.set_title("global+BN — the training loss (gray) glides smoothly down, oblivious;\n"
                  f"running_var (red ↑, shaded) and eval MRR (blue ↓) see-saw in lock-step  ·  "
                  f"corr(log running_var, MRR) = {r:+.2f}")
    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    if loss is not None:
        handles += ax3.get_legend_handles_labels()[0]; labels += ax3.get_legend_handles_labels()[1]
    ax1.legend(handles, labels, loc="center left", framealpha=0.9)
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"saved {out}  (corr={r:+.2f})")


def plot_data_diagnostic(extents, variances, out="figures/data_diagnostic.png"):
    """Per-sample: post-global-norm extent vs first_conv max-channel variance (the driver)."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(np.log10(extents), bins=60, color="#3b78b5")
    ax[0].set_title("post-global-norm extent (max|coord|) per sample")
    ax[0].set_xlabel("log10 extent"); ax[0].set_ylabel("count"); ax[0].grid(alpha=0.3)
    ax[0].axvline(0, color="k", ls="--", lw=1); ax[0].text(0.05, 0.9, "healthy ~1", transform=ax[0].transAxes)

    ax[1].scatter(extents, variances, s=8, alpha=0.4, color="#c0392b")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_title("first_conv max-channel variance vs extent\n(var grows ~ extent^2 -> outliers poison BN)")
    ax[1].set_xlabel("post-norm extent"); ax[1].set_ylabel("max-channel var"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")
