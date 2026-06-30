#!/usr/bin/env python
"""
One-command demo: trains the four variants on synthetic CAD-like data, reproduces
the BatchNorm running_var poisoning, and shows why each fix works.

    python run_demo.py              # full (~few min CPU)
    python run_demo.py --quick      # fast smoke (fewer steps/items)

Outputs:
    figures/data_diagnostic.png     extent histogram + var-vs-extent scatter
    figures/bn_anticorrelation.png  running_var vs eval MRR on one axis (the bug)
    figures/four_variants.png       running_var / MRR / pcpc / size-discrimination
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

from src.train import train_variant
from src.plots import plot_all, plot_data_diagnostic, plot_bn_anticorrelation
from src.data import ShapeDataset
from src.model import MiniPointEncoder

NUM_GROUP = 48
N_POINTS = 768
VARIANTS = {
    "global+BN":      dict(norm_mode="global",  norm_layer="bn", film=False, num_group=NUM_GROUP),
    "global+GN":      dict(norm_mode="global",  norm_layer="gn", film=False, num_group=NUM_GROUP),
    "unitbb+GN":      dict(norm_mode="unit_bb", norm_layer="gn", film=False, num_group=NUM_GROUP),
    "unitbb+GN+FiLM": dict(norm_mode="unit_bb", norm_layer="gn", film=True,  num_group=NUM_GROUP),
}


def data_diagnostic(device, quick):
    """Show the driver: under global norm, per-sample extent vs first_conv variance.
    Uses a dramatic outlier setting (4%, up to 1e4x) purely for a legible tail —
    training uses rarer/milder outliers so the model still learns (see VARIANTS run)."""
    ds = ShapeDataset(n_items=600 if quick else 1500, n_points=N_POINTS, seed=7,
                      with_outliers=True, outlier_frac=0.04, outlier_mult=(1e2, 1e4))
    pts = torch.stack([ds[i][0] for i in range(len(ds))])
    m = MiniPointEncoder(norm_mode="global", norm_layer="bn", num_group=NUM_GROUP).to(device)
    from torch.utils.data import DataLoader
    m.set_global_div(DataLoader(ds, batch_size=64))
    exts, vars = [], []
    with torch.no_grad():
        for i in range(0, len(pts), 64):
            b = pts[i:i+64].to(device)
            xyz = m.normalize(b)
            ext = xyz.abs().amax(dim=(1, 2)).cpu().numpy()
            _, fv = m(b, return_first_var=True)
            exts.append(ext); vars.append(fv.cpu().numpy())
    plot_data_diagnostic(np.concatenate(exts), np.concatenate(vars))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = args.steps or (150 if args.quick else 700)
    n_items = 800 if args.quick else 3072
    print(f"device={device}  steps={steps}  n_items={n_items}\n")

    print("== data diagnostic (why outliers poison the BN) ==")
    data_diagnostic(device, args.quick)

    results = {}
    for name, cfg in VARIANTS.items():
        print(f"\n== training {name}  {cfg} ==")
        hist, _ = train_variant(cfg, n_items=n_items, steps=steps, seed=args.seed,
                                device=device, eval_every=max(8, steps // 50))
        results[name] = hist

    plot_all(results, data_stats=None)
    plot_bn_anticorrelation(results["global+BN"])

    print("\n=== SUMMARY (last-3-eval mean) ===")
    print(f"  {'variant':16} {'MRR':>6} {'pcpc':>7} {'size_margin':>12} {'rv_max':>12}")
    for name, h in results.items():
        mrr = np.mean(h["mrr"][-3:]); pcpc = np.mean(h["pcpc"][-3:]); sm = np.mean(h["size_margin"][-3:])
        rv = np.array(h["bn_running_var"]); rvmax = np.nanmax(rv) if np.isfinite(rv).any() else float("nan")
        # volatility of MRR over the run = the headline symptom
        vol = np.std(h["mrr"][len(h["mrr"])//3:])
        print(f"  {name:16} {mrr:6.3f} {pcpc:7.3f} {sm:12.4f} {rvmax:12.2e}   MRR-vol={vol:.3f}")


if __name__ == "__main__":
    main()
