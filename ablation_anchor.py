#!/usr/bin/env python
"""
Ablation: is `global + GroupNorm`'s residual scale signal (size-margin ~0.35) really
a leak through the fixed anchor (constant 0.4 color + first-conv bias)?

We train `global + GroupNorm` twice — once with the anchor (the demo's default) and once
without it (`scale_anchor=False`: color->0, first-conv bias off, so first_conv is pure
scaling s*A and GroupNorm cancels s exactly). Prediction: shape/MRR roughly unchanged,
but size discrimination collapses 0.35 -> ~0, confirming the anchor is the leak.

    python ablation_anchor.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from src.train import train_variant

STEPS, NG = 1600, 48
device = "cuda" if torch.cuda.is_available() else "cpu"

runs = {
    "global+GN  (anchor: 0.4 color + bias)": dict(norm_mode="global", norm_layer="gn",
                                                  film=False, num_group=NG, scale_anchor=True),
    "global+GN  (anchor REMOVED)":           dict(norm_mode="global", norm_layer="gn",
                                                  film=False, num_group=NG, scale_anchor=False),
}

print(f"device={device}  steps={STEPS}\n")
results = {}
for name, cfg in runs.items():
    print(f"== {name}  {cfg} ==")
    hist, _ = train_variant(cfg, n_items=8000, steps=STEPS, seed=0, device=device,
                            eval_every=20)
    results[name] = hist

print("\n=== ABLATION RESULT (last-3-eval mean) ===")
print(f"  {'variant':40} {'MRR (shape)':>12} {'size-margin':>12}")
for name, h in results.items():
    mrr = float(np.mean(h["mrr"][-3:])); sm = float(np.mean(h["size_margin"][-3:]))
    print(f"  {name:40} {mrr:>12.3f} {sm:>12.4f}")
print("\nExpectation: MRR ~unchanged (shape still learned), size-margin collapses toward 0")
print("when the anchor is removed -> the 0.35 was a leak through the constant color + bias.")
