#!/usr/bin/env python
"""Double-check the 'why doesn't the loss spike?' claim with hard numbers.

Reproduces symptom_selfcontained's exact regime (global+BN, accum=12, batch=36,
frac=3e-4) but instruments the first few optimizer steps that contain an outlier
micro-batch (after warmup):
  - the 12 individual micro-batch losses (which one is poisoned, what it costs)
  - the logged window-mean loss vs the surrounding steps' means
  - running_var before the outlier micro-batch / right after it / at window end
  - eval MRR at the previous step vs right after the outlier step
  - grad-norm of the step vs the median step
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from src.data import ShapeDataset, Teacher, make_size_pairs
from src.model import MiniPointEncoder
from src.train import clip_loss, evaluate

dev = "cuda" if torch.cuda.is_available() else "cpu"
ACCUM, BATCH, STEPS, WARM = 12, 36, 600, 150
torch.manual_seed(0); np.random.seed(0)

ds = ShapeDataset(n_items=8000, n_points=768, outlier_frac=3e-4,
                  outlier_mult=(30.0, 6000.0), seed=0, with_outliers=True)
pts_all = torch.stack([ds[i][0] for i in range(len(ds))]).to(dev)
fam_all = torch.tensor([ds.fam[i] for i in range(len(ds))])
size_all = torch.from_numpy(ds.size).float().to(dev)
is_out = torch.from_numpy(ds.is_outlier)
teacher = Teacher(seed=1)
rng_b = torch.Generator().manual_seed(7)
print(f"outliers in dataset: {int(is_out.sum())}  extents(mult): see data")

eds = ShapeDataset(n_items=256, n_points=768, seed=999, with_outliers=False)
eval_pts = torch.stack([eds[i][0] for i in range(len(eds))])
eval_fam = torch.tensor([eds[i][1] for i in range(len(eds))])
eval_size = torch.tensor([eds[i][2] for i in range(len(eds))])
size_pairs = make_size_pairs(n_per_family=10, n_points=768)

cfg = dict(norm_mode="global", norm_layer="bn", film=False, num_group=48)
model = MiniPointEncoder(**cfg).to(dev)
clean = ~is_out
c = pts_all[clean] - pts_all[clean].mean(1, keepdim=True)
model.global_div = c.abs().amax(dim=(1, 2)).median().detach()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)

def rv():
    return model.first_conv[1].running_var.mean().item()

window_means, grad_norms = [], []
prev_mrr = None
dissected = 0
for step in range(1, STEPS + 1):
    opt.zero_grad()
    micro = []          # (loss, n_outliers, rv_before, rv_after)
    for m in range(ACCUM):
        idx = torch.randint(0, len(ds), (BATCH,), generator=rng_b)
        r0 = rv()
        emb = model(pts_all[idx], torch.log(size_all[idx]))
        T = teacher.embed(fam_all[idx].numpy(), size_all[idx].cpu().numpy()).to(dev)
        L = clip_loss(emb, T)
        (L / ACCUM).backward()
        micro.append((float(L.item()), int(is_out[idx].sum()), r0, rv()))
    g = float(torch.sqrt(sum(p.grad.detach().pow(2).sum()
                             for p in model.parameters() if p.grad is not None)).item())
    opt.step()
    wm = float(np.mean([m[0] for m in micro]))
    window_means.append(wm); grad_norms.append(g)
    has_out = any(m[1] > 0 for m in micro)

    if step > WARM and has_out and dissected < 2:
        dissected += 1
        mrr_after, _, _ = evaluate(model, eval_pts, eval_fam, eval_size, teacher, size_pairs, dev)
        base = float(np.median(window_means[WARM//2:-1])) if len(window_means) > 2 else float("nan")
        gmed = float(np.median(grad_norms[WARM//2:-1]))
        print(f"\n================ OUTLIER STEP {step} (dissection #{dissected}) ================")
        print(f"chance-ceiling loss for batch {BATCH}: ln({BATCH}) = {np.log(BATCH):.3f}")
        print(f"{'micro':>5} {'loss':>7} {'#outl':>5} {'rv_before':>12} {'rv_after':>12}")
        for i, (L, n, r0, r1) in enumerate(micro, 1):
            tag = "  <-- OUTLIER" if n > 0 else ""
            print(f"{i:>5} {L:>7.3f} {n:>5} {r0:>12.4e} {r1:>12.4e}{tag}")
        cleanL = [m[0] for m in micro if m[1] == 0]
        outL = [m[0] for m in micro if m[1] > 0]
        print(f"\nclean micro-batches : mean loss {np.mean(cleanL):.3f}  (n={len(cleanL)})")
        print(f"outlier micro-batch : loss {outL[0]:.3f}")
        print(f"logged window mean  : {wm:.3f}   vs run median {base:.3f}  -> ratio {wm/base:.2f}x")
        print(f"recent window means : {['%.3f'%v for v in window_means[-6:-1]]}")
        print(f"grad-norm this step : {g:.2f}   vs median {gmed:.2f}  -> ratio {g/gmed:.1f}x")
        print(f"running_var window  : start {micro[0][2]:.4e}  end {micro[-1][3]:.4e}"
              f"  -> jump x{micro[-1][3]/micro[0][2]:.0f}")
        print(f"eval MRR            : before-step {prev_mrr:.3f}  after-step {mrr_after:.3f}")
        if dissected == 2:
            break
    if step > WARM - 5:
        prev_mrr, _, _ = evaluate(model, eval_pts, eval_fam, eval_size, teacher, size_pairs, dev)

print(f"\nrun median window-loss: {np.median(window_means[WARM//2:]):.3f}"
      f"   max: {max(window_means[WARM//2:]):.3f}")
