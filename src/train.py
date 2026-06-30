"""
Train one variant and log, every `eval_every` steps:
  - bn_running_var   : mean of first_conv norm's running_var buffer (BN only; the poisoned stat)
  - retrieval_mrr    : PC->teacher retrieval MRR on a clean held-out eval set (eval mode)
  - pcpc_cos         : mean top-1 PC-PC cosine on eval set (collapse indicator; high == collapsed)
  - size_margin      : same-shape 3mm-vs-4mm separation the encoder achieves (scale discrimination)

The eval set has NO outliers, so any volatility there is caused purely by the
running_var buffer poisoned during training — exactly the mechanism under study.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ShapeDataset, Teacher, make_size_pairs, FAMILIES
from .model import MiniPointEncoder


def clip_loss(pc, txt, temp=0.07):
    logits = pc @ txt.t() / temp
    labels = torch.arange(len(pc), device=pc.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


@torch.no_grad()
def evaluate(model, eval_pts, eval_fam, eval_size, teacher, size_pairs, device):
    model.eval()
    log_s = torch.log(eval_size).to(device)
    emb = model(eval_pts.to(device), log_s)                       # eval mode -> uses running_var
    T = teacher.embed(eval_fam.numpy(), eval_size.numpy()).to(device)
    sims = emb @ T.t()
    rank = (sims.argsort(1, descending=True) == torch.arange(len(emb), device=device)[:, None]).float().argmax(1) + 1
    mrr = (1.0 / rank.float()).mean().item()
    pcsim = emb @ emb.t(); pcsim.fill_diagonal_(-1)
    pcpc = pcsim.max(1).values.mean().item()

    # size discrimination: same shape, 3mm vs 4mm -> do PC embeddings separate like the teacher wants?
    diffs = []
    for it in size_pairs:
        base = torch.from_numpy(it["base"])
        A = (base * it["sA"]).unsqueeze(0).to(device)
        B = (base * it["sB"]).unsqueeze(0).to(device)
        eA = model(A, torch.log(torch.tensor([it["sA"]], device=device)))
        eB = model(B, torch.log(torch.tensor([it["sB"]], device=device)))
        diffs.append((1 - (eA * eB).sum()).item())                # 1 - cos: bigger == better separated
    size_margin = float(np.mean(diffs))
    model.train()
    return mrr, pcpc, size_margin


def train_variant(cfg, n_items=3072, steps=700, batch=48, eval_every=20,
                  lr=2e-3, seed=0, device="cpu", log=print,
                  n_points=768, outlier_frac=0.0004, outlier_mult=(20.0, 2000.0),
                  snapshot=False):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = ShapeDataset(n_items=n_items, n_points=n_points, outlier_frac=outlier_frac,
                      outlier_mult=outlier_mult, seed=seed, with_outliers=True)
    # Precompute the whole train set once, then sample batches WITH REPLACEMENT.
    # With-replacement (Poisson) arrival gives variable gaps between outlier batches —
    # so some checkpoints land in long clean windows (running_var decays to its true
    # clean value -> good eval) and some right after an outlier (poisoned -> sharp drop).
    # Epoch-based shuffling instead forces the outlier in once per epoch, re-poisoning
    # running_var before it can recover, which makes BN *uniformly* bad rather than spiky.
    pts_all = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    fam_all = torch.tensor([ds.fam[i] for i in range(len(ds))])
    size_all = torch.from_numpy(ds.size).float().to(device)
    teacher = Teacher(seed=1)
    rng_b = torch.Generator().manual_seed(seed + 7)

    # fixed clean eval set (no outliers)
    eds = ShapeDataset(n_items=256, n_points=n_points, seed=999, with_outliers=False)
    eval_pts = torch.stack([eds[i][0] for i in range(len(eds))])
    eval_fam = torch.tensor([eds[i][1] for i in range(len(eds))])
    eval_size = torch.tensor([eds[i][2] for i in range(len(eds))])
    size_pairs = make_size_pairs(n_per_family=10, n_points=n_points)

    model = MiniPointEncoder(**cfg).to(device)
    if cfg["norm_mode"] == "global":   # divisor = median of per-sample extents over clean items
        clean = ~torch.from_numpy(ds.is_outlier)
        c = pts_all[clean] - pts_all[clean].mean(1, keepdim=True)
        model.global_div = c.abs().amax(dim=(1, 2)).median().detach()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {k: [] for k in ["step", "bn_running_var", "mrr", "pcpc", "size_margin", "batch_outlier", "loss"]}
    # per-STEP series (dense) — for the loss/grad-norm "everything looked fine" plot
    hist["loss_step"] = []; hist["grad_step"] = []
    snaps = {}                                            # step -> cpu state_dict copy (if snapshot)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(ds), (batch,), generator=rng_b)   # WITH replacement
        pts = pts_all[idx]; size = size_all[idx]; fam = fam_all[idx]
        emb = model(pts, torch.log(size))
        T = teacher.embed(fam.numpy(), size.cpu().numpy()).to(device)
        loss = clip_loss(emb, T)
        opt.zero_grad(); loss.backward()
        gnorm = float(torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in model.parameters()
                                     if p.grad is not None)).item())
        opt.step()
        hist["loss_step"].append(float(loss.item())); hist["grad_step"].append(gnorm)
        if step % eval_every == 0 or step == 1:
            mrr, pcpc, sm = evaluate(model, eval_pts, eval_fam, eval_size, teacher, size_pairs, device)
            rv = float("nan")
            if cfg["norm_layer"] == "bn":
                rv = model.first_conv[1].running_var.mean().item()
            b_out = float(torch.from_numpy(ds.is_outlier)[idx].float().mean())
            for k, v in zip(["step", "bn_running_var", "mrr", "pcpc", "size_margin", "batch_outlier", "loss"],
                            [step, rv, mrr, pcpc, sm, b_out, float(loss.item())]):
                hist[k].append(v)
            if snapshot:
                snaps[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(f"  step {step:4d}  loss {loss.item():.3f}  rv {rv:>10.3e}  "
                f"MRR {mrr:.3f}  pcpc {pcpc:.3f}  size_margin {sm:.4f}")
    return (hist, model, snaps) if snapshot else (hist, model)
