"""
Faithful mini point encoder, mirroring a standard patch-based point encoder:
  Group(FPS + kNN, per-patch recenter) -> first_conv[Conv1d(6,128)->Norm->ReLU->Conv1d]
  -> max-pool/concat -> second_conv[Conv1d(512,512)->Norm->ReLU->Conv1d] -> group tokens
  -> pool over groups -> MLP head -> L2-normalized embedding.

Configurable along the two design axes:
  norm_mode  : 'global'  (single dataset divisor; scale outliers survive)  -> the bug setup
               'unit_bb' (per-sample unit box; geometry is pure shape)     -> the fix setup
  norm_layer : 'bn' (BatchNorm1d, has running_var EMA -> poisonable)
               'gn' (GroupNorm, stateless, train==eval)
  film       : if True, inject log-scale via Fourier->MLP->FiLM on the group tokens
               (the explicit scale-conditioning path; only meaningful with unit_bb).
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ----------------------------- grouping (pure torch) -----------------------------
def fps(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling, index-0 start (matches the repo's fps_torch). [B,N,3]->idx[B,npoint]."""
    B, N, _ = xyz.shape
    dev = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=dev)
    dist = torch.full((B, N), 1e10, device=dev)
    far = torch.zeros(B, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev)
    for i in range(npoint):
        idx[:, i] = far
        c = xyz[ar, far, :].view(B, 1, 3)
        d = ((xyz - c) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.max(-1).indices
    return idx


def knn(k: int, xyz: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """k nearest neighbours of each center within xyz. -> idx [B,G,k]."""
    d = (centers.unsqueeze(2) - xyz.unsqueeze(1)).pow(2).sum(-1)   # [B,G,N]
    return d.topk(k, dim=-1, largest=False).indices


class Group(nn.Module):
    def __init__(self, num_group=64, group_size=16):
        super().__init__()
        self.num_group, self.group_size = num_group, group_size

    def forward(self, xyz, color):
        B, N, _ = xyz.shape
        cidx = fps(xyz, self.num_group)
        centers = torch.gather(xyz, 1, cidx.unsqueeze(-1).expand(-1, -1, 3))      # [B,G,3]
        nidx = knn(self.group_size, xyz, centers)                                 # [B,G,k]
        base = (torch.arange(B, device=xyz.device).view(-1, 1, 1) * N)
        flat = (nidx + base).view(-1)
        nbhd = xyz.reshape(B * N, 3)[flat].view(B, self.num_group, self.group_size, 3)
        ncol = color.reshape(B * N, 3)[flat].view(B, self.num_group, self.group_size, 3)
        nbhd = nbhd - centers.unsqueeze(2)                       # per-patch recenter (kills position, keeps scale)
        feats = torch.cat([nbhd, ncol], dim=-1)                 # [B,G,k,6]
        return centers, feats


# BN momentum is raised from PyTorch's default 0.1 to compress the running_var
# recovery time into this short (~700-step) demo. The real run used 0.1 but saved
# checkpoints 1000 steps apart, giving long clean windows for running_var to recover;
# here, with eval every ~14 steps, a higher momentum reproduces that "spike then
# recover" dynamic on the demo's timescale (shadow ~15 steps instead of ~60).
BN_MOMENTUM = 0.35

def norm1d(kind: str, c: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm1d(c, momentum=BN_MOMENTUM)
    if kind == "gn":
        return nn.GroupNorm(num_groups=min(32, c // 4), num_channels=c)
    raise ValueError(kind)


# ----------------------------- scale conditioning -----------------------------
class FourierScale(nn.Module):
    """log-scale scalar -> Fourier features -> MLP -> scale_emb. High freqs resolve tiny ratios."""
    def __init__(self, n_freq=16, out_dim=128, f_min=0.5, f_max=64.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(torch.log(torch.tensor(f_min)),
                                         torch.log(torch.tensor(f_max)), n_freq))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, 128), nn.SiLU(), nn.Linear(128, out_dim))

    def forward(self, log_s):                       # log_s [B]
        a = log_s.unsqueeze(-1) * self.freqs        # [B,n_freq]
        emb = torch.cat([a.sin(), a.cos()], -1)
        return self.mlp(emb)                         # [B,out_dim]


class FiLM(nn.Module):
    """h' = (1+gamma(c)) * h + beta(c), gamma/beta zero-init -> identity at start."""
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.to_gb.weight); nn.init.zeros_(self.to_gb.bias)

    def forward(self, h, c):                         # h [B,G,C], c [B,cond_dim]
        g, b = self.to_gb(c).chunk(2, -1)
        return (1 + g).unsqueeze(1) * h + b.unsqueeze(1)


# ----------------------------- the encoder -----------------------------
class MiniPointEncoder(nn.Module):
    def __init__(self, norm_mode="global", norm_layer="bn", film=False,
                 embed_dim=128, enc_dim=256, num_group=64, group_size=16,
                 scale_anchor=True):
        super().__init__()
        assert norm_mode in ("global", "unit_bb")
        self.norm_mode, self.norm_layer, self.use_film = norm_mode, norm_layer, film
        # scale_anchor: keep the fixed reference (constant 0.4 color + first-conv bias) that lets
        # a scale-invariant GroupNorm leak a faint absolute-scale cue under global norm. Set False
        # to remove both anchors (color->0, first-conv bias off) — first_conv becomes pure scaling
        # s*A, so GroupNorm cancels s exactly and the leak should vanish (ablation of the leak).
        self.scale_anchor = scale_anchor
        self.group = Group(num_group, group_size)
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1, bias=scale_anchor), norm1d(norm_layer, 128), nn.ReLU(True), nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), norm1d(norm_layer, 512), nn.ReLU(True), nn.Conv1d(512, enc_dim, 1))
        self.head = nn.Sequential(nn.Linear(enc_dim, 256), nn.GELU(), nn.Linear(256, embed_dim))
        if film:
            self.scale_embed = FourierScale(out_dim=128)
            self.film = FiLM(128, enc_dim)
        # global-norm divisor, set from the training set (mean of per-sample medians)
        self.register_buffer("global_div", torch.tensor(1.0))

    # ---- normalization of the raw physical point cloud ----
    def normalize(self, pts):                        # pts [B,N,3] physical scale
        c = pts.mean(1, keepdim=True)
        p = pts - c                                  # center (both modes)
        if self.norm_mode == "global":
            return p / self.global_div               # shared divisor -> outliers stay huge
        ext = p.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return p / (2 * ext)                          # unit box [-0.5,0.5] -> scale erased

    def set_global_div(self, loader, n_batches=8):
        meds = []
        with torch.no_grad():
            for bi, (pts, *_rest) in enumerate(loader):
                c = pts - pts.mean(1, keepdim=True)
                meds.append(c.abs().amax(dim=(1, 2)))
                if bi + 1 >= n_batches:
                    break
        self.global_div = torch.cat(meds).median().detach()

    def forward(self, pts, log_size=None, return_first_var=False):
        pts = pts.float()
        if log_size is not None:
            log_size = log_size.float()
        xyz = self.normalize(pts)
        color = torch.full_like(xyz, 0.4 if self.scale_anchor else 0.0)  # constant color (the fixed anchor)
        centers, feats = self.group(xyz, color)      # feats [B,G,k,6]
        B, G, k, _ = feats.shape
        x = feats.reshape(B * G, k, 6).transpose(1, 2)        # [B*G,6,k]
        bn_in = self.first_conv[0](x)                # Conv1d(6,128) output == the BN input
        first_var = None
        if return_first_var:                          # per-sample max-channel var (the BN statistic)
            v = bn_in.reshape(B, G, 128, k).var(dim=(1, 3), unbiased=False)  # [B,128]
            first_var = v.max(1).values.detach()
        f = self.first_conv[1:](bn_in)               # Norm->ReLU->Conv1d -> [B*G,256,k]
        fg = f.max(-1, keepdim=True).values
        f = torch.cat([fg.expand(-1, -1, k), f], 1)  # [B*G,512,k]
        tok = self.second_conv(f).max(-1).values.reshape(B, G, -1)            # group tokens [B,G,enc]
        if self.use_film and log_size is not None:
            tok = self.film(tok, self.scale_embed(log_size))
        emb = self.head(tok.mean(1))                 # pool groups -> head
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return (emb, first_var) if return_first_var else emb
