"""
Synthetic CAD-like point-cloud dataset that reproduces the failure mode.

Each object has:
  - a SHAPE family (cube / sphere / cylinder / torus / L-bracket)  -> the "what"
  - a physical SIZE s (the bbox half-extent in mm)                 -> the "how big"

Sizes are drawn log-normally around a median, EXCEPT a small fraction of
`outlier_frac` objects whose size is multiplied by a huge factor (1e2..1e4) —
the analogue of real-world-scale CAD uploads (meter-scale buildings, unit-error
models) that survive a single-divisor global normalization and poison the
first_conv BatchNorm.

A precomputed "teacher" embedding T(family, size_bucket) plays the role of the
text embedding (e.g. from a sentence/text model): it separates families strongly
and sizes weakly-but-present (categorical — adjacent sizes are distinguishable but
the gap doesn't grow much with the size ratio). The point encoder is trained
contrastively to match T, so to win it must recover BOTH shape and size.
"""
from __future__ import annotations
import numpy as np
import torch

FAMILIES = ["cube", "sphere", "cylinder", "torus", "lbracket"]
N_SIZE_BUCKETS = 12          # log-spaced size classes the teacher can distinguish


def _unit_shape(family: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return n points on a unit-scale instance of `family` (half-extent ~1)."""
    if family == "cube":
        p = rng.uniform(-1, 1, size=(n, 3))
        face = rng.integers(0, 3, n); sign = rng.choice([-1, 1], n)
        p[np.arange(n), face] = sign  # push to a face -> hollow box surface
    elif family == "sphere":
        p = rng.normal(size=(n, 3)); p /= np.linalg.norm(p, axis=1, keepdims=True)
    elif family == "cylinder":
        th = rng.uniform(0, 2*np.pi, n); z = rng.uniform(-1, 1, n)
        p = np.stack([np.cos(th), np.sin(th), z], 1)
    elif family == "torus":
        R, r = 0.7, 0.3
        u = rng.uniform(0, 2*np.pi, n); v = rng.uniform(0, 2*np.pi, n)
        p = np.stack([(R + r*np.cos(v))*np.cos(u),
                      (R + r*np.cos(v))*np.sin(u), r*np.sin(v)], 1)
    elif family == "lbracket":
        p = rng.uniform(-1, 1, size=(n, 3))
        # carve out one quadrant in x-y to make an L
        mask = (p[:, 0] > 0) & (p[:, 1] > 0)
        p[mask, rng.integers(0, 2, mask.sum())] *= -1
    else:
        raise ValueError(family)
    return p.astype(np.float32)


def size_to_bucket(s: np.ndarray) -> np.ndarray:
    """Map a physical size to a log-spaced bucket index in [0, N_SIZE_BUCKETS)."""
    lo, hi = np.log(1.0), np.log(50.0)            # the "normal" size range, mm
    b = np.floor((np.log(np.clip(s, 1e-3, None)) - lo) / (hi - lo) * N_SIZE_BUCKETS)
    return np.clip(b, 0, N_SIZE_BUCKETS - 1).astype(int)


class Teacher:
    """Precomputed target embeddings: strong family signal + weak categorical size signal."""
    def __init__(self, dim=128, size_weight=0.30, seed=0):
        g = np.random.default_rng(seed)
        self.e_fam = g.normal(size=(len(FAMILIES), dim)).astype(np.float32)
        self.e_size = g.normal(size=(N_SIZE_BUCKETS, dim)).astype(np.float32)
        self.size_weight = size_weight

    def embed(self, fam_idx: np.ndarray, size: np.ndarray) -> torch.Tensor:
        v = self.e_fam[fam_idx] + self.size_weight * self.e_size[size_to_bucket(size)]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return torch.from_numpy(v.astype(np.float32))


class ShapeDataset(torch.utils.data.Dataset):
    """
    Returns (points[N,3], family_idx, size, is_outlier). Points are at PHYSICAL
    scale (multiplied by size) — normalization happens later in the model, so we
    can swap global-norm vs unit-box without regenerating data.
    """
    def __init__(self, n_items=4096, n_points=1024, outlier_frac=0.006,
                 outlier_mult=(15.0, 5e3), seed=0, with_outliers=True):
        self.n_points = n_points
        rng = np.random.default_rng(seed)
        self.fam = rng.integers(0, len(FAMILIES), n_items)
        # normal sizes: log-normal centred ~8mm, spread within [1,50]
        s = np.exp(rng.normal(np.log(8.0), 0.6, n_items)).clip(1.0, 50.0)
        self.is_outlier = np.zeros(n_items, bool)
        if with_outliers:
            k = int(outlier_frac * n_items)
            idx = rng.choice(n_items, k, replace=False)
            mult = np.exp(rng.uniform(np.log(outlier_mult[0]), np.log(outlier_mult[1]), k))
            s[idx] *= mult
            self.is_outlier[idx] = True
        self.size = s.astype(np.float32)
        self.seed = seed

    def __len__(self): return len(self.fam)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)  # per-item reproducible sampling
        pts = _unit_shape(FAMILIES[self.fam[i]], self.n_points, rng) * self.size[i]
        return (torch.from_numpy(pts), int(self.fam[i]),
                float(self.size[i]), bool(self.is_outlier[i]))


def make_size_pairs(n_per_family=20, base_sizes=(3.0, 4.0), n_points=1024, seed=123):
    """
    Eval probe for SIZE discrimination: same family & same shape sampling, only the
    physical size differs (3mm vs 4mm) — the canonical 'tiny difference' case.
    Returns list of dicts with two point clouds A,B + their teacher embeddings.
    """
    rng = np.random.default_rng(seed)
    items = []
    for fam in range(len(FAMILIES)):
        for _ in range(n_per_family):
            base = _unit_shape(FAMILIES[fam], n_points, rng)  # SAME geometry for A,B
            items.append(dict(fam=fam, base=base.astype(np.float32),
                              sA=base_sizes[0], sB=base_sizes[1]))
    return items
