"""Training windows for the learned SRSR mask.

A window = a square quad crop (the network's input, with a halo so every mask
texel the merge reads has full receptive field) and the output pixels whose
merge reads only interior mask texels. For each window: the network inputs of
every comparison frame, the exact merge terms (srsr_merge.terms_uv), the
noise-free target and a ghost map.
"""
import collections
import os
import random

import numpy as np
import torch

import srsr_merge as sm


def frame_noise(meta, n):
    """Per frame, per CFA site (R G1 G2 B): Var(x) = a x + b in the merge's units
    (pre-white-balanced and exposure-matched raw), from the sensor-domain profile."""
    wb = np.asarray(meta["wb"], np.float64)
    w = np.array([wb[0], wb[1], wb[1], wb[2]]) / wb[1]
    E = meta["exposure"]
    a, b = [], []
    for f in range(n):
        g = 1.0
        if f > 0 and E[0] > 0 and E[f] > 0:
            g = min(1.5, max(0.67, E[0] / E[f]))
            if abs(g - 1.0) <= 0.002:
                g = 1.0
        a.append(g * w * meta["noise_a"])
        b.append(g * g * w * w * meta["noise_b"])
    return torch.tensor(np.array(a), dtype=torch.float32), torch.tensor(np.array(b), dtype=torch.float32)


class Item:
    """One decoded sample, reused for many windows."""

    def __init__(self, path):
        z = np.load(path)
        self.path = path
        self.S = sm.Sample(z)
        self.meta = sm.meta(z)
        self.masks = torch.from_numpy(z["masks"].astype(np.float32) / 255.0)       # [F, qH, qW]
        self.target = torch.from_numpy(z["target"].astype(np.float32))           # [oH, oW, 3]
        self.a, self.b = frame_noise(self.meta, self.S.n)
        self.rggb = [self.S.rggb(f).permute(2, 0, 1).contiguous() for f in range(self.S.n)]   # [4, qH, qW]


class Pool:
    """LRU of decoded samples (decoding a 20 MB npz costs ~0.5 s)."""

    def __init__(self, paths, size=12):
        self.paths, self.size = list(paths), size
        self.cache = collections.OrderedDict()

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        it = Item(path)
        self.cache[path] = it
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return it


def _crop_clamped(img, x0, y0, w, h):
    """img [C, H, W] -> [C, h, w] at (x0, y0), clamp-to-edge outside."""
    H, W = img.shape[-2:]
    xs = torch.arange(x0, x0 + w).clamp(0, W - 1)
    ys = torch.arange(y0, y0 + h).clamp(0, H - 1)
    return img[:, ys][:, :, xs]


def window(it, q, halo, margin, rng, x0=None, y0=None):
    """A window of q x q quads (incl. a halo of `halo` on each side). Returns a dict of tensors."""
    S, u = it.S, it.S.u
    qW, qH, oW, oH = u["quadW"], u["quadH"], u["outW"], u["outH"]
    q = min(q, qW, qH)
    if x0 is None:
        x0 = int(rng.integers(0, qW - q + 1))
        y0 = int(rng.integers(0, qH - q + 1))
    # Output pixels whose bilinear mask fetch reads only interior texels [x0+halo, x0+q-halo).
    lo_x, hi_x = (x0 + halo if x0 > 0 else 0), (x0 + q - halo if x0 + q < qW else qW)
    lo_y, hi_y = (y0 + halo if y0 > 0 else 0), (y0 + q - halo if y0 + q < qH else qH)

    def out_range(lo, hi, n_q, n_o):
        o = torch.arange(n_o, dtype=torch.float32)
        f = (o + 0.5) / n_o * n_q - 0.5
        b0 = torch.floor(f).long().clamp(0, n_q - 1)
        b1 = (torch.floor(f).long() + 1).clamp(0, n_q - 1)
        ok = (b0 >= lo) & (b1 < hi)
        idx = torch.nonzero(ok).flatten()
        return int(idx.min()), int(idx.max()) + 1

    ox0, ox1 = out_range(lo_x, hi_x, qW, oW)
    oy0, oy1 = out_range(lo_y, hi_y, qH, oH)
    yy, xx = torch.meshgrid(torch.arange(oy0, oy1, dtype=torch.float32),
                            torch.arange(ox0, ox1, dtype=torch.float32), indexing="ij")
    uv = torch.stack([(xx.reshape(-1) + 0.5) / oW, (yy.reshape(-1) + 0.5) / oH], 1)

    # Network inputs per comparison frame.
    qy, qx = torch.meshgrid(torch.arange(y0, y0 + q, dtype=torch.float32),
                            torch.arange(x0, x0 + q, dtype=torch.float32), indexing="ij")
    uvq = torch.stack([(qx.reshape(-1) + 0.5) / qW, (qy.reshape(-1) + 0.5) / qH], 1)
    movs, poss, flows = [], [], []
    for f in range(1, S.n):
        fl = S.flow_at(f, uvq) * torch.tensor([float(qW), float(qH)])          # quad px [q*q, 2]
        fl = fl.T.reshape(2, q, q)
        med = torch.round(fl.reshape(2, -1).median(1).values).long()
        mx0, my0 = x0 + int(med[0]) - margin, y0 + int(med[1]) - margin
        movs.append(_crop_clamped(it.rggb[f], mx0, my0, q + 2 * margin, q + 2 * margin))
        poss.append(torch.stack([qx - mx0 + fl[0], qy - my0 + fl[1]]))
        flows.append(fl)
    t = sm.terms_uv(S, uv)
    tgt = it.target[oy0:oy1, ox0:ox1].reshape(-1, 3)
    return dict(
        ref=_crop_clamped(it.rggb[0], x0, y0, q, q),
        mov=torch.stack(movs), pos=torch.stack(poss), flow=torch.stack(flows),
        analytic=it.masks[:, y0:y0 + q, x0:x0 + q][:, None],
        a=it.a, b=it.b, terms=t, uv=uv, x0=x0, y0=y0, out_hw=(oy1 - oy0, ox1 - ox0),
        target=tgt, sigma=noise_sigma(tgt, it.a[0], it.b[0]), meta=it.meta)


def noise_sigma(tgt, a, b):
    """Per-pixel noise std of one frame at the target's level, per output channel (R, G, B)."""
    ac = torch.stack([a[0], 0.5 * (a[1] + a[2]), a[3]])
    bc = torch.stack([b[0], 0.5 * (b[1] + b[2]), b[3]])
    return torch.sqrt((ac * tgt.clamp_min(0) + bc).clamp_min(1e-12))


def ghost_map(w):
    """Where keeping a comparison frame makes the merge visibly WORSE than the reference alone:
    max over frames of (|2-frame merge - target| - |reference only - target|) in noise std,
    channel-mean, per output pixel [P]. Uses only exact merge terms; independent of any mask."""
    t, tgt, sig = w["terms"], w["target"], w["sigma"]
    eps = t["u"]["dehomogenizeEpsilon"]
    ref_only = t["A0"] / t["B0"].clamp_min(eps)
    e0 = (ref_only - tgt).abs()
    two = (t["A0"][None] + t["A"]) / (t["B0"][None] + t["B"]).clamp_min(eps)
    ef = (two - tgt[None]).abs()
    return ((ef - e0[None]) / sig[None]).mean(-1).max(0).values


def split(paths_dir, val_frac=0.0):
    files = sorted(os.path.join(paths_dir, f) for f in os.listdir(paths_dir) if f.endswith(".npz"))
    random.Random(0).shuffle(files)
    nv = int(len(files) * val_frac)
    return files[nv:], files[:nv]
