"""Deliberate corruption of otherwise-good tile flow fields.

These teach *alignment failure* rather than "which photos are hard": the same
scene appears with clean and with corrupted correspondences, and only the
evidence in (ref, warped cmp, flow) can tell them apart.
"""
import numpy as np
import torch
import torch.nn.functional as F


def corrupt(tile_flow, ref_grey, tile, rng):
    """tile_flow [th,tw,2] float. Returns corrupted copy and the list of ops used."""
    f = tile_flow.clone()
    th, tw = f.shape[:2]
    ops = []

    def rvec(lo, hi, n=None):
        n = n or 1
        a = rng.uniform(0, 2 * np.pi, n)
        m = rng.uniform(lo, hi, n)
        return torch.from_numpy(np.stack([m * np.cos(a), m * np.sin(a)], -1)).float()

    def rmask(p):
        return torch.from_numpy(rng.random((th, tw)) < p)

    choices = ["subpixel_bias", "smooth_bias", "tile_errors", "inconsistent", "outliers", "edge_bias"]
    n = int(rng.integers(1, 3))
    for op in rng.choice(choices, n, replace=False):
        ops.append(str(op))
        if op == "subpixel_bias":
            f += rvec(0.1, 0.7)[0]
        elif op == "smooth_bias":
            g = torch.from_numpy(rng.normal(0, 1, (1, 2, 3, 3))).float() * rng.uniform(0.3, 2.0)
            g = F.interpolate(g, size=(th, tw), mode="bicubic", align_corners=True)[0]
            f += g.permute(1, 2, 0)
        elif op == "tile_errors":
            m = rmask(rng.uniform(0.05, 0.3))
            f[m] += rvec(0.5, 3.0, int(m.sum()))
        elif op == "inconsistent":
            m = rmask(rng.uniform(0.05, 0.2))
            f[m] = f[m] + torch.from_numpy(rng.uniform(-4, 4, (int(m.sum()), 2))).float()
        elif op == "outliers":
            k = int(rng.integers(1, max(2, th * tw // 16)))
            idx = rng.choice(th * tw, k, replace=False)
            ff = f.view(-1, 2)
            ff[torch.from_numpy(idx)] += rvec(4.0, 10.0, k)
        elif op == "edge_bias":
            H, W = ref_grey.shape
            gy = ref_grey[2:, 1:-1] - ref_grey[:-2, 1:-1]
            gx = ref_grey[1:-1, 2:] - ref_grey[1:-1, :-2]
            e = F.pad((gx * gx + gy * gy)[None, None], (1, 1, 1, 1))[0, 0]
            e = F.avg_pool2d(e[None, None, :th * tile, :tw * tile], tile)[0, 0]
            m = e > torch.quantile(e.flatten(), rng.uniform(0.5, 0.85))
            f[m] += rvec(0.25, 1.5, int(m.sum()))
    return f, ops
