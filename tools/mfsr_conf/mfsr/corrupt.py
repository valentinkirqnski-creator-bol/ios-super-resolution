"""Deliberate alignment errors, injected AFTER the app's aligner and BEFORE its analytic mask (conf_harness).

Two carriers, both consumed exactly the way the app consumes alignment:
  pert [H,W,2]  raw px; the harness samples it at every tile centre and ADDS it to that tile's vector, so it acts
                at tile granularity whatever tile size the app's SNR tuning picks. Fields are drawn on 16-px blocks
                (the smallest app tile), smooth fields per pixel.
  hpert [3,3]   a homography composed after the flow (lr_mov = Hp(H(lr + flow))): errors that vary INSIDE a tile
                (rotation / scale error), which no per-tile offset can express.

Types (magnitudes are continuous, log-uniform, so subtle errors near the detection limit are well represented):
  tile_subpix   10-50% of tiles, 0.03-1.0 px, random direction
  tile_large    5-20% of tiles, 1-12 px
  smooth        smooth random field (bilinear from a 3x3..6x6 control grid), 0.05-2 px peak
  rotation      0.03-0.6 deg about a random centre (+-0.4% scale)     -> tile-centre-correct, tile-edge-wrong
  block         a rectangular block of tiles (1/8..1/3 of the frame) with one wrong vector 0.2-6 px (wrong local motion)
  edge          tiles on strong reference edges (top 40% edge energy), 0.05-0.8 px ACROSS the edge (normal direction)
  edge_burst    the same edge slip per tile in 40-95% of the frames (+-15%): the merge cannot average it away
Occlusions / motion boundaries need no injection: moving-object scenes make the per-tile flow wrong on one side of
every object boundary by construction (scenarios.py).
"""
import math

import numpy as np

from .formation import rotation_h

BLOCK = 16
TYPES = ["tile_subpix", "tile_large", "smooth", "rotation", "block", "edge"]


def _logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _blocks(S):
    return (S + BLOCK - 1) // BLOCK


def _expand(bf, S):
    """[nb,nb,2] block field -> [S,S,2]."""
    return np.repeat(np.repeat(bf, BLOCK, 0), BLOCK, 1)[:S, :S]


def edge_info(ref_grey, S):
    """Per 16-px block of the reference grey (any resolution, resampled): edge energy rank in [0,1) and the dominant
    edge normal (unit vector, x/y)."""
    g = np.asarray(ref_grey, np.float64)
    if g.shape[0] != S:
        f = S // g.shape[0]
        g = np.repeat(np.repeat(g, f, 0), f, 1)
    gy, gx = np.gradient(g)
    nb = _blocks(S)
    pad = nb * BLOCK - S
    def bsum(a):
        a = np.pad(a, ((0, pad), (0, pad)))
        return a.reshape(nb, BLOCK, nb, BLOCK).sum((1, 3))
    sxx, syy, sxy = bsum(gx * gx), bsum(gy * gy), bsum(gx * gy)
    e = sxx + syy
    rank = np.empty(e.size)
    rank[np.argsort(e.reshape(-1))] = np.arange(e.size) / e.size
    tr, dd = sxx + syy, np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)
    l1 = 0.5 * (tr + dd)
    nx, ny = l1 - syy, sxy
    small = np.abs(nx) + np.abs(ny) < 1e-18
    nx, ny = np.where(small, sxy, nx), np.where(small, l1 - sxx, ny)
    nn = np.sqrt(nx * nx + ny * ny) + 1e-18
    return rank.reshape(nb, nb), np.stack([nx / nn, ny / nn], -1)


def draw(kind, rng, S, edge=None, burst_state=None):
    """One frame's error of the given kind. -> (pert [S,S,2] float32, hpert [3,3] float64)."""
    nb = _blocks(S)
    bf = np.zeros((nb, nb, 2))
    hp = np.eye(3)
    ang = rng.uniform(0, 2 * np.pi, (nb, nb))
    unit = np.stack([np.cos(ang), np.sin(ang)], -1)
    if kind == "tile_subpix":
        m = (rng.random((nb, nb)) < rng.uniform(0.1, 0.5)) * np.exp(rng.uniform(np.log(0.03), np.log(1.0), (nb, nb)))
        bf = unit * m[..., None]
    elif kind == "tile_large":
        m = (rng.random((nb, nb)) < rng.uniform(0.05, 0.2)) * np.exp(rng.uniform(np.log(1.0), np.log(12.0), (nb, nb)))
        bf = unit * m[..., None]
    elif kind == "smooth":
        g = int(rng.integers(3, 7))
        ctrl = rng.normal(0, 1, (g, g, 2))
        ctrl *= _logu(rng, 0.05, 2.0) / (np.abs(ctrl).max() + 1e-9)
        ys = np.linspace(0, g - 1, S)
        y0 = np.clip(np.floor(ys).astype(int), 0, g - 2)
        fy = ys - y0
        rows = ctrl[y0] * (1 - fy)[:, None, None] + ctrl[y0 + 1] * fy[:, None, None]          # [S,g,2]
        x0, fx = y0, fy
        field = rows[:, x0] * (1 - fx)[None, :, None] + rows[:, x0 + 1] * fx[None, :, None]   # [S,S,2]
        return field.astype(np.float32), hp
    elif kind == "rotation":
        deg = rng.choice([-1, 1]) * _logu(rng, 0.03, 0.6)
        cx, cy = rng.uniform(-0.5 * S, 1.5 * S), rng.uniform(-0.5 * S, 1.5 * S)
        hp = rotation_h(deg, cx, cy, 1.0 + rng.uniform(-0.004, 0.004))
    elif kind == "block":
        bh, bw = int(rng.integers(max(1, nb // 8), max(2, nb // 3) + 1)), int(rng.integers(max(1, nb // 8), max(2, nb // 3) + 1))
        by, bx = int(rng.integers(0, nb - bh + 1)), int(rng.integers(0, nb - bw + 1))
        a = rng.uniform(0, 2 * np.pi)
        bf[by:by + bh, bx:bx + bw] = _logu(rng, 0.2, 6.0) * np.array([math.cos(a), math.sin(a)])
    elif kind == "edge":
        rank, nrm = edge
        hit = (rank >= 0.6) & (rng.random((nb, nb)) < rng.uniform(0.3, 0.8))
        m = np.exp(rng.uniform(np.log(0.05), np.log(0.8), (nb, nb))) * rng.choice([-1, 1], (nb, nb))
        tang = 0.2 * m * rng.uniform(-1, 1, (nb, nb))
        bf = hit[..., None] * (m[..., None] * nrm + tang[..., None] * np.stack([-nrm[..., 1], nrm[..., 0]], -1))
    elif kind == "edge_burst":
        st = burst_state
        on = rng.random((nb, nb)) < st["pf"]
        jit = np.clip(1.0 + 0.15 * rng.normal(size=(nb, nb)), 0.5, 1.5)
        bf = (st["hit"] & on)[..., None] * st["vec"] * jit[..., None]
    else:
        raise ValueError(kind)
    return _expand(bf, S).astype(np.float32), hp


def edge_burst_state(rng, S, edge):
    rank, nrm = edge
    nb = _blocks(S)
    hit = (rank >= 0.55) & (rng.random((nb, nb)) < rng.uniform(0.1, 0.4))
    m = rng.uniform(0.08, 0.8, (nb, nb)) * rng.choice([-1, 1], (nb, nb))
    tang = 0.2 * m * rng.uniform(-1, 1, (nb, nb))
    vec = m[..., None] * nrm + tang[..., None] * np.stack([-nrm[..., 1], nrm[..., 0]], -1)
    return dict(hit=hit, vec=vec, pf=rng.uniform(0.4, 0.95, (nb, nb)))


def corrupt_burst(rng, N, S, ref_grey, p_frame, kinds, edge_burst=False):
    """Per comparison frame: with prob p_frame one (sometimes two) error kinds from `kinds`.
    -> pert [N,S,S,2], hpert [N,3,3], kinds_used list[str] per frame ('' = none)."""
    pert = np.zeros((N, S, S, 2), np.float32)
    hpert = np.tile(np.eye(3), (N, 1, 1))
    used = [""] * N
    edge = edge_info(ref_grey, S)
    st = edge_burst_state(rng, S, edge) if edge_burst else None
    for k in range(1, N):
        names = []
        if st is not None:
            p, h = draw("edge_burst", rng, S, edge, st)
            pert[k] += p
            names.append("edge_burst")
        if rng.random() < p_frame:
            for kind in rng.choice(kinds, size=1 + int(rng.random() < 0.25), replace=False):
                p, h = draw(str(kind), rng, S, edge)
                pert[k] += p
                hpert[k] = h @ hpert[k]
                names.append(str(kind))
        used[k] = "+".join(names)
    return pert, hpert, used
