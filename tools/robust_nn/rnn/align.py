"""Tile aligner run on the *noisy* synthetic frames, so training sees realistic
flow failures instead of perfect correspondences.

Built from scratch in the classic burst-pipeline style:
  * box-downsampled grey pyramid (factors 2,4,4 while the level stays >= 16 px)
  * per level: 3 upsampled candidates (parent + 2 nearest parents), pick the
    best, then exhaustive integer search (L2 coarse, L1 finest)
  * ICA / Lucas-Kanade sub-pixel refinement per finest tile
Output: one (dx,dy) per T x T tile in grey pixels, ref -> comparison.

ExternalAligner lets you substitute flows from the production implementation
(see README "Using the app's own aligner"): the strongest training signal is
the real aligner's real failures.
"""
import os
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

from .common import bilinear_clamp


def _down(x, f):
    return F.avg_pool2d(x[None, None], f)[0, 0]


def _tiles_idx(th, tw, t):
    yy = (torch.arange(th) * t).view(th, 1, 1, 1) + torch.arange(t).view(1, 1, t, 1)
    xx = (torch.arange(tw) * t).view(1, tw, 1, 1) + torch.arange(t).view(1, 1, 1, t)
    return yy, xx


def _cost(ref_t, cmp, yy, xx, dy, dx, norm):
    H, W = cmp.shape
    y = yy + dy.view(*dy.shape, 1, 1)
    x = xx + dx.view(*dx.shape, 1, 1)
    valid = ((y >= 0) & (y < H) & (x >= 0) & (x < W)).float()
    d = cmp[y.clamp(0, H - 1), x.clamp(0, W - 1)] - ref_t
    d = d.abs() if norm == 1 else d * d
    # score in-bounds pixels only; clamped borders otherwise form degenerate flat "matches"
    nv = valid.sum(dim=(-1, -2))
    c = (d * valid).sum(dim=(-1, -2)) / nv.clamp_min(1)
    return c + 1e3 * (nv < 0.75 * d.shape[-1] * d.shape[-2]).float()


def _match(ref, cmp, t, init, radius, norm):
    H, W = ref.shape
    th, tw = H // t, W // t
    yy, xx = _tiles_idx(th, tw, t)
    ref_t = ref[yy, xx]
    best = torch.full((th, tw), float("inf"))
    bo = torch.zeros(th, tw, 2, dtype=torch.long)
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            c = _cost(ref_t, cmp, yy, xx, init[..., 1] + oy, init[..., 0] + ox, norm)
            m = c < best
            best = torch.where(m, c, best)
            bo[m] = torch.tensor([ox, oy])
    return init + bo, best


def _candidates(ref, cmp, t, parent, f, t_par, norm):
    """Upsample parent tile flows (HDR+ 3-candidate scheme) and keep the best."""
    H, W = ref.shape
    th, tw = H // t, W // t
    pth, ptw = parent.shape[:2]
    cy = (torch.arange(th).float() + 0.5) * t / f / t_par
    cx = (torch.arange(tw).float() + 0.5) * t / f / t_par
    py = cy.floor().long().clamp(0, pth - 1)
    px = cx.floor().long().clamp(0, ptw - 1)
    ny = torch.where(cy - py.float() < 0.5, py - 1, py + 1).clamp(0, pth - 1)
    nx = torch.where(cx - px.float() < 0.5, px - 1, px + 1).clamp(0, ptw - 1)
    yy, xx = _tiles_idx(th, tw, t)
    ref_t = ref[yy, xx]
    best = None
    for (Y, X) in [(py[:, None].expand(th, tw), px[None, :].expand(th, tw)),
                   (ny[:, None].expand(th, tw), px[None, :].expand(th, tw)),
                   (py[:, None].expand(th, tw), nx[None, :].expand(th, tw))]:
        cand = torch.floor(parent[Y, X].float() * f + 0.5).long()
        c = _cost(ref_t, cmp, yy, xx, cand[..., 1], cand[..., 0], norm)
        if best is None:
            best, bc = cand, c
        else:
            m = c < bc
            best = torch.where(m[..., None], cand, best)
            bc = torch.where(m, c, bc)
    return best


def _ica(ref, cmp, t, flow, iters):
    H, W = ref.shape
    th, tw = H // t, W // t
    yy, xx = _tiles_idx(th, tw, t)
    yy, xx = yy.expand(th, tw, t, t), xx.expand(th, tw, t, t)
    rp = F.pad(ref[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    gx = 0.5 * (rp[1:-1, 2:] - rp[1:-1, :-2])
    gy = 0.5 * (rp[2:, 1:-1] - rp[:-2, 1:-1])
    gxt, gyt, rt = gx[yy, xx], gy[yy, xx], ref[yy, xx]
    hxx = (gxt * gxt).sum((-1, -2))
    hxy = (gxt * gyt).sum((-1, -2))
    hyy = (gyt * gyt).sum((-1, -2))
    lam = 0.01 * (hxx + hyy) + 1e-12          # damps the aperture direction (edges, flat tiles)
    hxx, hyy = hxx + lam, hyy + lam
    det = hxx * hyy - hxy * hxy
    T2 = t * t

    def warp_cost(f):
        J = bilinear_clamp(cmp[None], (xx.float() + f[..., 0, None, None]).reshape(1, th * tw, T2),
                           (yy.float() + f[..., 1, None, None]).reshape(1, th * tw, T2)).reshape(th, tw, t, t)
        r = rt - J
        return r, (r * r).mean((-1, -2))

    f0 = flow.float().clone()
    f = f0.clone()
    r, cost = warp_cost(f)
    for _ in range(iters):
        bx = (gxt * r).sum((-1, -2))
        by = (gyt * r).sum((-1, -2))
        d = torch.stack([(hyy * bx - hxy * by) / det, (hxx * by - hxy * bx) / det], -1).clamp(-1, 1)
        fn = f0 + (f + d - f0).clamp(-2, 2)
        rn, cn = warp_cost(fn)
        better = cn < cost                      # only accept steps that lower the tile SSD
        f = torch.where(better[..., None], fn, f)
        r = torch.where(better[..., None, None], rn, r)
        cost = torch.where(better, cn, cost)
    return f, cost


class TileAligner:
    def __init__(self, tile=16, coarse_tile=8, factors=(2, 4, 4), radii=(4, 2, 2), fine_radius=1,
                 ica_iters=3, min_size=16):
        self.tile, self.ct, self.factors = tile, coarse_tile, factors
        self.radii, self.fr, self.iters, self.min_size = radii, fine_radius, ica_iters, min_size

    def __call__(self, ref, cmp):
        """ref, cmp: grey [H,W] float tensors. Returns tile flow [th,tw,2] (x,y), cost [th,tw]."""
        pr, pc, fs = [ref], [cmp], []
        for f in self.factors:
            if min(pr[-1].shape) // f < self.min_size:
                break
            pr.append(_down(pr[-1], f))
            pc.append(_down(pc[-1], f))
            fs.append(f)
        L = len(pr) - 1
        flow, t_prev = None, None
        for lvl in range(L, -1, -1):
            t = self.tile if lvl == 0 else self.ct
            H, W = pr[lvl].shape
            th, tw = H // t, W // t
            if flow is None:
                init = torch.zeros(th, tw, 2, dtype=torch.long)
            else:
                init = _candidates(pr[lvl], pc[lvl], t, flow, fs[lvl], t_prev, 2 if lvl else 1)
            if lvl == 0:
                flow, _ = _match(pr[0], pc[0], t, init, self.fr, 1)
            else:
                flow, _ = _match(pr[lvl], pc[lvl], t, init, self.radii[min(L - lvl, len(self.radii) - 1)], 2)
            t_prev = t
        return _ica(pr[0], pc[0], self.tile, flow, self.iters)


class ExternalAligner:
    """Runs `cmd ref.f32 cmp.f32 W H tile out.f32`; out holds th*tw*2 float32 (x,y) grey px."""

    def __init__(self, cmd, tile=16):
        self.cmd, self.tile = cmd, tile

    def __call__(self, ref, cmp):
        H, W = ref.shape
        th, tw = H // self.tile, W // self.tile
        with tempfile.TemporaryDirectory() as d:
            pr, pc, po = (os.path.join(d, n) for n in ("ref.f32", "cmp.f32", "out.f32"))
            ref.numpy().astype(np.float32).tofile(pr)
            cmp.numpy().astype(np.float32).tofile(pc)
            subprocess.run([*self.cmd.split(), pr, pc, str(W), str(H), str(self.tile), po], check=True)
            fl = torch.from_numpy(np.fromfile(po, np.float32).reshape(th, tw, 2).copy())
        return fl, torch.zeros(th, tw)


def densify(tile_flow, H, W, tile, mode="nearest"):
    """Tile flow [th,tw,2] -> dense [2,H,W]. 'nearest' = what a non-overlapping
    tile merge uses; 'bilinear' interpolates between tile centres."""
    th, tw = tile_flow.shape[:2]
    if mode == "nearest":
        iy = (torch.arange(H) // tile).clamp(max=th - 1)
        ix = (torch.arange(W) // tile).clamp(max=tw - 1)
        return tile_flow[iy][:, ix].permute(2, 0, 1).contiguous()
    cy = ((torch.arange(H).float() + 0.5) / tile - 0.5).clamp(0, th - 1)
    cx = ((torch.arange(W).float() + 0.5) / tile - 0.5).clamp(0, tw - 1)
    out = []
    for c in range(2):
        img = tile_flow[..., c][None]
        out.append(bilinear_clamp(img, cx.view(1, 1, W).expand(1, H, W), cy.view(1, H, 1).expand(1, H, W))[0])
    return torch.stack(out)
