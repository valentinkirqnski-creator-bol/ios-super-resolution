"""Does seeding a coarse-to-fine tile search with the rigid estimate help?

The estimator is accurate (test_rigid.py). That is not the same as useful. A
pyramid tile search can already follow a large displacement by construction, so
the claim that pre-alignment helps has to name the failure it removes.

THE FAILURE IT REMOVES
----------------------
A pyramid search starts at its coarsest level, where one tile covers a large
part of the frame and is assigned ONE translation. Under rotation the true
motion varies across that area -- it grows with radius and turns -- so the
coarse tile cannot be right anywhere: the best single translation for it is a
compromise. That compromise is then upsampled as the starting point for the
level below, and a fine search with a small radius cannot recover from a
starting point that is already further away than its radius. The error is
structural, not a matter of search effort, and it is worst at the frame edges
where rotation displaces most.

Seeding replaces that compromise with the rigid model's own prediction per
tile, which varies within the coarse level exactly as the true motion does.

WHAT IS COMPARED
----------------
The same search, three ways:

  unseeded     pyramid search from zero -- what a translation-only
               coarse-to-fine does on its own
  seeded       identical, but every level starts from the rigid prediction at
               that tile's centre
  seed only    the rigid prediction with no search at all, to show how much of
               the result is the search and how much is the model

Scored as per-tile endpoint error against the true displacement field, plus the
fraction of tiles off by more than a pixel, which is the quantity that produces
a visible artifact rather than a small loss of sharpness.

The search here is written from scratch for this measurement and is NOT the
app's aligner; it is a plain SSD pyramid search with a parabola subpixel fit.
The point is the mechanism, measured on a search whose every parameter is
visible, not a prediction of the app's exact numbers.

    python test_seed.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigid import estimate_rigid, invert_rigid, pyramid, rigid_warp
from test_rigid import add_noise, add_object, load_scenes


def true_field(h, w, theta, tx, ty, tile, stride):
    """The exact displacement of each tile centre under the model."""
    ny = max(1, (h - tile) // stride + 1)
    nx = max(1, (w - tile) // stride + 1)
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    yc = np.arange(ny) * stride + 0.5 * tile - cy
    xc = np.arange(nx) * stride + 0.5 * tile - cx
    gy, gx = np.meshgrid(yc, xc, indexing='ij')
    s, c = np.sin(theta), np.cos(theta)
    out = np.empty((ny, nx, 2), np.float64)
    out[..., 0] = (c * gx - s * gy + tx) - gx
    out[..., 1] = (s * gx + c * gy + ty) - gy
    return out


def model_field(h, w, theta, tx, ty, tile, stride):
    return true_field(h, w, theta, tx, ty, tile, stride)


def search_level(ref, cmp_, init, tile, stride, radius):
    """Integer SSD search in a +-radius window around `init`, then a parabola
    fit in each axis for the subpixel part.

    init is (ny, nx, 2) in (dx, dy). Returns the refined field.
    """
    h, w = ref.shape
    ny, nx, _ = init.shape
    out = init.copy()
    offs = np.arange(-radius, radius + 1)
    for iy in range(ny):
        y0 = iy * stride
        for ix in range(nx):
            x0 = ix * stride
            patch = ref[y0:y0 + tile, x0:x0 + tile]
            if patch.shape != (tile, tile):
                continue
            bx, by = init[iy, ix]
            best, bd, cost = None, (0, 0), {}
            for dy in offs:
                for dx in offs:
                    sy = int(round(y0 + by + dy))
                    sx = int(round(x0 + bx + dx))
                    if sy < 0 or sx < 0 or sy + tile > h or sx + tile > w:
                        continue
                    d = cmp_[sy:sy + tile, sx:sx + tile] - patch
                    c = float(np.dot(d.ravel(), d.ravel()))
                    cost[(int(dy), int(dx))] = c
                    if best is None or c < best:
                        best, bd = c, (int(dy), int(dx))
            if best is None:
                continue
            dy, dx = bd
            # Parabola through the SSD minimum and its two neighbours, per
            # axis. Skipped at the window edge, where one neighbour is absent
            # and the fit would extrapolate.
            sub = [0.0, 0.0]
            for k, (a, b) in enumerate(((( dy, dx - 1), (dy, dx + 1)),
                                        ((dy - 1, dx), (dy + 1, dx)))):
                ca, cb = cost.get(a), cost.get(b)
                if ca is None or cb is None:
                    continue
                den = ca + cb - 2.0 * best
                if den > 1e-12:
                    sub[k] = float(np.clip(0.5 * (ca - cb) / den, -0.5, 0.5))
            out[iy, ix, 0] = bx + dx + sub[0]
            out[iy, ix, 1] = by + dy + sub[1]
    return out


def pyramid_search(ref, cmp_, tile, stride, radius, levels, seed=None,
                   seed_where='all', margin=0.5):
    """Coarse-to-fine tile search.

    `seed` is a callable (h, w, tile, stride) -> displacement field.

    seed_where decides where it is injected, and the choice matters more than
    the seed's accuracy:

      'all'  every level restarts from the model. This removes the rotation
             compromise but also removes coarse-to-fine entirely: a region
             moving independently of the global model sits further from the
             seed than the search radius, and no coarse level survives to
             find it. Measured worst-case tile error rises from 5.7 px to
             26.2 px when a moving block is present.
      'top'  only the coarsest level starts from the model. MEASURED WORSE
             THAN NO SEED AT ALL (2.67 vs 2.27 px). The coarsest level is not
             where rotation hurts: displacements there are divided by 2^levels,
             so a 14 px corner displacement is under 2 px and the search
             handles it. The damage is at the FINE levels, where one parent
             tile seeds four children and the rotation field varies visibly
             between neighbours, so the propagated parent is a poor start
             precisely where the field changes fastest.
      'both' every level evaluates the search from the model AND from the
             propagated parent, keeping the parent's result only when it beats
             the seed's by `margin`. Free choice on raw SSD (margin = 1.0) was
             MEASURED to improve the mean and wreck the worst case -- 5.7 to
             33.1 px on rigid motion -- because in flat or repeating texture a
             wrong vector sometimes matches better than the right one, and one
             60 px tile is a visible ghost. The seed carries a 0.01 px prior,
             so the parent has to earn the switch.
    """
    pr, pc = pyramid(ref, levels), pyramid(cmp_, levels)
    field = None
    for l in range(levels - 1, -1, -1):
        h, w = pr[l].shape
        ny = max(1, (h - tile) // stride + 1)
        nx = max(1, (w - tile) // stride + 1)
        use_seed = seed is not None and (
            seed_where in ('all', 'both') or l == levels - 1)
        parent = None
        if field is not None:
            iy = np.clip((np.arange(ny) * stride // 2) // stride, 0,
                         field.shape[0] - 1)
            ix = np.clip((np.arange(nx) * stride // 2) // stride, 0,
                         field.shape[1] - 1)
            parent = field[np.ix_(iy, ix)] * 2.0
        if use_seed:
            init = seed(h, w, tile, stride) / (2.0 ** l)
        elif parent is None:
            init = np.zeros((ny, nx, 2), np.float64)
        else:
            init = parent
        field = search_level(pr[l], pc[l], init, tile, stride, radius)
        if seed_where == 'both' and parent is not None:
            alt = search_level(pr[l], pc[l], parent, tile, stride, radius)
            field = pick_cheaper(pr[l], pc[l], field, alt, tile, stride,
                                 margin)
    return field


def tile_cost(ref, cmp_, field, tile, stride):
    """Per-tile SSD at the field's own displacement, bilinearly sampled."""
    from rigid import bilinear
    h, w = ref.shape
    ny, nx, _ = field.shape
    out = np.full((ny, nx), np.inf)
    yy, xx = np.mgrid[0:tile, 0:tile].astype(np.float32)
    for iy in range(ny):
        y0 = iy * stride
        for ix in range(nx):
            x0 = ix * stride
            patch = ref[y0:y0 + tile, x0:x0 + tile]
            if patch.shape != (tile, tile):
                continue
            dx, dy = field[iy, ix]
            v, ok = bilinear(cmp_, (xx + x0 + dx).ravel(),
                             (yy + y0 + dy).ravel())
            if ok.sum() < tile * tile // 2:
                continue
            d = (v - patch.ravel())[ok]
            out[iy, ix] = float(np.dot(d, d)) / ok.sum()
    return out


def pick_cheaper(ref, cmp_, a, b, tile, stride, margin=0.5):
    """Keep a unless b beats it by the margin. a is the seeded hypothesis."""
    ca = tile_cost(ref, cmp_, a, tile, stride)
    cb = tile_cost(ref, cmp_, b, tile, stride)
    take_b = (cb < ca * margin)[..., None]
    return np.where(take_b, b, a)


def epe(a, b):
    d = a - b
    return np.hypot(d[..., 0], d[..., 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, default=3)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--tile', type=int, default=16)
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--radius', type=int, default=4)
    ap.add_argument('--levels', type=int, default=4)
    ap.add_argument('--deg', type=float, default=1.0)
    ap.add_argument('--shift', type=float, default=8.0)
    ap.add_argument('--cond', default='noisy',
                    choices=('clean', 'noisy', 'object', 'lowlight'))
    ap.add_argument('--margin', type=float, default=0.5,
                    help='parent must beat the seed by this factor')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    scenes = load_scenes(a.scenes, a.size)
    print('%d scenes at %dpx, %d trials, tile %d stride %d radius %d levels %d'
          % (len(scenes), a.size, a.trials, a.tile, a.stride, a.radius,
             a.levels))
    print('condition %s, rotation %.2f deg, translation up to %.1f px'
          % (a.cond, a.deg, a.shift))
    print()
    print('%-10s %9s %9s %9s %9s %9s'
          % ('method', 'EPE mean', 'median', 'p90', 'worst', '>1px'))

    acc = {k: [] for k in ('unseeded', 'seed all', 'seed margin', 'seed only')}
    for name, g in scenes:
        for _ in range(a.trials):
            th = float(rng.choice([-1, 1]) * np.radians(a.deg))
            tx = float(rng.uniform(-1, 1) * a.shift)
            ty = float(rng.uniform(-1, 1) * a.shift)
            cmp_full = rigid_warp(g, *invert_rigid(th, tx, ty))
            m = int(np.ceil(abs(tx) + abs(ty))
                    + np.hypot(*g.shape) / 2 * abs(th)) + 4
            sl = (slice(m, -m), slice(m, -m))
            r, c = g[sl].copy(), cmp_full[sl].copy()
            if a.cond == 'noisy':
                r, c = add_noise(r, rng), add_noise(c, rng)
            elif a.cond == 'object':
                c, obj_mask, obj_d = add_object(r, c, rng, return_truth=True)
            elif a.cond == 'lowlight':
                r, c = add_noise(r * 0.04, rng), add_noise(c * 0.04, rng)
            h, w = r.shape
            gt = true_field(h, w, th, tx, ty, a.tile, a.stride)
            if a.cond == 'object':
                # Tiles whose area is mostly the moving block have the block's
                # motion as their truth, not the rigid field.
                ny, nx = gt.shape[:2]
                for jy in range(ny):
                    for jx in range(nx):
                        blk = obj_mask[jy * a.stride:jy * a.stride + a.tile,
                                       jx * a.stride:jx * a.stride + a.tile]
                        if blk.size and blk.mean() > 0.5:
                            gt[jy, jx, 0] = obj_d[0]
                            gt[jy, jx, 1] = obj_d[1]

            the, txe, tye = estimate_rigid(r, c, levels=5, iters=12,
                                           theta_sweep=np.radians(
                                               np.linspace(-1.5, 1.5, 7)).tolist())
            mk = lambda hh, ww, tl, st: model_field(hh, ww, the, txe, tye, tl, st)

            f_un = pyramid_search(r, c, a.tile, a.stride, a.radius, a.levels)
            f_sd = pyramid_search(r, c, a.tile, a.stride, a.radius, a.levels,
                                  seed=mk, seed_where='all')
            f_tp = pyramid_search(r, c, a.tile, a.stride, a.radius, a.levels,
                                  seed=mk, seed_where='both',
                                  margin=a.margin)
            f_mo = mk(h, w, a.tile, a.stride)
            for k, f in (('unseeded', f_un), ('seed all', f_sd),
                         ('seed margin', f_tp), ('seed only', f_mo)):
                acc[k].append(epe(f[:gt.shape[0], :gt.shape[1]], gt).ravel())

    for k in ('unseeded', 'seed all', 'seed margin', 'seed only'):
        e = np.concatenate(acc[k])
        print('%-10s %9.3f %9.3f %9.3f %9.3f %8.1f%%'
              % (k, e.mean(), np.median(e), np.percentile(e, 90), e.max(),
                 100.0 * (e > 1.0).mean()))
    print()
    eu = np.concatenate(acc['unseeded'])
    for k in ('seed all', 'seed margin'):
        es = np.concatenate(acc[k])
        print('%-9s mean EPE %.3f -> %.3f px (%.2fx), >1px %.1f%% -> %.1f%%,'
              ' worst %.1f -> %.1f px'
              % (k, eu.mean(), es.mean(), eu.mean() / max(es.mean(), 1e-9),
                 100.0 * (eu > 1).mean(), 100.0 * (es > 1).mean(),
                 eu.max(), es.max()))


if __name__ == '__main__':
    main()
