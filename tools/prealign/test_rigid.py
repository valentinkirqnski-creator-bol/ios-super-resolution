"""Does the estimator recover a known rotation and translation?

Run before anything is integrated. A pre-alignment stage is only worth having
if its estimate is accurate to well under the per-tile search's own capture
range; an estimate that is merely close is a seed that makes things worse,
because it moves every tile away from zero without landing near the truth.

The test images are real photographs -- the clean merged frames already in the
mask_refine dumps -- not synthetic texture, because a fit to a 3-parameter
model is easy on broadband noise and hard on the large flat regions and
repeating structure that real scenes contain.

The error that matters is reported in PIXELS AT THE FRAME CORNER, not in
degrees. An angular error is harmless in the middle of the frame and largest
where rotation already hurts most, so the corner displacement error is the
quantity the tile search actually sees:

    corner error = | predicted displacement - true displacement |
                   evaluated at the four corners, worst case

Four conditions, each with the same known transform:
  clean        the warp alone
  noisy        plus read/shot noise at a realistic level
  object       plus an independently moving rectangle (tests the Huber norm;
               a least-squares fit is dragged by this, a robust one is not)
  low light    scaled to a dark exposure, where gradients are weak

    python test_rigid.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_MR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'mask_refine')
sys.path.insert(0, _MR)

from rigid import (estimate_rigid, invert_rigid, rigid_warp, warp_coords,
                   tile_seed)


def load_scenes(n, size):
    """Grey frames from the clean merged images in the mask_refine dumps."""
    import data as D
    paths = sorted(glob.glob(os.path.join(_MR, 'data', 'burst_*.bin')))
    if not paths:
        raise SystemExit('no bursts in %s/data' % _MR)
    out = []
    step = max(1, len(paths) // max(n, 1))
    for p in paths[::step][:n]:
        d = D.read_burst(p)
        g = d['gt'].mean(axis=0)
        h, w = g.shape
        if min(h, w) < size:
            continue
        y = (h - size) // 2
        x = (w - size) // 2
        g = np.ascontiguousarray(g[y:y + size, x:x + size], np.float32)
        if g.mean() < 1e-4:
            continue                      # black crop, nothing to align to
        out.append((os.path.basename(p), g))
    if not out:
        raise SystemExit('no usable scenes')
    return out


def corner_error(h, w, th_t, tx_t, ty_t, th_e, tx_e, ty_e):
    """Worst-case displacement error over the four corners, in pixels."""
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    pts = np.array([[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]], np.float64)
    dy, dx = pts[:, 0] - cy, pts[:, 1] - cx
    def disp(th, tx, ty):
        s, c = np.sin(th), np.cos(th)
        return np.stack([c * dx - s * dy + tx - dx,
                         s * dx + c * dy + ty - dy], axis=1)
    e = disp(th_t, tx_t, ty_t) - disp(th_e, tx_e, ty_e)
    return float(np.max(np.hypot(e[:, 0], e[:, 1])))


def add_noise(img, rng, alpha=2.5e-5, beta=1e-6):
    var = np.maximum(alpha * np.maximum(img, 0.0) + beta, 0.0)
    return (img + rng.standard_normal(img.shape).astype(np.float32)
            * np.sqrt(var).astype(np.float32)).astype(np.float32)


def add_object(ref, cmp_, rng, frac=0.18, shift=9, return_truth=False):
    """An independently moving rectangle, pasted into the comparison frame at
    an offset. The global fit must ignore it.

    With return_truth, also returns (mask, (dx, dy)) describing the block's
    OWN motion. The paste sets cmp(p) = ref(p + shift) inside the block, and a
    tile search solves cmp(p + d) = ref(p), so the true displacement there is
    -shift in both axes -- not the rigid field. Scoring those tiles against the
    rigid field credits a global model for being wrong and penalises a search
    for being right.
    """
    h, w = cmp_.shape
    bh, bw = int(h * frac), int(w * frac)
    y = int(rng.integers(bh, h - 2 * bh))
    x = int(rng.integers(bw, w - 2 * bw))
    out = cmp_.copy()
    out[y:y + bh, x:x + bw] = ref[y + shift:y + shift + bh, x + shift:x + shift + bw]
    if not return_truth:
        return out
    m = np.zeros(cmp_.shape, bool)
    m[y:y + bh, x:x + bw] = True
    return out, m, (-float(shift), -float(shift))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, default=6)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--levels', type=int, default=5)
    ap.add_argument('--iters', type=int, default=12)
    ap.add_argument('--max-deg', type=float, default=1.2)
    ap.add_argument('--max-shift', type=float, default=12.0)
    ap.add_argument('--sweep', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trace', action='store_true')
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    scenes = load_scenes(a.scenes, a.size)
    sweep = (np.radians(np.linspace(-a.max_deg, a.max_deg, 7)).tolist()
             if a.sweep else None)
    print('%d scenes at %dpx, %d trials each, |theta| <= %.2f deg, |t| <= %.1f px'
          % (len(scenes), a.size, a.trials, a.max_deg, a.max_shift))
    print('rotation alone displaces a corner by %.1f px at %.2f deg'
          % (np.hypot(a.size, a.size) / 2 * np.radians(a.max_deg), a.max_deg))
    print()
    conds = ('clean', 'noisy', 'object', 'lowlight')
    print('%-10s %9s %9s %9s %9s' % ('condition', 'corner px', 'median',
                                     'p90', 'worst'))
    results = {}
    for cond in conds:
        errs, base = [], []
        for name, g in scenes:
            for _ in range(a.trials):
                th = float(rng.uniform(-1, 1) * np.radians(a.max_deg))
                tx = float(rng.uniform(-1, 1) * a.max_shift)
                ty = float(rng.uniform(-1, 1) * a.max_shift)
                ref = g
                # Resample by the INVERSE so that the transform the estimator
                # is defined to return is exactly (th, tx, ty). Warping by
                # (th, tx, ty) directly would make the truth its inverse, and
                # scoring against the forward parameters then reports roughly
                # twice the true displacement as error.
                cmp_ = rigid_warp(g, *invert_rigid(th, tx, ty))
                # The warp leaves an invalid border; fit on the inner region
                # so neither frame is scored against zeros. The crop is
                # symmetric, so it carries the rotation centre with it and
                # leaves both theta and t unchanged.
                m = int(np.ceil(abs(tx) + abs(ty))) + int(
                    np.hypot(*g.shape) / 2 * abs(th)) + 4
                sl = (slice(m, -m), slice(m, -m))
                r, c = ref[sl].copy(), cmp_[sl].copy()
                if cond == 'noisy':
                    r, c = add_noise(r, rng), add_noise(c, rng)
                elif cond == 'object':
                    c = add_object(r, c, rng)
                elif cond == 'lowlight':
                    r = add_noise(r * 0.04, rng)
                    c = add_noise(c * 0.04, rng)
                if a.trace:
                    print('  %s %s theta %+.4f t (%+.2f %+.2f)'
                          % (cond, name, th, tx, ty))
                the, txe, tye = estimate_rigid(
                    r, c, levels=a.levels, iters=a.iters,
                    theta_sweep=sweep, trace=a.trace)
                errs.append(corner_error(r.shape[0], r.shape[1],
                                         th, tx, ty, the, txe, tye))
                # What the tile search would face with no seed at all.
                base.append(corner_error(r.shape[0], r.shape[1],
                                         th, tx, ty, 0.0, 0.0, 0.0))
        e = np.array(errs)
        results[cond] = (e, np.array(base))
        print('%-10s %9.3f %9.3f %9.3f %9.3f'
              % (cond, e.mean(), np.median(e), np.percentile(e, 90), e.max()))

    print()
    print('A seed is useful only if the corner error is well inside the tile')
    print('search range. Reported against the uncorrected displacement:')
    for cond in conds:
        e, b = results[cond]
        print('  %-10s corner error %.3f px vs %.2f px uncorrected'
              '  ->  %.0fx reduction'
              % (cond, e.mean(), b.mean(), b.mean() / max(e.mean(), 1e-9)))


if __name__ == '__main__':
    main()
