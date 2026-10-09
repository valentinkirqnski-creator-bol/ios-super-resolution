"""Global rotation + translation between two frames, by coarse-to-fine
Gauss-Newton.

Written from scratch; no existing alignment code in this repo was consulted.

WHAT THIS IS FOR
----------------
The per-tile alignment that follows assigns ONE translation to each tile. A
rotation about the frame centre produces a displacement that grows with radius
and changes direction across the frame, so a translation-only search has to
rediscover it independently in every tile, at a displacement that is largest
exactly where the search range is most likely to be exceeded. Removing the
global part first leaves a residual the per-tile search is well suited to.

The estimate is a SEED, not a resampling. The comparison frames are CFA
mosaics: a rotation maps a pixel to a non-lattice position whose colour belongs
to a different Bayer phase, so warping the mosaic would interpolate across
colour phases and destroy the sub-pixel content the merge depends on. The three
parameters are handed to the tile search as a starting displacement instead,
and no pixel is ever resampled.

THE MODEL
---------
    p' = R(theta) (p - c) + c + t

with c the frame centre, R a rotation, t a translation in pixels. Three
parameters, fitted by minimising

    sum_p  rho( cmp(p') - ref(p) )

over a grey pyramid, with rho a Huber norm so that an independently moving
object or an occlusion cannot drag the global fit. The Jacobian is analytic:

    dp'/dtheta = R'(theta) (p - c),   R' = [[-s, -c], [c, -s]]
    dp'/dtx    = (1, 0)
    dp'/dty    = (0, 1)

so each iteration is a 3x3 normal-equation solve, and there is nothing to train
and nothing to tune beyond the iteration counts.

Scale is deliberately NOT a parameter. Hand-held bursts over a few hundred
milliseconds have negligible scale change, and a free scale parameter is the
one that most readily absorbs brightness mismatch into a spurious zoom.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- sampling

def bilinear(img, x, y):
    """Sample img at float coords. Returns (values, valid_mask).

    Out-of-bounds samples are reported invalid rather than clamped: clamping
    invents a plateau at the border, and a fit that is free to translate will
    happily slide into it to reduce the residual.
    """
    h, w = img.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    xi = np.clip(x0, 0, w - 2)
    yi = np.clip(y0, 0, h - 2)
    v00 = img[yi, xi]
    v01 = img[yi, xi + 1]
    v10 = img[yi + 1, xi]
    v11 = img[yi + 1, xi + 1]
    top = v00 + (v01 - v00) * fx
    bot = v10 + (v11 - v10) * fx
    return top + (bot - top) * fy, ok


def gradients(img):
    """Central differences, one-sided at the border."""
    gx = np.empty_like(img)
    gy = np.empty_like(img)
    gx[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    gx[:, 0] = img[:, 1] - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]
    gy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    gy[0, :] = img[1, :] - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]
    return gx, gy


def halve(img):
    """2x2 box downsample, dropping an odd last row/column."""
    h, w = (img.shape[0] // 2) * 2, (img.shape[1] // 2) * 2
    a = img[:h, :w].astype(np.float32)
    return 0.25 * (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2])


def pyramid(img, levels):
    """[finest, ..., coarsest]."""
    out = [np.ascontiguousarray(img, dtype=np.float32)]
    for _ in range(levels - 1):
        out.append(halve(out[-1]))
    return out


# ------------------------------------------------------------------ warp

def warp_coords(h, w, theta, tx, ty):
    """Coordinates in the comparison frame for every reference pixel."""
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    s, c = np.sin(theta), np.cos(theta)
    return (c * dx - s * dy + cx + tx,
            s * dx + c * dy + cy + ty)


def invert_rigid(theta, tx, ty):
    """Parameters of the inverse map.

    The convention matters and is easy to get backwards. estimate_rigid returns
    the transform W with cmp(W(p)) ~ ref(p), i.e. W maps a REFERENCE position
    to where that content sits in the COMPARISON frame -- which is the
    direction the tile search wants, since it looks up the comparison frame at
    reference position + displacement. To synthesise a comparison frame with a
    known W you must resample by W inverse, not by W.
    """
    s, c = np.sin(-theta), np.cos(-theta)
    return -theta, -(c * tx - s * ty), -(s * tx + c * ty)


def rigid_warp(img, theta, tx, ty):
    """Resample img under the model. For SYNTHESIS and diagnostics only --
    never applied to a CFA mosaic in the pipeline, for the reason in the module
    docstring."""
    h, w = img.shape
    x, y = warp_coords(h, w, theta, tx, ty)
    v, ok = bilinear(img, x, y)
    return np.where(ok, v, 0.0).astype(np.float32)


# -------------------------------------------------------------- the solver

def _huber_weights(r, k):
    a = np.abs(r)
    scale = k * (1.4826 * np.median(a) + 1e-8)
    return np.where(a <= scale, 1.0, scale / np.maximum(a, 1e-8)).astype(np.float32)


def _gauss_newton_level(ref, cmp_, theta, tx, ty, iters, huber, damp):
    h, w = ref.shape
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx0, dy0 = (xx - cx).ravel(), (yy - cy).ravel()
    gx, gy = gradients(cmp_)
    rf = ref.ravel()
    for _ in range(iters):
        s, c = np.sin(theta), np.cos(theta)
        px = c * dx0 - s * dy0 + cx + tx
        py = s * dx0 + c * dy0 + cy + ty
        v, ok = bilinear(cmp_, px, py)
        if ok.sum() < 64:
            break
        r = (v - rf)
        jx, _ = bilinear(gx, px, py)
        jy, _ = bilinear(gy, px, py)
        # dp'/dtheta
        dthx = -s * dx0 - c * dy0
        dthy = c * dx0 - s * dy0
        J = np.stack([jx * dthx + jy * dthy, jx, jy], axis=1)
        wt = _huber_weights(r[ok], huber)
        Jo, ro = J[ok], r[ok]
        Jw = Jo * wt[:, None]
        H = Jo.T @ Jw
        g = Jw.T @ ro
        H = H + damp * np.trace(H) / 3.0 * np.eye(3, dtype=H.dtype) + 1e-12 * np.eye(3)
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        theta += float(d[0])
        tx += float(d[1])
        ty += float(d[2])
        if abs(d[0]) < 1e-7 and abs(d[1]) < 1e-3 and abs(d[2]) < 1e-3:
            break
    return theta, tx, ty


def estimate_rigid(ref, cmp_, levels=5, iters=12, huber=2.0, damp=1e-3,
                   theta_sweep=None, trace=False):
    """-> (theta, tx, ty) mapping REFERENCE pixel coords into the comparison
    frame, at the resolution of the inputs.

    Coarse-to-fine: translation halves with each level, rotation does not. The
    coarsest level optionally sweeps a grid of starting angles, because the
    rotation parameter is the one with a genuine basin-of-attraction problem --
    a gradient step on theta is informative only once the frames overlap to
    within roughly a feature width at that scale.
    """
    pr = pyramid(ref, levels)
    pc = pyramid(cmp_, levels)
    theta, tx, ty = 0.0, 0.0, 0.0
    top = levels - 1
    if theta_sweep:
        best, bth = None, 0.0
        for th in theta_sweep:
            t2, x2, y2 = _gauss_newton_level(pr[top], pc[top], th, 0.0, 0.0,
                                            iters, huber, damp)
            x, y = warp_coords(*pr[top].shape, t2, x2, y2)
            v, ok = bilinear(pc[top], x.ravel(), y.ravel())
            if ok.sum() < 64:
                continue
            e = float(np.mean(np.abs(v[ok] - pr[top].ravel()[ok])))
            if best is None or e < best:
                best, bth, theta, tx, ty = e, th, t2, x2, y2
        if trace:
            print('    sweep picked theta0 %+.4f rad (err %.5f)' % (bth, best))
    for l in range(top, -1, -1):
        if l != top or not theta_sweep:
            theta, tx, ty = _gauss_newton_level(
                pr[l], pc[l], theta, tx, ty, iters, huber, damp)
        if trace:
            print('    level %d (%dx%d)  theta %+.5f rad (%+.3f deg)  t (%+.2f, %+.2f)'
                  % (l, pr[l].shape[1], pr[l].shape[0], theta,
                     np.degrees(theta), tx, ty))
        if l > 0:
            tx *= 2.0
            ty *= 2.0
    return theta, tx, ty


# ------------------------------------------------- what the seed is worth

def tile_seed(theta, tx, ty, h, w, tile, stride=None):
    """The rigid model evaluated at tile centres -> a per-tile starting
    displacement for the search that follows.

    Returns (ny, nx, 2) in (dx, dy) order: where the tile's centre in the
    reference is predicted to sit in the comparison frame, minus its reference
    position.
    """
    stride = tile if stride is None else stride
    ny = max(1, (h - tile) // stride + 1)
    nx = max(1, (w - tile) // stride + 1)
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    ty_c = (np.arange(ny) * stride + 0.5 * tile - cy).astype(np.float32)
    tx_c = (np.arange(nx) * stride + 0.5 * tile - cx).astype(np.float32)
    gy, gx = np.meshgrid(ty_c, tx_c, indexing='ij')
    s, c = np.sin(theta), np.cos(theta)
    out = np.empty((ny, nx, 2), np.float32)
    out[..., 0] = (c * gx - s * gy + tx) - gx
    out[..., 1] = (s * gx + c * gy + ty) - gy
    return out
