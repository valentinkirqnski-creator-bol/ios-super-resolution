"""Long-edge scenes for the misalignment-artifact NN (core/mann.h): the aperture-problem set for EDGES.

A camera-scale crop of a real DNG scene upsampled to `tex` texels per raw pixel, with large high-contrast shapes drawn
on top whose edges are long, straight or gently curved, at near-horizontal / near-vertical / any angle: car bodies
(rounded panels, dark windows with thin bright frames, dark trims), railings / fences (2-6 parallel bars with sparse
posts), thin line pairs (spoilers, wires, roof racks), building / roof edges. Along such an edge a tile's vector is
only constrained across it, so the aligner slides along it, and a slightly rotated or zoomed frame aligned with one
vector per tile breaks the edge into a stair of offset tiles or doubles it. Returned in the camera-native linear space
the burst renderer expects (as pattern_scene).
"""
import numpy as np
from scipy.ndimage import zoom

from rnn.pattern_scene import _colour_pair
from rnn.text_scene import _srgb_to_linear


def _cov(d, soft):
    """Coverage in [0, 1] of a signed distance (texels, negative inside) with a soft edge."""
    return np.clip(0.5 - d / max(soft, 1e-3), 0.0, 1.0)


def _rot(rng, near_axis=0.7):
    if rng.random() < near_axis:
        return float(rng.choice([0.0, np.pi / 2]) + rng.normal(0, 0.12))
    return float(rng.uniform(0, np.pi))


def _round_rect_d(u, v, hw, hh, r):
    """Signed distance to a rounded rectangle centred at 0 (half sizes hw, hh, corner radius r)."""
    qx, qy = np.abs(u) - (hw - r), np.abs(v) - (hh - r)
    out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return out + np.minimum(np.maximum(qx, qy), 0) - r


def _paint(img, a, col):
    img *= (1 - a[..., None])
    img += a[..., None] * col[None, None, :]


def _lin(rgb):
    return _srgb_to_linear(np.clip(np.asarray(rgb, np.float32), 0, 1))


def _shape(rng, h, w, tex):
    """One h x w texel patch: (rgb, alpha)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = _rot(rng)
    c, s = np.cos(th), np.sin(th)
    cy, cx = h / 2.0, w / 2.0
    u = (xx - cx) * c + (yy - cy) * s
    v = -(xx - cx) * s + (yy - cy) * c
    # gentle curvature (car roofs, bonnets): bend v by a parabola in u
    if rng.random() < 0.5:
        v = v + rng.uniform(-1, 1) * (u / max(w, h)) ** 2 * rng.uniform(0.02, 0.15) * max(w, h)
    soft = 0.7 * tex                                   # ~0.35 raw px edge
    rgb = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    kind = int(rng.integers(0, 4))
    if kind == 0:                                      # car body: panel + windows with frames + trims
        body = _lin(rng.uniform(0.55, 0.95, 3) if rng.random() < 0.6 else rng.uniform(0.05, 0.4, 3))
        hw, hh = 0.46 * w, 0.42 * h
        d = _round_rect_d(u, v, hw, hh, rng.uniform(0.05, 0.3) * min(hw, hh))
        a = _cov(d, soft)
        _paint(rgb, np.ones_like(a), body); alpha = a
        for _ in range(int(rng.integers(1, 4))):       # windows: dark glass, thin bright / dark frame
            ww, wh = rng.uniform(0.12, 0.4) * w, rng.uniform(0.08, 0.25) * h
            ox, oy = rng.uniform(-0.25, 0.25) * w, rng.uniform(-0.25, 0.2) * h
            dw = _round_rect_d(u - ox, v - oy, ww / 2, wh / 2, rng.uniform(0.1, 0.45) * min(ww, wh) / 2)
            fr = rng.uniform(1.0, 4.0) * tex
            frame = _lin(rng.uniform(0.0, 0.25, 3) * np.array([1.0, 0.7, 1.2]) if rng.random() < 0.5 else rng.uniform(0.6, 1.0, 3))
            _paint(rgb, _cov(dw - fr, soft) * a, frame)
            _paint(rgb, _cov(dw, soft) * a, _lin(rng.uniform(0.02, 0.15, 3)))
        for _ in range(int(rng.integers(0, 4))):       # trims / panel gaps / spoiler lines
            t0 = rng.uniform(-0.4, 0.4) * h; tw = rng.uniform(0.6, 3.5) * tex
            _paint(rgb, _cov(np.abs(v - t0) - tw / 2, soft) * a, _lin(rng.uniform(0.0, 0.3, 3)))
    elif kind == 1:                                    # railing / fence: parallel bars + sparse posts
        lo, hi = _colour_pair(rng)
        _paint(rgb, np.ones((h, w), np.float32), lo)
        nb = int(rng.integers(2, 7)); gap = rng.uniform(8, 40) * tex; bw = rng.uniform(2, 12) * tex
        m = np.zeros((h, w), np.float32)
        for i in range(nb):
            m = np.maximum(m, _cov(np.abs(v - (i - nb / 2) * gap) - bw / 2, soft))
        if rng.random() < 0.7:
            pg = rng.uniform(30, 120) * tex; pw = rng.uniform(2, 10) * tex
            ph = (u + rng.uniform(0, pg)) % pg
            m = np.maximum(m, _cov(np.abs(ph - pg / 2) - pw / 2, soft) * _cov(np.abs(v) - nb * gap / 2, soft))
        _paint(rgb, m, hi)
        alpha = np.ones((h, w), np.float32)
    elif kind == 2:                                    # thin line pairs (spoiler, roof rack, wires)
        lo, hi = _colour_pair(rng)
        _paint(rgb, np.ones((h, w), np.float32), lo)
        n = int(rng.integers(1, 4)); sp = rng.uniform(3, 14) * tex
        base = rng.uniform(-0.2, 0.2) * h
        for i in range(n):
            lw = rng.uniform(1.0, 4.0) * tex
            _paint(rgb, _cov(np.abs(v - base - i * sp) - lw / 2, soft), hi)
        alpha = np.ones((h, w), np.float32)
    else:                                              # building / roof edge: a big half-plane step
        lo, hi = _colour_pair(rng)
        a = _cov(v - rng.uniform(-0.2, 0.2) * h, soft)
        _paint(rgb, np.ones((h, w), np.float32), lo)
        _paint(rgb, a, hi)
        alpha = np.ones((h, w), np.float32)
    return rgb, alpha


def edge_scene(rng, sc_rgb, n_tex, tex, wb, cover=(0.35, 0.65)):
    """n_tex x n_tex texels at `tex` texels per raw px (see pattern_scene for the background)."""
    n_src = int(np.ceil(n_tex / (2.0 * tex))) + 2
    bh, bw = sc_rgb.shape[:2]
    y0, x0 = int(rng.integers(0, max(1, bh - n_src))), int(rng.integers(0, max(1, bw - n_src)))
    crop = sc_rgb[y0:y0 + n_src, x0:x0 + n_src]
    if crop.shape[0] < n_src or crop.shape[1] < n_src:
        crop = np.pad(crop, ((0, n_src - crop.shape[0]), (0, n_src - crop.shape[1]), (0, 0)), mode="reflect")
    bg = zoom(crop, (2.0 * tex, 2.0 * tex, 1), order=1)[:n_tex, :n_tex].astype(np.float32)
    inv_wb = 1.0 / np.asarray(wb, np.float32)
    target = rng.uniform(*cover) * n_tex * n_tex
    area = 0.0
    while area < target:
        ph = int(rng.uniform(80, 520) * tex); pw = int(rng.uniform(120, 700) * tex)
        ph, pw = min(ph, n_tex), min(pw, n_tex)
        py, px = int(rng.integers(0, n_tex - ph + 1)), int(rng.integers(0, n_tex - pw + 1))
        rgb, alpha = _shape(rng, ph, pw, tex)
        rgb = rgb * inv_wb[None, None, :]
        gain = float(np.clip(bg[py:py + ph, px:px + pw].mean() / max(rgb.mean(), 1e-4), 0.3, 3.0)) \
            if rng.random() < 0.5 else 1.0
        a = alpha[..., None]
        bg[py:py + ph, px:px + pw] = bg[py:py + ph, px:px + pw] * (1 - a) + rgb * gain * a
        area += ph * pw
    return np.ascontiguousarray(bg, dtype=np.float32)
