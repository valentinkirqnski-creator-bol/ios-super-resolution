"""Repeated-pattern scenes for the consensus refiner (cref v2): the aperture-problem set.

A camera-scale crop of a real DNG scene, upsampled to `tex` texels per raw pixel, with patches of repeating
structure drawn on top: line gratings at any angle with periods near the sensor's limit (2.2-8 raw px), fences /
railings / grids of bars (4-24 px), brick (courses 4-14 px), checkerboard / weave (2.5-10 px) and fine text. Block
matching and LK lock onto the wrong period or slide along the lines on exactly these (paper S.3, Fig. 27), so the
app's own aligner produces the real failures; the dataset tool adds period-sized tile slips on top.
Returned in the camera-native linear space the burst renderer expects (same as text_scene).
"""
import numpy as np
from scipy.ndimage import zoom

from rnn.text_scene import _srgb_to_linear, text_scene


def _soft_square(phase, duty, edge):
    """Square wave in [0, 1] of a phase in periods, with soft (texel-scale) edges."""
    f = phase - np.floor(phase)                      # [0, 1): the "on" segment is [0, duty)
    d = np.abs(f - duty / 2.0)
    d = np.minimum(d, 1.0 - d)                       # wrapped distance to the segment centre (periods)
    return np.clip((duty / 2.0 - d) / max(edge, 1e-3) + 0.5, 0.0, 1.0)


def _colour_pair(rng):
    lo = rng.uniform(0.02, 0.35, 3)
    hi = rng.uniform(0.45, 0.95, 3)
    if rng.random() < 0.3:                   # low contrast (fabric, mesh in shade)
        m = rng.uniform(0.15, 0.6)
        lo, hi = np.full(3, m) * rng.uniform(0.8, 1.0, 3), np.full(3, m) * rng.uniform(1.15, 1.6, 3)
    if rng.random() < 0.5:
        lo, hi = hi, lo
    return _srgb_to_linear(np.clip(lo, 0, 1)), _srgb_to_linear(np.clip(hi, 0, 1))


def _pattern(rng, h, w, tex):
    """One h x w texel patch: (rgb [h,w,3] linear sRGB-ish, alpha [h,w])."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    kind = int(rng.integers(0, 5))
    th = rng.uniform(0, np.pi) if rng.random() < 0.5 else rng.choice([0.0, np.pi / 2]) + rng.normal(0, 0.03)
    c, s = np.cos(th), np.sin(th)
    u, v = (xx * c + yy * s) / tex, (-xx * s + yy * c) / tex           # raw-px coordinates (edges ~0.35 px soft)
    if kind == 0:                                                      # line grating near the sensor limit
        p = float(np.exp(rng.uniform(np.log(2.2), np.log(8.0))))
        m = _soft_square(u / p, rng.uniform(0.3, 0.6), 0.35 / p)
    elif kind == 1:                                                    # fence / railing / grid of bars
        p1 = float(rng.uniform(4.0, 24.0))
        m = _soft_square(v / p1, rng.uniform(0.15, 0.4), 0.35 / p1)
        if rng.random() < 0.6:
            p2 = float(rng.uniform(4.0, 40.0))
            m = np.maximum(m, _soft_square(u / p2, rng.uniform(0.1, 0.35), 0.35 / p2))
    elif kind == 2:                                                    # brick: courses + staggered joints
        ph = float(rng.uniform(4.0, 14.0)); pw = ph * rng.uniform(2.0, 3.2)
        mort = rng.uniform(0.12, 0.25)
        row = np.floor(v / ph)
        m = np.maximum(_soft_square(v / ph, mort, 0.35 / ph),
                       _soft_square((u + 0.5 * pw * (row % 2)) / pw, mort * ph / pw, 0.35 / pw))
    elif kind == 3:                                                    # checkerboard / weave
        p = float(np.exp(rng.uniform(np.log(2.5), np.log(10.0))))
        a = _soft_square(u / p, 0.5, 0.35 / p); b = _soft_square(v / p, 0.5, 0.35 / p)
        m = a * (1 - b) + b * (1 - a)
    else:                                                              # fine text block
        t = text_scene(rng, h, w, tex, np.ones(3, np.float32), cap_raw=(2.2, 6.0))
        alpha = np.ones((h, w), np.float32)
        return t, alpha
    lo, hi = _colour_pair(rng)
    rgb = lo[None, None, :] * (1 - m[..., None]) + hi[None, None, :] * m[..., None]
    alpha = np.ones((h, w), np.float32)
    return rgb.astype(np.float32), alpha


def pattern_scene(rng, sc_rgb, n_tex, tex, wb, cover=(0.25, 0.55)):
    """n_tex x n_tex texels at `tex` texels per raw px: a real-scene crop (sc_rgb: one texel per RGGB quad, i.e.
    0.5 texel per raw px) upsampled by 2*tex, with pattern patches covering `cover` of the area."""
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
        ph = int(rng.uniform(60, 420) * tex); pw = int(rng.uniform(60, 420) * tex)
        ph, pw = min(ph, n_tex), min(pw, n_tex)
        py, px = int(rng.integers(0, n_tex - ph + 1)), int(rng.integers(0, n_tex - pw + 1))
        rgb, alpha = _pattern(rng, ph, pw, tex)
        rgb = rgb * inv_wb[None, None, :]
        # match the patch's brightness to the scene it sits on (keeps exposure realistic)
        gain = float(np.clip(bg[py:py + ph, px:px + pw].mean() / max(rgb.mean(), 1e-4), 0.3, 3.0)) \
            if rng.random() < 0.6 else 1.0
        a = alpha[..., None]
        bg[py:py + ph, px:px + pw] = bg[py:py + ph, px:px + pw] * (1 - a) + rgb * gain * a
        area += ph * pw
    return np.ascontiguousarray(bg, dtype=np.float32)
