"""Synthetic edge test chart: bars, squares and turned squares of ONE size range (raw px) in dark / bright colours on
a mid-grey background, so a ghosted-edge test can be read per edge size. Same output space as text_scene."""
import numpy as np
from PIL import Image, ImageDraw

from rnn.text_scene import _srgb_to_linear


def chart_scene(rng, H, W, tex_per_raw, wb, size_raw=(4.0, 8.0)):
    bg = tuple(int(v) for v in rng.uniform(110, 160, 3))
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    area = 0.0
    while area < 0.22 * H * W:
        s = float(rng.uniform(*size_raw)) * tex_per_raw                        # thickness / side in texels
        fg = tuple(int(v) for v in (rng.uniform(10, 60, 3) if rng.random() < 0.5 else rng.uniform(200, 250, 3)))
        cx, cy, th = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0, np.pi)
        kind = rng.integers(0, 3)
        if kind == 0:                                                          # bar: width s, long
            ln = max(6 * s, 48 * tex_per_raw) * rng.uniform(1, 3)
            u, v = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
            pts = [tuple(np.array([cx, cy]) + a * u * ln / 2 + b * v * s / 2) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            d.polygon(pts, fill=fg); area += ln * s
        else:                                                                  # square, axis-aligned or turned
            t = 0.0 if kind == 1 else th
            u, v = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
            pts = [tuple(np.array([cx, cy]) + (a * u + b * v) * s / 2) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            d.polygon(pts, fill=fg); area += s * s
    a = _srgb_to_linear(np.asarray(img, np.float32) / 255.0)
    return np.ascontiguousarray(a / np.asarray(wb, np.float32)[None, None, :], dtype=np.float32)
