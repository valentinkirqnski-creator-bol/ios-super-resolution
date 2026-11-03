"""Synthetic text scenes for the aliasing / super-resolution sets (v21).

Lines of text in several fonts, sizes and colours on paper-like or sign-like backgrounds, drawn at high
resolution so that, at the burst's texels-per-raw-pixel scale, the smallest text is a few raw pixels tall:
aliased and unreadable in any single raw frame, recoverable by merging sub-pixel-shifted frames at 2x.
Returned in the camera-native linear space the burst renderer expects (scene / white-balance gain).
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
FONTS = ["arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf", "georgia.ttf", "consola.ttf", "calibri.ttf",
         "tahoma.ttf", "trebuc.ttf", "segoeui.ttf"]
WORDS = ("the quick brown fox jumps over lazy dog super resolution burst merge align robust aliasing detail "
         "photo camera sensor pixel frame street sign exit open closed parking only stop main road north south "
         "east west hotel cafe bank pharmacy station platform gate floor level warning danger entrance "
         "monday tuesday friday sunday january october price total menu coffee tea water bread fresh daily").split()


def _phrase(rng):
    k = int(rng.integers(0, 4))
    if k == 0:    # code-like: plates, serials, phone numbers
        chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
        return " ".join("".join(rng.choice(list(chars), int(rng.integers(3, 8)))) for _ in range(int(rng.integers(2, 5))))
    ws = [str(rng.choice(WORDS)) for _ in range(int(rng.integers(3, 9)))]
    if k == 1:
        ws = [w.upper() for w in ws]
    elif k == 2:
        ws = [w.capitalize() for w in ws]
    return " ".join(ws)


def _srgb_to_linear(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def text_scene(rng, H, W, tex_per_raw, wb, cap_raw=(2.5, 7.0)):
    """rng: numpy Generator; H, W: texels; tex_per_raw: texels per raw pixel; wb: camera pre-white-balance gains;
    cap_raw: range of capital-letter heights in RAW pixels. Returns float32 [H, W, 3] camera-native linear."""
    light = rng.random() < 0.75
    bg = tuple(int(v) for v in (rng.uniform(200, 250, 3) if light else rng.uniform(10, 70, 3)))
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    y = int(rng.integers(0, 24))
    while y < H:
        cap = float(rng.uniform(*cap_raw))                       # capital height, raw px
        size = max(6, int(cap * tex_per_raw / 0.72))             # font size ~ cap height / 0.72
        font = ImageFont.truetype(os.path.join(FONT_DIR, str(rng.choice(FONTS))), size)
        line_h = int(size * rng.uniform(1.15, 1.6))
        if rng.random() < 0.2:                                   # a sign-like panel behind this line
            pc = tuple(int(v) for v in rng.uniform(0, 255, 3))
            d.rectangle([0, y, W, y + line_h], fill=pc)
            lum = 0.3 * pc[0] + 0.59 * pc[1] + 0.11 * pc[2]
            fg = (245, 245, 245) if lum < 128 else (15, 15, 15)
        elif light:
            fg = tuple(int(v) for v in (rng.uniform(0, 60, 3) if rng.random() < 0.7 else rng.uniform(0, 200, 3)))
        else:
            fg = tuple(int(v) for v in rng.uniform(190, 255, 3))
        x = -int(rng.integers(0, size * 3))
        while x < W:                                             # fill the line edge to edge
            t = _phrase(rng) + "   "
            d.text((x, y), t, fill=fg, font=font)
            x += int(d.textlength(t, font=font)) + 1
        y += line_h
    a = _srgb_to_linear(np.asarray(img, np.float32) / 255.0)
    return np.ascontiguousarray(a / np.asarray(wb, np.float32)[None, None, :], dtype=np.float32)
