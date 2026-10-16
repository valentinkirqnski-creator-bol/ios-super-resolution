"""Procedural clean high-resolution scene textures (linear RGB, float32).

Scenes are deliberately rich in the content that breaks alignment / robustness:
near-Nyquist stripes and zone plates (aliasing => correct-but-different samples),
bricks and text (repetitive => plausible wrong matches), hard dead-leaves edges
(tiny flow errors => visible ghosts), smooth gradients and flat dark regions
(noise must not read as motion). Optionally real photos can be mixed in.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def srgb_to_lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def rand_color(rng, lo=0.0, hi=1.0):
    return srgb_to_lin(rng.uniform(lo, hi, 3))


def _grid(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return yy, xx


def _mix(c1, c2, m):
    return (c1[None, None, :] + (c2 - c1)[None, None, :] * m[..., None]).astype(np.float32)


def pat_flat(rng, h, w):
    return np.broadcast_to(rand_color(rng), (h, w, 3)).astype(np.float32).copy()


def pat_stripes(rng, h, w):
    yy, xx = _grid(h, w)
    th = rng.uniform(0, np.pi)
    # periods down to 2.2 texels == 1.1 raw px: above Nyquist -> strong aliasing
    period = float(np.exp(rng.uniform(np.log(2.2), np.log(60.0))))
    ph = (xx * np.cos(th) + yy * np.sin(th)) / period * 2 * np.pi + rng.uniform(0, 2 * np.pi)
    s = np.sin(ph)
    m = (s > 0).astype(np.float32) if rng.random() < 0.6 else 0.5 + 0.5 * s
    return _mix(rand_color(rng), rand_color(rng), m)


def pat_checker(rng, h, w):
    yy, xx = _grid(h, w)
    th = rng.uniform(0, np.pi) if rng.random() < 0.5 else 0.0
    p = float(np.exp(rng.uniform(np.log(3.0), np.log(50.0))))
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    m = ((np.floor(u / p) + np.floor(v / p)) % 2).astype(np.float32)
    return _mix(rand_color(rng), rand_color(rng), m)


def pat_noise(rng, h, w):
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    base = rng.integers(2, 8)
    for o in range(int(rng.integers(3, 7))):
        n = int(base * 2 ** o)
        g = rng.random((n + 1, n + 1)).astype(np.float32)
        im = Image.fromarray(g).resize((w, h), Image.BICUBIC)
        acc += amp * np.asarray(im, np.float32)
        tot += amp
        amp *= rng.uniform(0.4, 0.75)
    acc = acc / tot
    acc = (acc - acc.min()) / max(1e-6, acc.max() - acc.min())
    return _mix(rand_color(rng), rand_color(rng), acc)


def pat_gradient(rng, h, w):
    yy, xx = _grid(h, w)
    th = rng.uniform(0, 2 * np.pi)
    u = xx * np.cos(th) + yy * np.sin(th)
    u = (u - u.min()) / max(1e-6, u.max() - u.min())
    return _mix(rand_color(rng), rand_color(rng), u.astype(np.float32))


def pat_bricks(rng, h, w):
    bw = float(rng.uniform(10, 60))
    bh = bw / rng.uniform(2.0, 3.2)
    mort = float(rng.uniform(1.0, 3.0))
    yy, xx = _grid(h, w)
    row = np.floor(yy / bh)
    xo = xx + (row % 2) * bw * 0.5
    col = np.floor(xo / bw)
    fy = yy - row * bh
    fx = xo - col * bw
    mortar = (fy < mort) | (fx < mort)
    base = rand_color(rng, 0.2, 0.9)
    jit = rng.uniform(0.75, 1.25, (int(h / bh) + 3, int((w + bw) / bw) + 3)).astype(np.float32)
    ji = jit[row.astype(int) % jit.shape[0], col.astype(int) % jit.shape[1]]
    img = base[None, None, :] * ji[..., None]
    img[mortar] = rand_color(rng, 0.0, 1.0)
    return img.astype(np.float32)


_FONT_CACHE = {}


def _font(size):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.load_default(size=size)
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def pat_text(rng, h, w):
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    size = int(rng.integers(8, 40))
    f = _font(size)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,-"
    y = int(rng.integers(-size, size))
    while y < h:
        s = "".join(rng.choice(list(alphabet), int(rng.integers(10, 80))))
        d.text((int(rng.integers(-20, 5)), y), s, fill=255, font=f)
        y += int(size * rng.uniform(1.0, 1.6))
    m = np.asarray(im, np.float32) / 255.0
    return _mix(rand_color(rng), rand_color(rng), m)


def pat_zoneplate(rng, h, w):
    yy, xx = _grid(h, w)
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    k = rng.uniform(0.002, 0.02)
    m = 0.5 + 0.5 * np.cos(k * ((yy - cy) ** 2 + (xx - cx) ** 2))
    return _mix(rand_color(rng), rand_color(rng), m.astype(np.float32))


PATTERNS = {
    "flat": (pat_flat, 0.33), "stripes": (pat_stripes, 0.12), "checker": (pat_checker, 0.05),
    "noise": (pat_noise, 0.15), "gradient": (pat_gradient, 0.10), "bricks": (pat_bricks, 0.08),
    "text": (pat_text, 0.11), "zone": (pat_zoneplate, 0.06),
}
REPETITIVE = ["bricks", "text", "stripes", "checker"]


def random_pattern(rng, h, w, only=None):
    names = only or list(PATTERNS)
    p = np.array([PATTERNS[n][1] for n in names], np.float64)
    n = names[rng.choice(len(names), p=p / p.sum())]
    return PATTERNS[n][0](rng, h, w)


def _shape_mask(rng, h, w):
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    kind = rng.integers(0, 3)
    if kind == 0:
        d.ellipse([0, 0, w - 1, h - 1], fill=255)
    elif kind == 1:
        d.rectangle([0, 0, w - 1, h - 1], fill=255)
    else:
        n = int(rng.integers(3, 8))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(0.5, 1.0, n)
        pts = [(w / 2 + 0.5 * w * r * np.cos(a), h / 2 + 0.5 * h * r * np.sin(a)) for a, r in zip(ang, rad)]
        d.polygon(pts, fill=255)
    return np.asarray(im, np.float32) / 255.0


def dead_leaves(rng, h, w, n_shapes=None, rmin=4, rmax=None, only=None):
    rmax = rmax or max(h, w) / 3
    img = random_pattern(rng, h, w, only)
    n_shapes = n_shapes or int(rng.integers(250, 700))
    for _ in range(n_shapes):
        r = rmin * (rmax / rmin) ** (rng.random() ** 1.3)   # many small, few big
        bh = max(3, int(r * rng.uniform(0.5, 2.0)))
        bw = max(3, int(r * rng.uniform(0.5, 2.0)))
        y0 = int(rng.integers(-bh // 2, h - bh // 2))
        x0 = int(rng.integers(-bw // 2, w - bw // 2))
        m = _shape_mask(rng, bh, bw)
        fill = random_pattern(rng, bh, bw, only)
        ys0, xs0 = max(0, y0), max(0, x0)
        ys1, xs1 = min(h, y0 + bh), min(w, x0 + bw)
        if ys1 <= ys0 or xs1 <= xs0:
            continue
        mm = m[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0, None]
        ff = fill[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        img[ys0:ys1, xs0:xs1] = img[ys0:ys1, xs0:xs1] * (1 - mm) + ff * mm
    return img


def add_highlights(rng, img, n=None):
    h, w, _ = img.shape
    n = n or int(rng.integers(2, 8))
    yy, xx = _grid(h, w)
    for _ in range(n):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(6, max(h, w) / 6)
        m = np.clip((r - np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)) / 2.0, 0, 1)
        gain = rng.uniform(2.0, 8.0)
        img = img * (1 + (gain - 1) * m[..., None])
    return img


def photo_scene(rng, h, w, scene_dir):
    files = [f for f in os.listdir(scene_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    im = Image.open(os.path.join(scene_dir, files[rng.integers(len(files))])).convert("RGB")
    s = max(h / im.height, w / im.width)
    if s > 1:
        im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.BICUBIC)
    y0 = int(rng.integers(0, im.height - h + 1))
    x0 = int(rng.integers(0, im.width - w + 1))
    a = np.asarray(im.crop((x0, y0, x0 + w, y0 + h)), np.float32) / 255.0
    return srgb_to_lin(a)


def background(rng, h, w, mode, scene_dir=None):
    if scene_dir and rng.random() < 0.5:
        return photo_scene(rng, h, w, scene_dir)
    if mode == "repetitive":
        if rng.random() < 0.4:
            return random_pattern(rng, h, w, REPETITIVE)
        # a repetitive field with a few big repetitive-filled leaves on top
        return dead_leaves(rng, h, w, n_shapes=int(rng.integers(3, 15)), rmin=h / 8, only=REPETITIVE)
    return dead_leaves(rng, h, w)


def object_rgba(rng, h, w, mode):
    """Foreground object: textured fill + soft-edged silhouette (RGBA)."""
    only = REPETITIVE if mode == "repetitive" and rng.random() < 0.5 else None
    fill = dead_leaves(rng, h, w, n_shapes=int(rng.integers(5, 60)), only=only) if rng.random() < 0.6 \
        else random_pattern(rng, h, w, only)
    a = _shape_mask(rng, h, w)
    return np.concatenate([fill, a[..., None]], axis=2).astype(np.float32)
