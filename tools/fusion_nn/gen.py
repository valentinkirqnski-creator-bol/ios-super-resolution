"""Synthetic RAW bursts from two real iPhone 15 frames, run through the app's Wronski stages.

    python gen.py --scenes DIR --out DIR --split train|val --n N --seed0 S [--workers 6]

Scenes: <name>.f32 (4032x3024 raw, linear, black-subtracted, prewhitened BGGR, the app's
loader via dng_dump) + <name>.meta. Each 2x2 Bayer quad becomes one linear RGB pixel
(R, mean of the two G, B): a 2016x1512 scene, much cleaner than a single raw sample.

One burst (8 frames, 256x256 raw on the scene grid, reference = frame 0):
  camera  : per frame similarity/affine (shift up to +-4 px, rotation, scale, shear)
  parallax: shared smooth inverse-depth map x per-frame camera translation
  objects : 0-2 soft-edged scene patches moving across the burst (occlusion, ghosts)
  exposure: per frame gain, compensated by its nominal value (+-2 % residual), sensor clip
  noise   : Poisson-Gaussian from the DNG noise profile x ISO factor 1..32, 12-bit codes
Target  : the scene in the reference geometry (objects at their frame-0 position).
Split   : per scene the right 20 % of columns is validation; a 192 px guard band keeps the
          training crops (and every pixel they sample, objects included) out of it.

Stored per burst (.npz, compressed): 12-bit codes + compensation gains, the app's tile flow,
analytic R (u8), kernel covariances (f16), target (f16), the true misalignment of the flow
the merge reads at every guide pixel (u8, 1/16 px; occlusions = 16 px), and for validation
the app's own merged image (parity).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.environ.get("WRONSKI_DUMP", os.path.join(HERE, "build", "wronski_dump.exe"))
SCENES = ("APC_1294", "APC_1186")          # burst12 frame 0, burst9 frame 0
S = 256                                     # burst size (raw px = scene px)
N = 8
MARGIN = 24                                 # scene px sampled around a crop (motion + objects)
GUARD = 192
VAL_FRAC = 0.2


def load_scene(d, name):
    m = open(os.path.join(d, name + ".meta")).read().split()
    H, W = int(m[0]), int(m[1])
    cfa = [[int(m[2]), int(m[3])], [int(m[4]), int(m[5])]]
    a, b = float(m[6]), float(m[7])
    wb = [float(x) for x in m[8:11]]
    r = np.fromfile(os.path.join(d, name + ".f32"), np.float32).reshape(H, W)
    rgb = np.zeros((3, H // 2, W // 2), np.float32)
    cnt = np.zeros(3)
    for dy in (0, 1):
        for dx in (0, 1):
            rgb[cfa[dy][dx]] += r[dy::2, dx::2]
            cnt[cfa[dy][dx]] += 1
    return dict(rgb=rgb / cnt[:, None, None].astype(np.float32), cfa=cfa, a=a, b=b, wb=wb)


def region(split, w, h):
    """x range of crop origins (incl. MARGIN on both sides) for a split; y range."""
    xv = int(w * (1 - VAL_FRAC))
    if split == "train":
        return (MARGIN, xv - GUARD - S - MARGIN), (MARGIN, h - S - MARGIN)
    return (xv + MARGIN, w - S - MARGIN), (MARGIN, h - S - MARGIN)


def smooth_field(rng, size, corr):
    g = max(2, int(np.ceil(size / corr)) + 3)
    z = rng.standard_normal((g, g)).astype(np.float32)
    zz = ndimage.zoom(z, size / (g - 3), order=3)
    o = int(round(zz.shape[0] / g * 1.5))
    f = zz[o:o + size, o:o + size]
    return f / (np.abs(f).max() + 1e-6)


def make_burst(sc, rng, split, hard=False):
    rgb = sc["rgb"]
    _, h, w = rgb.shape
    (xa, xb), (ya, yb) = region(split, w, h)
    x0 = int(rng.integers(xa, xb)) & ~1
    y0 = int(rng.integers(ya, yb)) & ~1
    k_iso = float(np.exp(rng.uniform(0, np.log(32))))
    bright = float(np.exp(rng.uniform(np.log(0.4), np.log(2.0))))
    a, b = sc["a"] * k_iso, sc["b"] * k_iso ** 2

    # scene window around the crop, prefiltered once for cubic B-spline sampling
    wx0, wy0 = x0 - MARGIN, y0 - MARGIN
    win = rgb[:, wy0:wy0 + S + 2 * MARGIN, wx0:wx0 + S + 2 * MARGIN] * bright
    coef = [ndimage.spline_filter(win[c], order=3) for c in range(3)]

    # camera motion
    # hard: more rotation / zoom / parallax and a per-frame non-rigid wobble, so the tile flow is
    # wrong by 0.2-1 px INSIDE tiles (small slips and shifted edges, as in real rotating bursts)
    strength = rng.choice([0.5, 1.0, 1.5], p=[0.3, 0.4, 0.3]) if hard else rng.choice([0.3, 1.0, 2.0], p=[0.3, 0.5, 0.2])
    rmax = 0.8 if hard else 0.5
    # hard: the slips come from rotation / parallax / wobble; the global shift stays small because
    # the app's pyramid on a 256 px crop is 4 px wide at its coarsest level (large shifts = gross
    # failures that a full 12 MP frame would not have)
    walk = np.cumsum(rng.normal(0, (0.4 if hard else 0.8) * strength, (N, 2)), 0)
    walk -= walk[0]
    tmax = 2 if hard else 4
    t = np.clip(walk + rng.uniform(-0.5, 0.5, (N, 2)), -tmax, tmax)
    t[0] = 0
    rot = np.deg2rad(np.clip(rng.normal(0, (0.25 if hard else 0.12) * strength, N).cumsum(), -rmax, rmax))
    rot[0] = 0
    scl = 1 + (np.clip(rng.normal(0, 0.002, N), -0.005, 0.005) if hard else np.clip(rng.normal(0, 0.0012, N), -0.003, 0.003))
    scl[0] = 1
    shr = np.clip(rng.normal(0, 0.0008, N), -0.002, 0.002)
    shr[0] = 0
    if rng.random() < (0.8 if hard else 0.5):   # parallax: inverse depth x camera translation
        depth = 0.5 * (smooth_field(rng, S, rng.uniform(16, 64) if hard else rng.uniform(24, 96)) + 1)
        par = (rng.uniform(0.2, 1.2) if hard else rng.uniform(0, 1.5)) * np.concatenate([[0], rng.uniform(-1, 1, N - 1)])[:, None] * \
            rng.standard_normal(2)[None] / np.sqrt(2)
    else:
        depth, par = np.zeros((S, S), np.float32), np.zeros((N, 2))

    # non-rigid wobble (hard): independent smooth sub-pixel field per frame and axis
    wob = [None] * N
    if hard and rng.random() < 0.7:
        amp = rng.uniform(0.1, 0.6)
        for f in range(1, N):
            wob[f] = [amp * smooth_field(rng, S, rng.uniform(16, 48)) for _ in range(2)]

    # moving objects: centre (crop coords) at frame 0, velocity, radii, source offset in the window
    objs = []
    for _ in range(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15] if hard else [0.4, 0.4, 0.2])):
        r = rng.uniform(8, 48, 2)
        ang = rng.uniform(0, 2 * np.pi)
        v = rng.uniform(0.3, 5) * np.array([np.cos(ang), np.sin(ang)])
        c0 = rng.uniform(0, S, 2)
        src = rng.uniform(r.max() / 2, S - r.max() / 2, 2) - c0       # texture from elsewhere in the crop
        objs.append((c0, v, r, src))

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ctr = (S - 1) / 2

    def cam(f, px, py):
        """crop px -> scene position (crop coords) seen by frame f at that pixel."""
        dx, dy = px - ctr, py - ctr
        c, s_ = np.cos(rot[f]) * scl[f], np.sin(rot[f]) * scl[f]
        sx = c * dx - s_ * dy + shr[f] * dy + ctr + t[f, 0] + depth_at(px, py) * par[f, 0]
        sy = s_ * dx + c * dy + ctr + t[f, 1] + depth_at(px, py) * par[f, 1]
        if wob[f] is not None:
            cy, cx = np.clip(py, 0, S - 1), np.clip(px, 0, S - 1)
            sx = sx + ndimage.map_coordinates(wob[f][0], [cy, cx], order=1, mode="nearest")
            sy = sy + ndimage.map_coordinates(wob[f][1], [cy, cx], order=1, mode="nearest")
        return sx, sy

    def depth_at(px, py):
        if not par.any():
            return 0.0
        return ndimage.map_coordinates(depth, [np.clip(py, 0, S - 1), np.clip(px, 0, S - 1)], order=1, mode="nearest")

    def obj_mask(f, sx, sy, o):
        c0, v, r, _ = o
        cx, cy = c0 + v * f
        d = np.sqrt(((sx - cx) / r[0]) ** 2 + ((sy - cy) / r[1]) ** 2)
        return np.clip((1 - d) * r.min() / 1.5 + 0.5, 0, 1)

    def render(f, sx, sy, chans):
        """scene value at crop coords (sx, sy) at time f, for per-pixel channels."""
        out = np.zeros(sx.shape, np.float32)
        for c in range(3):
            sel = chans == c
            if not sel.any():
                continue
            v = ndimage.map_coordinates(coef[c], [sy[sel] + MARGIN, sx[sel] + MARGIN], order=3,
                                        mode="mirror", prefilter=False)
            for o in objs:
                c0, vel, r, src = o
                m = obj_mask(f, sx[sel], sy[sel], o)
                if (m > 0).any():
                    ox = sx[sel] - (c0[0] + vel[0] * f) + c0[0] + src[0]
                    oy = sy[sel] - (c0[1] + vel[1] * f) + c0[1] + src[1]
                    vo = ndimage.map_coordinates(coef[c], [np.clip(oy + MARGIN, 0, S + 2 * MARGIN - 1),
                                                           np.clip(ox + MARGIN, 0, S + 2 * MARGIN - 1)],
                                                 order=3, mode="mirror", prefilter=False)
                    v = m * vo + (1 - m) * v
            out[sel] = v
        return out

    cfa = np.array(sc["cfa"])
    chans = cfa[(yy.astype(int) & 1), (xx.astype(int) & 1)]
    expo = np.ones(N)
    sel = rng.random(N) < 0.3
    expo[sel] = rng.uniform(0.85, 1.15, sel.sum())
    expo[0] = 1
    resid = np.clip(rng.normal(0, 0.01, N), -0.02, 0.02)
    resid[0] = 0
    codes = np.zeros((N, S, S), np.uint16)
    gains = np.zeros(N, np.float32)
    for f in range(N):
        sx, sy = cam(f, xx, yy) if f else (xx, yy)
        x = render(f, sx, sy, chans) * expo[f]
        noisy = x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(np.maximum(a * np.maximum(x, 0) + b, 0))
        codes[f] = np.round(np.clip(noisy, 0, 1) * 4095).astype(np.uint16)
        gains[f] = 1.0 / (expo[f] * (1 + resid[f]))
    tgt = np.stack([render(0, xx, yy, np.full((S, S), c)) for c in range(3)], -1)

    # true correspondence (frame position of each reference pixel) at guide-pixel centres
    gy, gx = np.mgrid[0:S // 2, 0:S // 2].astype(np.float32) * 2 + 0.5
    true = []
    for f in range(1, N):
        qx, qy = gx.copy(), gy.copy()
        for _ in range(6):                       # invert cam(f, q) = p
            sx, sy = cam(f, qx, qy)
            qx, qy = qx - (sx - gx), qy - (sy - gy)
        changed = np.zeros(gx.shape, bool)
        for o in objs:
            in0 = obj_mask(0, gx, gy, o) > 0.05
            c0, vel, r, _ = o
            # object pixels: the frame shows them displaced by vel*f in the scene
            px, py = qx.copy(), qy.copy()
            for _ in range(6):
                sx, sy = cam(f, px, py)
                px, py = px - (sx - (gx + vel[0] * f)), py - (sy - (gy + vel[1] * f))
            qx = np.where(in0, px, qx)
            qy = np.where(in0, py, qy)
            # background reference pixels the object covers in frame f: occluded
            sxf, syf = cam(f, qx, qy)
            changed |= (~in0) & (obj_mask(f, sxf, syf, o) > 0.05)
        true.append((qx - gx, qy - gy, changed))

    meta = dict(x0=x0, y0=y0, k_iso=k_iso, bright=bright, strength=float(strength), n_obj=len(objs), hard=hard,
                parallax=bool(par.any()))
    return codes, gains, tgt.astype(np.float32), (a, b), true, meta


def write_in(path, frames, sc, a, b):
    with open(path, "wb") as f:
        np.array([N, S, S, sc["cfa"][0][0], sc["cfa"][0][1], sc["cfa"][1][0], sc["cfa"][1][1]], np.int32).tofile(f)
        np.array([a, b, *sc["wb"], 1.0], np.float32).tofile(f)
        frames.astype(np.float32).tofile(f)


def read_out(path):
    buf = open(path, "rb").read()
    o = 0

    def ints(n):
        nonlocal o
        v = np.frombuffer(buf, np.int32, n, o)
        o += 4 * n
        return v

    def flts(n):
        nonlocal o
        v = np.frombuffer(buf, np.float32, n, o)
        o += 4 * n
        return v

    magic, n, tile, flags = ints(4)
    assert magic == 0x57524431, "bad dump"
    flows, Rs, covs = [], [], []
    for _ in range(1, n):
        ny, nx = ints(2)
        flows.append(flts(ny * nx * 2).reshape(ny, nx, 2))
        rh, rw, rc = ints(3)
        Rs.append(flts(rh * rw * rc).reshape(rh, rw, rc))
    for _ in range(n):
        ch, cw = ints(2)
        covs.append(flts(ch * cw * 4).reshape(ch, cw, 4)[..., [0, 1, 3]])
    hs, ws = ints(2)
    merged = flts(hs * ws * 3).reshape(hs, ws, 3)
    return dict(tile=int(tile), flags=int(flags), flows=np.stack(flows), R=np.stack(Rs), covs=np.stack(covs), merged=merged)


def ghost_map(true, flows, tile):
    """|flow the merge reads - true correspondence| at guide pixels, px; occluded = 16."""
    out = []
    G = S // 2
    for (tx, ty, changed), fl in zip(true, flows):
        ti = np.clip(np.floor((np.arange(G) * 2 + 0.5) / tile), 0, fl.shape[0] - 1).astype(int)
        tj = np.clip(np.floor((np.arange(G) * 2 + 0.5) / tile), 0, fl.shape[1] - 1).astype(int)
        v = fl[ti[:, None], tj[None, :]]
        e = np.hypot(v[..., 0] - tx, v[..., 1] - ty)
        e[changed] = 16
        out.append(np.clip(np.round(e * 16), 0, 255).astype(np.uint8))
    return np.stack(out)


_SC = {}


def work(job):
    scenes_dir, out_dir, split, seed, hard = job
    path = os.path.join(out_dir, "%s_%07d.npz" % (split, seed))
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    rng = np.random.default_rng(seed)
    name = SCENES[seed % len(SCENES)]
    if name not in _SC:
        _SC[name] = load_scene(scenes_dir, name)
    sc = _SC[name]
    codes, gains, tgt, (a, b), true, meta = make_burst(sc, rng, split, hard)
    frames = codes.astype(np.float32) / 4095 * gains[:, None, None]
    with tempfile.TemporaryDirectory(dir=out_dir) as td:
        fi, fo, fj = os.path.join(td, "in.bin"), os.path.join(td, "out.bin"), os.path.join(td, "jobs.txt")
        write_in(fi, frames, sc, a, b)
        open(fj, "w").write("%s %s\n" % (fi, fo))
        # wronski_dump very occasionally dies with a heap-corruption exit (0xC0000374) that
        # does not reproduce on the same input: retry, and check every output is finite.
        d = None
        for attempt in range(3):
            r = subprocess.run([DUMP, fj], capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(fo):
                d = read_out(fo)
                if all(np.isfinite(d[k]).all() for k in ("flows", "R", "covs", "merged")):
                    break
                d = None
            sys.stderr.write("seed %d: wronski_dump attempt %d failed (exit %d)\n" % (seed, attempt + 1, r.returncode))
        if d is None:
            sys.stderr.write("seed %d: skipped\n" % seed)
            return None, 0.0
    if d["flags"]:
        raise RuntimeError("flow mode not covered by the Python merge: flags %d" % d["flags"])
    params = np.array([a, b, d["tile"], *np.array(sc["cfa"]).ravel()], np.float64)
    z = dict(codes=codes, gain=gains, flow=d["flows"].astype(np.float32),
             R=np.round(np.clip(d["R"], 0, 1) * 255).astype(np.uint8), covs=d["covs"].astype(np.float16),
             target=tgt.astype(np.float16), ghost=ghost_map(true, d["flows"], d["tile"]), params=params,
             meta=np.array(repr(dict(meta, scene=name, seed=seed, split=split))))
    if seed < 4:
        z["merged"] = d["merged"].astype(np.float32)      # parity check
    elif split == "val":
        z["merged"] = d["merged"].astype(np.float16)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **z)
    os.replace(tmp, path)
    return path, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--hard", action="store_true", help="more rotation/zoom/parallax + non-rigid wobble")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.scenes, a.out, a.split, a.seed0 + i, a.hard) for i in range(a.n)]
    t0 = time.time()
    done = 0
    with Pool(a.workers) as p:
        for path, dt in p.imap_unordered(work, jobs, chunksize=2):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print("%d/%d  %.0fs  (last %.2fs)" % (done, len(jobs), time.time() - t0, dt), flush=True)


if __name__ == "__main__":
    main()
