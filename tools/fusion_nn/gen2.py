"""2x super-resolution training bursts: 3 MP synthetic RAW bursts from a 12 MP HR target.

    python gen2.py --scenes DIR --out DIR --split train|val --n N --seed0 S [--workers 6]

Scenes: <name>_hr.npy (hr_scene.py: the 12 MP DNG demosaiced, in the app's raw units) and
<name>.meta (CFA, noise profile, WB). HR target grid = 4032x3024 RGB; burst raw grid = half
(2016x1512, 3 MP). Raw pixel j sits on HR pixel 2j and integrates the HR image with a
[1,2,1]/4 tent per axis (the 2x2 HR area of a 3 MP pixel). The app merges the burst at
scale 2 (output pixel h <-> raw position h/2 <-> HR pixel h), so the target is the sharp HR
image itself, exactly aligned.

Per burst: 8 frames of 256x256 raw (target 512x512 RGB), reference = frame 0. Motion,
objects, exposure and noise as gen.py; every other burst pair uses the hard motion (more
rotation / zoom / parallax + non-rigid wobble: sub-pixel slips inside the app's tiles).
Then the app's Wronski stages (wronski_dump, scale 2). Stored: 12-bit codes + gains, tile
flow, kernels (f16), HR target (f16), true flow error at guide pixels (u8, 1/16 px), and for
validation the analytic R (u8) and the app's merged image (f16).
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

from gen import N, S, MARGIN, DUMP, SCENES, region, smooth_field, write_in, read_out, ghost_map

SCALE = 2


def load_scene(d, name):
    m = open(os.path.join(d, name + ".meta")).read().split()
    cfa = [[int(m[2]), int(m[3])], [int(m[4]), int(m[5])]]
    hr = np.load(os.path.join(d, name + "_hr.npy")).astype(np.float32)        # [3, H, W]
    return dict(hr=hr, cfa=cfa, a=float(m[6]), b=float(m[7]), wb=[float(x) for x in m[8:11]])


def make_burst(sc, rng, split, hard):
    hr = sc["hr"]
    _, H, W = hr.shape
    (xa, xb), (ya, yb) = region(split, W // 2, H // 2)          # raw-grid (3 MP) coordinates
    x0 = int(rng.integers(xa, xb)) & ~1
    y0 = int(rng.integers(ya, yb)) & ~1
    k_iso = float(np.exp(rng.uniform(0, np.log(32))))
    bright = float(np.exp(rng.uniform(np.log(0.4), np.log(2.0))))
    a, b = sc["a"] * k_iso, sc["b"] * k_iso ** 2

    # HR window around the crop (HR = 2 x raw coordinates); sharp (target) and tent (sensor)
    M2 = 2 * MARGIN
    win = hr[:, 2 * y0 - M2:2 * (y0 + S) + M2, 2 * x0 - M2:2 * (x0 + S) + M2] * bright
    tent = np.array([0.25, 0.5, 0.25], np.float32)
    blur = ndimage.convolve1d(ndimage.convolve1d(win, tent, axis=1, mode="mirror"), tent, axis=2, mode="mirror")
    coef_s = [ndimage.spline_filter(win[c], order=3) for c in range(3)]
    coef_b = [ndimage.spline_filter(blur[c], order=3) for c in range(3)]
    WS = win.shape[1]

    strength = rng.choice([0.5, 1.0, 1.5], p=[0.3, 0.4, 0.3]) if hard else rng.choice([0.3, 1.0, 2.0], p=[0.3, 0.5, 0.2])
    rmax = 0.8 if hard else 0.5
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
    if rng.random() < (0.8 if hard else 0.5):
        depth = 0.5 * (smooth_field(rng, S, rng.uniform(16, 64) if hard else rng.uniform(24, 96)) + 1)
        par = (rng.uniform(0.2, 1.2) if hard else rng.uniform(0, 1.5)) * np.concatenate(
            [[0], rng.uniform(-1, 1, N - 1)])[:, None] * rng.standard_normal(2)[None] / np.sqrt(2)
    else:
        depth, par = np.zeros((S, S), np.float32), np.zeros((N, 2))
    wob = [None] * N
    if hard and rng.random() < 0.7:
        amp = rng.uniform(0.1, 0.6)
        for f in range(1, N):
            wob[f] = [amp * smooth_field(rng, S, rng.uniform(16, 48)) for _ in range(2)]
    objs = []
    for _ in range(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15] if hard else [0.4, 0.4, 0.2])):
        r = rng.uniform(8, 48, 2)
        ang = rng.uniform(0, 2 * np.pi)
        v = rng.uniform(0.3, 5) * np.array([np.cos(ang), np.sin(ang)])
        c0 = rng.uniform(0, S, 2)
        src = rng.uniform(r.max() / 2, S - r.max() / 2, 2) - c0
        objs.append((c0, v, r, src))

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ctr = (S - 1) / 2

    def field_at(fld, px, py):
        return ndimage.map_coordinates(fld, [np.clip(py, 0, S - 1), np.clip(px, 0, S - 1)], order=1, mode="nearest")

    def cam(f, px, py):
        """raw crop px -> scene position (raw crop coords) seen by frame f at that pixel."""
        dx, dy = px - ctr, py - ctr
        c, s_ = np.cos(rot[f]) * scl[f], np.sin(rot[f]) * scl[f]
        dpt = field_at(depth, px, py) if par.any() else 0.0
        sx = c * dx - s_ * dy + shr[f] * dy + ctr + t[f, 0] + dpt * par[f, 0]
        sy = s_ * dx + c * dy + ctr + t[f, 1] + dpt * par[f, 1]
        if wob[f] is not None:
            sx = sx + field_at(wob[f][0], px, py)
            sy = sy + field_at(wob[f][1], px, py)
        return sx, sy

    def obj_mask(f, sx, sy, o):
        c0, v, r, _ = o
        cx, cy = c0 + v * f
        d = np.sqrt(((sx - cx) / r[0]) ** 2 + ((sy - cy) / r[1]) ** 2)
        return np.clip((1 - d) * r.min() / 1.5 + 0.5, 0, 1)

    def sample(coef, c, sx, sy):
        """scene value at raw crop coords (sx, sy): HR window pixel 2 * (s + MARGIN)."""
        return ndimage.map_coordinates(coef[c], [np.clip(2 * (sy + MARGIN), 0, WS - 1), np.clip(2 * (sx + MARGIN), 0, WS - 1)],
                                       order=3, mode="mirror", prefilter=False)

    def render(f, sx, sy, chans, coef):
        out = np.zeros(sx.shape, np.float32)
        for c in range(3):
            sel = chans == c
            if not sel.any():
                continue
            v = sample(coef, c, sx[sel], sy[sel])
            for o in objs:
                c0, vel, r, src = o
                m = obj_mask(f, sx[sel], sy[sel], o)
                if (m > 0).any():
                    ox = sx[sel] - (c0[0] + vel[0] * f) + c0[0] + src[0]
                    oy = sy[sel] - (c0[1] + vel[1] * f) + c0[1] + src[1]
                    v = m * sample(coef, c, ox, oy) + (1 - m) * v
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
        x = render(f, sx, sy, chans, coef_b) * expo[f]
        noisy = x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(np.maximum(a * np.maximum(x, 0) + b, 0))
        codes[f] = np.round(np.clip(noisy, 0, 1) * 4095).astype(np.uint16)
        gains[f] = 1.0 / (expo[f] * (1 + resid[f]))
    # target: output pixel h (HR grid) <-> raw position h / 2, sharp HR
    hy, hx = np.mgrid[0:2 * S, 0:2 * S].astype(np.float32) / 2
    tgt = np.stack([render(0, hx, hy, np.full(hx.shape, c), coef_s) for c in range(3)], -1)

    gy, gx = np.mgrid[0:S // 2, 0:S // 2].astype(np.float32) * 2 + 0.5
    true = []
    for f in range(1, N):
        qx, qy = gx.copy(), gy.copy()
        for _ in range(6):
            sx, sy = cam(f, qx, qy)
            qx, qy = qx - (sx - gx), qy - (sy - gy)
        changed = np.zeros(gx.shape, bool)
        for o in objs:
            in0 = obj_mask(0, gx, gy, o) > 0.05
            c0, vel, r, _ = o
            px, py = qx.copy(), qy.copy()
            for _ in range(6):
                sx, sy = cam(f, px, py)
                px, py = px - (sx - (gx + vel[0] * f)), py - (sy - (gy + vel[1] * f))
            qx = np.where(in0, px, qx)
            qy = np.where(in0, py, qy)
            sxf, syf = cam(f, qx, qy)
            changed |= (~in0) & (obj_mask(f, sxf, syf, o) > 0.05)
        true.append((qx - gx, qy - gy, changed))
    meta = dict(x0=x0, y0=y0, k_iso=k_iso, bright=bright, strength=float(strength), n_obj=len(objs), hard=hard,
                parallax=bool(par.any()), scale=SCALE)
    return codes, gains, tgt.astype(np.float32), (a, b), true, meta


_SC = {}


def work(job):
    scenes_dir, out_dir, split, seed = job
    path = os.path.join(out_dir, "%s_%07d.npz" % (split, seed))
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    rng = np.random.default_rng(seed)
    name = SCENES[seed % len(SCENES)]
    hard = (seed // 2) % 2 == 1
    if name not in _SC:
        _SC[name] = load_scene(scenes_dir, name)
    sc = _SC[name]
    codes, gains, tgt, (a, b), true, meta = make_burst(sc, rng, split, hard)
    frames = codes.astype(np.float32) / 4095 * gains[:, None, None]
    with tempfile.TemporaryDirectory(dir=out_dir) as td:
        fi, fo, fj = os.path.join(td, "in.bin"), os.path.join(td, "out.bin"), os.path.join(td, "jobs.txt")
        write_in(fi, frames, sc, a, b)
        with open(fi, "r+b") as f:                     # header: scale is the 6th float after 7 ints
            f.seek(7 * 4 + 5 * 4)
            np.array([SCALE], np.float32).tofile(f)
        open(fj, "w").write("%s %s\n" % (fi, fo))
        d = None
        for attempt in range(3):                      # wronski_dump's rare non-reproducible crash
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
    assert d["merged"].shape[:2] == (2 * S, 2 * S), d["merged"].shape
    assert d["R"].shape[1:3] == (S, S), "training files assume a raw-res R (merge tile lookup)"
    params = np.array([a, b, d["tile"], *np.array(sc["cfa"]).ravel(), SCALE], np.float64)
    z = dict(codes=codes, gain=gains, flow=d["flows"].astype(np.float32), covs=d["covs"].astype(np.float16),
             target=tgt.astype(np.float16), ghost=ghost_map(true, d["flows"], d["tile"]), params=params,
             meta=np.array(repr(dict(meta, scene=name, seed=seed, split=split))))
    if split == "val" or seed % 1000 < 2:
        z["R"] = np.round(np.clip(d["R"], 0, 1) * 255).astype(np.uint8)
        z["merged"] = d["merged"].astype(np.float32 if seed % 1000 < 2 else np.float16)
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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.scenes, a.out, a.split, a.seed0 + i) for i in range(a.n)]
    t0 = time.time()
    done = 0
    with Pool(a.workers) as p:
        for path, dt in p.imap_unordered(work, jobs, chunksize=2):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print("%d/%d  %.0fs  (last %.2fs)" % (done, len(jobs), time.time() - t0, dt), flush=True)


if __name__ == "__main__":
    main()
