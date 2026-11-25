"""Training data for the learned SRSR robustness mask.

One sample = one synthetic burst rendered from a real scene with exact
ground truth (tools/robust_nn/rnn/burst.py: layered world, exact per-frame
homographies, occlusion, moving objects), in the iPhone's raw format (CFA,
pre-white-balanced, the DNG noise profile scaled over ISO). The burst is run
through the REAL SRSR pipeline (srsr_dump: the app's SRSRKernels.metal on the
CPU), so the flow, the covariances and the analytic mask are exactly what the
phone would compute. Stored per sample (.npz):

  quads    [N, qH, qW, 4] u16   RGBA16Uint quad codes (slot 0 = reference)
  covs     [N, qH, qW, 4] f16   merge kernel covariances (half, as on the GPU)
  flows    [N-1, fh, fw, 4] f32 final flow textures (packed, merge reads .xy)
  masks    [N-1, qH, qW] u8     the analytic dilated keep mask (x255)
  merged   [oH, oW, 3] f16      the SRSR merge with the analytic mask (parity + baseline)
  target   [oH, oW, 3] f32      the noise-free reference view at the output grid, merge domain
  gt_flow  [N, 2, Hg, Wg] f16   exact correspondence (grey px of the uncropped raw grid)
  vis      [N, Hg, Wg] u8       visibility of the reference point in frame k
  uniforms / norms / geometry   the merge's constant buffers (raw bytes)
  meta     json: scenario, noise, wb, exposure, crop phase, scale

    python gen.py --out D:/srsr_nn_data/train --n 2000 --seed0 0
    python gen.py --out D:/srsr_nn_data/val   --n 100  --seed0 9000000
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "robust_nn"))
from rnn import scenarios                     # noqa: E402
from rnn.burst import Burst                   # noqa: E402

DUMP = os.environ.get("SRSR_DUMP_EXE", os.path.join(HERE, "build", "srsr_dump.exe"))
SCENES = os.environ.get("SRSR_SCENES", os.path.join(HERE, "data", "scenes"))


def load_scene(path_noext):
    """dng_dump output (the app loader: black-subtracted, white 1, pre-white-balanced)
    -> camera-native RGB, one texel per CFA quad, plus the capture's CFA / WB / noise."""
    h, w, c00, c01, c10, c11, a, b, w0, w1, w2, _pw = open(path_noext + ".meta").read().split()[:12]
    cfa = ((int(c00), int(c01)), (int(c10), int(c11)))
    wb = np.array([float(w0), float(w1), float(w2)], np.float32)
    r = np.fromfile(path_noext + ".f32", np.float32).reshape(int(h), int(w))
    r = r[: r.shape[0] // 2 * 2, : r.shape[1] // 2 * 2]
    q = {}
    for i in range(2):
        for j in range(2):
            q.setdefault(cfa[i][j], []).append(r[i::2, j::2])
    rgb = np.stack([np.mean(q[c], 0) / wb[c] for c in range(3)], -1)
    return np.clip(rgb, 0, None).astype(np.float32), cfa, wb, float(a), float(b)


_SCENE_CACHE = {}


def scene(name):
    if name not in _SCENE_CACHE:
        _SCENE_CACHE[name] = load_scene(os.path.join(SCENES, name))
    return _SCENE_CACHE[name]


def scene_names():
    return sorted(f[:-5] for f in os.listdir(SCENES) if f.endswith(".meta"))


def build(seed, size, scenes_list, val):
    """The burst of a seed: scene, scenario, noise, exposure all drawn from the seed."""
    rng = np.random.default_rng(seed)
    name = scenes_list[seed % len(scenes_list)]
    sc_rgb, cfa, wb, a0, b0 = scene(name)
    sc = scenarios.sample(rng)
    sc.update(n_frames=int(rng.integers(4, 9)))
    if rng.random() < 0.35:                                   # big hand shake
        sc.update(tremor=float(rng.uniform(3.0, 8.0)))
    k = float(np.exp(rng.uniform(0, np.log(30))))             # ISO multiplier on the DNG's profile
    if sc["name"] == "low_light":
        k = float(np.exp(rng.uniform(np.log(10), np.log(60))))
    sc.update(noise_a=a0 * k, noise_b=b0 * k * k)
    # Camera scale (0.5 texels / raw px = one DNG quad per synthetic quad) most of the time,
    # sometimes 2x shrunk (sharper, busier content per pixel).
    tex = 0.5 if rng.random() < 0.75 else 1.0
    b = Burst(sc, rng, Hr=size, Wr=size, bg_rgb=sc_rgb, cfa=cfa, prewhite=wb, tex_scale=tex)
    return b, sc, name, cfa, wb, a0 * k, b0 * k * k


def crop_phase(cfa):
    for i in range(2):
        for j in range(2):
            if cfa[i][j] == 0:
                return j, i          # (cropX, cropY): R at the origin
    raise ValueError("no red site")


def read_dump(d):
    idx = {}
    for ln in open(os.path.join(d, "index.txt")):
        name, w, h, nc, dt = ln.split()
        idx[name] = (int(w), int(h), int(nc), dt)

    def arr(name):
        w, h, nc, dt = idx[name]
        a = np.fromfile(os.path.join(d, name), np.uint16 if dt == "u16" else np.float32)
        return a.reshape(h, w, nc) if nc > 1 else a.reshape(h, w)

    n = sum(1 for k in idx if k.startswith("quad_"))
    out = dict(
        quads=np.stack([arr("quad_%02d.u16" % f) for f in range(n)]),
        covs=np.stack([arr("cov_%02d.f32" % f) for f in range(n)]).astype(np.float16),
        flows=np.stack([arr("flow_%02d.f32" % f) for f in range(1, n)]).astype(np.float32),
        masks=np.stack([np.rint(arr("mask_%02d.f32" % f) * 255) for f in range(1, n)]).astype(np.uint8),
        accw=np.rint(arr("accw.f32") * 255).astype(np.uint8),
        merged=arr("merged.f32").astype(np.float16),
        uniforms=np.fromfile(os.path.join(d, "uniforms.bin"), np.uint8),
        norms=np.fromfile(os.path.join(d, "norms.bin"), np.uint8),
        geometry=np.fromfile(os.path.join(d, "geometry.bin"), np.uint8),
    )
    return out


def gen_one(args):
    seed, out_dir, size, scale, scenes_list, val = args
    path = os.path.join(out_dir, "s%08d.npz" % seed)
    if os.path.exists(path):
        return path, 0.0, "exists"
    t0 = time.time()
    torch.set_num_threads(1)
    b, sc, scene_name, cfa, wb, na, nb = build(seed, size, scenes_list, val)
    r = b.render(out_scale=scale, seed=seed)
    tmp = tempfile.mkdtemp(prefix="srsr_nn_")
    try:
        bd, dd = os.path.join(tmp, "burst"), os.path.join(tmp, "dump")
        os.makedirs(bd)
        raws = r["raw"].numpy().astype(np.float32)
        N = raws.shape[0]
        for k in range(N):
            raws[k].tofile(os.path.join(bd, "frame_%02d.f32" % k))
        with open(os.path.join(bd, "meta.txt"), "w") as f:
            f.write("%d %d %d %d %d %d %d %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g 1\n" % (
                size, size, N, cfa[0][0], cfa[0][1], cfa[1][0], cfa[1][1],
                na, na, na, nb, nb, nb, wb[0], wb[1], wb[2]))
            f.write(" ".join("%.9g" % e for e in r["E"]) + "\n")
        p = subprocess.run([DUMP, bd, dd, str(scale)], capture_output=True, text=True)
        if p.returncode != 0:
            return path, time.time() - t0, "srsr_dump failed: " + (p.stderr or p.stdout)[-300:]
        dump = read_dump(dd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Target: the noise-free reference view at the output grid, in the merge's domain
    # (pre-white-balanced), shifted by the crop phase that puts R at the origin.
    cx, cy = crop_phase(cfa)
    oH, oW = dump["merged"].shape[:2]
    gt = r["gt"].numpy() * wb[:, None, None]                 # [3, size*scale, size*scale]
    gt = gt[:, cy * scale: cy * scale + oH, cx * scale: cx * scale + oW]
    meta = dict(seed=int(seed), scenario=sc["name"], scene=scene_name, size=size, scale=scale,
                crop=[cx, cy], cfa=[list(c) for c in cfa], wb=wb.tolist(), noise_a=na, noise_b=nb,
                exposure=[float(e) for e in r["E"]], n_frames=int(N))
    np.savez_compressed(path, **dump, target=np.ascontiguousarray(gt.transpose(1, 2, 0)).astype(np.float32),
                        gt_flow=r["gt_flow"].numpy().astype(np.float16), vis=(r["vis"].numpy() * 255).astype(np.uint8),
                        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    return path, time.time() - t0, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--size", type=int, default=768, help="raw crop (>= 512: the SRSR pyramid needs 8 coarsest px)")
    ap.add_argument("--scale", type=int, default=1, choices=[1, 2])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--val", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = scene_names()
    if not names:
        sys.exit("no scenes in %s (dng_dump <dng> <name>.f32 > <name>.meta)" % SCENES)
    jobs = [(a.seed0 + i, a.out, a.size, a.scale, names, a.val) for i in range(a.n)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (path, dt, msg) in enumerate(pool.imap_unordered(gen_one, jobs)):
            print("[%d/%d] %s %.1fs %s" % (i + 1, a.n, os.path.basename(path), dt, msg), flush=True)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
