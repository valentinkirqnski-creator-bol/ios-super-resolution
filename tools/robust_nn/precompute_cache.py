"""Run the app's stages (aligner + analytic mask, seeded deliberate corruption) ONCE per training variant and store
everything mask training needs, so a pixel-level net can take many passes over the data.

    python precompute_cache.py --src data/rp13_train,data/rp13_train_hdrp --out data/c15_train --variants 4
    python precompute_cache.py --src data/rp13_val --out data/c15_val --val     # same 60 samples/seeds as RP(val)
Stored per sample: ref grey, comparison greys, flow (guide px), analytic r0, corrected ground-truth mask
(occluded points judged by visible difference), hard-case maps (edge, rot, ferr), noise a/b, tile.
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("RNN_TP_SERVER", "1")
sys.path.insert(0, HERE)
import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402
import real_protocol as rp                                                # noqa: E402

OUT = None


EDGE_STEP = False
HOMOG = False
ORACLE = False
SMALL = False


def _init(out, edge_step=False, homog=False, oracle=False, small=False):
    global OUT, EDGE_STEP, HOMOG, ORACLE, SMALL
    OUT, EDGE_STEP, HOMOG, ORACLE, SMALL = out, edge_step, homog, oracle, small
    torch.set_num_threads(1)


def _job(job):
    f, name, corrupt, seed = job
    path = os.path.join(OUT, name + ".npz")
    if os.path.exists(path):
        return 0.0
    t0 = time.time()
    z = np.load(f)
    raw, gt = z["raw"], z["gt_flow"]
    al, be = float(z["alpha"]), float(z["beta"])
    for attempt in range(3):
        try:
            d = rp.run_tp_real(raw, gt, al, be, seed + attempt, corrupt, z["cfa"], z["wb"], nomerge=True,
                               edge_step=EDGE_STEP, homog=HOMOG, oracle=ORACLE,
                               small_only=SMALL)
            break
        except Exception:   # noqa: BLE001 - transient harness failure: retry with the next seed
            time.sleep(1.0)
    else:
        return -1.0
    clean, gtf = torch.from_numpy(z["clean_grey"]), torch.from_numpy(gt[1:])
    vis = torch.from_numpy(z["vis"][1:].astype(np.float32))
    geo = rp.geo_target(clean, d["flow"], gtf, vis, al, be, occ_look=True)
    hm = rp.hard_maps(clean, gtf, d["flow"], al, be)
    tmp = path[:-4] + ".tmp.npz"
    extra = dict(rflow=d["rflow"].numpy().astype(np.float32)) if "rflow" in d else {}
    np.savez(tmp, **extra, ref=d["ref_grey"].numpy().astype(np.float32), grey=d["grey"].numpy().astype(np.float32),
             flow=d["flow"].numpy().astype(np.float32), r0=d["r0"].numpy().astype(np.float16),
             geo=geo.numpy().astype(np.float16), edge=hm["edge"].numpy().astype(np.float16),
             rot=hm["rot"].numpy(), ferr=hm["ferr"].numpy().astype(np.float16),
             a=np.float32(al / 4.0), b=np.float32(be / 4.0), tile=np.int32(d["tile"]), corrupt=np.bool_(corrupt),
             scen=str(z["scenario"]))
    os.replace(tmp, path)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--corrupt_p", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--edge_step", action="store_true", help="v19: straight-edge step corruption (tp_real flag 8)")
    ap.add_argument("--homog", action="store_true", help="v20: the app's global homography path (tp_real flag 16)")
    ap.add_argument("--oracle", action="store_true", help="v21: perfect alignment (tp_real flag 32)")
    ap.add_argument("--limit", type=int, default=0, help="v24: use only the first N bursts of each source dir")
    ap.add_argument("--small", action="store_true", help="v22: only tiny misalignments when corrupting (flag 64)")
    ap.add_argument("--val_corrupt_only", action="store_true", help="v21: --val keeps only the corrupted half")
    ap.add_argument("--no_corrupt", action="store_true",
                    help="v20: never corrupt (perfect-alignment sets); --val then keeps only the aligner-only half")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for r in a.src.split(",")
                   for f in sorted(glob.glob(os.path.join(HERE, r, "*.npz")))[:a.limit or None])
    jobs = []
    if a.val:   # identical to RP(val)[idx]: idx < n aligner only, idx >= n corrupted; seed from default_rng(99 + idx)
        n = len(files)
        for idx in (range(n, 2 * n) if a.val_corrupt_only else range(n if a.no_corrupt else 2 * n)):
            seed = int(np.random.default_rng(99 + idx).integers(1, 1 << 30))
            jobs.append((files[idx % n], f"val{idx:03d}", idx >= n, seed))
    else:
        rng = np.random.default_rng(1234)
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            for v in range(a.variants):
                cor = bool(rng.random() < a.corrupt_p)
                jobs.append((f, f"{stem}_v{v}", cor and not a.no_corrupt, int(rng.integers(1, 1 << 30))))
    t0, bad = time.time(), 0
    with Pool(a.workers, initializer=_init, initargs=(a.out, a.edge_step, a.homog, a.oracle, a.small)) as p:
        for k, dt in enumerate(p.imap_unordered(_job, jobs, chunksize=2)):
            bad += dt < 0
            if k % 50 == 0:
                print(f"{k}/{len(jobs)}  {time.time() - t0:.0f}s  failed {bad}", flush=True)
    print(f"done {len(jobs)} in {time.time() - t0:.0f}s, failed {bad}", flush=True)


if __name__ == "__main__":
    main()
