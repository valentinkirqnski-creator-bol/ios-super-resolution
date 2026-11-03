"""Full-raw-resolution ground truth (hr_target.py) for every sample of a cache: rebuilds each source burst once from
its seed and renders the exact noise-free content the merge would pull for each cached variant's flow.
Writes data/<cache>_hr/<name>.npz with geo_hr [K,Hg,Wg] (float16).

    python precompute_hr.py data/c15_train data/c17_train data/c15_val data/c17_val --workers 4
"""
import argparse
import glob
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402


def _job(job):
    src, items = job
    torch.set_num_threads(1)
    from hr_target import hr_target, burst_of
    todo = [(c, o) for c, o in items if not os.path.exists(o)]
    if not todo:
        return 0
    b = burst_of(src)
    gt = torch.from_numpy(np.load(src)["gt_flow"][1:])
    for c, o in todo:
        z = np.load(c)
        t = hr_target(b, torch.from_numpy(z["flow"]), gt, 4 * float(z["a"]), 4 * float(z["b"]))
        tmp = o[:-4] + ".tmp.npz"
        np.savez(tmp, geo_hr=t.numpy().astype(np.float16))
        os.replace(tmp, o)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import train_pix as tp
    groups = defaultdict(list)
    for cdir in a.caches:
        out = os.path.join(HERE, cdir.rstrip("/\\") + "_hr")
        os.makedirs(out, exist_ok=True)
        for c in sorted(f for f in glob.glob(os.path.join(HERE, cdir, "*.npz")) if ".tmp." not in f):
            groups[tp._source(c)].append((c, os.path.join(out, os.path.basename(c))))
    jobs = sorted(groups.items())
    print(f"{sum(len(v) for v in groups.values())} samples from {len(jobs)} source bursts", flush=True)
    t0, done = time.time(), 0
    with Pool(a.workers) as p:
        for k, n in enumerate(p.imap_unordered(_job, jobs)):
            done += n
            if k % 25 == 0:
                print(f"{k}/{len(jobs)} sources, {done} samples, {time.time() - t0:.0f}s", flush=True)
    print(f"done {done} samples in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
