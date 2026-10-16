"""Generate synthetic training bursts.

    python gen_dataset.py --out data/train --n 1200 --seed0 0
    python gen_dataset.py --out data/val   --n 150  --seed0 1000000
    # optional: real photos as scene sources, or the app's own aligner
    python gen_dataset.py ... --scene_dir path/to/photos --align_cmd "path/to/aligner_cli"
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn.pipeline import make_example, to_record          # noqa: E402
from rnn.align import ExternalAligner                      # noqa: E402

ARGS = None


def _init(args):
    global ARGS
    ARGS = args
    torch.set_num_threads(1)


def _one(seed):
    path = os.path.join(ARGS.out, f"b{seed:08d}.npz")
    if os.path.exists(path):
        return seed, 0.0
    t = time.time()
    al = ExternalAligner(ARGS.align_cmd) if ARGS.align_cmd else None
    ex = make_example(seed, size=ARGS.size, aligner=al, scene_dir=ARGS.scene_dir)
    np.savez(path, **to_record(ex, ARGS.full))
    return seed, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--size", type=int, default=256, help="raw crop size (grey = size/2)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--scene_dir", default=None)
    ap.add_argument("--align_cmd", default=None)
    ap.add_argument("--full", action="store_true", help="also store RAW frames + GT (needed for --w_merge)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.n))
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        for i, (s, dt) in enumerate(pool.imap_unordered(_one, seeds)):
            if i % 50 == 0:
                print(f"{i}/{len(seeds)} seed={s} {dt:.2f}s/burst elapsed={time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
