"""Generate a dataset: synthetic bursts -> the app's stages (two alignment variants) -> labels -> data/<name>/<split>/.

    python gen_data.py configs/data_d1.yaml [--splits train val test] [--limit N] [--workers 2]

Resumable (existing samples are skipped). Each sample is reproducible from (config seed, split, index).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mfsr import config as C           # noqa: E402


def _job(args):
    split, idx, cfg, out = args
    import torch
    torch.set_num_threads(int(os.environ.get("MFSR_THREADS", "2")))
    from mfsr import sample
    if os.path.exists(out):
        return None
    try:
        meta, tm = sample.generate(split, idx, cfg, out, timing=True)
        return dict(split=split, idx=idx, category=meta["category"], N=meta["N"], tile=meta["tile"],
                    t=round(tm["total"], 1))
    except Exception as e:                                   # a failed sample is logged and skipped, never fatal
        return dict(split=split, idx=idx, error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    cfg = C.load_data(a.config)
    root = os.path.join(HERE, "data", cfg["name"])
    os.makedirs(root, exist_ok=True)
    C.save(cfg, os.path.join(root, "data_config.yaml"))
    jobs = []
    for split in a.splits:
        d = os.path.join(root, split)
        os.makedirs(d, exist_ok=True)
        n = cfg["counts"][split] if not a.limit else min(a.limit, cfg["counts"][split])
        jobs += [(split, i, cfg, os.path.join(d, f"{i:05d}.npz")) for i in range(n)]
    todo = [j for j in jobs if not os.path.exists(j[3])]
    print(f"{len(jobs)} samples, {len(todo)} to generate", flush=True)
    t0 = time.time()
    log = open(os.path.join(root, "gen_log.jsonl"), "a")
    w = a.workers or cfg["workers"]
    with Pool(w) as p:
        for k, r in enumerate(p.imap_unordered(_job, todo)):
            if r is None:
                continue
            log.write(json.dumps(r) + "\n")
            log.flush()
            if k % 10 == 0 or "error" in r:
                el = time.time() - t0
                print(f"{k + 1}/{len(todo)} {el:.0f}s ({el / (k + 1):.1f}s/sample) last={r}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
