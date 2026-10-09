"""Pack the per-burst .npz shards into memory-mapped .npy stacks.

    python pack.py --data data

The dataset is ~1.4 GB in fp16 and this box has 13.8 GB of RAM with rather less
free, so the trainer must not hold it as fp32 tensors (that would be 4 GB and
would page). Memory mapping one array per field lets the OS page cache hold what
it can and keeps the trainer's own footprint to a batch.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

FIELDS = {
    'feat': np.float16,
    'A': np.float16,
    'B': np.float16,
    'A_ref': np.float16,
    'B_ref': np.float16,
    'gt': np.float32,
    'edge': np.float16,
    'Rw': np.float16,
    'img': np.float16,
    'ferr': np.float16,
    'art': np.float16,
    'meta': np.float32,
}


def pack(split_dir, out_prefix):
    if not os.path.isdir(split_dir):
        return 0
    files = sorted(glob.glob(os.path.join(split_dir, '*.npz')))
    if not files:
        return 0
    z0 = np.load(files[0])
    shapes = {k: z0[k].shape for k in FIELDS}
    mm = {}
    for k, dt in FIELDS.items():
        path = out_prefix + '_' + k + '.npy'
        mm[k] = np.lib.format.open_memmap(
            path, mode='w+', dtype=dt, shape=(len(files),) + shapes[k])
    for i, f in enumerate(files):
        z = np.load(f)
        for k, dt in FIELDS.items():
            mm[k][i] = z[k].astype(dt)
    for k in mm:
        mm[k].flush()
    return len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    for split in ('train', 'val', 'test'):
        n = pack(os.path.join(root, split), os.path.join(root, split))
        print('%s: packed %d bursts' % (split, n))
    tot = sum(os.path.getsize(p) for p in glob.glob(os.path.join(root, '*.npy')))
    print('packed size %.2f GB' % (tot / 1e9))


if __name__ == '__main__':
    main()
