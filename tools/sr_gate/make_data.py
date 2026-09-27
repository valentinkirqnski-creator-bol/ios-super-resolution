"""Build the sr_gate training and validation sets from real Bayer DNGs.

    python make_data.py --out data --train 72 --val 24

Scenes are split by SOURCE FILE, so no validation burst is synthesised from a
photo any training burst saw. The validation set is a structured grid rather
than a random sample: every regime crossed with a ladder of per-tile flow
errors, so the report can say where a mask helps and where it hurts instead of
averaging the two into one number.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build
import srburst

HR = 320            # HR crop -> 160 raw -> 320 output, 10x10 tiles at ts=16
VAL_SIGMAS = (0.12, 0.35, 0.9, 2.2, 5.0)


def scene_cache(path, cache_dir):
    """Demosaicing a 12 MP DNG is ~1.3 s, and every burst from that photo would
    pay it again, so the scene is cached as a .npy once."""
    os.makedirs(cache_dir, exist_ok=True)
    key = os.path.join(cache_dir, os.path.basename(path) + '.npy')
    if os.path.exists(key):
        return np.load(key, mmap_mode='r')
    s = srburst.load_scene(path)
    if s is None:
        return None
    np.save(key, s)
    return np.load(key, mmap_mode='r')


def crop(scene, rng, hr=HR):
    h, w = scene.shape[:2]
    # Bayer phase must be preserved: a crop offset by an odd number of RAW
    # pixels relabels the CFA. HR is 2x raw, so the offset has to be a multiple
    # of 4 in HR coordinates for the raw origin to stay on an even pixel.
    y0 = int(rng.integers(0, (h - hr) // 4)) * 4
    x0 = int(rng.integers(0, (w - hr) // 4)) * 4
    c = np.ascontiguousarray(scene[y0:y0 + hr, x0:x0 + hr])
    return c


def interesting(c):
    """Skip crops that are flat or blown out -- nothing to learn about merging
    detail where there is none, and they would dominate the average."""
    g = c.mean(axis=2)
    if g.mean() < 0.02 or g.mean() > 0.9:
        return False
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    return (gx + gy) > 0.004


def save(d, path):
    np.savez_compressed(
        path,
        feat=d['feat'].astype(np.float16),
        Rw=d['Rw'].astype(np.float16),
        A=d['A'].astype(np.float16),
        B=d['B'].astype(np.float16),
        A_ref=d['A_ref'].astype(np.float16),
        B_ref=d['B_ref'].astype(np.float16),
        gt=d['gt'].astype(np.float32),
        edge=d['edge'].astype(np.float16),
        meta=np.array([d['regime'], d['sigma_flow'], d['noise_gain'],
                       d['tile_size']], np.float32),
    )


def one(args):
    kind, idx, dng, seed, out, cache, regime, sigma = args
    rng = np.random.default_rng(seed)
    scene = scene_cache(dng, cache)
    if scene is None:
        return None
    for _ in range(24):
        c = crop(scene, rng)
        if interesting(c):
            break
    else:
        return None
    spec = srburst.BurstSpec(rng, regime=regime)
    if sigma is not None:
        spec.sigma_flow = float(sigma)
    d = build.build_burst(c, spec, rng)
    path = os.path.join(out, kind, '%s_%04d.npz' % (kind, idx))
    save(d, path)
    return (kind, idx, d['regime'], d['sigma_flow'], d['noise_gain'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data')
    ap.add_argument('--train', type=int, default=72)
    ap.add_argument('--val', type=int, default=24)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--test', type=int, default=0,
                    help='also build a TEST grid, from the val scenes but with '
                         'different crops, seeds and regime draws. The '
                         'checkpoint is selected on val, so the headline number '
                         'has to come from something val selection never saw.')
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, a.out)
    cache = os.path.join(here, '.scene_cache')
    for k in ('train', 'val', 'test'):
        os.makedirs(os.path.join(out, k), exist_ok=True)

    dngs, n_in_tree = srburst.find_dngs(build.ROOT)
    # Split by source photo. The bursts/ captures and the hdrplus payload are
    # different scenes AND different sensors, so put some of each on both sides
    # rather than validating on a sensor never trained on.
    #
    # The holdout is drawn from the IN-TREE pool only, and the permutation is
    # over exactly that many files, so adding a source does NOT reshuffle it: the
    # six val/test scenes stay the six they have always been, and a number
    # measured before a source was added stays comparable with one measured
    # after. Out-of-tree sources are train-only, which does mean nothing here
    # measures generalisation TO them -- say so rather than implying otherwise.
    rng = np.random.default_rng(12345)
    order = list(rng.permutation(n_in_tree))
    val_files = [dngs[i] for i in order[:6]]
    train_files = [dngs[i] for i in order[6:]] + dngs[n_in_tree:]
    print('train scenes %d (%d in tree + %d train-only), val scenes %d'
          % (len(train_files), n_in_tree - 6, len(dngs) - n_in_tree,
             len(val_files)))
    for q in dngs[n_in_tree:]:
        print('   train-only source:', os.path.basename(q))

    jobs = []
    # training: regimes sampled with the moving-object case over-weighted,
    # because that is the one where a mask has to be SELECTIVE rather than
    # uniformly permissive or uniformly strict.
    weights = np.array([0.32, 0.20, 0.32, 0.16])
    for i in range(a.train):
        regime = int(np.searchsorted(np.cumsum(weights), rng.random()))
        jobs.append(('train', i, train_files[i % len(train_files)],
                     1000 + i, out, cache, regime, None))
    # validation: the full grid, so the report is a sweep and not an average
    k = 0
    for regime in (0, 1, 2, 3):
        for s in VAL_SIGMAS:
            for rep in range(max(1, a.val // 20)):
                jobs.append(('val', k, val_files[k % len(val_files)],
                             500000 + k, out, cache, regime, s))
                k += 1
    n_test = 0
    if a.test:
        for regime in (0, 1, 2, 3):
            for sv in VAL_SIGMAS:
                for rep in range(max(1, a.test // 20)):
                    jobs.append(('test', n_test,
                                 val_files[(n_test + 3) % len(val_files)],
                                 900000 + n_test * 7, out, cache, regime, sv))
                    n_test += 1
    print('bursts to build: %d train + %d val + %d test' % (a.train, k, n_test))

    # Warm the scene cache serially: rawpy is not fork-safe here and the
    # workers would otherwise each demosaic the same photo.
    t0 = time.time()
    for p in dngs:
        if scene_cache(p, cache) is None:
            print('  skipped (not Bayer):', os.path.basename(p))
    print('scene cache warm in %.1fs' % (time.time() - t0))

    t0 = time.time()
    done = 0
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs) as pool:
            for r in pool.imap_unordered(one, jobs):
                done += 1
                if r and done % 10 == 0:
                    print('  %d/%d  %.0fs' % (done, len(jobs), time.time() - t0),
                          flush=True)
    else:
        for j in jobs:
            one(j)
            done += 1
    print('built %d bursts in %.0fs' % (done, time.time() - t0))
    for k in ('train', 'val', 'test'):
        d = os.path.join(out, k)
        if not os.path.isdir(d):
            continue
        n = len(os.listdir(d))
        mb = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6
        print('%s: %d files, %.0f MB' % (k, n, mb))


if __name__ == '__main__':
    main()
