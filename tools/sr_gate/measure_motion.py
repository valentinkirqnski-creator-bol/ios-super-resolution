"""What motion is actually in the real bursts?

The synthetic regimes were guessed: the moving-object case used 1.5-10 raw px of
object velocity, and ours2 turned out to contain roughly 500 px of non-rigid
subject motion. Guessing again would be the same mistake, so this measures each
burst and reports the numbers the regimes should be built from.

Method: per-tile phase correlation on an 8x decimated grey, against frame 0.
The GLOBAL peak is the camera's background motion; the spread of
(local - global) is independent motion. Phase correlation rather than block
matching because the subject displacement is hundreds of raw pixels and a
search-window matcher would need a huge window to find it -- and, more to the
point, the real block matcher CANNOT find it either, which is why those tiles
lock onto the background in the first place.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_real_guide import load_raw

DEC = 8          # decimation of the raw for the correlation
WIN = 64         # correlation window, decimated px
STEP = 32        # tile stride, decimated px


def decimate(raw, f):
    h, w = raw.shape
    g = 0.25 * (raw[0::2, 0::2] + raw[0::2, 1::2] +
                raw[1::2, 0::2] + raw[1::2, 1::2])
    k = f // 2
    h2, w2 = (g.shape[0] // k) * k, (g.shape[1] // k) * k
    return g[:h2, :w2].reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3))


def pc_peak(a, b):
    """Phase-correlation displacement of b relative to a, sub-pixel free."""
    wnd = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    A = np.fft.rfft2((a - a.mean()) * wnd)
    B = np.fft.rfft2((b - b.mean()) * wnd)
    X = A * np.conj(B)
    X /= np.maximum(np.abs(X), 1e-9)
    c = np.fft.irfft2(X, s=a.shape)
    k = np.unravel_index(np.argmax(c), c.shape)
    dy = k[0] if k[0] <= a.shape[0] // 2 else k[0] - a.shape[0]
    dx = k[1] if k[1] <= a.shape[1] // 2 else k[1] - a.shape[1]
    return dy, dx, float(c.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', default='ours,ours2,ours3')
    a = ap.parse_args()
    root = 'C:/Users/valen/Downloads'

    for d in a.dirs.split(','):
        files = sorted(glob.glob(os.path.join(root, d, '*.dng')))
        if not files:
            print('%s: no dng' % d)
            continue
        ref = decimate(load_raw(os.path.join(root, d, os.path.basename(files[0])),
                                )[0], DEC)
        print()
        print('=== %s (%d frames, decimated %dx -> %s) ==='
              % (d, len(files), DEC, ref.shape))
        print('%-10s %-16s %-26s %-10s' %
              ('frame', 'global (raw px)', 'independent |d| raw px', 'moving'))
        all_ind = []
        for f in files[1:]:
            mov = decimate(load_raw(os.path.join(root, d, os.path.basename(f)))[0],
                           DEC)
            gy, gx, _ = pc_peak(ref, mov)
            loc = []
            for y in range(0, ref.shape[0] - WIN + 1, STEP):
                for x in range(0, ref.shape[1] - WIN + 1, STEP):
                    ty, tx, pk = pc_peak(ref[y:y + WIN, x:x + WIN],
                                         mov[y:y + WIN, x:x + WIN])
                    loc.append((ty - gy, tx - gx, pk))
            loc = np.array(loc)
            ind = np.hypot(loc[:, 0], loc[:, 1]) * DEC      # raw px
            all_ind.append(ind)
            frac = float((ind > 8).mean())
            print('%-10s (%+5.0f,%+5.0f)   p50 %5.0f  p90 %5.0f  max %5.0f   %5.1f%%'
                  % (os.path.basename(f)[4:8], gy * DEC, gx * DEC,
                     np.percentile(ind, 50), np.percentile(ind, 90), ind.max(),
                     100 * frac))
        ind = np.concatenate(all_ind)
        print('  ALL FRAMES  independent motion: p50 %.0f  p75 %.0f  p90 %.0f  '
              'p99 %.0f  max %.0f raw px'
              % tuple(np.percentile(ind, [50, 75, 90, 99, 100])))
        print('  tiles with independent motion > 8 raw px : %.1f%%'
              % (100 * float((ind > 8).mean())))
        print('  tiles with independent motion > 64 raw px: %.1f%%'
              % (100 * float((ind > 64).mean())))


if __name__ == '__main__':
    main()
