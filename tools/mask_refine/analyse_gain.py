"""What kind of burst carries the refinement headroom?

The smooth oracle's gain varies ~20x across windows (+0.10 to +2.18 dB). An
average over that is close to meaningless: if nearly all of it sits in one kind
of misalignment, the useful question is which, and whether anything observable
at inference time identifies it.

`gen_bursts.cpp` prints its regime choice but does not store it in the dump, so
the labels for the existing 6.5 GB are gone. They are not needed. The true flow
error `ferr` IS stored, per mask pixel, and the regimes differ in its spatial
structure rather than only its size:

  clean subpixel   small ferr, spatially flat
  rotation         ferr rising linearly across the window (it grows with
                   radius from the frame centre, and a crop sees a gradient)
  object/parallax  ferr concentrated in a compact, contiguous region with a
                   hard boundary -- high neighbour agreement in the thresholded
                   set, unlike scattered estimation noise

So each burst is described by statistics of ferr and of the burst's own noise
gain and brightness, and the oracle gain is correlated against them. Two
distinct things come out:

  * which descriptor predicts headroom -- i.e. where a refinement is worth
    having at all; and
  * whether that descriptor has an observable counterpart. ferr itself needs
    the true flow and is unavailable at inference, but a descriptor that is
    structural (a compact blob, a linear ramp) can often be recovered from the
    estimated flow alone, whereas one that is purely a magnitude cannot.

    python analyse_gain.py --data data --n 24
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from probe_distill import gate_gain, smooth_oracle


DESCRIPTORS = ('ferr_mean', 'ferr_p95', 'frac_gt1', 'ferr_ramp', 'ferr_blob',
               'flow_mag', 'flow_span', 'gain', 'bright', 'maskw')


def describe(d, w):
    """Burst-level descriptors on the window the oracle was fitted to."""
    ferr = w['ferr'][0].numpy()              # (N, h, wd)
    n, h, wd = ferr.shape
    yy, xx = np.mgrid[0:h, 0:wd]
    yy = (yy - yy.mean()) / max(h, 1)
    xx = (xx - xx.mean()) / max(wd, 1)

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        s = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / s) if s > 1e-12 else 0.0

    # Linear ramp across the window, the strongest axis, averaged over frames.
    ramp = np.mean([max(abs(corr(ferr[i], xx)), abs(corr(ferr[i], yy)))
                    for i in range(n)])

    # Contiguity of the high-error set: of the pixels above 1 px, what fraction
    # of their 4-neighbours are also above it. A compact blob scores high; the
    # same number of scattered pixels scores near the set's own density.
    hot = ferr > 1.0
    blob = 0.0
    if hot.any():
        agree = np.zeros_like(ferr)
        agree[:, 1:, :] += hot[:, :-1, :]
        agree[:, :-1, :] += hot[:, 1:, :]
        agree[:, :, 1:] += hot[:, :, :-1]
        agree[:, :, :-1] += hot[:, :, 1:]
        blob = float(agree[hot].mean() / 4.0 - hot.mean())

    fm = w['feat'][0, :, D.FEATURES.index('flow_mag')].numpy()
    fs = w['feat'][0, :, D.FEATURES.index('flow_span')].numpy()
    return dict(ferr_mean=float(ferr.mean()),
                ferr_p95=float(np.percentile(ferr, 95)),
                frac_gt1=float(hot.mean()),
                ferr_ramp=float(ramp),
                ferr_blob=blob,
                flow_mag=float(fm.mean()),
                flow_span=float(fs.mean()),
                gain=float(d['gain']),
                bright=float(w['gt'].mean()),
                maskw=float(w['Rw'].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--window', type=int, default=256)
    ap.add_argument('--down', type=int, default=4)
    ap.add_argument('--oracle-steps', type=int, default=250)
    ap.add_argument('--oracle-lr', type=float, default=0.3)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    bs = D.Bursts(a.data, cache=1, window=a.window, seed=11)
    n = min(a.n, len(bs))
    keys = ('feat', 'Rw', 'A_ref', 'B_ref', 'A', 'B', 'gt', 'ferr')
    print('%d bursts, %d px windows, oracle lattice %dx coarser'
          % (n, a.window, a.down), flush=True)
    print('%-5s %8s %7s' % ('burst', 'gain dB', 'kept')
          + ''.join('%10s' % k for k in DESCRIPTORS), flush=True)

    rows, descs = [], []
    t0 = time.time()
    for i in range(n):
        d = bs._get(i)
        w = dict(zip(keys, bs.batch(i)))
        g = smooth_oracle(w, a.down, a.oracle_steps, a.oracle_lr)
        gn, kept = gate_gain(w, g)
        de = describe(d, w)
        rows.append((gn, kept))
        descs.append(de)
        print('%-5d %+8.3f %7.3f' % (i, gn, kept)
              + ''.join('%10.4f' % de[k] for k in DESCRIPTORS), flush=True)
    print('elapsed %.0fs' % (time.time() - t0))

    gains = np.array([r[0] for r in rows], float)
    print()
    print('--- what predicts the headroom (Pearson, n=%d) ---' % n)
    order = []
    for k in DESCRIPTORS:
        v = np.array([d[k] for d in descs], float)
        if v.std() < 1e-12:
            order.append((0.0, k))
            continue
        c = float(np.corrcoef(gains, v)[0, 1])
        order.append((c, k))
    for c, k in sorted(order, key=lambda t: -abs(t[0])):
        print('  %-12s %+.3f' % (k, c))

    print()
    print('gain  mean %+.3f  median %+.3f  min %+.3f  max %+.3f  sd %.3f'
          % (gains.mean(), np.median(gains), gains.min(), gains.max(),
             gains.std()))
    top = np.argsort(-gains)
    half = gains[top[:max(1, n // 4)]].sum() / max(gains.sum(), 1e-9)
    print('top quartile of bursts holds %.0f%% of the total gain'
          % (100.0 * half))
    print()
    print('A descriptor that is STRUCTURAL (ferr_ramp, ferr_blob) can often be')
    print('recovered from the estimated flow alone. One that is purely a')
    print('MAGNITUDE (ferr_mean, ferr_p95) cannot -- it needs the true flow,')
    print('so a high correlation there is not a usable feature.')


if __name__ == '__main__':
    main()
