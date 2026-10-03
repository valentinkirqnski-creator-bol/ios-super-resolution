"""Where on a frame does the mask over-reject? By brightness, by position.

    python probe_scene.py --dng C:/Users/valen/Downloads/APC_1186.dng

Three questions, each from a user-visible complaint, and each answered against
the TRUE flow error (build_burst's ferr) so an undeserved rejection can be told
from a deserved one:

  brightness  mean R binned by local brightness, restricted to WELL-ALIGNED
              pixels (ferr below the 1.6 px threshold). Those pixels should all
              be merged whatever their brightness, so any downward slope here is
              pure over-rejection. A daylight frame with deep shadow spans a
              100:1 range inside one image, and the scene bank's crops are
              screened by make_data.interesting (mean > 0.02), so high-dynamic-
              range crops are the case training is thinnest on.

  column      mean R binned by x, again on well-aligned pixels only. A mask that
              is systematically stricter down one side would show as a slope or
              as an isolated end bin.

  radius      mean R and mean ferr binned by distance from frame centre. Under
              rotation the displacement grows with radius, so the within-tile
              part of it -- the part ONE motion vector per tile cannot represent
              -- grows too. If ferr rises with radius while R stays flat, the
              residual misalignment at the sides is a flow-resolution limit and
              not something the mask can be trained out of.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build as bld
import gate
import make_data
import srburst

FERR_OK = 1.6


def bins(vals, q):
    """Quantile edges, deduplicated (a mostly-flat channel gives ties)."""
    e = np.unique(np.quantile(vals, np.linspace(0, 1, q + 1)))
    return e if e.size >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dng', required=True)
    ap.add_argument('--ckpt', default='sr_gate.pt')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--hr', type=int, default=384)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=909)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    here = os.path.dirname(os.path.abspath(__file__))

    ck = torch.load(os.path.join(here, a.ckpt), map_location='cpu',
                    weights_only=True)
    net = gate.from_checkpoint(ck)
    n_in = ck['in_ch']

    full = srburst.load_scene(a.dng)
    if full is None:
        print('not real Bayer:', a.dng)
        return 1
    print('%s  %dx%d  mean %.4f' % (os.path.basename(a.dng), full.shape[1],
                                    full.shape[0], full.mean()))

    R, Bri, X, Rad, Fe, Rw = [], [], [], [], [], []
    for i in range(a.n):
        rng = np.random.default_rng(a.seed + i)
        # Crops taken WITHOUT the interesting() screen, so the shadowed parts of
        # the frame are represented instead of filtered out -- the screen is
        # what makes this scene under-covered in the first place.
        scene = make_data.crop(full, rng, a.hr)
        spec = srburst.BurstSpec(rng, regime=0)
        spec.theta = float(rng.uniform(0.004, 0.030))
        spec.trans = float(np.exp(rng.uniform(np.log(0.5), np.log(20.0))))
        spec.sigma_flow = float(np.exp(rng.uniform(np.log(0.03), np.log(0.3))))
        d = bld.build_burst(scene, spec, rng)
        with torch.no_grad():
            r = net(torch.from_numpy(d['feat'][:, :n_in])).numpy()[:, 0]
        # Local brightness on the mask lattice: channel 2 of the features is
        # SNR, so brightness comes from the guide means instead. Reconstruct it
        # from the ground truth, reduced to the mask grid.
        g = d['gt'].mean(axis=0)
        gh, gw = r.shape[1], r.shape[2]
        bri = g.reshape(gh, g.shape[0] // gh, gw, g.shape[1] // gw).mean(axis=(1, 3))
        yy, xx = np.mgrid[0:gh, 0:gw]
        rad = np.hypot(yy - gh / 2.0, xx - gw / 2.0) / (0.5 * np.hypot(gh, gw))
        nf = r.shape[0]
        R.append(r.ravel())
        Fe.append(d['ferr'].ravel())
        Rw.append(d['Rw'].ravel())
        Bri.append(np.tile(bri.ravel(), nf))
        X.append(np.tile((xx / max(gw - 1, 1)).ravel(), nf))
        Rad.append(np.tile(rad.ravel(), nf))
    R = np.concatenate(R); Fe = np.concatenate(Fe); Rw = np.concatenate(Rw)
    Bri = np.concatenate(Bri); X = np.concatenate(X); Rad = np.concatenate(Rad)
    ok = Fe <= FERR_OK
    print('%d mask pixels, %.1f%% well aligned (ferr <= %.1f px)'
          % (R.size, 100.0 * ok.mean(), FERR_OK))

    def table(title, key, sel, extra=None):
        e = bins(key[sel], 8)
        if e is None:
            print('\n%s: not enough spread' % title)
            return
        print('\n%s   (well-aligned pixels only)' % title)
        hdr = '%-16s %-9s %-9s %-10s' % ('bin', 'gate R', 'Wronski', '% of px')
        if extra is not None:
            hdr += '%-10s' % 'mean ferr'
        print(hdr)
        k, r, w = key[sel], R[sel], Rw[sel]
        ex = extra[sel] if extra is not None else None
        for b in range(e.size - 1):
            m = (k >= e[b]) & (k < e[b + 1] if b < e.size - 2 else k <= e[b + 1])
            if m.sum() < 50:
                continue
            row = '%-16s %-9.3f %-9.3f %-10.1f' % (
                '%.3f-%.3f' % (e[b], e[b + 1]), r[m].mean(), w[m].mean(),
                100.0 * m.sum() / m.size)
            if ex is not None:
                row += '%-10.3f' % ex[m].mean()
            print(row)

    table('mean R by LOCAL BRIGHTNESS', Bri, ok)
    table('mean R by COLUMN (x, 0=left)', X, ok)
    # Radius uses ALL pixels and reports ferr, because the question there is
    # whether the FLOW degrades toward the sides, not whether R does.
    e = bins(Rad, 6)
    print('\nmean R and TRUE flow error by RADIUS   (all pixels)')
    print('%-16s %-9s %-9s %-10s' % ('radius', 'gate R', 'mean ferr', '% > 1.6px'))
    for b in range(e.size - 1):
        m = (Rad >= e[b]) & (Rad < e[b + 1] if b < e.size - 2 else Rad <= e[b + 1])
        if m.sum() < 50:
            continue
        print('%-16s %-9.3f %-9.3f %-10.1f'
              % ('%.2f-%.2f' % (e[b], e[b + 1]), R[m].mean(), Fe[m].mean(),
                 100.0 * (Fe[m] > FERR_OK).mean()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
