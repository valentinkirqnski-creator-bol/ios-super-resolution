"""Which analytic mask is the gate actually being compared against?

    python baselines.py --data data --split test

The shipping mask is Eq. 6-9 with motion_geom_reject on, and two of its parts are
out of their calibrated domain on the full-resolution FFT guide: the geometry
threshold is in the DECIMATED guide's gradient units (stated in the commit that
pinned it), and Eq. 9's 5x5 minimum was designed for a lattice four times
coarser. A win over that mask is only interesting if it is also a win over its
better variants, so this scores them all on the same held-out bursts.

Two of the variants are reconstructed exactly from the stored features rather
than re-synthesised, which is possible because feature 0 IS exp(-d^2/sigma^2) and
feature 4 determines the s1/s2 test (it saturates only at a span of ts/2 = 8 raw
px, ten times r_Mt, where the test has long since fired):

    shipping      Eq. 6-9 + geometry rejection     (Config as pinned; stored Rw)
    no geom       Eq. 6-8 + the 5x5 minimum, geometry rejection off
    pointwise     Eq. 6-8, no minimum, no geometry rejection
    R = 1         merge everything
    gate          the network
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate
import srsim
import train as T

NAMES = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier'}
COLS = ['shipping', 'no geom', 'pointwise', 'R=1', 'gate']


def psnr_t(out, gt):
    return 10.0 * np.log10(1.0 / max(((out - gt) ** 2).mean().item(), 1e-12))


def analytic_from_features(feat, tile_size):
    """Eq. 6-8 recovered from the stored feature plane. feat: (N, C, h, w)."""
    exp_a = np.clip(feat[:, srsim.FEATURE_NAMES.index('exp_a')], 0.0, 1.0)
    span_f = feat[:, srsim.FEATURE_NAMES.index('span')]
    span = span_f / srsim.F_SPAN_SCALE * float(tile_size)
    s = np.where(span * span > srsim.R_MT ** 2, srsim.R_S1, srsim.R_S2)
    return np.clip(s * exp_a - srsim.R_T, 0.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--split', default='test')
    ap.add_argument('--ckpt', default='sr_gate.pt')
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    sp = T.Split(root, a.split)
    ck = torch.load(os.path.join(here, a.ckpt), map_location='cpu',
                    weights_only=True)
    net = gate.SRGate(in_ch=ck['in_ch'], width=ck['width'],
                      dilations=ck['dilations'])
    net.load_state_dict(ck['state_dict'])
    net.eval()

    rows = {}
    with torch.no_grad():
        for i in range(sp.n):
            d = sp.batch([i])
            args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
            gt = d['gt']
            ts = int(d['meta'][0][3])
            feat = d['feat'][0].numpy()
            r_pt = analytic_from_features(feat, ts)
            r_min = np.stack([srsim.local_min_5x5(r_pt[k])
                              for k in range(r_pt.shape[0])])
            v = {
                'shipping': psnr_t(gate.merge(*args, d['Rw']), gt),
                'no geom': psnr_t(gate.merge(*args,
                                             torch.from_numpy(r_min)[None]), gt),
                'pointwise': psnr_t(gate.merge(*args,
                                               torch.from_numpy(r_pt)[None]), gt),
                'R=1': psnr_t(gate.merge(*args, torch.ones_like(d['Rw'])), gt),
                'gate': psnr_t(gate.merge(*args, T.run_gate(net, d['feat'])), gt),
            }
            rows.setdefault((int(d['meta'][0][0]), float(d['meta'][0][1])),
                            []).append([v[c] for c in COLS])

    print('%-9s %-6s ' % ('regime', 'sig_f') +
          ' '.join('%-11s' % c for c in COLS))
    tot = []
    for key in sorted(rows):
        m = np.array(rows[key]).mean(axis=0)
        print('%-9s %-6.2f ' % (NAMES[key[0]], key[1]) +
              ' '.join('%-11.2f' % x for x in m))
        tot.append(m)
    tot = np.array(tot)
    print('%-9s %-6s ' % ('MEAN', '') +
          ' '.join('%-11.2f' % x for x in tot.mean(axis=0)))

    gi = COLS.index('gate')
    print()
    for ci, c in enumerate(COLS):
        if c == 'gate':
            continue
        diff = tot[:, gi] - tot[:, ci]
        print('gate vs %-10s %+.2f dB mean, %+.2f worst, wins %d/%d'
              % (c, diff.mean(), diff.min(), int((diff > 0).sum()), len(diff)))
    best = int(np.argmax(tot.mean(axis=0)[:COLS.index('R=1')]))
    print()
    print('strongest analytic variant: %s (%.2f dB)'
          % (COLS[best], tot.mean(axis=0)[best]))


if __name__ == '__main__':
    main()
