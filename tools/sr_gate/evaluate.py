"""Score a trained sr_gate on a held-out split.

    python evaluate.py --data data --split test --ckpt sr_gate.pt

Reports PSNR against the oracle-merge ground truth for the analytic mask, for
R = 1 (merge everything) and for the gate, plus edge and flat MSE separately.
Both of those matter and they pull in opposite directions: a mask that rejects
too much loses edge detail and adds noise in flat regions, a mask that rejects
too little doubles edges. A single average hides which one is happening.

R = 1 is in the table on purpose. It is the honest lower bound on what rejection
has to beat, and on this project's own bursts the analytic mask loses to it
across most of the sub-pixel band -- which is the thing the gate exists to fix.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate
import train as T

NAMES = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier'}


def mse(a, b, m=None):
    e = (a - b) ** 2
    if m is None:
        return float(e.mean())
    e = e.mean(axis=0)   # average the three colour channels, keep the plane
    return float((e * m).sum() / max(float(m.sum()), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--split', default='test')
    ap.add_argument('--ckpt', default='sr_gate.pt')
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    sp = T.Split(root, a.split)

    ck = torch.load(os.path.join(here, a.ckpt), map_location='cpu',
                    weights_only=True)
    net = gate.SRGate(in_ch=ck['in_ch'], width=ck['width'],
                      dilations=ck['dilations'])
    net.load_state_dict(ck['state_dict'])
    net.eval()
    print('%s split: %d bursts, %d parameters, %d training steps'
          % (a.split, sp.n, net.n_params(), ck.get('steps', -1)))

    rows = {}
    with torch.no_grad():
        for i in range(sp.n):
            d = sp.batch([i])
            args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
            Rg = T.run_gate(net, d['feat'])
            o_g = gate.merge(*args, Rg)[0].numpy()
            o_w = gate.merge(*args, d['Rw'])[0].numpy()
            o_1 = gate.merge(*args, torch.ones_like(d['Rw']))[0].numpy()
            gt = d['gt'][0].numpy()
            edge = d['edge'][0].numpy()
            # Edge = the top decile of the ground truth's gradient magnitude;
            # flat = the bottom half. Thresholds from the image itself so a dim
            # burst and a bright one are split at comparable structure.
            thi = np.percentile(edge, 90.0)
            tlo = np.percentile(edge, 50.0)
            me = (edge >= thi).astype(np.float64)
            mf = (edge <= tlo).astype(np.float64)
            key = (int(d['meta'][0][0]), float(d['meta'][0][1]))
            rows.setdefault(key, []).append((
                10 * np.log10(1 / max(mse(o_w, gt), 1e-12)),
                10 * np.log10(1 / max(mse(o_1, gt), 1e-12)),
                10 * np.log10(1 / max(mse(o_g, gt), 1e-12)),
                mse(o_w, gt, me), mse(o_g, gt, me),
                mse(o_w, gt, mf), mse(o_g, gt, mf),
                float(Rg.mean()), float(d['Rw'].mean()),
            ))

    print()
    print('%-9s %-6s %-8s %-8s %-8s %-10s %-8s %-9s %-9s %-6s' %
          ('regime', 'sig_f', 'Wronski', 'R=1', 'gate', 'vs Wronski', 'vs R=1',
           'edge MSE', 'flat MSE', 'meanR'))
    agg = []
    for key in sorted(rows):
        v = np.array(rows[key]).mean(axis=0)
        pw, p1, pg, ew, eg, fw, fg, mr, mw = v
        print('%-9s %-6.2f %-8.2f %-8.2f %-8.2f %+-10.2f %+-8.2f %+-8.1f%% %+-8.1f%% %-6.3f'
              % (NAMES[key[0]], key[1], pw, p1, pg, pg - pw, pg - p1,
                 100 * (eg / max(ew, 1e-12) - 1), 100 * (fg / max(fw, 1e-12) - 1),
                 mr))
        agg.append(v)
    agg = np.array(agg)
    pw, p1, pg = agg[:, 0], agg[:, 1], agg[:, 2]
    print('%-9s %-6s %-8.2f %-8.2f %-8.2f %+-10.2f %+-8.2f %+-8.1f%% %+-8.1f%%'
          % ('MEAN', '', pw.mean(), p1.mean(), pg.mean(),
             (pg - pw).mean(), (pg - p1).mean(),
             100 * (agg[:, 4].sum() / agg[:, 3].sum() - 1),
             100 * (agg[:, 6].sum() / agg[:, 5].sum() - 1)))
    print()
    print('gate beats Wronski in %d/%d cells; beats R=1 in %d/%d'
          % (int((pg > pw).sum()), len(pg), int((pg > p1).sum()), len(pg)))
    print('worst case vs Wronski %+.2f dB, worst case vs R=1 %+.2f dB'
          % ((pg - pw).min(), (pg - p1).min()))


if __name__ == '__main__':
    main()
