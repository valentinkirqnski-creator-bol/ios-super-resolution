"""Precompute the only-stricter ORACLE mask for every burst in a split.

probe_headroom.py measured that the oracle is worth +0.66 dB over the shipped
mask and buys it by removing only 6.4% of the mask's mass -- a sparse, surgical
edit. Training against the merged image failed to find any of it, and the reason
is visible in that run's log: the correction has one set of parameters to set the
overall level AND to find the sparse edit, and the level pull is much the louder
of the two. It converged to exact pass-through, meanR 0.310 against the base's
0.311.

So hand it the answer instead. For each burst, optimise R directly under

    0 <= R <= R_base

and store the result. Training then becomes dense supervised regression onto
R_oracle / R_base, which has no level/selectivity conflict in it at all: every
pixel carries its own target.

The safety property survives, twice over: the label is <= R_base by
construction, and SRGateOnlyStricter enforces it again at inference whatever the
regression does.

    python make_oracle.py --data data_se --split train --base sr_gate.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate
import train as T
from probe_headroom import oracle_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_se')
    ap.add_argument('--split', default='train')
    ap.add_argument('--base', default='sr_gate.pt')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    sp = T.Split(root, a.split)
    bp = a.base if os.path.isabs(a.base) else os.path.join(here, a.base)
    base = gate.from_checkpoint(torch.load(bp, map_location='cpu',
                                           weights_only=False))

    out = None
    t0 = time.time()
    for i in range(sp.n):
        d = sp.batch([i], ('feat', 'A', 'B', 'A_ref', 'B_ref', 'gt'))
        args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
        with torch.no_grad():
            Rb = T.run_gate(base, d['feat'])
        Ro = oracle_r(args, d['gt'], Rb, a.steps, a.lr)
        # Stored as the RATIO the correction has to predict, not as R itself, so
        # the label does not have to re-encode what the base already knows.
        g = (Ro / Rb.clamp_min(1e-6)).clamp(0.0, 1.0)
        if out is None:
            out = np.lib.format.open_memmap(
                os.path.join(root, '%s_orc.npy' % a.split), mode='w+',
                dtype=np.float32, shape=(sp.n,) + tuple(g.shape[1:]))
        out[i] = g[0].numpy()
        if i % 10 == 0 or i == sp.n - 1:
            el = time.time() - t0
            print('  %3d/%d  %4.0fs  (eta %4.0fs)  kept %.3f'
                  % (i + 1, sp.n, el, el / (i + 1) * (sp.n - i - 1),
                     float(g.mean())), flush=True)
    out.flush()
    print('wrote %s_orc.npy  shape %s' % (a.split, out.shape))


if __name__ == '__main__':
    main()
