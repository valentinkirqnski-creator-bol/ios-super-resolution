"""Is there anything for a refinement to find?

Three training configurations collapsed to the identity, including one with the
suppression penalty removed entirely. That is either the network failing to find
a signal, or there being no signal to find -- and those need opposite responses,
so this measures which.

The gate is optimised DIRECTLY, one free value per mask pixel, under exactly the
constraint the network operates under:

    M_final = M_wronski * g,   g in [0, 1]

with access to the ground truth. No features, no capacity limit, no
generalisation gap; it is allowed to cheat and overfit each burst completely.
Whatever it reaches is therefore an upper bound on every possible refinement of
this form, and the bound is tight enough to decide the question:

  * gain ~ 0  ->  the Wronski mask is already at the ceiling for this objective
                  on this data. No architecture or feature set helps, and the
                  honest answer is that the refinement has nothing to do.
  * gain large ->  the ceiling is real and the network or the features are the
                  limitation, which is a different problem with a different fix.

Reported against both halves of the objective, because they can disagree: a
refinement can reduce ghosting while costing PSNR, and a single number would
hide that.

    python probe_headroom.py --data data
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import losses as L


def psnr(a, b):
    m = float(((a - b) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


def oracle(A_ref, B_ref, A, B, gt, Rw, steps, lr, objective):
    """Free per-pixel gate in [0, 1], fitted to this burst with the target in
    hand. Parameterised as g so the constraint is a clamp and the step size is
    scale-free."""
    g = torch.ones_like(Rw, requires_grad=True)
    opt = torch.optim.Adam([g], lr=lr)
    for _ in range(steps):
        out = D.merge(A_ref, B_ref, A, B, Rw * g)
        if objective == 'mse':
            loss = ((out - gt) ** 2).mean()
        else:
            ow = D.merge(A_ref, B_ref, A, B, Rw)
            loss = (L.charbonnier(out, gt) /
                    L.charbonnier(ow, gt).detach().clamp_min(1e-8)
                    + L.ghost_loss(out, gt) /
                    L.ghost_loss(ow, gt).detach().clamp_min(1e-8))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            g.clamp_(0.0, 1.0)
    return (Rw * g).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--window', type=int, default=256)
    ap.add_argument('--objective', default='train',
                    choices=('train', 'mse'))
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    bs = D.Bursts(a.data, cache=2, window=a.window, seed=3)
    n = min(a.n, len(bs))
    print('oracle gate, %d bursts, %d windows of %d px, objective=%s'
          % (n, n, a.window, a.objective))
    print('%-6s %9s %9s %9s %9s %9s' %
          ('burst', 'PSNR_w', 'PSNR_or', 'gain dB', 'ghost_w', 'ghost_or'))
    rows = []
    for i in range(n):
        feat, Rw, A_ref, B_ref, A, B, gt, ferr = bs.batch(i)
        Mo = oracle(A_ref, B_ref, A, B, gt, Rw, a.steps, a.lr, a.objective)
        with torch.no_grad():
            ow = D.merge(A_ref, B_ref, A, B, Rw)
            oo = D.merge(A_ref, B_ref, A, B, Mo)
            pw, po = psnr(ow, gt), psnr(oo, gt)
            gw = float(L.ghost_loss(ow, gt))
            go = float(L.ghost_loss(oo, gt))
            kept = float(Mo.sum() / Rw.sum().clamp_min(1.0))
        rows.append((pw, po, po - pw, gw, go, kept))
        print('%-6d %9.2f %9.2f %+9.2f %9.5f %9.5f   kept %.3f'
              % (i, pw, po, po - pw, gw, go, kept))
    v = np.array(rows)
    print('%-6s %9.2f %9.2f %+9.2f %9.5f %9.5f   kept %.3f'
          % ('MEAN', v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(),
             v[:, 3].mean(), v[:, 4].mean(), v[:, 5].mean()))
    print()
    print('This is an UPPER BOUND on any refinement of the form M_w * g:')
    print('  the oracle sees the target and fits each burst freely.')
    print('  mean PSNR headroom   %+.3f dB' % v[:, 2].mean())
    print('  mean ghost reduction %+.1f%%'
          % (100.0 * (v[:, 3].mean() - v[:, 4].mean()) /
             max(v[:, 3].mean(), 1e-9)))
    print('  acceptance it keeps  %.3f' % v[:, 5].mean())


if __name__ == '__main__':
    main()
