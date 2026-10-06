"""How much can ANY only-stricter mask gain over the base? An upper bound.

A correction of the form R = R_base * sigmoid(corr) can only move R DOWN. So
before asking whether a 1761-parameter network finds something, ask whether
there is anything to find: optimise R directly, as free per-pixel variables,
under exactly that constraint

    0 <= R <= R_base

and report the gain. Projected gradient on R itself has no feature bottleneck,
no capacity limit and no generalisation gap -- it is allowed to cheat by looking
at the target. Whatever it reaches is therefore an upper bound on every possible
only-stricter model, and the bound is tight enough to be decisive:

  * if the ORACLE gains ~0, the base does not over-merge on this data and no
    network can help. The dataset is the problem, not the model.
  * if the ORACLE gains a lot, the headroom exists and the gap is the features
    or the capacity. That is a different problem with a different fix.

Run after training so the two do not compete for threads:

    python probe_headroom.py --data data_se --base sr_gate.pt
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

NAMES = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier', 4: 'parallax'}


def oracle_r(args, gt, R_base, steps, lr):
    """Projected gradient on R, constrained to [0, R_base]."""
    # Parameterise as R = R_base * g with g in [0, 1], which makes the
    # constraint a clamp on g and keeps the step size scale-free.
    g = torch.ones_like(R_base).requires_grad_(True)
    opt = torch.optim.Adam([g], lr=lr)
    for _ in range(steps):
        out = gate.merge(*args, R_base * g)
        loss = ((out - gt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            g.clamp_(0.0, 1.0)
    return (R_base * g).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_se')
    ap.add_argument('--base', default='sr_gate.pt')
    ap.add_argument('--split', default='val')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    sp = T.Split(root, a.split)
    bp = a.base if os.path.isabs(a.base) else os.path.join(here, a.base)
    base = gate.from_checkpoint(torch.load(bp, map_location='cpu',
                                           weights_only=False))

    print('%-9s %-6s %-8s %-8s %-8s %-9s %-7s %-7s %-7s' %
          ('regime', 'sig_f', 'base', 'oracle', 'R=1', 'headroom', 'meanRb',
           'meanRo', 'kept'))
    rows = []
    for i in range(sp.n):
        d = sp.batch([i])
        args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
        with torch.no_grad():
            Rb = T.run_gate(base, d['feat'])
            pb = T.psnr_t(gate.merge(*args, Rb), d['gt'])
            p1 = T.psnr_t(gate.merge(*args, torch.ones_like(Rb)), d['gt'])
        Ro = oracle_r(args, d['gt'], Rb, a.steps, a.lr)
        with torch.no_grad():
            po = T.psnr_t(gate.merge(*args, Ro), d['gt'])
        mb, mo = float(Rb.mean()), float(Ro.mean())
        # Fraction of the base's mass the oracle leaves alone. Near 1 means the
        # oracle wants the same mask the base already produces.
        kept = mo / max(mb, 1e-9)
        reg, sig = int(d['meta'][0][0]), float(d['meta'][0][1])
        print('%-9s %-6.2f %-8.2f %-8.2f %-8.2f %+-9.2f %-7.3f %-7.3f %-7.3f'
              % (NAMES[reg], sig, pb, po, p1, po - pb, mb, mo, kept),
              flush=True)
        rows.append((pb, po, p1, mb, mo))

    v = np.array(rows)
    print('%-9s %-6s %-8.2f %-8.2f %-8.2f %+-9.2f %-7.3f %-7.3f %-7.3f'
          % ('MEAN', '', v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(),
             (v[:, 1] - v[:, 0]).mean(), v[:, 3].mean(), v[:, 4].mean(),
             v[:, 4].mean() / max(v[:, 3].mean(), 1e-9)))
    print('\nheadroom is an UPPER BOUND on any only-stricter model: the oracle '
          'sees the target.')


if __name__ == '__main__':
    main()
