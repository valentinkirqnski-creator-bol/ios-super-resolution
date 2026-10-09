"""Can the features express the refinement, or can the merge loss not find it?

Five training runs against the merge collapsed to g = 1. Two explanations
remain and they need opposite fixes:

  (a) the features do not carry the decision, so no network of this shape can
      produce a useful gate; or
  (b) they do, but the gradient of the merge loss cannot find it -- the signal
      is there and the optimisation is the problem.

Training against the merge cannot tell these apart, because a collapse looks
identical either way. This takes the merge out of the loop. It fits a
SMOOTH-constrained oracle gate per burst (free parameters on a coarse lattice,
bilinearly upsampled -- the same kind of smooth field a convolutional net can
emit), which is the real learnable ceiling, and then trains the actual network
by plain supervised regression onto that gate. No merge in the loss, no
competing terms, no scale mismatch: just "predict this field from these
features".

Then the predicted gate goes back through the real merge, and three numbers
decide it:

  oracle        the smooth ceiling, gate fitted with the target in hand
  net on fit    the same windows the regression trained on -- EXPRESSIVITY.
                A network that cannot reach the ceiling on data it has
                memorised is short of information, not short of optimisation.
  net held out  windows it never saw -- generalisation.

  net-on-fit ~ oracle   -> (b). The features carry it; train by distillation.
  net-on-fit ~ 0        -> (a). The features do not carry it. More capacity,
                           more steps and better losses are all beside the
                           point; the inputs have to change.

    python probe_distill.py --data data
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import losses as L
from model import MaskRefineNet


def psnr(a, b):
    m = float(((a - b) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


def objective(out, gt, ow):
    """The training objective, as ratios to the Wronski merge."""
    return (L.charbonnier(out, gt) / L.charbonnier(ow, gt).detach().clamp_min(1e-8)
            + L.ghost_loss(out, gt) / L.ghost_loss(ow, gt).detach().clamp_min(1e-8))


def smooth_oracle(w, down, steps, lr):
    """Gate parameterised on a lattice `down` times coarser, then upsampled.

    This is the constraint a convolutional gate actually operates under, and it
    is why the free per-pixel oracle overstates the headroom: most of the free
    gate is per-pixel speckle, which no smooth field can reproduce and which
    only helps because the oracle can see the target.
    """
    Rw = w['Rw']
    A_ref, B_ref, A, B, gt = w['A_ref'], w['B_ref'], w['A'], w['B'], w['gt']
    b, n, h, wd = Rw.shape
    lh, lw = max(2, h // down), max(2, wd // down)
    p = torch.zeros(b * n, 1, lh, lw, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    ow = D.merge(A_ref, B_ref, A, B, Rw).detach()
    for _ in range(steps):
        g = torch.sigmoid(F.interpolate(p, size=(h, wd), mode='bilinear',
                                        align_corners=False)).view(b, n, h, wd)
        out = D.merge(A_ref, B_ref, A, B, Rw * g)
        loss = objective(out, gt, ow)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        g = torch.sigmoid(F.interpolate(p, size=(h, wd), mode='bilinear',
                                        align_corners=False)).view(b, n, h, wd)
    return g.detach()


def gate_gain(w, g):
    """PSNR of the merge under gate g, minus the Wronski merge."""
    with torch.no_grad():
        ow = D.merge(w['A_ref'], w['B_ref'], w['A'], w['B'], w['Rw'])
        og = D.merge(w['A_ref'], w['B_ref'], w['A'], w['B'], w['Rw'] * g)
        kept = float((w['Rw'] * g).sum() / w['Rw'].sum().clamp_min(1.0))
        return psnr(og, w['gt']) - psnr(ow, w['gt']), kept


def net_gate(net, w):
    b, n = w['feat'].shape[0], w['feat'].shape[1]
    g = net.gate(w['feat'].flatten(0, 1))
    return g[:, 0].view(b, n, *w['Rw'].shape[-2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--n-fit', type=int, default=6)
    ap.add_argument('--n-held', type=int, default=3)
    ap.add_argument('--window', type=int, default=256)
    ap.add_argument('--down', type=int, default=4,
                    help='oracle lattice coarsening, in mask pixels')
    ap.add_argument('--oracle-steps', type=int, default=250)
    ap.add_argument('--oracle-lr', type=float, default=0.3)
    ap.add_argument('--reg-steps', type=int, default=3000)
    ap.add_argument('--reg-lr', type=float, default=3e-3)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    bs = D.Bursts(a.data, cache=1, window=a.window, seed=11)
    n = a.n_fit + a.n_held
    if n > len(bs):
        raise SystemExit('only %d bursts' % len(bs))
    keys = ('feat', 'Rw', 'A_ref', 'B_ref', 'A', 'B', 'gt', 'ferr')
    wins = []
    t0 = time.time()
    for i in range(n):
        wins.append(dict(zip(keys, bs.batch(i))))
    print('%d windows of %d px loaded in %.0fs  (%d fit, %d held out)'
          % (n, a.window, time.time() - t0, a.n_fit, a.n_held), flush=True)

    print()
    print('--- smooth oracle (lattice %dx coarser than the mask) ---' % a.down,
          flush=True)
    print('%-6s %9s %9s' % ('win', 'gain dB', 'kept'), flush=True)
    for k, w in enumerate(wins):
        w['g'] = smooth_oracle(w, a.down, a.oracle_steps, a.oracle_lr)
        gn, kept = gate_gain(w, w['g'])
        w['gain'] = gn
        tag = '   (held out)' if k >= a.n_fit else ''
        print('%-6d %+9.3f %9.3f%s' % (k, gn, kept, tag), flush=True)
    fit, held = wins[:a.n_fit], wins[a.n_fit:]
    print('oracle   fit %+.3f dB   held %+.3f dB'
          % (np.mean([w['gain'] for w in fit]),
             np.mean([w['gain'] for w in held])), flush=True)

    # Supervised regression onto that gate. The merge is NOT in this loss.
    net = MaskRefineNet(in_ch=D.NUM_FEATURES)
    print()
    print('--- regression: %d params, features -> oracle gate ---'
          % net.n_params(), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.reg_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.reg_steps)
    for s in range(a.reg_steps):
        tot = 0.0
        opt.zero_grad()
        for w in fit:
            loss = ((net_gate(net, w) - w['g']) ** 2).mean() / len(fit)
            loss.backward()
            tot += float(loss.detach())
        opt.step()
        sched.step()
        if s % 500 == 0 or s == a.reg_steps - 1:
            print('  step %5d  mse %.5f  lr %.1e'
                  % (s, tot, sched.get_last_lr()[0]), flush=True)

    print()
    print('--- what the regression recovers ---', flush=True)
    print('%-6s %9s %9s %9s %9s' % ('win', 'oracle', 'net', 'frac', 'R2'),
          flush=True)
    rows = []
    for k, w in enumerate(wins):
        with torch.no_grad():
            g = net_gate(net, w)
        gn, kept = gate_gain(w, g)
        tgt = w['g']
        r2 = 1.0 - float(((g - tgt) ** 2).mean() / tgt.var().clamp_min(1e-12))
        frac = gn / w['gain'] if abs(w['gain']) > 1e-6 else float('nan')
        rows.append((w['gain'], gn, frac, r2))
        tag = '   (held out)' if k >= a.n_fit else ''
        print('%-6d %+9.3f %+9.3f %9.2f %9.3f   kept %.3f%s'
              % (k, w['gain'], gn, frac, r2, kept, tag), flush=True)
    v = np.array(rows, float)
    f, h = v[:a.n_fit], v[a.n_fit:]
    print()
    print('ON FIT WINDOWS (expressivity): oracle %+.3f dB  net %+.3f dB'
          '  = %.0f%%  R2 %.3f'
          % (f[:, 0].mean(), f[:, 1].mean(),
             100.0 * f[:, 1].mean() / max(abs(f[:, 0].mean()), 1e-9),
             f[:, 3].mean()))
    print('HELD OUT (generalisation):    oracle %+.3f dB  net %+.3f dB'
          '  = %.0f%%  R2 %.3f'
          % (h[:, 0].mean(), h[:, 1].mean(),
             100.0 * h[:, 1].mean() / max(abs(h[:, 0].mean()), 1e-9),
             h[:, 3].mean()))
    print()
    print('net on FIT ~ oracle  -> the features carry it; the merge-loss')
    print('                        gradient was the obstacle. Distil instead.')
    print('net on FIT ~ 0       -> the features do not carry it. The inputs')
    print('                        have to change; capacity and losses are')
    print('                        beside the point.')


if __name__ == '__main__':
    main()
