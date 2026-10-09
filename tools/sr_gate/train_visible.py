"""Train the mask against COUNTED visible defects instead of summed error.

This is the native version of rotation/parallax rejection: no geometry test, no
teacher, no label derived from another mask. The network gets the same eight
features it always had and a loss that measures what a viewer sees.

Why the loss had to change rather than the features or the data: the features
were measured sufficient (distilling the geometry test reaches 1.38% residual
disagreement from these eight inputs alone, and |E| does not saturate anywhere in
the motion range that matters -- 0.0% at sigma_flow 0.90), while every
merged-error loss was measured insufficient (the correction collapses to exact
pass-through, meanR 0.310 against the base's 0.311). Summing squared error prefers
a slightly soft frame that keeps a ghost over a sharp one that removed it,
because the ghost is a few pixels and the softness is all of them. That is
arithmetic, not weighting, so visible.py counts events instead.

Measured on the mask ladder in probe_visible.py, PSNR's optimum is base^0.5 --
MORE permissive than what ships -- while visible-events' optimum is base^2,
stricter. Opposite directions over the same ladder. That disagreement is the
whole reason this run exists.

Trained through SRGateOnlyStricter so it still cannot come out more permissive
than the shipped mask whatever the new loss decides, since that guarantee is a
property of the form and costs nothing to keep.

    python train_visible.py --data data_se --base sr_gate.pt --out sr_gate_vis.pt
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
import visible


def evaluate(net, val, base):
    """ghost%, lost% and PSNR for the model and the base, over the val grid."""
    net.eval()
    rows = []
    with torch.no_grad():
        for i in range(val.n):
            d = val.batch([i])
            args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
            Rb = T.run_gate(base, d['feat'])
            sig = visible.reference_sigma(
                gate.merge(*args, torch.zeros_like(Rb)), d['gt'])
            Rg = T.run_gate(net, d['feat'])
            og, ob = gate.merge(*args, Rg), gate.merge(*args, Rb)
            gg, lg = visible.visible(og, d['gt'], R_ref=Rb, sigma_ref=sig)
            gb, lb = visible.visible(ob, d['gt'], R_ref=Rb, sigma_ref=sig)
            rows.append((int(d['meta'][0][0]), float(d['meta'][0][1]),
                         100 * float(gg), 100 * float(lg), T.psnr_t(og, d['gt']),
                         100 * float(gb), 100 * float(lb), T.psnr_t(ob, d['gt']),
                         float(Rg.mean()), float(Rb.mean())))
    net.train()
    return rows


def report(rows):
    names = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier',
             4: 'parallax'}
    print('%-9s %-6s %-8s %-8s %-8s %-8s %-9s %-8s %-7s %-7s' %
          ('regime', 'sig_f', 'ghost', 'lost', 'b.ghost', 'b.lost', 'vis gain',
           'dB', 'meanR', 'b.meanR'))
    for r in sorted(rows, key=lambda q: (q[0], q[1])):
        reg, sig, gg, lg, pg, gb, lb, pb, mg, mb = r
        print('%-9s %-6.2f %-8.3f %-8.3f %-8.3f %-8.3f %+-9.3f %+-8.2f %-7.3f '
              '%-7.3f'
              % (names[reg], sig, gg, lg, gb, lb,
                 (gb + lb) - (gg + lg), pg - pb, mg, mb))
    v = np.array([[r[2] + r[3], r[5] + r[6], r[4] - r[7], r[8], r[9]]
                  for r in rows])
    print('%-9s %-6s %-8.3f %-8s %-8.3f %-8s %+-9.3f %+-8.2f %-7.3f %-7.3f'
          % ('MEAN', '', v[:, 0].mean(), '', v[:, 1].mean(), '',
             (v[:, 1] - v[:, 0]).mean(), v[:, 2].mean(), v[:, 3].mean(),
             v[:, 4].mean()))
    # Positive = fewer visible defects than the shipped mask. This is the
    # selection criterion; the dB column is printed because it is informative and
    # is EXPECTED to be negative -- that disagreement is the point.
    return float((v[:, 1] - v[:, 0]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_se')
    ap.add_argument('--base', default='sr_gate.pt')
    ap.add_argument('--out', default='sr_gate_vis.pt')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-every', type=float, default=420.0)
    ap.add_argument('--w-ghost', type=float, default=1.0)
    ap.add_argument('--w-lost', type=float, default=1.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    train = T.Split(root, 'train')
    val = T.Split(root, 'val')
    bp = a.base if os.path.isabs(a.base) else os.path.join(here, a.base)
    base = gate.from_checkpoint(torch.load(bp, map_location='cpu',
                                           weights_only=False))
    corr = gate.SRGate(in_ch=base.convs[0].weight.shape[1],
                       dilations=base.dilations, coarse=base.coarse,
                       pool=base.pool, coarse_dilations=base.coarse_dilations)
    net = gate.SRGateOnlyStricter(base, corr)
    print('visible-loss: %d trainable parameters, %d bursts'
          % (net.n_params(), train.n), flush=True)

    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad],
                           lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.steps,
                                                pct_start=0.15)
    best, t0, log_t, eval_t = -1e9, time.time(), time.time(), time.time()
    outp = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)

    for step in range(1, a.steps + 1):
        idx = sorted(rng.choice(train.n, size=a.batch, replace=False).tolist())
        d = train.batch(idx, ('feat', 'A', 'B', 'A_ref', 'B_ref', 'gt'))
        args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
        R = T.run_gate(net, d['feat'])
        with torch.no_grad():
            # The yardstick: reference-frame noise, divided down by the frame
            # count the FROZEN base merged. Fixed for the burst, so the model
            # cannot move the threshold it is being judged against.
            Rb = T.run_gate(net.base, d['feat'])
            sig = visible.reference_sigma(
                gate.merge(*args, torch.zeros_like(R)), d['gt'])
        out = gate.merge(*args, R)
        loss, g, l = visible.visible_loss(out, d['gt'], a.w_ghost, a.w_lost,
                                          R_ref=Rb, sigma_ref=sig)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [q for q in net.parameters() if q.requires_grad], 1.0)
        opt.step()
        sched.step()

        if time.time() - log_t > 45.0:
            log_t = time.time()
            print('  step %5d/%d  %4.0fs  ghost %.4f  lost %.4f  meanR %.3f  '
                  'lr %.1e'
                  % (step, a.steps, time.time() - t0, float(g), float(l),
                     R.mean().item(), sched.get_last_lr()[0]), flush=True)
        if time.time() - eval_t > a.eval_every:
            eval_t = time.time()
            s = report(evaluate(net, val, base))
            print('  [eval @ step %d] fewer visible defects than base: %+.3f%%'
                  % (step, s), flush=True)
            if s > best:
                best = s
                torch.save(net.checkpoint(steps=step, vis_gain=s), outp)
                print('  saved (best so far)', flush=True)

    s = report(evaluate(net, val, base))
    print('final: %+.3f%% fewer visible defects than base (best %+.3f%%)'
          % (s, best))
    if s >= best:
        torch.save(net.checkpoint(steps=a.steps, vis_gain=s), outp)
        print('saved final to', outp)
    else:
        print('kept the earlier checkpoint at', outp)


if __name__ == '__main__':
    main()
