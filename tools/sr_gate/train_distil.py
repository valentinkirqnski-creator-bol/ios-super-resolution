"""Train the only-stricter correction by regression onto the ORACLE mask.

Why not the merged-image loss train.py uses: it was tried, on exactly this data,
and the correction converged to doing nothing -- mean R 0.310 against the base's
0.311, vs base -0.00 dB on all 25 validation rows. The diagnosis is in
probe_headroom.py's numbers: the available gain is +0.66 dB and it is bought by
removing 6.4% of the mask's mass, so it is a SPARSE edit, while the same 1761
parameters also set the mask's overall LEVEL. Both pulls land on the same
weights and the level pull is far the stronger, so the sparse edit never
survives.

Regression onto the oracle has no such conflict. Every pixel carries its own
target, make_oracle.py already solved the hard part per burst, and the network's
only job is to find the part of that solution the 8 features can express.

Safety is unchanged and does not depend on this converging:

  * the label is R_oracle / R_base in [0, 1], so it can only ask for LESS;
  * SRGateOnlyStricter multiplies by a sigmoid regardless, so R <= R_base holds
    at inference for any weights at all.

A rotation emphasis is available with --regime-weight, because the oracle finds
real headroom on every rotation row (+0.35 to +0.95 dB) while finding essentially
none on parallax (+0.01 to +0.12) -- rejecting more cannot fix parallax, the base
already rejects 80-91% there.

    python make_oracle.py  --data data_se --split train
    python train_distil.py --data data_se --base sr_gate.pt --out sr_gate_rot.pt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_se')
    ap.add_argument('--base', default='sr_gate.pt')
    ap.add_argument('--out', default='sr_gate_rot.pt')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-every', type=float, default=300.0)
    ap.add_argument('--regime-weight', default='1,1,1,1,1',
                    help='per-regime sampling weight, in regime order '
                         'jitter,rot,object,outlier,parallax. "1,4,1,1,1" '
                         'quadruples the rotation share without rebuilding the '
                         'dataset.')
    ap.add_argument('--w-edge', type=float, default=3.0,
                    help='extra weight on pixels the oracle actually changed. '
                         'The label is 1.0 on ~94% of pixels, so a flat L1 is '
                         'dominated by the part that asks for nothing.')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    train = T.Split(root, 'train')
    val = T.Split(root, 'val')
    orc = np.load(os.path.join(root, 'train_orc.npy'), mmap_mode='r')
    assert orc.shape[0] == train.n, 'oracle labels do not match the split'

    bp = a.base if os.path.isabs(a.base) else os.path.join(here, a.base)
    base = gate.from_checkpoint(torch.load(bp, map_location='cpu',
                                           weights_only=False))
    corr = gate.SRGate(in_ch=base.convs[0].weight.shape[1],
                       dilations=base.dilations, coarse=base.coarse,
                       pool=base.pool, coarse_dilations=base.coarse_dilations)
    net = gate.SRGateOnlyStricter(base, corr)
    print('distil: %d trainable parameters, %d bursts, labels %s'
          % (net.n_params(), train.n, orc.shape), flush=True)

    # Per-burst sampling probability from the regime weights.
    rw = np.array([float(x) for x in a.regime_weight.split(',')])
    reg = train.d['meta'][:, 0].astype(int)
    p = rw[reg]
    p = p / p.sum()
    print('regime mix: %s  ->  effective shares %s'
          % (rw.tolist(),
             np.round([p[reg == k].sum() for k in range(5)], 3).tolist()),
          flush=True)

    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad],
                           lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.steps,
                                                pct_start=0.15)
    best, t0, log_t, eval_t = -1e9, time.time(), time.time(), time.time()
    outp = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)

    for step in range(1, a.steps + 1):
        idx = sorted(rng.choice(train.n, size=a.batch, replace=False,
                                p=p).tolist())
        d = train.batch(idx, ('feat',))
        tgt = torch.from_numpy(np.ascontiguousarray(orc[idx],
                                                    dtype=np.float32))
        feat = d['feat']
        want = net.convs[0].weight.shape[1]
        if feat.shape[2] > want:
            feat = feat[:, :, :want]
        b, n = feat.shape[0], feat.shape[1]
        x = feat.flatten(0, 1)
        with torch.no_grad():
            r0 = base(x)[:, 0]
        ratio = torch.sigmoid(corr.head(corr._trunk(x)))[:, 0]
        tgt = tgt.view(-1, tgt.shape[-2], tgt.shape[-1])
        # Weighted to the pixels the oracle moved: a flat L1 is dominated by the
        # ~94% it left at 1.0, and matching those is what "do nothing" already
        # does perfectly.
        w = 1.0 + a.w_edge * (1.0 - tgt)
        # Weighted by R_base too: a pixel the base already rejects cannot matter
        # downstream however wrong the ratio is there.
        loss = ((ratio - tgt).abs() * w * r0.detach()).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [q for q in net.parameters() if q.requires_grad], 1.0)
        opt.step()
        sched.step()

        if time.time() - log_t > 45.0:
            log_t = time.time()
            print('  step %5d/%d  %4.0fs  loss %.5f  ratio %.3f  tgt %.3f  '
                  'lr %.1e'
                  % (step, a.steps, time.time() - t0, loss.item(),
                     ratio.mean().item(), tgt.mean().item(),
                     sched.get_last_lr()[0]), flush=True)
        if time.time() - eval_t > a.eval_every:
            eval_t = time.time()
            g = T.report(T.evaluate(net, val, base))
            print('  [eval @ step %d] mean gain vs base %+.3f dB' % (step, g),
                  flush=True)
            if g > best:
                best = g
                torch.save(net.checkpoint(steps=step, val_gain=g), outp)
                print('  saved (best so far)', flush=True)

    g = T.report(T.evaluate(net, val, base))
    print('final mean gain vs base %+.3f dB (best %+.3f)' % (g, best))
    if g >= best:
        torch.save(net.checkpoint(steps=a.steps, val_gain=g), outp)
        print('saved final to', outp)
    else:
        print('kept the earlier checkpoint at', outp)


if __name__ == '__main__':
    main()
