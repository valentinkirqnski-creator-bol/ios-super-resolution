"""Teach the NN mask the rejections that geometry rejection makes, and no others.

The teacher is the ANALYTIC mask with motion_geom_reject on at threshold 0.0045
-- srsim.GEOM_REJECT_THRESHOLD, the same constant core/robustness.cpp uses, and
the setting that was observed to remove the rotation and parallax ghosts on real
bursts. Every packed split already stores it per burst as Rw (build.py:106 calls
wronski_robustness with geom_reject defaulting to True), so there is nothing to
precompute.

The label is where the two masks DISAGREE, in the one direction that is allowed:

    R_target = min(R_nn, Rw)          ratio = min(1, Rw / R_nn)

Two properties come out of the algebra rather than out of tuning, which is why
this is worth preferring to a merged-image loss:

  * STATIC SCENES ARE UNTOUCHED. Where geometry rejection does not fire, Rw >=
    R_nn over the pixels that matter, the min is R_nn, and the label is exactly
    1.0 -- it asks for no change. Nothing has to be balanced for this to hold.
  * IT CANNOT BECOME MORE PERMISSIVE. The ratio lies in [0, 1], and
    SRGateOnlyStricter multiplies the frozen base by a sigmoid regardless, so
    R <= R_base at inference for any weights at all.

And the point of distilling geometry rejection rather than simply enabling it:
it fires on flow-gradient x offset-from-tile-centre, which is nonzero wherever
the flow varies, including where the flow tracked the scene correctly. That is
its measured false-positive problem -- the robustness refinement net beat it with
5.7x fewer false rejections. Pushing it through 1761 parameters and 8 features
lets the net reproduce only the firings those features actually support; the ones
that are flow-gradient noise cannot survive the bottleneck. The intended result
is not a copy of geometry rejection but its true positives without its false ones.

    python train_geom.py --data data_se --base sr_gate.pt --out sr_gate_geom.pt
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
import srsim
import train as T


def labels(Rw, r0, eps=1e-4):
    """ratio = min(1, Rw / R_nn), the only-stricter form of min(R_nn, Rw)."""
    return (Rw / r0.clamp_min(eps)).clamp(0.0, 1.0)


def disagreement(split, base, n=None):
    """How often does geometry rejection reject where the NN does not?

    Reported before training because it decides whether there is anything to
    learn: if the two masks already agree, the label is 1.0 everywhere and the
    best possible model is the base unchanged.
    """
    n = n or split.n
    tot = fire = mass = 0.0
    with torch.no_grad():
        for i in range(n):
            d = split.batch([i], ('feat', 'Rw'))
            r0 = T.run_gate(base, d['feat'])
            lab = labels(d['Rw'], r0)
            # Weighted by R_nn: a pixel the NN already rejects cannot be changed
            # in any way that reaches the output.
            w = r0
            tot += float(w.sum())
            fire += float((w * (lab < 0.9)).sum())
            mass += float((w * lab).sum())
    return fire / max(tot, 1e-9), mass / max(tot, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_se')
    ap.add_argument('--base', default='sr_gate.pt')
    ap.add_argument('--out', default='sr_gate_geom.pt')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-every', type=float, default=300.0)
    ap.add_argument('--w-fire', type=float, default=4.0,
                    help='extra weight on the pixels where the teacher rejects '
                         'and the NN does not. Those are a small minority, and '
                         'a flat L1 over the rest is minimised by changing '
                         'nothing -- which is what the merged-image run already '
                         'converged to.')
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

    fire, keep = disagreement(train, base, min(train.n, 30))
    print('teacher: analytic mask + geometry rejection @ %.4f'
          % srsim.GEOM_REJECT_THRESHOLD)
    print('disagreement on 30 train bursts: teacher rejects where the NN does '
          'not on %.1f%% of the NN mask mass; label mean %.3f'
          % (100.0 * fire, keep), flush=True)
    if fire < 0.005:
        print('WARNING: the two masks already agree -- there is nothing here '
              'for the correction to learn.', flush=True)
    print('distil-geom: %d trainable parameters, %d bursts'
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
        d = train.batch(idx, ('feat', 'Rw'))
        feat = d['feat']
        want = net.convs[0].weight.shape[1]
        if feat.shape[2] > want:
            feat = feat[:, :, :want]
        x = feat.flatten(0, 1)
        with torch.no_grad():
            r0 = base(x)[:, 0]
        tgt = labels(d['Rw'].flatten(0, 1), r0)
        ratio = torch.sigmoid(corr.head(corr._trunk(x)))[:, 0]
        # Weighted twice: by R_nn, because a pixel the base already rejects
        # cannot reach the output however wrong the ratio is there; and up on
        # the pixels the teacher actually wants taken away, which are the
        # minority that carries the whole point.
        w = r0.detach() * (1.0 + a.w_fire * (1.0 - tgt))
        loss = ((ratio - tgt).abs() * w).mean() / w.mean().clamp_min(1e-6)
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
            # NOTE: selected on agreement with the teacher, NOT on dB. The dB is
            # printed because it is informative, but geometry rejection is known
            # to COST PSNR while removing ghosts a viewer sees -- that is the
            # whole reason this label exists rather than a merged-image loss.
            v, _ = disagreement(val, net, min(val.n, 12))
            print('  [eval @ step %d] teacher-vs-model residual disagreement '
                  '%.2f%%' % (step, 100.0 * v), flush=True)
            score = -v
            if score > best:
                best = score
                torch.save(net.checkpoint(steps=step, val_gain=g,
                                          residual=v), outp)
                print('  saved (best so far)', flush=True)

    g = T.report(T.evaluate(net, val, base))
    v, _ = disagreement(val, net, min(val.n, 12))
    print('final: %+.3f dB vs base, residual disagreement %.2f%%'
          % (g, 100.0 * v))
    if -v >= best:
        torch.save(net.checkpoint(steps=a.steps, val_gain=g, residual=v), outp)
        print('saved final to', outp)
    else:
        print('kept the earlier checkpoint at', outp)


if __name__ == '__main__':
    main()
