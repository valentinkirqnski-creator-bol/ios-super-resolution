"""Predict the ARTIFACT a merge would introduce; derive the mask from a policy.

The network does not output R. It outputs an estimate of how large an artifact
including frame n at pixel p would create, in units of the local noise sigma,
and the mask is then

    R = sigmoid((tau - a_hat) / beta)

with tau the artifact you are willing to tolerate. That split matters:

  * the LEARNED part answers "what will happen", which is a fact about the
    burst and does not change when your taste changes;
  * the POLICY part answers "how much do I accept", which is one scalar you
    turn against real bursts, after training, without retraining.

Train the network to output R directly and the tolerance is baked into the
weights, so every adjustment costs a training run.

ASYMMETRIC COST. Under-predicting the artifact means merging a ghost; over-
predicting means rejecting a frame that was fine. Those are not equally bad, and
a symmetric loss treats them as if they were. The loss is therefore the pinball
(quantile) loss at q, which penalises under-prediction q/(1-q) times more
heavily -- at the default q = 0.9, nine times. The network is thereby trained to
predict a high quantile of the artifact rather than its mean: it errs toward
"this will be worse than it looks".

The target comes from build.build_burst: the merge run twice with everything
fixed except the flow, differenced. See the comment there for why it is used
instead of |dF| or the merged-image error.

ACCEPTANCE. Reported and selected on artifacts, not PSNR:

    missed   % of pixels where the TRUE artifact clears vis_sigma and the mask
             still merges them (R > 0.5). This is the defect you see.
    kept     mean R where the true artifact is below 1 sigma -- the detail price
             paid for that.

PSNR is printed because it is informative and never optimised.

    python train_artifact.py --data data_ar --out sr_gate_art.pt
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


def policy(a_hat, tau=1.0, beta=0.5):
    """Artifact estimate (in noise sigma) -> mask. tau is the tolerance."""
    return torch.sigmoid((tau - a_hat) / beta)


def run_net(net, feat, img):
    """-> predicted log1p(artifact), (b, N, h, w)."""
    x = torch.cat([feat[:, :, :8], img], dim=2)
    b, n = x.shape[0], x.shape[1]
    r = net(x.flatten(0, 1))[:, 0]
    return r.view(b, n, x.shape[-2], x.shape[-1])


def pinball(pred, target, q=0.9):
    """Asymmetric L1: under-prediction costs q, over-prediction costs (1-q).

    At q = 0.9 a miss is 9x a false alarm, which is the asymmetry the problem
    actually has -- a ghost is a catastrophe and a slightly noisier patch is not.
    """
    d = target - pred
    return torch.maximum(q * d, (q - 1.0) * d).mean()


def metrics(a_hat, a_true, tau, beta, vis_sigma=3.0):
    """missed% and kept, the acceptance pair."""
    R = policy(a_hat, tau, beta)
    vis = a_true > vis_sigma
    safe = a_true < 1.0
    missed = float((vis & (R > 0.5)).float().sum() /
                   vis.float().sum().clamp_min(1.0))
    kept = float((R * safe.float()).sum() / safe.float().sum().clamp_min(1.0))
    return 100.0 * missed, kept, float(R.mean())


def evaluate(net, val, base, tau, beta, vis_sigma):
    rows = []
    net.eval()
    with torch.no_grad():
        for i in range(val.n):
            d = val.batch([i])
            args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
            a_hat = torch.expm1(run_net(net, d['feat'], d['img']).clamp(-8, 8))
            a_true = d['art']
            R = policy(a_hat, tau, beta)
            Rb = T.run_gate(base, d['feat'])
            m, k, mr = metrics(a_hat, a_true, tau, beta, vis_sigma)
            # The analytic mask scored on the same pair, measured directly
            # from its own R rather than through the policy.
            visb = a_true > vis_sigma
            safeb = a_true < 1.0
            mb = 100.0 * float((visb & (Rb > 0.5)).float().sum() /
                               visb.float().sum().clamp_min(1.0))
            kb = float((Rb * safeb.float()).sum() /
                       safeb.float().sum().clamp_min(1.0))
            rows.append((int(d['meta'][0][0]), float(d['meta'][0][1]),
                         m, k, mb, kb,
                         T.psnr_t(gate.merge(*args, R), d['gt']),
                         T.psnr_t(gate.merge(*args, Rb), d['gt']),
                         mr, float(Rb.mean())))
    net.train()
    return rows


def report(rows):
    names = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier',
             4: 'parallax'}
    print('%-9s %-6s %-9s %-7s %-9s %-7s %-8s %-8s %-7s %-7s' %
          ('regime', 'sig_f', 'missed%', 'kept', 'b.missed%', 'b.kept',
           'PSNR', 'base', 'meanR', 'b.meanR'))
    for r in sorted(rows, key=lambda q: (q[0], q[1])):
        print('%-9s %-6.2f %-9.2f %-7.3f %-9.2f %-7.3f %-8.2f %-8.2f %-7.3f '
              '%-7.3f' % (names[r[0]], r[1], r[2], r[3], r[4], r[5], r[6],
                          r[7], r[8], r[9]))
    v = np.array([[r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]]
                  for r in rows])
    print('%-9s %-6s %-9.2f %-7.3f %-9.2f %-7.3f %-8.2f %-8.2f %-7.3f %-7.3f'
          % ('MEAN', '', v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(),
             v[:, 3].mean(), v[:, 4].mean(), v[:, 5].mean(), v[:, 6].mean(),
             v[:, 7].mean()))
    # Selected on MISSED artifacts, with detail retention as the tie-break.
    # Not PSNR: the whole point is that PSNR trades a visible ghost for a
    # broadly smoother frame.
    return -(v[:, 0].mean() - 2.0 * v[:, 1].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_ar')
    ap.add_argument('--out', default='sr_gate_art.pt')
    ap.add_argument('--base', default='sr_gate.pt')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-every', type=float, default=420.0)
    ap.add_argument('--q', type=float, default=0.9,
                    help='pinball quantile. Higher = misses punished harder.')
    ap.add_argument('--tau', type=float, default=1.0,
                    help='tolerated artifact, in noise sigma. The policy knob.')
    ap.add_argument('--beta', type=float, default=0.5,
                    help='softness of the policy around tau')
    ap.add_argument('--vis-sigma', type=float, default=3.0,
                    help='artifact size counted as VISIBLE when scoring')
    ap.add_argument('--base-ch', type=int, default=16)
    ap.add_argument('--mid-ch', type=int, default=28)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    train = T.Split(root, 'train')
    val = T.Split(root, 'val')
    base = gate.from_checkpoint(torch.load(
        a.base if os.path.isabs(a.base) else os.path.join(here, a.base),
        map_location='cpu', weights_only=False))

    net = gate.SRGateUNet(base=a.base_ch, mid=a.mid_ch, linear_head=True)
    print('artifact net: %d params, %d in-ch, receptive field %d mask px'
          % (net.n_params(), net.in_ch, net.receptive_field()), flush=True)
    print('pinball q=%.2f (miss costs %.1fx a false alarm)  tau=%.2f sigma  '
          'beta=%.2f  visible>=%.1f sigma'
          % (a.q, a.q / (1 - a.q), a.tau, a.beta, a.vis_sigma), flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.steps,
                                                pct_start=0.15)
    best, t0, log_t, eval_t = -1e9, time.time(), time.time(), time.time()
    outp = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)

    for step in range(1, a.steps + 1):
        idx = sorted(rng.choice(train.n, size=a.batch, replace=False).tolist())
        d = train.batch(idx, ('feat', 'img', 'art'))
        pred = run_net(net, d['feat'], d['img'])
        tgt = torch.log1p(d['art'].clamp_min(0))
        loss = pinball(pred, tgt, a.q)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if time.time() - log_t > 45.0:
            log_t = time.time()
            with torch.no_grad():
                ah = torch.expm1(pred.clamp(-8, 8))
                m, k, mr = metrics(ah, d['art'], a.tau, a.beta, a.vis_sigma)
            print('  step %5d/%d  %4.0fs  pinball %.4f  pred %.3f  tgt %.3f  '
                  'missed %.1f%%  kept %.3f  meanR %.3f  lr %.1e'
                  % (step, a.steps, time.time() - t0, loss.item(),
                     float(ah.mean()), float(d['art'].mean()), m, k, mr,
                     sched.get_last_lr()[0]), flush=True)
        if time.time() - eval_t > a.eval_every:
            eval_t = time.time()
            sc = report(evaluate(net, val, base, a.tau, a.beta, a.vis_sigma))
            print('  [eval @ step %d] score %.3f' % (step, sc), flush=True)
            if sc > best:
                best = sc
                torch.save(gate.unet_checkpoint(
                    net, steps=step, score=sc, tau=a.tau, beta=a.beta,
                    q=a.q, vis_sigma=a.vis_sigma), outp)
                print('  saved (best so far)', flush=True)

    sc = report(evaluate(net, val, base, a.tau, a.beta, a.vis_sigma))
    print('final score %.3f (best %.3f)' % (sc, best))
    if sc >= best:
        torch.save(gate.unet_checkpoint(net, steps=a.steps, score=sc,
                                        tau=a.tau, beta=a.beta, q=a.q,
                                        vis_sigma=a.vis_sigma), outp)
        print('saved final to', outp)
    else:
        print('kept the earlier checkpoint at', outp)


if __name__ == '__main__':
    main()
