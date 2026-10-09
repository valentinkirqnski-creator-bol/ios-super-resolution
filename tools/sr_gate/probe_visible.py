"""Does the visible-events metric rank masks the way the EYE did?

A new loss is worth training against only if it reproduces judgements already
known to be right. The sharpest of those is on the parallax rows: merging
everything (R = 1) beats the shipped mask by about +14 dB of PSNR, and merging
everything is visibly ghosted. PSNR is not slightly wrong there, it is wrong by
fourteen decibels and in the wrong direction.

So the test is simple. Score a ladder of masks that runs from maximally
permissive to maximally strict:

    R = 1        merge every frame at full weight -- maximum ghosting
    R = base^p   the shipped mask raised to a power; p < 1 is more permissive
                 than it, p > 1 stricter, p = 1 is the shipped mask itself
    R = 0        reference frame only -- no ghosting possible, maximum noise

Powers of the shipped mask rather than other trained models on purpose: it needs
nothing but the mask that ships, so the ladder is reproducible and carries no
assumption from any earlier experiment.

What has to come out:

  * ghost% must RISE monotonically toward the permissive end. If it does not, the
    metric is not measuring ghosting.
  * on parallax, R = 1 must score WORSE than the shipped mask, which is the
    judgement PSNR gets backwards.
  * lost% must rise toward the strict end, or the metric would have no opinion
    about over-rejection and would be maximised by rejecting everything.

A metric that fails any of those is not worth a training run.

    python probe_visible.py --data data_se --base sr_gate.pt
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
import visible

NAMES = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier', 4: 'parallax'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_se')
    ap.add_argument('--base', default='sr_gate.pt')
    ap.add_argument('--split', default='val')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    sp = T.Split(root, a.split)
    bp = a.base if os.path.isabs(a.base) else os.path.join(here, a.base)
    base = gate.from_checkpoint(torch.load(bp, map_location='cpu',
                                           weights_only=False))

    # permissive -> strict
    ladder = [('R=1', None), ('base^0.25', 0.25), ('base^0.5', 0.5),
              ('base', 1.0), ('base^2', 2.0), ('base^4', 4.0), ('R=0', 0.0)]

    acc = {n: [] for n, _ in ladder}
    per_reg = {}
    with torch.no_grad():
        for i in range(sp.n):
            d = sp.batch([i])
            args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
            Rb = T.run_gate(base, d['feat'])
            reg = int(d['meta'][0][0])
            # Noise floor of the reference frame alone. A property of the burst,
            # not of any mask, so it is computed once and every rung of the
            # ladder is measured against the same one.
            out_ref = gate.merge(*args, torch.zeros_like(Rb))
            sig_ref = visible.reference_sigma(out_ref, d['gt'])
            for name, p in ladder:
                if p is None:
                    R = torch.ones_like(Rb)
                elif p == 0.0:
                    R = torch.zeros_like(Rb)
                else:
                    R = Rb.clamp(0, 1) ** p
                out = gate.merge(*args, R)
                # Yardstick fixed at the SHIPPED mask for every rung, so the ladder is
                # measured against one threshold rather than each rung's own.
                g, l = visible.visible(out, d['gt'], R_ref=Rb,
                                       sigma_ref=sig_ref)
                ps = T.psnr_t(out, d['gt'])
                acc[name].append((100 * float(g), 100 * float(l), ps,
                                  float(R.mean())))
                per_reg.setdefault((reg, name), []).append(
                    (100 * float(g), 100 * float(l), ps))

    print('ALL %d bursts' % sp.n)
    print('%-10s %-9s %-9s %-9s %-9s %-8s' %
          ('mask', 'ghost%', 'lost%', 'sum%', 'PSNR', 'meanR'))
    for name, _ in ladder:
        v = np.array(acc[name])
        print('%-10s %-9.3f %-9.3f %-9.3f %-9.2f %-8.3f'
              % (name, v[:, 0].mean(), v[:, 1].mean(),
                 v[:, 0].mean() + v[:, 1].mean(), v[:, 2].mean(),
                 v[:, 3].mean()))

    print('\nPARALLAX only -- the rows where PSNR prefers R=1 by ~14 dB')
    print('%-10s %-9s %-9s %-9s %-9s' %
          ('mask', 'ghost%', 'lost%', 'sum%', 'PSNR'))
    for name, _ in ladder:
        rows = per_reg.get((4, name), [])
        if not rows:
            continue
        v = np.array(rows)
        print('%-10s %-9.3f %-9.3f %-9.3f %-9.2f'
              % (name, v[:, 0].mean(), v[:, 1].mean(),
                 v[:, 0].mean() + v[:, 1].mean(), v[:, 2].mean()))

    g1 = np.array(acc['R=1'])[:, 0].mean()
    gb = np.array(acc['base'])[:, 0].mean()
    p1 = np.array([r[2] for r in per_reg.get((4, 'R=1'), [])]).mean()
    pb = np.array([r[2] for r in per_reg.get((4, 'base'), [])]).mean()
    v1 = np.array([r[0] + r[1] for r in per_reg.get((4, 'R=1'), [])]).mean()
    vb = np.array([r[0] + r[1] for r in per_reg.get((4, 'base'), [])]).mean()
    print('\nVERDICT')
    print('  ghost rises toward permissive:      %s  (R=1 %.3f%% vs base %.3f%%)'
          % ('YES' if g1 > gb else 'NO', g1, gb))
    print('  parallax: PSNR prefers R=1 by %+.1f dB' % (p1 - pb))
    print('  parallax: visible prefers %s  (R=1 %.3f%% vs base %.3f%%)'
          % ('base -- AGREES WITH THE EYE' if vb < v1 else 'R=1 -- SAME ERROR AS PSNR',
             v1, vb))


if __name__ == '__main__':
    main()
