"""Sweep the policy knob and compare the two masks on the SAME axis.

A single missed%/kept pair cannot distinguish "better detector" from "more
permissive". The artifact network and the analytic mask sit at whatever
operating point each happens to produce, so comparing them point to point says
almost nothing: a mask that merges more will always keep more detail and miss
more artifacts, whichever is the better predictor underneath.

Splitting prediction from policy is exactly what makes a fair comparison
possible. Sweeping tau traces the network's whole missed%/kept curve; the
analytic mask is one point. The question is then whether that point lies above
or below the curve -- that is, whether at the SAME detail retention the network
misses fewer visible artifacts.

Two curves are traced for the baseline too, so it is not judged at one point
either: the analytic mask scaled by a constant, and raised to a power. Neither
is how you would really tune it, but both move it along a permissiveness axis,
which is the fairest available stand-in.

    python sweep_tau.py --data data_ar --ckpt sr_gate_art.pt
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
import train_artifact as TA


def score_mask(R, art, vis_sigma):
    vis = art > vis_sigma
    safe = art < 1.0
    missed = float((vis & (R > 0.5)).float().sum() /
                   vis.float().sum().clamp_min(1.0))
    kept = float((R * safe.float()).sum() / safe.float().sum().clamp_min(1.0))
    return 100.0 * missed, kept, float(R.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_ar')
    ap.add_argument('--split', default='val')
    ap.add_argument('--ckpt', default='sr_gate_art.pt')
    ap.add_argument('--base', default='sr_gate.pt')
    ap.add_argument('--vis-sigma', type=float, default=3.0)
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--regime', type=int, default=-1,
                    help='restrict to one regime; -1 = all')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    sp = T.Split(root, a.split)
    net = gate.unet_from_checkpoint(torch.load(
        a.ckpt if os.path.isabs(a.ckpt) else os.path.join(here, a.ckpt),
        map_location='cpu', weights_only=False))
    base = gate.from_checkpoint(torch.load(
        a.base if os.path.isabs(a.base) else os.path.join(here, a.base),
        map_location='cpu', weights_only=False))

    A_HAT, A_TRUE, R_BASE = [], [], []
    with torch.no_grad():
        for i in range(sp.n):
            d = sp.batch([i], ('feat', 'img', 'art', 'meta'))
            if a.regime >= 0 and int(d['meta'][0][0]) != a.regime:
                continue
            A_HAT.append(torch.expm1(
                TA.run_net(net, d['feat'], d['img']).clamp(-8, 8)))
            A_TRUE.append(d['art'])
            R_BASE.append(T.run_gate(base, d['feat']))
    a_hat = torch.cat([x.flatten() for x in A_HAT])
    art = torch.cat([x.flatten() for x in A_TRUE])
    r_base = torch.cat([x.flatten() for x in R_BASE])
    print('%d bursts, %d mask pixels, %.2f%% carry a >%.0f sigma artifact'
          % (len(A_HAT), art.numel(), 100.0 * float((art > a.vis_sigma).float().mean()),
             a.vis_sigma))

    print('\nARTIFACT NET -- sweeping tau (tolerated artifact, in noise sigma)')
    print('%-8s %-9s %-8s %-8s' % ('tau', 'missed%', 'kept', 'meanR'))
    net_pts = []
    for tau in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        R = TA.policy(a_hat, tau, a.beta)
        m, k, mr = score_mask(R, art, a.vis_sigma)
        net_pts.append((k, m, mr, tau))
        print('%-8.2f %-9.2f %-8.3f %-8.3f' % (tau, m, k, mr))

    print('\nANALYTIC MASK -- scaled, and raised to a power')
    print('%-8s %-9s %-8s %-8s' % ('op', 'missed%', 'kept', 'meanR'))
    base_pts = []
    for lbl, R in ([('x%.2f' % s, (r_base * s).clamp(0, 1))
                    for s in (0.5, 0.75, 1.0)] +
                   [('^%.2f' % p, r_base.clamp(0, 1) ** p)
                    for p in (0.5, 0.25, 0.1)]):
        m, k, mr = score_mask(R, art, a.vis_sigma)
        base_pts.append((k, m, mr, lbl))
        print('%-8s %-9.2f %-8.3f %-8.3f' % (lbl, m, k, mr))

    # The comparison: at the baseline's detail retention, what does the net miss?
    print('\nAT MATCHED DETAIL RETENTION')
    print('%-22s %-10s %-10s %-10s' %
          ('baseline point', 'kept', 'base miss%', 'net miss%'))
    nk = np.array([p[0] for p in net_pts])
    nm = np.array([p[1] for p in net_pts])
    order = np.argsort(nk)
    for k, m, mr, lbl in base_pts:
        if k < nk.min() or k > nk.max():
            print('%-22s %-10.3f %-10.2f %s' % (lbl, k, m, 'outside swept range'))
            continue
        m_net = float(np.interp(k, nk[order], nm[order]))
        verdict = 'NET BETTER' if m_net < m else 'base better'
        print('%-22s %-10.3f %-10.2f %-10.2f  %s' % (lbl, k, m, m_net, verdict))


if __name__ == '__main__':
    main()
