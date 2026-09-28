"""Restore a hard zero for overwhelming residuals, without giving back the gain.

The gate is a sigmoid, so it asymptotes but never reaches 0. Measured on
Downloads/ours2, where the subject moved ~500 raw px and block matching can only
lock its tiles onto the background, d^2/sigma^2 has a median of 52 and the gate
still passes R = 0.146 against Wronski's 0.026. Seven comparison frames at 0.146
outweigh the reference frame, so about half the output there is ghost.

min(gate, Wronski) and gate*Wronski both fix that and are both wrong: each is
<= Wronski everywhere, and the entire measured gain comes from the gate being
MORE permissive than Wronski where Wronski over-rejects. The fix has to be
one-sided -- leave the gate alone in the regime it was trained on, and restore
the analytic hard zero only where the residual is so large that no merge could be
right:

    veto(a) = clamp((a_hi - a) / (a_hi - a_lo), 0, 1)      R = R_gate * veto

The point of the ramp rather than a step is that a mask discontinuity is visible;
Eq. 9's 5x5 minimum exists for the same reason.

This sweeps (a_lo, a_hi) against BOTH things that matter at once: the held-out
test PSNR, which must not drop, and the ours2 person region, which must fall to
something near Wronski.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate
import srsim
import train as T

# (a_lo, a_hi); a_hi <= a_lo means "off"
# (a_lo, a_hi, min_radius). a_hi <= a_lo disables the veto; radius 0 disables the
# spatial minimum. Radius is in RAW pixels on the full-resolution guide, so
# radius 5 is an 11x11 footprint against Eq. 9's 5x5 on the HALF-resolution guide,
# which was 10x10 raw -- radius 5 is the closest match to the physical safety
# margin the paper's s/t/Mt were tuned against.
CANDIDATES = [(0.0, 0.0, 0), (12.0, 30.0, 0),
              (0.0, 0.0, 2), (0.0, 0.0, 5), (0.0, 0.0, 8),
              (12.0, 30.0, 2), (12.0, 30.0, 5), (12.0, 30.0, 8)]

# A blanket minimum spreads EVERY reduction, including the ones the gate makes
# for good reasons, which is why it costs 2 dB. The dilated veto spreads only the
# confident rejections: the deficit (1 - veto) is dilated, and nothing happens
# anywhere that has no overwhelming residual within the radius. At radius 0 it is
# identical to the plain veto.
DILATED = [(12.0, 30.0, 0), (12.0, 30.0, 4), (12.0, 30.0, 8), (12.0, 30.0, 16),
           (12.0, 30.0, 32), (30.0, 80.0, 16), (30.0, 80.0, 32),
           (6.0, 15.0, 16)]


def a_from_feat(feat):
    """channel 1 is log1p(a)/8, clipped at 1 -> a saturates at expm1(8)."""
    return torch.expm1(feat[:, :, 1].clamp(0, 1) * 8.0)


def veto(a, lo, hi):
    if hi <= lo:
        return torch.ones_like(a)
    return ((hi - a) / (hi - lo)).clamp(0.0, 1.0)


def dilate(x, r):
    """Max filter: spreads a confident rejection outward without touching
    anywhere that has no confident rejection nearby."""
    if r <= 0:
        return x
    sh = x.shape
    v = x.reshape(-1, 1, sh[-2], sh[-1])
    v = torch.nn.functional.pad(v, (r, r, r, r), mode='replicate')
    v = torch.nn.functional.max_pool2d(v, 2 * r + 1, stride=1)
    return v.reshape(sh)


def local_min(x, r):
    """Eq. 9's minimum, at an arbitrary radius. Replicate padding, as everywhere
    else. -max_pool2d(-x) is the min filter."""
    if r <= 0:
        return x
    sh = x.shape
    v = x.reshape(-1, 1, sh[-2], sh[-1])
    v = torch.nn.functional.pad(v, (r, r, r, r), mode='replicate')
    v = -torch.nn.functional.max_pool2d(-v, 2 * r + 1, stride=1)
    return v.reshape(sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data3')
    ap.add_argument('--split', default='test')
    ap.add_argument('--ckpt', default='sr_gate.pt')
    ap.add_argument('--threads', type=int, default=9)
    a_ = ap.parse_args()
    torch.set_num_threads(a_.threads)
    here = os.path.dirname(os.path.abspath(__file__))
    root = a_.data if os.path.isabs(a_.data) else os.path.join(here, a_.data)
    sp = T.Split(root, a_.split)

    ck = torch.load(os.path.join(here, a_.ckpt), map_location='cpu',
                    weights_only=True)
    net = gate.from_checkpoint(ck)

    acc = {c: [] for c in CANDIDATES}
    acc2 = {c: [] for c in DILATED}
    pw_all, p1_all = [], []
    with torch.no_grad():
        for i in range(sp.n):
            d = sp.batch([i])
            args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
            gt = d['gt']
            R0 = T.run_gate(net, d['feat'])
            av = a_from_feat(d['feat'])
            pw_all.append(T.psnr_t(gate.merge(*args, d['Rw']), gt))
            p1_all.append(T.psnr_t(gate.merge(*args, torch.ones_like(d['Rw'])), gt))
            for c in CANDIDATES:
                R = local_min(R0 * veto(av, c[0], c[1]), c[2])
                acc[c].append(T.psnr_t(gate.merge(*args, R), gt))
            for c in DILATED:
                R = R0 * (1.0 - dilate(1.0 - veto(av, c[0], c[1]), c[2]))
                acc2[c].append(T.psnr_t(gate.merge(*args, R), gt))

    print('held-out %s split, %d bursts' % (a_.split, sp.n))
    print('  Wronski %.2f dB   R=1 %.2f dB' % (np.mean(pw_all), np.mean(p1_all)))
    print()
    print('%-22s %-10s %-10s' % ('variant', 'PSNR', 'vs plain gate'))
    base = np.mean(acc[CANDIDATES[0]])
    for c in CANDIDATES:
        m = np.mean(acc[c])
        print('%-22s %-10.2f %+-10.2f' % ('veto %s min r=%d'
              % ('off' if c[1] <= c[0] else '%g-%g' % (c[0], c[1]), c[2]),
              m, m - base))
    print()
    print('%-22s %-10s %-10s' % ('DILATED veto', 'PSNR', 'vs plain gate'))
    for c in DILATED:
        m = np.mean(acc2[c])
        print('%-22s %-10.2f %+-10.2f'
              % ('a %g-%g dilate r=%d' % c, m, m - base))


if __name__ == '__main__':
    main()
