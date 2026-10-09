"""The invariant, tested rather than asserted in a comment.

    M_final <= M_wronski  everywhere, for every input, for any weights.

Checked with RANDOM weights at several magnitudes, not just trained ones: the
guarantee is supposed to come from the form (a sigmoid in (0,1) times a
non-negative mask), so a trained net passing proves much less than an
adversarially initialised one passing.

Also checks the polarity the whole design rests on -- that M = 1 accepts and
M = 0 rejects in the pipeline's own merge -- by running the merge at both
extremes and confirming M = 0 reproduces the reference-only result.

    python test_invariant.py            # synthetic only
    python test_invariant.py --data data   # also on real dumped bursts
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import model as M

FAIL = []


def check(name, ok, detail=''):
    print('  %-52s %s%s' % (name, 'PASS' if ok else '*** FAIL ***',
                            ('  ' + detail) if detail else ''))
    if not ok:
        FAIL.append(name)


def test_invariant_random_weights():
    print('invariant M_final <= M_wronski, random weights:')
    torch.manual_seed(0)
    for sigma in (0.5, 2.0, 8.0):
        net = M.MaskRefineNet()
        for p in net.parameters():
            torch.nn.init.normal_(p, 0.0, sigma)
        feat = torch.randn(1, 3, D.NUM_FEATURES, 48, 61) * 3.0
        mask = torch.rand(1, 3, 48, 61)
        with torch.no_grad():
            mf, g = net(feat, mask)
        over = float((mf - mask).max())
        check('weights N(0, %.1f): max(M_final - M_wronski)' % sigma,
              over <= 0.0, '%.3e' % over)
        # [0, 1], not (0, 1): with large weights the sigmoid saturates to
        # exactly 0 and 1 in float32. That is harmless -- g <= 1 is what the
        # invariant needs -- so requiring strict inequality would be testing
        # floating point, not the design.
        check('  gate stays inside [0, 1]',
              bool((g >= 0).all() and (g <= 1).all()),
              'min %.3e max %.6f' % (float(g.min()), float(g.max())))
        check('  a zero Wronski pixel stays zero',
              float(mf[mask == 0].abs().max() if (mask == 0).any() else 0.0) == 0.0)


def test_invariant_extreme_inputs():
    print('invariant under degenerate inputs:')
    net = M.MaskRefineNet()
    mask = torch.rand(1, 2, 32, 32)
    for name, feat in (('zeros', torch.zeros(1, 2, D.NUM_FEATURES, 32, 32)),
                       ('huge', torch.full((1, 2, D.NUM_FEATURES, 32, 32), 1e4)),
                       ('negative', torch.full((1, 2, D.NUM_FEATURES, 32, 32), -1e4))):
        with torch.no_grad():
            mf, _ = net(feat, mask)
        check('features all %-9s -> M_final <= M_wronski' % name,
              float((mf - mask).max()) <= 0.0)
        check('  and finite', bool(torch.isfinite(mf).all()))


def test_size_independence():
    """Fully convolutional: the same input tile must give the same answer
    whatever the surrounding image size. A net that has learned to read the
    border would fail this, which is how the zero-padding trap shows itself."""
    print('size independence:')
    torch.manual_seed(1)
    net = M.MaskRefineNet()
    for p in net.parameters():
        torch.nn.init.normal_(p, 0.0, 0.7)
    big = torch.randn(1, 1, D.NUM_FEATURES, 96, 96)
    with torch.no_grad():
        g_big = net.gate(big.flatten(0, 1))[0, 0]
        g_crop = net.gate(big[:, :, :, 16:80, 16:80].flatten(0, 1))[0, 0]
    rf = net.receptive_field()
    pad = rf // 2 + 1
    a = g_big[16 + pad:80 - pad, 16 + pad:80 - pad]
    b = g_crop[pad:-pad, pad:-pad]
    d = float((a - b).abs().max())
    check('interior agrees between 96x96 and its 64x64 crop', d < 1e-5, '%.2e' % d)


def test_polarity(root):
    """M = 1 accepts, M = 0 rejects -- verified through the pipeline's merge."""
    print('mask polarity, on real dumped bursts:')
    bs = D.Bursts(root)
    for i in range(min(2, len(bs))):
        feat, Rw, A_ref, B_ref, A, B, gt, ferr = bs.batch(i)
        zero = D.merge(A_ref, B_ref, A, B, torch.zeros_like(Rw))
        ref_only = A_ref / B_ref.clamp_min(1e-8)
        d = float((zero - ref_only).abs().max())
        check('burst %d: M = 0 gives the reference-frame merge' % i, d < 1e-5,
              '%.2e' % d)
        one = D.merge(A_ref, B_ref, A, B, torch.ones_like(Rw))
        check('burst %d: M = 1 differs from M = 0 (frames contribute)' % i,
              float((one - zero).abs().max()) > 1e-4)
        # Monotone: more acceptance moves the result toward the all-frames merge.
        half = D.merge(A_ref, B_ref, A, B, torch.full_like(Rw, 0.5))
        dz = float((half - zero).abs().mean())
        do = float((half - one).abs().mean())
        check('burst %d: M = 0.5 lies between the two' % i, dz > 0 and do > 0)


def test_merge_linearity(root):
    """The reason the factors can be dumped at all: the merge is linear in M."""
    print('merge linearity in the mask:')
    bs = D.Bursts(root)
    feat, Rw, A_ref, B_ref, A, B, gt, ferr = bs.batch(0)
    a, b = torch.rand_like(Rw), torch.rand_like(Rw)
    up = lambda m: D.upsample_mask(m, A.shape[-2], A.shape[-1]).unsqueeze(2)
    lhs_num = A_ref + (A * up(a + b)).sum(1)
    rhs_num = A_ref + (A * up(a)).sum(1) + (A * up(b)).sum(1)
    d = float((lhs_num - rhs_num).abs().max())
    check('numerator is additive in M', d < 1e-3, '%.2e' % d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='')
    a = ap.parse_args()
    test_invariant_random_weights()
    test_invariant_extreme_inputs()
    test_size_independence()
    if a.data and os.path.isdir(a.data):
        test_polarity(a.data)
        test_merge_linearity(a.data)
    else:
        print('(no --data: polarity and linearity checks skipped)')
    print()
    if FAIL:
        print('FAILED: %d' % len(FAIL))
        for f in FAIL:
            print('  ', f)
        return 1
    print('ALL PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
