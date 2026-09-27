"""How much is there to win?

The merged image is differentiable in R, so the ceiling for ANY per-frame
per-pixel mask can be measured directly: optimise R itself, per burst, against
the ground truth. That is not a mask anyone could compute at inference time --
it has seen the answer -- but it bounds what a mask of this shape can do, and it
tells us whether the analytic mask is leaving 0.2 dB or 5 dB on the table.

Also checks the torch merge against the NumPy one, which is the thing training
depends on being exact.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build
import gate
import srburst


def psnr(a, b):
    m = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


def to_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


def main():
    dngs = srburst.find_dngs(build.ROOT)
    scene_full = srburst.load_scene(dngs[0])
    Hs = Ws = 384
    y0 = (scene_full.shape[0] - Hs) // 2
    x0 = (scene_full.shape[1] - Ws) // 2
    scene = np.ascontiguousarray(scene_full[y0:y0 + Hs, x0:x0 + Ws])

    names = {0: 'jitter', 1: 'rot/trans-only', 2: 'moving object', 3: 'outliers'}
    print('%-15s %-7s %-8s %-8s %-8s %-9s %-7s' %
          ('regime', 'sig_f', 'Wronski', 'R=1', 'oracleR', 'gain vs 1', 'meanRo'))
    for regime in (0, 1, 2, 3):
        for trial in range(2):
            rng = np.random.default_rng(100 + regime * 10 + trial)
            spec = srburst.BurstSpec(rng, regime=regime)
            d = build.build_burst(scene, spec, rng)

            A_ref = to_t(d['A_ref'])[None]
            B_ref = to_t(d['B_ref'])[None]
            A = to_t(d['A'])[None]
            B = to_t(d['B'])[None]
            gt = to_t(d['gt'])[None]

            # exactness check of the torch merge against the NumPy one
            if regime == 0 and trial == 0:
                R = to_t(d['Rw'])[None]
                o_t = gate.merge(A_ref, B_ref, A, B, R).numpy()[0]
                o_n = build.merged(d, d['Rw'])
                print('torch vs numpy merge: max abs diff %.3g'
                      % np.abs(o_t - o_n).max())

            logit = torch.zeros((1,) + d['Rw'].shape, requires_grad=True)
            with torch.no_grad():
                logit += 2.0
            opt = torch.optim.Adam([logit], lr=0.25)
            for _ in range(300):
                opt.zero_grad()
                out = gate.merge(A_ref, B_ref, A, B, torch.sigmoid(logit))
                loss = (out - gt).abs().mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                Ro = torch.sigmoid(logit)
                po = psnr(gate.merge(A_ref, B_ref, A, B, Ro).numpy()[0], d['gt'])
            ones = np.ones_like(d['Rw'])
            pw = psnr(build.merged(d, d['Rw']), d['gt'])
            p1 = psnr(build.merged(d, ones), d['gt'])
            print('%-15s %-7.2f %-8.2f %-8.2f %-8.2f %-9.2f %-7.3f' %
                  (names[regime], d['sigma_flow'], pw, p1, po, po - p1,
                   float(Ro.mean())))


if __name__ == '__main__':
    main()
