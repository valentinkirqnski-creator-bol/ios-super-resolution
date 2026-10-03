"""Which feature makes the gate reject camera rotation?

    python probe_rotation.py --ckpt sr_gate.pt --n 6

eval_goals.py measures that the gate is ~1.5 dB WORSE than merging everything
on a burst whose only motion is a rotation it estimated correctly. That says the
rejection is unearned but not where it comes from.

This holds one feature at a time at its "nothing to see here" value and reports
what mean R does. A channel whose neutralisation sends R toward 1 on the
rotation case is the one carrying the false alarm.

span and emag are the suspects worth naming in advance, because both are RAW
flow-field disagreement: under rotation the flow genuinely differs tile to tile,
so they fire on a burst that is perfectly alignable. If that is what this shows,
the fix is to replace them with the residual of a local AFFINE fit to the flow,
which rotation satisfies exactly. If it is not, the story was wrong and the
measurement says so before any of it gets built.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build as bld
import gate
import make_data
import srburst

NAMES = ['exp_a', 'log_a', 'snr', 'subpix', 'span', 'emag', 'grad', 'dir_e']
# The value each channel takes when it has nothing to report. 0 for everything
# that is a log1p of a ratio or a magnitude; exp_a is exp(-a), so its quiet
# value is 1 (zero residual), not 0.
NEUTRAL = [1.0, 0.0, None, 0.0, 0.0, 0.0, None, 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_gate.pt')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--hr', type=int, default=320)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=4242)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, '.scene_cache')

    ck = torch.load(os.path.join(here, a.ckpt), map_location='cpu',
                    weights_only=True)
    net = gate.from_checkpoint(ck)
    n_in = ck['in_ch']

    dngs, n_in_tree = srburst.find_dngs(bld.ROOT)
    order = list(np.random.default_rng(12345).permutation(n_in_tree))
    val_files = [dngs[i] for i in order[:6]]

    feats = []
    for i in range(a.n):
        rng = np.random.default_rng(a.seed + i)
        full = make_data.scene_cache(val_files[i % len(val_files)], cache)
        if full is None:
            continue
        for _ in range(24):
            scene = make_data.crop(full, rng, a.hr)
            if make_data.interesting(scene):
                break
        else:
            continue
        # A pure, correctly-estimated camera rotation. No object, no outliers,
        # and sigma_flow small so the per-tile flow really is right.
        s = srburst.BurstSpec(rng, regime=0)
        s.trans = float(np.exp(rng.uniform(np.log(0.3), np.log(8.0))))
        s.theta = float(rng.uniform(0.008, 0.020))
        s.sigma_flow = float(np.exp(rng.uniform(np.log(0.02), np.log(0.15))))
        d = bld.build_burst(scene, s, rng)
        feats.append(d['feat'][:, :n_in])
    f = np.concatenate(feats, axis=0)
    print('%d comparison frames, pure correctly-estimated rotation' % f.shape[0])

    def meanR(x):
        with torch.no_grad():
            return float(net(torch.from_numpy(x)).mean())

    base = meanR(f)
    print()
    print('%-10s %-10s %-10s' % ('held at', 'mean R', 'change'))
    print('%-10s %-10.3f %-10s' % ('(none)', base, '--'))
    rows = []
    for c in range(min(n_in, len(NAMES))):
        if NEUTRAL[c] is None:
            continue
        g = f.copy()
        g[:, c] = NEUTRAL[c]
        v = meanR(g)
        rows.append((v - base, NAMES[c], v))
    for dv, nm, v in sorted(rows, reverse=True):
        print('%-10s %-10.3f %+-10.3f' % (nm, v, dv))
    print()
    print('A large POSITIVE change means neutralising that channel stops the')
    print('rejection -- i.e. that channel is what reads legitimate rotation as')
    print('damage. snr and grad are skipped: they describe the scene, not the')
    print('motion, and have no "quiet" value that is not also a lie.')


if __name__ == '__main__':
    main()
