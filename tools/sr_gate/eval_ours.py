"""Score checkpoints on the out-of-tree bursts (Downloads/ours*).

    python eval_ours.py --ckpt sr_gate.pt --ckpt sr_gate_final.pt

Why this exists: the test split holds out scenes from the ORIGINAL three capture
sessions only, because out-of-tree sources are train-only. So it structurally
cannot show whether adding those captures helped on THAT kind of content -- the
one thing adding them was supposed to do.

Read the comparison for what it is. A model trained on data3 has SEEN these
scenes, so this is in-domain for it and out-of-domain for a model trained without
them. That is not a generalisation test; it is the question "does training on
these bursts make the mask better on these bursts", which is the question that
motivated adding them. The crops and motion seeds are fresh either way, so
neither model has seen these exact bursts.
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
import srburst
import train as T

NAMES = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier'}
SIGMAS = (0.12, 0.35, 0.9, 2.2, 5.0)


def psnr(a, b):
    m = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


def load(ckpt, here):
    ck = torch.load(os.path.join(here, ckpt), map_location='cpu', weights_only=True)
    net = gate.from_checkpoint(ck)
    return net, ck['in_ch']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=[])
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--hr', type=int, default=320)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    here = os.path.dirname(os.path.abspath(__file__))
    ckpts = a.ckpt or ['sr_gate.pt', 'sr_gate_final.pt']

    dngs, n_in_tree = srburst.find_dngs(bld.ROOT)
    scenes = dngs[n_in_tree:]
    assert scenes, 'no out-of-tree scenes found'
    print('out-of-tree scenes: %d' % len(scenes))

    nets = [(c,) + load(c, here) for c in ckpts]
    tot = {c: [] for c in ckpts}
    tot['Wronski'] = []
    tot['R=1'] = []

    k = 0
    print()
    print('%-9s %-6s %-9s %-9s %s' % ('regime', 'sig_f', 'Wronski', 'R=1',
                                      '  '.join('%-9s' % c[:9] for c in ckpts)))
    for regime in (0, 1, 2, 3):
        for sf in SIGMAS:
            for rep in range(a.reps):
                rng = np.random.default_rng(20260928 + k * 97 + rep)
                path = scenes[k % len(scenes)]
                sc = np.load(os.path.join(here, '.scene_cache',
                                          os.path.basename(path) + '.npy'),
                             mmap_mode='r')
                y0 = int(rng.integers(0, (sc.shape[0] - a.hr) // 4)) * 4
                x0 = int(rng.integers(0, (sc.shape[1] - a.hr) // 4)) * 4
                scene = np.ascontiguousarray(sc[y0:y0 + a.hr, x0:x0 + a.hr])
                spec = srburst.BurstSpec(rng, regime=regime)
                spec.sigma_flow = sf
                d = bld.build_burst(scene, spec, rng)

                pw = psnr(bld.merged(d, d['Rw']), d['gt'])
                p1 = psnr(bld.merged(d, np.ones_like(d['Rw'])), d['gt'])
                tot['Wronski'].append(pw)
                tot['R=1'].append(p1)
                cells = []
                feat = torch.from_numpy(d['feat'])[None]
                for c, net, in_ch in nets:
                    with torch.no_grad():
                        R = T.run_gate(net, feat)[0].numpy()
                    p = psnr(bld.merged(d, R), d['gt'])
                    tot[c].append(p)
                    cells.append(p)
                print('%-9s %-6.2f %-9.2f %-9.2f %s'
                      % (NAMES[regime], sf, pw, p1,
                         '  '.join('%-9.2f' % v for v in cells)), flush=True)
            k += 1

    print()
    print('%-9s %-6s %-9.2f %-9.2f %s'
          % ('MEAN', '', np.mean(tot['Wronski']), np.mean(tot['R=1']),
             '  '.join('%-9.2f' % np.mean(tot[c]) for c in ckpts)))
    base = np.array(tot[ckpts[0]])
    for c in ckpts[1:]:
        d_ = np.array(tot[c]) - base
        print('%s minus %s: %+.2f dB mean, %+.2f worst, wins %d/%d'
              % (c, ckpts[0], d_.mean(), d_.min(), int((d_ > 0).sum()), d_.size))


if __name__ == '__main__':
    main()
