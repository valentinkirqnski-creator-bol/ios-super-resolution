"""Mean R against measured misalignment, binned -- the whole response curve.

    python response_curve.py --dirs ours3 --ckpt sr_gate.pt

eval_real.py reports two endpoints, "reject" and "keep", and on a burst where
almost every tile is moving the keep class is nearly empty and its number means
very little. This bins every tile by its measured independent motion and reports
mean R per bin, which answers the question the endpoints cannot: does R fall off
WITH misalignment, or is it simply low everywhere?

A mask that is doing its job has a curve that starts high in the well-aligned
bins and decays. A mask that is over-rejecting is flat and low. The fraction of
tiles in each bin is printed too, because a bin holding 1% of the frame should
not drive a conclusion.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build as bld
import gate
import srsim
from diag_real_guide import load_raw
from eval_real import tile_motion
from measure_motion import DEC, decimate

BINS = [0, 2, 4, 8, 16, 32, 64, 128, 1e9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=[])
    ap.add_argument('--dirs', default='ours3')
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--search', type=float, default=32.0)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    here = os.path.dirname(os.path.abspath(__file__))
    root = 'C:/Users/valen/Downloads'
    ckpts = a.ckpt or ['sr_gate.pt']

    nets = []
    for c in ckpts:
        ck = torch.load(os.path.join(here, c), map_location='cpu',
                        weights_only=True)
        nets.append((c, gate.from_checkpoint(ck), ck['in_ch']))

    cfg = srsim.Cfg(noise_gain=1.0, tile_size=16).use_decimated_guide()
    std_c, diff_c = bld.sqrt_curves(cfg)

    acc = {}
    cnt = np.zeros(len(BINS) - 1)
    for d in a.dirs.split(','):
        files = sorted(glob.glob(os.path.join(root, d, '*.dng')))
        if not files:
            continue
        ref, _ = load_raw(files[0])
        ref_d = decimate(ref, DEC)
        gm_r, gv_r = srsim.local_stats_3x3(srsim.compute_guide_decimate3(ref))
        gh, gw = gm_r.shape[:2]
        h, w = ref.shape
        for f in files[1:1 + a.frames]:
            mov, _ = load_raw(f)
            ny, nx = h // 16, w // 16
            ind, flow = tile_motion(ref_d, decimate(mov, DEC), gh, gw, ny, nx,
                                    a.search)
            gm_c, _ = srsim.local_stats_3x3(srsim.compute_guide_decimate3(mov))
            d_sq, sig_sq, _ = srsim.compute_d_sigma(gm_r, gv_r, gm_c, flow, cfg,
                                                    std_c, diff_c)
            feat = srsim.build_features(d_sq, sig_sq, gm_r, gv_r, flow, cfg)
            masks = [('Wronski',
                      srsim.wronski_robustness(d_sq, sig_sq, flow, gm_r, cfg))]
            for nm, net, nin in nets:
                with torch.no_grad():
                    masks.append((nm, net(torch.from_numpy(
                        feat[:nin][None]))[0, 0].numpy()))
            for bi in range(len(BINS) - 1):
                m = (ind >= BINS[bi]) & (ind < BINS[bi + 1])
                if not m.any():
                    continue
                cnt[bi] += m.sum()
                for nm, R in masks:
                    acc.setdefault(nm, np.zeros((len(BINS) - 1, 2)))
                    acc[nm][bi, 0] += float(R[m].sum())
                    acc[nm][bi, 1] += m.sum()

    print('%s, %d comparison frames' % (a.dirs, a.frames))
    hdr = '%-16s' % 'misalignment'
    for nm in acc:
        hdr += '%-12s' % nm[:11]
    print(hdr + '%-10s' % '% of frame')
    tot = cnt.sum()
    for bi in range(len(BINS) - 1):
        if cnt[bi] == 0:
            continue
        hi = '%g' % BINS[bi + 1] if BINS[bi + 1] < 1e8 else 'inf'
        row = '%-16s' % ('%g-%s px' % (BINS[bi], hi))
        for nm in acc:
            v = acc[nm][bi]
            row += '%-12.3f' % (v[0] / max(v[1], 1))
        print(row + '%-10.1f' % (100.0 * cnt[bi] / max(tot, 1)))


if __name__ == '__main__':
    main()
