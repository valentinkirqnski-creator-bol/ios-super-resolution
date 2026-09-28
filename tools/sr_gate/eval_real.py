"""Acceptance test on the REAL bursts, with no hand-drawn regions.

    python eval_real.py --ckpt sr_gate_h_fine.pt --ckpt sr_gate_h_ms.pt

The synthetic test grid said +5.2 dB in 20/20 cells while the mask was passing a
ghost on a real subject, so a synthetic score is not evidence about real bursts.
This scores the real ones directly, and decides what SHOULD be rejected by
measuring it rather than by drawing a box: per tile, the phase-correlation
displacement minus the frame's global one is independent motion, and a tile with
a lot of it cannot be aligned by a translation the block matcher could find.

Two numbers per mask, and they pull against each other:

    reject  = 1 - mean R over tiles with independent motion > `--moving` px
              (higher is better: the ghost is being kept out)
    keep    = mean R over tiles with < 4 px
              (higher is better: aligned detail is still being merged)

Wronski is scored alongside as the reference both must beat to be worth having.
The flow handed to the mask is the GLOBAL background shift everywhere, which is
what block matching produces on a subject it cannot track -- the failure case.
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
from measure_motion import DEC, STEP, WIN, decimate, pc_peak


def tile_motion(ref_d, mov_d, gh, gw, ny, nx, search=32.0):
    """Independent motion per guide pixel, and the flow a real block matcher
    would produce, on the mask's tile grid.

    The flow matters as much as the motion map. Handing the mask a single global
    shift makes even the static background misaligned -- by up to DEC raw px of
    quantisation plus any rotation -- so every mask rejects everything and the
    "keep" number measures nothing. A real coarse-to-fine matcher follows smooth
    motion accurately and CANNOT follow a subject beyond its refinement range, so
    the per-tile vector is modelled as the local phase-correlation peak clamped
    to `search` raw px of the global one.
    """
    gy, gx, _ = pc_peak(ref_d, mov_d)
    ys = list(range(0, ref_d.shape[0] - WIN + 1, STEP))
    xs = list(range(0, ref_d.shape[1] - WIN + 1, STEP))
    m = np.zeros((len(ys), len(xs)), np.float32)
    fy_ = np.zeros((len(ys), len(xs)), np.float32)
    fx_ = np.zeros((len(ys), len(xs)), np.float32)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            ty, tx, _ = pc_peak(ref_d[y:y + WIN, x:x + WIN],
                                mov_d[y:y + WIN, x:x + WIN])
            m[iy, ix] = np.hypot(ty - gy, tx - gx) * DEC
            fy_[iy, ix] = (gy + np.clip(ty - gy, -search / DEC, search / DEC)) * DEC
            fx_[iy, ix] = (gx + np.clip(tx - gx, -search / DEC, search / DEC)) * DEC

    def up(a, H, W):
        yi = np.clip((np.arange(H) * a.shape[0] // H), 0, a.shape[0] - 1)
        xi = np.clip((np.arange(W) * a.shape[1] // W), 0, a.shape[1] - 1)
        return a[np.ix_(yi, xi)]

    flow = np.zeros((ny, nx, 2), np.float32)
    flow[..., 0] = up(fx_, ny, nx)
    flow[..., 1] = up(fy_, ny, nx)
    return up(m, gh, gw), flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=[])
    ap.add_argument('--dirs', default='ours,ours2,ours3')
    ap.add_argument('--frames', type=int, default=3, help='comparison frames')
    ap.add_argument('--moving', type=float, default=32.0)
    ap.add_argument('--static', type=float, default=4.0)
    ap.add_argument('--search', type=float, default=32.0,
                    help='raw px the modelled matcher can refine beyond '
                         'the global shift')
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

    print('%-8s %-6s %-10s %-8s %-8s   %s' %
          ('burst', 'frame', 'mask', 'reject', 'keep', '(reject = 1-meanR where '
           'independent motion > %g px; keep = meanR where < %g)'
           % (a.moving, a.static)))
    tot = {}
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
            mo = ind > a.moving
            st = ind < a.static
            for nm, R in masks:
                rej = 1.0 - float(R[mo].mean()) if mo.any() else float('nan')
                kep = float(R[st].mean()) if st.any() else float('nan')
                tot.setdefault(nm, []).append((rej, kep))
                print('%-8s %-6s %-10s %-8.3f %-8.3f' %
                      (d, os.path.basename(f)[4:8], nm[:10], rej, kep))
    print()
    print('%-20s %-10s %-10s %-10s' % ('MEAN over all', 'reject', 'keep',
                                       'reject+keep'))
    for nm, v in tot.items():
        v = np.array(v)
        print('%-20s %-10.3f %-10.3f %-10.3f'
              % (nm[:20], np.nanmean(v[:, 0]), np.nanmean(v[:, 1]),
                 np.nanmean(v[:, 0]) + np.nanmean(v[:, 1])))


if __name__ == '__main__':
    main()
