"""Does the mask reject only what it should? Measured against the TRUE flow.

    python eval_goals.py --ckpt sr_gate.pt --n 24

response_curve.py bins real bursts by a phase-correlation estimate of motion,
which is the best available there but is itself partly a texture detector -- and
on a real burst there is no way to know whether a large residual was damage or
aliasing. That is the one question every complaint about this mask comes down
to, so this harness asks it on SYNTHETIC bursts, where the motion was imposed
and `ferr` (build_burst) is the exact per-pixel error of the flow the merge
actually used.

Four numbers, one per thing the mask is asked to do:

  static      near-zero camera motion, correctly estimated. A mask doing its job
              is ~1 here. Anything well below is over-rejection with no cause,
              and it costs SR for nothing.
  rotation    real camera rotation, also correctly estimated (the per-tile flow
              is read off the true motion at tile centres). Rotation makes the
              flow field vary tile to tile legitimately, so this separates "the
              flow disagrees" from "the frame is unmergeable".
  misaligned  the pixels whose flow really is wrong. Must stay LOW, or the
              rebalance above was bought by giving up rejection.
  edge x real a 2x2 split: near a strong edge or not, flow really wrong or not.
              This is the "more precise around object edges, but only when it is
              a real misalignment" ask, and it is the only cell layout that can
              show it -- the wanted shape is low in (edge, real) and high in
              (edge, aligned), because an aliased edge is what SR feeds on.

Also reports merged-image PSNR overall and restricted to an edge band, against
Wronski and against R=1, so a gain in selectivity that costs image quality
cannot hide.
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

# Raw pixels of flow error above which a rejection is deserved.
#
# 1.6 px, because that is where this project MEASURED rejection to stop being
# net harmful on the merged image. Below it the offset is mostly signal the SR
# merge uses; above it the frame is placing detail in the wrong place. The
# threshold is the whole reason this harness can tell over-rejection from
# correct rejection, so it is a measured number and not a round one.
FERR_REAL = 1.6
# Edge strength above this quantile of the frame counts as "near an edge".
EDGE_Q = 0.80
# Below this quantile of edge strength the ground truth is flat, and anything
# the merge puts there is an artifact rather than a reconstruction error.
SMOOTH_Q = 0.50


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 0 else float(10.0 * np.log10(1.0 / mse))


def masks_for(d, net, n_in):
    """-> dict of name -> (N-1, gh, gw) mask."""
    out = {'R=1': np.ones_like(d['Rw']), 'Wronski': d['Rw']}
    if net is not None:
        with torch.no_grad():
            out['gate'] = net(
                torch.from_numpy(d['feat'][:, :n_in])).numpy()[:, 0]
    return out


def merged_psnr(d, mask):
    """PSNR of the merge: overall, on the edge band, and on the SMOOTH band.

    The smooth band is the one that matters for ghosting and it was missing.
    `edge` comes from the ground truth, so every edge-keyed figure asks "is the
    structure that should be here right?" -- and a ghost is structure that
    should NOT be here, lying in a region the GT calls flat. A wire smeared
    across open sky is invisible to the edge band and is a tiny area fraction of
    the whole frame, so the overall mean hides it too. That is how a model which
    ghosts visibly scored only 0.04 worse on the edge gap.
    """
    out = bld.srmerge.merge_from_ab(d['A_ref'], d['B_ref'], d['A'], d['B'],
                                    mask_to_output(d, mask))
    gt = d['gt']
    e = d['edge']
    band = e > np.quantile(e, EDGE_Q)
    smooth = e < np.quantile(e, SMOOTH_Q)
    b3 = np.broadcast_to(band, gt.shape)
    s3 = np.broadcast_to(smooth, gt.shape)
    mse_e = float(np.mean((out[b3] - gt[b3]) ** 2))
    mse_s = float(np.mean((out[s3] - gt[s3]) ** 2))
    return (psnr(out, gt),
            99.0 if mse_e <= 0 else float(10.0 * np.log10(1.0 / mse_e)),
            99.0 if mse_s <= 0 else float(10.0 * np.log10(1.0 / mse_s)))


def mask_to_output(d, mask):
    """The mask lives on the half-resolution guide; the merge samples it per
    output pixel. Nearest, exactly as accumulate_comp does on the device."""
    n, gh, gw = mask.shape
    Hs, Ws = d['A'].shape[2], d['A'].shape[3]
    yi = np.minimum((np.arange(Hs) * gh) // Hs, gh - 1)
    xi = np.minimum((np.arange(Ws) * gw) // Ws, gw - 1)
    return mask[:, yi][:, :, xi].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=[])
    ap.add_argument('--n', type=int, default=24,
                    help='bursts per case')
    ap.add_argument('--hr', type=int, default=320)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7700)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, '.scene_cache')

    nets = [(None, 'analytic only', 8)]
    for c in a.ckpt:
        ck = torch.load(os.path.join(here, c), map_location='cpu',
                        weights_only=True)
        nets.append((gate.from_checkpoint(ck), c, ck['in_ch']))

    dngs, n_in_tree = srburst.find_dngs(bld.ROOT)
    # The held-out scenes only, drawn exactly as make_data.py draws them, so
    # nothing here was trained on.
    order = list(np.random.default_rng(12345).permutation(n_in_tree))
    val_files = [dngs[i] for i in order[:6]]
    print('%d held-out scenes' % len(val_files))

    # Each case pins the motion it is about and leaves the rest of the
    # synthesiser alone. regime 0 throughout for static/rotation: a moving
    # object would put real damage in the frame and the question here is what
    # the mask does when there is NONE.
    def spec_static(rng):
        s = srburst.BurstSpec(rng, regime=0)
        s.trans = float(np.exp(rng.uniform(np.log(0.3), np.log(4.0))))
        s.theta = 0.0
        s.sigma_flow = float(np.exp(rng.uniform(np.log(0.02), np.log(0.25))))
        return s

    def spec_rotation(rng):
        s = srburst.BurstSpec(rng, regime=0)
        s.trans = float(np.exp(rng.uniform(np.log(0.3), np.log(8.0))))
        s.theta = float(rng.uniform(0.004, 0.020))
        s.sigma_flow = float(np.exp(rng.uniform(np.log(0.02), np.log(0.25))))
        return s

    def spec_object(rng):
        return srburst.BurstSpec(rng, regime=2)

    cases = [('static', spec_static), ('rotation', spec_rotation),
             ('object', spec_object)]

    # name -> case -> list of (sum R, count) and the psnr pairs
    acc = {}
    cells = {}
    for case, mk in cases:
        for i in range(a.n):
            rng = np.random.default_rng(a.seed + 1000 * len(case) + i)
            full = make_data.scene_cache(val_files[i % len(val_files)], cache)
            if full is None:
                continue
            # Same crop acceptance make_data uses, so a flat or blown patch
            # cannot dominate an average that is reported per case.
            for _ in range(24):
                scene = make_data.crop(full, rng, a.hr)
                if make_data.interesting(scene):
                    break
            else:
                continue
            d = bld.build_burst(scene, mk(rng), rng)
            ferr = d['ferr']
            e = d['edge']
            # The edge map is at output resolution; reduce to the mask lattice
            # by taking the max over each mask pixel's block, so a thin edge is
            # not averaged away.
            gh, gw = ferr.shape[1], ferr.shape[2]
            eb = e.reshape(gh, e.shape[0] // gh, gw, e.shape[1] // gw)
            eb = eb.max(axis=(1, 3))
            near_edge = eb > np.quantile(eb, EDGE_Q)
            real = ferr > FERR_REAL
            for net, name, n_in in nets:
                for mname, m in masks_for(d, net, n_in).items():
                    if net is None and mname == 'gate':
                        continue
                    key = (name if mname == 'gate' else mname)
                    if net is not None and mname != 'gate':
                        continue
                    s = acc.setdefault(key, {}).setdefault(case, np.zeros(2))
                    s[0] += float(m.sum())
                    s[1] += m.size
                    p, pe, ps = merged_psnr(d, m)
                    q = acc[key].setdefault(case + ':psnr', np.zeros(4))
                    q[0] += p
                    q[1] += pe
                    q[2] += 1
                    q[3] += ps
                    nb = np.broadcast_to(near_edge, m.shape)
                    for cell, sel in (('edge/real', nb & real),
                                      ('edge/aligned', nb & ~real),
                                      ('flat/real', ~nb & real),
                                      ('flat/aligned', ~nb & ~real)):
                        c = cells.setdefault(key, {}).setdefault(cell,
                                                                 np.zeros(2))
                        if sel.any():
                            c[0] += float(m[sel].sum())
                            c[1] += int(sel.sum())

    names = list(acc.keys())
    print()
    print('mean R by case (static/rotation want HIGH, object wants LOW where wrong)')
    hdr = '%-14s' % 'mask'
    for case, _ in cases:
        hdr += '%-12s' % case
    print(hdr)
    for nm in names:
        row = '%-14s' % nm[:13]
        for case, _ in cases:
            v = acc[nm].get(case)
            row += '%-12s' % ('%.3f' % (v[0] / max(v[1], 1)) if v is not None
                              else '-')
        print(row)

    print()
    print('merged PSNR, dB  (overall / edge band / SMOOTH band)')
    hdr = '%-14s' % 'mask'
    for case, _ in cases:
        hdr += '%-24s' % case
    print(hdr)
    for nm in names:
        row = '%-14s' % nm[:13]
        for case, _ in cases:
            q = acc[nm].get(case + ':psnr')
            row += '%-24s' % ('%.2f / %.2f / %.2f'
                              % (q[0] / q[2], q[1] / q[2], q[3] / q[2])
                              if q is not None and q[2] else '-')
        print(row)

    print()
    print('mean R by edge x real-misalignment  (ferr > %.1f px = real)' % FERR_REAL)
    order_cells = ['edge/real', 'edge/aligned', 'flat/real', 'flat/aligned']
    hdr = '%-14s' % 'mask'
    for c in order_cells:
        hdr += '%-15s' % c
    print(hdr)
    for nm in names:
        row = '%-14s' % nm[:13]
        for c in order_cells:
            v = cells.get(nm, {}).get(c)
            row += '%-15s' % ('%.3f' % (v[0] / max(v[1], 1))
                              if v is not None and v[1] else '-')
        print(row)
    print()
    print('wanted: edge/real LOW, edge/aligned HIGH. The gap between those two')
    print('columns IS the edge precision asked for; a mask that lowers both is')
    print('only being stricter, not more precise.')


if __name__ == '__main__':
    main()
