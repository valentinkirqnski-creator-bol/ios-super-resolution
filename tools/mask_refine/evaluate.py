"""Evaluate the refinement against the Wronski mask it refines.

Reports, per the brief:

  PSNR / SSIM        reconstruction against the ground-truth-aligned merge
  ghost              gradient energy the output has beside a true edge that the
                     target does not -- the doubled-edge measure, which PSNR is
                     largely blind to because a ghost is small in total error
  retained           sum(M_final) / sum(M_wronski): how much acceptance survived
  latency            per-megapixel forward time, measured not estimated
  params, peak mem   model size and the activation high-water mark

Split by burst regime (clean / parallax / object+occlusion / rotation) and by
flow-error size, because an average over all of them hides the case that matters:
a mask can win overall and still be worse on exactly the motion it was built for.

    python evaluate.py --data data --ckpt mask_refine.pt
    python evaluate.py --data data --ckpt mask_refine.pt --ablate
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import losses as L
import model as M


def psnr(a, b):
    m = float(((a - b) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


def ssim(a, b, C1=1e-4, C2=9e-4):
    """Global-window SSIM on the luminance. Enough to rank masks; not a
    replacement for a windowed implementation if an absolute number is wanted."""
    x = a.mean(1, keepdim=True)
    y = b.mean(1, keepdim=True)
    k = 11
    mu_x = F.avg_pool2d(x, k, 1, k // 2)
    mu_y = F.avg_pool2d(y, k, 1, k // 2)
    xx = F.avg_pool2d(x * x, k, 1, k // 2) - mu_x ** 2
    yy = F.avg_pool2d(y * y, k, 1, k // 2) - mu_y ** 2
    xy = F.avg_pool2d(x * y, k, 1, k // 2) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * xy + C2)) / \
        ((mu_x ** 2 + mu_y ** 2 + C1) * (xx + yy + C2))
    return float(s.mean())


def ghost(out, gt):
    return float(L.ghost_loss(out, gt))


def regime_of(d):
    """The generator does not record the regime, so it is inferred from what
    the burst shows: a large flow span means a field one vector per tile cannot
    describe (parallax or an independently moving object), and otherwise the
    size of the flow error separates clean from drifting."""
    span = float(np.percentile(np.abs(np.diff(d['flow'], axis=1)).sum(-1), 95))
    fe = float(np.median(d['ferr']))
    if span > 2.0:
        return 'boundary'
    return 'clean' if fe < 1.0 else 'drift'


def run(net, bursts, idx):
    rows = []
    with torch.no_grad():
        for i in idx:
            feat, Rw, A_ref, B_ref, A, B, gt, ferr = bursts.batch(i)
            Mf, _ = net(feat, Rw)
            o_f = D.merge(A_ref, B_ref, A, B, Mf)
            o_w = D.merge(A_ref, B_ref, A, B, Rw)
            o_1 = D.merge(A_ref, B_ref, A, B, torch.ones_like(Rw))
            d = bursts.items[i]
            rows.append(dict(
                regime=regime_of(d), ferr=float(np.median(d['ferr'])),
                psnr_f=psnr(o_f, gt), psnr_w=psnr(o_w, gt), psnr_1=psnr(o_1, gt),
                ssim_f=ssim(o_f, gt), ssim_w=ssim(o_w, gt),
                ghost_f=ghost(o_f, gt), ghost_w=ghost(o_w, gt),
                retained=float(Mf.sum() / Rw.sum().clamp_min(1.0)),
                meanW=float(Rw.mean()), meanF=float(Mf.mean())))
    return rows


def table(rows, title):
    print('\n%s' % title)
    print('%-10s %4s %8s %8s %8s %8s %8s %8s %8s' %
          ('group', 'n', 'PSNR', 'Wronski', 'gain', 'SSIM', 'W.SSIM',
           'ghost', 'retained'))
    groups = {}
    for r in rows:
        groups.setdefault(r['regime'], []).append(r)
    groups['ALL'] = rows
    for g in ('clean', 'drift', 'boundary', 'ALL'):
        rs = groups.get(g)
        if not rs:
            continue
        f = lambda k: float(np.mean([r[k] for r in rs]))
        print('%-10s %4d %8.2f %8.2f %+8.2f %8.4f %8.4f %8.4f %8.3f'
              % (g, len(rs), f('psnr_f'), f('psnr_w'),
                 f('psnr_f') - f('psnr_w'), f('ssim_f'), f('ssim_w'),
                 f('ghost_f'), f('retained')))
    print('  (ghost: lower is better. Wronski ghost %.4f)'
          % float(np.mean([r['ghost_w'] for r in rows])))
    # A mask that only ever merged everything would be a different thing again.
    print('  merge-everything baseline PSNR %.2f'
          % float(np.mean([r['psnr_1'] for r in rows])))


def cost(net):
    print('\ncost')
    print('  parameters            %d' % net.n_params())
    print('  receptive field       %d mask px' % net.receptive_field())
    x = torch.randn(1, D.NUM_FEATURES, 1024, 1024)
    with torch.no_grad():
        net.gate(x)                       # warm
        t = time.time()
        for _ in range(3):
            net.gate(x)
        dt = (time.time() - t) / 3.0
    print('  forward 1024x1024     %.0f ms  -> %.0f ms / MP (CPU, 1 thread set)'
          % (dt * 1e3, dt * 1e3))
    act = 0
    h = []

    def hook(m, i, o):
        nonlocal act
        act = max(act, o.numel() * 4)
    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            h.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        net.gate(x)
    for k in h:
        k.remove()
    print('  peak activation       %.1f MB for a 1 MP plane' % (act / 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--ckpt', default='mask_refine.pt')
    ap.add_argument('--ablate', action='store_true')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    here = os.path.dirname(os.path.abspath(__file__))

    ck = torch.load(a.ckpt if os.path.isabs(a.ckpt)
                    else os.path.join(here, a.ckpt),
                    map_location='cpu', weights_only=False)
    net = M.MaskRefineNet(in_ch=ck['in_ch'], width=ck['width'])
    net.load_state_dict(ck['state_dict'])
    net.eval()
    print('checkpoint: step %d, score %+.3f' % (ck['step'], ck['score']))

    bursts = D.Bursts(a.data)
    idx = list(range(len(bursts)))
    table(run(net, bursts, idx), 'all bursts in %s' % a.data)
    cost(net)

    if a.ablate:
        print('\nablation: each feature zeroed, PSNR gain over Wronski')
        print('%-16s %8s %8s' % ('zeroed', 'gain dB', 'retained'))
        base = run(net, bursts, idx)
        bg = float(np.mean([r['psnr_f'] - r['psnr_w'] for r in base]))
        print('%-16s %+8.2f %8.3f' % ('(none)', bg,
                                      float(np.mean([r['retained'] for r in base]))))
        for feat in D.FEATURES:
            b2 = D.Bursts(a.data, drop=(feat,))
            rs = run(net, b2, idx)
            g = float(np.mean([r['psnr_f'] - r['psnr_w'] for r in rs]))
            print('%-16s %+8.2f %8.3f   (%+.2f vs full)'
                  % (feat, g, float(np.mean([r['retained'] for r in rs])),
                     g - bg))


if __name__ == '__main__':
    main()
