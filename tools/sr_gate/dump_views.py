"""Write the training data out as PNGs you can actually look at.

Everything the gate trains on is linear float in .npy stacks, which is
unreadable. This renders, per burst:

    target.png      the TARGET -- merge of every frame with the TRUE flow, no
                    noise, nothing rejected. The best the pipeline could ever
                    produce for this burst, and what the loss compares against.
    ref_only.png    merge with R = 0: the reference frame alone. Noisy, soft,
                    no ghosts possible.
    merge_all.png   merge with R = 1: every frame at full weight, using the
                    ESTIMATED (wrong) flow. This is where the ghosts are.
    merge_gate.png  merge with the shipped neural mask.
    merge_wronski.png   merge with the analytic mask + geometry rejection.
    mask_gate.png   the neural mask itself, frame 0, white = merge, black = reject.
    mask_wronski.png    the analytic mask, same frame.
    ghost_all.png   |merge_all - target|, amplified. Makes the misalignment the
                    mask is supposed to catch directly visible.
    ghost_gate.png  |merge_gate - target|, same amplification, so the two are
                    comparable side by side.

The merges are reconstructed from the packed A/B factors exactly as training
does, so what you see is what the loss saw -- not a re-render through a
different path.

Linear light is displayed with a plain sRGB-ish gamma and a per-image 99.5th
percentile normalisation. That is for looking at only; no tone curve from the
app is involved, so do not judge colour or contrast from these.

    python dump_views.py --split val --index 0 1 2 --out views
    python dump_views.py --split train --index 7 --amp 12
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate
import train as T

NAMES = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier', 4: 'parallax'}


def to_png(x, path, gamma=2.2, pct=99.5, norm=None):
    """(3, H, W) or (H, W) linear float -> 8-bit PNG."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 3:
        a = np.transpose(a, (1, 2, 0))
    s = norm if norm is not None else max(np.percentile(a, pct), 1e-6)
    a = np.clip(a / s, 0.0, 1.0) ** (1.0 / gamma)
    PILImage.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)
    return s


def mask_png(R, path):
    """Mask as greyscale at its own resolution, nearest-upscaled x2 so one mask
    pixel is visibly one block rather than being resampled by the viewer."""
    a = np.clip(np.asarray(R, dtype=np.float32), 0.0, 1.0)
    a = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
    PILImage.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)


def render_from_dng(a, net, outd):
    """Synthesise bursts from one photo and render them, bypassing the packed
    dataset entirely. The burst is built by the SAME code the training set uses
    (build.build_burst), so what comes out is what training would have seen had
    this photo been in the source pool."""
    import build
    import make_data as MD
    import srburst

    scene = MD.scene_cache(a.dng, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '.scene_cache'))
    if scene is None:
        print('could not load', a.dng)
        return
    art_net = None
    ap_ = a.art_ckpt if os.path.isabs(a.art_ckpt) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), a.art_ckpt)
    if a.art_ckpt and os.path.exists(ap_):
        art_net = gate.unet_from_checkpoint(
            torch.load(ap_, map_location='cpu', weights_only=False))
    base = os.path.splitext(os.path.basename(a.dng))[0]
    rng = np.random.default_rng(a.seed)
    for k in range(a.crops):
        for _ in range(24):
            c = MD.crop(scene, rng)
            if MD.interesting(c):
                break
        spec = srburst.BurstSpec(rng, regime=a.regime)
        spec.sigma_flow = float(a.sigma)
        if a.gain > 0:
            spec.noise_gain = float(a.gain)
        d = build.build_burst(c, spec, rng)
        tag = '%s_crop%d_%s_sig%.2f_gain%.1f' % (
            base, k, NAMES[a.regime], spec.sigma_flow, d['noise_gain'])
        dd = os.path.join(outd, tag)
        os.makedirs(dd, exist_ok=True)
        t = lambda k_: torch.from_numpy(
            np.ascontiguousarray(d[k_], dtype=np.float32)).unsqueeze(0)
        args = (t('A_ref'), t('B_ref'), t('A'), t('B'))
        gt = t('gt')[0]
        with torch.no_grad():
            Rg = T.run_gate(net, t('feat'))
            Rw = t('Rw')
            m_ref = gate.merge(*args, torch.zeros_like(Rg))[0]
            m_all = gate.merge(*args, torch.ones_like(Rg))[0]
            m_gat = gate.merge(*args, Rg)[0]
            m_wro = gate.merge(*args, Rw)[0]
        # The artifact net, if its checkpoint is present: predict the artifact
        # each frame would contribute, then turn it into a mask through the
        # policy. tau is the only thing that chooses how strict this is.
        Ra = None
        if art_net is not None:
            with torch.no_grad():
                import train_artifact as TA
                a_hat = torch.expm1(
                    TA.run_net(art_net, t('feat'), t('img')).clamp(-8, 8))
                Ra = TA.policy(a_hat, a.tau, a.art_beta)
                m_art = gate.merge(*args, Ra)[0]

        s = to_png(gt, os.path.join(dd, 'target.png'))
        outs = [('ref_only', m_ref), ('merge_all', m_all),
                ('merge_gate', m_gat), ('merge_wronski', m_wro)]
        ghosts = [('ghost_all', m_all), ('ghost_gate', m_gat),
                  ('ghost_wronski', m_wro)]
        if Ra is not None:
            outs.append(('merge_artifact', m_art))
            ghosts.append(('ghost_artifact', m_art))
        for nm, im in outs:
            to_png(im, os.path.join(dd, nm + '.png'), norm=s)
        for nm, im in ghosts:
            to_png((im - gt).abs() * a.amp, os.path.join(dd, nm + '.png'),
                   norm=s)
        mask_png(Rg[0, a.frame], os.path.join(dd, 'mask_gate.png'))
        mask_png(Rw[0, a.frame], os.path.join(dd, 'mask_wronski.png'))
        extra = ''
        if Ra is not None:
            mask_png(Ra[0, a.frame], os.path.join(dd, 'mask_artifact.png'))
            # The artifact field itself, in noise sigma, clipped at 4 so the
            # scale is readable: white = the merge would damage this pixel.
            mask_png((a_hat[0, a.frame] / 4.0).clamp(0, 1),
                     os.path.join(dd, 'predicted_artifact.png'))
            mask_png((t('art')[0, a.frame] / 4.0).clamp(0, 1),
                     os.path.join(dd, 'true_artifact.png'))
            extra = '  artifact %.3f (tau %.2f)' % (float(Ra.mean()), a.tau)
        print('%s  align=%s  meanR gate %.3f  wronski %.3f%s  -> %s'
              % (tag, d.get('align', '?'), float(Rg.mean()), float(Rw.mean()),
                 extra, dd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_se')
    ap.add_argument('--split', default='val')
    ap.add_argument('--index', type=int, nargs='+', default=[0])
    ap.add_argument('--out', default='views')
    ap.add_argument('--ckpt', default='sr_gate.pt')
    ap.add_argument('--amp', type=float, default=8.0,
                    help='how much to amplify the error images. 8 means an '
                         'error of 1/8 full scale shows as white.')
    ap.add_argument('--frame', type=int, default=0,
                    help='which comparison frame to show the masks for')
    ap.add_argument('--dng', default='',
                    help='synthesise bursts from THIS source photo instead of '
                         'reading the packed split. Works for any DNG, '
                         'including ones outside the tree the builder scans -- '
                         'which is how you look at a scene the dataset does '
                         'not currently contain.')
    ap.add_argument('--regime', type=int, default=1,
                    help='0 jitter, 1 rotation, 2 moving object, 3 outlier, '
                         '4 parallax')
    ap.add_argument('--sigma', type=float, default=0.9,
                    help='per-tile flow error, in raw pixels')
    ap.add_argument('--gain', type=float, default=0.0,
                    help='noise gain; 0 leaves the random draw alone')
    ap.add_argument('--crops', type=int, default=3,
                    help='how many random crops of that photo to render')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--art-ckpt', default='sr_gate_art.pt',
                    help='the artifact-predicting U-Net. Rendered alongside the '
                         'shipped mask and the analytic one when present.')
    ap.add_argument('--tau', type=float, default=0.5,
                    help='tolerated artifact in noise sigma -- the policy knob. '
                         'Lower = stricter.')
    ap.add_argument('--art-beta', type=float, default=0.5)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    outd = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    net = gate.from_checkpoint(torch.load(
        a.ckpt if os.path.isabs(a.ckpt) else os.path.join(here, a.ckpt),
        map_location='cpu', weights_only=False))

    if a.dng:
        render_from_dng(a, net, outd)
        return

    sp = T.Split(root, a.split)
    for i in a.index:
        if i >= sp.n:
            print('index %d out of range (%d bursts)' % (i, sp.n))
            continue
        d = sp.batch([i])
        reg, sig = int(d['meta'][0][0]), float(d['meta'][0][1])
        ng = float(d['meta'][0][2])
        tag = '%s_%02d_%s_sig%.2f' % (a.split, i, NAMES[reg], sig)
        dd = os.path.join(outd, tag)
        os.makedirs(dd, exist_ok=True)

        args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
        with torch.no_grad():
            Rg = T.run_gate(net, d['feat'])
            Rw = d['Rw']
            gt = d['gt'][0]
            m_ref = gate.merge(*args, torch.zeros_like(Rg))[0]
            m_all = gate.merge(*args, torch.ones_like(Rg))[0]
            m_gat = gate.merge(*args, Rg)[0]
            m_wro = gate.merge(*args, Rw)[0]

        # One normalisation for every picture of this burst, so they are
        # directly comparable rather than each being auto-levelled.
        s = to_png(gt, os.path.join(dd, 'target.png'))
        for nm, im in (('ref_only', m_ref), ('merge_all', m_all),
                       ('merge_gate', m_gat), ('merge_wronski', m_wro)):
            to_png(im, os.path.join(dd, nm + '.png'), norm=s)
        # Errors against the target, shared amplification.
        for nm, im in (('ghost_all', m_all), ('ghost_gate', m_gat),
                       ('ghost_wronski', m_wro), ('ghost_ref', m_ref)):
            to_png((im - gt).abs() * a.amp, os.path.join(dd, nm + '.png'),
                   norm=s)
        mask_png(Rg[0, a.frame], os.path.join(dd, 'mask_gate.png'))
        mask_png(Rw[0, a.frame], os.path.join(dd, 'mask_wronski.png'))

        print('%s  noise_gain %.1f  meanR gate %.3f  wronski %.3f  -> %s'
              % (tag, ng, float(Rg.mean()), float(Rw.mean()), dd))

    print('\nnorm is the 99.5th percentile of each burst\'s target; the error '
          'images are amplified %gx.' % a.amp)


if __name__ == '__main__':
    main()
