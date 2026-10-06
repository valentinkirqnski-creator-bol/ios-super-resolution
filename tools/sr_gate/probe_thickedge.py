"""Do the sqrt guide and Eq. 9 explain the thick edges? Measured, not argued.

    python probe_thickedge.py --dng C:/Users/valen/Downloads/APC_1186.dng

Wronski's reply says the aperture problem is handled and that robustness was
enough to keep the other misalignment problems in check, so a thick edge in a
pipeline that implements the same equations points at an implementation
difference rather than a limit of the method. Two candidates survive scrutiny:

  sqrt guide   the paper computes sigma_ms and d_ms on a LINEAR half-resolution
               RGB guide and handles brightness dependence entirely through the
               Monte Carlo sigma_md / d_md curves. Computing the statistics in
               sqrt space is a variance-stabilising transform doing part of that
               job by another route, so d^2/sigma^2 is not the quantity Figure
               10 is calibrated against and the supplement's s/t/Mth no longer
               mean the same thing. It also survives switching the noise model
               off, because the transform is in the GUIDE and not in the
               correction.

  Eq. 9       the 5x5 minimum, whose stated purpose in the paper is exactly
               "misalignment in regions with high signal variance (like an edge
               on top of another one)" -- a doubled edge, in those words.

What is measured is EDGE THICKNESS, not PSNR. A thin doubled edge is a tiny
fraction of the pixels and moves a mean almost not at all, which is why every
PSNR-shaped number in this project has been blind to it. Instead, for pixels on
a strong ground-truth edge, this reports how much of the ground truth's own
gradient magnitude the merge retains: a thickened or doubled edge spreads the
same contrast over more pixels, so its peak gradient falls.

    edge sharpness = mean |grad(merged)| / mean |grad(gt)|   on the edge band

1.0 means the merge reproduces the reference's edges exactly. Lower is softer or
thicker. R=1 (merge everything) is the floor to beat, since it has every
misalignment in it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build as bld
import make_data
import srburst
import srmerge
import srsim


def grad_mag(img):
    g = img.mean(axis=0) if img.ndim == 3 else img
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    gy[1:-1, :] = 0.5 * (g[2:, :] - g[:-2, :])
    return np.sqrt(gx * gx + gy * gy)


def mask_for(b, cfg, std_c, diff_c, ref_m, ref_v, n, eq9):
    gm, _ = srsim.local_stats_3x3(srsim.compute_guide_decimate3(b['raws'][n]))
    d_sq, sig_sq, _ = srsim.compute_d_sigma(ref_m, ref_v, gm, b['flows'][n],
                                            cfg, std_c, diff_c)
    r = srsim.wronski_robustness(d_sq, sig_sq, b['flows'][n], ref_m, cfg)
    if not eq9:
        # Undo Eq. 9 by recomputing without it: wronski_robustness applies the
        # 5x5 minimum as its last step, so the no-Eq.9 mask is everything up to
        # that point. Reproduced here rather than adding a flag to the port,
        # which would change the function the C++ is checked against.
        S, _ = srsim.compute_s(b['flows'][n], srsim.R_MT, srsim.R_S1, srsim.R_S2)
        ts = cfg.tile_size
        ty, tx = srsim.tile_index_grids(d_sq.shape[0], d_sq.shape[1], ts, 3)
        ty = np.clip(ty, 0, b['flows'][n].shape[0] - 1)
        tx = np.clip(tx, 0, b['flows'][n].shape[1] - 1)
        s = S[ty, tx]
        with np.errstate(over='ignore'):
            r = s * np.exp(-np.minimum(d_sq.astype(np.float64) /
                                       np.maximum(sig_sq, 1e-20), 60.0)) - srsim.R_T
        r = np.clip(r, 0.0, 1.0).astype(np.float32)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dng', required=True)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--hr', type=int, default=512)
    ap.add_argument('--seed', type=int, default=515)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    full = srburst.load_scene(a.dng)
    if full is None:
        print('not real Bayer:', a.dng)
        return 1
    print('%s  %d bursts, hr=%d' % (os.path.basename(a.dng), a.n, a.hr))
    print('small misalignment at edges is the case under test, so sigma_flow is')
    print('held in the band that produces a visibly shifted edge.')

    acc = {}
    for i in range(a.n):
        rng = np.random.default_rng(a.seed + i)
        for _ in range(40):
            c = make_data.crop(full, rng, a.hr)
            if make_data.interesting(c) or make_data.textured(c):
                break
        spec = srburst.BurstSpec(rng, regime=0, level=2)
        spec.sigma_flow = float(np.exp(rng.uniform(np.log(0.3), np.log(0.9))))
        b = srburst.synth_burst(c, spec, rng, tile_size=16)

        h, w = b['h'], b['w']
        ST, N = 2, len(b['raws'])
        for sqrt_on in (True, False):
            cfg = srsim.Cfg(noise_gain=spec.noise_gain, tile_size=16)
            cfg.use_decimated_guide()
            cfg.guide_sqrt = sqrt_on
            std_c, diff_c = bld.sqrt_curves(cfg) if sqrt_on else bld.linear_curves(cfg)
            ref_m, ref_v = srsim.local_stats_3x3(
                srsim.compute_guide_decimate3(b['raws'][0]))
            covs_ref = srmerge.estimate_kernels(b['raws'][0], cfg)
            A_ref, B_ref = srmerge.accumulate_ref_ab(b['raws'][0], covs_ref, cfg, ST)
            Hs, Ws = (h * 2) // ST, (w * 2) // ST
            A = np.zeros((N - 1, 3, Hs, Ws), np.float32); B = np.zeros_like(A)
            Ag = np.zeros_like(A); Bg = np.zeros_like(A)
            for n in range(1, N):
                covs = srmerge.estimate_kernels(b['raws'][n], cfg)
                fx, fy = srmerge.tile_flow_at_output(b['flows'][n], h, w, 16, ST)
                A[n-1], B[n-1] = srmerge.accumulate_comp_ab(b['raws'][n], fx, fy, covs, cfg, ST)
                covs_c = srmerge.estimate_kernels(b['raws_clean'][n], cfg)
                tfx, tfy = b['true_flow'][n]
                Ag[n-1], Bg[n-1] = srmerge.accumulate_comp_ab(
                    b['raws_clean'][n], tfx[::ST, ::ST], tfy[::ST, ::ST], covs_c, cfg, ST)
            covs_rc = srmerge.estimate_kernels(b['raws_clean'][0], cfg)
            A_rc, B_rc = srmerge.accumulate_ref_ab(b['raws_clean'][0], covs_rc, cfg, ST)
            gt = srmerge.merge_from_ab(A_rc, B_rc, Ag, Bg,
                                       np.ones((N-1, Hs, Ws), np.float32))
            ggt = grad_mag(gt)
            band = ggt > np.quantile(ggt, 0.92)

            for eq9 in (True, False):
                R = np.stack([mask_for(b, cfg, std_c, diff_c, ref_m, ref_v, n, eq9)
                              for n in range(1, N)])
                gh, gw = R.shape[1], R.shape[2]
                yi = np.minimum((np.arange(Hs) * gh) // Hs, gh - 1)
                xi = np.minimum((np.arange(Ws) * gw) // Ws, gw - 1)
                Rout = R[:, yi][:, :, xi].astype(np.float32)
                out = srmerge.merge_from_ab(A_ref, B_ref, A, B, Rout)
                key = ('sqrt' if sqrt_on else 'linear') + (' +Eq9' if eq9 else ' -Eq9')
                v = acc.setdefault(key, np.zeros(4))
                v[0] += grad_mag(out)[band].mean() / max(ggt[band].mean(), 1e-9)
                v[1] += float(R.mean())
                v[2] += 1
                v[3] += 10*np.log10(1.0/max(float(np.mean((out-gt)**2)), 1e-12))
            if sqrt_on:
                one = srmerge.merge_from_ab(A_ref, B_ref, A, B,
                                            np.ones((N-1, Hs, Ws), np.float32))
                v = acc.setdefault('R=1 (merge all)', np.zeros(4))
                v[0] += grad_mag(one)[band].mean() / max(ggt[band].mean(), 1e-9)
                v[1] += 1.0
                v[2] += 1
                v[3] += 10*np.log10(1.0/max(float(np.mean((one-gt)**2)), 1e-12))

    print()
    print('%-20s %-16s %-10s %-10s' % ('variant', 'edge sharpness', 'mean R', 'PSNR dB'))
    for k in ('R=1 (merge all)', 'sqrt +Eq9', 'sqrt -Eq9', 'linear +Eq9', 'linear -Eq9'):
        v = acc.get(k)
        if v is None or v[2] == 0:
            continue
        print('%-20s %-16.4f %-10.3f %-10.2f' % (k, v[0]/v[2], v[1]/v[2], v[3]/v[2]))
    print()
    print('edge sharpness 1.0 = merge reproduces the reference edges exactly.')
    print('Higher is sharper. "sqrt +Eq9" is what ships today.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
