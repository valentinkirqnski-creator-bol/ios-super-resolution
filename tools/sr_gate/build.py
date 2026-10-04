"""Build one burst into everything training needs, and (as __main__) report the
reference points per regime so the dataset can be shown to contain a real
rejection signal before anything is trained.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srsim
import srmerge
import srburst

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))


_CURVES = {}


def sqrt_curves(cfg, n_patches=4000):
    """Per-channel sqrt-domain curves, cached: they depend only on the noise
    gain, and the Monte Carlo is 1001 bins x 4000 patches x 9 samples per
    channel -- far too slow to repeat for every burst."""
    key = tuple(round(v, 12) for v in cfg.alpha_ch + cfg.beta_ch)
    hit = _CURVES.get(key)
    if hit is None:
        std, diff = [], []
        for c in range(3):
            sc_, dc_ = srsim.build_noise_curves_sqrt(cfg.alpha_ch[c],
                                                     cfg.beta_ch[c],
                                                     n_patches=n_patches)
            std.append(sc_)
            diff.append(dc_)
        hit = (std, diff)
        _CURVES[key] = hit
    return hit


def psnr(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


def build_burst(scene, spec, rng, tile_size=16):
    """-> dict(feat, Rw, A, B, A_ref, B_ref, gt, meta)

    feat  (N-1, NUM_FEATURES, h, w)   the gate input, guide/raw lattice
    Rw    (N-1, h, w)                 the shipping analytic mask, for reference
    A, B  (N-1, 3, 2h, 2w)            the merge, factored so that
    A_ref, B_ref (3, 2h, 2w)            out = (A_ref + sum A_n R_n)/(B_ref + ...)
    gt    (3, 2h, 2w)                 noise-free frames, true flow, R == 1
    """
    b = srburst.synth_burst(scene, spec, rng, tile_size=tile_size)
    h, w = b['h'], b['w']
    N = len(b['raws'])

    cfg = srsim.Cfg(noise_gain=spec.noise_gain, tile_size=tile_size)
    cfg.use_decimated_guide()
    std_c, diff_c = sqrt_curves(cfg)
    cfg.tune_snr(b['raws'][0], std_c[1])
    # The synthesiser laid the flow out on a fixed tile grid, so keep the mask
    # on that grid rather than letting the SNR tune move it underneath.
    cfg.tile_size = tile_size

    # The guide is HALF resolution and three channels, so the mask lattice is
    # h/2 x w/2 and the loss grid is subsampled to match what that can express.
    ST = 2
    gh, gw = h // 2, w // 2
    Hs, Ws = (h * 2) // ST, (w * 2) // ST
    ref_m, ref_v = srsim.local_stats_3x3(srsim.compute_guide_decimate3(b['raws'][0]))
    covs_ref = srmerge.estimate_kernels(b['raws'][0], cfg)
    A_ref, B_ref = srmerge.accumulate_ref_ab(b['raws'][0], covs_ref, cfg, ST)

    A = np.zeros((N - 1, 3, Hs, Ws), np.float32)
    B = np.zeros_like(A)
    Ag = np.zeros_like(A)
    Bg = np.zeros_like(A)
    # Only the shipped channels are stored. srsim can build 12; channels 8-11
    # measured neutral and core/sr_gate_shared.h declares 8.
    # 9: channels 0-7 plus the per-channel max d^2/sigma^2. srsim can build 13;
    # 9-12 measured neutral and are not stored.
    NF = 9
    feat = np.zeros((N - 1, NF, gh, gw), np.float32)
    Rw = np.zeros((N - 1, gh, gw), np.float32)
    # How wrong the flow the merge ACTUALLY used is, per mask pixel, in raw
    # pixels. The synthesiser imposed the motion, so this is exact rather than
    # estimated -- which is what makes it possible to ask whether a rejection
    # was deserved. A large residual with a small ferr is aliasing the merge
    # wants; the same residual with a large ferr is damage. No statistic
    # available on a real burst can separate those two, and every question about
    # over-rejection is really a question about this quantity.
    ferr = np.zeros((N - 1, gh, gw), np.float32)

    for n in range(1, N):
        covs = srmerge.estimate_kernels(b['raws'][n], cfg)
        fx, fy = srmerge.tile_flow_at_output(b['flows'][n], h, w, cfg.tile_size,
                                             ST)
        A[n - 1], B[n - 1] = srmerge.accumulate_comp_ab(b['raws'][n], fx, fy,
                                                        covs, cfg, ST)

        covs_c = srmerge.estimate_kernels(b['raws_clean'][n], cfg)
        tfx, tfy = b['true_flow'][n]
        Ag[n - 1], Bg[n - 1] = srmerge.accumulate_comp_ab(
            b['raws_clean'][n], tfx[::ST, ::ST], tfy[::ST, ::ST], covs_c, cfg, ST)
        # Both are the per-output-pixel flow at stride ST, so they subtract
        # directly; [::2, ::2] then drops it onto the half-resolution mask
        # lattice the gate and Rw live on.
        dfx = fx - tfx[::ST, ::ST]
        dfy = fy - tfy[::ST, ::ST]
        ferr[n - 1] = np.sqrt(dfx * dfx + dfy * dfy)[::2, ::2]

        gm, gv = srsim.local_stats_3x3(
            srsim.compute_guide_decimate3(b['raws'][n]))
        d_sq, sig_sq, comps = srsim.compute_d_sigma(ref_m, ref_v, gm, b['flows'][n],
                                                    cfg, std_c, diff_c)
        feat[n - 1] = srsim.build_features(d_sq, sig_sq, ref_m, ref_v,
                                           b['flows'][n], cfg, comps)[:NF]
        Rw[n - 1] = srsim.wronski_robustness(d_sq, sig_sq, b['flows'][n], ref_m, cfg)

    covs_rc = srmerge.estimate_kernels(b['raws_clean'][0], cfg)
    A_rc, B_rc = srmerge.accumulate_ref_ab(b['raws_clean'][0], covs_rc, cfg, ST)
    ones_out = np.ones((N - 1, Hs, Ws), np.float32)
    gt = srmerge.merge_from_ab(A_rc, B_rc, Ag, Bg, ones_out)

    # An edge map on the ground truth, used to weight the loss and to report
    # edge MSE separately -- ghosting and lost detail both live on edges, and a
    # plain mean over a mostly flat image hides them.
    g = gt.mean(axis=0)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    gy[1:-1, :] = 0.5 * (g[2:, :] - g[:-2, :])
    edge = np.sqrt(gx * gx + gy * gy)

    return dict(feat=feat, Rw=Rw, A=A, B=B, A_ref=A_ref, B_ref=B_ref, gt=gt,
                edge=edge.astype(np.float32), ferr=ferr, h=h, w=w,
                stride=ST, guide_scale=2,
                tile_size=tile_size, regime=b['regime'],
                sigma_flow=b['sigma_flow'], noise_gain=b['noise_gain'],
                theta=b.get('theta', 0.0))


def merged(d, R):
    """R on the guide lattice (N-1, h/2, w/2) -> merged RGB output."""
    n = R.shape[0]
    Rout = np.stack([srmerge.sample_r_at_output(R[i], d['h'], d['w'],
                                                d.get('stride', 2),
                                                d.get('guide_scale', 2))
                     for i in range(n)], axis=0)
    return srmerge.merge_from_ab(d['A_ref'], d['B_ref'], d['A'], d['B'], Rout)


def main():
    dngs, _ = srburst.find_dngs(ROOT)
    scene_full = srburst.load_scene(dngs[0])
    Hs = Ws = 384
    y0 = (scene_full.shape[0] - Hs) // 2
    x0 = (scene_full.shape[1] - Ws) // 2
    scene = np.ascontiguousarray(scene_full[y0:y0 + Hs, x0:x0 + Ws])

    names = {0: 'jitter', 1: 'rot/trans-only', 2: 'moving object', 3: 'outliers'}
    print('%-15s %-8s %-8s %-8s %-8s %-8s %-6s' %
          ('regime', 'sig_f', 'Wronski', 'R=1', 'R=0 ref', 'best', 'meanRw'))
    for regime in (0, 1, 2, 3):
        for trial in range(2):
            rng = np.random.default_rng(100 + regime * 10 + trial)
            spec = srburst.BurstSpec(rng, regime=regime)
            d = build_burst(scene, spec, rng)
            ones = np.ones_like(d['Rw'])
            pw = psnr(merged(d, d['Rw']), d['gt'])
            p1 = psnr(merged(d, ones), d['gt'])
            p0 = psnr(merged(d, np.zeros_like(ones)), d['gt'])
            best = max(pw, p1, p0)
            tag = 'Wronski' if best == pw else ('R=1' if best == p1 else 'R=0')
            print('%-15s %-8.2f %-8.2f %-8.2f %-8.2f %-8s %-6.3f' %
                  (names[regime], d['sigma_flow'], pw, p1, p0, tag, d['Rw'].mean()))


if __name__ == '__main__':
    main()
