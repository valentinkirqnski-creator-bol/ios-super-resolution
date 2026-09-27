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
    std_c, diff_c = srsim.noise_curves_closed_form(cfg.alpha_rob, cfg.beta_rob)
    cfg.tune_snr(b['raws'][0], std_c)
    # The synthesiser laid the flow out on a fixed tile grid, so keep the mask
    # on that grid rather than letting the SNR tune move it underneath.
    cfg.tile_size = tile_size

    ref_m, ref_v = srsim.local_stats_3x3(srsim.compute_grey_fft(b['raws'][0]))
    covs_ref = srmerge.estimate_kernels(b['raws'][0], cfg)
    A_ref, B_ref = srmerge.accumulate_ref_ab(b['raws'][0], covs_ref, cfg)

    A = np.zeros((N - 1, 3, h * 2, w * 2), np.float32)
    B = np.zeros_like(A)
    Ag = np.zeros_like(A)
    Bg = np.zeros_like(A)
    feat = np.zeros((N - 1, srsim.NUM_FEATURES, h, w), np.float32)
    Rw = np.zeros((N - 1, h, w), np.float32)

    for n in range(1, N):
        covs = srmerge.estimate_kernels(b['raws'][n], cfg)
        fx, fy = srmerge.tile_flow_at_output(b['flows'][n], h, w, cfg.tile_size)
        A[n - 1], B[n - 1] = srmerge.accumulate_comp_ab(b['raws'][n], fx, fy, covs, cfg)

        covs_c = srmerge.estimate_kernels(b['raws_clean'][n], cfg)
        tfx, tfy = b['true_flow'][n]
        Ag[n - 1], Bg[n - 1] = srmerge.accumulate_comp_ab(b['raws_clean'][n], tfx,
                                                          tfy, covs_c, cfg)

        gm, _ = srsim.local_stats_3x3(srsim.compute_grey_fft(b['raws'][n]))
        d_sq, sig_sq = srsim.compute_d_sigma(ref_m, ref_v, gm, b['flows'][n], cfg,
                                             std_c, diff_c)
        feat[n - 1] = srsim.build_features(d_sq, sig_sq, ref_m, ref_v,
                                           b['flows'][n], cfg)
        Rw[n - 1] = srsim.wronski_robustness(d_sq, sig_sq, b['flows'][n], ref_m, cfg)

    covs_rc = srmerge.estimate_kernels(b['raws_clean'][0], cfg)
    A_rc, B_rc = srmerge.accumulate_ref_ab(b['raws_clean'][0], covs_rc, cfg)
    ones_out = np.ones((N - 1, h * 2, w * 2), np.float32)
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
                edge=edge.astype(np.float32), h=h, w=w,
                tile_size=tile_size, regime=b['regime'],
                sigma_flow=b['sigma_flow'], noise_gain=b['noise_gain'])


def merged(d, R):
    """R at guide/raw resolution (N-1, h, w) -> merged RGB output."""
    n = R.shape[0]
    Rout = np.stack([srmerge.sample_r_at_output(R[i], d['h'], d['w'])
                     for i in range(n)], axis=0)
    return srmerge.merge_from_ab(d['A_ref'], d['B_ref'], d['A'], d['B'], Rout)


def main():
    dngs = srburst.find_dngs(ROOT)
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
