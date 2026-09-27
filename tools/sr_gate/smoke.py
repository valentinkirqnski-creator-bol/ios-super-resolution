"""Smoke test: one synthetic burst end to end, Wronski vs an all-ones mask."""
import os
import sys
import time

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


def main():
    dngs = srburst.find_dngs(ROOT)
    print('DNGs found:', len(dngs))
    for d in dngs[:3]:
        print('  ', os.path.relpath(d, ROOT))
    t = time.time()
    scene_full = srburst.load_scene(dngs[0])
    print('scene', None if scene_full is None else scene_full.shape,
          'in %.1fs' % (time.time() - t))
    assert scene_full is not None

    # a 512x512 HR crop -> 256x256 raw -> 512x512 output
    Hs = Ws = 512
    y0 = (scene_full.shape[0] - Hs) // 2
    x0 = (scene_full.shape[1] - Ws) // 2
    scene = np.ascontiguousarray(scene_full[y0:y0 + Hs, x0:x0 + Ws])
    print('scene crop', scene.shape, 'mean %.4f' % scene.mean())

    rng = np.random.default_rng(0)
    spec = srburst.BurstSpec(rng, regime=0)
    spec.sigma_flow = 0.5
    spec.noise_gain = 8.0
    print('spec: sigma_flow %.2f noise_gain %.1f theta %.4f trans %.2f'
          % (spec.sigma_flow, spec.noise_gain, spec.theta, spec.trans))

    t = time.time()
    b = srburst.synth_burst(scene, spec, rng, tile_size=16)
    print('synth %.2fs' % (time.time() - t), 'raw', b['raws'][0].shape)

    cfg = srsim.Cfg(noise_gain=spec.noise_gain, tile_size=16)
    std_c, diff_c = srsim.noise_curves_closed_form(cfg.alpha_rob, cfg.beta_rob)
    snr = cfg.tune_snr(b['raws'][0], std_c)
    cfg.tile_size = 16   # pin for the smoke test
    print('snr %.1f k_detail %.3f k_denoise %.3f D_th %.3f D_tr %.3f'
          % (snr, cfg.k_detail, cfg.k_denoise, cfg.D_th, cfg.D_tr))

    h, w = b['h'], b['w']
    t = time.time()
    guide_ref = srsim.compute_grey_fft(b['raws'][0])
    ref_m, ref_v = srsim.local_stats_3x3(guide_ref)
    print('guide %.2fs  ref mean %.4f  ref var %.3g'
          % (time.time() - t, ref_m.mean(), ref_v.mean()))

    # reference contribution
    t = time.time()
    covs_ref = srmerge.estimate_kernels(b['raws'][0], cfg)
    A_ref, B_ref = srmerge.accumulate_ref_ab(b['raws'][0], covs_ref, cfg)
    print('ref A/B %.2fs  covs %s  B_ref mean %.4f'
          % (time.time() - t, covs_ref.shape, B_ref.mean()))

    N = len(b['raws'])
    A = np.zeros((N - 1, 3, h * 2, w * 2), np.float32)
    B = np.zeros_like(A)
    Ag = np.zeros_like(A)
    Bg = np.zeros_like(A)
    feats = np.zeros((N - 1, srsim.NUM_FEATURES, h, w), np.float32)
    Rw = np.zeros((N - 1, h, w), np.float32)
    t = time.time()
    for n in range(1, N):
        covs = srmerge.estimate_kernels(b['raws'][n], cfg)
        fx, fy = srmerge.tile_flow_at_output(b['flows'][n], h, w, cfg.tile_size)
        A[n - 1], B[n - 1] = srmerge.accumulate_comp_ab(b['raws'][n], fx, fy, covs, cfg)
        # ground truth: noise free frames, TRUE per-pixel flow, R == 1
        covs_c = srmerge.estimate_kernels(b['raws_clean'][n], cfg)
        tfx, tfy = b['true_flow'][n]
        Ag[n - 1], Bg[n - 1] = srmerge.accumulate_comp_ab(b['raws_clean'][n],
                                                          tfx, tfy, covs_c, cfg)
        gm, _ = srsim.local_stats_3x3(srsim.compute_grey_fft(b['raws'][n]))
        d_sq, sig_sq = srsim.compute_d_sigma(ref_m, ref_v, gm, b['flows'][n],
                                             cfg, std_c, diff_c)
        feats[n - 1] = srsim.build_features(d_sq, sig_sq, ref_m, ref_v,
                                            b['flows'][n], cfg)
        Rw[n - 1] = srsim.wronski_robustness(d_sq, sig_sq, b['flows'][n], ref_m, cfg)
    print('comp loop %.2fs' % (time.time() - t))

    A_ref_c, B_ref_c = srmerge.accumulate_ref_ab(b['raws_clean'][0],
                                                 srmerge.estimate_kernels(b['raws_clean'][0], cfg), cfg)
    ones = np.ones((N - 1, h, w), np.float32)
    ones_out = np.ones((N - 1, h * 2, w * 2), np.float32)
    gt = srmerge.merge_from_ab(A_ref_c, B_ref_c, Ag, Bg, ones_out)

    def merged(R):
        Rout = np.stack([srmerge.sample_r_at_output(R[n], h, w)
                         for n in range(N - 1)], axis=0)
        return srmerge.merge_from_ab(A_ref, B_ref, A, B, Rout)

    out_w = merged(Rw)
    out_1 = merged(ones)
    out_0 = merged(np.zeros_like(ones))
    print()
    print('GT       mean %.4f' % gt.mean())
    print('R=Wronski  PSNR %.2f dB   (mean R %.3f)' % (psnr(out_w, gt), Rw.mean()))
    print('R=1        PSNR %.2f dB' % psnr(out_1, gt))
    print('R=0 (ref)  PSNR %.2f dB' % psnr(out_0, gt))
    print()
    print('feature means:')
    for i, nm in enumerate(srsim.FEATURE_NAMES):
        print('  %-7s %.4f  (p99 %.4f)' % (nm, feats[:, i].mean(),
                                           np.percentile(feats[:, i], 99)))


if __name__ == '__main__':
    main()
