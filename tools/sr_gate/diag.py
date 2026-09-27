"""Sanity-check the ported Wronski baseline across the flow-error sweep.

If the port is right, R should sit near 1 when alignment is near perfect and
fall as the error grows. A mask that rejects everything at 0.5 px would be a
port bug -- or the known over-rejection of the FFT guide, which is what the
components column separates.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srsim
import srmerge
import srburst

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))


def main():
    dngs = srburst.find_dngs(ROOT)
    scene_full = srburst.load_scene(dngs[0])
    Hs = Ws = 384
    y0 = (scene_full.shape[0] - Hs) // 2
    x0 = (scene_full.shape[1] - Ws) // 2
    scene = np.ascontiguousarray(scene_full[y0:y0 + Hs, x0:x0 + Ws])

    print('%-9s %-7s %-7s %-7s %-7s %-7s %-9s' %
          ('sigma_f', 'meanR', 'noGeom', 'noGeomM', 'medA', 'p90A', 'geomRej%'))
    for sf in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0):
        rng = np.random.default_rng(7)
        spec = srburst.BurstSpec(rng, regime=0)
        spec.sigma_flow = sf
        spec.noise_gain = 8.0
        spec.theta = 0.0
        spec.trans = 3.0
        b = srburst.synth_burst(scene, spec, rng, tile_size=16)
        cfg = srsim.Cfg(noise_gain=spec.noise_gain, tile_size=16)
        std_c, diff_c = srsim.noise_curves_closed_form(cfg.alpha_rob, cfg.beta_rob)
        cfg.tune_snr(b['raws'][0], std_c)
        cfg.tile_size = 16

        ref_m, ref_v = srsim.local_stats_3x3(srsim.compute_grey_fft(b['raws'][0]))
        rs, rs_ng, rs_ng_nomin, amed, ap90, grej = [], [], [], [], [], []
        for n in range(1, len(b['raws'])):
            gm, _ = srsim.local_stats_3x3(srsim.compute_grey_fft(b['raws'][n]))
            d_sq, sig_sq = srsim.compute_d_sigma(ref_m, ref_v, gm, b['flows'][n],
                                                 cfg, std_c, diff_c)
            a = d_sq / np.maximum(sig_sq, 1e-20)
            amed.append(np.median(a))
            ap90.append(np.percentile(a, 90))
            rs.append(srsim.wronski_robustness(d_sq, sig_sq, b['flows'][n], ref_m,
                                               cfg, geom_reject=True).mean())
            r_ng = srsim.wronski_robustness(d_sq, sig_sq, b['flows'][n], ref_m,
                                            cfg, geom_reject=False)
            rs_ng.append(r_ng.mean())
            # same thing without Eq.9's 5x5 minimum, to see how much the
            # dilation itself costs
            S, _ = srsim.compute_s(b['flows'][n])
            ty, tx = srsim.tile_index_grids(*ref_m.shape, 16)
            raw_r = np.clip(S[ty, tx] * np.exp(-np.minimum(a, 700)) - srsim.R_T, 0, 1)
            rs_ng_nomin.append(raw_r.mean())
            rw = srsim.wronski_robustness(d_sq, sig_sq, b['flows'][n], ref_m, cfg, True)
            grej.append(100.0 * float(np.mean((r_ng > 0.5) & (rw < 0.5))))
        print('%-9.2f %-7.3f %-7.3f %-7.3f %-7.3f %-7.2f %-9.2f' %
              (sf, np.mean(rs), np.mean(rs_ng), np.mean(rs_ng_nomin),
               np.mean(amed), np.mean(ap90), np.mean(grej)))


if __name__ == '__main__':
    main()
