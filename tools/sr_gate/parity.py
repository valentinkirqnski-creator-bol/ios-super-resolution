"""Check that core/sr_gate.{h,cpp} computes the same function train.py fitted.

    python parity.py --exe sr_gate_parity.exe --ckpt sr_gate.pt

Dumps one real burst frame's Eq. 6 outputs, runs the C++ through them, and
compares BOTH the eight features and the final mask against the NumPy/torch
versions. A disagreement in the feature builder is the failure mode that would
otherwise go unnoticed: the network would still produce a plausible-looking mask,
just not the one it was trained to produce.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate
import srburst
import srmerge
import srsim
import build as bld


def dump(path, ref_m, ref_v, d_sq, sig_sq, flow, ts, alpha_sensor, beta_sensor):
    h, w = ref_m.shape
    ny, nx = flow.shape[:2]
    with open(path, 'wb') as f:
        f.write(b'SRGD')
        f.write(struct.pack('<5i', h, w, ny, nx, ts))
        f.write(struct.pack('<2f', alpha_sensor, beta_sensor))
        for a in (ref_m, ref_v, d_sq, sig_sq):
            f.write(np.ascontiguousarray(a, np.float32).tobytes())
        f.write(np.ascontiguousarray(flow, np.float32).tobytes())


def read_out(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'SRGO'
        h, w, c = struct.unpack('<3i', f.read(12))
        alpha_rob, beta_rob = struct.unpack('<2f', f.read(8))
        feat = np.frombuffer(f.read(h * w * c * 4), np.float32).reshape(h, w, c)
        mask = np.frombuffer(f.read(h * w * 4), np.float32).reshape(h, w)
    # C++ Image is channel interleaved; the Python features are channel first.
    return feat.transpose(2, 0, 1).copy(), mask.copy(), alpha_rob, beta_rob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default='sr_gate_parity.exe')
    ap.add_argument('--ckpt', default='sr_gate.pt')
    ap.add_argument('--hr', type=int, default=320)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    dngs, _ = srburst.find_dngs(bld.ROOT)
    scene_full = srburst.load_scene(dngs[0])
    y0 = (scene_full.shape[0] - a.hr) // 2
    x0 = (scene_full.shape[1] - a.hr) // 2
    scene = np.ascontiguousarray(scene_full[y0:y0 + a.hr, x0:x0 + a.hr])

    rng = np.random.default_rng(31337)
    spec = srburst.BurstSpec(rng, regime=2)
    spec.sigma_flow = 0.7
    spec.noise_gain = 6.0
    b = srburst.synth_burst(scene, spec, rng, tile_size=16)

    cfg = srsim.Cfg(noise_gain=spec.noise_gain, tile_size=16)
    std_c, diff_c = srsim.noise_curves_closed_form(cfg.alpha_rob, cfg.beta_rob)
    cfg.tune_snr(b['raws'][0], std_c)
    cfg.tile_size = 16
    ref_m, ref_v = srsim.local_stats_3x3(srsim.compute_grey_fft(b['raws'][0]))
    gm, gv = srsim.local_stats_3x3(srsim.compute_grey_fft(b['raws'][1]))
    d_sq, sig_sq, comps = srsim.compute_d_sigma(ref_m, ref_v, gm, b['flows'][1],
                                                cfg, std_c, diff_c)
    cvw = srsim.warp_sample_comp(gv, b['flows'][1], cfg)

    inp = os.path.join(here, 'parity_in.bin')
    outp = os.path.join(here, 'parity_out.bin')
    dump(inp, ref_m, ref_v, d_sq, sig_sq, b['flows'][1], cfg.tile_size,
         cfg.alpha_sensor, cfg.beta_sensor)

    exe = a.exe if os.path.isabs(a.exe) else os.path.join(here, a.exe)
    r = subprocess.run([exe, inp, outp], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip())
        return 1

    c_feat, c_mask, alpha_rob, beta_rob = read_out(outp)
    print('alpha_rob  python %.9g  c++ %.9g  rel %.2e'
          % (cfg.alpha_rob, alpha_rob, abs(alpha_rob / cfg.alpha_rob - 1)))
    print('beta_rob   python %.9g  c++ %.9g  rel %.2e'
          % (cfg.beta_rob, beta_rob, abs(beta_rob / cfg.beta_rob - 1)))

    p_feat = srsim.build_features(d_sq, sig_sq, ref_m, ref_v, b['flows'][1], cfg,
                                  comps, cvw)
    print()
    print('%-8s %-12s %-12s %-10s' % ('feature', 'max abs err', 'mean abs err',
                                      'range'))
    worst = 0.0
    # Only the channels the C++ actually emits. srsim can build more than the
    # shipped net consumes (8-11 measured neutral, not shipped), and the C++ plane
    # is sized by SRG_FEATURES, so the comparison is over that many.
    n_cmp = c_feat.shape[0]
    assert p_feat.shape[0] >= n_cmp, (p_feat.shape, c_feat.shape)
    print('comparing %d channels (C++ SRG_FEATURES); srsim builds %d'
          % (n_cmp, p_feat.shape[0]))
    for i, nm in enumerate(srsim.FEATURE_NAMES[:n_cmp]):
        e = np.abs(p_feat[i] - c_feat[i])
        worst = max(worst, float(e.max()))
        print('%-8s %-12.3e %-12.3e [%.3f, %.3f]'
              % (nm, e.max(), e.mean(), p_feat[i].min(), p_feat[i].max()))
    print('worst feature error %.3e' % worst)

    ck = torch.load(os.path.join(here, a.ckpt), map_location='cpu',
                    weights_only=True)
    net = gate.SRGate(in_ch=ck['in_ch'], width=ck['width'],
                      dilations=ck['dilations'])
    net.load_state_dict(ck['state_dict'])
    net.eval()
    with torch.no_grad():
        # torch on the C++ features isolates the inference from the features
        t_on_c = net(torch.from_numpy(c_feat[None]))[0, 0].numpy()
        t_on_p = net(torch.from_numpy(p_feat[:n_cmp][None]))[0, 0].numpy()
    print()
    print('mask: torch(py feats) vs c++            max %.3e  mean %.3e'
          % (np.abs(t_on_p - c_mask).max(), np.abs(t_on_p - c_mask).mean()))
    print('mask: torch(c++ feats) vs c++ (net only) max %.3e  mean %.3e'
          % (np.abs(t_on_c - c_mask).max(), np.abs(t_on_c - c_mask).mean()))
    print('mask means: torch %.6f  c++ %.6f' % (t_on_p.mean(), c_mask.mean()))

    ok = worst < 2e-5 and np.abs(t_on_c - c_mask).max() < 2e-5
    print()
    print('PARITY', 'OK' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
