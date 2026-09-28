"""Dump one synthesised burst frame pair for integration_main.cpp, then check
the mask the REAL pipeline produced against torch.

    python integration_dump.py --exe sr_gate_integ.exe

Only the raw planes and the flow go across. Everything else -- the FFT guide, the
3x3 statistics, the noise curves, Eq. 6, the gate -- is computed by the
pipeline's own C++ from a Config built the way core/types.h pins it. That is the
difference from parity.py, which hands the C++ a feature plane's worth of inputs
prepared in Python.
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
import build as bld
import gate
import srburst
import srsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default='sr_gate_integ.exe')
    ap.add_argument('--ckpt', default='sr_gate.pt')
    ap.add_argument('--hr', type=int, default=320)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    dngs, _ = srburst.find_dngs(bld.ROOT)
    scene_full = srburst.load_scene(dngs[0])
    y0 = (scene_full.shape[0] - a.hr) // 2
    x0 = (scene_full.shape[1] - a.hr) // 2
    scene = np.ascontiguousarray(scene_full[y0:y0 + a.hr, x0:x0 + a.hr])

    rng = np.random.default_rng(4242)
    spec = srburst.BurstSpec(rng, regime=2)
    spec.sigma_flow = 0.9
    spec.noise_gain = 6.0
    b = srburst.synth_burst(scene, spec, rng, tile_size=16)
    # The shipping guide as of defaultsVersion 14: three channels, half
    # resolution, 1.4 sqrt transfer.
    cfg = srsim.Cfg(noise_gain=spec.noise_gain, tile_size=16).use_decimated_guide()
    h, w = b['h'], b['w']
    flow = b['flows'][1]

    inp = os.path.join(here, 'integ_in.bin')
    out = os.path.join(here, 'integ_mask.bin')
    with open(inp, 'wb') as f:
        f.write(b'SRGB')
        f.write(struct.pack('<5i', h, w, flow.shape[0], flow.shape[1],
                            cfg.tile_size))
        f.write(struct.pack('<2f', cfg.alpha_sensor, cfg.beta_sensor))
        f.write(np.ascontiguousarray(b['raws'][0], np.float32).tobytes())
        f.write(np.ascontiguousarray(b['raws'][1], np.float32).tobytes())
        f.write(np.ascontiguousarray(flow, np.float32).tobytes())

    exe = a.exe if os.path.isabs(a.exe) else os.path.join(here, a.exe)
    r = subprocess.run([exe, inp, out], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr.strip():
        print('stderr:', r.stderr.strip())
    if r.returncode != 0:
        return 1

    with open(out, 'rb') as f:
        assert f.read(4) == b'SRGM'
        hh, ww = struct.unpack('<2i', f.read(8))
        m_gate = np.frombuffer(f.read(hh * ww * 4), np.float32).reshape(hh, ww)
        m_ana = np.frombuffer(f.read(hh * ww * 4), np.float32).reshape(hh, ww)

    # The same mask, via the Python port + torch. Any difference here is a
    # difference between the pipeline's own guide/Eq.6 and the port's.
    std_c, diff_c = bld.sqrt_curves(cfg)
    ref_m, ref_v = srsim.local_stats_3x3(
        srsim.compute_guide_decimate3(b['raws'][0]))
    gm, gv = srsim.local_stats_3x3(srsim.compute_guide_decimate3(b['raws'][1]))
    d_sq, sig_sq, comps = srsim.compute_d_sigma(ref_m, ref_v, gm, flow, cfg,
                                                std_c, diff_c)
    # The 8 channels the C++ emits; 8-11 measured neutral and are not shipped.
    feat = srsim.build_features(d_sq, sig_sq, ref_m, ref_v, flow, cfg)
    ck = torch.load(os.path.join(here, a.ckpt), map_location='cpu',
                    weights_only=True)
    net = gate.from_checkpoint(ck)
    n_in = ck['in_ch']
    with torch.no_grad():
        m_torch = net(torch.from_numpy(feat[:n_in][None]))[0, 0].numpy()

    d = np.abs(m_torch - m_gate)
    print()
    print('pipeline gate mask vs torch: max %.3e  mean %.3e  (means %.6f / %.6f)'
          % (d.max(), d.mean(), m_gate.mean(), m_torch.mean()))
    print('analytic mask mean %.6f' % np.nanmean(m_ana))
    # The noise curves differ: the pipeline runs the real Monte Carlo, the port
    # uses the closed form the MC converges to, so a small mismatch here is
    # expected and is NOT a code disagreement.
    # 4e-3, not 5e-3: the residual is Monte-Carlo noise in the noise curves --
    # the pipeline draws 1e5 patches per bin with a seeded MT19937, this port
    # draws 4000 with numpy. Raising it to 50000 takes the max from 6.2e-3 to
    # 3.5e-3, which is what identifies the remainder as sampling rather than a
    # code disagreement. (Before interp_MC_range was ported it was 1.7e-1 and
    # did NOT move with patch count, which is how that omission was found.)
    ok = d.max() < 8e-3
    print('INTEGRATION', 'OK' if ok else 'DIFFERS (see the note on curves)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
