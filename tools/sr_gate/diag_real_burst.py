"""What does the gate actually output on the ours2 person?

The guide is not blind to it -- d^2/sigma^2 has a median of 52 there, which is
overwhelming evidence. So either the gate rejects it and the ghost comes from
somewhere else, or the gate is under-rejecting far outside the range it was
trained on. This runs the real features through the real weights and says which.

Flow model: block matching cannot find a subject that has moved ~500 raw px with
a search range of tens, so its tiles lock onto the BACKGROUND. The per-tile flow
is therefore the global background shift everywhere -- which is exactly the
"object tiles lock onto the background" regime, at a displacement 50x larger than
any the training set contained.
"""
import os
import sys

import numpy as np
import torch

SR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SR)
import srsim
import gate

CKPT = os.environ.get('SR_GATE_CKPT', 'sr_gate.pt')
from diag_real_guide import load_raw, phase_shift


def main():
    ref, _ = load_raw('APC_1235.dng')
    cmp_, _ = load_raw('APC_1237.dng')
    h, w = ref.shape
    ga = 0.25 * (ref[0::2, 0::2] + ref[0::2, 1::2] + ref[1::2, 0::2] + ref[1::2, 1::2])
    gb = 0.25 * (cmp_[0::2, 0::2] + cmp_[0::2, 1::2] + cmp_[1::2, 0::2] + cmp_[1::2, 1::2])
    dy, dx = phase_shift(ga, gb)
    dy, dx = dy * 2, dx * 2
    print('background shift dy %d dx %d' % (dy, dx))

    cfg = srsim.Cfg(noise_gain=1.0, tile_size=16).use_decimated_guide()
    import build as bld
    std_c, diff_c = bld.sqrt_curves(cfg)

    ref_m, ref_v = srsim.local_stats_3x3(srsim.compute_guide_decimate3(ref))
    cmp_m, cmp_v = srsim.local_stats_3x3(srsim.compute_guide_decimate3(cmp_))

    ts = 16
    ny, nx = h // ts, w // ts
    flow = np.zeros((ny, nx, 2), np.float32)
    flow[..., 0] = dx          # every tile locked to the background
    flow[..., 1] = dy

    d_sq, sig_sq, comps = srsim.compute_d_sigma(ref_m, ref_v, cmp_m, flow, cfg,
                                                std_c, diff_c)
    feat = srsim.build_features(d_sq, sig_sq, ref_m, ref_v, flow, cfg)
    Rw = srsim.wronski_robustness(d_sq, sig_sq, flow, ref_m, cfg)

    ck = torch.load(os.path.join(SR, CKPT), map_location='cpu', weights_only=True)
    net = gate.from_checkpoint(ck)
    n_in = ck['in_ch']
    with torch.no_grad():
        Rg = net(torch.from_numpy(feat[:n_in][None]))[0, 0].numpy()

    # the veto: hard zero for overwhelming residuals, inert below a_lo
    A_LO, A_HI = 12.0, 30.0
    av = np.expm1(np.clip(feat[1], 0, 1) * 8.0)
    Rv = Rg * np.clip((A_HI - av) / (A_HI - A_LO), 0.0, 1.0)

    def locmin(x, r):
        p_ = np.pad(x, r, mode='edge')
        o = p_[0:x.shape[0], 0:x.shape[1]].copy()
        for i in range(2 * r + 1):
            for j in range(2 * r + 1):
                np.minimum(o, p_[i:i + x.shape[0], j:j + x.shape[1]], out=o)
        return o

    Rg_m5 = locmin(Rg, 2)
    Rv_m5 = locmin(Rv, 2)
    Rv_m8 = locmin(Rv, 8)

    gh, gw = ref_m.shape[:2]

    def box(x0, x1, y0, y1):
        return (slice(int(y0 / 570 * gh), int(y1 / 570 * gh)),
                slice(int(x0 / 760 * gw), int(x1 / 760 * gw)))

    regions = {
        'PERSON (moved ~500px)': box(345, 470, 130, 560),
        'dog (moved)':           box(495, 660, 390, 565),
        'sky (static)':          box(40, 300, 20, 130),
        'grass (static)':        box(40, 300, 300, 560),
    }
    a = d_sq / np.maximum(sig_sq, 1e-20)
    print()
    hdr = ('region', 'med a', 'Wronski', 'gate', '+veto', '+min5', 'veto+min5',
           'veto+min8')
    print('%-24s %-7s %-8s %-7s %-7s %-7s %-10s %-10s' % hdr)
    for nm, sl in regions.items():
        print('%-24s %-7.1f %-8.3f %-7.3f %-7.3f %-7.3f %-10.3f %-10.3f'
              % (nm, np.median(a[sl]), Rw[sl].mean(), Rg[sl].mean(),
                 Rv[sl].mean(), Rg_m5[sl].mean(), Rv_m5[sl].mean(),
                 Rv_m8[sl].mean()))

    print()
    print('whole frame: Wronski %.3f  gate %.3f  gate+veto %.3f'
          % (Rw.mean(), Rg.mean(), Rv.mean()))
    # How does the gate respond to a alone? Sweep the two ratio channels with the
    # rest held at the person region's medians.
    print()
    print('gate response to d^2/sigma^2 with other channels at PERSON medians:')
    sl = regions['PERSON (moved ~500px)']
    med = np.median(feat[:, sl[0], sl[1]].reshape(feat.shape[0], -1), axis=1)
    for av in (0.0, 1.0, 3.0, 10.0, 52.0, 300.0):
        f = np.tile(med[:, None, None], (1, 48, 48)).astype(np.float32)
        f[0] = np.exp(-min(av, 60.0))
        f[1] = min(np.log1p(av) / 8.0, 1.0)
        with torch.no_grad():
            r = net(torch.from_numpy(f[:n_in][None]))[0, 0].numpy()
        print('   a = %-8.1f exp_a %-8.4f log_a %-8.4f -> gate R %.3f'
              % (av, f[0, 0, 0], f[1, 0, 0], r.mean()))


if __name__ == '__main__':
    main()
