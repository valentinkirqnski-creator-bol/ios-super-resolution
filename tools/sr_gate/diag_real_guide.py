"""Does the shipping FFT guide see the person at all?

The shipping robustness guide (robustness_fft_guide_active) is a full-resolution
SINGLE-CHANNEL low pass of the raw: a luminance-ish image. The guide it replaced
on the Metal path was the half-resolution THREE-CHANNEL Bayer average, where
Eq. 6 sums d^2 over R, G and B.

So a subject that differs from what it is pasted over in COLOUR but not much in
LUMINANCE produces a large d on the three-channel guide and a small one on the
single-channel guide. If that is what is happening on ours2, no mask -- learned
or analytic -- can reject it, because the evidence is not in the guide.

Real frames, real motion, global background alignment by phase correlation.
"""
import os
import sys

import numpy as np
import rawpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srsim

D = 'C:/Users/valen/Downloads/ours2'


def load_raw(name, d=None):
    """Normalised, white-balanced raw plane, as raw_io hands the pipeline.
    `name` may be a bare filename (resolved against D) or a full path."""
    p = name if os.path.isabs(name) else os.path.join(d or D, name)
    r = rawpy.imread(p)
    raw = r.raw_image_visible.astype(np.float64)
    blk = float(r.black_level_per_channel[0])
    raw = np.clip((raw - blk) / (float(r.white_level) - blk), 0.0, 1.0)
    # prewhiten per CFA site, green = 1
    wb = np.asarray(r.camera_whitebalance, np.float64)
    g = wb[1] if wb[1] > 0 else 1.0
    pat = r.raw_pattern
    out = raw.copy()
    for i in range(2):
        for j in range(2):
            c = int(pat[i, j])
            ci = 1 if c == 3 else c          # rawpy: 3 is the second green
            out[i::2, j::2] *= wb[ci] / g
    return np.clip(out, 0.0, 1.0).astype(np.float32), pat


def phase_shift(a, b):
    """Global integer shift of b relative to a, by phase correlation."""
    A = np.fft.rfft2(a - a.mean())
    B = np.fft.rfft2(b - b.mean())
    X = A * np.conj(B)
    X /= np.maximum(np.abs(X), 1e-9)
    c = np.fft.irfft2(X, s=a.shape)
    k = np.unravel_index(np.argmax(c), c.shape)
    dy = k[0] if k[0] <= a.shape[0] // 2 else k[0] - a.shape[0]
    dx = k[1] if k[1] <= a.shape[1] // 2 else k[1] - a.shape[1]
    return dy, dx


def stats3(raw):
    """Half-res 3-channel Bayer guide (the guide Metal used before), R/G/B."""
    h, w = raw.shape
    g = np.zeros((h // 2, w // 2, 3), np.float32)
    g[..., 0] = raw[0::2, 0::2]                                  # R at (0,0)
    g[..., 1] = 0.5 * (raw[0::2, 1::2] + raw[1::2, 0::2])         # two greens
    g[..., 2] = raw[1::2, 1::2]                                   # B at (1,1)
    return g


def box3(x):
    p = np.pad(x.astype(np.float64), ((1, 1), (1, 1)) + ((0, 0),) * (x.ndim - 2),
               mode='edge')
    s = np.zeros_like(x, dtype=np.float64)
    s2 = np.zeros_like(x, dtype=np.float64)
    h, w = x.shape[:2]
    for i in range(3):
        for j in range(3):
            v = p[i:i + h, j:j + w]
            s += v
            s2 += v * v
    m = s / 9.0
    return m, np.maximum(s2 / 9.0 - m * m, 0.0)


def main():
    ref, pat = load_raw('APC_1235.dng')
    cmp_, _ = load_raw('APC_1237.dng')      # two frames apart: real motion
    h, w = ref.shape

    # global background alignment, on a decimated grey
    ga = 0.25 * (ref[0::2, 0::2] + ref[0::2, 1::2] + ref[1::2, 0::2] + ref[1::2, 1::2])
    gb = 0.25 * (cmp_[0::2, 0::2] + cmp_[0::2, 1::2] + cmp_[1::2, 0::2] + cmp_[1::2, 1::2])
    dy, dx = phase_shift(ga, gb)
    dy *= 2
    dx *= 2
    # keep the shift even so the CFA phase is preserved
    dy -= dy % 2
    dx -= dx % 2
    print('global background shift (raw px): dy %d dx %d' % (dy, dx))

    def shifted(a, sy, sx):
        return np.roll(np.roll(a, -sy, axis=0), -sx, axis=1)

    cmp_al = shifted(cmp_, dy, dx)

    alpha = srsim.ALPHA_DNG * (1.0 + 0.5 + 1.0) / 3.0
    beta = srsim.BETA_DNG * (1.0 + 0.5 + 1.0) / 3.0

    # ---- A: the SHIPPING guide, full-res single channel, linear -----------
    g1a = srsim.compute_grey_fft(ref)
    g1b = srsim.compute_grey_fft(cmp_al)
    m1a, v1a = box3(g1a)
    m1b, _ = box3(g1b)
    d1 = (m1a - m1b) ** 2
    a1 = alpha * 0.25
    b1 = beta * 0.25
    sc = srsim.noise_curves_closed_form(a1, b1)[0]
    sig_t1 = srsim.curve_lookup(sc, np.clip(m1a, 0, 1)) ** 2
    sig1 = np.maximum(v1a, sig_t1)
    ratio1 = d1 / np.maximum(sig1, 1e-20)

    # ---- B: the guide Metal used before, half-res three channel ----------
    g3a = stats3(ref)
    g3b = stats3(cmp_al)
    m3a, v3a = box3(g3a)
    m3b, _ = box3(g3b)
    d3 = ((m3a - m3b) ** 2).sum(axis=2)
    sc3 = srsim.noise_curves_closed_form(alpha, beta)[0]
    sig_t3 = sum(srsim.curve_lookup(sc3, np.clip(m3a[..., c], 0, 1)) ** 2
                 for c in range(3))
    sig3 = np.maximum(v3a.sum(axis=2), sig_t3)
    ratio3 = d3 / np.maximum(sig3, 1e-20)

    # ---- the person, from the preview: x 340-470 of 760, y 130-570 of 570
    def box(fr, x0, x1, y0, y1, W, H):
        return (slice(int(y0 / H * fr.shape[0]), int(y1 / H * fr.shape[0])),
                slice(int(x0 / W * fr.shape[1]), int(x1 / W * fr.shape[1])))

    per1 = box(ratio1, 345, 470, 130, 560, 760, 570)
    per3 = box(ratio3, 345, 470, 130, 560, 760, 570)
    sky1 = box(ratio1, 40, 300, 20, 130, 760, 570)
    sky3 = box(ratio3, 40, 300, 20, 130, 760, 570)

    # Wronski: R = clip(s2*exp(-a) - t, 0, 1) with s2 = 12, t = 0.12
    def meanR(rt):
        return float(np.clip(12.0 * np.exp(-np.minimum(rt, 700)) - 0.12, 0, 1).mean())

    print()
    print('%-28s %-12s %-12s %-12s' % ('', 'median a', 'frac a>2.4', 'mean Wronski R'))
    for nm, rt, sl in (('1-ch FFT guide  PERSON', ratio1, per1),
                       ('3-ch decimated  PERSON', ratio3, per3),
                       ('1-ch FFT guide  sky/bg', ratio1, sky1),
                       ('3-ch decimated  sky/bg', ratio3, sky3)):
        r = rt[sl]
        print('%-28s %-12.2f %-12.3f %-12.3f'
              % (nm, np.median(r), float((r > 2.4).mean()), meanR(r)))


if __name__ == '__main__':
    main()
