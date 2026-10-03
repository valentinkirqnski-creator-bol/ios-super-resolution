"""Does the channel-SUMMED d^2/sigma^2 miss a colour-only misalignment?

    python probe_colour.py

compute_d_sigma sums d^2 and sigma^2 across the three guide channels and applies
max()/shrinkage once to the totals. A white letter on a red field -- the crop
this was written for -- is the awkward case: across that boundary the RED channel
barely changes while green and blue change completely, so the signal lives in two
channels and the denominator collects variance from all three. If the summed
ratio is much smaller than the per-channel maximum, then a misaligned tile on a
strong colour edge reads as weaker evidence than it is, and the mask keeps it.

Builds a synthetic white-on-red field, applies a WHOLE-TILE offset to the
comparison frame (which is what the crop shows: a staircase along a near-vertical
stroke, stepping at the tile grid), and compares:

  summed      d^2/sigma^2 as Eq. 6 forms it, what the gate sees today
  per-ch max  max over channels of d_c^2/sigma_c^2
  chroma      the same on (R-G, B-G) only, i.e. luma removed entirely

A large gap between the first and the others is the case for giving the network
a colour-specific channel rather than hoping it infers one.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build as bld
import srsim


def field(h, w, stripe=96, shift=0):
    """White strokes on red, vertical, with a shift applied in raw px."""
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.55          # red field: high R, low G/B
    img[..., 1] = 0.06
    img[..., 2] = 0.07
    x = (np.arange(w) + shift) % stripe
    on = (x > stripe * 0.45) & (x < stripe * 0.80)
    img[:, on, :] = np.array([0.88, 0.86, 0.87], np.float32)   # near-white
    return img




def main():
    h = w = 256
    cfg = srsim.Cfg(noise_gain=2.0, tile_size=16).use_decimated_guide()
    std_c, diff_c = bld.sqrt_curves(cfg)

    ref_rgb = field(h, w, shift=0)
    # A whole-tile offset: 3 raw px, constant over the frame, which is what one
    # wrong motion vector per tile produces inside that tile.
    cmp_rgb = field(h, w, shift=3)

    # Guide means per channel, both frames, on the half-res 3-channel guide.
    gr = srsim.compute_guide_decimate3(_bayer(ref_rgb))
    gc = srsim.compute_guide_decimate3(_bayer(cmp_rgb))
    rm, rv = srsim.local_stats_3x3(gr)
    cm, _ = srsim.local_stats_3x3(gc)

    nch = rm.shape[2]
    print('guide %dx%d x%d' % (rm.shape[0], rm.shape[1], nch))

    # Per channel: squared mean difference and the Eq.6 denominator pieces.
    d_ch, s_ch = [], []
    for ch in range(nch):
        d = (rm[..., ch] - cm[..., ch]) ** 2
        b = np.clip(rm[..., ch], 0, 1)
        nv = srsim.guide_noise_var(cfg, b, ch, nch)
        s = np.maximum(rv[..., ch], nv)
        d_ch.append(d)
        s_ch.append(s)
    d_ch = np.stack(d_ch); s_ch = np.stack(s_ch)

    summed = d_ch.sum(axis=0) / np.maximum(s_ch.sum(axis=0), 1e-20)
    per_max = (d_ch / np.maximum(s_ch, 1e-20)).max(axis=0)
    # Chroma only: differences of (R-G) and (B-G), so a pure luma change cancels.
    dr = (rm[..., 0] - rm[..., 1]) - (cm[..., 0] - cm[..., 1])
    db = (rm[..., 2] - rm[..., 1]) - (cm[..., 2] - cm[..., 1])
    chroma = (dr * dr + db * db) / np.maximum(s_ch.sum(axis=0), 1e-20)

    # Report only where the edge actually is; a flat field says nothing.
    at_edge = d_ch.sum(axis=0) > np.quantile(d_ch.sum(axis=0), 0.90)
    print('\n%-14s %-12s %-12s %-12s' % ('', 'mean', 'p90', 'max'))
    for nm, v in (('summed', summed), ('per-ch max', per_max),
                  ('chroma', chroma)):
        e = v[at_edge]
        print('%-14s %-12.2f %-12.2f %-12.2f' % (nm, e.mean(),
                                                 np.percentile(e, 90), e.max()))
    print('\nper-ch max / summed  = %.2fx  at the edge'
          % (per_max[at_edge].mean() / max(summed[at_edge].mean(), 1e-9)))
    print('chroma     / summed  = %.2fx' % (chroma[at_edge].mean() /
                                            max(summed[at_edge].mean(), 1e-9)))
    print('\nexp(-a), which is feature 0, at the edge:')
    for nm, v in (('summed', summed), ('per-ch max', per_max)):
        print('  %-12s %.4f   (0 = reject, 1 = keep)'
              % (nm, float(np.exp(-np.minimum(v[at_edge], 60.0)).mean())))


def _bayer(rgb):
    """RGB -> a Bayer mosaic, the form compute_guide_decimate3 expects."""
    h, w = rgb.shape[:2]
    ch = srsim.CFA[np.arange(h)[:, None] & 1, np.arange(w)[None, :] & 1]
    return np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0]


if __name__ == '__main__':
    main()
