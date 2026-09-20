#pragma once
//
// Gradient estimate for the GEOMETRIC motion-rejection test only.
//
// Nothing else reads this. The Wronski robustness mask, the merge, the alignment
// and the flow all keep the gradients and the guide they already had; this
// replaces the two lines inside the geometry block that estimate |grad I|, and
// only those.
//
// WHY
//
// The geometry test is  reject if |grad I| * |E| > motion_geom_reject_threshold.
// |grad I| came from a central difference on the reference guide means, which in
// bright light is a fine estimate and in low light is mostly photon noise: a flat
// dark patch produces a gradient of order the noise sigma, and a large enough |E|
// then rejects tiles that are perfectly well aligned.
//
// WHAT THIS IS NOT
//
// It is not a noise-normalised score. The quantity handed to the threshold is
// still |grad I| in the same units, so motion_geom_reject_threshold keeps its
// calibrated meaning and is not touched. No noise floor is subtracted from the
// absolute criterion either. The only thing that changes is how well |grad I| is
// estimated.
//
// HOW
//
// A normalised 3x3 Sobel IS a central difference with [1,2,1] averaging along the
// perpendicular direction. On a linear ramp it returns exactly what the central
// difference returns -- identical units, identical scale, nothing to recalibrate
// -- while its noise variance is 3v/16 against the central difference's v/2, a
// 2.67x reduction. The averaging runs ALONG an edge, never across it, which is
// what makes this different from blurring the guide: a step edge keeps its full
// gradient.
//
// Averaging is not free on fine diagonal texture, though, so it is applied only
// where the sharp estimate is not trustworthy. The existing noise model gives the
// variance of a guide sample; from it comes the sigma of the sharp gradient, and
// the ratio of the two is an SNR. Above snr_hi the sharp estimate is returned
// unchanged, so bright scenes keep the behaviour they have today. Below snr_lo the
// averaged one is used. Between, they are mixed.
//
// The daylight case is safe by construction rather than by tuning: a pixel can
// only reject if |grad I| * |E| clears the threshold, and for the |E| this
// pipeline produces that needs a gradient far above the noise sigma -- which puts
// it well above snr_hi, where the blend is exactly zero.
//
#include "types.h"

#include <cmath>

namespace hhsr {

// Where the blend starts and ends, as a gradient SNR. Deliberately conservative:
// snr_hi is high enough that anything capable of tripping the geometry threshold
// in a bright scene sits above it and is returned untouched.
struct GeomGradParams {
    float snr_lo = 2.0f;   // at or below: fully averaged
    float snr_hi = 6.0f;   // at or above: the sharp estimate, unmodified
};

struct GeomGradient {
    f32 gx = 0.f;
    f32 gy = 0.f;
    f32 gmag = 0.f;         // what the geometry test should use
    f32 gmag_sharp = 0.f;   // what it used before, for diagnostics
    f32 sigma = 0.f;        // sigma of gmag_sharp, same units as gmag
    f32 snr = 0.f;
    f32 blend = 0.f;        // 0 = sharp and unchanged, 1 = fully averaged
};

// n[] is the 3x3 neighbourhood of guide CHANNEL 0, row-major:
//
//   n[0] n[1] n[2]     (y-1,x-1) (y-1,x) (y-1,x+1)
//   n[3] n[4] n[5]     (y  ,x-1) (y  ,x) (y  ,x+1)
//   n[6] n[7] n[8]     (y+1,x-1) (y+1,x) (y+1,x+1)
//
// The caller clamps the coordinates exactly as the existing geometry block clamps
// xl/xr/yu/yd, so edge behaviour is unchanged.
//
// sc carries the existing guide-pixel -> raw-pixel conversion: 2 for a 3-channel
// Bayer guide at half resolution, 1 otherwise. It divides the gradient here for
// the same reason and by the same amount it always did.
//
// sample_var is the variance of ONE guide means sample. RefStats::means is a 3x3
// local mean of the guide (local_stats_3x3), so that is guide_noise_var(...) / 9.
inline GeomGradient geom_gradient(const f32 n[9], f32 sc, f32 sample_var,
                                  const GeomGradParams& p, bool enabled) {
    GeomGradient g;
    const f32 inv_sc = (sc > 0.f) ? (1.f / sc) : 1.f;

    // The existing estimate, byte for byte: 0.5 * (right - left) / sc.
    g.gx = 0.5f * (n[5] - n[3]) * inv_sc;
    g.gy = 0.5f * (n[7] - n[1]) * inv_sc;
    g.gmag_sharp = std::sqrt(g.gx * g.gx + g.gy * g.gy);
    g.gmag = g.gmag_sharp;
    if (!enabled) return g;

    // Sigma of the sharp gradient magnitude. Each axis is a difference of two
    // samples scaled by a half, so var(gx) = (v + v)/4 = v/2 per axis, and the two
    // axes together give var(gx) + var(gy) = v -- so the sigma of gmag is simply
    // sqrt(v), in gradient units once inv_sc is applied.
    //
    // The two taps of a central difference on 3x3 means share a column, so they
    // are slightly correlated and this is a mild over-estimate. Over-estimating
    // sigma lowers the SNR and therefore averages a little sooner, which is the
    // harmless direction: it costs nothing where the gradient is strong.
    const f32 v = (sample_var > 0.f) ? sample_var : 0.f;
    g.sigma = std::sqrt(v) * inv_sc;
    if (!(g.sigma > 0.f)) return g;      // no noise model: leave it alone
    g.snr = g.gmag_sharp / g.sigma;

    const f32 lo = p.snr_lo;
    const f32 hi = (p.snr_hi > lo) ? p.snr_hi : (lo + 1e-6f);
    f32 t = (g.snr - lo) / (hi - lo);
    t = (t < 0.f) ? 0.f : ((t > 1.f) ? 1.f : t);
    g.blend = 1.f - t * t * (3.f - 2.f * t);   // smoothstep, 1 at low SNR
    if (g.blend <= 0.f) return g;              // bright: exactly as before

    // Normalised 3x3 Sobel. The 1/8 is what makes it agree with the central
    // difference on a ramp instead of being 4x larger.
    const f32 sob_x = ((n[2] + 2.f * n[5] + n[8]) -
                       (n[0] + 2.f * n[3] + n[6])) * (1.f / 8.f) * inv_sc;
    const f32 sob_y = ((n[6] + 2.f * n[7] + n[8]) -
                       (n[0] + 2.f * n[1] + n[2])) * (1.f / 8.f) * inv_sc;

    const f32 wb = g.blend;
    g.gx += (sob_x - g.gx) * wb;
    g.gy += (sob_y - g.gy) * wb;
    g.gmag = std::sqrt(g.gx * g.gx + g.gy * g.gy);
    return g;
}

}  // namespace hhsr
