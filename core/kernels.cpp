#include "stages.h"
#include "parallel.h"
#include "linalg.h"
#ifdef __APPLE__
#include "metal_gpu.h"
#endif

namespace hhsr {

// Multiplier that puts the kernel-estimate grey's NOISE at the amplitude
// D_th/D_tr are calibrated against. 1.0 means "leave it alone".
//
// D = clamp(1 - sqrt(l1)/D_tr + D_th, 0, 1) compares sqrt(l1) to D_tr ~ 1.1,
// which only means anything if the GAT actually mapped sensor noise to unit
// variance -- true ONLY when alpha/beta match the real noise. They frequently
// do not: a DNG without a NoiseProfile falls back to a single fixed alpha with
// no ISO scaling, and Config::has_noise_profile is used for logging only, so
// the mismatch is silent.
//
// The failure is not subtle. Measured on a flat patch of pure noise, model too
// small by a factor of:
//     1x   -> sigma 0.7500 isotropic     (denoise branch, correct)
//     4x   -> sigma 0.5451               (partly)
//     10x  -> sigma 0.1768, aspect 5.0   (DETAIL branch, on PURE NOISE)
// Beyond ~10x every pixel of a noisy frame takes the sharpest, most
// anisotropic kernel available, oriented by noise rather than by any edge --
// which reads as oversharpening, colour fringing, and anisotropy that looks
// random. D_tr would need to reach ~4 to compensate a 30x error and its slider
// stops at 2.0, so no setting can reach it.
//
// Measuring instead of trusting makes D scale-invariant. Robust estimator:
// the median absolute adjacent difference is dominated by flat areas, so edges
// and texture do not inflate it, and MAD/0.6745 recovers a Gaussian sigma.
// Computed from the RAW rather than the grey so the CPU and Metal paths can
// share it -- the GPU never has the grey host-side.
f32 kernel_noise_autoscale_factor(const Image& raw, const Config& cfg) {
    if (!cfg.kernel_noise_autoscale || raw.h < 8 || raw.w < 8) return 1.f;
    const f32 alpha = cfg.noise_alpha_gat(), beta = cfg.noise_beta_gat();
    if (!(alpha > 0.f) || !std::isfinite(alpha) || !std::isfinite(beta)) return 1.f;
    const f32 c = 0.375f * alpha * alpha + beta;
    auto g = [&](int y, int x) {
        return (2.f / alpha) * std::sqrt(std::max(0.f, alpha * raw.at(y, x) + c));
    };
    // Quad mean, mirroring compute_grey_decimate, then adjacent differences of
    // neighbouring quads. Subsampled: the noise level needs a distribution, not
    // every pixel.
    const int step = std::max(2, (std::min(raw.h, raw.w) / 256) * 2);
    std::vector<f32> d;
    d.reserve(4096);
    auto quad = [&](int y, int x) {
        if (!cfg.bayer_mode) return g(y, x);
        return 0.25f * (g(y, x) + g(y, x + 1) + g(y + 1, x) + g(y + 1, x + 1));
    };
    const int qs = cfg.bayer_mode ? 2 : 1;
    for (int y = 0; y + qs < raw.h - qs; y += step)
        for (int x = 0; x + 2 * qs < raw.w - qs; x += step)
            d.push_back(std::fabs(quad(y, x + qs) - quad(y, x)));
    if (d.size() < 64) return 1.f;
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    const f32 mad = d[d.size() / 2];
    // an adjacent difference of iid noise has variance 2*sigma^2
    const f32 meas = mad / 0.6745f / 1.41421356f;
    if (!(meas > 1e-8f) || !std::isfinite(meas)) return 1.f;
    // What a correct model would have produced: the GAT leaves unit variance
    // per raw sample and the quad decimation averages 4 of them.
    const f32 want = cfg.bayer_mode ? 0.5f : 1.f;
    const f32 k = want / meas;
    // Only correct GROSS mismatch -- within 2x the declaration is trusted, so a
    // well-formed NoiseProfile is left exactly alone.
    if (k >= 0.5f && k <= 2.f) return 1.f;
    if (!std::isfinite(k) || !(k > 0.f)) return 1.f;
    // Bounded, not just gross-mismatch-gated: a scene whose adjacent-quad
    // spacing (2 raw px, matching compute_grey_decimate) aliases against real
    // content -- period-2 checkerboard is the measured worst case, an 85x
    // apparent mismatch on a scene with an EXACT noise model -- reads as
    // noise no median can separate from signal at that scale, and would
    // otherwise scale the grey by two orders of magnitude on a single frame
    // of real, non-noisy texture. A genuine metadata error (wrong ISO's
    // NoiseProfile, a missing one) is a sigma-space mismatch of a few x at
    // most for any plausible ISO gap; clamping to 8x keeps correcting those
    // while bounding what an aliased scene can do to a kernel that was never
    // actually noisy.
    return clampf(k, 1.f / 8.f, 8.f);
}

CovField estimate_kernels(const Image& raw, const Config& cfg) {
#ifdef __APPLE__
    // Metal GPU only — same Alg. 5 math as the CPU path below (golden reference).
    CovField gpu = estimate_kernels_metal(raw, cfg);
    if (gpu.h > 0 && gpu.w > 0) return gpu;
    return CovField();
#else
    // Generalized Anscombe VST — matches utils_image.GAT / cuda_GAT.
    auto apply_gat = [](const Image& img, f32 alpha, f32 beta) {
        Image out(img.h, img.w, 1);
        f32 c = 0.375f * alpha * alpha + beta; // 3/8 * alpha^2 + beta
        for (size_t i = 0; i < img.data.size(); ++i) {
            f32 v = alpha * img.data[i] + c;
            out.data[i] = (2.f / alpha) * std::sqrt(std::max(0.f, v));
        }
        return out;
    };
    // Eq. 4's k1/k2 -- python-z kernels.py compute_k, verbatim:
    //     A = 1 + sqrt((l1 - l2)/(l1 + l2))
    //     D = clamp(1 - sqrt(l1)/D_tr + D_th, 0, 1)
    //     hard_threshold: A > 1.95 -> (1/k_shrink, k_stretch) else (1, 1)
    //     linear:         k1 = 1 + A/2*(1/k_shrink - 1)
    //                     k2 = 1 + A/2*(k_stretch - 1)
    //     k[i] = k_detail * ((1-D)*ki + D*k_denoise)
    //
    // Nothing else. python-z has no width floor, no continuous-anisotropy
    // override, no zero-floor remap and no stretch exponent, so neither does
    // this. Its selection_law knob (hard_threshold | linear, default linear)
    // is the one switch it does have, and that is Config::selection.
    auto compute_k = [](f32 l1, f32 l2, f32& k1, f32& k2, const Config& cfg) {
        const f32 tr = l1 + l2;
        // Flat patch: both eigenvalues ~0, so python-z computes 0/0 -> NaN and
        // relies on NaN > 1.95 being false to land on isotropic. Same result,
        // reached deliberately -- a NaN would otherwise propagate into k1/k2
        // through the linear branch, which python-z's numba path tolerates and
        // this one does not.
        f32 ratio = (tr > 1e-12f) ? std::max(0.f, (l1 - l2) / tr) : 0.f;
        if (!std::isfinite(ratio)) ratio = 0.f;
        const f32 A = 1.f + std::sqrt(ratio);
        const f32 D = std::min(1.f, std::max(0.f,
            1.f - std::sqrt(std::max(0.f, l1)) / cfg.D_tr + cfg.D_th));
        f32 kk1, kk2;
        if (cfg.selection == SelectionLaw::Linear) {
            kk1 = 1.f + 0.5f * A * (1.f / cfg.k_shrink - 1.f);
            kk2 = 1.f + 0.5f * A * (cfg.k_stretch - 1.f);
        } else if (A > 1.95f) {
            kk1 = 1.f / cfg.k_shrink; kk2 = cfg.k_stretch;
        } else {
            kk1 = 1.f; kk2 = 1.f;
        }
        k1 = cfg.k_detail * ((1.f - D) * kk1 + D * cfg.k_denoise);
        k2 = cfg.k_detail * ((1.f - D) * kk2 + D * cfg.k_denoise);
        // Width floor, applied here rather than only after inversion -- see
        // kMergeInvCovMax. Not python-z's; it is what keeps a collapsing
        // covariance out of the degeneracy fallback, and what keeps the
        // half accumulator's denominators representable.
        const f32 kmin = 1.f / std::sqrt(kMergeInvCovMax);
        k1 = std::max(k1, kmin);
        k2 = std::max(k2, kmin);
    };

    // python-z order: GAT the raw, then decimate to grey (kernels.py:80, :84).
    Image vst_raw = apply_gat(raw, cfg.noise_alpha_gat(), cfg.noise_beta_gat());
    Image grey = compute_grey_decimate(vst_raw, cfg.bayer_mode);

    const f32 nscale = kernel_noise_autoscale_factor(raw, cfg);
    if (nscale != 1.f) for (f32& v : grey.data) v *= nscale;

    Image grad = compute_gradients(grey); // [gh-1, gw-1, 2]

    int H = grey.h, W = grey.w;
    CovField covs(H, W);

    parallel_rows(H, cfg.num_threads, [&](int y) {
        for (int x = 0; x < W; ++x) {
            // Structure tensor over 2x2 gradient neighborhood (cuda_estimate_kernel)
            f32 s00 = 0, s01 = 0, s11 = 0;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    int gy = y - 1 + i, gx = x - 1 + j;
                    if (gy < 0 || gy >= grad.h || gx < 0 || gx >= grad.w) continue;
                    f32 gxv = grad.at(gy, gx, 0), gyv = grad.at(gy, gx, 1);
                    s00 += gxv * gxv;
                    s01 += gxv * gyv;
                    s11 += gyv * gyv;
                }
            }
            f32 l[2], e1[2], e2[2];
            eigen_elmts_2x2(s00, s01, s01, s11, l, e1, e2);

            // Python always runs compute_k (iso only affects merge, not estimate)
            f32 k1, k2;
            compute_k(l[0], l[1], k1, k2, cfg);

            f32 k1s = k1 * k1, k2s = k2 * k2;
            f32* c = covs.at(y, x);
            c[0] = k1s * e1[0] * e1[0] + k2s * e2[0] * e2[0];
            c[1] = k1s * e1[0] * e1[1] + k2s * e2[0] * e2[1];
            c[2] = c[1];
            c[3] = k1s * e1[1] * e1[1] + k2s * e2[1] * e2[1];
        }
    });
    return covs;
#endif
}

} // namespace hhsr
