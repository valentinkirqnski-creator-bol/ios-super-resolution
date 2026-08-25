#include "stages.h"
#include "parallel.h"
#include "linalg.h"
#ifdef __APPLE__
#include "metal_gpu.h"
#endif

namespace hhsr {

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
    // Eq. 4's k1/k2. This follows python-z kernels.py: A = 1 + sqrt(aniso),
    // then either linear interpolation with A/2 or the hard A > 1.95 branch.
    auto compute_k = [](f32 l1, f32 l2, f32& k1, f32& k2, const Config& cfg) {
        const f32 tr = l1 + l2;
        // Flat patch: both eigenvalues ~0. The old form computed 0/0 -> NaN,
        // and NaN > 1.95 is false, so it fell to isotropic by accident. A
        // continuous blend would propagate the NaN into k1/k2 instead, so the
        // degenerate case is now handled on purpose.
        f32 ratio = (tr > 1e-12f) ? std::max(0.f, (l1 - l2) / tr) : 0.f;
        if (!std::isfinite(ratio)) ratio = 0.f;
        const f32 A = 1.f + std::sqrt(ratio);
        f32 D = std::min(1.f, std::max(0.f, 1.f - std::sqrt(std::max(0.f, l1)) / cfg.D_tr + cfg.D_th));
        f32 kk1, kk2;
        if (cfg.kernel_google_s1) {
            // S.1's formulas as printed never produce a round kernel: at
            // A = 1 (isotropic content -- corners, punctual detail, distant
            // text) they still emit a k_shrink x k_stretch 4:1 ellipse whose
            // orientation is eigenvector noise, ~1 px of directional smear
            // that reads as denoising exactly where super-resolution should
            // be most visible. The paper's own prose says a clean corner
            // gets an ISOTROPIC kernel of std k_detail, contradicting its
            // formulas (the IPOL article documents the same defect), so
            // interpolate between the paper's own endpoints: round at A = 1,
            // the verbatim S.1 ellipse (1/(k_shrink*A), k_stretch*A at
            // A = 2) at full coherence. Sharp axis on e1, stretch on e2.
            const f32 t = clampf(A - 1.f, 0.f, 1.f);
            kk1 = 1.f / (1.f + t * (2.f * cfg.k_shrink - 1.f));
            kk2 = 1.f + t * (2.f * cfg.k_stretch - 1.f);
        } else if (cfg.selection == SelectionLaw::Linear || cfg.kernel_anisotropy_continuous) {
            // 0.5*A floors at 0.5; the zero-floor remap keeps the A=1.95
            // endpoint and sends isotropic content to round kernels. See
            // Config::kernel_anisotropy_zero_floor.
            f32 w = 0.5f * A;
            if (cfg.kernel_anisotropy_zero_floor) {
                const f32 t = clampf((A - 1.f) / 0.95f, 0.f, 1.f);
                const f32 g = std::max(1.f, cfg.kernel_stretch_gamma);
                w = 0.975f * ((g == 1.f) ? t : std::pow(t, g));
            }
            kk1 = 1.f + w * (1.f / cfg.k_shrink - 1.f);
            kk2 = 1.f + w * (cfg.k_stretch - 1.f);
        } else if (A > 1.95f) {
            kk1 = 1.f / cfg.k_shrink; kk2 = cfg.k_stretch;
        } else {
            kk1 = 1.f; kk2 = 1.f;
        }
        k1 = cfg.k_detail * ((1.f - D) * kk1 + D * cfg.k_denoise);
        k2 = cfg.k_detail * ((1.f - D) * kk2 + D * cfg.k_denoise);
    };

    // Google S.1 measures the tensor on the [0,1]-normalized image itself
    // (its D thresholds are in that image's gradient units); the IPOL
    // reference measures it in the GAT domain, where noise sigma ~ 1.
    Image vst_raw;
    if (!cfg.kernel_google_s1)
        vst_raw = apply_gat(raw, cfg.noise_alpha(), cfg.noise_beta());
    Image grey = compute_grey_decimate(cfg.kernel_google_s1 ? raw : vst_raw,
                                       cfg.bayer_mode);
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
