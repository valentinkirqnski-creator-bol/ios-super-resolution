#include "stages.h"
#include "parallel.h"
#include "linalg.h"
#ifdef __APPLE__
#include "metal_gpu.h"
#endif

namespace hhsr {

CovField estimate_kernels(const Image& raw, const Config& cfg) {
    // GCam ComputeGuideImage o_gauss_cov: build the merge kernel from the GCam structure
    // tensor instead of Alg.5. Runs on the host (CPU) on both platforms for now; a Metal
    // twin is TODO. Falls through to the standard estimator if it returns empty.
    if (cfg.robustness_gcam_enabled && cfg.gcam_sabre_kernel) {
        CovField k = compute_gcam_kernel_cov(raw, cfg);
        if (k.h > 0 && k.w > 0) return k;
    }
#ifdef __APPLE__
    // Metal GPU only — same Alg. 5 math as the CPU path below (golden reference).
    CovField gpu = estimate_kernels_metal(raw, cfg);
    if (gpu.h > 0 && gpu.w > 0) return gpu;
    return CovField();
#else
    // 1.4 kernels.py estimate_kernels: preprocessing "gat" is applied to the RAW MOSAIC,
    // per CFA site with that site's colour alpha/beta (utils_image.cuda_GAT), and only
    // THEN decimated to grey -- the GAT is a square root, so the order changes the
    // gradients. VST = alpha*v + 3/8*alpha^2 + beta (1.4's evaluation order), clamped at
    // 0, out = 2/alpha * sqrt(VST). alpha/beta per colour = the DNG profile scaled to
    // the raw's white balance (Config::noise_wb_gain), float32.
    auto apply_gat_raw = [](const Image& img, const Config& cfg) {
        Image out(img.h, img.w, 1);
        f32 ac[3], bc[3];
        for (int c = 0; c < 3; ++c) {
            const f32 g = cfg.noise_wb_gain(c);
            ac[c] = cfg.alpha_dng[c] * g;
            bc[c] = cfg.beta_dng[c] * g * g;
        }
        const f32 a1 = cfg.noise_alpha(), b1 = cfg.noise_beta();   // grey (non-Bayer) input
        parallel_rows(img.h, cfg.num_threads, [&](int y) {
            for (int x = 0; x < img.w; ++x) {
                const int c = cfg.cfa.p[y & 1][x & 1];
                const f32 alpha = cfg.bayer_mode ? ac[c] : a1;
                const f32 beta  = cfg.bayer_mode ? bc[c] : b1;
                const f32 vst = std::max(0.f, alpha * img.at(y, x) + 0.375f * alpha * alpha + beta);
                out.at(y, x) = 2.f / alpha * std::sqrt(vst);
            }
        });
        return out;
    };
    // Matches 460-main kernels.py compute_k.
    auto compute_k = [](f32 l1, f32 l2, f32& k1, f32& k2, const Config& cfg) {
        f32 A = 1.f + std::sqrt((l1 - l2) / (l1 + l2));
        f32 D = std::min(1.f, std::max(0.f, 1.f - std::sqrt(l1) / cfg.D_tr + cfg.D_th));
        f32 kk1, kk2;
        if (cfg.selection == SelectionLaw::HardThreshold) {
            // 460-main: snap to full anisotropy past A=1.95, else isotropic. A is
            // NaN on a flat patch (l1+l2==0); the > test is then false -> isotropic.
            if (A > 1.95f) { kk1 = 1.f / cfg.k_shrink; kk2 = cfg.k_stretch; }
            else           { kk1 = 1.f; kk2 = 1.f; }
        } else {
            // 1.4 default 'linear' (kernels.py::linear): a continuous ramp that
            // coincides with hard_threshold at A==1 (isotropic) and A==2 (full
            // stretch) but ramps in between instead of snapping at 1.95.
            //   k1 = (2-A) + (A-1)/k_shrink ; k2 = (2-A) + (A-1)*k_stretch
            // 1.4 leaves A=NaN (flat patch, l1+l2==0) unguarded, which poisons the
            // covariance; fall back to isotropic there, matching hard_threshold's
            // NaN behaviour and the intended flat-region kernel.
            if (l1 + l2 > 0.f && std::isfinite(A)) {
                kk1 = (2.f - A) + (A - 1.f) / cfg.k_shrink;
                kk2 = (2.f - A) + (A - 1.f) * cfg.k_stretch;
            } else { kk1 = 1.f; kk2 = 1.f; }
        }
        k1 = cfg.k_detail * ((1.f - D) * kk1 + D * cfg.k_denoise);
        k2 = cfg.k_detail * ((1.f - D) * kk2 + D * cfg.k_denoise);
    };

    Image vst = compute_grey_decimate(apply_gat_raw(raw, cfg), cfg.bayer_mode);  // GAT, then decimate
    Image grad = compute_gradients(vst); // [gh-1, gw-1, 2]
    const bool is_iso = (cfg.kernel == KernelShape::Iso);

    int H = vst.h, W = vst.w;
    CovField covs(H, W);

    parallel_rows(H, cfg.num_threads, [&](int y) {
        for (int x = 0; x < W; ++x) {
            if (is_iso) {   // cuda_estimate_kernel: iso -> diag(k_detail), done
                f32* c = covs.at(y, x);
                c[0] = cfg.k_detail; c[1] = 0.f; c[2] = 0.f; c[3] = cfg.k_detail;
                continue;
            }
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
