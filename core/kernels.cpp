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
    // Matches kernels.py compute_k / hard_threshold / linear (+ C++ float guards).
    auto compute_k = [](f32 l1, f32 l2, f32& k1, f32& k2, const Config& cfg) {
        l1 = std::max(0.f, l1);
        l2 = std::max(0.f, l2);
        f32 sum = l1 + l2;
        if (sum < 1e-12f) {
            k1 = k2 = cfg.k_detail * cfg.k_denoise;
            return;
        }
        f32 A = 1.f + std::sqrt(std::max(0.f, (l1 - l2) / sum));
        f32 D = clampf(1.f - std::sqrt(l1) / cfg.D_tr + cfg.D_th, 0.f, 1.f);
        f32 kk1, kk2;
        if (cfg.selection == SelectionLaw::HardThreshold) {
            if (A > 1.95f) { kk1 = 1.f / cfg.k_shrink; kk2 = cfg.k_stretch; }
            else           { kk1 = 1.f; kk2 = 1.f; }
        } else {
            kk1 = 1.f + A / 2.f * (1.f / cfg.k_shrink - 1.f);
            kk2 = 1.f + A / 2.f * (cfg.k_stretch - 1.f);
        }
        k1 = cfg.k_detail * ((1.f - D) * kk1 + D * cfg.k_denoise);
        k2 = cfg.k_detail * ((1.f - D) * kk2 + D * cfg.k_denoise);
        constexpr f32 k_min = 0.15f;
        k1 = std::max(k1, k_min);
        k2 = std::max(k2, k_min);
    };

    Image vst = apply_gat(raw, cfg.alpha, cfg.beta);
    Image grey = compute_grey_decimate(vst, cfg.bayer_mode);
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
