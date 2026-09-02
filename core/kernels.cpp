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
    // Generalized Anscombe VST, per CFA site. The raw entering here is the
    // loader's PREWHITENED raw -- site value x = k_c*v with k_c = wb_c/wb_1
    // (Config::noise_wb_gain) -- while (alpha, beta) describe the SENSOR
    // noise law Var(v) = alpha*v + beta. Under that gain the law transforms
    // exactly: Var(x) = (k_c*alpha)*x + k_c^2*beta, so each site must be
    // stabilized with its own (k_c*alpha, k_c^2*beta). The previous single
    // shared (alpha, beta) stabilized the wrong variable: post-GAT noise std
    // came out ~sqrt(k_c) per channel ({~1.4 R, 1 G, ~1.25 B} at daylight
    // WB) instead of the unit value D_th/D_tr are calibrated against, so
    // flat-region noise read as structure and k_denoise under-engaged.
    // Note (2/(k*a))*sqrt(k*a*(k*v) + ...) == (2/a)*sqrt(a*v + ...) up to
    // the same 3/8+beta offset in transformed units: the per-site transform
    // lands every channel back on the sensor-domain GAT of v, so the grey
    // sees channel-consistent signal as well as unit noise. Non-Bayer and
    // un-prewhitened inputs have k = 1 and are bit-identical to before.
    auto apply_gat = [&cfg](const Image& img, f32 alpha, f32 beta) {
        Image out(img.h, img.w, 1);
        f32 a_site[2][2], c_site[2][2];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                f32 k = 1.f;
                if (cfg.bayer_mode) {
                    const uint8_t ch = cfg.cfa.p[i][j];
                    if (ch < 3) k = cfg.noise_wb_gain((int)ch);
                }
                const f32 a = k * alpha;
                a_site[i][j] = a;
                c_site[i][j] = 0.375f * a * a + k * k * beta; // 3/8*a^2 + b
            }
        }
        for (int y = 0; y < img.h; ++y) {
            for (int x = 0; x < img.w; ++x) {
                const f32 a = a_site[y & 1][x & 1];
                const f32 v = a * img.at(y, x) + c_site[y & 1][x & 1];
                out.at(y, x) = (2.f / a) * std::sqrt(std::max(0.f, v));
            }
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
            // python-z "Adjust k1 k2 law" (commit 12ce005, 2026-08-31): lerp
            // from ISOTROPIC (1,1) at A=1 to the extremes (1/k_shrink,
            // k_stretch) at A=2. The previous law (1 + A/2*(extreme-1))
            // was already ~25% anisotropic at A=1 -- elongated kernels with
            // ARBITRARY eigenvector orientation on isotropic-gradient
            // pixels, exactly where orientation is meaningless.
            kk1 = (2.f - A) + (A - 1.f) / cfg.k_shrink;
            kk2 = (2.f - A) + (A - 1.f) * cfg.k_stretch;
        } else if (A > 1.95f) {
            kk1 = 1.f / cfg.k_shrink; kk2 = cfg.k_stretch;
        } else {
            kk1 = 1.f; kk2 = 1.f;
        }
        k1 = cfg.k_detail * ((1.f - D) * kk1 + D * cfg.k_denoise);
        k2 = cfg.k_detail * ((1.f - D) * kk2 + D * cfg.k_denoise);
    };

    // python-z order: GAT the raw, then decimate to grey (kernels.py:80, :84).
    // Base (alpha, beta) is python-z's single sensor-domain noise_model pair;
    // the per-site k_c transform inside apply_gat adapts it to the
    // prewhitened raw (a deliberate fix over python-z, which GATs its own
    // k-multiplied raw with the unscaled pair and so under-stabilizes R/B).
    Image vst_raw = apply_gat(raw, cfg.noise_alpha_shared(), cfg.noise_beta_shared());
    Image grey = compute_grey_decimate(vst_raw, cfg.bayer_mode);

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
