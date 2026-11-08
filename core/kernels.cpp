#include "stages.h"
#include "parallel.h"
#include "linalg.h"
#ifdef __APPLE__
#include "metal_gpu.h"
#endif

namespace hhsr {

static CovField estimate_kernels_cpu(const Image& raw, const Config& cfg);

CovField estimate_kernels(const Image& raw, const Config& cfg) {
    // GCam ComputeGuideImage o_gauss_cov: build the merge kernel from the GCam structure
    // tensor instead of Alg.5. Runs on the host (CPU) on both platforms for now; a Metal
    // twin is TODO. Falls through to the standard estimator if it returns empty.
    if (cfg.robustness_gcam_enabled && cfg.gcam_sabre_kernel) {
        CovField k = compute_gcam_kernel_cov(raw, cfg);
        if (k.h > 0 && k.w > 0) return k;
    }
#ifdef __APPLE__
    // bran (1.4) pipeline: the CPU estimator, because 1.4's GAT is float64 (alpha/beta
    // reach the CUDA kernel as a tuple of Python floats) and Metal has no double.
    if (cfg.bran14_kernels) return estimate_kernels_cpu(raw, cfg);
    // Metal GPU only — same Alg. 5 math as the CPU path below (golden reference).
    CovField gpu = estimate_kernels_metal(raw, cfg);
    if (gpu.h > 0 && gpu.w > 0) return gpu;
    return CovField();
#else
    return estimate_kernels_cpu(raw, cfg);
#endif
}

static CovField estimate_kernels_cpu(const Image& raw, const Config& cfg) {
    {
#if defined(__clang__)
    // 1.4 is compiled without fastmath: no a*b+c contraction (clang contracts by default).
#pragma clang fp contract(off)
#endif
    // 1.4 kernels.py estimate_kernels: preprocessing "gat" is applied to the RAW MOSAIC,
    // per CFA site with that site's colour alpha/beta (utils_image.cuda_GAT), and only
    // THEN decimated to grey -- the GAT is a square root, so the order changes the
    // gradients. VST = alpha*v + 3/8*alpha^2 + beta (1.4's evaluation order), clamped at
    // 0, out = 2/alpha * sqrt(VST). alpha/beta per colour = the DNG profile scaled to
    // the raw's white balance (Config::noise_wb_gain), float32.
    auto apply_gat_raw = [](const Image& img, const Config& cfg) {
        Image out(img.h, img.w, 1);
        if (cfg.bran14_kernels && cfg.bayer_mode && cfg.noise14_valid) {
            // 1.4 cuda_GAT exactly: i = px + py * (3 - 2 px) picks R, G1, G2, B from the
            // R, G1, B, G2 profile (the image is RGGB after cfa_to_rggb); alpha/beta are
            // float64, so VST and the sqrt are double, stored float32.
            const double* A = cfg.noise14_alpha;
            const double* B = cfg.noise14_beta;
            parallel_rows(img.h, cfg.num_threads, [&](int y) {
                const int py = y & 1;
                for (int x = 0; x < img.w; ++x) {
                    const int px = x & 1;
                    const int i = px + py * (3 - 2 * px);
                    const double a = A[i], b = B[i];
                    double vst = a * (double)img.at(y, x) + 3.0 / 8 * a * a + b;
                    vst = (0.0 > vst) ? 0.0 : vst;            // max(0, VST)
                    out.at(y, x) = (float)(2 / a * std::sqrt(vst));
                }
            });
            return out;
        }
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
            if (cfg.bran14_kernels && !cfg.bran14_flat_kernel_guard) {
                // 1.4 exactly: no guard, A = NaN on a flat patch propagates.
                kk1 = (2.f - A) + (A - 1.f) / cfg.k_shrink;
                kk2 = (2.f - A) + (A - 1.f) * cfg.k_stretch;
            } else if (l1 + l2 > 0.f && std::isfinite(A)) {
                kk1 = (2.f - A) + (A - 1.f) / cfg.k_shrink;
                kk2 = (2.f - A) + (A - 1.f) * cfg.k_stretch;
            } else { kk1 = 1.f; kk2 = 1.f; }
        }
        k1 = cfg.k_detail * ((1.f - D) * kk1 + D * cfg.k_denoise);
        k2 = cfg.k_detail * ((1.f - D) * kk2 + D * cfg.k_denoise);
    };

    if (cfg.bran14_kernels && cfg.bayer_mode) {
        // bran (1.4) kernels.py estimate_kernels with Numba's typing, step for step:
        //  - cuda_decimate_to_grey: c = 0 (int) += float32 -> float64 sum, / 4, stored f32;
        //  - torch conv2d [-.5,.5]/[.5,.5] then [.5;.5]/[-.5;.5] (groups=2): each tap
        //    product is exact, so gx = .5*t0 + .5*t0', t0 = -.5a + .5b; gy = -.5*t1 + .5*t1',
        //    t1 = .5a + .5b -- i.e. gy rounds as (bl + br) - (tl + tr);
        //  - structure tensor float32 (cuda.local f32 array);
        //  - get_real_polyroots_2 with int a = 1: 4*a*c and -b + sqrt(delta) are float64,
        //    the |r1| >= |r2| test is on the float64 roots, stored float32;
        //  - get_eigen_vect_2x2 float32;
        //  - compute_k: 1 + sqrt(f32) promotes A to float64; D, k1, k2 and k in float64
        //    with the Python-float k_detail / k_denoise / D_th / D_tr; k stored float32;
        //  - covs = k^2 e e^T in float32. No flat-patch guard (A = NaN propagates) unless
        //    Config::bran14_flat_kernel_guard.
        const Image g = apply_gat_raw(raw, cfg);
        const int gh = g.h / 2, gw = g.w / 2;
        Image grey(gh, gw, 1);
        parallel_rows(gh, cfg.num_threads, [&](int y) {
            for (int x = 0; x < gw; ++x) {
                double c = 0.0;
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j) c += (double)g.at(2 * y + i, 2 * x + j);
                grey.at(y, x) = (f32)(c / 4);
            }
        });
        const int Gh = gh - 1, Gw = gw - 1;
        Image grad(std::max(0, Gh), std::max(0, Gw), 2);
        parallel_rows(Gh, cfg.num_threads, [&](int y) {
            for (int x = 0; x < Gw; ++x) {
                const f32 tl = grey.at(y, x), tr = grey.at(y, x + 1);
                const f32 bl = grey.at(y + 1, x), br = grey.at(y + 1, x + 1);
                const f32 t0 = (-0.5f * tl) + (0.5f * tr), t0b = (-0.5f * bl) + (0.5f * br);
                const f32 t1 = (0.5f * tl) + (0.5f * tr), t1b = (0.5f * bl) + (0.5f * br);
                grad.at(y, x, 0) = (0.5f * t0) + (0.5f * t0b);
                grad.at(y, x, 1) = (-0.5f * t1) + (0.5f * t1b);
            }
        });
        const double kd = cfg.bran14_k_detail, kn = cfg.bran14_k_denoise;
        const double Dth = cfg.bran14_D_th, Dtr = cfg.bran14_D_tr;
        const double kst = cfg.k_stretch, ksh = cfg.k_shrink;
        const bool iso = (cfg.kernel == KernelShape::Iso);
        const bool hard = (cfg.selection == SelectionLaw::HardThreshold);
        const bool guard = cfg.bran14_flat_kernel_guard;
        CovField covs(gh, gw);
        parallel_rows(gh, cfg.num_threads, [&](int py) {
            for (int px = 0; px < gw; ++px) {
                f32* c = covs.at(py, px);
                if (iso) {
                    c[0] = (f32)kd; c[1] = 0.f; c[2] = 0.f; c[3] = (f32)kd;
                    continue;
                }
                f32 m00 = 0.f, m01 = 0.f, m10 = 0.f, m11 = 0.f;
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        const int x = px - 1 + j, y = py - 1 + i;
                        if (!(0 <= y && y < Gh && 0 <= x && x < Gw)) continue;
                        const f32 gx = grad.at(y, x, 0), gy = grad.at(y, x, 1);
                        m00 += gx * gx; m10 += gx * gy; m01 += gx * gy; m11 += gy * gy;
                    }
                }
                // get_eigen_val_2x2 / get_real_polyroots_2
                const f32 b = -(m00 + m11);
                const f32 cc = m00 * m11 - m01 * m10;
                double delta = (double)(b * b) - 4.0 * (double)cc;
                delta = (0.0 > delta) ? 0.0 : delta;             // max(delta, 0)
                const double r1 = (-(double)b + std::sqrt(delta)) / 2.0;
                const double r2 = (-(double)b - std::sqrt(delta)) / 2.0;
                f32 l[2];
                if (std::fabs(r1) >= std::fabs(r2)) { l[0] = (f32)r1; l[1] = (f32)r2; }
                else                                { l[0] = (f32)r2; l[1] = (f32)r1; }
                // get_eigen_vect_2x2
                f32 e1[2], e2[2];
                if (m01 == 0.f && m00 == m11) {
                    e1[0] = 1.f; e1[1] = 0.f; e2[0] = 0.f; e2[1] = 1.f;
                } else {
                    e1[0] = m00 + m01 - l[1];
                    e1[1] = m10 + m11 - l[1];
                    if (e1[0] == 0.f) {
                        e1[1] = 1.f; e2[0] = 1.f; e2[1] = 0.f;
                    } else if (e1[1] == 0.f) {
                        e1[0] = 1.f; e2[0] = 0.f; e2[1] = 1.f;
                    } else {
                        const f32 nrm = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1]);
                        e1[0] /= nrm; e1[1] /= nrm;
                        const double sign = std::copysign(1.0, (double)e1[0]);
                        e2[1] = std::fabs(e1[0]);
                        e2[0] = (f32)(-(double)e1[1] * sign);
                    }
                }
                // compute_k
                const f32 l1 = l[0], l2 = l[1];
                const double A = 1.0 + (double)std::sqrt((l1 - l2) / (l1 + l2));
                double D = 1.0 - (double)std::sqrt(l1) / Dtr + Dth;
                D = (0.0 > D) ? 0.0 : D;                          // max(0, D)
                D = (D < 1.0) ? D : 1.0;                          // min(1, D)
                double k1, k2;
                if (hard) {
                    if (A > 1.95) { k1 = 1.0 / ksh; k2 = kst; } else { k1 = 1.0; k2 = 1.0; }
                } else if (guard && !(std::isfinite(A) && (l1 + l2) > 0.f)) {
                    k1 = 1.0; k2 = 1.0;
                } else {
                    k1 = (2 - A) + (A - 1) / ksh;
                    k2 = (2 - A) + (A - 1) * kst;
                }
                const f32 ka = (f32)(kd * ((1 - D) * k1 + D * kn));
                const f32 kb = (f32)(kd * ((1 - D) * k2 + D * kn));
                const f32 k1s = ka * ka, k2s = kb * kb;
                c[0] = k1s * e1[0] * e1[0] + k2s * e2[0] * e2[0];
                c[1] = k1s * e1[0] * e1[1] + k2s * e2[0] * e2[1];
                c[2] = c[1];
                c[3] = k1s * e1[1] * e1[1] + k2s * e2[1] * e2[1];
            }
        });
        return covs;
    }

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
    }
}

} // namespace hhsr
