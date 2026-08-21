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
    // Eq. 4's k1/k2. The paper drives the kernel shape CONTINUOUSLY from the
    // structure tensor: "(l1 - l2)/(l1 + l2) is used to drive the desired
    // anisotropy of the kernels (Figure 8)", and Figure 8 samples that axis at
    // 0.1 / 0.5 / 0.9, which only makes sense if the shape varies across the
    // whole range.
    //
    // 460-main kernels.py instead switches at A > 1.95, i.e. anisotropy >
    // 0.9025, and is perfectly round below it. So every one of Figure 8's three
    // sampled edge strengths produced an identical isotropic kernel, and the
    // stretch appeared as an 8:1 jump (k_stretch 4.0 against 1/k_shrink 0.5)
    // at a single threshold.
    //
    // That matters beyond fidelity. Section 5.1.1 gives the anisotropic kernel
    // a specific job -- "they increase the algorithm's tolerance for small
    // misalignments and uneven coverage around edges" -- and with the switch
    // that tolerance only existed on near-perfect edges. A pole or wire at
    // anisotropy 0.6-0.8 got a round kernel and no tolerance at all, which is
    // exactly the content where misalignment is most visible. The threshold
    // also put a discontinuity in the reconstruction itself: neighbouring
    // pixels straddling 0.9025 were reconstructed with 1:1 and 8:1 kernels.
    //
    // Interpolating on a = sqrt(anisotropy) reproduces BOTH endpoints of the
    // old behaviour exactly -- a = 0 is isotropic, a = 1 is the full stretch --
    // and fills in between. sqrt rather than the raw ratio because the old
    // threshold was expressed on sqrt, so the existing k_stretch/k_shrink
    // tuning keeps its meaning at the top end.
    auto compute_k = [](f32 l1, f32 l2, f32& k1, f32& k2, const Config& cfg) {
        const f32 tr = l1 + l2;
        // Flat patch: both eigenvalues ~0. The old form computed 0/0 -> NaN,
        // and NaN > 1.95 is false, so it fell to isotropic by accident. A
        // continuous blend would propagate the NaN into k1/k2 instead, so the
        // degenerate case is now handled on purpose.
        f32 ratio = (tr > 1e-12f) ? std::max(0.f, (l1 - l2) / tr) : 0.f;
        if (!std::isfinite(ratio)) ratio = 0.f;
        const f32 a = std::sqrt(ratio);   // only the legacy threshold uses this
        f32 D = std::min(1.f, std::max(0.f, 1.f - std::sqrt(l1) / cfg.D_tr + cfg.D_th));
        f32 kk1, kk2;
        if (cfg.kernel_anisotropy_continuous) {
            // Blend on the RATIO itself, which is what the paper says drives
            // anisotropy. Blending on sqrt(ratio) would give a 2.3:1 kernel at
            // ratio 0.1, and at that anisotropy the dominant eigenvector is
            // noise-dominated -- stretching along it smears in an essentially
            // arbitrary direction. The ratio keeps weak gradients nearly round.
            kk1 = 1.f + ratio * (1.f / cfg.k_shrink - 1.f);
            kk2 = 1.f + ratio * (cfg.k_stretch - 1.f);
        } else if (1.f + a > 1.95f) {
            kk1 = 1.f / cfg.k_shrink; kk2 = cfg.k_stretch;
        } else {
            kk1 = 1.f; kk2 = 1.f;
        }
        k1 = cfg.k_detail * ((1.f - D) * kk1 + D * cfg.k_denoise);
        k2 = cfg.k_detail * ((1.f - D) * kk2 + D * cfg.k_denoise);
    };

    Image grey = compute_grey_decimate(raw, cfg.bayer_mode);
    Image vst = apply_gat(grey, cfg.noise_alpha(), cfg.noise_beta());
    Image grad = compute_gradients(vst); // [gh-1, gw-1, 2]

    int H = vst.h, W = vst.w;
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
