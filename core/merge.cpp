#include "stages.h"
#include "parallel.h"
#include "linalg.h"
#ifdef __APPLE__
#include "metal_gpu.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace hhsr {

namespace {

static inline f32 denoise_power_merge(f32 r_acc, f32 power_max, f32 max_frame_count) {
    return (r_acc <= max_frame_count) ? power_max : 1.f;
}

static inline int denoise_range_merge(f32 r_acc, int rad_max, f32 max_frame_count) {
    return (r_acc <= max_frame_count) ? rad_max : 1;
}

// Guard against singular/near-singular covariance inversions producing
// infinitely sharp kernels. That can leave R/B denominators at zero while G
// receives weight, showing up as green or black speckles.
static inline void soften_inv_cov(f32& ixx, f32& ixy, f32& iyy) {
    constexpr f32 k_max_abs = 32.f;
    f32 m = std::max(std::fabs(ixx), std::max(std::fabs(iyy), std::fabs(ixy)));
    if (!(m > k_max_abs) || !std::isfinite(m)) {
        if (!std::isfinite(ixx) || !std::isfinite(ixy) || !std::isfinite(iyy)) {
            ixx = 2.f;
            ixy = 0.f;
            iyy = 2.f;
        }
        return;
    }
    f32 s = k_max_abs / m;
    ixx *= s;
    ixy *= s;
    iyy *= s;
}

static inline int cuda_round_to_int(f32 x) {
    return (int)std::lround(x);
}

// Bilinear cov sample + invert.
// ref (accumulate_ref): floor indices + modf fracs; invert_2x2 → I on singular.
// comp (accumulate): int() indices + modf fracs; raw 1/det.
static inline void interp_inv_cov(const CovField& covs, f32 kmap_i, f32 kmap_j,
                                  f32& ixx, f32& ixy, f32& iyy, bool raw_det) {
    // math.modf: fractional part keeps sign of value
    f32 frac_x = kmap_j - std::trunc(kmap_j);
    f32 frac_y = kmap_i - std::trunc(kmap_i);
    int fx, fy;
    if (raw_det) {
        // Python accumulate: floor_x = int(max(math.floor(grey_pos), 0))
        fx = std::max((int)std::floor(kmap_j), 0);
        fy = std::max((int)std::floor(kmap_i), 0);
    } else {
        // Python accumulate_ref: floor_x = int(max(math.floor(grey_pos), 0))
        fx = std::max((int)std::floor(kmap_j), 0);
        fy = std::max((int)std::floor(kmap_i), 0);
    }
    int cx = std::min(fx + 1, covs.w - 1), cy = std::min(fy + 1, covs.h - 1);

    const f32* tl = covs.at(fy, fx);
    const f32* tr = covs.at(fy, cx);
    const f32* bl = covs.at(cy, fx);
    const f32* br = covs.at(cy, cx);

    auto lerp2 = [&](int idx) {
        f32 top = tl[idx] + frac_x * (tr[idx] - tl[idx]);
        f32 bot = bl[idx] + frac_x * (br[idx] - bl[idx]);
        return top + frac_y * (bot - top);
    };
    f32 xx = lerp2(0), xy = lerp2(1), yy = lerp2(3);
    if (raw_det) {
        f32 det = xx * yy - xy * xy;
        if (std::fabs(det) > 1e-10f) {
            f32 inv_det = 1.f / det;
            ixx =  inv_det * yy;
            ixy = -inv_det * xy;
            iyy =  inv_det * xx;
        } else {
            ixx = 1.f;
            ixy = 0.f;
            iyy = 1.f;
        }
    } else {
        invert_sym_2x2(xx, xy, yy, ixx, ixy, iyy);
    }
    soften_inv_cov(ixx, ixy, iyy);
}

// Alg. 4 — matches handheld_super_resolution/merge.py accumulate().
// On Apple, merge_comp_band_metal runs the same math on GPU.
static void accumulate_comp(const Image& img, const FlowField& flow, const CovField& covs,
                            const Image& robustness, int tile_size,
                            Image& num, Image& den, int y0, const Config& cfg) {
    const int band_h = num.h, Ws = num.w;
    const int lr_h = img.h, lr_w = img.w;
    const int nch = cfg.bayer_mode ? 3 : 1;
    const bool iso = (cfg.kernel == KernelShape::Iso);
    const f32 scale = cfg.scale;

    parallel_rows(band_h, cfg.num_threads, [&](int local_i) {
        const int hr_i = y0 + local_i;
        for (int hr_j = 0; hr_j < Ws; ++hr_j) {
            // Python accumulate(): coarse_ref_sub_pos = output_pixel / scale.
            const f32 lr_x = (f32)hr_j / scale;
            const f32 lr_y = (f32)hr_i / scale;

            // Python: px = int(lr_x // tile_size); no clamp on flow tile index
            const int px = (int)(lr_x / (f32)tile_size);
            const int py = (int)(lr_y / (f32)tile_size);
            const f32 flowx = flow.dx(py, px);
            const f32 flowy = flow.dy(py, px);

            int i_r, j_r;
            if (cfg.bayer_mode) {
                i_r = std::min(std::max(cuda_round_to_int((lr_y - 0.5f) / 2.f), 0),
                               robustness.h - 1);
                j_r = std::min(std::max(cuda_round_to_int((lr_x - 0.5f) / 2.f), 0),
                               robustness.w - 1);
            } else {
                i_r = std::min(std::max(cuda_round_to_int(lr_y), 0), robustness.h - 1);
                j_r = std::min(std::max(cuda_round_to_int(lr_x), 0), robustness.w - 1);
            }
            const f32 local_r = robustness.at(i_r, j_r);

            const f32 lr_mov_x = lr_x + flowx;
            const f32 lr_mov_y = lr_y + flowy;
            if (!(lr_mov_x >= 0.f && lr_mov_x < (f32)lr_w &&
                  lr_mov_y >= 0.f && lr_mov_y < (f32)lr_h))
                continue;

            f32 ixx = 0.f, ixy = 0.f, iyy = 0.f;
            if (!iso) {
                f32 kmap_j, kmap_i;
                if (cfg.bayer_mode) {
                    // Python: grey_pos = (patch_center_pos - 0.5) / 2.
                    kmap_j = (lr_mov_x - 0.5f) / 2.f;
                    kmap_i = (lr_mov_y - 0.5f) / 2.f;
                } else {
                    // Python: grey_pos is exactly the coarse/warped grid.
                    kmap_j = lr_mov_x;
                    kmap_i = lr_mov_y;
                }
                interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy, /*raw_det=*/true);
            }

            const int center_j = cuda_round_to_int(lr_mov_x);
            const int center_i = cuda_round_to_int(lr_mov_y);

            f32 val[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int j = center_j + dj;
                    const int i = center_i + di;
                    if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

                    const int channel = cfg.bayer_mode ? cfg.cfa.p[i & 1][j & 1] : 0;
                    const f32 c = img.at(i, j);

                    const f32 dist_x = (f32)j - lr_mov_x;
                    const f32 dist_y = (f32)i - lr_mov_y;
                    f32 z;
                    if (iso) z = 2.f * (dist_x * dist_x + dist_y * dist_y);
                    else     z = ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
                    z = std::max(0.f, z);
                    const f32 w = std::exp(-0.5f * z);

                    val[channel] += w * local_r * c;
                    acc[channel] += w * local_r;
                }
            }
            for (int ch = 0; ch < nch; ++ch) {
                num.at(local_i, hr_j, ch) += val[ch];
                den.at(local_i, hr_j, ch) += acc[ch];
            }
        }
    });
}

// Alg. 11 — matches handheld_super_resolution/merge.py accumulate_ref().
static void accumulate_ref(const Image& img, const CovField& covs, const Image* acc_rob,
                           Image& num, Image& den, int y0, const Config& cfg) {
    const int band_h = num.h, Ws = num.w;
    const int lr_h = img.h, lr_w = img.w;
    const int nch = cfg.bayer_mode ? 3 : 1;
    const bool iso = (cfg.kernel == KernelShape::Iso);
    const f32 scale = cfg.scale;

    const bool robustness_denoise = cfg.accumulated_robustness_denoiser_enabled;
    const int rad_max = (int)cfg.acc_rob_rad_max;
    const f32 max_multiplier = cfg.acc_rob_max_multiplier;
    const f32 max_frame_count = cfg.acc_rob_max_frame_count;

    parallel_rows(band_h, cfg.num_threads, [&](int local_i) {
        const int hr_i = y0 + local_i;
        for (int hr_j = 0; hr_j < Ws; ++hr_j) {
            // Python: coarse_ref_sub_pos = output_pixel / scale  (no +0.5)
            const f32 coarse_x = (f32)hr_j / scale;
            const f32 coarse_y = (f32)hr_i / scale;

            f32 local_acc_r = 0.f;
            f32 additional_denoise_power = 1.f;
            int rad = 1;
            if (robustness_denoise && acc_rob) {
                // Python: acc_rob[min(round(coarse_y), h-1), min(round(coarse_x), w-1)]
                // (high clamp only — no max(0,·))
                f32 acc_y = coarse_y;
                f32 acc_x = coarse_x;
                if (cfg.bayer_mode) {
                    acc_y = (coarse_y - 0.5f) / 2.f;
                    acc_x = (coarse_x - 0.5f) / 2.f;
                }
                const int ay = std::min(std::max(cuda_round_to_int(acc_y), 0), acc_rob->h - 1);
                const int ax = std::min(std::max(cuda_round_to_int(acc_x), 0), acc_rob->w - 1);
                local_acc_r = acc_rob->at(ay, ax);
                additional_denoise_power =
                    denoise_power_merge(local_acc_r, max_multiplier, max_frame_count);
                rad = denoise_range_merge(local_acc_r, rad_max, max_frame_count);
            }

            f32 ixx = 0.f, ixy = 0.f, iyy = 0.f;
            if (!iso) {
                f32 kmap_j, kmap_i;
                if (cfg.bayer_mode) {
                    // Python: grey_pos = (coarse - 0.5) / 2
                    kmap_j = (coarse_x - 0.5f) / 2.f;
                    kmap_i = (coarse_y - 0.5f) / 2.f;
                } else {
                    // Python: grey_pos = coarse  (no -0.5)
                    kmap_j = coarse_x;
                    kmap_i = coarse_y;
                }
                interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy, /*raw_det=*/false);
            }

            // Python: center = round(coarse)
            const int center_j = cuda_round_to_int(coarse_x);
            const int center_i = cuda_round_to_int(coarse_y);

            f32 val[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
            for (int di = -rad; di <= rad; ++di) {
                for (int dj = -rad; dj <= rad; ++dj) {
                    const int j = center_j + dj;
                    const int i = center_i + di;
                    if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

                    const int channel = cfg.bayer_mode ? cfg.cfa.p[i & 1][j & 1] : 0;
                    const f32 c = img.at(i, j);

                    const f32 dist_x = (f32)j - coarse_x;
                    const f32 dist_y = (f32)i - coarse_y;
                    f32 y;
                    if (iso) y = std::max(0.f, 2.f * (dist_x * dist_x + dist_y * dist_y));
                    else     y = std::max(0.f, ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y +
                                                 iyy * dist_y * dist_y);
                    y /= additional_denoise_power;
                    const f32 w = std::exp(-0.5f * y);

                    val[channel] += c * w;
                    acc[channel] += w;
                }
            }

            // Python: overwrite when robustness_denoise and local_acc_r < max_frame_count
            const bool overwrite =
                robustness_denoise && acc_rob && local_acc_r < max_frame_count;
            for (int ch = 0; ch < nch; ++ch) {
                if (overwrite) {
                    num.at(local_i, hr_j, ch) = val[ch];
                    den.at(local_i, hr_j, ch) = acc[ch];
                } else {
                    num.at(local_i, hr_j, ch) += val[ch];
                    den.at(local_i, hr_j, ch) += acc[ch];
                }
            }
        }
    });
}

} // namespace

void merge_comp_band(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                     const Image& robustness, int tile_size,
                     Image& num_band, Image& den_band, int y0, const Config& cfg,
                     int frame_id) {
#ifdef __APPLE__
    // Metal GPU only — same Alg. 4 math as accumulate_comp (incl. per-pixel robustness).
    if (!merge_comp_band_metal(comp_raw, flow, covs, robustness, tile_size,
                               num_band, den_band, y0, cfg, frame_id)) {
        return;
    }
#else
    (void)frame_id;
    accumulate_comp(comp_raw, flow, covs, robustness, tile_size, num_band, den_band, y0, cfg);
#endif
}

void merge_ref_band(const Image& ref_raw, const CovField& covs,
                    Image& num_band, Image& den_band, int y0, const Config& cfg,
                    const Image* acc_rob) {
#ifdef __APPLE__
    // Metal GPU only — same Alg. 11 math. Waits for this band (sync API).
    // pipeline_paths Apple path calls merge_ref_band_metal directly for async overlap.
    if (!merge_ref_band_metal(ref_raw, covs, num_band, den_band, y0, cfg, acc_rob) ||
        !metal_merge_wait_inflight()) {
        return;
    }
#else
    accumulate_ref(ref_raw, covs, acc_rob, num_band, den_band, y0, cfg);
#endif
}

void merge_comp(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                const Image& robustness, int tile_size,
                Image& num, Image& den, const Config& cfg) {
    merge_comp_band(comp_raw, flow, covs, robustness, tile_size, num, den, 0, cfg);
}

void merge_ref(const Image& ref_raw, const CovField& covs,
               Image& num, Image& den, const Config& cfg, const Image* acc_rob) {
    merge_ref_band(ref_raw, covs, num, den, 0, cfg, acc_rob);
}

void accumulate_diag(const Image& num, const Image& den, AccumDiag& d) {
    const int nch = std::min(3, num.c);
    const size_t n = (size_t)num.h * (size_t)num.w;
    for (size_t p = 0; p < n; ++p) {
        ++d.pixels;
        f32 dens[3] = {0, 0, 0};
        for (int ch = 0; ch < nch; ++ch) {
            f32 nv = num.data[p * (size_t)num.c + (size_t)ch];
            f32 dv = den.data[p * (size_t)den.c + (size_t)ch];
            dens[ch] = dv;
            if (!std::isfinite(nv)) ++d.num_nonfinite[ch];
            if (!std::isfinite(dv)) ++d.den_nonfinite[ch];
            else if (dv == 0.f) ++d.den_zero[ch];
            else if (dv > 0.f && dv < 1e-12f) ++d.den_tiny[ch];
        }
        if (nch >= 3) {
            if (dens[0] == 0.f && dens[1] == 0.f && dens[2] == 0.f) ++d.rgb_all_zero;
            else if (dens[0] == 0.f && dens[1] > 0.f && dens[2] == 0.f) ++d.only_green;
        }
    }
}

std::string format_accum_diag(const AccumDiag& d) {
    if (d.pixels == 0) return "accum: empty";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "accum den0 R/G/B=%zu/%zu/%zu onlyG=%zu all0=%zu nanDen=%zu",
        d.den_zero[0], d.den_zero[1], d.den_zero[2],
        d.only_green, d.rgb_all_zero,
        d.den_nonfinite[0] + d.den_nonfinite[1] + d.den_nonfinite[2]);
    return std::string(buf);
}

} // namespace hhsr
