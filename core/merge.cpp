#include "stages.h"
#include "parallel.h"

namespace hhsr {

namespace {

static inline f32 denoise_power_merge(f32 r_acc, f32 power_max, f32 max_frame_count) {
    return (r_acc <= max_frame_count) ? power_max : 1.f;
}

static inline int denoise_range_merge(f32 r_acc, int rad_max, f32 max_frame_count) {
    return (r_acc <= max_frame_count) ? rad_max : 1;
}

static inline bool interp_inv_cov(const CovField& covs, f32 kmap_i, f32 kmap_j,
                                  f32& ixx, f32& ixy, f32& iyy) {
    f32 frac_x = kmap_j - std::floor(kmap_j);
    f32 frac_y = kmap_i - std::floor(kmap_i);
    int fx = std::max((int)kmap_j, 0), fy = std::max((int)kmap_i, 0);
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
    f32 det = xx * yy - xy * xy;
    
    // If the ellipse collapses (area approaches 0), fall back to an isotropic circular filter
    if (std::abs(det) > 1e-10f) {
        f32 inv = 1.f / det;
        ixx = inv * yy;
        ixy = -inv * xy;
        iyy = inv * xx;
    } else {
        ixx = 1.f;
        ixy = 0.f;
        iyy = 1.f;
    }
    return true;
}

// Alg. 4 — matches handheld_super_resolution/merge.py accumulate().
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
            const f32 lr_x = (hr_j + 0.5f) / scale;
            const f32 lr_y = (hr_i + 0.5f) / scale;

            const int px = (int)(lr_x / (f32)tile_size);
            const int py = (int)(lr_y / (f32)tile_size);
            const int tpy = std::min(py, flow.ny - 1);
            const int tpx = std::min(px, flow.nx - 1);
            const f32 flowx = flow.dx(tpy, tpx);
            const f32 flowy = flow.dy(tpy, tpx);

            // Robustness map is at guide resolution (half-size in bayer mode),
            // matching Python's cuda_robustness_threshold which operates on the guide image.
            const f32 rob_scale = cfg.bayer_mode ? 0.5f : 1.0f;
            const int i_r = std::min((int)(lr_y * rob_scale), robustness.h - 1);
            const int j_r = std::min((int)(lr_x * rob_scale), robustness.w - 1);
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
                    kmap_j = lr_mov_x / 2.f - 0.5f;
                    kmap_i = lr_mov_y / 2.f - 0.5f;
                } else {
                    kmap_j = lr_mov_x - 0.5f;
                    kmap_i = lr_mov_y - 0.5f;
                }
                if (!interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy)) continue;
            }

            const int center_j = (int)lr_mov_x;
            const int center_i = (int)lr_mov_y;
            const f32 lr_mov_j = lr_mov_x - 0.5f;
            const f32 lr_mov_i = lr_mov_y - 0.5f;

            f32 val[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int j = center_j + dj;
                    const int i = center_i + di;
                    if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

                    const int channel = cfg.bayer_mode ? cfg.cfa.p[i & 1][j & 1] : 0;
                    const f32 c = img.at(i, j);

                    const f32 dist_x = (f32)j - lr_mov_j;
                    const f32 dist_y = (f32)i - lr_mov_i;
                    f32 z;
                    if (iso) z = 2.f * (dist_x * dist_x + dist_y * dist_y);
                    else     z = ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
                    z = std::max(0.f, z);
                    f32 w = std::exp(-0.5f * z);
                    if (w == 0.0f && di == 0 && dj == 0) w = 1.0f;

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

static inline int python_round(float x) {
    float floor_x = std::floor(x);
    float frac = x - floor_x;
    if (frac > 0.5f) return (int)floor_x + 1;
    if (frac < 0.5f) return (int)floor_x;
    int a = (int)floor_x;
    return (a % 2 == 0) ? a : a + 1;
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
            const f32 coarse_x = hr_j / scale;
            const f32 coarse_y = hr_i / scale;

            f32 local_acc_r = 0.f;
            f32 additional_denoise_power = 1.f;
            int rad = 1;
            if (robustness_denoise && acc_rob) {
                // acc_rob is accumulated guide-res R maps — half-size in bayer mode
                const f32 rob_scale = cfg.bayer_mode ? 0.5f : 1.0f;
                const int ay = std::min((int)std::nearbyint(coarse_y * rob_scale), acc_rob->h - 1);
                const int ax = std::min((int)std::nearbyint(coarse_x * rob_scale), acc_rob->w - 1);
                local_acc_r = acc_rob->at(std::max(0, ay), std::max(0, ax));
                additional_denoise_power =
                    denoise_power_merge(local_acc_r, max_multiplier, max_frame_count);
                rad = denoise_range_merge(local_acc_r, rad_max, max_frame_count);

            }

            f32 ixx = 0.f, ixy = 0.f, iyy = 0.f;
            if (!iso) {
                f32 kmap_j, kmap_i;
                if (cfg.bayer_mode) {
                    kmap_j = coarse_x / 2.f - 0.5f;
                    kmap_i = coarse_y / 2.f - 0.5f;
                } else {
                    kmap_j = coarse_x - 0.5f;
                    kmap_i = coarse_y - 0.5f;
                }
                if (!interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy)) continue;
            }

            const int center_j = python_round(coarse_x);
            const int center_i = python_round(coarse_y);

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
                    f32 w = std::exp(-0.5f * y);
                    if (w == 0.0f && di == 0 && dj == 0) w = 1.0f;

                    val[channel] += c * w;
                    acc[channel] += w;
                }
            }

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
                     Image& num_band, Image& den_band, int y0, const Config& cfg) {
    accumulate_comp(comp_raw, flow, covs, robustness, tile_size, num_band, den_band, y0, cfg);
}

void merge_ref_band(const Image& ref_raw, const CovField& covs,
                    Image& num_band, Image& den_band, int y0, const Config& cfg,
                    const Image* acc_rob) {
    accumulate_ref(ref_raw, covs, acc_rob, num_band, den_band, y0, cfg);
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

} // namespace hhsr
