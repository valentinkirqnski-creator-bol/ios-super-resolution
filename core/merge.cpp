#include "stages.h"
#include "parallel.h"

namespace hhsr {

namespace {

static inline f32 sample_image_bilinear(const Image& img, f32 y, f32 x, int ch = 0) {
    y = clampf(y, 0.f, (f32)(img.h - 1));
    x = clampf(x, 0.f, (f32)(img.w - 1));
    const int y0 = (int)std::floor(y);
    const int x0 = (int)std::floor(x);
    const int y1 = std::min(y0 + 1, img.h - 1);
    const int x1 = std::min(x0 + 1, img.w - 1);
    const f32 fy = y - y0;
    const f32 fx = x - x0;
    const f32 top = img.at(y0, x0, ch) * (1.f - fx) + img.at(y0, x1, ch) * fx;
    const f32 bot = img.at(y1, x0, ch) * (1.f - fx) + img.at(y1, x1, ch) * fx;
    return top * (1.f - fy) + bot * fy;
}

static void sample_flow_bilinear(const FlowField& flow, int tile_size, f32 grey_x, f32 grey_y,
                                 f32& out_dx, f32& out_dy) {
    const f32 fx = grey_x / (f32)tile_size;
    const f32 fy = grey_y / (f32)tile_size;
    const int px0 = std::min((int)std::floor(fx), flow.nx - 1);
    const int py0 = std::min((int)std::floor(fy), flow.ny - 1);
    const int px1 = std::min(px0 + 1, flow.nx - 1);
    const int py1 = std::min(py0 + 1, flow.ny - 1);
    const f32 wx = fx - px0;
    const f32 wy = fy - py0;

    const f32 dx00 = flow.dx(py0, px0), dy00 = flow.dy(py0, px0);
    const f32 dx10 = flow.dx(py0, px1), dy10 = flow.dy(py0, px1);
    const f32 dx01 = flow.dx(py1, px0), dy01 = flow.dy(py1, px0);
    const f32 dx11 = flow.dx(py1, px1), dy11 = flow.dy(py1, px1);

    const f32 top_dx = dx00 + wx * (dx10 - dx00);
    const f32 bot_dx = dx01 + wx * (dx11 - dx01);
    const f32 top_dy = dy00 + wx * (dy10 - dy00);
    const f32 bot_dy = dy01 + wx * (dy11 - dy01);
    out_dx = top_dx + wy * (bot_dx - top_dx);
    out_dy = top_dy + wy * (bot_dy - top_dy);
}

static f32 comp_merge_weight(f32 frame_r, f32 gate_r, const Config& cfg) {
    if (gate_r < cfg.motion_comp_hard_cutoff) return 0.f;
    const f32 feather = smoothstepf(cfg.motion_feather_low, cfg.motion_feather_high, gate_r);
    return feather * frame_r * gate_r;
}

} // namespace

// Bilinearly interpolate the 2x2 covariance field at (kmap_i, kmap_j) and
// return its inverse [ixx, ixy, iyy]. Returns false if degenerate.
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
    if (det == 0.f) return false;
    f32 inv = 1.f / det;
    ixx =  inv * yy; ixy = -inv * xy; iyy = inv * xx;
    return true;
}

// Core accumulation shared by comp and ref frames (Alg. 4 / Alg. 11).
// num_band/den_band cover global output rows [y0, y0 + band.h).
static void accumulate(const Image& img, const FlowField* flow, const CovField& covs,
                       const CovField* ref_covs, const Image* robustness, const Image* rob_min,
                       int tile_size, Image& num, Image& den, int y0, const Config& cfg, f32 ref_rob) {
    int band_h = num.h, Ws = num.w;
    int lr_h = img.h, lr_w = img.w;
    int nch = cfg.bayer_mode ? 3 : 1;
    bool iso = (cfg.kernel == KernelShape::Iso);
    f32 scale = cfg.scale;
    const bool is_comp = (robustness != nullptr);
    int up = cfg.bayer_mode ? 2 : 1;

    parallel_rows(band_h, cfg.num_threads, [&](int local_i) {
        int hr_i = y0 + local_i;
        for (int hr_j = 0; hr_j < Ws; ++hr_j) {
            f32 lr_x = (hr_j + 0.5f) / scale;
            f32 lr_y = (hr_i + 0.5f) / scale;
            const f32 grey_x = lr_x / up;
            const f32 grey_y = lr_y / up;

            f32 merge_weight = ref_rob;
            f32 kernel_denoise_power = 1.f;
            f32 gate_r = 1.f;
            bool use_iso_kernel = iso;
            if (is_comp) {
                const f32 frame_r = sample_image_bilinear(*robustness, grey_y, grey_x);
                gate_r = frame_r;
                if (rob_min)
                    gate_r = std::min(gate_r, sample_image_bilinear(*rob_min, grey_y, grey_x));
                merge_weight = comp_merge_weight(frame_r, gate_r, cfg);
                if (merge_weight <= 1e-5f) continue;
                use_iso_kernel = iso || gate_r < cfg.motion_iso_threshold;
                const f32 uncertain = 1.f - smoothstepf(cfg.motion_feather_low, cfg.motion_feather_high, gate_r);
                kernel_denoise_power = 1.f + uncertain * std::max(0.f, cfg.motion_kernel_widen_max - 1.f);
            }

            f32 flowx = 0.f, flowy = 0.f;
            if (flow) {
                f32 fdx = 0.f, fdy = 0.f;
                sample_flow_bilinear(*flow, tile_size, grey_x, grey_y, fdx, fdy);
                flowx = fdx * up;
                flowy = fdy * up;
            }

            f32 sample_x = lr_x + flowx;
            f32 sample_y = lr_y + flowy;
            if (flow) {
                if (!(sample_x >= 0 && sample_x < lr_w && sample_y >= 0 && sample_y < lr_h))
                    continue;
            }

            const bool use_ref_kernels =
                is_comp && cfg.merge_comp_with_ref_kernels && ref_covs != nullptr;
            const CovField& active_covs = use_ref_kernels ? *ref_covs : covs;
            const f32 cov_x = use_ref_kernels ? lr_x : sample_x;
            const f32 cov_y = use_ref_kernels ? lr_y : sample_y;

            f32 ixx = 0, ixy = 0, iyy = 0;
            if (!use_iso_kernel) {
                f32 kmap_j, kmap_i;
                if (cfg.bayer_mode) { kmap_j = cov_x / 2 - 0.5f; kmap_i = cov_y / 2 - 0.5f; }
                else                { kmap_j = cov_x - 0.5f;     kmap_i = cov_y - 0.5f; }
                if (!interp_inv_cov(active_covs, kmap_i, kmap_j, ixx, ixy, iyy)) continue;
            }

            int center_j = (int)sample_x, center_i = (int)sample_y;
            f32 lr_mov_j = sample_x - 0.5f, lr_mov_i = sample_y - 0.5f;

            f32 val[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int j = center_j + dj, i = center_i + di;
                    if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

                    int channel = cfg.bayer_mode ? cfg.cfa.p[i & 1][j & 1] : 0;
                    f32 c = img.at(i, j);

                    f32 dist_x = j - lr_mov_j, dist_y = i - lr_mov_i;
                    f32 z;
                    if (use_iso_kernel) z = 2.f * (dist_x * dist_x + dist_y * dist_y);
                    else              z = ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
                    z = std::max(0.f, z) / kernel_denoise_power;
                    f32 w = std::exp(-0.5f * z);

                    val[channel] += w * merge_weight * c;
                    acc[channel] += w * merge_weight;
                }
            }
            for (int ch = 0; ch < nch; ++ch) {
                num.at(local_i, hr_j, ch) += val[ch];
                den.at(local_i, hr_j, ch) += acc[ch];
            }
        }
    });
}

static f32 sample_acc_rob_nearest(const Image& acc_rob, f32 grey_y, f32 grey_x) {
    const int y = std::min((int)std::lround(grey_y), acc_rob.h - 1);
    const int x = std::min((int)std::lround(grey_x), acc_rob.w - 1);
    return acc_rob.at(std::max(0, y), std::max(0, x));
}

static int ref_merge_radius(f32 acc_r, const Config& cfg) {
    if (!cfg.accumulated_robustness_merge_enabled) return 1;
    return (acc_r <= cfg.acc_rob_frame_threshold) ? cfg.acc_rob_rad_max : 1;
}

static f32 ref_cov_denoise_power(f32 acc_r, const Config& cfg) {
    if (!cfg.accumulated_robustness_merge_enabled) return 1.f;
    return (acc_r <= cfg.acc_rob_frame_threshold) ? cfg.acc_rob_cov_multiplier : 1.f;
}

static bool ref_overwrites_comp(f32 acc_r, const Config& cfg) {
    return cfg.accumulated_robustness_merge_enabled &&
           acc_r < cfg.acc_rob_frame_threshold;
}

// Alg. 11: reference accumulation with paper single-frame fallback (§5.2, §6.2).
static void accumulate_ref(const Image& img, const CovField& covs, const Image* acc_rob,
                           Image& num, Image& den, int y0, const Config& cfg) {
    int band_h = num.h, Ws = num.w;
    int lr_h = img.h, lr_w = img.w;
    int nch = cfg.bayer_mode ? 3 : 1;
    bool iso = (cfg.kernel == KernelShape::Iso);
    f32 scale = cfg.scale;
    int up = cfg.bayer_mode ? 2 : 1;

    parallel_rows(band_h, cfg.num_threads, [&](int local_i) {
        int hr_i = y0 + local_i;
        for (int hr_j = 0; hr_j < Ws; ++hr_j) {
            f32 lr_x = (hr_j + 0.5f) / scale;
            f32 lr_y = (hr_i + 0.5f) / scale;
            const f32 grey_x = lr_x / up;
            const f32 grey_y = lr_y / up;

            f32 local_acc_r = 0.f;
            if (acc_rob) local_acc_r = sample_acc_rob_nearest(*acc_rob, grey_y, grey_x);
            const int rad = ref_merge_radius(local_acc_r, cfg);
            const f32 cov_denoise = ref_cov_denoise_power(local_acc_r, cfg);
            const bool overwrite = ref_overwrites_comp(local_acc_r, cfg);

            f32 coarse_x = lr_x, coarse_y = lr_y;
            int center_j = (int)std::lround(coarse_x);
            int center_i = (int)std::lround(coarse_y);

            f32 ixx = 0, ixy = 0, iyy = 0;
            if (!iso) {
                f32 kmap_j, kmap_i;
                if (cfg.bayer_mode) { kmap_j = coarse_x / 2 - 0.5f; kmap_i = coarse_y / 2 - 0.5f; }
                else                { kmap_j = coarse_x - 0.5f;     kmap_i = coarse_y - 0.5f; }
                if (!interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy)) continue;
            }

            f32 val[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
            for (int di = -rad; di <= rad; ++di) {
                for (int dj = -rad; dj <= rad; ++dj) {
                    int j = center_j + dj, i = center_i + di;
                    if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

                    int channel = cfg.bayer_mode ? cfg.cfa.p[i & 1][j & 1] : 0;
                    f32 c = img.at(i, j);

                    f32 dist_x = j - coarse_x, dist_y = i - coarse_y;
                    f32 z;
                    if (iso) z = 2.f * (dist_x * dist_x + dist_y * dist_y);
                    else     z = ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
                    z = std::max(0.f, z) / cov_denoise;
                    f32 w = std::exp(-0.5f * z);

                    val[channel] += w * c;
                    acc[channel] += w;
                }
            }

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

void merge_comp_band(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                     const CovField& ref_covs, const Image& robustness, const Image* rob_min,
                     int tile_size, Image& num_band, Image& den_band, int y0, const Config& cfg) {
    accumulate(comp_raw, &flow, covs, &ref_covs, &robustness, rob_min, tile_size,
               num_band, den_band, y0, cfg, 1.f);
}

void merge_ref_band(const Image& ref_raw, const CovField& covs,
                    Image& num_band, Image& den_band, int y0, const Config& cfg,
                    const Image* acc_rob) {
    accumulate_ref(ref_raw, covs, acc_rob, num_band, den_band, y0, cfg);
}

void merge_comp(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                const CovField& ref_covs, const Image& robustness, const Image* rob_min,
                int tile_size, Image& num, Image& den, const Config& cfg) {
    merge_comp_band(comp_raw, flow, covs, ref_covs, robustness, rob_min, tile_size,
                    num, den, 0, cfg);
}

void merge_ref(const Image& ref_raw, const CovField& covs,
               Image& num, Image& den, const Config& cfg,
               const Image* acc_rob) {
    merge_ref_band(ref_raw, covs, num, den, 0, cfg, acc_rob);
}

} // namespace hhsr
