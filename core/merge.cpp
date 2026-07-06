#include "stages.h"
#include "parallel.h"

namespace hhsr {

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
                       const Image* robustness, int tile_size,
                       Image& num, Image& den, int y0, const Config& cfg, f32 ref_rob) {
    int band_h = num.h, Ws = num.w;
    int lr_h = img.h, lr_w = img.w;
    int nch = cfg.bayer_mode ? 3 : 1;
    bool iso = (cfg.kernel == KernelShape::Iso);
    f32 scale = cfg.scale;
    // Alignment runs on grey images (half raw resolution for bayer). Flow tiles
    // and flow displacements are in grey-pixel units, so convert to/from raw.
    int up = cfg.bayer_mode ? 2 : 1;

    parallel_rows(band_h, cfg.num_threads, [&](int local_i) {
        int hr_i = y0 + local_i;
        for (int hr_j = 0; hr_j < Ws; ++hr_j) {
            f32 lr_x = (hr_j + 0.5f) / scale;
            f32 lr_y = (hr_i + 0.5f) / scale;

            f32 flowx = 0.f, flowy = 0.f;
            if (flow) {
                // Index the flow tile in grey coordinates.
                int px = std::min((int)((lr_x / up) / tile_size), flow->nx - 1);
                int py = std::min((int)((lr_y / up) / tile_size), flow->ny - 1);
                // Grey-pixel displacement -> raw-pixel displacement.
                flowx = flow->dx(py, px) * up;
                flowy = flow->dy(py, px) * up;
            }

            f32 local_r = ref_rob;
            if (robustness) {
                // Robustness is stored at grey resolution (raw / up).
                int i_r = std::min((int)(lr_y / up), robustness->h - 1);
                int j_r = std::min((int)(lr_x / up), robustness->w - 1);
                local_r = robustness->at(i_r, j_r);
            }

            f32 lr_mov_x = lr_x + flowx;
            f32 lr_mov_y = lr_y + flowy;
            if (!(lr_mov_x >= 0 && lr_mov_x < lr_w && lr_mov_y >= 0 && lr_mov_y < lr_h))
                continue;

            f32 ixx = 0, ixy = 0, iyy = 0;
            if (!iso) {
                f32 kmap_j, kmap_i;
                if (cfg.bayer_mode) { kmap_j = lr_mov_x / 2 - 0.5f; kmap_i = lr_mov_y / 2 - 0.5f; }
                else                { kmap_j = lr_mov_x - 0.5f;     kmap_i = lr_mov_y - 0.5f; }
                if (!interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy)) continue;
            }

            int center_j = (int)lr_mov_x, center_i = (int)lr_mov_y;
            f32 lr_mov_j = lr_mov_x - 0.5f, lr_mov_i = lr_mov_y - 0.5f;

            f32 val[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int j = center_j + dj, i = center_i + di;
                    if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

                    int channel = cfg.bayer_mode ? cfg.cfa.p[i & 1][j & 1] : 0;
                    f32 c = img.at(i, j);

                    f32 dist_x = j - lr_mov_j, dist_y = i - lr_mov_i;
                    f32 z;
                    if (iso) z = 2.f * (dist_x * dist_x + dist_y * dist_y);
                    else     z = ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
                    z = std::max(0.f, z);
                    f32 w = std::exp(-0.5f * z);

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

void merge_comp_band(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                     const Image& robustness, int tile_size,
                     Image& num_band, Image& den_band, int y0, const Config& cfg) {
    accumulate(comp_raw, &flow, covs, &robustness, tile_size, num_band, den_band, y0, cfg, 1.f);
}

void merge_ref_band(const Image& ref_raw, const CovField& covs,
                    Image& num_band, Image& den_band, int y0, const Config& cfg) {
    accumulate(ref_raw, nullptr, covs, nullptr, 1, num_band, den_band, y0, cfg, 1.f);
}

void merge_comp(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                const Image& robustness, int tile_size,
                Image& num, Image& den, const Config& cfg) {
    merge_comp_band(comp_raw, flow, covs, robustness, tile_size, num, den, 0, cfg);
}

void merge_ref(const Image& ref_raw, const CovField& covs,
               Image& num, Image& den, const Config& cfg) {
    merge_ref_band(ref_raw, covs, num, den, 0, cfg);
}

} // namespace hhsr
