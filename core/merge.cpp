#include "stages.h"
#include "parallel.h"
#include "linalg.h"
#ifdef __APPLE__
#include "metal_gpu.h"
#endif
#ifdef _WIN32
#include "d3d11_merge.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace hhsr {

namespace {

// The accumulated-robustness adaptive denoiser (reference-kernel enlargement
// driven by the per-pixel accumulated robustness) was removed: the reference
// merge now always reconstructs with a fixed radius-1 kernel and no denoise
// power, so no extra spatial smoothing is applied on top of the multi-frame
// merge.

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

static inline f32 sample_robustness_bilinear(const Image& robustness, f32 y, f32 x, int ch = 0) {
    if (robustness.h <= 0 || robustness.w <= 0) return 0.f;
    if (ch >= robustness.c) ch = robustness.c - 1;   // 1ch mask -> broadcast
    y = std::min(std::max(y, 0.f), (f32)(robustness.h - 1));
    x = std::min(std::max(x, 0.f), (f32)(robustness.w - 1));
    const int y0 = (int)std::floor(y);
    const int x0 = (int)std::floor(x);
    const int y1 = std::min(y0 + 1, robustness.h - 1);
    const int x1 = std::min(x0 + 1, robustness.w - 1);
    const f32 fy = y - (f32)y0;
    const f32 fx = x - (f32)x0;
    const f32 top = robustness.at(y0, x0, ch) +
                    (robustness.at(y0, x1, ch) - robustness.at(y0, x0, ch)) * fx;
    const f32 bot = robustness.at(y1, x0, ch) +
                    (robustness.at(y1, x1, ch) - robustness.at(y1, x0, ch)) * fx;
    return top + (bot - top) * fy;
}

// Bilinear cov sample + invert.
// ref (accumulate_ref): floor indices + modf fracs; invert_2x2 → I on singular.
// comp (accumulate): int() indices + modf fracs; raw 1/det.
static inline void interp_inv_cov(const CovField& covs, f32 kmap_i, f32 kmap_j,
                                  f32& ixx, f32& ixy, f32& iyy, bool raw_det,
                                  bool no_invert = false, bool soften = true) {
    // math.modf: fractional part keeps sign of value
    f32 frac_x = kmap_j - std::trunc(kmap_j);
    f32 frac_y = kmap_i - std::trunc(kmap_i);
    int fx, fy;
    if (raw_det) {
        // Python accumulate: floor_x = max(int(kmap_j), 0) — no high clamp
        fx = std::max((int)kmap_j, 0);
        fy = std::max((int)kmap_i, 0);
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
    if (no_invert) {
        // GCam merge kernel: the field already holds the PRECISION (inverse covariance);
        // GCam bilinearly interpolates it and uses it directly (texture(kernel_covariance)
        // + UNPACK, no inversion). Return the interpolated precision as-is -- no invert, no
        // soften (GCam does neither).
        ixx = xx; ixy = xy; iyy = yy;
        return;
    }
    if (raw_det) {
        f32 det = xx * yy - xy * xy;
        // merge_match_14 (soften off): 1.4 inverts unguarded, inv_det = 1.0 / det
        // ("invertible by design") -- a flat patch's NaN covariance stays NaN and the
        // z = max(0, z) clamp then gives it weight 1, as in 1.4.
        if (!soften || std::fabs(det) > 1e-10f) {
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
    // soften = false under Config::merge_match_14: 1.4 uses inv_det * cov as-is; the cap
    // at 32 would reshape the sharp anisotropic 1.4 kernels (measured: ~20% of pixels).
    if (soften) soften_inv_cov(ixx, ixy, iyy);
}

// Alg. 4 — matches handheld_super_resolution/merge.py accumulate().
// ---------------------------------------------------------------------------
// bran (1.4) merge.py accumulate(), with Numba's typing (Config::bran14_kernels on a CPU
// merge). scale reaches the kernel as a Python float, so lr = (hr + 0.5) / scale, the
// moved position, the kernel-map position, the covariance lerp (float32 corners, float64
// fractions), the inverse (unguarded 1.0 / det), the distances, z and exp() are all
// float64; val / acc are float32 local arrays (+= in float64, stored float32); num / den
// get float32 adds. The reference goes through the same kernel with a zero flow and
// r = 1 (super_resolution.main: dummy_alignment / dummy_r).

// CPython / Numba float floor division (float_floor_div): fmod-based, not floor(a / b).
static inline double py_floordiv(double a, double b) {
    double mod = std::fmod(a, b);
    double div = (a - mod) / b;
    if (mod != 0.0 && ((b < 0) != (mod < 0))) div -= 1.0;
    double fl;
    if (div != 0.0) {
        fl = std::floor(div);
        if (div - fl > 0.5) fl += 1.0;
    } else {
        fl = std::copysign(0.0, a / b);
    }
    return fl;
}

static void accumulate14_exact(const Image& img, const FlowField* flow, const CovField& covs,
                               const Image* r, int tile_size, Image& num, Image& den, int y0,
                               const Config& cfg) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int band_h = num.h, Ws = num.w;
    const int lr_h = img.h, lr_w = img.w;
    const double scale = (double)cfg.scale;
    const double ts = (double)tile_size;
    parallel_rows(band_h, cfg.num_threads, [&](int local_i) {
        const int hr_i = y0 + local_i;
        for (int hr_j = 0; hr_j < Ws; ++hr_j) {
            const double lr_x = ((double)hr_j + 0.5) / scale;
            const double lr_y = ((double)hr_i + 0.5) / scale;
            float flowx = 0.f, flowy = 0.f;
            if (flow) {
                const int px = (int)py_floordiv(lr_x, ts);
                const int py = (int)py_floordiv(lr_y, ts);
                flowx = flow->dx(py, px);
                flowy = flow->dy(py, px);
            }
            float local_r = 1.f;
            if (r) {
                const int i_r = std::min((int)(py_floordiv(lr_y, 2.0) - 0.5), lr_h / 2 - 1);
                const int j_r = std::min((int)(py_floordiv(lr_x, 2.0) - 0.5), lr_w / 2 - 1);
                local_r = r->at(i_r, j_r);
            }
            const double lr_mov_x = lr_x + (double)flowx;
            const double lr_mov_y = lr_y + (double)flowy;
            if (!(0 <= lr_mov_x && lr_mov_x < lr_w && 0 <= lr_mov_y && lr_mov_y < lr_h)) continue;

            const double kmap_j = lr_mov_x / 2 - 0.5;
            const double kmap_i = lr_mov_y / 2 - 0.5;
            double ip;
            const double frac_x = std::modf(kmap_j, &ip);
            const double frac_y = std::modf(kmap_i, &ip);
            const int floor_x = std::max((int)kmap_j, 0);
            const int floor_y = std::max((int)kmap_i, 0);
            const int ceil_x = std::min(floor_x + 1, covs.w - 1);
            const int ceil_y = std::min(floor_y + 1, covs.h - 1);
            const f32* tr = covs.at(floor_y, floor_x);
            const f32* tl = covs.at(floor_y, ceil_x);
            const f32* br = covs.at(ceil_y, floor_x);
            const f32* bl = covs.at(ceil_y, ceil_x);
            auto lerp = [&](int k) {
                const double top = (double)tr[k] + frac_x * (double)(tl[k] - tr[k]);
                const double bot = (double)br[k] + frac_x * (double)(bl[k] - br[k]);
                return top + frac_y * (bot - top);
            };
            const double cxx = lerp(0), cxy = lerp(1), cyy = lerp(3);
            const double det = cxx * cyy - cxy * cxy;
            const double inv_det = 1.0 / det;
            const double ixx = inv_det * cyy;
            const double ixy = -inv_det * cxy;
            const double iyy = inv_det * cxx;

            const int center_j = (int)lr_mov_x;
            const int center_i = (int)lr_mov_y;
            const double lr_mov_j = lr_mov_x - 0.5;
            const double lr_mov_i = lr_mov_y - 0.5;
            float val[3] = {0.f, 0.f, 0.f}, acc[3] = {0.f, 0.f, 0.f};
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int j = center_j + dj;
                    const int i = center_i + di;
                    if (!(0 <= j && j < lr_w && 0 <= i && i < lr_h)) continue;
                    const int channel = i % 2 + j % 2;      // RGGB hardcoded
                    const float c = img.at(i, j);
                    const double dist_x = (double)j - lr_mov_j;
                    const double dist_y = (double)i - lr_mov_i;
                    double z = ixx * dist_x * dist_x + 2 * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
                    z = (z > 0) ? z : 0.0;                   // max(0, z): NaN -> 0
                    const double w = std::exp(-0.5 * z);
                    val[channel] = (float)((double)val[channel] + w * (double)local_r * (double)c);
                    acc[channel] = (float)((double)acc[channel] + w * (double)local_r);
                }
            }
            for (int ch = 0; ch < 3; ++ch) {
                num.at(local_i, hr_j, ch) += val[ch];
                den.at(local_i, hr_j, ch) += acc[ch];
            }
        }
    });
}

// On Apple, merge_comp_band_metal runs the same math on GPU.
static void accumulate_comp(const Image& img, const FlowField& flow, const CovField& covs,
                            const Image& robustness, int tile_size,
                            Image& num, Image& den, int y0, const Config& cfg,
                            const Mask8* rob8 = nullptr) {
    const int band_h = num.h, Ws = num.w;
    const int lr_h = img.h, lr_w = img.w;
    const int nch = cfg.bayer_mode ? 3 : 1;
    const bool iso = (cfg.kernel == KernelShape::Iso);
    const f32 scale = cfg.scale;
    // gcam_sabre_merge_res: the mask is a compact 8-bit output-grid mask (Mask8),
    // read 1:1 at the output pixel (v/255). Avoids a 48MP fp32 mask in memory.
    const bool use_m8 = (rob8 && !rob8->empty());
    const bool m14 = cfg.merge_match_14;   // 1.4 merge.py conventions (see Config)
    if (cfg.bran14_kernels && m14 && cfg.bayer_mode && !use_m8) {
        accumulate14_exact(img, &flow, covs, &robustness, tile_size, num, den, y0, cfg);
        return;
    }

    parallel_rows(band_h, cfg.num_threads, [&](int local_i) {
        const int hr_i = y0 + local_i;
        for (int hr_j = 0; hr_j < Ws; ++hr_j) {
            // 460 accumulate(): coarse_ref_sub_pos = output_pixel / scale.
            // 1.4 accumulate() (merge_match_14): lr = (hr + 0.5) / scale -- pixel centres.
            const f32 lr_x = m14 ? ((f32)hr_j + 0.5f) / scale : (f32)hr_j / scale;
            const f32 lr_y = m14 ? ((f32)hr_i + 0.5f) / scale : (f32)hr_i / scale;

            // Python: px = int(lr_x // tile_size); no clamp on flow tile index
            const int px = (int)(lr_x / (f32)tile_size);
            const int py = (int)(lr_y / (f32)tile_size);
            // Bilinear between tile centres: block matching gives one vector
            // per tile, and consuming it nearest makes the warp piecewise
            // constant, which rotation turns into a visible tile grid. The
            // mask samples the SAME way, so it still grades the fetch that
            // actually happens. See FlowField::sample_bilinear.
            f32 flowx, flowy;
            if (flow.has_dense()) {
                // Dense per-pixel flow (Config::dense_flow_enabled): the finest
                // correspondence, one vector per pixel. Takes precedence over the
                // tile/affine path; the mask samples it the same way below.
                flow.sample_dense(lr_y, lr_x, flowx, flowy);
            } else if (cfg.affine_flow_enabled && flow.has_affine()) {
                // Per-tile affine: displacement varies across the tile so
                // rotation/yaw/pitch warp correctly. At a tile centre this
                // equals flow.dx/dy(py,px), so flag-off behaviour is unchanged.
                flow.sample_affine(lr_y, lr_x, tile_size, flowx, flowy);
            } else if (cfg.gcam_flow_bilinear_active()) {
                // gcam_sabre_flow_bilinear: smooth per-pixel field (tile vectors
                // interpolated between tile centres) -- the same field the gcam
                // robustness operator warps with (gcam_tile_bilinear), so the mask
                // grades this exact fetch.
                flow.sample_bilinear(lr_y, lr_x, tile_size, flowx, flowy);
            } else {
                flowx = flow.dx(py, px);
                flowy = flow.dy(py, px);
            }

            // Which coordinate space R lives in is decided by R's ACTUAL
            // dimensions, not Config::robustness_raw_resolution_active():
            // the raw-resolution path returns an empty image when the hires
            // ref stats are missing and silently falls through to the
            // guide-resolution path, so the flag can say "raw" while the mask
            // handed to us is guide. Trusting the flag there would sample R
            // at half the correct position everywhere.
            f32 local_r[3];
            if (use_m8) {
                // gcam_sabre_merge_res: compact 8-bit output-grid mask, read 1:1.
                const int iy = std::min(rob8->h - 1, std::max(0, hr_i));
                const int ix = std::min(rob8->w - 1, std::max(0, hr_j));
                const f32 r = rob8->keep(iy, ix);
                local_r[0] = local_r[1] = local_r[2] = r;
            } else {
            // Which coordinate space R lives in is decided by R's ACTUAL
            // dimensions, not Config::robustness_raw_resolution_active():
            // the raw-resolution path returns an empty image when the hires
            // ref stats are missing and silently falls through to the
            // guide-resolution path, so the flag can say "raw" while the mask
            // handed to us is guide. Trusting the flag there would sample R
            // at half the correct position everywhere.
            const bool rob_is_raw = (robustness.h == lr_h && robustness.w == lr_w);
            f32 rob_y = lr_y, rob_x = lr_x;
            if (!rob_is_raw && cfg.bayer_mode) {
                rob_y = (lr_y - 0.5f) / 2.f;
                rob_x = (lr_x - 0.5f) / 2.f;
            }
            // Config::merge_robustness_bilinear documents why this is no longer
            // nearest by default, and what the nearest branch was protecting.
            // Per-channel robustness (ISA-style): one weight per R/G/B. A 1ch
            // mask broadcasts (local_r[*] equal). Indexed below by the raw
            // sample's CFA channel.
            const int rnc = robustness.c;
            // robustness_guide_res_bilinear: the guide-res mask bilinearly
            // upsampled to the merge grid IS this bilinear fetch.
            if (cfg.merge_robustness_bilinear || cfg.robustness_guide_res_bilinear ||
                (cfg.gcam_sabre_bilinear_upscale && !cfg.gcam_sabre_raw_res)) {
                for (int rc = 0; rc < 3; ++rc)
                    local_r[rc] = sample_robustness_bilinear(robustness, rob_y, rob_x, rc);
            } else if (m14) {
                // 1.4: Bayer guide-res mask -> r[min(int(lr_y//2 - 0.5), lr_h//2 - 1)]
                // (floor division, minus 0.5, int() truncation); a raw-res mask (and grey
                // mode) -> r[min(int(lr_y), lr_h - 1)].
                int iy, ix;
                if (!rob_is_raw && cfg.bayer_mode) {
                    iy = (int)(std::floor(lr_y / 2.f) - 0.5f);
                    ix = (int)(std::floor(lr_x / 2.f) - 0.5f);
                } else {
                    iy = (int)lr_y;
                    ix = (int)lr_x;
                }
                iy = std::max(0, std::min(iy, robustness.h - 1));
                ix = std::max(0, std::min(ix, robustness.w - 1));
                for (int rc = 0; rc < 3; ++rc)
                    local_r[rc] = robustness.at(iy, ix, rc < rnc ? rc : rnc - 1);
            } else {
                const int iy = std::min(robustness.h - 1,
                                        std::max(0, (int)std::floor(rob_y + 0.5f)));
                const int ix = std::min(robustness.w - 1,
                                        std::max(0, (int)std::floor(rob_x + 0.5f)));
                for (int rc = 0; rc < 3; ++rc)
                    local_r[rc] = robustness.at(iy, ix, rc < rnc ? rc : rnc - 1);
            }
            }

            f32 lr_mov_x = lr_x + flowx;
            f32 lr_mov_y = lr_y + flowy;
            // Global homography warp-then-refine: the comparison was warped into
            // the reference frame before align(), so the actual comp position is
            // H*(lr + residual_flow). Compose it back here (and the covariance is
            // then fetched at the correct comp position too).
            if (flow.has_global_h) {
                f32 hx, hy;
                apply_homography(flow.global_h, lr_mov_x, lr_mov_y, hx, hy);
                lr_mov_x = hx; lr_mov_y = hy;
            }
            if (cfg.gcam_sabre_kernel) {
                // GCam Merge MirrorUVs: reflect the warped sample back into the frame
                // (inset by the 3x3 footprint, [1, lr-2]) instead of skipping out-of-frame
                // samples, so border pixels still merge. One reflection per side, then
                // clamp. Reads the real mirrored pixel (correct CFA phase).
                auto mir = [](f32 v, f32 hi) {
                    if (v < 1.f) v = 2.f - v;
                    if (v > hi)  v = 2.f * hi - v;
                    return clampf(v, 1.f, hi);
                };
                lr_mov_x = mir(lr_mov_x, (f32)(lr_w - 2));
                lr_mov_y = mir(lr_mov_y, (f32)(lr_h - 2));
            } else if (!(lr_mov_x >= 0.f && lr_mov_x < (f32)lr_w &&
                         lr_mov_y >= 0.f && lr_mov_y < (f32)lr_h)) {
                continue;
            }

            f32 ixx = 0.f, ixy = 0.f, iyy = 0.f;
            if (!iso || m14) {   // 1.4: iso reads the diag(k_detail) covariance too
                f32 kmap_j, kmap_i;
                if (cfg.bayer_mode && m14) {
                    // 1.4 merge.py: kmap = lr_mov/2 - 0.5 (with lr at pixel centres).
                    kmap_j = lr_mov_x / 2.f - 0.5f;
                    kmap_i = lr_mov_y / 2.f - 0.5f;
                } else if (cfg.bayer_mode) {
                    // 460-parity: a half-res grey cell g sits at coarse 2g+0.5,
                    // so g = (x-0.5)/2 (merge.py accumulate, and the ref path
                    // below). Was x/2 - 0.5 -- a 0.25-cell (0.5 raw-px) offset on
                    // every comparison-frame covariance fetch.
                    kmap_j = (lr_mov_x - 0.5f) / 2.f;
                    kmap_i = (lr_mov_y - 0.5f) / 2.f;
                } else {
                    kmap_j = lr_mov_x - 0.5f;
                    kmap_i = lr_mov_y - 0.5f;
                }
                interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy, /*raw_det=*/true,
                               /*no_invert=*/cfg.gcam_sabre_kernel, /*soften=*/!m14);
            }

            // GCam SampleNeighborhoodRBF anchors at floor(coord) and measures the kernel
            // offset from each neighbour's pixel CENTRE (pos+0.5): offset = (j+0.5) - coord.
            // Our default path anchors at round() and uses offset = j - coord (a benign
            // half-pixel). Match GCam exactly on the gcam-kernel path.
            // 1.4 (merge_match_14): center = int(lr_mov) (>= 0 here, so == floor) and
            // dist = j - (lr_mov - 0.5) == (j + 0.5) - lr_mov -- the same as the GCam form.
            const bool gk = cfg.gcam_sabre_kernel;
            const bool fl = gk || m14;
            const f32 ctr = fl ? 0.5f : 0.f;
            const int center_j = fl ? (int)std::floor(lr_mov_x) : cuda_round_to_int(lr_mov_x);
            const int center_i = fl ? (int)std::floor(lr_mov_y) : cuda_round_to_int(lr_mov_y);

            f32 val[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int j = center_j + dj;
                    const int i = center_i + di;
                    if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

                    const int channel = cfg.bayer_mode ? cfg.cfa.p[i & 1][j & 1] : 0;
                    const f32 c = img.at(i, j);

                    const f32 dist_x = ((f32)j + ctr) - lr_mov_x;
                    const f32 dist_y = ((f32)i + ctr) - lr_mov_y;
                    f32 z;
                    if (iso && !m14) z = 2.f * (dist_x * dist_x + dist_y * dist_y);
                    else     z = ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
                    z = std::max(0.f, z);
                    // GCam GetKernelWeight = exp2(-0.5*kd) + 5e-5. On the gcam-kernel path
                    // the stored covariance folds in ln2, so exp(-0.5*z) == exp2(...); add
                    // GCam's kEpsilon so far taps keep a tiny floor (matches the merge).
                    const f32 w = std::exp(-0.5f * z) + (cfg.gcam_sabre_kernel ? 5e-5f : 0.f);

                    const f32 lr_c = local_r[channel < 3 ? channel : 2];
                    val[channel] += w * lr_c * c;
                    acc[channel] += w * lr_c;
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

    (void)acc_rob;  // Adaptive denoiser removed: the reference merge is a fixed
                    // radius-1 kernel with no accumulated-robustness enlargement.
    // merge_match_14: 1.4 merges the reference through merge.py accumulate() with a zero
    // alignment and r = 1 (super_resolution.py), so the reference takes exactly the
    // comparison-frame conventions: pixel-centre lr, kmap = lr/2 - 0.5, raw-det inverse,
    // int() centre with (j + 0.5) - lr distances, iso via the diag(k_detail) covariance.
    const bool m14 = cfg.merge_match_14;
    if (cfg.bran14_kernels && m14 && cfg.bayer_mode) {
        accumulate14_exact(img, nullptr, covs, nullptr, 1, num, den, y0, cfg);
        return;
    }

    parallel_rows(band_h, cfg.num_threads, [&](int local_i) {
        const int hr_i = y0 + local_i;
        for (int hr_j = 0; hr_j < Ws; ++hr_j) {
            // 460: coarse_ref_sub_pos = output_pixel / scale  (no +0.5); 1.4: (hr + 0.5)/scale
            const f32 coarse_x = m14 ? ((f32)hr_j + 0.5f) / scale : (f32)hr_j / scale;
            const f32 coarse_y = m14 ? ((f32)hr_i + 0.5f) / scale : (f32)hr_i / scale;

            const f32 additional_denoise_power = 1.f;
            const int rad = 1;

            f32 ixx = 0.f, ixy = 0.f, iyy = 0.f;
            if (!iso || m14) {
                f32 kmap_j, kmap_i;
                if (m14) {
                    // 1.4 accumulate(): Bayer lr/2 - 0.5, grey lr - 0.5
                    kmap_j = cfg.bayer_mode ? coarse_x / 2.f - 0.5f : coarse_x - 0.5f;
                    kmap_i = cfg.bayer_mode ? coarse_y / 2.f - 0.5f : coarse_y - 0.5f;
                } else if (cfg.bayer_mode) {
                    // Python: grey_pos = (coarse - 0.5) / 2
                    kmap_j = (coarse_x - 0.5f) / 2.f;
                    kmap_i = (coarse_y - 0.5f) / 2.f;
                } else {
                    // Python: grey_pos = coarse  (no -0.5)
                    kmap_j = coarse_x;
                    kmap_i = coarse_y;
                }
                interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy, /*raw_det=*/m14,
                               /*no_invert=*/cfg.gcam_sabre_kernel, /*soften=*/!m14);
            }

            // Python: center = round(coarse). GCam merge kernel / 1.4: floor anchor +
            // (j+0.5) pixel-centre offset (1.4: int(lr) and j - (lr - 0.5)).
            const bool gk = cfg.gcam_sabre_kernel;
            const bool fl = gk || m14;
            const f32 ctr = fl ? 0.5f : 0.f;
            const int center_j = fl ? (int)std::floor(coarse_x) : cuda_round_to_int(coarse_x);
            const int center_i = fl ? (int)std::floor(coarse_y) : cuda_round_to_int(coarse_y);

            f32 val[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
            for (int di = -rad; di <= rad; ++di) {
                for (int dj = -rad; dj <= rad; ++dj) {
                    const int j = center_j + dj;
                    const int i = center_i + di;
                    if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

                    const int channel = cfg.bayer_mode ? cfg.cfa.p[i & 1][j & 1] : 0;
                    const f32 c = img.at(i, j);

                    const f32 dist_x = ((f32)j + ctr) - coarse_x;
                    const f32 dist_y = ((f32)i + ctr) - coarse_y;
                    f32 y;
                    if (iso && !m14) y = std::max(0.f, 2.f * (dist_x * dist_x + dist_y * dist_y));
                    else     y = std::max(0.f, ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y +
                                                 iyy * dist_y * dist_y);
                    y /= additional_denoise_power;
                    // GCam GetKernelWeight: exp2(-0.5 kd) + 5e-5 (exp2 reconciled via the
                    // ln2-scaled precision on the gcam-kernel path).
                    const f32 w = std::exp(-0.5f * y) + (gk ? 5e-5f : 0.f);

                    val[channel] += c * w;
                    acc[channel] += w;
                }
            }

            // The reference frame always accumulates into the running merge.
            for (int ch = 0; ch < nch; ++ch) {
                num.at(local_i, hr_j, ch) += val[ch];
                den.at(local_i, hr_j, ch) += acc[ch];
            }
        }
    });
}

} // namespace

bool merge_comp_band(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                     const Image& robustness, int tile_size,
                     Image& num_band, Image& den_band, int y0, const Config& cfg,
                     int frame_id, const Mask8* rob8) {
    // gcam_sabre_merge_res 8-bit output-grid mask: run the CPU accumulate (which reads
    // the Mask8). On device the frames are non-resident for merge_res, so comp_raw is
    // on the host. The GPU merge kernels read a float mask and don't handle Mask8.
    if (rob8 && !rob8->empty()) {
        accumulate_comp(comp_raw, flow, covs, robustness, tile_size, num_band, den_band, y0, cfg, rob8);
        return true;
    }
#ifdef __APPLE__
    // bran pipeline, exact mode: 1.4's float64 accumulate on the CPU.
    if (cfg.bran14_kernels && cfg.bran14_exact_cpu) {
        accumulate_comp(comp_raw, flow, covs, robustness, tile_size, num_band, den_band, y0, cfg, rob8);
        return true;
    }
    // Metal GPU only — same Alg. 4 math as accumulate_comp (incl. per-pixel robustness).
    return merge_comp_band_metal(comp_raw, flow, covs, robustness, tile_size,
                                 num_band, den_band, y0, cfg, frame_id);
#else
#ifdef _WIN32
    // Windows D3D11 GPU merge (opt-in). Skipped for the Mask8 (merge-res) path --
    // the shader reads a float mask -- so that falls through to the CPU accumulate.
    if (cfg.merge_backend_d3d11 && (!rob8 || rob8->empty()) && d3d11_merge_available() &&
        merge_comp_band_d3d11(comp_raw, flow, covs, robustness, tile_size,
                              num_band, den_band, y0, cfg, frame_id))
        return true;
#endif
    (void)frame_id;
    accumulate_comp(comp_raw, flow, covs, robustness, tile_size, num_band, den_band, y0, cfg, rob8);
    return true;
#endif
}

void merge_ref_band(const Image& ref_raw, const CovField& covs,
                    Image& num_band, Image& den_band, int y0, const Config& cfg,
                    const Image* acc_rob) {
#ifdef __APPLE__
    if (cfg.bran14_kernels && cfg.bran14_exact_cpu) {   // bran exact mode: CPU float64
        accumulate_ref(ref_raw, covs, acc_rob, num_band, den_band, y0, cfg);
        return;
    }
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

// Every field is a COUNT, so the order pixels are visited in cannot change the
// result -- which is what makes the chunked form below exactly equal to the
// serial one it replaces.
static void accumulate_diag_span(const f32* nump, const f32* denp,
                                 size_t first, size_t last, int c, AccumDiag& d) {
    const int nch = std::min(3, c);
    for (size_t p = first; p < last; ++p) {
        ++d.pixels;
        f32 dens[3] = {0, 0, 0};
        for (int ch = 0; ch < nch; ++ch) {
            f32 nv = nump[p * (size_t)c + (size_t)ch];
            f32 dv = denp[p * (size_t)c + (size_t)ch];
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

static void accumulate_diag_merge(AccumDiag& d, const AccumDiag& s) {
    d.pixels += s.pixels;
    for (int ch = 0; ch < 3; ++ch) {
        d.den_zero[ch] += s.den_zero[ch];
        d.den_tiny[ch] += s.den_tiny[ch];
        d.den_nonfinite[ch] += s.den_nonfinite[ch];
        d.num_nonfinite[ch] += s.num_nonfinite[ch];
    }
    d.only_green += s.only_green;
    d.rgb_all_zero += s.rgb_all_zero;
}

// Chunked and row-parallel.
//
// The online merge calls this over the WHOLE output -- 48.8M pixels, reading
// both accumulators -- where the banded path only ever saw one band. Serial and
// branch-heavy, that was several hundred milliseconds of a burst, spent on a
// diagnostic string and inside no profiler bucket.
//
// Exact, not approximate: the fields are counts and addition of counts is
// associative, so chunking and summing gives the same numbers in any order.
void accumulate_diag_ptr(const f32* nump, const f32* denp, size_t n,
                         int c, AccumDiag& d) {
    if (!nump || !denp || n == 0 || c <= 0) return;
    // Below this the dispatch costs more than the scan.
    constexpr size_t kSerialLimit = 1u << 16;
    if (n <= kSerialLimit) {
        accumulate_diag_span(nump, denp, 0, n, c, d);
        return;
    }
    const int nchunks = 16;
    const size_t chunk = (n + (size_t)nchunks - 1u) / (size_t)nchunks;
    std::vector<AccumDiag> parts((size_t)nchunks);
    parallel_rows(nchunks, 0, [&](int k) {
        const size_t first = (size_t)k * chunk;
        if (first >= n) return;
        const size_t last = std::min(n, first + chunk);
        accumulate_diag_span(nump, denp, first, last, c, parts[(size_t)k]);
    });
    for (const AccumDiag& p : parts) accumulate_diag_merge(d, p);
}

void accumulate_diag(const Image& num, const Image& den, AccumDiag& d) {
    accumulate_diag_ptr(num.data.data(), den.data.data(),
                        (size_t)num.h * (size_t)num.w, num.c, d);
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
