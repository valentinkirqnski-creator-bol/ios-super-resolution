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

// Enlargement of the reference merge kernel, adapted to the accumulated
// robustness -- "we locally enlarge the merging kernel of the reference frame
// when the frame count is low".
//
// Derived rather than thresholded. Merging k frames cuts noise variance by 1/k;
// averaging over a kernel of area A does the same by 1/A. Dividing the
// Mahalanobis distance by m sends Sigma -> m*Sigma, so the kernel area scales as
// m. Matching the two gives m = N/k. The reference frame always contributes, so
// the effective frame count at a pixel is r_acc + 1, and N is the whole burst.
//
// Everything falls out of that: fully merged pixels get m = 1 and are left
// alone, single-frame pixels get m = N, and it normalises itself to burst
// length with no parameter to set. power_max is only a cap.
//
// The reference implementation uses a step -- power_max below a frame-count
// threshold, 1 above -- which is a simplification of the same idea. A step also
// puts a visible seam between enlarged and un-enlarged regions, which this
// avoids.
// Reference-implementation behaviour, kept for acc_rob_adaptive == false: full
// enlargement below a frame-count threshold and none above, with the
// accumulator overwritten below it.
static inline f32 denoise_power_step(f32 r_acc, f32 power_max, f32 max_frame_count) {
    return (r_acc <= max_frame_count) ? power_max : 1.f;
}

static inline int denoise_range_step(f32 r_acc, int rad_max, f32 max_frame_count) {
    return (r_acc <= max_frame_count) ? rad_max : 1;
}

static inline f32 denoise_power_merge(f32 r_acc, f32 power_max, f32 burst_frames) {
    if (!(burst_frames > 1.f)) return 1.f;
    const f32 effective = std::max(1.f, r_acc + 1.f);
    const f32 cap = std::max(1.f, power_max);
    return std::min(std::max(burst_frames / effective, 1.f), cap);
}

// sigma grows as sqrt(m), so the window has to grow with it or the widened
// kernel is truncated and the enlargement does nothing.
//
// Floored for pixels with under two contributing frames. The law above answers
// a noise question, but chroma is a sampling one and does not scale with burst
// length: a 3x3 window on a Bayer lattice holds as few as one red and one blue
// sample, so a single-frame pixel reconstructs colour from one measurement
// however short the burst was. 5x5 takes that to at least four. Without this a
// 2-frame burst asks for m=2, which rounds back down to 3x3 in exactly the
// case that is most prone to demosaicking artifacts.
static inline int denoise_range_merge(f32 power, f32 r_acc, int rad_max) {
    int r = (int)std::lround(std::sqrt(std::max(1.f, power)));
    if (r_acc + 1.f < 2.f) r = std::max(r, rad_max);
    return std::min(std::max(r, 1), std::max(1, rad_max));
}

// Guard against singular/near-singular covariance inversions producing
// infinitely sharp kernels. That can leave R/B denominators at zero while G
// receives weight, showing up as green or black speckles.
//
// A kernel this sharp contributes weight to essentially one raw sample. Under
// a Bayer CFA that sample belongs to one channel, so the other two accumulate
// nothing at that output pixel and their denominators stay at zero -- hence
// the speckle rather than a general softness. Clamping the largest magnitude
// to k_max_abs bounds the kernel's minimum width instead of rejecting the
// tile, so the sample still contributes, just over more than one pixel.
//
// Keep in step with the Metal twin in HHSRKernels.metal and the mirror in
// tools/validate_merge_equiv.py: all three are compared against each other.
// The ceiling is mode-dependent, and its upper bound is set by COVERAGE, not
// by taste: the fast-weights tap cutoff (z > 16 = beyond 4 sigma) and the
// fp16 accumulator both need the sharpest kernel to still reach the nearest
// same-color sample, which sits at ~0.2-0.4 px across an 8-frame Bayer
// merge. 512 (sigma floor 0.044 px) violated that -- every off-site colour
// channel lost all its taps and the denominator zeroed, which is exactly the
// green/black per-channel speckle the guard exists to prevent. 128 floors
// sigma at 0.088 px: z at 0.35 px is ~15.8 (inside the cutoff), the weight
// ~4e-4 (fp16-safe), and the resolution cost vs 0.044 px is 97% vs 98%
// contrast at 1 px strokes -- nothing, for artifact-free coverage.
//
// BOTH modes use 128 now. The legacy value 32 (sigma floor 0.177 px) was a
// silent divergence from the 460-main reference, which has NO width floor:
// at k_detail 0.17 its across-edge kernels run at 0.085 px and ours were
// doubled to 0.177 -- every edge ~2x softer than the reference at identical
// settings, which is why matching its detail took k_detail ~0.10 here. At
// 128 the same k_detail means the same kernel as 460-main down to the
// coverage bound, and the reference-pass coverage floor (keyed on this
// value being > 64) guards the denominators in both modes.
static inline f32 merge_soften_max_inv(const Config& cfg) {
    (void)cfg;
    return 128.f;
}

static inline void soften_inv_cov(f32& ixx, f32& ixy, f32& iyy, f32 k_max_abs) {
    if (!std::isfinite(ixx) || !std::isfinite(ixy) || !std::isfinite(iyy)) {
        ixx = 2.f;
        ixy = 0.f;
        iyy = 2.f;
        return;
    }
    // Clamp EIGENVALUES, not the whole matrix. The previous form rescaled all
    // three entries by one factor, which bounds the sharp axis (what the
    // speckle fix needs) but widens the already-wide axis by the same factor
    // -- pure blur with no coverage benefit. Measured on an edge kernel with
    // the continuous-anisotropy defaults: sigma 0.085 x 0.68 px became
    // 0.177 x 1.41 px, doubling the along-edge width for nothing, which is
    // the reported over-smoothing. Bounding each eigenvalue independently
    // gives 0.177 x 0.68: the across-edge axis widened exactly to the
    // coverage floor, the along-edge axis untouched.
    const f32 mean = 0.5f * (ixx + iyy);
    const f32 half_diff = 0.5f * (ixx - iyy);
    const f32 disc = std::sqrt(half_diff * half_diff + ixy * ixy);
    const f32 l1 = mean + disc;                    // largest eigenvalue
    if (!(l1 > k_max_abs)) return;                 // nothing too sharp
    const f32 l2 = mean - disc;
    const f32 c1 = k_max_abs;
    const f32 c2 = std::min(l2, k_max_abs);
    // Eigenvector of l1. Both constructions degenerate only when the matrix
    // is (near-)isotropic diagonal, where a plain per-entry clamp is exact.
    f32 vx = ixy;
    f32 vy = l1 - ixx;
    f32 n2 = vx * vx + vy * vy;
    if (!(n2 > 0.f)) {
        vx = l1 - iyy;
        vy = ixy;
        n2 = vx * vx + vy * vy;
    }
    if (!(n2 > 0.f)) {
        ixx = std::min(ixx, k_max_abs);
        iyy = std::min(iyy, k_max_abs);
        return;
    }
    const f32 inv_n2 = 1.f / n2;
    const f32 d = c1 - c2;
    ixx = c2 + d * (vx * vx * inv_n2);
    ixy = d * (vx * vy * inv_n2);
    iyy = c2 + d * (vy * vy * inv_n2);
}

static inline int cuda_round_to_int(f32 x) {
    return (int)std::lround(x);
}

static inline f32 sample_robustness_bilinear(const Image& robustness, f32 y, f32 x) {
    if (robustness.h <= 0 || robustness.w <= 0) return 0.f;
    y = std::min(std::max(y, 0.f), (f32)(robustness.h - 1));
    x = std::min(std::max(x, 0.f), (f32)(robustness.w - 1));
    const int y0 = (int)std::floor(y);
    const int x0 = (int)std::floor(x);
    const int y1 = std::min(y0 + 1, robustness.h - 1);
    const int x1 = std::min(x0 + 1, robustness.w - 1);
    const f32 fy = y - (f32)y0;
    const f32 fx = x - (f32)x0;
    const f32 top = robustness.at(y0, x0) +
                    (robustness.at(y0, x1) - robustness.at(y0, x0)) * fx;
    const f32 bot = robustness.at(y1, x0) +
                    (robustness.at(y1, x1) - robustness.at(y1, x0)) * fx;
    return top + (bot - top) * fy;
}

// Bilinear cov sample + invert.
// ref (accumulate_ref): floor indices + modf fracs; invert_2x2 → I on singular.
// comp (accumulate): int() indices + modf fracs; raw 1/det.
static inline void interp_inv_cov(const CovField& covs, f32 kmap_i, f32 kmap_j,
                                  f32& ixx, f32& ixy, f32& iyy, bool raw_det,
                                  f32 soften_max) {
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
    soften_inv_cov(ixx, ixy, iyy, soften_max);
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
    const bool fastw = cfg.merge_fast_weights;

    parallel_rows(band_h, cfg.num_threads, [&](int local_i) {
        const int hr_i = y0 + local_i;
        for (int hr_j = 0; hr_j < Ws; ++hr_j) {
            // output_pixel / scale -- the SAME expression accumulate_ref uses
            // for coarse_x/coarse_y below. These two must agree: the reference
            // frame and every comparison frame are accumulated into one shared
            // num/den grid, so sampling them at different positions offsets
            // every comparison frame against the reference they are being
            // fused onto.
            //
            // b65b51f had this as (hr + 0.5) / scale "to match python-z's
            // accumulate" while leaving accumulate_ref on hr / scale, which
            // put the two half an output pixel apart -- 0.25 raw px at 2x --
            // and that mismatch is the sub-pixel misalignment. It did not
            // actually match python-z either: python-z pairs its (hr+0.5)/s
            // with a compensating `lr_mov_j = lr_mov_x - 0.5` before the
            // distance (merge.py accumulate, line 398), which this port has
            // never had, so the +0.5 alone landed a further half pixel out.
            //
            // 460-main -- which this function's header says it mirrors --
            // uses output_pixel / scale in BOTH accumulate and accumulate_ref
            // (merge.py:143 and :399). Verified by reading both references.
            const f32 lr_x = (f32)hr_j / scale;
            const f32 lr_y = (f32)hr_i / scale;

            // Python: px = int(lr_x // tile_size); no clamp on flow tile index
            const int px = (int)(lr_x / (f32)tile_size);
            const int py = (int)(lr_y / (f32)tile_size);
            // Bilinear between tile centres: block matching gives one vector
            // per tile, and consuming it nearest makes the warp piecewise
            // constant, which rotation turns into a visible tile grid. The
            // mask samples the SAME way, so it still grades the fetch that
            // actually happens. See FlowField::sample_bilinear.
            // Flow hypotheses. Normally one (sampled); in overlapped-tile
            // merge mode (Config::flow_overlap_merge) up to four -- the
            // covering half-pitch tiles' OWN vectors, Hann-crossfaded
            // (cos^2/sin^2 per axis == the 50%-overlap raised cosine,
            // partition of unity). Duplicates are merged so smooth regions
            // cost a single gather.
            f32 hx[4], hy[4], hw[4] = {1.f, 0.f, 0.f, 0.f};
            int nhyp = 1;
            if (cfg.overlap_merge_active() && flow.has_fine()) {
                const f32 P = 0.5f * (f32)tile_size;
                const f32 tcy = lr_y / P - 0.5f;
                const f32 tcx = lr_x / P - 0.5f;
                const int cy0i = (int)std::floor(tcy), cx0i = (int)std::floor(tcx);
                const f32 ay = tcy - (f32)cy0i, ax = tcx - (f32)cx0i;
                auto cl = [](int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); };
                const int iy0 = cl(cy0i, flow.fine_ny), iy1 = cl(cy0i + 1, flow.fine_ny);
                const int ix0 = cl(cx0i, flow.fine_nx), ix1 = cl(cx0i + 1, flow.fine_nx);
                const f32 sy = std::sin(1.57079632679f * ay);
                const f32 sx = std::sin(1.57079632679f * ax);
                const f32 wy1 = sy * sy, wy0 = 1.f - wy1;
                const f32 wx1 = sx * sx, wx0 = 1.f - wx1;
                const int idx4[4][2] = {{iy0, ix0}, {iy0, ix1}, {iy1, ix0}, {iy1, ix1}};
                const f32 w4[4] = {wy0 * wx0, wy0 * wx1, wy1 * wx0, wy1 * wx1};
                for (int a = 0; a < 4; ++a) {
                    const size_t o = ((size_t)idx4[a][0] * flow.fine_nx + idx4[a][1]) * 2u;
                    hx[a] = flow.fine_flow[o + 0];
                    hy[a] = flow.fine_flow[o + 1];
                    hw[a] = w4[a];
                }
                // CLUSTER, not exact-dedup: measured lattice vectors are
                // never bit-identical (independent sub-pixel fits), so an
                // exact test never fires and every pixel paid 4 gathers --
                // the merge stopped hiding behind the CPU. Hypotheses within
                // a quarter pixel are the same motion to sub-quantisation
                // accuracy; fold them into their weighted mean. Smooth
                // regions collapse to ONE gather; only real disagreements
                // (>= the boundary threshold) stay separate.
                for (int a = 1; a < 4; ++a)
                    for (int b = 0; b < a; ++b)
                        if (hw[a] > 0.f && hw[b] > 0.f &&
                            std::fabs(hx[a] - hx[b]) < 0.25f &&
                            std::fabs(hy[a] - hy[b]) < 0.25f) {
                            const f32 wsum = hw[b] + hw[a];
                            hx[b] = (hx[b] * hw[b] + hx[a] * hw[a]) / wsum;
                            hy[b] = (hy[b] * hw[b] + hy[a] * hw[a]) / wsum;
                            hw[b] = wsum;
                            hw[a] = 0.f;
                            break;
                        }
                nhyp = 4;
            } else if (cfg.flow_bilinear_sampling) {
                flow.sample_bilinear(lr_y, lr_x, tile_size, hx[0], hy[0]);
            } else {
                // sample_nearest, not flow.dx(py,px): follows the fine grid
                // when one exists, which every GPU path already does.
                flow.sample_nearest(lr_y, lr_x, tile_size, hx[0], hy[0]);
            }

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
            const f32 local_r = sample_robustness_bilinear(robustness, rob_y, rob_x);

            f32 val[3] = {0, 0, 0}, acc[3] = {0, 0, 0};
            for (int h = 0; h < nhyp; ++h) {
            if (hw[h] <= 0.f) continue;
            if (fastw && nhyp > 1 && hw[h] < 0.05f) continue;
            const f32 flowx = hx[h], flowy = hy[h];
            const f32 wsc = hw[h];
            const f32 lr_mov_x = lr_x + flowx;
            const f32 lr_mov_y = lr_y + flowy;
            if (!(lr_mov_x >= 0.f && lr_mov_x < (f32)lr_w &&
                  lr_mov_y >= 0.f && lr_mov_y < (f32)lr_h))
                continue;

            f32 ixx = 0.f, ixy = 0.f, iyy = 0.f;
            if (!iso) {
                // Raw position -> covariance/grey grid. Must be the SAME
                // mapping accumulate_ref uses below, for the same reason the
                // sampling position must: at zero flow a comparison frame
                // describes the same scene point as the reference, so both
                // have to look up the same kernel shape.
                //
                // 460-main uses (pos - 0.5)/2 (bayer) and pos (non-bayer) in
                // BOTH accumulate and accumulate_ref -- merge.py:446 and :162.
                // This path previously used python-z's accumulate form
                // (pos/2 - 0.5, pos - 0.5, merge.py:350), which the reference
                // pass never matched: a constant 0.25 grey px = 0.5 raw px
                // offset, so comparison frames were reconstructed with a
                // kernel measured half a raw pixel away from the one the
                // reference frame used. python-z is internally inconsistent
                // here too -- its own accumulate_ref uses the 460-main form.
                f32 kmap_j, kmap_i;
                if (cfg.bayer_mode) {
                    kmap_j = (lr_mov_x - 0.5f) / 2.f;
                    kmap_i = (lr_mov_y - 0.5f) / 2.f;
                } else {
                    kmap_j = lr_mov_x;
                    kmap_i = lr_mov_y;
                }
                interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy, /*raw_det=*/true,
                               merge_soften_max_inv(cfg));
            }

            const int center_j = cuda_round_to_int(lr_mov_x);
            const int center_i = cuda_round_to_int(lr_mov_y);

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
                    if (fastw && z > 16.f) continue;
                    const f32 w = std::exp(-0.5f * z);

                    val[channel] += w * local_r * wsc * c;
                    acc[channel] += w * local_r * wsc;
                }
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
    // See the coverage-floor comment in the tap loop below. Always active
    // now that the ceiling is 128 in every mode.
    const f32 cover_eps = (merge_soften_max_inv(cfg) > 64.f) ? 5e-4f : 0.f;
    const int band_h = num.h, Ws = num.w;
    const int lr_h = img.h, lr_w = img.w;
    const int nch = cfg.bayer_mode ? 3 : 1;
    const bool iso = (cfg.kernel == KernelShape::Iso);
    const f32 scale = cfg.scale;

    const bool robustness_denoise = cfg.accumulated_robustness_denoiser_enabled;
    const int rad_max = (int)cfg.acc_rob_rad_max;
    const f32 max_multiplier = cfg.acc_rob_max_multiplier;
    const f32 burst_frames = (f32)cfg.burst_frame_count;
    const bool adaptive = cfg.acc_rob_adaptive;
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
                // acc_rob inherits whatever resolution the per-frame R had --
                // read that off its own dimensions rather than the config
                // flag, same reasoning as accumulate_comp's rob_is_raw above.
                const bool acc_is_raw =
                    (acc_rob->h == lr_h && acc_rob->w == lr_w);
                if (!acc_is_raw && cfg.bayer_mode) {
                    acc_y = (coarse_y - 0.5f) / 2.f;
                    acc_x = (coarse_x - 0.5f) / 2.f;
                }
                const int ay = std::min(std::max(cuda_round_to_int(acc_y), 0), acc_rob->h - 1);
                const int ax = std::min(std::max(cuda_round_to_int(acc_x), 0), acc_rob->w - 1);
                local_acc_r = acc_rob->at(ay, ax);
                if (adaptive) {
                    additional_denoise_power =
                        denoise_power_merge(local_acc_r, max_multiplier, burst_frames);
                    rad = denoise_range_merge(additional_denoise_power, local_acc_r, rad_max);
                } else {
                    additional_denoise_power =
                        denoise_power_step(local_acc_r, max_multiplier, max_frame_count);
                    rad = denoise_range_step(local_acc_r, rad_max, max_frame_count);
                }
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
                interp_inv_cov(covs, kmap_i, kmap_j, ixx, ixy, iyy, /*raw_det=*/false,
                               merge_soften_max_inv(cfg));
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
                    f32 w = std::exp(-0.5f * y);
                    // Coverage floor: a tiny isotropic sigma=1
                    // reference contribution guarantees every colour channel
                    // of every output pixel a nonzero denominator, whatever
                    // the sharp kernels and robustness rejection left behind.
                    // 5e-4 relative: invisible (<1%) wherever any real tap
                    // survives, decisive where none does -- the residual
                    // green/black per-channel holes. The 3x3 window (rad >=
                    // 1) always contains all four CFA sites, so the floor
                    // closes every hole. Twin in merge_accumulate_ref_body.
                    if (cover_eps > 0.f)
                        w += cover_eps * std::exp(-0.5f * (dist_x * dist_x +
                                                           dist_y * dist_y));

                    val[channel] += c * w;
                    acc[channel] += w;
                }
            }

            // The reference overwrites below its threshold, discarding whatever
            // the comparison frames contributed. That is a no-op at both ends --
            // where nothing merged their weights are exactly zero, since
            // merge_comp_contrib returns early on r <= 0, and where everything
            // merged the threshold is not met -- so it only ever bites in
            // between, where it throws away real signal. The adaptive path
            // therefore always accumulates: a hard data cliff in the middle of a
            // continuous enlargement would defeat the point.
            const bool overwrite = !adaptive && robustness_denoise && acc_rob &&
                                   local_acc_r < max_frame_count;
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

bool merge_comp_band(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                     const Image& robustness, int tile_size,
                     Image& num_band, Image& den_band, int y0, const Config& cfg,
                     int frame_id) {
#ifdef __APPLE__
    // Metal GPU only — same Alg. 4 math as accumulate_comp (incl. per-pixel robustness).
    return merge_comp_band_metal(comp_raw, flow, covs, robustness, tile_size,
                                 num_band, den_band, y0, cfg, frame_id);
#else
    (void)frame_id;
    accumulate_comp(comp_raw, flow, covs, robustness, tile_size, num_band, den_band, y0, cfg);
    return true;
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

void accumulate_diag_ptr(const f32* nump, const f32* denp, size_t n,
                         int c, AccumDiag& d) {
    // Chunked parallel scan. Every field is an additive integer count, so
    // per-chunk partials summed in chunk order are bit-identical to the
    // serial scan -- this was a SERIAL walk over 144M values at 48MP,
    // costing a few hundred ms of the encode tail for one status line.
    const int nch_par = std::min(3, c);
    const size_t kChunk = 1u << 20;
    if (n > 2 * kChunk) {
        const size_t nchunks = (n + kChunk - 1) / kChunk;
        std::vector<AccumDiag> parts(nchunks);
        parallel_rows((int)nchunks, 0, [&](int ci) {
            const size_t p0 = (size_t)ci * kChunk;
            const size_t p1 = std::min(n, p0 + kChunk);
            accumulate_diag_ptr(nump + p0 * (size_t)c, denp + p0 * (size_t)c,
                                p1 - p0, c, parts[(size_t)ci]);
        });
        for (const AccumDiag& q : parts) {
            d.pixels += q.pixels;
            for (int ch = 0; ch < 3; ++ch) {
                d.den_zero[ch] += q.den_zero[ch];
                d.den_tiny[ch] += q.den_tiny[ch];
                d.den_nonfinite[ch] += q.den_nonfinite[ch];
                d.num_nonfinite[ch] += q.num_nonfinite[ch];
            }
            d.only_green += q.only_green;
            d.rgb_all_zero += q.rgb_all_zero;
        }
        return;
    }
    const int nch = nch_par;
    for (size_t p = 0; p < n; ++p) {
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
