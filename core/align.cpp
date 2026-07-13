#include "stages.h"
#include "parallel.h"
#include "linalg.h"
#include <complex>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace hhsr {

// ============================================================================
// Sobel gradients — matches Python's F.conv2d(image, SOBEL_X/Y, padding='same')
// Produces same-sized output (unlike compute_gradients which shrinks by 1).
// SOBEL_X = [[-1,0,1]], SOBEL_Y = [[-1],[0],[1]]
// ============================================================================
static Image compute_sobel_gradx(const Image& img) {
    Image out(img.h, img.w, 1);
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            // kernel [-1, 0, 1] centred at x, padding='same' => clamp
            int xm = std::max(0, x - 1);
            int xp = std::min(img.w - 1, x + 1);
            out.at(y, x) = -img.at(y, xm) + img.at(y, xp);
        }
    }
    return out;
}

static Image compute_sobel_grady(const Image& img) {
    Image out(img.h, img.w, 1);
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            // kernel [[-1],[0],[1]] centred at y, padding='same' => clamp
            int ym = std::max(0, y - 1);
            int yp = std::min(img.h - 1, y + 1);
            out.at(y, x) = -img.at(ym, x) + img.at(yp, x);
        }
    }
    return out;
}

// ============================================================================
// Bilinear interpolation matching the Python ICA kernel exactly.
// Python: floor_x = x + int(alignment[0]), frac_x = modf(alignment[0])
//         OOB => 0.0 (NOT clamp-to-edge).
// ============================================================================
static inline f32 sample_oob_zero(const Image& img, int iy, int ix) {
    if (iy < 0 || iy >= img.h || ix < 0 || ix >= img.w) return 0.f;
    return img.at(iy, ix);
}

static inline f32 bilinear_oob_zero(const Image& img, int pixel_y, int pixel_x,
                                     int floor_off_y, int floor_off_x,
                                     f32 frac_x, f32 frac_y) {
    int fy = pixel_y + floor_off_y;
    int fx = pixel_x + floor_off_x;
    f32 m00 = sample_oob_zero(img, fy,     fx    );
    f32 m01 = sample_oob_zero(img, fy,     fx + 1);
    f32 m10 = sample_oob_zero(img, fy + 1, fx    );
    f32 m11 = sample_oob_zero(img, fy + 1, fx + 1);

    f32 lerpx_top = m00 + (m01 - m00) * frac_x;
    f32 lerpx_bot = m10 + (m11 - m10) * frac_x;
    return lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
}

// ============================================================================
// Clamp-to-edge bilinear for L2 block matching patch extraction
// (matches Python's extract_flow_patches which uses .clamp(0, shape-1))
// ============================================================================
static inline f32 sample_clamp(const Image& img, int iy, int ix) {
    iy = std::max(0, std::min(img.h - 1, iy));
    ix = std::max(0, std::min(img.w - 1, ix));
    return img.at(iy, ix);
}

// ============================================================================
// FFT-based L2 block matching — matches Python's align_lvl_block_matching_L2
//
// Algorithm:
// 1. For each tile, extract a (2R+ts) × (2R+ts) search patch from moving,
//    centred on the tile position + current flow (rounded to int), clamped.
// 2. The reference tile is zero-padded to (2R+ts) × (2R+ts).
// 3. Cross-correlation via conj(ref_fft) * mov_fft → irfft2 → fftshift.
// 4. Crop to (2R+1) × (2R+1) valid region.
// 5. Compute windowed L2 norm of search patches via box-sum.
// 6. L2_error = L2_search - 2 * corrs. Argmin gives best shift.
// ============================================================================

// Pad dimension to next power of 2 (FFT requirement for radix-2)
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// 2D real-to-complex FFT: input real[h*w], output complex[h*w] (full-size, not rfft)
static void rfft2d_full(const f32* real_in, int h, int w,
                        std::vector<std::complex<f32>>& out) {
    out.resize((size_t)h * w);
    for (int i = 0; i < h * w; ++i) out[i] = {real_in[i], 0.f};
    fft2d(out, h, w, false);
}

// 2D IFFT → real: takes complex[h*w], writes real output
static void irfft2d_full(std::vector<std::complex<f32>>& data, int h, int w,
                         std::vector<f32>& real_out) {
    fft2d(data, h, w, true);
    real_out.resize((size_t)h * w);
    for (int i = 0; i < h * w; ++i) real_out[i] = data[i].real();
}

static void block_match_level_L2(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;
    int R = search_radius;
    int search_size = 2 * R + ts;       // size of search patch
    int corr_size = 2 * R + 1;          // output correlation map size

    // Pad to power of 2 for FFT
    int fft_h = next_pow2(search_size);
    int fft_w = next_pow2(search_size);

    // Process each tile
    parallel_rows(ny, num_threads, [&](int ty) {
        // Allocate per-thread FFT buffers
        std::vector<std::complex<f32>> ref_fft((size_t)fft_h * fft_w);
        std::vector<std::complex<f32>> mov_fft((size_t)fft_h * fft_w);
        std::vector<f32> corr_real;
        std::vector<f32> ref_tile_padded((size_t)fft_h * fft_w, 0.f);
        std::vector<f32> mov_patch((size_t)fft_h * fft_w, 0.f);

        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;

            // Round current flow to integer (matching Python's flow.round().long())
            int flow_dx = (int)std::round(flow.dx(ty, tx));
            int flow_dy = (int)std::round(flow.dy(ty, tx));

            // --- Prepare zero-padded reference tile ---
            // Reference tile at (oy, ox) of size ts×ts, zero-padded with R on each side
            // to make (2R+ts) × (2R+ts), then further zero-padded to fft_h × fft_w.
            std::fill(ref_tile_padded.begin(), ref_tile_padded.end(), 0.f);
            for (int i = 0; i < ts; ++i) {
                for (int j = 0; j < ts; ++j) {
                    int ry = oy + i;
                    int rx = ox + j;
                    if (ry < ref.h && rx < ref.w) {
                        ref_tile_padded[(size_t)(i + R) * fft_w + (j + R)] = ref.at(ry, rx);
                    }
                }
            }

            // --- Extract search patch from moving image ---
            // Centred on tile position + flow, size (2R+ts) × (2R+ts), clamped.
            std::fill(mov_patch.begin(), mov_patch.end(), 0.f);
            for (int i = 0; i < search_size; ++i) {
                for (int j = 0; j < search_size; ++j) {
                    int my = oy + flow_dy + i - R;
                    int mx = ox + flow_dx + j - R;
                    // Clamp to image bounds (matching Python's .clamp(0, shape-1))
                    my = std::max(0, std::min(moving.h - 1, my));
                    mx = std::max(0, std::min(moving.w - 1, mx));
                    mov_patch[(size_t)i * fft_w + j] = moving.at(my, mx);
                }
            }

            // --- FFT cross-correlation ---
            rfft2d_full(ref_tile_padded.data(), fft_h, fft_w, ref_fft);
            rfft2d_full(mov_patch.data(), fft_h, fft_w, mov_fft);

            // conj(ref_fft) * mov_fft
            std::vector<std::complex<f32>> cross((size_t)fft_h * fft_w);
            for (size_t i = 0; i < cross.size(); ++i) {
                cross[i] = std::conj(ref_fft[i]) * mov_fft[i];
            }

            // IFFT → real correlation
            irfft2d_full(cross, fft_h, fft_w, corr_real);

            // fftshift
            // We need to fftshift the real-valued correlation. Reuse complex fftshift
            // by temporarily wrapping in complex, or do it directly on real.
            // Direct real fftshift:
            {
                // fftshift on real buffer of size fft_h × fft_w
                std::vector<f32> shifted((size_t)fft_h * fft_w);
                int half_h = fft_h / 2;
                int half_w = fft_w / 2;
                for (int y = 0; y < fft_h; ++y) {
                    for (int x = 0; x < fft_w; ++x) {
                        int sy = (y + half_h) % fft_h;
                        int sx = (x + half_w) % fft_w;
                        shifted[(size_t)y * fft_w + x] = corr_real[(size_t)sy * fft_w + sx];
                    }
                }
                corr_real = shifted;
            }

            // --- Crop to valid region ---
            // Python: pre_crop_size = corrs.shape[-1] (= fft_w since we used fft_w)
            // crop = (pre_crop_size - 1 - corr_size) // 2
            // corrs = corrs[..., crop+1:crop+corr_size+1, crop+1:crop+corr_size+1]
            int pre_crop_size = fft_w; // square, so same for h and w
            int crop = (pre_crop_size - 1 - corr_size) / 2;
            int crop_y0 = crop + 1;
            int crop_x0 = crop + 1;
            // Also do the same for height dimension (fft_h)
            int pre_crop_h = fft_h;
            int crop_h = (pre_crop_h - 1 - corr_size) / 2;
            int crop_hy0 = crop_h + 1;

            std::vector<f32> corrs((size_t)corr_size * corr_size);
            for (int i = 0; i < corr_size; ++i) {
                for (int j = 0; j < corr_size; ++j) {
                    corrs[(size_t)i * corr_size + j] =
                        corr_real[(size_t)(crop_hy0 + i) * fft_w + (crop_x0 + j)];
                }
            }

            // --- Compute windowed L2 norm of search patches ---
            // Box-filter convolution: sum of squares over ts×ts windows within
            // the search_size×search_size search patch. Output is corr_size×corr_size.
            std::vector<f32> L2_search((size_t)corr_size * corr_size, 0.f);
            for (int i = 0; i < corr_size; ++i) {
                for (int j = 0; j < corr_size; ++j) {
                    f32 sum_sq = 0.f;
                    for (int ki = 0; ki < ts; ++ki) {
                        for (int kj = 0; kj < ts; ++kj) {
                            int si = i + ki;
                            int sj = j + kj;
                            f32 v = mov_patch[(size_t)si * fft_w + sj];
                            sum_sq += v * v;
                        }
                    }
                    L2_search[(size_t)i * corr_size + j] = sum_sq;
                }
            }

            // --- L2 error = L2_search - 2 * corrs ---
            // (Omits the constant ref L2 norm which doesn't affect argmin)
            f32 best_err = 1e30f;
            int best_dy = 0, best_dx = 0;
            for (int i = 0; i < corr_size; ++i) {
                for (int j = 0; j < corr_size; ++j) {
                    f32 err = L2_search[(size_t)i * corr_size + j]
                            - 2.f * corrs[(size_t)i * corr_size + j];
                    if (err < best_err) {
                        best_err = err;
                        best_dy = i - corr_size / 2;
                        best_dx = j - corr_size / 2;
                    }
                }
            }

            // Update flow: += delta (matching Python: alignment[:,:,0] += dx)
            flow.dx(ty, tx) += (f32)best_dx;
            flow.dy(ty, tx) += (f32)best_dy;
        }
    });
}

// ============================================================================
// L1 block matching — matches Python's align_lvl_block_matching_L1
// Rounds flow to integer, OOB => 0.0.
// ============================================================================
static void block_match_level_L1(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;
    int R = search_radius;

    parallel_rows(ny, num_threads, [&](int ty) {
        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;

            // Round current flow (matching Python: round(alignments[...]))
            int flow_dx = (int)std::round(flow.dx(ty, tx));
            int flow_dy = (int)std::round(flow.dy(ty, tx));

            f32 best_err = 1e30f;
            int best_sx = 0, best_sy = 0;

            for (int sdy = -R; sdy <= R; ++sdy) {
                for (int sdx = -R; sdx <= R; ++sdx) {
                    f32 l1_sum = 0.f;
                    for (int i = 0; i < ts; ++i) {
                        int ry = oy + i;
                        if (ry >= ref.h) break;
                        for (int j = 0; j < ts; ++j) {
                            int rx = ox + j;
                            if (rx >= ref.w) break;

                            int my = ry + flow_dy + sdy;
                            int mx = rx + flow_dx + sdx;
                            // OOB => 0.0 (matching Python CUDA kernel)
                            f32 mv = (my >= 0 && my < moving.h &&
                                      mx >= 0 && mx < moving.w)
                                     ? moving.at(my, mx) : 0.f;
                            l1_sum += std::fabs(ref.at(ry, rx) - mv);
                        }
                    }
                    if (l1_sum < best_err) {
                        best_err = l1_sum;
                        best_sy = sdy;
                        best_sx = sdx;
                    }
                }
            }

            // Update: flow = rounded_flow + best_shift (matching Python)
            flow.dx(ty, tx) = (f32)(flow_dx + best_sx);
            flow.dy(ty, tx) = (f32)(flow_dy + best_sy);
        }
    });
}

// ============================================================================
// Per-level ICA refinement — matches Python's align_lvl_ica + ica_kernel_*
//
// Uses Sobel gradients (same size as image), bilinear interpolation with
// modf decomposition and OOB=>0, and Cramer's rule update.
// ============================================================================
static void ica_refine_level(const Image& ref, const Image& gradx,
                              const Image& grady, const Image& moving,
                              FlowField& flow, int tile_size, int n_iter,
                              int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;

    // Pre-compute per-tile Hessian and its inverse
    // Hessian H = [[sum(gx*gx), sum(gx*gy)], [sum(gx*gy), sum(gy*gy)]]
    parallel_rows(ny, num_threads, [&](int ty) {
        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;

            // Accumulate Hessian over tile
            f32 h00 = 0, h01 = 0, h10 = 0, h11 = 0;
            for (int i = 0; i < ts; ++i) {
                int py = oy + i;
                if (py >= ref.h) break;
                for (int j = 0; j < ts; ++j) {
                    int px = ox + j;
                    if (px >= ref.w) break;
                    f32 gx = gradx.at(py, px);
                    f32 gy = grady.at(py, px);
                    h00 += gx * gx;
                    h01 += gx * gy;
                    h10 += gx * gy;
                    h11 += gy * gy;
                }
            }

            // Check singularity (matching Python: abs(det) < 1e-10)
            f32 det = h00 * h11 - h01 * h10;
            if (std::fabs(det) < 1e-10f) continue;
            f32 det_inv = 1.f / det;

            // ICA iterations
            f32 fx = flow.dx(ty, tx);
            f32 fy = flow.dy(ty, tx);

            for (int it = 0; it < n_iter; ++it) {
                // Decompose flow via modf (matching Python: frac_x, _ = math.modf(s_alignment[0]))
                f32 int_part_x, int_part_y;
                f32 frac_x = std::modf(fx, &int_part_x);
                f32 frac_y = std::modf(fy, &int_part_y);
                int floor_off_x = (int)int_part_x;
                int floor_off_y = (int)int_part_y;

                f32 B0 = 0.f, B1 = 0.f;
                for (int i = 0; i < ts; ++i) {
                    int py = oy + i;
                    if (py >= ref.h) break;
                    for (int j = 0; j < ts; ++j) {
                        int px = ox + j;
                        if (px >= ref.w) break;

                        // Bilinear interpolation of moving at (py + flow_y, px + flow_x)
                        // floor_x = px + int(alignment[0]), frac from modf
                        f32 mov_interp = bilinear_oob_zero(moving, py, px,
                                                            floor_off_y, floor_off_x,
                                                            frac_x, frac_y);

                        f32 gradt = mov_interp - ref.at(py, px);
                        f32 gx = gradx.at(py, px);
                        f32 gy = grady.at(py, px);

                        // Accumulate: B += -grad * gradt (matching Python)
                        B0 += -gx * gradt;
                        B1 += -gy * gradt;
                    }
                }

                // Update flow: Cramer's rule (matching Python)
                // alignment[0] += det_inv * (A11 * B0 - A01 * B1)
                // alignment[1] += det_inv * (-A10 * B0 + A00 * B1)
                fx += det_inv * (h11 * B0 - h01 * B1);
                fy += det_inv * (-h10 * B0 + h00 * B1);
            }

            flow.dx(ty, tx) = fx;
            flow.dy(ty, tx) = fy;
        }
    });
}

// ============================================================================
// Flow upscaling — matches Python's upscale_lvl
//
// Python logic:
//   repeat_factor = upsample_factor // (new_tile_size // prev_tile_size)
//   upsampled = F.interpolate(alignments, scale_factor=repeat_factor, mode='nearest')
//   upsampled *= upsample_factor
//   # zero-pad if size mismatch
// ============================================================================
static FlowField upscale_flow(const FlowField& in, int target_ny, int target_nx,
                               int upsample_factor, int new_tile_size,
                               int prev_tile_size) {
    // Compute repeat factor matching Python
    int tile_ratio = new_tile_size / std::max(1, prev_tile_size);
    int repeat_factor = upsample_factor / std::max(1, tile_ratio);
    if (repeat_factor < 1) repeat_factor = 1;

    // Nearest-neighbour upscale by repeat_factor
    int up_ny = in.ny * repeat_factor;
    int up_nx = in.nx * repeat_factor;
    FlowField upsampled(up_ny, up_nx);
    for (int ty = 0; ty < up_ny; ++ty) {
        for (int tx = 0; tx < up_nx; ++tx) {
            int sy = ty / repeat_factor;
            int sx = tx / repeat_factor;
            sy = std::min(sy, in.ny - 1);
            sx = std::min(sx, in.nx - 1);
            upsampled.dx(ty, tx) = in.dx(sy, sx) * (f32)upsample_factor;
            upsampled.dy(ty, tx) = in.dy(sy, sx) * (f32)upsample_factor;
        }
    }

    // Pad to target size if needed (matching Python: F.pad with zeros)
    FlowField out(target_ny, target_nx);
    for (int ty = 0; ty < target_ny; ++ty) {
        for (int tx = 0; tx < target_nx; ++tx) {
            if (ty < up_ny && tx < up_nx) {
                out.dx(ty, tx) = upsampled.dx(ty, tx);
                out.dy(ty, tx) = upsampled.dy(ty, tx);
            }
            // else remains 0 (from FlowField constructor)
        }
    }
    return out;
}

// ============================================================================
// Main alignment entry point — matches Python's align() in alignment.py
//
// For each pyramid level (coarse → fine):
//   1. Upscale flow from previous level
//   2. Block match (L1 or L2 based on config)
//   3. ICA refinement
// ============================================================================
FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg, int tile_size) {
    Pyramid mov_pyr = build_pyramid(moving_grey, cfg.bm_factors);

    int nlev = (int)ref_pyr.levels.size();
    FlowField flow; // empty => zero init at coarsest level

    // Coarse (last) -> fine (first).
    // C++ pyramid: levels[0] = finest (factor=1), levels[nlev-1] = coarsest (factor=32)
    // Config arrays: index 0 = finest, index nlev-1 = coarsest (fine-to-coarse order)
    for (int lvl = nlev - 1; lvl >= 0; --lvl) {
        const Image& r = ref_pyr.levels[lvl];
        const Image& m = mov_pyr.levels[lvl];

        int ts = (lvl < (int)cfg.bm_tile_sizes.size())
                     ? std::max(4, cfg.bm_tile_sizes[lvl])
                     : std::max(4, tile_size);
        int radius = (lvl < (int)cfg.bm_search_radii.size())
                     ? cfg.bm_search_radii[lvl] : 2;

        int ny = std::max(1, r.h / ts);
        int nx = std::max(1, r.w / ts);

        // Upscale flow from previous level or init to zero
        if (flow.nx == 0) {
            flow = FlowField(ny, nx);
        } else {
            // We just processed level lvl+1 (one coarser). Now transitioning to lvl.
            // Python: upsample_factor = factors[l+1] where l+1 is the coarser level
            //         in the reversed (coarse-first) list. In our fine-to-coarse arrays,
            //         the coarser level is lvl+1, so upsample_factor = factors[lvl+1].
            int upsample_factor = ((lvl + 1) < (int)cfg.bm_factors.size())
                                  ? cfg.bm_factors[lvl + 1] : 1;
            int prev_ts = ((lvl + 1) < (int)cfg.bm_tile_sizes.size())
                          ? cfg.bm_tile_sizes[lvl + 1]
                          : ts;
            flow = upscale_flow(flow, ny, nx, upsample_factor, ts, prev_ts);
        }

        // Determine metric for this level
        std::string metric = "L2";
        if (lvl < (int)cfg.bm_metrics.size()) {
            metric = cfg.bm_metrics[lvl];
        }

        // Block matching
        if (metric == "L1") {
            block_match_level_L1(r, m, ts, radius, flow, cfg.num_threads);
        } else {
            block_match_level_L2(r, m, ts, radius, flow, cfg.num_threads);
        }

        // Per-level ICA refinement (matching Python: align_lvl_ica at every level)
        Image gx = compute_sobel_gradx(r);
        Image gy = compute_sobel_grady(r);
        ica_refine_level(r, gx, gy, m, flow, ts, cfg.ica_n_iter, cfg.num_threads);
    }

    return flow;
}

} // namespace hhsr
