#include "stages.h"
#include "parallel.h"
#include <cmath>
#include <limits>
#include <vector>

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
            // kernel [-1, 0, 1] centred at x, padding='same' => zero padding
            f32 vm = (x - 1 >= 0) ? img.at(y, x - 1) : 0.f;
            f32 vp = (x + 1 < img.w) ? img.at(y, x + 1) : 0.f;
            out.at(y, x) = -vm + vp;
        }
    }
    return out;
}

static Image compute_sobel_grady(const Image& img) {
    Image out(img.h, img.w, 1);
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            // kernel [[-1],[0],[1]] centred at y, padding='same' => zero padding
            f32 vm = (y - 1 >= 0) ? img.at(y - 1, x) : 0.f;
            f32 vp = (y + 1 < img.h) ? img.at(y + 1, x) : 0.f;
            out.at(y, x) = -vm + vp;
        }
    }
    return out;
}

// ============================================================================
// Bilinear interpolation matching Python ICA (ICA.py):
//   floor_x = x + int(alignment[0]);  frac = modf(alignment)
//   floor/ceil clamped to image bounds (clamp-to-edge), NOT OOB→0.
// ============================================================================
static inline f32 bilinear_clamp(const Image& img, int pixel_y, int pixel_x,
                                 int floor_off_y, int floor_off_x,
                                 f32 frac_x, f32 frac_y) {
    int floor_y = pixel_y + floor_off_y;
    int floor_x = pixel_x + floor_off_x;
    floor_x = std::max(0, std::min(img.w - 1, floor_x));
    floor_y = std::max(0, std::min(img.h - 1, floor_y));
    int ceil_x = std::max(0, std::min(img.w - 1, floor_x + 1));
    int ceil_y = std::max(0, std::min(img.h - 1, floor_y + 1));

    f32 m00 = img.at(floor_y, floor_x);
    f32 m01 = img.at(floor_y, ceil_x);
    f32 m10 = img.at(ceil_y, floor_x);
    f32 m11 = img.at(ceil_y, ceil_x);

    f32 lerpx_top = m00 + (m01 - m00) * frac_x;
    f32 lerpx_bot = m10 + (m11 - m10) * frac_x;
    return lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
}

// ============================================================================
// FFT-based L2 block matching — matches Python's align_lvl_block_matching_L2
//
// PyTorch path: corrs = fftshift(irfft2(conj(rfft2(ref)) * rfft2(mov), s=N))
// For real signals that equals circular cross-correlation:
//   corr[dy,dx] = sum_{i,j} ref[i,j] * mov[(i+dy)%N, (j+dx)%N]
// which is DFT-exact (same math as rfft2/irfft2), without library float drift.
// ============================================================================

static void block_match_level_L2(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;
    int R = search_radius;
    int search_size = 2 * R + ts;       // N — same as Python
    int corr_size = 2 * R + 1;
    const int N = search_size;

    struct RowBuffers {
        std::vector<f32> ref_tile_padded;
        std::vector<f32> mov_patch;
        std::vector<f32> corr;      // N×N circular correlation (pre-fftshift)
        std::vector<f32> shifted;   // after fftshift
        std::vector<f32> corrs;     // cropped corr_size×corr_size
        std::vector<f32> L2_search;

        RowBuffers(int n, int c_size)
            : ref_tile_padded(n * n, 0.f), mov_patch(n * n, 0.f),
              corr(n * n), shifted(n * n),
              corrs(c_size * c_size), L2_search(c_size * c_size, 0.f) {}
    };

    std::vector<RowBuffers> buffers;
    buffers.reserve(ny);
    for (int i = 0; i < ny; ++i)
        buffers.emplace_back(N, corr_size);

    parallel_rows(ny, num_threads, [&](int ty) {
        RowBuffers& b = buffers[ty];

        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;

            // torch: flow.round().long() — round half to even
            int flow_dx = (int)std::rint((float)flow.dx(ty, tx));
            int flow_dy = (int)std::rint((float)flow.dy(ty, tx));

            // Zero-padded reference tile at offset R (pad (R,R,R,R) in Python)
            std::fill(b.ref_tile_padded.begin(), b.ref_tile_padded.end(), 0.f);
            for (int i = 0; i < ts; ++i) {
                for (int j = 0; j < ts; ++j) {
                    int ry = oy + i;
                    int rx = ox + j;
                    if (ry < ref.h && rx < ref.w)
                        b.ref_tile_padded[(size_t)(i + R) * N + (j + R)] = ref.at(ry, rx);
                }
            }

            // Search patch — clamp to edge (extract_flow_patches)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    int my = std::max(0, std::min(moving.h - 1, oy + flow_dy + i - R));
                    int mx = std::max(0, std::min(moving.w - 1, ox + flow_dx + j - R));
                    b.mov_patch[(size_t)i * N + j] = moving.at(my, mx);
                }
            }

            // Circular cross-correlation ≡ irfft2(conj(rfft2(ref))*rfft2(mov))
            // corr[dy,dx] = sum ref[i,j] * mov[(i+dy)%N,(j+dx)%N]
            // Restrict sum to nonzero ref support [R,R+ts)²
            for (int dy = 0; dy < N; ++dy) {
                for (int dx = 0; dx < N; ++dx) {
                    f32 s = 0.f;
                    for (int i = 0; i < ts; ++i) {
                        int ri = i + R;
                        int mi = ri + dy;
                        if (mi >= N) mi -= N;
                        for (int j = 0; j < ts; ++j) {
                            int rj = j + R;
                            int mj = rj + dx;
                            if (mj >= N) mj -= N;
                            s += b.ref_tile_padded[(size_t)ri * N + rj]
                               * b.mov_patch[(size_t)mi * N + mj];
                        }
                    }
                    b.corr[(size_t)dy * N + dx] = s;
                }
            }

            // torch.fft.fftshift: roll by n // 2  →  shifted[y] = corr[(y+n/2)%n]
            {
                int shift = N / 2;
                for (int y = 0; y < N; ++y) {
                    for (int x = 0; x < N; ++x) {
                        int sy = (y + shift) % N;
                        int sx = (x + shift) % N;
                        b.shifted[(size_t)y * N + x] = b.corr[(size_t)sy * N + sx];
                    }
                }
            }

            // crop = (N - 1 - corr_size) // 2; then [crop+1 : crop+corr_size+1]
            int crop = (N - 1 - corr_size) / 2;
            int crop0 = crop + 1;
            for (int i = 0; i < corr_size; ++i)
                for (int j = 0; j < corr_size; ++j)
                    b.corrs[(size_t)i * corr_size + j] =
                        b.shifted[(size_t)(crop0 + i) * N + (crop0 + j)];

            // Box-filter sum of squares over ts×ts windows
            std::fill(b.L2_search.begin(), b.L2_search.end(), 0.f);
            for (int i = 0; i < corr_size; ++i) {
                for (int j = 0; j < corr_size; ++j) {
                    f32 sum_sq = 0.f;
                    for (int ki = 0; ki < ts; ++ki)
                        for (int kj = 0; kj < ts; ++kj) {
                            f32 v = b.mov_patch[(size_t)(i + ki) * N + (j + kj)];
                            sum_sq += v * v;
                        }
                    b.L2_search[(size_t)i * corr_size + j] = sum_sq;
                }
            }

            f32 best_err = 1e30f;
            int best_dy = 0, best_dx = 0;
            for (int i = 0; i < corr_size; ++i) {
                for (int j = 0; j < corr_size; ++j) {
                    f32 err = b.L2_search[(size_t)i * corr_size + j]
                            - 2.f * b.corrs[(size_t)i * corr_size + j];
                    if (err < best_err) {
                        best_err = err;
                        best_dy = i - corr_size / 2;
                        best_dx = j - corr_size / 2;
                    }
                }
            }

            flow.dx(ty, tx) += (f32)best_dx;
            flow.dy(ty, tx) += (f32)best_dy;
        }
    });
}

// ============================================================================
// L1 block matching — matches Python's align_lvl_block_matching_L1 CUDA kernels
// including the broken argmin (`if err < min` never updates err).
//
// cuda_L1_local_search16: err starts as s_err[0,0]
// cuda_L1_local_search32/64: err starts as +inf → condition never fires → shift 0
// ============================================================================
static void block_match_level_L1(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;
    int R = search_radius;
    int corr = 2 * R + 1;

    parallel_rows(ny, num_threads, [&](int ty) {
        std::vector<f32> s_err((size_t)corr * corr);
        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;

            // CUDA round(): half away from zero
            int flow_dx = (int)std::lround((float)flow.dx(ty, tx));
            int flow_dy = (int)std::lround((float)flow.dy(ty, tx));

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
                            f32 mv = (my >= 0 && my < moving.h &&
                                      mx >= 0 && mx < moving.w)
                                     ? moving.at(my, mx) : 0.f;
                            l1_sum += std::fabs(ref.at(ry, rx) - mv);
                        }
                    }
                    s_err[(size_t)(sdy + R) * corr + (sdx + R)] = l1_sum;
                }
            }

            // --- Python CUDA argmin bug (verbatim) ---
            //   min = s_err[i,j]
            //   if err < min:  # never assigns err = min
            //       min_shift = (j-R, i-R)
            // ts==16: err = s_err[0,0];  ts==32/64: err = +inf
            f32 err = (ts == 16)
                          ? s_err[0]
                          : std::numeric_limits<f32>::infinity();
            int min_shift_x = 0, min_shift_y = 0; // unset locals → 0 in practice
            for (int i = 0; i < corr; ++i) {
                for (int j = 0; j < corr; ++j) {
                    f32 min_v = s_err[(size_t)i * corr + j];
                    if (err < min_v) {
                        min_shift_y = i - R;
                        min_shift_x = j - R;
                    }
                }
            }

            flow.dx(ty, tx) = (f32)(flow_dx + min_shift_x);
            flow.dy(ty, tx) = (f32)(flow_dy + min_shift_y);
        }
    });
}

// ============================================================================
// Per-level ICA refinement — matches Python's align_lvl_ica + ica_kernel_*
//
// Uses Sobel gradients (same size as image), bilinear with modf + clamp-to-edge,
// and Cramer's rule update.
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
                // Matches Python's math.modf(alignment[0]) where negative values yield negative fractions
                f32 frac_x = fx - std::trunc(fx);
                int floor_off_x = (int)std::trunc(fx);
                f32 frac_y = fy - std::trunc(fy);
                int floor_off_y = (int)std::trunc(fy);

                f32 B0 = 0.f, B1 = 0.f;
                for (int i = 0; i < ts; ++i) {
                    int py = oy + i;
                    if (py >= ref.h) break;
                    for (int j = 0; j < ts; ++j) {
                        int px = ox + j;
                        if (px >= ref.w) break;

                        f32 mov_interp = bilinear_clamp(moving, py, px,
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

    int up_ny = in.ny * repeat_factor;
    int up_nx = in.nx * repeat_factor;
    FlowField upsampled(up_ny, up_nx);

    // Nearest-neighbour upscale by repeat_factor matching Python config default
    for (int ty = 0; ty < up_ny; ++ty) {
        for (int tx = 0; tx < up_nx; ++tx) {
            int sy = std::min(in.ny - 1, ty / repeat_factor);
            int sx = std::min(in.nx - 1, tx / repeat_factor);

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
