#include "stages.h"
#include "parallel.h"
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace hhsr {

namespace {

// CUDA-style pairwise tree reduce (ica_kernel_8/16 shared-mem loop).
static f32 butterfly_reduce_sum(std::vector<f32>& s, int n) {
    int N = n / 2;
    while (N > 0) {
        for (int tid = 0; tid < N; ++tid)
            s[(size_t)tid] += s[(size_t)tid + N];
        N /= 2;
    }
    return s[0];
}

// CUDA shfl_down_sync warp reduce then sum warp leaders (L1/ICA 32/64).
// vals[tid] = per-thread contribution; n_threads multiple of 32.
static f32 warp_then_block_reduce_sum(std::vector<f32>& vals, int n_threads) {
    constexpr int WARP = 32;
    std::vector<f32> tmp(vals.size());
    for (int base = 0; base < n_threads; base += WARP) {
        for (int offset = WARP / 2; offset > 0; offset /= 2) {
            tmp = vals;
            for (int lane = 0; lane < WARP; ++lane) {
                int src = lane + offset;
                f32 add = (src < WARP) ? tmp[(size_t)base + src] : 0.f;
                vals[(size_t)base + lane] = tmp[(size_t)base + lane] + add;
            }
        }
    }
    f32 sum = vals[0];
    for (int w = WARP; w < n_threads; w += WARP)
        sum += vals[(size_t)w];
    return sum;
}

} // namespace

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
//   tile 8 (ica_kernel_8): clamp-to-edge
//   tile 16/32/64: OOB samples → 0.0
// ============================================================================
static inline f32 sample_or_zero(const Image& img, int y, int x) {
    return (y >= 0 && y < img.h && x >= 0 && x < img.w) ? img.at(y, x) : 0.f;
}

static inline f32 bilinear_ica(const Image& img, int pixel_y, int pixel_x,
                               int floor_off_y, int floor_off_x,
                               f32 frac_x, f32 frac_y, bool clamp_edge) {
    int floor_y = pixel_y + floor_off_y;
    int floor_x = pixel_x + floor_off_x;

    f32 m00, m01, m10, m11;
    if (clamp_edge) {
        int fy = std::max(0, std::min(img.h - 1, floor_y));
        int fx = std::max(0, std::min(img.w - 1, floor_x));
        int cy = std::max(0, std::min(img.h - 1, fy + 1));
        int cx = std::max(0, std::min(img.w - 1, fx + 1));
        m00 = img.at(fy, fx);
        m01 = img.at(fy, cx);
        m10 = img.at(cy, fx);
        m11 = img.at(cy, cx);
    } else {
        m00 = sample_or_zero(img, floor_y + 0, floor_x + 0);
        m01 = sample_or_zero(img, floor_y + 0, floor_x + 1);
        m10 = sample_or_zero(img, floor_y + 1, floor_x + 0);
        m11 = sample_or_zero(img, floor_y + 1, floor_x + 1);
    }

    f32 lerpx_top = m00 + (m01 - m00) * frac_x;
    f32 lerpx_bot = m10 + (m11 - m10) * frac_x;
    return lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
}

// ============================================================================
// FFT-based L2 block matching — matches Python's align_lvl_block_matching_L2
//
// PyTorch: corrs = fftshift(irfft2(conj(rfft2(ref)) * rfft2(mov), s=N))
// C++:     same via full complex fft2 (real input) — same DFT ops/order family
//          as rfft2/irfft2, not spatial accumulation.
// ============================================================================

static void block_match_level_L2(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;
    int R = search_radius;
    int search_size = 2 * R + ts;
    int corr_size = 2 * R + 1;
    const int N = search_size;
    const size_t NN = (size_t)N * N;

    struct RowBuffers {
        std::vector<f32> ref_tile_padded;
        std::vector<f32> mov_patch;
        std::vector<std::complex<f32>> F_ref;
        std::vector<std::complex<f32>> F_mov;
        std::vector<f32> corr;
        std::vector<f32> shifted;
        std::vector<f32> corrs;
        std::vector<f32> L2_search;
        std::vector<std::complex<f32>> row_buf;
        std::vector<std::complex<f32>> dft_buf;

        RowBuffers(int n, int c_size)
            : ref_tile_padded(n * n, 0.f), mov_patch(n * n, 0.f),
              F_ref(n * n), F_mov(n * n),
              corr(n * n), shifted(n * n),
              corrs(c_size * c_size), L2_search(c_size * c_size, 0.f),
              row_buf(n), dft_buf(n) {}
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

            std::fill(b.ref_tile_padded.begin(), b.ref_tile_padded.end(), 0.f);
            for (int i = 0; i < ts; ++i) {
                for (int j = 0; j < ts; ++j) {
                    int ry = oy + i;
                    int rx = ox + j;
                    if (ry < ref.h && rx < ref.w)
                        b.ref_tile_padded[(size_t)(i + R) * N + (j + R)] = ref.at(ry, rx);
                }
            }

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    int my = std::max(0, std::min(moving.h - 1, oy + flow_dy + i - R));
                    int mx = std::max(0, std::min(moving.w - 1, ox + flow_dx + j - R));
                    b.mov_patch[(size_t)i * N + j] = moving.at(my, mx);
                }
            }

            // irfft2(conj(rfft2(ref)) * rfft2(mov)) via full fft2 of real arrays
            for (size_t i = 0; i < NN; ++i) {
                b.F_ref[i] = {b.ref_tile_padded[i], 0.f};
                b.F_mov[i] = {b.mov_patch[i], 0.f};
            }
            fft2d(b.F_ref, N, N, false, &b.row_buf, &b.dft_buf);
            fft2d(b.F_mov, N, N, false, &b.row_buf, &b.dft_buf);
            for (size_t i = 0; i < NN; ++i)
                b.F_ref[i] = std::conj(b.F_ref[i]) * b.F_mov[i];
            fft2d(b.F_ref, N, N, true, &b.row_buf, &b.dft_buf);
            for (size_t i = 0; i < NN; ++i)
                b.corr[i] = b.F_ref[i].real();

            // torch.fft.fftshift: out[i] = in[(i + n//2) % n]  (even N: same as roll)
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

            int crop = (N - 1 - corr_size) / 2;
            int crop0 = crop + 1;
            for (int i = 0; i < corr_size; ++i)
                for (int j = 0; j < corr_size; ++j)
                    b.corrs[(size_t)i * corr_size + j] =
                        b.shifted[(size_t)(crop0 + i) * N + (crop0 + j)];

            // Valid box filter on search_area.square() (conv2d padding=valid)
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

            // torch.argmin — first minimum
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
// cuda_L1_local_search16: err starts as s_err[0,0]; L1 sum via warp reduce
// cuda_L1_local_search32/64: err starts as +inf → condition never fires →
//   min_shift never written in CUDA (UB); Numba locals are typically 0 → we use 0
// ============================================================================
static void block_match_level_L1(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;
    int R = search_radius;
    int corr = 2 * R + 1;
    const int n_threads = ts * ts; // 1 thread/pixel (ts=64 uses 1024 with 4px/thread in CUDA;
                                   // we still reduce over ts*ts pixel contribs in CUDA order)

    parallel_rows(ny, num_threads, [&](int ty) {
        std::vector<f32> s_err((size_t)corr * corr);
        std::vector<f32> per_thread((size_t)std::max(n_threads, 32));
        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;

            // CUDA round(): half away from zero
            int flow_dx = (int)std::lround((float)flow.dx(ty, tx));
            int flow_dy = (int)std::lround((float)flow.dy(ty, tx));

            for (int sdy = -R; sdy <= R; ++sdy) {
                for (int sdx = -R; sdx <= R; ++sdx) {
                    // Per-pixel |ref-mov| then CUDA-order block reduce
                    std::fill(per_thread.begin(), per_thread.end(), 0.f);
                    for (int i = 0; i < ts; ++i) {
                        int ry = oy + i;
                        for (int j = 0; j < ts; ++j) {
                            int rx = ox + j;
                            int tid = i * ts + j; // ty*TILE + tx with ty=i, tx=j
                            f32 rv = (ry < ref.h && rx < ref.w) ? ref.at(ry, rx) : 0.f;
                            int my = ry + flow_dy + sdy;
                            int mx = rx + flow_dx + sdx;
                            f32 mv = (my >= 0 && my < moving.h && mx >= 0 && mx < moving.w)
                                         ? moving.at(my, mx) : 0.f;
                            per_thread[(size_t)tid] = std::fabs(rv - mv);
                        }
                    }
                    f32 l1_sum;
                    if (ts == 16 || ts == 32 || ts == 64) {
                        // Pad to multiple of 32 for warp reduce (ts=16 → 256)
                        int nt = n_threads;
                        if (nt % 32) nt = ((nt + 31) / 32) * 32;
                        if ((int)per_thread.size() < nt) per_thread.resize(nt, 0.f);
                        l1_sum = warp_then_block_reduce_sum(per_thread, nt);
                    } else {
                        // ts==8 not implemented in Python L1; sequential fallback
                        l1_sum = 0.f;
                        for (int t = 0; t < n_threads; ++t) l1_sum += per_thread[(size_t)t];
                    }
                    s_err[(size_t)(sdy + R) * corr + (sdx + R)] = l1_sum;
                }
            }

            // --- Python CUDA argmin bug (verbatim) ---
            f32 err = (ts == 16)
                          ? s_err[0]
                          : std::numeric_limits<f32>::infinity();
            // Uninitialized in CUDA when the if never fires; device locals → 0
            int min_shift_x = 0, min_shift_y = 0;
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
// B accumulation uses the same reduce trees as the CUDA kernels (butterfly for
// 8/16, warp+block for 32/64).
// ============================================================================
static void ica_refine_level(const Image& ref, const Image& gradx,
                              const Image& grady, const Image& moving,
                              FlowField& flow, int tile_size, int n_iter,
                              int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;
    const bool clamp_edge = (ts == 8);
    const int n_pix = ts * ts;

    parallel_rows(ny, num_threads, [&](int ty) {
        std::vector<f32> s_B0((size_t)std::max(n_pix, 1024));
        std::vector<f32> s_B1((size_t)std::max(n_pix, 1024));

        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;

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

            f32 det = h00 * h11 - h01 * h10;
            if (std::fabs(det) < 1e-10f) continue;
            f32 det_inv = 1.f / det;

            f32 fx = flow.dx(ty, tx);
            f32 fy = flow.dy(ty, tx);

            for (int it = 0; it < n_iter; ++it) {
                f32 frac_x = fx - std::trunc(fx);
                int floor_off_x = (int)std::trunc(fx);
                f32 frac_y = fy - std::trunc(fy);
                int floor_off_y = (int)std::trunc(fy);

                std::fill(s_B0.begin(), s_B0.end(), 0.f);
                std::fill(s_B1.begin(), s_B1.end(), 0.f);

                if (ts == 64) {
                    // ica_kernel_64: 16×64 threads, each owns 4 vertical pixels
                    for (int tyy = 0; tyy < 16; ++tyy) {
                        for (int txx = 0; txx < 64; ++txx) {
                            int ti = tyy * 64 + txx;
                            int px = ox + txx;
                            int py0 = oy + tyy * 4;
                            f32 B0 = 0.f, B1 = 0.f;
                            for (int k = 0; k < 4; ++k) {
                                int py = py0 + k;
                                if (py >= ref.h || px >= ref.w) continue;
                                f32 mov_interp = bilinear_ica(moving, py, px,
                                                              floor_off_y, floor_off_x,
                                                              frac_x, frac_y, false);
                                f32 gradt = mov_interp - ref.at(py, px);
                                f32 gx = gradx.at(py, px);
                                f32 gy = grady.at(py, px);
                                B0 += -gx * gradt;
                                B1 += -gy * gradt;
                            }
                            s_B0[(size_t)ti] = B0;
                            s_B1[(size_t)ti] = B1;
                        }
                    }
                    // Warp reduce with warp0 kept in "register" path:
                    // same as warp_then_block_reduce_sum (adds vals[0]+vals[32]+...)
                    f32 B0 = warp_then_block_reduce_sum(s_B0, 1024);
                    f32 B1 = warp_then_block_reduce_sum(s_B1, 1024);
                    fx += det_inv * (h11 * B0 - h01 * B1);
                    fy += det_inv * (-h10 * B0 + h00 * B1);
                } else {
                    for (int i = 0; i < ts; ++i) {
                        int py = oy + i;
                        for (int j = 0; j < ts; ++j) {
                            int px = ox + j;
                            int tid = i * ts + j;
                            if (py >= ref.h || px >= ref.w) {
                                s_B0[(size_t)tid] = 0.f;
                                s_B1[(size_t)tid] = 0.f;
                                continue;
                            }
                            f32 mov_interp = bilinear_ica(moving, py, px,
                                                          floor_off_y, floor_off_x,
                                                          frac_x, frac_y, clamp_edge);
                            f32 gradt = mov_interp - ref.at(py, px);
                            f32 gx = gradx.at(py, px);
                            f32 gy = grady.at(py, px);
                            s_B0[(size_t)tid] = -gx * gradt;
                            s_B1[(size_t)tid] = -gy * gradt;
                        }
                    }
                    f32 B0, B1;
                    if (ts == 8 || ts == 16) {
                        B0 = butterfly_reduce_sum(s_B0, n_pix);
                        B1 = butterfly_reduce_sum(s_B1, n_pix);
                    } else {
                        // ts == 32 (and any other): warp + block reduce
                        int nt = n_pix;
                        if (nt % 32) nt = ((nt + 31) / 32) * 32;
                        B0 = warp_then_block_reduce_sum(s_B0, nt);
                        B1 = warp_then_block_reduce_sum(s_B1, nt);
                    }
                    fx += det_inv * (h11 * B0 - h01 * B1);
                    fy += det_inv * (-h10 * B0 + h00 * B1);
                }
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
