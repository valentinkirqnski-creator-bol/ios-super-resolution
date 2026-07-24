#include "stages.h"
#include "parallel.h"
#include "debug_utils.h"
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#ifdef __APPLE__
#include "metal_gpu.h"
#endif

namespace hhsr {

namespace {

// ============================================================================
// CUDA reduce emulation — same addition *order* as the Python Numba kernels
// (butterfly shared-mem for ICA 8/16; shfl_down_sync warps + sequential warp
// leaders for L1 / ICA 32/64). Still CPU float32, but the tree matches CUDA.
// ============================================================================

// ica_kernel_8/16: while N>0: if tid<N: s[tid]+=s[tid+N]; N/=2
static f32 butterfly_reduce_sum(std::vector<f32>& s, int n) {
    for (int N = n / 2; N > 0; N /= 2) {
        for (int tid = 0; tid < N; ++tid)
            s[(size_t)tid] += s[(size_t)tid + N];
    }
    return s[0];
}

// One warp: `v += __shfl_down_sync(0xffffffff, v, offset)` for offset=16..1.
// Out-of-range source returns the lane's own value (CUDA shfl_down rule).
// Returns what lane 0 holds after the reduce (the only value the kernels use).
static f32 cuda_shfl_down_warp_sum_lane0(const f32* v32) {
    constexpr int WARP = 32;
    f32 lane[WARP];
    for (int i = 0; i < WARP; ++i) lane[i] = v32[i];
    for (int offset = WARP / 2; offset > 0; offset /= 2) {
        f32 next[WARP];
        for (int i = 0; i < WARP; ++i) {
            int src = i + offset;
            f32 shfl = (src < WARP) ? lane[src] : lane[i];
            next[i] = lane[i] + shfl;
        }
        for (int i = 0; i < WARP; ++i) lane[i] = next[i];
    }
    return lane[0];
}

// L1 + ICA-32: each warp → lane0 sum, store compact s[w]=warp_sum[w], then
// tid0: total = s[0]; for w in 1..nwarps-1: total += s[w]
// (same order as writing s_l1_map[tid] at tid%32==0 and summing 0,32,64,…)
static f32 warp_then_block_reduce_sum(std::vector<f32>& vals, int n_threads) {
    constexpr int WARP = 32;
    const int nwarps = n_threads / WARP;
    std::vector<f32> warp_sums((size_t)std::max(1, nwarps), 0.f);
    for (int w = 0; w < nwarps; ++w)
        warp_sums[(size_t)w] =
            cuda_shfl_down_warp_sum_lane0(vals.data() + (size_t)w * WARP);
    f32 total = warp_sums[0];
    for (int w = 1; w < nwarps; ++w)
        total += warp_sums[(size_t)w];
    return total;
}

// ica_kernel_64: warp0 sum stays in register; warps 1.. written to shared;
// tid0: B0 += s_B0[i] for i in 1..nwarps-1  (identical sequential order).
static f32 warp_reduce_ica64(std::vector<f32>& vals /* >= 1024 */) {
    constexpr int WARP = 32;
    constexpr int N_THREADS = 1024;
    constexpr int NWARPS = N_THREADS / WARP;
    f32 warp0 = cuda_shfl_down_warp_sum_lane0(vals.data());
    f32 total = warp0;
    for (int w = 1; w < NWARPS; ++w)
        total += cuda_shfl_down_warp_sum_lane0(vals.data() + (size_t)w * WARP);
    return total;
}

// Torch / NumPy round-half-to-even (L2 extract_flow_patches: flow.round()).
static inline int torch_round_to_int(f32 x) {
    return (int)std::rint(x);
}

// CUDA round(): half away from zero (L1 kernels).
static inline int cuda_round_to_int(f32 x) {
    return (int)std::lround(x);
}

// Per-tile Hessian [ny,nx,2,2] packed as 00,01,10,11 — matches init_ica once per level.
struct HessianField {
    int ny = 0, nx = 0;
    std::vector<f32> data; // ny*nx*4
    f32* at(int ty, int tx) { return &data[((size_t)ty * nx + tx) * 4]; }
    const f32* at(int ty, int tx) const { return &data[((size_t)ty * nx + tx) * 4]; }
};

static HessianField compute_hessian(const Image& gradx, const Image& grady, int ts) {
    // Python init_ica: ny, nx = h // tile_size, w // tile_size
    int ny = gradx.h / ts;
    int nx = gradx.w / ts;
    HessianField H;
    H.ny = ny;
    H.nx = nx;
    H.data.assign((size_t)ny * nx * 4, 0.f);
    for (int ty = 0; ty < ny; ++ty) {
        for (int tx = 0; tx < nx; ++tx) {
            f32* h = H.at(ty, tx);
            int oy = ty * ts, ox = tx * ts;
            for (int i = 0; i < ts; ++i) {
                int py = oy + i;
                if (py >= gradx.h) break;
                for (int j = 0; j < ts; ++j) {
                    int px = ox + j;
                    if (px >= gradx.w) break;
                    f32 gx = gradx.at(py, px);
                    f32 gy = grady.at(py, px);
                    h[0] += gx * gx;
                    h[1] += gx * gy;
                    h[2] += gx * gy;
                    h[3] += gy * gy;
                }
            }
        }
    }
    return H;
}

} // namespace

// ============================================================================
// Sobel — F.conv2d(..., padding='same') with zero padding
// ============================================================================
static Image compute_sobel_gradx(const Image& img) {
    Image out(img.h, img.w, 1);
    parallel_rows(img.h, 0, [&](int y) {
        for (int x = 0; x < img.w; ++x) {
            f32 vm = (x - 1 >= 0) ? img.at(y, x - 1) : 0.f;
            f32 vp = (x + 1 < img.w) ? img.at(y, x + 1) : 0.f;
            out.at(y, x) = -vm + vp;
        }
    });
    return out;
}

static Image compute_sobel_grady(const Image& img) {
    Image out(img.h, img.w, 1);
    parallel_rows(img.h, 0, [&](int y) {
        for (int x = 0; x < img.w; ++x) {
            f32 vm = (y - 1 >= 0) ? img.at(y - 1, x) : 0.f;
            f32 vp = (y + 1 < img.h) ? img.at(y + 1, x) : 0.f;
            out.at(y, x) = -vm + vp;
        }
    });
    return out;
}

// ICA bilinear: tile 8 clamp-to-edge; 16/32/64 OOB → 0
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
// L2 BM — Torch: rfft2 / irfft2 / fftshift / L2_search-2*corr / argmin
// Metal uses the same formulas; FFT numerics ≠ Torch. Set HHSR_L2_CPU=1 to
// force the vDSP CPU path (closer to Torch; still float ε vs CUDA).
// ============================================================================
static bool env_flag_on(const char* name) {
    const char* e = std::getenv(name);
    return e && e[0] == '1' && e[1] == '\0';
}

static void block_match_level_L2_cpu(const Image& ref, const Image& moving,
                                     int tile_size, int search_radius,
                                     FlowField& flow, int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;
    int R = search_radius;
    int search_size = 2 * R + ts;
    int corr_size = 2 * R + 1;
    const int N = search_size;
    const int wh = N / 2 + 1;
    const size_t NWh = (size_t)N * wh;

    struct RowBuffers {
        std::vector<f32> ref_tile_padded;
        std::vector<f32> mov_patch;
        std::vector<std::complex<f32>> F_ref;
        std::vector<std::complex<f32>> F_mov;
        std::vector<f32> corr;
        std::vector<f32> corrs;
        std::vector<f32> L2_search;
        RowBuffers(int n, int c_size, int wh_)
            : ref_tile_padded(n * n, 0.f), mov_patch(n * n, 0.f),
              F_ref((size_t)n * wh_), F_mov((size_t)n * wh_),
              corr(n * n), corrs(c_size * c_size),
              L2_search(c_size * c_size, 0.f) {}
    };

    std::vector<RowBuffers> buffers;
    buffers.reserve((size_t)ny);
    for (int i = 0; i < ny; ++i)
        buffers.emplace_back(N, corr_size, wh);

    parallel_rows(ny, num_threads, [&](int ty) {
        RowBuffers& b = buffers[(size_t)ty];
        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;

            // torch: flow.round().long() — round half to even
            int flow_dx = torch_round_to_int(flow.dx(ty, tx));
            int flow_dy = torch_round_to_int(flow.dy(ty, tx));

            std::fill(b.ref_tile_padded.begin(), b.ref_tile_padded.end(), 0.f);
            for (int i = 0; i < ts; ++i) {
                for (int j = 0; j < ts; ++j) {
                    int ry = oy + i, rx = ox + j;
                    if (ry < ref.h && rx < ref.w)
                        b.ref_tile_padded[(size_t)(i + R) * N + (j + R)] = ref.at(ry, rx);
                }
            }

            // extract_flow_patches: clamp to edges
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    int my = std::max(0, std::min(moving.h - 1, oy + flow_dy + i - R));
                    int mx = std::max(0, std::min(moving.w - 1, ox + flow_dx + j - R));
                    b.mov_patch[(size_t)i * N + j] = moving.at(my, mx);
                }
            }

            // corrs = fftshift(irfft2(conj(rfft2(ref)) * rfft2(mov), s=N))
            rfft2(b.ref_tile_padded.data(), N, N, b.F_ref);
            rfft2(b.mov_patch.data(), N, N, b.F_mov);
            for (size_t i = 0; i < NWh; ++i)
                b.F_ref[i] = std::conj(b.F_ref[i]) * b.F_mov[i];
            irfft2(b.F_ref, N, N, b.corr);
            fftshift2d_real(b.corr, N, N);

            int crop = (N - 1 - corr_size) / 2;
            int crop0 = crop + 1;
            for (int i = 0; i < corr_size; ++i)
                for (int j = 0; j < corr_size; ++j)
                    b.corrs[(size_t)i * corr_size + j] =
                        b.corr[(size_t)(crop0 + i) * N + (crop0 + j)];

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

static void block_match_level_L2(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, int num_threads) {
#ifdef __APPLE__
    // Default: Metal (same math, different FFT). HHSR_L2_CPU=1 → vDSP path.
    if (!env_flag_on("HHSR_L2_CPU") && !env_flag_on("HHSR_ALIGN_CPU")) {
        if (block_match_level_L2_metal(ref, moving, tile_size, search_radius, flow))
            return;
    }
#endif
    block_match_level_L2_cpu(ref, moving, tile_size, search_radius, flow, num_threads);
}

// ============================================================================
// L1 BM — CUDA kernels + broken argmin. ts=64 uses 1024 threads × 4 px.
// ============================================================================
static void block_match_level_L1(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, int num_threads) {
#ifdef __APPLE__
    if (!env_flag_on("HHSR_L1_CPU") && !env_flag_on("HHSR_ALIGN_CPU") &&
        block_match_level_L1_metal(ref, moving, tile_size, search_radius, flow))
        return;
#endif
    int ny = flow.ny, nx = flow.nx;
    int ts = tile_size;
    int R = search_radius;
    int corr = 2 * R + 1;

    parallel_rows(ny, num_threads, [&](int ty) {
        std::vector<f32> s_err((size_t)corr * corr);
        std::vector<f32> per_thread(1024);
        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;
            int flow_dx = cuda_round_to_int(flow.dx(ty, tx));
            int flow_dy = cuda_round_to_int(flow.dy(ty, tx));

            for (int sdy = -R; sdy <= R; ++sdy) {
                for (int sdx = -R; sdx <= R; ++sdx) {
                    f32 l1_sum = 0.f;
                    if (ts == 64) {
                        // cuda_L1_local_search64: ignores shift_x/y in the L1 sum
                        // (block_matching.py:307-310) — every shift gets the same
                        // cost; with the broken argmin this is round(flow) only.
                        std::fill(per_thread.begin(), per_thread.end(), 0.f);
                        for (int tyy = 0; tyy < 16; ++tyy) {
                            for (int txx = 0; txx < 64; ++txx) {
                                int ti = tyy * 64 + txx;
                                int px = ox + txx;
                                int py0 = oy + tyy * 4;
                                f32 acc = 0.f;
                                for (int k = 0; k < 4; ++k) {
                                    int py = py0 + k;
                                    f32 rv = (py < ref.h && px < ref.w) ? ref.at(py, px) : 0.f;
                                    int my = py + flow_dy; // no + sdy
                                    int mx = px + flow_dx; // no + sdx
                                    f32 mv = (my >= 0 && my < moving.h && mx >= 0 && mx < moving.w)
                                                 ? moving.at(my, mx) : 0.f;
                                    acc += std::fabs(rv - mv);
                                }
                                per_thread[(size_t)ti] = acc;
                            }
                        }
                        l1_sum = warp_then_block_reduce_sum(per_thread, 1024);
                    } else {
                        // ts==16 Python cuda_L1_local_search16 loads shared mov
                        // WITHOUT -search_radius, then indexes with +search_radius
                        // → effective sample at (ry+flow+sdy+R, rx+flow+sdx+R).
                        // ts==32/64 load with -R so the +R index is correct L1.
                        const int off = (ts == 16) ? R : 0;
                        const int n_threads = ts * ts;
                        if ((int)per_thread.size() < n_threads)
                            per_thread.resize((size_t)n_threads);
                        std::fill(per_thread.begin(), per_thread.begin() + n_threads, 0.f);
                        for (int i = 0; i < ts; ++i) {
                            int ry = oy + i;
                            for (int j = 0; j < ts; ++j) {
                                int rx = ox + j;
                                int tid = i * ts + j;
                                f32 rv = (ry < ref.h && rx < ref.w) ? ref.at(ry, rx) : 0.f;
                                int my = ry + flow_dy + sdy + off;
                                int mx = rx + flow_dx + sdx + off;
                                f32 mv = (my >= 0 && my < moving.h && mx >= 0 && mx < moving.w)
                                             ? moving.at(my, mx) : 0.f;
                                per_thread[(size_t)tid] = std::fabs(rv - mv);
                            }
                        }
                        if (ts == 16 || ts == 32) {
                            int nt = n_threads;
                            if (nt % 32) nt = ((nt + 31) / 32) * 32;
                            if ((int)per_thread.size() < nt) per_thread.resize(nt, 0.f);
                            l1_sum = warp_then_block_reduce_sum(per_thread, nt);
                        } else {
                            for (int t = 0; t < n_threads; ++t)
                                l1_sum += per_thread[(size_t)t];
                        }
                    }
                    s_err[(size_t)(sdy + R) * corr + (sdx + R)] = l1_sum;
                }
            }

            // Python CUDA argmin bug (verbatim)
            f32 err = (ts == 16) ? s_err[0] : std::numeric_limits<f32>::infinity();
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
// ICA — precomputed Hessian (init_ica); CUDA reduce trees
// ============================================================================
static void ica_refine_level(const Image& ref, const Image& gradx,
                              const Image& grady, const Image& moving,
                              const HessianField& hessian,
                              FlowField& flow, int tile_size, int n_iter,
                              int num_threads) {
#ifdef __APPLE__
    // Metal ica_kernel_8/16 — same math/order as CPU path below / Python ICA.py.
    if (ica_refine_level_metal(ref, gradx, grady, hessian.data, moving, flow,
                               tile_size, n_iter))
        return;
#endif
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

            const f32* h = hessian.at(ty, tx);
            f32 h00 = h[0], h01 = h[1], h10 = h[2], h11 = h[3];
            f32 det = h00 * h11 - h01 * h10;
            if (std::fabs(det) < 1e-10f) continue;
            f32 det_inv = 1.f / det;

            f32 fx = flow.dx(ty, tx);
            f32 fy = flow.dy(ty, tx);

            for (int it = 0; it < n_iter; ++it) {
                // math.modf + int() truncation toward zero (ICA.py)
                f32 frac_x = fx - std::trunc(fx);
                int floor_off_x = (int)std::trunc(fx);
                f32 frac_y = fy - std::trunc(fy);
                int floor_off_y = (int)std::trunc(fy);

                std::fill(s_B0.begin(), s_B0.end(), 0.f);
                std::fill(s_B1.begin(), s_B1.end(), 0.f);

                if (ts == 64) {
                    // ica_kernel_64 sliding bilinear — port Python verbatim
                    // (including the floor_y+=1 / read floor_y+1 quirk).
                    for (int tyy = 0; tyy < 16; ++tyy) {
                        for (int txx = 0; txx < 64; ++txx) {
                            int ti = tyy * 64 + txx;
                            int px = ox + txx;
                            int py0 = oy + tyy * 4;
                            int floor_x = px + floor_off_x;
                            int floor_y = py0 + floor_off_y;
                            f32 m10 = sample_or_zero(moving, floor_y, floor_x);
                            f32 m11 = sample_or_zero(moving, floor_y, floor_x + 1);
                            f32 lerpx_bot = m10 + (m11 - m10) * frac_x;
                            f32 B0 = 0.f, B1 = 0.f;
                            for (int k = 0; k < 4; ++k) {
                                int py = py0 + k;
                                floor_y += 1;
                                m10 = sample_or_zero(moving, floor_y + 1, floor_x);
                                m11 = sample_or_zero(moving, floor_y + 1, floor_x + 1);
                                f32 lerpx_top = lerpx_bot;
                                lerpx_bot = m10 + (m11 - m10) * frac_x;
                                f32 mov_interp = lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
                                if (py < ref.h && px < ref.w) {
                                    f32 gradt = mov_interp - ref.at(py, px);
                                    B0 += -gradx.at(py, px) * gradt;
                                    B1 += -grady.at(py, px) * gradt;
                                }
                            }
                            s_B0[(size_t)ti] = B0;
                            s_B1[(size_t)ti] = B1;
                        }
                    }
                    f32 B0 = warp_reduce_ica64(s_B0);
                    f32 B1 = warp_reduce_ica64(s_B1);
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
                            s_B0[(size_t)tid] = -gradx.at(py, px) * gradt;
                            s_B1[(size_t)tid] = -grady.at(py, px) * gradt;
                        }
                    }
                    f32 B0, B1;
                    if (ts == 8 || ts == 16) {
                        B0 = butterfly_reduce_sum(s_B0, n_pix);
                        B1 = butterfly_reduce_sum(s_B1, n_pix);
                    } else {
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

// upscale_lvl — nearest (default.yaml flow_upscale_mode)
static FlowField upscale_flow(const FlowField& in, int target_ny, int target_nx,
                               int upsample_factor, int new_tile_size,
                               int prev_tile_size) {
    int tile_ratio = new_tile_size / std::max(1, prev_tile_size);
    int repeat_factor = upsample_factor / std::max(1, tile_ratio);
    if (repeat_factor < 1) repeat_factor = 1;

    int up_ny = in.ny * repeat_factor;
    int up_nx = in.nx * repeat_factor;
    FlowField upsampled(up_ny, up_nx);
    for (int ty = 0; ty < up_ny; ++ty) {
        for (int tx = 0; tx < up_nx; ++tx) {
            int sy = std::min(in.ny - 1, ty / repeat_factor);
            int sx = std::min(in.nx - 1, tx / repeat_factor);
            upsampled.dx(ty, tx) = in.dx(sy, sx) * (f32)upsample_factor;
            upsampled.dy(ty, tx) = in.dy(sy, sx) * (f32)upsample_factor;
        }
    }

    FlowField out(target_ny, target_nx);
    for (int ty = 0; ty < target_ny; ++ty) {
        for (int tx = 0; tx < target_nx; ++tx) {
            if (ty < up_ny && tx < up_nx) {
                out.dx(ty, tx) = upsampled.dx(ty, tx);
                out.dy(ty, tx) = upsampled.dy(ty, tx);
            }
        }
    }
    return out;
}

// Ref Sobel+Hessian are independent of the moving frame — reuse across
// comparison frames in one burst. Cleared before merge (see clear_align_ref_ica_cache).
struct RefIcaLevel {
    Image gx, gy;
    HessianField hess;
};
struct RefIcaBurstCache {
    const void* key = nullptr;
    std::vector<RefIcaLevel> levels;
};
static RefIcaBurstCache g_ref_ica_cache;

void clear_align_ref_ica_cache() {
    g_ref_ica_cache = {};
#ifdef __APPLE__
    metal_clear_ref_ica_cache();
#endif
}

// ============================================================================
// align() — Python alignment.align
// ref_grey must already be circular-padded (init_alignment); moving is NOT.
// ============================================================================
FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg, int tile_size) {
    (void)ref_grey; // pyramid already built from padded ref
    int nlev = (int)ref_pyr.levels.size();

#ifdef __APPLE__
    // Default iOS path: Metal alignment. HHSR_ALIGN_CPU=1 forces the C++ path.
    if (!env_flag_on("HHSR_ALIGN_CPU")) {
        FlowField flow_gpu;
        if (align_metal(ref_pyr, moving_grey, cfg, tile_size, flow_gpu))
            return flow_gpu;
    }
#endif

    // CPU path: cache ref Sobel+Hessian across comparison frames.
    // C++ pyramid is fine-first (levels[0]=finest). Python build_gaussian_pyramid
    // returns coarse-first (pyramid[::-1]); its list_id = n-1-l then maps
    // coarse→params[n-1], fine→params[0]. With fine-first storage that is simply
    // params[lvl] (arrays are fine→coarse in default.yaml).
    if (g_ref_ica_cache.key != (const void*)&ref_pyr ||
        (int)g_ref_ica_cache.levels.size() != nlev) {
        g_ref_ica_cache.key = (const void*)&ref_pyr;
        g_ref_ica_cache.levels.assign((size_t)nlev, RefIcaLevel{});
        for (int lvl = 0; lvl < nlev; ++lvl) {
            const Image& r = ref_pyr.levels[lvl];
            int ts = (lvl < (int)cfg.bm_tile_sizes.size())
                         ? cfg.bm_tile_sizes[lvl] : tile_size;
            RefIcaLevel& L = g_ref_ica_cache.levels[(size_t)lvl];
            L.gx = compute_sobel_gradx(r);
            L.gy = compute_sobel_grady(r);
            L.hess = compute_hessian(L.gx, L.gy, ts);
        }
        // Python dumps pyramid/grads at enum i==0 after reverse = coarsest.
        if (nlev > 0) {
            const RefIcaLevel& Lc = g_ref_ica_cache.levels[(size_t)nlev - 1];
            debug_dump_bin("cpp_gradx_0", Lc.gx.data.data(), Lc.gx.data.size());
            debug_dump_bin("cpp_grady_0", Lc.gy.data.data(), Lc.gy.data.size());
        }
    }

    // Moving: unpadded grey → pyramid (matches Python align())
    Pyramid mov_pyr = build_pyramid(moving_grey, cfg.bm_factors);

    FlowField flow;

    for (int lvl = nlev - 1; lvl >= 0; --lvl) {
        const Image& r = ref_pyr.levels[lvl];
        const Image& m = mov_pyr.levels[lvl];

        int ts = (lvl < (int)cfg.bm_tile_sizes.size())
                     ? cfg.bm_tile_sizes[lvl] : tile_size;
        int radius = (lvl < (int)cfg.bm_search_radii.size())
                     ? cfg.bm_search_radii[lvl] : 2;

        // Tile grid from padded ref level (Python: h // tile_size)
        int ny = r.h / ts;
        int nx = r.w / ts;

        if (flow.nx == 0) {
            flow = FlowField(ny, nx);
        } else {
            int upsample_factor = ((lvl + 1) < (int)cfg.bm_factors.size())
                                  ? cfg.bm_factors[lvl + 1] : 1;
            int prev_ts = ((lvl + 1) < (int)cfg.bm_tile_sizes.size())
                          ? cfg.bm_tile_sizes[lvl + 1]
                          : ts;
            flow = upscale_flow(flow, ny, nx, upsample_factor, ts, prev_ts);
        }

        std::string metric = "L2";
        if (lvl < (int)cfg.bm_metrics.size())
            metric = cfg.bm_metrics[lvl];

        if (metric == "L1")
            block_match_level_L1(r, m, ts, radius, flow, cfg.num_threads);
        else
            block_match_level_L2(r, m, ts, radius, flow, cfg.num_threads);

        const RefIcaLevel& L = g_ref_ica_cache.levels[(size_t)lvl];
        ica_refine_level(r, L.gx, L.gy, m, L.hess, flow, ts, cfg.ica_n_iter,
                         cfg.num_threads);
    }

    return flow;
}

} // namespace hhsr
