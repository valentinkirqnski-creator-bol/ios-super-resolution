#include "stages.h"
#include "parallel.h"
#include "debug_utils.h"
#include "prof.h"
#include <atomic>
#include <cmath>
#include <mutex>
#include <complex>
#include <cstdlib>
#include <limits>
#include <algorithm>
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
    // Python init_ICA: ny, nx = ceil(h / tile_size), ceil(w / tile_size).
    int ny = (gradx.h + ts - 1) / ts;
    int nx = (gradx.w + ts - 1) / ts;
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

// ICA damping ratio: the eigenvalue ratio the solve is regularized toward,
// i.e. the ratio below which a tile counts as 1D. 0 disables.
static f32 ica_damp_ratio(const Config& cfg) {
    if (!cfg.ica_regularize_enabled) return 0.f;
    const f32 r = cfg.flow_regularize_aperture_ratio;
    if (!std::isfinite(r) || r <= 0.f) return 0.f;
    return std::min(r, 1.f);
}

// Per-iteration step bound: whatever the block matching that preceded ICA could
// have reached. A larger step means ICA has left the region the search actually
// evaluated. 0 disables.
static f32 ica_max_step(const Config& cfg, int search_radius) {
    if (!cfg.ica_regularize_enabled) return 0.f;
    return (f32)std::max(1, search_radius);
}

static f32 clamped_ambiguity_ratio(const Config& cfg) {
    f32 ambiguity_ratio = cfg.flow_reject_1d_ambiguity_ratio;
    if (!std::isfinite(ambiguity_ratio)) ambiguity_ratio = 1.10f;
    return std::max(ambiguity_ratio, 1.f);
}

// Least-squares bivariate quadratic over the 3x3 cost neighbourhood of a
// block-matching winner: f(x,y) = a x^2 + b y^2 + c xy + d x + e y + g fitted
// to the nine costs D[(y+1)*3+(x+1)], x,y in {-1,0,1}, minimum at
// mu = -H^-1 g with H = [[2a, c],[c, 2b]], g = [d, e]. This is the sub-pixel
// estimator Wronski's alignment specifies (ImageStackAlignator kernel.cu
// carries the same fit); the closed forms below are the normal equations of
// that least-squares problem on the 3x3 lattice.
//
// Returns false -- leave the integer result alone -- unless every cost is
// finite, the fitted Hessian is positive definite (a true 2-D minimum, not a
// ridge or saddle), and the sub-cell minimum lies within half a cell of the
// winner on both axes (the winner IS the integer argmin, so a fit that says
// otherwise is noise, not signal).
static bool quadratic_subpixel_3x3(const f32* D, f32& mu_x, f32& mu_y) {
    for (int i = 0; i < 9; ++i)
        if (!std::isfinite(D[i])) return false;
    const f32 P   = D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7] + D[8];
    const f32 Sx  = (D[2] + D[5] + D[8]) - (D[0] + D[3] + D[6]);
    const f32 Sy  = (D[6] + D[7] + D[8]) - (D[0] + D[1] + D[2]);
    const f32 Sxx = D[0] + D[2] + D[3] + D[5] + D[6] + D[8];
    const f32 Syy = D[0] + D[1] + D[2] + D[6] + D[7] + D[8];
    const f32 Sxy = (D[0] + D[8]) - (D[2] + D[6]);
    const f32 d = Sx / 6.f;
    const f32 e = Sy / 6.f;
    const f32 c = Sxy / 4.f;
    const f32 a = 0.5f * Sxx - P / 3.f;
    const f32 b = 0.5f * Syy - P / 3.f;
    const f32 h11 = 2.f * a, h22 = 2.f * b;
    const f32 det = h11 * h22 - c * c;
    if (!(h11 > 0.f) || !(h22 > 0.f) || !(det > 1e-12f)) return false;
    const f32 mx = -(h22 * d - c * e) / det;
    const f32 my = -(h11 * e - c * d) / det;
    if (!(std::fabs(mx) <= 0.5f) || !(std::fabs(my) <= 0.5f)) return false;
    mu_x = mx;
    mu_y = my;
    return true;
}

static uint32_t match_is_ambiguous(f32 best_cost, f32 second_cost,
                                   f32 ambiguity_ratio) {
    if (!std::isfinite(best_cost) || !std::isfinite(second_cost)) return 0u;
    if (best_cost < 0.f) best_cost = 0.f;
    if (second_cost < 0.f) second_cost = 0.f;
    const f32 denom = std::max(best_cost, 1e-12f);
    return (second_cost / denom) < ambiguity_ratio ? 1u : 0u;
}

}  // namespace
// Closed here on purpose: compute_motion_irregular is declared in stages.h and
// called from pipeline_paths, so it has to have external linkage. Leaving it in
// the anonymous namespace above defined hhsr::{anonymous}::compute_motion_irregular,
// a distinct overload from the declared hhsr:: one, and every call in this file
// then saw both and was ambiguous. The anonymous namespace reopens directly
// after it.

// Wronski's motion prior, measured on the grid alignment produced.
//
// Identical arithmetic to compute_s in robustness.cpp; the point is purely
// where it runs. Robustness sees the flow only after flow_to_raw_tile_grid has
// duplicated each grey tile across a 2x2 block of raw tiles and scaled the
// displacements by 2, and the 3x3 span measured there is not the same quantity.
// sx/sy convert the stored displacements into the units r_Mt is defined in.
//
// The reference keeps its flow array in RAW pixels throughout -- robustness.py
// divides by 2 when the guide is half resolution ("the flow must be divided by
// 2"), and compute_s consumes that same array unscaled. So M_th is a raw-pixel
// threshold. Alignment on the decimate grey stores half-resolution
// displacements, and measuring their span directly would compare grey pixels
// against a raw-pixel threshold -- an effective 2x loosening on that path and
// no change on FFT, which is exactly the kind of silent unit mismatch that
// makes r_Mt look inert.
std::vector<uint32_t> compute_motion_irregular(const FlowField& flow, f32 Mt,
                                               f32 sx, f32 sy, int num_threads) {
    const size_t n = (size_t)std::max(0, flow.ny) * (size_t)std::max(0, flow.nx);
    std::vector<uint32_t> out;
    if (n == 0 || flow.flow.empty()) return out;
    if (!std::isfinite(Mt)) Mt = 0.f;
    if (!std::isfinite(sx) || sx <= 0.f) sx = 1.f;
    if (!std::isfinite(sy) || sy <= 0.f) sy = 1.f;
    out.assign(n, 0u);
    // Distribution of M itself, not just how many cleared the threshold. One
    // run then answers whether r_Mt is in the right range at all: if the
    // percentiles sit far below it nothing will ever fire whatever it is set
    // to, and if they sit far above it everything fires. Comparing the flagged
    // percentage at two thresholds cannot distinguish those from a threshold
    // that is simply mistuned.
    const bool want_stats = prof_enabled();
    std::vector<f32> m_sq;
    if (want_stats) m_sq.assign(n, 0.f);
    const f32 inf = std::numeric_limits<f32>::infinity();
    parallel_rows(flow.ny, num_threads, [&](int ty) {
        f32 vx[9], vy[9];
        for (int tx = 0; tx < flow.nx; ++tx) {
            int cnt = 0;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    const int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= flow.ny || xx < 0 || xx >= flow.nx) continue;
                    vx[cnt] = flow.dx(yy, xx) * sx;
                    vy[cnt] = flow.dy(yy, xx) * sy;
                    ++cnt;
                }
            }

            f32 mnx = inf, mny = inf, mxx = -inf, mxy = -inf;
            for (int k = 0; k < cnt; ++k) {
                mnx = std::min(mnx, vx[k]);  mxx = std::max(mxx, vx[k]);
                mny = std::min(mny, vy[k]);  mxy = std::max(mxy, vy[k]);
            }
            const f32 d0 = (cnt > 0) ? (mxx - mnx) : 0.f;
            const f32 d1 = (cnt > 0) ? (mxy - mny) : 0.f;
            const f32 m2 = d0 * d0 + d1 * d1;
            const size_t idx = (size_t)ty * flow.nx + tx;
            out[idx] = (m2 > Mt * Mt) ? 1u : 0u;
            if (want_stats) m_sq[idx] = m2;
        }
    });
    if (want_stats && n > 0) {
        size_t flagged = 0;
        for (uint32_t v : out) flagged += (v != 0u) ? 1u : 0u;
        auto pct_at = [&](double q) {
            const size_t k = std::min(n - 1, (size_t)(q * (double)(n - 1)));
            std::nth_element(m_sq.begin(), m_sq.begin() + k, m_sq.end());
            return std::sqrt((double)m_sq[k]);   // stored squared
        };
        // Order matters: nth_element partially sorts, so ascending quantiles
        // stay valid on the progressively partitioned range.
        const double p50 = pct_at(0.50);
        const double p90 = pct_at(0.90);
        const double pmax = pct_at(1.00);
        prof_add_cpu("motionM#r_Mt-used", (double)Mt);
        prof_add_cpu("motionM#scale-x", (double)sx);
        prof_add_cpu("motionM#pct-irregular", 100.0 * (double)flagged / (double)n);
        prof_add_cpu("motionM#p50", p50);
        prof_add_cpu("motionM#p90", p90);
        prof_add_cpu("motionM#max", pmax);
    }
    return out;
}

namespace {  // reopened -- everything below is file-local again

// Measured with sx = sy = 1, which is already raw units on the FFT grey.
// flow_to_raw_tile_grid recomputes with the true scale when they differ.
static void mark_motion_irregular_tiles(FlowField& flow, const Config& cfg) {
    flow.motion_irregular =
        compute_motion_irregular(flow, cfg.r_Mt, 1.f, 1.f, cfg.num_threads);
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
                                     FlowField& flow, f32 ambiguity_ratio,
                                     bool write_ambiguity, int num_threads) {
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
    // Size it if this is the first level, but do NOT clear it otherwise: the
    // value already there was propagated down from the coarser level by
    // upscale_flow_460, and a wrong match up there is what produces the large
    // errors. The per-tile write below ORs into it.
    if (write_ambiguity) {
        const size_t n_amb = (size_t)std::max(0, ny) * (size_t)std::max(0, nx);
        if (flow.match_ambiguous.size() != n_amb)
            flow.match_ambiguous.assign(n_amb, 0u);
    }

    parallel_rows(ny, num_threads, [&](int ty) {
        RowBuffers& b = buffers[(size_t)ty];
        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * ts;
            int ox = tx * ts;

            // torch: flow.round().long() — round half to even
            int flow_dx = torch_round_to_int(flow.dx(ty, tx));
            int flow_dy = torch_round_to_int(flow.dy(ty, tx));

            std::fill(b.ref_tile_padded.begin(), b.ref_tile_padded.end(), 0.f);
            f32 ref_sq = 0.f;
            for (int i = 0; i < ts; ++i) {
                for (int j = 0; j < ts; ++j) {
                    int ry = oy + i, rx = ox + j;
                    if (ry < ref.h && rx < ref.w) {
                        f32 rv = ref.at(ry, rx);
                        b.ref_tile_padded[(size_t)(i + R) * N + (j + R)] = rv;
                        ref_sq += rv * rv;
                    }
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

            f32 best_rank = 1e30f;
            f32 second_rank = 1e30f;
            f32 best_cost = std::numeric_limits<f32>::infinity();
            f32 second_cost = std::numeric_limits<f32>::infinity();
            int best_dy = 0, best_dx = 0;
            for (int i = 0; i < corr_size; ++i) {
                for (int j = 0; j < corr_size; ++j) {
                    f32 err = b.L2_search[(size_t)i * corr_size + j]
                            - 2.f * b.corrs[(size_t)i * corr_size + j];
                    f32 cost = std::max(0.f, ref_sq + err);
                    if (err < best_rank) {
                        second_rank = best_rank;
                        second_cost = best_cost;
                        best_rank = err;
                        best_cost = cost;
                        best_dy = i - corr_size / 2;
                        best_dx = j - corr_size / 2;
                    } else if (err < second_rank) {
                        second_rank = err;
                        second_cost = cost;
                    }
                }
            }
            flow.dx(ty, tx) += (f32)best_dx;
            flow.dy(ty, tx) += (f32)best_dy;
            if (write_ambiguity)
                flow.ambiguous(ty, tx) |=
                    match_is_ambiguous(best_cost, second_cost, ambiguity_ratio);
        }
    });
}

static void block_match_level_direct_460(const Image& ref, const Image& moving,
                                         int tile_size, int search_radius,
                                         FlowField& flow, bool l1,
                                         f32 ambiguity_ratio, bool write_ambiguity,
                                         bool fallback_on_ambiguous,
                                         int num_threads, bool subpixel) {
    const int ny = flow.ny, nx = flow.nx;
    const int ts = tile_size;
    const int R = search_radius;
    // 13x13 covers every shipped radius (<= 4 raw -> <= 4 grey); a larger
    // radius just skips the sub-pixel fit rather than overrunning the buffer.
    const int span = 2 * R + 1;
    const bool want_fit = subpixel && span >= 3 && span <= 13;
    // Size it if this is the first level, but do NOT clear it otherwise: the
    // value already there was propagated down from the coarser level by
    // upscale_flow_460, and a wrong match up there is what produces the large
    // errors. The per-tile write below ORs into it.
    if (write_ambiguity) {
        const size_t n_amb = (size_t)std::max(0, ny) * (size_t)std::max(0, nx);
        if (flow.match_ambiguous.size() != n_amb)
            flow.match_ambiguous.assign(n_amb, 0u);
    }
    parallel_rows(ny, num_threads, [&](int ty) {
        for (int tx = 0; tx < nx; ++tx) {
            const int oy = ty * ts;
            const int ox = tx * ts;
            const f32 local_fx = flow.dx(ty, tx);
            const f32 local_fy = flow.dy(ty, tx);
            // Round the search centre, do not truncate. The reference rounds
            // (extract_flow_patches does flow.round(); the L1 kernels do
            // round(alignment)), and truncation displaces the centre by up to a
            // full pixel where rounding caps it at half. Level 0 searches with
            // radius 1, so a whole-pixel offset can put the true optimum outside
            // the window entirely, leaving ICA to recover a full pixel from the
            // edge of its basin.
            //
            // Only fractional input flow is affected, so this is inert on the
            // FFT path -- every level there receives integer flow -- and live on
            // the decimate path, where per-level ICA hands each level a
            // fractional estimate. Hoisted out of the inner loops; it was being
            // recomputed per sampled pixel.
            const int base_fx = cuda_round_to_int(local_fx);
            const int base_fy = cuda_round_to_int(local_fy);
            f32 min_dist = std::numeric_limits<f32>::infinity();
            f32 second_dist = std::numeric_limits<f32>::infinity();
            int min_shift_x = 0, min_shift_y = 0;
            f32 cost_surf[13 * 13];
            for (int sdy = -R; sdy <= R; ++sdy) {
                for (int sdx = -R; sdx <= R; ++sdx) {
                    f32 dist = 0.f;
                    bool valid = true;
                    for (int i = 0; i < ts && valid; ++i) {
                        for (int j = 0; j < ts; ++j) {
                            const int rx = ox + j;
                            const int ry = oy + i;
                            const int mx = rx + base_fx + sdx;
                            const int my = ry + base_fy + sdy;
                            if (!(rx >= 0 && rx < ref.w && ry >= 0 && ry < ref.h &&
                                  mx >= 0 && mx < moving.w && my >= 0 && my < moving.h)) {
                                valid = false;
                                break;
                            }
                            const f32 diff = ref.at(ry, rx) - moving.at(my, mx);
                            dist += l1 ? std::fabs(diff) : diff * diff;
                        }
                    }
                    if (!valid) dist = std::numeric_limits<f32>::infinity();
                    if (want_fit)
                        cost_surf[(sdy + R) * span + (sdx + R)] = dist;
                    if (dist < min_dist) {
                        second_dist = min_dist;
                        min_dist = dist;
                        min_shift_y = sdy;
                        min_shift_x = sdx;
                    } else if (dist < second_dist) {
                        second_dist = dist;
                    }
                }
            }
            const uint32_t ambiguous =
                (write_ambiguity || fallback_on_ambiguous)
                    ? match_is_ambiguous(min_dist, second_dist, ambiguity_ratio)
                    : 0u;
            if (fallback_on_ambiguous && ambiguous != 0u) {
                // ImageStackAlignator's rule: no precise shift determinable
                // (near-tied cost surface: flat patch, aperture, repetition)
                // -> apply NO shift; the tile keeps its seed, i.e. the
                // upsampled previous-level flow or the global initial
                // estimate at the coarsest level.
                flow.dx(ty, tx) = local_fx;
                flow.dy(ty, tx) = local_fy;
            } else {
                f32 sub_x = 0.f, sub_y = 0.f;
                if (want_fit &&
                    min_shift_x > -R && min_shift_x < R &&
                    min_shift_y > -R && min_shift_y < R) {
                    f32 D[9];
                    for (int yy = -1; yy <= 1; ++yy)
                        for (int xx = -1; xx <= 1; ++xx)
                            D[(yy + 1) * 3 + (xx + 1)] =
                                cost_surf[(min_shift_y + yy + R) * span +
                                          (min_shift_x + xx + R)];
                    (void)quadratic_subpixel_3x3(D, sub_x, sub_y);
                }
                flow.dx(ty, tx) = local_fx + (f32)min_shift_x + sub_x;
                flow.dy(ty, tx) = local_fy + (f32)min_shift_y + sub_y;
            }
            if (write_ambiguity)
                flow.ambiguous(ty, tx) |= ambiguous;
        }
    });
}

static void block_match_level_L2(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, const Config& cfg,
                                  int num_threads) {
    const bool write_ambiguity = cfg.flow_reject_ambiguous_enabled;
    const bool fallback = cfg.align_ambiguous_fallback_enabled;
    const f32 ambiguity_ratio = clamped_ambiguity_ratio(cfg);
#ifdef __APPLE__
    // 460-main direct local L2 search. HHSR_L2_CPU=1 forces CPU direct path.
    if (!env_flag_on("HHSR_L2_CPU") && !env_flag_on("HHSR_ALIGN_CPU")) {
        if (block_match_level_L2_metal(ref, moving, tile_size, search_radius, flow,
                                       ambiguity_ratio, write_ambiguity, fallback,
                                       cfg.bm_subpixel_quadratic))
            return;
    }
#endif
    block_match_level_direct_460(ref, moving, tile_size, search_radius, flow,
                                 false, ambiguity_ratio, write_ambiguity, fallback,
                                 num_threads, cfg.bm_subpixel_quadratic);
}

// ============================================================================
// L1 BM — CUDA kernels + broken argmin. ts=64 uses 1024 threads × 4 px.
// ============================================================================
static void block_match_level_L1(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, const Config& cfg,
                                  int num_threads) {
    const bool write_ambiguity = cfg.flow_reject_ambiguous_enabled;
    const bool fallback = cfg.align_ambiguous_fallback_enabled;
    const f32 ambiguity_ratio = clamped_ambiguity_ratio(cfg);
#ifdef __APPLE__
    if (!env_flag_on("HHSR_L1_CPU") && !env_flag_on("HHSR_ALIGN_CPU") &&
        block_match_level_L1_metal(ref, moving, tile_size, search_radius, flow,
                                   ambiguity_ratio, write_ambiguity, fallback,
                                   cfg.bm_subpixel_quadratic))
        return;
#endif
    block_match_level_direct_460(ref, moving, tile_size, search_radius, flow,
                                 true, ambiguity_ratio, write_ambiguity, fallback,
                                 num_threads, cfg.bm_subpixel_quadratic);
    return;
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
                              int num_threads,
                              f32 damp_ratio, f32 max_step) {
#ifdef __APPLE__
    // Metal ica_kernel_8/16 — same math/order as CPU path below / Python ICA.py.
    if (ica_refine_level_metal(ref, gradx, grady, hessian.data, moving, flow,
                               tile_size, n_iter, damp_ratio, max_step))
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
            // Levenberg-Marquardt damping toward the aperture ratio. The
            // Hessian is fixed across the iterations below, so this is done
            // once per tile. See Config::ica_regularize_enabled.
            if (damp_ratio > 0.f) {
                const f32 tr = h00 + h11;
                const f32 d0 = h00 * h11 - h01 * h10;
                const f32 disc = std::sqrt(std::max(0.f, tr * tr * 0.25f - d0));
                const f32 l1 = tr * 0.5f + disc;
                const f32 l2 = tr * 0.5f - disc;
                const f32 lam = damp_ratio * l1 - l2;   // > 0 only when l2/l1 < ratio
                if (lam > 0.f) { h00 += lam; h11 += lam; }
            }
            f32 det = h00 * h11 - h01 * h10;
            if (std::fabs(det) < 1e-10f) continue;
            f32 det_inv = 1.f / det;

            f32 fx = flow.dx(ty, tx);
            f32 fy = flow.dy(ty, tx);

            for (int it = 0; it < n_iter; ++it) {
                // floor, not trunc. ICA.py uses math.modf + int(), which
                // truncates toward zero, so a negative displacement such as
                // -0.25 yields offset 0 and fraction -0.25. bilinear_ica then
                // evaluates m00 + (m01 - m00) * (-0.25): the right position, but
                // reached by extrapolating from the pair to the RIGHT of it
                // instead of interpolating from the pair that straddles it.
                //
                // Positive displacements interpolate, negative ones extrapolate,
                // so the converged sub-pixel flow differs by direction. Measured
                // on a synthetic shift at ts=16: 0.075px of spread between +0.75
                // and -0.75, against 0.0025px with floor. It does not reduce the
                // peak error -- both variants carry the same ~0.11px bilinear
                // shrinkage toward zero -- but it makes that error symmetric, so
                // frames drifting left and right land consistently rather than
                // scattering by direction.
                const f32 base_x = std::floor(fx);
                const f32 base_y = std::floor(fy);
                f32 frac_x = fx - base_x;
                int floor_off_x = (int)base_x;
                f32 frac_y = fy - base_y;
                int floor_off_y = (int)base_y;

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
                    f32 dfx = det_inv * (h11 * B0 - h01 * B1);
                    f32 dfy = det_inv * (-h10 * B0 + h00 * B1);
                    if (max_step > 0.f) {
                        const f32 st = std::sqrt(dfx * dfx + dfy * dfy);
                        if (st > max_step) {
                            const f32 k = max_step / st;
                            dfx *= k;
                            dfy *= k;
                        }
                    }
                    fx += dfx;
                    fy += dfy;
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
                    f32 dfx = det_inv * (h11 * B0 - h01 * B1);
                    f32 dfy = det_inv * (-h10 * B0 + h00 * B1);
                    if (max_step > 0.f) {
                        const f32 st = std::sqrt(dfx * dfx + dfy * dfy);
                        if (st > max_step) {
                            const f32 k = max_step / st;
                            dfx *= k;
                            dfy *= k;
                        }
                    }
                    fx += dfx;
                    fy += dfy;
                }
            }

            flow.dx(ty, tx) = fx;
            flow.dy(ty, tx) = fy;
        }
    });
}

// upscale_lvl — 460-main candidate upscaling.
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
            if (in.match_ambiguous.size() == (size_t)in.ny * (size_t)in.nx)
                upsampled.ambiguous(ty, tx) = in.ambiguous(sy, sx);
        }
    }

    FlowField out(target_ny, target_nx);
    for (int ty = 0; ty < target_ny; ++ty) {
        for (int tx = 0; tx < target_nx; ++tx) {
            if (ty < up_ny && tx < up_nx) {
                out.dx(ty, tx) = upsampled.dx(ty, tx);
                out.dy(ty, tx) = upsampled.dy(ty, tx);
                if (upsampled.match_ambiguous.size() == (size_t)up_ny * (size_t)up_nx)
                    out.ambiguous(ty, tx) = upsampled.ambiguous(ty, tx);
            }
        }
    }
    return out;
}

static FlowField upscale_flow_460(const Image& ref, const Image& moving,
                                  const FlowField& in, int target_ny, int target_nx,
                                  int upsample_factor, int new_tile_size,
                                  int prev_tile_size,
                                  bool fallback_on_ambiguous = false,
                                  f32 ambiguity_ratio = 1.10f) {
    int tile_ratio = new_tile_size / std::max(1, prev_tile_size);
    int repeat_factor = upsample_factor / std::max(1, tile_ratio);
    if (repeat_factor < 1) repeat_factor = 1;

    FlowField out(target_ny, target_nx);
    for (int ty = 0; ty < target_ny; ++ty) {
        for (int tx = 0; tx < target_nx; ++tx) {
            // The target grid is floor(level_dims / tile_size), but the level
            // dims come out of downsample_by, which shrinks the image by a valid
            // convolution (4*factor pixels) before subsampling. target_n* is
            // therefore always a little larger than repeat_factor * in.n*, at
            // every level and every shipped configuration -- circular padding
            // only guarantees divisibility at level 0.
            //
            // Those uncovered tiles form a strip along the bottom and right
            // edges. They used to be reset to zero motion, which the finest
            // level could not recover: its search radius is 1, so a burst with
            // any real handheld motion left that strip unaligned. Clamping to
            // the nearest covered coarse tile hands them the neighbouring
            // estimate instead, and the three-candidate test below still
            // re-verifies it against the image. upscale_flow(), the sibling
            // above, has always clamped.
            const int prev_x = std::min(tx / repeat_factor, in.nx - 1);
            const int prev_y = std::min(ty / repeat_factor, in.ny - 1);
            const int ups_x = tx % repeat_factor;
            const int ups_y = ty % repeat_factor;
            const int x_shift = (2 * ups_x + 1 > repeat_factor) ? 1 : -1;
            const int y_shift = (2 * ups_y + 1 > repeat_factor) ? 1 : -1;
            const int cand_y = std::max(0, std::min(in.ny - 1, prev_y + y_shift));
            const int cand_x = std::max(0, std::min(in.nx - 1, prev_x + x_shift));
            f32 cand[3][2] = {
                {in.dx(prev_y, prev_x) * (f32)upsample_factor,
                 in.dy(prev_y, prev_x) * (f32)upsample_factor},
                {in.dx(cand_y, prev_x) * (f32)upsample_factor,
                 in.dy(cand_y, prev_x) * (f32)upsample_factor},
                {in.dx(prev_y, cand_x) * (f32)upsample_factor,
                 in.dy(prev_y, cand_x) * (f32)upsample_factor},
            };

            const int ox = tx * new_tile_size;
            const int oy = ty * new_tile_size;
            f32 best_dist = std::numeric_limits<f32>::infinity();
            f32 second_dist = std::numeric_limits<f32>::infinity();
            int best_i = 0;
            for (int ci = 0; ci < 3; ++ci) {
                f32 dist = 0.f;
                bool valid = true;
                for (int i = 0; i < new_tile_size && valid; ++i) {
                    for (int j = 0; j < new_tile_size; ++j) {
                        const int rx = ox + j;
                        const int ry = oy + i;
                        const int mx = rx + (int)cand[ci][0];
                        const int my = ry + (int)cand[ci][1];
                        if (!(rx >= 0 && rx < ref.w && ry >= 0 && ry < ref.h &&
                              mx >= 0 && mx < moving.w && my >= 0 && my < moving.h)) {
                            valid = false;
                            break;
                        }
                        dist += std::fabs(ref.at(ry, rx) - moving.at(my, mx));
                    }
                }
                if (!valid) dist = std::numeric_limits<f32>::infinity();
                if (dist < best_dist) {
                    second_dist = best_dist;
                    best_dist = dist;
                    best_i = ci;
                } else if (dist < second_dist) {
                    second_dist = dist;
                }
            }
            // ImageStackAlignator's rule extended to the candidate test: when
            // a NEIGHBOUR tile's flow wins over the parent's by less than the
            // ambiguity ratio, the cost surface cannot distinguish them --
            // keep the parent (the previous level's own estimate) rather
            // than inheriting a neighbour's unrelated motion. The measured
            // catastrophic case this targets: neighbour (82,-154) beating
            // parent (88,112) at 0.4546 vs 0.4968 -- ratio 1.093.
            if (fallback_on_ambiguous && best_i != 0 &&
                match_is_ambiguous(best_dist, second_dist, ambiguity_ratio) != 0u)
                best_i = 0;
            out.dx(ty, tx) = cand[best_i][0];
            out.dy(ty, tx) = cand[best_i][1];
            if (in.match_ambiguous.size() == (size_t)in.ny * (size_t)in.nx) {
                int am_y = prev_y;
                int am_x = prev_x;
                if (best_i == 1) {
                    am_y = cand_y;
                    am_x = prev_x;
                } else if (best_i == 2) {
                    am_y = prev_y;
                    am_x = cand_x;
                }
                out.ambiguous(ty, tx) = in.ambiguous(am_y, am_x);
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

// Reference Sobel gradients for the dense Lucas-Kanade stage, held across the
// burst: they depend only on the reference, so recomputing them per comparison
// frame would be pure waste -- and on the Metal path it would also re-upload
// two full-resolution buffers every frame. Keyed on the grey's data pointer
// AND size, and dropped at burst start, so allocator address reuse cannot
// alias one burst's gradients onto the next.
static std::mutex g_dlk_grad_mu;
static const f32* g_dlk_grad_key = nullptr;
static size_t g_dlk_grad_n = 0;
static Image g_dlk_gx, g_dlk_gy;

void clear_align_ref_ica_cache() {
    g_ref_ica_cache = {};
    {
        std::lock_guard<std::mutex> lk(g_dlk_grad_mu);
        g_dlk_grad_key = nullptr;
        g_dlk_grad_n = 0;
        g_dlk_gx = Image();
        g_dlk_gy = Image();
    }
#ifdef __APPLE__
    metal_clear_ref_ica_cache();
    metal_clear_densify_ref_cache();
    metal_clear_dense_lk_cache();
#endif
}

// align() — Python alignment.align
// ref_grey must already be circular-padded (init_alignment); moving is NOT.
// ============================================================================
// ---------------------------------------------------------------------------
// Layer 0: global rigid pre-alignment (ImageStackAlignator PreAlignment.cs)
// ---------------------------------------------------------------------------
//
// ScanAngles: for every candidate angle, rotate the moving frame about its
// centre, cross-correlate against the reference in the Fourier domain
// (conjugate multiply then inverse transform -- kernel.cu
// conjugateComplexMulKernel does a plain correlation, NOT a whitened phase
// correlation), and take the peak. The peak VALUE selects the angle; the peak
// LOCATION gives the translation. Coarse sweep at 5x the increment, then a
// fine sweep around the winner, exactly as PreAlignment.ScanAngles does.
//
// What the peak means, in this codebase's flow convention (a moving-frame
// position is a reference position plus u):
//     mov(q)          = ref(R_-th (q - c - t) + c)
//     W_ph(mov)(p)    = mov(R_ph (p - c) + c)        [warp by candidate ph]
//     W_th(mov)(p)    = ref(p - R_-th t)
// so correlating ref against W_th(mov) peaks at s = R_-th t, hence
// t = R_th s. Sign conventions are pinned by prealign_selftest(), which
// synthesises a known (theta, t) and checks both come back.
//
// Working resolution is PA_N square. The source is box-averaged down to it
// (area resampling, so a 16x decimation of a 12MP grey does not alias) and
// padded to preserve aspect, which keeps the scale uniform -- a rotation by
// theta about the grey centre is then a rotation by the same theta about the
// PA centre.
namespace {

constexpr int PA_N = 256;

inline f32 pa_bilinear(const f32* img, int h, int w, f32 y, f32 x) {
    if (!(y >= 0.f) || !(x >= 0.f) ||
        y > (f32)(h - 1) || x > (f32)(w - 1)) return 0.f;
    const int y0 = (int)y, x0 = (int)x;
    const int y1 = (y0 + 1 < h) ? y0 + 1 : y0;
    const int x1 = (x0 + 1 < w) ? x0 + 1 : x0;
    const f32 ay = y - (f32)y0, ax = x - (f32)x0;
    const f32 a = img[(size_t)y0 * w + x0], b = img[(size_t)y0 * w + x1];
    const f32 c = img[(size_t)y1 * w + x0], d = img[(size_t)y1 * w + x1];
    return (a * (1.f - ax) + b * ax) * (1.f - ay) +
           (c * (1.f - ax) + d * ax) * ay;
}

// Area-average `src` into a centred oh x ow region of a PA_N x PA_N buffer and
// subtract the region mean, so the zeros outside the region -- and the zeros a
// rotation later brings in -- all sit at the image's own DC level.
//
// The mean has to go before anything else: with zero padding the overlap area
// shrinks as |d| grows, so a non-zero mean makes the correlation fall off with
// displacement and biases the peak toward d = 0 -- which would report every
// frame as unrotated and untranslated.
void pa_decimate(const Image& src, std::vector<f32>& out,
                 int& oh, int& ow, int& oy0, int& ox0) {
    out.assign((size_t)PA_N * PA_N, 0.f);
    const int h = src.h, w = src.w;
    oh = ow = 0; oy0 = ox0 = 0;
    if (h <= 0 || w <= 0) return;
    const f32 s = (f32)PA_N / (f32)std::max(h, w);
    oh = std::max(1, std::min(PA_N, (int)std::lround((double)h * s)));
    ow = std::max(1, std::min(PA_N, (int)std::lround((double)w * s)));
    oy0 = (PA_N - oh) / 2; ox0 = (PA_N - ow) / 2;

    double sum = 0.0;
    for (int y = 0; y < oh; ++y) {
        const int sy0 = (int)((int64_t)y * h / oh);
        const int sy1 = std::max(sy0 + 1, (int)((int64_t)(y + 1) * h / oh));
        for (int x = 0; x < ow; ++x) {
            const int sx0 = (int)((int64_t)x * w / ow);
            const int sx1 = std::max(sx0 + 1, (int)((int64_t)(x + 1) * w / ow));
            double acc = 0.0;
            for (int sy = sy0; sy < sy1; ++sy)
                for (int sx = sx0; sx < sx1; ++sx)
                    acc += (double)src.data[(size_t)sy * w + sx];
            const f32 v = (f32)(acc / (double)((sy1 - sy0) * (sx1 - sx0)));
            out[(size_t)(oy0 + y) * PA_N + (ox0 + x)] = v;
            sum += (double)v;
        }
    }
    const f32 mean = (f32)(sum / (double)(oh * ow));
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
            out[(size_t)(oy0 + y) * PA_N + (ox0 + x)] -= mean;
}

// Separable Hann over the region, zeroing everything outside it. Returns the
// L2 norm of the result.
//
// Applied AFTER the rotation, never before: rotating an already-windowed
// buffer carries the taper around with the content, so the correlation energy
// would depend on the angle and the raw peak would always prefer 0 degrees.
// That is precisely what the self-test caught -- +0.6 and +0.3 degrees both
// came back as 0.00 while -0.8 happened to survive. Dividing the peak by the
// two norms (a normalised cross-correlation on the peak alone) closes the
// same hole from the other side; both are kept because either one failing
// silently would restore the bias.
f32 pa_window(std::vector<f32>& b, int oh, int ow, int oy0, int ox0) {
    double n2 = 0.0;
    for (int y = 0; y < PA_N; ++y) {
        const bool in_y = (y >= oy0 && y < oy0 + oh);
        const f32 wy = in_y ? (0.5f - 0.5f * std::cos(6.28318530718f *
                          ((f32)(y - oy0) + 0.5f) / (f32)oh)) : 0.f;
        for (int x = 0; x < PA_N; ++x) {
            const bool in_x = (x >= ox0 && x < ox0 + ow);
            f32& v = b[(size_t)y * PA_N + x];
            if (!in_y || !in_x) { v = 0.f; continue; }
            const f32 wx = 0.5f - 0.5f * std::cos(6.28318530718f *
                           ((f32)(x - ox0) + 0.5f) / (f32)ow);
            v *= wy * wx;
            n2 += (double)v * (double)v;
        }
    }
    return (f32)std::sqrt(n2);
}

// out(p) = in(R_phi (p - cN) + cN), bilinear, zero outside.
void pa_rotate(const std::vector<f32>& in, std::vector<f32>& out, f32 phi) {
    out.assign((size_t)PA_N * PA_N, 0.f);
    const f32 cs = std::cos(phi), sn = std::sin(phi);
    const f32 cN = 0.5f * (f32)PA_N;
    for (int y = 0; y < PA_N; ++y) {
        const f32 py = (f32)y - cN;
        for (int x = 0; x < PA_N; ++x) {
            const f32 px = (f32)x - cN;
            const f32 sx = cs * px - sn * py + cN;
            const f32 sy = sn * px + cs * py + cN;
            out[(size_t)y * PA_N + x] = pa_bilinear(in.data(), PA_N, PA_N, sy, sx);
        }
    }
}

// Parabolic vertex through three samples, clamped to the sample spacing.
inline f32 pa_subpixel(f32 vm, f32 v0, f32 vp) {
    const f32 den = vm - 2.f * v0 + vp;
    if (!(std::fabs(den) > 1e-20f)) return 0.f;
    const f32 d = 0.5f * (vm - vp) / den;
    return (d > 1.f) ? 1.f : ((d < -1.f) ? -1.f : d);
}

} // namespace

RigidModel estimate_global_rigid(const Image& ref_grey, const Image& moving_grey,
                                 const Config& cfg) {
    RigidModel m;
    m.cx = 0.5f * (f32)ref_grey.w;
    m.cy = 0.5f * (f32)ref_grey.h;
    if (ref_grey.h <= 0 || ref_grey.w <= 0 ||
        moving_grey.h != ref_grey.h || moving_grey.w != ref_grey.w)
        return m;

    std::vector<f32> ref_pa, mov_pa;
    int oh = 0, ow = 0, oy0 = 0, ox0 = 0;
    int moh = 0, mow = 0, moy0 = 0, mox0 = 0;
    pa_decimate(ref_grey, ref_pa, oh, ow, oy0, ox0);
    pa_decimate(moving_grey, mov_pa, moh, mow, moy0, mox0);
    if (oh <= 0 || ow <= 0) return m;

    const f32 norm_ref = pa_window(ref_pa, oh, ow, oy0, ox0);
    if (!(norm_ref > 0.f)) return m;

    std::vector<std::complex<f32>> F_ref, F_mov;
    rfft2(ref_pa.data(), PA_N, PA_N, F_ref);

    std::vector<f32> rot, corr;

    // One correlation at angle phi: returns the normalised peak value, and
    // writes the sub-pixel peak displacement s into (sy, sx).
    auto score = [&](f32 phi, f32& sy, f32& sx) -> f32 {
        pa_rotate(mov_pa, rot, phi);
        const f32 norm_mov = pa_window(rot, moh, mow, moy0, mox0);
        if (!(norm_mov > 0.f)) { sy = sx = 0.f; return -std::numeric_limits<f32>::max(); }
        rfft2(rot.data(), PA_N, PA_N, F_mov);
        for (size_t i = 0; i < F_mov.size() && i < F_ref.size(); ++i)
            F_mov[i] = std::conj(F_ref[i]) * F_mov[i];
        irfft2(F_mov, PA_N, PA_N, corr);
        int by = 0, bx = 0;
        f32 best = -std::numeric_limits<f32>::max();
        for (int y = 0; y < PA_N; ++y)
            for (int x = 0; x < PA_N; ++x) {
                const f32 v = corr[(size_t)y * PA_N + x];
                if (v > best) { best = v; by = y; bx = x; }
            }
        auto wrap = [](int i) { return (i > PA_N / 2) ? i - PA_N : i; };
        const int ym = (by + PA_N - 1) % PA_N, yp = (by + 1) % PA_N;
        const int xm = (bx + PA_N - 1) % PA_N, xp = (bx + 1) % PA_N;
        sy = (f32)wrap(by) + pa_subpixel(corr[(size_t)ym * PA_N + bx], best,
                                         corr[(size_t)yp * PA_N + bx]);
        sx = (f32)wrap(bx) + pa_subpixel(corr[(size_t)by * PA_N + xm], best,
                                         corr[(size_t)by * PA_N + xp]);
        return best / (norm_ref * norm_mov);
    };

    const f32 deg = 3.14159265358979f / 180.f;
    const f32 incr = std::max(0.01f, cfg.prealign_rot_incr) * deg;
    const f32 range = std::max(incr, cfg.prealign_rot_range * deg);

    f32 best_v = -std::numeric_limits<f32>::max(), best_a = 0.f;
    f32 best_sy = 0.f, best_sx = 0.f;
    for (f32 a = -range; a <= range + 1e-6f; a += 5.f * incr) {
        f32 sy, sx;
        const f32 v = score(a, sy, sx);
        if (v > best_v) { best_v = v; best_a = a; best_sy = sy; best_sx = sx; }
    }
    const f32 zero = best_a, fine_range = 10.f * incr;
    for (f32 a = zero - fine_range; a <= zero + fine_range + 1e-6f; a += incr) {
        f32 sy, sx;
        const f32 v = score(a, sy, sx);
        if (v > best_v) { best_v = v; best_a = a; best_sy = sy; best_sx = sx; }
    }

    // t = R_theta * s, then PA pixels back to grey pixels. Per-axis scale:
    // aspect is preserved to within the rounding of oh/ow, so the two factors
    // agree to well under a percent and the angle is unaffected.
    const f32 cs = std::cos(best_a), sn = std::sin(best_a);
    const f32 tx_pa = cs * best_sx - sn * best_sy;
    const f32 ty_pa = sn * best_sx + cs * best_sy;
    m.theta = best_a;
    m.tx = tx_pa * ((f32)ref_grey.w / (f32)ow);
    m.ty = ty_pa * ((f32)ref_grey.h / (f32)oh);
    m.peak = best_v;
    m.valid = true;
    return m;
}

// Cumulative downscale of pyramid level `lvl` relative to the alignment grey.
//
// build_pyramid applies bm_factors[0] to level 0 as well (a no-op at the
// default 1), then bm_factors[k] from level k-1 to level k -- so the scale is
// the running product from 0, not from 1. With {1,2,4,4} the levels sit at
// 1x, 2x, 8x, 32x.
f32 pyramid_level_scale(const Config& cfg, int lvl) {
    f32 s = 1.f;
    for (int k = 0; k <= lvl && k < (int)cfg.bm_factors.size(); ++k)
        s *= (f32)std::max(1, cfg.bm_factors[k]);
    return s;
}

// Layer 1: start the coarsest block-matching level from the global rigid
// model instead of from zero, so the search sees only the residual.
//
// This is ImageStackAlignator's convertToTilesOverlapBorder (kernel.cu
// 203-220), which evaluates the model at each tile's CENTRE and offsets that
// tile's fetch by it. Seeding the flow is the same operation expressed in
// this codebase's terms: block matching searches around the current vector,
// so a seeded tile compares the reference tile against an already
// model-compensated moving tile.
//
// Note what this does and does not fix. It removes the large, ambiguous part
// of the search (a rotation reaches ~44 raw px in the corners at 1 degree)
// and gives ICA a linearisation point that is actually near the answer. It
// does NOT remove the ramp WITHIN a tile: after subtracting the model at the
// tile centre the field still varies +/-0.14 px across a 16-px tile at 1
// degree, and block matching still answers that with one constant. ISA has
// the same property; what removes it there is the dense per-pixel
// Lucas-Kanade, not the pre-alignment.
void seed_flow_from_rigid(FlowField& flow, const RigidModel& model,
                          int level_tile_size, f32 level_scale) {
    if (!model.valid || flow.ny <= 0 || flow.nx <= 0 ||
        level_tile_size <= 0 || !(level_scale > 0.f)) return;
    for (int ty = 0; ty < flow.ny; ++ty) {
        const f32 gy = ((f32)ty + 0.5f) * (f32)level_tile_size * level_scale;
        for (int tx = 0; tx < flow.nx; ++tx) {
            const f32 gx = ((f32)tx + 0.5f) * (f32)level_tile_size * level_scale;
            f32 dx = 0.f, dy = 0.f;
            model.displacement(gy, gx, dx, dy);
            flow.dx(ty, tx) = dx / level_scale;
            flow.dy(ty, tx) = dy / level_scale;
        }
    }
}

FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg, int tile_size) {
    int nlev = (int)ref_pyr.levels.size();

    // Layer 0. One model per comparison frame, estimated on the alignment
    // greys before any tiling, and handed to whichever path runs.
    RigidModel rigid;
    if (cfg.prealign_enabled)
        rigid = estimate_global_rigid(ref_grey, moving_grey, cfg);

#ifdef __APPLE__
    // Default iOS path: Metal alignment. HHSR_ALIGN_CPU=1 forces the C++ path.
    if (!env_flag_on("HHSR_ALIGN_CPU")) {
        FlowField flow_gpu;
        if (align_metal(ref_pyr, ref_grey, moving_grey, cfg, tile_size, flow_gpu,
                        rigid)) {
            mark_motion_irregular_tiles(flow_gpu, cfg);
            return flow_gpu;
        }
    }
#endif

    // CPU path: cache ref Sobel+Hessian across comparison frames.
    // C++ pyramid is fine-first (levels[0]=finest). Python build_gaussian_pyramid
    // returns coarse-first (pyramid[::-1]); its list_id = n-1-l then maps
    // coarse→params[n-1], fine→params[0]. With fine-first storage that is simply
    // params[lvl] (arrays are fine→coarse in default.yaml).
    // Sobel gradients and the ICA Hessian depend only on the reference, so they
    // are built once per burst and reused by every comparison frame. Keyed on
    // the pyramid's address and level count; clear_align_ref_ica_cache() drops
    // it between bursts.
    const bool ica_all = cfg.ica_every_level();
    if (ica_all && (g_ref_ica_cache.key != (const void*)&ref_pyr ||
                    (int)g_ref_ica_cache.levels.size() != nlev)) {
        g_ref_ica_cache.key = (const void*)&ref_pyr;
        g_ref_ica_cache.levels.assign((size_t)nlev, RefIcaLevel{});
        for (int lvl = 0; lvl < nlev; ++lvl) {
            // Level 0 on the FFT grey is refined by the single finest pass
            // below, which builds its own gradients; caching a second
            // full-resolution set here is the allocation that fails.
            if (lvl == 0 && cfg.ica_per_level_coarse_only()) continue;
            const Image& r = ref_pyr.levels[lvl];
            int ts = cfg.grey_tile_size((lvl < (int)cfg.bm_tile_sizes.size())
                         ? cfg.bm_tile_sizes[lvl] : tile_size);
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

    // 460-main pads alternate images circularly before pyramid construction.
    // Pad with the tile size block matching ACTUALLY uses on this grey --
    // grey-domain, not raw. Padding with the raw tile size on the half-res
    // decimate grey appended wrapped rows the grid didn't need (1512 is
    // already divisible by 8), manufacturing a garbage bottom tile row whose
    // flow seeded and neighboured the real bottom row through the
    // coarse-to-fine upsample.
    Image moving_padded = pad_image_circular(moving_grey, cfg.grey_tile_size(tile_size));
    Pyramid mov_pyr = build_pyramid(moving_padded, cfg.bm_factors);

    FlowField flow;

    for (int lvl = nlev - 1; lvl >= 0; --lvl) {
        const Image& r = ref_pyr.levels[lvl];
        const Image& m = mov_pyr.levels[lvl];

        // Grey-domain tile size: bm_tile_sizes is in RAW pixels, the pyramid
        // is built on the alignment grey. See Config::grey_tile_size.
        int ts = cfg.grey_tile_size((lvl < (int)cfg.bm_tile_sizes.size())
                     ? cfg.bm_tile_sizes[lvl] : tile_size);
        // Grey-domain search radius -- bm_search_radii is in RAW pixels, same
        // as bm_tile_sizes. See Config::grey_search_radius.
        int radius = cfg.grey_search_radius(
            (lvl < (int)cfg.bm_search_radii.size()) ? cfg.bm_search_radii[lvl] : 2);

        // Tile grid from padded ref level (Python: h // tile_size)
        int ny = r.h / ts;
        int nx = r.w / ts;

        if (flow.nx == 0) {
            flow = FlowField(ny, nx);
            // Layer 1: seed the coarsest level with the global model.
            seed_flow_from_rigid(flow, rigid, ts, pyramid_level_scale(cfg, lvl));
        } else {
            int upsample_factor = ((lvl + 1) < (int)cfg.bm_factors.size())
                                  ? cfg.bm_factors[lvl + 1] : 1;
            int prev_ts = ((lvl + 1) < (int)cfg.bm_tile_sizes.size())
                          ? cfg.grey_tile_size(cfg.bm_tile_sizes[lvl + 1])
                          : ts;
            flow = upscale_flow_460(r, m, flow, ny, nx, upsample_factor, ts, prev_ts,
                                    cfg.align_ambiguous_fallback_enabled,
                                    clamped_ambiguity_ratio(cfg));
        }

        std::string metric = "L2";
        if (lvl < (int)cfg.bm_metrics.size())
            metric = cfg.bm_metrics[lvl];

        if (metric == "L1")
            block_match_level_L1(r, m, ts, radius, flow, cfg, cfg.num_threads);
        else
            block_match_level_L2(r, m, ts, radius, flow, cfg, cfg.num_threads);

        // alignment.py align_lvl: block matching, then ICA, at this level --
        // before the flow is upscaled and its error multiplied.
        if (ica_all && lvl < (int)g_ref_ica_cache.levels.size() &&
            !(lvl == 0 && cfg.ica_per_level_coarse_only())) {
            const RefIcaLevel& L = g_ref_ica_cache.levels[(size_t)lvl];
            if (L.hess.ny == flow.ny && L.hess.nx == flow.nx)
                ica_refine_level(r, L.gx, L.gy, m, L.hess, flow, ts,
                                 cfg.ica_n_iter, cfg.num_threads,
                                 ica_damp_ratio(cfg),
                                 ica_max_step(cfg, radius));
        }
    }

    // Full-res FFT grey keeps the single finest-level refinement, unchanged.
    // With per-level ICA the loop above already refined level 0, and repeating
    // it here would run ICA twice on the finest scale.
    // Coarse-only leaves the finest level to this pass, so it must still run.
    if (!ica_all || cfg.ica_per_level_coarse_only()) {
        Image gx = compute_sobel_gradx(ref_grey);
        Image gy = compute_sobel_grady(ref_grey);
        HessianField finest_hess = compute_hessian(gx, gy, cfg.grey_tile_size(tile_size));
        debug_dump_bin("cpp_gradx_ica", gx.data.data(), gx.data.size());
        debug_dump_bin("cpp_grady_ica", gy.data.data(), gy.data.size());
        const int finest_radius = cfg.grey_search_radius(
            cfg.bm_search_radii.empty() ? 1 : cfg.bm_search_radii[0]);
        ica_refine_level(ref_grey, gx, gy, moving_grey, finest_hess, flow,
                         cfg.grey_tile_size(tile_size),
                         cfg.ica_n_iter, cfg.num_threads,
                         ica_damp_ratio(cfg), ica_max_step(cfg, finest_radius));
    }
    mark_motion_irregular_tiles(flow, cfg);
    return flow;
}

// Re-express a flow field estimated on the grey onto a raw-resolution tile
// grid, scaling displacements by the resolution ratio.
//
// Alignment runs on the grey. With the Bayer quad average that is half
// resolution, so the field is on a half-res tile grid carrying half-res
// displacements, while robustness (via upscale_warp_stats, which runs over
// guide.h * 2 = raw height) and the merge kernel both index the flow as
// raw_coordinate / tile_size and add the displacement in raw pixels. Without
// this conversion every motion vector would be half its true value.
//
// Each raw tile takes the flow of whichever grey tile covers its centre, so the
// mapping is a pure nearest-tile lookup plus a scale -- no interpolation, which
// would invent displacements the search never evaluated.
//
// Returns the input unchanged when the grey is already full resolution, so the
// FFT path is unaffected.
// tile_size is in RAW pixels (the raw tile grid this builds); guide_tile_size
// is the size the alignment actually used on the grey. They differ by
// Config::alignment_grey_scale on the decimate path -- passing one value for
// both is what made a raw tile map to the wrong source tile.
FlowField flow_to_raw_tile_grid(const FlowField& flow, int raw_h, int raw_w,
                                int grey_h, int grey_w, int tile_size,
                                f32 r_Mt, int num_threads, int guide_tile_size) {
    if (guide_tile_size <= 0) guide_tile_size = tile_size;
    if (flow.nx <= 0 || flow.ny <= 0 || flow.flow.empty() ||
        raw_h <= 0 || raw_w <= 0 || grey_h <= 0 || grey_w <= 0 ||
        tile_size <= 0)
        return flow;

    const f32 sx = (f32)raw_w / (f32)grey_w;
    const f32 sy = (f32)raw_h / (f32)grey_h;
    if (std::fabs(sx - 1.f) < 1e-6f && std::fabs(sy - 1.f) < 1e-6f)
        return flow;

    const int raw_ny = (raw_h + tile_size - 1) / tile_size;
    const int raw_nx = (raw_w + tile_size - 1) / tile_size;
    FlowField out(raw_ny, raw_nx);
    // Re-measured here rather than carried, because only this function knows
    // the grey-to-raw scale that puts the span into the units r_Mt is defined
    // in. Measured on the INPUT grid so the 3x3 neighbourhood spans three
    // distinct alignment tiles -- after the mapping below, three consecutive
    // raw tiles cover only two, which is a narrower and noisier estimate of the
    // same quantity. Then mapped with the displacements.
    std::vector<uint32_t> src_irregular;
    if (flow.has_motion_prior())
        src_irregular = compute_motion_irregular(flow, r_Mt, sx, sy, num_threads);
    const bool carry_motion = src_irregular.size() == (size_t)flow.ny * flow.nx;
    if (carry_motion) out.motion_irregular.assign((size_t)raw_ny * raw_nx, 0u);
    for (int ty = 0; ty < raw_ny; ++ty) {
        const f32 raw_cy = ((f32)ty + 0.5f) * (f32)tile_size;
        const int gy = std::max(0, std::min(flow.ny - 1,
            (int)std::floor((raw_cy / sy) / (f32)guide_tile_size)));
        for (int tx = 0; tx < raw_nx; ++tx) {
            const f32 raw_cx = ((f32)tx + 0.5f) * (f32)tile_size;
            const int gx = std::max(0, std::min(flow.nx - 1,
                (int)std::floor((raw_cx / sx) / (f32)guide_tile_size)));
            out.dx(ty, tx) = flow.dx(gy, gx) * sx;
            out.dy(ty, tx) = flow.dy(gy, gx) * sy;
            if (flow.match_ambiguous.size() == (size_t)flow.ny * (size_t)flow.nx)
                out.ambiguous(ty, tx) = flow.ambiguous(gy, gx);
            if (carry_motion)
                out.irregular(ty, tx) = src_irregular[(size_t)gy * flow.nx + gx];
        }
    }
    return out;
}

// Full-resolution ICA polish (Config::align_fullres_polish): one final ICA
// refinement of the RAW-grid flow at raw resolution, on the band-limited
// full-res FFT grey both frames were converted to -- the same image the FFT
// path measures on. Run after flow_to_raw_tile_grid and before the boundary
// densify, while the seed is the finished decimate estimate. On device the
// Metal path (prep_level_ica_gpu + ica_bufs, ref side cached per burst) does
// the work; the CPU body below exists for host tools and forced-CPU debug,
// built from the same Sobel/Hessian/ica_refine_level pieces the CPU align
// driver uses at its own finest level.
bool flow_fullres_ica_polish(const Image& ref_grey_full, const Image& mov_grey_full,
                             FlowField& flow, int tile_size, const Config& cfg) {
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty() || tile_size <= 0 ||
        ref_grey_full.h <= 0 || ref_grey_full.w <= 0 ||
        mov_grey_full.h <= 0 || mov_grey_full.w <= 0)
        return false;
#ifdef __APPLE__
    if (!env_flag_on("HHSR_ALIGN_CPU"))
        return ica_fullres_polish_metal(ref_grey_full, mov_grey_full, flow,
                                        tile_size, cfg);
#endif
    const int ny = (ref_grey_full.h + tile_size - 1) / tile_size;
    const int nx = (ref_grey_full.w + tile_size - 1) / tile_size;
    if (ny != flow.ny || nx != flow.nx) return false;
    Image gx = compute_sobel_gradx(ref_grey_full);
    Image gy = compute_sobel_grady(ref_grey_full);
    HessianField hess = compute_hessian(gx, gy, tile_size);
    if (hess.ny != flow.ny || hess.nx != flow.nx) return false;
    ica_refine_level(ref_grey_full, gx, gy, mov_grey_full, hess, flow,
                     tile_size, cfg.ica_n_iter, cfg.num_threads,
                     ica_damp_ratio(cfg), ica_max_step(cfg, 1));
    return true;
}
// Build the boundary-selected half-pitch refinement of a RAW-grid flow field
// (FlowField::fine_*, consumed transparently by sample_bilinear and the GPU
// hosts). Runs after flow_to_raw_tile_grid, on the same greys alignment used.
//
// Each fine cell (pitch tile_size/2) looks at the four tile vectors its
// centre sits between -- the exact four bilinear sampling reads there:
//  - If they agree within cfg.flow_select_threshold raw px (Chebyshev), the
//    cell stores their bilinear blend at its centre, and consumers sampling
//    the fine grid reproduce the coarse sampling to first order: the fine
//    lattice is quarter-shifted from the coarse one, so the match is exact
//    for a locally linear field away from image borders (measured 6e-8) and
//    bounded by ~1/8 of the flow's per-tile second difference elsewhere --
//    far below the staircase this machinery removes. Sub-tile gradients
//    (rotation) survive by construction.
//  - Otherwise the cell sits on a motion boundary, where a blend produces
//    flow belonging to NEITHER side. Each of the four vectors is scored by L1
//    residual over the cell's footprint on the alignment grey (rounded
//    integer warp -- this selects between motions pixels apart, it does not
//    re-estimate sub-pixel), and the best single vector is stored verbatim,
//    fractional part included. Ties keep the lowest candidate index; a
//    candidate whose warp leaves the moving grey is skipped; if none can be
//    scored the blend stands, which is today's behaviour.
void flow_densify_boundary_select(FlowField& flow,
                                  const Image& ref_grey, const Image& mov_grey,
                                  int raw_h, int raw_w, int tile_size,
                                  const Config& cfg) {
    flow.fine_ny = 0;
    flow.fine_nx = 0;
    flow.fine_flow.clear();
    flow.fine_div = 2;
    // Layer 3 replaces this whole stage: instead of deciding per half-pitch
    // cell between a blend and one of the four neighbouring tile vectors, it
    // re-estimates the flow densely. Dispatched from here so all seven call
    // sites keep working unchanged.
    //
    // Deliberately ahead of the flow_bilinear_sampling gate: at one cell per
    // grey pixel the lattice step under rotation is 0.035 raw px (0.28 sigma
    // against the merge kernel), so consuming the field nearest is no longer
    // the staircase that gate exists to avoid.
    if (cfg.flow_dense_lk_enabled) {
        flow_densify_lucas_kanade(flow, ref_grey, mov_grey, raw_h, raw_w,
                                  tile_size, cfg);
        return;
    }
    const bool overlap_all = cfg.overlap_merge_active();
    if ((!cfg.flow_boundary_selection && !overlap_all) ||
        !cfg.flow_bilinear_sampling)
        return;
    // Even pitch only: the fine grid's pitch is tile_size/2 and the GPU hosts
    // pass it as an integer. Every shipped tile size (16/32/64) is even.
    if (flow.ny <= 0 || flow.nx <= 0 || tile_size < 2 || (tile_size % 2) != 0 ||
        ref_grey.h <= 0 || ref_grey.w <= 0 || ref_grey.c != 1 ||
        mov_grey.h <= 0 || mov_grey.w <= 0 || mov_grey.c != 1 ||
        raw_h <= 0 || raw_w <= 0)
        return;
#ifdef __APPLE__
    // The measurement branches below are the last CPU hotspot of the align
    // stage (~200ms/frame on violent bursts with the overlap merge on); the
    // GPU twin runs one thread per fine cell with identical math. The CPU
    // body remains for host tools and HHSR_ALIGN_CPU debugging.
    if (!env_flag_on("HHSR_ALIGN_CPU") &&
        flow_densify_select_metal(flow, ref_grey, mov_grey, raw_h, raw_w,
                                  tile_size, cfg))
        return;
#endif
    const f32 ts2 = 0.5f * (f32)tile_size;
    const int fny = std::max(1, (int)std::ceil((f32)raw_h / ts2));
    const int fnx = std::max(1, (int)std::ceil((f32)raw_w / ts2));
    // Raw -> grey scale (0.5 on the decimate grey, 1 on the FFT grey). The
    // flow's displacements are raw px (flow_to_raw_tile_grid scaled them), so
    // the residual test converts both positions and displacements.
    const f32 gsy = (f32)mov_grey.h / (f32)raw_h;
    const f32 gsx = (f32)mov_grey.w / (f32)raw_w;
    const f32 thr = std::max(0.f, cfg.flow_select_threshold);
    std::vector<f32> fine((size_t)fny * (size_t)fnx * 2u, 0.f);
    parallel_rows(fny, cfg.num_threads, [&](int fy) {
        for (int fx = 0; fx < fnx; ++fx) {
            // Same lattice mapping as FlowField::sample_grid at the cell centre.
            const f32 cy = ((f32)fy + 0.5f) * ts2;
            const f32 cx = ((f32)fx + 0.5f) * ts2;
            const f32 tcy = cy / (f32)tile_size - 0.5f;
            const f32 tcx = cx / (f32)tile_size - 0.5f;
            const int y0 = (int)std::floor(tcy), x0 = (int)std::floor(tcx);
            const f32 ay = tcy - (f32)y0, ax = tcx - (f32)x0;
            auto cl = [](int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); };
            const int iy0 = cl(y0, flow.ny), iy1 = cl(y0 + 1, flow.ny);
            const int ix0 = cl(x0, flow.nx), ix1 = cl(x0 + 1, flow.nx);
            const f32 vx[4] = {flow.dx(iy0, ix0), flow.dx(iy0, ix1),
                               flow.dx(iy1, ix0), flow.dx(iy1, ix1)};
            const f32 vy[4] = {flow.dy(iy0, ix0), flow.dy(iy0, ix1),
                               flow.dy(iy1, ix0), flow.dy(iy1, ix1)};
            f32 spread = 0.f;
            for (int a = 0; a < 4; ++a)
                for (int b = a + 1; b < 4; ++b)
                    spread = std::max(spread,
                                      std::max(std::fabs(vx[a] - vx[b]),
                                               std::fabs(vy[a] - vy[b])));
            const f32 tx0 = vx[0] + (vx[1] - vx[0]) * ax;
            const f32 bx0 = vx[2] + (vx[3] - vx[2]) * ax;
            const f32 ty0 = vy[0] + (vy[1] - vy[0]) * ax;
            const f32 by0 = vy[2] + (vy[3] - vy[2]) * ax;
            f32 out_x = tx0 + (bx0 - tx0) * ay;
            f32 out_y = ty0 + (by0 - ty0) * ay;
            if (overlap_all && spread > thr) {
                // Overlapped-tile measurement (Config::flow_overlap_merge),
                // for cells whose four neighbour vectors DISAGREE: the merge
                // consumes these nearest, one per covering tile, so each must
                // be the cell's own measurement on its FULL tile-sized window
                // (Ts = tile_size at stride Ts/2 -- HDR+'s layout). Cells
                // whose neighbours agree keep the blend: measuring a motion
                // all four tiles already agree on adds only measurement noise
                // and, being never bit-equal, defeated the merge's hypothesis
                // clustering -- every pixel paid 4 gathers and the merge
                // stopped hiding behind the CPU (measured: 'much slower').
                const int win = std::max(2, (int)std::lround((f32)tile_size * gsy));
                const int gy0 = (int)std::lround(((f32)fy + 0.5f) * ts2 * gsy) - win / 2;
                const int gx0 = (int)std::lround(((f32)fx + 0.5f) * ts2 * gsx) - win / 2;
                auto cost_at = [&](int idy, int idx) -> f32 {
                    f32 dist = 0.f;
                    for (int i = 0; i < win; ++i) {
                        for (int j = 0; j < win; ++j) {
                            const int ry = gy0 + i, rx = gx0 + j;
                            const int my = ry + idy, mx = rx + idx;
                            if (!(ry >= 0 && ry < ref_grey.h &&
                                  rx >= 0 && rx < ref_grey.w &&
                                  my >= 0 && my < mov_grey.h &&
                                  mx >= 0 && mx < mov_grey.w))
                                return std::numeric_limits<f32>::infinity();
                            dist += std::fabs(ref_grey.at(ry, rx) -
                                              mov_grey.at(my, mx));
                        }
                    }
                    return dist;
                };
                // Seed: best of the four neighbour vectors on this window;
                // border cells where nothing scores keep the blend.
                f32 best_cost = std::numeric_limits<f32>::infinity();
                int seed_y = 0, seed_x = 0;
                bool have_seed = false;
                for (int c = 0; c < 4; ++c) {
                    const int idx = (int)std::lround(vx[c] * gsx);
                    const int idy = (int)std::lround(vy[c] * gsy);
                    const f32 d = cost_at(idy, idx);
                    if (d < best_cost) {
                        best_cost = d;
                        seed_y = idy;
                        seed_x = idx;
                        have_seed = true;
                        out_x = vx[c];
                        out_y = vy[c];
                    }
                }
                // One 3x3 integer refinement plus the quadratic sub-cell fit.
                // The measurement replaces the seed only when it MOVES: a
                // grey-integer step is 2 raw px of quantisation on decimate,
                // so a confirmed seed keeps its ICA-refined fraction.
                if (have_seed) {
                    f32 surf[9];
                    int bo = 4;
                    f32 bo_cost = best_cost;
                    for (int oy = -1; oy <= 1; ++oy)
                        for (int ox = -1; ox <= 1; ++ox) {
                            const int s = (oy + 1) * 3 + (ox + 1);
                            surf[s] = (ox == 0 && oy == 0)
                                          ? best_cost
                                          : cost_at(seed_y + oy, seed_x + ox);
                            if (surf[s] < bo_cost) { bo_cost = surf[s]; bo = s; }
                        }
                    if (bo != 4) {
                        seed_y += bo / 3 - 1;
                        seed_x += bo % 3 - 1;
                        for (int oy = -1; oy <= 1; ++oy)
                            for (int ox = -1; ox <= 1; ++ox)
                                surf[(oy + 1) * 3 + (ox + 1)] =
                                    cost_at(seed_y + oy, seed_x + ox);
                        if (std::isfinite(surf[4])) {
                            f32 sub_x = 0.f, sub_y = 0.f;
                            (void)quadratic_subpixel_3x3(surf, sub_x, sub_y);
                            out_x = ((f32)seed_x + sub_x) / gsx;
                            out_y = ((f32)seed_y + sub_y) / gsy;
                        }
                    }
                }
            } else if (spread > thr) {
                // Cell footprint on the grey.
                const int gy0 = (int)std::floor((f32)fy * ts2 * gsy);
                const int gx0 = (int)std::floor((f32)fx * ts2 * gsx);
                const int gh = std::max(1, (int)std::lround(ts2 * gsy));
                const int gw = std::max(1, (int)std::lround(ts2 * gsx));
                f32 best_cost = std::numeric_limits<f32>::infinity();
                for (int c = 0; c < 4; ++c) {
                    const int idx = (int)std::lround(vx[c] * gsx);
                    const int idy = (int)std::lround(vy[c] * gsy);
                    f32 dist = 0.f;
                    bool valid = true;
                    for (int i = 0; i < gh && valid; ++i) {
                        for (int j = 0; j < gw; ++j) {
                            const int ry = gy0 + i, rx = gx0 + j;
                            const int my = ry + idy, mx = rx + idx;
                            if (!(ry >= 0 && ry < ref_grey.h &&
                                  rx >= 0 && rx < ref_grey.w &&
                                  my >= 0 && my < mov_grey.h &&
                                  mx >= 0 && mx < mov_grey.w)) {
                                valid = false;
                                break;
                            }
                            dist += std::fabs(ref_grey.at(ry, rx) -
                                              mov_grey.at(my, mx));
                        }
                    }
                    if (valid && dist < best_cost) {
                        best_cost = dist;
                        out_x = vx[c];
                        out_y = vy[c];
                    }
                }
            }
            fine[((size_t)fy * fnx + fx) * 2 + 0] = out_x;
            fine[((size_t)fy * fnx + fx) * 2 + 1] = out_y;
        }
    });
    flow.fine_flow = std::move(fine);
    flow.fine_ny = fny;
    flow.fine_nx = fnx;
}

// ---------------------------------------------------------------------------
// Layer 3: dense Lucas-Kanade refinement (ImageStackAlignator lucasKanadeOptim)
// ---------------------------------------------------------------------------
//
// opticalFlow.cu:190 runs one thread PER PIXEL over the dense flow field,
// solving a 2x2 normal-equation system over a sliding window. That is the
// layer that actually removes the tile lattice from the result, and it is the
// one this port never had: ica_refine_level solves once per TILE, so the
// output is a lattice of independent constants no matter how it is later
// interpolated.
//
// Two error terms die here that no interpolator can touch:
//
//  - Independent per-tile estimation noise. Neighbouring tiles share no pixels,
//    so their errors are uncorrelated and show as a step of ~eps*sqrt(2) at
//    every seam, with exactly the tile pitch. Interpolation rounds the step's
//    edges; the field is still wrong by eps inside each tile.
//  - Attribution bias. A per-tile solve minimises sum |grad I . u + I_t|^2
//    over the tile, so the vector it returns is the field's value at the
//    tile's GRADIENT-ENERGY CENTROID, not at its centre -- but it is stored
//    and interpolated as if it belonged at the centre. With a field gradient
//    of theta = 0.0175 px/px, a centroid 5 px off-centre biases that tile by
//    0.087 px. Content-dependent, so it differs tile to tile: another grid.
//
// Solving per cell at one cell per grey pixel makes both terms local instead
// of tile-periodic. The window still spans several pixels, so neighbouring
// cells share most of their data and their errors are strongly correlated --
// which is the point: correlated error is smooth, and smooth error is not a
// visible seam.
//
// Departures from ISA, both deliberate:
//  - It solves with an SVD and a minDet cutoff; this uses Levenberg damping
//    (A + lambda*I, lambda from the trace). Same intent -- do not invert an
//    aperture-problem system -- but it degrades toward the seeded value
//    continuously instead of switching at a threshold, which matters when the
//    switch itself would be a visible boundary.
//  - Cells are seeded by sampling the coarse field, which for the rigid part
//    of the motion is exact: bilinear interpolation reproduces a linear
//    function, and a rotation's displacement field IS linear in position.
// Pitch selection for the dense lattice. Finest first: the lattice step under
// rotation is theta * pitch, so a smaller pitch is strictly better for the
// artifact, and the only reason to back off is the buffer. A 12MP raw at
// pitch 2 costs 24 MB; a 48MP raw would cost 96 MB, which is why this is a
// budget and not a constant.
//
// Shared with the Metal twin so the two cannot disagree about the lattice
// they are filling -- a mismatch there would not fail, it would silently
// write a correct field at the wrong pitch.
bool flow_dense_lk_lattice(int raw_h, int raw_w, int tile_size,
                           const Config& cfg,
                           int& div, int& fny, int& fnx, f32& pitch) {
    div = 0; fny = 0; fnx = 0; pitch = 0.f;
    if (raw_h <= 0 || raw_w <= 0 || tile_size < 2) return false;
    const size_t budget = (size_t)std::max(1, cfg.flow_dense_lk_max_mb)
                          * 1024u * 1024u;
    for (int d : {8, 4, 2}) {
        if (tile_size % d != 0) continue;
        const f32 p = (f32)tile_size / (f32)d;
        const int ny = std::max(1, (int)std::ceil((f32)raw_h / p));
        const int nx = std::max(1, (int)std::ceil((f32)raw_w / p));
        if ((size_t)ny * (size_t)nx * 2u * sizeof(f32) > budget) continue;
        div = d; pitch = p; fny = ny; fnx = nx;
        return true;
    }
    return false;
}

void flow_densify_lucas_kanade(FlowField& flow,
                               const Image& ref_grey, const Image& mov_grey,
                               int raw_h, int raw_w, int tile_size,
                               const Config& cfg) {
    flow.fine_flow.clear();
    flow.fine_div = 2;
    if (flow.ny <= 0 || flow.nx <= 0 || tile_size < 2 ||
        ref_grey.h <= 0 || ref_grey.w <= 0 || ref_grey.c != 1 ||
        mov_grey.h <= 0 || mov_grey.w <= 0 || mov_grey.c != 1 ||
        raw_h <= 0 || raw_w <= 0)
        return;

    int div = 0, fny = 0, fnx = 0;
    f32 pitch = 0.f;
    if (!flow_dense_lk_lattice(raw_h, raw_w, tile_size, cfg,
                               div, fny, fnx, pitch))
        return;             // nothing fits: leave the coarse field alone

    // Raw -> grey scale (0.5 on the decimate grey, 1 on the FFT grey). The
    // flow carries RAW-px displacements (flow_to_raw_tile_grid scaled them),
    // so positions and displacements both convert.
    const f32 gsy = (f32)ref_grey.h / (f32)raw_h;
    const f32 gsx = (f32)ref_grey.w / (f32)raw_w;
    if (!(gsy > 0.f) || !(gsx > 0.f)) return;

    // Reference gradients, built once per burst (see g_dlk_grad_*). Copied
    // out under the lock rather than aliased: the caller may run comparison
    // frames concurrently, and a later burst start would otherwise free these
    // while a worker is still reading them.
    Image gx, gy;
    {
        std::lock_guard<std::mutex> lk(g_dlk_grad_mu);
        if (g_dlk_grad_key != ref_grey.data.data() ||
            g_dlk_grad_n != ref_grey.data.size()) {
            g_dlk_gx = compute_sobel_gradx(ref_grey);
            g_dlk_gy = compute_sobel_grady(ref_grey);
            g_dlk_grad_key = ref_grey.data.data();
            g_dlk_grad_n = ref_grey.data.size();
        }
        gx = g_dlk_gx;
        gy = g_dlk_gy;
    }
    if (gx.h != ref_grey.h || gx.w != ref_grey.w ||
        gy.h != ref_grey.h || gy.w != ref_grey.w) return;

#ifdef __APPLE__
    // One thread per lattice cell with identical math; the CPU body below
    // remains for host tools and HHSR_ALIGN_CPU debugging. At one cell per
    // grey pixel this stage is far too heavy to leave on the CPU.
    if (!env_flag_on("HHSR_ALIGN_CPU") &&
        flow_dense_lk_metal(flow, ref_grey, mov_grey, gx, gy,
                            raw_h, raw_w, tile_size, cfg))
        return;
#endif

    const int hw = std::max(1, cfg.flow_dense_lk_half_window);
    const int iters = std::max(1, cfg.flow_dense_lk_iters);
    const f32 damp = std::max(0.f, cfg.flow_dense_lk_damping);
    const f32 max_step_g = std::max(0.f, cfg.flow_dense_lk_max_step) *
                           0.5f * (gsy + gsx);

    std::vector<f32> fine((size_t)fny * (size_t)fnx * 2u, 0.f);

    auto samp = [](const Image& im, f32 y, f32 x) -> f32 {
        // Clamped bilinear: a window that runs off the moving frame keeps
        // contributing edge data rather than dropping to zero, which would
        // read as a huge temporal gradient and throw the solve.
        const f32 cy = (y < 0.f) ? 0.f : (y > (f32)(im.h - 1) ? (f32)(im.h - 1) : y);
        const f32 cx = (x < 0.f) ? 0.f : (x > (f32)(im.w - 1) ? (f32)(im.w - 1) : x);
        const int y0 = (int)cy, x0 = (int)cx;
        const int y1 = (y0 + 1 < im.h) ? y0 + 1 : y0;
        const int x1 = (x0 + 1 < im.w) ? x0 + 1 : x0;
        const f32 ay = cy - (f32)y0, ax = cx - (f32)x0;
        return (im.at(y0, x0) * (1.f - ax) + im.at(y0, x1) * ax) * (1.f - ay) +
               (im.at(y1, x0) * (1.f - ax) + im.at(y1, x1) * ax) * ay;
    };

    parallel_rows(fny, cfg.num_threads, [&](int fy) {
        for (int fx = 0; fx < fnx; ++fx) {
            // Cell centre in raw px, then on the alignment grey.
            const f32 ry = ((f32)fy + 0.5f) * pitch;
            const f32 rx = ((f32)fx + 0.5f) * pitch;
            f32 dxr = 0.f, dyr = 0.f;
            // Bilinear explicitly, NOT flow.sample_bilinear: the pipeline sets
            // FlowField::sample_bicubic from Config::flow_bicubic_sampling
            // BEFORE this stage runs (pipeline.cpp), so going through the
            // dispatcher would seed with Catmull-Rom here while the Metal twin
            // seeds bilinearly -- a divergence visible only with Bicubic Flow
            // on. The order is immaterial to the result (this is the starting
            // point of an iterative solve, and the dense field is what
            // consumers interpolate afterwards), so both are pinned to the
            // cheaper one rather than teaching the shader a second form.
            FlowField::sample_grid(flow.flow.data(), flow.ny, flow.nx,
                                   (f32)tile_size, ry, rx, dxr, dyr);

            const f32 cy = ry * gsy, cx = rx * gsx;
            f32 ux = dxr * gsx, uy = dyr * gsy;   // grey-px displacement

            for (int it = 0; it < iters; ++it) {
                f32 a00 = 0.f, a01 = 0.f, a11 = 0.f, b0 = 0.f, b1 = 0.f;
                for (int wy = -hw; wy <= hw; ++wy) {
                    const f32 py = cy + (f32)wy;
                    if (py < 0.f || py > (f32)(ref_grey.h - 1)) continue;
                    for (int wx = -hw; wx <= hw; ++wx) {
                        const f32 px = cx + (f32)wx;
                        if (px < 0.f || px > (f32)(ref_grey.w - 1)) continue;
                        const f32 ix = samp(gx, py, px);
                        const f32 iy = samp(gy, py, px);
                        const f32 it_ = samp(mov_grey, py + uy, px + ux) -
                                        samp(ref_grey, py, px);
                        a00 += ix * ix; a01 += ix * iy; a11 += iy * iy;
                        b0  -= ix * it_; b1 -= iy * it_;
                    }
                }
                // Levenberg damping off the trace: scale-free, and it leaves
                // an aperture-problem cell sitting at its seeded value rather
                // than sliding along the edge.
                const f32 lam = damp * 0.5f * (a00 + a11);
                const f32 m00 = a00 + lam, m11 = a11 + lam;
                const f32 det = m00 * m11 - a01 * a01;
                if (!(std::fabs(det) > 1e-20f)) break;
                f32 sx_ = ( m11 * b0 - a01 * b1) / det;
                f32 sy_ = (-a01 * b0 + m00 * b1) / det;
                if (max_step_g > 0.f) {
                    const f32 mag = std::sqrt(sx_ * sx_ + sy_ * sy_);
                    if (mag > max_step_g) {
                        const f32 k = max_step_g / mag;
                        sx_ *= k; sy_ *= k;
                    }
                }
                if (!std::isfinite(sx_) || !std::isfinite(sy_)) break;
                ux += sx_; uy += sy_;
            }

            const size_t o = ((size_t)fy * fnx + fx) * 2u;
            fine[o + 0] = ux / gsx;   // back to raw px
            fine[o + 1] = uy / gsy;
        }
    });

    flow.fine_flow.swap(fine);
    flow.fine_ny = fny;
    flow.fine_nx = fnx;
    flow.fine_div = div;
}

} // namespace hhsr
