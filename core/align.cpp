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

static f32 clamped_aperture_ratio(const Config& cfg) {
    f32 aperture_ratio = cfg.flow_regularize_aperture_ratio;
    if (!std::isfinite(aperture_ratio)) aperture_ratio = 0.15f;
    return std::min(std::max(aperture_ratio, 0.f), 1.f);
}

// ICA damping ratio: the eigenvalue ratio the solve is regularized toward.
// Shares flow_regularize_aperture_ratio with mark_aperture_limited_tiles, since
// both express the same thing -- the ratio below which a tile is 1D. 0 disables.
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

static uint32_t match_is_ambiguous(f32 best_cost, f32 second_cost,
                                   f32 ambiguity_ratio) {
    if (!std::isfinite(best_cost) || !std::isfinite(second_cost)) return 0u;
    if (best_cost < 0.f) best_cost = 0.f;
    if (second_cost < 0.f) second_cost = 0.f;
    const f32 denom = std::max(best_cost, 1e-12f);
    return (second_cost / denom) < ambiguity_ratio ? 1u : 0u;
}

static bool hessian_tile_is_1d(const HessianField& hess, int ty, int tx,
                               f32 aperture_ratio) {
    const f32* h = hess.at(ty, tx);
    const f32 h00 = h[0], h01 = h[1], h11 = h[3];
    const f32 tr = h00 + h11;
    const f32 det = h00 * h11 - h01 * h[2];
    if (!(tr > 0.f)) return false; // Flat: weak in both directions, not a 1D edge.
    const f32 disc = std::sqrt(std::max(0.f, tr * tr * 0.25f - det));
    const f32 l1 = tr * 0.5f + disc;
    const f32 l2 = tr * 0.5f - disc;
    return (l1 > 0.f) && (l2 < aperture_ratio * l1);
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

// Same measurement, second and much larger threshold -- see
// Config::motion_magnitude_veto_enabled and FlowField::motion_magnitude_reject.
static void mark_motion_magnitude_reject_tiles(FlowField& flow, const Config& cfg) {
    flow.motion_magnitude_reject =
        compute_motion_irregular(flow, cfg.motion_magnitude_veto_px, 1.f, 1.f,
                                 cfg.num_threads);
}

static void mark_aperture_limited_tiles(FlowField& flow, const HessianField* hess,
                                        const Config& cfg) {
    const size_t n = (size_t)std::max(0, flow.ny) * (size_t)std::max(0, flow.nx);
    flow.aperture_limited.assign(n, 0u);
    if (!cfg.flow_reject_1d_enabled) return;
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty()) return;
    if (!hess || hess->ny != flow.ny || hess->nx != flow.nx ||
        hess->data.size() < n * 4u)
        return;

    const f32 aperture_ratio = clamped_aperture_ratio(cfg);
    std::atomic<long long> n_marked{0};
    parallel_rows(flow.ny, cfg.num_threads, [&](int ty) {
        for (int tx = 0; tx < flow.nx; ++tx) {
            if (hessian_tile_is_1d(*hess, ty, tx, aperture_ratio)) {
                flow.aperture(ty, tx) = 1u;
                n_marked.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });
    if (prof_enabled()) {
        prof_add_cpu("flow1d#enabled", 1.0);
        prof_add_cpu("flow1d#aperture-ratio", (double)aperture_ratio);
        prof_add_cpu("flow1d#rejected-tiles", (double)n_marked.load());
    }
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
                                         int num_threads) {
    const int ny = flow.ny, nx = flow.nx;
    const int ts = tile_size;
    const int R = search_radius;
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
            // ImageStackAlignator's "no reasonable peak -> fall back" --
            // see Config::align_ambiguous_fallback_enabled. Computed
            // whenever either consumer wants it: write_ambiguity persists
            // the flag for downstream robustness demotion, fallback_on_
            // ambiguous uses it here to decide whether to trust this
            // level's match at all.
            const bool ambiguous_here = (write_ambiguity || fallback_on_ambiguous) &&
                match_is_ambiguous(min_dist, second_dist, ambiguity_ratio) != 0u;
            if (fallback_on_ambiguous && ambiguous_here) {
                // Leave the seed as-is: the coarser level's estimate, or at
                // the coarsest level, the global/thumbnail pre-alignment
                // initial flow.
                flow.dx(ty, tx) = local_fx;
                flow.dy(ty, tx) = local_fy;
            } else {
                flow.dx(ty, tx) = local_fx + (f32)min_shift_x;
                flow.dy(ty, tx) = local_fy + (f32)min_shift_y;
            }
            if (write_ambiguity)
                flow.ambiguous(ty, tx) |= ambiguous_here ? 1u : 0u;
        }
    });
}

static void block_match_level_L2(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, const Config& cfg,
                                  int num_threads) {
    const bool write_ambiguity = cfg.flow_reject_ambiguous_enabled;
    const f32 ambiguity_ratio = clamped_ambiguity_ratio(cfg);
    const bool fallback_on_ambiguous = cfg.align_ambiguous_fallback_enabled;
#ifdef __APPLE__
    // 460-main direct local L2 search. HHSR_L2_CPU=1 forces CPU direct path.
    if (!env_flag_on("HHSR_L2_CPU") && !env_flag_on("HHSR_ALIGN_CPU")) {
        if (block_match_level_L2_metal(ref, moving, tile_size, search_radius, flow,
                                       ambiguity_ratio, write_ambiguity,
                                       fallback_on_ambiguous))
            return;
    }
#endif
    block_match_level_direct_460(ref, moving, tile_size, search_radius, flow,
                                 false, ambiguity_ratio, write_ambiguity,
                                 fallback_on_ambiguous, num_threads);
}

// ============================================================================
// L1 BM — CUDA kernels + broken argmin. ts=64 uses 1024 threads × 4 px.
// ============================================================================
static void block_match_level_L1(const Image& ref, const Image& moving,
                                  int tile_size, int search_radius,
                                  FlowField& flow, const Config& cfg,
                                  int num_threads) {
    const bool write_ambiguity = cfg.flow_reject_ambiguous_enabled;
    const f32 ambiguity_ratio = clamped_ambiguity_ratio(cfg);
    const bool fallback_on_ambiguous = cfg.align_ambiguous_fallback_enabled;
#ifdef __APPLE__
    if (!env_flag_on("HHSR_L1_CPU") && !env_flag_on("HHSR_ALIGN_CPU") &&
        block_match_level_L1_metal(ref, moving, tile_size, search_radius, flow,
                                   ambiguity_ratio, write_ambiguity,
                                   fallback_on_ambiguous))
        return;
#endif
    block_match_level_direct_460(ref, moving, tile_size, search_radius, flow,
                                 true, ambiguity_ratio, write_ambiguity,
                                 fallback_on_ambiguous, num_threads);
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
            if (in.aperture_limited.size() == (size_t)in.ny * (size_t)in.nx)
                upsampled.aperture(ty, tx) = in.aperture(sy, sx);
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
                if (upsampled.aperture_limited.size() == (size_t)up_ny * (size_t)up_nx)
                    out.aperture(ty, tx) = upsampled.aperture(ty, tx);
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
                                  int prev_tile_size) {
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
                    best_dist = dist;
                    best_i = ci;
                }
            }
            out.dx(ty, tx) = cand[best_i][0];
            out.dy(ty, tx) = cand[best_i][1];
            if (in.aperture_limited.size() == (size_t)in.ny * (size_t)in.nx) {
                int ap_y = prev_y;
                int ap_x = prev_x;
                if (best_i == 1) {
                    ap_y = cand_y;
                    ap_x = prev_x;
                } else if (best_i == 2) {
                    ap_y = prev_y;
                    ap_x = cand_x;
                }
                out.aperture(ty, tx) = in.aperture(ap_y, ap_x);
            }
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

void clear_align_ref_ica_cache() {
    g_ref_ica_cache = {};
#ifdef __APPLE__
    metal_clear_ref_ica_cache();
#endif
}

FlowField make_global_initial_flow(int ny, int nx, int tile_size, int abs_factor,
                                   int finest_h, int finest_w,
                                   f32 initial_dx, f32 initial_dy,
                                   f32 initial_rotation_rad) {
    FlowField flow(ny, nx);
    const f32 factor = (f32)std::max(1, abs_factor);
    const f32 inv_factor = 1.f / factor;
    const f32 dx = std::isfinite(initial_dx) ? initial_dx : 0.f;
    const f32 dy = std::isfinite(initial_dy) ? initial_dy : 0.f;
    const f32 a = std::isfinite(initial_rotation_rad) ? initial_rotation_rad : 0.f;
    const f32 ca = std::cos(a);
    const f32 sa = std::sin(a);
    const f32 cx = 0.5f * (f32)std::max(0, finest_w - 1);
    const f32 cy = 0.5f * (f32)std::max(0, finest_h - 1);

    for (int ty = 0; ty < ny; ++ty) {
        for (int tx = 0; tx < nx; ++tx) {
            const f32 x = ((f32)tx * (f32)tile_size + 0.5f * (f32)tile_size) * factor;
            const f32 y = ((f32)ty * (f32)tile_size + 0.5f * (f32)tile_size) * factor;
            const f32 rx = x - cx;
            const f32 ry = y - cy;
            const f32 rot_x = ca * rx - sa * ry - rx;
            const f32 rot_y = sa * rx + ca * ry - ry;
            flow.dx(ty, tx) = (dx + rot_x) * inv_factor;
            flow.dy(ty, tx) = (dy + rot_y) * inv_factor;
        }
    }
    return flow;
}

// align() — Python alignment.align
// ref_grey must already be circular-padded (init_alignment); moving is NOT.
// ============================================================================
FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg, int tile_size,
                f32 initial_dx, f32 initial_dy, f32 initial_rotation_rad) {
    int nlev = (int)ref_pyr.levels.size();

#ifdef __APPLE__
    // Default iOS path: Metal alignment. HHSR_ALIGN_CPU=1 forces the C++ path.
    if (!env_flag_on("HHSR_ALIGN_CPU")) {
        FlowField flow_gpu;
        if (align_metal(ref_pyr, ref_grey, moving_grey, cfg, tile_size, flow_gpu,
                        initial_dx, initial_dy, initial_rotation_rad)) {
            if (cfg.flow_reject_1d_enabled) {
                Image gx = compute_sobel_gradx(ref_grey);
                Image gy = compute_sobel_grady(ref_grey);
                HessianField hess = compute_hessian(gx, gy, tile_size);
                mark_aperture_limited_tiles(flow_gpu, &hess, cfg);
            } else {
                mark_aperture_limited_tiles(flow_gpu, nullptr, cfg);
            }
            mark_motion_irregular_tiles(flow_gpu, cfg);
            mark_motion_magnitude_reject_tiles(flow_gpu, cfg);
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

    // 460-main pads alternate images circularly before pyramid construction.
    Image moving_padded = pad_image_circular(moving_grey, tile_size);
    Pyramid mov_pyr = build_pyramid(moving_padded, cfg.bm_factors);

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
            const int abs_factor = (lvl < (int)ref_pyr.abs_factors.size())
                                   ? ref_pyr.abs_factors[(size_t)lvl] : 1;
            flow = make_global_initial_flow(ny, nx, ts, abs_factor,
                                            ref_grey.h, ref_grey.w,
                                            initial_dx, initial_dy,
                                            initial_rotation_rad);
        } else {
            int upsample_factor = ((lvl + 1) < (int)cfg.bm_factors.size())
                                  ? cfg.bm_factors[lvl + 1] : 1;
            int prev_ts = ((lvl + 1) < (int)cfg.bm_tile_sizes.size())
                          ? cfg.bm_tile_sizes[lvl + 1]
                          : ts;
            flow = upscale_flow_460(r, m, flow, ny, nx, upsample_factor, ts, prev_ts);
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
    HessianField finest_hess;
    if (!ica_all || cfg.ica_per_level_coarse_only()) {
        Image gx = compute_sobel_gradx(ref_grey);
        Image gy = compute_sobel_grady(ref_grey);
        finest_hess = compute_hessian(gx, gy, tile_size);
        debug_dump_bin("cpp_gradx_ica", gx.data.data(), gx.data.size());
        debug_dump_bin("cpp_grady_ica", gy.data.data(), gy.data.size());
        const int finest_radius = cfg.bm_search_radii.empty() ? 1
                                                              : cfg.bm_search_radii[0];
        ica_refine_level(ref_grey, gx, gy, moving_grey, finest_hess, flow, tile_size,
                         cfg.ica_n_iter, cfg.num_threads,
                         ica_damp_ratio(cfg), ica_max_step(cfg, finest_radius));
    }
    const HessianField* mark_hess = nullptr;
    if (!finest_hess.data.empty()) {
        mark_hess = &finest_hess;
    } else if (!g_ref_ica_cache.levels.empty() &&
               g_ref_ica_cache.levels[0].hess.ny == flow.ny &&
               g_ref_ica_cache.levels[0].hess.nx == flow.nx) {
        mark_hess = &g_ref_ica_cache.levels[0].hess;
    }
    mark_aperture_limited_tiles(flow, mark_hess, cfg);
    mark_motion_irregular_tiles(flow, cfg);
    mark_motion_magnitude_reject_tiles(flow, cfg);
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
FlowField flow_to_raw_tile_grid(const FlowField& flow, int raw_h, int raw_w,
                                int grey_h, int grey_w, int tile_size,
                                f32 r_Mt, f32 motion_magnitude_veto_px,
                                int num_threads) {
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
    // Same re-measurement, second threshold -- see
    // Config::motion_magnitude_veto_enabled.
    std::vector<uint32_t> src_magnitude_reject;
    if (flow.has_motion_magnitude_prior())
        src_magnitude_reject = compute_motion_irregular(flow, motion_magnitude_veto_px,
                                                        sx, sy, num_threads);
    const bool carry_magnitude =
        src_magnitude_reject.size() == (size_t)flow.ny * flow.nx;
    if (carry_magnitude)
        out.motion_magnitude_reject.assign((size_t)raw_ny * raw_nx, 0u);
    for (int ty = 0; ty < raw_ny; ++ty) {
        const f32 raw_cy = ((f32)ty + 0.5f) * (f32)tile_size;
        const int gy = std::max(0, std::min(flow.ny - 1,
            (int)std::floor((raw_cy / sy) / (f32)tile_size)));
        for (int tx = 0; tx < raw_nx; ++tx) {
            const f32 raw_cx = ((f32)tx + 0.5f) * (f32)tile_size;
            const int gx = std::max(0, std::min(flow.nx - 1,
                (int)std::floor((raw_cx / sx) / (f32)tile_size)));
            out.dx(ty, tx) = flow.dx(gy, gx) * sx;
            out.dy(ty, tx) = flow.dy(gy, gx) * sy;
            if (flow.aperture_limited.size() == (size_t)flow.ny * (size_t)flow.nx)
                out.aperture(ty, tx) = flow.aperture(gy, gx);
            if (flow.match_ambiguous.size() == (size_t)flow.ny * (size_t)flow.nx)
                out.ambiguous(ty, tx) = flow.ambiguous(gy, gx);
            if (carry_motion)
                out.irregular(ty, tx) = src_irregular[(size_t)gy * flow.nx + gx];
            if (carry_magnitude)
                out.motion_magnitude_reject[(size_t)ty * raw_nx + tx] =
                    src_magnitude_reject[(size_t)gy * flow.nx + gx];
        }
    }
    return out;
}

// Builds a raw-pixel tile-grid FlowField from a dense per-guide-pixel flow
// field produced by an external neural flow estimator (PWCNet), re-using
// flow_to_raw_tile_grid's grey-to-raw scaling so the result is a drop-in
// replacement for align()'s output at any downstream consumer
// (compute_robustness, merge, ...).
//
// dense_flow: dx plane (guide_h*guide_w floats) followed by dy plane
// (guide_h*guide_w floats), values in GUIDE-pixel units -- the layout a
// Core ML (1,2,guide_h,guide_w) MLMultiArray output has.
//
// aperture_limited / match_ambiguous are left unset (all zero): those are
// specific to the block matcher's own candidate search and have no
// equivalent for a dense CNN flow field -- match_ambiguous-based rejection
// (robustness.cpp) simply never fires for tiles sourced this way.
// motion_irregular is likewise left unmeasured here; flow_to_raw_tile_grid
// only recomputes it when the input already carries one, so downstream
// compute_s falls back to its own derivation from the raw-tile output, same
// as any other flow source that doesn't pre-measure it.
FlowField flow_from_dense_guide(const f32* dense_flow, int guide_h, int guide_w,
                                int raw_h, int raw_w, int tile_size,
                                f32 r_Mt, f32 motion_magnitude_veto_px,
                                int num_threads) {
    if (!dense_flow || guide_h <= 0 || guide_w <= 0 || raw_h <= 0 || raw_w <= 0 || tile_size <= 0)
        return FlowField();

    const f32* dx_plane = dense_flow;
    const f32* dy_plane = dense_flow + (size_t)guide_h * (size_t)guide_w;

    const int gny = (guide_h + tile_size - 1) / tile_size;
    const int gnx = (guide_w + tile_size - 1) / tile_size;
    FlowField flow_guide(gny, gnx);

    // Average-pool the dense flow into tile_size x tile_size guide-pixel
    // blocks -- the same granularity flow_to_raw_tile_grid expects an input
    // tile grid to already be at (it re-derives the grey/raw ratio from
    // guide_h/guide_w and this same tile_size).
    parallel_rows(gny, num_threads, [&](int ty) {
        const int y0 = ty * tile_size;
        const int y1 = std::min(guide_h, y0 + tile_size);
        for (int tx = 0; tx < gnx; ++tx) {
            const int x0 = tx * tile_size;
            const int x1 = std::min(guide_w, x0 + tile_size);
            double sum_dx = 0.0, sum_dy = 0.0;
            int n = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const size_t idx = (size_t)y * guide_w + x;
                    sum_dx += dx_plane[idx];
                    sum_dy += dy_plane[idx];
                    ++n;
                }
            }
            if (n > 0) {
                flow_guide.dx(ty, tx) = (f32)(sum_dx / n);
                flow_guide.dy(ty, tx) = (f32)(sum_dy / n);
            }
        }
    });

    return flow_to_raw_tile_grid(flow_guide, raw_h, raw_w, guide_h, guide_w,
                                 tile_size, r_Mt, motion_magnitude_veto_px,
                                 num_threads);
}

// Redundant frame-to-previous-frame corroboration of this frame's own
// direct-to-reference flow -- see FlowField::chain_inconsistent for the
// reasoning (ImageStackAlignator's shift(i->ref) == shift(i->i-1) +
// shift(i-1->ref) consistency principle).
//
// prev_guide/cur_guide: guide-resolution images (compute_guide's output --
// raw/2 for Bayer, raw resolution otherwise) for the previous and current
// comparison frames, any channel count. flow_cur_to_ref/flow_prev_to_ref:
// both frames' own direct-to-reference flow, already expressed on the SAME
// raw tile grid (flow_to_raw_tile_grid's output) -- so composing the
// predicted relative shift is a same-index lookup, not a warp/resample.
//
// Cost: one single-level, no-pyramid, no-ICA block match at guide
// resolution -- the same per-tile SAD kernel block_match_level_L1 already
// runs at finer pyramid levels, just run once more here instead of a second
// full align().
std::vector<uint32_t> compute_chain_closure(const Image& prev_guide, const Image& cur_guide,
                                            const FlowField& flow_cur_to_ref,
                                            const FlowField& flow_prev_to_ref,
                                            int raw_h, int raw_w, int raw_tile_size,
                                            f32 closure_threshold_px,
                                            int base_search_radius_px,
                                            int num_threads) {
    const int ny = flow_cur_to_ref.ny, nx = flow_cur_to_ref.nx;
    std::vector<uint32_t> out;
    if (ny <= 0 || nx <= 0 || flow_cur_to_ref.flow.empty()) return out;
    if (flow_prev_to_ref.ny != ny || flow_prev_to_ref.nx != nx ||
        flow_prev_to_ref.flow.empty())
        return out;
    if (prev_guide.h <= 0 || prev_guide.w <= 0 ||
        cur_guide.h != prev_guide.h || cur_guide.w != prev_guide.w ||
        cur_guide.c != prev_guide.c || raw_h <= 0 || raw_w <= 0 || raw_tile_size <= 0)
        return out;

    // Single-channel average -- a rough SAD corroboration match needs only
    // enough signal to localize a shift, not full colour.
    Image prev_grey(prev_guide.h, prev_guide.w, 1);
    Image cur_grey(cur_guide.h, cur_guide.w, 1);
    const f32 inv_c = 1.f / (f32)std::max(1, prev_guide.c);
    for (int y = 0; y < prev_guide.h; ++y) {
        for (int x = 0; x < prev_guide.w; ++x) {
            f32 sp = 0.f, sc = 0.f;
            for (int ch = 0; ch < prev_guide.c; ++ch) {
                sp += prev_guide.at(y, x, ch);
                sc += cur_guide.at(y, x, ch);
            }
            prev_grey.at(y, x) = sp * inv_c;
            cur_grey.at(y, x) = sc * inv_c;
        }
    }

    const f32 guide_sx = (f32)prev_guide.w / (f32)raw_w;
    const f32 guide_sy = (f32)prev_guide.h / (f32)raw_h;
    const f32 guide_s = 0.5f * (guide_sx + guide_sy);
    const int guide_ts = std::max(1, (int)std::lround((f32)raw_tile_size * guide_s));
    const int guide_radius = std::max(1, (int)std::lround((f32)base_search_radius_px * guide_s));

    // Seed every tile with the predicted relative shift: what frame i's
    // motion relative to frame i-1 would be if both frames' own alignment to
    // the reference is honest.
    FlowField probe(ny, nx);
    for (int ty = 0; ty < ny; ++ty) {
        for (int tx = 0; tx < nx; ++tx) {
            const f32 pred_dx_raw = flow_cur_to_ref.dx(ty, tx) - flow_prev_to_ref.dx(ty, tx);
            const f32 pred_dy_raw = flow_cur_to_ref.dy(ty, tx) - flow_prev_to_ref.dy(ty, tx);
            probe.dx(ty, tx) = pred_dx_raw * guide_sx;
            probe.dy(ty, tx) = pred_dy_raw * guide_sy;
        }
    }

    // Same ambiguity ratio clamped_ambiguity_ratio() falls back to -- this
    // corroboration match has no Config of its own to read one from.
    // fallback_on_ambiguous=false: this probe wants the raw best-effort
    // match to compare against the prediction, not align.cpp's own
    // seed-preserving fallback -- an ambiguous read is handled by this
    // function's own caller-side check instead (see the ambiguous flag read
    // below).
    block_match_level_direct_460(prev_grey, cur_grey, guide_ts, guide_radius, probe,
                                 /*l1=*/true, /*ambiguity_ratio=*/1.10f,
                                 /*write_ambiguity=*/true,
                                 /*fallback_on_ambiguous=*/false, num_threads);

    out.assign((size_t)ny * nx, 0u);
    const f32 th_sq = closure_threshold_px * closure_threshold_px;
    for (int ty = 0; ty < ny; ++ty) {
        for (int tx = 0; tx < nx; ++tx) {
            const f32 pred_dx_raw = flow_cur_to_ref.dx(ty, tx) - flow_prev_to_ref.dx(ty, tx);
            const f32 pred_dy_raw = flow_cur_to_ref.dy(ty, tx) - flow_prev_to_ref.dy(ty, tx);
            const f32 actual_dx_raw = probe.dx(ty, tx) / guide_sx;
            const f32 actual_dy_raw = probe.dy(ty, tx) / guide_sy;
            const f32 res_dx = actual_dx_raw - pred_dx_raw;
            const f32 res_dy = actual_dy_raw - pred_dy_raw;
            const f32 closure_sq = res_dx * res_dx + res_dy * res_dy;
            const size_t idx = (size_t)ty * nx + tx;
            // An ambiguous corroboration read (near-tied minima) neither
            // confirms nor refutes the seed -- do not flag on inconclusive
            // evidence.
            const bool ambiguous = idx < probe.match_ambiguous.size() &&
                                   probe.match_ambiguous[idx] != 0u;
            out[idx] = (!ambiguous && closure_sq > th_sq) ? 1u : 0u;
        }
    }
    return out;
}

} // namespace hhsr
