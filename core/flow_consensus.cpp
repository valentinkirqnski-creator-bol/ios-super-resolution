#include "flow_consensus.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hhsr {

namespace {

inline f32 sample_grey_clamped(const Image& g, int y, int x) {
    if (y < 0) y = 0; else if (y >= g.h) y = g.h - 1;
    if (x < 0) x = 0; else if (x >= g.w) x = g.w - 1;
    return g.data[(size_t)y * g.w + x];
}

// SSD of a tile window of `a` (top-left ay,ax) against `b` shifted by (sx,sy),
// both greys single channel. Window is [0,ts) x [0,ts) from the tile origin.
inline f32 tile_ssd(const Image& a, const Image& b, int ay, int ax,
                    int sx, int sy, int ts) {
    f32 acc = 0.f;
    for (int yy = 0; yy < ts; ++yy) {
        const int gy = ay + yy;
        for (int xx = 0; xx < ts; ++xx) {
            const int gx = ax + xx;
            const f32 av = sample_grey_clamped(a, gy, gx);
            const f32 bv = sample_grey_clamped(b, gy + sy, gx + sx);
            const f32 d = av - bv;
            acc += d * d;
        }
    }
    return acc;
}

// Variance of a's tile window -- a flat window carries no matchable texture.
inline f32 tile_variance(const Image& a, int ay, int ax, int ts) {
    f32 s = 0.f, s2 = 0.f;
    const f32 inv = 1.f / (f32)(ts * ts);
    for (int yy = 0; yy < ts; ++yy)
        for (int xx = 0; xx < ts; ++xx) {
            const f32 v = sample_grey_clamped(a, ay + yy, ax + xx);
            s += v; s2 += v * v;
        }
    const f32 m = s * inv;
    return std::max(s2 * inv - m * m, 0.f);
}

} // namespace

CoarseFlow coarse_block_match(const Image& a_grey, const Image& b_grey,
                              int tile_size, int radius,
                              f32 seed_x, f32 seed_y, int num_threads) {
    CoarseFlow out;
    if (a_grey.h <= 0 || a_grey.w <= 0 || tile_size <= 0) return out;
    const int ny = a_grey.h / tile_size;
    const int nx = a_grey.w / tile_size;
    if (ny <= 0 || nx <= 0) return out;
    out.flow = FlowField(ny, nx);
    out.valid.assign((size_t)ny * nx, 0u);

    const int sx0 = (int)std::lround(seed_x);
    const int sy0 = (int)std::lround(seed_y);
    // A near-flat window (relative to the pair's dynamic range) cannot be
    // matched; the threshold is a small fraction of the global variance so it
    // adapts to exposure instead of being a fixed grey level.
    f32 gmean = 0.f;
    for (f32 v : a_grey.data) gmean += v;
    gmean /= (f32)std::max<size_t>(1, a_grey.data.size());
    f32 gvar = 0.f;
    for (f32 v : a_grey.data) { const f32 d = v - gmean; gvar += d * d; }
    gvar /= (f32)std::max<size_t>(1, a_grey.data.size());
    const f32 var_floor = std::max(1.0e-8f, 0.02f * gvar);

    parallel_rows(ny, num_threads, [&](int ty) {
        for (int tx = 0; tx < nx; ++tx) {
            const int ay = ty * tile_size;
            const int ax = tx * tile_size;
            if (tile_variance(a_grey, ay, ax, tile_size) < var_floor)
                continue;  // stays invalid (valid[]=0)

            // Integer search over the window around the seed. Track the best and
            // the best NON-adjacent competitor so an ambiguous (repeated-pattern)
            // match can be flagged.
            f32 best = std::numeric_limits<f32>::infinity();
            f32 second = std::numeric_limits<f32>::infinity();
            int bsx = sx0, bsy = sy0;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int sx = sx0 + dx, sy = sy0 + dy;
                    const f32 c = tile_ssd(a_grey, b_grey, ay, ax, sx, sy, tile_size);
                    if (c < best) {
                        // Demote the old best to "second" only if it is far from
                        // the new best, so the parabola's own neighbours do not
                        // count as a competing minimum.
                        if (std::abs(sx - bsx) > 1 || std::abs(sy - bsy) > 1)
                            second = std::min(second, best);
                        best = c; bsx = sx; bsy = sy;
                    } else if ((std::abs(sx - bsx) > 1 || std::abs(sy - bsy) > 1)) {
                        second = std::min(second, c);
                    }
                }
            }
            if (!std::isfinite(best)) continue;

            // Parabolic sub-pixel refine in x and y from the three SSDs around
            // the integer minimum.
            const f32 cxm = tile_ssd(a_grey, b_grey, ay, ax, bsx - 1, bsy, tile_size);
            const f32 cxp = tile_ssd(a_grey, b_grey, ay, ax, bsx + 1, bsy, tile_size);
            const f32 cym = tile_ssd(a_grey, b_grey, ay, ax, bsx, bsy - 1, tile_size);
            const f32 cyp = tile_ssd(a_grey, b_grey, ay, ax, bsx, bsy + 1, tile_size);
            f32 fx = (f32)bsx, fy = (f32)bsy;
            const f32 dxden = (cxm - 2.f * best + cxp);
            if (std::abs(dxden) > 1e-12f) {
                f32 off = 0.5f * (cxm - cxp) / dxden;
                if (off > 1.f) off = 1.f; else if (off < -1.f) off = -1.f;
                fx += off;
            }
            const f32 dyden = (cym - 2.f * best + cyp);
            if (std::abs(dyden) > 1e-12f) {
                f32 off = 0.5f * (cym - cyp) / dyden;
                if (off > 1.f) off = 1.f; else if (off < -1.f) off = -1.f;
                fy += off;
            }
            out.flow.dx(ty, tx) = fx;
            out.flow.dy(ty, tx) = fy;
            // Confident when the best minimum is clearly below its competitor.
            // (second == inf, i.e. no competitor found, is maximally confident.)
            const bool unique = !(second < 1.25f * best);
            out.valid[(size_t)ty * nx + tx] = unique ? 1u : 0u;
        }
    });
    return out;
}

TrajectoryConsensus solve_trajectory_consensus(
    const std::vector<CoarseFlow>& direct,
    const std::vector<CoarseFlow>& chain,
    int ref_index, f32 robust_c, int iters, int num_threads) {
    const int n = (int)direct.size();
    TrajectoryConsensus res;
    std::vector<FlowField>& out = res.flow;
    if (n <= 0) return res;
    // Grid from the first populated measurement.
    int ny = 0, nx = 0;
    for (const auto& d : direct) if (d.flow.ny > 0) { ny = d.flow.ny; nx = d.flow.nx; break; }
    if (ny <= 0 || nx <= 0) return res;
    if (ref_index < 0 || ref_index >= n) ref_index = 0;

    out.assign((size_t)n, FlowField(ny, nx));
    res.trusted.assign((size_t)n, std::vector<uint8_t>((size_t)ny * nx, 0u));
    const int K = n - 1;                 // unknowns: every frame but the reference
    if (K <= 0) return res;
    auto uidx = [ref_index](int f) { return f < ref_index ? f : f - 1; };
    const f32 c2 = std::max(1e-6f, robust_c * robust_c);
    const int it = std::max(1, iters);

    parallel_rows(ny, num_threads, [&](int ty) {
        // Per-tile scratch reused across the row.
        std::vector<f32> ux(K), uy(K);
        std::vector<f32> A((size_t)K * K), bx(K), by(K);
        for (int tx = 0; tx < nx; ++tx) {
            const size_t ti = (size_t)ty * nx + tx;

            // Gather this tile's measurements.
            // Direct: unknown f -> value, valid.
            // Chain:  pair (f, f+1) -> value, valid.
            // Seed the unknowns from the direct measurement where present.
            for (int f = 0; f < n; ++f) {
                if (f == ref_index) continue;
                const int ui = uidx(f);
                if (f < (int)direct.size() && !direct[f].valid.empty() &&
                    direct[f].valid[ti]) {
                    ux[ui] = direct[f].flow.dx(ty, tx);
                    uy[ui] = direct[f].flow.dy(ty, tx);
                } else {
                    ux[ui] = 0.f; uy[ui] = 0.f;
                }
            }

            for (int iter = 0; iter < it; ++iter) {
                std::fill(A.begin(), A.end(), 0.f);
                std::fill(bx.begin(), bx.end(), 0.f);
                std::fill(by.begin(), by.end(), 0.f);
                // Tiny Tikhonov term keeps A invertible for unconstrained tiles
                // (the solution then stays at the seed / zero).
                for (int i = 0; i < K; ++i) A[(size_t)i * K + i] += 1e-3f;

                // Iteration 0 solves a plain (unweighted) least squares; only
                // from that balanced solution do the later iterations reweight.
                // Seeding the weights from the raw direct measurements instead
                // would give a blunder residual 0 and full weight on iter 0 --
                // the outlier would then look like truth and the honest chain
                // constraints like the outliers. The uniform first pass is what
                // lets the redundancy out-vote a single bad measurement.
                const bool uniform = (iter == 0);

                // Direct equations: u_f = D_f.
                for (int f = 0; f < n; ++f) {
                    if (f == ref_index) continue;
                    if (f >= (int)direct.size() || direct[f].valid.empty() ||
                        !direct[f].valid[ti]) continue;
                    const int ui = uidx(f);
                    const f32 dxv = direct[f].flow.dx(ty, tx);
                    const f32 dyv = direct[f].flow.dy(ty, tx);
                    const f32 rx = ux[ui] - dxv, ry = uy[ui] - dyv;
                    const f32 w = uniform ? 1.f : c2 / (c2 + rx * rx + ry * ry);  // Cauchy
                    A[(size_t)ui * K + ui] += w;
                    bx[ui] += w * dxv;
                    by[ui] += w * dyv;
                }
                // Chain equations: u_{f+1} - u_f = C_f (ref pinned to 0).
                for (int f = 0; f + 1 < n; ++f) {
                    if (f >= (int)chain.size() || chain[f].valid.empty() ||
                        !chain[f].valid[ti]) continue;
                    const f32 cxv = chain[f].flow.dx(ty, tx);
                    const f32 cyv = chain[f].flow.dy(ty, tx);
                    const bool a_ref = (f == ref_index);
                    const bool b_ref = (f + 1 == ref_index);
                    const f32 ua_x = a_ref ? 0.f : ux[uidx(f)];
                    const f32 ua_y = a_ref ? 0.f : uy[uidx(f)];
                    const f32 ub_x = b_ref ? 0.f : ux[uidx(f + 1)];
                    const f32 ub_y = b_ref ? 0.f : uy[uidx(f + 1)];
                    const f32 rx = (ub_x - ua_x) - cxv, ry = (ub_y - ua_y) - cyv;
                    const f32 w = uniform ? 1.f : c2 / (c2 + rx * rx + ry * ry);
                    // Accumulate w * c c^T and w * c * rhs, c in {+1 at b, -1 at a}.
                    if (!a_ref && !b_ref) {
                        const int ia = uidx(f), ib = uidx(f + 1);
                        A[(size_t)ia * K + ia] += w;
                        A[(size_t)ib * K + ib] += w;
                        A[(size_t)ia * K + ib] -= w;
                        A[(size_t)ib * K + ia] -= w;
                        bx[ia] -= w * cxv; bx[ib] += w * cxv;
                        by[ia] -= w * cyv; by[ib] += w * cyv;
                    } else if (a_ref && !b_ref) {   // u_b = C_f
                        const int ib = uidx(f + 1);
                        A[(size_t)ib * K + ib] += w;
                        bx[ib] += w * cxv; by[ib] += w * cyv;
                    } else if (!a_ref && b_ref) {   // -u_a = C_f
                        const int ia = uidx(f);
                        A[(size_t)ia * K + ia] += w;
                        bx[ia] -= w * cxv; by[ia] -= w * cyv;
                    }
                }

                // Solve A u = b for both components (A is SPD, small). In-place
                // Gaussian elimination with partial pivoting on an augmented
                // [A | bx by] -- K is the burst size, so this is a few hundred
                // flops per tile.
                // Build augmented copy.
                // (Reuse A as the working matrix; keep bx/by as RHS columns.)
                bool ok = true;
                for (int col = 0; col < K && ok; ++col) {
                    // Pivot.
                    int piv = col; f32 pv = std::abs(A[(size_t)col * K + col]);
                    for (int r = col + 1; r < K; ++r) {
                        const f32 a = std::abs(A[(size_t)r * K + col]);
                        if (a > pv) { pv = a; piv = r; }
                    }
                    if (pv < 1e-20f) { ok = false; break; }
                    if (piv != col) {
                        for (int c = 0; c < K; ++c)
                            std::swap(A[(size_t)col * K + c], A[(size_t)piv * K + c]);
                        std::swap(bx[col], bx[piv]);
                        std::swap(by[col], by[piv]);
                    }
                    const f32 inv = 1.f / A[(size_t)col * K + col];
                    for (int r = col + 1; r < K; ++r) {
                        const f32 factor = A[(size_t)r * K + col] * inv;
                        if (factor == 0.f) continue;
                        for (int c = col; c < K; ++c)
                            A[(size_t)r * K + c] -= factor * A[(size_t)col * K + c];
                        bx[r] -= factor * bx[col];
                        by[r] -= factor * by[col];
                    }
                }
                if (ok) {
                    for (int r = K - 1; r >= 0; --r) {
                        f32 sx = bx[r], sy = by[r];
                        for (int c = r + 1; c < K; ++c) {
                            sx -= A[(size_t)r * K + c] * ux[c];
                            sy -= A[(size_t)r * K + c] * uy[c];
                        }
                        const f32 d = A[(size_t)r * K + r];
                        if (std::abs(d) < 1e-20f) { ok = false; break; }
                        ux[r] = sx / d;
                        uy[r] = sy / d;
                    }
                }
                if (!ok) break;  // keep the seed for this tile
            }

            // Write the consensus ref->f displacement per frame, plus a trusted
            // flag: 1 only where the frame's tile is backed by >= 2 valid
            // measurements (its direct, and/or the two chains touching it), so
            // the trajectory could be cross-checked. A lone direct measurement
            // is NOT trusted -- there is nothing to out-vote a blunder with, so
            // the caller keeps the refined align flow there.
            auto direct_valid = [&](int f) {
                return f != ref_index && f < (int)direct.size() &&
                       !direct[f].valid.empty() && direct[f].valid[ti];
            };
            auto chain_valid = [&](int f) {  // pair (f, f+1)
                return f >= 0 && f + 1 < n && f < (int)chain.size() &&
                       !chain[f].valid.empty() && chain[f].valid[ti];
            };
            for (int f = 0; f < n; ++f) {
                if (f == ref_index) { out[f].dx(ty, tx) = 0.f; out[f].dy(ty, tx) = 0.f; continue; }
                const int ui = uidx(f);
                out[f].dx(ty, tx) = ux[ui];
                out[f].dy(ty, tx) = uy[ui];
                int touching = (direct_valid(f) ? 1 : 0) +
                               (chain_valid(f - 1) ? 1 : 0) +
                               (chain_valid(f) ? 1 : 0);
                res.trusted[f][ti] = (touching >= 2) ? 1u : 0u;
            }
        }
    });
    return res;
}

int apply_consensus_correction(FlowField& align_flow, const FlowField& consensus,
                               const std::vector<uint8_t>& trusted, f32 reject_px) {
    const int ay = align_flow.ny, ax = align_flow.nx;
    const int cy = consensus.ny, cx = consensus.nx;
    if (ay <= 0 || ax <= 0 || cy <= 0 || cx <= 0 ||
        trusted.size() != (size_t)cy * cx)
        return 0;
    // The consensus grid (the coarse pre-pass) and the align grid (the finest
    // alignment level, padded) need not match exactly, so map each align tile to
    // its covering consensus tile by proportional position. Nearest is fine: the
    // consensus is a coarse per-tile field and the reject_px gate is forgiving.
    const f32 sy = (f32)cy / (f32)ay, sx = (f32)cx / (f32)ax;
    const f32 r2 = reject_px * reject_px;
    int replaced = 0;
    for (int ty = 0; ty < ay; ++ty) {
        int gy = (int)((ty + 0.5f) * sy);
        if (gy < 0) gy = 0; else if (gy >= cy) gy = cy - 1;
        for (int tx = 0; tx < ax; ++tx) {
            int gx = (int)((tx + 0.5f) * sx);
            if (gx < 0) gx = 0; else if (gx >= cx) gx = cx - 1;
            if (!trusted[(size_t)gy * cx + gx]) continue;
            const f32 cdx = consensus.dx(gy, gx), cdy = consensus.dy(gy, gx);
            const f32 ddx = align_flow.dx(ty, tx) - cdx;
            const f32 ddy = align_flow.dy(ty, tx) - cdy;
            if (ddx * ddx + ddy * ddy > r2) {
                align_flow.dx(ty, tx) = cdx;
                align_flow.dy(ty, tx) = cdy;
                ++replaced;
            }
        }
    }
    return replaced;
}

} // namespace hhsr
