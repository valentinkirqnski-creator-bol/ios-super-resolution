// feat_metric.cpp — feature-metric alignment refinement (Config::feat_metric_refine_enabled).
//
// A tiny learned CNN (core/feat_metric_shared.h + feat_metric_weights.h) maps the
// reference and comparison grey to FM_NF feature planes; a per-tile AFFINE
// Lucas-Kanade (6-DOF: translation + 2x2 linear part) then refines the flow ON
// THOSE FEATURES, seeded from the finalized tile/affine flow. The scene-aware
// features disambiguate handheld / parallax / repetitive content where raw
// intensity LK cannot (measured +16..37% endpoint over classical on held-out
// real bursts; the affine model keeps rotation from regressing). Output is a
// per-pixel dense field (FlowField::dense), same contract as dense_lk_refine —
// merge and robustness sample it via sample_dense with no other change.
//
// Runs on the HOST (like dense_lk_refine) even on device, so it is memory-guarded
// and downsample-aware at the call site (pipeline_paths.cpp). This file is
// resolution-agnostic: it uses ref_grey/comp_grey dims vs raw_h/raw_w for the
// grey->raw scale, exactly as dense_lk_refine does.
#include "types.h"
#include "stages.h"
#include "parallel.h"
#include "feat_metric_shared.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace hhsr {

// Must match the training constants (scratchpad/step2_affine.py).
static const int   FM_TILE   = 16;     // grey px; the net was trained at this tile
static const float FM_LAM_T  = 1e-3f;  // translation Levenberg damping
static const float FM_LAM_A  = 3e-2f;  // linear-part damping (weaker-conditioned)
static const float FM_STEP_T = 2.0f;   // per-iter translation clamp (px)
static const float FM_STEP_A = 0.15f;  // per-iter linear-part clamp
static const float FM_ACCEPT_MARGIN = 0.03f; // per-tile SSD improvement to keep affine

// ---- feature extraction: three 3x3 convs, full planes (banding handled by the
// downsample at the call site). Output planar [FM_NF][h][w]. -----------------
static void fm_features(const Image& grey, std::vector<f32>& out, int nt) {
    const int w = grey.w, h = grey.h;
    const size_t N = (size_t)h * w;
    FMPlane gp{grey.data.data(), w, h};
    std::vector<f32> a((size_t)FM_H1 * N), b((size_t)FM_H2 * N);
    parallel_rows(h, nt, [&](int y) {
        float t[FM_H1];
        for (int x = 0; x < w; ++x) {
            fm_conv3x3_at(gp, 1, FM_H1, FM_C1_W, FM_C1_B, true, w, h, x, y, t);
            for (int o = 0; o < FM_H1; ++o) a[((size_t)o * h + y) * w + x] = t[o];
        }
    });
    FMPlane ap{a.data(), w, h};
    parallel_rows(h, nt, [&](int y) {
        float t[FM_H2];
        for (int x = 0; x < w; ++x) {
            fm_conv3x3_at(ap, FM_H1, FM_H2, FM_C2_W, FM_C2_B, true, w, h, x, y, t);
            for (int o = 0; o < FM_H2; ++o) b[((size_t)o * h + y) * w + x] = t[o];
        }
    });
    FMPlane bp{b.data(), w, h};
    out.assign((size_t)FM_NF * N, 0.f);
    parallel_rows(h, nt, [&](int y) {
        float t[FM_NF];
        for (int x = 0; x < w; ++x) {
            fm_conv3x3_at(bp, FM_H2, FM_NF, FM_C3_W, FM_C3_B, false, w, h, x, y, t);
            for (int o = 0; o < FM_NF; ++o) out[((size_t)o * h + y) * w + x] = t[o];
        }
    });
}

static inline f32 fm_bilin(const f32* plane, int w, int h, f32 x, f32 y) {
    x = std::min(std::max(x, 0.f), (f32)(w - 1));
    y = std::min(std::max(y, 0.f), (f32)(h - 1));
    const int x0 = (int)x, y0 = (int)y;
    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    const f32 fx = x - (f32)x0, fy = y - (f32)y0;
    const f32 t = plane[(size_t)y0 * w + x0] + (plane[(size_t)y0 * w + x1] - plane[(size_t)y0 * w + x0]) * fx;
    const f32 bb = plane[(size_t)y1 * w + x0] + (plane[(size_t)y1 * w + x1] - plane[(size_t)y1 * w + x0]) * fx;
    return t + (bb - t) * fy;
}

// Solve a 6x6 SPD system H x = g by Cholesky (H damped -> always PD). Returns
// false if not decomposable (then x is left zero).
static bool fm_solve6(const f32 H[36], const f32 g[6], f32 x[6]) {
    f32 L[36] = {0};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            f32 s = H[i * 6 + j];
            for (int k = 0; k < j; ++k) s -= L[i * 6 + k] * L[j * 6 + k];
            if (i == j) {
                if (s <= 1e-12f) return false;
                L[i * 6 + j] = std::sqrt(s);
            } else {
                L[i * 6 + j] = s / L[j * 6 + j];
            }
        }
    }
    f32 yv[6];
    for (int i = 0; i < 6; ++i) {
        f32 s = g[i];
        for (int k = 0; k < i; ++k) s -= L[i * 6 + k] * yv[k];
        yv[i] = s / L[i * 6 + i];
    }
    for (int i = 5; i >= 0; --i) {
        f32 s = yv[i];
        for (int k = i + 1; k < 6; ++k) s -= L[k * 6 + i] * x[k];
        x[i] = s / L[i * 6 + i];
    }
    return true;
}

void feat_metric_refine(FlowField& flow, const Image& ref_grey,
                        const Image& comp_grey, const Config& cfg,
                        int raw_tile_size, int raw_h, int raw_w) {
    if (!cfg.feat_metric_refine_enabled) return;
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty() || raw_tile_size <= 0) return;
    if (ref_grey.data.size() != (size_t)ref_grey.h * ref_grey.w * ref_grey.c ||
        comp_grey.data.size() != (size_t)comp_grey.h * comp_grey.w * comp_grey.c)
        return;
    const int h = ref_grey.h, w = ref_grey.w;
    if (h < 8 || w < 8 || comp_grey.h != h || comp_grey.w != w) return;
    const int nt = cfg.num_threads;
    const size_t N = (size_t)h * w;
    const f32 sx = (w > 0) ? (f32)raw_w / (f32)w : 1.f;
    const f32 sy = (h > 0) ? (f32)raw_h / (f32)h : 1.f;
    const int iters = std::max(1, cfg.feat_metric_iters);
    const int tile = FM_TILE;
    const int ny = (h + tile - 1) / tile, nx = (w + tile - 1) / tile;
    const f32 half = 0.5f * (f32)tile;

    // Feature planes.
    std::vector<f32> Fr, Fc;
    fm_features(ref_grey, Fr, nt);
    fm_features(comp_grey, Fc, nt);

    // Per-pixel intra-tile normalized coords u,v in [-1,1] (precomputed by row/col).
    std::vector<f32> uu(w), vv(h);
    for (int x = 0; x < w; ++x) { int tx = x / tile; uu[x] = ((f32)x - ((f32)tx * tile + half - 0.5f)) / half; }
    for (int y = 0; y < h; ++y) { int ty = y / tile; vv[y] = ((f32)y - ((f32)ty * tile + half - 0.5f)) / half; }

    // Seed per-tile affine params p[ny*nx*6] = (dx,dy,a,b,c,d); a..d = 0.
    std::vector<f32> p((size_t)ny * nx * 6, 0.f);
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            const f32 gy = std::min((f32)h - 1.f, (ty + 0.5f) * tile);
            const f32 gx = std::min((f32)w - 1.f, (tx + 0.5f) * tile);
            const f32 raw_y = 0.5f * (sy - 1.f) + sy * gy;
            const f32 raw_x = 0.5f * (sx - 1.f) + sx * gx;
            f32 drx, dry;
            flow.sample_affine(raw_y, raw_x, raw_tile_size, drx, dry);
            const size_t t = ((size_t)ty * nx + tx) * 6;
            p[t + 0] = drx / sx;
            p[t + 1] = dry / sy;
        }

    // The seeded per-pixel dense (for the accept gate): tile-constant translation.
    auto disp_at = [&](const f32* pp, int x, int y, f32& dx, f32& dy) {
        const size_t t = ((size_t)(y / tile) * nx + (x / tile)) * 6;
        const f32 u = uu[x], v = vv[y];
        dx = pp[t + 0] + pp[t + 2] * u + pp[t + 3] * v;
        dy = pp[t + 1] + pp[t + 4] * u + pp[t + 5] * v;
    };
    std::vector<f32> seed(ny * (size_t)nx * 2);
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            const size_t t = ((size_t)ty * nx + tx);
            seed[t * 2 + 0] = p[t * 6 + 0];
            seed[t * 2 + 1] = p[t * 6 + 1];
        }

    std::vector<f32> Fcw((size_t)FM_NF * N);
    for (int it = 0; it < iters; ++it) {
        // Warp all comp feature channels by the current per-pixel affine disp.
        parallel_rows(h, nt, [&](int y) {
            for (int x = 0; x < w; ++x) {
                f32 dx, dy; disp_at(p.data(), x, y, dx, dy);
                const f32 wx = (f32)x + dx, wy = (f32)y + dy;
                for (int c = 0; c < FM_NF; ++c)
                    Fcw[((size_t)c * h + y) * w + x] = fm_bilin(Fc.data() + (size_t)c * N, w, h, wx, wy);
            }
        });
        // Per-tile normal equations over the tile's pixels and all channels.
        parallel_rows(ny, nt, [&](int ty) {
            for (int tx = 0; tx < nx; ++tx) {
                f32 H[36] = {0}, g[6] = {0};
                const int y0 = ty * tile, y1 = std::min(y0 + tile, h);
                const int x0 = tx * tile, x1 = std::min(x0 + tile, w);
                for (int y = y0; y < y1; ++y) {
                    // Central difference with ZERO at the image border, matching the
                    // training reference (torch grads() leaves row/col 0 and N-1 at 0).
                    const bool yb = (y == 0 || y == h - 1);
                    const int ym = y - 1, yp = y + 1;
                    for (int x = x0; x < x1; ++x) {
                        const bool xb = (x == 0 || x == w - 1);
                        const int xm = x - 1, xp = x + 1;
                        const f32 u = uu[x], v = vv[y];
                        for (int c = 0; c < FM_NF; ++c) {
                            const f32* P = Fcw.data() + (size_t)c * N;
                            const f32 gx = xb ? 0.f : 0.5f * (P[(size_t)y * w + xp] - P[(size_t)y * w + xm]);
                            const f32 gy = yb ? 0.f : 0.5f * (P[(size_t)yp * w + x] - P[(size_t)ym * w + x]);
                            const f32 r = P[(size_t)y * w + x] - Fr[((size_t)c * h + y) * w + x];
                            const f32 J[6] = {gx, gy, gx * u, gx * v, gy * u, gy * v};
                            for (int i = 0; i < 6; ++i) {
                                g[i] -= J[i] * r;
                                for (int j = i; j < 6; ++j) H[i * 6 + j] += J[i] * J[j];
                            }
                        }
                    }
                }
                for (int i = 0; i < 6; ++i) for (int j = i + 1; j < 6; ++j) H[j * 6 + i] = H[i * 6 + j];
                const f32 d6[6] = {FM_LAM_T, FM_LAM_T, FM_LAM_A, FM_LAM_A, FM_LAM_A, FM_LAM_A};
                for (int i = 0; i < 6; ++i) H[i * 6 + i] += d6[i];
                f32 step[6] = {0};
                if (fm_solve6(H, g, step)) {
                    for (int i = 0; i < 2; ++i) step[i] = std::min(std::max(step[i], -FM_STEP_T), FM_STEP_T);
                    for (int i = 2; i < 6; ++i) step[i] = std::min(std::max(step[i], -FM_STEP_A), FM_STEP_A);
                    const size_t t = ((size_t)ty * nx + tx) * 6;
                    for (int i = 0; i < 6; ++i) p[t + i] += step[i];
                }
            }
        });
    }

    // Per-tile SSD accept on the GREY: keep the affine result only where it
    // improves the photometric match over the seed, else revert that tile to its
    // seed translation. Protects clean/flat/low-light tiles (where the 6 DOF would
    // overfit noise) and bounds the stage so it can never make the merge worse.
    // Disabled (feat_metric_accept_gate=false) only for cross-language parity.
    const bool gate = cfg.feat_metric_accept_gate;
    std::vector<f32> dense(N * 2);
    parallel_rows(ny, nt, [&](int ty) {
        for (int tx = 0; tx < nx; ++tx) {
            const int y0 = ty * tile, y1 = std::min(y0 + tile, h);
            const int x0 = tx * tile, x1 = std::min(x0 + tile, w);
            const size_t ti = (size_t)ty * nx + tx;
            bool keep = true;
            if (gate) {
                f32 ss_seed = 0.f, ss_feat = 0.f;
                for (int y = y0; y < y1; ++y)
                    for (int x = x0; x < x1; ++x) {
                        const f32 rv = ref_grey.at(y, x);
                        f32 dx, dy; disp_at(p.data(), x, y, dx, dy);
                        const f32 ef = fm_bilin(comp_grey.data.data(), w, h, (f32)x + dx, (f32)y + dy) - rv;
                        const f32 sx0 = seed[ti * 2 + 0], sy0 = seed[ti * 2 + 1];
                        const f32 es = fm_bilin(comp_grey.data.data(), w, h, (f32)x + sx0, (f32)y + sy0) - rv;
                        ss_seed += es * es; ss_feat += ef * ef;
                    }
                keep = (ss_feat < (1.f - FM_ACCEPT_MARGIN) * ss_seed);
            }
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x) {
                    const size_t i = (size_t)y * w + x;
                    if (keep) { f32 dx, dy; disp_at(p.data(), x, y, dx, dy); dense[i * 2] = dx; dense[i * 2 + 1] = dy; }
                    else { dense[i * 2] = seed[ti * 2 + 0]; dense[i * 2 + 1] = seed[ti * 2 + 1]; }
                }
        }
    });

    flow.dense = std::move(dense);
    flow.dense_h = h; flow.dense_w = w;
    flow.dense_sx = sx; flow.dense_sy = sy;
}

} // namespace hhsr
