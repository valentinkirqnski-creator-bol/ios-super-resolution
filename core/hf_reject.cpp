#include "hf_reject.h"
#include <algorithm>
#include <cmath>

namespace hhsr {

namespace {

inline int clampi(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

// Luminance-like signal from the guide. The guide is linear camera RGB (green
// already combined), so a fixed sRGB luma weighting is not meaningful here; the
// channel mean is a scale-consistent luminance proxy. Single-channel guide passes
// through.
Image guide_luma(const Image& g) {
    if (g.c == 1) return g;
    Image y(g.h, g.w, 1);
    const f32 inv = 1.f / (f32)std::max(1, g.c);
    for (int i = 0; i < g.h; ++i)
        for (int j = 0; j < g.w; ++j) {
            f32 s = 0.f;
            for (int c = 0; c < g.c; ++c) s += g.at(i, j, c);
            y.at(i, j) = s * inv;
        }
    return y;
}

// Separable 3x3 Gaussian [1 2 1]/4 each axis = [1 2 1; 2 4 2; 1 2 1]/16, edge-clamped.
Image lowpass3x3(const Image& y) {
    static const f32 k[3] = {1.f, 2.f, 1.f};
    Image tmp(y.h, y.w, 1), out(y.h, y.w, 1);
    for (int i = 0; i < y.h; ++i)             // horizontal
        for (int j = 0; j < y.w; ++j) {
            f32 s = 0.f;
            for (int dj = -1; dj <= 1; ++dj) s += k[dj + 1] * y.at(i, clampi(j + dj, y.w - 1));
            tmp.at(i, j) = s * 0.25f;
        }
    for (int i = 0; i < y.h; ++i)             // vertical
        for (int j = 0; j < y.w; ++j) {
            f32 s = 0.f;
            for (int di = -1; di <= 1; ++di) s += k[di + 1] * tmp.at(clampi(i + di, y.h - 1), j);
            out.at(i, j) = s * 0.25f;
        }
    return out;
}

// Local mean and (population) variance over a (2r+1)^2 edge-clamped window.
void local_mean_var(const Image& y, int r, Image& var_out) {
    var_out = Image(y.h, y.w, 1);
    const f32 n = (f32)((2 * r + 1) * (2 * r + 1));
    for (int i = 0; i < y.h; ++i)
        for (int j = 0; j < y.w; ++j) {
            f32 s = 0.f, s2 = 0.f;
            for (int di = -r; di <= r; ++di) {
                const int yy = clampi(i + di, y.h - 1);
                for (int dj = -r; dj <= r; ++dj) {
                    const f32 v = y.at(yy, clampi(j + dj, y.w - 1));
                    s += v; s2 += v * v;
                }
            }
            const f32 m = s / n;
            var_out.at(i, j) = std::max(0.f, s2 / n - m * m);
        }
}

}  // namespace

Image hf_variance_loss(const Image& guide, int win, f32 min_var) {
    Image L(guide.h <= 0 ? 0 : guide.h, guide.w <= 0 ? 0 : guide.w, 1);
    if (guide.h <= 0 || guide.w <= 0) return L;
    const int r = std::max(1, win / 2);
    const Image Y  = guide_luma(guide);
    const Image YL = lowpass3x3(Y);
    Image vb, va;
    local_mean_var(Y,  r, vb);
    local_mean_var(YL, r, va);
    const f32 eps = 1e-8f;
    for (int i = 0; i < guide.h; ++i)
        for (int j = 0; j < guide.w; ++j) {
            const f32 Vb = vb.at(i, j), Va = va.at(i, j);
            f32 l = (Vb < min_var) ? 0.f : (Vb - Va) / (Vb + eps);
            L.at(i, j) = std::min(1.f, std::max(0.f, l));
        }
    return L;
}

std::vector<f32> hf_flow_variation(const FlowField& flow) {
    const int ny = flow.ny, nx = flow.nx;
    std::vector<f32> D((size_t)std::max(0, ny) * (size_t)std::max(0, nx), 0.f);
    if (ny <= 0 || nx <= 0 || flow.flow.size() < (size_t)ny * nx * 2) return D;
    auto uv = [&](int ty, int tx, f32& u, f32& v) {
        const size_t o = ((size_t)ty * nx + tx) * 2;
        u = flow.flow[o + 0]; v = flow.flow[o + 1];
    };
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            f32 ui, vi; uv(ty, tx, ui, vi);
            f32 acc = 0.f; int cnt = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy) continue;
                    const int jy = ty + dy, jx = tx + dx;
                    if (jy < 0 || jy >= ny || jx < 0 || jx >= nx) continue;
                    f32 uj, vj; uv(jy, jx, uj, vj);
                    acc += std::sqrt((ui - uj) * (ui - uj) + (vi - vj) * (vi - vj));
                    ++cnt;
                }
            D[(size_t)ty * nx + tx] = cnt ? acc / (f32)cnt : 0.f;
        }
    return D;
}

Image hf_build_mask(const Image& Lmap, const std::vector<f32>& D,
                    int flow_ny, int flow_nx, int guide_tile_size, f32 tl, f32 td) {
    Image H(Lmap.h, Lmap.w, 1);
    if (Lmap.h <= 0 || Lmap.w <= 0 || guide_tile_size <= 0 ||
        flow_ny <= 0 || flow_nx <= 0 ||
        D.size() < (size_t)flow_ny * flow_nx)
        return H;  // all-zero: reject nothing
    for (int i = 0; i < Lmap.h; ++i) {
        const int ty = clampi(i / guide_tile_size, flow_ny - 1);
        for (int j = 0; j < Lmap.w; ++j) {
            if (Lmap.at(i, j) <= tl) continue;             // Test A fails
            const int tx = clampi(j / guide_tile_size, flow_nx - 1);
            if (D[(size_t)ty * flow_nx + tx] <= td) continue;  // Test B fails
            H.at(i, j) = 1.f;                              // both high -> reject
        }
    }
    return H;
}

void hf_apply_mask(Image& R, const Image& H) {
    if (R.h <= 0 || R.w <= 0 || H.h <= 0 || H.w <= 0) return;
    for (int i = 0; i < R.h; ++i) {
        const int gi = clampi((int)((int64_t)i * H.h / R.h), H.h - 1);
        for (int j = 0; j < R.w; ++j) {
            const int gj = clampi((int)((int64_t)j * H.w / R.w), H.w - 1);
            if (H.at(gi, gj) > 0.5f)
                for (int c = 0; c < R.c; ++c) R.at(i, j, c) = 0.f;
        }
    }
}

void hf_reject_apply(Image& R, const Image& guide, const FlowField& flow,
                     int guide_tile_size, const Config& cfg) {
    if (!cfg.hf_reject_enabled) return;
    const Image L = hf_variance_loss(guide, cfg.hf_reject_win, cfg.hf_reject_min_var);
    const std::vector<f32> D = hf_flow_variation(flow);
    const Image H = hf_build_mask(L, D, flow.ny, flow.nx, guide_tile_size,
                                  cfg.hf_reject_tl, cfg.hf_reject_td);
    hf_apply_mask(R, H);
}

}  // namespace hhsr
