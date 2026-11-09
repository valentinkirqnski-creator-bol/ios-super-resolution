#include "ref_demosaic.h"
#include "parallel.h"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace hhsr {

namespace {
Image g_ref_demo;

// Mirror about the edge pixel: keeps the Bayer parity of the index.
inline int mir(int v, int n) {
    if (v < 0) v = -v;
    if (v >= n) v = 2 * (n - 1) - v;
    return std::min(std::max(v, 0), n - 1);
}
}  // namespace

void merge_set_ref_demosaic(Image&& rgb) { g_ref_demo = std::move(rgb); }
void merge_clear_ref_demosaic() { g_ref_demo = Image(); }
const Image* merge_ref_demosaic() {
    return (g_ref_demo.h > 0 && g_ref_demo.w > 0 && g_ref_demo.c == 3 &&
            g_ref_demo.data.size() == (size_t)g_ref_demo.h * g_ref_demo.w * 3)
               ? &g_ref_demo : nullptr;
}

Image demosaic_ref_rgb(const Image& raw, const Config& cfg) {
    const int H = raw.h, W = raw.w, nt = cfg.num_threads;
    if (!cfg.bayer_mode || raw.c != 1 || H < 8 || W < 8 || raw.data.size() < (size_t)H * W)
        return Image();
    auto ch = [&](int y, int x) { return (int)cfg.cfa.p[y & 1][x & 1]; };
    // Colour-difference interpolation wants white-balanced data. The loader already
    // applied wb[c]/wb[G] when raw_prewhitened; otherwise apply it here and undo it
    // on the way out, so the result stays in the merge's units.
    float g[3] = {1.f, 1.f, 1.f};
    if (!cfg.raw_prewhitened && cfg.white_balance[1] > 0.f)
        for (int c = 0; c < 3; ++c)
            g[c] = cfg.white_balance[c] > 0.f ? cfg.white_balance[c] / cfg.white_balance[1] : 1.f;
    std::vector<float> X((size_t)H * W);
    parallel_rows(H, nt, [&](int y) {
        for (int x = 0; x < W; ++x) X[(size_t)y * W + x] = std::max(0.f, raw.at(y, x)) * g[ch(y, x)];
    });
    auto at = [&](const std::vector<float>& P, int y, int x) {
        return P[(size_t)mir(y, H) * W + mir(x, W)];
    };

    // 1. Direction discrimination (every pixel): Hamilton-Adams gradients, box-summed
    //    3x3; VH = V / (V + H) -> 1 = strong vertical change, interpolate horizontally.
    std::vector<float> gv((size_t)H * W), gh((size_t)H * W), VH((size_t)H * W);
    parallel_rows(H, nt, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const float c = at(X, y, x);
            gv[(size_t)y * W + x] = std::fabs(at(X, y - 1, x) - at(X, y + 1, x)) +
                                    std::fabs(2.f * c - at(X, y - 2, x) - at(X, y + 2, x));
            gh[(size_t)y * W + x] = std::fabs(at(X, y, x - 1) - at(X, y, x + 1)) +
                                    std::fabs(2.f * c - at(X, y, x - 2) - at(X, y, x + 2));
        }
    });
    parallel_rows(H, nt, [&](int y) {
        for (int x = 0; x < W; ++x) {
            float v = 0.f, h = 0.f;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j) { v += at(gv, y + i, x + j); h += at(gh, y + i, x + j); }
            VH[(size_t)y * W + x] = v / std::max(1e-20f, v + h);
        }
    });

    // 2. Green everywhere: Hamilton-Adams directional estimates at R/B sites, softly
    //    blended by VH, clamped to the local green range (no overshoot halos).
    std::vector<float> G((size_t)H * W);
    parallel_rows(H, nt, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const size_t o = (size_t)y * W + x;
            if (ch(y, x) == 1) { G[o] = X[o]; continue; }
            const float c = X[o];
            const float n = at(X, y - 1, x), s = at(X, y + 1, x), w = at(X, y, x - 1), e = at(X, y, x + 1);
            const float ev = 0.5f * (n + s) + 0.25f * (2.f * c - at(X, y - 2, x) - at(X, y + 2, x));
            const float eh = 0.5f * (w + e) + 0.25f * (2.f * c - at(X, y, x - 2) - at(X, y, x + 2));
            const float vh = VH[o];
            float gval = vh * eh + (1.f - vh) * ev;
            const float lo = std::min(std::min(n, s), std::min(w, e));
            const float hi = std::max(std::max(n, s), std::max(w, e));
            G[o] = std::min(std::max(gval, lo), hi);
        }
    });

    // 3. The opposite chroma at R/B sites: diagonal colour differences, weighted by
    //    the diagonal gradients (NW-SE vs NE-SW).
    std::vector<float> R((size_t)H * W), B((size_t)H * W);
    parallel_rows(H, nt, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const size_t o = (size_t)y * W + x;
            const int c = ch(y, x);
            if (c == 1) continue;
            const float d_nw = at(X, y - 1, x - 1) - at(G, y - 1, x - 1);
            const float d_se = at(X, y + 1, x + 1) - at(G, y + 1, x + 1);
            const float d_ne = at(X, y - 1, x + 1) - at(G, y - 1, x + 1);
            const float d_sw = at(X, y + 1, x - 1) - at(G, y + 1, x - 1);
            const float gp = std::fabs(at(X, y - 1, x - 1) - at(X, y + 1, x + 1)) +
                             std::fabs(2.f * G[o] - at(G, y - 1, x - 1) - at(G, y + 1, x + 1));
            const float gq = std::fabs(at(X, y - 1, x + 1) - at(X, y + 1, x - 1)) +
                             std::fabs(2.f * G[o] - at(G, y - 1, x + 1) - at(G, y + 1, x - 1));
            const float pq = gp / std::max(1e-20f, gp + gq);   // 1 -> NW-SE changes: use NE-SW
            const float other = G[o] + pq * 0.5f * (d_ne + d_sw) + (1.f - pq) * 0.5f * (d_nw + d_se);
            if (c == 0) { R[o] = X[o]; B[o] = std::max(0.f, other); }
            else        { B[o] = X[o]; R[o] = std::max(0.f, other); }
        }
    });

    // 4. R and B at green sites: N/S and E/W neighbours are all R/B sites (full
    //    chroma after step 3); colour differences blended by VH.
    Image out(H, W, 3);
    parallel_rows(H, nt, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const size_t o = (size_t)y * W + x;
            float r, b;
            if (ch(y, x) == 1) {
                const float gn = at(G, y - 1, x), gs = at(G, y + 1, x), gw = at(G, y, x - 1), ge = at(G, y, x + 1);
                const float rv = 0.5f * ((at(R, y - 1, x) - gn) + (at(R, y + 1, x) - gs));
                const float rh = 0.5f * ((at(R, y, x - 1) - gw) + (at(R, y, x + 1) - ge));
                const float bv = 0.5f * ((at(B, y - 1, x) - gn) + (at(B, y + 1, x) - gs));
                const float bh = 0.5f * ((at(B, y, x - 1) - gw) + (at(B, y, x + 1) - ge));
                const float vh = VH[o];
                r = std::max(0.f, G[o] + vh * rh + (1.f - vh) * rv);
                b = std::max(0.f, G[o] + vh * bh + (1.f - vh) * bv);
            } else {
                r = R[o];
                b = B[o];
            }
            out.at(y, x, 0) = r / g[0];
            out.at(y, x, 1) = G[o] / g[1];
            out.at(y, x, 2) = b / g[2];
        }
    });
    return out;
}

}  // namespace hhsr
