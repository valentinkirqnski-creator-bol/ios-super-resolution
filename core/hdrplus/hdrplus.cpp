// HDR+ pipeline CPU twin. See hdrplus.h for the contract and the faithfulness
// notes. Every function reproduces the matching Halide Func from
// hdr-plus-master/src (align.cpp, merge.cpp, finish.cpp, util.cpp), including
// its integer domain, cast semantics and boundary conditions.
#include "hdrplus.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <thread>
#include <atomic>
#include <type_traits>

namespace hhsr {
namespace hdrplus {
namespace {

// Dynamic work-stealing parallel-for. Every use below splits an independent
// axis (comp frames, output rows, filter rows), so results are bit-identical to
// the serial order regardless of thread count.
template <class F>
void parallel_for(int begin, int end, F f) {
    const int n = end - begin;
    if (n <= 0) return;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    int nt = (int)std::min<unsigned>(hw, (unsigned)n);
    if (nt <= 1) { for (int i = begin; i < end; ++i) f(i); return; }
    std::atomic<int> next{begin};
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (int t = 0; t < nt; ++t)
        ts.emplace_back([&]{ int i; while ((i = next.fetch_add(1)) < end) f(i); });
    for (auto& th : ts) th.join();
}

// ---- Halide numeric semantics ------------------------------------------------
// Halide integer division/modulo are FLOOR (round toward -inf), unlike C++.
inline int fdiv(int a, int b) { int q = a / b, r = a % b; if (r != 0 && ((r < 0) != (b < 0))) --q; return q; }
inline int fmod_(int a, int b) { int r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r; }
// u16(x): truncate toward zero, keep low 16 bits (Halide cast, wraps — not sat).
inline uint16_t u16c(double x) { return (uint16_t)(int64_t)x; }
// u16_sat / u8_sat: truncate toward zero, then clamp.
inline uint16_t u16s(double x) { int64_t v = (int64_t)x; return (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v)); }
inline uint8_t  u8s (double x) { int64_t v = (int64_t)x; return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
inline int32_t  i32c(double x) { return (int32_t)x; }

// mirror_interior: reflect without duplicating the edge sample. clamp == repeat_edge.
inline int mir(int i, int n) {
    if (n <= 1) return 0;
    int p = 2 * (n - 1);
    int m = ((i % p) + p) % p;
    return m < n ? m : p - m;
}
inline int clmp(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

// ---- A simple uint16 / float 2D plane ---------------------------------------
struct U16Plane {
    int w = 0, h = 0;
    std::vector<uint16_t> d;
    U16Plane() = default;
    U16Plane(int w_, int h_) : w(w_), h(h_), d((size_t)w_ * h_, 0) {}
    inline uint16_t& at(int x, int y) { return d[(size_t)y * w + x]; }
    inline uint16_t  at(int x, int y) const { return d[(size_t)y * w + x]; }
    // mirror_interior, with an interior fast path (the common case in block
    // matching) that skips the modulo in mir().
    inline uint16_t  mget(int x, int y) const {
        if ((unsigned)x < (unsigned)w && (unsigned)y < (unsigned)h)
            return d[(size_t)y * w + x];
        return d[(size_t)mir(y, h) * w + mir(x, w)];
    }
    inline uint16_t  cget(int x, int y) const { return d[(size_t)clmp(y, h) * w + clmp(x, w)]; } // repeat_edge
};

// ---- util.cpp: downsamplers --------------------------------------------------
// box_down2: 2x2 average, /4, u16 (util.cpp:22).
U16Plane box_down2(const U16Plane& s) {
    U16Plane o(s.w / 2, s.h / 2);
    for (int y = 0; y < o.h; ++y)
        for (int x = 0; x < o.w; ++x) {
            uint32_t sum = (uint32_t)s.mget(2 * x, 2 * y) + s.mget(2 * x + 1, 2 * y) +
                           s.mget(2 * x, 2 * y + 1) + s.mget(2 * x + 1, 2 * y + 1);
            o.at(x, y) = u16c(sum / 4u);
        }
    return o;
}
// gauss_down4: 5x5 integer gauss, stride 4, /159, u16 (util.cpp:40-78).
U16Plane gauss_down4(const U16Plane& s) {
    static const int k[5][5] = {
        {2, 4, 5, 4, 2}, {4, 9, 12, 9, 4}, {5, 12, 15, 12, 5}, {4, 9, 12, 9, 4}, {2, 4, 5, 4, 2}};
    U16Plane o(s.w / 4, s.h / 4);
    for (int y = 0; y < o.h; ++y)
        for (int x = 0; x < o.w; ++x) {
            uint32_t sum = 0;
            for (int ry = -2; ry <= 2; ++ry)
                for (int rx = -2; rx <= 2; ++rx)
                    sum += (uint32_t)s.mget(4 * x + rx, 4 * y + ry) * (uint32_t)k[ry + 2][rx + 2];
            o.at(x, y) = u16c(sum / 159u);
        }
    return o;
}

// ---- align.cpp ---------------------------------------------------------------
constexpr int T_SIZE_2 = 16;      // overlap / pyramid tile size (align.h)
constexpr int DOWNSAMPLE_RATE = 4;
inline int prev_tile(int t) { return fdiv(t - 1, DOWNSAMPLE_RATE); }
inline int idx_layer(int t, int i) { return t * T_SIZE_2 / 2 + i; } // t*8 + i
inline int idx_im(int t, int i) { return t * T_SIZE_2 + i; }        // t*16 + i

struct Pyr { U16Plane l0, l1, l2; };
Pyr build_pyramid(const U16Plane& raw) {
    Pyr p;
    p.l0 = box_down2(raw);
    p.l1 = gauss_down4(p.l0);
    p.l2 = gauss_down4(p.l1);
    return p;
}

// A tile-offset array over an index range that may start at -1 (Halide reaches
// one tile below zero through prev_tile(0) = floor(-1/4) = -1).
struct AlignGrid {
    int x0 = 0, y0 = 0, nx = 0, ny = 0; // index range [x0, x0+nx), [y0, y0+ny)
    std::vector<int16_t> dx, dy;
    void alloc(int x0_, int y0_, int nx_, int ny_) {
        x0 = x0_; y0 = y0_; nx = nx_; ny = ny_;
        dx.assign((size_t)nx * ny, 0); dy.assign((size_t)nx * ny, 0);
    }
    inline int idx(int tx, int ty) const {
        int ix = tx - x0, iy = ty - y0;
        ix = ix < 0 ? 0 : (ix >= nx ? nx - 1 : ix); // defensive; callers stay in range
        iy = iy < 0 ? 0 : (iy >= ny ? ny - 1 : iy);
        return iy * nx + ix;
    }
    inline int gdx(int tx, int ty) const { return dx[idx(tx, ty)]; }
    inline int gdy(int tx, int ty) const { return dy[idx(tx, ty)]; }
};

// align_layer (align.cpp:15). For each output tile, search +/-4 around the
// (scaled, clamped) previous-level offset and take the argmin L1 tile score.
// prevConst == true means the previous alignment is the constant (0,0).
void align_layer(const U16Plane& ref, const U16Plane& alt,
                 const AlignGrid* prev, bool prevConst,
                 int pmin, int pmax,          // symmetric per-axis clamp (min==min.x==min.y)
                 int pmaxX,                   // (kept for clarity; pmin/pmax used both axes)
                 AlignGrid& out) {
    (void)pmaxX;
    const int dw = ref.w, dh = ref.h;
    for (int ty = out.y0; ty < out.y0 + out.ny; ++ty) {
        for (int tx = out.x0; tx < out.x0 + out.nx; ++tx) {
            int pox = 0, poy = 0;
            if (!prevConst) {
                int ptx = prev_tile(tx), pty = prev_tile(ty);
                int cvx = prev->gdx(ptx, pty), cvy = prev->gdy(ptx, pty);
                cvx = std::max(pmin, std::min(pmax, cvx));
                cvy = std::max(pmin, std::min(pmax, cvy));
                pox = DOWNSAMPLE_RATE * cvx;
                poy = DOWNSAMPLE_RATE * cvy;
            }
            long best = std::numeric_limits<long>::max();
            int bxi = -4, byi = -4;
            // RDom r1(-4,8,-4,8): x (xi) changes fastest -> yi outer, xi inner;
            // argmin keeps the first (lowest yi, then xi) minimum.
            for (int yi = -4; yi <= 3; ++yi) {
                for (int xi = -4; xi <= 3; ++xi) {
                    long score = 0;
                    for (int ry = 0; ry < 16; ++ry) {
                        int y0 = idx_layer(ty, ry);
                        int ay = y0 + poy + yi;
                        for (int rx = 0; rx < 16; ++rx) {
                            int x0 = idx_layer(tx, rx);
                            int ax = x0 + pox + xi;
                            int rv = ref.mget(x0, y0);
                            int av = alt.mget(ax, ay);
                            score += std::abs(rv - av);
                        }
                    }
                    if (score < best) { best = score; bxi = xi; byi = yi; }
                }
            }
            int o = (ty - out.y0) * out.nx + (tx - out.x0);
            out.dx[o] = (int16_t)(bxi + pox);
            out.dy[o] = (int16_t)(byi + poy);
        }
    }
}

} // namespace

// ----------------------------------------------------------------------------
void align_and_merge(const std::vector<const uint16_t*>& frames, int w, int h,
                     std::vector<uint16_t>& merged_out) {
    const int N = (int)frames.size();
    merged_out.assign((size_t)w * h, 0);
    if (N == 0) return;

    // Raw frames as mirror-interior planes (imgs_mirror in the source).
    std::vector<U16Plane> raw(N);
    for (int n = 0; n < N; ++n) {
        raw[n] = U16Plane(w, h);
        std::memcpy(raw[n].d.data(), frames[n], (size_t)w * h * sizeof(uint16_t));
    }
    if (N == 1) { merged_out = raw[0].d; return; }

    // Reference pyramid (frame 0) once.
    Pyr pref = build_pyramid(raw[0]);

    // Tile counts (align.cpp:120-121). Raw tiles: 32px, 16px overlap.
    const int num_tx = w / T_SIZE_2 - 1;
    const int num_ty = h / T_SIZE_2 - 1;
    if (num_tx <= 0 || num_ty <= 0) { merged_out = raw[0].d; return; }

    // Coarse index ranges (prev_tile reaches down to -1).
    const int n1x_hi = prev_tile(num_tx - 1), n1y_hi = prev_tile(num_ty - 1);
    const int n2x_hi = prev_tile(n1x_hi),     n2y_hi = prev_tile(n1y_hi);

    // merge_temporal half-res layer == align layer_0 (reuse per frame below).
    // Precompute cosine window (merge_spatial, T_SIZE = 32).
    double cosw[32];
    for (int v = 0; v < 32; ++v) cosw[v] = 0.5 - 0.5 * std::cos(2.0 * 3.141592 * (v + 0.5) / 32.0);

    // Spatial-merge accumulators. output(x,y) = u16(sum of 4 cosine-weighted
    // tiles). We accumulate per comparison frame's contribution incrementally
    // so only one comp pyramid is resident at a time (low RAM).
    std::vector<double> acc((size_t)w * h, 0.0);

    // Per-tile total weight (sum over comps + 1 for the reference) and, per
    // comp frame processed in turn, that comp's per-tile weight. We need all
    // comps' weights before we can divide, so collect weights first, then
    // blend. Weights are tiny (num_tx*num_ty per frame).
    const int TX0 = -1, TY0 = -1;                 // tiles span [-1 .. num_tx] for spatial
    const int TNX = num_tx - TX0 + 1;             // includes num_tx
    const int TNY = num_ty - TY0 + 1;
    auto tIdx = [&](int tx, int ty) { return (size_t)(ty - TY0) * TNX + (tx - TX0); };

    std::vector<float> total_w((size_t)TNX * TNY, 1.0f); // +1 for the reference
    std::vector<std::vector<float>> comp_w(N);           // comp_w[n][tile]
    std::vector<AlignGrid> align0(N);                    // raw per-tile offsets (2*align_0), per comp

    // Pass 1: per comp frame, build pyramid, alignment, and per-tile weights.
    // Parallel over comp frames (each writes only its own align0[n]/comp_w[n]);
    // total_w is reduced serially afterwards to avoid a race.
    parallel_for(1, N, [&](int n) {
        Pyr pn = build_pyramid(raw[n]);

        AlignGrid a2, a1, a0;
        a2.alloc(-1, -1, n2x_hi - (-1) + 1, n2y_hi - (-1) + 1);
        a1.alloc(-1, -1, n1x_hi - (-1) + 1, n1y_hi - (-1) + 1);
        a0.alloc(0, 0, num_tx, num_ty);
        // Level 2 (coarsest): prev = const 0, clamp [0,0].
        align_layer(pref.l2, pn.l2, nullptr, true, 0, 0, 0, a2);
        // Level 1: prev = a2, clamp min_2 = -4, max_2 = 3.
        align_layer(pref.l1, pn.l1, &a2, false, -4, 3, 0, a1);
        // Level 0: prev = a1, clamp min_1 = -20, max_1 = 15.
        align_layer(pref.l0, pn.l0, &a1, false, -20, 15, 0, a0);

        // Final raw alignment = 2 * alignment_0 (align.cpp:126).
        AlignGrid& ar = align0[n];
        ar.alloc(0, 0, num_tx, num_ty);
        for (int i = 0; i < num_tx * num_ty; ++i) { ar.dx[i] = (int16_t)(2 * a0.dx[i]); ar.dy[i] = (int16_t)(2 * a0.dy[i]); }

        // Per-tile temporal weight (merge.cpp:46-71) over the spatial tile span.
        comp_w[n].assign((size_t)TNX * TNY, 0.0f);
        const U16Plane& layRef = pref.l0;
        const U16Plane& layN   = pn.l0;
        const int dwL = layRef.w, dhL = layRef.h;
        for (int ty = TY0; ty <= num_ty; ++ty) {
            for (int tx = TX0; tx <= num_tx; ++tx) {
                // alignment is repeat_edge over [0,num_tx)x[0,num_ty).
                int ctx = clmp(tx, num_tx), cty = clmp(ty, num_ty);
                int ox = std::max(-168, std::min(126, (int)ar.dx[cty * num_tx + ctx]));
                int oy = std::max(-168, std::min(126, (int)ar.dy[cty * num_tx + ctx]));
                int ohx = fdiv(ox, 2), ohy = fdiv(oy, 2);
                long sumL1 = 0;
                for (int ry = 0; ry < 16; ++ry) {
                    int ly = idx_layer(ty, ry);
                    int aly = ly + ohy;
                    for (int rx = 0; rx < 16; ++rx) {
                        int lx = idx_layer(tx, rx);
                        int alx = lx + ohx;
                        int rv = layRef.d[(size_t)mir(ly, dhL) * dwL + mir(lx, dwL)];
                        int av = layN.d[(size_t)mir(aly, dhL) * dwL + mir(alx, dwL)];
                        sumL1 += std::abs(rv - av);
                    }
                }
                int dist = (int)(sumL1 / 256);                 // integer /256
                double norm_dist = std::max(1.0, dist / 8.0 - 10.0 / 8.0);
                float wv = (norm_dist > (300.0 - 10.0)) ? 0.0f : (float)(1.0 / norm_dist);
                comp_w[n][tIdx(tx, ty)] = wv;
            }
        }
    });

    // Reduce total weight = reference(1) + sum over comp frames.
    for (int n = 1; n < N; ++n)
        for (size_t t = 0; t < total_w.size(); ++t)
            total_w[t] += comp_w[n][t];

    // Pass 2: spatial merge. For each raw output pixel, blend the 4 tiles it
    // touches; each tile's temporal value is the weighted sum over comps + ref.
    // We fold temporal + spatial together (merge.cpp output + merge_spatial).
    // To keep only one comp pyramid resident we would stream, but temporal
    // needs every comp; the comp RAW planes are already resident (needed for
    // the pixel-domain fetch imgs_mirror(.,.,n)), so iterate pixels directly.
    parallel_for(0, h, [&](int y) {
        for (int x = 0; x < w; ++x) {
            int tx0 = fdiv(x, 16) - 1, tx1 = fdiv(x, 16);
            int ty0 = fdiv(y, 16) - 1, ty1 = fdiv(y, 16);
            int ix0 = fmod_(x, 16) + 16, ix1 = fmod_(x, 16);
            int iy0 = fmod_(y, 16) + 16, iy1 = fmod_(y, 16);
            // Four overlapping tiles: (tx0,ty0),(tx1,ty0),(tx0,ty1),(tx1,ty1)
            // with inner indices paired (idx_0 with tile_0, idx_1 with tile_1).
            struct TT { int tx, ty, ix, iy; } tts[4] = {
                {tx0, ty0, ix0, iy0}, {tx1, ty0, ix1, iy0},
                {tx0, ty1, ix0, iy1}, {tx1, ty1, ix1, iy1}};
            double out = 0.0;
            for (const TT& t : tts) {
                double wgt = cosw[t.ix] * cosw[t.iy];
                if (wgt == 0.0) continue;
                // temporal value at (tx,ty, ix,iy):
                float tot = total_w[tIdx(t.tx, t.ty)];
                int rxi = idx_im(t.tx, t.ix), ryi = idx_im(t.ty, t.iy);
                double tv = (double)raw[0].mget(rxi, ryi) / tot; // ref / total
                int ctx = clmp(t.tx, num_tx), cty = clmp(t.ty, num_ty);
                for (int n = 1; n < N; ++n) {
                    int ox = align0[n].dx[cty * num_tx + ctx];
                    int oy = align0[n].dy[cty * num_tx + ctx];
                    int av = raw[n].mget(rxi + ox, ryi + oy);
                    tv += (double)comp_w[n][tIdx(t.tx, t.ty)] * av / tot;
                }
                out += wgt * tv;
            }
            merged_out[(size_t)y * w + x] = u16c(out);
        }
    });
}

// ============================ finish ========================================
namespace {

// shift_bayer_to_rggb (finish.cpp:747). Returns the offset (ox,oy) so the
// mosaic reads as RGGB: out(x,y) = in(x+ox, y+oy).
void rggb_shift(int cfa, int& ox, int& oy) {
    switch (cfa) {
        case CFA_RGGB: ox = 0; oy = 0; break;
        case CFA_GRBG: ox = 1; oy = 0; break;
        case CFA_GBRG: ox = 0; oy = 1; break;
        case CFA_BGGR: ox = 1; oy = 1; break;
        default:       ox = 0; oy = 0; break;
    }
}

// gauss_7x7 separable (util.cpp:140). Templated on sample type; boundary is
// repeat_edge (combine wraps its inputs that way). Works on a full WxH plane.
const double K7[7] = {0.026267, 0.100742, 0.225511, 0.29496, 0.225511, 0.100742, 0.026267};

// gamma_correct / gamma_inverse scalars (util.cpp:217/254), u16 in -> u16 out.
inline uint16_t gamma_correct1(int v) {
    if (v < 200) return u16c(12.92 * v);
    return u16c(680.552897 * std::pow((double)v, 0.416667) - 3604.425);
}
inline uint16_t gamma_inverse1(int v) {
    if (v < 2575) return u16c(0.0774 * v);
    return u16c(std::pow((double)v / 65535.0 + 0.055, 2.4) * 57632.49226);
}

// Separable 7-tap gauss over a full plane, repeat_edge boundary (combine wraps
// its inputs that way). Parallel over rows; u16 input truncates to u16 like the
// Halide u16(val), float input stays float.
template <class T>
std::vector<T> gauss7_plane(const std::vector<T>& in, int w, int h) {
    std::vector<double> bx((size_t)w * h);
    parallel_for(0, h, [&](int y) {
        for (int x = 0; x < w; ++x) {
            double s = 0; for (int r = -3; r <= 3; ++r) s += (double)in[(size_t)y * w + clmp(x + r, w)] * K7[r + 3];
            bx[(size_t)y * w + x] = s;
        }
    });
    std::vector<T> out((size_t)w * h);
    parallel_for(0, h, [&](int y) {
        for (int x = 0; x < w; ++x) {
            double s = 0; for (int r = -3; r <= 3; ++r) s += bx[(size_t)clmp(y + r, h) * w + x] * K7[r + 3];
            if constexpr (std::is_same<T, uint16_t>::value) out[(size_t)y * w + x] = u16c(s);
            else out[(size_t)y * w + x] = (T)s;
        }
    });
    return out;
}

} // namespace

void finish(const uint16_t* merged, int w, int h, const Params& p,
            std::vector<uint8_t>& rgb_out) {
    // ---- shift to RGGB + black/white level + white balance ------------------
    // These three combine into one pass over the mosaic (finish.cpp:779-784).
    int sox, soy; rggb_shift(p.cfa_pattern, sox, soy);
    const double white_factor = 65535.0 / ((double)p.white_point - (double)p.black_point);
    // wb on the RGGB grid: (0,0)=R, (1,0)=G0, (0,1)=G1, (1,1)=B.
    auto bwl_wb = [&](int x, int y) -> uint16_t {
        // shifted sample with mirror_interior over the mosaic:
        int sx = mir(x + sox, w), sy = mir(y + soy, h);
        int raw = merged[(size_t)sy * w + sx];
        uint16_t lvl = u16s(((double)raw - (double)p.black_point) * white_factor);
        float g = ((y & 1) == 0) ? (((x & 1) == 0) ? p.wb_r : p.wb_g0)
                                 : (((x & 1) == 0) ? p.wb_g1 : p.wb_b);
        return u16s((double)g * (double)lvl);
    };
    U16Plane wb(w, h);
    parallel_for(0, h, [&](int y) {
        for (int x = 0; x < w; ++x) wb.at(x, y) = bwl_wb(x, y);
    });

    // ---- Malvar demosaic (finish.cpp:70) ------------------------------------
    static const int f0[5][5] = {{0,0,-1,0,0},{0,0,2,0,0},{-1,2,4,2,-1},{0,0,2,0,0},{0,0,-1,0,0}};
    static const int f1[5][5] = {{0,0,1,0,0},{0,-2,0,-2,0},{-2,8,10,8,-2},{0,-2,0,-2,0},{0,0,1,0,0}};
    static const int f2[5][5] = {{0,0,-2,0,0},{0,-2,8,-2,0},{1,0,10,0,1},{0,-2,8,-2,0},{0,0,-2,0,0}};
    static const int f3[5][5] = {{0,0,-3,0,0},{0,4,0,4,0},{-3,0,12,0,-3},{0,4,0,4,0},{0,0,-3,0,0}};
    auto conv5 = [&](const int kf[5][5], int fs, int x, int y) -> uint16_t {
        int acc = 0;
        for (int ry = -2; ry <= 2; ++ry)
            for (int rx = -2; rx <= 2; ++rx)
                acc += (int)wb.mget(x + rx, y + ry) * kf[ry + 2][rx + 2];
        return u16s((double)acc / (double)fs);
    };
    // RGB result as three interleaved planes.
    std::vector<uint16_t> rgb((size_t)w * h * 3, 0);
    parallel_for(0, h, [&](int y) {
        bool R_row = (y % 2 == 0), B_row = !R_row;
        for (int x = 0; x < w; ++x) {
            bool R_col = (x % 2 == 0), B_col = !R_col;
            uint16_t d0 = conv5(f0, 8, x, y);
            uint16_t d1 = conv5(f1, 16, x, y);
            uint16_t d2 = conv5(f2, 16, x, y);
            uint16_t d3 = conv5(f3, 16, x, y);
            uint16_t self = wb.at(x, y);
            uint16_t Rr, Gg, Bb;
            // R channel (finish.cpp:180)
            if (R_row && R_col) Rr = self;
            else if (R_row && B_col) Rr = d1;
            else if (B_row && R_col) Rr = d2;
            else Rr = d3;
            // G channel
            if ((R_row && R_col) || (B_row && B_col)) Gg = d0;
            else Gg = self;
            // B channel
            if (B_row && B_col) Bb = self;
            else if (B_row && R_col) Bb = d1;
            else if (R_row && B_col) Bb = d2;
            else Bb = d3;
            size_t o = ((size_t)y * w + x) * 3;
            rgb[o + 0] = Rr; rgb[o + 1] = Gg; rgb[o + 2] = Bb;
        }
    });

    // ---- sRGB CCM (finish.cpp:619): out_c = sum_r ccm[r*3+c] * in_r ---------
    std::vector<uint16_t> srgb((size_t)w * h * 3, 0);
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        double in0 = rgb[i * 3 + 0], in1 = rgb[i * 3 + 1], in2 = rgb[i * 3 + 2];
        for (int c = 0; c < 3; ++c) {
            double v = p.ccm[0 * 3 + c] * in0 + p.ccm[1 * 3 + c] * in1 + p.ccm[2 * 3 + c] * in2;
            srgb[i * 3 + c] = u16s(v);
        }
    }

    // ---- tone_map (finish.cpp:543) -----------------------------------------
    // grayscale = (r+g+b)/3 (integer); normal_dist LUT; 3-pass exposure fusion.
    std::vector<uint16_t> gray((size_t)w * h);
    for (size_t i = 0; i < (size_t)w * h; ++i)
        gray[i] = u16c(((uint32_t)srgb[i*3] + srgb[i*3+1] + srgb[i*3+2]) / 3u);

    static std::vector<float> ndist;
    if (ndist.empty()) {
        ndist.resize(65536);
        for (int v = 0; v < 65536; ++v)
            ndist[v] = (float)std::exp(-12.5 * std::pow((double)v / 65535.0 - 0.5, 2.0));
    }

    const double comp = p.compression, gainv = p.gain;
    const int num_passes = 3;
    const double comp_const = 1.0 + comp / num_passes;
    const double gain_const = 1.0 + gainv / num_passes;
    const double comp_slope = (comp - comp_const) / (num_passes - 1);
    const double gain_slope = (gainv - gain_const) / (num_passes - 1);

    std::vector<uint16_t> dark = gray; // dark starts as grayscale

    // gauss_7x7 over a full plane (repeat_edge). Thin wrappers over the shared
    // parallel helper above, keeping the combine() call sites unchanged.
    auto gauss7_u16 = [&](const std::vector<uint16_t>& in) { return gauss7_plane(in, w, h); };
    auto gauss7_f   = [&](const std::vector<float>& in)    { return gauss7_plane(in, w, h); };

    // combine() of two gamma-corrected greyscales with the normal_dist mask,
    // 2-layer Laplacian (num_layers = 2 -> one band in the loop + residual).
    auto combine = [&](const std::vector<uint16_t>& im1, const std::vector<uint16_t>& im2) {
        std::vector<float> mask1((size_t)w * h), mask2((size_t)w * h);
        for (size_t i = 0; i < (size_t)w * h; ++i) {
            float w1 = ndist[im1[i]], w2 = ndist[im2[i]];
            float m1 = w1 / (w1 + w2);
            mask1[i] = m1; mask2[i] = 1.0f - m1;
        }
        std::vector<uint16_t> unb1 = im1, unb2 = im2;
        std::vector<uint16_t> bl1 = gauss7_u16(im1), bl2 = gauss7_u16(im2);
        std::vector<long> accum((size_t)w * h, 0);
        // layer 1 band:
        for (size_t i = 0; i < (size_t)w * h; ++i) {
            int lap1 = (int)unb1[i] - (int)bl1[i];
            int lap2 = (int)unb2[i] - (int)bl2[i];
            accum[i] += (long)i32c((double)lap1 * mask1[i]) + (long)i32c((double)lap2 * mask2[i]);
        }
        bl1 = gauss7_u16(bl1); bl2 = gauss7_u16(bl2);
        mask1 = gauss7_f(mask1); mask2 = gauss7_f(mask2);
        // residual (lowest band):
        std::vector<uint16_t> out((size_t)w * h);
        for (size_t i = 0; i < (size_t)w * h; ++i) {
            long a = accum[i] + (long)i32c((double)bl1[i] * mask1[i]) + (long)i32c((double)bl2[i] * mask2[i]);
            out[i] = u16s((double)a);
        }
        return out;
    };

    for (int pass = 0; pass < num_passes; ++pass) {
        double norm_comp = pass * comp_slope + comp_const;
        double norm_gain = pass * gain_slope + gain_const;
        std::vector<uint16_t> bright((size_t)w * h);
        for (size_t i = 0; i < (size_t)w * h; ++i) bright[i] = u16s(norm_comp * (double)(uint32_t)dark[i]);
        std::vector<uint16_t> darkg((size_t)w * h), brightg((size_t)w * h);
        for (size_t i = 0; i < (size_t)w * h; ++i) { darkg[i] = gamma_correct1(dark[i]); brightg[i] = gamma_correct1(bright[i]); }
        std::vector<uint16_t> comb = combine(darkg, brightg);
        for (size_t i = 0; i < (size_t)w * h; ++i) dark[i] = u16s(norm_gain * (double)(uint32_t)gamma_inverse1(comb[i]));
    }

    // Reintroduce colour: out_c = sat16(in_c * dark / max(1,gray)) (finish.cpp:600).
    std::vector<uint16_t> toned((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        uint32_t g = std::max(1u, (uint32_t)gray[i]);
        for (int c = 0; c < 3; ++c)
            toned[i * 3 + c] = u16s((double)((uint64_t)(uint32_t)srgb[i*3+c] * (uint32_t)dark[i] / g));
    }

    // ---- gamma_correct (3ch) then contrast (strength 5, black 2000) --------
    for (size_t i = 0; i < (size_t)w * h * 3; ++i) toned[i] = gamma_correct1(toned[i]);

    const float strength = 5.f; const int black_level = 2000;
    const double scale = 0.8 + 0.3 / std::min(1.f, strength); // = 1.1
    const double inner_constant = 3.141592 / (2.0 * scale);
    const double sin_constant = std::sin(inner_constant);
    const double slope = 65535.0 / (2.0 * sin_constant);
    const double constant = slope * sin_constant;
    const double factor = 3.141592 / (scale * 65535.0);
    const double white_scale = 65535.0 / (65535.0 - black_level);

    rgb_out.assign((size_t)w * h * 3, 0);
    for (size_t i = 0; i < (size_t)w * h * 3; ++i) {
        double val = factor * (double)toned[i];
        uint16_t s = u16s(slope * std::sin(val - inner_constant) + constant);
        uint16_t c2 = u16s(((double)(int)s - black_level) * white_scale);
        rgb_out[i] = u8s((double)(c2 / 256)); // u8bit_interleaved: in/256 integer
    }
}

bool process(const std::vector<const uint16_t*>& frames, int w, int h,
             const Params& p,
             std::vector<uint16_t>& merged_out,
             std::vector<uint8_t>& rgb_out,
             bool allow_gpu) {
    if (frames.empty() || w <= 0 || h <= 0) return false;
#if defined(__APPLE__)
    // Try the Metal path; it returns false (and we fall back) if Metal is
    // unavailable or the runtime-compiled kernels fail.
    if (allow_gpu && process_gpu(frames, w, h, p, merged_out, rgb_out)) return true;
#else
    (void)allow_gpu;
#endif
    align_and_merge(frames, w, h, merged_out);
    finish(merged_out.data(), w, h, p, rgb_out);
    return true;
}

} // namespace hdrplus
} // namespace hhsr
