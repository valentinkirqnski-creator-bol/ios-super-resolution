// Misalignment-artifact NN (see mann.h). CPU features + windowed inference over the
// irregular tiles only; on Apple the convs run on the GPU (cref_conv kernel).
#include "mann.h"
#include "cref.h"            // cref_flow_disagreement
#include "mann_weights.h"
#include "parallel.h"
#if defined(__APPLE__)
#include "metal_gpu.h"
#endif
#include <algorithm>
#include <cmath>

namespace hhsr {

const int kMannDil[kMannLayers] = {1, 2, 4, 1};

bool mann_available() { return kMannTrained; }

float mann_motion_thresh(int tile_size, const Config& cfg) {
    // Same conversion as compute_robustness_gcam_sabre (GCam UV threshold -> raw px of
    // our 3x3 tile neighbourhood).
    constexpr float kGcamTilesAcross = 4128.f / 16.f;
    return cfg.gcam_sabre_motion_thresh * (float)std::max(1, tile_size) * kGcamTilesAcross;
}

std::vector<float> mann_tile_irregularity(const FlowField& flow) {
    const int ny = flow.ny, nx = flow.nx;
    std::vector<float> irr((size_t)std::max(0, ny) * std::max(0, nx), 0.f);
    if (ny <= 0 || nx <= 0 || flow.flow.size() < (size_t)ny * nx * 2) return irr;
    auto cl = [](int a, int hi) { return a < 0 ? 0 : (a >= hi ? hi - 1 : a); };
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            float mnx = 1e30f, mxx = -1e30f, mny = 1e30f, mxy = -1e30f;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int qy = cl(ty + dy, ny), qx = cl(tx + dx, nx);
                    const float vx = flow.dx(qy, qx), vy = flow.dy(qy, qx);
                    mnx = std::min(mnx, vx); mxx = std::max(mxx, vx);
                    mny = std::min(mny, vy); mxy = std::max(mxy, vy);
                }
            const float rx = mxx - mnx, ry = mxy - mny;
            irr[(size_t)ty * nx + tx] = std::sqrt(rx * rx + ry * ry);
        }
    return irr;
}

static inline float mn_c(float x) { return std::min(8.f, std::max(-8.f, x)) * 0.125f; }
static inline float smooth01(float t) { t = std::min(1.f, std::max(0.f, t)); return t * t * (3.f - 2.f * t); }

// Bilinear sample of a tile-grid scalar / the flow at grey pixel (y, x) (tile centres at
// ((t + 0.5) ts), edge-clamped).
struct TileBil { int i0, i1, j0, j1; float wy, wx; };
static inline TileBil tile_bil(int y, int x, int ts, int ny, int nx) {
    TileBil b;
    float fy = ((float)y + 0.5f) / (float)ts - 0.5f, fx = ((float)x + 0.5f) / (float)ts - 0.5f;
    fy = std::min((float)(ny - 1), std::max(0.f, fy));
    fx = std::min((float)(nx - 1), std::max(0.f, fx));
    b.i0 = std::min(ny - 1, (int)fy); b.j0 = std::min(nx - 1, (int)fx);
    b.i1 = std::min(ny - 1, b.i0 + 1); b.j1 = std::min(nx - 1, b.j0 + 1);
    b.wy = fy - (float)b.i0; b.wx = fx - (float)b.j0;
    return b;
}
static inline float bil_at(const float* g, int nx, int stride, int off, const TileBil& b) {
    const float a = g[((size_t)b.i0 * nx + b.j0) * stride + off], c = g[((size_t)b.i0 * nx + b.j1) * stride + off];
    const float d = g[((size_t)b.i1 * nx + b.j0) * stride + off], e = g[((size_t)b.i1 * nx + b.j1) * stride + off];
    const float t = a + (c - a) * b.wx, u = d + (e - d) * b.wx;
    return t + (u - t) * b.wy;
}

void mann_features(const Image& ref_grey, const Image& comp_aligned, const Image& R_grey,
                   const FlowField& flow, const std::vector<float>& M,
                   const std::vector<float>& tdis, int tile_size, const Config& cfg,
                   int wy0, int wy1, int wx0, int wx1,
                   std::vector<float>& out, std::vector<float>& gate_out) {
    const int h = ref_grey.h, w = ref_grey.w;
    const int rows = wy1 - wy0, cols = wx1 - wx0;
    const size_t plane = (size_t)std::max(0, rows) * std::max(0, cols);
    out.assign((size_t)kMannIn * plane, 0.f);
    gate_out.assign(plane, 0.f);
    if (rows <= 0 || cols <= 0) return;
    const int ts = std::max(1, tile_size), ny = flow.ny, nx = flow.nx;
    const bool have_flow = ny > 0 && nx > 0 && flow.flow.size() >= (size_t)ny * nx * 2 &&
                           M.size() == (size_t)ny * nx;
    const bool have_d = have_flow && tdis.size() == (size_t)ny * nx;
    const float T = mann_motion_thresh(ts, cfg) * std::max(1e-3f, cfg.mann_gate);
    const float g_lo = 0.75f * T, g_hi = 1.25f * T;
    auto cy = [&](int y) { return std::min(h - 1, std::max(0, y)); };
    auto cx = [&](int x) { return std::min(w - 1, std::max(0, x)); };
    // Aux region A = the rect +-2 (clamped): the low-pass L (5x5 binomial) and the 3x3
    // high-pass hp of ref and comp, each evaluated at clamped image coordinates (the
    // 5x5 statistics windows of the rect read A at clamped coordinates).
    const int ay0 = std::max(0, wy0 - 2), ay1 = std::min(h, wy1 + 2);
    const int ax0 = std::max(0, wx0 - 2), ax1 = std::min(w, wx1 + 2);
    const int ar = ay1 - ay0, ac = ax1 - ax0;
    const size_t ap = (size_t)ar * ac;
    std::vector<float> Lr(ap), Lc(ap), Hr(ap), Hc(ap);
    static const float kb[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
    parallel_rows(ar, cfg.num_threads, [&](int r) {
        const int y = ay0 + r;
        for (int c = 0; c < ac; ++c) {
            const int x = ax0 + c;
            float lr = 0.f, lc = 0.f;
            for (int i = -2; i <= 2; ++i) {
                const int yy = cy(y + i);
                float tr = 0.f, tc = 0.f;
                for (int j = -2; j <= 2; ++j) {
                    const int xx = cx(x + j);
                    tr += kb[j + 2] * ref_grey.at(yy, xx);
                    tc += kb[j + 2] * comp_aligned.at(yy, xx);
                }
                lr += kb[i + 2] * tr; lc += kb[i + 2] * tc;
            }
            float br = 0.f, bc = 0.f;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j) {
                    br += ref_grey.at(cy(y + i), cx(x + j));
                    bc += comp_aligned.at(cy(y + i), cx(x + j));
                }
            const size_t o = (size_t)r * ac + c;
            Lr[o] = lr; Lc[o] = lc;
            Hr[o] = ref_grey.at(y, x) - br * (1.f / 9.f);
            Hc[o] = comp_aligned.at(y, x) - bc * (1.f / 9.f);
        }
    });
    const float na = std::max(0.f, cfg.alpha_dng[1]), nb = std::max(1e-12f, cfg.beta_dng[1]);
    const float kE = (70.f / 256.f) * (70.f / 256.f);   // noise energy kept by the 2-D binomial
    const float kH = 8.f / 9.f;                         // noise energy kept by the 3x3 high-pass
    // VL = 1 - var(L)/var, noise-corrected (noise variance k * nv), 0 below SNR 4.
    auto vl = [&](float s1, float s2, float l1, float l2, float nv) {
        const float m = s1 / 25.f, ml = l1 / 25.f;
        const float v = std::max(0.f, s2 / 25.f - m * m) - nv;
        const float vlp = std::max(0.f, std::max(0.f, l2 / 25.f - ml * ml) - kE * nv);
        return (v > 4.f * nv) ? std::min(1.f, std::max(0.f, 1.f - vlp / v)) : 0.f;
    };
    parallel_rows(rows, cfg.num_threads, [&](int r) {
        const int y = wy0 + r;
        for (int c = 0; c < cols; ++c) {
            const int x = wx0 + c;
            float sr1 = 0, sr2 = 0, sc1 = 0, sc2 = 0, sd1 = 0, sd2 = 0;
            float lr1 = 0, lr2 = 0, lc1 = 0, lc2 = 0, ld1 = 0, ld2 = 0;
            float hdd = 0, hrr = 0, hcc = 0;
            for (int i = -2; i <= 2; ++i) {
                const int yy = cy(y + i);
                const size_t ro = (size_t)(yy - ay0) * ac;
                for (int j = -2; j <= 2; ++j) {
                    const int xx = cx(x + j);
                    const size_t o = ro + (size_t)(xx - ax0);
                    const float vr = ref_grey.at(yy, xx), vc = comp_aligned.at(yy, xx), vd = vc - vr;
                    const float a = Lr[o], b = Lc[o], d = b - a;
                    sr1 += vr; sr2 += vr * vr; sc1 += vc; sc2 += vc * vc; sd1 += vd; sd2 += vd * vd;
                    lr1 += a; lr2 += a * a; lc1 += b; lc2 += b * b; ld1 += d; ld2 += d * d;
                    const float hr = Hr[o], hc = Hc[o], hd = hc - hr;
                    hdd += hd * hd; hrr += hr * hr; hcc += hc * hc;
                }
            }
            const float m = sr1 / 25.f;
            const float nv = na * std::max(m, 0.f) + nb;
            const float rf = ref_grey.at(y, x), cp = comp_aligned.at(y, x);
            const float isd = 1.f / std::sqrt(2.f * nv), is1 = 1.f / std::sqrt(nv);
            const size_t ao = (size_t)(y - ay0) * ac + (size_t)(x - ax0);
            const size_t o = (size_t)r * cols + c;
            out[o] = std::min(1.f, std::max(0.f, R_grey.at(y, x)));
            out[plane + o] = mn_c((cp - rf) * isd);
            out[2 * plane + o] = mn_c(Hr[ao] * is1);
            out[3 * plane + o] = mn_c(Hc[ao] * is1);
            out[4 * plane + o] = vl(sr1, sr2, lr1, lr2, nv);
            out[5 * plane + o] = vl(sc1, sc2, lc1, lc2, nv);
            out[6 * plane + o] = vl(sd1, sd2, ld1, ld2, 2.f * nv);
            {
                const float nh = 2.f * kH * nv;
                const float den = (hrr + hcc) / 25.f - nh;
                const float num = std::max(0.f, hdd / 25.f - nh);
                out[7 * plane + o] = den > 4.f * nv ? std::min(2.f, num / den) * 0.5f : 0.f;
            }
            float m8 = 0.f, seam = 0.f, dd = 0.f, g = 0.f;
            if (have_flow) {
                const TileBil b = tile_bil(y, x, ts, ny, nx);
                const float Mb = bil_at(M.data(), nx, 1, 0, b);
                m8 = std::min(4.f, Mb / std::max(1e-6f, T)) * 0.25f;
                const int ty = std::min(ny - 1, y / ts), tx = std::min(nx - 1, x / ts);
                const float fbx = bil_at(flow.flow.data(), nx, 2, 0, b), fby = bil_at(flow.flow.data(), nx, 2, 1, b);
                seam = std::min(2.f, std::hypot(fbx - flow.dx(ty, tx), fby - flow.dy(ty, tx))) * 0.5f;
                if (have_d) dd = std::min(4.f, tdis[(size_t)ty * nx + tx]) * 0.25f;
                g = smooth01((Mb - g_lo) / std::max(1e-6f, g_hi - g_lo));
            }
            out[8 * plane + o] = m8;
            out[9 * plane + o] = seam;
            out[10 * plane + o] = dd;
            gate_out[o] = g;
        }
    });
}

// One 3x3 dilated conv + ReLU over a window (plane-major), replicate padding at the
// window edges (the window is clamped to the image, so these are the image edges or
// halo columns/rows the outputs never depend on).
static void mann_conv3(const std::vector<float>& in, int cin, std::vector<float>& out,
                       int rows, int cols, const float* W, const float* B, int dil, int nt) {
    const size_t plane = (size_t)rows * cols;
    out.assign((size_t)kMannCh * plane, 0.f);
    parallel_rows(rows, nt, [&](int y) {
        float acc[kMannCh];
        for (int x = 0; x < cols; ++x) {
            for (int co = 0; co < kMannCh; ++co) acc[co] = B[co];
            for (int ky = 0; ky < 3; ++ky) {
                const int yy = std::min(rows - 1, std::max(0, y + (ky - 1) * dil));
                for (int kx = 0; kx < 3; ++kx) {
                    const int xx = std::min(cols - 1, std::max(0, x + (kx - 1) * dil));
                    const size_t q = (size_t)yy * cols + xx;
                    for (int ci = 0; ci < cin; ++ci) {
                        const float v = in[(size_t)ci * plane + q];
                        if (v == 0.f) continue;
                        const float* wp = W + ((size_t)ci * 9 + ky * 3 + kx);
                        for (int co = 0; co < kMannCh; ++co) acc[co] += wp[(size_t)co * cin * 9] * v;
                    }
                }
            }
            for (int co = 0; co < kMannCh; ++co) out[(size_t)co * plane + (size_t)y * cols + x] = std::max(0.f, acc[co]);
        }
    });
}

bool mann_refine(Image& R, const Image& ref_grey, const Image& comp_aligned,
                 const FlowField& flow, int tile_size, const Config& cfg) {
    if (!kMannTrained) return false;
    const int h = ref_grey.h, w = ref_grey.w;
    if (h <= 0 || w <= 0 || R.h <= 0 || R.w <= 0 || R.data.size() != (size_t)R.h * R.w * R.c) return false;
    if (comp_aligned.h != h || comp_aligned.w != w) return false;
    const int ny = flow.ny, nx = flow.nx, ts = std::max(1, tile_size);
    if (ny <= 0 || nx <= 0 || flow.flow.size() < (size_t)ny * nx * 2) return false;
    const std::vector<float> M = mann_tile_irregularity(flow);
    const std::vector<float> tdis = cref_flow_disagreement(flow);
    const float T = mann_motion_thresh(ts, cfg) * std::max(1e-3f, cfg.mann_gate);
    // Tiles whose pixels can have g > 0: the bilinear M there is at most the max M of the
    // tile and its neighbours, so a tile is live if any 3x3 neighbour has M > 0.75 T.
    std::vector<uint8_t> live((size_t)ny * nx, 0);
    size_t n_live = 0;
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            bool l = false;
            for (int i = -1; i <= 1 && !l; ++i)
                for (int j = -1; j <= 1 && !l; ++j) {
                    const int qy = std::min(ny - 1, std::max(0, ty + i)), qx = std::min(nx - 1, std::max(0, tx + j));
                    l = M[(size_t)qy * nx + qx] > 0.75f * T;
                }
            live[(size_t)ty * nx + tx] = l; n_live += l;
        }
    if (!n_live) return true;   // no irregular tile: R unchanged
    // Mask on the grey grid (nearest; identity when R is already raw/grey-res).
    Image Rg(h, w, 1);
    parallel_rows(h, cfg.num_threads, [&](int y) {
        const int sy = std::min(R.h - 1, (int)((int64_t)y * R.h / h));
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(R.w - 1, (int)((int64_t)x * R.w / w));
            Rg.at(y, x) = R.at(sy, sx, 0);
        }
    });
    Image Rout = Rg;
    auto tile_y0 = [&](int ty) { return std::min(h, ty * ts); };
    auto tile_y1 = [&](int ty) { return ty >= ny - 1 ? h : std::min(h, (ty + 1) * ts); };
    auto tile_x0 = [&](int tx) { return std::min(w, tx * ts); };
    auto tile_x1 = [&](int tx) { return tx >= nx - 1 ? w : std::min(w, (tx + 1) * ts); };
    const int band_tiles = std::max(1, 128 / ts), halo = kMannHalo;
    std::vector<float> f, gt, a, b, dlt;
    for (int tr0 = 0; tr0 < ny; tr0 += band_tiles) {
        const int tr1 = std::min(ny, tr0 + band_tiles);
        int txa = nx, txb = -1;
        for (int ty = tr0; ty < tr1; ++ty)
            for (int tx = 0; tx < nx; ++tx)
                if (live[(size_t)ty * nx + tx]) { txa = std::min(txa, tx); txb = std::max(txb, tx); }
        if (txb < 0) continue;
        const int y0 = tile_y0(tr0), y1 = tile_y1(tr1 - 1), x0 = tile_x0(txa), x1 = tile_x1(txb);
        if (y1 <= y0 || x1 <= x0) continue;
        const int wy0 = std::max(0, y0 - halo), wy1 = std::min(h, y1 + halo);
        const int wx0 = std::max(0, x0 - halo), wx1 = std::min(w, x1 + halo);
        const int rows = wy1 - wy0, cols = wx1 - wx0;
        mann_features(ref_grey, comp_aligned, Rg, flow, M, tdis, ts, cfg, wy0, wy1, wx0, wx1, f, gt);
        const size_t plane = (size_t)rows * cols;
        bool gpu = false;
#if defined(__APPLE__)
        gpu = mann_window_metal(f, rows, cols, a);
#endif
        if (!gpu) {
            const float* Ws[kMannLayers] = {kMannW0, kMannW1, kMannW2, kMannW3};
            const float* Bs[kMannLayers] = {kMannB0, kMannB1, kMannB2, kMannB3};
            mann_conv3(f, kMannIn, a, rows, cols, Ws[0], Bs[0], kMannDil[0], cfg.num_threads);
            mann_conv3(a, kMannCh, b, rows, cols, Ws[1], Bs[1], kMannDil[1], cfg.num_threads);
            mann_conv3(b, kMannCh, a, rows, cols, Ws[2], Bs[2], kMannDil[2], cfg.num_threads);
            mann_conv3(a, kMannCh, b, rows, cols, Ws[3], Bs[3], kMannDil[3], cfg.num_threads);
            a.swap(b);
        }
        // Head + blend on the band's own pixels, only where the gate is open.
        parallel_rows(y1 - y0, cfg.num_threads, [&](int r) {
            const int y = y0 + r;
            for (int x = x0; x < x1; ++x) {
                const size_t q = (size_t)(y - wy0) * cols + (size_t)(x - wx0);
                const float g = gt[q];
                if (g <= 0.f) continue;
                float d = kMannBO;
                for (int c = 0; c < kMannCh; ++c) d += kMannWO[c] * a[(size_t)c * plane + q];
                const float rg = Rg.at(y, x);
                const float r0 = std::min(0.98f, std::max(0.02f, rg));
                const float z = std::log(r0 / (1.f - r0)) + d - cfg.mann_strictness;
                const float rn = 1.f / (1.f + std::exp(-z));
                Rout.at(y, x) = rg + g * (rn - rg);
            }
        });
    }
    // Back to R's grid: mean of the grey pixels each R pixel covers (identity at raw res).
    if (R.h == h && R.w == w) {
        for (size_t i = 0; i < (size_t)h * w; ++i)
            for (int c = 0; c < R.c; ++c) R.data[i * R.c + c] = Rout.data[i];
    } else {
        parallel_rows(R.h, cfg.num_threads, [&](int Y) {
            const int ya = (int)((int64_t)Y * h / R.h), yb = std::max(ya + 1, (int)((int64_t)(Y + 1) * h / R.h));
            for (int X = 0; X < R.w; ++X) {
                const int xa = (int)((int64_t)X * w / R.w), xb = std::max(xa + 1, (int)((int64_t)(X + 1) * w / R.w));
                float s = 0.f;
                int cnt = 0;
                for (int y = ya; y < std::min(h, yb); ++y)
                    for (int x = xa; x < std::min(w, xb); ++x) { s += Rout.at(y, x); ++cnt; }
                const float v = cnt ? s / (float)cnt : R.at(Y, X, 0);
                for (int c = 0; c < R.c; ++c) R.at(Y, X, c) = v;
            }
        });
    }
    return true;
}

}  // namespace hhsr
