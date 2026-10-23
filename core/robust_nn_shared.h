// robust_nn_shared.h — robustness-mask refinement network, per-pixel code.
//
// Dual-language: this exact file compiles as C++ (CPU reference, parity tests)
// and as Metal Shading Language (core/HHSRKernels.metal, tools' robust_nn.metal).
// The feature arithmetic mirrors tools/robust_nn/rnn/features.py
// operation-for-operation (same tap order, same expression trees) — change both
// or neither; tools/robust_nn/parity_test.py checks them against each other.
//
// Inputs are read through small accessor objects so the same arithmetic runs on
// plain buffers (CPU, standalone kernels) and on threadgroup tiles (the app's
// fused kernel). Every accessor is called with coordinates ALREADY CLAMPED to
// the image (clamp-to-edge semantics are applied here, not in the accessor).
//   image accessor : float operator()(int x, int y) const
//   flow accessor  : void  operator()(int x, int y, thread float& dx, thread float& dy) const
//   feature acc.   : float operator()(int c, int x, int y) const
//
// Domain contract (GREY resolution, grey = mean of each RGGB 2x2 quad):
//   ref, cmp : black-subtracted, normalised so the white level is 1.0
//   flow     : ref -> cmp in grey px: cmp is sampled at (x+dx, y+dy). Use the
//              same per-pixel vector the merge uses.
//   r0       : the analytic robustness, in [0,1]
//   noise    : grey-domain variance var(v) = noise_a * v + noise_b
//              (per-sample raw a,b of an RGGB quad mean: (aR+2aG+aB)/16 etc.)
//   flow_step: spacing in px for the flow-consistency features (tile stride)
#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define RNN_DEVICE device
#define RNN_THREAD thread
#define RNN_TGROUP threadgroup
#define RNN_CONSTANT constant
#define RNN_SQRT(x) precise::sqrt(x)
#define RNN_LOG(x) precise::log(x)
#define RNN_EXP(x) precise::exp(x)
#define RNN_FLOOR(x) floor(x)
#else
#include <cmath>
#define RNN_DEVICE
#define RNN_THREAD
#define RNN_TGROUP
#define RNN_CONSTANT static const
#define RNN_SQRT(x) std::sqrt(x)
#define RNN_LOG(x) std::log(x)
#define RNN_EXP(x) std::exp(x)
#define RNN_FLOOR(x) std::floor(x)
#endif

#ifdef RNN_WEIGHTS_HEADER
#include RNN_WEIGHTS_HEADER
#else
#include "robust_nn_weights.h"
#endif

struct RNNParams {
    int width;
    int height;
    int flow_step;
    float noise_a;
    float noise_b;
    float clip_level;
    int pad0;
    int pad1;
};

inline int rnn_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float rnn_minf(float a, float b) { return a < b ? a : b; }
inline float rnn_maxf(float a, float b) { return a > b ? a : b; }
inline float rnn_clampf(float v, float lo, float hi) { return rnn_minf(rnn_maxf(v, lo), hi); }

// ---- plain-buffer accessors -------------------------------------------------------
struct RNNImg {
    const RNN_DEVICE float* p;
    int w;
    float operator()(int x, int y) const { return p[y * w + x]; }
};
struct RNNFlowInterleaved {          // (dx,dy) per pixel
    const RNN_DEVICE float* p;
    int w;
    void operator()(int x, int y, RNN_THREAD float& dx, RNN_THREAD float& dy) const {
        int i = 2 * (y * w + x);
        dx = p[i];
        dy = p[i + 1];
    }
};
struct RNNFeatPlanar {               // [RNN_NF][h][w]
    const RNN_DEVICE float* p;
    int w, h;
    float operator()(int c, int x, int y) const { return p[(c * h + y) * w + x]; }
};

template <typename IA>
inline float rnn_bilinear(IA img, int W, int H, float x, float y) {
    x = rnn_clampf(x, 0.0f, (float)(W - 1));
    y = rnn_clampf(y, 0.0f, (float)(H - 1));
    float x0f = RNN_FLOOR(x), y0f = RNN_FLOOR(y);
    float fx = x - x0f, fy = y - y0f;
    int x0 = (int)x0f, y0 = (int)y0f;
    int x1 = x0 + 1 < W - 1 ? x0 + 1 : W - 1;
    int y1 = y0 + 1 < H - 1 ? y0 + 1 : H - 1;
    float v00 = img(x0, y0), v01 = img(x1, y0);
    float v10 = img(x0, y1), v11 = img(x1, y1);
    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) + fy * ((1.0f - fx) * v10 + fx * v11);
}

// Pass 1: J(p) = cmp(p + flow(p)).
template <typename IA, typename FA>
inline float rnn_warp_at(IA cmp, FA flow, int W, int H, int x, int y) {
    float dx, dy;
    flow(x, y, dx, dy);
    return rnn_bilinear(cmp, W, H, (float)x + dx, (float)y + dy);
}

// Pass 2: the 14 evidence features at (x,y) (in-image). r0c = r0 at (x,y).
template <typename RA, typename JA, typename FA>
inline void rnn_features_at(RA ref, JA J, FA flow, float r0v, RNNParams P, int x, int y, RNN_THREAD float* f) {
    const int W = P.width, H = P.height;
    float smr = 0.0f, smj = 0.0f, sci = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int qx = rnn_clampi(x + dx, 0, W - 1), qy = rnn_clampi(y + dy, 0, H - 1);
            float a = ref(qx, qy);
            float b = J(qx, qy);
            smr = smr + a;
            smj = smj + b;
            sci = sci + ((rnn_maxf(a, b) > P.clip_level) ? 1.0f : 0.0f);
        }
    float mu_r = smr / 9.0f, mu_j = smj / 9.0f;
    float s2 = rnn_maxf(P.noise_a * rnn_maxf(mu_r, 0.0f) + P.noise_b, 1e-10f);

    float Rc = rnn_clampf(r0v, 1e-3f, 1.0f - 1e-3f);
    f[0] = 0.25f * RNN_LOG(Rc / (1.0f - Rc));
    float dm = mu_j - mu_r;
    f[1] = RNN_LOG(1.0f + dm * dm * 9.0f / s2);

    float rr = 0.0f, Sxx = 0.0f, Syy = 0.0f, Sxy = 0.0f, bx = 0.0f, by = 0.0f;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            int qx = rnn_clampi(x + dx, 0, W - 1), qy = rnn_clampi(y + dy, 0, H - 1);
            float r = J(qx, qy) - ref(qx, qy);
            float gx = 0.5f * (ref(rnn_clampi(qx + 1, 0, W - 1), qy) - ref(rnn_clampi(qx - 1, 0, W - 1), qy));
            float gy = 0.5f * (ref(qx, rnn_clampi(qy + 1, 0, H - 1)) - ref(qx, rnn_clampi(qy - 1, 0, H - 1)));
            rr = rr + r * r;
            Sxx = Sxx + gx * gx;
            Syy = Syy + gy * gy;
            Sxy = Sxy + gx * gy;
            bx = bx + gx * r;
            by = by + gy * r;
        }
    rr = rr / 25.0f; Sxx = Sxx / 25.0f; Syy = Syy / 25.0f; Sxy = Sxy / 25.0f; bx = bx / 25.0f; by = by / 25.0f;
    f[2] = RNN_LOG(1.0f + rr / s2);
    f[3] = RNN_LOG(1.0f + (Sxx + Syy) / s2);
    float lam = 0.5f * s2;
    float A11 = Sxx + lam, A22 = Syy + lam;
    float det = A11 * A22 - Sxy * Sxy;
    float u = (A22 * bx - Sxy * by) / det;
    float v = (A11 * by - Sxy * bx) / det;
    f[4] = rnn_minf(RNN_SQRT(u * u + v * v), 4.0f);
    f[5] = rnn_clampf((bx * u + by * v) / (rr + 1e-3f * s2), 0.0f, 1.0f);
    float tr = Sxx + Syy;
    float dd = RNN_SQRT((Sxx - Syy) * (Sxx - Syy) + 4.0f * Sxy * Sxy);
    f[6] = dd / (tr + 2.0f * s2);
    f[7] = 0.25f * RNN_LOG(rnn_maxf(mu_r, 1e-4f));
    f[8] = sci / 9.0f;

    // flow consistency: least-squares affine fit over a 5x5 lattice of step fs
    const int fs = P.flow_step;
    float n = 0.0f, sx = 0.0f, sy = 0.0f, sxx = 0.0f, sxy = 0.0f, syy = 0.0f;
    float bx0 = 0.0f, bx1 = 0.0f, bx2 = 0.0f, by0 = 0.0f, by1 = 0.0f, by2 = 0.0f;
    for (int j = -2; j <= 2; ++j)
        for (int i = -2; i <= 2; ++i) {
            int qx = rnn_clampi(x + i * fs, 0, W - 1), qy = rnn_clampi(y + j * fs, 0, H - 1);
            float tx, ty;
            flow(qx, qy, tx, ty);
            float ddx = (float)(qx - x) / (float)fs, ddy = (float)(qy - y) / (float)fs;
            n = n + 1.0f;
            sx = sx + ddx; sy = sy + ddy;
            sxx = sxx + ddx * ddx; sxy = sxy + ddx * ddy; syy = syy + ddy * ddy;
            bx0 = bx0 + tx; bx1 = bx1 + ddx * tx; bx2 = bx2 + ddy * tx;
            by0 = by0 + ty; by1 = by1 + ddx * ty; by2 = by2 + ddy * ty;
        }
    float c00 = sxx * syy - sxy * sxy;
    float c01 = sy * sxy - sx * syy;
    float c02 = sx * sxy - sxx * sy;
    float c11 = n * syy - sy * sy;
    float c12 = sx * sy - n * sxy;
    float c22 = n * sxx - sx * sx;
    float fdet = n * c00 + sx * c01 + sy * c02;
    bool ok = fdet > 1e-3f;
    float inv = ok ? 1.0f / fdet : 0.0f;
    float ax0 = ok ? (c00 * bx0 + c01 * bx1 + c02 * bx2) * inv : bx0 / n;
    float ax1 = (c01 * bx0 + c11 * bx1 + c12 * bx2) * inv;
    float ax2 = (c02 * bx0 + c12 * bx1 + c22 * bx2) * inv;
    float ay0 = ok ? (c00 * by0 + c01 * by1 + c02 * by2) * inv : by0 / n;
    float ay1 = (c01 * by0 + c11 * by1 + c12 * by2) * inv;
    float ay2 = (c02 * by0 + c12 * by1 + c22 * by2) * inv;
    float e2 = 0.0f;
    for (int j = -2; j <= 2; ++j)
        for (int i = -2; i <= 2; ++i) {
            int qx = rnn_clampi(x + i * fs, 0, W - 1), qy = rnn_clampi(y + j * fs, 0, H - 1);
            float tx, ty;
            flow(qx, qy, tx, ty);
            float ddx = (float)(qx - x) / (float)fs, ddy = (float)(qy - y) / (float)fs;
            float ex = tx - (ax0 + ax1 * ddx + ax2 * ddy);
            float ey = ty - (ay0 + ay1 * ddx + ay2 * ddy);
            e2 = e2 + (ex * ex + ey * ey);
        }
    float fx0, fy0;
    flow(x, y, fx0, fy0);
    float cx = fx0 - ax0, cy = fy0 - ay0;
    f[9] = RNN_LOG(1.0f + RNN_SQRT(cx * cx + cy * cy));
    f[10] = RNN_LOG(1.0f + RNN_SQRT(e2 / n));
    f[11] = RNN_LOG(1.0f + RNN_SQRT(ax1 * ax1 + ax2 * ax2 + ay1 * ay1 + ay2 * ay2));
    float md = 0.0f;
    for (int t = 0; t < 4; ++t) {
        int ox = (t == 0) ? 1 : ((t == 1) ? -1 : 0);
        int oy = (t == 2) ? 1 : ((t == 3) ? -1 : 0);
        int qx = rnn_clampi(x + ox * fs, 0, W - 1), qy = rnn_clampi(y + oy * fs, 0, H - 1);
        float tx, ty;
        flow(qx, qy, tx, ty);
        float ex = fx0 - tx, ey = fy0 - ty;
        md = rnn_maxf(md, RNN_SQRT(ex * ex + ey * ey));
    }
    f[12] = RNN_LOG(1.0f + md);

    float r9 = 0.0f;
    for (int dy = -4; dy <= 4; dy += 2)
        for (int dx = -4; dx <= 4; dx += 2) {
            int qx = rnn_clampi(x + dx, 0, W - 1), qy = rnn_clampi(y + dy, 0, H - 1);
            float r = J(qx, qy) - ref(qx, qy);
            r9 = r9 + r * r;
        }
    f[13] = RNN_LOG(1.0f + (r9 / 25.0f) / s2);
}

// Pass 3: the network at (x,y). feat(c, x, y) over clamped coords. Returns refined R.
template <typename FTA>
inline float rnn_net_at(FTA feat, float r0v, int W, int H, int x, int y) {
    float h1[RNN_HID];
    for (int h = 0; h < RNN_HID; ++h) h1[h] = RNN_B1[h];
    for (int ky = 0; ky < 3; ++ky) {
        int yy = rnn_clampi(y + ky - 1, 0, H - 1);
        for (int kx = 0; kx < 3; ++kx) {
            int xx = rnn_clampi(x + kx - 1, 0, W - 1);
            for (int c = 0; c < RNN_NF; ++c) {
                float v = feat(c, xx, yy);
                for (int h = 0; h < RNN_HID; ++h) h1[h] += RNN_W1[((h * RNN_NF + c) * 3 + ky) * 3 + kx] * v;
            }
        }
    }
    for (int h = 0; h < RNN_HID; ++h) h1[h] = rnn_maxf(h1[h], 0.0f);
    float d = RNN_B3[0];
    for (int o = 0; o < RNN_HID; ++o) {
        float a = RNN_B2[o];
        for (int i = 0; i < RNN_HID; ++i) a += RNN_W2[o * RNN_HID + i] * h1[i];
        d += RNN_W3[o] * rnn_maxf(a, 0.0f);
    }
    float Rc = rnn_clampf(r0v, 1e-3f, 1.0f - 1e-3f);
    float z = RNN_LOG(Rc / (1.0f - Rc)) + d;
    return 1.0f / (1.0f + RNN_EXP(-z));
}

#ifndef RNN_CTX
#define RNN_CTX 0
#endif
#if RNN_CTX > 0
// ---- v31 tile branch (RNN_CTX > 0; mirrors rnn/model.py RobustNetTile) ------------------------
// A wrong tile vector moves the WHOLE flow tile, and a zipper is the border between a right and a
// wrong tile, so the net also sees its own tile: the mean and max of the 14 standardised features
// over the pixel's flow tile (flow_step guide px, grid origin 0) -> K1 (1x1) -> ReLU -> K2 (3x3 over
// the tile grid, clamp-to-edge: the 8 neighbour tiles) -> ReLU -> K3 (1x1) added to layer 2 before
// its ReLU. Nothing crosses a tile border except the 3x3 tile step.

// Per-tile summary from the tile's mean / max of the RAW features (standardised here).
inline void rnn_tile_summary(RNN_THREAD const float* avg, RNN_THREAD const float* mx, RNN_THREAD float* c) {
    for (int o = 0; o < RNN_CTX; ++o) {
        float a = RNN_BK1[o];
        for (int i = 0; i < RNN_NF; ++i) {
            float xa = (avg[i] - RNN_MU[i]) / RNN_SD[i];
            float xm = (mx[i] - RNN_MU[i]) / RNN_SD[i];
            a += RNN_K1[o * 2 * RNN_NF + i] * xa + RNN_K1[o * 2 * RNN_NF + RNN_NF + i] * xm;
        }
        c[o] = rnn_maxf(a, 0.0f);
    }
}

struct RNNTileA {                    // [tile][RNN_CTX], tiles row-major over a nxg-wide grid
    const RNN_DEVICE float* p;
    int nxg;
    float operator()(int c, int tx, int ty) const { return p[(ty * nxg + tx) * RNN_CTX + c]; }
};

// 3x3 over the tile grid + ReLU, clamp-to-edge (== replicate padding).
template <typename TA>
inline void rnn_tile_conv_at(TA tileA, int nxg, int nyg, int tx, int ty, RNN_THREAD float* out) {
    for (int o = 0; o < RNN_CTX; ++o) out[o] = RNN_BK2[o];
    for (int ky = 0; ky < 3; ++ky) {
        int yy = rnn_clampi(ty + ky - 1, 0, nyg - 1);
        for (int kx = 0; kx < 3; ++kx) {
            int xx = rnn_clampi(tx + kx - 1, 0, nxg - 1);
            for (int c = 0; c < RNN_CTX; ++c) {
                float v = tileA(c, xx, yy);
                for (int o = 0; o < RNN_CTX; ++o) out[o] += RNN_K2[((o * RNN_CTX + c) * 3 + ky) * 3 + kx] * v;
            }
        }
    }
    for (int o = 0; o < RNN_CTX; ++o) out[o] = rnn_maxf(out[o], 0.0f);
}

// The network at (x,y) with the pixel's tile term tb[RNN_CTX] (K2's output for its tile).
template <typename FTA>
inline float rnn_net_tile_at(FTA feat, RNN_THREAD const float* tb, float r0v, int W, int H, int x, int y) {
    float h1[RNN_HID];
    for (int h = 0; h < RNN_HID; ++h) h1[h] = RNN_B1[h];
    for (int ky = 0; ky < 3; ++ky) {
        int yy = rnn_clampi(y + ky - 1, 0, H - 1);
        for (int kx = 0; kx < 3; ++kx) {
            int xx = rnn_clampi(x + kx - 1, 0, W - 1);
            for (int c = 0; c < RNN_NF; ++c) {
                float v = feat(c, xx, yy);
                for (int h = 0; h < RNN_HID; ++h) h1[h] += RNN_W1[((h * RNN_NF + c) * 3 + ky) * 3 + kx] * v;
            }
        }
    }
    for (int h = 0; h < RNN_HID; ++h) h1[h] = rnn_maxf(h1[h], 0.0f);
    float d = RNN_B3[0];
    for (int o = 0; o < RNN_HID; ++o) {
        float a = RNN_B2[o];
        for (int i = 0; i < RNN_HID; ++i) a += RNN_W2[o * RNN_HID + i] * h1[i];
        for (int c = 0; c < RNN_CTX; ++c) a += RNN_K3[o * RNN_CTX + c] * tb[c];
        d += RNN_W3[o] * rnn_maxf(a, 0.0f);
    }
    float Rc = rnn_clampf(r0v, 1e-3f, 1.0f - 1e-3f);
    float z = RNN_LOG(Rc / (1.0f - Rc)) + d;
    return 1.0f / (1.0f + RNN_EXP(-z));
}
#endif

// ---- v15: pixel-seeing refinement (Config::robust_nn_v15_enabled) -----------------------
// Mirrors tools/robust_nn/rnn/pixnet.py (PixNet.forward, pixel_inputs) op for op over its OWN
// v14 base (RNN15B_*, so a later change of the main weights cannot change v15's base). The app
// runs it as banded kernels (HHSRKernels.metal, rnn15_*); tools/robust_nn/app_kernel_test15.py
// checks those kernels, band edges included, against PyTorch.
#ifdef RNN15_WEIGHTS_HEADER
#include RNN15_WEIGHTS_HEADER
#else
#include "robust_nn15_weights.h"
#endif

#ifdef __METAL_VERSION__
#define RNN_CPTR constant
#else
#define RNN_CPTR const
#endif

// Base (v14) logit at (x,y) with PixNet's clamp of the base probability to [1e-4, 1 - 1e-4].
template <typename FTA>
inline float rnn15_base_logit_at(FTA feat, float r0v, int W, int H, int x, int y) {
    float h1[RNN15B_HID];
    for (int h = 0; h < RNN15B_HID; ++h) h1[h] = RNN15B_B1[h];
    for (int ky = 0; ky < 3; ++ky) {
        int yy = rnn_clampi(y + ky - 1, 0, H - 1);
        for (int kx = 0; kx < 3; ++kx) {
            int xx = rnn_clampi(x + kx - 1, 0, W - 1);
            for (int c = 0; c < RNN15_NF; ++c) {
                float v = feat(c, xx, yy);
                for (int h = 0; h < RNN15B_HID; ++h) h1[h] += RNN15B_W1[((h * RNN15_NF + c) * 3 + ky) * 3 + kx] * v;
            }
        }
    }
    for (int h = 0; h < RNN15B_HID; ++h) h1[h] = rnn_maxf(h1[h], 0.0f);
    float d = RNN15B_B3[0];
    for (int o = 0; o < RNN15B_HID; ++o) {
        float a = RNN15B_B2[o];
        for (int i = 0; i < RNN15B_HID; ++i) a += RNN15B_W2[o * RNN15B_HID + i] * h1[i];
        d += RNN15B_W3[o] * rnn_maxf(a, 0.0f);
    }
    float Rc = rnn_clampf(r0v, 1e-3f, 1.0f - 1e-3f);
    float z = RNN_LOG(Rc / (1.0f - Rc)) + d;
    float R = rnn_clampf(1.0f / (1.0f + RNN_EXP(-z)), 1e-4f, 1.0f - 1e-4f);
    return RNN_LOG(R) - RNN_LOG(1.0f - R);
}

// The RNN15_CIN PixNet inputs at (x,y): standardised raw features (feat holds UNstandardised
// features), 4 noise-normalised pixel channels, base logit / 4.
template <typename RA, typename JA, typename FTA>
inline void rnn15_inputs_at(RA ref, JA J, FTA feat, float r0v, RNNParams P, int x, int y, RNN_THREAD float* xin) {
    const int W = P.width, H = P.height;
    for (int c = 0; c < RNN15_NF; ++c) xin[c] = (feat(c, x, y) - RNN15_MU[c]) / RNN15_SD[c];
    float smr = 0.0f, smj = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int qx = rnn_clampi(x + dx, 0, W - 1), qy = rnn_clampi(y + dy, 0, H - 1);
            smr = smr + ref(qx, qy);
            smj = smj + J(qx, qy);
        }
    float mr = smr / 9.0f, mj = smj / 9.0f;
    float sig = RNN_SQRT(rnn_maxf(P.noise_a * rnn_maxf(mr, 0.0f) + P.noise_b, 1e-10f));
    float ir = ref(x, y), jv = J(x, y);
    xin[RNN15_NF + 0] = rnn_clampf((jv - ir) / sig, -10.0f, 10.0f);
    xin[RNN15_NF + 1] = rnn_clampf((ir - mr) / sig, -10.0f, 10.0f);
    xin[RNN15_NF + 2] = rnn_clampf((jv - mj) / sig, -10.0f, 10.0f);
    xin[RNN15_NF + 3] = rnn_clampf((mj - mr) / sig, -10.0f, 10.0f);
    xin[RNN15_NF + 4] = rnn15_base_logit_at(feat, r0v, W, H, x, y) / 4.0f;
}

// One dilated 3x3 conv + ReLU at (x,y), clamp-to-edge (== replicate padding).
template <typename XA>
inline void rnn15_conv_at(XA in, int cin, RNN_CPTR float* w, RNN_CPTR float* b, int dil, int W, int H, int x, int y,
                          RNN_THREAD float* out) {
    for (int o = 0; o < RNN15_CH; ++o) out[o] = b[o];
    for (int ky = 0; ky < 3; ++ky) {
        int yy = rnn_clampi(y + (ky - 1) * dil, 0, H - 1);
        for (int kx = 0; kx < 3; ++kx) {
            int xx = rnn_clampi(x + (kx - 1) * dil, 0, W - 1);
            for (int c = 0; c < cin; ++c) {
                float v = in(c, xx, yy);
                for (int o = 0; o < RNN15_CH; ++o) out[o] += w[((o * cin + c) * 3 + ky) * 3 + kx] * v;
            }
        }
    }
    for (int o = 0; o < RNN15_CH; ++o) out[o] = rnn_maxf(out[o], 0.0f);
}

// Head: R = sigmoid(base logit + delta).
template <typename HA>
inline float rnn15_out_at(HA h, float zb, int x, int y) {
    float d = RNN15_B5[0];
    for (int c = 0; c < RNN15_CH; ++c) d += RNN15_W5[c] * h(c, x, y);
    return 1.0f / (1.0f + RNN_EXP(-(zb + d)));
}
