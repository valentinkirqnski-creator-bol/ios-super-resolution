// fusion_nn_shared.h -- the fusion network's per-pixel layers (Config::fusion_nn_enabled).
//
// Dual-language: this exact file compiles as C++ (core/fusion_nn.cpp: CPU path and the PC
// parity test against PyTorch, tools/fusion_nn/parity_app.py) and as Metal
// (core/HHSRKernels.metal: fnn_stem / fnn_block / fnn_head), so the two paths cannot drift.
//
// Layout: a band of `rows` image rows starting at image row `row0`, `w` columns, channels
// interleaved per pixel ([row][x][ch]). Zero padding at the IMAGE border (PyTorch padding=d);
// taps outside the band but inside the image are clamped to the band -- only rows within the
// receptive field (16) of the band edge are affected, which is why callers keep a 16-row halo.
// Weights are packed channel-inner (see fusion_nn.cpp, Packed): stem [9][CIN][C],
// depthwise [9][C], pointwise [in][out].
#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define FNN_IN device const
#define FNN_OUT device
#define FNN_W constant
#define FNN_MAX(a, b) max(a, b)
#define FNN_CLAMPI(v, lo, hi) clamp(v, lo, hi)
#else
#include <algorithm>
#define FNN_IN const
#define FNN_OUT
#define FNN_W const
#define FNN_MAX(a, b) std::max(a, b)
#define FNN_CLAMPI(v, lo, hi) std::min(std::max(v, lo), hi)
#endif

#define FNN_K_CIN 8
#define FNN_K_C 12

struct FnnBand {
    int w;       // columns
    int rows;    // rows in the band buffers
    int row0;    // image row of band row 0
    int img_h;   // image rows (zero padding outside)
    int dil;     // dilation (blocks)
    int pad0, pad1, pad2;
};

inline void fnn_stem_px(FNN_IN float* X, FNN_OUT float* A, FNN_W float* Wp, FNN_W float* Bp,
                        FnnBand p, int x, int r) {
    float acc[FNN_K_C];
    for (int o = 0; o < FNN_K_C; ++o) acc[o] = Bp[o];
    const int y = p.row0 + r;
    for (int ky = 0; ky < 3; ++ky) {
        const int yy = y + ky - 1;
        if (yy < 0 || yy >= p.img_h) continue;
        const int rr = FNN_CLAMPI(yy - p.row0, 0, p.rows - 1);
        for (int kx = 0; kx < 3; ++kx) {
            const int xx = x + kx - 1;
            if (xx < 0 || xx >= p.w) continue;
            FNN_IN float* in = X + ((rr * p.w + xx) * FNN_K_CIN);
            const int t = ky * 3 + kx;
            for (int i = 0; i < FNN_K_CIN; ++i) {
                const float v = in[i];
                FNN_W float* wp = Wp + ((t * FNN_K_CIN + i) * FNN_K_C);
                for (int o = 0; o < FNN_K_C; ++o) acc[o] += wp[o] * v;
            }
        }
    }
    FNN_OUT float* out = A + ((r * p.w + x) * FNN_K_C);
    for (int o = 0; o < FNN_K_C; ++o) out[o] = FNN_MAX(acc[o], 0.f);
}

// out = in + relu(pointwise(depthwise_dilated(in)))
inline void fnn_block_px(FNN_IN float* in, FNN_OUT float* out, FNN_W float* dw, FNN_W float* dwb,
                         FNN_W float* pw, FNN_W float* pwb, FnnBand p, int x, int r) {
    float t[FNN_K_C];
    for (int c = 0; c < FNN_K_C; ++c) t[c] = dwb[c];
    const int y = p.row0 + r;
    const int d = p.dil;
    for (int ky = 0; ky < 3; ++ky) {
        const int yy = y + (ky - 1) * d;
        if (yy < 0 || yy >= p.img_h) continue;
        const int rr = FNN_CLAMPI(yy - p.row0, 0, p.rows - 1);
        for (int kx = 0; kx < 3; ++kx) {
            const int xx = x + (kx - 1) * d;
            if (xx < 0 || xx >= p.w) continue;
            FNN_IN float* v = in + ((rr * p.w + xx) * FNN_K_C);
            FNN_W float* wk = dw + ((ky * 3 + kx) * FNN_K_C);
            for (int c = 0; c < FNN_K_C; ++c) t[c] += wk[c] * v[c];
        }
    }
    float u[FNN_K_C];
    for (int o = 0; o < FNN_K_C; ++o) u[o] = pwb[o];
    for (int i = 0; i < FNN_K_C; ++i) {
        const float v = t[i];
        FNN_W float* wi = pw + (i * FNN_K_C);
        for (int o = 0; o < FNN_K_C; ++o) u[o] += wi[o] * v;
    }
    FNN_IN float* src = in + ((r * p.w + x) * FNN_K_C);
    FNN_OUT float* dst = out + ((r * p.w + x) * FNN_K_C);
    for (int o = 0; o < FNN_K_C; ++o) dst[o] = src[o] + FNN_MAX(u[o], 0.f);
}

// 1x1 head (weights [C] then bias at index C)
inline float fnn_head_px(FNN_IN float* in, FNN_W float* hw, FnnBand p, int x, int r) {
    FNN_IN float* v = in + ((r * p.w + x) * FNN_K_C);
    float l = hw[FNN_K_C];
    for (int c = 0; c < FNN_K_C; ++c) l += hw[c] * v[c];
    return l;
}
