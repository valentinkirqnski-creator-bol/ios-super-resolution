// feat_metric_shared.h — feature-metric alignment: the learned feature
// extractor, per-pixel code.
//
// Dual-language: this exact file compiles as C++ (CPU reference / harness) and
// as Metal Shading Language (core/HHSRKernels.metal). The CNN arithmetic mirrors
// the PyTorch FeatNet in scratchpad/step2_affine.py op-for-op: three 3x3 convs
// (1->8 relu, 8->8 relu, 8->4), REFLECT padding (PyTorch padding_mode="reflect",
// i.e. reflect-101: index -1 -> 1, N -> N-2), cross-correlation (no kernel flip).
// Change both languages or neither.
//
// Orchestration (which differs per platform) builds three feature planes by
// applying fm_conv3x3_at over the grey, each pass reading the previous plane.
// The per-tile affine Lucas-Kanade that consumes the features lives in
// feat_metric.cpp (CPU) and its Metal twin — it is tile-parallel, not per-pixel,
// so it is not in this shared header.
//
// Domain contract (GREY resolution, same grey the block matcher used):
//   grey : single channel, the values compute_grey() produces.
//   The feature planes are stored planar [FM_NF][h][w] in grey-pixel order.
#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define FM_DEVICE device
#define FM_THREAD thread
#define FM_CONSTANT constant
#define FM_CPTR constant
#else
#include <algorithm>
#define FM_DEVICE
#define FM_THREAD
#define FM_CONSTANT static const
#define FM_CPTR const
#endif

#ifdef FM_WEIGHTS_HEADER
#include FM_WEIGHTS_HEADER
#else
#include "feat_metric_weights.h"
#endif

// reflect-101 index clamp, matching torch padding_mode="reflect" for a border of
// 1: -1 -> 1, N -> N-2. Degenerates safely to 0 for N == 1.
inline int fm_reflect(int i, int N) {
    if (N <= 1) return 0;
    while (i < 0 || i >= N) {
        if (i < 0) i = -i;
        if (i >= N) i = 2 * N - 2 - i;
    }
    return i;
}

// One 3x3 cross-correlation + bias (+ optional ReLU) at (x,y), all `cout`
// output channels. `in` is a planar accessor float(c,x,y) over clamped-or-
// reflected coords; weights are [out][in][ky][kx]. out[] must hold `cout`.
template <typename IA>
inline void fm_conv3x3_at(IA in, int cin, int cout, FM_CPTR float* w, FM_CPTR float* b,
                          bool relu, int W, int H, int x, int y, FM_THREAD float* out) {
    for (int o = 0; o < cout; ++o) out[o] = b[o];
    for (int ky = 0; ky < 3; ++ky) {
        int yy = fm_reflect(y + ky - 1, H);
        for (int kx = 0; kx < 3; ++kx) {
            int xx = fm_reflect(x + kx - 1, W);
            for (int c = 0; c < cin; ++c) {
                float v = in(c, xx, yy);
                for (int o = 0; o < cout; ++o)
                    out[o] += w[((o * cin + c) * 3 + ky) * 3 + kx] * v;
            }
        }
    }
    if (relu)
        for (int o = 0; o < cout; ++o)
            if (out[o] < 0.0f) out[o] = 0.0f;
}

// Planar feature/grey plane accessor: [c][H][W] row-major. Coords must already
// be in-range (callers pass fm_reflect'd coords).
struct FMPlane {
    const FM_DEVICE float* p;
    int w, h;
    float operator()(int c, int x, int y) const { return p[(c * h + y) * w + x]; }
};
