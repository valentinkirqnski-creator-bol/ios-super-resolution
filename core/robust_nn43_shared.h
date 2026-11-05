// robust_nn43_shared.h — robust_nn v43 network (tools/robust_nn rnn/model.py RobustNet43), per guide pixel.
//
// Dual-language like robust_nn_shared.h (include that first: it supplies RNN_CONSTANT / RNN_LOG / RNN_EXP,
// rnn_clampi, rnn_maxf and the v40 features). v43 = one learned weight per 48 MP OUTPUT pixel from statistics
// at 12 MP:
//   inputs  : the v40 features of the guide pixel (RNN_NF of them, read through `feat`) and its quad's four
//             raw-resolution analytic values (Algorithm 6, 8-bit), as 0.25 * logit(clamp(q, 1e-3, 1 - 1e-3)),
//             channel 2 * dy + dx for raw pixel (2y + dy, 2x + dx), read through `rawf`
//   layers  : 3x3 conv over clamped coords (RNN43_NIN inputs) -> ReLU -> 1x1 -> ReLU -> 1x1 to 16 outputs
//   outputs : d[4 * dy + dx] = the correction of output pixel (4y + dy, 4x + dx) of the 2x output grid
// The caller forms R(o) = sigmoid(logit(q of o's raw pixel) + d) and runs Eq. 9's minimum on the output grid.
// Mirrors RobustNet43.forward operation for operation; tools/robust_nn/app_kernel_test_v43.py checks it.
#pragma once

#ifdef RNN43_WEIGHTS_HEADER
#include RNN43_WEIGHTS_HEADER
#else
#include "robust_nn43_weights.h"
#endif

// The network's input for one raw analytic value stored as 8 bits.
inline float rnn43_raw_feature(float q) {
    float Rc = q < 1e-3f ? 1e-3f : (q > 1.0f - 1e-3f ? 1.0f - 1e-3f : q);
    return 0.25f * RNN_LOG(Rc / (1.0f - Rc));
}

// feat(c, x, y): v40 feature c (< RNN_NF) at clamped guide coords; rawf(k, x, y): raw input k (0..3) there.
template <typename FTA, typename RTA>
inline void rnn43_net_at(FTA feat, RTA rawf, int W, int H, int x, int y, RNN_THREAD float* d) {
    float h1[RNN43_HID];
    for (int h = 0; h < RNN43_HID; ++h) h1[h] = RNN43_B1[h];
    for (int ky = 0; ky < 3; ++ky) {
        int yy = rnn_clampi(y + ky - 1, 0, H - 1);
        for (int kx = 0; kx < 3; ++kx) {
            int xx = rnn_clampi(x + kx - 1, 0, W - 1);
            for (int c = 0; c < RNN43_NIN; ++c) {
                float v = (c < RNN43_NIN - 4) ? feat(c, xx, yy) : rawf(c - (RNN43_NIN - 4), xx, yy);
                for (int h = 0; h < RNN43_HID; ++h) h1[h] += RNN43_W1[((h * RNN43_NIN + c) * 3 + ky) * 3 + kx] * v;
            }
        }
    }
    for (int h = 0; h < RNN43_HID; ++h) h1[h] = rnn_maxf(h1[h], 0.0f);
    float h2[RNN43_HID];
    for (int o = 0; o < RNN43_HID; ++o) {
        float a = RNN43_B2[o];
        for (int i = 0; i < RNN43_HID; ++i) a += RNN43_W2[o * RNN43_HID + i] * h1[i];
        h2[o] = rnn_maxf(a, 0.0f);
    }
    for (int k = 0; k < 16; ++k) {
        float a = RNN43_B3[k];
        for (int i = 0; i < RNN43_HID; ++i) a += RNN43_W3[k * RNN43_HID + i] * h2[i];
        d[k] = a;
    }
}

// R of one output pixel from its raw pixel's 8-bit analytic value and its correction.
inline float rnn43_weight(float q, float d) {
    float Rc = q < 1e-3f ? 1e-3f : (q > 1.0f - 1e-3f ? 1.0f - 1e-3f : q);
    float z = RNN_LOG(Rc / (1.0f - Rc)) + d;
    return 1.0f / (1.0f + RNN_EXP(-z));
}
