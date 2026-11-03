// robust_nn.metal — three passes per comparison frame, all at grey resolution.
//   rnn_warp     : J = cmp(p + flow(p))                         (1 float / px scratch)
//   rnn_features : 14 evidence features                          (RNN_NF planes scratch)
//   rnn_infer    : 3x3 conv -> 1x1 -> 1x1 -> sigmoid(logit(r0) + delta)
// Compile the library with fast math OFF (MTLCompileOptions.fastMathEnabled = NO,
// or mathMode = .safe) to stay close to the CPU reference; the header already
// uses precise:: for log/exp/sqrt.
#include "robust_nn_shared.h"

kernel void rnn_warp(const device float* cmp [[buffer(0)]],
                     const device float* flow [[buffer(1)]],
                     device float* J [[buffer(2)]],
                     constant RNNParams& P [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x >= P.width || (int)gid.y >= P.height) return;
    J[gid.y * P.width + gid.x] = rnn_warp_at(RNNImg{cmp, P.width}, RNNFlowInterleaved{flow, P.width},
                                              P.width, P.height, (int)gid.x, (int)gid.y);
}

kernel void rnn_features(const device float* ref [[buffer(0)]],
                         const device float* J [[buffer(1)]],
                         const device float* flow [[buffer(2)]],
                         const device float* r0 [[buffer(3)]],
                         device float* feat [[buffer(4)]],
                         constant RNNParams& P [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x >= P.width || (int)gid.y >= P.height) return;
    float f[RNN_NF];
    rnn_features_at(RNNImg{ref, P.width}, RNNImg{J, P.width}, RNNFlowInterleaved{flow, P.width},
                    r0[gid.y * P.width + gid.x], P, (int)gid.x, (int)gid.y, f);
    const uint plane = (uint)(P.width * P.height);
    const uint i = gid.y * P.width + gid.x;
    for (int c = 0; c < RNN_NF; ++c) feat[c * plane + i] = f[c];
}

kernel void rnn_infer(const device float* feat [[buffer(0)]],
                      const device float* r0 [[buffer(1)]],
                      device float* r_out [[buffer(2)]],
                      constant RNNParams& P [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x >= P.width || (int)gid.y >= P.height) return;
    r_out[gid.y * P.width + gid.x] = rnn_net_at(RNNFeatPlanar{feat, P.width, P.height},
                                                r0[gid.y * P.width + gid.x], P.width, P.height,
                                                (int)gid.x, (int)gid.y);
}
