#include "robust_nn.h"

#include <cstring>

namespace robust_nn {

void refine(const float* ref, const float* cmp, const float* flow_xy, const float* r0, const RNNParams& P,
            float* r_out, Scratch* scratch, float* feat_out) {
    Scratch local;
    Scratch& s = scratch ? *scratch : local;
    const int W = P.width, H = P.height;
    const size_t plane = (size_t)W * H;
    s.J.resize(plane);
    s.feat.resize(plane * RNN_NF);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            s.J[(size_t)y * W + x] = rnn_warp_at(RNNImg{cmp, W}, RNNFlowInterleaved{flow_xy, W}, W, H, x, y);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float f[RNN_NF];
            rnn_features_at(RNNImg{ref, W}, RNNImg{s.J.data(), W}, RNNFlowInterleaved{flow_xy, W},
                            r0[(size_t)y * W + x], P, x, y, f);
            for (int c = 0; c < RNN_NF; ++c) s.feat[c * plane + (size_t)y * W + x] = f[c];
        }

#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            r_out[(size_t)y * W + x] =
                rnn_net_at(RNNFeatPlanar{s.feat.data(), W, H}, r0[(size_t)y * W + x], W, H, x, y);

    if (feat_out) std::memcpy(feat_out, s.feat.data(), plane * RNN_NF * sizeof(float));
}

}  // namespace robust_nn
