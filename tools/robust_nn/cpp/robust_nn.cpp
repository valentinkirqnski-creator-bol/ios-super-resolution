#include "robust_nn.h"

#include <algorithm>
#include <cstring>

namespace robust_nn {

void refine(const float* ref, const float* cmp, const float* flow_xy, const float* r0, const RNNParams& P,
            float* r_out, Scratch* scratch, float* feat_out, const float* feat_flow_xy) {
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
            rnn_features_at(RNNImg{ref, W}, RNNImg{s.J.data(), W},
                            RNNFlowInterleaved{feat_flow_xy ? feat_flow_xy : flow_xy, W},
                            r0[(size_t)y * W + x], P, x, y, f);
            for (int c = 0; c < RNN_NF; ++c) s.feat[c * plane + (size_t)y * W + x] = f[c];
        }

#if RNN_CTX > 0
    // v31 tile branch, as the app's rnn_tile_pool / rnn_tile_conv passes: mean and max of the features over
    // each flow tile (P.flow_step px, grid origin 0, in-image pixels only) -> K1, then 3x3 over the tile grid -> K2.
    const int fs = std::max(1, P.flow_step);
    const int nxg = (W + fs - 1) / fs, nyg = (H + fs - 1) / fs;
    std::vector<float> tileA((size_t)nxg * nyg * RNN_CTX), tileB((size_t)nxg * nyg * RNN_CTX);
#pragma omp parallel for schedule(static)
    for (int ty = 0; ty < nyg; ++ty)
        for (int tx = 0; tx < nxg; ++tx) {
            float avg[RNN_NF], mx[RNN_NF];
            for (int c = 0; c < RNN_NF; ++c) { avg[c] = 0.f; mx[c] = -1e30f; }
            int n = 0;
            for (int y = ty * fs; y < std::min(H, (ty + 1) * fs); ++y)
                for (int x = tx * fs; x < std::min(W, (tx + 1) * fs); ++x, ++n)
                    for (int c = 0; c < RNN_NF; ++c) {
                        float v = s.feat[c * plane + (size_t)y * W + x];
                        avg[c] += v;
                        mx[c] = rnn_maxf(mx[c], v);
                    }
            for (int c = 0; c < RNN_NF; ++c) avg[c] /= (float)std::max(n, 1);
            rnn_tile_summary(avg, mx, &tileA[((size_t)ty * nxg + tx) * RNN_CTX]);
        }
#pragma omp parallel for schedule(static)
    for (int ty = 0; ty < nyg; ++ty)
        for (int tx = 0; tx < nxg; ++tx)
            rnn_tile_conv_at(RNNTileA{tileA.data(), nxg}, nxg, nyg, tx, ty, &tileB[((size_t)ty * nxg + tx) * RNN_CTX]);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            r_out[(size_t)y * W + x] = rnn_net_tile_at(RNNFeatPlanar{s.feat.data(), W, H},
                                                       &tileB[((size_t)(y / fs) * nxg + x / fs) * RNN_CTX],
                                                       r0[(size_t)y * W + x], W, H, x, y);
#else
#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            r_out[(size_t)y * W + x] =
                rnn_net_at(RNNFeatPlanar{s.feat.data(), W, H}, r0[(size_t)y * W + x], W, H, x, y);
#endif

    if (feat_out) std::memcpy(feat_out, s.feat.data(), plane * RNN_NF * sizeof(float));
}

}  // namespace robust_nn
