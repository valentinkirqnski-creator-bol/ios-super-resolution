// Dumps the shared header's per-pixel features (core/robust_nn_shared.h rnn_features_at) for parity against
// rnn/features.py. in.bin: int32 W,H,fs; float a,b,clip; ref[H*W]; J[H*W]; flow[H*W*2] (dx,dy interleaved);
// r0[H*W]  ->  out.bin: float feats[RNN_NF][H][W], then R[H][W] (rnn_net_at). J is passed in (already warped) so only the features are compared.
#include <cstdio>
#include <vector>
#include "robust_nn_shared.h"

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 3;
    int hdr[3];
    float sc[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::fread(sc, 4, 3, f) != 3) return 4;
    const int W = hdr[0], H = hdr[1], fs = hdr[2];
    const size_t n = (size_t)W * H;
    std::vector<float> ref(n), J(n), fl(2 * n), r0(n);
    if (std::fread(ref.data(), 4, n, f) != n || std::fread(J.data(), 4, n, f) != n ||
        std::fread(fl.data(), 4, 2 * n, f) != 2 * n || std::fread(r0.data(), 4, n, f) != n) return 5;
    std::fclose(f);
    RNNParams P = {W, H, fs, sc[0], sc[1], sc[2], 0, 0};
    RNNImg ra = {ref.data(), W}, ja = {J.data(), W};
    RNNFlowInterleaved fa = {fl.data(), W};
    std::vector<float> out((size_t)RNN_NF * n);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float v[RNN_NF];
            rnn_features_at(ra, ja, fa, r0[(size_t)y * W + x], P, x, y, v);
            for (int c = 0; c < RNN_NF; ++c) out[(size_t)c * n + (size_t)y * W + x] = v[c];
        }
    std::vector<float> R(n);
    RNNFeatPlanar fp = {out.data(), W, H};
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) R[(size_t)y * W + x] = rnn_net_at(fp, r0[(size_t)y * W + x], W, H, x, y);
    FILE* o = std::fopen(argv[2], "wb");
    std::fwrite(out.data(), 4, out.size(), o);
    std::fwrite(R.data(), 4, R.size(), o);
    std::fclose(o);
    return 0;
}
