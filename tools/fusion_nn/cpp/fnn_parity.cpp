// Parity + timing of core/fusion_nn.cpp (the app's CPU network) against tools/fusion_nn.
//   fnn_parity <in.bin> <out.bin>     in: int32 gh, gw, is_ref, threads; f32 a, b;
//                                         f32 ref_gat, frame_gat, mean_gat [gh][gw][3]
//                                     out: f32 logits [gh][gw]; stderr: seconds
#include "fusion_nn.h"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace hhsr;

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    int hdr[4];
    float ab[2];
    if (std::fread(hdr, 4, 4, f) != 4 || std::fread(ab, 4, 2, f) != 2) return 3;
    const int gh = hdr[0], gw = hdr[1];
    Image g[3];
    for (auto& im : g) {
        im = Image(gh, gw, 3);
        if (std::fread(im.data.data(), 4, im.data.size(), f) != im.data.size()) return 4;
    }
    std::fclose(f);
    const auto t0 = std::chrono::steady_clock::now();
    Image l = fusion_nn_logits(g[0], g[1], g[2], hdr[2] != 0, ab[0], ab[1], hdr[3]);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "%.3f\n", s);
    FILE* o = std::fopen(argv[2], "wb");
    std::fwrite(l.data.data(), 4, l.data.size(), o);
    std::fclose(o);
    return 0;
}
