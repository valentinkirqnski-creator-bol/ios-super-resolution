// Does core/sr_gate_unet.cpp compute the same function PyTorch trained?
//
// The weights are exported as a flat blob and indexed by hand in C++ and again
// in Metal. A transposed filter, an off-by-one offset or a wrong concatenation
// order all produce a plausible-looking mask and no error, so the forward pass
// is checked numerically against the framework that fitted it rather than
// eyeballed.
//
//   in:  i32 h, i32 w, i32 c, f32 tau, f32 beta, then h*w*c f32 features
//   out: h*w f32 mask
//
// tools/sr_gate/unet_parity.py writes the input, runs this, and compares.
#include "sr_gate.h"
#include "types.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace hhsr;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: unet_parity in.bin out.bin\n");
        return 2;
    }
    std::FILE* fi = std::fopen(argv[1], "rb");
    if (!fi) return 1;
    int32_t h = 0, w = 0, c = 0;
    float tau = 1.f, beta = 0.5f;
    if (std::fread(&h, 4, 1, fi) != 1 || std::fread(&w, 4, 1, fi) != 1 ||
        std::fread(&c, 4, 1, fi) != 1 || std::fread(&tau, 4, 1, fi) != 1 ||
        std::fread(&beta, 4, 1, fi) != 1) {
        std::printf("short header\n");
        return 1;
    }
    if (c != SRGU_IN) {
        std::printf("expected %d channels, got %d\n", SRGU_IN, c);
        return 1;
    }
    Image feat(h, w, c);
    if (std::fread(feat.data.data(), sizeof(f32),
                   (size_t)h * w * c, fi) != (size_t)h * w * c) {
        std::printf("short features\n");
        return 1;
    }
    std::fclose(fi);

    if (!sr_gate_unet_available()) {
        std::printf("weights unavailable\n");
        return 1;
    }
    Image R = sr_gate_unet_infer_cpu(feat, tau, beta);
    if (R.h != h || R.w != w) {
        std::printf("inference returned %dx%d\n", R.h, R.w);
        return 1;
    }
    std::FILE* fo = std::fopen(argv[2], "wb");
    if (!fo) return 1;
    std::fwrite(R.data.data(), sizeof(f32), (size_t)h * w, fo);
    std::fclose(fo);
    return 0;
}
