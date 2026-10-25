// frame_dump in.dng out.f32 : one frame through the app's own loader (core/raw_io.cpp load_raw_frame: black-subtracted,
// white = 1, pre-white-balanced as the app delivers it), written as float32 H x W. Prints
// "H W cfa00 cfa01 cfa10 cfa11 alpha beta wb0 wb1 wb2 prewhitened exposure" (exposure = shutter x ISO, 0 = unknown).
#include <cstdio>
#include "raw_io.h"
using namespace hhsr;
int main(int argc, char** argv) {
    if (argc < 3) return 1;
    Config cfg;
    Image r = load_raw_frame(argv[1], cfg, true);
    if (r.h <= 0) return 2;
    FILE* f = std::fopen(argv[2], "wb");
    if (!f) return 3;
    std::fwrite(r.data.data(), 4, r.data.size(), f);
    std::fclose(f);
    std::printf("%d %d %d %d %d %d %g %g %g %g %g %d %g\n", r.h, r.w, cfg.cfa.p[0][0], cfg.cfa.p[0][1],
                cfg.cfa.p[1][0], cfg.cfa.p[1][1], cfg.alpha_dng[1], cfg.beta_dng[1], cfg.white_balance[0],
                cfg.white_balance[1], cfg.white_balance[2], (int)cfg.raw_prewhitened, (double)r.exposure);
    return 0;
}
