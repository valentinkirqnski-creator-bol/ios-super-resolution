// dng_dump in.dng out.f32 : the app's own decode (core/raw_io.cpp), black-subtracted,
// white = 1, written as float32 H x W; prints "H W cfa00 cfa01 cfa10 cfa11 alpha beta wb0 wb1 wb2".
#include <cstdio>
#include "raw_io.h"
using namespace hhsr;
int main(int argc, char** argv) {
    Config cfg;
    Image r = load_raw_frame(argv[1], cfg, true);
    FILE* f = std::fopen(argv[2], "wb");
    std::fwrite(r.data.data(), 4, r.data.size(), f);
    std::fclose(f);
    std::printf("%d %d %d %d %d %d %g %g %g %g %g %d\n", r.h, r.w, cfg.cfa.p[0][0], cfg.cfa.p[0][1], cfg.cfa.p[1][0],
                cfg.cfa.p[1][1], cfg.alpha_dng[1], cfg.beta_dng[1], cfg.white_balance[0], cfg.white_balance[1],
                cfg.white_balance[2], (int)cfg.raw_prewhitened);
    return 0;
}
