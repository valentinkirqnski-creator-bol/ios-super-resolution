// Coordinate-bug test on the device pipeline (process_burst_loader_to_dng).
// Frames are exact crops of ONE real raw frame: frame k = crop at (y0 + dy_k, x0 + dx_k), even shifts (CFA phase
// kept), identical noise. So perfect alignment is the integer shift, robustness must keep everything, and the
// merge must equal the reference-only result up to kernel weighting -- no sub-pixel offset, no doubling.
// usage: coord_test <dng> <out.dng> <mode: A identical | B shifted | C refonly> [gcam] [scale]
#include "pipeline.h"
#include "raw_io.h"
#include "types.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
using namespace hhsr;

int main(int argc, char** argv) {
    if (argc < 4) return 1;
    const std::string path = argv[1], out = argv[2];
    const char mode = argv[3][0];
    const bool gcam = argc > 4 && std::strcmp(argv[4], "gcam") == 0;
    const float scale = argc > 5 ? (float)std::atof(argv[5]) : 2.f;
    Config meta;
    Image full = load_raw_frame(path, meta, true);
    if (full.h <= 0) { std::printf("load failed\n"); return 2; }
    const int S = 1536;
    const int y0 = ((full.h / 2 - S / 2) / 2) * 2, x0 = ((full.w / 2 - S / 2) / 2) * 2;
    const int N = 5;
    const int sh[N][2] = {{0, 0}, {2, 0}, {0, 2}, {4, -6}, {-2, 4}};   // (dy, dx) raw px, even
    auto crop = [&](int dy, int dx) {
        Image im(S, S, 1);
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x) im.at(y, x) = full.at(y0 + dy + y, x0 + dx + x);
        return im;
    };
    RawFrameLoaderFn loader = [&](int index, Config& c, bool is_reference, int, int) {
        if (is_reference) { Image tmp = load_raw_frame(path, c, true); (void)tmp; }
        if (mode == 'A') return crop(0, 0);
        if (mode == 'C' && index != 0) {          // unmergeable: a different far-away region, rotated content
            Image im = crop(0, 0);
            for (int y = 0; y < S; ++y)
                for (int x = 0; x < S; ++x) im.at(y, x) = full.at(64 + (x / 2) * 2 + (y & 1), 64 + (y / 2) * 2 + (x & 1));
            return im;
        }
        return crop(sh[index][0], sh[index][1]);
    };
    Config cfg;
    cfg.scale = scale;
    cfg.dng_codec = Config::DNG_CODEC_NONE;
    cfg.dng_store_unwhitened = false;
    cfg.robustness_gcam_enabled = gcam;
    if (std::getenv("CT_DEBUG")) cfg.robustness_save_mask = true;
    std::printf("mode %c gcam %d scale %.1f frame %dx%d crop at %d,%d\n", mode, (int)gcam, scale, full.w, full.h, x0, y0);
    Image prev = process_burst_loader_to_dng(N, loader, cfg, out, nullptr, 256);
    std::printf("done %dx%d\n", prev.w, prev.h);
    return prev.h > 0 ? 0 : 3;
}
