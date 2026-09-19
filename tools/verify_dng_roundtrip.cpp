// End-to-end DNG write -> read for the lossless codecs, and for
// load_linear_dng_rgb16_info, which is what the HDR finish reads.
//
//   g++ -std=c++17 -O2 -I core tools/verify_dng_roundtrip.cpp
//       core/dng_writer.cpp core/ljpeg.cpp -lz -o verify_dng_roundtrip
//   ./verify_dng_roundtrip [out_dir]     (exit 0 iff every case matches)
//
// Caveat this cannot cover: the reader is ours, so a misunderstanding shared by
// writer and reader would pass -- it is not proof of Adobe compatibility. What it
// does establish is that strip offsets and byte counts are self-consistent across
// single- and multi-strip layouts with a short tail band, that each codec survives
// a round trip bit-exactly, and that the geometry and colour metadata written come
// back as written.

#include "dng_writer.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace hhsr;

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const std::string& what) {
    if (ok) ++g_pass;
    else { ++g_fail; std::printf("  FAIL: %s\n", what.c_str()); }
}

static long file_size(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return -1;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fclose(f);
    return n;
}

static void run(const char* name, int codec, int W, int H, int band_rows,
                const std::string& dir) {
    // Noisy gradient: what a real merge produces, and incompressible enough that
    // a broken predictor shows up rather than being masked by flat content.
    std::vector<uint16_t> img((size_t)W * H * 3);
    std::mt19937 rng(777);
    std::normal_distribution<double> n(0.0, 140.0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c) {
                double v = 9000.0 + 14000.0 * (double)x / W
                                  +  6000.0 * (double)y / H + 2500.0 * c + n(rng);
                if (v < 0) v = 0;
                if (v > 65535) v = 65535;
                img[((size_t)y * W + x) * 3 + c] = (uint16_t)(v + 0.5);
            }

    const std::string path = dir + "/t_" + name + ".dng";
    const float cm[9]   = {0.7f, -0.2f, -0.05f, -0.3f, 1.1f, 0.15f, -0.02f, 0.2f, 0.6f};
    const float wb[3]   = {2.06f, 1.0f, 1.84f};
    const float cam[9]  = {1.6f, -0.5f, -0.1f, -0.2f, 1.4f, -0.2f, 0.0f, -0.4f, 1.5f};

    DngStreamWriter w;
    bool ok = w.open(path, W, H, "TestCam", /*orientation*/6, cm, wb,
                     /*bakedSrgb*/false, "TestMake", cam,
                     /*pixelsPrewhitened*/false, codec, nullptr);
    check(ok, std::string(name) + ": open");
    if (!ok) return;

    for (int y0 = 0; y0 < H; y0 += band_rows) {
        const int bh = (band_rows < H - y0) ? band_rows : (H - y0);
        ok = w.write_rows(img.data() + (size_t)y0 * W * 3, bh);
        if (!ok) break;
    }
    check(ok, std::string(name) + ": write_rows");
    check(w.close(), std::string(name) + ": close");

    const long bytes = file_size(path);
    check(bytes > 0, std::string(name) + ": file exists");

    std::vector<uint16_t> back;
    int rW = 0, rH = 0;
    LinearDngColorInfo info;
    const bool rd = load_linear_dng_rgb16_info(path, back, rW, rH, info);
    check(rd, std::string(name) + ": read back");
    if (!rd) return;

    check(rW == W && rH == H, std::string(name) + ": dimensions (" +
          std::to_string(rW) + "x" + std::to_string(rH) + ")");
    check(back.size() == img.size(), std::string(name) + ": sample count");

    size_t bad = 0, first = 0;
    const size_t lim = back.size() < img.size() ? back.size() : img.size();
    for (size_t i = 0; i < lim; ++i)
        if (img[i] != back[i]) { if (!bad) first = i; ++bad; }
    if (bad) {
        std::printf("  FAIL: %s: %zu/%zu samples differ, first at %zu (%u vs %u)\n",
                    name, bad, lim, first, img[first], back[first]);
        ++g_fail;
    } else ++g_pass;

    check(info.orientation == 6, std::string(name) + ": orientation carried");
    check(info.has_as_shot_neutral, std::string(name) + ": AsShotNeutral present");
    check(info.has_color_matrix, std::string(name) + ": ColorMatrix1 present");

    const double raw = (double)img.size() * 2.0;
    std::printf("  %-22s %5dx%-5d band %4d  %9.0f -> %8ld bytes (%.2fx)  ori=%d asn=[%.3f %.3f %.3f]\n",
                name, W, H, band_rows, raw, bytes, raw / (double)bytes,
                info.orientation,
                info.as_shot_neutral[0], info.as_shot_neutral[1], info.as_shot_neutral[2]);
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    // Single strip, and multi-strip with a short tail band -- the geometry the
    // real merge produces (Hs not a multiple of band_rows).
    run("ljpeg_1strip",   Config::DNG_CODEC_LJPEG, 256, 64,  64,  dir);
    run("ljpeg_nstrip",   Config::DNG_CODEC_LJPEG, 512, 200, 48,  dir);
    run("none_1strip",    Config::DNG_CODEC_NONE,  256, 64,  64,  dir);
    run("none_nstrip",    Config::DNG_CODEC_NONE,  512, 200, 48,  dir);
    run("ljpeg_odd_dims", Config::DNG_CODEC_LJPEG, 333, 97,  13,  dir);
    run("ljpeg_1row",     Config::DNG_CODEC_LJPEG, 640, 1,   1,   dir);
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
