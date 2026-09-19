// Render check for core/finish_hdr.cpp -- the HDR finish behind the JPG export.
//
//   g++ -std=c++17 -O2 -I core tools/verify_finish_hdr.cpp
//       core/finish_hdr.cpp core/dng_writer.cpp core/ljpeg.cpp -lz -o verify_finish_hdr
//   ./verify_finish_hdr [out_dir]        (exit 0 iff every check passes)
//
// Writes synthetic LinearRaw DNGs with the real writer and renders them with the
// real finish, so the whole path runs. Checking a return code is not the point;
// these are the properties that were actually wrong at one time or another:
//
//   * NEUTRAL STAYS NEUTRAL. The un-white-balanced container -- what the app
//     writes by default, since dng_store_unwhitened and raw_prewhitened are both
//     true -- stores a neutral grey as (g/2.06, g, g/1.84). Every per-pixel
//     statistic has to allow for that. The black level is the loud one: measured
//     as a low percentile of the per-pixel MINIMUM channel it reads the red level
//     of the darkest content, and subtracting that from all three sends red to
//     zero and the shadows green.
//   * BLOWN HIGHLIGHTS STAY NEUTRAL TO THE RIM. The clip detector reads cam_max
//     from the stored sample and the chroma denoiser runs first and writes that
//     buffer back. The interior of a blown region is safe -- the blur returns its
//     own value -- so only a scan of the rim catches the ring where the flag was
//     lost and the pink came back.
//   * THE TWO CONTAINERS AGREE. The same scene stored camera-native or
//     pre-balanced must render to the same picture; that is what the column-scale
//     reasoning in finish_hdr.h is for. This is the check that fails loudest when
//     a measurement is made in the wrong space, because the prewhitened half is
//     unaffected by such a bug and the native half is not.
//
// Plus geometry edges, degenerate content (all black, all white, low key with no
// true black to find), EV monotonicity, determinism, and the failure paths.

#include "dng_writer.h"
#include "finish_hdr.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace hhsr;

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const std::string& what) {
    if (ok) ++g_pass;
    else { ++g_fail; std::printf("  FAIL: %s\n", what.c_str()); }
}

static const float kWB[3] = {2.06f, 1.0f, 1.84f};   // real gains for this sensor

enum Content { kRamp, kAllBlack, kAllWhite, kLowKey, kNeutralStep, kBlownDisc };

// Camera-native samples. `peak` keeps the ramp clear of full scale so the
// prewhitened variant can multiply the gains in without clipping.
static std::vector<uint16_t> make_pixels(int W, int H, Content content,
                                         bool clipped_patch, double peak) {
    std::vector<uint16_t> img((size_t)W * H * 3);
    std::mt19937 rng(4242);
    std::normal_distribution<double> n(0.0, 60.0);
    const double dx = (W > 1) ? (double)(W - 1) : 1.0;
    const double dy = (H > 1) ? (double)(H - 1) : 1.0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const double fx = (double)x / dx;
            const double fy = (double)y / dy;
            for (int c = 0; c < 3; ++c) {
                double v;
                switch (content) {
                    case kAllBlack: v = 0.0; break;
                    case kAllWhite: v = 65535.0; break;
                    // Deliberately no black content and no highlight: black_max
                    // and display_black_max must both refuse to invent one.
                    // Divided by the gain, so the scene really is neutral grey:
                    // equal stored channels would be a violently magenta scene
                    // once the 2.06 / 1.00 / 1.84 gains go on.
                    case kLowKey:   v = (1400.0 + 500.0 * fx + n(rng)) / kWB[c]; break;
                    // A hard step between two neutral greys, no noise. Neutral
                    // in must be neutral out; any channel spread in the render
                    // is a tint some measurement invented.
                    case kNeutralStep:
                        v = ((fx < 0.5) ? 3000.0 : 30000.0) / kWB[c];
                        break;
                    // A sensor-blown disc on neutral ground. Curved, so the
                    // boundary is sampled at every orientation rather than only
                    // axis-aligned.
                    case kBlownDisc: {
                        const double rx = (fx - 0.5) * (W > 1 ? W : 1);
                        const double ry = (fy - 0.5) * (H > 1 ? H : 1);
                        const double rad = 0.28 * (double)((W < H) ? W : H);
                        v = (std::sqrt(rx * rx + ry * ry) <= rad) ? 65535.0
                                                                  : 20000.0 / kWB[c];
                        break;
                    }
                    default:
                        v = (120.0 + peak * fx * fx) * (1.0 + 0.25 * (c - 1) * fy) / kWB[c]
                            + n(rng);
                        break;
                }
                if (clipped_patch && fx > 0.75 && fy < 0.25) v = 65535.0;
                if (v < 0) v = 0;
                if (v > 65535) v = 65535;
                img[((size_t)y * W + x) * 3 + c] = (uint16_t)(v + 0.5);
            }
        }
    }
    return img;
}

static bool write_dng(const std::string& path, const std::vector<uint16_t>& img,
                      int W, int H, int orientation, bool prewhitened) {
    const float cm[9]  = {0.7f, -0.2f, -0.05f, -0.3f, 1.1f, 0.15f, -0.02f, 0.2f, 0.6f};
    const float cam[9] = {1.6f, -0.5f, -0.1f, -0.2f, 1.4f, -0.2f, 0.0f, -0.4f, 1.5f};
    DngStreamWriter w;
    if (!w.open(path, W, H, "TestCam", orientation, cm, kWB, /*bakedSrgb*/false,
                "TestMake", cam, prewhitened, Config::DNG_CODEC_LJPEG, nullptr))
        return false;
    const int band = 32;
    for (int y0 = 0; y0 < H; y0 += band) {
        const int bh = (band < H - y0) ? band : (H - y0);
        if (!w.write_rows(img.data() + (size_t)y0 * W * 3, bh)) return false;
    }
    return w.close();
}

struct Stats { double mean; int mn, mx; double pct0, pct255; };
static Stats stats_of(const std::vector<uint8_t>& rgb) {
    Stats s{0.0, 255, 0, 0.0, 0.0};
    long a0 = 0, a255 = 0;
    double sum = 0;
    for (uint8_t v : rgb) {
        sum += v;
        if (v < s.mn) s.mn = v;
        if (v > s.mx) s.mx = v;
        if (v == 0) ++a0;
        if (v == 255) ++a255;
    }
    s.mean = sum / (double)rgb.size();
    s.pct0 = 100.0 * a0 / (double)rgb.size();
    s.pct255 = 100.0 * a255 / (double)rgb.size();
    return s;
}

// Largest channel spread among the bright pixels of the clipped patch: a
// sensor-blown site must render neutral, not pink.
static int clip_spread(const std::vector<uint8_t>& rgb, int W, int H, long& found) {
    int worst = 0;
    found = 0;
    const int rows = (H / 5 > 0) ? H / 5 : 1;
    for (int y = 0; y < rows; ++y) {
        for (int x = (int)(W * 0.80); x < W; ++x) {
            const size_t o = ((size_t)y * W + x) * 3;
            const int r = rgb[o], g = rgb[o + 1], b = rgb[o + 2];
            const int hi = r > g ? (r > b ? r : b) : (g > b ? g : b);
            const int lo = r < g ? (r < b ? r : b) : (g < b ? g : b);
            if (hi > 180) { if (hi - lo > worst) worst = hi - lo; ++found; }
        }
    }
    return worst;
}

static int max_spread(const std::vector<uint8_t>& rgb) {
    int worst = 0;
    for (size_t i = 0; i + 2 < rgb.size(); i += 3) {
        const int r = rgb[i], g = rgb[i + 1], b = rgb[i + 2];
        const int hi = r > g ? (r > b ? r : b) : (g > b ? g : b);
        const int lo = r < g ? (r < b ? r : b) : (g < b ? g : b);
        if (hi - lo > worst) worst = hi - lo;
    }
    return worst;
}

struct Expect { bool highlights = true; bool shadows = true; bool range = true; };

static bool render(const char* name, int W, int H, Content content, bool clipped,
                   bool prewhitened, int orientation, const FinishHdrParams& p,
                   const std::string& dir, std::vector<uint8_t>& out,
                   double peak = 40000.0) {
    const std::string path = dir + "/hdr_" + name + ".dng";
    std::vector<uint16_t> img = make_pixels(W, H, content, clipped, peak);
    if (prewhitened)
        for (size_t i = 0; i < img.size(); ++i) {
            const double v = img[i] * (double)kWB[i % 3];
            img[i] = (uint16_t)(v > 65535.0 ? 65535.0 : v + 0.5);
        }
    if (!write_dng(path, img, W, H, orientation, prewhitened)) {
        check(false, std::string(name) + ": write DNG");
        return false;
    }
    int rW = 0, rH = 0, ori = 0;
    const bool ok = finish_hdr_from_dng(path, p, out, rW, rH, ori);
    check(ok, std::string(name) + ": finish_hdr_from_dng");
    if (!ok) return false;
    check(rW == W && rH == H, std::string(name) + ": dimensions");
    check(out.size() == (size_t)W * H * 3, std::string(name) + ": buffer size");
    check(ori == orientation, std::string(name) + ": orientation carried");
    return out.size() == (size_t)W * H * 3;
}

static void run(const char* name, int W, int H, Content content, bool clipped,
                const FinishHdrParams& p, const std::string& dir,
                Expect e = Expect{}, bool prewhitened = false, int orientation = 6) {
    std::vector<uint8_t> rgb;
    if (!render(name, W, H, content, clipped, prewhitened, orientation, p, dir, rgb)) return;

    const Stats s = stats_of(rgb);
    if (e.highlights) check(s.mx > 200, std::string(name) +
        ": reaches highlights (max " + std::to_string(s.mx) + ")");
    if (e.shadows) check(s.mn < 40, std::string(name) +
        ": reaches shadows (min " + std::to_string(s.mn) + ")");
    if (e.range) check(s.mean > 20.0 && s.mean < 235.0, std::string(name) +
        ": mean in range (" + std::to_string((int)s.mean) + ")");
    check(s.pct255 < 40.0, std::string(name) + ": not blown out");

    std::string spread = "n/a";
    if (clipped) {
        long found = 0;
        const int worst = clip_spread(rgb, W, H, found);
        spread = std::to_string(worst);
        check(found > 0, std::string(name) + ": found bright clipped samples");
        check(worst <= 24, std::string(name) +
              ": clipped highlights neutral (spread " + spread + ")");
    }
    std::printf("  %-18s %5dx%-5d mean %6.1f  min %3d  max %3d  %6.2f%%@0 %6.2f%%@255  clipspread %s\n",
                name, W, H, s.mean, s.mn, s.mx, s.pct0, s.pct255, spread.c_str());
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    const FinishHdrParams def;

    std::printf("-- content and geometry --\n");
    run("default",  320, 240, kRamp, true,  def, dir);
    run("noclip",   320, 240, kRamp, false, def, dir);
    run("tiny",       8,   8, kRamp, false, def, dir);
    run("row",      256,   1, kRamp, false, def, dir);
    run("column",     1, 256, kRamp, false, def, dir);
    run("odd_dims", 333,  97, kRamp, true,  def, dir);
    run("ori1",     320, 240, kRamp, false, def, dir, Expect{}, false, 1);

    std::printf("-- degenerate content --\n");
    {
        // Must not crash and must not invent content that is not there.
        Expect e; e.highlights = false; e.range = false;
        run("all_black", 128, 128, kAllBlack, false, def, dir, e);
    }
    {
        // A uniform frame renders at the auto key -- mid grey -- because its
        // log-average IS its own value. That is what an auto key does; what
        // matters here is that it stays neutral and does not go black.
        Expect e; e.shadows = false; e.highlights = false;
        run("all_white", 128, 128, kAllWhite, false, def, dir, e);
        std::vector<uint8_t> rgb;
        if (render("all_white_hue", 128, 128, kAllWhite, false, false, 6, def, dir, rgb)) {
            const int worst = max_spread(rgb);
            check(worst <= 8, "all_white: neutral (spread " + std::to_string(worst) + ")");
            std::printf("     all_white channel spread %d\n", worst);
        }
    }
    {
        // Low-key: no true black in the scene, so black_max / display_black_max
        // must refuse to crush the shadows to manufacture one.
        // Low-key by definition: no true black AND no highlight, so neither end
        // of the range is expected to be reached.
        Expect e; e.shadows = false; e.highlights = false;
        run("low_key", 256, 192, kLowKey, false, def, dir, e);
        std::vector<uint8_t> rgb;
        if (render("low_key_floor", 256, 192, kLowKey, false, false, 6, def, dir, rgb)) {
            const Stats s = stats_of(rgb);
            check(s.pct0 < 2.0, "low_key: shadows not crushed (" +
                  std::to_string(s.pct0) + "% at 0)");
        }
    }

    std::printf("-- neutral stays neutral --\n");
    {
        // The un-white-balanced container stores a neutral grey as
        // (g/2.06, g, g/1.84). Every per-pixel statistic -- the black level, the
        // chroma opponent split -- has to allow for that, or neutral content
        // comes back tinted. The black level is the loud one: measured as a low
        // percentile of the per-pixel MINIMUM channel it reads the red level of
        // the darkest content, and subtracting that from all three sends red to
        // zero and the shadows green.
        struct Case { const char* name; bool prewhitened; float denoise; };
        const Case cases[] = {
            {"native, denoise off",  false, 0.f},
            {"native, denoise 0.40", false, 0.40f},
            {"prewhitened",          true,  0.40f},
        };
        for (const Case& c : cases) {
            FinishHdrParams pp = def;
            pp.chroma_denoise = c.denoise;
            std::vector<uint8_t> rgb;
            const char* tag = c.prewhitened ? "step_pw"
                            : (c.denoise > 0.f ? "step_dn" : "step");
            if (!render(tag, 256, 128, kNeutralStep, false, c.prewhitened, 1, pp, dir, rgb))
                continue;
            const int worst = max_spread(rgb);
            check(worst <= 2, std::string("neutral step stays neutral (") + c.name +
                  "): spread " + std::to_string(worst));
            std::printf("     %-22s worst channel spread %d\n", c.name, worst);
        }
    }

    std::printf("-- blown highlights stay neutral to the rim --\n");
    {
        // The clip detector reads cam_max from the STORED sample, and the chroma
        // denoiser runs first and writes that buffer back. Inside a large blown
        // region the blur returns the region own value and nothing moves, but
        // within a filter radius of the edge it mixes in the surroundings -- and
        // if that drops a stored channel below clip_threshold the neutral pull
        // switches off for those pixels alone, leaving a magenta ring one radius
        // wide. The interior alone does not catch it; the rim has to be scanned.
        for (int variant = 0; variant < 2; ++variant) {
            FinishHdrParams pp = def;
            pp.chroma_denoise = variant ? 0.40f : 0.f;
            std::vector<uint8_t> rgb;
            const int DW = 256, DH = 256;
            if (!render(variant ? "disc_dn" : "disc", DW, DH, kBlownDisc, false,
                        false, 1, pp, dir, rgb))
                continue;
            const double cx = DW * 0.5, cy = DH * 0.5;
            const double rad = 0.28 * DW;
            int worst_in = 0, worst_rim = 0;
            for (int y = 0; y < DH; ++y)
                for (int x = 0; x < DW; ++x) {
                    const size_t o = ((size_t)y * DW + x) * 3;
                    const int r = rgb[o], g = rgb[o + 1], b = rgb[o + 2];
                    const int hi = r > g ? (r > b ? r : b) : (g > b ? g : b);
                    const int lo = r < g ? (r < b ? r : b) : (g < b ? g : b);
                    const double d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                    if (d <= rad - 20.0) { if (hi - lo > worst_in) worst_in = hi - lo; }
                    else if (d <= rad + 20.0) { if (hi - lo > worst_rim) worst_rim = hi - lo; }
                }
            check(worst_in <= 2, std::string("blown disc interior neutral (denoise ") +
                  (variant ? "0.40" : "off") + "): spread " + std::to_string(worst_in));
            check(worst_rim <= 4, std::string("blown disc RIM neutral (denoise ") +
                  (variant ? "0.40" : "off") + "): spread " + std::to_string(worst_rim));
            std::printf("     denoise %-4s  interior %d  rim %d\n",
                        variant ? "0.40" : "off", worst_in, worst_rim);
        }
    }

    std::printf("-- parameters --\n");
    FinishHdrParams flat = def;
    flat.shadow_lift = 0.f; flat.highlight_rolloff = 0.f; flat.local_strength = 0.f;
    flat.contrast = 0.f; flat.vibrance = 0.f; flat.chroma_denoise = 0.f;
    run("no_tonemap", 320, 240, kRamp, true, flat, dir);

    FinishHdrParams strong = def;
    strong.shadow_lift = 1.f; strong.highlight_rolloff = 1.f; strong.local_strength = 1.f;
    strong.contrast = 1.f; strong.vibrance = 1.f; strong.saturation = 1.5f;
    strong.chroma_denoise = 1.f;
    run("strong", 320, 240, kRamp, true, strong, dir);

    FinishHdrParams ev = def; ev.exposure_ev = 2.f;
    run("ev_plus2", 320, 240, kRamp, true, ev, dir);
    ev.exposure_ev = -2.f;
    {
        Expect e; e.highlights = false;
        run("ev_minus2", 320, 240, kRamp, true, ev, dir, e);
    }

    // EV must be monotone.
    {
        std::vector<uint8_t> lo, mid, hi;
        FinishHdrParams a = def; a.exposure_ev = -1.f;
        FinishHdrParams b = def; b.exposure_ev = +1.f;
        if (render("ev_m1", 200, 150, kRamp, false, false, 6, a, dir, lo) &&
            render("ev_0",  200, 150, kRamp, false, false, 6, def, dir, mid) &&
            render("ev_p1", 200, 150, kRamp, false, false, 6, b, dir, hi)) {
            check(stats_of(lo).mean < stats_of(mid).mean, "EV: -1 darker than 0");
            check(stats_of(mid).mean < stats_of(hi).mean, "EV: 0 darker than +1");
        }
    }

    std::printf("-- containers --\n");
    {
        // The same scene stored camera-native vs pre-balanced must render alike.
        // peak is held well under full scale / max(gain) so the prewhitened
        // variant does not clip -- the two containers would then hold different
        // scenes and the comparison would be meaningless.
        const double kSafePeak = 24000.0;
        FinishHdrParams nocd = def;
        nocd.chroma_denoise = 0.f;
        for (int variant = 0; variant < 2; ++variant) {
            const FinishHdrParams& pp = variant ? nocd : def;
            const char* tag = variant ? "chroma_denoise=0" : "defaults       ";
            std::vector<uint8_t> native, white;
            if (!render(variant ? "cnat2" : "cnat", 256, 192, kRamp, false, false, 6,
                        pp, dir, native, kSafePeak) ||
                !render(variant ? "cwht2" : "cwht", 256, 192, kRamp, false, true, 6,
                        pp, dir, white, kSafePeak) ||
                native.size() != white.size())
                continue;
            double sad = 0.0;
            int worst = 0;
            for (size_t i = 0; i < native.size(); ++i) {
                const int d = std::abs((int)native[i] - (int)white[i]);
                sad += d;
                if (d > worst) worst = d;
            }
            const double mae = sad / (double)native.size();
            std::printf("     %s: MAE %.2f, worst %d\n", tag, mae, worst);
            check(mae < 1.5, std::string("container parity (") + tag +
                  "): MAE " + std::to_string(mae));
            check(worst <= 8, std::string("container parity (") + tag +
                  "): worst " + std::to_string(worst));
        }
        run("prewhitened", 320, 240, kRamp, false, def, dir, Expect{}, true);
    }

    std::printf("-- determinism --\n");
    {
        std::vector<uint8_t> a, b;
        if (render("det_a", 200, 150, kRamp, true, false, 6, def, dir, a) &&
            render("det_b", 200, 150, kRamp, true, false, 6, def, dir, b))
            check(a == b, "two renders of the same DNG are identical");
    }

    std::printf("-- failure modes --\n");
    {
        std::vector<uint8_t> rgb;
        int W = 1, H = 1, ori = 1;
        check(!finish_hdr_from_dng(dir + "/does_not_exist.dng", def, rgb, W, H, ori),
              "missing file returns false");
        check(rgb.empty() && W == 0 && H == 0, "missing file leaves outputs cleared");
        const std::string junk = dir + "/junk.dng";
        FILE* f = std::fopen(junk.c_str(), "wb");
        if (f) { std::fputs("not a tiff at all, not even close", f); std::fclose(f); }
        check(!finish_hdr_from_dng(junk, def, rgb, W, H, ori), "garbage file returns false");
    }

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
