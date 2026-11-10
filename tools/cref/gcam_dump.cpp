// Dump the GCam CPU pieces (guide, prefiltered guide, noise LUT, dilate, dilate 2x, full CPU
// operator mask) for a real frame pair, plus timings, to verify optimisations are bit-identical.
// usage: gcam_dump <ref.dng> <comp.dng> <out.bin>
#include "stages.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <chrono>
#include <cstdio>
#include <vector>
using namespace hhsr;
static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    Config cfg;
    Image ref = load_raw_frame(argv[1], cfg, true);
    Config c2 = cfg;
    Image comp = load_raw_frame(argv[2], c2, false);
    cfg.burst_frame_count = 8;
    tune_config_snr(ref, cfg);
    cfg.robustness_gcam_enabled = true;
    const int ts = 16;
    FILE* fo = std::fopen(argv[3], "wb");
    auto put = [&](const char* name, const Image& im, double ms) {
        std::printf("%-28s %8.1f ms  %dx%dx%d\n", name, ms, im.w, im.h, im.c);
        std::fwrite(im.data.data(), 4, im.data.size(), fo);
    };
    double t;
    for (int nm = 0; nm < 2; ++nm) {
        cfg.gcam_sabre_noise_model = nm == 1;
        t = now(); GcamNoiseLut lut = build_gcam_noise_lut(cfg);
        std::printf("noise lut (model %d)          %8.1f ms\n", nm, now() - t);
        std::fwrite(lut.v, 4, 80, fo);
        t = now(); Image g0 = compute_gcam_guide(comp, cfg, false); put(nm ? "guide (nm)" : "guide", g0, now() - t);
        t = now(); Image g1 = compute_gcam_guide(ref, cfg, true); put(nm ? "guide prefilter (nm)" : "guide prefilter", g1, now() - t);
    }
    cfg.gcam_sabre_noise_model = false;
    // A synthetic flow (shift + noise) so the operator produces a non-trivial mask.
    FlowField f(comp.h / ts, comp.w / ts);
    for (int t2 = 0; t2 < f.ny * f.nx; ++t2) { f.flow[2 * t2] = (float)((t2 * 7919) % 13) * 0.25f - 1.5f; f.flow[2 * t2 + 1] = (float)((t2 * 104729) % 11) * 0.2f - 1.f; }
    RefStats rs = init_robustness(ref, cfg);
    t = now(); Image R = compute_robustness(comp, rs, f, ts, cfg); put("CPU operator mask", R, now() - t);
    t = now(); Image d = robustness_gcam_dilate(R, cfg); put("dilate", d, now() - t);
    Config c3 = cfg; c3.gcam_sabre_dilate_2x = true;
    t = now(); Image d2 = robustness_gcam_dilate(R, c3); put("dilate 2x", d2, now() - t);
    t = now(); Image m = robustness_gcam_min_window(R, 5); put("min window 5", m, now() - t);
    std::fclose(fo);
    return 0;
}
