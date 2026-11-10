// Run the (mechanically translated) Metal rob_gcam_guide + rob_gcam_prefilter on the CPU and compare with
// compute_gcam_guide(comp, cfg, true), for both noise models. usage: gguide_test <ref.dng> <comp.dng>
#include "stages.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
using namespace hhsr;
#include "gguide_emu.inc"

int main(int argc, char** argv) {
    Config cfg;
    Image ref = load_raw_frame(argv[1], cfg, true);
    Config c2 = cfg;
    Image comp = load_raw_frame(argv[2], c2, false);
    cfg.burst_frame_count = 8;
    tune_config_snr(ref, cfg);
    cfg.robustness_gcam_enabled = true;
    for (int nm = 0; nm < 2; ++nm) {
        cfg.gcam_sabre_noise_model = nm == 1;
        const Image cpu = compute_gcam_guide(comp, cfg, true);
        const uint32_t gh = comp.h / 2, gw = comp.w / 2;
        GcamNoiseLut lut = cfg.gcam_sabre_noise_model ? build_gcam_noise_lut(cfg) : GcamNoiseLut{};
        GcamGuideParams p{};
        p.gh = gh; p.gw = gw; p.raw_w = comp.w; p.use_mc = nm;
        p.cfa00 = cfg.cfa.p[0][0]; p.cfa01 = cfg.cfa.p[0][1]; p.cfa10 = cfg.cfa.p[1][0]; p.cfa11 = cfg.cfa.p[1][1];
        p.green_factor = cfg.gcam_sabre_green_factor; p.clip_point = cfg.gcam_sabre_clip_point;
        p.na = cfg.noise_alpha_robustness(); p.nb = cfg.noise_beta_robustness();
        std::vector<float> g0((size_t)gh * gw * 4), out((size_t)gh * gw * 4);
        for (uint32_t y = 0; y < gh; ++y)
            for (uint32_t x = 0; x < gw; ++x) rob_gcam_guide(g0.data(), comp.data.data(), lut.v, p, U2{x, y});
        const U2 dims{gw, gh};
        for (uint32_t y = 0; y < gh; ++y)
            for (uint32_t x = 0; x < gw; ++x) rob_gcam_prefilter(out.data(), g0.data(), dims, U2{x, y});
        size_t ndiff = 0; double maxrel = 0;
        for (size_t i = 0; i < out.size(); ++i)
            if (out[i] != cpu.data[i]) {
                ++ndiff;
                maxrel = std::max(maxrel, (double)std::fabs(out[i] - cpu.data[i]) / (1e-12 + std::fabs(cpu.data[i])));
            }
        std::printf("noise model %d: guide %ux%ux4  values differing %zu of %zu  max rel diff %.3g\n", nm, gw, gh, ndiff,
                    out.size(), maxrel);
    }
    return 0;
}
