// Parity harness: run core/sr_gate.cpp on a dump from tools/sr_gate/parity.py
// and write its features and mask back out for comparison.
//
// This is the check that makes the training worth anything. train.py fits a
// function of eight features computed in NumPy; the app computes those features
// in C++ and evaluates the same network in C++ and Metal. If the two feature
// builders disagree anywhere -- a rounding mode, an edge clamp, a tile index --
// the weights are fitted to inputs the device never produces.
//
// Build (WinLibs g++ / MinGW-w64 UCRT; every TU needs the SAME -std, see the
// note in tools/rob_refine on why mixing c++17 and gnu++17 corrupts the heap):
//   g++ -O2 -std=gnu++17 -Icore -pthread -c core/sr_gate.cpp -o sr_gate.o
//   g++ -O2 -std=gnu++17 -Icore -pthread -c tools/sr_gate/parity_main.cpp -o pm.o
//   g++ -O2 -std=gnu++17 -static -pthread sr_gate.o pm.o -o sr_gate_parity.exe

#include "sr_gate.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using hhsr::Config;
using hhsr::f32;
using hhsr::FlowField;
using hhsr::Image;

namespace {

bool rd(FILE* f, void* p, size_t n) { return std::fread(p, 1, n, f) == n; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: sr_gate_parity <in.bin> <out.bin>\n");
        return 2;
    }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    char magic[4];
    int32_t hdr[5];
    float ab[2];
    if (!rd(f, magic, 4) || std::memcmp(magic, "SRGD", 4) != 0 ||
        !rd(f, hdr, sizeof(hdr)) || !rd(f, ab, sizeof(ab))) {
        std::fprintf(stderr, "bad header\n");
        return 2;
    }
    const int h = hdr[0], w = hdr[1], ny = hdr[2], nx = hdr[3], ts = hdr[4];
    const float alpha_sensor = ab[0], beta_sensor = ab[1];

    Image ref_means(h, w, 1), ref_vars(h, w, 1), d_sq(h, w, 1), sigma_sq(h, w, 1);
    const size_t n = (size_t)h * w * sizeof(float);
    if (!rd(f, ref_means.data.data(), n) || !rd(f, ref_vars.data.data(), n) ||
        !rd(f, d_sq.data.data(), n) || !rd(f, sigma_sq.data.data(), n)) {
        std::fprintf(stderr, "short read on planes\n");
        return 2;
    }
    FlowField flow(ny, nx);
    if (!rd(f, flow.flow.data(), (size_t)ny * nx * 2 * sizeof(float))) {
        std::fprintf(stderr, "short read on flow\n");
        return 2;
    }
    std::fclose(f);

    // The shipping configuration, which is the domain the gate was trained in.
    Config cfg;
    cfg.bayer_mode = true;
    cfg.grey_method = hhsr::GreyMethod::FFT;
    cfg.robustness_fft_guide = true;          // -> robustness_fft_guide_active()
    cfg.raw_prewhitened = false;             // wb gains already folded in
    cfg.debug_noise_model_disabled = false;
    cfg.num_threads = 0;
    for (int c = 0; c < 3; ++c) {
        cfg.alpha_dng[c] = alpha_sensor;
        cfg.beta_dng[c] = beta_sensor;
    }
    std::printf("cfg alpha_rob %.9g  beta_rob %.9g  (fft energy %.3f)\n",
                (double)cfg.noise_alpha_robustness(),
                (double)cfg.noise_beta_robustness(),
                (double)cfg.fft_guide_noise_energy());
    std::printf("sr_gate_available %d\n", (int)hhsr::sr_gate_available() ? 1 : 0);

    Image feat = hhsr::build_sr_gate_features(ref_means, ref_vars, d_sq, sigma_sq,
                                              flow, ts, cfg);
    if (feat.h != h || feat.w != w || feat.c != hhsr::kSrGateFeatures) {
        std::fprintf(stderr, "feature build failed\n");
        return 1;
    }
    Image mask = hhsr::sr_gate_infer_cpu(feat);
    if (mask.h != h || mask.w != w) {
        std::fprintf(stderr, "inference failed\n");
        return 1;
    }

    FILE* o = std::fopen(argv[2], "wb");
    if (!o) return 2;
    std::fwrite("SRGO", 1, 4, o);
    int32_t oh[3] = {h, w, hhsr::kSrGateFeatures};
    std::fwrite(oh, sizeof(oh), 1, o);
    float rob[2] = {cfg.noise_alpha_robustness(), cfg.noise_beta_robustness()};
    std::fwrite(rob, sizeof(rob), 1, o);
    std::fwrite(feat.data.data(), sizeof(float), feat.data.size(), o);
    std::fwrite(mask.data.data(), sizeof(float), mask.data.size(), o);
    std::fclose(o);
    std::printf("wrote %s  mean mask %.6f\n", argv[2],
                [&] { double s = 0; for (f32 v : mask.data) s += v;
                      return s / (double)mask.data.size(); }());
    return 0;
}
