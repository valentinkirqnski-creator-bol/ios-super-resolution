// End-to-end check of the sr_gate hook through the REAL pipeline entry points.
//
// parity.py proves core/sr_gate.cpp computes the trained function. This proves
// the hook in compute_robustness actually reaches it, in the shipping
// configuration, with the Config plumbing and the guide that the pipeline
// itself builds -- not a dump prepared by a Python script.
//
// Reads a burst dumped by integration_dump.py (raw planes only; everything else
// is computed here by the pipeline's own code) and prints the mean mask, the
// fraction fully trusted and the fraction fully rejected for the analytic mask
// and for the gate.
//
// Build (same -std for every TU; see tools/rob_refine on why mixing them
// corrupts the heap):
//   g++ -O2 -std=gnu++17 -Icore -Itools/rob_refine -pthread -c \
//       core/robustness.cpp core/grey_pyramid.cpp core/sr_gate.cpp \
//       tools/rob_refine/host_stubs.cpp tools/sr_gate/integration_main.cpp
//   g++ -O2 -std=gnu++17 -static -pthread *.o -o tools/sr_gate/sr_gate_integ.exe

#include "stages.h"
#include "sr_gate.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace hhsr;

namespace {

bool rd(FILE* f, void* p, size_t n) { return std::fread(p, 1, n, f) == n; }

struct Stats {
    double mean = 0, frac_one = 0, frac_zero = 0;
    int nan_count = 0;
};

Stats mask_stats(const Image& r) {
    Stats s;
    double acc = 0;
    size_t one = 0, zero = 0;
    for (f32 v : r.data) {
        if (!(v == v)) {
            ++s.nan_count;
            continue;
        }
        acc += v;
        if (v > 0.99f) ++one;
        if (v < 0.01f) ++zero;
    }
    const double n = (double)r.data.size();
    s.mean = acc / n;
    s.frac_one = (double)one / n;
    s.frac_zero = (double)zero / n;
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: sr_gate_integ <burst.bin>\n");
        return 2;
    }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    char magic[4];
    int32_t hdr[5];
    float ab[2];
    if (!rd(f, magic, 4) || std::memcmp(magic, "SRGB", 4) != 0 ||
        !rd(f, hdr, sizeof(hdr)) || !rd(f, ab, sizeof(ab))) {
        std::fprintf(stderr, "bad header\n");
        return 2;
    }
    const int h = hdr[0], w = hdr[1], ny = hdr[2], nx = hdr[3], ts = hdr[4];
    Image ref(h, w, 1), comp(h, w, 1);
    const size_t plane = (size_t)h * w * sizeof(float);
    FlowField flow(ny, nx);
    if (!rd(f, ref.data.data(), plane) || !rd(f, comp.data.data(), plane) ||
        !rd(f, flow.flow.data(), (size_t)ny * nx * 2 * sizeof(float))) {
        std::fprintf(stderr, "short read\n");
        return 2;
    }
    std::fclose(f);

    // The shipping configuration, as core/types.h pins it.
    Config cfg;
    cfg.bayer_mode = true;
    cfg.scale = 2.f;
    cfg.grey_method = GreyMethod::FFT;
    cfg.robustness_enabled = true;
    // The shipping configuration as of defaultsVersion 14: raw-resolution OFF,
    // so compute_guide builds the three-channel half-resolution sqrt guide.
    cfg.robustness_raw_resolution_enabled = false;
    cfg.robustness_guide_sqrt = true;
    cfg.guide_curve = 1;
    cfg.raw_prewhitened = false;
    cfg.num_threads = 0;
    for (int c = 0; c < 3; ++c) {
        cfg.alpha_dng[c] = ab[0];
        cfg.beta_dng[c] = ab[1];
    }
    std::printf("guide: fft_active=%d raw_res_active=%d sqrt_active=%d\n",
                (int)cfg.robustness_fft_guide_active(),
                (int)cfg.robustness_raw_resolution_active(),
                (int)cfg.robustness_guide_sqrt_active());
    std::printf("alpha_rob %.9g beta_rob %.9g\n",
                (double)cfg.noise_alpha_robustness(),
                (double)cfg.noise_beta_robustness());
    std::printf("sr_gate_available %d\n", (int)sr_gate_available());

    RefStats st = init_robustness(ref, cfg);
    if (st.means.h != h / 2 || st.means.w != w / 2 || st.means.c != 3) {
        std::fprintf(stderr, "guide is %dx%dx%d, expected %dx%dx3 -- the "
                     "shipping predicate did not select the FFT guide\n",
                     st.means.h, st.means.w, st.means.c, h / 2, w / 2);
        return 1;
    }
    std::printf("guide %dx%d x%d ok\n", st.means.h, st.means.w, st.means.c);

    cfg.sr_gate_enabled = false;
    Image r_analytic = compute_robustness(comp, st, flow, ts, cfg, nullptr);
    cfg.sr_gate_enabled = true;
    Image r_gate = compute_robustness(comp, st, flow, ts, cfg, nullptr);
    // The mask lives on the GUIDE lattice, which is half the raw size on the
    // three-channel decimated guide.
    if (r_analytic.h != st.means.h || r_gate.h != st.means.h) {
        std::fprintf(stderr, "mask shape wrong: analytic %dx%d gate %dx%d, "
                     "guide %dx%d\n", r_analytic.h, r_analytic.w, r_gate.h,
                     r_gate.w, st.means.h, st.means.w);
        return 1;
    }

    const Stats a = mask_stats(r_analytic);
    const Stats g = mask_stats(r_gate);
    std::printf("\n%-10s %-9s %-9s %-9s %-6s\n", "mask", "mean", "R>0.99", "R<0.01",
                "NaN");
    std::printf("%-10s %-9.4f %-9.4f %-9.4f %-6d\n", "analytic", a.mean,
                a.frac_one, a.frac_zero, a.nan_count);
    std::printf("%-10s %-9.4f %-9.4f %-9.4f %-6d\n", "sr_gate", g.mean,
                g.frac_one, g.frac_zero, g.nan_count);

    // Did the hook actually change anything? A gate that silently declined
    // would return the analytic mask and look fine.
    double max_diff = 0;
    for (size_t i = 0; i < r_gate.data.size(); ++i) {
        const f32 x = r_gate.data[i], y = r_analytic.data[i];
        if (x == x && y == y) max_diff = std::fmax(max_diff, std::fabs(x - y));
    }
    std::printf("\nmax |gate - analytic| = %.4f  -> hook %s\n", max_diff,
                max_diff > 1e-4 ? "FIRED" : "DID NOT FIRE");

    // Dump the gate mask so parity against torch can be checked on THIS path.
    FILE* o = std::fopen(argc > 2 ? argv[2] : "integ_mask.bin", "wb");
    if (o) {
        std::fwrite("SRGM", 1, 4, o);
        int32_t oh[2] = {r_gate.h, r_gate.w};
        std::fwrite(oh, sizeof(oh), 1, o);
        std::fwrite(r_gate.data.data(), sizeof(float), r_gate.data.size(), o);
        std::fwrite(r_analytic.data.data(), sizeof(float), r_analytic.data.size(), o);
        std::fclose(o);
    }
    return max_diff > 1e-4 ? 0 : 1;
}
