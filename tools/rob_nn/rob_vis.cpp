// Dump what the masks actually see and decide, on a REAL burst, so the result
// can be looked at rather than only scored.
//
// Writes, per comparison frame and at guide resolution:
//   <out>_fNN.feat   the 18 feature planes, interleaved float32
//   <out>_fNN.an     the analytic mask (Wronski Eq. 5-9), float32
//   <out>_ref.rgb    the reference guide image, float32 RGB, for context
// plus a text sidecar with the dimensions.
//
// The learned mask is NOT run here: there is no Core ML runtime on the machine
// this is built for, and reimplementing the network in C++ purely to visualise
// it would be a second implementation to keep in step. tools/rob_nn/vis_rob.py
// loads these planes, runs the trained checkpoint, and writes the images.
#include "stages.h"
#include "parallel.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace hhsr;

namespace {
bool write_f32(const std::string& path, const Image& im) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(im.data.data(), sizeof(f32), im.data.size(), f);
    std::fclose(f);
    return true;
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        std::printf("usage: rob_vis out_prefix ref.dng comp1.dng [comp2.dng ...]\n"
                    "  env: ROB_GAIN (1) emulated 1/k exposure\n");
        return 1;
    }
    const std::string out = argv[1];
    float gain = 1.f;
    if (const char* v = std::getenv("ROB_GAIN")) gain = (float)std::atof(v);

    Config cfg;
    cfg.scale = 2.f; cfg.bayer_mode = true; cfg.grey_method = GreyMethod::Decimate;
    cfg.r_t = 0.12f; cfg.r_s1 = 2.0f; cfg.r_s2 = 12.0f; cfg.r_Mt = 0.8f;
    cfg.num_threads = 0; cfg.alignment_tile_size = 16;

    Image ref = load_raw_frame(argv[2], cfg, true, 0, 0);
    if (ref.w <= 0) { std::printf("decode failed\n"); return 1; }
    Config work = cfg;
    work.burst_frame_count = argc - 2;
    tune_config_snr(ref, work);
    work.r_t = cfg.r_t; work.r_s1 = cfg.r_s1; work.r_s2 = cfg.r_s2;
    work.use_neural_robustness = true;
    const int ts = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];

    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
    RefStats rs = init_robustness(ref, work);
    ensure_robustness_nn_ref_hf(rs, work);
    if (rs.means.data.empty()) { std::printf("ref stats empty\n"); return 1; }
    const int gh = rs.means.h, gw = rs.means.w;

    Image ref_guide = compute_guide(ref, work);
    write_f32(out + "_ref.rgb", ref_guide);
    std::printf("guide %dx%d, tile %d, %d channels\n", gw, gh, ts, kRobustnessNnChannels);

    for (int a = 3; a < argc; ++a) {
        Image comp = load_raw_frame(argv[a], cfg, true, 0, 0);
        if (comp.w <= 0) continue;
        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
        FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, ts, 0.f, 0.f, 0.f);
        flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w, ts,
                                     work.r_Mt, work.num_threads, work.grey_tile_size(ts));

        // The features exactly as the app would build them, whole plane.
        Config w2 = work;
        w2.use_neural_robustness = false;      // so this call takes the analytic path
        Image an = compute_robustness(comp, rs, flow, ts, w2, nullptr);

        RefStats cs = init_robustness(comp, work);
        // Channels 18-19. Must be measured here too: leaving them zero would
        // feed the network a constant on two inputs it was trained to use, and
        // the failure would look like a mysteriously bad mask rather than a
        // missing feature.
        const std::vector<f32> mq =
            measure_match_quality(ref_grey, comp_grey, flow, ts, work);
        Image feat = build_robustness_nn_features(rs, cs.means, flow, ts, work, 0,
                                                  /*raw_res=*/false, /*rows=*/gh,
                                                  /*planar=*/false, &mq);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "_f%02d", a - 3);
        write_f32(out + buf + ".feat", feat);
        write_f32(out + buf + ".an", an);
        double m = 0; size_t n = 0;
        for (size_t i = 0; i < an.data.size(); ++i) { m += an.data[i]; ++n; }
        std::printf("  frame %d: analytic mask mean %.3f\n", a - 3, m / (double)n);
    }

    if (FILE* f = std::fopen((out + ".dims").c_str(), "w")) {
        std::fprintf(f, "h %d\nw %d\nfeat_c %d\nframes %d\n",
                     gh, gw, kRobustnessNnChannels, argc - 3);
        std::fclose(f);
    }
    std::printf("wrote %s_*\n", out.c_str());
    return 0;
}
