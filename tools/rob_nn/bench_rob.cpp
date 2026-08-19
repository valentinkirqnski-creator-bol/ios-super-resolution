// What does the learned mask actually COST per comparison frame?
//
// The budget is 200 ms for EVERYTHING the mask adds to a burst, not 200 ms of
// Core ML. Inference is the part that is easy to think about and, on the
// evidence below, not the part that dominates. This tool times the pieces that
// run on the CPU and are therefore measurable on any host:
//
//   1. compute_guide + local_stats_3x3 on the comparison frame -- the mask's
//      own copy of the comparison statistics, per frame
//   2. ensure_robustness_nn_ref_hf -- feature channel 17, ONCE per burst
//   3. build_robustness_nn_features -- the 18 planes, per frame, including the
//      per-pixel robustness_analytic_R behind channels 13-14
//   4. the strip loop's halo overlap, which rebuilds kRobustnessNnHalo rows
//      twice at every strip boundary
//   5. the interleaved -> NCHW transpose the Core ML wrapper performs while
//      filling MLMultiArray. Not Core ML's own time, but time the mask spends,
//      and it moves C*H*W floats through a strided write pattern.
//
// What it CANNOT measure is the Apple Neural Engine, so the inference number is
// absent rather than guessed. Anything reported here is host wall-clock on an
// x86 CPU and is NOT an iPhone figure; it bounds the CPU-side work and shows
// which term dominates, which is the actionable part.
#include "stages.h"
#include "parallel.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace hhsr;

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::printf("usage: bench_rob ref.dng comp.dng [reps]\n");
        return 1;
    }
    const int reps = (argc > 3) ? std::atoi(argv[3]) : 3;

    Config cfg;
    cfg.scale = 2.f; cfg.bayer_mode = true; cfg.grey_method = GreyMethod::Decimate;
    cfg.r_t = 0.12f; cfg.r_s1 = 2.0f; cfg.r_s2 = 12.0f; cfg.r_Mt = 0.8f;
    cfg.num_threads = 0; cfg.alignment_tile_size = 16;

    Image ref = load_raw_frame(argv[1], cfg, true, 0, 0);
    Image comp = load_raw_frame(argv[2], cfg, true, 0, 0);
    if (ref.w <= 0 || comp.w <= 0) { std::printf("decode failed\n"); return 1; }
    Config work = cfg;
    work.burst_frame_count = 2;
    tune_config_snr(ref, work);
    work.r_t = cfg.r_t; work.r_s1 = cfg.r_s1; work.r_s2 = cfg.r_s2;
    work.use_neural_robustness = true;
    const int ts = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];

    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
    Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
    FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, ts, 0.f, 0.f, 0.f);
    flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w, ts,
                                 work.r_Mt, work.num_threads, work.grey_tile_size(ts));

    RefStats rs = init_robustness(ref, work);
    if (rs.means.data.empty()) { std::printf("ref stats empty\n"); return 1; }
    const int gh = rs.means.h, gw = rs.means.w;
    std::printf("guide %dx%d (%.2f MP), %d feature channels, %d threads\n",
                gw, gh, gh * gw / 1e6, kRobustnessNnChannels,
                work.num_threads ? work.num_threads : -1);

    double t_hf = 0, t_stats = 0, t_feat = 0, t_fill = 0;

    // Warm-up, timed separately and EXCLUDED from the per-frame numbers.
    //
    // The first call to robustness_analytic_R builds the Monte Carlo noise
    // curves for the three guide channels. That is a large one-time cost, it
    // is paid once per (alpha, beta) rather than once per frame, and the
    // ANALYTIC mask pays exactly the same cost -- so it is not something the
    // learned mask adds. Folding it into the per-frame average is how this
    // bench first reported 2425 ms/frame for work that actually takes 175:
    // the giveaway was that reps x reported was constant.
    {
        auto t0 = Clock::now();
        Image cw;
        RefStats cwarm = init_robustness(comp, work);
        cw = std::move(cwarm.means);
        build_robustness_nn_features(rs, cw, flow, ts, work, 0, false);
        std::printf("one-time noise-curve build + first call: %.0f ms "
                    "(shared with the analytic mask, not added by the model)\n",
                    ms_since(t0));
    }

    for (int r = 0; r < reps; ++r) {
        {   // once per burst
            RefStats tmp = rs;
            tmp.nn_hf = Image();
            auto t0 = Clock::now();
            ensure_robustness_nn_ref_hf(tmp, work);
            t_hf += ms_since(t0);
        }
        ensure_robustness_nn_ref_hf(rs, work);

        Image cm;
        {   // Per comparison frame. init_robustness is compute_guide followed
            // by the same 3x3 local statistics the mask path builds for the
            // comparison frame (its other outputs are off in this Config), so
            // timing it times the real code rather than a copy of it --
            // local_stats_3x3 itself is file-static in robustness.cpp.
            auto t0 = Clock::now();
            RefStats cs = init_robustness(comp, work);
            t_stats += ms_since(t0);
            cm = std::move(cs.means);
        }
        if (cm.h != gh || cm.w != gw) { std::printf("comp stats shape mismatch\n"); return 1; }

        // The real strip loop, halo overlap included.
        const int strip_h = kRobustnessNnStripRows + 2 * kRobustnessNnHalo;
        std::vector<f32> nchw((size_t)kRobustnessNnChannels * strip_h * gw);
        auto t0 = Clock::now();
        double fill = 0;
        for (int y0 = 0; y0 < gh; y0 += kRobustnessNnStripRows) {
            const int top = std::min(std::max(y0 - kRobustnessNnHalo, 0), gh - strip_h);
            if (gh < strip_h) break;
            Image feat = build_robustness_nn_features(rs, cm, flow, ts, work, top, false);
            // The transpose the Core ML wrapper does on the way in. Written the
            // same way -- channel-major over an interleaved source -- so the
            // cache behaviour is representative.
            auto t1 = Clock::now();
            const f32* src = feat.data.data();
            const size_t C = kRobustnessNnChannels, H = feat.h, W = feat.w;
            for (size_t c = 0; c < C; ++c)
                for (size_t y = 0; y < H; ++y) {
                    f32* dst = nchw.data() + (c * H + y) * W;
                    const f32* sp = src + (y * W) * C + c;
                    for (size_t x = 0; x < W; ++x) dst[x] = sp[x * C];
                }
            fill += ms_since(t1);
        }
        t_feat += ms_since(t0) - fill;
        t_fill += fill;
    }

    const double n = reps;
    std::printf("\nper comparison frame (host CPU wall clock, NOT an iPhone number)\n");
    std::printf("  comparison guide + 3x3 stats      %7.1f ms\n", t_stats / n);
    std::printf("  build_robustness_nn_features      %7.1f ms   (all strips, halo included)\n", t_feat / n);
    std::printf("  interleaved -> NCHW transpose     %7.1f ms   (Core ML input fill)\n", t_fill / n);
    std::printf("  ------------------------------------------\n");
    std::printf("  CPU-side total per frame          %7.1f ms\n",
                (t_stats + t_feat + t_fill) / n);
    std::printf("  once per burst: channel 17 cache  %7.1f ms\n", t_hf / n);
    std::printf("\n  Core ML inference is NOT included -- it runs on the ANE and\n");
    std::printf("  cannot be measured on this host. The numbers above are the\n");
    std::printf("  work the mask does BEFORE the model is even called.\n");
    const double per_burst5 = 5.0 * (t_stats + t_feat + t_fill) / n + t_hf / n;
    std::printf("\n  a 6-frame burst (5 comparison frames) would spend %.0f ms\n", per_burst5);
    std::printf("  of CPU on this host, against a 200 ms budget for everything.\n");
    return 0;
}
