// Merge a real burst with a chosen robustness mask, and dump both the merged
// image and the mask that produced it.
//
// The point is to see the mask's consequence, not only its score. Two modes:
//
//   analytic   compute_robustness as the app runs it (Wronski Eq. 5-9)
//   learned    per-frame masks read from <prefix>_maskNN.f32, guide
//              resolution, one float per pixel -- written by vis_rob.py after
//              running the trained checkpoint over the dumped features
//
// The learned mask is fed in from a file rather than run here because there is
// no Core ML runtime on this host, and reimplementing the network in C++ to
// visualise it would be a second implementation to keep in step with the one
// that ships. The alignment is deterministic and identical in both tools, so
// the masks line up with the frames they were built from.
//
// Outputs (float32, raw dumps; tools/rob_nn/vis_rob.py renders the PNGs):
//   <prefix>_merged.f32   scale*h x scale*w x 3
//   <prefix>_accmask.f32  guide h x w, MEAN of R over the comparison frames
//   <prefix>_merge.dims   dimensions and which mask was used
#include "stages.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace hhsr;

namespace {
bool write_f32(const std::string& p, const std::vector<f32>& v) {
    FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) return false;
    std::fwrite(v.data(), sizeof(f32), v.size(), f);
    std::fclose(f);
    return true;
}
bool read_f32(const std::string& p, std::vector<f32>& v, size_t n) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    v.resize(n);
    const size_t got = std::fread(v.data(), sizeof(f32), n, f);
    std::fclose(f);
    return got == n;
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 5) {
        std::printf("usage: rob_merge out_prefix analytic|learned ref.dng comp1.dng ...\n"
                    "  env: ROB_SCALE (1)\n");
        return 1;
    }
    const std::string out = argv[1];
    const std::string mode = argv[2];
    const bool learned = (mode == "learned");
    float scale = 1.f;
    if (const char* v = std::getenv("ROB_SCALE")) scale = (float)std::atof(v);
    // The gate the app applies. This tool rendered UNGATED until now, so every
    // image judged by eye showed behaviour the pipeline does not have: the
    // merge never sees a value below the gate. Defaults to Config's own value
    // so the picture and the app cannot drift apart.
    Config gate_cfg;
    float gate = gate_cfg.rob_nn_gate;
    if (const char* v = std::getenv("ROB_GATE")) gate = (float)std::atof(v);

    Config cfg;
    cfg.scale = scale;
    cfg.bayer_mode = true;
    cfg.grey_method = GreyMethod::Decimate;
    cfg.r_t = 0.12f; cfg.r_s1 = 2.0f; cfg.r_s2 = 12.0f; cfg.r_Mt = 0.8f;
    cfg.num_threads = 0; cfg.alignment_tile_size = 16;

    std::vector<Image> burst;
    for (int a = 3; a < argc; ++a) {
        Image im = load_raw_frame(argv[a], cfg, true, 0, 0);
        if (im.w <= 0) { std::printf("decode failed %s\n", argv[a]); return 1; }
        burst.push_back(std::move(im));
    }
    const int n = (int)burst.size();
    const Image& ref = burst[0];

    Config work = cfg;
    work.burst_frame_count = n;
    tune_config_snr(ref, work);
    work.r_t = cfg.r_t; work.r_s1 = cfg.r_s1; work.r_s2 = cfg.r_s2;
    work.scale = scale;
    const int ts = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];

    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
    RefStats ref_stats = init_robustness(ref, work);
    CovField ref_covs = estimate_kernels(ref, work);

    const int Hs = (int)std::lround(work.scale * ref.h);
    const int Ws = (int)std::lround(work.scale * ref.w);
    const int nch = work.bayer_mode ? 3 : 1;
    Image num(Hs, Ws, nch), den(Hs, Ws, nch);

    const int gh = ref.h / 2, gw = ref.w / 2;
    std::vector<f32> acc((size_t)gh * gw, 0.f);
    Image acc_rob(gh, gw, 1);
    int merged = 0;

    for (int k = 1; k < n; ++k) {
        Image comp_grey = compute_grey(burst[k], work.bayer_mode, work.grey_method);
        FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, ts);
        flow = flow_to_raw_tile_grid(flow, burst[k].h, burst[k].w,
                                     comp_grey.h, comp_grey.w, ts,
                                     work.r_Mt, work.num_threads,
                                     work.grey_tile_size(ts));
        Image rob;
        if (learned) {
            char nb[32];
            std::snprintf(nb, sizeof(nb), "_mask%02d.f32", k - 1);
            std::vector<f32> m;
            if (!read_f32(out + nb, m, (size_t)gh * gw)) {
                // Refuse rather than quietly fall back. A silent fallback to the
                // analytic mask is exactly what cost real debugging time before:
                // the output looks plausible and is not what it claims to be.
                std::printf("MISSING %s%s -- run vis_rob.py first. Refusing to "
                            "fall back to the analytic mask.\n", out.c_str(), nb);
                return 1;
            }
            rob = Image(gh, gw, 1);
            rob.data = std::move(m);
        } else {
            Config wa = work;
            wa.use_neural_robustness = false;
            rob = compute_robustness(burst[k], ref_stats, flow, ts, wa);
        }
        if (rob.h != gh || rob.w != gw) {
            std::printf("mask %dx%d != guide %dx%d\n", rob.w, rob.h, gw, gh);
            return 1;
        }
        double m = 0;
        size_t gated = 0;
        for (size_t i = 0; i < rob.data.size(); ++i) {
            f32 r = rob.data[i];
            if (!std::isfinite(r)) r = 0.f;         // see README: the CPU
            r = std::min(std::max(r, 0.f), 1.f);    // analytic path emits inf
            // Below the gate the pixel is not merged at all; at or above it
            // the mask's own value is kept, so the rolloff survives inside the
            // trusted band. Exactly what compute_robustness does. Applied to
            // the LEARNED mask only -- the analytic mask has no such gate in
            // the pipeline, and adding one here would flatter it.
            if (learned && gate > 0.f && r < gate) { r = 0.f; ++gated; }
            rob.data[i] = r;
            acc[i] += r;
            m += r;
        }
        if (learned && gate > 0.f)
            std::printf("    gate %.4f rejected %.1f%% of this frame\n",
                        gate, 100.0 * (double)gated / (double)rob.data.size());
        std::printf("  frame %d: mask mean %.3f\n", k, m / (double)acc.size());
        CovField covs = estimate_kernels(burst[k], work);
        merge_comp(burst[k], flow, covs, rob, ts, num, den, work);
        ++merged;
    }
    for (size_t i = 0; i < acc.size(); ++i) acc[i] /= (float)std::max(merged, 1);
    acc_rob.data.assign(acc.begin(), acc.end());
    // merge_ref takes the SUM of the comparison masks, matching pipeline.cpp.
    Image acc_sum(gh, gw, 1);
    for (size_t i = 0; i < acc.size(); ++i) acc_sum.data[i] = acc[i] * (f32)merged;
    merge_ref(ref, ref_covs, num, den, work, &acc_sum);

    std::vector<f32> outimg((size_t)Hs * Ws * nch);
    for (size_t i = 0; i < outimg.size(); ++i) {
        const f32 d = den.data[i];
        outimg[i] = (d > 0.f) ? num.data[i] / d : 0.f;
    }
    write_f32(out + "_merged.f32", outimg);
    write_f32(out + "_accmask.f32", acc);
    if (FILE* f = std::fopen((out + "_merge.dims").c_str(), "w")) {
        std::fprintf(f, "h %d\nw %d\nc %d\nmask_h %d\nmask_w %d\nmask %s\nframes %d\n",
                     Hs, Ws, nch, gh, gw, learned ? "learned" : "analytic", merged);
        std::fclose(f);
    }
    double am = 0;
    for (float v : acc) am += v;
    size_t rej = 0;
    for (float v : acc) if (v < 0.5f) ++rej;
    std::printf("%s mask%s, %d comparison frames merged, accumulated mask mean "
                "%.3f, rejected (acc R < 0.5) %.1f%%\n",
                learned ? "LEARNED" : "ANALYTIC",
                (learned && gate > 0.f) ? " (gated)" : "", merged,
                am / (double)acc.size(), 100.0 * (double)rej / (double)acc.size());
    std::printf("wrote %s_merged.f32 (%dx%dx%d) and %s_accmask.f32 (%dx%d)\n",
                out.c_str(), Ws, Hs, nch, out.c_str(), gw, gh);
    return 0;
}
