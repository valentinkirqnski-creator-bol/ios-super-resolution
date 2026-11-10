// Time the per-frame CPU pieces of the GCam robustness path on a real 12MP frame pair.
#include "stages.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include "fb_confidence.h"
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
    const int ts = cfg.bm_tile_sizes.empty() ? 16 : cfg.bm_tile_sizes[0];
    std::printf("frame %dx%d threads %d raw_res %d noise_model %d structure_match %d\n", ref.w, ref.h, cfg.num_threads,
                (int)cfg.gcam_sabre_raw_res, (int)cfg.gcam_sabre_noise_model, (int)cfg.gcam_structure_match);
    double t = now();
    RefStats rs = init_robustness(ref, cfg);
    std::printf("init_robustness (ref)        %8.1f ms\n", now() - t);
    t = now(); Image rg = compute_grey(ref, cfg.bayer_mode, cfg.grey_method);
    Image cg = compute_grey(comp, cfg.bayer_mode, cfg.grey_method);
    std::printf("compute_grey x2              %8.1f ms\n", now() - t);
    Pyramid pyr = build_pyramid(pad_image_circular(rg, cfg.grey_tile_size(ts)), cfg.bm_factors);
    t = now(); FlowField f = align(pyr, rg, cg, cfg, ts); clear_align_ref_ica_cache();
    f = flow_to_raw_tile_grid(f, comp.h, comp.w, cg.h, cg.w, ts, cfg.r_Mt, cfg.num_threads, cfg.grey_tile_size(ts));
    median3x3_tile_flow(f);
    std::printf("align (CPU)                  %8.1f ms\n", now() - t);
    t = now(); GcamNoiseLut lut = build_gcam_noise_lut(cfg); (void)lut;
    std::printf("build_gcam_noise_lut         %8.1f ms\n", now() - t);
    t = now(); Image g = compute_gcam_guide(comp, cfg, true);
    std::printf("compute_gcam_guide (comp)    %8.1f ms\n", now() - t);
    t = now(); std::vector<f32> irr = compute_flow_irregularity(f);
    std::printf("compute_flow_irregularity    %8.1f ms\n", now() - t);
    t = now(); Image R = compute_robustness(comp, rs, f, ts, cfg);
    std::printf("compute_robustness (CPU op)  %8.1f ms  -> %dx%d\n", now() - t, R.w, R.h);
    t = now(); Image d = robustness_gcam_dilate(R, cfg);
    std::printf("robustness_gcam_dilate       %8.1f ms\n", now() - t);
    std::vector<uint8_t> rows; double af = 0; (void)rows; (void)af;
    t = now();
    std::printf("robustness_row_activity      %8.1f ms\n", now() - t);
    t = now(); std::vector<float> e = fb_tile_error(rg, cg, f, comp.h, comp.w, ts, cfg);
    std::printf("fb_tile_error (4x4)          %8.1f ms\n", now() - t);
    t = now(); fb_apply_confidence(&R, nullptr, e, f, comp.h, comp.w, ts, cfg);
    std::printf("fb_apply_confidence          %8.1f ms\n", now() - t);
    t = now(); CovField cv = estimate_kernels(comp, cfg);
    std::printf("estimate_kernels (CPU)       %8.1f ms\n", now() - t);
    return 0;
}
