// FB confidence on a REAL burst (no injected slips): per comparison frame, the app's flow
// (+ GCam median), the GCam / Wronski masks, the local and full round-trip errors, and how
// much the confidence actually darkens the mask (where it flags vs where the mask already is).
// usage: fb_diag <burst_dir> [n_frames=3] [tau=0.2]
#include "fb_confidence.h"
#include "stages.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
using namespace hhsr;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    for (auto& e : fs::directory_iterator(argv[1]))
        if (e.path().extension() == ".dng" || e.path().extension() == ".DNG") paths.push_back(e.path().string());
    std::sort(paths.begin(), paths.end());
    const int nf = argc > 2 ? std::atoi(argv[2]) : 3;
    const float tau = argc > 3 ? (float)std::atof(argv[3]) : 0.2f;
    Config cfg;
    Image ref = load_raw_frame(paths[0], cfg, true);
    cfg.burst_frame_count = (int)paths.size();
    tune_config_snr(ref, cfg);
    cfg.fb_conf_tau_px = tau;
    const int ts = cfg.bm_tile_sizes.empty() ? 16 : cfg.bm_tile_sizes[0];
    Image rg = compute_grey(ref, cfg.bayer_mode, cfg.grey_method);
    Pyramid pyr = build_pyramid(pad_image_circular(rg, cfg.grey_tile_size(ts)), cfg.bm_factors);
    for (int gc = 0; gc < 2; ++gc) {
        Config cv = cfg;
        cv.robustness_gcam_enabled = gc == 1;
        RefStats rs = init_robustness(ref, cv);
        for (int k = 1; k <= nf && k < (int)paths.size(); ++k) {
            Config c2 = cv;
            Image comp = load_raw_frame(paths[k], c2, false);
            Image cg = compute_grey(comp, cv.bayer_mode, cv.grey_method);
            FlowField f = align(pyr, rg, cg, cv, ts);
            clear_align_ref_ica_cache();
            f = flow_to_raw_tile_grid(f, comp.h, comp.w, cg.h, cg.w, ts, cv.r_Mt, cv.num_threads, cv.grey_tile_size(ts));
            if (gc) median3x3_tile_flow(f);
            Image R = compute_robustness(comp, rs, f, ts, cv);
            for (int full = 0; full < 4; ++full) {
                Config cf = cv; cf.fb_conf_full = full == 3;
                const int subs[4] = {1, 2, 4, 1};
                cf.fb_conf_subdiv = subs[full];
                std::vector<float> e = fb_tile_error(rg, cg, f, comp.h, comp.w, ts, cf);
                Image R2 = R;
                fb_apply_confidence(&R2, nullptr, e, f, comp.h, comp.w, ts, cf);
                // per-tile stats
                std::vector<float> es = e; std::sort(es.begin(), es.end());
                size_t n03 = 0, n1 = 0, nzero = 0;
                for (float v : e) { n03 += v > 0.3f; n1 += v > 1.f; nzero += v == 0.f; }
                double m0 = 0, m1 = 0, flag_m0 = 0, flag_m1 = 0, kept_new = 0; size_t nflag = 0, nkept = 0;
                const int sub = cf.fb_conf_subdiv; const int pnx = f.nx * sub; const float ps = (float)ts / sub;
                const float ky = (float)comp.h / R.h, kx = (float)comp.w / R.w;
                for (int y = 0; y < R.h; ++y)
                    for (int x = 0; x < R.w; ++x) {
                        const float a = R.at(y, x), b = R2.at(y, x);
                        m0 += a; m1 += b;
                        const int ty = std::min(f.ny * sub - 1, (int)(y * ky / ps)), tx = std::min(pnx - 1, (int)(x * kx / ps));
                        if (e[(size_t)ty * pnx + tx] > 0.3f) { flag_m0 += a; flag_m1 += b; ++nflag; }
                        if (a > 0.5f) { ++nkept; kept_new += a - b; }
                    }
                const double np = (double)R.h * R.w;
                std::printf("%s frame %d %s: tiles e=0 %.0f%%  e>0.3 %.1f%%  e>1 %.1f%%  e p90 %.2f p99 %.2f | mask mean %.3f -> %.3f"
                            " | in flagged (e>0.3) px: mask %.3f -> %.3f\n",
                            gc ? "GCam   " : "Wronski", k, full == 3 ? "FULL  1x1" : (full == 0 ? "local 1x1" : (full == 1 ? "local 2x2" : "local 4x4")), 100.0 * nzero / e.size(),
                            100.0 * n03 / e.size(), 100.0 * n1 / e.size(), es[es.size() * 9 / 10], es[es.size() * 99 / 100],
                            m0 / np, m1 / np, nflag ? flag_m0 / nflag : 0.0, nflag ? flag_m1 / nflag : 0.0, nkept ? 100.0 * kept_new / nkept : 0.0);
            }
        }
    }
    return 0;
}
