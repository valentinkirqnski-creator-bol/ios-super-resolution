// Flow of exact even-shifted crops of one real frame (app aligner, as the dataset tool / pipeline stages).
#include "stages.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <algorithm>
#include <cstdio>
#include <vector>
using namespace hhsr;

int main(int argc, char** argv) {
    Config cfg;
    Image full = load_raw_frame(argv[1], cfg, true);
    const int S = 1536;
    const int y0 = ((full.h / 2 - S / 2) / 2) * 2, x0 = ((full.w / 2 - S / 2) / 2) * 2;
    auto crop = [&](int dy, int dx) {
        Image im(S, S, 1);
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x) im.at(y, x) = full.at(y0 + dy + y, x0 + dx + x);
        return im;
    };
    cfg.burst_frame_count = 5;
    Image ref = crop(0, 0);
    tune_config_snr(ref, cfg);
    const int ts = cfg.bm_tile_sizes.empty() ? 16 : cfg.bm_tile_sizes[0];
    Image rg = compute_grey(ref, cfg.bayer_mode, cfg.grey_method);
    Pyramid pyr = build_pyramid(pad_image_circular(rg, cfg.grey_tile_size(ts)), cfg.bm_factors);
    std::printf("grey %dx%d ts %d bayer %d\n", rg.w, rg.h, ts, (int)cfg.bayer_mode);
    const int sh[4][2] = {{2, 0}, {0, 2}, {4, -6}, {-2, 4}};
    for (auto& s : sh) {
        Image cg = compute_grey(crop(s[0], s[1]), cfg.bayer_mode, cfg.grey_method);
        FlowField f = align(pyr, rg, cg, cfg, ts);
        clear_align_ref_ica_cache();
        FlowField fr = flow_to_raw_tile_grid(f, S, S, cg.h, cg.w, ts, cfg.r_Mt, cfg.num_threads, cfg.grey_tile_size(ts));
        for (int pass = 0; pass < 2; ++pass) {
            const FlowField& q = pass ? fr : f;
            std::vector<float> vx, vy;
            for (int ty = 2; ty < q.ny - 2; ++ty)
                for (int tx = 2; tx < q.nx - 2; ++tx) { vx.push_back(q.dx(ty, tx)); vy.push_back(q.dy(ty, tx)); }
            std::sort(vx.begin(), vx.end()); std::sort(vy.begin(), vy.end());
            const size_t m = vx.size();
            std::printf("shift dy %+d dx %+d  %s grid %dx%d: dx p10 %+.4f p50 %+.4f p90 %+.4f | dy p10 %+.4f p50 %+.4f p90 %+.4f\n",
                        s[0], s[1], pass ? "RAW " : "GREY", q.nx, q.ny, vx[m / 10], vx[m / 2], vx[m * 9 / 10],
                        vy[m / 10], vy[m / 2], vy[m * 9 / 10]);
        }
    }
    return 0;
}
