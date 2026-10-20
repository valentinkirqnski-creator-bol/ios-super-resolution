// burst_compare — a real DNG burst through the app's own CPU pipeline stages
// (core/pipeline.cpp process_burst: decode, grey, pyramid, align,
// compute_robustness, estimate_kernels, merge_comp/merge_ref), accumulating
// three merges side by side:
//    ones : every sample at weight 1 (no rejection: shows the ghosting)
//    r0   : the app's analytic robustness mask (what ships)
//    nn   : r0 refined by robust_nn (CPU reference of the Metal kernel), then
//           optional sharpening [lo, hi]
//
//   burst_compare out_dir crop_x crop_y crop_w crop_h scale lo hi dng...
// writes out_dir/{ones,r0,nn}.f32 (Hs x Ws x 3), out_dir/masks.f32
// (per comp frame: r0 then nn, guide res), out_dir/meta.txt
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "raw_io.h"
#include "snr_tuning.h"
#include "robust_nn.h"
#include "stages.h"
#include "types.h"

using namespace hhsr;

static Image crop(const Image& im, int x0, int y0, int w, int h) {
    Image o(h, w, im.c);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < im.c; ++c) o.at(y, x, c) = im.at(y0 + y, x0 + x, c);
    return o;
}

// The guide-pixel correspondence rob_make_mask / rnn_refine_mask use.
static void guide_flow(const FlowField& fl, int tile, const Config& cfg, int x, int y, float& dx, float& dy) {
    const float ry = 2.f * y + 0.5f, rx = 2.f * x + 0.5f;
    float fx, fy;
    if (cfg.affine_flow_enabled && fl.has_affine()) {
        fl.sample_affine(ry, rx, tile, fx, fy);
    } else {
        int py = std::min((int)(ry / tile), fl.ny - 1), px = std::min((int)(rx / tile), fl.nx - 1);
        fx = fl.dx(py, px);
        fy = fl.dy(py, px);
    }
    float sx = x + 0.5f * fx, sy = y + 0.5f * fy;
    if (fl.has_global_h) {
        float hx, hy;
        apply_homography(fl.global_h, 2.f * sx, 2.f * sy, hx, hy);
        sx = 0.5f * hx;
        sy = 0.5f * hy;
    }
    dx = sx - x;
    dy = sy - y;
}

static std::vector<float> guide_grey(const Image& raw, const Config& cfg) {
    Image g = compute_guide(raw, cfg);   // camera-native R, mean G, B at quad resolution
    std::vector<float> o((size_t)g.h * g.w);
    for (int y = 0; y < g.h; ++y)
        for (int x = 0; x < g.w; ++x)
            o[(size_t)y * g.w + x] = 0.25f * g.at(y, x, 0) + 0.5f * g.at(y, x, 1) + 0.25f * g.at(y, x, 2);
    return o;
}

int main(int argc, char** argv) {
    if (argc < 11) {
        std::fprintf(stderr, "usage: out_dir crop_x crop_y crop_w crop_h scale lo hi dng...\n");
        return 1;
    }
    const std::string out = argv[1];
    int cx = std::atoi(argv[2]), cy = std::atoi(argv[3]);   // negative = centred
    const int cw = std::atoi(argv[4]) & ~1, ch = std::atoi(argv[5]) & ~1;
    const float scale = (float)std::atof(argv[6]);
    const float lo = (float)std::atof(argv[7]), hi = (float)std::atof(argv[8]);
    std::vector<std::string> paths(argv + 9, argv + argc);

    Config cfg;
    std::vector<Image> burst = load_raw_burst(paths, cfg);
    // BC_GAIN_NORM=1: match every frame's brightness to the reference, one gain per CFA site (median of the
    // inner frame, robust to the burst's small motion) - tests whether exposure drift causes the over-rejection.
    if (std::getenv("BC_GAIN_NORM") && burst.size() > 1) {   // the app's own brightness matching (core/raw_io)
        float med[4], mx[4];
        raw_site_stats(burst[0], med, mx);
        for (size_t k = 1; k < burst.size(); ++k) match_raw_exposure(burst[k], med, mx);
    }
    if (cx < 0) cx = (burst[0].w - cw) / 2;
    if (cy < 0) cy = (burst[0].h - ch) / 2;
    cx &= ~1;
    cy &= ~1;
    std::fprintf(stderr, "sensor %dx%d, crop at (%d,%d)\n", burst[0].w, burst[0].h, cx, cy);
    for (auto& b : burst) b = crop(b, cx, cy, cw, ch);
    cfg.scale = scale;
    Config work = cfg;
    work.burst_frame_count = (int)burst.size();
    tune_config_snr(burst[0], work);
    const Image& ref = burst[0];
    const int n = (int)burst.size();
    const int tile = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    std::fprintf(stderr, "frames %d crop %dx%d scale %.1f tile %d bayer %d alpha_dng %.3g %.3g %.3g\n", n, cw, ch,
                 scale, tile, (int)work.bayer_mode, work.alpha_dng[0], work.alpha_dng[1], work.alpha_dng[2]);

    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Image ref_grey_padded = pad_image_circular(ref_grey, work.grey_tile_size(tile));
    Pyramid ref_pyr = build_pyramid(ref_grey_padded, work.bm_factors);
    RefStats ref_stats = init_robustness(ref, work);
    CovField ref_covs = estimate_kernels(ref, work);

    const int Hs = (int)std::lround(work.scale * ref.h), Ws = (int)std::lround(work.scale * ref.w);
    Image num[3] = {Image(Hs, Ws, 3), Image(Hs, Ws, 3), Image(Hs, Ws, 3)};
    Image den[3] = {Image(Hs, Ws, 3), Image(Hs, Ws, 3), Image(Hs, Ws, 3)};

    const std::vector<float> rgrey = guide_grey(ref, work);
    const int gw = ref.w / 2, gh = ref.h / 2;
    RNNParams P{};
    P.width = gw;
    P.height = gh;
    P.flow_step = std::max(1, tile / 2);
    P.noise_a = (work.alpha_dng[0] + 2.f * work.alpha_dng[1] + work.alpha_dng[2]) / 16.f;
    P.noise_b = (work.beta_dng[0] + 2.f * work.beta_dng[1] + work.beta_dng[2]) / 16.f;
    P.clip_level = 0.9f;
    FILE* fm = std::fopen((out + "/masks.f32").c_str(), "wb");
    robust_nn::Scratch scratch;

    for (int k = 1; k < n; ++k) {
        const Image& comp = burst[k];
        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
        FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, tile);
        flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w, tile, work.r_Mt,
                                     work.num_threads, work.grey_tile_size(tile));
        Image rob = compute_robustness(comp, ref_stats, flow, tile, work);   // the app's analytic mask
        if (rob.c != 1 || rob.h != comp.h || rob.w != comp.w) {
            std::fprintf(stderr, "unexpected mask %dx%dx%d\n", rob.h, rob.w, rob.c);
            return 3;
        }
        // robust_nn on the guide grid (the analytic mask is constant per RGGB quad)
        std::vector<float> cgrey = guide_grey(comp, work), fxy((size_t)gw * gh * 2), r0g((size_t)gw * gh),
            rn((size_t)gw * gh);
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                guide_flow(flow, tile, work, x, y, fxy[2 * ((size_t)y * gw + x)], fxy[2 * ((size_t)y * gw + x) + 1]);
                r0g[(size_t)y * gw + x] = rob.at(2 * y, 2 * x);
            }
        robust_nn::refine(rgrey.data(), cgrey.data(), fxy.data(), r0g.data(), P, rn.data(), &scratch);
        if (hi > lo)
            for (auto& v : rn) v = std::min(std::max((v - lo) / (hi - lo), 0.f), 1.f);
        std::fwrite(r0g.data(), 4, r0g.size(), fm);
        std::fwrite(rn.data(), 4, rn.size(), fm);
        Image rnn(comp.h, comp.w, 1), ones(comp.h, comp.w, 1);
        for (int y = 0; y < comp.h; ++y)
            for (int x = 0; x < comp.w; ++x) {
                rnn.at(y, x) = rn[(size_t)std::min(y / 2, gh - 1) * gw + std::min(x / 2, gw - 1)];
                ones.at(y, x) = 1.f;
            }
        CovField covs = estimate_kernels(comp, work);
        merge_comp(comp, flow, covs, ones, tile, num[0], den[0], work);
        merge_comp(comp, flow, covs, rob, tile, num[1], den[1], work);
        merge_comp(comp, flow, covs, rnn, tile, num[2], den[2], work);
        double m0 = 0, m1 = 0;
        for (size_t i = 0; i < r0g.size(); ++i) m0 += r0g[i], m1 += rn[i];
        std::fprintf(stderr, "frame %d: mean R analytic %.3f  nn %.3f\n", k, m0 / r0g.size(), m1 / rn.size());
    }
    std::fclose(fm);
    const char* names[3] = {"ones", "r0", "nn"};
    for (int v = 0; v < 3; ++v) {
        merge_ref(ref, ref_covs, num[v], den[v], work, nullptr);
        std::vector<float> o(num[v].data.size());
        for (size_t i = 0; i < o.size(); ++i) o[i] = den[v].data[i] > 0.f ? num[v].data[i] / den[v].data[i] : 0.f;
        FILE* f = std::fopen((out + "/" + names[v] + ".f32").c_str(), "wb");
        std::fwrite(o.data(), 4, o.size(), f);
        std::fclose(f);
    }
    FILE* f = std::fopen((out + "/meta.txt").c_str(), "w");
    std::fprintf(f, "%d %d %d %d %d %f %f %f\n", Hs, Ws, gh, gw, n, work.white_balance[0], work.white_balance[1],
                 work.white_balance[2]);
    std::fclose(f);
    return 0;
}
