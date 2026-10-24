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
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "global_homography.h"
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

// The guide-pixel correspondence rob_make_mask / rnn_refine_mask use. use_h = false: the tile/affine flow
// alone, without the global homography (the residual after the global warp).
static void guide_flow(const FlowField& fl, int tile, const Config& cfg, int x, int y, float& dx, float& dy,
                       bool use_h = true) {
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
    if (use_h && fl.has_global_h) {
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
    // BC_GAIN_NORM=1: match every frame's brightness to the reference like the app does (one gain from the
    // frames' shutter x ISO metadata) - tests whether exposure drift causes the over-rejection.
    if (std::getenv("BC_GAIN_NORM") && burst.size() > 1 && burst[0].exposure > 0.f) {   // the app's matching
        float med[4], mx[4];
        raw_site_stats(burst[0], med, mx);
        for (size_t k = 1; k < burst.size(); ++k)
            if (burst[k].exposure > 0.f) {
                const float g = std::min(1.5f, std::max(0.67f, burst[0].exposure / burst[k].exposure));
                std::fprintf(stderr, "frame %zu exposure gain %.4f\n", k, g);
                if (std::fabs(g - 1.f) > 0.002f) {
                    float sat[4];   // per-site saturation, as the app (pipeline_paths.cpp) caps now
                    for (int s = 0; s < 4; ++s) {
                        int col = (int)cfg.cfa.p[s / 2][s % 2];
                        if (col < 0) col = 0;
                        if (col > 2) col = 1;
                        sat[s] = cfg.white_balance[1] > 0.f ? cfg.white_balance[col] / cfg.white_balance[1] : mx[s];
                    }
                    apply_raw_gain(burst[k], g, sat);
                }
            }
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
    // BC_MERGES (e.g. "nn" or "r0,nn"): only these merges, to fit full frames in memory. Default: all three.
    const char* bm = std::getenv("BC_MERGES");
    const std::string sel = bm ? bm : "ones,r0,nn";
    const bool want[3] = {sel.find("ones") != std::string::npos, sel.find("r0") != std::string::npos,
                          sel.find("nn") != std::string::npos};
    Image num[3], den[3];
    for (int v = 0; v < 3; ++v)
        if (want[v]) { num[v] = Image(Hs, Ws, 3); den[v] = Image(Hs, Ws, 3); }

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
        // BC_HOMOG=1: the app's global homography (Config::global_homography_enabled, estimate_global_warp in
        // core/pipeline_paths.cpp): warp the comparison grey onto the reference, align the residual, carry H on
        // the flow so the masks and the merge sample comp at H * (p + flow). Full-resolution grey only, as in the app.
        const bool homog = std::getenv("BC_HOMOG") && comp_grey.h == comp.h && comp_grey.w == comp.w &&
                           ref_grey.h == ref.h && ref_grey.w == ref.w;
        f32 Hg[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        FlowField flow;
        if (homog) {
            estimate_global_homography(ref_grey, comp_grey, work, Hg);
            flow = align(ref_pyr, ref_grey, warp_grey_by_homography(comp_grey, Hg), work, tile);
            std::fprintf(stderr, "frame %d homography %.5f %.5f %.2f / %.5f %.5f %.2f / %.2e %.2e\n", k, Hg[0],
                         Hg[1], Hg[2], Hg[3], Hg[4], Hg[5], Hg[6], Hg[7]);
        } else {
            if (std::getenv("BC_HOMOG")) std::fprintf(stderr, "frame %d: grey not full resolution, no homography\n", k);
            flow = align(ref_pyr, ref_grey, comp_grey, work, tile);
        }
        flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w, tile, work.r_Mt,
                                     work.num_threads, work.grey_tile_size(tile));
        if (homog) {
            for (int i = 0; i < 9; ++i) flow.global_h[i] = Hg[i];
            flow.has_global_h = true;
            // what the homography takes out: rotation (deg), scale, and the translation at the frame centre
            const double rot = std::atan2((double)Hg[3], (double)Hg[0]) * 180.0 / 3.14159265358979;
            const double scl = std::sqrt((double)Hg[0] * Hg[0] + (double)Hg[3] * Hg[3]);
            const double cx = 0.5 * comp.w, cy = 0.5 * comp.h;
            const double wq = Hg[6] * cx + Hg[7] * cy + Hg[8];
            const double hx = (Hg[0] * cx + Hg[1] * cy + Hg[2]) / wq, hy = (Hg[3] * cx + Hg[4] * cy + Hg[5]) / wq;
            std::fprintf(stderr, "frame %d: homography rotation %.3f deg, scale %.4f, centre shift (%.1f, %.1f) raw px\n",
                         k, rot, scl, hx - cx, hy - cy);
        }
        Image rob = compute_robustness(comp, ref_stats, flow, tile, work);   // the app's analytic mask
        if (rob.c != 1 || rob.h != comp.h || rob.w != comp.w) {
            std::fprintf(stderr, "unexpected mask %dx%dx%d\n", rob.h, rob.w, rob.c);
            return 3;
        }
        // robust_nn on the guide grid (the analytic mask is constant per RGGB quad)
        // BC_RESID_FEAT=1: the flow-consistency features read the residual flow (no global homography); the warp
        // still samples through the homography.
        const bool resid = std::getenv("BC_RESID_FEAT") && flow.has_global_h;
        std::vector<float> cgrey = guide_grey(comp, work), fxy((size_t)gw * gh * 2), r0g((size_t)gw * gh),
            rn((size_t)gw * gh), fres(resid ? (size_t)gw * gh * 2 : 0);
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                guide_flow(flow, tile, work, x, y, fxy[2 * ((size_t)y * gw + x)], fxy[2 * ((size_t)y * gw + x) + 1]);
                if (resid)
                    guide_flow(flow, tile, work, x, y, fres[2 * ((size_t)y * gw + x)],
                               fres[2 * ((size_t)y * gw + x) + 1], false);
                r0g[(size_t)y * gw + x] = rob.at(2 * y, 2 * x);
            }
        robust_nn::refine(rgrey.data(), cgrey.data(), fxy.data(), r0g.data(), P, rn.data(), &scratch, nullptr,
                          resid ? fres.data() : nullptr);
        if (std::getenv("BC_DUMP_INPUTS")) {   // the mask net's guide-res inputs per frame (for Python mask models)
            char nm[64];
            std::snprintf(nm, sizeof nm, "/inputs_f%d.f32", k);
            FILE* fi = std::fopen((out + nm).c_str(), "wb");
            int hdr[3] = {gw, gh, P.flow_step};
            float prm[3] = {P.noise_a, P.noise_b, P.clip_level};
            std::fwrite(hdr, 4, 3, fi); std::fwrite(prm, 4, 3, fi);
            std::fwrite(rgrey.data(), 4, rgrey.size(), fi); std::fwrite(cgrey.data(), 4, cgrey.size(), fi);
            std::fwrite(fxy.data(), 4, fxy.size(), fi);
            std::fwrite(resid ? fres.data() : fxy.data(), 4, fxy.size(), fi);
            std::fwrite(r0g.data(), 4, r0g.size(), fi);
            std::fclose(fi);
        }
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
        // BC_GUIDED_UP=1: edge-guided, min-preserving upsample of the guide-resolution NN mask to raw resolution.
        // Instead of one value per RGGB quad (a 2-raw-px staircase), each raw pixel takes a joint-bilateral
        // average of the 3x3 guide cells around it, weighted by how similar the reference's same-CFA-phase raw
        // sample in that cell is to this pixel (differences measured against the sensor noise, BC_GUIDED_TAU sigmas,
        // default 3) and by a Gaussian of the distance (1 guide px). The result is then the MINIMUM with the quad
        // value, so rejection is only ever extended along the edge the net flagged, never eroded.
        if (std::getenv("BC_GUIDED_UP")) {
            const float tau = std::getenv("BC_GUIDED_TAU") ? (float)std::atof(std::getenv("BC_GUIDED_TAU")) : 3.f;
            const float na = (work.alpha_dng[0] + work.alpha_dng[1] + work.alpha_dng[2]) / 3.f;
            const float nb = (work.beta_dng[0] + work.beta_dng[1] + work.beta_dng[2]) / 3.f;
            Image up(comp.h, comp.w, 1);
#pragma omp parallel for schedule(static)
            for (int y = 0; y < comp.h; ++y)
                for (int x = 0; x < comp.w; ++x) {
                    const float ip = ref.at(y, x);
                    const float s2 = tau * tau * std::max(na * std::max(ip, 0.f) + nb, 1e-10f);
                    const float gyf = (y - 0.5f) * 0.5f, gxf = (x - 0.5f) * 0.5f;   // the merge's guide coordinate
                    const int gy0 = std::min(gh - 1, std::max(0, (int)std::floor(gyf + 0.5f)));
                    const int gx0 = std::min(gw - 1, std::max(0, (int)std::floor(gxf + 0.5f)));
                    float wsum = 0.f, rsum = 0.f;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int qy = gy0 + dy, qx = gx0 + dx;
                            if (qy < 0 || qy >= gh || qx < 0 || qx >= gw) continue;
                            const int ry = std::min(comp.h - 1, 2 * qy + (y & 1)), rx = std::min(comp.w - 1, 2 * qx + (x & 1));
                            const float d = ref.at(ry, rx) - ip;
                            const float sy = (float)qy - gyf, sx = (float)qx - gxf;
                            const float w = std::exp(-0.5f * (sx * sx + sy * sy)) * std::exp(-0.5f * d * d / s2);
                            wsum += w;
                            rsum += w * rn[(size_t)qy * gw + qx];
                        }
                    const float jb = wsum > 0.f ? rsum / wsum : rnn.at(y, x);
                    up.at(y, x) = std::min(jb, rnn.at(y, x));
                }
            rnn = up;
        }
        if (std::getenv("BC_DUMP_RAWMASK")) {   // the frame-averaged raw-resolution NN mask -> nn_rawmask.f32
            static std::vector<double> acc; static int nacc = 0;
            if (acc.empty()) acc.assign(rnn.data.size(), 0.0);
            for (size_t i = 0; i < rnn.data.size(); ++i) acc[i] += rnn.data[i];
            ++nacc;
            if (k == (int)burst.size() - 1) {
                std::vector<float> o(acc.size());
                for (size_t i = 0; i < o.size(); ++i) o[i] = (float)(acc[i] / nacc);
                FILE* fr = std::fopen((out + "/nn_rawmask.f32").c_str(), "wb");
                std::fwrite(o.data(), 4, o.size(), fr);
                std::fclose(fr);
            }
        }
        CovField covs = estimate_kernels(comp, work);
        if (want[0]) merge_comp(comp, flow, covs, ones, tile, num[0], den[0], work);
        if (want[1]) merge_comp(comp, flow, covs, rob, tile, num[1], den[1], work);
        if (want[2]) merge_comp(comp, flow, covs, rnn, tile, num[2], den[2], work);
        double m0 = 0, m1 = 0;
        for (size_t i = 0; i < r0g.size(); ++i) m0 += r0g[i], m1 += rn[i];
        std::fprintf(stderr, "frame %d: mean R analytic %.3f  nn %.3f\n", k, m0 / r0g.size(), m1 / rn.size());
    }
    std::fclose(fm);
    const char* names[3] = {"ones", "r0", "nn"};
    for (int v = 0; v < 3; ++v) {
        if (!want[v]) continue;
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
