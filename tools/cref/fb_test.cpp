// FB confidence test on exact even-shifted crops of a real frame: app aligner flow, then
// (a) the true flow, (b) slips injected on random tiles (across / along the local edge),
// and the measured round-trip error per slip size. Also a real frame pair (argv[2]).
#include "fb_confidence.h"
#include "stages.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include "parallel.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace hhsr;

static void pct(const char* tag, std::vector<float> v) {
    if (v.empty()) { std::printf("%s: none\n", tag); return; }
    std::sort(v.begin(), v.end());
    const size_t m = v.size();
    std::printf("%-34s n %5zu  e p10 %.3f p50 %.3f p90 %.3f | C=exp(-e^2/2*0.75^2) p50 %.3f p10 %.3f\n", tag, m,
                v[m / 10], v[m / 2], v[m * 9 / 10], std::exp(-v[m / 2] * v[m / 2] / 1.125f),
                std::exp(-v[m * 9 / 10] * v[m * 9 / 10] / 1.125f));
}

int main(int argc, char** argv) {
    Config cfg;
    Image full = load_raw_frame(argv[1], cfg, true);
    const int S = 1536;
    const int y0 = ((full.h / 2 - S / 2) / 2) * 2, x0 = ((full.w / 2 - S / 2) / 2) * 2;
    auto crop = [&](const Image& f, int dy, int dx) {
        Image im(S, S, 1);
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x) im.at(y, x) = f.at(y0 + dy + y, x0 + dx + x);
        return im;
    };
    cfg.burst_frame_count = 5;
    Image ref = crop(full, 0, 0);
    tune_config_snr(ref, cfg);
    if (const char* f = std::getenv("FULL")) cfg.fb_conf_full = f[0] == '1';
    if (const char* f = std::getenv("SUB")) cfg.fb_conf_subdiv = std::atoi(f);
    const int ts = cfg.bm_tile_sizes.empty() ? 16 : cfg.bm_tile_sizes[0];
    Image rg = compute_grey(ref, cfg.bayer_mode, cfg.grey_method);
    Pyramid pyr = build_pyramid(pad_image_circular(rg, cfg.grey_tile_size(ts)), cfg.bm_factors);
    auto flow_of = [&](const Image& cg) {
        FlowField f = align(pyr, rg, cg, cfg, ts);
        clear_align_ref_ica_cache();
        return flow_to_raw_tile_grid(f, S, S, cg.h, cg.w, ts, cfg.r_Mt, cfg.num_threads, cfg.grey_tile_size(ts));
    };
    Image cg = compute_grey(crop(full, 4, -6), cfg.bayer_mode, cfg.grey_method);
    FlowField f = flow_of(cg);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> e0 = fb_tile_error(rg, cg, f, S, S, ts, cfg);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("mode %s  tiles %dx%d  time %.0f ms (%d threads)\n", cfg.fb_conf_full ? "FULL" : "local", f.nx, f.ny, ms, cfg.num_threads);
    pct("exact shift, app flow", e0);
    // Injected slips on random tiles: across / along the patch's dominant edge.
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const float mags[] = {0.1f, 0.3f, 0.6f, 1.0f, 1.5f, 2.5f, 4.0f};
    for (int along = 0; along < 2; ++along)
        for (float m : mags) {
            FlowField g = f;
            std::vector<int> tiles;
            std::vector<float> coh;
            for (int t = 0; t < g.ny * g.nx; ++t) {
                const int ty = t / g.nx, tx = t % g.nx;
                if (ty < 2 || tx < 2 || ty >= g.ny - 2 || tx >= g.nx - 2 || U(rng) > 0.08f) continue;
                // edge direction from the ref structure tensor
                double jxx = 0, jyy = 0, jxy = 0;
                for (int y = ty * ts; y < (ty + 1) * ts; ++y)
                    for (int x = tx * ts; x < (tx + 1) * ts; ++x) {
                        const double gx = rg.at(y, x + 1) - rg.at(y, x - 1), gy = rg.at(y + 1, x) - rg.at(y - 1, x);
                        jxx += gx * gx; jyy += gy * gy; jxy += gx * gy;
                    }
                const double tr = jxx + jyy, ds = std::sqrt(std::max(0.0, 0.25 * (jxx - jyy) * (jxx - jyy) + jxy * jxy));
                const double l1 = 0.5 * tr + ds, l2 = 0.5 * tr - ds;
                if (tr <= 0) continue;
                double nxv = jxy, nyv = l1 - jxx;
                if (std::fabs(nxv) + std::fabs(nyv) < 1e-15) { nxv = 1; nyv = 0; }
                const double nn = std::hypot(nxv, nyv);
                nxv /= nn; nyv /= nn;   // normal (across the edge)
                const double dxv = along ? -nyv : nxv, dyv = along ? nxv : nyv;
                g.flow[2 * t] += (float)(m * dxv); g.flow[2 * t + 1] += (float)(m * dyv);
                tiles.push_back(t);
                coh.push_back((float)((l1 - l2) / (l1 + l2)));
            }
            std::vector<float> e = fb_tile_error(rg, cg, g, S, S, ts, cfg), es, es_edge;
            const int sub = std::max(1, cfg.fb_conf_subdiv), pnx = g.nx * sub;
            for (size_t i = 0; i < tiles.size(); ++i) {
                const int ty = tiles[i] / g.nx, tx = tiles[i] % g.nx;
                float mx = 0.f;   // the tile's worst piece
                for (int a2 = 0; a2 < sub; ++a2)
                    for (int b2 = 0; b2 < sub; ++b2) mx = std::max(mx, e[(size_t)(ty * sub + a2) * pnx + tx * sub + b2]);
                es.push_back(mx);
                if (coh[i] > 0.6f) es_edge.push_back(mx);
            }
            char tag[96];
            std::snprintf(tag, sizeof tag, "slip %.1f px %s, all tiles", m, along ? "ALONG edge " : "ACROSS edge");
            pct(tag, es);
            std::snprintf(tag, sizeof tag, "slip %.1f px %s, edge tiles", m, along ? "ALONG edge " : "ACROSS edge");
            pct(tag, es_edge);
        }
    if (argc > 2) {   // real pair: ref frame vs another frame of the burst
        Config c2 = cfg;
        Image f2 = load_raw_frame(argv[2], c2, false);
        Image c2g = compute_grey(crop(f2, 0, 0), cfg.bayer_mode, cfg.grey_method);
        FlowField fr = flow_of(c2g);
        auto t1 = std::chrono::steady_clock::now();
        std::vector<float> er = fb_tile_error(rg, c2g, fr, S, S, ts, cfg);
        const double ms2 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::printf("real pair (%.0f ms):\n", ms2);
        pct("real burst pair", er);
        size_t n15 = 0, n4 = 0;
        for (float v : er) { n15 += v > 1.5f; n4 += v > 4.f; }
        std::printf("real pair: tiles e>1.5 %.1f%%  e>4 %.1f%%\n", 100.0 * n15 / er.size(), 100.0 * n4 / er.size());
    }
    return 0;
}
