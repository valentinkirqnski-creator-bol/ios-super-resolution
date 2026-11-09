// Misalignment-artifact NN dataset harness (core/mann.h features), derived from the cref tool.
// Adds EDGE slips (aperture problem): along-edge slides, across-edge shifts (doubled edges) and
// ramped offsets along runs of edge tiles (tile stairs). Crop planes: [kMannIn] features, z, gate.
// For one rendered burst: the app's own alignment, injected tile slips (known), both
// masks (Wronski / Alg.6 default and GCam), the warped greys + consensus, the cref
// features, and per raw pixel the ghost size z = |clean_k(p+used) - clean_k(p+gt)| /
// sigma_q (sigma_q = one-frame quad noise; z = 50 where the pixel is not visible in
// frame k). Writes crops: per crop int32 {variant, frame, y, x} then f16 planes
// [kCrefIn + 1][S][S] (features, then z).
// usage: cref_dataset in.bin out.crops seed [crops_per_frame=16] [S=80]
#include "stages.h"
#include "cref.h"
#include "mann.h"
#include "snr_tuning.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace hhsr;

static uint16_t f2h(float f) {   // IEEE half (round-to-nearest-even), enough for features in [-1,1] and z
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t hm = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (hm & 1u))) ++hm;
        return (uint16_t)(sign | hm);
    }
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t hm = m >> 13;
    const uint32_t rem = m & 0x1fffu;
    uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | hm);
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) ++h;
    return h;
}

static float bil(const float* p, int h, int w, float y, float x) {
    y = std::min(std::max(y, 0.f), (float)(h - 1));
    x = std::min(std::max(x, 0.f), (float)(w - 1));
    const int y0 = std::min(h - 2, (int)y), x0 = std::min(w - 2, (int)x);
    const float fy = y - y0, fx = x - x0;
    const float* r0 = p + (size_t)y0 * w;
    const float* r1 = r0 + w;
    const float t = r0[x0] + (r0[x0 + 1] - r0[x0]) * fx, b = r1[x0] + (r1[x0 + 1] - r1[x0]) * fx;
    return t + (b - t) * fy;
}

// The pipeline's warp_comp_grey_to_ref: nearest tile flow, bilinear sample (grey == raw grid).
static Image warp_grey(const Image& g, const FlowField& f, int ts) {
    Image o(g.h, g.w, 1);
    for (int y = 0; y < g.h; ++y) {
        const int ty = std::min(y / ts, f.ny - 1);
        for (int x = 0; x < g.w; ++x) {
            const int tx = std::min(x / ts, f.nx - 1);
            o.at(y, x) = bil(g.data.data(), g.h, g.w, (float)y + f.dy(ty, tx), (float)x + f.dx(ty, tx));
        }
    }
    return o;
}

int main(int argc, char** argv) {
    if (argc < 4) return 1;
    FILE* fi = std::fopen(argv[1], "rb");
    if (!fi) return 2;
    int32_t hi[7];
    float hf[5];
    std::fread(hi, 4, 7, fi);
    std::fread(hf, 4, 5, fi);
    const int N = hi[0], H = hi[1], W = hi[2], Hg = H / 2, Wg = W / 2;
    std::vector<Image> raw(N);
    for (int k = 0; k < N; ++k) {
        raw[k] = Image(H, W, 1);
        std::fread(raw[k].data.data(), 4, (size_t)H * W, fi);
    }
    std::vector<float> gt((size_t)N * 2 * Hg * Wg), clean((size_t)N * Hg * Wg), vis((size_t)N * Hg * Wg);
    std::fread(gt.data(), 4, gt.size(), fi);
    std::fread(clean.data(), 4, clean.size(), fi);
    std::fread(vis.data(), 4, vis.size(), fi);
    std::fclose(fi);
    const uint32_t seed = (uint32_t)std::atoi(argv[3]);
    const int per_frame = argc > 4 ? std::atoi(argv[4]) : 16;
    const int S = argc > 5 ? std::atoi(argv[5]) : 80;

    Config cfg;
    cfg.bayer_mode = true;
    cfg.cfa.p[0][0] = (uint8_t)hi[3]; cfg.cfa.p[0][1] = (uint8_t)hi[4];
    cfg.cfa.p[1][0] = (uint8_t)hi[5]; cfg.cfa.p[1][1] = (uint8_t)hi[6];
    for (int c = 0; c < 3; ++c) { cfg.alpha_dng[c] = hf[0]; cfg.beta_dng[c] = hf[1]; cfg.white_balance[c] = hf[2 + c]; }
    cfg.has_noise_profile = true;
    cfg.raw_prewhitened = true;
    cfg.burst_frame_count = N;
    cfg.snr_auto_tune = false;            // manual tile size (the app's default 16), no per-burst Monte Carlo
    cfg.alignment_tile_size = 16;
    tune_config_snr(raw[0], cfg);
    const int ts = cfg.bm_tile_sizes.empty() ? 16 : cfg.bm_tile_sizes[0];

    const Image& ref = raw[0];
    Image ref_grey = compute_grey(ref, cfg.bayer_mode, cfg.grey_method);
    if (ref_grey.h != H || ref_grey.w != W) { std::fprintf(stderr, "grey not raw-res\n"); return 3; }
    Pyramid ref_pyr = build_pyramid(pad_image_circular(ref_grey, cfg.grey_tile_size(ts)), cfg.bm_factors);

    // Flows (app aligner) + injected slips, shared by both mask variants.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::vector<FlowField> flows(N);
    std::vector<Image> greys(N);
    double err_sum = 0; size_t err_n = 0;
    for (int k = 1; k < N; ++k) {
        greys[k] = compute_grey(raw[k], cfg.bayer_mode, cfg.grey_method);
        FlowField f = align(ref_pyr, ref_grey, greys[k], cfg, ts);
        f = flow_to_raw_tile_grid(f, H, W, greys[k].h, greys[k].w, ts, cfg.r_Mt, cfg.num_threads, cfg.grey_tile_size(ts));
        flows[k] = f;
        clear_align_ref_ica_cache();
        if (std::getenv("CREF_CONV") && k == 1) {
            std::vector<float> vx, vy;
            for (int i = 0; i < f.ny * f.nx; ++i) { vx.push_back(f.flow[2 * i]); vy.push_back(f.flow[2 * i + 1]); }
            std::sort(vx.begin(), vx.end()); std::sort(vy.begin(), vy.end());
            const size_t m = vx.size();
            std::fprintf(stderr, "frame1 flow dx p10 %.2f p50 %.2f p90 %.2f | dy p10 %.2f p50 %.2f p90 %.2f | grid %dx%d grey %dx%d ts %d\n",
                         vx[m / 10], vx[m / 2], vx[m * 9 / 10], vy[m / 10], vy[m / 2], vy[m * 9 / 10], f.nx, f.ny, greys[k].w, greys[k].h, ts);
        }
    }
    if (std::getenv("CREF_CONV")) {   // convention check: app flow vs gt under 4 hypotheses (before slips)
        double e[4] = {0, 0, 0, 0}; size_t n = 0; std::vector<float> ev, ev_tex;
        for (int k = 1; k < N; ++k) {
            const float* gk = gt.data() + (size_t)k * 2 * Hg * Wg;
            for (int ty = 0; ty < flows[k].ny; ++ty)
                for (int tx = 0; tx < flows[k].nx; ++tx) {
                    const int gy = std::min((int)((ty + 0.5f) * ts * 0.5f), Hg - 1), gx = std::min((int)((tx + 0.5f) * ts * 0.5f), Wg - 1);
                    const float g0 = 2.f * gk[(size_t)gy * Wg + gx], g1 = 2.f * gk[(size_t)Hg * Wg + (size_t)gy * Wg + gx];
                    const float fx = flows[k].dx(ty, tx), fy = flows[k].dy(ty, tx);
                    ev.push_back(std::hypot(fx - g0, fy - g1)); e[0] += ev.back();
                    {   // texture: clean-grey gradient energy over the tile vs the noise variance
                        const float* c0 = clean.data();
                        double ge = 0; int cnt = 0;
                        for (int yy = ty * ts / 2; yy < std::min(Hg - 1, (ty + 1) * ts / 2); ++yy)
                            for (int xx = tx * ts / 2; xx < std::min(Wg - 1, (tx + 1) * ts / 2); ++xx) {
                                const float dx = c0[(size_t)yy * Wg + xx + 1] - c0[(size_t)yy * Wg + xx];
                                const float dy = c0[(size_t)(yy + 1) * Wg + xx] - c0[(size_t)yy * Wg + xx];
                                ge += dx * dx + dy * dy; ++cnt;
                            }
                        const float m = c0[(size_t)gy * Wg + gx];
                        const double nv = (hf[0] * std::max(0.f, m) + hf[1]) * 0.25;
                        if (cnt && ge / cnt > 4.0 * nv) ev_tex.push_back(ev.back());
                    } e[1] += std::hypot(fx + g0, fy + g1);
                    e[2] += std::hypot(fx - g1, fy - g0); e[3] += std::hypot(fx + g1, fy + g0);
                    ++n;
                }
        }
        std::sort(ev.begin(), ev.end());
        std::sort(ev_tex.begin(), ev_tex.end());
        size_t lt = 0; for (float v : ev_tex) lt += v < 0.5f;
        if (!ev_tex.empty()) std::fprintf(stderr, "TEXTURED tiles %zu/%zu: err median %.3f frac<0.5 %.3f\n", ev_tex.size(), ev.size(), ev_tex[ev_tex.size() / 2], (double)lt / ev_tex.size());
        size_t lo = 0; for (float v : ev) lo += v < 0.5f;
        std::fprintf(stderr, "err median %.3f p90 %.3f frac<0.5 %.3f  ", ev[ev.size() / 2], ev[ev.size() * 9 / 10], (double)lo / ev.size());
        std::fprintf(stderr, "conv: (+x,+y) %.3f  (-x,-y) %.3f  swapped %.3f  swapped- %.3f\n", e[0] / n, e[1] / n, e[2] / n, e[3] / n);
    }
    // Slips: per frame with p 0.7, a fraction 0.05-0.35 of tiles gets a slip of 0.1-3 px (log-uniform), random
    // angle; plus burst-consistent slips (one per tile, in 40-95% of frames, +-15% jitter) on 5-15% of tiles.
    const int ny = flows[1].ny, nx = flows[1].nx;
    std::vector<float> cdx((size_t)ny * nx, 0.f), cdy((size_t)ny * nx, 0.f), cpf((size_t)ny * nx, 0.f);
    {
        const float frac = 0.05f + 0.10f * U(rng);
        for (size_t t = 0; t < cdx.size(); ++t)
            if (U(rng) < frac) {
                const float m = std::exp(std::log(0.1f) + U(rng) * (std::log(3.f) - std::log(0.1f)));
                const float a = 6.2831853f * U(rng);
                cdx[t] = m * std::cos(a); cdy[t] = m * std::sin(a); cpf[t] = 0.4f + 0.55f * U(rng);
            }
    }
    for (int k = 1; k < N; ++k) {
        FlowField& f = flows[k];
        const bool corrupt = U(rng) < 0.7f;
        const float frac = 0.05f + 0.30f * U(rng);
        for (int ty = 0; ty < f.ny; ++ty)
            for (int tx = 0; tx < f.nx; ++tx) {
                const size_t t = (size_t)ty * f.nx + tx;
                if (corrupt && U(rng) < frac) {
                    const float m = std::exp(std::log(0.1f) + U(rng) * (std::log(3.f) - std::log(0.1f)));
                    const float a = 6.2831853f * U(rng);
                    f.dx(ty, tx) += m * std::cos(a); f.dy(ty, tx) += m * std::sin(a);
                }
                if (cpf[t] > 0.f && U(rng) < cpf[t]) {
                    const float j = 1.f + 0.15f * (2.f * U(rng) - 1.f);
                    f.dx(ty, tx) += cdx[t] * j; f.dy(ty, tx) += cdy[t] * j;
                }
            }
    }
    // v2: period slips -- on repeated-pattern tiles, the wrong-period lock block matching makes (the
    // aperture problem, paper S.3). The reference's HF variance loss (feature 7) picks pattern tiles; the
    // period is the strongest secondary peak of the tile's normalised autocorrelation (24x24 window, shifts
    // up to 12 px, |shift| >= 2). Per frame with p 0.6, each pattern tile slips by +-P (sometimes 2P) with
    // p 0.3, half the time together with a neighbour (blocky clusters).
    size_t n_pattern_tiles = 0, n_period_slips = 0;
    {
        std::vector<float> hf_plane((size_t)H * W), dis_dummy((size_t)H * W);
        cref_aux_rows(ref_grey, std::vector<float>(), 0, 0, ts, cfg, 0, H, dis_dummy.data(), hf_plane.data());
        std::vector<float> Px((size_t)ny * nx, 0.f), Py((size_t)ny * nx, 0.f);
        #pragma omp parallel for schedule(dynamic, 8) reduction(+ : n_pattern_tiles)
        for (int t = 0; t < ny * nx; ++t) {
            const int ty = t / nx, tx = t % nx;
            double hs = 0; int hc = 0;
            for (int y = ty * ts; y < std::min(H, (ty + 1) * ts); ++y)
                for (int x = tx * ts; x < std::min(W, (tx + 1) * ts); ++x) { hs += hf_plane[(size_t)y * W + x]; ++hc; }
            if (!hc || hs / hc < 0.35) continue;
            const int cy = ty * ts + ts / 2, cx = tx * ts + ts / 2, R = 12, Wn = 12;
            auto G = [&](int y, int x) { return ref_grey.at(std::min(H - 1, std::max(0, y)), std::min(W - 1, std::max(0, x))); };
            std::vector<float> ncc((2 * R + 1) * (2 * R + 1), -1.f);
            float peak = -1.f;
            for (int sy = -R; sy <= R; ++sy)
                for (int sx = -R; sx <= R; ++sx) {
                    if (std::max(std::abs(sx), std::abs(sy)) < 2) continue;
                    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0; int n = 0;
                    for (int y = cy - Wn; y < cy + Wn; ++y)
                        for (int x = cx - Wn; x < cx + Wn; ++x) {
                            const double va = G(y, x), vb = G(y + sy, x + sx);
                            sa += va; sb += vb; saa += va * va; sbb += vb * vb; sab += va * vb; ++n;
                        }
                    const double cov = sab / n - (sa / n) * (sb / n);
                    const double den = std::sqrt(std::max(1e-20, (saa / n - (sa / n) * (sa / n)) * (sbb / n - (sb / n) * (sb / n))));
                    const float v = (float)(cov / den);
                    ncc[(sy + R) * (2 * R + 1) + sx + R] = v;
                    peak = std::max(peak, v);
                }
            // The period: the SHORTEST shift whose correlation is within 0.05 of the best (a grating
            // repeats at P, 2P, ...; a wrong lock is usually the nearest repeat). Needs a clear repeat.
            int bx = 0, by = 0, bd = 1 << 30;
            if (peak > 0.6f)
                for (int sy = -R; sy <= R; ++sy)
                    for (int sx = -R; sx <= R; ++sx) {
                        const float v = ncc[(sy + R) * (2 * R + 1) + sx + R];
                        if (v >= peak - 0.05f && sx * sx + sy * sy < bd) { bd = sx * sx + sy * sy; bx = sx; by = sy; }
                    }
            if (bx || by) { Px[t] = (float)bx; Py[t] = (float)by; ++n_pattern_tiles; }
        }
        for (int k = 1; k < N; ++k) {
            if (U(rng) >= 0.6f) continue;
            FlowField& f = flows[k];
            for (int t = 0; t < ny * nx; ++t) {
                if ((Px[t] == 0.f && Py[t] == 0.f) || U(rng) >= 0.3f) continue;
                const float sgn = U(rng) < 0.5f ? -1.f : 1.f, mult = U(rng) < 0.2f ? 2.f : 1.f;
                const float ddx = sgn * mult * Px[t], ddy = sgn * mult * Py[t];
                f.flow[2 * t] += ddx; f.flow[2 * t + 1] += ddy; ++n_period_slips;
                if (U(rng) < 0.5f) {   // a neighbour with the same pattern locks the same way
                    const int ty = t / nx, tx = t % nx;
                    const int qy = std::min(ny - 1, std::max(0, ty + (U(rng) < 0.5f ? -1 : 1) * (U(rng) < 0.5f)));
                    const int qx = std::min(nx - 1, std::max(0, tx + (U(rng) < 0.5f ? -1 : 1)));
                    const int q = qy * nx + qx;
                    if (q != t && (Px[q] != 0.f || Py[q] != 0.f)) { f.flow[2 * q] += ddx; f.flow[2 * q + 1] += ddy; }
                }
            }
        }
    }

    // EDGE slips (mann): tiles dominated by one long edge (structure tensor coherence > 0.6, gradient energy
    // > 16x noise). Per frame with p 0.6: each edge tile with p 0.2 slides ALONG its edge by 0.5-4 px (harmless
    // on a straight edge) plus, with p 0.7, ACROSS it by 0.2-1.5 px (log-uniform; a shifted / doubled edge).
    // Per frame with p 0.5: tile stairs -- runs of 3-7 edge tiles walked along the edge direction get a ramp of
    // across-edge offsets (step 0.15-0.7 px per tile, centred), as a slight roll aligned per tile produces.
    size_t n_edge_tiles = 0, n_edge_slips = 0, n_stairs = 0;
    {
        std::vector<float> ex((size_t)ny * nx, 0.f), ey((size_t)ny * nx, 0.f);
        std::vector<uint8_t> is_edge((size_t)ny * nx, 0);
        for (int t = 0; t < ny * nx; ++t) {
            const int ty = t / nx, tx = t % nx;
            double jxx = 0, jyy = 0, jxy = 0, mean = 0; int n = 0;
            for (int y = ty * ts; y < std::min(H - 1, (ty + 1) * ts); ++y)
                for (int x = tx * ts; x < std::min(W - 1, (tx + 1) * ts); ++x) {
                    const double gx = ref_grey.at(y, x + 1) - ref_grey.at(y, x), gy = ref_grey.at(y + 1, x) - ref_grey.at(y, x);
                    jxx += gx * gx; jyy += gy * gy; jxy += gx * gy; mean += ref_grey.at(y, x); ++n;
                }
            if (!n) continue;
            jxx /= n; jyy /= n; jxy /= n; mean /= n;
            const double tr = jxx + jyy, dt = std::sqrt(std::max(0.0, (jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy));
            const double l1 = 0.5 * (tr + dt), l2 = 0.5 * (tr - dt);
            const double nv = hf[0] * std::max(0.0, mean) + hf[1];
            if (tr < 8.0 * 2.0 * nv || (l1 - l2) / std::max(1e-12, l1 + l2) < 0.6) continue;
            // normal = eigenvector of l1; edge direction = perpendicular
            double nxv = jxy, nyv = l1 - jxx;
            if (std::fabs(nxv) + std::fabs(nyv) < 1e-12) { nxv = 1; nyv = 0; if (jyy > jxx) { nxv = 0; nyv = 1; } }
            const double nn = std::hypot(nxv, nyv);
            ex[t] = (float)(-nyv / nn); ey[t] = (float)(nxv / nn);   // along the edge
            is_edge[t] = 1; ++n_edge_tiles;
        }
        auto logu = [&](float lo, float hi) { return std::exp(std::log(lo) + U(rng) * (std::log(hi) - std::log(lo))); };
        for (int k = 1; k < N; ++k) {
            FlowField& f = flows[k];
            if (U(rng) < 0.6f)
                for (int t = 0; t < ny * nx; ++t) {
                    if (!is_edge[t] || U(rng) >= 0.2f) continue;
                    const float a = (U(rng) < 0.5f ? -1.f : 1.f) * (0.5f + 3.5f * U(rng));
                    const float b = U(rng) < 0.7f ? (U(rng) < 0.5f ? -1.f : 1.f) * logu(0.2f, 1.5f) : 0.f;
                    // along = (ex, ey); across = the normal (ey, -ex)
                    f.flow[2 * t] += a * ex[t] + b * ey[t];
                    f.flow[2 * t + 1] += a * ey[t] - b * ex[t];
                    ++n_edge_slips;
                }
            if (U(rng) < 0.5f && n_edge_tiles) {
                const int runs = 1 + (int)(U(rng) * std::max<size_t>(1, n_edge_tiles / 12));
                for (int rr = 0; rr < runs; ++rr) {
                    int t = (int)(U(rng) * ny * nx) % (ny * nx);
                    for (int tries = 0; tries < 50 && !is_edge[t]; ++tries) t = (int)(U(rng) * ny * nx) % (ny * nx);
                    if (!is_edge[t]) break;
                    const int len = 3 + (int)(U(rng) * 5);
                    std::vector<int> run{t};
                    float py = (float)(t / nx), px = (float)(t % nx);
                    for (int s2 = 1; s2 < len; ++s2) {
                        py += ey[t]; px += ex[t];
                        const int qy = (int)std::lround(py), qx = (int)std::lround(px);
                        if (qy < 0 || qy >= ny || qx < 0 || qx >= nx) break;
                        const int q = qy * nx + qx;
                        if (q != run.back()) run.push_back(q);
                    }
                    if (run.size() < 3) continue;
                    const float step = (U(rng) < 0.5f ? -1.f : 1.f) * (0.15f + 0.55f * U(rng));
                    const float c0 = 0.5f * (float)(run.size() - 1);
                    for (size_t i = 0; i < run.size(); ++i) {
                        const int q = run[i];
                        const float b = step * ((float)i - c0);
                        f.flow[2 * q] += b * ey[t];
                        f.flow[2 * q + 1] += -b * ex[t];
                    }
                    ++n_stairs;
                }
            }
        }
    }

    // Ghost size z per raw pixel and frame, for a given set of merged flows.
    const float a = hf[0], b = hf[1];
    auto ghost = [&](const std::vector<FlowField>& fl, std::vector<std::vector<float>>& Z) {
    Z.assign(N, std::vector<float>());
    for (int k = 1; k < N; ++k) {
        const std::vector<FlowField>& flows = fl;
        Z[k].assign((size_t)H * W, 0.f);
        const float* ck = clean.data() + (size_t)k * Hg * Wg;
        const float* gk = gt.data() + (size_t)k * 2 * Hg * Wg;
        const float* vk = vis.data() + (size_t)k * Hg * Wg;
        const float* c0 = clean.data();
        for (int y = 0; y < H; ++y) {
            const int ty = std::min(y / ts, ny - 1);
            for (int x = 0; x < W; ++x) {
                const int tx = std::min(x / ts, nx - 1);
                const float gy = (y + 0.5f) * 0.5f - 0.5f, gx = (x + 0.5f) * 0.5f - 0.5f;
                const float tgx = bil(gk, Hg, Wg, gy, gx), tgy = bil(gk + (size_t)Hg * Wg, Hg, Wg, gy, gx);
                const float ux = 0.5f * flows[k].dx(ty, tx), uy = 0.5f * flows[k].dy(ty, tx);
                const float vis_k = bil(vk, Hg, Wg, gy, gx);
                const float d = bil(ck, Hg, Wg, gy + uy, gx + ux) - bil(ck, Hg, Wg, gy + tgy, gx + tgx);
                const float m = std::max(0.f, bil(c0, Hg, Wg, gy, gx));
                const float sq = std::sqrt((a * m + b) * 0.25f);
                float z = std::fabs(d) / std::max(sq, 1e-6f);
                if (vis_k < 0.5f) z = 50.f;
                Z[k][(size_t)y * W + x] = std::min(50.f, z);
                if (y % 8 == 0 && x % 8 == 0) {
                    err_sum += std::hypot(2.f * tgx - flows[k].dx(ty, tx), 2.f * tgy - flows[k].dy(ty, tx));
                    ++err_n;
                }
            }
        }
    }
    };

    FILE* fo = std::fopen(argv[2], "wb");
    if (!fo) return 4;
    constexpr int kPl = kMannIn + 2;   // features, z, gate
    std::vector<uint16_t> buf((size_t)kPl * S * S);
    size_t ncrops = 0;
    for (int variant = 0; variant < 2; ++variant) {
        Config cv = cfg;
        cv.robustness_gcam_enabled = (variant == 1);
        // CREF_HF=1: the masks with HF artifact rejection ON (paper S.3 rule) -- same seeds,
        // flows, slips, labels and crop positions; only feature 0 (R) differs.
        if (std::getenv("CREF_HF")) cv.hf_artifact_removal_enabled = true;
        if (const char* t = std::getenv("CREF_HF_T")) cv.hf_variance_loss_threshold = (float)std::atof(t);
        // The flows this variant MERGES with (GCam: the pipeline's 3x3 median), and everything derived
        // from them: the warped greys, the consensus, the ghost labels, the flow-disagreement feature.
        std::vector<FlowField> fv = flows;
        if (cv.robustness_gcam_enabled)
            for (int k = 1; k < N; ++k) median3x3_tile_flow(fv[k]);
        std::vector<Image> aligned(N);
        for (int k = 1; k < N; ++k) aligned[k] = warp_grey(greys[k], fv[k], ts);
        std::vector<std::vector<float>> Z;
        err_sum = 0; err_n = 0;
        ghost(fv, Z);
        std::fprintf(stderr, "burst %s variant %d N=%d %dx%d ts=%d mean |flow-gt| %.3f raw px pattern tiles %zu period slips %zu edge tiles %zu edge slips %zu stairs %zu\n",
                     argv[1], variant, N, W, H, ts, err_sum / std::max<size_t>(1, err_n), n_pattern_tiles, n_period_slips,
                     n_edge_tiles, n_edge_slips, n_stairs);
        RefStats rs = init_robustness(ref, cv);
        for (int k = 1; k < N; ++k) {
            const FlowField& f = fv[k];
            Image R = compute_robustness(raw[k], rs, f, ts, cv);
            if (R.h <= 0) continue;
            Image Rg(H, W, 1);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    Rg.at(y, x) = R.at(std::min(R.h - 1, (int)((int64_t)y * R.h / H)), std::min(R.w - 1, (int)((int64_t)x * R.w / W)));
            std::vector<float> F, G;
            const std::vector<float> tdis = cref_flow_disagreement(f);
            const std::vector<float> M = mann_tile_irregularity(f);
            mann_features(ref_grey, aligned[k], Rg, f, M, tdis, ts, cv, 0, H, 0, W, F, G);
            const size_t plane = (size_t)H * W;
            // Crops: 1/4 uniform, 3/4 centred on gated pixels (g > 0.5), preferring harmful or edge ones.
            std::vector<size_t> hot, gated;
            for (size_t i = 0; i < plane; i += 7)
                if (G[i] > 0.5f) {
                    gated.push_back(i);
                    if (Z[k][i] > 2.f || std::fabs(F[2 * plane + i]) > 0.25f) hot.push_back(i);
                }
            if (hot.empty()) hot = gated;
            for (int c = 0; c < per_frame; ++c) {
                int y0, x0;
                if ((c & 3) && !hot.empty()) {
                    const size_t i = hot[(size_t)(U(rng) * hot.size()) % hot.size()];
                    y0 = (int)(i / W) - S / 2; x0 = (int)(i % W) - S / 2;
                } else {
                    y0 = (int)(U(rng) * (H - S)); x0 = (int)(U(rng) * (W - S));
                }
                y0 = std::min(H - S, std::max(0, y0)); x0 = std::min(W - S, std::max(0, x0));
                for (int ch = 0; ch < kPl; ++ch)
                    for (int y = 0; y < S; ++y)
                        for (int x = 0; x < S; ++x) {
                            const size_t q = (size_t)(y0 + y) * W + (x0 + x);
                            const float v = ch < kMannIn ? F[(size_t)ch * plane + q] : (ch == kMannIn ? Z[k][q] : G[q]);
                            buf[((size_t)ch * S + y) * S + x] = f2h(v);
                        }
                const int32_t hd[4] = {variant, k, y0, x0};
                std::fwrite(hd, 4, 4, fo);
                std::fwrite(buf.data(), 2, buf.size(), fo);
                ++ncrops;
            }
        }
    }
    std::fclose(fo);
    std::fprintf(stderr, "wrote %zu crops of %d px\n", ncrops, S);
    return 0;
}
