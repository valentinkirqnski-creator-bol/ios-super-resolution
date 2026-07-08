#include "stages.h"
#include "parallel.h"

namespace hhsr {

// Alg. 7: guide image at half resolution. For bayer, average the two greens
// and keep R/B, undoing white balance. For grey mode the input is the guide.
static Image compute_guide(const Image& raw, const Config& cfg) {
    if (!cfg.bayer_mode) {
        Image g(raw.h, raw.w, 1);
        g.data = raw.data;
        return g;
    }
    int gh = raw.h / 2, gw = raw.w / 2;
    Image guide(gh, gw, 3);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            f32 gsum = 0.f;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    uint8_t c = cfg.cfa.p[i][j];
                    f32 v = raw.at(2 * y + i, 2 * x + j) / cfg.white_balance[c];
                    if (c == 1) gsum += v;
                    else        guide.at(y, x, c) = v;
                }
            }
            guide.at(y, x, 1) = 0.5f * gsum;
        }
    }
    return guide;
}

// Alg. 8: 3x3 local mean (mu) and variance (sigma^2) per channel.
static void local_stats_3x3(const Image& guide, Image& means, Image& vars) {
    means = Image(guide.h, guide.w, guide.c);
    vars  = Image(guide.h, guide.w, guide.c);
    for (int ch = 0; ch < guide.c; ++ch) {
        for (int y = 0; y < guide.h; ++y) {
            for (int x = 0; x < guide.w; ++x) {
                f32 s = 0.f, s2 = 0.f;
                for (int i = -1; i <= 1; ++i) {
                    int yy = (int)clampf((f32)(y + i), 0.f, (f32)(guide.h - 1));
                    for (int j = -1; j <= 1; ++j) {
                        int xx = (int)clampf((f32)(x + j), 0.f, (f32)(guide.w - 1));
                        f32 v = guide.at(yy, xx, ch);
                        s += v; s2 += v * v;
                    }
                }
                f32 m = s / 9.f;
                means.at(y, x, ch) = m;
                vars.at(y, x, ch) = s2 / 9.f - m * m;
            }
        }
    }
}

static inline f32 bilinear(const Image& img, f32 y, f32 x, int ch) {
    y = clampf(y, 0.f, (f32)(img.h - 1));
    x = clampf(x, 0.f, (f32)(img.w - 1));
    int y0 = (int)std::floor(y), x0 = (int)std::floor(x);
    int y1 = std::min(y0 + 1, img.h - 1), x1 = std::min(x0 + 1, img.w - 1);
    f32 fy = y - y0, fx = x - x0;
    f32 top = img.at(y0, x0, ch) * (1 - fx) + img.at(y0, x1, ch) * fx;
    f32 bot = img.at(y1, x0, ch) * (1 - fx) + img.at(y1, x1, ch) * fx;
    return top * (1 - fy) + bot * fy;
}

static void sample_flow_bilinear(const FlowField& flow, int tile_size, f32 grey_x, f32 grey_y,
                                 f32& out_dx, f32& out_dy) {
    const f32 fx = grey_x / (f32)tile_size;
    const f32 fy = grey_y / (f32)tile_size;
    const int px0 = std::min((int)std::floor(fx), flow.nx - 1);
    const int py0 = std::min((int)std::floor(fy), flow.ny - 1);
    const int px1 = std::min(px0 + 1, flow.nx - 1);
    const int py1 = std::min(py0 + 1, flow.ny - 1);
    const f32 wx = fx - px0;
    const f32 wy = fy - py0;

    const f32 dx00 = flow.dx(py0, px0), dy00 = flow.dy(py0, px0);
    const f32 dx10 = flow.dx(py0, px1), dy10 = flow.dy(py0, px1);
    const f32 dx01 = flow.dx(py1, px0), dy01 = flow.dy(py1, px0);
    const f32 dx11 = flow.dx(py1, px1), dy11 = flow.dy(py1, px1);

    const f32 top_dx = dx00 + wx * (dx10 - dx00);
    const f32 bot_dx = dx01 + wx * (dx11 - dx01);
    const f32 top_dy = dy00 + wx * (dy10 - dy00);
    const f32 bot_dy = dy01 + wx * (dy11 - dy01);
    out_dx = top_dx + wy * (bot_dx - top_dx);
    out_dy = top_dy + wy * (bot_dy - top_dy);
}

static f32 sample_tile_map_bilinear(const std::vector<f32>& map, int tny, int tnx, f32 tyf, f32 txf) {
    const int px0 = std::min((int)std::floor(txf), tnx - 1);
    const int py0 = std::min((int)std::floor(tyf), tny - 1);
    const int px1 = std::min(px0 + 1, tnx - 1);
    const int py1 = std::min(py0 + 1, tny - 1);
    const f32 wx = txf - px0;
    const f32 wy = tyf - py0;
    const f32 v00 = map[(size_t)py0 * tnx + px0];
    const f32 v10 = map[(size_t)py0 * tnx + px1];
    const f32 v01 = map[(size_t)py1 * tnx + px0];
    const f32 v11 = map[(size_t)py1 * tnx + px1];
    const f32 top = v00 + wx * (v10 - v00);
    const f32 bot = v01 + wx * (v11 - v01);
    return top + wy * (bot - top);
}

// Light 3x3 blur softens tile/block boundaries in the robustness mask.
static Image blur_robustness_3x3(const Image& src) {
    Image out(src.h, src.w, 1);
    for (int y = 0; y < src.h; ++y) {
        for (int x = 0; x < src.w; ++x) {
            f32 s = 0.f;
            for (int i = -1; i <= 1; ++i) {
                int yy = (int)clampf((f32)(y + i), 0.f, (f32)(src.h - 1));
                for (int j = -1; j <= 1; ++j) {
                    int xx = (int)clampf((f32)(x + j), 0.f, (f32)(src.w - 1));
                    s += src.at(yy, xx);
                }
            }
            out.at(y, x) = s / 9.f;
        }
    }
    return out;
}

RefStats init_robustness(const Image& ref_raw, const Config& cfg) {
    RefStats st;
    if (!cfg.robustness_enabled) return st;
    Image guide = compute_guide(ref_raw, cfg);
    local_stats_3x3(guide, st.means, st.stds);
    return st;
}

Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg) {
    int gh = cfg.bayer_mode ? comp_raw.h / 2 : comp_raw.h;
    int gw = cfg.bayer_mode ? comp_raw.w / 2 : comp_raw.w;

    if (!cfg.robustness_enabled) {
        // Neutral robustness of 1 everywhere, at grey resolution.
        Image r(gh, gw, 1);
        std::fill(r.data.begin(), r.data.end(), 1.f);
        return r;
    }

    Image guide = compute_guide(comp_raw, cfg);
    Image cmeans, cvars;
    local_stats_3x3(guide, cmeans, cvars);
    int nc = guide.c;

    // Flow tile size expressed in grey pixels (alignment runs on grey).
    int gts = std::max(1, tile_size);
    int tny = flow.ny, tnx = flow.nx;

    // Flow-irregularity map S over the tile grid (Alg. 6 penalty).
    std::vector<f32> S((size_t)tny * tnx, cfg.r_s2);
    for (int ty = 0; ty < tny; ++ty) {
        for (int tx = 0; tx < tnx; ++tx) {
            f32 mnx = 1e30f, mny = 1e30f, mxx = -1e30f, mxy = -1e30f;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= tny || xx < 0 || xx >= tnx) continue;
                    f32 fx = flow.dx(yy, xx), fy = flow.dy(yy, xx);
                    mnx = std::min(mnx, fx); mxx = std::max(mxx, fx);
                    mny = std::min(mny, fy); mxy = std::max(mxy, fy);
                }
            }
            f32 d0 = mxx - mnx, d1 = mxy - mny;
            S[(size_t)ty * tnx + tx] = (d0 * d0 + d1 * d1 > cfg.r_Mt * cfg.r_Mt) ? cfg.r_s1 : cfg.r_s2;
        }
    }

    // R at grey resolution.
    Image R(gh, gw, 1);
    parallel_rows(gh, cfg.num_threads, [&](int y) {
        for (int x = 0; x < gw; ++x) {
            f32 fx = 0.f, fy = 0.f;
            sample_flow_bilinear(flow, gts, (f32)x, (f32)y, fx, fy);
            const f32 s = sample_tile_map_bilinear(S, tny, tnx, (f32)y / gts, (f32)x / gts);

            f32 d_sq = 0.f, sigma_sq = 0.f;
            for (int c = 0; c < nc; ++c) {
                f32 rm = ref_stats.means.at(std::min(y, ref_stats.means.h - 1),
                                            std::min(x, ref_stats.means.w - 1), c);
                f32 cm = bilinear(cmeans, y + fy, x + fx, c);
                f32 dp = std::fabs(rm - cm);

                // Noise model: sigma^2 = alpha*I + beta (affine). This replaces
                // the Monte-Carlo std/diff curves of the reference with a direct
                // evaluation (documented approximation).
                f32 brightness = clampf(rm, 0.f, 1.f);
                f32 sigma_t_sq = std::max(cfg.alpha * brightness + cfg.beta, 1e-8f);
                f32 d_t_sq = sigma_t_sq;

                f32 sigma_p_sq = ref_stats.stds.at(std::min(y, ref_stats.stds.h - 1),
                                                   std::min(x, ref_stats.stds.w - 1), c);
                sigma_sq += std::max(sigma_p_sq, sigma_t_sq);

                f32 dp_sq = dp * dp;
                f32 shrink = dp_sq / (dp_sq + d_t_sq);
                d_sq += dp_sq * shrink * shrink;
            }
            R.at(y, x) = clampf(s * std::exp(-d_sq / std::max(sigma_sq, 1e-8f)) - cfg.r_t, 0.f, 1.f);
        }
    });

    // Alg. 9: 5x5 local minimum. Returned at grey resolution; merge indexes it
    // in grey coordinates (raw / up), which halves its memory footprint.
    Image r_grey(gh, gw, 1);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            f32 mn = 1e30f;
            for (int i = -2; i <= 2; ++i) {
                int yy = (int)clampf((f32)(y + i), 0.f, (f32)(gh - 1));
                for (int j = -2; j <= 2; ++j) {
                    int xx = (int)clampf((f32)(x + j), 0.f, (f32)(gw - 1));
                    mn = std::min(mn, R.at(yy, xx));
                }
            }
            r_grey.at(y, x) = mn;
        }
    }
    return blur_robustness_3x3(r_grey);
}

} // namespace hhsr
