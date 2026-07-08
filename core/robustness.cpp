#include "stages.h"
#include "parallel.h"
#include <cmath>

namespace hhsr {

namespace {

static inline f32 dogson_quadratic(f32 x) {
    f32 ax = std::fabs(x);
    if (ax <= 0.5f) return -2.f * ax * ax + 1.f;
    if (ax <= 1.5f) return ax * ax - 2.5f * ax + 1.5f;
    return 0.f;
}

static int bayer_upscale_factor(const Image& guide_stats, int raw_h, int raw_w) {
    if (guide_stats.h * 2 == raw_h && guide_stats.w * 2 == raw_w) return 2;
    return 1;
}

struct NoiseCurves {
    std::vector<f32> std_curve;
    std::vector<f32> diff_curve;
};

static NoiseCurves make_noise_curves(f32 alpha, f32 beta) {
    NoiseCurves nc;
    nc.std_curve.resize(1001);
    nc.diff_curve.resize(1001);
    for (int i = 0; i <= 1000; ++i) {
        f32 b = i / 1000.f;
        f32 sigma = std::sqrt(std::max(0.f, alpha * b + beta));
        nc.std_curve[(size_t)i] = sigma;
        nc.diff_curve[(size_t)i] = sigma * 0.7978845608f;
    }
    return nc;
}

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
                        s += v;
                        s2 += v * v;
                    }
                }
                f32 m = s / 9.f;
                means.at(y, x, ch) = m;
                vars.at(y, x, ch) = s2 / 9.f - m * m;
            }
        }
    }
}

static f32 sample_dogson(const Image& stats, f32 LR_y, f32 LR_x, int ch) {
    if (!(LR_y >= 0.f && LR_y < (f32)stats.h && LR_x >= 0.f && LR_x < (f32)stats.w))
        return 1e30f;
    int center_y = (int)std::lround(LR_y);
    int center_x = (int)std::lround(LR_x);
    f32 w_acc = 0.f, buf = 0.f;
    for (int i = -1; i <= 1; ++i) {
        int y_ = (int)clampf((f32)(center_y + i), 0.f, (f32)(stats.h - 1));
        f32 dy = (f32)y_ - LR_y;
        f32 wy = dogson_quadratic(dy);
        for (int j = -1; j <= 1; ++j) {
            int x_ = (int)clampf((f32)(center_x + j), 0.f, (f32)(stats.w - 1));
            f32 dx = (f32)x_ - LR_x;
            f32 w = wy * dogson_quadratic(dx);
            buf += stats.at(y_, x_, ch) * w;
            w_acc += w;
        }
    }
    return (w_acc > 1e-12f) ? buf / w_acc : 1e30f;
}

static Image upscale_warp_stats(const Image& guide_stats, int raw_h, int raw_w,
                                bool is_ref, const FlowField* flow, int tile_size,
                                int num_threads) {
    const int nc = guide_stats.c;
    const int s = bayer_upscale_factor(guide_stats, raw_h, raw_w);
    Image out(raw_h, raw_w, nc);

    parallel_rows(raw_h, num_threads, [&](int y) {
        for (int x = 0; x < raw_w; ++x) {
            f32 flow_x = 0.f, flow_y = 0.f;
            if (!is_ref && flow) {
                int patch_idy = std::min(y / tile_size, flow->ny - 1);
                int patch_idx = std::min(x / tile_size, flow->nx - 1);
                flow_x = flow->dx(patch_idy, patch_idx);
                flow_y = flow->dy(patch_idy, patch_idx);
            }
            f32 LR_y = (y + flow_y + 0.5f) / (f32)s - 0.5f;
            f32 LR_x = (x + flow_x + 0.5f) / (f32)s - 0.5f;
            for (int ch = 0; ch < nc; ++ch) {
                f32 v = sample_dogson(guide_stats, LR_y, LR_x, ch);
                out.at(y, x, ch) = (v > 1e20f) ? 0.f : v;
            }
        }
    });
    return out;
}

static void apply_noise_model(const Image& d_p, const Image& ref_means, const Image& ref_vars,
                              const NoiseCurves& nc, Image& d_sq, Image& sigma_sq) {
    const int nc_ch = ref_means.c;
    d_sq = Image(ref_means.h, ref_means.w, 1);
    sigma_sq = Image(ref_means.h, ref_means.w, 1);
    for (int y = 0; y < ref_means.h; ++y) {
        for (int x = 0; x < ref_means.w; ++x) {
            f32 d_sq_ = 0.f, sigma_sq_ = 0.f;
            for (int ch = 0; ch < nc_ch; ++ch) {
                f32 brightness = ref_means.at(y, x, ch);
                int id_noise = (int)std::lround(1000.f * brightness);
                id_noise = std::max(0, std::min(1000, id_noise));
                f32 sigma_t = nc.std_curve[(size_t)id_noise];
                f32 d_t = nc.diff_curve[(size_t)id_noise];
                f32 sigma_p_sq = ref_vars.at(y, x, ch);
                sigma_sq_ += std::max(sigma_p_sq, sigma_t * sigma_t);
                f32 d_p_ = d_p.at(y, x, ch);
                f32 d_p_sq = d_p_ * d_p_;
                f32 shrink = d_p_sq / (d_p_sq + d_t * d_t);
                d_sq_ += d_p_sq * shrink * shrink;
            }
            d_sq.at(y, x) = d_sq_;
            sigma_sq.at(y, x) = sigma_sq_;
        }
    }
}

static std::vector<f32> compute_s(const FlowField& flow, f32 Mt, f32 s1, f32 s2) {
    std::vector<f32> S((size_t)flow.ny * flow.nx, s2);
    for (int ty = 0; ty < flow.ny; ++ty) {
        for (int tx = 0; tx < flow.nx; ++tx) {
            f32 mnx = 1e30f, mny = 1e30f, mxx = -1e30f, mxy = -1e30f;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= flow.ny || xx < 0 || xx >= flow.nx) continue;
                    f32 fx = flow.dx(yy, xx), fy = flow.dy(yy, xx);
                    mnx = std::min(mnx, fx);
                    mxx = std::max(mxx, fx);
                    mny = std::min(mny, fy);
                    mxy = std::max(mxy, fy);
                }
            }
            f32 d0 = mxx - mnx, d1 = mxy - mny;
            S[(size_t)ty * flow.nx + tx] = (d0 * d0 + d1 * d1 > Mt * Mt) ? s1 : s2;
        }
    }
    return S;
}

static Image local_min_5x5(const Image& R) {
    Image r(R.h, R.w, 1);
    for (int y = 0; y < R.h; ++y) {
        for (int x = 0; x < R.w; ++x) {
            f32 mn = 1e30f;
            for (int i = -2; i <= 2; ++i) {
                int yy = (int)clampf((f32)(y + i), 0.f, (f32)(R.h - 1));
                for (int j = -2; j <= 2; ++j) {
                    int xx = (int)clampf((f32)(x + j), 0.f, (f32)(R.w - 1));
                    mn = std::min(mn, R.at(yy, xx));
                }
            }
            r.at(y, x) = mn;
        }
    }
    return r;
}

} // namespace

RefStats init_robustness(const Image& ref_raw, const Config& cfg) {
    RefStats st;
    if (!cfg.robustness_enabled) return st;
    Image guide = compute_guide(ref_raw, cfg);
    Image means, vars;
    local_stats_3x3(guide, means, vars);
    st.means = upscale_warp_stats(means, ref_raw.h, ref_raw.w, true, nullptr, 0, cfg.num_threads);
    st.stds  = upscale_warp_stats(vars, ref_raw.h, ref_raw.w, true, nullptr, 0, cfg.num_threads);
    return st;
}

Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg) {
    if (!cfg.robustness_enabled) {
        Image r(comp_raw.h, comp_raw.w, 1);
        std::fill(r.data.begin(), r.data.end(), 1.f);
        return r;
    }

    const NoiseCurves nc = make_noise_curves(cfg.alpha, cfg.beta);

    Image guide = compute_guide(comp_raw, cfg);
    Image comp_means, comp_vars;
    local_stats_3x3(guide, comp_means, comp_vars);
    comp_means = upscale_warp_stats(comp_means, comp_raw.h, comp_raw.w, false, &flow,
                                    tile_size, cfg.num_threads);

    Image d_p(comp_raw.h, comp_raw.w, ref_stats.means.c);
    for (int y = 0; y < comp_raw.h; ++y) {
        for (int x = 0; x < comp_raw.w; ++x) {
            for (int ch = 0; ch < d_p.c; ++ch)
                d_p.at(y, x, ch) = std::fabs(ref_stats.means.at(y, x, ch) - comp_means.at(y, x, ch));
        }
    }

    Image d_sq, sigma_sq;
    apply_noise_model(d_p, ref_stats.means, ref_stats.stds, nc, d_sq, sigma_sq);

    std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2);

    Image R(comp_raw.h, comp_raw.w, 1);
    for (int y = 0; y < comp_raw.h; ++y) {
        for (int x = 0; x < comp_raw.w; ++x) {
            int patch_idy = std::min(y / tile_size, flow.ny - 1);
            int patch_idx = std::min(x / tile_size, flow.nx - 1);
            f32 s = S[(size_t)patch_idy * flow.nx + patch_idx];
            f32 sig = std::max(sigma_sq.at(y, x), 1e-8f);
            R.at(y, x) = clampf(s * std::exp(-d_sq.at(y, x) / sig) - cfg.r_t, 0.f, 1.f);
        }
    }
    return local_min_5x5(R);
}

} // namespace hhsr
