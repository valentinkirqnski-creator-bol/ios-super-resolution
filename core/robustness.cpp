#include "stages.h"
#include "parallel.h"
#include <limits>
#include <random>
#include <cmath>

namespace hhsr {

namespace {

static inline f32 dogson_quadratic(f32 x) {
    // Matches dogson_quadratic_kernel in utils_image.py
    f32 ax = std::fabs(x);
    if (ax <= 0.5f) return -2.f * ax * ax + 1.f;
    if (ax <= 1.5f) return ax * ax - 2.5f * ax + 1.5f;
    return 0.f;
}

struct NoiseCurves {
    std::vector<f32> std_curve;
    std::vector<f32> diff_curve;
};

// Mirrors fast_monte_carlo.py
static constexpr int k_n_patches = 100000; // n_patches = int(1e5)
static constexpr int k_n_brightness = 1000;
static constexpr f32 k_tol = 3.f;

static void get_non_linearity_bound(f32 alpha, f32 beta, f32 tol, f32& xmin, f32& xmax) {
    f32 tol_sq = tol * tol;
    xmin = tol_sq / 2.f * (alpha + std::sqrt(tol_sq * alpha * alpha + 4.f * beta));
    // Same formula as Python; clamp under the sqrt only for float safety when inner < 0.
    f32 inner = std::pow(2.f + tol_sq * alpha, 2.f) - 4.f * (1.f + tol_sq * beta);
    xmax = (2.f + tol_sq * alpha - std::sqrt(std::max(0.f, inner))) / 2.f;
}

static void unitary_MC(f32 alpha, f32 beta, f32 b, f32& diff_mean, f32& std_mean) {
    // Matches fast_monte_carlo.unitary_MC (population std, ddof=0).
    f32 scale = std::sqrt(std::max(0.f, alpha * b + beta));

    // Seeded for determinism; Python uses unseeded np.random so curves won't bit-match,
    // but sample count / estimator match.
    std::mt19937 gen(1337 + (int)(b * 1000));
    std::normal_distribution<f32> dist(0.f, 1.f);

    double sum_std = 0;
    double sum_diff = 0;

    for (int i = 0; i < k_n_patches; ++i) {
        f32 p1[9], p2[9];
        f32 m1 = 0, m2 = 0;
        for (int j = 0; j < 9; ++j) {
            f32 v1 = std::max(0.f, std::min(1.f, b + scale * dist(gen)));
            f32 v2 = std::max(0.f, std::min(1.f, b + scale * dist(gen)));
            p1[j] = v1; p2[j] = v2;
            m1 += v1; m2 += v2;
        }
        m1 /= 9.f; m2 /= 9.f;
        f32 s1 = 0, s2 = 0;
        for (int j = 0; j < 9; ++j) {
            s1 += (p1[j] - m1) * (p1[j] - m1);
            s2 += (p2[j] - m2) * (p2[j] - m2);
        }
        s1 = std::sqrt(s1 / 9.f);
        s2 = std::sqrt(s2 / 9.f);

        sum_std += 0.5 * (double)(s1 + s2);
        sum_diff += (double)std::fabs(m1 - m2);
    }

    diff_mean = (f32)(sum_diff / k_n_patches);
    std_mean = (f32)(sum_std / k_n_patches);
}

// Matches fast_monte_carlo.interp_MC + run_fast_MC overwrite of [imin:imax].
static void interp_MC_range(NoiseCurves& nc, int imin, int imax) {
    f32 s_min = nc.std_curve[imin];
    f32 s_max = nc.std_curve[imax];
    f32 d_min = nc.diff_curve[imin];
    f32 d_max = nc.diff_curve[imax];
    // Python: brightness_l = brightness[imin-1:imax+2], norm vs endpoints
    f32 b0 = (imin - 1) / (f32)k_n_brightness;
    f32 b1 = (imax + 1) / (f32)k_n_brightness;
    f32 denom = b1 - b0;

    f32 s2_min = s_min * s_min;
    f32 s2_max = s_max * s_max;
    f32 d2_min = d_min * d_min;
    f32 d2_max = d_max * d_max;

    for (int i = imin; i <= imax; ++i) {
        f32 b = i / (f32)k_n_brightness;
        f32 norm_b = (b - b0) / denom;
        f32 s2 = norm_b * (s2_max - s2_min) + s2_min;
        f32 d2 = norm_b * (d2_max - d2_min) + d2_min;
        nc.std_curve[i] = std::sqrt(std::max(0.f, s2));
        nc.diff_curve[i] = std::sqrt(std::max(0.f, d2));
    }
}

static NoiseCurves make_noise_curves(f32 alpha, f32 beta) {
    // Cache like Python (curves built once per alpha/beta, reused every frame).
    static NoiseCurves cached;
    static f32 cached_alpha = std::numeric_limits<f32>::quiet_NaN();
    static f32 cached_beta  = std::numeric_limits<f32>::quiet_NaN();
    if (alpha == cached_alpha && beta == cached_beta)
        return cached;

    NoiseCurves nc;
    nc.std_curve.resize((size_t)k_n_brightness + 1);
    nc.diff_curve.resize((size_t)k_n_brightness + 1);

    f32 xmin, xmax;
    get_non_linearity_bound(alpha, beta, k_tol, xmin, xmax);

    int imin = (int)std::ceil(xmin * (f32)k_n_brightness) + 1;
    int imax = (int)std::floor(xmax * (f32)k_n_brightness) - 1;

    // Python: if imin > n_brightness_levels: full regular MC
    const bool full_mc = (imin > k_n_brightness) || (imin >= imax) || (imin < 1) || (imax > k_n_brightness - 1);

    if (full_mc) {
        parallel_rows(k_n_brightness + 1, 0, [&](int i) {
            f32 b = i / (f32)k_n_brightness;
            unitary_MC(alpha, beta, b, nc.diff_curve[i], nc.std_curve[i]);
        });
    } else {
        // MC on non-linear parts: [0, imin] and [imax, 1000]
        parallel_rows(k_n_brightness + 1, 0, [&](int i) {
            if (i <= imin || i >= imax) {
                f32 b = i / (f32)k_n_brightness;
                unitary_MC(alpha, beta, b, nc.diff_curve[i], nc.std_curve[i]);
            }
        });
        // Overwrite [imin, imax] inclusive (matches run_fast_MC)
        interp_MC_range(nc, imin, imax);
    }

    cached = nc;
    cached_alpha = alpha;
    cached_beta = beta;
    return nc;
}

static Image compute_guide(const Image& raw, const Config& cfg) {
    if (!cfg.bayer_mode) {
        // Python: guide_img = raw.reshape((1, H, W))
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
                // Stored variance (sigma^2), same as Python local_stds
                vars.at(y, x, ch) = s2 / 9.f - m * m;
            }
        }
    }
}

static f32 sample_dogson(const Image& stats, f32 LR_y, f32 LR_x, int ch) {
    // Python OOB: HR[...] = 1/0  (+inf)
    if (!(LR_y >= 0.f && LR_y < (f32)stats.h && LR_x >= 0.f && LR_x < (f32)stats.w))
        return std::numeric_limits<f32>::infinity();

    // CUDA round / std::lround: half away from zero
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
    // Python: buffer[c] / w_acc (no zero check)
    return buf / w_acc;
}

static Image upscale_warp_stats(const Image& guide_stats,
                                bool is_ref, const FlowField* flow, int tile_size,
                                int num_threads) {
    const int nc = guide_stats.c;
    // Match Python upscale_warp_stats sizing: 3ch -> 2x, else same size
    const int out_h = (nc == 3) ? guide_stats.h * 2 : guide_stats.h;
    const int out_w = (nc == 3) ? guide_stats.w * 2 : guide_stats.w;
    // Match Python cuda_uspcale_dogson which hardcodes s = 2
    const f32 s = 2.f;
    Image out(out_h, out_w, nc);

    parallel_rows(out_h, num_threads, [&](int y) {
        for (int x = 0; x < out_w; ++x) {
            f32 flow_x = 0.f, flow_y = 0.f;
            if (!is_ref && flow) {
                // Python: patch_idy = int(y // tile_size)  (no clamp)
                int patch_idy = y / tile_size;
                int patch_idx = x / tile_size;
                flow_x = flow->dx(patch_idy, patch_idx);
                flow_y = flow->dy(patch_idy, patch_idx);
            }
            f32 LR_y = (y + flow_y + 0.5f) / s - 0.5f;
            f32 LR_x = (x + flow_x + 0.5f) / s - 0.5f;
            for (int ch = 0; ch < nc; ++ch) {
                out.at(y, x, ch) = sample_dogson(guide_stats, LR_y, LR_x, ch);
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
                // Python: id_noise = round(1000 * brightness)
                int id_noise = (int)std::lround(1000.f * brightness);
                id_noise = std::max(0, std::min(k_n_brightness, id_noise));
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
    const f32 inf = std::numeric_limits<f32>::infinity();
    std::vector<f32> S((size_t)flow.ny * flow.nx, s2);
    for (int ty = 0; ty < flow.ny; ++ty) {
        for (int tx = 0; tx < flow.nx; ++tx) {
            // Python: mini = +1/0, maxi = -1/0
            f32 mnx = inf, mny = inf, mxx = -inf, mxy = -inf;
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
    const f32 inf = std::numeric_limits<f32>::infinity();
    for (int y = 0; y < R.h; ++y) {
        for (int x = 0; x < R.w; ++x) {
            f32 mn = inf;
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
    st.means = upscale_warp_stats(means, true, nullptr, 0, cfg.num_threads);
    st.stds  = upscale_warp_stats(vars, true, nullptr, 0, cfg.num_threads);
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
    // Python discards comp local stds
    (void)comp_vars;
    comp_means = upscale_warp_stats(comp_means, false, &flow, tile_size, cfg.num_threads);

    const int h = comp_means.h, w = comp_means.w;
    Image d_p(h, w, ref_stats.means.c);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < d_p.c; ++ch)
                d_p.at(y, x, ch) = std::fabs(ref_stats.means.at(y, x, ch) - comp_means.at(y, x, ch));
        }
    }

    Image d_sq, sigma_sq;
    apply_noise_model(d_p, ref_stats.means, ref_stats.stds, nc, d_sq, sigma_sq);

    std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2);

    Image R(h, w, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Python: patch_idy = int(idy // tile_size)  (no clamp)
            int patch_idy = y / tile_size;
            int patch_idx = x / tile_size;
            f32 s = S[(size_t)patch_idy * flow.nx + patch_idx];
            f32 sig = sigma_sq.at(y, x);
            R.at(y, x) = clampf(s * std::exp(-d_sq.at(y, x) / sig) - cfg.r_t, 0.f, 1.f);
        }
    }
    return local_min_5x5(R);
}

} // namespace hhsr
