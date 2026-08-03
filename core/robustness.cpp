#include "stages.h"
#include "parallel.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cmath>
#include <string>
#include <vector>
#include <utility>
#ifdef __APPLE__
#include "metal_gpu.h"
#endif

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

// ============================================================================
// NumPy RandomState (legacy / frozen @ 1.16) — MT19937 + polar Box-Muller.
// Same generator family as np.random.randn. Seeded per brightness for a
// deterministic app default.
//
// Stock Python run_fast_MC is *unseeded* + multiprocessed, so curves differ
// every Python run. Bit-match a specific Python run by loading a dump:
//   HHSR_NOISE_CURVES_DIR=/path  with std_curve.bin + diff_curve.bin
//   (1001 float32 each) and optional meta.txt (alpha=… / beta=…).
// Export without editing the Python package:
//   tools/export_noise_curves.py
//   tools/run_sr_dump_noise_curves.py  (captures curves from one pipeline run)
// ============================================================================
struct NumpyRandomState {
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu;
    static constexpr uint32_t UPPER_MASK = 0x80000000u;
    static constexpr uint32_t LOWER_MASK = 0x7fffffffu;

    uint32_t key[N]{};
    int pos = N;
    int has_gauss = 0;
    double gauss = 0.0;

    explicit NumpyRandomState(uint32_t seed) { rk_seed(seed); }

    void rk_seed(uint32_t seed) {
        seed &= 0xffffffffu;
        for (int i = 0; i < N; ++i) {
            key[i] = seed;
            seed = (1812433253u * (seed ^ (seed >> 30)) + (uint32_t)i + 1u) & 0xffffffffu;
        }
        pos = N;
        has_gauss = 0;
        gauss = 0.0;
    }

    uint32_t rk_random() {
        uint32_t y;
        if (pos == N) {
            int i;
            for (i = 0; i < N - M; ++i) {
                y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
                key[i] = key[i + M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            for (; i < N - 1; ++i) {
                y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
                key[i] = key[i + (M - N)] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            y = (key[N - 1] & UPPER_MASK) | (key[0] & LOWER_MASK);
            key[N - 1] = key[M - 1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            pos = 0;
        }
        y = key[pos++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    double rk_double() {
        // NumPy randomkit: (a*2^26 + b) / 2^53
        long a = (long)(rk_random() >> 5);
        long b = (long)(rk_random() >> 6);
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    double rk_gauss() {
        if (has_gauss) {
            const double tmp = gauss;
            gauss = 0.0;
            has_gauss = 0;
            return tmp;
        }
        double f, x1, x2, r2;
        do {
            x1 = 2.0 * rk_double() - 1.0;
            x2 = 2.0 * rk_double() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        f = std::sqrt(-2.0 * std::log(r2) / r2);
        gauss = f * x1;
        has_gauss = 1;
        return f * x2;
    }
};

static void get_non_linearity_bound(f32 alpha, f32 beta, f32 tol, f32& xmin, f32& xmax) {
    // float64 like NumPy for the bound indices
    double a = (double)alpha, b = (double)beta, t = (double)tol;
    double tol_sq = t * t;
    xmin = (f32)(tol_sq / 2.0 * (a + std::sqrt(tol_sq * a * a + 4.0 * b)));
    double inner = std::pow(2.0 + tol_sq * a, 2.0) - 4.0 * (1.0 + tol_sq * b);
    xmax = (f32)((2.0 + tol_sq * a - std::sqrt(std::max(0.0, inner))) / 2.0);
}

static void unitary_MC(f32 alpha, f32 beta, f32 b, f32& diff_mean, f32& std_mean) {
    // Same estimator as fast_monte_carlo.unitary_MC (population std, |Δμ|),
    // same draw order (all patch1 then all patch2). RNG seed is C++-only.
    const double bd = (double)b;
    const double scale = std::sqrt(std::max(0.0, (double)alpha * bd + (double)beta));
    const uint32_t seed = 1337u + (uint32_t)std::lround(bd * (double)k_n_brightness);
    NumpyRandomState rng(seed);

    const int n = k_n_patches;
    auto fill_patch_stats = [&](std::vector<double>& means, std::vector<double>& stds) {
        means.resize((size_t)n);
        stds.resize((size_t)n);
        for (int i = 0; i < n; ++i) {
            double p[9];
            double m = 0.0;
            for (int j = 0; j < 9; ++j) {
                double v = bd + scale * rng.rk_gauss();
                p[j] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
                m += p[j];
            }
            m /= 9.0;
            double s = 0.0;
            for (int j = 0; j < 9; ++j) {
                double d = p[j] - m;
                s += d * d;
            }
            means[(size_t)i] = m;
            stds[(size_t)i] = std::sqrt(s / 9.0);
        }
    };

    // C-order (N,3,3): entire patch1 stream, then patch2 — same as NumPy randn.
    std::vector<double> m1, s1, m2, s2;
    fill_patch_stats(m1, s1);
    fill_patch_stats(m2, s2);

    double sum_std = 0.0;
    double sum_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_std += 0.5 * (s1[(size_t)i] + s2[(size_t)i]);
        sum_diff += std::fabs(m1[(size_t)i] - m2[(size_t)i]);
    }

    diff_mean = (f32)(sum_diff / n);
    std_mean = (f32)(sum_std / n);
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

static bool read_f32_bin(const std::string& path, std::vector<f32>& out, size_t expect) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.resize(expect);
    size_t n = std::fread(out.data(), sizeof(f32), expect, f);
    std::fclose(f);
    return n == expect;
}

static bool meta_matches(const std::string& dir, f32 alpha, f32 beta) {
    std::string path = dir + "/meta.txt";
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return true; // no meta → accept dump as authoritative
    double a = 0.0, b = 0.0;
    char line[256];
    bool got_a = false, got_b = false;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "alpha=%lf", &a) == 1) got_a = true;
        if (std::sscanf(line, "beta=%lf", &b) == 1) got_b = true;
    }
    std::fclose(f);
    if (!got_a || !got_b) return true;
    // Relative tolerance — DNG α/β are float32-ish.
    auto close = [](double x, double y) {
        double d = std::fabs(x - y);
        return d <= 1e-9 || d <= 1e-5 * std::max(std::fabs(x), std::fabs(y));
    };
    return close(a, (double)alpha) && close(b, (double)beta);
}

static std::string noise_curves_search_dir() {
    if (const char* env = std::getenv("HHSR_NOISE_CURVES_DIR"))
        return std::string(env);
    if (const char* dbg = std::getenv("HHSR_DEBUG_DIR"))
        return std::string(dbg) + "/noise_curves";
#ifdef __APPLE__
    if (const char* home = std::getenv("HOME"))
        return std::string(home) + "/Documents/noise_curves";
#endif
    return "noise_curves";
}

// Load Python-dumped curves (same unseeded np.random stream as that run).
static bool try_load_python_noise_curves(f32 alpha, f32 beta, NoiseCurves& nc) {
    const std::string dir = noise_curves_search_dir();
    if (!meta_matches(dir, alpha, beta)) return false;
    const size_t n = (size_t)k_n_brightness + 1;
    std::vector<f32> stdc, diffc;
    if (!read_f32_bin(dir + "/std_curve.bin", stdc, n)) return false;
    if (!read_f32_bin(dir + "/diff_curve.bin", diffc, n)) return false;
    nc.std_curve = std::move(stdc);
    nc.diff_curve = std::move(diffc);
    std::printf("[noise] Loaded Python curves from %s (%zu bins)\n", dir.c_str(), n);
    return true;
}

static const NoiseCurves& make_noise_curves(f32 alpha, f32 beta) {
    // Cache like Python (curves built once per alpha/beta, reused every frame).
    static NoiseCurves cached;
    static f32 cached_alpha = std::numeric_limits<f32>::quiet_NaN();
    static f32 cached_beta  = std::numeric_limits<f32>::quiet_NaN();
    static bool cached_from_file = false;
    if (alpha == cached_alpha && beta == cached_beta)
        return cached;

    NoiseCurves nc;
    if (try_load_python_noise_curves(alpha, beta, nc)) {
        cached = std::move(nc);
        cached_alpha = alpha;
        cached_beta = beta;
        cached_from_file = true;
        return cached;
    }
    (void)cached_from_file;

    nc.std_curve.resize((size_t)k_n_brightness + 1);
    nc.diff_curve.resize((size_t)k_n_brightness + 1);

    f32 xmin, xmax;
    get_non_linearity_bound(alpha, beta, k_tol, xmin, xmax);

    int imin = (int)std::ceil(xmin * (f32)k_n_brightness) + 1;
    int imax = (int)std::floor(xmax * (f32)k_n_brightness) - 1;

    // Python run_fast_MC: only this gate triggers full regular MC
    const bool full_mc = (imin > k_n_brightness);

    if (full_mc) {
        parallel_rows(k_n_brightness + 1, 0, [&](int i) {
            f32 b = i / (f32)k_n_brightness;
            unitary_MC(alpha, beta, b, nc.diff_curve[(size_t)i], nc.std_curve[(size_t)i]);
        });
    } else {
        // MC on non-linear parts: [0, imin] and [imax, 1000]
        parallel_rows(k_n_brightness + 1, 0, [&](int i) {
            if (i <= imin || i >= imax) {
                f32 b = i / (f32)k_n_brightness;
                unitary_MC(alpha, beta, b, nc.diff_curve[(size_t)i], nc.std_curve[(size_t)i]);
            }
        });
        // Overwrite [imin, imax] inclusive (matches run_fast_MC)
        interp_MC_range(nc, imin, imax);
    }

    cached = std::move(nc);
    cached_alpha = alpha;
    cached_beta = beta;
    cached_from_file = false;
    return cached;
}

} // namespace

f32 noise_std_at_brightness(f32 brightness, f32 alpha, f32 beta) {
    // Python: id_noise = round(1000*brightness); std = std_curve[id_noise] — no clamp
    const NoiseCurves& nc = make_noise_curves(alpha, beta);
    int id = (int)std::lround(1000.f * brightness);
    return nc.std_curve[(size_t)id];
}

void fetch_noise_curves(f32 alpha, f32 beta,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve) {
    const NoiseCurves& nc = make_noise_curves(alpha, beta);
    std_curve = nc.std_curve;
    diff_curve = nc.diff_curve;
}

namespace {

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
                    f32 v = raw.at(2 * y + i, 2 * x + j);
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

static Image local_lowpass_gaussian5x5(const Image& guide) {
    static constexpr f32 k[5] = {1.f, 4.f, 6.f, 4.f, 1.f};
    Image out(guide.h, guide.w, guide.c);
    for (int ch = 0; ch < guide.c; ++ch) {
        for (int y = 0; y < guide.h; ++y) {
            for (int x = 0; x < guide.w; ++x) {
                f32 s = 0.f;
                for (int i = -2; i <= 2; ++i) {
                    int yy = (int)clampf((f32)(y + i), 0.f, (f32)(guide.h - 1));
                    f32 wy = k[i + 2];
                    for (int j = -2; j <= 2; ++j) {
                        int xx = (int)clampf((f32)(x + j), 0.f, (f32)(guide.w - 1));
                        s += wy * k[j + 2] * guide.at(yy, xx, ch);
                    }
                }
                out.at(y, x, ch) = s / 256.f;
            }
        }
    }
    return out;
}

static f32 guide_noise_var(const Config& cfg, int nch, int ch, f32 brightness) {
    if (!std::isfinite(brightness)) brightness = 0.f;
    brightness = clampf(brightness, 0.f, 1.f);
    f32 v = std::max(cfg.alpha * brightness + cfg.beta, 0.f);
    if (nch == 3 && ch == 1)
        v *= 0.5f; // green guide channel is the average of two Bayer greens.
    return v;
}

static Image high_frequency_loss_map_adaptive(const Image& means, const Image& vars,
                                              const Image& lp_vars, const Config& cfg) {
    Image loss(vars.h, vars.w, 1);
    constexpr f32 kLocalVarianceNoiseScale = 8.f / 9.f;
    constexpr f32 kGaussian5x5NoiseEnergy = 4900.f / 65536.f;
    constexpr f32 kMinTextureSnr = 4.f;
    for (int y = 0; y < vars.h; ++y) {
        for (int x = 0; x < vars.w; ++x) {
            f32 var = 0.f, lp_var = 0.f;
            f32 noise_var = 0.f, lp_noise_var = 0.f;
            for (int ch = 0; ch < vars.c; ++ch) {
                var += std::max(vars.at(y, x, ch), 0.f);
                lp_var += std::max(lp_vars.at(y, x, ch), 0.f);
                const f32 n = guide_noise_var(cfg, vars.c, ch, means.at(y, x, ch));
                noise_var += kLocalVarianceNoiseScale * n;
                lp_noise_var += kLocalVarianceNoiseScale * kGaussian5x5NoiseEnergy * n;
            }
            const f32 signal_var = std::max(var - noise_var, 0.f);
            const f32 signal_lp_var = std::max(lp_var - lp_noise_var, 0.f);
            const f32 min_signal_var = kMinTextureSnr * std::max(noise_var, 1.0e-20f);
            loss.at(y, x) = (signal_var > min_signal_var)
                ? std::max((signal_var - signal_lp_var) / signal_var, 0.f)
                : 0.f;
        }
    }
    return loss;
}

static f32 guide_edge_strength_sq(const Image& means, int y, int x) {
    if (means.h <= 0 || means.w <= 0 || means.c <= 0) return 0.f;
    const int xm = (int)clampf((f32)(x - 1), 0.f, (f32)(means.w - 1));
    const int xp = (int)clampf((f32)(x + 1), 0.f, (f32)(means.w - 1));
    const int ym = (int)clampf((f32)(y - 1), 0.f, (f32)(means.h - 1));
    const int yp = (int)clampf((f32)(y + 1), 0.f, (f32)(means.h - 1));
    f32 edge_sq = 0.f;
    for (int ch = 0; ch < means.c; ++ch) {
        const f32 gx = 0.5f * (means.at(y, xp, ch) - means.at(y, xm, ch));
        const f32 gy = 0.5f * (means.at(yp, x, ch) - means.at(ym, x, ch));
        edge_sq = std::max(edge_sq, gx * gx + gy * gy);
    }
    return edge_sq;
}

static f32 guide_edge_strength_sq_neighborhood(const Image& means, int y, int x, int radius) {
    radius = std::max(0, std::min(2, radius));
    f32 edge_sq = 0.f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int yy = (int)clampf((f32)(y + dy), 0.f, (f32)(means.h - 1));
        for (int dx = -radius; dx <= radius; ++dx) {
            const int xx = (int)clampf((f32)(x + dx), 0.f, (f32)(means.w - 1));
            edge_sq = std::max(edge_sq, guide_edge_strength_sq(means, yy, xx));
        }
    }
    return edge_sq;
}

static f32 guide_brightness(const Image& means, int y, int x) {
    if (means.h <= 0 || means.w <= 0 || means.c <= 0 ||
        y < 0 || y >= means.h || x < 0 || x >= means.w)
        return 0.f;
    f32 sum = 0.f;
    for (int ch = 0; ch < means.c; ++ch)
        sum += means.at(y, x, ch);
    return clampf(sum / (f32)means.c, 0.f, 1.f);
}

static bool motion_edge_reject(const Image& ref_means, const Image& comp_means,
                               const std::vector<uint32_t>& motion_irregular,
                               size_t pidx, int y, int x, int new_y, int new_x,
                               f32 residual_ratio, const Config& cfg) {
    if (!cfg.motion_edge_rejection_enabled) return false;
    if (pidx >= motion_irregular.size() || motion_irregular[pidx] == 0u) return false;
    if (!std::isfinite(residual_ratio) ||
        residual_ratio <= cfg.motion_edge_residual_threshold)
        return false;

    const int edge_radius = std::max(0, cfg.motion_edge_neighborhood_radius);
    f32 edge_sq = guide_edge_strength_sq_neighborhood(ref_means, y, x, edge_radius);
    f32 brightness = guide_brightness(ref_means, y, x);
    if (new_y >= 0 && new_y < comp_means.h && new_x >= 0 && new_x < comp_means.w)
    {
        edge_sq = std::max(edge_sq,
                           guide_edge_strength_sq_neighborhood(comp_means, new_y, new_x,
                                                               edge_radius));
        brightness = std::max(brightness, guide_brightness(comp_means, new_y, new_x));
    }
    const f32 noise_var = std::max(0.f, cfg.alpha * brightness + cfg.beta);
    const f32 noise_edge_floor =
        std::max(0.f, cfg.motion_edge_noise_floor_multiplier) * std::sqrt(noise_var);
    const f32 th = std::max(cfg.motion_edge_threshold, 0.f);
    const f32 effective_th = std::max(th, noise_edge_floor);
    return edge_sq > effective_th * effective_th;
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
            if (!is_ref && flow && tile_size > 0 && flow->ny > 0 && flow->nx > 0 &&
                !flow->flow.empty()) {
                // Python: patch_idy = int(y // tile_size)  (no clamp)
                int patch_idy = y / tile_size;
                int patch_idx = x / tile_size;
                if (patch_idy >= 0 && patch_idy < flow->ny &&
                    patch_idx >= 0 && patch_idx < flow->nx) {
                    flow_x = flow->dx(patch_idy, patch_idx);
                    flow_y = flow->dy(patch_idy, patch_idx);
                }
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
                // Python: id_noise = round(1000 * brightness) — no clamp.
                int id_noise = (int)std::lround(1000.f * brightness);
                // Host: same bins as Python for finite brightness in range; avoid crash on +inf.
                if (!std::isfinite(brightness))
                    id_noise = 0;
                else if (id_noise < 0)
                    id_noise = 0;
                else if (id_noise >= (int)nc.std_curve.size())
                    id_noise = (int)nc.std_curve.size() - 1;
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

static std::vector<f32> compute_s(const FlowField& flow, f32 Mt, f32 s1, f32 s2,
                                  std::vector<uint32_t>* irregular_out = nullptr) {
    const f32 inf = std::numeric_limits<f32>::infinity();
    std::vector<f32> S((size_t)flow.ny * flow.nx, s2);
    if (irregular_out) irregular_out->assign((size_t)flow.ny * flow.nx, 0u);
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
            const bool irregular = d0 * d0 + d1 * d1 > Mt * Mt;
            const size_t idx = (size_t)ty * flow.nx + tx;
            S[idx] = irregular ? s1 : s2;
            if (irregular_out) (*irregular_out)[idx] = irregular ? 1u : 0u;
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
    if (!cfg.robustness_enabled) return RefStats();
#ifdef __APPLE__
    // Metal GPU only — same math as the CPU path below (golden reference).
    RefStats gpu = init_robustness_metal(ref_raw, cfg);
    if (gpu.means.h > 0 && gpu.means.w > 0) return gpu;
    return RefStats();
#else
    RefStats st;
    Image guide = compute_guide(ref_raw, cfg);
    Image means, vars;
    local_stats_3x3(guide, means, vars);
    if (cfg.hf_artifact_removal_enabled) {
        Image lp_guide = local_lowpass_gaussian5x5(guide);
        Image lp_means, lp_vars;
        local_stats_3x3(lp_guide, lp_means, lp_vars);
        st.hf_loss = high_frequency_loss_map_adaptive(means, vars, lp_vars, cfg);
    }
    // 460-main keeps robustness local statistics on the guide grid
    // (H/2 x W/2 x RGB for Bayer), not upsampled back to raw resolution.
    st.means = std::move(means);
    st.stds  = std::move(vars);
    return st;
#endif
}

Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg) {
    if (!cfg.robustness_enabled) {
        Image guide = compute_guide(comp_raw, cfg);
        Image r(guide.h, guide.w, 1);
        std::fill(r.data.begin(), r.data.end(), 1.f);
        return r;
    }
    // Empty flow (e.g. grey/align failed) — do not index flow.flow.data()==nullptr.
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty() || tile_size <= 0) {
        // Do not fully trust comps when alignment produced no flow (Python has no
        // such bandage; ones here made the mask white and let ghosts through).
        Image guide = compute_guide(comp_raw, cfg);
        Image r(guide.h, guide.w, 1);
        std::fill(r.data.begin(), r.data.end(), 0.f);
        return r;
    }

#ifdef __APPLE__
    // Metal GPU only — same Alg. robustness math as the CPU path below.
    Image gpu = compute_robustness_metal(comp_raw, ref_stats, flow, tile_size, cfg);
    if (gpu.h > 0 && gpu.w > 0) return gpu;
    return Image();
#else

    const NoiseCurves& nc = make_noise_curves(cfg.alpha, cfg.beta);

    Image guide = compute_guide(comp_raw, cfg);
    Image comp_means, comp_vars;
    local_stats_3x3(guide, comp_means, comp_vars);
    Image comp_hf_loss;
    if (cfg.hf_artifact_removal_enabled) {
        Image lp_guide = local_lowpass_gaussian5x5(guide);
        Image lp_means, lp_vars;
        local_stats_3x3(lp_guide, lp_means, lp_vars);
        comp_hf_loss = high_frequency_loss_map_adaptive(comp_means, comp_vars, lp_vars, cfg);
    }

    const int h = comp_means.h, w = comp_means.w;
    Image d_p(h, w, ref_stats.means.c);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            f32 flow_x = 0.f, flow_y = 0.f;
            int patch_idy = 0, patch_idx = 0;
            if (d_p.c == 1) {
                patch_idy = y / tile_size;
                patch_idx = x / tile_size;
                flow_x = flow.dx(patch_idy, patch_idx);
                flow_y = flow.dy(patch_idy, patch_idx);
            } else {
                patch_idy = (int)((2.f * (f32)y + 0.5f) / (f32)tile_size);
                patch_idx = (int)((2.f * (f32)x + 0.5f) / (f32)tile_size);
                flow_x = 0.5f * flow.dx(patch_idy, patch_idx);
                flow_y = 0.5f * flow.dy(patch_idy, patch_idx);
            }

            const int new_x = (int)std::lround((f32)x + flow_x);
            const int new_y = (int)std::lround((f32)y + flow_y);
            const bool inbound = (new_x >= 0 && new_x < w && new_y >= 0 && new_y < h);
            for (int ch = 0; ch < d_p.c; ++ch) {
                d_p.at(y, x, ch) = inbound
                    ? std::fabs(ref_stats.means.at(y, x, ch) - comp_means.at(new_y, new_x, ch))
                    : std::numeric_limits<f32>::infinity();
            }
        }
    }

    Image d_sq, sigma_sq;
    apply_noise_model(d_p, ref_stats.means, ref_stats.stds, nc, d_sq, sigma_sq);

    std::vector<uint32_t> motion_irregular;
    std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2,
                                   (cfg.motion_edge_rejection_enabled ||
                                    cfg.hf_artifact_removal_enabled)
                                       ? &motion_irregular
                                       : nullptr);
    if (!cfg.motion_edge_rejection_enabled && !cfg.hf_artifact_removal_enabled)
        motion_irregular.assign(S.size(), 0u);

    Image R(h, w, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int patch_idy, patch_idx;
            if (ref_stats.means.c == 3) {
                patch_idy = (int)((2.f * (f32)y + 0.5f) / (f32)tile_size);
                patch_idx = (int)((2.f * (f32)x + 0.5f) / (f32)tile_size);
            } else {
                patch_idy = y / tile_size;
                patch_idx = x / tile_size;
            }
            f32 flow_x = 0.f, flow_y = 0.f;
            if (ref_stats.means.c == 3) {
                flow_x = 0.5f * flow.dx(patch_idy, patch_idx);
                flow_y = 0.5f * flow.dy(patch_idy, patch_idx);
            } else {
                flow_x = flow.dx(patch_idy, patch_idx);
                flow_y = flow.dy(patch_idy, patch_idx);
            }
            const int new_x = (int)std::lround((f32)x + flow_x);
            const int new_y = (int)std::lround((f32)y + flow_y);
            const size_t pidx = (size_t)patch_idy * flow.nx + patch_idx;
            f32 s = S[pidx];
            f32 sig = sigma_sq.at(y, x);
            const f32 ratio = (sig > 0.f && std::isfinite(sig))
                ? d_sq.at(y, x) / sig
                : (d_sq.at(y, x) > 0.f ? std::numeric_limits<f32>::infinity() : 0.f);
            const bool residual_high =
                std::isfinite(ratio) && ratio > cfg.motion_edge_residual_threshold;
            const bool hf_reject =
                cfg.hf_artifact_removal_enabled &&
                pidx < motion_irregular.size() &&
                motion_irregular[pidx] != 0u &&
                residual_high &&
                (
                    (!ref_stats.hf_loss.data.empty() &&
                     ref_stats.hf_loss.at(y, x) > cfg.hf_variance_loss_threshold) ||
                    (new_x >= 0 && new_x < w && new_y >= 0 && new_y < h &&
                     !comp_hf_loss.data.empty() &&
                     comp_hf_loss.at(new_y, new_x) > cfg.hf_variance_loss_threshold)
                );
            const bool edge_reject =
                motion_edge_reject(ref_stats.means, comp_means, motion_irregular,
                                   pidx, y, x, new_y, new_x, ratio, cfg);
            R.at(y, x) = (hf_reject || edge_reject)
                ? 0.f
                : clampf(s * std::exp(-d_sq.at(y, x) / sig) - cfg.r_t, 0.f, 1.f);
        }
    }
    return local_min_5x5(R);
#endif
}

} // namespace hhsr
