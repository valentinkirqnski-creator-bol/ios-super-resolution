#include "stages.h"
#include "parallel.h"
#include "pixel4a_noise_curves.h"
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

static int closest_pixel4a_curve_iso(int iso) {
    int best = pixel4a_noise::kIsos[0];
    int best_d = std::abs(iso - best);
    for (int i = 1; i < pixel4a_noise::kIsoCount; ++i) {
        int d = std::abs(iso - pixel4a_noise::kIsos[i]);
        if (d < best_d) {
            best = pixel4a_noise::kIsos[i];
            best_d = d;
        }
    }
    return best;
}

static bool load_bundled_pixel4a_noise_curves(int iso, NoiseCurves& nc) {
    iso = closest_pixel4a_curve_iso(iso);
    int idx = pixel4a_noise::index_for_iso(iso);
    if (idx < 0) return false;
    const float* std_curve = pixel4a_noise::kStdCurves[idx];
    const float* diff_curve = pixel4a_noise::kDiffCurves[idx];
    nc.std_curve.assign(std_curve, std_curve + pixel4a_noise::kBins);
    nc.diff_curve.assign(diff_curve, diff_curve + pixel4a_noise::kBins);
    return true;
}

// Pure builder, no caching -- shared by the single-slot and per-channel
// cache wrappers below so the ~1e5-patch Monte Carlo logic exists once.
static NoiseCurves build_noise_curves(f32 alpha, f32 beta) {
    NoiseCurves nc;
    if (try_load_python_noise_curves(alpha, beta, nc))
        return nc;

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

    return nc;
}

static const NoiseCurves& make_noise_curves(f32 alpha, f32 beta) {
    // Cache like Python (curves built once per alpha/beta, reused every frame).
    static NoiseCurves cached;
    static f32 cached_alpha = std::numeric_limits<f32>::quiet_NaN();
    static f32 cached_beta  = std::numeric_limits<f32>::quiet_NaN();
    if (alpha == cached_alpha && beta == cached_beta)
        return cached;
    cached = build_noise_curves(alpha, beta);
    cached_alpha = alpha;
    cached_beta = beta;
    return cached;
}

static const NoiseCurves& make_noise_curves(const Config& cfg) {
    if (cfg.debug_pixel4a_noise_profile) {
        static NoiseCurves cached_pixel4a;
        static int cached_iso = 0;
        int iso = cfg.debug_pixel4a_noise_curve_iso > 0
            ? cfg.debug_pixel4a_noise_curve_iso
            : 100;
        iso = closest_pixel4a_curve_iso(iso);
        if (!cached_pixel4a.std_curve.empty() && cached_iso == iso)
            return cached_pixel4a;

        NoiseCurves nc;
        if (load_bundled_pixel4a_noise_curves(iso, nc)) {
            cached_pixel4a = std::move(nc);
            cached_iso = iso;
            std::printf("[noise] Loaded bundled Pixel 4a ISO %d curves (%d bins)\n",
                        iso, pixel4a_noise::kBins);
            return cached_pixel4a;
        }
    }
    return make_noise_curves(cfg.noise_alpha(), cfg.noise_beta());
}

// Per-guide-channel curve, 3 independently cached slots (one per CFA colour)
// rather than routing through the single-slot cache above: R/G/B typically
// have different alpha'/beta' after white balance, so 3 calls through a
// 1-slot cache would evict and rebuild the Monte Carlo curve on every call --
// 3x the cost every frame instead of once per burst. debug_pixel4a_noise_
// profile has no per-channel data (it's a fixed bundled table for parity
// checks against the reference implementation), so every channel shares
// that one curve, same as before this function existed.
static const NoiseCurves& make_noise_curves_channel(f32 alpha, f32 beta, int ch) {
    static NoiseCurves cached[3];
    static f32 cached_alpha[3] = {std::numeric_limits<f32>::quiet_NaN(),
                                  std::numeric_limits<f32>::quiet_NaN(),
                                  std::numeric_limits<f32>::quiet_NaN()};
    static f32 cached_beta[3] = {std::numeric_limits<f32>::quiet_NaN(),
                                 std::numeric_limits<f32>::quiet_NaN(),
                                 std::numeric_limits<f32>::quiet_NaN()};
    ch = std::max(0, std::min(2, ch));
    if (alpha == cached_alpha[ch] && beta == cached_beta[ch])
        return cached[ch];
    cached[ch] = build_noise_curves(alpha, beta);
    cached_alpha[ch] = alpha;
    cached_beta[ch] = beta;
    return cached[ch];
}

static const NoiseCurves& make_noise_curves_channel(const Config& cfg, int ch) {
    if (cfg.debug_pixel4a_noise_profile)
        return make_noise_curves(cfg);
    return make_noise_curves_channel(cfg.noise_alpha_ch(ch), cfg.noise_beta_ch(ch), ch);
}

} // namespace

// Python indexes std_curve[round(1000*brightness)] with no clamp, which is safe
// there only because the loader clipped every sample to [0,1]. Now that white
// balance is applied without an upper clip -- a 2x red gain puts any raw value
// above 0.49 past 1.0 -- the frame mean these are called with can exceed 1 on a
// bright scene, and the curve has 1001 entries. Clamped rather than left to
// read past the end.
static inline size_t noise_curve_index(f32 brightness, size_t n) {
    if (!std::isfinite(brightness) || n == 0) return 0;
    const long id = std::lround(1000.f * brightness);
    if (id < 0) return 0;
    return (size_t)std::min<long>(id, (long)n - 1);
}

f32 noise_std_at_brightness(f32 brightness, f32 alpha, f32 beta) {
    const NoiseCurves& nc = make_noise_curves(alpha, beta);
    return nc.std_curve[noise_curve_index(brightness, nc.std_curve.size())];
}

f32 noise_std_at_brightness(f32 brightness, const Config& cfg) {
    const NoiseCurves& nc = make_noise_curves(cfg);
    return nc.std_curve[noise_curve_index(brightness, nc.std_curve.size())];
}

void fetch_noise_curves(f32 alpha, f32 beta,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve) {
    const NoiseCurves& nc = make_noise_curves(alpha, beta);
    std_curve = nc.std_curve;
    diff_curve = nc.diff_curve;
}

void fetch_noise_curves(const Config& cfg,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve) {
    const NoiseCurves& nc = make_noise_curves(cfg);
    std_curve = nc.std_curve;
    diff_curve = nc.diff_curve;
}

void fetch_noise_curves_channel(const Config& cfg, int ch,
                                std::vector<f32>& std_curve, std::vector<f32>& diff_curve) {
    const NoiseCurves& nc = make_noise_curves_channel(cfg, ch);
    std_curve = nc.std_curve;
    diff_curve = nc.diff_curve;
}

// Not in the anonymous namespace below: neural_flow's caller (pipeline_paths.cpp)
// needs the exact same guide image the classical robustness path scores
// against, rather than re-deriving its own and risking the two drifting.
Image compute_guide(const Image& raw, const Config& cfg) {
    if (!cfg.bayer_mode) {
        // Python: guide_img = raw.reshape((1, H, W))
        Image g(raw.h, raw.w, 1);
        g.data = raw.data;
        return g;
    }
    int gh = raw.h / 2, gw = raw.w / 2;
    Image guide(gh, gw, 3);
    // Divisor per colour taken from the CFA rather than assumed. Bit-identical
    // to the previous 0.5*gsum for any Bayer pattern -- scaling by 1 and by 1/2
    // are both exact in IEEE754 -- but it stays correct, and stays in step with
    // Config::noise_guide_weight, if a pattern ever arrives with a different
    // count.
    f32 inv[3];
    for (int c = 0; c < 3; ++c) {
        const int n = cfg.cfa.count((uint8_t)c);
        inv[c] = (n > 0) ? 1.f / (f32)n : 0.f;
    }
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            f32 sum[3] = {0.f, 0.f, 0.f};
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    const uint8_t c = cfg.cfa.p[i][j];
                    if (c < 3) sum[c] += raw.at(2 * y + i, 2 * x + j);
                }
            }
            for (int c = 0; c < 3; ++c) guide.at(y, x, c) = sum[c] * inv[c];
        }
    }
    return guide;
}

namespace {

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

// Defined below, next to the other noise-model helpers.
static f32 guide_noise_var(const Config& cfg, int nch, int ch, f32 brightness);

static Image high_frequency_loss_map_adaptive(const Image& means, const Image& vars,
                                              const Image& lp_vars, const Config& cfg) {
    Image loss(vars.h, vars.w, 1);
    constexpr f32 kLocalVarianceNoiseScale = 8.f / 9.f;
    constexpr f32 kGaussian5x5NoiseEnergy = 4900.f / 65536.f;
    const f32 kMinTextureSnr = std::max(cfg.hf_min_texture_snr, 0.f);
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

static f32 guide_noise_var(const Config& cfg, int nch, int ch, f32 brightness) {
    if (!std::isfinite(brightness)) brightness = 0.f;
    brightness = clampf(brightness, 0.f, 1.f);
    f32 v = std::max(cfg.noise_alpha() * brightness + cfg.noise_beta(), 0.f);
    if (nch == 3 && ch == 1)
        v *= 0.5f; // green guide channel is the average of two Bayer greens.
    return v;
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

static f32 sample_bilinear_or_inf(const Image& img, f32 y, f32 x, int ch) {
    if (!(y >= 0.f && y < (f32)img.h && x >= 0.f && x < (f32)img.w))
        return std::numeric_limits<f32>::infinity();
    const int y0 = (int)std::floor(y);
    const int x0 = (int)std::floor(x);
    const int y1 = std::min(y0 + 1, img.h - 1);
    const int x1 = std::min(x0 + 1, img.w - 1);
    const f32 fy = y - (f32)y0;
    const f32 fx = x - (f32)x0;
    const f32 top = img.at(y0, x0, ch) +
                    (img.at(y0, x1, ch) - img.at(y0, x0, ch)) * fx;
    const f32 bot = img.at(y1, x0, ch) +
                    (img.at(y1, x1, ch) - img.at(y1, x0, ch)) * fx;
    return top + (bot - top) * fy;
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
    const f32 noise_var =
        std::max(0.f, cfg.noise_alpha() * brightness + cfg.noise_beta());
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

// nc_ch: one curve pointer per guide channel (ref_means.c of them -- 3 for
// Bayer, 1 otherwise), each channel scored against its own curve rather
// than all channels sharing one built from the cross-channel mean of
// alpha'/beta'. See Config::noise_alpha_ch/noise_beta_ch and
// make_noise_curves_channel.
static void apply_noise_model(const Image& d_p, const Image& ref_means, const Image& ref_vars,
                              const NoiseCurves* const nc_ch[3], Image& d_sq, Image& sigma_sq) {
    const int n_ch = ref_means.c;
    d_sq = Image(ref_means.h, ref_means.w, 1);
    sigma_sq = Image(ref_means.h, ref_means.w, 1);
    for (int y = 0; y < ref_means.h; ++y) {
        for (int x = 0; x < ref_means.w; ++x) {
            // Eq. 6 aggregates each term into ONE scalar across channels
            // first (sigma = sqrt(sum of per-channel variances), Eq 6's d/
            // d_ms/d_md are bare per-pixel scalars, not per-channel), and
            // only then applies max()/shrinkage once. Summing per-channel
            // max(measured, noise-floor) instead of max-of-sums is not the
            // same computation: max(a,b) >= a and >= b always, so
            // sum(max(a_c,b_c)) >= max(sum(a_c), sum(b_c)) in every case,
            // strictly greater whenever which term dominates differs across
            // channels (e.g. a colored edge: real structure in one channel,
            // near-zero in the others). That inflates sigma^2, which shrinks
            // d^2/sigma^2 and makes R more forgiving than Eq. 6 specifies
            // exactly in that mixed-channel-dominance case.
            f32 sigma_ms_sq = 0.f, sigma_md_sq = 0.f;
            f32 d_ms_sq = 0.f, d_md_sq = 0.f;
            for (int ch = 0; ch < n_ch; ++ch) {
                const NoiseCurves& nc = *nc_ch[ch];
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
                sigma_ms_sq += ref_vars.at(y, x, ch);
                sigma_md_sq += sigma_t * sigma_t;
                f32 d_p_ = d_p.at(y, x, ch);
                d_ms_sq += d_p_ * d_p_;
                d_md_sq += d_t * d_t;
            }
            f32 sigma_sq_ = std::max(sigma_ms_sq, sigma_md_sq);
            f32 shrink = d_ms_sq / (d_ms_sq + d_md_sq);
            f32 d_sq_ = d_ms_sq * shrink * shrink;
            d_sq.at(y, x) = d_sq_;
            sigma_sq.at(y, x) = sigma_sq_;
        }
    }
}

// Same computation as apply_noise_model, with the d_p array folded in: the
// per-channel |ref - comp| is derived inline from ref_means/comp_means rather
// than read from a materialised buffer. At raw resolution that buffer is
// H*W*3 floats (146 MB on a 12 MP frame) written once and read once by the
// very next stage -- the single biggest allocation in the raw-resolution
// robustness path, and the one that pushed a multi-frame burst past the
// memory ceiling. Arithmetic is identical to apply_noise_model above
// (same sum-across-channels-then-combine-once order); an out-of-bounds
// Dodgson sample arrives as +inf in comp_means and must stay +inf in the
// difference so R lands at 0 downstream.
static void apply_noise_model_fused(const Image& ref_means, const Image& comp_means,
                                    const Image& ref_vars,
                                    const NoiseCurves* const nc_ch[3],
                                    Image& d_sq, Image& sigma_sq, int num_threads) {
    const int n_ch = ref_means.c;
    d_sq = Image(ref_means.h, ref_means.w, 1);
    sigma_sq = Image(ref_means.h, ref_means.w, 1);
    parallel_rows(ref_means.h, num_threads, [&](int y) {
        for (int x = 0; x < ref_means.w; ++x) {
            f32 sigma_ms_sq = 0.f, sigma_md_sq = 0.f;
            f32 d_ms_sq = 0.f, d_md_sq = 0.f;
            for (int ch = 0; ch < n_ch; ++ch) {
                const NoiseCurves& nc = *nc_ch[ch];
                f32 brightness = ref_means.at(y, x, ch);
                int id_noise = (int)std::lround(1000.f * brightness);
                if (!std::isfinite(brightness))
                    id_noise = 0;
                else if (id_noise < 0)
                    id_noise = 0;
                else if (id_noise >= (int)nc.std_curve.size())
                    id_noise = (int)nc.std_curve.size() - 1;
                f32 sigma_t = nc.std_curve[(size_t)id_noise];
                f32 d_t = nc.diff_curve[(size_t)id_noise];
                sigma_ms_sq += ref_vars.at(y, x, ch);
                sigma_md_sq += sigma_t * sigma_t;
                const f32 comp = comp_means.at(y, x, ch);
                f32 d_p_ = std::isfinite(comp)
                    ? std::fabs(brightness - comp)
                    : std::numeric_limits<f32>::infinity();
                d_ms_sq += d_p_ * d_p_;
                d_md_sq += d_t * d_t;
            }
            f32 sigma_sq_ = std::max(sigma_ms_sq, sigma_md_sq);
            f32 shrink = d_ms_sq / (d_ms_sq + d_md_sq);
            d_sq.at(y, x) = d_ms_sq * shrink * shrink;
            sigma_sq.at(y, x) = sigma_sq_;
        }
    });
}

static std::vector<uint32_t> compute_tile_residual_high(const Image& d_sq,
                                                        const Image& sigma_sq,
                                                        const FlowField& flow,
                                                        int tile_size,
                                                        int guide_channels,
                                                        f32 residual_threshold,
                                                        bool already_raw_res = false) {
    const size_t n_tiles = (size_t)std::max(0, flow.ny) * (size_t)std::max(0, flow.nx);
    std::vector<uint32_t> out(n_tiles, 0u);
    if (n_tiles == 0 || tile_size <= 0 || !std::isfinite(residual_threshold))
        return out;

    std::vector<uint32_t> count(n_tiles, 0u);
    std::vector<uint32_t> high_count(n_tiles, 0u);
    for (int y = 0; y < d_sq.h; ++y) {
        for (int x = 0; x < d_sq.w; ++x) {
            int ty, tx;
            // already_raw_res: d_sq/sigma_sq were computed directly at raw
            // resolution (Config::robustness_raw_resolution_active), so the
            // guide->raw 2x+0.5 conversion below would double-scale them --
            // a plain tile_size divide is already the raw tile grid.
            if (!already_raw_res && guide_channels == 3) {
                ty = (int)((2.f * (f32)y + 0.5f) / (f32)tile_size);
                tx = (int)((2.f * (f32)x + 0.5f) / (f32)tile_size);
            } else {
                ty = y / tile_size;
                tx = x / tile_size;
            }
            if (ty < 0 || ty >= flow.ny || tx < 0 || tx >= flow.nx) continue;
            const size_t pidx = (size_t)ty * flow.nx + tx;
            const f32 sig = sigma_sq.at(y, x);
            const f32 dsq = d_sq.at(y, x);
            const f32 ratio = (sig > 0.f && std::isfinite(sig))
                ? dsq / sig
                : (dsq > 0.f ? std::numeric_limits<f32>::infinity() : 0.f);
            if (!std::isfinite(ratio)) continue;
            ++count[pidx];
            if (ratio > residual_threshold) ++high_count[pidx];
        }
    }

    for (size_t i = 0; i < n_tiles; ++i) {
        if (count[i] == 0) continue;
        const uint32_t need = std::max<uint32_t>(
            2u, (uint32_t)std::ceil((double)count[i] * 0.10));
        out[i] = high_count[i] >= need ? 1u : 0u;
    }
    return out;
}

static std::vector<f32> compute_s(const FlowField& flow, f32 Mt, f32 s1, f32 s2,
                                  std::vector<uint32_t>* irregular_out = nullptr) {
    const f32 inf = std::numeric_limits<f32>::infinity();
    std::vector<f32> S((size_t)flow.ny * flow.nx, s2);
    if (irregular_out) irregular_out->assign((size_t)flow.ny * flow.nx, 0u);
    // Measured on the alignment grid by mark_motion_irregular_tiles and carried
    // through flow_to_raw_tile_grid. Re-deriving it from a field whose tiles
    // have been duplicated 2x and whose displacements have been scaled 2x
    // measures a different span; see the note on FlowField::motion_irregular.
    if (flow.has_motion_prior()) {
        for (size_t i = 0; i < S.size(); ++i) {
            const bool irregular = flow.motion_irregular[i] != 0u;
            S[i] = irregular ? s1 : s2;
            if (irregular_out) (*irregular_out)[i] = irregular ? 1u : 0u;
        }
        return S;
    }
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
    // 460-main keeps robustness local statistics on the guide grid
    // (H/2 x W/2 x RGB for Bayer), not upsampled back to raw resolution.
    st.means = std::move(means);
    st.stds  = std::move(vars);
    if (cfg.hf_artifact_removal_enabled) {
        Image lp_guide = local_lowpass_gaussian5x5(guide);
        Image lp_means, lp_vars;
        local_stats_3x3(lp_guide, lp_means, lp_vars);
        st.hf_loss = high_frequency_loss_map_adaptive(st.means, st.stds, lp_vars, cfg);
    }
    if (cfg.robustness_raw_resolution_active()) {
        // is_ref=true: no flow warp, just the Dodgson upscale (Algorithm 6
        // never warps the reference's own stats -- only Gn's). Once per
        // burst here, not once per comparison frame.
        st.means_hires = upscale_warp_stats(st.means, /*is_ref=*/true, nullptr,
                                            0, cfg.num_threads);
        st.stds_hires = upscale_warp_stats(st.stds, /*is_ref=*/true, nullptr,
                                           0, cfg.num_threads);
    }
    return st;
#endif
}

// Algorithm 6, read literally: d^2/sigma^2/R computed at RAW resolution,
// reached by Dodgson-quadratic upscaling the guide-resolution local stats
// (warping the comparison frame's stats by the flow in the process) rather
// than computing everything directly at guide resolution the way
// compute_robustness below does. See Config::robustness_raw_resolution_
// enabled for why this exists and why it's decimate-only; RefStats::
// means_hires/stds_hires for the reference side (upscaled once per burst in
// init_robustness, not once per comparison frame).
//
// hf_artifact_removal_enabled's noise floor still reads ref_stats.hf_loss,
// which is guide-resolution (its own Dodgson upscale would be a further
// feature, not built here) -- mapped down from the raw pixel to its parent
// guide pixel for that one lookup. Neither that nor motion_edge_rejection_
// enabled are the common case this toggle is meant for; both stay correct,
// just at their existing granularity rather than the new one.
static Image compute_robustness_raw_res(const Image& comp_raw, const RefStats& ref_stats,
                                        const FlowField& flow, int tile_size,
                                        const Config& cfg, Image* s_select_out) {
    if (ref_stats.means_hires.h <= 0 || ref_stats.means_hires.w <= 0 ||
        ref_stats.stds_hires.h <= 0 || ref_stats.stds_hires.w <= 0) {
        // init_robustness didn't populate the hires stats (e.g. robustness
        // was off when the burst started and got toggled mid-burst) --
        // no raw-res path to run.
        return Image();
    }

    const NoiseCurves* nc_ch[3] = {nullptr, nullptr, nullptr};
    if (ref_stats.means.c == 3) {
        for (int ch = 0; ch < 3; ++ch)
            nc_ch[ch] = &make_noise_curves_channel(cfg, ch);
    } else {
        nc_ch[0] = &make_noise_curves(cfg);
    }

    // Comparison frame's own local stats, still built at guide resolution
    // (compute_guide/local_stats_3x3 are unchanged) -- only the upscale step
    // is new. is_ref=false: warped by the flow during the upscale, so every
    // raw pixel of comp_means already sits in the reference's coordinate
    // frame -- no separate shift-and-bilinear-sample afterward. comp_vars_
    // guide is a byproduct of local_stats_3x3 but, like the guide-resolution
    // path above, is never read -- apply_noise_model only ever uses the
    // REFERENCE's variance (ref_vars), so it isn't worth its own upscale pass.
    // Memory: at raw resolution every full-size buffer here is H*W*3 floats
    // -- 146 MB each on a 12 MP frame. The scopes free each guide-resolution
    // intermediate the moment it is dead, and the d_p difference array is
    // never materialised at all (folded into apply_noise_model_fused), so the
    // peak is the upscaled comp_means plus the two scalar outputs rather than
    // the whole chain at once.
    Image comp_means;
    {
        Image comp_means_guide;
        {
            Image guide = compute_guide(comp_raw, cfg);
            Image comp_vars_guide; // byproduct, never read -- freed with this scope
            local_stats_3x3(guide, comp_means_guide, comp_vars_guide);
        }
        comp_means = upscale_warp_stats(comp_means_guide, /*is_ref=*/false, &flow,
                                        tile_size, cfg.num_threads);
    }

    const Image& ref_means = ref_stats.means_hires;
    const Image& ref_vars = ref_stats.stds_hires;
    const int h = ref_means.h, w = ref_means.w;
    const int nch = ref_means.c;
    if (comp_means.h != h || comp_means.w != w || comp_means.c != nch)
        return Image();

    Image d_sq, sigma_sq;
    apply_noise_model_fused(ref_means, comp_means, ref_vars, nc_ch, d_sq, sigma_sq,
                            cfg.num_threads);
    std::vector<uint32_t> tile_residual_high;
    if (cfg.flow_reject_1d_enabled) {
        tile_residual_high = compute_tile_residual_high(
            d_sq, sigma_sq, flow, tile_size, ref_stats.means.c,
            cfg.flow_reject_1d_residual_threshold, /*already_raw_res=*/true);
    }

    std::vector<uint32_t> motion_irregular;
    std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2,
                                   (cfg.motion_edge_rejection_enabled ||
                                    cfg.hf_artifact_removal_enabled)
                                       ? &motion_irregular
                                       : nullptr);
    if (!cfg.motion_edge_rejection_enabled && !cfg.hf_artifact_removal_enabled)
        motion_irregular.assign(S.size(), 0u);

    const bool have_hf_loss = !ref_stats.hf_loss.data.empty();

    Image R(h, w, 1);
    if (s_select_out) *s_select_out = Image(h, w, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Raw resolution throughout -- a plain tile_size divide is
            // already the raw tile grid, no guide->raw rescale needed.
            const int patch_idy = y / tile_size;
            const int patch_idx = x / tile_size;
            const size_t pidx = (size_t)patch_idy * flow.nx + patch_idx;
            if (patch_idy < 0 || patch_idy >= flow.ny ||
                patch_idx < 0 || patch_idx >= flow.nx) {
                R.at(y, x) = 0.f;
                if (s_select_out) s_select_out->at(y, x) = 0.f;
                continue;
            }
            f32 s = S[pidx];
            f32 sig = sigma_sq.at(y, x);
            const f32 ratio = (sig > 0.f && std::isfinite(sig))
                ? d_sq.at(y, x) / sig
                : (d_sq.at(y, x) > 0.f ? std::numeric_limits<f32>::infinity() : 0.f);
            const bool residual_high =
                std::isfinite(ratio) && ratio > cfg.motion_edge_residual_threshold;
            (void)residual_high;
            const int gy = std::min(std::max(y / 2, 0), std::max(0, ref_stats.hf_loss.h - 1));
            const int gx = std::min(std::max(x / 2, 0), std::max(0, ref_stats.hf_loss.w - 1));
            const bool hf_reject =
                cfg.hf_artifact_removal_enabled &&
                pidx < motion_irregular.size() && motion_irregular[pidx] != 0u &&
                have_hf_loss &&
                ref_stats.hf_loss.at(gy, gx) > cfg.hf_variance_loss_threshold;
            // comp_means is already warped into the reference's coordinate
            // frame (the Dodgson upscale above), so the "moved" position for
            // the edge-strength neighbourhood lookup is just (y,x) again.
            const bool edge_reject =
                motion_edge_reject(ref_means, comp_means, motion_irregular,
                                   pidx, y, x, y, x, ratio, cfg);
            const bool aperture_limited =
                cfg.flow_reject_1d_enabled &&
                pidx < flow.aperture_limited.size() &&
                pidx < tile_residual_high.size() &&
                flow.aperture_limited[pidx] != 0u &&
                tile_residual_high[pidx] != 0u;
            if (aperture_limited) s = std::min(s, cfg.r_s1);
            const bool match_ambiguous =
                cfg.flow_reject_ambiguous_enabled &&
                pidx < flow.match_ambiguous.size() &&
                flow.match_ambiguous[pidx] != 0u;
            if (match_ambiguous) s = std::min(s, cfg.r_s1);
            const bool hard_reject = hf_reject || edge_reject;
            f32 r_val = hard_reject
                ? 0.f
                : clampf(s * std::exp(-d_sq.at(y, x) / sig) - cfg.r_t, 0.f, 1.f);
            // An out-of-bounds Dodgson sample writes +inf into comp_means by
            // design ("infinite will imply R = 0"), which makes d_sq +inf and
            // the Wiener shrink inf/inf = NaN, so r_val is NaN. The Python
            // clamp (CUDA fmaxf/fminf) returns the non-NaN operand and yields
            // the intended 0; clampf's comparisons are both false for NaN and
            // would return NaN, poisoning every merge accumulator that touches
            // this pixel.
            if (!std::isfinite(r_val)) r_val = 0.f;
            R.at(y, x) = r_val;
            if (s_select_out) s_select_out->at(y, x) = (s <= cfg.r_s1) ? 1.f : 0.f;
        }
    }
    // 5x5 min at RAW resolution -- 5x5 raw px, the IPOL reading of Eq. 9.
    // The guide-resolution path below min-pools the same window on the guide
    // grid, where it spans an effective 10x10 raw px.
    return local_min_5x5(R);
}

Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg,
                         Image* s_select_out) {
    if (!cfg.robustness_enabled) {
        Image guide = compute_guide(comp_raw, cfg);
        Image r(guide.h, guide.w, 1);
        std::fill(r.data.begin(), r.data.end(), 1.f);
        // Nothing was scored, so no prior was chosen. Report s2 uniformly so the
        // split masks stay well-formed and still sum to the combined one.
        if (s_select_out) *s_select_out = Image(guide.h, guide.w, 1);
        return r;
    }
    // Empty flow (e.g. grey/align failed) — do not index flow.flow.data()==nullptr.
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty() || tile_size <= 0) {
        // Do not fully trust comps when alignment produced no flow (Python has no
        // such bandage; ones here made the mask white and let ghosts through).
        Image guide = compute_guide(comp_raw, cfg);
        Image r(guide.h, guide.w, 1);
        std::fill(r.data.begin(), r.data.end(), 0.f);
        if (s_select_out) *s_select_out = Image(guide.h, guide.w, 1);
        return r;
    }

#ifdef __APPLE__
    // Metal GPU only — same Alg. robustness math as the CPU path below.
    Image gpu = compute_robustness_metal(comp_raw, ref_stats, flow, tile_size, cfg,
                                         s_select_out);
    if (gpu.h > 0 && gpu.w > 0) return gpu;
    return Image();
#else
    if (cfg.robustness_raw_resolution_active()) {
        Image raw_res = compute_robustness_raw_res(comp_raw, ref_stats, flow, tile_size,
                                                    cfg, s_select_out);
        if (raw_res.h > 0 && raw_res.w > 0) return raw_res;
        // Falls through to the guide-resolution path below if the hires ref
        // stats weren't populated (e.g. robustness was off when the burst
        // started).
    }

    // One curve per guide channel (3 for Bayer, matching R/(G1+G2)/2/B; 1
    // otherwise) rather than one curve shared by all channels -- see
    // Config::noise_alpha_ch/noise_beta_ch and make_noise_curves_channel.
    const NoiseCurves* nc_ch[3] = {nullptr, nullptr, nullptr};
    if (ref_stats.means.c == 3) {
        for (int ch = 0; ch < 3; ++ch)
            nc_ch[ch] = &make_noise_curves_channel(cfg, ch);
    } else {
        nc_ch[0] = &make_noise_curves(cfg);
    }

    Image guide = compute_guide(comp_raw, cfg);
    Image comp_means, comp_vars;
    local_stats_3x3(guide, comp_means, comp_vars);

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

            const f32 sample_x = (f32)x + flow_x;
            const f32 sample_y = (f32)y + flow_y;
            for (int ch = 0; ch < d_p.c; ++ch) {
                const f32 comp = sample_bilinear_or_inf(comp_means, sample_y, sample_x, ch);
                const f32 dp = std::isfinite(comp)
                    ? std::fabs(ref_stats.means.at(y, x, ch) - comp)
                    : std::numeric_limits<f32>::infinity();
                d_p.at(y, x, ch) = dp;
            }
        }
    }

    Image d_sq, sigma_sq;
    apply_noise_model(d_p, ref_stats.means, ref_stats.stds, nc_ch, d_sq, sigma_sq);
    std::vector<uint32_t> tile_residual_high;
    if (cfg.flow_reject_1d_enabled) {
        tile_residual_high = compute_tile_residual_high(
            d_sq, sigma_sq, flow, tile_size, ref_stats.means.c,
            cfg.flow_reject_1d_residual_threshold);
    }

    std::vector<uint32_t> motion_irregular;
    std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2,
                                   (cfg.motion_edge_rejection_enabled ||
                                    cfg.hf_artifact_removal_enabled)
                                       ? &motion_irregular
                                       : nullptr);
    if (!cfg.motion_edge_rejection_enabled && !cfg.hf_artifact_removal_enabled)
        motion_irregular.assign(S.size(), 0u);

    Image R(h, w, 1);
    if (s_select_out) *s_select_out = Image(h, w, 1);
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
            // Both required: an almost-entirely-high-frequency patch, and a
            // large local variation in the alignment vector field -- "the same
            // as used in the motion prior", i.e. the r_Mt test. Hair is
            // high-frequency but tracks cleanly; a noisy flat wall varies but
            // has no real high-frequency signal.
            const bool hf_reject =
                cfg.hf_artifact_removal_enabled &&
                pidx < motion_irregular.size() && motion_irregular[pidx] != 0u &&
                !ref_stats.hf_loss.data.empty() &&
                ref_stats.hf_loss.at(y, x) > cfg.hf_variance_loss_threshold;
            const bool edge_reject =
                motion_edge_reject(ref_stats.means, comp_means, motion_irregular,
                                   pidx, y, x, new_y, new_x, ratio, cfg);
            // Aperture-limited tiles are demoted to the irregular-motion prior
            // rather than discarded. The tile is not wrong, it is unverifiable:
            // the gradient constrains motion across the edge but not along it,
            // so one component of the flow is unmeasured rather than measured
            // badly. Zeroing threw away the component that WAS measured, and on
            // scenes with long edges -- architecture, horizons, railings -- that
            // removed enough of the burst to cost real detail. s1 keeps the tile
            // contributing while holding it to the same standard as any other
            // tile whose motion estimate is not trusted.
            const bool aperture_limited =
                cfg.flow_reject_1d_enabled &&
                pidx < flow.aperture_limited.size() &&
                pidx < tile_residual_high.size() &&
                flow.aperture_limited[pidx] != 0u &&
                tile_residual_high[pidx] != 0u;
            // min, not assignment: a tile already flagged motion-irregular must
            // not be promoted, and lower s is strictly stricter here.
            if (aperture_limited) s = std::min(s, cfg.r_s1);
            // Block matching found two near-equal minima here, so the offset it
            // picked is not distinguishable from at least one other. Demote to
            // the strict prior. This is the one input to the mask that does not
            // come from the image residual, which matters because the residual
            // cannot see this failure: the wrong offset was selected precisely
            // for producing a small difference.
            const bool match_ambiguous =
                cfg.flow_reject_ambiguous_enabled &&
                pidx < flow.match_ambiguous.size() &&
                flow.match_ambiguous[pidx] != 0u;
            if (match_ambiguous) s = std::min(s, cfg.r_s1);
            const bool hard_reject = hf_reject || edge_reject;
            f32 r_val = hard_reject
                ? 0.f
                : clampf(s * std::exp(-d_sq.at(y, x) / sig) - cfg.r_t, 0.f, 1.f);
            R.at(y, x) = r_val;
            // Compared against r_s1 rather than recomputing the conditions, so
            // the record cannot drift from the value actually used above.
            if (s_select_out) s_select_out->at(y, x) = (s <= cfg.r_s1) ? 1.f : 0.f;
        }
    }
    return local_min_5x5(R);
#endif
}

} // namespace hhsr
