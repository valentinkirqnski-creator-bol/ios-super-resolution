#include "stages.h"
#include "robustness_nn.h"
#include "debug_utils.h"
#include "parallel.h"
#include "pixel4a_noise_curves.h"
#include <algorithm>
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
    // WB-scaled per-channel alpha/beta, matching the WB'd guide. Only mask
    // paths call this wrapper.
    return make_noise_curves_channel(cfg.noise_alpha_ch_robustness(ch),
                                     cfg.noise_beta_ch_robustness(ch), ch);
}

// Mask-only variants: honour Config::debug_noise_model_disabled by building
// the curves from alpha = beta = 0 (so sigma_t = d_t = 0 in every bin),
// while make_noise_curves(cfg) itself stays ungated -- it is shared with SNR
// auto-tuning via noise_std_at_brightness, and gating it there changed the
// alignment tile size and merge constants along with the mask (measured:
// tile 16 -> 32), which is exactly what a diagnostic probe must not do.
static const NoiseCurves& mask_noise_curves(const Config& cfg) {
    if (cfg.debug_noise_model_disabled)
        return make_noise_curves(0.f, 0.f);
    if (cfg.debug_pixel4a_noise_profile)
        return make_noise_curves(cfg);
    return make_noise_curves(cfg.noise_alpha_robustness(), cfg.noise_beta_robustness());
}
static const NoiseCurves& mask_noise_curves_channel(const Config& cfg, int ch) {
    if (cfg.debug_noise_model_disabled)
        return make_noise_curves_channel(0.f, 0.f, ch);
    return make_noise_curves_channel(cfg, ch);
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

// Only the Metal robustness-mask host consumes these two, so they honour the
// mask-only noise-model kill switch; SNR reads noise_std_at_brightness, which
// stays ungated.
void fetch_noise_curves(const Config& cfg,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve) {
    const NoiseCurves& nc = mask_noise_curves(cfg);
    std_curve = nc.std_curve;
    diff_curve = nc.diff_curve;
}

void fetch_noise_curves_channel(const Config& cfg, int ch,
                                std::vector<f32>& std_curve, std::vector<f32>& diff_curve) {
    const NoiseCurves& nc = mask_noise_curves_channel(cfg, ch);
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

// See stages.h. One plane, guide resolution, unsmoothed -- the input the
// spatial residual channels and measure_match_quality both read. A plain mean
// over the guide colour planes rather than a luminance weighting: the
// robustness decision is about geometry, and an unweighted mean keeps the
// green channel from dominating what is really a structure detector.
Image guide_luma(const Image& guide) {
    Image l(guide.h, guide.w, 1);
    if (guide.c <= 0) return l;
    const f32 inv = 1.f / (f32)guide.c;
    for (int y = 0; y < guide.h; ++y)
        for (int x = 0; x < guide.w; ++x) {
            f32 s = 0.f;
            for (int c = 0; c < guide.c; ++c) s += guide.at(y, x, c);
            l.at(y, x) = s * inv;
        }
    return l;
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
    f32 v = std::max(cfg.noise_alpha_robustness() * brightness +
                     cfg.noise_beta_robustness(), 0.f);
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
        std::max(0.f, cfg.noise_alpha_robustness() * brightness +
                          cfg.noise_beta_robustness());
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

// Raw-resolution Eq. 9: the min is computed on the GUIDE lattice -- exactly
// where Wronski computes it, with his 5x5-guide = 10x10-raw footprint --
// then nearest-upsampled back to raw. The raw-res R first collapses
// 2x2 -> guide by MIN, so the denser raw-resolution detections are
// preserved (one rejected raw pixel still darkens its guide cell). Net:
// R is evaluated per raw pixel (the toggle's point), but the spatial
// safety margin is Wronski's own, on Wronski's own grid.
static Image local_min_5x5_on_guide(const Image& R);

RefStats init_robustness(const Image& ref_raw, const Config& cfg) {
    if (!cfg.robustness_enabled) return RefStats();
    // The learned correction AND the deterministic shape check both need the
    // reference guide UNSMOOTHED (shifted edges survive here; 3x3 means erase
    // them). This is the last place the reference raw is in scope. It has to
    // happen on both paths: on Metal the means and stds live on the GPU, and
    // no readback of them can reconstruct a plane that was never boxed.
    auto fill_nn_luma = [&](RefStats& st) {
        if (!cfg.use_neural_robustness && !cfg.robustness_shape_check_enabled)
            return;
        st.nn_luma = guide_luma(compute_guide(ref_raw, cfg));
    };
#ifdef __APPLE__
    // Metal GPU only — same math as the CPU path below (golden reference).
    RefStats gpu = init_robustness_metal(ref_raw, cfg);
    if (gpu.means.h > 0 && gpu.means.w > 0) { fill_nn_luma(gpu); return gpu; }
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
    fill_nn_luma(st);
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
            nc_ch[ch] = &mask_noise_curves_channel(cfg, ch);
    } else {
        nc_ch[0] = &mask_noise_curves(cfg);
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
    // Eq. 9 on Wronski's own lattice: 2x2 min-reduce to guide, 5x5 min
    // there (= the paper's 10x10-raw footprint), nearest-upsample back.
    // See local_min_5x5_on_guide.
    return local_min_5x5_on_guide(R);
}

Image robustness_local_min_on_guide(const Image& R) {
    return local_min_5x5_on_guide(R);
}

static Image local_min_5x5_on_guide(const Image& R) {
    const int gh = R.h / 2, gw = R.w / 2;
    if (gh <= 0 || gw <= 0) return local_min_5x5(R);
    Image G(gh, gw, 1);
    for (int gy = 0; gy < gh; ++gy) {
        for (int gx = 0; gx < gw; ++gx) {
            f32 m = R.at(2 * gy, 2 * gx);
            m = std::min(m, R.at(2 * gy, 2 * gx + 1));
            m = std::min(m, R.at(2 * gy + 1, 2 * gx));
            m = std::min(m, R.at(2 * gy + 1, 2 * gx + 1));
            G.at(gy, gx) = m;
        }
    }
    Image M = local_min_5x5(G);
    Image out(R.h, R.w, 1);
    for (int y = 0; y < R.h; ++y) {
        const int gy = std::min(gh - 1, y / 2);
        for (int x = 0; x < R.w; ++x)
            out.at(y, x) = M.at(gy, std::min(gw - 1, x / 2));
    }
    return out;
}

f32 robustness_analytic_R(const f32* ref_mean, const f32* ref_var,
                          const f32* comp_mean, f32 Mspan, const Config& cfg,
                          f32* ratio_out) {
    f32 d_ms_sq = 0.f, sig_ms_sq = 0.f, sig_md_sq = 0.f, d_md_sq = 0.f;
    for (int c = 0; c < 3; ++c) {
        const f32 dm = ref_mean[c] - comp_mean[c];
        d_ms_sq += dm * dm;
        sig_ms_sq += std::max(ref_var[c], 0.f);
        const NoiseCurves& nc = mask_noise_curves_channel(cfg, c);
        int id = (int)std::lround(1000.f * ref_mean[c]);
        id = std::min(std::max(id, 0), (int)nc.std_curve.size() - 1);
        const f32 st = nc.std_curve[(size_t)id];
        const f32 dt = nc.diff_curve[(size_t)id];
        sig_md_sq += st * st;
        d_md_sq   += dt * dt;
    }
    const f32 sig_sq = std::max(sig_ms_sq, sig_md_sq);
    const f32 shrink = d_ms_sq / std::max(d_ms_sq + d_md_sq, 1e-12f);
    const f32 d_sq = d_ms_sq * shrink * shrink;
    const f32 ratio = (sig_sq > 0.f) ? d_sq / sig_sq : 0.f;
    if (ratio_out) *ratio_out = std::isfinite(ratio) ? ratio : 1e6f;
    const f32 s_sel = (Mspan > cfg.r_Mt) ? cfg.r_s1 : cfg.r_s2;
    const f32 r = s_sel * std::exp(-ratio) - cfg.r_t;
    return std::isfinite(r) ? clampf(r, 0.f, 1.f) : 0.f;
}

// See stages.h. Two scale-free statistics from the block-matching cost surface
// around the offset the mask is being asked to judge.
//
// This is carried over from an earlier experiment (branch matchq-mask-4da49a9)
// where it was the one feature change that lifted the accuracy curve instead
// of sliding along it. Its first version was WRONG and the measurement caught
// it: comparing the chosen offset against a ring 2-3 px away scored
// uncorrupted tiles 0.09 and corrupted ones 0.78, backwards. It was measuring
// local STEEPNESS -- as large at a wrong offset sitting on a gradient as at a
// right one -- and its exclusion ring hid the true optimum whenever the error
// was smaller than the ring. Both statistics below are anchored on the MINIMUM
// over a real search window instead, which is the question that matters.
//
//   [0]  log(cost at the chosen offset / best cost anywhere in the window).
//        0 when the offset being judged IS the best correspondence available,
//        positive in proportion to how much better something nearby would have
//        been. This is what detects a misaligned tile.
//
//   [1]  log(best rival outside the winner basin / best cost). How UNIQUE that
//        best match is: large when the minimum is sharp and isolated, near 0
//        when several candidates tie. This is the evidence that does NOT
//        confuse aliasing with misalignment -- aliasing changes the residual
//        at the bottom of the cost surface without flattening it, so a
//        correctly aligned aliased tile keeps a sharp isolated minimum, while
//        a tile matched onto similar-looking content elsewhere does not.
//
// Both are ratios of two costs measured on the same tile, so tile contrast and
// brightness cancel and no contrast-dependent statistic is reintroduced. Both
// are floored at the sensor noise, because two costs that are both within the
// noise are not meaningfully different and without the floor a flat tile
// manufactures enormous ratios out of nothing.
//
// MEASURED ON THE FLOW BEING JUDGED, never on a known-good one. Measuring at
// the true offset would hand the network the answer and score wonderfully
// while being useless in the app.
std::vector<f32> measure_match_quality(const Image& ref_luma, const Image& comp_luma,
                                       const FlowField& flow, int tile_size,
                                       const Config& cfg) {
    std::vector<f32> out((size_t)flow.ny * flow.nx * 2, 0.f);
    if (ref_luma.h <= 0 || comp_luma.h <= 0 || tile_size <= 0) return out;
    if (ref_luma.data.empty() || comp_luma.data.empty()) return out;
    // The flow is on a RAW tile grid carrying RAW displacements; the luma
    // planes are guide resolution, i.e. half of raw.
    const f32 g_per_raw = 0.5f;
    const int tg = std::max(2, (int)std::lround(tile_size * g_per_raw));
    const int RS = 4;                       // search radius, guide px
    const int W = 2 * RS + 1;
    // Every second pixel in each axis. The cost surface is a tile-level
    // statistic and 16 samples of an 8x8 tile estimate it to well inside the
    // noise, while the full scan is 81 offsets x 64 pixels x 47k tiles = 247
    // MOP per comparison frame, which the 200 ms budget for the whole
    // correction cannot afford four times over.
    const int St = 2;
    const f32 alpha = cfg.noise_alpha_robustness(), beta = cfg.noise_beta_robustness();
    parallel_rows(flow.ny, cfg.num_threads, [&](int ty) {
        std::vector<double> c((size_t)W * W, 1e30);
        for (int tx = 0; tx < flow.nx; ++tx) {
            const int oy = (int)std::lround(ty * tile_size * g_per_raw);
            const int ox = (int)std::lround(tx * tile_size * g_per_raw);
            const f32 fx = flow.dx(ty, tx) * g_per_raw;
            const f32 fy = flow.dy(ty, tx) * g_per_raw;
            double sr = 0; int nref = 0;
            std::fill(c.begin(), c.end(), 1e30);
            for (int ddy = -RS; ddy <= RS; ++ddy)
                for (int ddx = -RS; ddx <= RS; ++ddx) {
                    double acc = 0; int n = 0;
                    for (int i = 0; i < tg; i += St)
                        for (int j = 0; j < tg; j += St) {
                            const int ry = oy + i, rx = ox + j;
                            if (ry < 0 || rx < 0 || ry >= ref_luma.h || rx >= ref_luma.w) continue;
                            const int my = (int)std::lround(ry + fy) + ddy;
                            const int mx = (int)std::lround(rx + fx) + ddx;
                            if (my < 0 || mx < 0 || my >= comp_luma.h || mx >= comp_luma.w) continue;
                            const f32 r = ref_luma.at(ry, rx);
                            acc += std::fabs(r - comp_luma.at(my, mx));
                            if (ddy == 0 && ddx == 0) { sr += r; ++nref; }
                            ++n;
                        }
                    if (n >= 8) c[(size_t)(ddy + RS) * W + (ddx + RS)] = acc / (double)n;
                }
            if (nref < 8) continue;
            const double mean_ref = sr / nref;
            const double c0 = c[(size_t)RS * W + RS];
            if (c0 > 1e29) continue;
            double cmin = 1e30; int by = 0, bx = 0;
            for (int ddy = -RS; ddy <= RS; ++ddy)
                for (int ddx = -RS; ddx <= RS; ++ddx) {
                    const double v = c[(size_t)(ddy + RS) * W + (ddx + RS)];
                    if (v < cmin) { cmin = v; by = ddy; bx = ddx; }
                }
            // Best rival OUTSIDE the winner basin, so the runner-up is a
            // genuinely different correspondence rather than the same one a
            // pixel over.
            double crival = 1e30;
            for (int ddy = -RS; ddy <= RS; ++ddy)
                for (int ddx = -RS; ddx <= RS; ++ddx) {
                    const int dy2 = ddy - by, dx2 = ddx - bx;
                    if (dy2 * dy2 + dx2 * dx2 < 4) continue;      // >= 2 px away
                    crival = std::min(crival, c[(size_t)(ddy + RS) * W + (ddx + RS)]);
                }
            const f32 nsig = std::sqrt(std::max(
                alpha * (f32)std::min(std::max(mean_ref, 0.0), 1.0) + beta, 0.f));
            const double floor_c = std::max((double)nsig, 1e-7);
            const double den = std::max(cmin, floor_c);
            const f32 offby = (f32)std::log(std::max(std::max(c0, floor_c) / den, 1.0));
            const f32 uniq = (crival < 1e29)
                ? (f32)std::log(std::max(std::max(crival, floor_c) / den, 1.0)) : 0.f;
            out[((size_t)ty * flow.nx + tx) * 2 + 0] =
                std::isfinite(offby) ? std::min(offby, 8.f) : 0.f;
            out[((size_t)ty * flow.nx + tx) * 2 + 1] =
                std::isfinite(uniq) ? std::min(uniq, 8.f) : 0.f;
        }
    });
    return out;
}

void ensure_robustness_nn_ref_hf(RefStats& rs, const Config& cfg) {
    const Image& rm = rs.means;
    if (rs.nn_hf.h == rm.h && rs.nn_hf.w == rm.w && !rs.nn_hf.data.empty()) return;
    if (rm.h <= 0 || rm.w <= 0 || rm.c != 3 ||
        rm.data.size() < (size_t)rm.h * rm.w * rm.c)
        return;                       // GPU-resident stats; caller fetches first
    const int h = rm.h, w = rm.w;
    Image hf(h, w, 1);
    // Separable 5x5 box over the three mean planes, then the mean absolute
    // deviation of the centre from it. Separable because the naive form is
    // 25 taps per channel and this runs on the full guide plane.
    Image blur(h, w, 3);
    parallel_rows(h, cfg.num_threads, [&](int y) {
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                f32 s = 0.f;
                for (int j = -2; j <= 2; ++j)
                    s += rm.at(y, std::min(std::max(x + j, 0), w - 1), c);
                blur.at(y, x, c) = s * 0.2f;
            }
    });
    parallel_rows(h, cfg.num_threads, [&](int y) {
        for (int x = 0; x < w; ++x) {
            f32 acc = 0.f;
            for (int c = 0; c < 3; ++c) {
                f32 s = 0.f;
                for (int i = -2; i <= 2; ++i)
                    s += blur.at(std::min(std::max(y + i, 0), h - 1), x, c);
                acc += std::fabs(rm.at(y, x, c) - s * 0.2f);
            }
            hf.at(y, x) = acc / 3.f;
        }
    });
    rs.nn_hf = std::move(hf);
}

Image build_robustness_nn_features(const RefStats& ref_stats, const Image& comp_means,
                                   const FlowField& flow, int tile_size,
                                   const Config& cfg, int y0, bool raw_res,
                                   int rows, const Image* comp_luma,
                                   const std::vector<f32>* match_q) {
    const Image& rm = raw_res ? ref_stats.means_hires : ref_stats.means;
    const Image& rv = raw_res ? ref_stats.stds_hires : ref_stats.stds;
    if (rm.h <= 0 || rm.w <= 0 || rm.c != 3 || rv.c != 3) return Image();
    // Dimensions are not enough. On the Metal path init_robustness returns a
    // RefStats with h/w/c filled and the pixel vectors EMPTY -- the statistics
    // stay resident on the GPU. Indexing that reads off the end of an empty
    // vector on every pixel, which is an out-of-bounds read on the first
    // comparison frame, not a graceful failure. Check the storage, not the
    // shape.
    if (rm.data.size() < (size_t)rm.h * rm.w * rm.c ||
        rv.data.size() < (size_t)rv.h * rv.w * rv.c)
        return Image();
    if (comp_means.h != rm.h || comp_means.w != rm.w || comp_means.c != 3) return Image();
    if (flow.ny <= 0 || flow.nx <= 0 || tile_size <= 0) return Image();

    const int h = rm.h, w = rm.w;
    // Channel 17's cache. Built once per burst by ensure_robustness_nn_ref_hf;
    // at raw resolution there is no cached plane, so the channel is zero there
    // (the raw-res path is off by default and was never trained with it).
    const Image* hf_plane = &ref_stats.nn_hf;
    const bool have_hf = !raw_res && hf_plane->h == rm.h && hf_plane->w == rm.w &&
                         hf_plane->data.size() >= (size_t)rm.h * rm.w;
    // Channels 20-26. Guide-resolution planes only: at raw resolution the
    // comparison statistics have already been Dodgson-upscaled AND warped, so
    // an unwarped luma plane would be in a different coordinate system from
    // everything around it -- the exact double-warp confusion this file has
    // been bitten by before. The raw-res path is off by default and was never
    // trained with these channels, so they are simply zero there.
    const Image* rl = &ref_stats.nn_luma;
    const bool have_sp = !raw_res && comp_luma != nullptr &&
                         rl->h == rm.h && rl->w == rm.w &&
                         rl->data.size() >= (size_t)rm.h * rm.w &&
                         comp_luma->h == rm.h && comp_luma->w == rm.w &&
                         comp_luma->data.size() >= (size_t)rm.h * rm.w;
    // Channels 18-19.
    const bool have_mq = match_q != nullptr &&
                         match_q->size() >= (size_t)flow.ny * flow.nx * 2;
    // rows == 0 selects the on-device strip height. The training generator
    // passes the whole plane instead: it writes crops from arbitrary origins,
    // and stitching them out of strips would only add a way to get the
    // indexing wrong in the one place where features and labels must line up.
    const int strip_h = (rows > 0) ? std::min(rows, h)
                                   : (kRobustnessNnStripRows + 2 * kRobustnessNnHalo);
    Image feat(strip_h, w, kRobustnessNnChannels);

    // ---- per-tile flow statistics, computed once ---------------------------
    //
    // These used to be recomputed inside the pixel loop, four times per pixel
    // (once per bilinear corner) at 3 MP -- a 3x3 scan of the flow field per
    // corner. The tile grid is ~47k entries against ~3M pixels, so hoisting it
    // is both much cheaper and the only way the three new channels below are
    // affordable at all.
    const int tny = flow.ny, tnx = flow.nx;
    std::vector<f32> t_span((size_t)tny * tnx, 0.f);   // Eq. 7's M
    std::vector<f32> t_resid((size_t)tny * tnx, 0.f);  // |flow - local median|
    std::vector<f32> t_rspan((size_t)tny * tnx, 0.f);  // span of that residual
    parallel_rows(tny, cfg.num_threads, [&](int ty) {
        f32 bx[9], by[9];
        for (int tx = 0; tx < tnx; ++tx) {
            f32 mnx = std::numeric_limits<f32>::infinity(), mny = mnx;
            f32 mxx = -mnx, mxy = -mnx;
            int n = 0;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j) {
                    const int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= tny || xx < 0 || xx >= tnx) continue;
                    const f32 vx = flow.dx(yy, xx), vy = flow.dy(yy, xx);
                    mnx = std::min(mnx, vx); mxx = std::max(mxx, vx);
                    mny = std::min(mny, vy); mxy = std::max(mxy, vy);
                    bx[n] = vx; by[n] = vy; ++n;
                }
            const f32 dxs = (mxx > mnx) ? (mxx - mnx) : 0.f;
            const f32 dys = (mxy > mny) ? (mxy - mny) : 0.f;
            t_span[(size_t)ty * tnx + tx] = std::sqrt(dxs * dxs + dys * dys);

            // Component-wise median of the 3x3 neighbourhood: a robust local
            // model of what the flow "should" be here. The residual against it
            // is the channel that separates the two cases the current mask
            // conflates. Under smooth camera rotation every neighbour agrees
            // with the local trend, so the residual is ~0 while the SPAN is
            // large -- which is precisely why keying on the span alone (all
            // the analytic mask can do, and measured at >70% of tiles over
            // r_Mt on these bursts) rejects rotation. One tile that has locked
            // onto the wrong match disagrees with its neighbours and the
            // residual is the size of the error.
            std::nth_element(bx, bx + n / 2, bx + n);
            std::nth_element(by, by + n / 2, by + n);
            const f32 ex = flow.dx(ty, tx) - bx[n / 2];
            const f32 ey = flow.dy(ty, tx) - by[n / 2];
            t_resid[(size_t)ty * tnx + tx] = std::sqrt(ex * ex + ey * ey);
        }
    });
    // Second pass: how rough is the neighbourhood's residual in general. Gives
    // the network a local scale to judge channel 15 against, so a large
    // residual in an area where everything is rough (real parallax, foliage)
    // reads differently from the same residual in a smoothly-flowing area.
    parallel_rows(tny, cfg.num_threads, [&](int ty) {
        for (int tx = 0; tx < tnx; ++tx) {
            f32 mx = 0.f;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j) {
                    const int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= tny || xx < 0 || xx >= tnx) continue;
                    mx = std::max(mx, t_resid[(size_t)yy * tnx + xx]);
                }
            t_rspan[(size_t)ty * tnx + tx] = mx;
        }
    });
    // y0 is the first source row of the window, which the caller keeps fully
    // inside the image. That matters: the window's edges then coincide with
    // the image's, so the convolutions' zero-padding is the same padding
    // whole-plane inference would apply. Extending past the edge instead --
    // by replicating or zero-filling rows here -- does NOT reproduce it,
    // because those rows carry bias-driven activations into the next layer
    // where whole-plane inference has true zeros. Verified bit-identical to
    // whole-plane output with this windowing, and visibly seamed without it.
    parallel_rows(strip_h, cfg.num_threads, [&](int sy) {
        const int y = std::min(std::max(y0 + sy, 0), h - 1);
        for (int x = 0; x < w; ++x) {
            // Guide pixel (y,x) covers raw pixels (2y,2x); the flow grid is
            // indexed in raw pixels, same convention as the analytic mask.
            // The flow field is one vector per tile. Sampling it nearest --
            // which is what the MERGE correctly does, since it must fetch the
            // pixel the search actually evaluated -- makes three of the input
            // channels piecewise constant over 16 raw pixels, and the network
            // draws what it is shown: a mask tiled into visible squares with
            // stair-stepped edges. For the DECISION "is this motion plausible
            // and consistent with its neighbours" the tile grid is an artifact
            // of the search, not a property of the scene, so the flow-derived
            // channels are sampled bilinearly between tile centres. This does
            // not change what the merge fetches; only what the mask reasons
            // about.
            // At raw resolution a pixel IS a raw pixel, so the tile index is a
            // plain divide; at guide resolution it covers two.
            const f32 pos_scale = raw_res ? 1.f : 2.f;
            const f32 tcy = (pos_scale * (f32)y) / (f32)tile_size - 0.5f;
            const f32 tcx = (pos_scale * (f32)x) / (f32)tile_size - 0.5f;
            const int t0y = (int)std::floor(tcy), t0x = (int)std::floor(tcx);
            const f32 ay = tcy - (f32)t0y, ax = tcx - (f32)t0x;
            auto tclamp = [](int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); };
            const int iy0 = tclamp(t0y, flow.ny), iy1 = tclamp(t0y + 1, flow.ny);
            const int ix0 = tclamp(t0x, flow.nx), ix1 = tclamp(t0x + 1, flow.nx);
            auto bilerp = [&](f32 v00, f32 v01, f32 v10, f32 v11) {
                const f32 top = v00 + (v01 - v00) * ax;
                const f32 bot = v10 + (v11 - v10) * ax;
                return top + (bot - top) * ay;
            };
            const f32 fx = bilerp(flow.dx(iy0, ix0), flow.dx(iy0, ix1),
                                  flow.dx(iy1, ix0), flow.dx(iy1, ix1));
            const f32 fy = bilerp(flow.dy(iy0, ix0), flow.dy(iy0, ix1),
                                  flow.dy(iy1, ix0), flow.dy(iy1, ix1));
            // Nearest tile is still needed wherever a genuinely per-tile
            // quantity is required.
            const int pty = tclamp((int)std::floor(tcy + 0.5f), flow.ny);
            const int ptx = tclamp((int)std::floor(tcx + 0.5f), flow.nx);

            // Local span of the flow field, Wronski Eq. 7 -- the same motion
            // statistic the analytic mask reduces to a binary s1/s2 choice.
            // Handed over as a continuous value so the network can grade it,
            // and bilinearly blended for the same reason as the flow above.
            auto tlook = [&](const std::vector<f32>& t) {
                return bilerp(t[(size_t)iy0 * tnx + ix0], t[(size_t)iy0 * tnx + ix1],
                              t[(size_t)iy1 * tnx + ix0], t[(size_t)iy1 * tnx + ix1]);
            };
            const f32 Mspan = tlook(t_span);

            // Comparison statistics sampled where the flow points, in guide
            // units (half the raw displacement), matching the generator.
            // upscale_warp_stats has already applied the flow when building the
            // hires comparison statistics, so at raw resolution the correct
            // sample sits at (y,x); shifting again would double-apply it.
            int qy = y, qx = x;
            if (!raw_res) {
                qy = std::min(std::max((int)std::lround((f32)y + 0.5f * fy), 0), h - 1);
                qx = std::min(std::max((int)std::lround((f32)x + 0.5f * fx), 0), w - 1);
            }

            f32 brightness = 0.f;
            for (int c = 0; c < 3; ++c) brightness += rm.at(y, x, c);
            brightness = clampf(brightness / 3.f, 0.f, 1.f);
            // Gated accessors, so the noise plane vanishes with the mask
            // noise-model toggle exactly as the analytic path's does.
            const f32 nsig = std::sqrt(std::max(cfg.noise_alpha_robustness() * brightness +
                                                cfg.noise_beta_robustness(), 0.f));

            f32* o = &feat.at(sy, x, 0);
            for (int c = 0; c < 3; ++c) o[c] = rm.at(y, x, c);
            // stds holds VARIANCE (see local_stats_3x3); the generator fed the
            // network standard deviations, so take the root here too.
            for (int c = 0; c < 3; ++c) o[3 + c] = std::sqrt(std::max(rv.at(y, x, c), 0.f));
            for (int c = 0; c < 3; ++c) o[6 + c] = comp_means.at(qy, qx, c);
            o[9] = fx;
            o[10] = fy;
            o[11] = Mspan;
            o[12] = nsig;

            // Channels 13-14: the analytic mask's own answer. Passing channel
            // 14 straight through IS the analytic mask, so the network starts
            // from it and only has to learn where to depart -- rather than
            // rebuilding Eq. 5-9 from scratch and doing worse in the regions
            // the closed form already handles.
            //
            // This is what fixes smooth areas. There sigma_ms collapses toward
            // zero and the decision rests entirely on the noise floor
            // sigma_md, which encodes sensor physics the network cannot infer
            // from a 3x3 neighbourhood at any amount of training. Handing it
            // the finished ratio puts that knowledge into the input.
            f32 rmean[3], rvar[3], cmean[3];
            for (int c = 0; c < 3; ++c) {
                rmean[c] = rm.at(y, x, c);
                rvar[c]  = rv.at(y, x, c);
                cmean[c] = comp_means.at(qy, qx, c);
            }
            f32 ratio = 0.f;
            const f32 r_an = robustness_analytic_R(rmean, rvar, cmean, Mspan,
                                                   cfg, &ratio);
            // log1p: the ratio is unbounded and heavy-tailed, and the input
            // normalisation is fitted over the training set, not per image.
            o[13] = std::log1p(std::max(ratio, 0.f));
            o[14] = r_an;

            // Channels 15-16: the flow's local CONSISTENCY, which is the
            // evidence neither the analytic mask nor the previous feature set
            // carried. Channel 11 (the span M) confuses "smooth camera
            // rotation" with "one tile is wrong" -- both make neighbouring
            // vectors differ -- and on these bursts it fires on 70-100% of
            // tiles, so as a rejection cue it is close to a constant. The
            // residual against the local median is near zero for the first
            // case and the size of the error for the second; the residual's
            // own neighbourhood span gives the local scale to read it against.
            o[15] = tlook(t_resid);
            o[16] = tlook(t_rspan);

            // Channel 17: local high-frequency energy of the reference. The
            // mean says how bright, the std how contrasty, but neither says
            // how FINE the structure is -- and that is what decides whether a
            // subpixel error costs anything. A thin rod and a soft gradient
            // can carry the same local std.
            //
            // Read from the per-burst cache, never recomputed here: it depends
            // only on the reference, and a 5x5 box over 3 channels inside this
            // loop would cost 75 reads/px on every comparison frame.
            o[17] = have_hf ? hf_plane->at(y, x) : 0.f;

            // Channels 18-19: the shape of the block-matching cost surface at
            // the tile this pixel sits in. Bilinear between tile centres for
            // the same reason the flow channels are -- a piecewise-constant
            // input makes the network draw the tile grid.
            if (have_mq) {
                auto mq = [&](int t, int u, int k) {
                    return (*match_q)[((size_t)t * tnx + u) * 2 + k];
                };
                o[18] = bilerp(mq(iy0, ix0, 0), mq(iy0, ix1, 0),
                               mq(iy1, ix0, 0), mq(iy1, ix1, 0));
                o[19] = bilerp(mq(iy0, ix0, 1), mq(iy0, ix1, 1),
                               mq(iy1, ix0, 1), mq(iy1, ix1, 1));
            } else {
                o[18] = 0.f; o[19] = 0.f;
            }

            // Channels 20-26: the spatial residual. Everything above is a
            // local statistic and a one-pixel edge shift moves none of them
            // appreciably; this is the evidence that does move.
            //
            // The comparison luma is fetched BILINEARLY at the sub-pixel
            // position the flow points to, not at the nearest pixel. Rounding
            // here would quantise the residual to whole guide pixels, i.e. to
            // two RAW pixels, and the errors this correction exists to catch
            // start below one.
            if (have_sp) {
                const f32 sy = (f32)y + 0.5f * fy;
                const f32 sx = (f32)x + 0.5f * fx;
                const f32 cy = std::min(std::max(sy, 0.f), (f32)(h - 1));
                const f32 cx = std::min(std::max(sx, 0.f), (f32)(w - 1));
                const int by0 = (int)cy, bx0 = (int)cx;
                const int by1 = std::min(by0 + 1, h - 1), bx1 = std::min(bx0 + 1, w - 1);
                const f32 wy = cy - (f32)by0, wx = cx - (f32)bx0;
                const f32 ctop = comp_luma->at(by0, bx0) +
                                 (comp_luma->at(by0, bx1) - comp_luma->at(by0, bx0)) * wx;
                const f32 cbot = comp_luma->at(by1, bx0) +
                                 (comp_luma->at(by1, bx1) - comp_luma->at(by1, bx0)) * wx;
                const f32 cl = ctop + (cbot - ctop) * wy;
                const f32 rlv = rl->at(y, x);
                const f32 res = rlv - cl;
                const int xm = (x > 0) ? x - 1 : 0, xp = (x < w - 1) ? x + 1 : w - 1;
                const int ym = (y > 0) ? y - 1 : 0, yp = (y < h - 1) ? y + 1 : h - 1;
                const f32 gx = 0.5f * (rl->at(y, xp) - rl->at(y, xm));
                const f32 gy = 0.5f * (rl->at(yp, x) - rl->at(ym, x));
                o[20] = rlv;
                o[21] = cl;
                o[22] = res;
                o[23] = gx;
                o[24] = gy;
                // Regularised displacement estimate. For a pure translation e
                // along the edge normal the residual is r ~ e.g, so e ~ r|g| /
                // |g|^2 -- but that divides by zero on flat content, where the
                // residual is pure noise and the answer must be "no evidence",
                // not "infinity". Tikhonov with the NOISE as the regulariser
                // gives exactly that: it decays smoothly to 0 as the gradient
                // falls below what the sensor can distinguish from grain.
                const f32 g2 = gx * gx + gy * gy;
                const f32 gm = std::sqrt(g2);
                const f32 eps = std::max(4.f * nsig * nsig, 1e-8f);
                const f32 disp = res * gm / (g2 + eps);
                o[25] = std::min(std::max(disp, -4.f), 4.f);
                o[26] = res / std::max(nsig, 1e-6f);
            } else {
                for (int c = 20; c < 27; ++c) o[c] = 0.f;
            }
        }
    });
    return feat;
}

Image compute_robustness_analytic(const Image& comp_raw, const RefStats& ref_stats,
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
            nc_ch[ch] = &mask_noise_curves_channel(cfg, ch);
    } else {
        nc_ch[0] = &mask_noise_curves(cfg);
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
            // The same NaN the raw-resolution path above already guards, on
            // the path that actually runs by default -- it was only ever fixed
            // on one of the two. d_sq/sig is 0/0 wherever the expected noise
            // variance collapses to zero and the difference is zero with it,
            // and inf/inf wherever an out-of-frame sample writes +inf into
            // comp_means; clampf's comparisons are both false for NaN, so the
            // NaN survives into R and poisons every merge accumulator that
            // touches the pixel. Measured on ok/burst1-3: 169 pixels in a 2 M
            // sample, i.e. rare, silent, and permanent where it happens.
            if (!std::isfinite(r_val)) r_val = 0.f;
            R.at(y, x) = r_val;
            // Compared against r_s1 rather than recomputing the conditions, so
            // the record cannot drift from the value actually used above.
            if (s_select_out) s_select_out->at(y, x) = (s <= cfg.r_s1) ? 1.f : 0.f;
        }
    }
    return local_min_5x5(R);
#endif
}

// Softstep in [0,1]: 0 below lo, 1 above hi, Hermite between.
static f32 shape_soft01(f32 x, f32 lo, f32 hi) {
    if (!(hi > lo)) return (x >= hi) ? 1.f : 0.f;
    f32 t = (x - lo) / (hi - lo);
    t = clampf(t, 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

static f32 shape_sample_luma(const Image& luma, f32 y, f32 x) {
    if (!(y >= 0.f && y < (f32)luma.h && x >= 0.f && x < (f32)luma.w))
        return std::numeric_limits<f32>::quiet_NaN();
    const int y0 = (int)std::floor(y);
    const int x0 = (int)std::floor(x);
    const int y1 = std::min(y0 + 1, luma.h - 1);
    const int x1 = std::min(x0 + 1, luma.w - 1);
    const f32 fy = y - (f32)y0;
    const f32 fx = x - (f32)x0;
    const f32 top = luma.at(y0, x0) + (luma.at(y0, x1) - luma.at(y0, x0)) * fx;
    const f32 bot = luma.at(y1, x0) + (luma.at(y1, x1) - luma.at(y1, x0)) * fx;
    return top + (bot - top) * fy;
}

// Per-tile |V - median(V_neighbours)| in RAW pixels. Smooth rotation keeps
 // this near zero; a single wrong lock does not. Same statistic as NN
// feature channel 15, without the bilinear blend -- one value per tile.
static std::vector<f32> shape_flow_median_residual(const FlowField& flow,
                                                    int num_threads) {
    const int ny = flow.ny, nx = flow.nx;
    std::vector<f32> out((size_t)std::max(0, ny) * (size_t)std::max(0, nx), 0.f);
    if (ny <= 0 || nx <= 0 || flow.flow.empty()) return out;
    parallel_rows(ny, num_threads, [&](int ty) {
        f32 bx[9], by[9];
        for (int tx = 0; tx < nx; ++tx) {
            int n = 0;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j) {
                    const int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= ny || xx < 0 || xx >= nx) continue;
                    bx[n] = flow.dx(yy, xx);
                    by[n] = flow.dy(yy, xx);
                    ++n;
                }
            if (n <= 0) continue;
            std::nth_element(bx, bx + n / 2, bx + n);
            std::nth_element(by, by + n / 2, by + n);
            const f32 ex = flow.dx(ty, tx) - bx[n / 2];
            const f32 ey = flow.dy(ty, tx) - by[n / 2];
            out[(size_t)ty * nx + tx] = std::sqrt(ex * ex + ey * ey);
        }
    });
    return out;
}

// Deterministic C_shape in [0,1]. Multiplies R_analytic; never raises it.
//
// Cue 1 (always): noise-normalized |ref - flow-warped comp| on UNSMOOTHED
// guide luma over a 3x3 window, gated by gradient-direction agreement so
// correctly aligned noise does not look like misalignment.
//
// Cue 2 (optional): local flow residual vs 3x3 median -- rotation-safe;
 // only strengthens a penalty that the photometric cue already supports.
static Image compute_shape_confidence(const Image& R_normal,
                                      const Image& ref_luma,
                                      const Image& comp_luma,
                                      const FlowField& flow,
                                      int tile_size,
                                      const Config& cfg) {
    const int h = R_normal.h, w = R_normal.w;
    Image C(h, w, 1);
    std::fill(C.data.begin(), C.data.end(), 1.f);
    if (h <= 0 || w <= 0 || tile_size <= 0) return C;
    if (ref_luma.h != h || ref_luma.w != w || comp_luma.h != h || comp_luma.w != w)
        return C;
    if (ref_luma.data.empty() || comp_luma.data.empty()) return C;
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty()) return C;

    const bool bayer = (cfg.bayer_mode);
    const f32 alpha = cfg.noise_alpha_robustness();
    const f32 beta = cfg.noise_beta_robustness();
    const f32 r_gate = std::max(cfg.shape_r_gate, 0.f);
    const f32 resid_lo = std::max(cfg.shape_resid_lo, 0.f);
    const f32 resid_hi = std::max(cfg.shape_resid_hi, resid_lo + 1e-3f);
    const f32 cos_lo = cfg.shape_grad_cos_lo;
    const f32 cos_hi = std::max(cfg.shape_grad_cos_hi, cos_lo + 1e-3f);
    const f32 min_edge = std::max(cfg.shape_min_edge_snr, 0.f);
    const f32 strength = clampf(cfg.shape_strength, 0.f, 1.f);
    const bool use_flow = cfg.shape_use_flow_geometry;
    const f32 flow_lo = std::max(cfg.shape_flow_resid_lo, 0.f);
    const f32 flow_hi = std::max(cfg.shape_flow_resid_hi, flow_lo + 1e-3f);

    std::vector<f32> flow_resid;
    if (use_flow)
        flow_resid = shape_flow_median_residual(flow, cfg.num_threads);

    parallel_rows(h, cfg.num_threads, [&](int y) {
        for (int x = 0; x < w; ++x) {
            const f32 Rn = R_normal.at(y, x);
            if (!(Rn > r_gate) || !std::isfinite(Rn)) {
                C.at(y, x) = 1.f;
                continue;
            }

            // Same tile indexing the analytic mask uses on the guide grid.
            int patch_idy, patch_idx;
            f32 flow_x, flow_y;
            if (bayer) {
                patch_idy = (int)((2.f * (f32)y + 0.5f) / (f32)tile_size);
                patch_idx = (int)((2.f * (f32)x + 0.5f) / (f32)tile_size);
                if (patch_idy < 0 || patch_idy >= flow.ny ||
                    patch_idx < 0 || patch_idx >= flow.nx) {
                    C.at(y, x) = 1.f;
                    continue;
                }
                // Raw displacement -> guide pixels (half). Sample comparison
                // ONCE at this offset -- do not also use a pre-warped plane.
                flow_x = 0.5f * flow.dx(patch_idy, patch_idx);
                flow_y = 0.5f * flow.dy(patch_idy, patch_idx);
            } else {
                patch_idy = y / tile_size;
                patch_idx = x / tile_size;
                if (patch_idy < 0 || patch_idy >= flow.ny ||
                    patch_idx < 0 || patch_idx >= flow.nx) {
                    C.at(y, x) = 1.f;
                    continue;
                }
                flow_x = flow.dx(patch_idy, patch_idx);
                flow_y = flow.dy(patch_idy, patch_idx);
            }

            f32 abs_sum = 0.f, ref_sum = 0.f;
            int n = 0;
            bool any_oob = false;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ry = y + di, rx = x + dj;
                    if (ry < 0 || ry >= h || rx < 0 || rx >= w) continue;
                    const f32 rv = ref_luma.at(ry, rx);
                    const f32 cv = shape_sample_luma(comp_luma,
                                                     (f32)ry + flow_y,
                                                     (f32)rx + flow_x);
                    if (!std::isfinite(cv)) { any_oob = true; continue; }
                    abs_sum += std::fabs(rv - cv);
                    ref_sum += rv;
                    ++n;
                }
            }
            if (n < 5) {
                // Mostly out of frame under this flow -- already handled by
                // the analytic path via +inf means; leave C alone.
                C.at(y, x) = any_oob ? 1.f : 1.f;
                continue;
            }

            const f32 mean_ref = clampf(ref_sum / (f32)n, 0.f, 1.f);
            // Luma is the mean of guide RGB; green is already half-weighted
            // in the Bayer guide. Match measure_match_quality's sigma.
            const f32 sigma = std::sqrt(std::max(alpha * mean_ref + beta, 0.f));
            const f32 sigma_floor = std::max(sigma, 1e-6f);
            const f32 resid_snr = (abs_sum / (f32)n) / sigma_floor;

            // Central-difference gradients on the unsmoothed planes.
            auto ref_at = [&](int yy, int xx) -> f32 {
                yy = std::min(std::max(yy, 0), h - 1);
                xx = std::min(std::max(xx, 0), w - 1);
                return ref_luma.at(yy, xx);
            };
            const f32 gx_r = 0.5f * (ref_at(y, x + 1) - ref_at(y, x - 1));
            const f32 gy_r = 0.5f * (ref_at(y + 1, x) - ref_at(y - 1, x));
            const f32 mag_r = std::sqrt(gx_r * gx_r + gy_r * gy_r);
            const f32 edge_snr = mag_r / sigma_floor;
            if (!(edge_snr > min_edge)) {
                C.at(y, x) = 1.f;   // no structure to verify
                continue;
            }

            const f32 c_xm = shape_sample_luma(comp_luma, (f32)y + flow_y,
                                               (f32)(x - 1) + flow_x);
            const f32 c_xp = shape_sample_luma(comp_luma, (f32)y + flow_y,
                                               (f32)(x + 1) + flow_x);
            const f32 c_ym = shape_sample_luma(comp_luma, (f32)(y - 1) + flow_y,
                                               (f32)x + flow_x);
            const f32 c_yp = shape_sample_luma(comp_luma, (f32)(y + 1) + flow_y,
                                               (f32)x + flow_x);
            f32 bad_grad = 0.f;
            if (std::isfinite(c_xm) && std::isfinite(c_xp) &&
                std::isfinite(c_ym) && std::isfinite(c_yp)) {
                const f32 gx_c = 0.5f * (c_xp - c_xm);
                const f32 gy_c = 0.5f * (c_yp - c_ym);
                const f32 mag_c = std::sqrt(gx_c * gx_c + gy_c * gy_c);
                // Weak warped gradient: photometric residual must carry the
                // case; do not invent a gradient mismatch from noise.
                if (mag_c / sigma_floor > 0.5f * min_edge) {
                    const f32 denom = std::max(mag_r * mag_c, 1e-20f);
                    const f32 cos_sim = clampf((gx_r * gx_c + gy_r * gy_c) / denom,
                                               -1.f, 1.f);
                    // High cosine -> good; low cosine -> bad.
                    bad_grad = shape_soft01(cos_hi - cos_sim, 0.f, cos_hi - cos_lo);
                }
            }

            const f32 bad_photo = shape_soft01(resid_snr, resid_lo, resid_hi);

            f32 bad_flow = 0.f;
            if (use_flow && !flow_resid.empty()) {
                const size_t pidx = (size_t)patch_idy * flow.nx + patch_idx;
                if (pidx < flow_resid.size())
                    bad_flow = shape_soft01(flow_resid[pidx], flow_lo, flow_hi);
            }

            // Strong penalty only from combined evidence. Photometric alone
            // is never enough (aligned noise); geometry/grad alone is never
            // enough (rotation / weak texture).
            f32 struct_cue = bad_grad;
            if (use_flow)
                struct_cue = std::max(bad_grad, bad_flow);
            const f32 evidence = bad_photo * struct_cue;

            // Fade in with how strongly the analytic mask trusts the pixel,
            // so we mostly police high-R camouflaged failures.
            const f32 trust = shape_soft01(Rn, r_gate, std::min(1.f, r_gate + 0.35f));
            f32 c = 1.f - strength * evidence * trust;
            if (!std::isfinite(c)) c = 1.f;
            C.at(y, x) = clampf(c, 0.f, 1.f);
        }
    });
    return C;
}

// R_final = R_analytic * C_shape (* C_nn when the learned path is on).
//
// The analytic mask above is computed first and is never modified. Shape and
// network contribute only C in [0,1], and the ONLY thing done with either is a
// multiply -- so neither can raise R for any pixel, on any input.
//
// Eq. 9's 5x5 minimum has already been applied inside the analytic mask, and
// is deliberately NOT applied again to C. That step spreads rejection outward
// from a pointwise test with no spatial context; dilating C would double-count.
Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg,
                         Image* s_select_out) {
    Image R = compute_robustness_analytic(comp_raw, ref_stats, flow, tile_size,
                                          cfg, s_select_out);
    if (!cfg.robustness_enabled) return R;
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty() || tile_size <= 0) return R;
    if (R.h <= 0 || R.w <= 0) return R;

    // ---- deterministic shape confidence (optional) ------------------------
    // Guide resolution only: same reason as the learned path -- at raw
    // resolution upscale_warp_stats has already applied the flow, and the
    // unsmoothed luma plane lives at guide resolution.
    if (cfg.robustness_shape_check_enabled &&
        !cfg.robustness_raw_resolution_active()) {
#ifdef __APPLE__
        RefStats* mutable_stats = const_cast<RefStats*>(&ref_stats);
        if (mutable_stats->nn_luma.data.empty() &&
            mutable_stats->means.data.empty() &&
            !metal_fetch_host_ref_stats(*mutable_stats)) {
            // No host luma available; leave the analytic mask alone.
        } else
#endif
        {
            const Image* rl = &ref_stats.nn_luma;
            if (rl->h != R.h || rl->w != R.w ||
                rl->data.size() < (size_t)R.h * R.w) {
                // Shape check was toggled on after init_robustness, or luma
                // was never filled. Means are boxed and cannot reconstruct
                // unsmoothed structure. Fail closed: skip.
                rl = nullptr;
            }
            if (rl) {
                Image comp_luma = guide_luma(compute_guide(comp_raw, cfg));
                if (comp_luma.h == R.h && comp_luma.w == R.w) {
                    Image C = compute_shape_confidence(R, *rl, comp_luma, flow,
                                                       tile_size, cfg);
                    for (size_t i = 0; i < R.data.size() && i < C.data.size(); ++i)
                        R.data[i] *= clampf(C.data[i], 0.f, 1.f);
                    if (cfg.save_shape_rob_mask)
                        debug_dump_bin("rob_shape_c", C.data.data(), C.data.size());
                }
            }
        }
    }

    if (!cfg.use_neural_robustness) return R;
    // Degenerate inputs: the analytic mask has already answered "trust
    // everything" or "trust nothing" without looking at the flow, and the
    // correction has nothing to correct.
    if (!robustness_nn_available()) return R;

#ifdef __APPLE__
    // See metal_fetch_host_ref_stats: on this path the reference stats are
    // GPU-resident by design, so bring them across before building features
    // from them. One readback per burst, not per frame -- the copy stays in
    // ref_stats.
    RefStats* mutable_stats = const_cast<RefStats*>(&ref_stats);
    if (mutable_stats->means.data.empty() && !metal_fetch_host_ref_stats(*mutable_stats))
        return R;   // no host stats, no features; the analytic mask stands
#endif
    // GUIDE RESOLUTION ONLY, deliberately. At raw resolution
    // upscale_warp_stats has already applied the flow to the comparison
    // statistics, so the spatial channels would need an unwarped luma plane in
    // a coordinate system nothing else in the feature vector uses -- and the
    // model was never trained with them zeroed. Running it there would be a
    // different model from the one that was measured, so the raw-resolution
    // toggle simply gets the analytic mask.
    if (cfg.robustness_raw_resolution_active()) return R;
    // init_robustness fills this from the reference raw. Empty means the
    // correction was switched on mid-burst, after the reference was processed,
    // in which case seven of the twenty-seven channels would be zero. Fall
    // back rather than run it blind.
    if (ref_stats.nn_luma.h != R.h || ref_stats.nn_luma.w != R.w ||
        ref_stats.nn_luma.data.size() < (size_t)R.h * R.w)
        return R;

    Image cm_nn, cl_nn;
    {
        Image guide_nn = compute_guide(comp_raw, cfg);
        cl_nn = guide_luma(guide_nn);
        Image cv_nn;   // variance is not a feature; freed with this scope
        local_stats_3x3(guide_nn, cm_nn, cv_nn);
    }
    // Feature channel 17 depends only on the reference, so it is built here
    // (once per burst -- the guard inside makes repeat calls free) rather than
    // inside the per-pixel loop of every comparison frame.
    ensure_robustness_nn_ref_hf(*const_cast<RefStats*>(&ref_stats), cfg);
    // Channels 18-19, one measurement per tile per comparison frame.
    const std::vector<f32> mq = measure_match_quality(ref_stats.nn_luma, cl_nn,
                                                      flow, tile_size, cfg);

    const int nh = cm_nn.h, nw = cm_nn.w;
    // The correction is only meaningful on the grid the analytic mask lives
    // on. A mismatch means the raw-resolution toggle and the mask disagree
    // (compute_robustness_raw_res silently returns guide resolution when the
    // hires reference stats are missing), and resampling C to fit would apply
    // it half a plane out of position.
    if (nh != R.h || nw != R.w) return R;

    const int strip_h = kRobustnessNnStripRows + 2 * kRobustnessNnHalo;
    Image nn_c(nh, nw, 1);
    // Every window is strip_h tall and fully inside the image, so Core ML sees
    // one input shape for the whole burst (no reshape per strip) and the
    // result matches whole-plane inference exactly. An image shorter than one
    // window would make that impossible; there is nothing to save there
    // either, so leave the analytic mask alone.
    bool ok = (nh >= strip_h && nw > 0);
    for (int y0 = 0; ok && y0 < nh; y0 += kRobustnessNnStripRows) {
        const int top = std::min(std::max(y0 - kRobustnessNnHalo, 0), nh - strip_h);
        Image feat = build_robustness_nn_features(ref_stats, cm_nn, flow, tile_size,
                                                  cfg, top, /*raw_res=*/false, /*rows=*/0,
                                                  &cl_nn, &mq);
        Image strip;
        if (feat.h != strip_h || !robustness_nn_infer(feat, strip) ||
            strip.h != strip_h || strip.w != nw) {
            ok = false;
            break;
        }
        const int rows = std::min(kRobustnessNnStripRows, nh - y0);
        for (int r = 0; r < rows; ++r)
            std::memcpy(&nn_c.at(y0 + r, 0), &strip.at(y0 - top + r, 0),
                        (size_t)nw * sizeof(f32));
    }
    if (!ok) return R;   // fails closed: the analytic mask is the fallback

    // The multiply. Clamped defensively so a model that somehow emits a value
    // outside [0,1] -- a bad export, a numerical edge -- still cannot raise R.
    for (size_t i = 0; i < R.data.size() && i < nn_c.data.size(); ++i)
        R.data[i] *= clampf(nn_c.data[i], 0.f, 1.f);
    // Saved as C, not as R_final: the question this dump exists to answer is
    // "what did the network do", and R_final mixes that with what the analytic
    // mask already did.
    if (cfg.save_nn_rob_mask)
        debug_dump_bin("rob_nn_c", nn_c.data.data(), nn_c.data.size());
    return R;
}

} // namespace hhsr
