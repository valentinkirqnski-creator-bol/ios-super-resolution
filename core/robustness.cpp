#include "stages.h"
#include "parallel.h"
#include "prof.h"
#include <mutex>
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

// Pure builder, no caching -- shared by the single-slot and per-channel
// cache wrappers below so the ~1e5-patch Monte Carlo logic exists once.
// Disk cache for the Monte Carlo curves, keyed on the EXACT float bits of
// (alpha, beta). unitary_MC is fully deterministic -- the RNG seed is a fixed
// function of the brightness bin -- so a cached table is bit-identical to a
// rebuild; the cache changes when the answer arrives, never what it is.
//
// Why it exists: the build is ~1.8M gaussian draws per non-linear bin and was
// measured at 2.0-2.8 s PER CHANNEL cold, i.e. up to ~7.5 s hiding inside the
// first frame's comp:robustness. In-process statics already amortise repeat
// bursts; this amortises across app launches. The directory is provided by
// the app (Caches/); when unset -- host tools, tests -- behaviour is exactly
// as before.
static std::string g_noise_cache_dir;

static std::string noise_cache_file(f32 alpha, f32 beta) {
    uint32_t ab, bb;
    std::memcpy(&ab, &alpha, 4);
    std::memcpy(&bb, &beta, 4);
    char name[96];
    std::snprintf(name, sizeof(name), "/mccurve_v1_%08x_%08x_%d.bin", ab, bb,
                  k_n_patches);
    return g_noise_cache_dir + name;
}

static bool noise_cache_load(f32 alpha, f32 beta, NoiseCurves& nc) {
    if (g_noise_cache_dir.empty()) return false;
    FILE* f = std::fopen(noise_cache_file(alpha, beta).c_str(), "rb");
    if (!f) return false;
    const size_t n = (size_t)k_n_brightness + 1;
    nc.std_curve.resize(n);
    nc.diff_curve.resize(n);
    const bool ok = std::fread(nc.std_curve.data(), sizeof(f32), n, f) == n &&
                    std::fread(nc.diff_curve.data(), sizeof(f32), n, f) == n &&
                    std::fgetc(f) == EOF;   // reject truncated/oversized files
    std::fclose(f);
    if (!ok) { nc.std_curve.clear(); nc.diff_curve.clear(); }
    return ok;
}

static void noise_cache_store(f32 alpha, f32 beta, const NoiseCurves& nc) {
    if (g_noise_cache_dir.empty()) return;
    // Write-to-temp + rename so a mid-write kill can never leave a short file
    // that a later launch would half-trust.
    const std::string path = noise_cache_file(alpha, beta);
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return;
    const size_t n = (size_t)k_n_brightness + 1;
    const bool ok = std::fwrite(nc.std_curve.data(), sizeof(f32), n, f) == n &&
                    std::fwrite(nc.diff_curve.data(), sizeof(f32), n, f) == n;
    std::fclose(f);
    if (ok) std::rename(tmp.c_str(), path.c_str());
    else std::remove(tmp.c_str());
}

static NoiseCurves build_noise_curves(f32 alpha, f32 beta) {
    NoiseCurves nc;
    if (try_load_python_noise_curves(alpha, beta, nc))
        return nc;
    if (noise_cache_load(alpha, beta, nc)) {
        std::printf("[noise] MC curve cache HIT (a=%g b=%g)\n", alpha, beta);
        return nc;
    }
    std::printf("[noise] MC curve cache MISS (a=%g b=%g)%s -- building, ~2-3 s\n",
                alpha, beta,
                g_noise_cache_dir.empty() ? " [cache dir NOT SET]" : "");
    // Bucketed so a profile shows the Monte-Carlo build as its own line
    // instead of hiding inside whichever caller (SNR tuning or the first
    // robustness call) happened to trigger it.
    const double t_build = prof_now_ms();

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

    noise_cache_store(alpha, beta, nc);
    prof_add_cpu("rob:noise-mc-build", prof_now_ms() - t_build);
    return nc;
}

// Guards the two lazy curve caches below. Needed since the burst prewarms the
// curves on a worker thread (see robustness_prewarm_noise_curves) while the
// main thread may request the same curves for frame analysis. The lock covers
// the lookup AND the build, so a concurrent request for a key being built
// blocks until it is cached rather than building it twice. Returned
// references stay valid after unlock because within one burst every caller
// asks for the same keys, so the slots are never rebuilt underneath them --
// the same invariant the unlocked single-thread version relied on.
static std::mutex g_noise_curves_mu;

static const NoiseCurves& make_noise_curves(f32 alpha, f32 beta) {
    // Cache like Python (curves built once per alpha/beta, reused every frame).
    std::lock_guard<std::mutex> lock(g_noise_curves_mu);
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
    return make_noise_curves(cfg.noise_alpha(), cfg.noise_beta());
}

// Per-guide-channel curve, 3 independently cached slots (one per CFA colour)
// rather than routing through the single-slot cache above: R/G/B typically
// have different alpha'/beta' after white balance, so 3 calls through a
// 1-slot cache would evict and rebuild the Monte Carlo curve on every call --
// 3x the cost every frame instead of once per burst.
static const NoiseCurves& make_noise_curves_channel(f32 alpha, f32 beta, int ch) {
    std::lock_guard<std::mutex> lock(g_noise_curves_mu);
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
    return make_noise_curves(cfg.noise_alpha_robustness(), cfg.noise_beta_robustness());
}
static const NoiseCurves& mask_noise_curves_channel(const Config& cfg, int ch) {
    if (cfg.debug_noise_model_disabled)
        return make_noise_curves_channel(0.f, 0.f, ch);
    return make_noise_curves_channel(cfg, ch);
}

} // namespace

// Defined OUTSIDE the anonymous namespace above, deliberately. Inside it the
// symbol gets internal linkage -- hhsr::(anonymous)::robustness_set_noise_...
// -- and the app links against hhsr::robustness_set_noise_cache_dir, which is
// exactly the undefined-symbol failure CI produced. Host -fsyntax-only cannot
// catch linkage, so this class of slip only surfaces at the device link.
// Anonymous-namespace members stay visible unqualified from the enclosing
// namespace in the same TU, so the assignment still reaches g_noise_cache_dir.
void robustness_set_noise_cache_dir(const std::string& dir) { g_noise_cache_dir = dir; }

// Builds (or cache-loads) every noise curve the burst's robustness calls will
// request, through the exact same accessors they use -- same keys, same
// debug_noise_model_disabled gating, same values. Meant to run on a worker
// thread right after SNR tuning has fixed the noise model: on a curve-cache
// MISS (any new ISO changes alpha/beta and with them the cache key) the
// Monte-Carlo builds otherwise run synchronously inside the FIRST comparison
// frame's robustness call, which is the "analyzing frame 2" freeze. The
// g_noise_curves_mu lock inside the accessors makes the concurrent warm safe:
// if the frame catches up it waits for the in-flight build, never duplicates
// it. Also outside the anonymous namespace: pipeline callers link against
// hhsr::robustness_prewarm_noise_curves.
void robustness_prewarm_noise_curves(const Config& cfg) {
    (void)make_noise_curves(cfg);
    (void)mask_noise_curves(cfg);
    for (int ch = 0; ch < 3; ++ch)
        (void)mask_noise_curves_channel(cfg, ch);
}


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

// Not in the anonymous namespace below: pipeline callers need the exact same
// guide image the classical robustness path scores against.
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
    // count. undo[c] folds in Config::guide_wb_undo -- python-z divides each
    // site by wb[c] before averaging (robustness.py compute_guide_image);
    // since that factor is constant per channel it commutes with the /n
    // average, so multiplying the finished sum once is bit-identical to
    // dividing each of the n samples first.
    f32 undo[3];
    for (int c = 0; c < 3; ++c) {
        const int n = cfg.cfa.count((uint8_t)c);
        undo[c] = (n > 0) ? cfg.guide_wb_undo(c) / (f32)n : 0.f;
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
            for (int c = 0; c < 3; ++c) guide.at(y, x, c) = sum[c] * undo[c];
        }
    }
    return guide;
}

namespace {

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
                                int num_threads, bool bilinear_flow) {
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
                // y, x are RAW here, so sample_bilinear takes them directly.
                if (bilinear_flow) {
                    flow->sample_bilinear((f32)y, (f32)x, tile_size, flow_x, flow_y);
                } else {
                    // Python: patch_idy = int(y // tile_size)  (no clamp)
                    int patch_idy = y / tile_size;
                    int patch_idx = x / tile_size;
                    if (patch_idy >= 0 && patch_idy < flow->ny &&
                        patch_idx >= 0 && patch_idx < flow->nx) {
                        flow_x = flow->dx(patch_idy, patch_idx);
                        flow_y = flow->dy(patch_idy, patch_idx);
                    }
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
            // python-z cuda_apply_noise_model, read literally: each channel
            // gets its OWN max(measured, noise-floor) and its OWN Wiener
            // shrink, using that channel's own d_t, and only the RESULTS are
            // summed across channels. Not max-of-sums / one-shrink-of-sums --
            // those are a different, later computation this port used to run
            // instead (see git history), on the (debatable, but not
            // reference-matching) argument that Eq. 6's d/sigma are bare
            // per-pixel scalars rather than per-channel. Matching python-z
            // here means matching ITS order, not re-deriving one from the
            // paper text.
            f32 sigma_sq_ = 0.f, d_sq_ = 0.f;
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
                f32 sigma_p_sq = ref_vars.at(y, x, ch);
                f32 d_p_sq = d_p.at(y, x, ch) * d_p.at(y, x, ch);
                sigma_sq_ += std::max(sigma_p_sq, sigma_t * sigma_t);
                f32 shrink = d_p_sq / (d_p_sq + d_t * d_t);
                d_sq_ += d_p_sq * shrink * shrink;
            }
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
// memory ceiling. Arithmetic is identical to apply_noise_model above: each
// channel's own max(measured, floor) and own Wiener shrink, summed after. An
// out-of-bounds Dodgson sample arrives as +inf in comp_means and must stay
// +inf in the difference so R lands at 0 downstream.
static void apply_noise_model_fused(const Image& ref_means, const Image& comp_means,
                                    const Image& ref_vars,
                                    const NoiseCurves* const nc_ch[3],
                                    Image& d_sq, Image& sigma_sq, int num_threads) {
    const int n_ch = ref_means.c;
    d_sq = Image(ref_means.h, ref_means.w, 1);
    sigma_sq = Image(ref_means.h, ref_means.w, 1);
    parallel_rows(ref_means.h, num_threads, [&](int y) {
        for (int x = 0; x < ref_means.w; ++x) {
            f32 sigma_sq_ = 0.f, d_sq_ = 0.f;
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
                const f32 comp = comp_means.at(y, x, ch);
                f32 d_p_ = std::isfinite(comp)
                    ? std::fabs(brightness - comp)
                    : std::numeric_limits<f32>::infinity();
                f32 sigma_p_sq = ref_vars.at(y, x, ch);
                f32 d_p_sq = d_p_ * d_p_;
                sigma_sq_ += std::max(sigma_p_sq, sigma_t * sigma_t);
                f32 shrink = d_p_sq / (d_p_sq + d_t * d_t);
                d_sq_ += d_p_sq * shrink * shrink;
            }
            d_sq.at(y, x) = d_sq_;
            sigma_sq.at(y, x) = sigma_sq_;
        }
    });
}

static std::vector<f32> compute_s(const FlowField& flow, f32 Mt, f32 s1, f32 s2,
                                  std::vector<uint32_t>* irregular_out = nullptr) {
    const f32 inf = std::numeric_limits<f32>::infinity();
    std::vector<f32> S((size_t)flow.ny * flow.nx, s2);
    if (irregular_out) irregular_out->assign((size_t)flow.ny * flow.nx, 0u);
    // python-z parity: cuda_compute_s always recomputes the Eq. 7/8 span
    // live, on the exact flow array compute_robustness receives -- which is
    // `flow` right here, so the live loop below already operates on the
    // correct array with no extra plumbing needed. This used to shortcut to
    // FlowField::motion_irregular, a value measured earlier in alignment on
    // an unduplicated, unscaled grid (see that field's own comment) --
    // deliberate at the time, but not what python-z does, so it is gone.
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
    if (cfg.robustness_raw_resolution_active()) {
        // is_ref=true: no flow warp, just the Dodgson upscale (Algorithm 6
        // never warps the reference's own stats -- only Gn's). Once per
        // burst here, not once per comparison frame.
        st.means_hires = upscale_warp_stats(st.means, /*is_ref=*/true, nullptr,
                                            0, cfg.num_threads,
                                            cfg.flow_bilinear_sampling);
        st.stds_hires = upscale_warp_stats(st.stds, /*is_ref=*/true, nullptr,
                                           0, cfg.num_threads,
                                           cfg.flow_bilinear_sampling);
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
// init_robustness, not once per comparison frame.
static Image compute_robustness_raw_res(const Image& comp_raw, const RefStats& ref_stats,
                                        const FlowField& flow, int tile_size,
                                        const Config& cfg) {
    if (ref_stats.means_hires.h <= 0 || ref_stats.means_hires.w <= 0 ||
        ref_stats.stds_hires.h <= 0 || ref_stats.stds_hires.w <= 0) {
        // init_robustness didn't populate the hires stats (e.g. robustness
        // was off when the burst started and got toggled mid-burst) --
        // no raw-res path to run.
        return Image();
    }

    // 832f7b8-style per-channel curves, restored on request: each guide
    // channel scored against its own WB-scaled (alpha', beta') curve
    // (Config::noise_alpha_ch_robustness -- alpha_dng[c] * wb_gain *
    // guide_weight, same accessor 832f7b8 used), NOT python-z's single
    // shared unscaled curve (3585bef item 2). Deliberate deviation from
    // python-z; the guide-resolution path below always kept this form.
    const NoiseCurves* nc_ch[3] = {nullptr, nullptr, nullptr};
    for (int ch = 0; ch < 3; ++ch)
        nc_ch[ch] = &mask_noise_curves_channel(cfg, ch);

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
        // Checker follows doer: warp the comparison stats with the SAME
        // flow field the merge will apply -- the exact gate accumulate_comp
        // uses (flow_bilinear_sampling, or the dense-lattice trusted
        // pairing when a fine lattice exists). R certifying one flow while
        // the merge fetches samples with another means merged samples were
        // never verified at all between tile centres. When the toggle is
        // off this is nearest per-tile, which is also python-z parity
        // (cuda_uspcale_dogson's bare patch_idy = int(y // tile_size) --
        // python-z has no interpolation option, but its merge consumes the
        // same nearest flow, so checker == doer holds there too).
        const bool merge_samples_flow =
            cfg.flow_bilinear_sampling ||
            ((cfg.flow_dense_lattice_trusted() || cfg.overlap_merge_active()) &&
             flow.has_fine());
        comp_means = upscale_warp_stats(comp_means_guide, /*is_ref=*/false, &flow,
                                        tile_size, cfg.num_threads,
                                        /*bilinear_flow=*/merge_samples_flow);
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

    std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2);

    Image R(h, w, 1);
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
                continue;
            }
            f32 s = S[pidx];
            f32 sig = sigma_sq.at(y, x);
            const bool match_ambiguous =
                cfg.flow_reject_ambiguous_enabled &&
                pidx < flow.match_ambiguous.size() &&
                flow.match_ambiguous[pidx] != 0u;
            if (match_ambiguous) s = std::min(s, cfg.r_s1);
            f32 r_val = clampf(s * std::exp(-d_sq.at(y, x) / sig) - cfg.r_t, 0.f, 1.f);
            // An out-of-bounds Dodgson sample writes +inf into comp_means by
            // design ("infinite will imply R = 0"), which makes d_sq +inf and
            // the Wiener shrink inf/inf = NaN, so r_val is NaN. The Python
            // clamp (CUDA fmaxf/fminf) returns the non-NaN operand and yields
            // the intended 0; clampf's comparisons are both false for NaN and
            // would return NaN, poisoning every merge accumulator that touches
            // this pixel.
            if (!std::isfinite(r_val)) r_val = 0.f;
            R.at(y, x) = r_val;
        }
    }
    // 832f7b8-style Eq. 9, restored on request: the guide-grid erosion
    // (2x2 min-reduce -> 5x5 min on the coarser grid -> nearest upsample,
    // an effective ~10-12 raw-px minimum footprint) instead of python-z's
    // literal 5x5 on raw R (3585bef item 5's filter change). The wider
    // erosion spreads every rejection over roughly double the area, which
    // is most of why the 832f7b8-era mask read darker. Deliberate deviation
    // from python-z; local_min_5x5(R) is the python-z-exact form if parity
    // is wanted back.
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
    if (cfg.robustness_raw_resolution_active()) {
        Image raw_res = compute_robustness_raw_res(comp_raw, ref_stats, flow, tile_size, cfg);
        if (raw_res.h > 0 && raw_res.w > 0) return raw_res;
    }

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
    // Checker follows doer here too: with the overlapped-tile merge active
    // there is no single fetch per pixel, so the mask grades the lattice
    // expectation (sample_bilinear auto-selects the fine grid) -- same
    // documented approximation as the raw-res path.
    const bool sample_flow_g =
        cfg.flow_bilinear_sampling ||
        (cfg.overlap_merge_active() && flow.has_fine());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            f32 flow_x = 0.f, flow_y = 0.f;
            int patch_idy = 0, patch_idx = 0;
            if (d_p.c == 1) {
                patch_idy = y / tile_size;
                patch_idx = x / tile_size;
                if (sample_flow_g)
                    flow.sample_bilinear((f32)y, (f32)x, tile_size, flow_x, flow_y);
                else {
                    flow_x = flow.dx(patch_idy, patch_idx);
                    flow_y = flow.dy(patch_idy, patch_idx);
                }
            } else {
                patch_idy = (int)((2.f * (f32)y + 0.5f) / (f32)tile_size);
                patch_idx = (int)((2.f * (f32)x + 0.5f) / (f32)tile_size);
                if (sample_flow_g) {
                    f32 rdx, rdy;
                    flow.sample_bilinear(2.f * (f32)y + 0.5f, 2.f * (f32)x + 0.5f,
                                         tile_size, rdx, rdy);
                    flow_x = 0.5f * rdx; flow_y = 0.5f * rdy;
                } else {
                    flow_x = 0.5f * flow.dx(patch_idy, patch_idx);
                    flow_y = 0.5f * flow.dy(patch_idy, patch_idx);
                }
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
    std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2);

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
            const size_t pidx = (size_t)patch_idy * flow.nx + patch_idx;
            f32 s = S[pidx];
            f32 sig = sigma_sq.at(y, x);
            const bool match_ambiguous =
                cfg.flow_reject_ambiguous_enabled &&
                pidx < flow.match_ambiguous.size() &&
                flow.match_ambiguous[pidx] != 0u;
            if (match_ambiguous) s = std::min(s, cfg.r_s1);
            f32 r_val = clampf(s * std::exp(-d_sq.at(y, x) / sig) - cfg.r_t, 0.f, 1.f);
            R.at(y, x) = r_val;
        }
    }
    return local_min_5x5(R);
#endif
}

} // namespace hhsr
