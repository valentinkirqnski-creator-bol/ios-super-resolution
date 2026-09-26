#include "stages.h"
#include "robustness_nn.h"
#include "robustness_refine_shared.h"
#include "parallel.h"
#include "pixel4a_noise_curves.h"
#include "prof.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cmath>
#include <memory>
#include <mutex>
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

static void unitary_MC(f32 alpha, f32 beta, f32 b, f32& diff_mean, f32& std_mean,
                       bool sqrt_domain = false) {
    // Same estimator as fast_monte_carlo.unitary_MC (population std, |Δμ|),
    // same draw order (all patch1 then all patch2). RNG seed is C++-only.
    // sqrt_domain: 1.4's cuda_compute_guide_image applies sqrt to the clipped
    // raw before the 3x3 patch statistics, so the curve here must too --
    // stored at the LATENT brightness bin b; the mask indexes it by the guide
    // mean SQUARED (guide = sqrt(latent)). See Config::robustness_guide_sqrt.
    const double bd = (double)b;
    const double scale = std::sqrt(std::max(0.0, (double)alpha * bd + (double)beta));
    const uint32_t seed = 1337u + (uint32_t)std::lround(bd * (double)k_n_brightness);
    NumpyRandomState rng(seed);

    const int n = k_n_patches;
    auto fill_patch_stats = [&](std::vector<double>& means, std::vector<double>& stds) {
        // resize() on an already-large vector is a no-op, which is the point:
        // the four buffers below are thread_local scratch (see the call site),
        // so a 70-bin curve stops allocating and freeing 4 x 800KB per bin.
        means.resize((size_t)n);
        stds.resize((size_t)n);
        for (int i = 0; i < n; ++i) {
            double p[9];
            double m = 0.0;
            for (int j = 0; j < 9; ++j) {
                double v = bd + scale * rng.rk_gauss();
                v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
                if (sqrt_domain) v = std::sqrt(v);
                p[j] = v;
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
    //
    // thread_local, not local: build_noise_curves_batch runs one bin per
    // worker thread, and a 70-bin curve allocated and freed 4 x 800KB per bin
    // on the same allocate-and-touch path that costs ~1.24 GB/s on device.
    // Reused across bins on the same thread; the contents are fully
    // overwritten by fill_patch_stats before any read, so nothing carries over.
    static thread_local std::vector<double> m1, s1, m2, s2;
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
// sqrt_domain: build the curve in 1.4's sqrt-guide domain (a separate LUT
// from the linear one SNR/kernel tuning use). The interpolation shortcut is
// domain-agnostic (it lerps sigma^2/d^2 between the MC'd non-linear ends),
// so sqrt uses it too; only the Python-dump load is linear-only.
// One curve to fill. Several are prepared together so their Monte-Carlo bins
// can share a single parallel dispatch -- see build_noise_curves_batch.
struct CurveBuildSpec {
    NoiseCurves* out = nullptr;
    f32 alpha = 0.f;
    f32 beta = 0.f;
    bool sqrt_domain = false;
    bool needs_mc = false;   // false when zeros or a Python dump already filled it
    bool full_mc = false;
    int imin = 0;
    int imax = 0;
};

// Everything that decides WHICH bins need the Monte Carlo, without running it.
// Returns with spec.out already filled when no MC is needed at all.
static void prepare_noise_curve_spec(CurveBuildSpec& spec) {
    NoiseCurves& nc = *spec.out;
    nc = NoiseCurves();

    // No noise model (alpha = beta = 0, e.g. Disable Noise Model): every
    // sample equals the brightness, so patch std and diff are exactly 0 in
    // both domains. Return zeros WITHOUT running the ~1e5-patch MC over 1001
    // bins -- that build produced only zeros yet stalled the first
    // comparison frame ("Frame 2: analyze" hang), most visibly on the sqrt
    // path, which used to force the full-bin MC.
    if (!(spec.alpha > 0.f) && !(spec.beta > 0.f)) {
        nc.std_curve.assign((size_t)k_n_brightness + 1, 0.f);
        nc.diff_curve.assign((size_t)k_n_brightness + 1, 0.f);
        spec.needs_mc = false;
        return;
    }

    if (!spec.sqrt_domain && try_load_python_noise_curves(spec.alpha, spec.beta, nc)) {
        spec.needs_mc = false;
        return;
    }

    nc.std_curve.resize((size_t)k_n_brightness + 1);
    nc.diff_curve.resize((size_t)k_n_brightness + 1);

    f32 xmin, xmax;
    get_non_linearity_bound(spec.alpha, spec.beta, k_tol, xmin, xmax);

    spec.imin = (int)std::ceil(xmin * (f32)k_n_brightness) + 1;
    spec.imax = (int)std::floor(xmax * (f32)k_n_brightness) - 1;
    // Python run_fast_MC: only this gate triggers full regular MC
    spec.full_mc = (spec.imin > k_n_brightness);
    spec.needs_mc = true;
}

// Build any number of curves in ONE parallel dispatch over the bins that
// actually run the Monte Carlo.
//
// The previous form dispatched 1001 iterations per curve, of which only the
// non-linear ends -- about 70 bins at moderate ISO, ~210 at high ISO -- did
// any work, and those sit at i <= imin and i >= imax, i.e. at the two ENDS of
// the index range. dispatch_apply hands out contiguous ranges, so one or two
// workers received every heavy bin while the rest returned immediately. With
// three guide channels built one after another that happened three times over.
//
// Here the work is enumerated first and dispatched as a flat job list, so
// every worker gets an equal share and all three channels overlap. Purely a
// scheduling change: each job calls the same unitary_MC with the same
// (alpha, beta, bin), whose RNG is seeded from the bin alone
// (1337 + lround(b * 1000)), and writes only its own two slots. Bit-identical
// to the serial result, in any execution order.
static void build_noise_curves_batch(CurveBuildSpec* specs, int n) {
    if (!specs || n <= 0) return;
    const double t0 = prof_now_ms();

    struct Job { int spec; int bin; };
    std::vector<Job> jobs;
    jobs.reserve((size_t)n * 256u);

    for (int s = 0; s < n; ++s) {
        prepare_noise_curve_spec(specs[s]);
        if (!specs[s].needs_mc) continue;
        const CurveBuildSpec& sp = specs[s];
        for (int i = 0; i <= k_n_brightness; ++i) {
            // Same predicate as the two branches it replaces.
            if (sp.full_mc || i <= sp.imin || i >= sp.imax)
                jobs.push_back(Job{s, i});
        }
    }

    if (!jobs.empty()) {
        parallel_rows((int)jobs.size(), 0, [&](int j) {
            const Job& job = jobs[(size_t)j];
            const CurveBuildSpec& sp = specs[job.spec];
            const f32 b = job.bin / (f32)k_n_brightness;
            unitary_MC(sp.alpha, sp.beta, b,
                       sp.out->diff_curve[(size_t)job.bin],
                       sp.out->std_curve[(size_t)job.bin],
                       sp.sqrt_domain);
        });
    }

    for (int s = 0; s < n; ++s) {
        // Overwrite [imin, imax] inclusive (matches run_fast_MC)
        if (specs[s].needs_mc && !specs[s].full_mc)
            interp_MC_range(*specs[s].out, specs[s].imin, specs[s].imax);
    }

    if (!jobs.empty()) {
        prof_add_cpu("robustness:noise-curves(mc)", prof_now_ms() - t0);
        prof_add_cpu("robustness:noise-curves#bins", (double)jobs.size());
    }
}

// ---------------------------------------------------------------------------
// Curve cache.
//
// One append-only table keyed by (alpha, beta, sqrt_domain), shared by the
// single-slot and per-channel accessors. Three properties matter:
//
//   Never evicted. The accessors hand back a reference that compute_robustness
//   holds across a whole frame, so a slot that could be rebuilt underneath it
//   would dangle. Entries are ~8KB (two 1001-float curves), so a session's
//   worth costs well under a megabyte -- and a burst at an ISO already seen
//   finds its curves built, which is most of what made repeat shots fast and
//   first-of-a-new-ISO shots slow.
//
//   Deduplicated on the key, not the channel. The channel index only selects
//   which alpha/beta to ask for; two channels with equal parameters (the
//   debug_noise_model_disabled case, where both are 0) now share one build
//   instead of producing two identical ones.
//
//   Locked across the build, not just the lookup. prewarm_noise_curves runs on
//   a background thread while the first comparison frame may ask for the same
//   curve on the pipeline thread; without the lock both would run the same
//   Monte Carlo. Holding it means the pipeline thread waits for the prewarm
//   rather than duplicating it, which is the intended behaviour.
// ---------------------------------------------------------------------------
struct CurveCacheEntry {
    f32 alpha = 0.f;
    f32 beta = 0.f;
    bool sqrt_domain = false;
    NoiseCurves nc;
};

static std::mutex g_curve_mu;
// unique_ptr so appending cannot move the entries a caller already holds.
static std::vector<std::unique_ptr<CurveCacheEntry>> g_curve_cache;

static const NoiseCurves* curve_lookup_locked(f32 alpha, f32 beta, bool sqrt_domain) {
    for (const auto& e : g_curve_cache) {
        // Exact float equality, matching the previous per-slot caches: these
        // keys are derived deterministically from the same Config fields.
        if (e->alpha == alpha && e->beta == beta && e->sqrt_domain == sqrt_domain)
            return &e->nc;
    }
    return nullptr;
}

// Build every requested key that is not cached yet, in ONE batch so their
// Monte-Carlo bins share a single parallel dispatch. Caller holds g_curve_mu.
static void curve_build_missing_locked(const f32* alphas, const f32* betas,
                                       const bool* sqrts, int n) {
    std::vector<CurveCacheEntry*> fresh;
    std::vector<CurveBuildSpec> specs;
    fresh.reserve((size_t)n);
    specs.reserve((size_t)n);

    for (int i = 0; i < n; ++i) {
        if (curve_lookup_locked(alphas[i], betas[i], sqrts[i])) continue;
        // Another entry queued in this same call with the same key.
        bool dup = false;
        for (CurveCacheEntry* f : fresh) {
            if (f->alpha == alphas[i] && f->beta == betas[i] &&
                f->sqrt_domain == sqrts[i]) { dup = true; break; }
        }
        if (dup) continue;
        g_curve_cache.push_back(std::unique_ptr<CurveCacheEntry>(new CurveCacheEntry()));
        CurveCacheEntry* e = g_curve_cache.back().get();
        e->alpha = alphas[i];
        e->beta = betas[i];
        e->sqrt_domain = sqrts[i];
        fresh.push_back(e);
    }
    if (fresh.empty()) return;

    for (CurveCacheEntry* e : fresh) {
        CurveBuildSpec spec;
        spec.out = &e->nc;
        spec.alpha = e->alpha;
        spec.beta = e->beta;
        spec.sqrt_domain = e->sqrt_domain;
        specs.push_back(spec);
    }
    build_noise_curves_batch(specs.data(), (int)specs.size());
}

static const NoiseCurves& noise_curves_cached(f32 alpha, f32 beta, bool sqrt_domain) {
    std::lock_guard<std::mutex> lk(g_curve_mu);
    if (const NoiseCurves* hit = curve_lookup_locked(alpha, beta, sqrt_domain))
        return *hit;
    curve_build_missing_locked(&alpha, &beta, &sqrt_domain, 1);
    const NoiseCurves* built = curve_lookup_locked(alpha, beta, sqrt_domain);
    // curve_build_missing_locked always appends on a miss; the fallback keeps
    // the reference valid rather than dereferencing null if that ever changes.
    static const NoiseCurves kEmpty;
    return built ? *built : kEmpty;
}

static const NoiseCurves& make_noise_curves(f32 alpha, f32 beta) {
    return noise_curves_cached(alpha, beta, /*sqrt_domain=*/false);
}

static const NoiseCurves& make_noise_curves(const Config& cfg) {
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
    // ch is no longer a cache dimension: the shared table keys on
    // (alpha, beta, sqrt_domain), which is all the channel index ever selected.
    (void)ch;
    return noise_curves_cached(alpha, beta, /*sqrt_domain=*/false);
}

static const NoiseCurves& make_noise_curves_channel(const Config& cfg, int ch) {
    // WB-scaled per-channel alpha/beta, matching the WB'd guide. Only mask
    // paths call this wrapper.
    return make_noise_curves_channel(cfg.noise_alpha_ch_robustness(ch),
                                     cfg.noise_beta_ch_robustness(ch), ch);
}

// 1.4 sqrt-guide parity: SEPARATE sqrt-domain caches (single-slot and
// per-channel), so the robustness curves live in the sqrt domain while the
// linear curves SNR/kernel tuning share stay untouched.
static const NoiseCurves& make_noise_curves_sqrt(f32 alpha, f32 beta) {
    return noise_curves_cached(alpha, beta, /*sqrt_domain=*/true);
}
static const NoiseCurves& make_noise_curves_channel_sqrt(f32 alpha, f32 beta, int ch) {
    (void)ch;   // see make_noise_curves_channel
    return noise_curves_cached(alpha, beta, /*sqrt_domain=*/true);
}

// Mask-only variants: honour Config::debug_noise_model_disabled by building
// the curves from alpha = beta = 0 (so sigma_t = d_t = 0 in every bin),
// while make_noise_curves(cfg) itself stays ungated -- it is shared with SNR
// auto-tuning via noise_std_at_brightness, and gating it there changed the
// alignment tile size and merge constants along with the mask (measured:
// tile 16 -> 32), which is exactly what a diagnostic probe must not do.
// robustness_guide_sqrt routes to the sqrt-domain caches (1.4 parity).
static const NoiseCurves& mask_noise_curves(const Config& cfg) {
    const bool sq = cfg.robustness_guide_sqrt;
    if (cfg.debug_noise_model_disabled)
        return sq ? make_noise_curves_sqrt(0.f, 0.f) : make_noise_curves(0.f, 0.f);
    const f32 a = cfg.noise_alpha_robustness(), b = cfg.noise_beta_robustness();
    return sq ? make_noise_curves_sqrt(a, b) : make_noise_curves(a, b);
}
static const NoiseCurves& mask_noise_curves_channel(const Config& cfg, int ch) {
    const bool sq = cfg.robustness_guide_sqrt;
    if (cfg.debug_noise_model_disabled)
        return sq ? make_noise_curves_channel_sqrt(0.f, 0.f, ch)
                  : make_noise_curves_channel(0.f, 0.f, ch);
    return sq ? make_noise_curves_channel_sqrt(cfg.noise_alpha_ch_robustness(ch),
                                               cfg.noise_beta_ch_robustness(ch), ch)
              : make_noise_curves_channel(cfg, ch);
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

// Build every noise curve this burst will ask for, in one batch, ahead of the
// stage that needs it.
//
// This is the largest single cost in the burst and it used to land on the first
// comparison frame: ~1.8 million seeded Gaussian draws per brightness bin,
// over the ~70 non-linear bins (more at high ISO), once per guide channel,
// plus the linear curve noise_std_at_brightness needs for the status line.
// Neither was inside a profiler bucket.
//
// It is also why repeat shots were fast and the first shot at a new ISO was
// slow: the cache key is the WB-scaled per-channel alpha/beta, so any change
// in ISO or white balance was a miss.
//
// Nothing here depends on pixel data -- only on the reference frame's
// NoiseProfile and white balance -- so the caller runs it on a background
// thread as soon as the reference metadata is known, and it overlaps the
// reference grey, pyramid, statistics and kernels. The cache lock makes the
// first comparison frame wait for this rather than duplicate it.
//
// Idempotent, and bit-identical to building the curves lazily: same keys, same
// per-bin seeds, same arithmetic.
void prewarm_noise_curves(const Config& cfg) {
    // Mirror exactly which keys the burst requests. Order does not matter --
    // the batch dedupes -- but the set has to match, or a curve still builds
    // on the shutter path.
    // The linear curve goes first, in its own lock scope. The pipeline thread
    // asks for it within milliseconds of this starting -- the status line calls
    // noise_std_at_brightness, and so does tune_config_snr when SNR auto-tuning
    // is on -- and it must not end up waiting behind the three mask curves it
    // does not need. One batch under one lock would have done exactly that.
    {
        const f32 a0 = cfg.noise_alpha();
        const f32 b0 = cfg.noise_beta();
        const bool sq0 = false;
        std::lock_guard<std::mutex> lk(g_curve_mu);
        curve_build_missing_locked(&a0, &b0, &sq0, 1);
    }

    // The mask's curves: mask_noise_curves_channel / mask_noise_curves. Not
    // needed until the first comparison frame is scored, which is several
    // hundred milliseconds of reference work away.
    f32 a[4], b[4];
    bool sq[4];
    int n = 0;
    const bool msq = cfg.robustness_guide_sqrt;
    if (cfg.debug_noise_model_disabled) {
        a[n] = 0.f; b[n] = 0.f; sq[n] = msq; ++n;
    } else if (cfg.bayer_mode) {
        for (int ch = 0; ch < 3; ++ch) {
            a[n] = cfg.noise_alpha_ch_robustness(ch);
            b[n] = cfg.noise_beta_ch_robustness(ch);
            sq[n] = msq;
            ++n;
        }
    } else {
        a[n] = cfg.noise_alpha_robustness();
        b[n] = cfg.noise_beta_robustness();
        sq[n] = msq;
        ++n;
    }

    std::lock_guard<std::mutex> lk(g_curve_mu);
    curve_build_missing_locked(a, b, sq, n);
}

// Not in the anonymous namespace below: neural_flow's caller (pipeline_paths.cpp)
// needs the exact same guide image the classical robustness path scores
// against, rather than re-deriving its own and risking the two drifting.
// Guide transfer curve: 0 linear, 1 sqrt (variance-stabilizing, 1.4), 2 gamma
// 1/2.2, 3 IEC sRGB. Twin of _apply_guide_curve in the Metal kernel.
static inline f32 apply_guide_curve(f32 v, int curve) {
    switch (curve) {
        case 1: return std::sqrt(std::max(0.f, v));
        case 2: { f32 vc = std::min(std::max(v, 0.f), 1.f); return std::pow(vc, 1.f / 2.2f); }
        case 3: { f32 vc = std::min(std::max(v, 0.f), 1.f);
                  return vc <= 0.0031308f ? 12.92f * vc
                                          : 1.055f * std::pow(vc, 1.f / 2.4f) - 0.055f; }
        default: return v; // 0 = linear
    }
}

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
    // 1.4 parity: 1.4 builds the guide from the UN-prewhitened raw. The port's
    // loader prewhitens (site *= white_balance[c]/white_balance[1]) and the
    // rest of the pipeline depends on that, so undo it HERE, in the guide
    // only, to recover sqrt(raw) exactly as 1.4's cuda_compute_guide_image
    // sees it. undo = wb[1]/wb[c] (green = 1); disabled when the raw was not
    // prewhitened. This makes the noise-off mask's guide domain match 1.4.
    // Un-prewhiten to camera-native (1.4 parity) UNLESS the guide is asked to
    // keep white balance (guide_white_balance) -- a real-RGB guide wants WB, and
    // the port's raw is already white-balanced by the loader's prewhitening.
    // Real-RGB guide (Config::real_rgb_guide). Self-contained: it does not read
    // guide_white_balance / guide_color_matrix / guide_curve at all, and returns
    // before them, so the two schemes cannot interleave.
    //
    // white balance kept -> camera->sRGB -> IEC sRGB curve. The loader
    // prewhitens (raw_io.cpp: site *= wb[c]/wb[1]), so on that path WB is
    // already in the samples and "keep" means doing nothing; a raw that arrived
    // un-prewhitened gets the gains applied here instead, so the guide is the
    // same picture either way.
    if (cfg.real_rgb_guide) {
        f32 wb[3] = {1.f, 1.f, 1.f};
        if (!cfg.raw_prewhitened) {
            const f32 g = cfg.white_balance[1];
            for (int c = 0; c < 3; ++c) {
                const f32 wc = cfg.white_balance[c];
                wb[c] = (std::isfinite(g) && std::isfinite(wc) && g > 0.f) ? (wc / g) : 1.f;
            }
        }
        // No matrix available means no real RGB to speak of; WB plus the curve
        // is still a better-conditioned guide than sensor space, so carry on
        // rather than silently reverting to the 1.4 guide the caller did not ask
        // for.
        const bool ccm = cfg.has_cam_to_srgb;
        const float* M = cfg.cam_to_srgb;
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                f32 sum[3] = {0.f, 0.f, 0.f};
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j) {
                        const uint8_t c = cfg.cfa.p[i][j];
                        if (c < 3) sum[c] += raw.at(2 * y + i, 2 * x + j);
                    }
                f32 rgb[3] = {sum[0] * inv[0] * wb[0],
                              sum[1] * inv[1] * wb[1],
                              sum[2] * inv[2] * wb[2]};
                if (ccm) {
                    const f32 r = rgb[0], gr = rgb[1], b = rgb[2];
                    rgb[0] = M[0] * r + M[1] * gr + M[2] * b;
                    rgb[1] = M[3] * r + M[4] * gr + M[5] * b;
                    rgb[2] = M[6] * r + M[7] * gr + M[8] * b;
                }
                for (int c = 0; c < 3; ++c)
                    guide.at(y, x, c) = apply_guide_curve(rgb[c], 3);  // IEC sRGB
            }
        }
        return guide;
    }

    f32 wbu[3] = {1.f, 1.f, 1.f};
    if (cfg.raw_prewhitened && !cfg.guide_white_balance) {
        const f32 g = cfg.white_balance[1];
        for (int c = 0; c < 3; ++c) {
            const f32 wc = cfg.white_balance[c];
            wbu[c] = (std::isfinite(g) && std::isfinite(wc) && wc > 0.f) ? (g / wc) : 1.f;
        }
    }
    // Effective transfer curve: -1 auto follows robustness_guide_sqrt so the
    // default (no colour flags) stays byte-identical.
    int curve = cfg.guide_curve;
    if (curve < 0) curve = cfg.robustness_guide_sqrt ? 1 : 0;
    const bool apply_ccm = cfg.guide_color_matrix && cfg.has_cam_to_srgb;
    const float* M = cfg.cam_to_srgb; // camera -> linear sRGB (same as the ISP)
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            f32 sum[3] = {0.f, 0.f, 0.f};
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    const uint8_t c = cfg.cfa.p[i][j];
                    if (c < 3) sum[c] += raw.at(2 * y + i, 2 * x + j);
                }
            }
            f32 rgb[3] = {sum[0] * inv[0] * wbu[0],
                          sum[1] * inv[1] * wbu[1],
                          sum[2] * inv[2] * wbu[2]};
            if (apply_ccm) {
                const f32 r = rgb[0], gr = rgb[1], b = rgb[2];
                rgb[0] = M[0] * r + M[1] * gr + M[2] * b;
                rgb[1] = M[3] * r + M[4] * gr + M[5] * b;
                rgb[2] = M[6] * r + M[7] * gr + M[8] * b;
            }
            for (int c = 0; c < 3; ++c)
                guide.at(y, x, c) = apply_guide_curve(rgb[c], curve);
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
    f32 v = std::max(cfg.noise_alpha_robustness() * brightness +
                     cfg.noise_beta_robustness(), 0.f);
    if (nch == 3 && ch == 1)
        v *= 0.5f; // green guide channel is the average of two Bayer greens.
    return v;
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
// Precomputed 1.4 single-curve noise LUT (monte_carlo.py NoiseLut). sigma_sq /
// d_sq already encode the 3-channel SUM, indexed by the mean sqrt-domain guide
// brightness. Loaded once from a raw .bin (magic "N14L", int32 bins, f32
// alpha_rgbg[4], f32 beta_rgbg[4], f32 sigma_noise_sq[bins], f32 d_noise_sq[bins]).
struct NoiseLut14 {
    bool valid = false;
    int  bins = 0;
    f32  alpha_rgbg[4] = {0,0,0,0};
    f32  beta_rgbg[4]  = {0,0,0,0};
    std::vector<f32> sigma_sq;
    std::vector<f32> d_sq;
};

static bool load_noise_lut14_file(const std::string& path, NoiseLut14& lut) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4]; int32_t bins = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "N14L", 4) == 0 &&
              std::fread(&bins, sizeof(int32_t), 1, f) == 1 && bins > 1 && bins < 100000;
    if (ok) {
        lut.bins = bins;
        lut.sigma_sq.resize((size_t)bins);
        lut.d_sq.resize((size_t)bins);
        ok = std::fread(lut.alpha_rgbg, sizeof(f32), 4, f) == 4 &&
             std::fread(lut.beta_rgbg,  sizeof(f32), 4, f) == 4 &&
             std::fread(lut.sigma_sq.data(), sizeof(f32), (size_t)bins, f) == (size_t)bins &&
             std::fread(lut.d_sq.data(),     sizeof(f32), (size_t)bins, f) == (size_t)bins;
    }
    std::fclose(f);
    lut.valid = ok;
    return ok;
}

// Cached once: HHSR_NOISE_LUT14, else <noise_curves_search_dir>/noise_lut.bin.
// Invalid (absent) -> the per-channel runtime model is used instead.
static const NoiseLut14& active_noise_lut14() {
    static NoiseLut14 lut;
    static bool tried = false;
    if (!tried) {
        tried = true;
        std::string path;
        if (const char* e = std::getenv("HHSR_NOISE_LUT14")) path = e;
        else path = noise_curves_search_dir() + "/noise_lut.bin";
        if (load_noise_lut14_file(path, lut))
            std::printf("[noise] Loaded 1.4 noise LUT (%d bins) from %s\n",
                        lut.bins, path.c_str());
    }
    return lut;
}

static void apply_noise_model(const Image& d_p, const Image& ref_means, const Image& ref_vars,
                              const NoiseCurves* const nc_ch[3], Image& d_sq, Image& sigma_sq,
                              bool sqrt_index = false) {
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
                // sqrt guide (1.4): the curve is keyed by LATENT brightness,
                // and the guide mean is sqrt(latent), so index by mean^2.
                f32 bidx = sqrt_index ? brightness * brightness : brightness;
                int id_noise = (int)std::lround(1000.f * bidx);
                // Host: same bins as Python for finite brightness in range; avoid crash on +inf.
                if (!std::isfinite(bidx))
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

// 1.4's noise correction (monte_carlo.py + cuda_compute_d_sigma) via the
// precomputed single-curve LUT: d^2 = Σ_c (Δμ_c)^2 with a Wiener shrink toward
// d_noise_sq(mean brightness); σ^2 = max(Σ_c var_c, sigma_noise_sq(mean
// brightness)). The LUT curves already encode the 3-channel sum and are indexed
// by the mean sqrt-domain guide brightness (round((bins-1)*mean)), so this is
// bit-identical to 1.4 when the LUT is 1.4's own output. Drop-in for
// apply_noise_model when a 1.4 LUT is loaded (per-channel model is the fallback).
static void apply_noise_model_1p4(const Image& d_p, const Image& ref_means,
                                  const Image& ref_vars, const NoiseLut14& lut,
                                  Image& d_sq, Image& sigma_sq) {
    const int n_ch = ref_means.c;
    const int bins = lut.bins;
    d_sq = Image(ref_means.h, ref_means.w, 1);
    sigma_sq = Image(ref_means.h, ref_means.w, 1);
    for (int y = 0; y < ref_means.h; ++y) {
        for (int x = 0; x < ref_means.w; ++x) {
            f32 dq = 0.f, sq = 0.f, bright = 0.f;
            for (int ch = 0; ch < n_ch; ++ch) {
                const f32 dpc = d_p.at(y, x, ch);
                dq += dpc * dpc;
                sq += ref_vars.at(y, x, ch);
                bright += ref_means.at(y, x, ch);
            }
            bright /= (f32)n_ch;
            if (!std::isfinite(bright)) bright = 0.f;
            bright = bright < 0.f ? 0.f : (bright > 1.f ? 1.f : bright);
            int idx = (int)std::lround((f32)(bins - 1) * bright);
            if (idx < 0) idx = 0; else if (idx >= bins) idx = bins - 1;
            sq = std::max(sq, lut.sigma_sq[(size_t)idx]);
            // dq stays +inf for an out-of-bounds sample -> exp(-inf)=0 -> R=0.
            if (std::isfinite(dq) && dq > 0.f) {
                const f32 shrink = dq / (dq + lut.d_sq[(size_t)idx]);
                dq *= shrink * shrink;
            }
            d_sq.at(y, x) = dq;
            sigma_sq.at(y, x) = sq;
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
                                    Image& d_sq, Image& sigma_sq, int num_threads,
                                    bool sqrt_index = false) {
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
                // sqrt guide (1.4): index by mean^2 -- see apply_noise_model.
                f32 bidx = sqrt_index ? brightness * brightness : brightness;
                int id_noise = (int)std::lround(1000.f * bidx);
                if (!std::isfinite(bidx))
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

// 1.4 single-curve noise correction, fused (Δμ derived inline from ref/comp
// means, no materialised d_p). Raw-resolution twin of apply_noise_model_1p4.
static void apply_noise_model_fused_1p4(const Image& ref_means, const Image& comp_means,
                                        const Image& ref_vars, const NoiseLut14& lut,
                                        Image& d_sq, Image& sigma_sq, int num_threads) {
    const int n_ch = ref_means.c;
    const int bins = lut.bins;
    d_sq = Image(ref_means.h, ref_means.w, 1);
    sigma_sq = Image(ref_means.h, ref_means.w, 1);
    parallel_rows(ref_means.h, num_threads, [&](int y) {
        for (int x = 0; x < ref_means.w; ++x) {
            f32 dq = 0.f, sq = 0.f, bright = 0.f;
            for (int ch = 0; ch < n_ch; ++ch) {
                const f32 r = ref_means.at(y, x, ch);
                const f32 comp = comp_means.at(y, x, ch);
                const f32 dpc = std::isfinite(comp) ? std::fabs(r - comp)
                                                    : std::numeric_limits<f32>::infinity();
                dq += dpc * dpc;
                sq += ref_vars.at(y, x, ch);
                bright += r;
            }
            bright /= (f32)n_ch;
            if (!std::isfinite(bright)) bright = 0.f;
            bright = bright < 0.f ? 0.f : (bright > 1.f ? 1.f : bright);
            int idx = (int)std::lround((f32)(bins - 1) * bright);
            if (idx < 0) idx = 0; else if (idx >= bins) idx = bins - 1;
            sq = std::max(sq, lut.sigma_sq[(size_t)idx]);
            if (std::isfinite(dq) && dq > 0.f) {
                const f32 shrink = dq / (dq + lut.d_sq[(size_t)idx]);
                dq *= shrink * shrink;
            }
            d_sq.at(y, x) = dq;
            sigma_sq.at(y, x) = sq;
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

// Bilinear sample of the per-tile motion scale S at a tile coordinate (already
// tile-centred: tc = raw_pos/tile_size - 0.5, same convention as
// FlowField::sample_bilinear). Turns the per-tile s into the per-pixel s the
// paper specifies (Config::robustness_per_pixel_s), removing the tile-block seam.
static inline f32 sample_s_bilinear(const std::vector<f32>& S, int ny, int nx,
                                    f32 tcy, f32 tcx) {
    if (ny <= 0 || nx <= 0) return 0.f;
    const int y0 = (int)std::floor(tcy), x0 = (int)std::floor(tcx);
    const f32 ay = tcy - (f32)y0, ax = tcx - (f32)x0;
    auto cl = [](int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); };
    const int iy0 = cl(y0, ny), iy1 = cl(y0 + 1, ny);
    const int ix0 = cl(x0, nx), ix1 = cl(x0 + 1, nx);
    const f32 s00 = S[(size_t)iy0 * nx + ix0], s01 = S[(size_t)iy0 * nx + ix1];
    const f32 s10 = S[(size_t)iy1 * nx + ix0], s11 = S[(size_t)iy1 * nx + ix1];
    const f32 top = s00 + (s01 - s00) * ax;
    const f32 bot = s10 + (s11 - s10) * ax;
    return top + (bot - top) * ay;
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
                                            0, cfg.num_threads, false);
        st.stds_hires = upscale_warp_stats(st.stds, /*is_ref=*/true, nullptr,
                                           0, cfg.num_threads, false);
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
// guide pixel for that one lookup. Not the common case this toggle is meant
// for; it stays correct, just at its existing granularity rather than the new one.
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

    const NoiseLut14& lut14 = active_noise_lut14();
    const bool use_lut = lut14.valid && !cfg.debug_noise_model_disabled;
    const NoiseCurves* nc_ch[3] = {nullptr, nullptr, nullptr};
    if (!use_lut) {
        if (ref_stats.means.c == 3) {
            for (int ch = 0; ch < 3; ++ch)
                nc_ch[ch] = &mask_noise_curves_channel(cfg, ch);
        } else {
            nc_ch[0] = &mask_noise_curves(cfg);
        }
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
                                        tile_size, cfg.num_threads, false);
    }

    const Image& ref_means = ref_stats.means_hires;
    const Image& ref_vars = ref_stats.stds_hires;
    const int h = ref_means.h, w = ref_means.w;
    const int nch = ref_means.c;
    if (comp_means.h != h || comp_means.w != w || comp_means.c != nch)
        return Image();

    Image d_sq, sigma_sq;
    if (use_lut)
        apply_noise_model_fused_1p4(ref_means, comp_means, ref_vars, lut14,
                                    d_sq, sigma_sq, cfg.num_threads);
    else
        apply_noise_model_fused(ref_means, comp_means, ref_vars, nc_ch, d_sq, sigma_sq,
                                cfg.num_threads, cfg.robustness_guide_sqrt);
    std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2);

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

Image build_robustness_nn_features(const RefStats& ref_stats, const Image& comp_means,
                                   const FlowField& flow, int tile_size,
                                   const Config& cfg, int y0, bool raw_res) {
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
    const int strip_h = kRobustnessNnStripRows + 2 * kRobustnessNnHalo;
    Image feat(strip_h, w, kRobustnessNnChannels);
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
            auto span_at = [&](int cy, int cx) {
                f32 mnx = std::numeric_limits<f32>::infinity(), mny = mnx;
                f32 mxx = -mnx, mxy = -mnx;
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j) {
                        const int yy = cy + i, xx = cx + j;
                        if (yy < 0 || yy >= flow.ny || xx < 0 || xx >= flow.nx) continue;
                        const f32 vx = flow.dx(yy, xx), vy = flow.dy(yy, xx);
                        mnx = std::min(mnx, vx); mxx = std::max(mxx, vx);
                        mny = std::min(mny, vy); mxy = std::max(mxy, vy);
                    }
                const f32 dxs = (mxx > mnx) ? (mxx - mnx) : 0.f;
                const f32 dys = (mxy > mny) ? (mxy - mny) : 0.f;
                return std::sqrt(dxs * dxs + dys * dys);
            };
            const f32 Mspan = bilerp(span_at(iy0, ix0), span_at(iy0, ix1),
                                     span_at(iy1, ix0), span_at(iy1, ix1));

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
        }
    });
    return feat;
}

// The analytic mask exactly as it has always been. compute_robustness below
// is a thin wrapper that runs the learned refinement on top of whatever this
// returns, so every path through here -- Metal, raw-resolution, the learned
// replacement, the two degenerate early returns -- gets the same treatment
// without each having to remember to ask for it.
// refined_on_gpu is set when the Metal path's rob_refine_mask kernel has
// already applied the learned refinement, so the wrapper below does not apply
// it a second time.
static Image compute_robustness_core(const Image& comp_raw, const RefStats& ref_stats,
                                     const FlowField& flow, int tile_size,
                                     const Config& cfg, Image* s_select_out,
                                     bool* refined_on_gpu) {
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

    // Learned mask (robustness_nn.h) in place of Eq. 5-9. Tried before both
    // the Metal and CPU analytic paths, and falls through to them whenever the
    // model is absent, fails to load, or the features cannot be built -- so a
    // missing model degrades to the reference behaviour rather than to no mask.
    //
    // Eq. 9's 5x5 minimum is deliberately NOT applied on top. That step exists
    // to spread rejection outward from a pointwise test with no spatial
    // context; this network has a ~30 raw-pixel receptive field and was
    // trained and measured to emit the final per-pixel decision, so dilating
    // it again would double-count and would not match the reported numbers.
    if (cfg.use_neural_robustness && robustness_nn_available()) {
#ifdef __APPLE__
        // See metal_fetch_host_ref_stats: on this path the reference stats are
        // GPU-resident by design, so bring them across before building
        // features from them. One readback per burst, not per frame -- the
        // copy stays in ref_stats.
        RefStats* mutable_stats = const_cast<RefStats*>(&ref_stats);
        if (mutable_stats->means.data.empty() &&
            !metal_fetch_host_ref_stats(*mutable_stats)) {
            // Could not get them; the analytic path below reads the same
            // statistics straight from the GPU and is unaffected.
            return compute_robustness_metal(comp_raw, ref_stats, flow, tile_size,
                                            cfg, s_select_out);
        }
#endif
        // Honour the raw-resolution toggle. Previously this hook ran before the
        // raw-res dispatch and always returned a 3 MP mask, so enabling that
        // setting alongside the learned mask silently did nothing.
        const bool nn_raw = cfg.robustness_raw_resolution_active() &&
                            ref_stats.means_hires.h > 0 &&
                            ref_stats.means_hires.data.size() ==
                                (size_t)ref_stats.means_hires.h *
                                ref_stats.means_hires.w * ref_stats.means_hires.c;
        Image cm_nn;
        {
            Image guide_nn = compute_guide(comp_raw, cfg);
            Image cv_nn;   // variance is not a feature; freed with this scope
            local_stats_3x3(guide_nn, cm_nn, cv_nn);
            if (nn_raw) {
                // Dodgson upscale + flow warp into the reference frame, the
                // same transform the analytic raw-res path applies.
                Image hires = upscale_warp_stats(cm_nn, /*is_ref=*/false, &flow,
                                                 tile_size, cfg.num_threads, false);
                cm_nn = std::move(hires);
            }
        }
        const int nh = cm_nn.h, nw = cm_nn.w;
        const int strip_h = kRobustnessNnStripRows + 2 * kRobustnessNnHalo;
        Image nn_mask(nh, nw, 1);
        // Every window is strip_h tall and fully inside the image, so Core ML
        // sees one input shape for the whole burst (no reshape per strip) and
        // the result matches whole-plane inference exactly. An image shorter
        // than one window would make that impossible; there is nothing to
        // save there either, so fall back to the analytic mask.
        bool ok = (nh >= strip_h && nw > 0);
        for (int y0 = 0; ok && y0 < nh; y0 += kRobustnessNnStripRows) {
            const int top = std::min(std::max(y0 - kRobustnessNnHalo, 0), nh - strip_h);
            Image feat = build_robustness_nn_features(ref_stats, cm_nn, flow,
                                                      tile_size, cfg, top, nn_raw);
            Image strip;
            if (feat.h != strip_h || !robustness_nn_infer(feat, strip) ||
                strip.h != strip_h || strip.w != nw) {
                ok = false;
                break;
            }
            const int rows = std::min(kRobustnessNnStripRows, nh - y0);
            for (int r = 0; r < rows; ++r)
                std::memcpy(&nn_mask.at(y0 + r, 0),
                            &strip.at(y0 - top + r, 0),
                            (size_t)nw * sizeof(f32));
        }
        if (ok) {
            // The learned mask makes no s1/s2 choice, so report the strict
            // prior uniformly rather than leaving the split masks undefined.
            if (s_select_out) {
                *s_select_out = Image(nn_mask.h, nn_mask.w, 1);
                std::fill(s_select_out->data.begin(), s_select_out->data.end(), 1.f);
            }
            return nn_mask;
        }
    }

#ifdef __APPLE__
    // Metal GPU path — same Alg. robustness math as the CPU path below. The
    // Metal noise kernel only implements the per-channel model, so when a 1.4
    // single-curve LUT is loaded (Config parity mode) skip Metal and run the
    // VERIFIED CPU path below instead. Robustness then runs on CPU for this
    // frame, but the expensive runtime Monte-Carlo curve build is skipped (the
    // LUT replaces it), so it is not necessarily slower overall.
    if (!active_noise_lut14().valid) {
        Image gpu = compute_robustness_metal(comp_raw, ref_stats, flow, tile_size, cfg,
                                             s_select_out, refined_on_gpu);
        if (gpu.h > 0 && gpu.w > 0) return gpu;
        return Image();
    }
#endif
    if (cfg.robustness_raw_resolution_active()) {
        Image raw_res = compute_robustness_raw_res(comp_raw, ref_stats, flow, tile_size,
                                                    cfg, s_select_out);
        if (raw_res.h > 0 && raw_res.w > 0) return raw_res;
        // Falls through to the guide-resolution path below if the hires ref
        // stats weren't populated (e.g. robustness was off when the burst
        // started).
    }

    // 1.4 single-curve LUT (parity) vs the per-channel runtime model. When the
    // LUT is used the per-channel Monte-Carlo curves are NOT built at all, so
    // the runtime MC cost disappears.
    const NoiseLut14& lut14 = active_noise_lut14();
    const bool use_lut = lut14.valid && !cfg.debug_noise_model_disabled;
    // One curve per guide channel (3 for Bayer, matching R/(G1+G2)/2/B; 1
    // otherwise) rather than one curve shared by all channels -- see
    // Config::noise_alpha_ch/noise_beta_ch and make_noise_curves_channel.
    const NoiseCurves* nc_ch[3] = {nullptr, nullptr, nullptr};
    if (!use_lut) {
        if (ref_stats.means.c == 3) {
            for (int ch = 0; ch < 3; ++ch)
                nc_ch[ch] = &mask_noise_curves_channel(cfg, ch);
        } else {
            nc_ch[0] = &mask_noise_curves(cfg);
        }
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
            // Same sampling as the merge, deliberately: the mask must score
            // the correspondence the merge will actually fetch.
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
    // 1.4 parity: use the LUT's exact single-curve correction, else per-channel.
    if (use_lut)
        apply_noise_model_1p4(d_p, ref_stats.means, ref_stats.stds, lut14, d_sq, sigma_sq);
    else
        apply_noise_model(d_p, ref_stats.means, ref_stats.stds, nc_ch, d_sq, sigma_sq,
                          cfg.robustness_guide_sqrt);
    std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2);

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
            const size_t pidx = (size_t)patch_idy * flow.nx + patch_idx;
            f32 s = S[pidx];
            f32 sig = sigma_sq.at(y, x);
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
            // Geometry-aware rejection (Config::motion_geom_reject_enabled):
            // reject where the per-tile TRANSLATION is a poor model of the local
            // motion. E = flow-gradient * offset-from-tile-centre (within-tile
            // motion variation, from neighbouring tile vectors), weighted by the
            // reference gradient so only visible-edge errors reject. Inert under
            // translation (flow gradient ~ 0). Validated on APC bursts.
            bool geom_reject = false;
            if (cfg.motion_geom_reject_enabled && flow.ny >= 3 && flow.nx >= 3) {
                auto clt = [](int a, int hi) { return a < 0 ? 0 : (a >= hi ? hi - 1 : a); };
                const int ptu = clt(patch_idy - 1, flow.ny), ptd = clt(patch_idy + 1, flow.ny);
                const int pxl = clt(patch_idx - 1, flow.nx), pxr = clt(patch_idx + 1, flow.nx);
                const f32 inv2ts = 1.f / (2.f * (f32)tile_size);
                const f32 gdxdx = (flow.dx(patch_idy, pxr) - flow.dx(patch_idy, pxl)) * inv2ts;
                const f32 gdydx = (flow.dy(patch_idy, pxr) - flow.dy(patch_idy, pxl)) * inv2ts;
                const f32 gdxdy = (flow.dx(ptd, patch_idx) - flow.dx(ptu, patch_idx)) * inv2ts;
                const f32 gdydy = (flow.dy(ptd, patch_idx) - flow.dy(ptu, patch_idx)) * inv2ts;
                const f32 sc = (ref_stats.means.c == 3) ? 2.f : 1.f;
                const f32 rawx = sc * (f32)x + 0.5f * (sc - 1.f);
                const f32 rawy = sc * (f32)y + 0.5f * (sc - 1.f);
                const f32 u = rawx - ((f32)patch_idx + 0.5f) * (f32)tile_size;
                const f32 v = rawy - ((f32)patch_idy + 0.5f) * (f32)tile_size;
                const f32 ex = gdxdx * u + gdxdy * v, ey = gdydx * u + gdydy * v;
                const f32 Emag = std::sqrt(ex * ex + ey * ey);
                const int xl = std::max(0, x - 1), xr = std::min(w - 1, x + 1);
                const int yu = std::max(0, y - 1), yd = std::min(h - 1, y + 1);
                const f32 gix = 0.5f * (ref_stats.means.at(y, xr, 0) - ref_stats.means.at(y, xl, 0)) / sc;
                const f32 giy = 0.5f * (ref_stats.means.at(yd, x, 0) - ref_stats.means.at(yu, x, 0)) / sc;
                const f32 gmag = std::sqrt(gix * gix + giy * giy);
                // Absolute criterion (|grad I| * |E|): preserves the good-light
                // behaviour exactly -- the bright-scene rejections you already get
                // stay. ALWAYS applied.
                geom_reject = (gmag * Emag) > cfg.motion_geom_reject_threshold;
                // Relative (exposure-invariant) criterion ADDED on top: contrast
                // (|grad g|/g, noise-floor-subtracted) * |E|. This is what catches
                // the low-light misalignments the absolute form misses (its
                // gradient shrinks in dim scenes). Union, so nothing good-light is
                // lost. bri = local guide mean; nsig = per-raw-pixel guide noise
                // sigma (same 1/sc units as gmag). See types.h.
                if (!geom_reject && cfg.motion_geom_relative) {
                    const f32 bri = guide_brightness(ref_stats.means, y, x);
                    const f32 nsig = std::sqrt(guide_noise_var(cfg, ref_stats.means.c, 0, bri)) / sc;
                    const f32 gmag_dn = std::max(0.f, gmag - cfg.motion_geom_noise_floor_mult * nsig);
                    geom_reject = (gmag_dn / (bri + 1e-4f)) * Emag >
                                  cfg.motion_geom_reject_threshold_relative;
                }
            }
            const bool hard_reject = geom_reject;
            f32 r_val = hard_reject
                ? 0.f
                : clampf(s * std::exp(-d_sq.at(y, x) / sig) - cfg.r_t, 0.f, 1.f);
            R.at(y, x) = r_val;
            if (s_select_out) s_select_out->at(y, x) = (s <= cfg.r_s1) ? 1.f : 0.f;
        }
    }
    return local_min_5x5(R);
}

// ======================= learned refinement (Config::
// robustness_refine_nn_enabled) ===========================================
//
// See stages.h for the channel contract and types.h for why this stage
// exists at all. The short version: Eq. 5-9 decides from a photometric
// statistic, and the residual misalignment left by ONE FLOW VECTOR PER TILE
// under rotation or parallax makes that statistic more confident, not less
// -- sigma picks up the edge's own texture faster than d picks up a
// fraction-of-a-pixel shift. Everything below is in service of handing a
// network the geometric evidence instead, and of making sure it can only
// ever subtract.

std::vector<f32> robustness_motion_prior(const FlowField& flow, const Config& cfg) {
    return compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2);
}

void robustness_correspondence(const Image& ref_means, const Image& ref_vars,
                               const Image& comp_means, const FlowField& flow,
                               int tile_size, bool raw_res, const Config& cfg,
                               Image& d_sq, Image& sigma_sq) {
    d_sq = Image(); sigma_sq = Image();
    if (ref_means.h <= 0 || ref_means.w <= 0 || ref_means.c <= 0) return;
    if (comp_means.c != ref_means.c) return;
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty() || tile_size <= 0) return;

    Image d_p(ref_means.h, ref_means.w, ref_means.c);
    parallel_rows(ref_means.h, cfg.num_threads, [&](int y) {
        for (int x = 0; x < ref_means.w; ++x) {
            f32 fx = 0.f, fy = 0.f;
            if (!raw_res) {
                // Nearest tile, and the displacement halved into guide units:
                // the same sampling compute_robustness_core uses, because the
                // mask has to score the correspondence the MERGE will fetch.
                const int pty = std::min(flow.ny - 1,
                    std::max(0, (int)((2.f * (f32)y + 0.5f) / (f32)tile_size)));
                const int ptx = std::min(flow.nx - 1,
                    std::max(0, (int)((2.f * (f32)x + 0.5f) / (f32)tile_size)));
                fx = 0.5f * flow.dx(pty, ptx);
                fy = 0.5f * flow.dy(pty, ptx);
            }
            for (int ch = 0; ch < ref_means.c; ++ch) {
                const f32 cv = raw_res
                    ? comp_means.at(y, x, ch)
                    : sample_bilinear_or_inf(comp_means, (f32)y + fy, (f32)x + fx, ch);
                d_p.at(y, x, ch) = std::isfinite(cv)
                    ? std::fabs(ref_means.at(y, x, ch) - cv)
                    : std::numeric_limits<f32>::infinity();
            }
        }
    });
    const NoiseLut14& lut14 = active_noise_lut14();
    if (lut14.valid && !cfg.debug_noise_model_disabled) {
        apply_noise_model_1p4(d_p, ref_means, ref_vars, lut14, d_sq, sigma_sq);
    } else {
        const NoiseCurves* nc_ch[3] = {nullptr, nullptr, nullptr};
        if (ref_means.c == 3)
            for (int ch = 0; ch < 3; ++ch) nc_ch[ch] = &mask_noise_curves_channel(cfg, ch);
        else
            nc_ch[0] = &mask_noise_curves(cfg);
        apply_noise_model(d_p, ref_means, ref_vars, nc_ch, d_sq, sigma_sq,
                          cfg.robustness_guide_sqrt);
    }
}

static_assert(RR_CHANNELS == kRobustnessRefineChannels,
              "robustness_refine_shared.h and types.h disagree on the channel "
              "count; the trained weights match one of them");

Image build_robustness_refine_features(const RefStats& ref_stats,
                                       const Image& comp_means, const Image& R,
                                       const Image& d_sq, const Image& sigma_sq,
                                       const FlowField& flow, int tile_size,
                                       const Config& cfg, int y0, int strip_h) {
    if (R.h <= 0 || R.w <= 0 || strip_h <= 0) return Image();
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty() || tile_size <= 0) return Image();

    // Which set of reference statistics this mask was made from. Decided by
    // matching R's own dimensions rather than by reading the toggle, for the
    // same reason merge.cpp does: the raw-resolution path silently falls back
    // to guide resolution when the hires stats are missing, so the flag is not
    // the authority on what actually ran.
    const bool raw_res = (R.h != ref_stats.means.h || R.w != ref_stats.means.w);
    const Image& rm = raw_res ? ref_stats.means_hires : ref_stats.means;
    const Image& rv = raw_res ? ref_stats.stds_hires : ref_stats.stds;
    if (rm.h != R.h || rm.w != R.w || rm.c < 1) return Image();
    // Dimensions are not enough: on the Metal path RefStats can carry shape
    // with the pixel vectors empty (the statistics stay GPU-resident), and
    // indexing that reads off the end of an empty vector on every pixel.
    if (rm.data.size() < (size_t)rm.h * rm.w * rm.c ||
        rv.data.size() < (size_t)rv.h * rv.w * rv.c)
        return Image();
    if (comp_means.c != rm.c || comp_means.h <= 0 || comp_means.w <= 0) return Image();
    if (d_sq.h != R.h || d_sq.w != R.w || sigma_sq.h != R.h || sigma_sq.w != R.w)
        return Image();

    const int h = R.h, w = R.w, nch = rm.c;
    // Raw pixels per feature pixel. Every length reaching rr_features is
    // converted with it, so |E| here and motion_geom_reject_threshold there
    // are the same quantity.
    const f32 sc = raw_res ? 1.f : 2.f;
    const f32 inv2ts = 1.f / (2.f * (f32)tile_size);

    const std::vector<f32> S = compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2);

    // ---- two scalar luma planes for the strip, built once -----------------
    // Everything spatial in rr_features -- the gradient, the 3x3 structure
    // tensor, the Laplacian, the local mean residual -- would otherwise
    // re-read nch interleaved channels for every tap, which is ~200 scattered
    // loads per output pixel and, measured, 1.8 s per 3 MP frame against
    // 0.27 s this way. The Metal kernel does gather per thread instead: on a
    // GPU the scattered reads are cheap and a prepass would cost a buffer.
    //
    // The comparison plane is sampled WHERE THE FLOW POINTS, which is also a
    // correctness fix: the local mean residual previously differenced the
    // reference against the comparison frame at the same coordinate, with no
    // flow applied at all, so channel 19 was measuring the scene's global
    // motion rather than the residual after alignment.
    //
    // Two rows of margin: the structure tensor evaluates a gradient at y+-1,
    // and that gradient reaches y+-2.
    const int MARGIN = 2;
    const int lum_h = strip_h + 2 * MARGIN;
    std::vector<f32> refl((size_t)lum_h * w), cmpl((size_t)lum_h * w);
    // Divide rather than multiply by a reciprocal: rr_luma in the shared header
    // divides, and the two differ by up to one ULP. Measured, that alone was
    // the entire disagreement between this path and the GPU one on 28 of 3.28
    // million pixels, so matching it is free and makes the parity check exact
    // enough to be worth trusting.
    const f32 fnch = (f32)nch;
    parallel_rows(lum_h, cfg.num_threads, [&](int ly) {
        const int y = std::min(std::max(y0 - MARGIN + ly, 0), h - 1);
        f32* rrow = &refl[(size_t)ly * w];
        f32* crow = &cmpl[(size_t)ly * w];
        for (int x = 0; x < w; ++x) {
            f32 s = 0.f;
            for (int c = 0; c < nch; ++c) s += rm.at(y, x, c);
            rrow[x] = s / fnch;

            f32 cs = 0.f;
            bool ok = true;
            if (raw_res) {
                // upscale_warp_stats already applied the flow when building
                // the hires comparison statistics, so the correct sample sits
                // at (y,x) and shifting again would double-apply it.
                for (int c = 0; c < nch; ++c) cs += comp_means.at(y, x, c);
            } else {
                const f32 rawy = sc * (f32)y + 0.5f * (sc - 1.f);
                const f32 rawx = sc * (f32)x + 0.5f * (sc - 1.f);
                auto cl0 = [](int a, int hi) { return a < 0 ? 0 : (a >= hi ? hi - 1 : a); };
                const int pty = cl0((int)((rawy + 0.5f) / (f32)tile_size), flow.ny);
                const int ptx = cl0((int)((rawx + 0.5f) / (f32)tile_size), flow.nx);
                const f32 fyg = 0.5f * flow.dy(pty, ptx);
                const f32 fxg = 0.5f * flow.dx(pty, ptx);
                for (int c = 0; c < nch; ++c) {
                    const f32 v = sample_bilinear_or_inf(comp_means, (f32)y + fyg,
                                                         (f32)x + fxg, c);
                    if (!std::isfinite(v)) { ok = false; break; }
                    cs += v;
                }
            }
            // The flow points outside the comparison frame. Eq. 6 gives that
            // pixel d = inf and R = 0, so the refinement has nothing left to
            // take; matching the reference makes the residual zero rather than
            // infinite, which keeps the features finite.
            crow[x] = ok ? cs / fnch : rrow[x];
        }
    });
    auto plane_at = [&](const std::vector<f32>& p, int sy, int x) -> f32 {
        const int ly = std::min(std::max(sy + MARGIN, 0), lum_h - 1);
        return p[(size_t)ly * w + (size_t)std::min(std::max(x, 0), w - 1)];
    };

    Image feat(strip_h, w, kRobustnessRefineChannels);
    parallel_rows(strip_h, cfg.num_threads, [&](int sy) {
        const int y = std::min(std::max(y0 + sy, 0), h - 1);
        for (int x = 0; x < w; ++x) {
            RefineInputs in;
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                    in.refl[i][j] = plane_at(refl, sy + i - 2, x + j - 2);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    in.cmpl[i][j] = plane_at(cmpl, sy + i - 1, x + j - 1);

            // ---- tile addressing. Nearest tile, deliberately: this is the
            // vector the merge will actually fetch with, and the quantity
            // being judged is how badly THAT vector misrepresents the motion
            // inside its own tile. Smoothing it between tile centres would
            // describe a flow field the merge does not use.
            const f32 rawy = sc * (f32)y + 0.5f * (sc - 1.f);
            const f32 rawx = sc * (f32)x + 0.5f * (sc - 1.f);
            auto clt = [](int a, int hi) { return a < 0 ? 0 : (a >= hi ? hi - 1 : a); };
            const int pty = clt((int)((rawy + 0.5f) / (f32)tile_size), flow.ny);
            const int ptx = clt((int)((rawx + 0.5f) / (f32)tile_size), flow.nx);
            const size_t pidx = (size_t)pty * flow.nx + ptx;
            const int ptu = clt(pty - 1, flow.ny), ptd = clt(pty + 1, flow.ny);
            const int pxl = clt(ptx - 1, flow.nx), pxr = clt(ptx + 1, flow.nx);

            in.R = R.at(y, x);
            in.d_sq = d_sq.at(y, x);
            in.sigma_sq = sigma_sq.at(y, x);
            f32 noise_var_sum = 0.f;
            for (int c = 0; c < nch; ++c)
                noise_var_sum += guide_noise_var(cfg, nch, c, rm.at(y, x, c));
            in.noise_var_sum = noise_var_sum;
            in.gdxdx = (flow.dx(pty, pxr) - flow.dx(pty, pxl)) * inv2ts;
            in.gdydx = (flow.dy(pty, pxr) - flow.dy(pty, pxl)) * inv2ts;
            in.gdxdy = (flow.dx(ptd, ptx) - flow.dx(ptu, ptx)) * inv2ts;
            in.gdydy = (flow.dy(ptd, ptx) - flow.dy(ptu, ptx)) * inv2ts;
            in.u = rawx - ((f32)ptx + 0.5f) * (f32)tile_size;
            in.v = rawy - ((f32)pty + 0.5f) * (f32)tile_size;
            in.tile_size = (f32)tile_size;

            f32 mnx = std::numeric_limits<f32>::infinity(), mny = mnx;
            f32 mxx = -mnx, mxy = -mnx;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j) {
                    const int yy = pty + i, xx = ptx + j;
                    if (yy < 0 || yy >= flow.ny || xx < 0 || xx >= flow.nx) continue;
                    const f32 vx = flow.dx(yy, xx), vy = flow.dy(yy, xx);
                    mnx = std::min(mnx, vx); mxx = std::max(mxx, vx);
                    mny = std::min(mny, vy); mxy = std::max(mxy, vy);
                }
            const f32 spx = (mxx > mnx) ? (mxx - mnx) : 0.f;
            const f32 spy = (mxy > mny) ? (mxy - mny) : 0.f;
            in.Mspan = std::sqrt(spx * spx + spy * spy);

            f32 s_prior = S[pidx];
            if (cfg.flow_reject_ambiguous_enabled &&
                pidx < flow.match_ambiguous.size() && flow.match_ambiguous[pidx] != 0u)
                s_prior = std::min(s_prior, cfg.r_s1);
            in.s_prior = s_prior;
            in.sc = sc;
            in.nch = nch;

            rr_features(&in, &feat.at(sy, x, 0));
        }
    });
    return feat;
}

bool apply_robustness_refinement(Image& R, const Image& comp_raw,
                                 const RefStats& ref_stats, const FlowField& flow,
                                 int tile_size, const Config& cfg,
                                 float* changed_frac) {
    if (changed_frac) *changed_frac = 0.f;
    if (!cfg.robustness_refine_nn_enabled) return false;
    if (R.h <= 0 || R.w <= 0 || R.c != 1) return false;
    // compute_robustness_metal returns dimensions with no pixels when the mask
    // stays GPU-resident for the merge. Nothing on the host can refine that --
    // and nothing needs to, because the GPU path refines in its own kernel; the
    // only way to arrive here with an empty mask is that kernel being absent,
    // in which case leaving R alone is the correct fail-closed behaviour.
    if (R.data.size() < (size_t)R.h * (size_t)R.w) return false;
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty() || tile_size <= 0) return false;
    if (!robustness_refine_available()) return false;

#ifdef __APPLE__
    // The Metal path keeps the reference statistics on the GPU; bring them
    // across once per burst, exactly as the replacement network's caller does.
    if (ref_stats.means.data.empty()) {
        RefStats* mutable_stats = const_cast<RefStats*>(&ref_stats);
        if (!metal_fetch_host_ref_stats(*mutable_stats)) return false;
    }
#endif
    const bool raw_res = (R.h != ref_stats.means.h || R.w != ref_stats.means.w);
    const Image& rm = raw_res ? ref_stats.means_hires : ref_stats.means;
    const Image& rv = raw_res ? ref_stats.stds_hires : ref_stats.stds;
    if (rm.h != R.h || rm.w != R.w) return false;
    if (rm.data.size() < (size_t)rm.h * rm.w * rm.c ||
        rv.data.size() < (size_t)rv.h * rv.w * rv.c)
        return false;

    // Comparison statistics on the same lattice as R, by the same route the
    // mask took to get there.
    Image comp_means;
    {
        Image guide = compute_guide(comp_raw, cfg);
        Image comp_vars;
        local_stats_3x3(guide, comp_means, comp_vars);
        if (raw_res)
            comp_means = upscale_warp_stats(comp_means, /*is_ref=*/false, &flow,
                                            tile_size, cfg.num_threads, false);
    }
    if (comp_means.h <= 0 || comp_means.w <= 0) return false;

    // Eq. 6 again, for the two channels that have to be the mask's own
    // numbers rather than a reimplementation of them.
    Image d_sq, sigma_sq;
    robustness_correspondence(rm, rv, comp_means, flow, tile_size, raw_res, cfg,
                              d_sq, sigma_sq);
    if (d_sq.h != rm.h || sigma_sq.h != rm.h) return false;

    // Strips, for the same reason the replacement network uses them: the
    // weights are negligible but the activations are not, and this pipeline
    // is already close enough to the footprint limit that a gigabyte of
    // intermediates is a jetsam kill rather than a slowdown.
    const int strip_rows = kRobustnessRefineStripRows;
    const int halo = kRobustnessRefineHalo;
    const int strip_h = strip_rows + 2 * halo;
    if (R.h < strip_h) return false;

    Image refined(R.h, R.w, 1);
    const f32 kappa = clampf(cfg.robustness_refine_max_reduction, 0.f, 1.f);
    const f32 dead = clampf(cfg.robustness_refine_deadzone, 0.f, 1.f);
    size_t changed = 0;

    for (int y0 = 0; y0 < R.h; y0 += strip_rows) {
        const int top = std::min(std::max(y0 - halo, 0), R.h - strip_h);
        Image feat = build_robustness_refine_features(ref_stats, comp_means, R, d_sq,
                                                      sigma_sq, flow, tile_size, cfg,
                                                      top, strip_h);
        Image q;
        if (feat.h != strip_h || !robustness_refine_infer(feat, q) ||
            q.h != strip_h || q.w != R.w) {
            // Any failure leaves R exactly as the analytic mask produced it.
            return false;
        }
        const int rows = std::min(strip_rows, R.h - y0);
        for (int r = 0; r < rows; ++r) {
            const f32* qp = &q.at(y0 - top + r, 0);
            const f32* rp = &R.at(y0 + r, 0);
            f32* op = &refined.at(y0 + r, 0);
            for (int x = 0; x < R.w; ++x) {
                // q is confidence that the pixel should be KEPT. Every part of
                // this is deliberate: the reduction is bounded by kappa, it
                // MULTIPLIES R rather than replacing it (so R == 0 stays 0 and
                // the network can never resurrect a pixel Eq. 5-9 rejected),
                // and anything inside the dead zone passes through bit for bit
                // so the stage stays a sparse correction rather than a new
                // mask wearing the old one as a hat.
                const f32 drop = 1.f - clampf(qp[x], 0.f, 1.f);
                if (drop <= dead) { op[x] = rp[x]; continue; }
                op[x] = rp[x] * (1.f - kappa * drop);
                if (rp[x] > 0.f && op[x] != rp[x]) ++changed;
            }
        }
    }
    R = std::move(refined);
    if (changed_frac)
        *changed_frac = (f32)((double)changed / ((double)R.h * (double)R.w));
    return true;
}

Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg,
                         Image* s_select_out) {
    bool refined_on_gpu = false;
    Image R = compute_robustness_core(comp_raw, ref_stats, flow, tile_size, cfg,
                                      s_select_out, &refined_on_gpu);
    // The Metal path refines inside its own command buffer -- one kernel over
    // the finished mask, with the guide, the statistics and the flow already in
    // GPU buffers and nothing read back. Doing it again here would apply the
    // reduction twice, and would also be the expensive way round: this CPU
    // implementation has to rebuild on the host everything that path already
    // holds.
    if (!refined_on_gpu && R.h > 0 && R.w > 0)
        apply_robustness_refinement(R, comp_raw, ref_stats, flow, tile_size, cfg);
    return R;
}

} // namespace hhsr
