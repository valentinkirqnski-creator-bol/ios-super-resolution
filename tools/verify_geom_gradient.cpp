// Verifies core/geom_gradient.h -- the noise-aware gradient used ONLY by the
// geometric motion-rejection test.
//
//   g++ -std=c++17 -O2 -I core tools/verify_geom_gradient.cpp -o verify_geom_gradient
//   ./verify_geom_gradient        (exit 0 iff every acceptance criterion holds)
//
// The three cases that decide whether the change is safe:
//   A  bright/low noise  -- every pixel able to reject must be UNCHANGED
//   B  low light + edge  -- a real edge must keep its gradient
//   C  low light + flat  -- noise must be less able to fake an edge
//
#include "geom_gradient.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace hhsr;

// The two profiles from the device: bright main camera, and the noisier front one.
struct NoiseModel { const char* name; float alpha, beta; };
static const NoiseModel kBright{"bright  (main, ISO low)", 0.000465193f, 1.07414e-06f};
static const NoiseModel kDark  {"dark    (front, high gain)", 0.00107903f, 3.58883e-06f};

static const float kThreshold = 0.0045f;   // motion_geom_reject_threshold, UNCHANGED

struct Field { int w, h; std::vector<float> v; };

// A scene in guide units, then noise, then the 3x3 local mean.
static Field make_means(int w, int h, const NoiseModel& nm, int kind,
                        float level, float step, uint32_t seed) {
    Field clean{w, h, std::vector<float>((size_t)w * h, 0.f)};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float s = level;
            if (kind == 1) s = (x < w / 2) ? level : level + step;          // step edge
            else if (kind == 2) s = level + step * (float)x / (float)(w - 1); // ramp
            clean.v[(size_t)y * w + x] = s;
        }
    // Photon + read noise from the model the robustness path uses.
    std::mt19937 rng(seed);
    Field noisy = clean;
    for (size_t i = 0; i < noisy.v.size(); ++i) {
        const float b = std::min(std::max(clean.v[i], 0.f), 1.f);
        const float sd = std::sqrt(std::max(nm.alpha * b + nm.beta, 0.f));
        std::normal_distribution<float> g(0.f, sd);
        noisy.v[i] = clean.v[i] + g(rng);
    }
    // local_stats_3x3: the means image the geometry test actually reads.
    Field means = noisy;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0.f;
            int n = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int yy = std::min(h - 1, std::max(0, y + dy));
                    const int xx = std::min(w - 1, std::max(0, x + dx));
                    acc += noisy.v[(size_t)yy * w + xx];
                    ++n;
                }
            means.v[(size_t)y * w + x] = acc / (float)n;
        }
    return means;
}

static void gather(const Field& f, int x, int y, float n[9]) {
    int k = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const int yy = std::min(f.h - 1, std::max(0, y + dy));
            const int xx = std::min(f.w - 1, std::max(0, x + dx));
            n[k++] = f.v[(size_t)yy * f.w + xx];
        }
}

// Pixels are bucketed by the SHARP gradient's SNR, because that is what decides
// whether the blend touches them at all:
//   LOW    snr <= snr_lo   fully averaged
//   MID    between         partially
//   HIGH   snr >= snr_hi   blend is exactly 0, must be bit-identical
// Averaging gmag over a whole step-edge field tells you nothing -- the flat
// majority swamps the edge.
enum Bucket { LOW = 0, MID = 1, HIGH = 2, NB = 3 };

struct Bstat {
    double old_g = 0, new_g = 0, sigma = 0;
    long px = 0, old_rej = 0, new_rej = 0, changed = 0;
    double old_peak = 0, new_peak = 0;
};

struct Result { Bstat b[NB]; };

static Result run(const Field& means, const NoiseModel& nm, float sc, float Emag,
                  const GeomGradParams& gp) {
    Result r;
    for (int y = 2; y < means.h - 2; ++y)
        for (int x = 2; x < means.w - 2; ++x) {
            float n[9];
            gather(means, x, y, n);
            const float bri = std::min(std::max(n[4], 0.f), 1.f);
            const float sample_var = std::max(nm.alpha * bri + nm.beta, 0.f) / 9.f;
            const GeomGradient o = geom_gradient(n, sc, sample_var, gp, false);
            const GeomGradient g = geom_gradient(n, sc, sample_var, gp, true);
            const int bi = (g.snr <= gp.snr_lo) ? LOW : ((g.snr >= gp.snr_hi) ? HIGH : MID);
            Bstat& b = r.b[bi];
            const double os = (double)o.gmag * Emag, ns = (double)g.gmag * Emag;
            const bool orej = os > kThreshold, nrej = ns > kThreshold;
            b.old_g += o.gmag; b.new_g += g.gmag; b.sigma += g.sigma;
            b.old_peak = std::max(b.old_peak, os);
            b.new_peak = std::max(b.new_peak, ns);
            b.old_rej += orej ? 1 : 0; b.new_rej += nrej ? 1 : 0;
            b.changed += (orej != nrej) ? 1 : 0;
            ++b.px;
        }
    for (int i = 0; i < NB; ++i) {
        const double d = (double)std::max(1L, r.b[i].px);
        r.b[i].old_g /= d; r.b[i].new_g /= d; r.b[i].sigma /= d;
    }
    return r;
}

static const char* kBname[NB] = {"LOW  snr<=lo", "MID  lo..hi ", "HIGH snr>=hi"};

static void report(const char* label, const Field& f, const NoiseModel& nm,
                   float Emag, const GeomGradParams& gp) {
    const Result r = run(f, nm, 2.f, Emag, gp);
    std::printf("  %s   Emag %.3f\n", label, Emag);
    for (int i = 0; i < NB; ++i) {
        const Bstat& b = r.b[i];
        if (b.px == 0) continue;
        std::printf("      %s  %6.1f%% of px | gmag %.5f -> %.5f (%+5.1f%%) | "
                    "reject %5.2f%% -> %5.2f%% | flipped %5.3f%%\n",
                    kBname[i], 100.0 * (double)b.px / (double)((f.h - 4) * (f.w - 4)),
                    b.old_g, b.new_g,
                    b.old_g > 0 ? 100.0 * (b.new_g - b.old_g) / b.old_g : 0.0,
                    100.0 * (double)b.old_rej / (double)b.px,
                    100.0 * (double)b.new_rej / (double)b.px,
                    100.0 * (double)b.changed / (double)b.px);
    }
}

static int g_fail = 0;

int main() {
    GeomGradParams gp;   // the shipped snr_lo / snr_hi
    std::printf("threshold %.4f (unchanged), sc = 2 (Bayer guide), snr_lo %.1f snr_hi %.1f\n",
                kThreshold, gp.snr_lo, gp.snr_hi);

    // ---- A. bright, low noise -------------------------------------------
    // A real edge and a textured ramp at daylight brightness. Emag swept across
    // the value that straddles the threshold, which is where a decision can flip.
    std::printf("\nA. BRIGHT / LOW NOISE  (%s)\n", kBright.name);
    {
        const Field edge = make_means(160, 160, kBright, 1, 0.35f, 0.25f, 11);
        for (float e : {0.005f, 0.02f, 0.05f, 0.2f}) report("bright step edge", edge, kBright, e, gp);
        const Field ramp = make_means(160, 160, kBright, 2, 0.30f, 0.40f, 12);
        for (float e : {0.02f, 0.2f}) report("bright ramp (fine texture)", ramp, kBright, e, gp);
        const Field flat = make_means(160, 160, kBright, 0, 0.30f, 0.f, 13);
        for (float e : {0.2f, 1.0f}) report("bright flat", flat, kBright, e, gp);
    }

    // ---- B. low light with real edges ------------------------------------
    std::printf("\nB. LOW LIGHT + REAL EDGE  (%s)\n", kDark.name);
    {
        const Field edge = make_means(160, 160, kDark, 1, 0.04f, 0.10f, 21);
        for (float e : {0.02f, 0.05f, 0.2f}) report("dark step edge", edge, kDark, e, gp);
    }

    // ---- C. low light, flat and noisy ------------------------------------
    std::printf("\nC. LOW LIGHT / FLAT NOISY  (%s)\n", kDark.name);
    {
        const Field flat = make_means(160, 160, kDark, 0, 0.04f, 0.f, 31);
        for (float e : {0.2f, 0.5f, 1.0f, 2.0f}) report("dark flat (noise only)", flat, kDark, e, gp);
    }

    // ---- acceptance ------------------------------------------------------
    std::printf("\n-- acceptance --\n");
    {
        // A. The load-bearing invariant: a pixel whose sharp gradient is already
        // trustworthy (snr >= snr_hi) must come out bit-identical, at any Emag.
        // Those are the only pixels that reject in a bright scene.
        const Field be = make_means(160, 160, kBright, 1, 0.35f, 0.25f, 41);
        const Field br = make_means(160, 160, kBright, 2, 0.30f, 0.40f, 42);
        long hi_flips = 0, hi_px = 0, all_flips = 0, all_px = 0;
        for (float e : {0.002f, 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.5f, 1.0f})
            for (const Field* f : {&be, &br}) {
                const Result r = run(*f, kBright, 2.f, e, gp);
                hi_flips += r.b[HIGH].changed; hi_px += r.b[HIGH].px;
                for (int i = 0; i < NB; ++i) { all_flips += r.b[i].changed; all_px += r.b[i].px; }
            }
        const double pct = 100.0 * (double)all_flips / (double)std::max(1L, all_px);
        const bool ok = (hi_flips == 0) && pct < 0.5;
        std::printf("  A  bright: HIGH-snr flips %ld / %ld (must be 0); all flips %.3f%%  %s\n",
                    hi_flips, hi_px, pct, ok ? "OK" : "FAIL");
        if (!ok) ++g_fail;
    }
    {
        // C. Noise-only dark region: the noise-driven gradient must come down.
        const Field flat = make_means(160, 160, kDark, 0, 0.04f, 0.f, 43);
        const Result r = run(flat, kDark, 2.f, 1.0f, gp);
        double og = 0, ng = 0; long px = 0;
        for (int i = 0; i < NB; ++i) { og += r.b[i].old_g * (double)r.b[i].px;
                                      ng += r.b[i].new_g * (double)r.b[i].px;
                                      px += r.b[i].px; }
        og /= (double)px; ng /= (double)px;
        const double drop = 100.0 * (og - ng) / std::max(1e-12, og);
        const bool ok = ng < og;
        std::printf("  C  dark flat: noise gmag %.5f -> %.5f (-%.1f%%)  %s\n",
                    og, ng, drop, ok ? "OK" : "FAIL");
        if (!ok) ++g_fail;
        std::printf("     NOTE: independent samples would predict -39%%. means is already a"
                    "\n           3x3 local mean (local_stats_3x3), so neighbouring samples are"
                    "\n           correlated and a further 3x3 Sobel recovers far less.\n");
    }
    {
        // B. A real dark edge must keep its peak score.
        const Field edge = make_means(160, 160, kDark, 1, 0.04f, 0.10f, 44);
        const Result r = run(edge, kDark, 2.f, 0.05f, gp);
        double op = 0, np = 0;
        for (int i = 0; i < NB; ++i) { op = std::max(op, r.b[i].old_peak);
                                      np = std::max(np, r.b[i].new_peak); }
        const double keep = 100.0 * np / std::max(1e-12, op);
        const bool ok = keep > 90.0;
        std::printf("  B  dark edge: peak score kept %.1f%% of old  %s\n", keep, ok ? "OK" : "FAIL");
        if (!ok) ++g_fail;
    }

    std::printf("\n%s\n", g_fail ? "FAILURES" : "all acceptance criteria met");
    return g_fail ? 1 : 0;
}
