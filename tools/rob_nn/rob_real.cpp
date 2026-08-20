// Training-set generator for the learned robustness mask, built from REAL
// bursts with only the FLOW corrupted.
//
// Why this replaces rob_dataset.cpp
// ---------------------------------
// rob_dataset.cpp synthesises the comparison frame: it warps a real reference
// by a known rigid transform, resampling each CFA phase, and adds noise drawn
// from the DNG model. That makes the true flow known analytically, which is
// what a label needs -- but it also means every comparison frame the network
// ever sees was produced by that renderer. The network learned the renderer's
// statistics (its interpolation blur, its uncorrelated modelled noise) rather
// than the sensor's, and generalised badly: on the burst it was trained on it
// did well, on unseen bursts it merged sky into mountain.
//
// This generator keeps the imagery entirely real -- both frames are captured
// raws, with their real noise, real optics, real parallax -- and corrupts only
// the flow field, which is the one quantity we are allowed to change without
// touching a pixel.
//
//   1. run the real aligner on a real (reference, comparison) pair. Call the
//      result flow_base.
//   2. verify flow_base per tile PHOTOMETRICALLY (below) and drop the tiles it
//      does not vouch for. flow_base is ground truth only where it is right.
//   3. corrupt flow_base into flow_est with a deliberately chosen failure
//      pattern and magnitude.
//   4. label harm(p) = |comp(p + flow_est) - comp(p + flow_base)|, both samples
//      taken from the SAME captured comparison frame.
//
// Step 4 is exact. When flow_est == flow_base the two samples are literally the
// same fetch, so harm is 0 and the ideal R is 1.0 -- the label reaches the top
// of the range, which is what the mask must learn to emit and currently never
// does. It is also non-circular in the same way rob_dataset's was: comparing
// the fetch against the REFERENCE would punish aliasing, the frame-to-frame
// difference super-resolution exists to exploit. Comparing the fetch against
// the CORRECT fetch is zero however aliased the content.
//
// Why the validity mask is not optional
// -------------------------------------
// flow_base is ground truth only where the aligner actually succeeded. Measured
// on these three bursts (probe2), the median tile's photometric residual sits
// at 1.0x the expected noise sigma -- the aligner is excellent -- but the 90th
// percentile reaches 2.7x, so roughly a tenth of tiles are not to be trusted.
// At such a tile both label directions are poison: leaving it uncorrupted
// labels a genuinely bad fetch "safe", and corrupting it measures deviation
// from a wrong origin. Those tiles are therefore given weight 0 and excluded
// from training and evaluation rather than guessed at.
//
// Note what is NOT done: tiles that fail the check are not labelled harmful.
// That test is d^2/sigma^2 at raw resolution -- the analytic mask's own
// statistic -- and labelling from it would train the network to copy the
// analytic mask, which is exactly what it must not do.
//
// Why a global affine model is not used anywhere
// ----------------------------------------------
// The obvious shortcut -- fit a rigid/affine camera model to flow_base and call
// that the truth -- was measured and rejected. On these bursts the affine model
// is photometrically much WORSE than the raw per-tile flow (median residual
// 1.4-2.5x noise sigma against the flow's 1.0x), because handheld translation
// over a scene with depth produces real parallax that no global model
// describes. Its per-tile deviation runs to 5-20 px and is CORRECT. Feeding
// "deviation from the global model" to the network as evidence would have
// taught it a false alarm.
//
// What the label is FOR, now that the mask is multiplicative
// ----------------------------------------------------------
// The network no longer emits the mask. It emits a correction C in [0,1] and
// the merge uses R_analytic * C, so the quantity to supervise is not "is this
// pixel safe" -- the analytic mask already answers that correctly most of the
// time -- but "by how much is the analytic mask WRONG here, in the unsafe
// direction". Hence CH_RNORM: the record carries the analytic mask's own
// finished answer, and train_rob.py forms the target as the ratio
// R_ideal / R_normal clipped to 1. Where the closed form already rejects, the
// target is exactly 1 and the network is asked to do nothing at all.
//
// Output: 35 float32 channels per guide pixel -- 27 inputs (see
// build_robustness_nn_features in core/stages.h, which must agree exactly) and
// 8 analysis channels. Records are square crops; a per-record text sidecar
// carries burst/frame/pattern/crop/phase so evaluation can slice by failure
// type, by scene, by held-out region and by corrupted-versus-paired-clean
// without re-reading the pixels.
#include "stages.h"
#include "parallel.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace hhsr;

namespace {

// ---------------------------------------------------------------- sampling

// Bilinear sample of one CFA colour plane. Pixels of a given Bayer phase form a
// regular half-resolution lattice; sampling within that lattice keeps every
// fetch a true mosaic sample, which is what the merge itself does.
float sample_phase(const Image& raw, float sy, float sx, int oy, int ox) {
    const float fi = (sy - (float)oy) * 0.5f, fj = (sx - (float)ox) * 0.5f;
    const int i0 = (int)std::floor(fi), j0 = (int)std::floor(fj);
    const float ai = fi - (float)i0, aj = fj - (float)j0;
    const int imax = (raw.h - 1 - oy) / 2, jmax = (raw.w - 1 - ox) / 2;
    auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
    const int i1 = cl(i0 + 1, imax), j1 = cl(j0 + 1, jmax);
    const int ic = cl(i0, imax), jc = cl(j0, jmax);
    auto at = [&](int i, int j) { return raw.at(2 * i + oy, 2 * j + ox); };
    const float top = at(ic, jc) + (at(ic, j1) - at(ic, jc)) * aj;
    const float bot = at(i1, jc) + (at(i1, j1) - at(i1, jc)) * aj;
    return top + (bot - top) * ai;
}

// The same bilinear fetch, but keeping the footprint it used.
//
// The label needs the noise floor of the DIFFERENCE between two fetches, and
// that floor is not 2*sigma^2. Two bilinear samples of the same lattice have
// (a) variance Sum(w^2)*sigma^2 each, which is as low as 0.25*sigma^2 for a
// half-pixel offset, and (b) CORRELATED noise whenever their 2x2 footprints
// overlap -- which for a sub-pixel error is almost completely.
//
// Assuming a flat 2*sigma^2 was measured to destroy the labels outright: the
// 0.25-1 px and 1-4 px error bins came out with exactly zero harm at the 99th
// percentile, i.e. every real difference had been subtracted away, and the
// training set contained essentially no harmful pixels at all. Since the
// weights factorise over rows and columns, the exact covariance is cheap.
struct Tap {
    float v;              // the interpolated value
    int r[2], c[2];       // lattice rows/cols touched
    float wr[2], wc[2];   // their weights
};

Tap sample_tap(const Image& raw, float sy, float sx, int oy, int ox) {
    Tap t;
    const float fi = (sy - (float)oy) * 0.5f, fj = (sx - (float)ox) * 0.5f;
    const int i0 = (int)std::floor(fi), j0 = (int)std::floor(fj);
    const float ai = fi - (float)i0, aj = fj - (float)j0;
    const int imax = (raw.h - 1 - oy) / 2, jmax = (raw.w - 1 - ox) / 2;
    auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
    t.r[0] = cl(i0, imax); t.r[1] = cl(i0 + 1, imax);
    t.c[0] = cl(j0, jmax); t.c[1] = cl(j0 + 1, jmax);
    t.wr[0] = 1.f - ai; t.wr[1] = ai;
    t.wc[0] = 1.f - aj; t.wc[1] = aj;
    t.v = 0.f;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            t.v += t.wr[a] * t.wc[b] * raw.at(2 * t.r[a] + oy, 2 * t.c[b] + ox);
    return t;
}

// Sum of w_a * w_b over taps that land on the SAME lattice site. With a == b
// this is Sum(w^2), the variance-reduction factor of one interpolation; with
// a != b it is the covariance factor between two. Factorises into rows times
// columns because the bilinear weights do.
float tap_overlap(const Tap& a, const Tap& b) {
    float fr = 0.f, fc = 0.f;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            if (a.r[i] == b.r[j]) fr += a.wr[i] * b.wr[j];
            if (a.c[i] == b.c[j]) fc += a.wc[i] * b.wc[j];
        }
    return fr * fc;
}

// Sub-pixel registration error the aligner is allowed to have while still
// counting as ground truth, in raw pixels. Not a fudge factor: block matching
// resolves motion to a fraction of a pixel, and the merge is built to absorb
// exactly that much. What it must NOT absorb is a tile fetched from the wrong
// place, which is what the validity test is there to catch.
constexpr float kRegSlopPx = 0.5f;

// Sentinel for a warped sample that fell outside the comparison frame.
constexpr float kNoSample = -1e30f;

// True while (y,x) + displacement stays inside the frame with a margin. A fetch
// that leaves the sensor is clamped by sample_phase, so `got` and `want` can
// agree there for no better reason than both having run off the same edge --
// which would label a gross error harmless.
bool in_frame(const Image& im, float y, float x, float m = 2.f) {
    return y >= m && x >= m && y < (float)im.h - m && x < (float)im.w - m;
}

// Guide-resolution 3x3 mean/variance, matching robustness.cpp's local_stats_3x3.
void local_stats(const Image& g, Image& means, Image& vars, int nthreads) {
    means = Image(g.h, g.w, g.c);
    vars = Image(g.h, g.w, g.c);
    parallel_rows(g.h, nthreads, [&](int y) {
        for (int ch = 0; ch < g.c; ++ch)
            for (int x = 0; x < g.w; ++x) {
                float s = 0.f, s2 = 0.f;
                for (int i = -1; i <= 1; ++i) {
                    const int yy = std::min(std::max(y + i, 0), g.h - 1);
                    for (int j = -1; j <= 1; ++j) {
                        const int xx = std::min(std::max(x + j, 0), g.w - 1);
                        const float v = g.at(yy, xx, ch);
                        s += v; s2 += v * v;
                    }
                }
                const float m = s / 9.f;
                means.at(y, x, ch) = m;
                vars.at(y, x, ch) = std::max(s2 / 9.f - m * m, 0.f);
            }
    });
}

// ------------------------------------------------------- exposure emulation
//
// All three source bursts are bright daylight (ISO 50, 1/3800-1/6000 s, read
// straight off the DNG tags). The user's failure cases include LOW LIGHT WITH
// MOTION, and a network trained only on high-SNR frames has to extrapolate the
// one statistic the whole mask rests on -- the relation between a colour
// difference d and the noise sigma it should be judged against. So the light
// level is augmented rather than left as captured.
//
// This is NOT a return to synthesising imagery. The scene, the optics, the
// parallax and the sensor's own noise all stay exactly as captured; only the
// exposure is changed, and it is changed the way physics changes it.
//
// Emulating a 1/k exposure of the same scene:
//
//   signal    v  ->  v / k                       (k times fewer photons)
//   variance          alpha*(v/k) + beta         (what a real 1/k capture has)
//
// alpha is the per-electron shot term and beta the per-readout term, so
// NEITHER depends on exposure time -- a shorter exposure of the same scene
// obeys the same law at a lower signal. That is what makes this exact rather
// than a model: the augmented frame is a valid sample of the SAME alpha/beta
// the pipeline already reads from the DNG, so Config needs no adjustment and
// feature channel 12 (expected sigma at this brightness) stays consistent with
// the pixels by construction. Getting that wrong would teach the network a
// false d-versus-sigma relationship, which is the one thing it must get right.
//
// The frame already carries its own daylight noise, and scaling shrinks it:
// after v -> v/k the existing variance is (alpha*v + beta)/k^2. So the noise
// that must be ADDED is the difference,
//
//   delta = alpha*(v/k) + beta - (alpha*v + beta)/k^2
//
// which is non-negative for every k >= 1. Adding the full target variance
// instead -- the obvious mistake -- would leave the frame noisier than its own
// label claims.
//
// Nothing is clipped. Post-black-subtraction values are legitimately negative
// where noise crosses zero, and clamping them would truncate exactly the part
// of the distribution that low-light detail lives in and bias the local means
// the whole mask reads.
void emulate_exposure(Image& raw, float k, const Config& cfg, uint32_t seed) {
    if (k <= 1.0f) return;
    const float inv_k = 1.0f / k, inv_k2 = inv_k * inv_k;
    parallel_rows(raw.h, cfg.num_threads, [&](int y) {
        std::mt19937 rng(seed * 2654435761u + (uint32_t)y);
        std::normal_distribution<float> gauss(0.f, 1.f);
        for (int x = 0; x < raw.w; ++x) {
            // Per-CFA-site noise law. noise_alpha_ch/noise_beta_ch fold in the
            // guide's quad-averaging weight, which halves green -- correct for
            // a guide pixel, wrong for the raw site being drawn here.
            const int c = cfg.cfa.p[y & 1][x & 1];
            const float g = cfg.noise_wb_gain(c);
            const float a = cfg.alpha_dng[c] * g;
            const float b = cfg.beta_dng[c] * g * g;
            const float v = raw.at(y, x);
            const float vs = v * inv_k;
            const float have = (a * std::max(v, 0.f) + b) * inv_k2;
            const float want = a * std::max(vs, 0.f) + b;
            const float delta = std::max(want - have, 0.f);
            raw.at(y, x) = vs + gauss(rng) * std::sqrt(delta);
        }
    });
}

// ------------------------------------------------------------ flow helpers

struct TileGrid {
    int ny = 0, nx = 0, ts = 0;
    std::vector<float> dx, dy;
    float& X(int t, int s) { return dx[(size_t)t * nx + s]; }
    float& Y(int t, int s) { return dy[(size_t)t * nx + s]; }
    float X(int t, int s) const { return dx[(size_t)t * nx + s]; }
    float Y(int t, int s) const { return dy[(size_t)t * nx + s]; }
};

TileGrid to_grid(const FlowField& f, int ts) {
    TileGrid g; g.ny = f.ny; g.nx = f.nx; g.ts = ts;
    g.dx.resize((size_t)f.ny * f.nx);
    g.dy.resize((size_t)f.ny * f.nx);
    for (int t = 0; t < f.ny; ++t)
        for (int s = 0; s < f.nx; ++s) {
            g.X(t, s) = f.dx(t, s);
            g.Y(t, s) = f.dy(t, s);
        }
    return g;
}

void to_flow(const TileGrid& g, FlowField& f) {
    for (int t = 0; t < g.ny; ++t)
        for (int s = 0; s < g.nx; ++s) {
            f.dx(t, s) = g.X(t, s);
            f.dy(t, s) = g.Y(t, s);
        }
}

// ------------------------------------------------------- corruption patterns
//
// Independent random vectors per tile are the easy case and not the one that
// costs photographs. What breaks a merge in the field has structure: a whole
// region locked onto the wrong match, a smoothly wrong field from a bad
// coarse level, a boundary where two motions meet. The menu below is those
// patterns, each drawn at a magnitude from the same ladder so failure TYPE and
// failure SIZE vary independently.
enum Pattern {
    P_NONE = 0,      // uncorrupted: the positive class, real flow, real motion
    P_SINGLE,        // one isolated tile
    P_NEIGHBOURS,    // a few adjacent tiles, independent errors
    P_GROUP_SAME,    // a rectangular group locked onto one shared wrong motion
    P_SMOOTH,        // smooth low-frequency wrong field, everywhere
    P_ABRUPT,        // half-plane boundary: flow jumps across a line
    P_ROTATION,      // spurious rotation about a random centre
    P_TRANS_ROT,     // translation plus rotation
    P_MAGNITUDE,     // right direction, wrong length
    P_DIRECTION,     // right length, wrong direction
    P_EDGE_ALIGNED,  // error along the dominant local edge direction
    P_EDGE_PERP,     // error across it
    P_SIMILAR,       // error to a location whose content STATISTICS match
    P_GLOBAL,        // the whole field off by one constant vector
    P_COUNT
};

const char* pattern_name(int p) {
    static const char* n[P_COUNT] = {
        "none", "single", "neighbours", "group_same", "smooth", "abrupt",
        "rotation", "trans_rot", "magnitude", "direction", "edge_aligned",
        "edge_perp", "similar", "global"};
    return (p >= 0 && p < P_COUNT) ? n[p] : "?";
}

// The error ladder, in RAW pixels. The bottom rungs matter as much as the top:
// a 0.05 px error is genuinely harmless and the mask must say so decisively,
// which is the half of the range the current model never reaches.
struct Band { float lo, hi; };
const Band kBands[] = {
    {0.05f, 0.10f}, {0.10f, 0.25f}, {0.25f, 0.50f}, {0.50f, 1.00f},
    {1.00f, 2.00f}, {2.00f, 4.00f}, {4.00f, 16.f},  {16.f, 64.f},
    {64.f, 256.f},
};
constexpr int kNumBands = (int)(sizeof(kBands) / sizeof(kBands[0]));

} // namespace

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        std::printf("usage: rob_real out_prefix burst_id frame1.dng frame2.dng ...\n"
                    "  env: ROB_CROP (128) ROB_CROPS (4) ROB_VARIANTS (24)\n"
                    "       ROB_REFS (2)   ROB_SEED (0)  ROB_APPEND (0)\n"
                    "       ROB_HARD (path to a hard-pattern list, one id per line)\n");
        return 1;
    }
    const std::string out_prefix = argv[1];
    const int burst_id = std::atoi(argv[2]);
    std::vector<std::string> files;
    for (int i = 3; i < argc; ++i) files.push_back(argv[i]);
    if (files.size() < 2) { std::printf("need at least 2 frames\n"); return 1; }

    // Exposure ladder. 1 = the frames as captured; k > 1 emulates a 1/k
    // exposure, i.e. dimmer and correspondingly noisier. A sweep rather than
    // two clusters at the extremes, so the network interpolates across the
    // noise range instead of memorising two operating points.
    std::vector<float> gains;
    if (const char* v = std::getenv("ROB_GAINS")) {
        std::string t(v);
        size_t i = 0;
        while (i < t.size()) {
            size_t j = t.find(',', i);
            if (j == std::string::npos) j = t.size();
            gains.push_back((float)std::atof(t.substr(i, j - i).c_str()));
            i = j + 1;
        }
    }
    if (gains.empty()) gains.push_back(1.f);

    int crop = 128, crops_per_variant = 4, variants = 24, n_refs = 2;
    uint32_t seed0 = 0;
    bool append = false;
    if (const char* v = std::getenv("ROB_CROP")) crop = std::atoi(v);
    if (const char* v = std::getenv("ROB_CROPS")) crops_per_variant = std::atoi(v);
    if (const char* v = std::getenv("ROB_VARIANTS")) variants = std::atoi(v);
    if (const char* v = std::getenv("ROB_REFS")) n_refs = std::atoi(v);
    if (const char* v = std::getenv("ROB_SEED")) seed0 = (uint32_t)std::atoi(v);
    if (const char* v = std::getenv("ROB_APPEND")) append = std::atoi(v) != 0;

    // Hard-example mining feeds a pattern histogram back in: the ids listed are
    // drawn far more often than the rest, so a retrain concentrates on the
    // failures the previous round actually made instead of re-covering ground
    // the model already has.
    std::vector<int> hard_patterns;
    if (const char* hp = std::getenv("ROB_HARD")) {
        if (FILE* f = std::fopen(hp, "r")) {
            int id;
            while (std::fscanf(f, "%d", &id) == 1)
                if (id >= 0 && id < P_COUNT) hard_patterns.push_back(id);
            std::fclose(f);
            std::printf("hard-example list: %zu entries from %s\n",
                        hard_patterns.size(), hp);
        }
    }

    // Per-burst personality. The three bursts must not be three draws from one
    // distribution or the model can memorise them as one scene type; each gets
    // its own pattern emphasis, its own band emphasis and its own corruption
    // density, so "generalises across bursts" is a real question at eval time.
    const int bmod = burst_id % 3;
    const float band_bias[3] = {0.0f, 0.35f, -0.30f};   // toward larger/smaller errors
    const float density[3]   = {0.28f, 0.16f, 0.42f};   // fraction of tiles hit

    const int NCH = 35;   // 27 inputs + 8 analysis channels
    const int CH_HARM = 27, CH_RIDEAL = 28, CH_FERR = 29, CH_W = 30, CH_REP = 31;
    const int CH_OCC = 32, CH_DIS = 33;
    // R_normal: the analytic mask this correction MULTIPLIES, written by the
    // shared compute_robustness_analytic rather than re-derived, and taken
    // after Eq. 9's 5x5 minimum -- because that, not the pointwise Eq. 5 in
    // input channel 14, is the value the app multiplies at inference. The
    // training target is a ratio against it, so getting the wrong one here
    // would silently train the network to correct a mask nobody runs.
    const int CH_RNORM = 34;

    const std::string bin_path = out_prefix + ".f32";
    FILE* fout = std::fopen(bin_path.c_str(), append ? "ab" : "wb");
    if (!fout) { std::printf("cannot open %s\n", bin_path.c_str()); return 1; }
    FILE* fidx = std::fopen((out_prefix + ".idx").c_str(), append ? "a" : "w");
    if (!fidx) { std::printf("cannot open idx\n"); return 1; }

    Config cfg;
    cfg.scale = 2.f;
    cfg.bayer_mode = true;
    cfg.grey_method = GreyMethod::Decimate;
    cfg.r_t = 0.12f; cfg.r_s1 = 2.0f; cfg.r_s2 = 12.0f; cfg.r_Mt = 0.8f;
    cfg.num_threads = 0;
    cfg.alignment_tile_size = 16;

    // Record ids must be GLOBAL, not per-invocation. They restarted at 0 on
    // every append, so with three bursts appended in turn the same id appeared
    // three times, several index rows pointed at one data row, and the trainer
    // -- which reshapes the blob and indexes it by this id -- paired features
    // from one record with metadata from another. That silently scrambles the
    // per-burst and per-pattern breakouts AND leaks data across the train/eval
    // split, since one data row can then be reached from both sides.
    size_t total_records = 0;
    if (append) {
        if (FILE* xf = std::fopen((out_prefix + ".idx").c_str(), "r")) {
            int ch;
            while ((ch = std::fgetc(xf)) != EOF)
                if (ch == 10) ++total_records;
            std::fclose(xf);
            std::printf("appending after %zu existing records\n", total_records);
        }
    }
    int gh_all = 0, gw_all = 0;

    const int nf = (int)files.size();
    const int refs_to_use = std::min(n_refs, nf - 1);

    for (int ri = 0; ri < refs_to_use; ++ri) {
        // Spread the reference choices across the burst rather than always
        // using frame 0: the last frame of a handheld burst is a very
        // different alignment problem from the second.
        const int ref_idx = (ri * (nf - 1)) / std::max(1, refs_to_use - 1 ? refs_to_use - 1 : 1);
        const int refi = std::min(ref_idx, nf - 1);

        Image ref_native = load_raw_frame(files[refi], cfg, true, 0, 0);
        if (ref_native.w <= 0) { std::printf("  decode failed\n"); continue; }


        // ---- pass 1: alignment ground truth, measured at the NATIVE exposure
        //
        // The true correspondence between two frames is a property of the
        // scene, not of how much light was let in, so it is measured once on
        // the frames as captured and reused at every emulated exposure. Doing
        // it the other way round -- re-aligning each darkened copy -- was tried
        // first and is wrong twice over: the aligner has less signal to work
        // with, so flow_base degrades exactly where the labels most need to be
        // right, and the photometric validity test silently stops working,
        // because it divides by a sigma that grew with the noise. Measured, it
        // passed 75.8% of tiles at native exposure and 99.4% at 1/16 -- it was
        // not verifying anything any more, it was waving everything through.
        //
        // So truth is established once, at the best signal-to-noise the data
        // has, and only the FEATURES are recomputed from the darkened frames.
        Config wn = cfg;
        wn.burst_frame_count = nf;
        tune_config_snr(ref_native, wn);
        wn.r_t = cfg.r_t; wn.r_s1 = cfg.r_s1; wn.r_s2 = cfg.r_s2;
        const int ts_truth = wn.bm_tile_sizes.empty() ? 16 : wn.bm_tile_sizes[0];
        Image refg_n = compute_grey(ref_native, wn.bayer_mode, wn.grey_method);
        Pyramid refp_n = build_pyramid(refg_n, wn.bm_factors);

        // ---- scene motion and occlusion -----------------------------------
        //
        // The corruption label answers exactly one question: what does THIS
        // flow error cost? It is measured comp-versus-comp, so at zero
        // corruption it is identically zero -- including over a walking person,
        // where the baseline flow is already wrong and merging would ghost.
        // Trained naively on a burst with independent subject motion, the
        // network would learn that walking people are safe to merge, which is
        // the opposite of what it must learn.
        //
        // That is a genuinely different failure -- the correspondence is
        // invalid however good the flow is -- so it gets its own label
        // component rather than being folded into the harm measure.
        //
        // Detected by MULTI-FRAME AGREEMENT. For reference pixel p, warp every
        // comparison frame into the reference and take d_n(p) = warped_n(p) -
        // ref(p). Then:
        //
        //   * a frame whose d_n is an OUTLIER against the other frames' median
        //     is occluded in that frame -- the background vanished behind the
        //     subject there, in that frame only.
        //   * a pixel where EVERY frame's d agrees and is large is one where
        //     the REFERENCE is the odd view: the subject was there in the
        //     reference and has moved away in all the others. Merging any
        //     comparison frame there is equally wrong. This is the
        //     disocclusion case, and it is the one that ghosts a walking
        //     person, because the aligner matched the background under it.
        //
        // Why this is not the circularity the harm label avoids. A plain
        // ref-versus-comp difference punishes ordinary aliasing and sub-pixel
        // sampling -- the very signal super-resolution exists to exploit --
        // and would reject texture. Subtracting the across-frame median
        // removes whatever is systematic between the reference and the
        // comparisons, and the gradient allowance below absorbs the rest, so
        // only a frame that disagrees with the CONSENSUS is flagged. It is
        // also privileged information: the consensus comes from frames the
        // network never sees at inference, so it is supervision rather than a
        // statistic the model could have computed for itself.
        //
        // Needs at least three comparison frames for the median to mean
        // anything; with fewer the scores stay zero and the burst simply
        // contributes no occlusion supervision (burst6 has two).
        std::vector<std::vector<float>> warped_g((size_t)nf);
        std::vector<float> occ_score, dis_score;   // per comp, filled below
        std::vector<std::vector<float>> occ_map((size_t)nf), dis_map((size_t)nf);
        std::vector<FlowField> flow_truth((size_t)nf);
        std::vector<TileGrid> base_truth((size_t)nf);
        std::vector<std::vector<float>> valid_truth((size_t)nf);
        for (int ci = 0; ci < nf; ++ci) {
            if (ci == refi) continue;
            Image cn = load_raw_frame(files[ci], cfg, true, 0, 0);
            if (cn.w <= 0) continue;
            Image cgn = compute_grey(cn, wn.bayer_mode, wn.grey_method);
            FlowField f = align(refp_n, refg_n, cgn, wn, ts_truth, 0.f, 0.f, 0.f);
            f = flow_to_raw_tile_grid(f, cn.h, cn.w, cgn.h, cgn.w, ts_truth,
                                      wn.r_Mt, wn.num_threads,
                                      wn.grey_tile_size(ts_truth));
            const TileGrid b = to_grid(f, ts_truth);
            const int TY = b.ny, TX = b.nx;
            std::vector<float> tv((size_t)TY * TX, 0.f);
            parallel_rows(TY, wn.num_threads, [&](int t) {
                for (int s = 0; s < TX; ++s) {
                    const float fx = b.X(t, s), fy = b.Y(t, s);
                    double sd = 0, sn = 0; int n = 0; bool oob = false;
                    for (int i = 1; i < ts_truth && !oob; i += 2)
                        for (int j = 1; j < ts_truth; j += 2) {
                            const int py = t * ts_truth + i, px = s * ts_truth + j;
                            if (py < 2 || px < 2 ||
                                py >= ref_native.h - 2 || px >= ref_native.w - 2) continue;
                            if (!in_frame(cn, (float)py + fy, (float)px + fx, 3.f)) { oob = true; break; }
                            const float rv = ref_native.at(py, px);
                            sd += std::fabs(sample_phase(cn, (float)py + fy, (float)px + fx,
                                                         py & 1, px & 1) - rv);
                            const int cc = wn.cfa.p[py & 1][px & 1];
                            const float wg = wn.noise_wb_gain(cc);
                            const float nsg = std::sqrt(std::max(
                                wn.alpha_dng[cc] * wg * std::max(rv, 0.f)
                                + wn.beta_dng[cc] * wg * wg, 0.f));
                            // Allow for sub-pixel registration slop as well as
                            // noise. A tile can be correctly aligned to within
                            // a fraction of a pixel and still leave a residual
                            // proportional to its own gradient, so a pure
                            // noise-scaled test is really a TEXTURE test: it
                            // keeps flat tiles and throws away detailed ones.
                            // Measured before this term, the tiles it dropped
                            // were 2.6x more textured than the ones it kept
                            // (local std 2.19 vs 0.84 sigma) and 80% of the
                            // dropped pixels were the structured ones -- so the
                            // training set was being filtered down to exactly
                            // the content on which the mask has nothing to
                            // decide. Same-phase neighbours are 2 raw px apart,
                            // hence the /4 for a per-pixel derivative.
                            const float gx = 0.25f * (ref_native.at(py, px + 2)
                                                      - ref_native.at(py, px - 2));
                            const float gy = 0.25f * (ref_native.at(py + 2, px)
                                                      - ref_native.at(py - 2, px));
                            const float slop = kRegSlopPx * std::sqrt(gx * gx + gy * gy);
                            sn += std::sqrt(nsg * nsg + slop * slop);
                            ++n;
                        }
                    // Two correctly aligned frames still differ by their own
                    // independent noise: E|a-b| = sigma*sqrt(2)*sqrt(2/pi) =
                    // 1.13*sigma, so a perfect tile scores 1.13, not 1.0. The
                    // cut at 1.6 therefore keeps everything within about 1.4x
                    // of ideal and drops the tail the aligner cannot vouch for.
                    tv[(size_t)t * TX + s] = (!oob && n >= 8 && sn > 0 && sd / sn < 1.6) ? 1.f : 0.f;
                }
            });
            const double vf = std::accumulate(tv.begin(), tv.end(), 0.0) / (double)tv.size();
            std::printf("[burst %d ref %d comp %d] tiles %dx%d  trustworthy %.1f%% "
                        "(native exposure)\n", burst_id, refi, ci, TX, TY, 100.0 * vf);
            // The comparison frame's guide, warped into the reference by its
            // own flow. Green only: it carries the best signal-to-noise of the
            // three and one plane is enough to spot content that is not there.
            {
                Image cg2 = compute_guide(cn, wn);
                const int GH2 = cg2.h, GW2 = cg2.w;
                std::vector<float> W((size_t)GH2 * GW2, 0.f);
                parallel_rows(GH2, wn.num_threads, [&](int gy) {
                    for (int gx = 0; gx < GW2; ++gx) {
                        const int t = std::min(b.ny - 1, std::max(0, (2 * gy) / ts_truth));
                        const int u = std::min(b.nx - 1, std::max(0, (2 * gx) / ts_truth));
                        const float sy = (float)gy + 0.5f * b.Y(t, u);
                        const float sx = (float)gx + 0.5f * b.X(t, u);
                        const int y0i = (int)std::floor(sy), x0i = (int)std::floor(sx);
                        const float ay = sy - (float)y0i, ax = sx - (float)x0i;
                        auto cl2 = [](int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); };
                        const int ya = cl2(y0i, GH2), yb = cl2(y0i + 1, GH2);
                        const int xa = cl2(x0i, GW2), xb = cl2(x0i + 1, GW2);
                        // A warp that leaves the frame is CLAMPED by the
                        // index arithmetic, which silently returns edge
                        // content and reads as a colossal disagreement. On a
                        // burst with large motion that painted whole bands of
                        // the frame edge as "occluded". Mark it instead, and
                        // keep it out of the consensus entirely.
                        if (sy < 1.f || sx < 1.f || sy >= (float)GH2 - 1 ||
                            sx >= (float)GW2 - 1) {
                            W[(size_t)gy * GW2 + gx] = kNoSample;
                            continue;
                        }
                        const float t0 = cg2.at(ya, xa, 1) + (cg2.at(ya, xb, 1) - cg2.at(ya, xa, 1)) * ax;
                        const float t1 = cg2.at(yb, xa, 1) + (cg2.at(yb, xb, 1) - cg2.at(yb, xa, 1)) * ax;
                        W[(size_t)gy * GW2 + gx] = t0 + (t1 - t0) * ay;
                    }
                });
                warped_g[(size_t)ci] = std::move(W);
            }
            flow_truth[(size_t)ci] = std::move(f);
            base_truth[(size_t)ci] = std::move(b);
            valid_truth[(size_t)ci] = std::move(tv);
        }

        // ---- consensus across the warped frames -----------------------------
        {
            Image refg = compute_guide(ref_native, wn);
            const int GH2 = refg.h, GW2 = refg.w;
            std::vector<int> have;
            for (int ci = 0; ci < nf; ++ci)
                if (!warped_g[(size_t)ci].empty()) have.push_back(ci);
            const bool enough = have.size() >= 3;
            for (int ci : have) {
                occ_map[(size_t)ci].assign((size_t)GH2 * GW2, 0.f);
                dis_map[(size_t)ci].assign((size_t)GH2 * GW2, 0.f);
            }
            if (!enough) {
                std::printf("[burst %d ref %d] only %zu comparison frames -- no "
                            "occlusion supervision from this reference\n",
                            burst_id, refi, have.size());
            } else {
                const float ga = wn.noise_alpha_robustness(), gb = wn.noise_beta_robustness();
                parallel_rows(GH2, wn.num_threads, [&](int gy) {
                    std::vector<float> dv;
                    for (int gx = 0; gx < GW2; ++gx) {
                        const size_t q = (size_t)gy * GW2 + gx;
                        const float rv = refg.at(gy, gx, 1);
                        dv.clear();
                        int nseen = 0;
                        for (int ci : have) {
                            const float v = warped_g[(size_t)ci][q];
                            dv.push_back(v == kNoSample ? kNoSample : v - rv);
                            if (v != kNoSample) ++nseen;
                        }
                        if (nseen < 3) continue;      // scores stay 0
                        std::vector<float> srt;
                        for (float v : dv) if (v != kNoSample) srt.push_back(v);
                        std::nth_element(srt.begin(), srt.begin() + srt.size() / 2, srt.end());
                        const float med = srt[srt.size() / 2];
                        // Noise floor of a frame difference, plus what a
                        // half-raw-pixel registration error would produce on
                        // this much local gradient. Without the second term
                        // this is a texture detector, not an occlusion
                        // detector -- the same trap the validity test fell
                        // into.
                        const int yl = std::max(gy - 1, 0), yr = std::min(gy + 1, GH2 - 1);
                        const int xl = std::max(gx - 1, 0), xr = std::min(gx + 1, GW2 - 1);
                        const float gxg = 0.5f * (refg.at(gy, xr, 1) - refg.at(gy, xl, 1));
                        const float gyg = 0.5f * (refg.at(yr, gx, 1) - refg.at(yl, gx, 1));
                        const float slop = 0.25f * std::sqrt(gxg * gxg + gyg * gyg);
                        const float nsg = std::sqrt(std::max(ga * std::max(rv, 0.f) + gb, 0.f));
                        const float den = std::sqrt(2.f * nsg * nsg + slop * slop + 1e-20f);
                        // Only where there is real structure to ghost.
                        //
                        // A smooth bright sky has almost no gradient, so any
                        // flow error there fetches nearly the same value and
                        // the merge is unharmed -- but the sky is also where
                        // the aligner has nothing to lock onto, so its flow is
                        // arbitrary and its frame-to-frame differences are
                        // large and inconsistent. Ungated, that painted a
                        // third of a mountain burst as unmergeable, sky
                        // included. Independently moving objects -- the case
                        // this component exists for -- always carry structure.
                        const float gmag = std::sqrt(gxg * gxg + gyg * gyg);
                        if (gmag < 2.f * nsg) continue;
                        for (size_t k = 0; k < have.size(); ++k) {
                            if (dv[k] == kNoSample) continue;
                            occ_map[(size_t)have[k]][q] = std::fabs(dv[k] - med) / den;
                            dis_map[(size_t)have[k]][q] = std::fabs(med) / den;
                        }
                    }
                });
                double occ_hi = 0, dis_hi = 0; size_t np = (size_t)GH2 * GW2;
                for (size_t q = 0; q < np; ++q) {
                    if (occ_map[(size_t)have[0]][q] > 4.f) occ_hi += 1;
                    if (dis_map[(size_t)have[0]][q] > 4.f) dis_hi += 1;
                }
                // Visual validation hook. A bad occlusion detector would
                // inject systematically wrong labels into every sample from a
                // burst with subject motion, so the maps get looked at before
                // they are trained on.
                if (const char* dp = std::getenv("ROB_DUMP_OCC")) {
                    std::string base(dp);
                    auto dump = [&](const std::string& nm, const std::vector<float>& v) {
                        if (FILE* fo = std::fopen((base + nm).c_str(), "wb")) {
                            std::fwrite(v.data(), sizeof(float), v.size(), fo);
                            std::fclose(fo);
                        }
                    };
                    for (size_t k = 0; k < have.size() && k < 3; ++k) {
                        char nb[32];
                        std::snprintf(nb, sizeof(nb), "_occ%zu.f32", k);
                        dump(nb, occ_map[(size_t)have[k]]);
                        std::snprintf(nb, sizeof(nb), "_dis%zu.f32", k);
                        dump(nb, dis_map[(size_t)have[k]]);
                        std::snprintf(nb, sizeof(nb), "_warp%zu.f32", k);
                        dump(nb, warped_g[(size_t)have[k]]);
                    }
                    std::vector<float> rg((size_t)GH2 * GW2);
                    for (int y = 0; y < GH2; ++y)
                        for (int x = 0; x < GW2; ++x) rg[(size_t)y * GW2 + x] = refg.at(y, x, 1);
                    dump("_refg.f32", rg);
                    if (FILE* fo = std::fopen((base + "_occ.dims").c_str(), "w")) {
                        std::fprintf(fo, "h %d\nw %d\nn %zu\n", GH2, GW2,
                                     std::min<size_t>(have.size(), 3));
                        std::fclose(fo);
                    }
                    std::printf("  dumped occlusion maps to %s_*\n", base.c_str());
                }
                std::printf("[burst %d ref %d] consensus over %zu frames: "
                            "occluded %.2f%%, ref-anomalous %.2f%% (frame %d)\n",
                            burst_id, refi, have.size(), 100.0 * occ_hi / np,
                            100.0 * dis_hi / np, have[0]);
                // The warped guides existed only to form the consensus above
                // and are 12 MB each. Holding all seven through the whole
                // variant loop is 85 MB of a budget that is already the
                // binding constraint on this host.
                for (auto& v : warped_g) { std::vector<float>().swap(v); }
            }
        }

      // ---- pass 2: features at each emulated exposure -----------------------
      for (size_t gi = 0; gi < gains.size(); ++gi) {
        const float gain = gains[gi];
        std::printf("[burst %d] reference %s  exposure 1/%.3g\n",
                    burst_id, files[refi].c_str(), gain);
        Image ref = ref_native;
        emulate_exposure(ref, gain, cfg, seed0 + 7717u * (uint32_t)(refi + 1)
                                       + 131u * (uint32_t)gi);

        Config work = cfg;
        work.burst_frame_count = nf;
        tune_config_snr(ref, work);
        work.r_t = cfg.r_t; work.r_s1 = cfg.r_s1; work.r_s2 = cfg.r_s2;
        // The flow grid was built at the native exposure's tile size, so the
        // features must be indexed on that same grid whatever this exposure's
        // SNR tuning would have picked on its own.
        const int ts = ts_truth;

        Image ref_guide = compute_guide(ref, work);
        // The pipeline's own reference statistics, not a reimplementation of
        // them: init_robustness is what the app calls, and channel 17's cache
        // is filled by the same helper the app uses. A second implementation
        // here is exactly how training and inference drift apart.
        work.use_neural_robustness = true;
        RefStats rs = init_robustness(ref, work);
        ensure_robustness_nn_ref_hf(rs, work);
        if (rs.means.data.empty()) { std::printf("  ref stats empty\n"); continue; }
        const Image& ref_means = rs.means;

        const int gh = ref_guide.h, gw = ref_guide.w;
        const int oh = std::min(crop, gh), ow = std::min(crop, gw);
        if (gh_all && (gh_all != oh || gw_all != ow)) {
            std::printf("  record shape changed; skipping\n");
            continue;
        }
        gh_all = oh; gw_all = ow;

        // Per-tile dominant edge orientation from the reference, for the
        // along-edge / across-edge corruption patterns. A displacement along a
        // straight edge is nearly invisible photometrically and is exactly the
        // case that survives the analytic test.
        const int tny_hint = (ref.h + ts - 1) / ts, tnx_hint = (ref.w + ts - 1) / ts;
        std::vector<float> edge_ang((size_t)tny_hint * tnx_hint, 0.f);
        std::vector<float> edge_str((size_t)tny_hint * tnx_hint, 0.f);
        // Per-tile self-similarity: how strongly the tile matches itself under a
        // small shift. High = repetitive texture (brickwork, foliage, railings),
        // where a wrong match locks onto the neighbouring period and the
        // photometry agrees. Analysis channel, used to break out results.
        std::vector<float> tile_rep((size_t)tny_hint * tnx_hint, 0.f);
        parallel_rows(tny_hint, work.num_threads, [&](int t) {
            for (int s = 0; s < tnx_hint; ++s) {
                double gxx = 0, gyy = 0, gxy = 0;
                const int gy0 = (t * ts) / 2, gx0 = (s * ts) / 2, gts = ts / 2;
                for (int i = 1; i < gts - 1; ++i)
                    for (int j = 1; j < gts - 1; ++j) {
                        const int y = std::min(gy0 + i, gh - 2), x = std::min(gx0 + j, gw - 2);
                        if (y < 1 || x < 1) continue;
                        float lx = 0.f, ly = 0.f;
                        for (int c = 0; c < 3; ++c) {
                            lx += ref_guide.at(y, x + 1, c) - ref_guide.at(y, x - 1, c);
                            ly += ref_guide.at(y + 1, x, c) - ref_guide.at(y - 1, x, c);
                        }
                        gxx += (double)lx * lx; gyy += (double)ly * ly; gxy += (double)lx * ly;
                    }
                const double th = 0.5 * std::atan2(2.0 * gxy, gxx - gyy);
                const double tr = gxx + gyy;
                const double det = gxx * gyy - gxy * gxy;
                const double coh = (tr > 1e-12) ? std::sqrt(std::max(tr * tr - 4 * det, 0.0)) / tr : 0.0;
                edge_ang[(size_t)t * tnx_hint + s] = (float)th;
                edge_str[(size_t)t * tnx_hint + s] = (float)coh;

                float best = 0.f;
                for (int k = 0; k < 4; ++k) {
                    static const int dyv[4] = {0, 3, 3, -3}, dxv[4] = {3, 0, 3, 3};
                    double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0; int n = 0;
                    for (int i = 0; i < gts; ++i)
                        for (int j = 0; j < gts; ++j) {
                            const int y = gy0 + i, x = gx0 + j;
                            const int y2 = y + dyv[k], x2 = x + dxv[k];
                            if (y < 0 || x < 0 || y >= gh || x >= gw) continue;
                            if (y2 < 0 || x2 < 0 || y2 >= gh || x2 >= gw) continue;
                            const float a = ref_guide.at(y, x, 1), b2 = ref_guide.at(y2, x2, 1);
                            sa += a; sb += b2; sab += (double)a * b2;
                            saa += (double)a * a; sbb += (double)b2 * b2; ++n;
                        }
                    if (n < 8) continue;
                    const double ma = sa / n, mb = sb / n;
                    const double ca = saa / n - ma * ma, cbb = sbb / n - mb * mb;
                    const double cab = sab / n - ma * mb;
                    const double dd = std::sqrt(std::max(ca, 0.0) * std::max(cbb, 0.0));
                    if (dd > 1e-6) best = std::max(best, (float)(cab / dd));
                }
                tile_rep[(size_t)t * tnx_hint + s] = best;
            }
        });

        for (int ci = 0; ci < nf; ++ci) {
            if (ci == refi || valid_truth[(size_t)ci].empty()) continue;
            Image comp = load_raw_frame(files[ci], cfg, true, 0, 0);
            if (comp.w <= 0) { std::printf("  decode failed %s\n", files[ci].c_str()); continue; }
            // Same light level as the reference, independent noise realisation
            // -- which is what a second exposure of the same scene is.
            emulate_exposure(comp, gain, cfg, seed0 + 991u * (uint32_t)(ci + 1)
                                            + 131u * (uint32_t)gi + 5u);

            // Truth from pass 1. The geometry does not change with exposure.
            const FlowField& flow0 = flow_truth[(size_t)ci];
            const TileGrid& base = base_truth[(size_t)ci];
            const std::vector<float>& tile_valid = valid_truth[(size_t)ci];
            const int TNY = base.ny, TNX = base.nx;

            Image comp_guide = compute_guide(comp, work);
            Image comp_means, comp_vars;
            local_stats(comp_guide, comp_means, comp_vars, work.num_threads);
            // Feature channels 20-26 and the match-quality search both read
            // the UNSMOOTHED guide luma, via the same shared helper the app
            // uses. comp_means is a 3x3 box and cannot substitute: the box is
            // precisely what erases the shifted edge these channels exist to
            // see.
            Image comp_luma = guide_luma(comp_guide);

            for (int vi = 0; vi < variants; ++vi) {
                const uint32_t seed = seed0 + (uint32_t)(burst_id * 1000003 + refi * 10007
                                                       + ci * 101 + vi) * 2654435761u;
                std::mt19937 rng(seed);
                std::uniform_real_distribution<float> u01(0.f, 1.f);
                std::uniform_real_distribution<float> u11(-1.f, 1.f);

                int pat;
                if (vi == 0 || (vi % 4) == 2) {
                    // Every (ref, comp) pair contributes one completely
                    // uncorrupted variant. This is the whole point of the new
                    // strategy: real imagery, real handheld motion including
                    // real camera ROTATION (measured at 0.25-2.0 deg on these
                    // bursts), labelled R = 1.0 exactly. Without a healthy
                    // supply of these the model has no reason to ever emit the
                    // top of the range, and rotation keeps reading as suspect.
                    pat = P_NONE;
                } else if (!hard_patterns.empty() && u01(rng) < 0.6f) {
                    pat = hard_patterns[rng() % hard_patterns.size()];
                } else {
                    // Bias the menu per burst so the three are not three draws
                    // from one distribution.
                    pat = 1 + (int)(rng() % (P_COUNT - 1));
                    if ((int)(rng() % 3) == bmod) pat = 1 + (int)(rng() % (P_COUNT - 1));
                }

                int band = (int)(rng() % kNumBands);
                {   // per-burst push up or down the ladder, plus an
                    // exposure-dependent push. In dim light a small
                    // misalignment genuinely causes no measurable damage --
                    // the noise is larger than the difference -- so at 1/16
                    // exposure the sub-pixel rungs produce almost no harmful
                    // pixels (measured: 0.02% against 1.54% at native). That
                    // is a real information limit, not something to train
                    // around, but it does mean the low-light samples must be
                    // drawn from the part of the ladder where damage IS
                    // detectable, or they teach nothing about rejection at all.
                    const float b = (float)band + band_bias[bmod] * 3.f + u11(rng)
                                  + std::log2(std::max(gain, 1.f));
                    band = std::min(kNumBands - 1, std::max(0, (int)std::lround(b)));
                }
                const float mag_lo = kBands[band].lo, mag_hi = kBands[band].hi;
                auto draw_mag = [&]() { return mag_lo + u01(rng) * (mag_hi - mag_lo); };

                TileGrid g = base;
                const float dens = density[bmod] * (0.4f + 1.6f * u01(rng));

                auto hit = [&](int t, int s, float ex, float ey) {
                    if (t < 0 || t >= TNY || s < 0 || s >= TNX) return;
                    g.X(t, s) += ex; g.Y(t, s) += ey;
                };

                switch (pat) {
                case P_NONE:
                    break;
                case P_SINGLE: {
                    const int n = 1 + (int)(dens * TNY * TNX * 0.20f);
                    for (int k = 0; k < n; ++k) {
                        const int t = (int)(rng() % TNY), s = (int)(rng() % TNX);
                        const float a = u01(rng) * 6.2831853f, m = draw_mag();
                        hit(t, s, m * std::cos(a), m * std::sin(a));
                    }
                    break;
                }
                case P_NEIGHBOURS: {
                    const int n = 1 + (int)(dens * TNY * TNX * 0.020f);
                    for (int k = 0; k < n; ++k) {
                        const int t0 = (int)(rng() % TNY), s0 = (int)(rng() % TNX);
                        const int r = 1 + (int)(rng() % 3);
                        for (int dt = -r; dt <= r; ++dt)
                            for (int ds = -r; ds <= r; ++ds) {
                                const float a = u01(rng) * 6.2831853f, m = draw_mag();
                                hit(t0 + dt, s0 + ds, m * std::cos(a), m * std::sin(a));
                            }
                    }
                    break;
                }
                case P_GROUP_SAME: {
                    // The realistic block failure: a whole region locks onto one
                    // wrong match, so the flow inside it is perfectly smooth and
                    // perfectly consistent -- every "is my neighbourhood
                    // coherent" test passes, and only the photometry can object.
                    const int n = 2 + (int)(rng() % 6);
                    for (int k = 0; k < n; ++k) {
                        const int hgt = 2 + (int)(rng() % 20), wid = 2 + (int)(rng() % 20);
                        const int t0 = (int)(rng() % std::max(1, TNY - hgt));
                        const int s0 = (int)(rng() % std::max(1, TNX - wid));
                        const float a = u01(rng) * 6.2831853f, m = draw_mag();
                        const float ex = m * std::cos(a), ey = m * std::sin(a);
                        for (int t = t0; t < t0 + hgt; ++t)
                            for (int s = s0; s < s0 + wid; ++s) hit(t, s, ex, ey);
                    }
                    break;
                }
                case P_SMOOTH: {
                    // A few low-frequency sinusoids: the signature of a coarse
                    // pyramid level that converged to the wrong basin and was
                    // then refined smoothly on top of its own error.
                    const int nh = 2 + (int)(rng() % 3);
                    std::vector<float> ax(nh), ay(nh), kx(nh), ky(nh), ph(nh);
                    for (int k = 0; k < nh; ++k) {
                        ax[k] = u11(rng); ay[k] = u11(rng);
                        kx[k] = u11(rng) * 6.f / std::max(1, TNX);
                        ky[k] = u11(rng) * 6.f / std::max(1, TNY);
                        ph[k] = u01(rng) * 6.2831853f;
                    }
                    const float m = draw_mag();
                    for (int t = 0; t < TNY; ++t)
                        for (int s = 0; s < TNX; ++s) {
                            float ex = 0.f, ey = 0.f;
                            for (int k = 0; k < nh; ++k) {
                                const float p = kx[k] * s + ky[k] * t + ph[k];
                                ex += ax[k] * std::sin(p);
                                ey += ay[k] * std::cos(p);
                            }
                            hit(t, s, m * ex / nh, m * ey / nh);
                        }
                    break;
                }
                case P_ABRUPT: {
                    // Two motions meeting at a line -- an occlusion boundary the
                    // aligner split the wrong way.
                    const float a = u01(rng) * 6.2831853f;
                    const float nxd = std::cos(a), nyd = std::sin(a);
                    const float c = (u01(rng) * TNX * nxd) + (u01(rng) * TNY * nyd);
                    const float b = u01(rng) * 6.2831853f, m = draw_mag();
                    const float ex = m * std::cos(b), ey = m * std::sin(b);
                    for (int t = 0; t < TNY; ++t)
                        for (int s = 0; s < TNX; ++s)
                            if (s * nxd + t * nyd > c) hit(t, s, ex, ey);
                    break;
                }
                case P_ROTATION:
                case P_TRANS_ROT: {
                    // A SPURIOUS rotation, i.e. one on top of the real motion.
                    // The uncorrupted variants already supply correct rotation;
                    // this supplies the wrong kind, so the network has to learn
                    // the difference from evidence rather than from "rotational
                    // structure means danger".
                    const float cy = u01(rng) * TNY, cx = u01(rng) * TNX;
                    const float rmax = std::sqrt((float)(TNY * TNY + TNX * TNX)) * 0.5f;
                    const float m = draw_mag();
                    const float th = (m / std::max(rmax * ts, 1.f)) * (u01(rng) < 0.5f ? -1.f : 1.f);
                    float gx = 0.f, gy = 0.f;
                    if (pat == P_TRANS_ROT) {
                        const float a = u01(rng) * 6.2831853f;
                        gx = m * 0.7f * std::cos(a); gy = m * 0.7f * std::sin(a);
                    }
                    for (int t = 0; t < TNY; ++t)
                        for (int s = 0; s < TNX; ++s) {
                            const float dy = (t - cy) * ts, dx = (s - cx) * ts;
                            hit(t, s, gx - th * dy, gy + th * dx);
                        }
                    break;
                }
                case P_MAGNITUDE:
                case P_DIRECTION: {
                    const int n = 1 + (int)(dens * TNY * TNX * 0.25f);
                    for (int k = 0; k < n; ++k) {
                        const int t = (int)(rng() % TNY), s = (int)(rng() % TNX);
                        const float fx = base.X(t, s), fy = base.Y(t, s);
                        const float len = std::sqrt(fx * fx + fy * fy);
                        if (len < 1e-3f) continue;
                        const float m = draw_mag();
                        if (pat == P_MAGNITUDE) {
                            const float sgn = (u01(rng) < 0.5f) ? -1.f : 1.f;
                            hit(t, s, sgn * m * fx / len, sgn * m * fy / len);
                        } else {
                            // Rotate the vector so its length is preserved and
                            // only its direction is wrong -- the error is m px.
                            const float dth = 2.f * std::asin(std::min(1.f, m / (2.f * len)))
                                            * (u01(rng) < 0.5f ? -1.f : 1.f);
                            const float c = std::cos(dth), sn2 = std::sin(dth);
                            hit(t, s, (c * fx - sn2 * fy) - fx, (sn2 * fx + c * fy) - fy);
                        }
                    }
                    break;
                }
                case P_EDGE_ALIGNED:
                case P_EDGE_PERP: {
                    const int n = 1 + (int)(dens * TNY * TNX * 0.25f);
                    for (int k = 0; k < n; ++k) {
                        const int t = (int)(rng() % TNY), s = (int)(rng() % TNX);
                        const size_t ei = (size_t)std::min(t, tny_hint - 1) * tnx_hint
                                        + std::min(s, tnx_hint - 1);
                        if (edge_str[ei] < 0.3f) continue;   // no dominant edge here
                        // edge_ang is the structure-tensor principal direction,
                        // i.e. the gradient (across-edge) direction.
                        float a = edge_ang[ei];
                        if (pat == P_EDGE_ALIGNED) a += 1.5707963f;
                        const float m = draw_mag() * (u01(rng) < 0.5f ? -1.f : 1.f);
                        hit(t, s, m * std::cos(a), m * std::sin(a));
                    }
                    break;
                }
                case P_SIMILAR: {
                    // The case that cost the user a photograph: a wrong match
                    // whose content has the SAME statistics as the right one --
                    // blue sky landing where mountain green was, flat shadow on
                    // flat shadow. d^2/sigma^2 is genuinely small there, so no
                    // amount of tuning the analytic mask reaches it, and a
                    // network trained only on random displacements never meets
                    // it either because a random displacement almost always
                    // lands somewhere that looks different.
                    //
                    // Constructed by searching, per tile, for the displacement
                    // in the ladder's band whose guide statistics best match the
                    // correct fetch. The result is a large, genuinely damaging
                    // geometric error that is photometrically camouflaged.
                    const int n = 1 + (int)(dens * TNY * TNX * 0.25f);
                    for (int k = 0; k < n; ++k) {
                        const int t = (int)(rng() % TNY), s = (int)(rng() % TNX);
                        const int cy = (t * ts + ts / 2) / 2, cx = (s * ts + ts / 2) / 2;
                        const int qy0 = std::min(std::max((int)(cy + 0.5f * base.Y(t, s)), 0), gh - 1);
                        const int qx0 = std::min(std::max((int)(cx + 0.5f * base.X(t, s)), 0), gw - 1);
                        float want[3];
                        for (int c = 0; c < 3; ++c) want[c] = comp_means.at(qy0, qx0, c);
                        float best_d = 1e30f, bex = 0.f, bey = 0.f;
                        for (int trial = 0; trial < 48; ++trial) {
                            const float a = u01(rng) * 6.2831853f;
                            const float m = draw_mag();
                            const float ex = m * std::cos(a), ey = m * std::sin(a);
                            const int qy = std::min(std::max((int)(qy0 + 0.5f * ey), 0), gh - 1);
                            const int qx = std::min(std::max((int)(qx0 + 0.5f * ex), 0), gw - 1);
                            float d = 0.f;
                            for (int c = 0; c < 3; ++c) {
                                const float dv = comp_means.at(qy, qx, c) - want[c];
                                d += dv * dv;
                            }
                            if (d < best_d) { best_d = d; bex = ex; bey = ey; }
                        }
                        hit(t, s, bex, bey);
                    }
                    break;
                }
                case P_GLOBAL: {
                    const float a = u01(rng) * 6.2831853f, m = draw_mag();
                    const float ex = m * std::cos(a), ey = m * std::sin(a);
                    for (int t = 0; t < TNY; ++t)
                        for (int s = 0; s < TNX; ++s) hit(t, s, ex, ey);
                    break;
                }
                default: break;
                }

                // The corrupted field, in the pipeline's own container, so the
                // feature builder below is fed exactly what the app would feed
                // it. motion_irregular is deliberately dropped: it was measured
                // on the uncorrupted field and would otherwise be a channel of
                // ground truth leaking into the inputs.
                // ------------------------------------------------ crops
                // Where did this variant's corruption actually land? A 128 px
                // crop covers 16x16 tiles, so uniformly random crops of a
                // pattern that touches a few hundred tiles out of 47k contain
                // no harmful pixel at all -- measured at under 3% harmful,
                // which trains a mask that has never seen the thing it exists
                // to catch. Most crops are therefore centred on a corrupted
                // tile. A third stay uniform so the set still contains large
                // genuinely-clean expanses and the value the model learns to
                // emit there is not an extrapolation.
                // Selected by MEASURED HARM, not by injected error. Those are
                // not the same set and the difference is most of the training
                // signal: corrupting a tile of clear sky by 100 px fetches an
                // identical pixel and is correctly labelled safe, so a crop
                // centred there teaches nothing about rejection. Before this
                // probe, 20% of pixels carried an injected error and only 0.48%
                // of them any actual damage.
                //
                // A cheap probe -- a few samples per corrupted tile, the same
                // harm measure the label uses -- finds the tiles where the
                // corruption genuinely landed on different content. Crops are
                // then centred there. This changes only WHICH pixels are
                // looked at, never what they are labelled.
                std::vector<int> hit_tiles;
                for (int t = 0; t < TNY; ++t)
                    for (int s = 0; s < TNX; ++s) {
                        if (tile_valid[(size_t)t * TNX + s] <= 0.f) continue;
                        const float ex = g.X(t, s) - base.X(t, s);
                        const float ey = g.Y(t, s) - base.Y(t, s);
                        if (ex * ex + ey * ey <= 4e-4f) continue;
                        int hits = 0;
                        for (int i = 3; i < ts && hits < 2; i += 5)
                            for (int j = 3; j < ts; j += 5) {
                                const int py = t * ts + i, px = s * ts + j;
                                if (py >= comp.h || px >= comp.w) continue;
                                if (!in_frame(comp, py + g.Y(t, s), px + g.X(t, s), 2.f) ||
                                    !in_frame(comp, py + base.Y(t, s), px + base.X(t, s), 2.f))
                                    continue;
                                const Tap G = sample_tap(comp, (float)py + g.Y(t, s),
                                                         (float)px + g.X(t, s), py & 1, px & 1);
                                const Tap W = sample_tap(comp, (float)py + base.Y(t, s),
                                                         (float)px + base.X(t, s), py & 1, px & 1);
                                const int cc = work.cfa.p[py & 1][px & 1];
                                const float wg = work.noise_wb_gain(cc);
                                const float A = work.alpha_dng[cc] * wg;
                                const float B = work.beta_dng[cc] * wg * wg;
                                const float vg = std::max(A * std::max(G.v, 0.f) + B, 0.f);
                                const float vw = std::max(A * std::max(W.v, 0.f) + B, 0.f);
                                const float nv = std::max(tap_overlap(G, G) * vg
                                                        + tap_overlap(W, W) * vw
                                                        - 2.f * tap_overlap(G, W) * std::min(vg, vw), 0.f);
                                const float d = G.v - W.v;
                                if (d * d > nv + vw) ++hits;   // damage beyond noise
                            }
                        if (hits >= 2) hit_tiles.push_back(t * TNX + s);
                    }

                // ---------------------------------------- paired phases
                //
                // Every corrupted record is emitted TWICE: once with the
                // corrupted flow and once with the baseline flow, over the SAME
                // crop of the SAME frame pair at the SAME exposure and the same
                // noise realisation. Only the correspondence differs.
                //
                // That pairing is what forces the distinction the model has to
                // make. These bursts contain real camera rotation (0.25-2.0 deg
                // measured), real curved motion and real parallax, and a set
                // that shows the network rotation only in its corrupted
                // variants teaches "rotation is dangerous" -- which is wrong,
                // and is a mask that throws away good frames. Holding
                // everything but the flow constant leaves the correspondence as
                // the only thing that can explain the label.
                //
                // The crop origins are chosen ONCE, from the corrupted variant
                // (where the harm probe knows which tiles matter), and reused
                // by both phases.
                std::vector<std::pair<int,int>> origins;
                for (int k = 0; k < crops_per_variant; ++k) {
                    int cy0 = 0, cx0 = 0;
                    // Retry for a crop the validity mask actually vouches for:
                    // a record that is 95% weight-0 costs a full record of disk
                    // and contributes almost nothing to the gradient.
                    float best_valid = -1.f; int best_y = 0, best_x = 0;
                    for (int attempt = 0; attempt < 8; ++attempt) {
                        if (!hit_tiles.empty() && u01(rng) < 0.67f) {
                            const int ti = hit_tiles[rng() % hit_tiles.size()];
                            const int tcy = (ti / TNX) * ts / 2, tcx = (ti % TNX) * ts / 2;
                            cy0 = tcy - oh / 2 + (int)(u11(rng) * oh * 0.25f);
                            cx0 = tcx - ow / 2 + (int)(u11(rng) * ow * 0.25f);
                        } else {
                            cy0 = (gh > oh) ? (int)(u01(rng) * (float)(gh - oh)) : 0;
                            cx0 = (gw > ow) ? (int)(u01(rng) * (float)(gw - ow)) : 0;
                        }
                        cy0 = std::min(std::max(cy0, 0), std::max(0, gh - oh));
                        cx0 = std::min(std::max(cx0, 0), std::max(0, gw - ow));
                        // Cheap proxy for the record's weight: the tile validity
                        // over the tiles the crop covers.
                        float acc = 0.f; int n = 0;
                        for (int t = (2 * cy0) / ts; t <= (2 * (cy0 + oh) - 1) / ts; ++t)
                            for (int s = (2 * cx0) / ts; s <= (2 * (cx0 + ow) - 1) / ts; ++s) {
                                if (t < 0 || t >= TNY || s < 0 || s >= TNX) continue;
                                acc += tile_valid[(size_t)t * TNX + s]; ++n;
                            }
                        const float vf = n ? acc / n : 0.f;
                        if (vf > best_valid) { best_valid = vf; best_y = cy0; best_x = cx0; }
                        if (vf > 0.6f) break;
                    }
                    cy0 = best_y; cx0 = best_x;
                    if (best_valid < 0.15f) continue;   // nothing usable here
                    origins.emplace_back(cy0, cx0);
                }
                if (origins.empty()) continue;

                // P_NONE is already the uncorrupted case, so its pair would be
                // a bit-identical duplicate: one phase only.
                const int nphase = (pat == P_NONE) ? 1 : 2;
                for (int phase = 0; phase < nphase; ++phase) {
                const TileGrid& gp = (phase == 0) ? g : base;
                FlowField fph = flow0;
                fph.motion_irregular.clear();
                to_flow(gp, fph);

                // Channels 18-19, measured on the flow BEING JUDGED. Measuring
                // it on the baseline would hand the network the answer and
                // score wonderfully while being useless in the app.
                const std::vector<f32> mq = measure_match_quality(rs.nn_luma, comp_luma,
                                                                  fph, ts, work);
                // The mask the correction multiplies, from the shared code the
                // app runs. Eq. 9's 5x5 minimum is inside it.
                Image rnorm = compute_robustness_analytic(comp, rs, fph, ts, work);
                const bool have_rn = (rnorm.h == gh && rnorm.w == gw &&
                                      rnorm.data.size() >= (size_t)gh * gw);
                if (!have_rn) { std::printf("  analytic mask empty\n"); continue; }

                for (size_t oi = 0; oi < origins.size(); ++oi) {
                    const int cy0 = origins[oi].first, cx0 = origins[oi].second;
                    // Inputs 0..26 come from the SHARED builder in
                    // core/robustness.cpp, not from a copy of it here. The
                    // previous generator reimplemented the layout and the two
                    // could drift silently; a single call cannot.
                    //
                    // Built over the CROP's rows, not the whole plane. At 27
                    // channels a full guide-resolution feature tensor is 329 MB
                    // and this host has under a gigabyte free -- it threw
                    // std::bad_alloc on the first variant. The builder is
                    // pointwise in y given the tile grid, so a row window
                    // starting at cy0 is bit-identical to the same rows of the
                    // full plane, which is what makes this a memory change and
                    // not a numerical one.
                    Image feat = build_robustness_nn_features(rs, comp_means, fph, ts,
                                                              work, cy0, /*raw_res=*/false,
                                                              /*rows=*/oh, &comp_luma, &mq);
                    if (feat.h != oh || feat.c != kRobustnessNnChannels) {
                        std::printf("  feature builder returned %dx%dx%d, expected %dx%dx%d\n",
                                    feat.h, feat.w, feat.c, oh, gw, kRobustnessNnChannels);
                        std::fclose(fout); std::fclose(fidx); return 1;
                    }
                    std::vector<float> rec((size_t)oh * ow * NCH, 0.f);

                    double sum_ferr = 0.0, sum_w = 0.0, sum_bad = 0.0, sum_r = 0.0;
                    parallel_rows(oh, work.num_threads, [&](int iy) {
                        const int gy = cy0 + iy;
                        for (int ix = 0; ix < ow; ++ix) {
                            const int gx = cx0 + ix;
                            const int ry = 2 * gy, rx = 2 * gx;
                            const int pty = std::min(TNY - 1, std::max(0, ry / ts));
                            const int ptx = std::min(TNX - 1, std::max(0, rx / ts));
                            // The merge fetches with the NEAREST tile's vector,
                            // so the label must be measured with that one --
                            // even though the feature channels interpolate.
                            const float fex = gp.X(pty, ptx), fey = gp.Y(pty, ptx);
                            const float ftx = base.X(pty, ptx), fty = base.Y(pty, ptx);

                            // HARM, measured as the photometric consequence of
                            // the corrupted correspondence -- never as the size
                            // of the injected error. The two come apart
                            // constantly and in both directions: 3 px of error
                            // across flat sky fetches an identical pixel and
                            // does no damage, while 0.3 px across a thin wire
                            // fetches something completely different. The
                            // injected magnitudes exist to spread the harm
                            // distribution, they are not the target.
                            //
                            // Both samples come from the SAME comparison frame.
                            // Taking one from the reference instead would make
                            // the label punish ordinary aliasing and genuine
                            // inter-frame difference -- exactly the signal
                            // super-resolution exists to exploit.
                            float harm2 = 0.f, nvar = 0.f;
                            bool oob = false;
                            for (int i = 0; i < 2 && !oob; ++i)
                                for (int j = 0; j < 2; ++j) {
                                    const int py = ry + i, px = rx + j;
                                    if (py >= comp.h || px >= comp.w) { oob = true; break; }
                                    if (!in_frame(comp, py + fey, px + fex, 2.f) ||
                                        !in_frame(comp, py + fty, px + ftx, 2.f)) { oob = true; break; }
                                    const Tap G = sample_tap(comp, (float)py + fey,
                                                             (float)px + fex, py & 1, px & 1);
                                    const Tap W = sample_tap(comp, (float)py + fty,
                                                             (float)px + ftx, py & 1, px & 1);
                                    const float d = G.v - W.v;
                                    harm2 += d * d;

                                    // Per-CFA-site noise law: G.v and W.v are
                                    // RAW samples, and noise_alpha() carries the
                                    // guide's quad-averaging weight, which
                                    // understates a single site (green by 2x).
                                    const int cc = work.cfa.p[py & 1][px & 1];
                                    const float wg = work.noise_wb_gain(cc);
                                    const float A = work.alpha_dng[cc] * wg;
                                    const float B = work.beta_dng[cc] * wg * wg;
                                    const float vg = std::max(A * std::max(G.v, 0.f) + B, 0.f);
                                    const float vw = std::max(A * std::max(W.v, 0.f) + B, 0.f);
                                    // Var(G - W) with the interpolation
                                    // footprints and their overlap accounted
                                    // for. When the flow is untouched the two
                                    // fetches are identical, every term cancels
                                    // and this is exactly 0 -- which is what
                                    // lets the label reach 1.0 and gives the
                                    // mask the top of its range to learn.
                                    nvar += tap_overlap(G, G) * vg
                                          + tap_overlap(W, W) * vw
                                          - 2.f * tap_overlap(G, W) * std::min(vg, vw);
                                }
                            harm2 *= 0.25f; nvar *= 0.25f;
                            nvar = std::max(nvar, 0.f);

                            // The excess of the measured difference over the
                            // noise the sensor would have produced anyway. A
                            // difference smaller than that noise is not damage:
                            // it is the same content read twice.
                            const float excess = std::max(harm2 - nvar, 0.f);
                            // Scale the excess by the sensor's own sigma, so
                            // "harmful" means "differs from the correct fetch by
                            // more than noise", at any light level.
                            const int cc0 = work.cfa.p[ry & 1][rx & 1];
                            const float wg0 = work.noise_wb_gain(cc0);
                            const float nsig = std::sqrt(std::max(
                                work.alpha_dng[cc0] * wg0 * std::max(ref_means.at(gy, gx, 1), 0.f)
                                + work.beta_dng[cc0] * wg0 * wg0, 1e-12f));
                            const float harm = std::sqrt(excess);

                            // Ideal R: full trust while the mis-fetch stays
                            // within the sensor's own noise, rolling off past
                            // it. Exactly 1.0 when the flow is untouched.
                            const float z = harm / std::max(nsig, 1e-9f);
                            const float r_harm = std::exp(-(z * z) / 4.0f);   // z0 = 2

                            // The second label component: content that is not
                            // in this comparison frame cannot be merged from
                            // it, however good the flow is. Both thresholds sit
                            // well above the noise so that ordinary aliasing
                            // and registration slop never trip them; the
                            // reference-anomaly test is the more conservative
                            // of the two because it fires on agreement rather
                            // than on an outlier.
                            float occ_s = 0.f, dis_s = 0.f;
                            if (!occ_map[(size_t)ci].empty()) {
                                const size_t q = (size_t)gy * gw + gx;
                                occ_s = occ_map[(size_t)ci][q];
                                dis_s = dis_map[(size_t)ci][q];
                            }
                            // 4 rather than 3 sigma on the outlier test. At 3
                            // it speckled through foliage on the motorway
                            // burst -- fast parallax through leaves genuinely
                            // does disagree frame to frame, but not so
                            // reliably that it should be labelled unmergeable
                            // pixel by pixel. The reference-anomaly test needs
                            // no such tightening: it lands cleanly on the
                            // moving cars and nowhere else.
                            const float eo = std::max(occ_s - 4.f, 0.f);
                            const float ed = std::max(dis_s - 4.f, 0.f);
                            const float r_scene = std::exp(-(eo * eo + ed * ed) / 4.0f);

                            // The merge is only safe if BOTH hold: the flow
                            // fetched the right place AND there is something
                            // there to fetch.
                            const float r_ideal = std::min(r_harm, r_scene);

                            // Weight 0 where flow_base is not vouched for, or
                            // where either fetch left the sensor.
                            float wt = tile_valid[(size_t)pty * TNX + ptx];
                            if (oob) wt = 0.f;
                            // Drop a halo at the crop edge: the network sees a
                            // 8-guide-pixel radius, and inside the halo part of
                            // that context is outside the record.
                            if (iy < kRobustnessNnHalo || ix < kRobustnessNnHalo ||
                                iy >= oh - kRobustnessNnHalo || ix >= ow - kRobustnessNnHalo)
                                wt = 0.f;

                            float* o = &rec[((size_t)iy * ow + ix) * NCH];
                            const f32* f = &feat.at(iy, gx, 0);
                            for (int c = 0; c < kRobustnessNnChannels; ++c) o[c] = f[c];
                            o[CH_HARM] = harm;
                            o[CH_RIDEAL] = r_ideal;
                            o[CH_FERR] = std::sqrt((fex - ftx) * (fex - ftx) +
                                                   (fey - fty) * (fey - fty));
                            o[CH_W] = wt;
                            o[CH_REP] = tile_rep[(size_t)std::min(pty, tny_hint - 1) * tnx_hint +
                                                 std::min(ptx, tnx_hint - 1)];
                            o[CH_OCC] = occ_s;
                            o[CH_DIS] = dis_s;
                            o[CH_RNORM] = rnorm.at(gy, gx);
                        }
                    });
                    for (size_t p = 0; p < (size_t)oh * ow; ++p) {
                        const float* o = &rec[p * NCH];
                        sum_w += o[CH_W];
                        sum_ferr += o[CH_W] * o[CH_FERR];
                        sum_r += o[CH_W] * o[CH_RIDEAL];
                        if (o[CH_W] > 0.f && o[CH_RIDEAL] < 0.5f) sum_bad += 1.0;
                    }
                    std::fwrite(rec.data(), sizeof(float), rec.size(), fout);
                    // Sidecar: everything evaluation needs to slice the set by
                    // scene, by failure type and by held-out region without
                    // touching the pixels again.
                    std::fprintf(fidx, "%zu %d %d %d %d %d %d %d %d %.4f %.4f %.4f %.3f %d\n",
                                 total_records, burst_id, refi, ci, vi, pat, band,
                                 cy0, cx0, sum_w > 0 ? sum_ferr / sum_w : 0.0,
                                 sum_w > 0 ? sum_bad / sum_w : 0.0,
                                 sum_w / ((double)oh * ow), gain, phase);
                    ++total_records;
                }
                }
            }
        }
      }   // gains
    }
    std::fclose(fout);
    std::fclose(fidx);
    // In append mode this run's total_records is only ITS OWN contribution --
    // writing it would tell the trainer the set is one burst long and silently
    // train on a fraction of the data. Count what is actually in the file.
    // Counted from the index, one line per record, NOT from the size of the
    // blob: std::ftell returns a 32-bit long on this toolchain and silently
    // overflows past 2 GB, which is every real dataset. It reported 60 records
    // for a 12 GB file.
    size_t records_in_file = 0;
    if (FILE* xf = std::fopen((out_prefix + ".idx").c_str(), "r")) {
        int ch;
        while ((ch = std::fgetc(xf)) != EOF)
            if (ch == 10) ++records_in_file;   // '\n'
        std::fclose(xf);
    }
    if (records_in_file == 0) records_in_file = total_records;
    if (FILE* mf = std::fopen((out_prefix + ".meta").c_str(), "w")) {
        std::fprintf(mf, "guide_h %d\nguide_w %d\nchannels %d\nrecords %zu\npixels %zu\n",
                     gh_all, gw_all, NCH, records_in_file,
                     records_in_file * (size_t)gh_all * gw_all);
        std::fclose(mf);
    }
    std::printf("wrote %s (%zu records of %dx%d x %d ch)\n",
                bin_path.c_str(), total_records, gw_all, gh_all, NCH);
    return 0;
}
