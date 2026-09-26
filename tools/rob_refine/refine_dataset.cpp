// Training-set generator for the robustness REFINEMENT network.
//
// Not the same job as tools/rob_nn/rob_dataset.cpp. That one trains a
// replacement for Wronski Eq. 5-9 and therefore deliberately corrupts the
// flow across the whole error spectrum, out to 250 px, so the network learns
// to score gross failures. This one trains a stage that sits ON TOP of the
// analytic mask and may only subtract from it, aimed at the one failure the
// analytic mask cannot see:
//
//     the flow field carries ONE VECTOR PER TILE, so wherever the real motion
//     has rotation or parallax the true displacement varies across the tile
//     and the merge fetches content that is a fraction of a pixel out --
//     worst at the tile edges, zero at its centre.
//
// A fraction of a pixel is exactly the regime where d^2/sigma^2 becomes
// *more* confident rather than less: sigma picks up the edge's own texture
// faster than d picks up the shift, so R rises toward 1 on precisely the
// edges that are being doubled (rotation-tile-artifact; measured: d 0.0195
// against sigma 0.059 at a rotation-smeared roofline, giving R ~ 1).
//
// So the motion here is generated the other way round from rob_dataset: the
// aligner is left alone to make whatever error it really makes, and the
// synthetic warps are parameterised directly by WITHIN-TILE VARIATION -- how
// many pixels the true displacement changes across one tile -- swept
// log-uniformly from 0.02 px (indistinguishable from perfect) to 3 px, with a
// deliberate mass at exactly zero.
//
// Labels
// ------
// Ground truth is the same non-circular construction rob_dataset uses, and
// for the same reason: labelling by "does the comparison frame differ from
// the reference" would punish aliasing, which is the frame-to-frame
// difference super-resolution exists to exploit, and would train the network
// to reject exactly the signal we want. Instead the comparison frame is
// synthesised from a known warp, so
//
//     Delta = | comp(p + flow_estimated) - comp(p + flow_true) |
//
// is what we fetched against what we should have fetched, both from the same
// frame. It is zero whenever the flow is right, however aliased the content.
//
// The target is then the merge outcome the user actually cares about rather
// than a hand-drawn falloff. A sample whose noise variance is sigma^2 and
// whose misalignment contributes a squared error Delta^2 has mean squared
// error sigma^2 + Delta^2; inverse-MSE weighting against a perfectly aligned
// sample of the same exposure gives it relative weight
//
//     R* = sigma^2 / (sigma^2 + Delta^2) = 1 / (1 + Delta^2/sigma^2)
//
// which is 1 when the flow is right, 1/2 when the misalignment equals one
// noise sigma, and needs no tuning constant. That is "will including this
// sample make the merged pixel worse", stated as a weight. It is also
// deliberately CONSERVATIVE: a systematic misalignment does not average out
// over frames the way independent noise does, so the true optimum is lower
// than this. Erring high is the right direction for a stage whose failure
// mode is throwing away good detail.
//
// The training target for the network is not R* itself but the ratio
//
//     q_keep* = clamp(R* / R_wronski, 0, 1)
//
// computed in train_refine.py -- the multiplier that would turn the analytic
// mask into the ideal one, clamped so the network can never ask for R to go
// UP. Where the analytic mask is already at or below R* the target is exactly
// 1 and the correct answer is to do nothing.
//
// Output: kRobustnessRefineChannels + 5 float32 channels per sampled pixel at
// guide resolution. See LAYOUT below. Channel 0 is R with motion_geom_reject
// OFF and the first channel after the features is the same mask with it ON;
// no other feature channel depends on that toggle, so swapping channel 0 for
// that one at training time gives the exact feature tensor for either
// baseline and both can be evaluated from one dataset.
#include "stages.h"
#include "parallel.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace hhsr;

namespace {

// ---------------------------------------------------------------- LAYOUT
// 0..N-1 build_robustness_refine_features, channel 0 = R (geom reject OFF),
//        where N = kRobustnessRefineChannels
// N      R with motion_geom_reject ON
// N+1    R*, the ideal merge weight from ground truth
// N+2    |flow_est - flow_true| at this pixel, raw px   (analysis only)
// N+3    Delta, the mis-fetch magnitude, intensity units (analysis only)
// N+4    sigma used to normalise it                     (analysis only)
constexpr int kOutChannels = kRobustnessRefineChannels + 5;
constexpr int kChRGeom = kRobustnessRefineChannels;
constexpr int kChRStar = kRobustnessRefineChannels + 1;
constexpr int kChFlowErr = kRobustnessRefineChannels + 2;
constexpr int kChDelta = kRobustnessRefineChannels + 3;
constexpr int kChSigma = kRobustnessRefineChannels + 4;

// Bilinear sample of one CFA colour plane. The pixels of a given Bayer phase
// (oy, ox) form a regular half-resolution lattice; sampling within that
// lattice keeps the synthetic frame a true Bayer image rather than a
// demosaic-remosaic round trip, which would erase the aliasing the whole
// algorithm depends on -- and would quietly delete the single most important
// class of pixel this network must learn NOT to reject.
float sample_phase(const Image& raw, float sy, float sx, int oy, int ox) {
    const float fi = (sy - (float)oy) * 0.5f;
    const float fj = (sx - (float)ox) * 0.5f;
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

// The motion model.
//
//   T(p) = Rot(theta) * (p - c) + c + t + invdepth(p) * parallax
//
// Rigid rotation about the frame centre, a global translation, and a
// parallax term: a smooth inverse-depth field times a fixed camera baseline.
// The last one is what makes this generator different. A rigid rotation
// produces a within-tile displacement variation that is the SAME everywhere
// in the frame (magnitude theta * tile_size), which a network could learn to
// read off the global flow field alone. Parallax makes the variation depend
// on the scene, so the only way to predict it at a given pixel is from the
// local evidence -- which is the thing being trained.
//
// The inverse-depth field is a handful of low-frequency sinusoids plus one
// optional slab. Sinusoids give smooth gradients (the subtle case); the slab
// gives a depth discontinuity, where the flow field is genuinely
// discontinuous and no per-tile vector can be right for both sides.
struct MotionField {
    float cos_t = 1.f, sin_t = 0.f;
    float cy = 0.f, cx = 0.f;
    float ty = 0.f, tx = 0.f;
    float pvy = 0.f, pvx = 0.f;            // parallax baseline, raw px
    float a[3] = {0, 0, 0};                // sinusoid amplitudes, sum <= 1
    float fy[3] = {0, 0, 0}, fx[3] = {0, 0, 0};  // cycles per frame
    float ph[3] = {0, 0, 0};
    bool  slab = false;
    float slab_y0 = 0, slab_y1 = 0, slab_x0 = 0, slab_x1 = 0, slab_z = 0.f;

    // Inverse depth in [0,1]; 0 is infinitely far (no parallax), 1 nearest.
    float invdepth(float y, float x) const {
        if (slab && y >= slab_y0 && y < slab_y1 && x >= slab_x0 && x < slab_x1)
            return slab_z;
        float s = 0.f;
        const float ny = y / std::max(2.f * cy, 1.f), nx = x / std::max(2.f * cx, 1.f);
        for (int i = 0; i < 3; ++i)
            s += a[i] * std::sin(6.2831853f * (fy[i] * ny + fx[i] * nx) + ph[i]);
        return 0.5f + 0.5f * clampf(s, -1.f, 1.f);
    }

    // Forward: where reference content at p lands in the comparison frame.
    // This is exactly what the merge means by flow -- it fetches comp at
    // (p + flow) -- so flow_true(p) = T(p) - p.
    void fwd(float y, float x, float& oy, float& ox) const {
        const float dy = y - cy, dx = x - cx;
        const float z = invdepth(y, x);
        oy = cos_t * dy - sin_t * dx + cy + ty + z * pvy;
        ox = sin_t * dy + cos_t * dx + cx + tx + z * pvx;
    }

    // Inverse, used to synthesise: comp(q) = ref(T^-1 q). The parallax term
    // makes T non-rigid, so there is no closed form; three fixed-point steps
    // from the rigid inverse are more than enough because the parallax
    // displacement is small (a few px) and the field is smooth, so the map is
    // a contraction with a tiny constant.
    void inv(float y, float x, float& oy, float& ox) const {
        float py = 0.f, px = 0.f;
        for (int it = 0; it < 3; ++it) {
            const float dy = y - cy - ty - py, dx = x - cx - tx - px;
            oy =  cos_t * dy + sin_t * dx + cy;
            ox = -sin_t * dy + cos_t * dx + cx;
            const float z = invdepth(oy, ox);
            py = z * pvy; px = z * pvx;
        }
    }
};

// A rectangle of the comparison frame filled with content copied from
// elsewhere in the reference: a stand-in for the scene changing between
// frames. Without these the training set contains only camera motion, and a
// network trained on it would learn "the geometry looks plausible, therefore
// merge", which is the mistake that ghosts a moving subject. They also give
// the mismatch channel (23) something to separate from the geometric case.
struct OccRect { int y0, y1, x0, x1, sy, sx; };

bool in_rect(const std::vector<OccRect>& rs, float y, float x, int* which = nullptr) {
    for (size_t i = 0; i < rs.size(); ++i) {
        const OccRect& r = rs[i];
        if (y >= (float)r.y0 && y < (float)r.y1 && x >= (float)r.x0 && x < (float)r.x1) {
            if (which) *which = (int)i;
            return true;
        }
    }
    return false;
}

Image synth_frame(const Image& ref, const MotionField& X, const Config& cfg,
                  uint32_t seed, const std::vector<OccRect>& occ) {
    Image out(ref.h, ref.w, 1);
    const float alpha = cfg.noise_alpha(), beta = cfg.noise_beta();
    // Deterministic per row, so the thread count can never change the data.
    parallel_rows(ref.h, cfg.num_threads, [&](int y) {
        std::mt19937 rng(seed * 2654435761u + (uint32_t)y);
        std::normal_distribution<float> gauss(0.f, 1.f);
        for (int x = 0; x < ref.w; ++x) {
            float sy, sx;
            int oi = 0;
            if (in_rect(occ, (float)y, (float)x, &oi)) {
                // Unrelated content, kept on the same Bayer phase so the
                // patch is still a valid mosaic rather than a colour shift.
                sy = (float)(y + occ[oi].sy);
                sx = (float)(x + occ[oi].sx);
            } else {
                X.inv((float)y, (float)x, sy, sx);
            }
            float v = sample_phase(ref, sy, sx, y & 1, x & 1);
            // Heteroscedastic sensor noise: var = alpha*signal + beta. The
            // reference already carries its own, so this is the second
            // independent realisation a real second exposure would have --
            // and it is what stops the network from learning that a zero
            // residual is achievable.
            const float var = std::max(alpha * std::max(v, 0.f) + beta, 0.f);
            v += gauss(rng) * std::sqrt(var);
            out.at(y, x) = std::max(v, 0.f);
        }
    });
    return out;
}

// Guide-resolution 3x3 mean/variance, matching robustness.cpp's
// local_stats_3x3 (which is file-static, so it cannot simply be called).
void local_stats(const Image& g, Image& means, Image& vars) {
    means = Image(g.h, g.w, g.c);
    vars = Image(g.h, g.w, g.c);
    for (int ch = 0; ch < g.c; ++ch)
        for (int y = 0; y < g.h; ++y)
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
}

float envf(const char* k, float d) {
    if (const char* v = std::getenv(k)) return (float)std::atof(v);
    return d;
}
int envi(const char* k, int d) {
    if (const char* v = std::getenv(k)) return std::atoi(v);
    return d;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        std::printf("usage: refine_dataset out_prefix n_frames ref.dng [more_refs.dng...]\n"
                    "env: ROB_REFINE_STRIDE (2)    spatial decimation of the output\n"
                    "     ROB_REFINE_WTV_MAX (0.8) max within-tile variation, px/tile\n"
                    "     ROB_REFINE_SHIFT (12)    max global translation, raw px\n"
                    "     ROB_REFINE_JITTER (0.15) fraction of tiles given a small\n"
                    "                              extra offset; 0 isolates the\n"
                    "                              aligner's own error\n"
                    "     ROB_REFINE_REGIME (-1)   force one motion regime (0..7)\n");
        return 1;
    }
    const std::string out_prefix = argv[1];
    const int n_frames = std::atoi(argv[2]);
    std::vector<std::string> refs;
    for (int i = 3; i < argc; ++i) refs.push_back(argv[i]);

    const int stride = std::max(1, envi("ROB_REFINE_STRIDE", 2));
    // Maximum within-tile displacement variation, px across one tile.
    // At tile_size 16 this is a rotation of wtv/16 radians, so 0.8 px/tile is
    // 2.9 degrees -- already past what a handheld burst does between frames,
    // and comfortably past where the per-tile translation model breaks. The
    // first draft used 3.0, which is 10.7 degrees: measured, that produced a
    // 41 px mean flow error and moved the whole frame into the gross-failure
    // regime this stage is explicitly not for.
    const float wtv_max = envf("ROB_REFINE_WTV_MAX", 0.8f);
    // Frame-to-frame handheld translation, raw px. Deliberately modest:
    // measured on this harness, +-30 px drives the aligner into outright
    // search failure over flat content and the resulting 5-40 px flow errors
    // become 43% of all the correctable mass, drowning the sub-pixel regime
    // this stage exists for. Gross failures are the replacement network's job
    // (tools/rob_nn); train_refine.py additionally stratifies by flow error so
    // the remainder cannot dominate either.
    const float shift_max = envf("ROB_REFINE_SHIFT", 12.f);
    // Fraction of tiles given a small extra offset. Set to 0 to measure the
    // aligner's own error alone -- which is also the sanity check that matters
    // most here: with no jitter and a static camera, the label must say
    // "reject nothing", because the only frame-to-frame difference is noise
    // and Bayer aliasing. If it does not, the label is punishing aliasing and
    // every number downstream of it is worthless.
    const float jitter_frac = envf("ROB_REFINE_JITTER", 0.15f);

    const std::string bin_path = out_prefix + ".f32";
    FILE* fout = std::fopen(bin_path.c_str(), "wb");
    if (!fout) { std::printf("cannot open %s\n", bin_path.c_str()); return 1; }
    size_t total_rows = 0;
    int oh_all = 0, ow_all = 0, frames_written = 0;

    for (size_t ri = 0; ri < refs.size(); ++ri) {
        Config cfg;
        cfg.scale = 2.f;
        cfg.bayer_mode = true;
        cfg.grey_method = GreyMethod::Decimate;
        cfg.r_t = 0.12f; cfg.r_s1 = 2.0f; cfg.r_s2 = 12.0f; cfg.r_Mt = 0.8f;
        cfg.num_threads = 0;
        cfg.alignment_tile_size = 16;

        std::printf("[ref %zu] decoding %s\n", ri, refs[ri].c_str());
        Image ref = load_raw_frame(refs[ri], cfg, true, 0, 0);
        if (ref.w <= 0) { std::printf("  decode failed\n"); continue; }
        Config work = cfg;
        work.burst_frame_count = n_frames + 1;
        tune_config_snr(ref, work);
        work.r_t = cfg.r_t; work.r_s1 = cfg.r_s1; work.r_s2 = cfg.r_s2;
        const int ts = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];

        // align() caches the reference's Sobel gradients and ICA Hessian,
        // keyed on the ADDRESS of the Pyramid it was given. ref_pyr below is a
        // loop local, so the second and every later reference is handed back
        // the FIRST reference's derivatives and aligns against them. Measured
        // before this call existed: a static camera on reference 0 gave a mean
        // flow error of 0.207 px, and references 1..N all converged to
        // 0.589 px with 42% of the frame labelled unmergeable instead of 6%.
        clear_align_ref_ica_cache();
        Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
        Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
        RefStats ref_stats = init_robustness(ref, work);
        const int gh = ref_stats.means.h, gw = ref_stats.means.w;
        const int oh = (gh + stride - 1) / stride, ow = (gw + stride - 1) / stride;
        oh_all = oh; ow_all = ow;
        std::printf("[ref %zu] %dx%d raw, guide %dx%d, ts=%d, out %dx%d (stride %d)\n",
                    ri, ref.w, ref.h, gw, gh, ts, ow, oh, stride);

        for (int fi = 0; fi < n_frames; ++fi) {
            const uint32_t seed = (uint32_t)(ri * 1000 + fi + 1);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            std::uniform_real_distribution<float> u01(0.f, 1.f);

            // Systematic coverage of the regimes rather than random draws, so
            // every one is guaranteed present for every reference instead of
            // merely likely. The regime index cycles on the frame number.
            //
            //  0  static camera. The ONLY difference between the frames is
            //     noise and Bayer aliasing. Nothing here may be rejected, and
            //     a generator without this case trains a network that quietly
            //     eats fine detail in still scenes.
            //  1  pure subpixel translation. Maximal aliasing difference,
            //     zero within-tile variation -- the other half of the
            //     false-positive pressure, and the case that teaches the net
            //     that a large residual on a sharp edge is not by itself
            //     evidence of anything.
            //  2  pure large translation, still zero within-tile variation.
            //  3  small rotation only: the target regime, subtle end.
            //  4  rotation + translation, stronger.
            //  5  parallax only, smooth: within-tile variation that is not
            //     predictable from the global motion.
            //  6  parallax with a depth slab: a genuine flow discontinuity.
            //  7  everything at once, plus scene change.
            const int forced = envi("ROB_REFINE_REGIME", -1);
            const int regime = (forced >= 0 && forced < 8) ? forced : (fi % 8);
            MotionField X;
            X.cy = 0.5f * (float)ref.h; X.cx = 0.5f * (float)ref.w;

            // Within-tile variation, in px across one tile, swept
            // log-uniformly. This -- not the rotation angle, not the shift --
            // is the quantity the stage exists to judge, so it is the one
            // that gets balanced coverage.
            float wtv = 0.f;
            if (regime >= 3)
                wtv = 0.02f * std::pow(wtv_max / 0.02f, u01(rng));
            float rot_share = 0.f, par_share = 0.f;
            switch (regime) {
                case 3: case 4: rot_share = 1.f; break;
                case 5: case 6: par_share = 1.f; break;
                case 7: rot_share = u01(rng); par_share = 1.f - rot_share; break;
                default: break;
            }

            const float theta = (rot_share * wtv / (float)ts) * (u(rng) < 0.f ? -1.f : 1.f);
            X.cos_t = std::cos(theta); X.sin_t = std::sin(theta);

            if (regime == 1) {
                // Subpixel only: the aliasing case, deliberately never a whole
                // pixel, because a whole-pixel shift on this lattice is the
                // one translation that does NOT change the sampling phase.
                X.ty = u(rng) * 0.9f; X.tx = u(rng) * 0.9f;
            } else if (regime != 0) {
                X.ty = u(rng) * shift_max; X.tx = u(rng) * shift_max;
            }

            if (par_share > 0.f) {
                // Choose the sinusoids first, then scale the baseline so the
                // field's actual maximum gradient delivers the requested
                // within-tile variation. Solving for it rather than guessing
                // is what keeps the sweep honest across frequencies.
                float amp_sum = 0.f, grad_norm = 0.f;
                for (int i = 0; i < 3; ++i) {
                    X.a[i] = u01(rng);
                    amp_sum += X.a[i];
                    X.fy[i] = u(rng) * 4.f;
                    X.fx[i] = u(rng) * 4.f;
                    X.ph[i] = u01(rng) * 6.2831853f;
                }
                for (int i = 0; i < 3; ++i) X.a[i] /= std::max(amp_sum, 1e-6f);
                // |d invdepth / d pixel| <= 0.5 * sum a_i * 2pi * |f| / span
                for (int i = 0; i < 3; ++i) {
                    const f32 gy = 6.2831853f * X.fy[i] / (float)ref.h;
                    const f32 gx = 6.2831853f * X.fx[i] / (float)ref.w;
                    grad_norm += 0.5f * X.a[i] * std::sqrt(gy * gy + gx * gx);
                }
                const float want_grad = par_share * wtv / (float)ts;   // px per px
                const float baseline = want_grad / std::max(grad_norm, 1e-9f);
                const float ang = u01(rng) * 6.2831853f;
                X.pvy = baseline * std::sin(ang);
                X.pvx = baseline * std::cos(ang);
                if (regime == 6 || regime == 7) {
                    // A depth slab. Its edge is a step in the flow field, so
                    // the tiles straddling it cannot be right for both sides
                    // whatever the aligner does -- the case where rejection
                    // really is the only available answer.
                    X.slab = true;
                    const int hgt = 200 + (int)(u01(rng) * 900.f);
                    const int wid = 200 + (int)(u01(rng) * 900.f);
                    X.slab_y0 = (float)(int)(u01(rng) * (float)(ref.h - hgt - 1));
                    X.slab_x0 = (float)(int)(u01(rng) * (float)(ref.w - wid - 1));
                    X.slab_y1 = X.slab_y0 + (float)hgt;
                    X.slab_x1 = X.slab_x0 + (float)wid;
                    X.slab_z = u01(rng);
                }
            }

            std::vector<OccRect> occ;
            const int n_occ = (regime == 7) ? 3 : 0;
            for (int k = 0; k < n_occ; ++k) {
                const int hgt = 64 + (int)(u01(rng) * 400.f);
                const int wid = 64 + (int)(u01(rng) * 400.f);
                OccRect r;
                r.y0 = (int)(u01(rng) * (float)(ref.h - hgt - 1));
                r.x0 = (int)(u01(rng) * (float)(ref.w - wid - 1));
                r.y0 &= ~1; r.x0 &= ~1;          // keep Bayer phase
                r.y1 = r.y0 + hgt; r.x1 = r.x0 + wid;
                r.sy = (((int)(u(rng) * 600.f)) & ~1);
                r.sx = (((int)(u(rng) * 600.f)) & ~1);
                if (r.y0 + r.sy < 0) r.sy = -r.y0;
                if (r.y1 + r.sy >= ref.h) r.sy = ref.h - 1 - r.y1;
                if (r.x0 + r.sx < 0) r.sx = -r.x0;
                if (r.x1 + r.sx >= ref.w) r.sx = ref.w - 1 - r.x1;
                r.sy &= ~1; r.sx &= ~1;
                occ.push_back(r);
            }

            double jitter_pct = 0.0;
            Image comp = synth_frame(ref, X, work, seed, occ);
            Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
            FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, ts, 0.f, 0.f, 0.f);
            flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w,
                                         ts, work.r_Mt, work.num_threads,
                                         work.grey_tile_size(ts));

            // The real aligner, left alone, is good: on most tiles its error
            // is well under a pixel, which is the regime we want but also
            // means the dataset would be almost entirely "no correction
            // needed". A minority of tiles get a small extra offset so the
            // decision boundary is populated from both sides. Deliberately
            // SMALL -- 0 to 2 px. Larger errors belong to the replacement
            // network in tools/rob_nn; this stage only ever gets to see
            // pixels the analytic mask has already decided to trust.
            {
                std::mt19937 jr(seed * 7919u + 13u);
                long long n_jittered = 0;
                std::uniform_real_distribution<float> j01(0.f, 1.f);
                std::uniform_real_distribution<float> ang(0.f, 6.2831853f);
                for (int ty = 0; ty < flow.ny; ++ty)
                    for (int tx = 0; tx < flow.nx; ++tx) {
                        if (j01(jr) > jitter_frac) continue;
                        // Log-uniform, not uniform. Uniform over 0..2 px puts
                        // the average jittered tile a whole pixel out, which
                        // is the moderate band; the band this stage works in
                        // is tenths of a pixel, and it needs to be populated
                        // by something other than the tail of a uniform draw.
                        const float mag = 0.05f * std::pow(40.f, j01(jr));
                        const float a = ang(jr);
                        flow.dx(ty, tx) += mag * std::cos(a);
                        flow.dy(ty, tx) += mag * std::sin(a);
                        ++n_jittered;
                    }
                jitter_pct = 100.0 * (double)n_jittered /
                             std::max(1.0, (double)flow.ny * flow.nx);
            }

            // The two analytic baselines. Both masks, same frame, same flow:
            // channel 0 of the features is the first, channel 24 the second,
            // and every other feature channel is independent of the toggle,
            // so training and evaluation can construct either baseline's
            // exact feature tensor from this one record.
            Config cfg_plain = work; cfg_plain.motion_geom_reject_enabled = false;
            Config cfg_geom  = work; cfg_geom.motion_geom_reject_enabled  = true;
            Image R_plain = compute_robustness(comp, ref_stats, flow, ts, cfg_plain);
            Image R_geom  = compute_robustness(comp, ref_stats, flow, ts, cfg_geom);
            if (R_plain.h != gh || R_geom.h != gh) {
                std::printf("  robustness returned %dx%d, expected %dx%d -- skipping\n",
                            R_plain.h, R_plain.w, gh, gw);
                continue;
            }

            // Eq. 6's outputs, for the feature builder. Same construction the
            // shipped apply_robustness_refinement uses.
            Image comp_guide = compute_guide(comp, work);
            Image comp_means, comp_vars;
            local_stats(comp_guide, comp_means, comp_vars);
            comp_vars = Image();
            Image d_sq, sigma_sq;
            robustness_correspondence(ref_stats.means, ref_stats.stds, comp_means,
                                      flow, ts, /*raw_res=*/false, work, d_sq, sigma_sq);

            Image feat = build_robustness_refine_features(
                ref_stats, comp_means, R_plain, d_sq, sigma_sq, flow, ts, work,
                /*y0=*/0, /*strip_h=*/gh);
            if (feat.h != gh || feat.c != kRobustnessRefineChannels) {
                std::printf("  feature builder returned %dx%dx%d -- skipping\n",
                            feat.h, feat.w, feat.c);
                continue;
            }

            // ---- label ----
            const float alpha = work.noise_alpha(), beta = work.noise_beta();
            std::vector<float> rec((size_t)oh * ow * kOutChannels, 0.f);
            double sum_flow_err = 0.0, sum_wtv = 0.0;
            size_t n_reject = 0, n_px = 0;
            // Pixels the flow got essentially right that the label nonetheless
            // marks down. Should be a rounding error; anything else means the
            // label is reading aliasing or noise as misalignment.
            size_t n_alias_hit = 0;

            parallel_rows(oh, work.num_threads, [&](int oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    const int gy = std::min(oy * stride, gh - 1);
                    const int gx = std::min(ox * stride, gw - 1);
                    const int ry = 2 * gy, rx = 2 * gx;
                    const int pty = std::min(flow.ny - 1, std::max(0, ry / ts));
                    const int ptx = std::min(flow.nx - 1, std::max(0, rx / ts));
                    const float fex = flow.dx(pty, ptx), fey = flow.dy(pty, ptx);

                    // Ground-truth flow, analytic, at this pixel -- not at the
                    // tile centre. The gap between the two IS the error this
                    // stage exists to find.
                    float tyv, txv;
                    X.fwd((float)ry, (float)rx, tyv, txv);
                    const float ftx = txv - (float)rx, fty = tyv - (float)ry;
                    const bool occluded = in_rect(occ, tyv, txv);

                    // Delta: what we fetch against what we should have
                    // fetched, both from the comparison frame. Measured at RAW
                    // resolution over the quad, so an error in structure finer
                    // than the guide's 3x3 mean survives into the label even
                    // though it is invisible to d.
                    float d2 = 0.f, nvar = 0.f;
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j) {
                            const int py = ry + i, px = rx + j;
                            const float got = sample_phase(comp, (float)py + fey,
                                                           (float)px + fex, py & 1, px & 1);
                            const float want = sample_phase(comp, (float)py + fty,
                                                            (float)px + ftx, py & 1, px & 1);
                            const float e = got - want;
                            d2 += e * e;
                            nvar += std::max(alpha * std::max(want, 0.f) + beta, 0.f);
                        }
                    d2 *= 0.25f; nvar *= 0.25f;
                    // Inverse-MSE weight; see the header. No tuning constant.
                    const float r_star = occluded ? 0.f
                        : (float)(nvar / std::max((double)nvar + (double)d2, 1e-20));

                    float* o = &rec[((size_t)oy * ow + ox) * kOutChannels];
                    const float* f = &feat.at(gy, gx, 0);
                    for (int c = 0; c < kRobustnessRefineChannels; ++c) o[c] = f[c];
                    o[kChRGeom] = R_geom.at(gy, gx);
                    o[kChRStar] = r_star;
                    o[kChFlowErr] = std::sqrt((fex - ftx) * (fex - ftx) +
                                              (fey - fty) * (fey - fty));
                    o[kChDelta] = std::sqrt(d2);
                    o[kChSigma] = std::sqrt(nvar);
                    (void)n_px;
                }
            });

            for (int oy = 0; oy < oh; ++oy)
                for (int ox = 0; ox < ow; ++ox) {
                    const float* o = &rec[((size_t)oy * ow + ox) * kOutChannels];
                    sum_flow_err += o[kChFlowErr];
                    sum_wtv += o[6];                    // |E|
                    if (o[kChRStar] < 0.5f && o[0] > 0.5f) ++n_reject;
                    if (o[kChFlowErr] < 0.05f && o[kChRStar] < 0.9f) ++n_alias_hit;
                }
            std::fwrite(rec.data(), sizeof(float), rec.size(), fout);
            total_rows += (size_t)oh * ow;
            ++frames_written;
            const double npx = (double)oh * ow;
            std::printf("[ref %zu frame %2d] regime %d wtv=%.3f px/tile rot=%+.4fdeg "
                        "shift=(%+.1f,%+.1f) par=%.2f occ=%d jit=%.0f%% | meanFlowErr=%.3fpx "
                        "mean|E|=%.3fpx | analytic mask trusts %.2f%% that ground truth "
                        "says to drop; label marks down %.3f%% of correctly-fetched pixels\n",
                        ri, fi, regime, wtv, theta * 180.f / 3.14159265f, X.tx, X.ty,
                        std::sqrt(X.pvx * X.pvx + X.pvy * X.pvy), n_occ, jitter_pct,
                        sum_flow_err / npx, sum_wtv / npx,
                        100.0 * (double)n_reject / npx,
                        100.0 * (double)n_alias_hit / npx);
        }
    }
    std::fclose(fout);
    const std::string meta_path = out_prefix + ".meta";
    if (FILE* mf = std::fopen(meta_path.c_str(), "w")) {
        std::fprintf(mf, "height %d\nwidth %d\nchannels %d\nfeature_channels %d\n"
                         "frames %d\npixels %zu\n",
                     oh_all, ow_all, kOutChannels, kRobustnessRefineChannels,
                     frames_written, total_rows);
        std::fclose(mf);
    }
    std::printf("wrote %s (%d frames, %zu pixels x %d ch)\n",
                bin_path.c_str(), frames_written, total_rows, kOutChannels);
    return 0;
}
