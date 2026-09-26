// Judge the refinement on the MERGED IMAGE, against a known ground truth.
//
// Every number this work has produced so far is against R*, a surrogate label
// the dataset generator computes from the true displacement. R* is defensible
// -- it is the inverse-MSE optimal weight, and it is not circular -- but it is
// still a proxy, and a stage can score well on a proxy while making the picture
// no better. The complaint that started this was about the picture: edges still
// thickening under rotation.
//
// So: synthesise a burst from one reference by a KNOWN motion field, and build
// the ground truth by warping each synthesised frame back through the exact
// inverse of that field and merging those. That is what the merge would produce
// if alignment were perfect -- same kernels, same accumulator, same output
// lattice, same per-frame noise -- so every difference from it is attributable
// to flow error and to what the robustness mask did about it, and to nothing
// else. Merging the reference alone would NOT do: a correct multi-frame merge
// is better than one frame, so the reference is not the target.
//
// Four configurations, which is the comparison the brief asked for:
//
//   0  Wronski only
//   1  Wronski + geometry rejection
//   2  Wronski + learned refinement
//   3  Wronski + geometry rejection + learned refinement
//
// and five measurements on each, because "MSE went down" does not distinguish
// a sharper picture from a blurrier one that happens to hug the mean:
//
//   PSNR        over the whole frame
//   edge        squared error restricted to where the ground truth has edges
//   thin        squared error restricted to thin lines, which are what a
//               too-eager mask eats first
//   thicken     gradient energy the merge put NEXT to a ground-truth edge that
//               the ground truth does not have there. This is the doubled-edge
//               measurement: a doubled edge is not a large error at the edge,
//               it is gradient where there should be none beside it.
//   flat        squared error on flat content -- the noise floor, which says
//               whether rejections cost SNR
//
// each also split by where in its alignment tile the pixel sits (centre,
// interior, edge, corner), because the failure being chased is by construction
// worst far from the tile centre and a frame-wide average hides it.
#include "types.h"
#include "stages.h"
#include "raw_io.h"
#include "parallel.h"
#include "snr_tuning.h"
#include "refine_host.h"
#include "robustness_refine_shared.h"
#include "robustness_refine_weights.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static_assert(hhsr::kRobustnessRefineWeightCount == RR_WEIGHTS_N,
              "regenerate robustness_refine_weights.h: "
              "python tools/rob_refine/export_metal_weights.py <ckpt>");

using namespace hhsr;

namespace {


// ---- sampling and synthesis, twins of refine_dataset.cpp ------------------
//
// Deliberately duplicated rather than shared: the two harnesses must be able to
// disagree. If this one ever stops reproducing the other's geometry that is a
// finding, and a shared helper would hide it.
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

struct MotionField {
    float cos_t = 1.f, sin_t = 0.f;
    float cy = 0.f, cx = 0.f;
    float ty = 0.f, tx = 0.f;

    void set_rotation(float deg) {
        const float r = deg * 3.14159265358979f / 180.f;
        cos_t = std::cos(r); sin_t = std::sin(r);
    }
    // Where reference content at p lands in the comparison frame.
    void fwd(float y, float x, float& oy, float& ox) const {
        const float dy = y - cy, dx = x - cx;
        oy = cos_t * dy - sin_t * dx + cy + ty;
        ox = sin_t * dy + cos_t * dx + cx + tx;
    }
    // comp(q) = ref(inv(q)). Rigid, so this is exact -- no fixed point needed.
    void inv(float y, float x, float& oy, float& ox) const {
        const float dy = y - cy - ty, dx = x - cx - tx;
        oy =  cos_t * dy + sin_t * dx + cy;
        ox = -sin_t * dy + cos_t * dx + cx;
    }
};

// One synthesised exposure, and the same exposure warped back to the
// reference's geometry. The second is the ground truth's input: identical
// content, identical noise realisation, identical resampling -- differing from
// the first ONLY in that its alignment is exact rather than per-tile.
//
// Resampling the same way in both directions matters. If the ground truth were
// built from the reference instead, it would be one resampling sharper than
// anything the merge could produce, and every configuration would look bad for
// a reason that has nothing to do with robustness.
void synth_pair(const Image& ref, const MotionField& X, const Config& cfg,
                uint32_t seed, Image& comp, Image& aligned) {
    comp = Image(ref.h, ref.w, 1);
    const float alpha = cfg.noise_alpha(), beta = cfg.noise_beta();
    parallel_rows(ref.h, cfg.num_threads, [&](int y) {
        std::mt19937 rng(seed * 2654435761u + (uint32_t)y);
        std::normal_distribution<float> gauss(0.f, 1.f);
        for (int x = 0; x < ref.w; ++x) {
            float sy, sx;
            X.inv((float)y, (float)x, sy, sx);
            float v = sample_phase(ref, sy, sx, y & 1, x & 1);
            const float var = std::max(alpha * std::max(v, 0.f) + beta, 0.f);
            v += gauss(rng) * std::sqrt(var);
            comp.at(y, x) = std::max(v, 0.f);
        }
    });
    aligned = Image(ref.h, ref.w, 1);
    parallel_rows(ref.h, cfg.num_threads, [&](int y) {
        for (int x = 0; x < ref.w; ++x) {
            float fy, fx;
            X.fwd((float)y, (float)x, fy, fx);
            aligned.at(y, x) = std::max(sample_phase(comp, fy, fx, y & 1, x & 1), 0.f);
        }
    });
}

// ---- the network, on the host, from the compiled-in weights ---------------
struct HostNet {
    bool infer(const Image& feat, Image& out) const {
        if (feat.c != RR_CHANNELS) return false;
        out = Image(feat.h, feat.w, 1);
        const float* w = kRobustnessRefineWeights;
        parallel_rows(feat.h, 0, [&](int y) {
            float f[RR_CHANNELS];
            for (int x = 0; x < feat.w; ++x) {
                for (int c = 0; c < RR_CHANNELS; ++c) f[c] = feat.at(y, x, c);
                out.at(y, x) = rr_eval(f, w);
            }
        });
        return true;
    }
};
HostNet g_net;
bool host_infer(const Image& feat, Image& out) { return g_net.infer(feat, out); }

// ---- merging ---------------------------------------------------------------
Image merge_burst(const Image& ref, const CovField& ref_covs,
                  const std::vector<Image>& comps,
                  const std::vector<FlowField>& flows,
                  const std::vector<CovField>& covs,
                  const std::vector<Image>& robs,
                  int ts, const Config& cfg) {
    const int Hs = (int)std::lround(cfg.scale * ref.h);
    const int Ws = (int)std::lround(cfg.scale * ref.w);
    const int nch = cfg.bayer_mode ? 3 : 1;
    Image num(Hs, Ws, nch), den(Hs, Ws, nch);
    for (size_t k = 0; k < comps.size(); ++k)
        merge_comp(comps[k], flows[k], covs[k], robs[k], ts, num, den, cfg);
    merge_ref(ref, ref_covs, num, den, cfg, nullptr);
    Image out(Hs, Ws, nch);
    for (size_t i = 0; i < out.data.size(); ++i) {
        const f32 d = den.data[i];
        out.data[i] = (d > 0.f) ? num.data[i] / d : 0.f;
    }
    return out;
}

// Luma of a merged RGB frame, which is what every metric below is computed on.
// A per-channel metric would triple the output and say the same thing: the
// failure being measured is geometric, so it lands on all three.
Image luma_of(const Image& rgb) {
    Image g(rgb.h, rgb.w, 1);
    parallel_rows(rgb.h, 0, [&](int y) {
        for (int x = 0; x < rgb.w; ++x) {
            if (rgb.c == 1) { g.at(y, x) = rgb.at(y, x, 0); continue; }
            g.at(y, x) = 0.299f * rgb.at(y, x, 0) + 0.587f * rgb.at(y, x, 1)
                       + 0.114f * rgb.at(y, x, 2);
        }
    });
    return g;
}

void grad_lap(const Image& g, Image& gmag, Image& lap) {
    gmag = Image(g.h, g.w, 1);
    lap = Image(g.h, g.w, 1);
    parallel_rows(g.h, 0, [&](int y) {
        const int yu = std::max(0, y - 1), yd = std::min(g.h - 1, y + 1);
        for (int x = 0; x < g.w; ++x) {
            const int xl = std::max(0, x - 1), xr = std::min(g.w - 1, x + 1);
            const float gx = 0.5f * (g.at(y, xr) - g.at(y, xl));
            const float gy = 0.5f * (g.at(yd, x) - g.at(yu, x));
            gmag.at(y, x) = std::sqrt(gx * gx + gy * gy);
            lap.at(y, x) = g.at(y, xr) + g.at(y, xl) + g.at(yd, x) + g.at(yu, x)
                         - 4.f * g.at(y, x);
        }
    });
}

// Where in its alignment tile a MERGED pixel sits. The merge lattice is
// cfg.scale times the raw one, so the raw position is out_pos / scale; the
// classification is on the offset from the tile centre, normalised, because
// that is the axis along which one-vector-per-tile is expected to fail.
enum Zone { ZONE_CENTRE = 0, ZONE_INTERIOR, ZONE_EDGE, ZONE_CORNER, ZONE_N };
const char* kZoneName[ZONE_N] = {"centre", "interior", "edge", "corner"};

int zone_of(int y, int x, float scale, int ts) {
    const float ry = (float)y / scale, rx = (float)x / scale;
    const float u = std::fabs(std::fmod(rx, (float)ts) - 0.5f * (float)ts) / (0.5f * (float)ts);
    const float v = std::fabs(std::fmod(ry, (float)ts) - 0.5f * (float)ts) / (0.5f * (float)ts);
    const float m = std::max(u, v), n = std::min(u, v);
    if (m < 0.35f) return ZONE_CENTRE;
    if (n > 0.65f) return ZONE_CORNER;
    if (m > 0.65f) return ZONE_EDGE;
    return ZONE_INTERIOR;
}

struct Acc {
    double se = 0.0; size_t n = 0;
    void add(double e) { se += e * e; ++n; }
    double mean() const { return n ? se / (double)n : 0.0; }
};

struct Metrics {
    Acc all, edge, thin, flat;
    Acc zone[ZONE_N];
    double thicken = 0.0; size_t thicken_n = 0;
    double psnr(double peak) const {
        const double m = all.mean();
        return m > 0.0 ? 10.0 * std::log10(peak * peak / m) : 99.0;
    }
};

// One configuration's output against the ground truth.
//
// The masks are derived from the GROUND TRUTH, never from the output being
// judged: "edge" must mean the same set of pixels for every configuration, or
// a configuration that blurs an edge away would be scored on fewer of them and
// look better for having destroyed the evidence.
Metrics score(const Image& out_rgb, const Image& gt_rgb,
              const Image& gt_g, const Image& gt_gmag, const Image& gt_lap,
              float edge_thr, float thin_thr, float scale, int ts) {
    Image og = luma_of(out_rgb), ogm, olap;
    grad_lap(og, ogm, olap);
    Metrics m;
    // Single-threaded: the accumulators are small and shared, and this runs
    // once per configuration on a frame that took a minute to merge.
    for (int y = 1; y < gt_g.h - 1; ++y) {
        for (int x = 1; x < gt_g.w - 1; ++x) {
            const double e = (double)og.at(y, x) - (double)gt_g.at(y, x);
            m.all.add(e);
            m.zone[zone_of(y, x, scale, ts)].add(e);
            const float gm = gt_gmag.at(y, x);
            if (gm > edge_thr) {
                m.edge.add(e);
                if (std::fabs(gt_lap.at(y, x)) > thin_thr) m.thin.add(e);
            } else if (gm < 0.25f * edge_thr) {
                m.flat.add(e);
                // Thickening: this pixel is NOT an edge in the ground truth,
                // but it neighbours one. Gradient the merge put here is
                // gradient the scene does not have -- a doubled or smeared
                // edge, which is exactly the reported artifact and is
                // invisible to a plain squared error at the edge itself.
                bool near_edge = false;
                for (int i = -2; i <= 2 && !near_edge; ++i)
                    for (int j = -2; j <= 2; ++j) {
                        const int yy = std::min(std::max(y + i, 0), gt_g.h - 1);
                        const int xx = std::min(std::max(x + j, 0), gt_g.w - 1);
                        if (gt_gmag.at(yy, xx) > edge_thr) { near_edge = true; break; }
                    }
                if (near_edge) {
                    const double ex = (double)ogm.at(y, x) - (double)gm;
                    if (ex > 0.0) m.thicken += ex;
                    ++m.thicken_n;
                }
            }
        }
    }
    (void)gt_rgb;
    return m;
}

float percentile(const Image& g, float frac) {
    std::vector<float> v;
    v.reserve((size_t)((g.h / 7) * (g.w / 7) + 16));
    for (int y = 0; y < g.h; y += 7)
        for (int x = 0; x < g.w; x += 7) v.push_back(std::fabs(g.at(y, x)));
    if (v.empty()) return 0.f;
    size_t k = (size_t)(frac * (float)(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + (long)k, v.end());
    return v[k];
}

float envf(const char* k, float d) {
    const char* s = std::getenv(k); return s ? (float)std::atof(s) : d;
}
int envi(const char* k, int d) {
    const char* s = std::getenv(k); return s ? std::atoi(s) : d;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::printf(
            "usage: merge_eval ref.dng [rot_deg ...]\n"
            "  With no angles, sweeps 0.05 0.2 0.5 1.0 2.0 4.0 degrees.\n"
            "env: MERGE_EVAL_FRAMES (5)   comparison frames per burst\n"
            "     MERGE_EVAL_SHIFT (6)    translation added to every frame, raw px\n"
            "     MERGE_EVAL_GEOM (0.0045) geometry rejection threshold\n");
        return 1;
    }
    const int n_frames = std::max(1, envi("MERGE_EVAL_FRAMES", 5));
    const float shift = envf("MERGE_EVAL_SHIFT", 6.f);
    const float geom_thr = envf("MERGE_EVAL_GEOM", 0.0045f);

    std::vector<float> angles;
    for (int i = 2; i < argc; ++i) angles.push_back((float)std::atof(argv[i]));
    if (angles.empty()) angles = {0.05f, 0.2f, 0.5f, 1.0f, 2.0f, 4.0f};

    Config cfg;
    cfg.scale = 2.f;
    cfg.bayer_mode = true;
    cfg.grey_method = GreyMethod::Decimate;
    cfg.r_t = 0.12f; cfg.r_s1 = 2.0f; cfg.r_s2 = 12.0f; cfg.r_Mt = 0.8f;
    cfg.num_threads = 0;
    cfg.alignment_tile_size = 16;

    std::printf("decoding %s\n", argv[1]);
    Image ref = load_raw_frame(argv[1], cfg, true, 0, 0);
    if (ref.w <= 0) { std::printf("decode failed\n"); return 1; }

    Config work = cfg;
    work.burst_frame_count = n_frames + 1;
    tune_config_snr(ref, work);
    work.r_t = cfg.r_t; work.r_s1 = cfg.r_s1; work.r_s2 = cfg.r_s2;
    const int ts = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    std::printf("%dx%d raw, ts=%d, %d comparison frames, scale %.0f\n",
                ref.w, ref.h, ts, n_frames, (double)work.scale);

    clear_align_ref_ica_cache();
    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
    RefStats ref_stats = init_robustness(ref, work);
    CovField ref_covs = estimate_kernels(ref, work);

    struct Cfg4 { const char* name; bool geom; bool nn; };
    const Cfg4 CONFIGS[4] = {
        {"Wronski only",        false, false},
        {"+ geom reject",       true,  false},
        {"+ NN",                false, true },
        {"+ geom reject + NN",  true,  true },
    };

    for (float deg : angles) {
        MotionField X;
        X.cy = 0.5f * (float)ref.h; X.cx = 0.5f * (float)ref.w;
        X.set_rotation(deg);

        std::vector<Image> comps(n_frames), aligned(n_frames);
        std::vector<FlowField> flows(n_frames), zero_flows(n_frames);
        std::vector<CovField> covs(n_frames);
        double mean_flow_err = 0.0, mean_true_disp = 0.0;

        for (int k = 0; k < n_frames; ++k) {
            MotionField Xk = X;
            // Every frame gets its own translation as well, so the burst is a
            // burst and not the same displacement N times -- with one identical
            // comparison frame repeated, the merge cannot average anything and
            // the whole measurement degenerates.
            const float ph = 6.2831853f * (float)k / (float)n_frames;
            Xk.ty = shift * std::sin(ph);
            Xk.tx = shift * std::cos(ph);
            synth_pair(ref, Xk, work, (uint32_t)(k + 1), comps[k], aligned[k]);

            Image cg = compute_grey(comps[k], work.bayer_mode, work.grey_method);
            flows[k] = align(ref_pyr, ref_grey, cg, work, ts);
            covs[k] = estimate_kernels(comps[k], work);

            zero_flows[k] = flows[k];
            std::fill(zero_flows[k].flow.begin(), zero_flows[k].flow.end(), 0.f);

            // What the per-tile flow actually costs, sampled coarsely: the true
            // displacement at a pixel against the vector its tile will be
            // fetched with. This is the quantity the whole stage is about, and
            // printing it keeps the angles honest about what they produced.
            double se = 0.0, st = 0.0; size_t n = 0;
            for (int y = 0; y < ref.h; y += 32)
                for (int x = 0; x < ref.w; x += 32) {
                    float fy, fx;
                    Xk.fwd((float)y, (float)x, fy, fx);
                    const float tdy = fy - (float)y, tdx = fx - (float)x;
                    const int tyi = std::min(y / ts, flows[k].ny - 1);
                    const int txi = std::min(x / ts, flows[k].nx - 1);
                    const float ey = flows[k].dy(tyi, txi) - tdy;
                    const float ex = flows[k].dx(tyi, txi) - tdx;
                    se += std::sqrt(ex * ex + ey * ey);
                    st += std::sqrt(tdx * tdx + tdy * tdy);
                    ++n;
                }
            mean_flow_err += se / (double)n;
            mean_true_disp += st / (double)n;
        }
        mean_flow_err /= n_frames;
        mean_true_disp /= n_frames;

        // ---- ground truth: the same frames, aligned exactly ---------------
        const int gh = ref_stats.means.h, gw = ref_stats.means.w;
        std::vector<Image> ones(n_frames);
        std::vector<CovField> acovs(n_frames);
        for (int k = 0; k < n_frames; ++k) {
            ones[k] = Image(gh, gw, 1);
            std::fill(ones[k].data.begin(), ones[k].data.end(), 1.f);
            acovs[k] = estimate_kernels(aligned[k], work);
        }
        Image gt = merge_burst(ref, ref_covs, aligned, zero_flows, acovs, ones, ts, work);
        aligned.clear(); aligned.shrink_to_fit();
        acovs.clear(); acovs.shrink_to_fit();

        Image gt_g = luma_of(gt), gt_gmag, gt_lap;
        grad_lap(gt_g, gt_gmag, gt_lap);
        const float edge_thr = percentile(gt_gmag, 0.90f);
        const float thin_thr = percentile(gt_lap, 0.95f);
        const float peak = percentile(gt_g, 0.999f);

        std::printf("\n================================================================"
                    "================================\n");
        std::printf("  rotation %.2f deg | mean true displacement %.2f px | "
                    "mean per-tile flow error %.3f px\n",
                    (double)deg, mean_true_disp, mean_flow_err);
        std::printf("  edge threshold %.5f (top 10%%), thin %.5f (top 5%% |lap|), "
                    "peak %.4f\n", (double)edge_thr, (double)thin_thr, (double)peak);
        std::printf("================================================================"
                    "================================\n");
        std::printf("  %-21s %8s %10s %10s %10s %10s\n",
                    "configuration", "PSNR", "edge MSE", "thin MSE", "thicken", "flat MSE");

        Metrics base;
        for (int c = 0; c < 4; ++c) {
            Config run = work;
            run.motion_geom_reject_enabled = CONFIGS[c].geom;
            run.motion_geom_reject_threshold = geom_thr;
            run.robustness_refine_nn_enabled = CONFIGS[c].nn;
            robustness_refine_set_host(CONFIGS[c].nn ? &host_infer : nullptr);

            std::vector<Image> robs(n_frames);
            for (int k = 0; k < n_frames; ++k) {
                robs[k] = compute_robustness(comps[k], ref_stats, flows[k], ts, run);
                if (CONFIGS[c].nn)
                    apply_robustness_refinement(robs[k], comps[k], ref_stats,
                                                flows[k], ts, run, nullptr);
            }
            Image out = merge_burst(ref, ref_covs, comps, flows, covs, robs, ts, run);
            Metrics m = score(out, gt, gt_g, gt_gmag, gt_lap, edge_thr, thin_thr,
                              work.scale, ts);
            if (c == 0) base = m;
            auto pct = [](double v, double b) {
                return b > 0.0 ? 100.0 * (v - b) / b : 0.0;
            };
            std::printf("  %-21s %8.3f %10.3e %10.3e %10.3e %10.3e\n",
                        CONFIGS[c].name, m.psnr((double)peak), m.edge.mean(),
                        m.thin.mean(),
                        m.thicken_n ? m.thicken / (double)m.thicken_n : 0.0,
                        m.flat.mean());
            if (c > 0)
                std::printf("  %-21s %+8.2f %9.1f%% %9.1f%% %9.1f%% %9.1f%%\n", "  vs Wronski",
                            m.psnr((double)peak) - base.psnr((double)peak),
                            pct(m.edge.mean(), base.edge.mean()),
                            pct(m.thin.mean(), base.thin.mean()),
                            pct(m.thicken_n ? m.thicken / (double)m.thicken_n : 0.0,
                                base.thicken_n ? base.thicken / (double)base.thicken_n : 0.0),
                            pct(m.flat.mean(), base.flat.mean()));
            std::printf("  %-21s", "   by tile zone MSE");
            for (int z = 0; z < ZONE_N; ++z)
                std::printf(" %s %9.3e", kZoneName[z], m.zone[z].mean());
            std::printf("\n");
        }
        robustness_refine_set_host(nullptr);
    }
    return 0;
}
