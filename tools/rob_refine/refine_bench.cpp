// Exercise and measure the SHIPPED refinement path off-device.
//
// train_refine.py and eval_refine.py measure the network. They do not measure
// apply_robustness_refinement: the feature builder it calls, the strip loop,
// the bounded multiply, the dead zone, the cost of running all of that on a
// 3 MP plane once per comparison frame. Core ML is Apple-only, so the only
// way to exercise that code anywhere else is to give it a host evaluator for
// the same weights -- which is what this does, via the refine_host.h hook the
// test harness installs in place of robustness_nn.mm.
//
// What it checks:
//
//   identity     with the toggle off, R must come back BIT-IDENTICAL to the
//                mask the pipeline produces today. This is the "do not
//                destroy the current behaviour" guarantee, tested rather
//                than asserted.
//   one-sided    R_final <= R everywhere, and R == 0 stays 0. The network is
//                not trusted to respect that; the pipeline enforces it, and
//                this checks the enforcement.
//   sparsity     what fraction of pixels the stage actually moves.
//   cost         wall time and the peak extra allocation of the stage, which
//                is what the 200 ms / 200 MB budget is about. The numbers
//                here are a desktop CPU running the network in scalar C++;
//                on device the same graph runs on the ANE or GPU through
//                Core ML. Treat the feature-builder time as indicative and
//                the network time as an upper bound.
//
// Usage:  refine_bench ref.dng refinenet_mlp_geom.bin [rotation_deg]
#include "stages.h"
#include "parallel.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include "refine_host.h"
#include "robustness_refine_shared.h"
#include "robustness_refine_weights.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace hhsr;

namespace {

// ---- the host stand-in for Core ML --------------------------------------
struct HostNet {
    int arch = 0, in_ch = 0, width = 0;
    std::vector<int> log_ch;
    std::vector<float> mu, sd;
    std::vector<std::vector<float>> w;   // parameters in nn.Sequential order
    bool loaded = false;

    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::printf("cannot open %s\n", path.c_str()); return false; }
        char magic[4];
        int n_log = 0, n_par = 0;
        bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "RFN1", 4) == 0;
        ok = ok && std::fread(&arch, 4, 1, f) == 1;
        ok = ok && std::fread(&in_ch, 4, 1, f) == 1;
        ok = ok && std::fread(&width, 4, 1, f) == 1;
        ok = ok && std::fread(&n_log, 4, 1, f) == 1;
        if (!ok || in_ch != kRobustnessRefineChannels) {
            std::printf("bad weight file (in_ch %d, expected %d)\n",
                        in_ch, kRobustnessRefineChannels);
            std::fclose(f);
            return false;
        }
        log_ch.resize(n_log);
        ok = ok && std::fread(log_ch.data(), 4, n_log, f) == (size_t)n_log;
        mu.resize(in_ch); sd.resize(in_ch);
        ok = ok && std::fread(mu.data(), 4, in_ch, f) == (size_t)in_ch;
        ok = ok && std::fread(sd.data(), 4, in_ch, f) == (size_t)in_ch;
        ok = ok && std::fread(&n_par, 4, 1, f) == 1;
        for (int i = 0; ok && i < n_par; ++i) {
            int n = 0;
            ok = std::fread(&n, 4, 1, f) == 1;
            if (!ok || n <= 0 || n > (1 << 24)) { ok = false; break; }
            w.emplace_back(n);
            ok = std::fread(w.back().data(), 4, n, f) == (size_t)n;
        }
        std::fclose(f);
        loaded = ok;
        if (ok)
            std::printf("host net: arch %s, in_ch %d, width %d, %d tensors\n",
                        arch == 0 ? "mlp" : "cnn", in_ch, width, n_par);
        return ok;
    }

    // Pointwise only. The convolutional variant is not implemented here on
    // purpose: shipping it needs kRobustnessRefineHalo raised to match, and a
    // bench that quietly ran a different receptive field from the device
    // would be worse than no bench.
    bool infer(const Image& feat, Image& out) const {
        if (!loaded || arch != 0 || w.size() != 6) return false;
        const int H = feat.h, W_ = feat.w, C = feat.c, wd = width;
        Image r(H, W_, 1);
        parallel_rows(H, 0, [&](int y) {
            std::vector<float> h0((size_t)wd), h1((size_t)wd), x((size_t)C);
            for (int px = 0; px < W_; ++px) {
                const f32* in = &feat.at(y, px, 0);
                for (int c = 0; c < C; ++c) {
                    float v = in[c];
                    for (int k : log_ch)
                        if (k == c) { v = std::copysign(std::log1p(std::fabs(v)), v); break; }
                    x[c] = (v - mu[c]) / sd[c];
                }
                for (int o = 0; o < wd; ++o) {
                    float s = w[1][o];
                    const float* row = &w[0][(size_t)o * C];
                    for (int c = 0; c < C; ++c) s += row[c] * x[c];
                    h0[o] = s > 0.f ? s : 0.f;
                }
                for (int o = 0; o < wd; ++o) {
                    float s = w[3][o];
                    const float* row = &w[2][(size_t)o * wd];
                    for (int c = 0; c < wd; ++c) s += row[c] * h0[c];
                    h1[o] = s > 0.f ? s : 0.f;
                }
                float s = w[5][0];
                for (int c = 0; c < wd; ++c) s += w[4][c] * h1[c];
                r.at(y, px) = 1.f / (1.f + std::exp(-s));
            }
        });
        out = std::move(r);
        return true;
    }
};

HostNet g_net;
bool host_infer(const Image& feat, Image& out) { return g_net.infer(feat, out); }

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

double ms_now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::printf("usage: refine_bench ref.dng refinenet_mlp_geom.bin [rot_deg]\n");
        return 1;
    }
    const float rot_deg = (argc > 3) ? (float)std::atof(argv[3]) : 1.0f;
    if (!g_net.load(argv[2])) return 1;

    Config cfg;
    cfg.scale = 2.f;
    cfg.bayer_mode = true;
    cfg.grey_method = GreyMethod::Decimate;
    cfg.r_t = 0.12f; cfg.r_s1 = 2.0f; cfg.r_s2 = 12.0f; cfg.r_Mt = 0.8f;
    cfg.num_threads = 0;
    cfg.alignment_tile_size = 16;

    Image ref = load_raw_frame(argv[1], cfg, true, 0, 0);
    if (ref.w <= 0) { std::printf("decode failed\n"); return 1; }
    Config work = cfg;
    work.burst_frame_count = 2;
    tune_config_snr(ref, work);
    work.r_t = cfg.r_t; work.r_s1 = cfg.r_s1; work.r_s2 = cfg.r_s2;
    work.motion_geom_reject_enabled = true;
    const int ts = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];

    // A comparison frame with a known rotation: the regime the stage is for.
    const float th = rot_deg * 3.14159265f / 180.f;
    const float ct = std::cos(th), st = std::sin(th);
    const float cy = 0.5f * (float)ref.h, cx = 0.5f * (float)ref.w;
    Image comp(ref.h, ref.w, 1);
    const float alpha = work.noise_alpha(), beta = work.noise_beta();
    parallel_rows(ref.h, work.num_threads, [&](int y) {
        std::mt19937 rng(7u + (uint32_t)y);
        std::normal_distribution<float> g(0.f, 1.f);
        for (int x = 0; x < ref.w; ++x) {
            const float dy = (float)y - cy, dx = (float)x - cx;
            const float sy = ct * dy + st * dx + cy;
            const float sx = -st * dy + ct * dx + cx;
            float v = sample_phase(ref, sy, sx, y & 1, x & 1);
            v += g(rng) * std::sqrt(std::max(alpha * std::max(v, 0.f) + beta, 0.f));
            comp.at(y, x) = std::max(v, 0.f);
        }
    });

    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
    RefStats ref_stats = init_robustness(ref, work);
    Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
    FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, ts, 0.f, 0.f, 0.f);
    flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w,
                                 ts, work.r_Mt, work.num_threads,
                                 work.grey_tile_size(ts));
    std::printf("frame %dx%d, guide %dx%d, tile %d, rotation %.2f deg\n",
                ref.w, ref.h, ref_stats.means.w, ref_stats.means.h, ts, rot_deg);

    // ---- 1. identity when the toggle is off -----------------------------
    double t0 = ms_now();
    Image R_base = compute_robustness(comp, ref_stats, flow, ts, work);
    const double t_base = ms_now() - t0;
    robustness_refine_set_host(nullptr);
    Config off = work; off.robustness_refine_nn_enabled = true;
    Image R_nomodel = compute_robustness(comp, ref_stats, flow, ts, off);
    size_t diff_nomodel = 0;
    for (size_t i = 0; i < R_base.data.size(); ++i)
        if (std::memcmp(&R_base.data[i], &R_nomodel.data[i], sizeof(f32)) != 0)
            ++diff_nomodel;
    std::printf("\nidentity: toggle on but no model loaded -> %zu of %zu pixels "
                "differ (must be 0)\n", diff_nomodel, R_base.data.size());

    // ---- 2. the refinement itself ---------------------------------------
    robustness_refine_set_host(&host_infer);
    Config on = work;
    on.robustness_refine_nn_enabled = true;
    // Left at the Config defaults deliberately: the point of this bench is to
    // measure what the app will actually do, not a hand-picked setting.

    Image R_ref = R_base;
    float changed = 0.f;
    t0 = ms_now();
    const bool ran = apply_robustness_refinement(R_ref, comp, ref_stats, flow, ts,
                                                 on, &changed);
    const double t_refine = ms_now() - t0;
    if (!ran) { std::printf("refinement declined to run\n"); return 1; }

    size_t violated = 0, resurrected = 0, moved = 0;
    double sum_base = 0, sum_ref = 0, max_drop = 0;
    for (size_t i = 0; i < R_base.data.size(); ++i) {
        const f32 a = R_base.data[i], b = R_ref.data[i];
        if (!std::isfinite(a)) continue;
        if (b > a + 1e-7f) ++violated;
        if (a <= 0.f && b > 0.f) ++resurrected;
        if (std::fabs(b - a) > 1e-7f) ++moved;
        sum_base += a; sum_ref += b;
        max_drop = std::max(max_drop, (double)(a - b));
    }
    const double npx = (double)R_base.data.size();
    std::printf("\none-sided: R_final > R at %zu pixels (must be 0)\n", violated);
    std::printf("           R was 0 and became positive at %zu pixels (must be 0)\n",
                resurrected);
    std::printf("           largest single reduction %.4f (cap is kappa = %.2f)\n",
                max_drop, on.robustness_refine_max_reduction);
    std::printf("\nsparsity:  %.2f%% of pixels moved (reported %.2f%%)\n",
                100.0 * (double)moved / npx, 100.0 * changed);
    std::printf("           mean R %.4f -> %.4f\n", sum_base / npx, sum_ref / npx);

    // ---- 3. cost ---------------------------------------------------------
    // Split the stage so the two halves can be judged separately: the feature
    // builder is ours and runs on the CPU on device too, while the network
    // part is what Core ML will move to the ANE.
    Image comp_means;
    {
        // robustness.cpp's local_stats_3x3 is file-static, so this repeats it
        // rather than exporting a symbol purely for a bench.
        Image g = compute_guide(comp, work);
        comp_means = Image(g.h, g.w, g.c);
        parallel_rows(g.h, work.num_threads, [&](int y) {
            for (int x = 0; x < g.w; ++x)
                for (int ch = 0; ch < g.c; ++ch) {
                    float s = 0.f;
                    for (int i = -1; i <= 1; ++i) {
                        const int yy = std::min(std::max(y + i, 0), g.h - 1);
                        for (int j = -1; j <= 1; ++j) {
                            const int xx = std::min(std::max(x + j, 0), g.w - 1);
                            s += g.at(yy, xx, ch);
                        }
                    }
                    comp_means.at(y, x, ch) = s / 9.f;
                }
        });
    }
    Image d_sq, sigma_sq;
    t0 = ms_now();
    robustness_correspondence(ref_stats.means, ref_stats.stds, comp_means, flow,
                              ts, false, work, d_sq, sigma_sq);
    const double t_corr = ms_now() - t0;
    const int strip_h = kRobustnessRefineStripRows + 2 * kRobustnessRefineHalo;
    t0 = ms_now();
    Image feat = build_robustness_refine_features(ref_stats, comp_means, R_base,
                                                  d_sq, sigma_sq, flow, ts, work,
                                                  0, strip_h);
    const double t_feat_strip = ms_now() - t0;
    const double strips = std::ceil((double)R_base.h / kRobustnessRefineStripRows);

    // ---- strips must equal whole-plane inference -------------------------
    // The replacement network's README documents at length how easily this
    // goes wrong: pad a window past the image edge and bias-driven activations
    // reach the next layer where whole-plane inference has true zeros, which
    // shows up as visible seams (max |diff| 0.95 measured, for one of the two
    // obvious wrong ways to do it). The pointwise model cannot suffer that --
    // its receptive field is one pixel -- but the FEATURE BUILDER still reads
    // a neighbourhood for the structure tensor and the local residual, so its
    // strip decomposition is where an equivalent bug would live instead.
    {
        Image whole = build_robustness_refine_features(ref_stats, comp_means, R_base,
                                                       d_sq, sigma_sq, flow, ts, work,
                                                       0, R_base.h);
        double worst = 0.0;
        int worst_ch = -1, worst_row = -1;
        if (whole.h == R_base.h) {
            for (int y0s = 0; y0s < R_base.h; y0s += kRobustnessRefineStripRows) {
                const int top = std::min(std::max(y0s - kRobustnessRefineHalo, 0),
                                         R_base.h - strip_h);
                Image st = build_robustness_refine_features(ref_stats, comp_means,
                                                            R_base, d_sq, sigma_sq,
                                                            flow, ts, work, top, strip_h);
                if (st.h != strip_h) { worst = -1; break; }
                const int rows = std::min(kRobustnessRefineStripRows, R_base.h - y0s);
                for (int r = 0; r < rows; ++r)
                    for (int x = 0; x < R_base.w; ++x)
                        for (int c = 0; c < kRobustnessRefineChannels; ++c) {
                            const double d = std::fabs(st.at(y0s - top + r, x, c) -
                                                       whole.at(y0s + r, x, c));
                            if (d > worst) { worst = d; worst_ch = c; worst_row = y0s + r; }
                        }
            }
        }
        std::printf("\nstrips: max |feature difference| against whole-plane "
                    "%.3g (channel %d, row %d)\n", worst, worst_ch, worst_row);
    }

    const double feat_mb = (double)strip_h * R_base.w * kRobustnessRefineChannels
                           * sizeof(f32) / (1024.0 * 1024.0);
    const double plane_mb = (double)R_base.h * R_base.w * sizeof(f32)
                            / (1024.0 * 1024.0);
    std::printf("\ncost per comparison frame, %.0f strips of %d rows:\n",
                strips, strip_h);
    std::printf("  analytic mask (baseline, for scale)   %8.1f ms\n", t_base);
    std::printf("  Eq. 6 correspondence                  %8.1f ms\n", t_corr);
    std::printf("  feature builder, all strips           %8.1f ms\n",
                t_feat_strip * strips);
    std::printf("  whole stage incl. scalar-C++ network  %8.1f ms\n", t_refine);
    std::printf("\n  peak extra allocation:\n");
    std::printf("    one feature strip   %7.1f MB  (%d ch x %d x %d)\n",
                feat_mb, kRobustnessRefineChannels, strip_h, R_base.w);
    std::printf("    d_sq + sigma_sq     %7.1f MB\n", 2 * plane_mb);
    std::printf("    comparison means    %7.1f MB\n", 3 * plane_mb);
    std::printf("    refined mask        %7.1f MB\n", plane_mb);
    std::printf("    total               %7.1f MB\n",
                feat_mb + 6 * plane_mb);

    // ---- 4. the GPU path's own code, run here ---------------------------
    //
    // rob_refine_mask is fifteen lines around rr_refine_pixel, and
    // rr_refine_pixel lives in robustness_refine_shared.h, which compiles for
    // both targets. So the entire GPU code path -- its gather, its
    // transcription of Eq. 6, the features, the network, the bounded multiply
    // -- can be run right here against the independently written CPU path.
    //
    // What this does NOT cover is the Metal compile itself and Metal's own
    // floating-point behaviour (fast-math reassociation, a 1-ULP divide). Those
    // need a device. What it does cover is every way the two paths could
    // disagree about WHAT to compute, which is the failure that would ship
    // silently.
    {
        std::vector<float> stdc, diffc, stdall, diffall;
        const int curve_nch = std::max(1, std::min(3, ref_stats.means.c));
        for (int ch = 0; ch < curve_nch; ++ch) {
            fetch_noise_curves_channel(work, ch, stdc, diffc);
            stdall.insert(stdall.end(), stdc.begin(), stdc.end());
            diffall.insert(diffall.end(), diffc.begin(), diffc.end());
        }
        const int curve_n = stdc.empty() ? 0 : (int)stdc.size();
        std::vector<f32> S = robustness_motion_prior(flow, work);
        std::vector<uint32_t> amb = flow.match_ambiguous;
        const bool amb_on = work.flow_reject_ambiguous_enabled &&
                            amb.size() == (size_t)flow.ny * flow.nx;
        if (amb.size() != (size_t)flow.ny * flow.nx)
            amb.assign((size_t)flow.ny * flow.nx, 0u);

        RefineParams rp{};
        rp.h = R_base.h;
        rp.w = R_base.w;
        rp.nch = ref_stats.means.c;
        rp.tile_size = ts;
        rp.flow_ny = flow.ny;
        rp.flow_nx = flow.nx;
        rp.curve_n = curve_n;
        rp.sqrt_index = work.robustness_guide_sqrt ? 1 : 0;
        rp.ambiguous_enabled = amb_on ? 1 : 0;
        rp.r_s1 = work.r_s1;
        rp.alpha = work.noise_alpha_robustness();
        rp.beta = work.noise_beta_robustness();
        rp.kappa = on.robustness_refine_max_reduction;
        rp.deadzone = on.robustness_refine_deadzone;

        if (curve_n <= 0) {
            std::printf("\nGPU parity: no noise curves (1.4 LUT mode?) -- skipped\n");
        } else {
            Image gpu(R_base.h, R_base.w, 1);
            const double t_gpu0 = ms_now();
            parallel_rows(R_base.h, work.num_threads, [&](int y) {
                for (int x = 0; x < R_base.w; ++x)
                    gpu.at(y, x) = rr_refine_pixel(
                        ref_stats.means.data.data(), ref_stats.stds.data.data(),
                        comp_means.data.data(), stdall.data(), diffall.data(),
                        S.data(), flow.flow.data(), amb.data(), kRobustnessRefineWeights,
                        &rp, y, x, R_base.at(y, x));
            });
            const double t_gpu = ms_now() - t_gpu0;

            // A pixel whose predicted reduction lands exactly on the dead zone
            // is a coin flip: one path takes the pass-through branch, the other
            // the scaled one, so they differ by the entire step
            // kappa * deadzone * R at once. That is inherent to a hard
            // threshold, not an arithmetic disagreement, so it is counted
            // separately -- otherwise one boundary pixel in three million makes
            // the max-difference figure look like a defect.
            const double step = (double)on.robustness_refine_max_reduction *
                                (double)on.robustness_refine_deadzone;
            double worst = 0.0;
            int wy = -1, wx = -1;
            size_t disagree = 0, flips = 0;
            for (int y = 0; y < R_base.h; ++y)
                for (int x = 0; x < R_base.w; ++x) {
                    const double a = R_ref.at(y, x), b = gpu.at(y, x);
                    if (!std::isfinite(a) || !std::isfinite(b)) continue;
                    const double d = std::fabs(a - b);
                    const double base = R_base.at(y, x);
                    if (d > 1e-5 &&
                        std::fabs(d - step * base) < 0.02 * step * base + 1e-6) {
                        ++flips;
                        continue;
                    }
                    if (d > worst) { worst = d; wy = y; wx = x; }
                    if (d > 1e-5) ++disagree;
                }
            std::printf("\nGPU parity (the kernel's own shared code, run here):\n");
            std::printf("  max |R_gpu - R_cpu| %.3g at (%d,%d)\n", worst, wy, wx);
            std::printf("  pixels differing by more than 1e-5: %zu of %zu\n",
                        disagree, (size_t)R_base.h * R_base.w);
            std::printf("  dead-zone boundary flips, %.3f of R each: %zu\n",
                        step, flips);
            std::printf("  whole stage as the kernel does it, single pass, no\n"
                        "  feature plane and no readback: %.1f ms on this CPU\n", t_gpu);
        }
    }

    return 0;
}
