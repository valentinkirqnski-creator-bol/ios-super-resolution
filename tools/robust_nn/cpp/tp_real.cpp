// tp_real — one training/eval sample for the robust_nn target protocol, built
// from the APP's own stages on a synthetic burst whose exact flow is known.
//
//   tp_real in.bin out.bin
// in : int32 N, H, W, seed, corrupt, cfa00, cfa01, cfa10, cfa11
//      float32 scale, alpha, beta, wb0, wb1, wb2, prewhitened
//      raw[N][H][W]                (as the app's loader delivers it: CFA as given, pre-WB'd if prewhitened)
//      gt[N][H/2][W/2][2]          (exact ref->cmp flow, GREY px; frame 0 ignored)
// Per comparison frame the app's aligner runs on the noisy frames (its real
// errors), optionally followed by deliberate tile corruption (seeded); the
// app's analytic mask (compute_robustness) scores that flow; the app's merge
// gives each frame's contribution with the USED flow and with the EXACT flow.
// out: A_ref,B_ref [Ho][Wo][3]
//      for n=1..N-1: A_used,B_used,A_gt,B_gt [Ho][Wo][3]
//      ref_grey [gh][gw]; for n=1..N-1: grey, r0, flow_x, flow_y, corrupt [gh][gw]
//      (+ residual_flow_x, residual_flow_y [gh][gw] per frame with the homography flag 16)
//      int32 tile
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>
#include <chrono>
static double tnow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define TPT(name) do { if (std::getenv("TP_TIMING")) std::fprintf(stderr, "%-22s %.3fs\n", name, tnow() - _t0); _t0 = tnow(); } while (0)

#include "global_homography.h"
#include "snr_tuning.h"
#include "stages.h"
#include "types.h"

using namespace hhsr;

// The guide-pixel correspondence the app's mask kernels use (RnnAppFlow): nearest tile flow halved to guide px,
// then the global homography when one was estimated. use_h = false: the residual tile flow alone.
static void guide_flow(const FlowField& fl, int tile, int x, int y, float& dx, float& dy, bool use_h = true) {
    const float ry = 2.f * y + 0.5f, rx = 2.f * x + 0.5f;
    int py = std::min((int)(ry / tile), fl.ny - 1), px = std::min((int)(rx / tile), fl.nx - 1);
    dx = 0.5f * fl.dx(py, px);
    dy = 0.5f * fl.dy(py, px);
    if (use_h && fl.has_global_h) {
        float hx, hy;
        apply_homography(fl.global_h, 2.f * (x + dx), 2.f * (y + dy), hx, hy);
        dx = 0.5f * hx - x;
        dy = 0.5f * hy - y;
    }
}

// Corruption magnitude in RAW px: half tiny (0.02..0.1, below/at the aligner's own precision),
// half 0.1..12 log-uniform. Tiny errors are priced by the merge like any other: harmful on sharp
// detail, harmless on flat areas.
static float tiny_or_large(std::mt19937& rng) {
    std::uniform_real_distribution<float> U(0.f, 1.f);
    if (U(rng) < 0.5f) return 0.02f + 0.08f * U(rng);
    return 0.1f * std::exp(U(rng) * std::log(120.f));
}

static void put(FILE* o, const std::vector<float>& v) { std::fwrite(v.data(), 4, v.size(), o); }

static int process(const char* in_path, const char* out_path);

// Persistent mode (--server): reads "in out" path pairs from stdin, one per line, answers "OK <rc>" per job.
// One-time costs (noise-curve Monte-Carlo per noise level, FFT/thread setup) are then paid once per worker.
int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--server") {
        char a[4096], b[4096];
        while (std::scanf("%4095s %4095s", a, b) == 2) {
            const int rc = process(a, b);
            std::printf("OK %d\n", rc);
            std::fflush(stdout);
        }
        return 0;
    }
    if (argc < 3) return 1;
    return process(argv[1], argv[2]);
}

static int process(const char* in_path, const char* out_path) {
    clear_align_ref_ica_cache();   // keyed on the pyramid's ADDRESS: must not leak between jobs
    FILE* f = std::fopen(in_path, "rb");
    if (!f) return 2;
    int32_t hi[9];
    float hf[7];
    if (std::fread(hi, 4, 9, f) != 9 || std::fread(hf, 4, 7, f) != 7) return 3;
    const int N = hi[0], H = hi[1], W = hi[2], seed = hi[3];
    const int corrupt = hi[4] & 1;
    const bool nomerge = (hi[4] & 2) != 0;   // mask-only training: skip all merges (the slow part)
    // v19: straight-edge steps. In corrupted frames, tiles crossed by a dominant straight edge are shifted a
    // fraction of a pixel ACROSS that edge, each by its own amount, so a long straight edge comes out as
    // offset segments at tile borders (the "shifted straight" edges seen at 48 MP).
    const bool edge_step = (hi[4] & 8) != 0;
    // v20: the app's global homography (Config::global_homography_enabled): each comparison grey is warped onto
    // the reference by estimate_global_homography before align(), and the flow carries H. Per frame the output
    // then also holds the residual tile flow (without H), which the net's flow features read.
    const bool homog = (hi[4] & 16) != 0;
    // v21: perfect alignment. Every tile's vector is set to the exact flow at the tile centre (exact for the
    // translation-only bursts this is used with), before any deliberate corruption: any rejection of an
    // uncorrupted frame is then the mask's own error.
    const bool oracle = (hi[4] & 32) != 0;
    // v22: only TINY misalignments in corrupted frames - tiles on edges shifted 0.05-0.5 raw px (log-uniform, any
    // direction), plus the straight-edge steps when flag 8 is set; none of the large / block shifts.
    const bool small_only = (hi[4] & 64) != 0;
    const int Hg = H / 2, Wg = W / 2;
    std::vector<Image> raw(N, Image(H, W, 1));
    for (auto& r : raw)
        if (std::fread(r.data.data(), 4, r.data.size(), f) != r.data.size()) return 4;
    std::vector<float> gt((size_t)N * Hg * Wg * 2);
    if (std::fread(gt.data(), 4, gt.size(), f) != gt.size()) return 5;
    std::fclose(f);

    double _t0 = tnow();
    TPT("read input");
    Config cfg;
    cfg.bayer_mode = true;
    cfg.cfa.p[0][0] = (uint8_t)hi[5];
    cfg.cfa.p[0][1] = (uint8_t)hi[6];
    cfg.cfa.p[1][0] = (uint8_t)hi[7];
    cfg.cfa.p[1][1] = (uint8_t)hi[8];
    cfg.scale = hf[0];
    for (int c = 0; c < 3; ++c) {
        cfg.alpha_dng[c] = hf[1];
        cfg.beta_dng[c] = hf[2];
        cfg.white_balance[c] = hf[3 + c];
    }
    cfg.has_noise_profile = true;
    cfg.raw_prewhitened = hf[6] != 0.f;
    Config work = cfg;
    work.burst_frame_count = N;
    // The SNR auto-tune builds a Monte-Carlo noise curve (~4 s) and its result is the same every time a burst is
    // reused, so the caller may pass the cached result: TP_TUNE="Ts k_detail k_denoise D_th D_tr".
    if (const char* tc = std::getenv("TP_TUNE")) {
        int Ts = 16;
        std::sscanf(tc, "%d %f %f %f %f", &Ts, &work.k_detail, &work.k_denoise, &work.D_th, &work.D_tr);
        work.snr_auto_tune = false;
        work.alignment_tile_size = Ts;
        tune_config_snr(raw[0], work);          // applies the manual tile size only
    } else {
        tune_config_snr(raw[0], work);
    }
    if (std::getenv("TP_PRINT_TUNE"))
        std::fprintf(stderr, "TUNE %d %.9g %.9g %.9g %.9g\n", work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0],
                     work.k_detail, work.k_denoise, work.D_th, work.D_tr);
    const int tile = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    TPT("tune_config_snr");

    const Image& ref = raw[0];
    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Pyramid ref_pyr = build_pyramid(pad_image_circular(ref_grey, work.grey_tile_size(tile)), work.bm_factors);
    TPT("ref grey+pyramid");
    RefStats ref_stats = init_robustness(ref, work);
    TPT("init_robustness");
    CovField ref_covs;
    if (!nomerge) ref_covs = estimate_kernels(ref, work);
    const int Ho = (int)std::lround(work.scale * H), Wo = (int)std::lround(work.scale * W);

    FILE* o = std::fopen(out_path, "wb");
    if (!nomerge) {
        Image num(Ho, Wo, 3), den(Ho, Wo, 3);
        merge_ref(ref, ref_covs, num, den, work, nullptr);
        put(o, num.data);
        put(o, den.data);
    }
    auto guide_grey = [&](const Image& r) {
        Image g = compute_guide(r, work);
        std::vector<float> v((size_t)Hg * Wg);
        for (int y = 0; y < Hg; ++y)
            for (int x = 0; x < Wg; ++x)
                v[(size_t)y * Wg + x] = 0.25f * g.at(y, x, 0) + 0.5f * g.at(y, x, 1) + 0.25f * g.at(y, x, 2);
        return v;
    };
    std::mt19937 rng((uint32_t)seed);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    Image ones(H, W, 1);
    for (auto& x : ones.data) x = 1.f;
    std::vector<std::vector<float>> per;   // grey, r0, fx, fy, corr per frame
    // v28: TP_SMALL_CONSISTENT=1 - some slipped tiles get ONE slip for the whole burst, applied (+-15%) in 40-95% of
    // the frames, so the merge shows a clear double / shifted / zippery edge instead of a blur. Own rng: the main
    // draws are unchanged.
    const bool small_cons = std::getenv("TP_SMALL_CONSISTENT") != nullptr;
    std::mt19937 crng((uint32_t)seed ^ 0x9e3779b9u);
    std::normal_distribution<float> Nc(0.f, 1.f);
    std::vector<float> cs_dx, cs_dy, cs_pf;
    for (int n = 1; n < N; ++n) {
        const Image& comp = raw[n];
        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
        TPT("comp grey");
        const bool warp_h = homog && comp_grey.h == comp.h && comp_grey.w == comp.w &&
                            ref_grey.h == ref.h && ref_grey.w == ref.w;   // full-resolution grey only, as in the app
        f32 Hm[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        FlowField flow;
        if (warp_h) {
            estimate_global_homography(ref_grey, comp_grey, work, Hm);
            flow = align(ref_pyr, ref_grey, warp_grey_by_homography(comp_grey, Hm), work, tile);
        } else {
            flow = align(ref_pyr, ref_grey, comp_grey, work, tile);
        }
        TPT("align");
        flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w, tile, work.r_Mt,
                                     work.num_threads, work.grey_tile_size(tile));
        if (warp_h) {
            for (int i = 0; i < 9; ++i) flow.global_h[i] = Hm[i];
            flow.has_global_h = true;
        }
        if (oracle) {
            const float* g = gt.data() + (size_t)n * Hg * Wg * 2;
            for (int ty = 0; ty < flow.ny; ++ty)
                for (int tx = 0; tx < flow.nx; ++tx) {
                    const int gy = std::min((int)((ty + 0.5f) * tile * 0.5f), Hg - 1);
                    const int gx = std::min((int)((tx + 0.5f) * tile * 0.5f), Wg - 1);
                    const size_t i = ((size_t)ty * flow.nx + tx) * 2, s = ((size_t)gy * Wg + gx) * 2;
                    flow.flow[i] = 2.f * g[s];            // grey px -> raw px
                    flow.flow[i + 1] = 2.f * g[s + 1];
                }
            flow.global_h[0] = flow.global_h[4] = flow.global_h[8] = 1.f;
            flow.global_h[1] = flow.global_h[2] = flow.global_h[3] = flow.global_h[5] = 0.f;
            flow.global_h[6] = flow.global_h[7] = 0.f;
            flow.has_global_h = false;
        }
        std::vector<float> tcorr((size_t)flow.ny * flow.nx, 0.f);
        // Per-tile edge strength of the reference (mean squared grey gradient) as a rank in [0,1),
        // used for the EXTRA edge-targeted corruption below (the random corruption is unchanged).
        std::vector<float> erank((size_t)flow.ny * flow.nx, 0.5f);
        // per-tile structure tensor of the reference grey (edge_step only): dominant edge normal + coherence
        std::vector<double> sxx(erank.size(), 0.0), syy(erank.size(), 0.0), sxy(erank.size(), 0.0);
        {
            std::vector<double> esum(erank.size(), 0.0), ecnt(erank.size(), 0.0);
            for (int gy = 1; gy + 1 < ref_grey.h; ++gy)
                for (int gx = 1; gx + 1 < ref_grey.w; ++gx) {
                    const float dx = ref_grey.at(gy, gx + 1) - ref_grey.at(gy, gx - 1);
                    const float dy = ref_grey.at(gy + 1, gx) - ref_grey.at(gy - 1, gx);
                    const int ty = std::min((int)((2.f * gy + 0.5f) / tile), flow.ny - 1);
                    const int tx = std::min((int)((2.f * gx + 0.5f) / tile), flow.nx - 1);
                    esum[(size_t)ty * flow.nx + tx] += dx * dx + dy * dy;
                    ecnt[(size_t)ty * flow.nx + tx] += 1.0;
                    sxx[(size_t)ty * flow.nx + tx] += dx * dx;
                    syy[(size_t)ty * flow.nx + tx] += dy * dy;
                    sxy[(size_t)ty * flow.nx + tx] += dx * dy;
                }
            std::vector<size_t> idx(erank.size());
            for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                return esum[a] / std::max(1.0, ecnt[a]) < esum[b] / std::max(1.0, ecnt[b]);
            });
            for (size_t r = 0; r < idx.size(); ++r) erank[idx[r]] = (float)r / (float)idx.size();
        }
        const bool frame_corrupted = corrupt && U(rng) > 0.25f;
        if (frame_corrupted && small_only) {
            // TP_SMALL_RANGE="lo,hi" (raw px, v24) overrides the 0.05-0.5 px log-uniform slip range
            float slo = 0.05f, shi = 0.5f;
            if (const char* sr = std::getenv("TP_SMALL_RANGE")) std::sscanf(sr, "%f,%f", &slo, &shi);
            // v28: TP_SMALL_ANY=1 corrupts tiles anywhere (not only the stronger-edge half), TP_SMALL_UNIFORM=1 draws
            // the slip uniformly in [lo, hi] instead of log-uniformly
            const bool any_tile = std::getenv("TP_SMALL_ANY") != nullptr;
            const bool uni = std::getenv("TP_SMALL_UNIFORM") != nullptr;
            const float pfrac = (0.1f + 0.3f * U(rng)) * (small_cons ? 0.5f : 1.f);
            for (int ty = 0; ty < flow.ny; ++ty)
                for (int tx = 0; tx < flow.nx; ++tx) {
                    const size_t t = (size_t)ty * flow.nx + tx;
                    if ((!any_tile && erank[t] < 0.5f) || U(rng) >= pfrac) continue;
                    const float u0 = U(rng);
                    const float mag = uni ? slo + (shi - slo) * u0 : slo * std::exp(u0 * std::log(shi / slo));
                    const float ang = 6.2831853f * U(rng);
                    flow.flow[2 * t] += mag * std::cos(ang);
                    flow.flow[2 * t + 1] += mag * std::sin(ang);
                    tcorr[t] = 1.f;
                }
        }
        if (corrupt && small_only && small_cons) {
            float slo = 0.05f, shi = 0.5f;
            if (const char* sr = std::getenv("TP_SMALL_RANGE")) std::sscanf(sr, "%f,%f", &slo, &shi);
            const bool any_tile = std::getenv("TP_SMALL_ANY") != nullptr;
            const size_t nt = (size_t)flow.nx * flow.ny;
            if (cs_pf.size() != nt) {   // drawn once per burst
                cs_dx.assign(nt, 0.f); cs_dy.assign(nt, 0.f); cs_pf.assign(nt, 0.f);
                const float cfrac = 0.05f + 0.15f * U(crng);
                for (size_t t = 0; t < nt; ++t) {
                    if ((!any_tile && erank[t] < 0.5f) || U(crng) >= cfrac) continue;
                    const float mag = slo + (shi - slo) * U(crng), ang = 6.2831853f * U(crng);
                    cs_dx[t] = mag * std::cos(ang); cs_dy[t] = mag * std::sin(ang);
                    cs_pf[t] = 0.4f + 0.55f * U(crng);
                }
            }
            for (size_t t = 0; t < nt; ++t) {
                if (cs_pf[t] <= 0.f || U(crng) >= cs_pf[t]) continue;
                const float j = std::min(1.5f, std::max(0.5f, 1.f + 0.15f * Nc(crng)));
                flow.flow[2 * t] += cs_dx[t] * j;
                flow.flow[2 * t + 1] += cs_dy[t] * j;
                tcorr[t] = 1.f;
            }
        }
        if (frame_corrupted && !small_only) {   // deliberate corruption on top of the aligner's own errors
            const bool block = U(rng) < 0.3f;
            const float frac = 0.05f + 0.35f * U(rng);
            int by0 = 0, bx0 = 0, bh = 0, bw = 0;
            float bmag = 0, bang = 0;
            if (block) {
                bh = 1 + (int)(U(rng) * std::max(1, flow.ny / 2));
                bw = 1 + (int)(U(rng) * std::max(1, flow.nx / 2));
                by0 = (int)(U(rng) * std::max(1, flow.ny - bh + 1));
                bx0 = (int)(U(rng) * std::max(1, flow.nx - bw + 1));
                bmag = tiny_or_large(rng);
                bang = 6.2831853f * U(rng);
            }
            for (int ty = 0; ty < flow.ny; ++ty)
                for (int tx = 0; tx < flow.nx; ++tx) {
                    bool hit;
                    float mag, ang;
                    if (block) {
                        hit = ty >= by0 && ty < by0 + bh && tx >= bx0 && tx < bx0 + bw;
                        mag = bmag;
                        ang = bang;
                    } else {
                        hit = U(rng) < frac;
                        mag = tiny_or_large(rng);
                        ang = 6.2831853f * U(rng);
                    }
                    if (!hit) continue;
                    const size_t i = ((size_t)ty * flow.nx + tx) * 2;
                    flow.flow[i] += mag * std::cos(ang);
                    flow.flow[i + 1] += mag * std::sin(ang);
                    tcorr[(size_t)ty * flow.nx + tx] = 1.f;
                }
        }
        if (frame_corrupted && !small_only && U(rng) < 0.7f) {   // PLUS extra corrupted tiles on the strongest edges
            const float efrac = 0.05f + 0.20f * U(rng);
            for (int ty = 0; ty < flow.ny; ++ty)
                for (int tx = 0; tx < flow.nx; ++tx) {
                    const size_t t = (size_t)ty * flow.nx + tx;
                    if (erank[t] < 0.7f || U(rng) >= efrac) continue;
                    const float mag = tiny_or_large(rng), ang = 6.2831853f * U(rng);
                    flow.flow[2 * t] += mag * std::cos(ang);
                    flow.flow[2 * t + 1] += mag * std::sin(ang);
                    tcorr[t] = 1.f;
                }
        }
        if (frame_corrupted && edge_step) {   // straight-edge steps (see edge_step above)
            const float pfrac = 0.3f + 0.5f * U(rng);
            for (int ty = 0; ty < flow.ny; ++ty)
                for (int tx = 0; tx < flow.nx; ++tx) {
                    const size_t t = (size_t)ty * flow.nx + tx;
                    const double a = sxx[t], b = syy[t], c = sxy[t];
                    const double tr = a + b, dd = std::sqrt((a - b) * (a - b) + 4.0 * c * c);
                    const double l1 = 0.5 * (tr + dd), l2 = 0.5 * (tr - dd);
                    const double coh = tr > 1e-12 ? ((l1 - l2) / (l1 + l2)) * ((l1 - l2) / (l1 + l2)) : 0.0;
                    if (erank[t] < 0.6f || coh < 0.5 || U(rng) >= pfrac) continue;
                    // edge normal = eigenvector of l1 (the gradient direction)
                    float nx = (float)(l1 - b), ny = (float)c;
                    if (std::fabs(nx) + std::fabs(ny) < 1e-12f) { nx = (float)c; ny = (float)(l1 - a); }
                    const float nn = std::sqrt(nx * nx + ny * ny) + 1e-12f;
                    nx /= nn; ny /= nn;
                    const float mag = 0.08f * std::exp(U(rng) * std::log(7.5f)) * (U(rng) < 0.5f ? -1.f : 1.f);
                    const float tang = 0.2f * mag * (2.f * U(rng) - 1.f);
                    flow.flow[2 * t] += mag * nx - tang * ny;
                    flow.flow[2 * t + 1] += mag * ny + tang * nx;
                    tcorr[t] = 1.f;
                }
        }
        TPT("flow grid+corrupt");
        Image rob = compute_robustness(comp, ref_stats, flow, tile, work);
        TPT("compute_robustness");
        if (!nomerge) {
        CovField covs = estimate_kernels(comp, work);
        Image nu(Ho, Wo, 3), du(Ho, Wo, 3), ng(Ho, Wo, 3), dg(Ho, Wo, 3);
        merge_comp(comp, flow, covs, ones, tile, nu, du, work);
        FlowField gfl;
        gfl.dense_h = Hg;
        gfl.dense_w = Wg;
        gfl.dense_sx = gfl.dense_sy = 2.f;
        const float* src = gt.data() + (size_t)n * Hg * Wg * 2;
        gfl.dense.assign(src, src + (size_t)Hg * Wg * 2);
        merge_comp(comp, gfl, covs, ones, tile, ng, dg, work);
        put(o, nu.data);
        put(o, du.data);
        put(o, ng.data);
        put(o, dg.data);
        }
        std::vector<float> r0((size_t)Hg * Wg), fx(r0.size()), fy(r0.size()), cm(r0.size()),
            rfx(homog ? r0.size() : 0), rfy(homog ? r0.size() : 0);
        for (int y = 0; y < Hg; ++y)
            for (int x = 0; x < Wg; ++x) {
                const size_t i = (size_t)y * Wg + x;
                r0[i] = rob.at(std::min(2 * y, rob.h - 1), std::min(2 * x, rob.w - 1));
                guide_flow(flow, tile, x, y, fx[i], fy[i]);
                if (homog) guide_flow(flow, tile, x, y, rfx[i], rfy[i], false);
                const int py = std::min((int)((2.f * y + 0.5f) / tile), flow.ny - 1);
                const int px = std::min((int)((2.f * x + 0.5f) / tile), flow.nx - 1);
                cm[i] = tcorr[(size_t)py * flow.nx + px];
            }
        per.push_back(guide_grey(comp));
        per.push_back(r0);
        per.push_back(fx);
        per.push_back(fy);
        per.push_back(cm);
        if (homog) {
            per.push_back(rfx);
            per.push_back(rfy);
        }
    }
    TPT("per-frame rest");
    put(o, guide_grey(ref));
    for (auto& v : per) put(o, v);
    int32_t t = tile;
    std::fwrite(&t, 4, 1, o);
    std::fclose(o);
    return 0;
}
