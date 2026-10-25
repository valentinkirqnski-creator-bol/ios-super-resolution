// conf_harness -- the APP's own stages on one synthetic burst, for tools/mfsr_conf.
//
// Runs exactly what the app runs before the merge (core/: tune_config_snr, the
// aligner with the optional global homography, the analytic robustness mask,
// the kernel covariances), lets the caller inject a deliberate alignment error
// AFTER the aligner and BEFORE the mask, and returns everything the PyTorch
// side needs to reproduce the app's merge exactly for any robustness map
// (mfsr/merge.py is a parity-tested replica of merge.cpp's accumulate_comp /
// accumulate_ref; it needs the tile flow, the homography and the covariances).
//
//   conf_harness in.bin out.bin          one job
//   conf_harness --server                "in out" path pairs on stdin, "OK <rc>" per job
//
// in.bin
//   int32  N, H, W, seed, flags, cfa00, cfa01, cfa10, cfa11, tile (0 = SNR auto), 6 x reserved   (16)
//   f32    scale, alpha, beta, wb0, wb1, wb2, prewhitened,
//          k_detail, k_denoise, D_th, D_tr, k_stretch, k_shrink (used when tile > 0), 3 x reserved (16)
//   f32    raw[N][H][W]          noisy frames as the app's loader delivers them (CFA as given)
//   f32    gt[N][H][W][2]        exact ref->frame displacement per reference RAW pixel, raw px (frame 0 ignored)
//   f32    pert[N][H][W][2]      deliberate error, raw px, sampled at each tile centre and ADDED to its vector
//   f32    hpert[N][9]           deliberate error as a homography composed AFTER the flow (row-major, identity = none)
// flags
//   1   global homography (Config::global_homography_enabled path: H estimated, comp grey warped, then align)
//   2   oracle tiles: every tile vector = the exact displacement at the tile centre (replaces the aligner)
//   4   merge parity: also output the app's merge contributions with R = 1 (merge_ref / merge_comp)
//   8   with flag 4: f32 rmap[N-1][H][W] follows hpert in in.bin and the parity merge_comp uses it as the
//       robustness mask (raw resolution) instead of R = 1 -- tests the recombination against the real merge
// out.bin
//   int32  tile, ny, nx, cov_h, cov_w
//   f32    k_detail, k_denoise, D_th, D_tr
//   f32    ref_grey_full[H][W], ref_cov[cov_h][cov_w][4]
//   per comparison frame n = 1..N-1:
//     f32  flow[ny][nx][2] (USED: aligner/oracle + pert), flow_aligner[ny][nx][2] (before pert),
//          H[9] (USED: estimated homography composed with hpert), has_h,
//          r0[H][W] (the app's analytic robustness for the used correspondence, raw resolution),
//          cov[cov_h][cov_w][4], grey_full[H][W]
//   if flags & 4:  A_ref, B_ref [Ho][Wo][3]; per frame A, B [Ho][Wo][3]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "global_homography.h"
#include "snr_tuning.h"
#include "stages.h"
#include "types.h"

using namespace hhsr;

static void put(FILE* o, const std::vector<float>& v) { std::fwrite(v.data(), 4, v.size(), o); }
static void put(FILE* o, const float* p, size_t n) { std::fwrite(p, 4, n, o); }

static void matmul3(const f32 A[9], const f32 B[9], f32 C[9]) {   // C = A * B (B applied first)
    f32 t[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            t[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c] + A[r * 3 + 1] * B[1 * 3 + c] + A[r * 3 + 2] * B[2 * 3 + c];
    std::memcpy(C, t, sizeof(t));
}

static bool is_identity(const f32 Hm[9]) {
    static const f32 I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < 9; ++i)
        if (std::fabs(Hm[i] - I[i]) > 1e-9f) return false;
    return true;
}

static int process(const char* in_path, const char* out_path) {
    clear_align_ref_ica_cache();   // keyed on the pyramid's ADDRESS: must not leak between jobs
    FILE* f = std::fopen(in_path, "rb");
    if (!f) return 2;
    int32_t hi[16];
    float hf[16];
    if (std::fread(hi, 4, 16, f) != 16 || std::fread(hf, 4, 16, f) != 16) { std::fclose(f); return 3; }
    const int N = hi[0], H = hi[1], W = hi[2], flags = hi[4], tile_force = hi[9];
    const bool homog = (flags & 1) != 0, oracle = (flags & 2) != 0, parity = (flags & 4) != 0;
    if (N < 2 || H < 32 || W < 32 || (H & 1) || (W & 1)) { std::fclose(f); return 4; }
    const size_t HW = (size_t)H * W;
    std::vector<Image> raw(N, Image(H, W, 1));
    for (auto& r : raw)
        if (std::fread(r.data.data(), 4, HW, f) != HW) { std::fclose(f); return 5; }
    std::vector<float> gt((size_t)N * HW * 2), pert((size_t)N * HW * 2), hpert((size_t)N * 9);
    if (std::fread(gt.data(), 4, gt.size(), f) != gt.size() ||
        std::fread(pert.data(), 4, pert.size(), f) != pert.size() ||
        std::fread(hpert.data(), 4, hpert.size(), f) != hpert.size()) { std::fclose(f); return 6; }
    const bool use_rmap = parity && (flags & 8) != 0;
    std::vector<Image> rmap;
    if (use_rmap) {
        rmap.assign(N - 1, Image(H, W, 1));
        for (auto& r : rmap)
            if (std::fread(r.data.data(), 4, HW, f) != HW) { std::fclose(f); return 10; }
    }
    std::fclose(f);

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
    if (tile_force > 0) {
        // Caller-fixed tuning (reproducible, and skips the ~4 s Monte-Carlo noise curve per job).
        work.snr_auto_tune = false;
        work.alignment_tile_size = tile_force;
        work.k_detail = hf[7];
        work.k_denoise = hf[8];
        work.D_th = hf[9];
        work.D_tr = hf[10];
        if (hf[11] > 0.f) work.k_stretch = hf[11];
        if (hf[12] > 0.f) work.k_shrink = hf[12];
    }
    tune_config_snr(raw[0], work);
    const int tile = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];

    const Image& ref = raw[0];
    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Pyramid ref_pyr = build_pyramid(pad_image_circular(ref_grey, work.grey_tile_size(tile)), work.bm_factors);
    RefStats ref_stats = init_robustness(ref, work);
    CovField ref_covs = estimate_kernels(ref, work);
    const int Ho = (int)std::lround(work.scale * H), Wo = (int)std::lround(work.scale * W);

    // Full-resolution grey for the outputs (the FFT grey is full resolution; resample if a build differs).
    auto full_grey = [&](const Image& g) {
        std::vector<float> v(HW);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int gy = std::min(g.h - 1, y * g.h / H), gx = std::min(g.w - 1, x * g.w / W);
                v[(size_t)y * W + x] = g.at(gy, gx, 0);
            }
        return v;
    };
    // Raw pixel at a tile centre, clamped into the frame.
    auto centre = [&](int t, int dim) { return std::min(dim - 1, std::max(0, (int)((t + 0.5f) * tile))); };

    struct Frame {
        FlowField flow;
        std::vector<float> flow_al;
        Image rob;
        CovField covs;
        std::vector<float> grey;
    };
    std::vector<Frame> fr(N);
    for (int n = 1; n < N; ++n) {
        const Image& comp = raw[n];
        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
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
        flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w, tile, work.r_Mt,
                                     work.num_threads, work.grey_tile_size(tile));
        if (warp_h) {
            for (int i = 0; i < 9; ++i) flow.global_h[i] = Hm[i];
            flow.has_global_h = true;
        }
        const float* g = gt.data() + (size_t)n * HW * 2;
        const float* pe = pert.data() + (size_t)n * HW * 2;
        if (oracle) {
            for (int ty = 0; ty < flow.ny; ++ty)
                for (int tx = 0; tx < flow.nx; ++tx) {
                    const size_t s = ((size_t)centre(ty, H) * W + centre(tx, W)) * 2;
                    const size_t i = ((size_t)ty * flow.nx + tx) * 2;
                    flow.flow[i] = g[s];
                    flow.flow[i + 1] = g[s + 1];
                }
            for (int i = 0; i < 9; ++i) flow.global_h[i] = (i % 4 == 0) ? 1.f : 0.f;
            flow.has_global_h = false;
        }
        fr[n].flow_al = flow.flow;
        for (int ty = 0; ty < flow.ny; ++ty)
            for (int tx = 0; tx < flow.nx; ++tx) {
                const size_t s = ((size_t)centre(ty, H) * W + centre(tx, W)) * 2;
                const size_t i = ((size_t)ty * flow.nx + tx) * 2;
                flow.flow[i] += pe[s];
                flow.flow[i + 1] += pe[s + 1];
            }
        const float* hp = hpert.data() + (size_t)n * 9;
        if (!is_identity(hp)) {
            f32 C[9];
            matmul3(hp, flow.global_h, C);   // lr_mov = Hp(H(lr + flow))
            for (int i = 0; i < 9; ++i) flow.global_h[i] = C[i];
            flow.has_global_h = true;
        }
        fr[n].rob = compute_robustness(comp, ref_stats, flow, tile, work);
        fr[n].covs = estimate_kernels(comp, work);
        fr[n].grey = full_grey(comp_grey);
        fr[n].flow = std::move(flow);
    }

    FILE* o = std::fopen(out_path, "wb");
    if (!o) return 7;
    const int32_t hdr[5] = {tile, fr[1].flow.ny, fr[1].flow.nx, ref_covs.h, ref_covs.w};
    std::fwrite(hdr, 4, 5, o);
    const float tune[4] = {work.k_detail, work.k_denoise, work.D_th, work.D_tr};
    std::fwrite(tune, 4, 4, o);
    put(o, full_grey(ref_grey));
    put(o, ref_covs.cov);
    for (int n = 1; n < N; ++n) {
        const Frame& F = fr[n];
        if (F.flow.ny != hdr[1] || F.flow.nx != hdr[2]) { std::fclose(o); return 9; }
        put(o, F.flow.flow);
        put(o, F.flow_al);
        put(o, F.flow.global_h, 9);
        const float hh = F.flow.has_global_h ? 1.f : 0.f;
        put(o, &hh, 1);
        std::vector<float> r0(HW, 0.f);   // the mask may come back at guide resolution on some paths: resample
        const Image& R = F.rob;
        if (R.h > 0)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const int ry = std::min(R.h - 1, y * R.h / H), rx = std::min(R.w - 1, x * R.w / W);
                    r0[(size_t)y * W + x] = R.at(ry, rx, 0);
                }
        put(o, r0);
        if (F.covs.h != ref_covs.h || F.covs.w != ref_covs.w) { std::fclose(o); return 8; }
        put(o, F.covs.cov);
        put(o, F.grey);
    }
    if (parity) {
        Image ones(H, W, 1);
        std::fill(ones.data.begin(), ones.data.end(), 1.f);
        Image num(Ho, Wo, 3), den(Ho, Wo, 3);
        merge_ref(ref, ref_covs, num, den, work, nullptr);
        put(o, num.data);
        put(o, den.data);
        for (int n = 1; n < N; ++n) {
            Image nu(Ho, Wo, 3), du(Ho, Wo, 3);
            merge_comp(raw[n], fr[n].flow, fr[n].covs, use_rmap ? rmap[n - 1] : ones, tile, nu, du, work);
            put(o, nu.data);
            put(o, du.data);
        }
    }
    std::fclose(o);
    return 0;
}

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
