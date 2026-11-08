// Parity of the consensus refiner's three implementations on one real-looking frame:
// CPU cref_refine (core/cref.cpp), the cref_* Metal kernels run through a C++ shim with
// the host's banding (metal_gpu.mm cref_refine_bands_metal logic), and PyTorch (done in
// cref_parity.py from the dumped features). Built with -I<dir holding a RANDOM-weights
// cref_weights.h> ahead of core/.
// usage: cref_parity burst.bin outdir
#include "stages.h"
#include "cref.h"
#include "cref_weights.h"
#include "snr_tuning.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace msl {
typedef unsigned int uint;
struct uint2 { uint x, y; };
using std::sqrt; using std::exp; using std::log;
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline float min(float a, float b) { return a < b ? a : b; }
inline float max(float a, float b) { return a > b ? a : b; }
inline int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
#define kernel
#define device
#define constant const
#define buffer(n)
#define thread_position_in_grid
#include "cref_kernels_emu.inc"
#undef kernel
#undef device
#undef constant
}  // namespace msl
using namespace hhsr;

static void dump(const std::string& p, const std::vector<float>& v) {
    FILE* f = std::fopen(p.c_str(), "wb"); std::fwrite(v.data(), 4, v.size(), f); std::fclose(f);
}

int main(int argc, char** argv) {
    FILE* fi = std::fopen(argv[1], "rb");
    int32_t hi[7]; float hf[5];
    std::fread(hi, 4, 7, fi); std::fread(hf, 4, 5, fi);
    const int N = hi[0], H = hi[1], W = hi[2];
    std::vector<Image> raw(N);
    for (int k = 0; k < N; ++k) { raw[k] = Image(H, W, 1); std::fread(raw[k].data.data(), 4, (size_t)H * W, fi); }
    std::fclose(fi);
    const std::string out = argv[2];
    Config cfg;
    cfg.bayer_mode = true;
    cfg.cfa.p[0][0] = (uint8_t)hi[3]; cfg.cfa.p[0][1] = (uint8_t)hi[4]; cfg.cfa.p[1][0] = (uint8_t)hi[5]; cfg.cfa.p[1][1] = (uint8_t)hi[6];
    for (int c = 0; c < 3; ++c) { cfg.alpha_dng[c] = hf[0]; cfg.beta_dng[c] = hf[1]; cfg.white_balance[c] = hf[2 + c]; }
    cfg.has_noise_profile = true; cfg.raw_prewhitened = true; cfg.burst_frame_count = N;
    cfg.snr_auto_tune = false; cfg.alignment_tile_size = 16;
    tune_config_snr(raw[0], cfg);
    const int ts = cfg.bm_tile_sizes.empty() ? 16 : cfg.bm_tile_sizes[0];
    Image ref_grey = compute_grey(raw[0], true, cfg.grey_method);
    Pyramid pyr = build_pyramid(pad_image_circular(ref_grey, cfg.grey_tile_size(ts)), cfg.bm_factors);
    std::vector<Image> al(N);
    std::vector<FlowField> fl(N);
    Image cons(H, W, 1);
    for (int k = 1; k < N; ++k) {
        Image g = compute_grey(raw[k], true, cfg.grey_method);
        FlowField f = align(pyr, ref_grey, g, cfg, ts);
        f = flow_to_raw_tile_grid(f, H, W, g.h, g.w, ts, cfg.r_Mt, cfg.num_threads, cfg.grey_tile_size(ts));
        clear_align_ref_ica_cache();
        Image a(H, W, 1);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int ty = std::min(y / ts, f.ny - 1), tx = std::min(x / ts, f.nx - 1);
                float sy = std::min(std::max(y + f.dy(ty, tx), 0.f), (float)(H - 1)), sx = std::min(std::max(x + f.dx(ty, tx), 0.f), (float)(W - 1));
                const int y0 = std::min(H - 2, (int)sy), x0 = std::min(W - 2, (int)sx);
                const float fy = sy - y0, fx = sx - x0;
                const float t = g.at(y0, x0) + (g.at(y0, x0 + 1) - g.at(y0, x0)) * fx, b = g.at(y0 + 1, x0) + (g.at(y0 + 1, x0 + 1) - g.at(y0 + 1, x0)) * fx;
                a.at(y, x) = t + (b - t) * fy;
            }
        al[k] = a; fl[k] = f;
        for (size_t i = 0; i < a.data.size(); ++i) cons.data[i] += a.data[i] / (float)(N - 1);
    }
    RefStats rs = init_robustness(raw[0], cfg);
    Image R = compute_robustness(raw[1], rs, fl[1], ts, cfg);
    Image Rg(H, W, 1);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) Rg.at(y, x) = R.at(std::min(R.h - 1, y * R.h / H), std::min(R.w - 1, x * R.w / W));
    std::vector<float> F;
    cref_features(ref_grey, al[1], cons, N - 1, Rg, cfg, 0, H, 0, F);
    dump(out + "/feats.f32", F);
    Image Rc = Rg;
    if (!cref_refine(Rc, ref_grey, al[1], cons, N - 1, cfg)) { std::printf("cref_refine failed\n"); return 1; }
    dump(out + "/rcpu.f32", Rc.data);

    // ---- Metal kernels through the shim, with the host's banding ----
    std::vector<float> Re((size_t)H * W, 0.f);
    const int band = 256, halo = kCrefHalo;
    const Image* src[4] = {&ref_grey, &al[1], &cons, &Rg};
    for (int y0 = 0; y0 < H; y0 += band) {
        const int y1 = std::min(H, y0 + band);
        msl::CrefParams p{};
        p.w = W; p.h = H; p.rows = (y1 - y0) + 2 * halo; p.halo = halo; p.y0 = y0;
        const int slo = std::max(0, y0 - halo - 1), shi = std::min(H - 1, y1 + halo);
        p.slo = slo; p.srows = shi - slo + 1; p.out_rows = y1 - y0;
        p.vlo = std::max(0, halo - y0); p.vhi = std::min((int)p.rows - 1, halo + (H - 1 - y0));
        p.na = std::max(0.f, cfg.alpha_dng[1]); p.nb = std::max(1e-12f, cfg.beta_dng[1]); p.nfr = (float)(N - 1);
        std::vector<float> stg((size_t)4 * p.srows * W), Fb((size_t)6 * p.rows * W), A((size_t)8 * p.rows * W), B(A.size()), O((size_t)p.out_rows * W);
        for (int pl = 0; pl < 4; ++pl)
            std::copy(src[pl]->data.begin() + (size_t)slo * W, src[pl]->data.begin() + (size_t)(slo + p.srows) * W, stg.begin() + (size_t)pl * p.srows * W);
        for (uint32_t y = 0; y < p.rows; ++y) for (uint32_t x = 0; x < p.w; ++x) msl::cref_feat(stg.data(), Fb.data(), p, msl::uint2{x, y});
        const float* Ws[4] = {kCrefW0, kCrefW1, kCrefW2, kCrefW3};
        const float* Bs[4] = {kCrefB0, kCrefB1, kCrefB2, kCrefB3};
        float* ins[4] = {Fb.data(), A.data(), B.data(), A.data()};
        float* outs[4] = {A.data(), B.data(), A.data(), B.data()};
        for (int l = 0; l < 4; ++l) {
            msl::CrefParams q = p; q.cin = l == 0 ? kCrefIn : kCrefCh; q.dil = kCrefDil[l];
            for (uint32_t y = 0; y < q.rows; ++y) for (uint32_t x = 0; x < q.w; ++x) msl::cref_conv(ins[l], outs[l], Ws[l], Bs[l], q, msl::uint2{x, y});
        }
        const float bo = kCrefBO;
        for (uint32_t y = 0; y < p.out_rows; ++y) for (uint32_t x = 0; x < p.w; ++x) msl::cref_head(B.data(), Fb.data(), O.data(), kCrefWO, p, bo, msl::uint2{x, y});
        std::copy(O.begin(), O.end(), Re.begin() + (size_t)y0 * W);
    }
    double md = 0;
    for (size_t i = 0; i < Re.size(); ++i) md = std::max(md, (double)std::fabs(Re[i] - Rc.data[i]));
    std::printf("metal-shim vs CPU: max |diff| %.3g\n", md);
    dump(out + "/rmetal.f32", Re);
    std::printf("H %d W %d\n", H, W);
    return 0;
}
