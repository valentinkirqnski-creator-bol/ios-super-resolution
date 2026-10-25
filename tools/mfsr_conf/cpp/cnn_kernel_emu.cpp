// Runs the app's ConfNet Metal kernels (rnn15_warp, rnn15_feat, cnn_inputs, cnn_enc, cnn_f1, cnn_head in
// core/HHSRKernels.metal) on the CPU through the MSL shim of tools/robust_nn/cpp/app_kernel_emu15.cpp, with the band
// loop of metal_gpu.mm's cnn_refine_raw_res mirrored line for line. Band buffers are refilled with NaN before every
// band, so a stage that reads a row no earlier stage computed poisons the output.
//   cnn_kernel_emu in.bin out.bin band_rows
// in.bin: int32 gw, gh, tile, ny, nx, homog, combine; f32 noise_a, noise_b, hmat[9];
//         f32 ref_grey[gh][gw], comp_grey[gh][gw], flow[ny][nx][2] (raw px), r0[gh][gw]
// out.bin: f32 R[gh][gw]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "robust_nn_shared.h"

// ---- MSL shim (as app_kernel_emu15.cpp) -----------------------------------------------
using uint = unsigned int;
struct uint2 { uint x, y; };
#define kernel
#define device
#define constant const
#define threadgroup
#define thread
enum class mem_flags { mem_threadgroup };
inline void threadgroup_barrier(mem_flags) {}
using std::min;
inline int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float min(float a, float b) { return a < b ? a : b; }
inline void flow_sample_affine(device const float* flow, device const float* jac,
                               uint fny, uint fnx, float raw_y, float raw_x, float ts,
                               thread float& odx, thread float& ody) {
    int px = clamp(int(std::floor(raw_x / ts)), 0, int(fnx) - 1);
    int py = clamp(int(std::floor(raw_y / ts)), 0, int(fny) - 1);
    uint fi = (uint(py) * fnx + uint(px)) * 2u;
    uint ji = (uint(py) * fnx + uint(px)) * 4u;
    float offx = raw_x / ts - 0.5f - float(px);
    float offy = raw_y / ts - 0.5f - float(py);
    odx = flow[fi + 0u] + jac[ji + 0u] * offx + jac[ji + 1u] * offy;
    ody = flow[fi + 1u] + jac[ji + 2u] * offx + jac[ji + 3u] * offy;
}
#pragma GCC diagnostic ignored "-Wattributes"
#include "conf_nn_shared.h"
#include RNN_APP_BLOCK   // the exact kernel source of HHSRKernels.metal from the robust_nn block on

static_assert(sizeof(CnnBand) == 64, "CnnBand must stay 64 bytes (metal_gpu.mm CnnBandCPU)");
static_assert(sizeof(CnnBand) == sizeof(Rnn15Band), "warp / feature kernels read the band as Rnn15Band");

int main(int argc, char** argv) {
    if (argc < 4) return 1;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    int hdr[7];
    float prm[11];
    if (std::fread(hdr, 4, 7, f) != 7 || std::fread(prm, 4, 11, f) != 11) return 3;
    const int gw = hdr[0], gh = hdr[1], tile = hdr[2], ny = hdr[3], nx = hdr[4], homog = hdr[5], combine = hdr[6];
    const size_t n = (size_t)gw * gh;
    std::vector<float> ref(n), cmp(n), flow(2 * (size_t)ny * nx), r0(n), out(n, -1.f);
    if (std::fread(ref.data(), 4, n, f) != n || std::fread(cmp.data(), 4, n, f) != n ||
        std::fread(flow.data(), 4, flow.size(), f) != flow.size() || std::fread(r0.data(), 4, n, f) != n)
        return 4;
    std::fclose(f);
    std::vector<float> guide(3 * n);
    for (size_t i = 0; i < n; ++i) guide[3 * i] = guide[3 * i + 1] = guide[3 * i + 2] = cmp[i];
    std::vector<float> jac(4 * (size_t)ny * nx, 0.f);
    RnnAppParams p{};
    p.width = gw; p.height = gh; p.flow_step = std::max(1, tile / 2);
    p.noise_a = prm[0]; p.noise_b = prm[1]; p.clip_level = 0.9f;
    p.flow_ny = (uint)ny; p.flow_nx = (uint)nx; p.tile_size = (float)tile; p.affine = 0; p.homog = (uint)homog;
    for (int i = 0; i < 9; ++i) p.hmat[i] = prm[2 + i];
    const int B = std::atoi(argv[3]);

    // ---- metal_gpu.mm cnn_refine_raw_res, band loop ----
    const int d[4] = {CNN_DIL0, CNN_DIL1, CNN_DIL2, CNN_DIL3};
    const int h_f1 = 1, h_e4 = 1, h_e3 = h_e4 + d[3], h_e2 = h_e3 + d[2], h_e1 = h_e2 + d[1], h_x = h_e1 + d[0];
    const int h_f = h_x, h_j = h_f + 4;
    const int h_a = h_e1 > h_e3 ? h_e1 : h_e3;
    const int h_b = h_e2 > h_e4 ? h_e2 : h_e4;
    const size_t Wz = (size_t)gw;
    std::vector<float> gj((B + 2 * h_j) * Wz), gf(RNN_NF * (B + 2 * h_f) * Wz), gx(CNN_NG * (B + 2 * h_x) * Wz),
        ga(CNN_CH * (B + 2 * h_a) * Wz), gb(CNN_CH * (B + 2 * h_b) * Wz);
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    auto rows = [&](int a, int b) { return std::max(0, std::min(gh, b) - std::max(0, a)); };
    for (int y0 = 0; y0 < gh; y0 += B) {
        for (auto* v : {&gj, &gf, &gx, &ga, &gb}) std::fill(v->begin(), v->end(), NaN);
        const int y1 = std::min(gh, y0 + B);
        CnnBand bd{};
        bd.combine = combine;
        bd.jb = std::max(0, y0 - h_j); bd.jn = rows(y0 - h_j, y1 + h_j);
        bd.fb = std::max(0, y0 - h_f); bd.fn = rows(y0 - h_f, y1 + h_f);
        bd.xb = std::max(0, y0 - h_x); bd.xn = rows(y0 - h_x, y1 + h_x);
        const int ab = std::max(0, y0 - h_a), an = rows(y0 - h_a, y1 + h_a);
        const int bb = std::max(0, y0 - h_b), bn = rows(y0 - h_b, y1 + h_b);
        auto grid = [&](int lo, int hi, auto&& k) {
            bd.lo = std::max(0, lo);
            bd.hi = std::min(gh, hi);
            const CnnBand& kb = bd;
            for (int yy = 0; yy < bd.hi - bd.lo; ++yy)
                for (int xx = 0; xx < gw; ++xx) k(kb, uint2{(uint)xx, (uint)yy});
        };
        // the warp / feature kernels take Rnn15Band: same 64-byte layout (static_assert above)
        auto as15 = [](const CnnBand& b) -> const Rnn15Band& { return *reinterpret_cast<const Rnn15Band*>(&b); };
        grid(y0 - h_j, y1 + h_j, [&](const CnnBand& kb, uint2 g) {
            rnn15_warp(guide.data(), gj.data(), flow.data(), jac.data(), p, as15(kb), g); });
        grid(y0 - h_f, y1 + h_f, [&](const CnnBand& kb, uint2 g) {
            rnn15_feat(ref.data(), gj.data(), r0.data(), gf.data(), flow.data(), jac.data(), p, as15(kb), g); });
        grid(y0 - h_x, y1 + h_x, [&](const CnnBand& kb, uint2 g) {
            cnn_inputs(ref.data(), gj.data(), gf.data(), gx.data(), p, kb, g); });
        const int halo_out[4] = {h_e1, h_e2, h_e3, h_e4};
        for (int l = 0; l < 4; ++l) {
            const bool from_x = (l == 0), to_a = (l % 2 == 0);
            bd.layer = l;
            bd.dil = d[l];
            bd.ib = from_x ? bd.xb : (to_a ? bb : ab);
            bd.in_ = from_x ? bd.xn : (to_a ? bn : an);
            bd.ob = to_a ? ab : bb;
            bd.on = to_a ? an : bn;
            const float* in = from_x ? gx.data() : (to_a ? gb.data() : ga.data());
            float* o = to_a ? ga.data() : gb.data();
            grid(y0 - halo_out[l], y1 + halo_out[l], [&](const CnnBand& kb, uint2 g) { cnn_enc(in, o, p, kb, g); });
        }
        bd.ib = bb; bd.in_ = bn; bd.ob = ab; bd.on = an;
        grid(y0 - h_f1, y1 + h_f1, [&](const CnnBand& kb, uint2 g) { cnn_f1(gb.data(), ga.data(), p, kb, g); });
        bd.ib = ab; bd.in_ = an;
        grid(y0, y1, [&](const CnnBand& kb, uint2 g) { cnn_head(ga.data(), r0.data(), out.data(), p, kb, g); });
    }
    FILE* o = std::fopen(argv[2], "wb");
    if (!o) return 5;
    std::fwrite(out.data(), 4, n, o);
    std::fclose(o);
    return 0;
}
