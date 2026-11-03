// Runs the app's v15 banded Metal kernels (rnn15_*, core/HHSRKernels.metal) on the CPU through the
// same MSL shim as app_kernel_emu.cpp, with the band loop of metal_gpu.mm's rnn15_refine_raw_res
// mirrored line for line. Band buffers are refilled with NaN before every band, so a stage that
// reads a row no earlier stage computed poisons the output instead of silently reusing stale data.
//   app_kernel_emu15 in.bin out.bin [band_rows lo hi]   (in.bin as for app_kernel_emu)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "robust_nn_shared.h"

// ---- MSL shim (as app_kernel_emu.cpp) ----------------------------------------------
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
#include RNN_APP_BLOCK   // the exact kernel source appended to HHSRKernels.metal

static_assert(sizeof(Rnn15Band) == 64, "Rnn15Band must stay 64 bytes (metal_gpu.mm Rnn15BandCPU)");

int main(int argc, char** argv) {
    FILE* f = std::fopen(argv[1], "rb");
    int hdr[3];
    float prm[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::fread(prm, 4, 3, f) != 3) return 3;
    const int W = hdr[0], H = hdr[1], fs = hdr[2];
    const size_t n = (size_t)W * H;
    std::vector<float> ref(n), cmp(n), flow(2 * n), r0(n), out(n, -1.f);
    if (std::fread(ref.data(), 4, n, f) != n || std::fread(cmp.data(), 4, n, f) != n ||
        std::fread(flow.data(), 4, 2 * n, f) != 2 * n || std::fread(r0.data(), 4, n, f) != n)
        return 4;
    std::fclose(f);
    std::vector<float> guide(3 * n);
    for (size_t i = 0; i < n; ++i) guide[3 * i] = guide[3 * i + 1] = guide[3 * i + 2] = cmp[i];
    const int ny = (H + fs - 1) / fs, nx = (W + fs - 1) / fs;
    std::vector<float> tflow(2 * (size_t)ny * nx), jac(4 * (size_t)ny * nx, 0.f);
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            size_t s = 2 * ((size_t)std::min(ty * fs, H - 1) * W + std::min(tx * fs, W - 1));
            tflow[2 * ((size_t)ty * nx + tx)] = 2.f * flow[s];
            tflow[2 * ((size_t)ty * nx + tx) + 1] = 2.f * flow[s + 1];
        }
    RnnAppParams p{};
    p.width = W; p.height = H; p.flow_step = fs;
    p.noise_a = prm[0]; p.noise_b = prm[1]; p.clip_level = prm[2];
    p.flow_ny = ny; p.flow_nx = nx; p.tile_size = 2.f * fs; p.affine = 0; p.homog = 0;
    const int B = argc > 3 ? std::atoi(argv[3]) : 32;
    p.sharp_lo = argc > 5 ? std::atof(argv[4]) : 0.f;
    p.sharp_hi = argc > 5 ? std::atof(argv[5]) : 0.f;

    const size_t Wz = (size_t)W;
    std::vector<float> gj((B + 26) * Wz), gf(RNN_NF * (B + 18) * Wz), gx(RNN15_CIN * (B + 16) * Wz),
        ga(RNN15_CH * (B + 14) * Wz), gb(RNN15_CH * (B + 10) * Wz);
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    auto rows = [&](int a, int b) { return std::max(0, std::min(H, b) - std::max(0, a)); };
    const int halo[4] = {7, 5, 1, 0};
    for (int y0 = 0; y0 < H; y0 += B) {
        for (auto* v : {&gj, &gf, &gx, &ga, &gb}) std::fill(v->begin(), v->end(), NaN);
        const int y1 = std::min(H, y0 + B);
        Rnn15Band bd{};   // same fields as metal_gpu.mm's Rnn15BandCPU
        bd.jb = std::max(0, y0 - 13); bd.jn = rows(y0 - 13, y1 + 13);
        bd.fb = std::max(0, y0 - 9);  bd.fn = rows(y0 - 9, y1 + 9);
        bd.xb = std::max(0, y0 - 8);  bd.xn = rows(y0 - 8, y1 + 8);
        const int ab = std::max(0, y0 - 7), an = rows(y0 - 7, y1 + 7);
        const int bb = std::max(0, y0 - 5), bn = rows(y0 - 5, y1 + 5);
        auto grid = [&](int lo, int hi, auto&& k) {
            bd.lo = std::max(0, lo);
            bd.hi = std::min(H, hi);
            if (std::getenv("EMU_TRACE"))
                std::fprintf(stderr, "band %d rows [%d,%d) layer %d\n", y0, bd.lo, bd.hi, bd.layer);
            const Rnn15Band& kb = bd;
            for (int yy = 0; yy < bd.hi - bd.lo; ++yy)
                for (int xx = 0; xx < W; ++xx) k(kb, uint2{(uint)xx, (uint)yy});
        };
        grid(y0 - 13, y1 + 13, [&](const Rnn15Band& kb, uint2 g) {
            rnn15_warp(guide.data(), gj.data(), tflow.data(), jac.data(), p, kb, g); });
        grid(y0 - 9, y1 + 9, [&](const Rnn15Band& kb, uint2 g) {
            rnn15_feat(ref.data(), gj.data(), r0.data(), gf.data(), tflow.data(), jac.data(), p, kb, g); });
        grid(y0 - 8, y1 + 8, [&](const Rnn15Band& kb, uint2 g) {
            rnn15_inputs(ref.data(), gj.data(), gf.data(), r0.data(), gx.data(), p, kb, g); });
        const int dil[4] = {1, 2, 4, 1};
        for (int l = 0; l < 4; ++l) {
            const bool from_x = (l == 0), to_a = (l % 2 == 0);
            bd.layer = l + 1;
            bd.dil = dil[l];
            bd.ib = from_x ? bd.xb : (to_a ? bb : ab);
            bd.in_ = from_x ? bd.xn : (to_a ? bn : an);
            bd.ob = to_a ? ab : bb;
            bd.on = to_a ? an : bn;
            const float* in = from_x ? gx.data() : (to_a ? gb.data() : ga.data());
            float* o = to_a ? ga.data() : gb.data();
            grid(y0 - halo[l], y1 + halo[l], [&](const Rnn15Band& kb, uint2 g) { rnn15_conv(in, o, p, kb, g); });
        }
        bd.ib = bb;
        bd.in_ = bn;
        grid(y0, y1, [&](const Rnn15Band& kb, uint2 g) { rnn15_out(gb.data(), gx.data(), out.data(), p, kb, g); });
    }
    FILE* o = std::fopen(argv[2], "wb");
    std::fwrite(out.data(), 4, n, o);
    std::fclose(o);
    return 0;
}
