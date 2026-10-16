// Runs the app's fused Metal kernel (rnn_refine_mask, core/HHSRKernels.metal)
// on the CPU through a thin MSL shim, single-threaded (barriers become no-ops),
// to check its tiling / halo indexing / flow lookup against the reference.
//   app_kernel_emu in.bin out.bin    (same in.bin format as robust_nn_cli; the
//   dense flow must be nearest-tile with tile = flow_step grey px)
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "robust_nn_shared.h"

// ---- MSL shim ----------------------------------------------------------------
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
inline float floor_(float v) { return std::floor(v); }
// copied verbatim (minus address spaces) from HHSRKernels.metal
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

int main(int argc, char** argv) {
    FILE* f = std::fopen(argv[1], "rb");
    int hdr[3];
    float prm[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::fread(prm, 4, 3, f) != 3) return 3;
    const int W = hdr[0], H = hdr[1], fs = hdr[2];
    size_t n = (size_t)W * H;
    std::vector<float> ref(n), cmp(n), flow(2 * n), r0(n), out(n, -1.f);
    if (std::fread(ref.data(), 4, n, f) != n || std::fread(cmp.data(), 4, n, f) != n ||
        std::fread(flow.data(), 4, 2 * n, f) != 2 * n || std::fread(r0.data(), 4, n, f) != n)
        return 4;
    std::fclose(f);
    // app-side inputs: 3-channel guide whose grey is cmp, per-tile RAW flow
    std::vector<float> guide(3 * n);
    for (size_t i = 0; i < n; ++i) guide[3 * i] = guide[3 * i + 1] = guide[3 * i + 2] = cmp[i];
    const int ny = (H + fs - 1) / fs, nx = (W + fs - 1) / fs;
    std::vector<float> tflow(2 * (size_t)ny * nx), jac(4 * (size_t)ny * nx, 0.f);
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            size_t s = 2 * ((size_t)std::min(ty * fs, H - 1) * W + std::min(tx * fs, W - 1));
            tflow[2 * ((size_t)ty * nx + tx)] = 2.f * flow[s];          // grey -> raw px
            tflow[2 * ((size_t)ty * nx + tx) + 1] = 2.f * flow[s + 1];
        }
    RnnAppParams p{};
    p.width = W; p.height = H; p.flow_step = fs;
    p.noise_a = prm[0]; p.noise_b = prm[1]; p.clip_level = prm[2];
    p.flow_ny = ny; p.flow_nx = nx; p.tile_size = 2.f * fs; p.affine = 0; p.homog = 0;
    p.sharp_lo = argc > 4 ? std::atof(argv[3]) : 0.f;   // optional: lo hi
    p.sharp_hi = argc > 4 ? std::atof(argv[4]) : 0.f;
    for (uint ty = 0; ty < (uint)((H + 15) / 16); ++ty)
        for (uint tx = 0; tx < (uint)((W + 15) / 16); ++tx)
            rnn_refine_mask(ref.data(), guide.data(), r0.data(), out.data(), tflow.data(), jac.data(), p,
                            uint2{tx, ty}, 0u, uint2{1u, 1u});
    FILE* o = std::fopen(argv[2], "wb");
    std::fwrite(out.data(), 4, n, o);
    std::fclose(o);
    return 0;
}
