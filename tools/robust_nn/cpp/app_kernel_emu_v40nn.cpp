// Runs nn-mask's robust_nn v40 Metal kernels (rnn_grey_from_raw + rnn_refine_mask, the exact source appended to
// core/HHSRKernels.metal) on the CPU through a thin MSL shim, single-threaded per threadgroup (barriers no-ops), to
// check the raw -> grey step, the tiling / halo indexing and the flow lookup against PyTorch.
//   app_kernel_emu_v40nn in.bin out.bin
// in.bin: int32 W,H,fs; float a,b,clip,lo,hi,u00,u01,u10,u11; raw[2H*2W]; ref_grey[H*W]; tile_flow[ny*nx*2] (raw px);
//         r0[H*W]  -> out.bin: float cmp_grey[H*W] (from rnn_grey_from_raw), R[H*W]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "robust_nn_shared.h"

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
using std::max;
inline int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
#pragma GCC diagnostic ignored "-Wattributes"
#include RNN_APP_BLOCK

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 3;
    int hdr[3];
    float prm[9];
    if (std::fread(hdr, 4, 3, f) != 3 || std::fread(prm, 4, 9, f) != 9) return 3;
    const int W = hdr[0], H = hdr[1], fs = hdr[2];
    const int ny = (H + fs - 1) / fs, nx = (W + fs - 1) / fs;
    const size_t n = (size_t)W * H, nr = 4 * n, nt = 2 * (size_t)ny * nx;
    std::vector<float> raw(nr), ref(n), tflow(nt), r0(n), cg(n, -1.f), out(n, -1.f);
    if (std::fread(raw.data(), 4, nr, f) != nr || std::fread(ref.data(), 4, n, f) != n ||
        std::fread(tflow.data(), 4, nt, f) != nt || std::fread(r0.data(), 4, n, f) != n) return 4;
    std::fclose(f);
    RnnGreyParams g{(uint)(2 * W), (uint)(2 * H), (uint)W, (uint)H, prm[5], prm[6], prm[7], prm[8]};
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) rnn_grey_from_raw(raw.data(), cg.data(), g, uint2{(uint)x, (uint)y});
    RnnAppParams p{};
    p.width = W; p.height = H; p.flow_step = fs;
    p.noise_a = prm[0]; p.noise_b = prm[1]; p.clip_level = prm[2];
    p.flow_ny = ny; p.flow_nx = nx; p.tile_size = 2.f * fs;
    p.sharp_lo = prm[3]; p.sharp_hi = prm[4];
    for (int ty = 0; ty < (H + 15) / 16; ++ty)
        for (int tx = 0; tx < (W + 15) / 16; ++tx)
            rnn_refine_mask(ref.data(), cg.data(), r0.data(), out.data(), tflow.data(), p, uint2{(uint)tx, (uint)ty},
                            0u, uint2{1u, 1u});
    FILE* o = std::fopen(argv[2], "wb");
    std::fwrite(cg.data(), 4, n, o);
    std::fwrite(out.data(), 4, n, o);
    std::fclose(o);
    return 0;
}
