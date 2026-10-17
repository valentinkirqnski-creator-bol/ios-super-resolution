// merge_dump — the APP's own merge (core/merge.cpp accumulate_comp/_ref, core/
// kernels.cpp steerable covariances) run with R = 1, dumping each frame's
// contribution separately. accumulate_comp samples R once per output pixel and
// multiplies it into every tap, so for any mask
//
//     merged = (A_ref + sum_n R_n(p) A_n) / (B_ref + sum_n R_n(p) B_n)
//
// exactly; tools/robust_nn recombines these in torch.
//
//   merge_dump in.bin out.bin
// in : int32 N, H, W, V; float32 scale;
//      raw[N][H][W]                      (Bayer RGGB, black-subtracted, white = 1)
//      flow[V][N][H/2][W/2][2]           (dense ref->cmp, GREY px; frame 0 ignored)
// out: A_ref[Ho][Wo][3], B_ref[Ho][Wo][3], then for v < V, n = 1..N-1:
//      A[v][n][Ho][Wo][3], B[v][n][Ho][Wo][3]
#include <cstdint>
#include <cstdio>
#include <vector>

#include "stages.h"
#include "types.h"

using namespace hhsr;

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    int32_t hdr[4];
    float scale;
    if (std::fread(hdr, 4, 4, f) != 4 || std::fread(&scale, 4, 1, f) != 1) return 3;
    const int N = hdr[0], H = hdr[1], W = hdr[2], V = hdr[3];
    const int Hg = H / 2, Wg = W / 2;
    std::vector<Image> raw(N, Image(H, W, 1));
    for (auto& r : raw)
        if (std::fread(r.data.data(), 4, r.data.size(), f) != r.data.size()) return 4;
    std::vector<float> flows((size_t)V * N * Hg * Wg * 2);
    if (std::fread(flows.data(), 4, flows.size(), f) != flows.size()) return 5;
    std::fclose(f);

    Config cfg;
    cfg.bayer_mode = true;
    cfg.scale = scale;
    const int Ho = (int)(H * scale), Wo = (int)(W * scale);
    CovField covs = estimate_kernels(raw[0], cfg);

    FILE* o = std::fopen(argv[2], "wb");
    {
        Image num(Ho, Wo, 3), den(Ho, Wo, 3);
        merge_ref_band(raw[0], covs, num, den, 0, cfg, nullptr);
        std::fwrite(num.data.data(), 4, num.data.size(), o);
        std::fwrite(den.data.data(), 4, den.data.size(), o);
    }
    Image ones(H, W, 1);
    for (auto& x : ones.data) x = 1.f;
    for (int v = 0; v < V; ++v)
        for (int n = 1; n < N; ++n) {
            FlowField fl;
            fl.dense_h = Hg;
            fl.dense_w = Wg;
            fl.dense_sx = fl.dense_sy = 2.f;
            const float* src = flows.data() + ((size_t)v * N + n) * Hg * Wg * 2;
            fl.dense.assign(src, src + (size_t)Hg * Wg * 2);
            Image num(Ho, Wo, 3), den(Ho, Wo, 3);
            merge_comp_band(raw[n], fl, covs, ones, /*tile_size=*/16, num, den, 0, cfg, -1);
            std::fwrite(num.data.data(), 4, num.data.size(), o);
            std::fwrite(den.data.data(), 4, den.data.size(), o);
        }
    std::fclose(o);
    return 0;
}
