// Parity: mann_refine (windowed over live tiles, banded, gated) vs a full-image reference.
// Synthetic ref / comp greys with edges + noise, a random flow with irregular patches, a random
// mask. Writes: features [11][H][W] + gate [H][W] (full image, the dataset path) and R before /
// after mann_refine, for mann_parity.py to compare against torch on the full image.
#include "mann.h"
#include "cref.h"
#include <cmath>
#include <cstdio>
#include <random>
using namespace hhsr;

int main(int argc, char** argv) {
    const int H = 300, W = 420, ts = 16;
    std::mt19937 rng(7);
    std::normal_distribution<float> N01(0.f, 1.f);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    Config cfg;
    cfg.alpha_dng[1] = 1e-4f; cfg.beta_dng[1] = 1e-6f;
    cfg.num_threads = 4;
    Image ref(H, W, 1), comp(H, W, 1), R(H, W, 1);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float e = (x + 0.3f * y) > 200 ? 0.6f : 0.2f;            // a slanted edge
            const float bar = ((y / 9) % 3 == 0) ? 0.25f : 0.f;             // bars
            const float base = e + bar;
            const float sh = (x > 250 && y > 120) ? 1.3f : 0.f;             // a shifted region in comp
            const float bc = ((x + sh + 0.3f * y) > 200 ? 0.6f : 0.2f) + bar;
            ref.at(y, x) = base + 0.01f * N01(rng);
            comp.at(y, x) = bc + 0.01f * N01(rng);
            R.at(y, x) = U(rng);
        }
    FlowField f;
    f.ny = (H + ts - 1) / ts; f.nx = (W + ts - 1) / ts;
    f.ny = H / ts; f.nx = W / ts;   // tiles cover the image; the last row / col takes the remainder
    f.flow.assign((size_t)f.ny * f.nx * 2, 0.f);
    for (int t = 0; t < f.ny * f.nx; ++t) {
        const int ty = t / f.nx, tx = t % f.nx;
        const bool irr = (ty > 4 && ty < 11 && tx > 3 && tx < 12) || (ty > 13 && tx > 19);
        if (irr) { f.flow[2 * t] = 2.f * N01(rng); f.flow[2 * t + 1] = 2.f * N01(rng); }
    }
    const std::vector<float> M = mann_tile_irregularity(f);
    const std::vector<float> tdis = cref_flow_disagreement(f);
    std::vector<float> F, G;
    mann_features(ref, comp, R, f, M, tdis, ts, cfg, 0, H, 0, W, F, G);
    Image Rn = R;
    const bool ok = mann_refine(Rn, ref, comp, f, ts, cfg);
    FILE* fo = std::fopen(argc > 1 ? argv[1] : "mann_parity.bin", "wb");
    const int hd[3] = {H, W, (int)ok};
    std::fwrite(hd, 4, 3, fo);
    std::fwrite(F.data(), 4, F.size(), fo);
    std::fwrite(G.data(), 4, G.size(), fo);
    std::fwrite(R.data.data(), 4, R.data.size(), fo);
    std::fwrite(Rn.data.data(), 4, Rn.data.size(), fo);
    std::fclose(fo);
    size_t gated = 0;
    for (float g : G) gated += g > 0.f;
    std::printf("ok %d gated px %zu of %d\n", (int)ok, gated, H * W);
    return 0;
}
