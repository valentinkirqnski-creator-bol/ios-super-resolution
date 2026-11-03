// Parity / benchmarking CLI.
//   robust_nn_cli in.bin out.bin
// in.bin : int32 W,H,flow_step; float32 noise_a,noise_b,clip; then ref[WH], cmp[WH], flow[2WH], r0[WH]
// out.bin: float32 R[WH] followed by features[NF*WH] (planar)
#include <chrono>
#include <cstdio>
#include <vector>

#include "robust_nn.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s in.bin out.bin\n", argv[0]);
        return 1;
    }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    int hdr[3];
    float prm[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::fread(prm, 4, 3, f) != 3) return 3;
    RNNParams P{};
    P.width = hdr[0];
    P.height = hdr[1];
    P.flow_step = hdr[2];
    P.noise_a = prm[0];
    P.noise_b = prm[1];
    P.clip_level = prm[2];
    size_t n = (size_t)P.width * P.height;
    std::vector<float> ref(n), cmp(n), flow(2 * n), r0(n), out(n), feat(n * RNN_NF);
    if (std::fread(ref.data(), 4, n, f) != n || std::fread(cmp.data(), 4, n, f) != n ||
        std::fread(flow.data(), 4, 2 * n, f) != 2 * n || std::fread(r0.data(), 4, n, f) != n)
        return 4;
    std::fclose(f);
    robust_nn::Scratch s;
    auto t0 = std::chrono::steady_clock::now();
    robust_nn::refine(ref.data(), cmp.data(), flow.data(), r0.data(), P, out.data(), &s, feat.data());
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "refine %dx%d: %.2f ms\n", P.width, P.height,
                 std::chrono::duration<double, std::milli>(t1 - t0).count());
    FILE* o = std::fopen(argv[2], "wb");
    std::fwrite(out.data(), 4, n, o);
    std::fwrite(feat.data(), 4, n * RNN_NF, o);
    std::fclose(o);
    return 0;
}
