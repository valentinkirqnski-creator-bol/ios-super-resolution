// Tile-seam leak test on the CPU merge: comp = exact copy of the ref crop, flow 0 everywhere
// except one tile T with a WRONG vector (+3, +2 px). The mask rejects T exactly (each mask
// pixel graded with its own tile, as Wronski / GCam / FB confidence do). Any output pixel
// whose value differs from the merge with T's vector corrected is a leak of the wrong vector.
// Raw-res mask and guide-res mask; scale 2.
#include "stages.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <cmath>
#include <cstdio>
using namespace hhsr;

int main(int argc, char** argv) {
    Config cfg;
    Image full = load_raw_frame(argv[1], cfg, true);
    const int S = 512, ts = 16;
    Image ref(S, S, 1);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) ref.at(y, x) = full.at(full.h / 2 - S / 2 + y, full.w / 2 - S / 2 + x);
    cfg.scale = 2.f;
    cfg.burst_frame_count = 2;
    tune_config_snr(ref, cfg);
    CovField covs = estimate_kernels(ref, cfg);
    FlowField fw(S / ts, S / ts), fc(S / ts, S / ts);
    const int Ty = 10, Tx = 10;
    fw.dx(Ty, Tx) = 3.f; fw.dy(Ty, Tx) = 2.f;
    const int Hs = S * 2, Ws = S * 2;
    for (int guide = 0; guide < 2; ++guide) {
        const int mh = guide ? S / 2 : S;
        Image R(mh, mh, 1);
        for (int y = 0; y < mh; ++y)
            for (int x = 0; x < mh; ++x) {
                // the tile this mask pixel belongs to (raw y, or raw 2g + 0.5 for guide)
                const float ry = guide ? 2.f * y + 0.5f : (float)y, rx = guide ? 2.f * x + 0.5f : (float)x;
                const bool inT = (int)(ry / ts) == Ty && (int)(rx / ts) == Tx;
                R.at(y, x) = inT ? 0.f : 1.f;
            }
        Image n1(Hs, Ws, 3), d1(Hs, Ws, 3), n2(Hs, Ws, 3), d2(Hs, Ws, 3);
        merge_comp(ref, fw, covs, R, ts, n1, d1, cfg);
        merge_comp(ref, fc, covs, R, ts, n2, d2, cfg);
        int leak = 0, lymin = 1 << 30, lymax = -1, lxmin = 1 << 30, lxmax = -1;
        double worst = 0;
        for (int y = 0; y < Hs; ++y)
            for (int x = 0; x < Ws; ++x) {
                bool diff = false;
                for (int c = 0; c < 3; ++c) {
                    const float a = n1.at(y, x, c), b = n2.at(y, x, c);
                    if (std::fabs(a - b) > 1e-6f * (1.f + std::fabs(b))) { diff = true; worst = std::max(worst, (double)std::fabs(a - b)); }
                }
                if (diff) { ++leak; lymin = std::min(lymin, y); lymax = std::max(lymax, y); lxmin = std::min(lxmin, x); lxmax = std::max(lxmax, x); }
            }
        std::printf("%s mask: leaked output px %d", guide ? "guide-res" : "raw-res  ", leak);
        if (leak) std::printf("  rows %d..%d cols %d..%d (tile T covers output %d..%d)", lymin, lymax, lxmin, lxmax, Ty * ts * 2, (Ty + 1) * ts * 2 - 1);
        std::printf("\n");
    }
    return 0;
}
