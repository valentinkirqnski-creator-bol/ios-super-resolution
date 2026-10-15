// Convolutional ghost detector inference (build 117). See stages.h.
//
// Runs the tools/ghost_nn CNN host-side at 1/GHOST_DS of the grey resolution
// and darkens the robustness mask where it detects a ghost. Architecture is
// fixed to match tools/ghost_nn/train_ghost.py GhostCNN: five conv layers,
// dilations 1,2,4,1 (3x3) then a 1x1, ReLU between, sigmoid out. Weights +
// input normalisation come from core/ghost_cnn_weights.h (auto-generated).
#include "stages.h"
#include "types.h"
#include "parallel.h"
#include "ghost_cnn_weights.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace hhsr {

namespace {

// Noise model (matches tools/nn_mask_refine/common.py and the device merge).
constexpr float GN_A = 2.0e-3f, GN_B = 1.0e-3f;

inline int reflect_idx(int i, int n) {
    if (n <= 1) return 0;
    // reflect without repeating the edge (torch 'reflect' pad semantics)
    int period = 2 * (n - 1);
    i = i % period; if (i < 0) i += period;
    return i < n ? i : period - i;
}

// 3x3 dilated conv, reflect pad, optional ReLU. in/out are (C,H,W) row-major.
void conv3x3(const std::vector<float>& in, std::vector<float>& out,
             const float* Wt, const float* Bs, int Cin, int Cout,
             int H, int W, int dil, bool relu, int nthreads) {
    out.assign((size_t)Cout * H * W, 0.f);
    const size_t plane = (size_t)H * W;
    parallel_rows(H, nthreads, [&](int y) {
        int yy[3];
        for (int ky = 0; ky < 3; ++ky) yy[ky] = reflect_idx(y + (ky - 1) * dil, H);
        for (int x = 0; x < W; ++x) {
            int xx[3];
            for (int kx = 0; kx < 3; ++kx) xx[kx] = reflect_idx(x + (kx - 1) * dil, W);
            for (int oc = 0; oc < Cout; ++oc) {
                const float* w = Wt + (size_t)oc * Cin * 9;
                float s = Bs[oc];
                for (int ic = 0; ic < Cin; ++ic) {
                    const float* inp = &in[(size_t)ic * plane];
                    const float* wc = w + ic * 9;
                    s += wc[0]*inp[(size_t)yy[0]*W+xx[0]] + wc[1]*inp[(size_t)yy[0]*W+xx[1]] + wc[2]*inp[(size_t)yy[0]*W+xx[2]]
                       + wc[3]*inp[(size_t)yy[1]*W+xx[0]] + wc[4]*inp[(size_t)yy[1]*W+xx[1]] + wc[5]*inp[(size_t)yy[1]*W+xx[2]]
                       + wc[6]*inp[(size_t)yy[2]*W+xx[0]] + wc[7]*inp[(size_t)yy[2]*W+xx[1]] + wc[8]*inp[(size_t)yy[2]*W+xx[2]];
                }
                if (relu && s < 0.f) s = 0.f;
                out[(size_t)oc * plane + (size_t)y * W + x] = s;
            }
        }
    });
}

// 1x1 conv to a single channel + sigmoid.
void conv1x1_sigmoid(const std::vector<float>& in, std::vector<float>& out,
                     const float* Wt, const float* Bs, int Cin, int H, int W, int nthreads) {
    out.assign((size_t)H * W, 0.f);
    const size_t plane = (size_t)H * W;
    parallel_rows(H, nthreads, [&](int y) {
        for (int x = 0; x < W; ++x) {
            float s = Bs[0];
            const size_t p = (size_t)y * W + x;
            for (int ic = 0; ic < Cin; ++ic) s += Wt[ic] * in[(size_t)ic * plane + p];
            out[p] = 1.f / (1.f + std::exp(-s));
        }
    });
}

} // namespace

void ghost_detect_refine(Image& R, const Image& ref_grey,
                         const Image& comp_aligned_grey, const Config& cfg) {
    if (!cfg.ghost_detector_enabled) return;
    const int H = R.h, W = R.w, c = std::max(1, R.c);
    if (H <= 0 || W <= 0) return;
    // GPU-resident mask (header only): safe no-op (the flag forces host mask).
    if (R.data.size() != (size_t)H * W * c) return;
    if (ref_grey.h != H || ref_grey.w != W ||
        comp_aligned_grey.h != H || comp_aligned_grey.w != W) return;

    const int DS = GHOST_DS < 1 ? 1 : GHOST_DS;
    const int Hs = H / DS, Ws = W / DS;
    if (Hs < 4 || Ws < 4) return;

    // Build the four input channels at the working resolution by box-averaging
    // the FULL-res channels over DS x DS blocks (matches train_ghost._ds, which
    // averages the already-formed [ref, comp, res/sig, R] channels). Then apply
    // the stored per-channel normalisation.
    std::vector<float> in((size_t)4 * Hs * Ws, 0.f);
    const float inv = 1.f / (float)(DS * DS);
    parallel_rows(Hs, cfg.num_threads, [&](int ys) {
        for (int xs = 0; xs < Ws; ++xs) {
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (int dy = 0; dy < DS; ++dy) {
                const int y = ys * DS + dy;
                for (int dx = 0; dx < DS; ++dx) {
                    const int x = xs * DS + dx;
                    const float rf = ref_grey.at(y, x);
                    const float cm = comp_aligned_grey.at(y, x);
                    const float sig = std::sqrt(std::max(GN_A * std::max(rf, 0.f) + GN_B, 1e-12f));
                    a0 += rf;
                    a1 += cm;
                    a2 += (rf - cm) / std::max(sig, 1e-4f);
                    a3 += R.at(y, x, 0);
                }
            }
            const size_t p = (size_t)ys * Ws + xs;
            const size_t pl = (size_t)Hs * Ws;
            in[0 * pl + p] = (a0 * inv - GHOST_FMU[0]) / GHOST_FSD[0];
            in[1 * pl + p] = (a1 * inv - GHOST_FMU[1]) / GHOST_FSD[1];
            in[2 * pl + p] = (a2 * inv - GHOST_FMU[2]) / GHOST_FSD[2];
            in[3 * pl + p] = (a3 * inv - GHOST_FMU[3]) / GHOST_FSD[3];
        }
    });

    // Five-layer CNN (dilations 1,2,4,1 then 1x1), matching GhostCNN.
    std::vector<float> h1, h2, h3, h4, g;
    conv3x3(in, h1, GHOST_W1, GHOST_B1, 4, 24, Hs, Ws, 1, true, cfg.num_threads);
    conv3x3(h1, h2, GHOST_W2, GHOST_B2, 24, 24, Hs, Ws, 2, true, cfg.num_threads);
    conv3x3(h2, h3, GHOST_W3, GHOST_B3, 24, 24, Hs, Ws, 4, true, cfg.num_threads);
    conv3x3(h3, h4, GHOST_W4, GHOST_B4, 24, 16, Hs, Ws, 1, true, cfg.num_threads);
    conv1x1_sigmoid(h4, g, GHOST_W5, GHOST_B5, 16, Hs, Ws, cfg.num_threads);

    // Upsample g (bilinear) to full res and darken: R' = R * (1 - strength*g).
    const float strength = cfg.ghost_strength > 0.f ? cfg.ghost_strength : 1.0f;
    parallel_rows(H, cfg.num_threads, [&](int y) {
        // map full-res y to working-res coordinate (align block centres)
        float fy = ((float)y + 0.5f) / (float)DS - 0.5f;
        int y0 = (int)std::floor(fy); float wy = fy - (float)y0;
        int y0c = std::min(std::max(y0, 0), Hs - 1);
        int y1c = std::min(std::max(y0 + 1, 0), Hs - 1);
        for (int x = 0; x < W; ++x) {
            float fx = ((float)x + 0.5f) / (float)DS - 0.5f;
            int x0 = (int)std::floor(fx); float wx = fx - (float)x0;
            int x0c = std::min(std::max(x0, 0), Ws - 1);
            int x1c = std::min(std::max(x0 + 1, 0), Ws - 1);
            const float g00 = g[(size_t)y0c * Ws + x0c], g01 = g[(size_t)y0c * Ws + x1c];
            const float g10 = g[(size_t)y1c * Ws + x0c], g11 = g[(size_t)y1c * Ws + x1c];
            const float gv = (g00 * (1 - wx) + g01 * wx) * (1 - wy)
                           + (g10 * (1 - wx) + g11 * wx) * wy;
            float keep = 1.f - strength * gv;
            if (keep < 0.f) keep = 0.f; else if (keep > 1.f) keep = 1.f;
            for (int ch = 0; ch < c; ++ch) R.at(y, x, ch) *= keep;
        }
    });
}

} // namespace hhsr
