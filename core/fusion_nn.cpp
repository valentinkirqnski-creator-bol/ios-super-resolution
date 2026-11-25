// Fusion NN -- see fusion_nn.h. Mirrors tools/fusion_nn/fusion.py (guides, warp_guides,
// net_inputs, FusionNet) operation for operation; tools/fusion_nn/parity_app.py checks the
// logits against PyTorch.
#include "fusion_nn.h"
#include "fusion_nn_shared.h"
#include "fusion_nn_weights.h"
#include "parallel.h"
#if defined(__APPLE__)
#include "metal_gpu.h"
#endif

#include <algorithm>
#include <cmath>
#include <vector>

namespace hhsr {

namespace {

constexpr int CIN = FNN_CIN;   // 8
constexpr int C = FNN_C;       // 12
constexpr int NB = FNN_NBLOCKS;
constexpr int HALO = 16;       // receptive field: stem 1 + dilations 1+2+4+8
constexpr int BAND = 64;

static_assert(CIN == FNN_K_CIN && C == FNN_K_C, "fusion_nn_weights.h does not match fusion_nn_shared.h");

// Weights re-laid for channel-inner loops: [tap][in][out] for convs, [tap][c] for depthwise.
struct Packed {
    float stem[9][CIN][C];
    float stem_b[C];
    float dw[NB][9][C];
    float dw_b[NB][C];
    float pw[NB][C][C];        // [in][out]
    float pw_b[NB][C];
    float head[C + 1];         // weights, then bias
    Packed() {
        for (int o = 0; o < C; ++o) stem_b[o] = fnn_stem_b[o];
        for (int c = 0; c < C; ++c) head[c] = fnn_head_w[c];
        head[C] = fnn_head_b[0];
        for (int b = 0; b < NB; ++b)
            for (int c = 0; c < C; ++c) { dw_b[b][c] = fnn_dw_b[b][c]; pw_b[b][c] = fnn_pw_b[b][c]; }
        for (int o = 0; o < C; ++o)
            for (int i = 0; i < CIN; ++i)
                for (int t = 0; t < 9; ++t) stem[t][i][o] = fnn_stem_w[(o * CIN + i) * 9 + t];
        for (int b = 0; b < NB; ++b) {
            for (int c = 0; c < C; ++c)
                for (int t = 0; t < 9; ++t) dw[b][t][c] = fnn_dw_w[b][c * 9 + t];
            for (int o = 0; o < C; ++o)
                for (int i = 0; i < C; ++i) pw[b][i][o] = fnn_pw_w[b][o * C + i];
        }
    }
};

const Packed& packed() {
    static const Packed p;
    return p;
}

// One band through the network on the CPU (the shared per-pixel code the GPU also runs).
void band_cpu(const std::vector<float>& X, std::vector<float>& A, std::vector<float>& B2, FnnBand p,
              int keep0, int keep_rows, float* out, int threads) {
    const Packed& P = packed();
    parallel_rows(p.rows, threads, [&](int r) {
        for (int x = 0; x < p.w; ++x) fnn_stem_px(X.data(), A.data(), &P.stem[0][0][0], P.stem_b, p, x, r);
    });
    std::vector<float>* cur = &A;
    std::vector<float>* nxt = &B2;
    for (int b = 0; b < NB; ++b) {
        p.dil = fnn_dil[b];
        const float* in = cur->data();
        float* o = nxt->data();
        parallel_rows(p.rows, threads, [&](int r) {
            for (int x = 0; x < p.w; ++x)
                fnn_block_px(in, o, &P.dw[b][0][0], P.dw_b[b], &P.pw[b][0][0], P.pw_b[b], p, x, r);
        });
        std::swap(cur, nxt);
    }
    const float* fin = cur->data();
    parallel_rows(keep_rows, threads, [&](int r) {
        for (int x = 0; x < p.w; ++x) out[(size_t)r * p.w + x] = fnn_head_px(fin, P.head, p, x, keep0 + r);
    });
}

inline float gat1(float x, float a, float b) {
    const float v = a * std::max(x, 0.f) + 0.375f * a * a + b;
    return (2.f / a) * std::sqrt(std::max(v, 0.f));
}

}  // namespace

Image fusion_nn_guide(const Image& raw, const Config& cfg) {
    const int gh = raw.h / 2, gw = raw.w / 2;
    Image g(gh, gw, 3);
    int cnt[3] = {0, 0, 0};
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) cnt[std::min(2, (int)cfg.cfa.p[dy][dx])]++;
    float inv[3];
    for (int c = 0; c < 3; ++c) inv[c] = cnt[c] > 0 ? 1.f / (float)cnt[c] : 0.f;
    parallel_rows(gh, cfg.num_threads, [&](int y) {
        for (int x = 0; x < gw; ++x) {
            float s[3] = {0.f, 0.f, 0.f};
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    s[std::min(2, (int)cfg.cfa.p[dy][dx])] += raw.at(2 * y + dy, 2 * x + dx);
            for (int c = 0; c < 3; ++c) g.at(y, x, c) = s[c] * inv[c];
        }
    });
    return g;
}

bool fusion_nn_flow_supported(const FlowField& flow) {
    return flow.ny > 0 && flow.nx > 0 && !flow.flow.empty() && !flow.has_dense() &&
           !flow.has_affine() && !flow.has_global_h;
}

Image fusion_nn_warp_guide(const Image& guide, const FlowField& flow, int tile_size, int threads) {
    const int gh = guide.h, gw = guide.w;
    Image out(gh, gw, 3);
    parallel_rows(gh, threads, [&](int y) {
        const int ty = std::min(flow.ny - 1, std::max(0, (int)std::floor((2.f * y + 0.5f) / (float)tile_size)));
        for (int x = 0; x < gw; ++x) {
            const int tx = std::min(flow.nx - 1, std::max(0, (int)std::floor((2.f * x + 0.5f) / (float)tile_size)));
            float sx = (float)x + 0.5f * flow.dx(ty, tx);
            float sy = (float)y + 0.5f * flow.dy(ty, tx);
            sx = std::min(std::max(sx, 0.f), (float)(gw - 1));
            sy = std::min(std::max(sy, 0.f), (float)(gh - 1));
            const int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
            const int x1 = std::min(x0 + 1, gw - 1), y1 = std::min(y0 + 1, gh - 1);
            const float fx = sx - (float)x0, fy = sy - (float)y0;
            for (int c = 0; c < 3; ++c) {
                const float top = guide.at(y0, x0, c) * (1.f - fx) + guide.at(y0, x1, c) * fx;
                const float bot = guide.at(y1, x0, c) * (1.f - fx) + guide.at(y1, x1, c) * fx;
                out.at(y, x, c) = top * (1.f - fy) + bot * fy;
            }
        }
    });
    return out;
}

void fusion_nn_gat(Image& g, float a, float b, int threads) {
    const size_t row = (size_t)g.w * (size_t)g.c;
    parallel_rows(g.h, threads, [&](int y) {
        float* p = g.data.data() + (size_t)y * row;
        for (size_t i = 0; i < row; ++i) p[i] = gat1(p[i], a, b);
    });
}

bool fusion_nn_noise(const Config& cfg, float& a, float& b) {
    a = cfg.alpha_dng[1];
    b = cfg.beta_dng[1];
    return std::isfinite(a) && std::isfinite(b) && a > 0.f && b >= 0.f;
}

Image fusion_nn_logits(const Image& ref_gat, const Image& frame_gat, const Image& mean_gat,
                       bool is_ref, float a, float b, int threads) {
    const int gh = ref_gat.h, gw = ref_gat.w;
    Image logits(gh, gw, 1);
    if (gh <= 0 || gw <= 0 || frame_gat.h != gh || frame_gat.w != gw || mean_gat.h != gh || mean_gat.w != gw)
        return Image();
    const float inv_one = 1.f / gat1(1.f, a, b);
    const int max_rows = BAND + 2 * HALO;
    std::vector<float> X((size_t)max_rows * gw * CIN), A((size_t)max_rows * gw * C), B2((size_t)max_rows * gw * C);

    for (int y0 = 0; y0 < gh; y0 += BAND) {
        const int y1 = std::min(gh, y0 + BAND);
        const int e0 = std::max(0, y0 - HALO), e1 = std::min(gh, y1 + HALO), er = e1 - e0;
        // ---- inputs: is-ref, brightness, (f - ref) / 4, (f - mean) / 4, clamped to +-8
        parallel_rows(er, threads, [&](int r) {
            const int y = e0 + r;
            for (int x = 0; x < gw; ++x) {
                float* o = &X[((size_t)r * gw + x) * CIN];
                const float r0 = ref_gat.at(y, x, 0), r1 = ref_gat.at(y, x, 1), r2 = ref_gat.at(y, x, 2);
                o[0] = is_ref ? 1.f : 0.f;
                o[1] = (r0 + r1 + r2) * (1.f / 3.f) * inv_one;
                for (int c = 0; c < 3; ++c) {
                    const float f = frame_gat.at(y, x, c);
                    o[2 + c] = std::min(8.f, std::max(-8.f, (f - ref_gat.at(y, x, c)) * 0.25f));
                    o[5 + c] = std::min(8.f, std::max(-8.f, (f - mean_gat.at(y, x, c)) * 0.25f));
                }
            }
        });
        FnnBand p{gw, er, e0, gh, 1, 0, 0, 0};
        float* out = &logits.data[(size_t)y0 * gw];
        bool done = false;
#if defined(__APPLE__)
        done = fusion_nn_band_metal(X.data(), er, gw, e0, gh, y0 - e0, y1 - y0, out);
#endif
        if (!done) band_cpu(X, A, B2, p, y0 - e0, y1 - y0, out, threads);
    }
    return logits;
}

Image fusion_nn_mask(const Image& l_f, const Image& l_ref, int h, int w, int c, int threads) {
    const int gh = l_f.h, gw = l_f.w;
    if (gh <= 0 || gw <= 0 || l_ref.h != gh || l_ref.w != gw || h <= 0 || w <= 0 || c <= 0) return Image();
    Image R(h, w, c);
    const bool direct = (h == gh && w == gw);
    const float sy = (float)gh / (float)h, sx = (float)gw / (float)w;
    parallel_rows(h, threads, [&](int y) {
        for (int x = 0; x < w; ++x) {
            float dlt;
            if (direct) {
                dlt = l_f.at(y, x) - l_ref.at(y, x);
            } else {
                // logits are bilinear in guide space (training: bilinear upsampling of the logits)
                const float gy = std::min(std::max(((float)y + 0.5f) * sy - 0.5f, 0.f), (float)(gh - 1));
                const float gx = std::min(std::max(((float)x + 0.5f) * sx - 0.5f, 0.f), (float)(gw - 1));
                const int y0 = (int)std::floor(gy), x0 = (int)std::floor(gx);
                const int y1 = std::min(y0 + 1, gh - 1), x1 = std::min(x0 + 1, gw - 1);
                const float fy = gy - (float)y0, fx = gx - (float)x0;
                auto lerp2 = [&](const Image& L) {
                    const float top = L.at(y0, x0) * (1.f - fx) + L.at(y0, x1) * fx;
                    const float bot = L.at(y1, x0) * (1.f - fx) + L.at(y1, x1) * fx;
                    return top * (1.f - fy) + bot * fy;
                };
                dlt = lerp2(l_f) - lerp2(l_ref);
            }
            // weight relative to the reference; the bound only guards against overflow
            const float r = std::exp(std::min(std::max(dlt, -30.f), 6.f));
            for (int k = 0; k < c; ++k) R.at(y, x, k) = r;
        }
    });
    return R;
}

size_t fusion_nn_frame_bytes(int raw_h, int raw_w, int mask_h, int mask_w, int mask_c) {
    const size_t g = (size_t)(raw_h / 2) * (size_t)(raw_w / 2);
    const size_t band = (size_t)(BAND + 2 * HALO) * (size_t)(raw_w / 2) * (CIN + 2 * C);
    return sizeof(float) * (2 * 3 * g + g + band + (size_t)mask_h * mask_w * mask_c);
}

}  // namespace hhsr

namespace hhsr {

FusionNnWeights fusion_nn_weights() {
    const Packed& P = packed();
    FusionNnWeights w;
    w.stem = &P.stem[0][0][0];
    w.stem_b = P.stem_b;
    for (int b = 0; b < NB; ++b) {
        w.dw[b] = &P.dw[b][0][0];
        w.dw_b[b] = P.dw_b[b];
        w.pw[b] = &P.pw[b][0][0];
        w.pw_b[b] = P.pw_b[b];
        w.dil[b] = fnn_dil[b];
    }
    w.head = P.head;
    return w;
}

}  // namespace hhsr
