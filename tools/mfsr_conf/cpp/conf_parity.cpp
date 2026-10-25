// conf_parity in.bin out.bin -- CPU reference of the device ConfNet path (core/conf_nn_shared.h), for
// tools/mfsr_conf/parity_cnn.py. Build with -DCNN_WEIGHTS_HEADER="<exported weights>".
//
// in.bin: int32 K, gh, gw, H, W, flow_step; f32 noise_a, noise_b, clip;
//         f32 ref_g[gh][gw], comp_g[K][gh][gw], flow_warp[K][gh][gw][2], flow_feat[K][gh][gw][2], r0g[K][gh][gw],
//             xg_py[K][CNN_NG][gh][gw], If[H][W], jf[K][H][W], r0raw[K][H][W]
// out.bin: f32 xg_cpp[K][CNN_NG][gh][gw]   (features + pixel channels from the greys / flows)
//          f32 xr_cpp[K][CNN_NR][H][W]     (raw inputs from If / jf)
//          f32 logits[K][2][H][W]          (the network run on xg_py and xr_cpp; guide logits upsampled when CNN_UP 0)
//          f32 R[K][H][W]                  (cnn_weight of the merge logit)
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "conf_nn_shared.h"

struct Planar {     // [C][h][w] accessor
    const float* p;
    int w, h;
    float operator()(int c, int x, int y) const { return p[((size_t)c * h + y) * w + x]; }
};

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    int hi[6];
    float hf[3];
    if (std::fread(hi, 4, 6, f) != 6 || std::fread(hf, 4, 3, f) != 3) return 3;
    const int K = hi[0], gh = hi[1], gw = hi[2], H = hi[3], W = hi[4], fstep = hi[5];
    const size_t G = (size_t)gh * gw, R = (size_t)H * W;
    auto rd = [&](size_t n) { std::vector<float> v(n); if (std::fread(v.data(), 4, n, f) != n) std::exit(4); return v; };
    std::vector<float> ref_g = rd(G), comp_g = rd(K * G), fw = rd(K * G * 2), ff = rd(K * G * 2), r0g = rd(K * G),
                       xg_py = rd((size_t)K * CNN_NG * G), If = rd(R), jf = rd(K * R), r0raw = rd(K * R);
    std::fclose(f);
    RNNParams P{gw, gh, fstep, hf[0], hf[1], hf[2], 0, 0};

    std::vector<float> xg_cpp((size_t)K * CNN_NG * G), xr_cpp((size_t)K * CNN_NR * R);
    for (int k = 0; k < K; ++k) {
        RNNImg ref{ref_g.data(), gw}, cmp{comp_g.data() + k * G, gw};
        RNNFlowInterleaved flw{fw.data() + k * G * 2, gw}, flf{ff.data() + k * G * 2, gw};
        std::vector<float> J(G), feat((size_t)14 * G);
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) J[(size_t)y * gw + x] = rnn_warp_at(cmp, flw, gw, gh, x, y);
        RNNImg Ja{J.data(), gw};
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                float fv[RNN_NF];
                rnn_features_at(ref, Ja, flf, r0g[k * G + (size_t)y * gw + x], P, x, y, fv);
                for (int c = 0; c < 14; ++c) feat[(size_t)c * G + (size_t)y * gw + x] = fv[c];
            }
        Planar fa{feat.data(), gw, gh};
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                float xin[CNN_NG];
                cnn_inputs_at(ref, Ja, fa, P, x, y, xin);
                for (int c = 0; c < CNN_NG; ++c) xg_cpp[((size_t)k * CNN_NG + c) * G + (size_t)y * gw + x] = xin[c];
            }
        RNNImg Ia{If.data(), W};
        for (int Y = 0; Y < H; ++Y)
            for (int X = 0; X < W; ++X) {
                float xr[CNN_NR];
                cnn_raw_inputs_at(Ia, jf[k * R + (size_t)Y * W + X], P.noise_a, P.noise_b, W, H, X, Y, xr);
                for (int c = 0; c < CNN_NR; ++c) xr_cpp[((size_t)k * CNN_NR + c) * R + (size_t)Y * W + X] = xr[c];
            }
    }

    // network on xg_py (isolates the network from the input path)
    std::vector<float> enc((size_t)K * CNN_CH * G), tmp((size_t)CNN_CH * G);
    for (int k = 0; k < K; ++k) {
        std::vector<float> cur(xg_py.begin() + (size_t)k * CNN_NG * G, xg_py.begin() + (size_t)(k + 1) * CNN_NG * G);
        int cin = CNN_NG;
        for (int l = 0; l < 4; ++l) {
            Planar in{cur.data(), gw, gh};
            for (int y = 0; y < gh; ++y)
                for (int x = 0; x < gw; ++x) {
                    float o[CNN_CH];
                    cnn_encoder_layer_at(in, l, gw, gh, x, y, o);
                    for (int c = 0; c < CNN_CH; ++c) tmp[(size_t)c * G + (size_t)y * gw + x] = o[c];
                }
            cur = tmp;
            cin = CNN_CH;
        }
        (void)cin;
        std::copy(cur.begin(), cur.end(), enc.begin() + (size_t)k * CNN_CH * G);
    }
    std::vector<float> mean((size_t)CNN_CH * G, 0.f), mx((size_t)CNN_CH * G, -1e30f);
    for (int k = 0; k < K; ++k)
        for (size_t i = 0; i < (size_t)CNN_CH * G; ++i) {
            mean[i] += enc[(size_t)k * CNN_CH * G + i];
            mx[i] = rnn_maxf(mx[i], enc[(size_t)k * CNN_CH * G + i]);
        }
    for (auto& v : mean) v /= (float)K;
    std::vector<float> logits((size_t)K * 2 * R), Rw((size_t)K * R);
    for (int k = 0; k < K; ++k) {
        std::vector<float> g1((size_t)CNN_CH * G), lgg((size_t)2 * G), hu((size_t)(CNN_UP > 0 ? CNN_UP : 1) * G);
        for (size_t i = 0; i < G; ++i) {
            float fk[CNN_CH], mn[CNN_CH], mm[CNN_CH], o[CNN_CH];
            for (int c = 0; c < CNN_CH; ++c) {
                fk[c] = enc[((size_t)k * CNN_CH + c) * G + i];
                mn[c] = mean[(size_t)c * G + i];
                mm[c] = mx[(size_t)c * G + i];
            }
            cnn_fuse1(fk, mn, mm, o);
            for (int c = 0; c < CNN_CH; ++c) g1[(size_t)c * G + i] = o[c];
        }
        Planar ga{g1.data(), gw, gh};
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                float hid[CNN_CH], lg[2];
                cnn_head_at(ga, gw, gh, x, y, hid, lg);
                lgg[(size_t)y * gw + x] = lg[0];
                lgg[G + (size_t)y * gw + x] = lg[1];
#if CNN_UP > 0
                float h_[CNN_UP];
                cnn_u0(hid, h_);
                for (int c = 0; c < CNN_UP; ++c) hu[(size_t)c * G + (size_t)y * gw + x] = h_[c];
#endif
            }
        for (int Y = 0; Y < H; ++Y)
            for (int X = 0; X < W; ++X) {
                const size_t gi = (size_t)(Y / 2) * gw + (X / 2);
                float lg_here[2] = {lgg[gi], lgg[G + gi]}, out[2] = {lg_here[0], lg_here[1]};
#if CNN_UP > 0
                const float* xr_k = xr_cpp.data() + (size_t)k * CNN_NR * R;
                auto U = [&](int c, int xx, int yy) -> float {
                    const size_t g2 = (size_t)(yy / 2) * gw + (xx / 2);
                    if (c < CNN_UP) return hu[(size_t)c * G + g2];
                    if (c < CNN_UP + CNN_NR) return xr_k[(size_t)(c - CNN_UP) * R + (size_t)yy * W + xx];
                    return lgg[(size_t)(c - CNN_UP - CNN_NR) * G + g2];
                };
                cnn_raw_refine_at(U, lg_here, W, H, X, Y, out);
#endif
                logits[((size_t)k * 2 + 0) * R + (size_t)Y * W + X] = out[0];
                logits[((size_t)k * 2 + 1) * R + (size_t)Y * W + X] = out[1];
                Rw[(size_t)k * R + (size_t)Y * W + X] = cnn_weight(out[1], r0raw[(size_t)k * R + (size_t)Y * W + X]);
            }
    }
    FILE* o = std::fopen(argv[2], "wb");
    if (!o) return 5;
    std::fwrite(xg_cpp.data(), 4, xg_cpp.size(), o);
    std::fwrite(xr_cpp.data(), 4, xr_cpp.size(), o);
    std::fwrite(logits.data(), 4, logits.size(), o);
    std::fwrite(Rw.data(), 4, Rw.size(), o);
    std::fclose(o);
    return 0;
}
