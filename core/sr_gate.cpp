#include "sr_gate.h"

#include "sr_gate_weights.h"
#include "stages.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace hhsr {

namespace {

constexpr int kIn = SRG_FEATURES;
constexpr int kW = SRG_WIDTH;
constexpr int kDil[SRG_LAYERS] = {SRG_DIL0, SRG_DIL1, SRG_DIL2};
constexpr int kWOff[SRG_LAYERS] = {SRG_OFF_W0, SRG_OFF_W1, SRG_OFF_W2};
constexpr int kBOff[SRG_LAYERS] = {SRG_OFF_B0, SRG_OFF_B1, SRG_OFF_B2};

static_assert(SRG_WEIGHTS_N == kSrGateWeightCount,
              "core/sr_gate_weights.h does not match the layout in "
              "sr_gate_shared.h -- re-run tools/sr_gate/export_weights.py");

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// One dilated 3x3 ReLU layer over a band of rows.
//
// src holds rows [src_y0, src_y0 + src_rows) of the layer input, channel
// interleaved with stride w * in_ch; dst receives rows
// [dst_y0, dst_y0 + dst_rows). Row indices are in IMAGE coordinates, and taps
// outside the image are clamped -- which is what the replicate padding the
// network was trained with does, and what every other window in robustness.cpp
// does. Because every tap is clamped into [0, h-1], and every band is itself
// clipped to [0, h), a clamped tap always lands inside src.
//
// Twin of sr_gate_conv in HHSRKernels.metal.
void conv_relu_band(const f32* src, int src_y0,
                    f32* dst, int dst_y0, int dst_rows,
                    int w, int h, int in_ch, int out_ch, int dil,
                    const f32* wgt, const f32* bias, int num_threads) {
    parallel_rows(dst_rows, num_threads, [&](int r) {
        const int y = dst_y0 + r;
        f32* out = dst + (size_t)r * w * out_ch;
        for (int x = 0; x < w; ++x) {
            f32 acc[kW];
            for (int o = 0; o < out_ch; ++o) acc[o] = bias[o];
            for (int ky = 0; ky < 3; ++ky) {
                const int yy = clampi(y + (ky - 1) * dil, 0, h - 1);
                const f32* row = src + (size_t)(yy - src_y0) * w * in_ch;
                for (int kx = 0; kx < 3; ++kx) {
                    const int xx = clampi(x + (kx - 1) * dil, 0, w - 1);
                    const f32* v = row + (size_t)xx * in_ch;
                    const int kidx = ky * 3 + kx;
                    for (int o = 0; o < out_ch; ++o) {
                        const f32* wo = wgt + ((size_t)o * in_ch) * 9;
                        f32 s = 0.f;
                        for (int i = 0; i < in_ch; ++i)
                            s += wo[(size_t)i * 9 + kidx] * v[i];
                        acc[o] += s;
                    }
                }
            }
            f32* o_px = out + (size_t)x * out_ch;
            for (int o = 0; o < out_ch; ++o)
                o_px[o] = acc[o] > 0.f ? acc[o] : 0.f;
        }
    });
}

}  // namespace

const f32* sr_gate_weights(int* count) {
    if (count) *count = kSrGateWeightCount;
    return kSrGateWeights;
}

bool sr_gate_available() {
    return kSrGateWeightCount == SRG_WEIGHTS_N;
}

// --------------------------------------------------------------------------
// features
// --------------------------------------------------------------------------

Image build_sr_gate_features(const Image& ref_means, const Image& ref_vars,
                             const Image& d_sq, const Image& sigma_sq,
                             const FlowField& flow, int tile_size,
                             const Config& cfg) {
    const int h = ref_means.h, w = ref_means.w, nch = ref_means.c;
    if (h <= 0 || w <= 0 || tile_size <= 0 || flow.ny <= 0 || flow.nx <= 0 ||
        flow.flow.empty() || (nch != 1 && nch != 3) ||
        ref_vars.h != h || ref_vars.w != w || ref_vars.c != nch ||
        d_sq.h != h || d_sq.w != w || sigma_sq.h != h || sigma_sq.w != w)
        return Image();

    // The 3x3 tile flow span, once per tile rather than once per pixel.
    std::vector<f32> span((size_t)flow.ny * flow.nx, 0.f);
    for (int ty = 0; ty < flow.ny; ++ty) {
        for (int tx = 0; tx < flow.nx; ++tx) {
            f32 mnx = std::numeric_limits<f32>::infinity(), mxx = -mnx;
            f32 mny = mnx, mxy = -mnx;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    const int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= flow.ny || xx < 0 || xx >= flow.nx) continue;
                    const f32 fx = flow.dx(yy, xx), fy = flow.dy(yy, xx);
                    mnx = std::min(mnx, fx);
                    mxx = std::max(mxx, fx);
                    mny = std::min(mny, fy);
                    mxy = std::max(mxy, fy);
                }
            }
            const f32 d0 = mxx - mnx, d1 = mxy - mny;
            span[(size_t)ty * flow.nx + tx] = std::sqrt(d0 * d0 + d1 * d1);
        }
    }

    const f32 alpha = cfg.noise_alpha_robustness();
    const f32 beta = cfg.noise_beta_robustness();
    const f32 inv2ts = 1.f / (2.f * (f32)tile_size);
    const bool have_grad = (flow.ny >= 3 && flow.nx >= 3);
    // Guide-to-raw scale, the same sc compute_robustness_core uses: the
    // three-channel guide is half resolution, the FFT guide is not.
    const f32 sc = (nch == 3) ? 2.f : 1.f;

    Image feat(h, w, SRG_FEATURES);
    parallel_rows(h, cfg.num_threads, [&](int y) {
        for (int x = 0; x < w; ++x) {
            int ty, tx;
            if (nch == 1) {
                ty = y / tile_size;
                tx = x / tile_size;
            } else {
                ty = (int)((2.f * (f32)y + 0.5f) / (f32)tile_size);
                tx = (int)((2.f * (f32)x + 0.5f) / (f32)tile_size);
            }
            ty = clampi(ty, 0, flow.ny - 1);
            tx = clampi(tx, 0, flow.nx - 1);

            SrGateInputs in;
            in.d_sq = d_sq.at(y, x);
            in.sigma_sq = sigma_sq.at(y, x);
            in.tile_size = (f32)tile_size;
            in.raw_fx = flow.dx(ty, tx);
            in.raw_fy = flow.dy(ty, tx);
            in.span = span[(size_t)ty * flow.nx + tx];

            f32 var_sum = 0.f, nvar_sum = 0.f, bri_sum = 0.f;
            for (int ch = 0; ch < nch; ++ch) {
                const f32 b = clampf(std::isfinite(ref_means.at(y, x, ch))
                                         ? ref_means.at(y, x, ch) : 0.f,
                                     0.f, 1.f);
                f32 nv = std::max(alpha * b + beta, 0.f);
                // The green guide channel is the average of two Bayer greens,
                // so half the variance (guide_noise_var).
                if (nch == 3 && ch == 1) nv *= 0.5f;
                nvar_sum += nv;
                var_sum += std::max(ref_vars.at(y, x, ch), 0.f);
                bri_sum += b;
            }
            in.var_sum = var_sum;
            in.nvar_sum = nvar_sum;

            f32 ex = 0.f, ey = 0.f;
            if (have_grad) {
                const int ptu = clampi(ty - 1, 0, flow.ny - 1);
                const int ptd = clampi(ty + 1, 0, flow.ny - 1);
                const int pxl = clampi(tx - 1, 0, flow.nx - 1);
                const int pxr = clampi(tx + 1, 0, flow.nx - 1);
                const f32 gdxdx = (flow.dx(ty, pxr) - flow.dx(ty, pxl)) * inv2ts;
                const f32 gdydx = (flow.dy(ty, pxr) - flow.dy(ty, pxl)) * inv2ts;
                const f32 gdxdy = (flow.dx(ptd, tx) - flow.dx(ptu, tx)) * inv2ts;
                const f32 gdydy = (flow.dy(ptd, tx) - flow.dy(ptu, tx)) * inv2ts;
                const f32 rawx = sc * (f32)x + 0.5f * (sc - 1.f);
                const f32 rawy = sc * (f32)y + 0.5f * (sc - 1.f);
                const f32 uu = rawx - ((f32)tx + 0.5f) * (f32)tile_size;
                const f32 vv = rawy - ((f32)ty + 0.5f) * (f32)tile_size;
                ex = gdxdx * uu + gdxdy * vv;
                ey = gdydx * uu + gdydy * vv;
            }
            in.ex = ex;
            in.ey = ey;

            // Channel 0's gradient, /sc, and the noise sigma in those same
            // units at the cross-channel mean brightness -- the same pairing
            // the geometry test in compute_robustness_core uses.
            const int xl = std::max(0, x - 1), xr = std::min(w - 1, x + 1);
            const int yu = std::max(0, y - 1), yd = std::min(h - 1, y + 1);
            in.gix = 0.5f * (ref_means.at(y, xr, 0) - ref_means.at(y, xl, 0)) / sc;
            in.giy = 0.5f * (ref_means.at(yd, x, 0) - ref_means.at(yu, x, 0)) / sc;
            const f32 bri = bri_sum / (f32)nch;
            in.nsig = std::sqrt(std::max(alpha * bri + beta, 1.0e-20f)) / sc;

            sr_gate_features_from(&in, &feat.at(y, x, 0));
        }
    });
    return feat;
}

// --------------------------------------------------------------------------
// inference
// --------------------------------------------------------------------------

Image sr_gate_infer_cpu(const Image& feat) {
    const int h = feat.h, w = feat.w;
    if (h <= 0 || w <= 0 || feat.c != SRG_FEATURES || !sr_gate_available())
        return Image();

    // Row bands, because a 12 MP plane would otherwise need three full-size
    // 8-channel intermediates (1.2 GB). To produce output rows [y0, y1) the
    // layers need, working backwards through the dilations:
    //     a2 [y0, y1)  a1 [y0-3, y1+3)  a0 [y0-5, y1+5)  feat [y0-6, y1+6)
    // each clipped to [0, h). Same arithmetic as the Metal path.
    const int band = 64;
    const int pad1 = kDil[2];                 // 3
    const int pad0 = kDil[2] + kDil[1];       // 5
    std::vector<f32> buf0((size_t)(band + 2 * pad0) * w * kW);
    std::vector<f32> buf1((size_t)(band + 2 * pad1) * w * kW);
    std::vector<f32> buf2((size_t)band * w * kW);

    const f32* W = kSrGateWeights;
    Image out(h, w, 1);

    for (int y0 = 0; y0 < h; y0 += band) {
        const int y1 = std::min(y0 + band, h);
        const int a0y0 = std::max(0, y0 - pad0), a0y1 = std::min(h, y1 + pad0);
        const int a1y0 = std::max(0, y0 - pad1), a1y1 = std::min(h, y1 + pad1);

        conv_relu_band(feat.data.data(), 0, buf0.data(), a0y0, a0y1 - a0y0,
                       w, h, kIn, kW, kDil[0], W + kWOff[0], W + kBOff[0], 0);
        conv_relu_band(buf0.data(), a0y0, buf1.data(), a1y0, a1y1 - a1y0,
                       w, h, kW, kW, kDil[1], W + kWOff[1], W + kBOff[1], 0);
        conv_relu_band(buf1.data(), a1y0, buf2.data(), y0, y1 - y0,
                       w, h, kW, kW, kDil[2], W + kWOff[2], W + kBOff[2], 0);

        const f32* hw = W + SRG_OFF_HW;
        const f32 hb = W[SRG_OFF_HB];
        parallel_rows(y1 - y0, 0, [&](int r) {
            const f32* src = buf2.data() + (size_t)r * w * kW;
            for (int x = 0; x < w; ++x) {
                const f32* v = src + (size_t)x * kW;
                f32 s = hb;
                for (int i = 0; i < kW; ++i) s += hw[i] * v[i];
                out.at(y0 + r, x) = 1.f / (1.f + std::exp(-s));
            }
        });
    }
    return out;
}

Image sr_gate_mask(const Image& ref_means, const Image& ref_vars,
                   const Image& d_sq, const Image& sigma_sq,
                   const FlowField& flow, int tile_size, const Config& cfg) {
    if (!cfg.sr_gate_enabled || !sr_gate_available()) return Image();
    // The shipped weights were trained on the SHIPPING guide: one channel, raw
    // resolution, linear (robustness_fft_guide_active). The feature builder
    // above also handles the three-channel decimated guide, and both Metal and
    // the CPU compute it identically, but no weights have been fitted in that
    // domain -- three residuals summed instead of one, a different lattice, a
    // different noise scale -- so decline rather than run a trained function
    // out of the domain it was measured in.
    // The shipped weights are fitted on the THREE-channel half-resolution guide:
    // the 1.4 sqrt guide Metal has always built, which is what the shipping
    // config selects with robustness_raw_resolution OFF. A quarter of the pixels
    // of the full-resolution FFT guide, no extra FFT per frame, the resident mask
    // slice stays at 97 MB, and d^2 sums over R, G and B so the mask sees colour
    // differences a single-channel luminance guide cannot.
    //
    // build_sr_gate_features handles the one-channel FFT guide too, identically
    // on both backends, but no weights have been fitted there since the guide
    // changed -- so decline rather than run a trained function outside the domain
    // it was measured in.
    if (ref_means.c != 3) return Image();
    Image feat = build_sr_gate_features(ref_means, ref_vars, d_sq, sigma_sq,
                                       flow, tile_size, cfg);
    if (feat.h <= 0) return Image();
    return sr_gate_infer_cpu(feat);
}

}  // namespace hhsr
