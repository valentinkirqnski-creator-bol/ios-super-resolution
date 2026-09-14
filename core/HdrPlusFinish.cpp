#include "HdrPlusFinish.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "parallel.h"

namespace hhsr {
namespace {

constexpr float kMax = 65535.f;

// Reference capture (handheld_sr_x2) WB + camera->sRGB, used when opts leaves
// them null. Matches the DNG's private tag 65000.
constexpr float kRefWB[3] = {2.0249f, 1.0f, 1.8623f};
constexpr float kRefCcm[9] = {   // row-major: out_i = sum_j ccm[i*3+j]*in_j
     1.36753f, -0.12669f, -0.24083f,
    -0.23965f,  1.58429f, -0.34463f,
    -0.02004f, -0.47008f,  1.49012f
};

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float sat16(float v) { return clampf(v, 0.f, kMax); }

// HDR+ util.cpp gamma_correct (linear [0,65535] -> IEC sRGB gamma [0,65535]).
inline float gamma_correct(float v) {
    if (v < 200.f) return sat16(12.92f * v);
    return sat16(680.552897f * std::pow(std::max(v, 0.f), 0.416667f) - 3604.425f);
}
// HDR+ util.cpp gamma_inverse (gamma [0,65535] -> linear [0,65535]).
inline float gamma_inverse(float v) {
    if (v < 2575.f) return sat16(0.0774f * v);
    return sat16(std::pow(v / kMax + 0.055f, 2.4f) * 57632.49226f);
}
// HDR+ finish.cpp brighten.
inline float brighten(float v, float g) { return sat16(g * v); }

// HDR+ finish.cpp tone_map luma-weight distribution (exposure-fusion paper).
inline float normal_dist(float v) {
    const float t = v / kMax - 0.5f;
    return std::exp(-12.5f * t * t);
}

// gauss_7x7 (util.cpp): separable 1D kernel, std dev 4/3, repeat-edge borders.
const float kG7[7] = {0.026267f, 0.100742f, 0.225511f, 0.29496f,
                      0.225511f, 0.100742f, 0.026267f};

void gauss_7x7(const std::vector<float>& src, std::vector<float>& dst, int W, int H) {
    std::vector<float> tmp((size_t)W * H);
    // horizontal (rows independent)
    parallel_rows(H, 0, [&](int y) {
        auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
        const float* s = &src[(size_t)y * W];
        float* t = &tmp[(size_t)y * W];
        for (int x = 0; x < W; ++x) {
            float a = 0.f;
            for (int k = -3; k <= 3; ++k) a += s[cl(x + k, W - 1)] * kG7[k + 3];
            t[x] = a;
        }
    });
    // vertical (output rows independent)
    dst.assign((size_t)W * H, 0.f);
    parallel_rows(H, 0, [&](int y) {
        auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
        float* d = &dst[(size_t)y * W];
        for (int x = 0; x < W; ++x) {
            float a = 0.f;
            for (int k = -3; k <= 3; ++k) a += tmp[(size_t)cl(y + k, H - 1) * W + x] * kG7[k + 3];
            d[x] = a;
        }
    });
}

// HDR+ finish.cpp combine: 2-level Laplacian exposure fusion of two greyscale
// images weighted by dist(). Mirrors the num_layers==2 unrolling exactly:
//   out = (im1 - G(im1))*m1 + (im2 - G(im2))*m2
//       + G(G(im1))*G(m1)   + G(G(im2))*G(m2)
// where m1/m2 are the per-pixel dist weights and G is gauss_7x7.
void combine(const std::vector<float>& im1, const std::vector<float>& im2,
             int W, int H, std::vector<float>& out) {
    const size_t N = (size_t)W * H;
    std::vector<float> m1(N), m2(N);
    for (size_t i = 0; i < N; ++i) {
        const float w1 = normal_dist(im1[i]);
        const float w2 = normal_dist(im2[i]);
        const float s = w1 + w2;
        m1[i] = (s > 1e-12f) ? w1 / s : 0.5f;
        m2[i] = 1.f - m1[i];
    }
    std::vector<float> b1, b2;                 // G(im1), G(im2)
    gauss_7x7(im1, b1, W, H);
    gauss_7x7(im2, b2, W, H);

    out.assign(N, 0.f);
    // Detail band: (im - G(im)) * mask (original masks).
    for (size_t i = 0; i < N; ++i)
        out[i] = (im1[i] - b1[i]) * m1[i] + (im2[i] - b2[i]) * m2[i];

    // Advance one octave: base = G(G(im)), masks -> G(mask).
    std::vector<float> bb1, bb2, gm1, gm2;
    gauss_7x7(b1, bb1, W, H);
    gauss_7x7(b2, bb2, W, H);
    gauss_7x7(m1, gm1, W, H);
    gauss_7x7(m2, gm2, W, H);
    for (size_t i = 0; i < N; ++i)
        out[i] = sat16(out[i] + bb1[i] * gm1[i] + bb2[i] * gm2[i]);
}

// HDR+ finish.cpp tone_map: iterative exposure-fusion dynamic-range compression
// on the luma, reintroducing colour by ratio. rgb is [0,65535] linear sRGB.
void tone_map(std::vector<float>& rgb, int W, int H, float comp, float gain) {
    const size_t N = (size_t)W * H;
    std::vector<float> gray(N), dark(N);
    for (size_t i = 0; i < N; ++i)
        gray[i] = (rgb[i * 3 + 0] + rgb[i * 3 + 1] + rgb[i * 3 + 2]) / 3.f;
    dark = gray;

    const int num_passes = 3;
    const float comp_const = 1.f + comp / num_passes;
    const float gain_const = 1.f + gain / num_passes;
    const float comp_slope = (comp - comp_const) / (num_passes - 1);
    const float gain_slope = (gain - gain_const) / (num_passes - 1);

    std::vector<float> bright(N), dg(N), bg(N), fused;
    for (int pass = 0; pass < num_passes; ++pass) {
        const float norm_comp = pass * comp_slope + comp_const;
        const float norm_gain = pass * gain_slope + gain_const;
        for (size_t i = 0; i < N; ++i) {
            bright[i] = brighten(dark[i], norm_comp);
            dg[i] = gamma_correct(dark[i]);
            bg[i] = gamma_correct(bright[i]);
        }
        combine(dg, bg, W, H, fused);          // dark_gamma <- fused
        for (size_t i = 0; i < N; ++i)
            dark[i] = brighten(gamma_inverse(fused[i]), norm_gain);
    }

    // Reintroduce colour: scale each channel by dark/grayscale.
    for (size_t i = 0; i < N; ++i) {
        const float k = dark[i] / std::max(1.f, gray[i]);
        rgb[i * 3 + 0] = sat16(rgb[i * 3 + 0] * k);
        rgb[i * 3 + 1] = sat16(rgb[i * 3 + 1] * k);
        rgb[i * 3 + 2] = sat16(rgb[i * 3 + 2] * k);
    }
}

// HDR+ finish.cpp contrast: scaled-cosine S-curve, then black subtract + scale.
inline float contrast_curve(float v, float slope, float inner_constant,
                            float constant, float factor,
                            float black_level, float white_scale) {
    const float s = slope * std::sin(factor * v - inner_constant) + constant;
    return sat16((sat16(s) - black_level) * white_scale);
}

}  // namespace

void hdrplus_finish(const uint16_t* rgb16, int W, int H,
                    const HdrPlusFinishOpts& opts,
                    std::vector<uint8_t>& out_rgb8) {
    out_rgb8.clear();
    if (!rgb16 || W <= 0 || H <= 0) return;
    const size_t N = (size_t)W * H;
    const float* wb = opts.wb ? opts.wb : kRefWB;
    const float* ccm = opts.ccm ? opts.ccm : kRefCcm;

    // Stages 1-2: white balance + CCM -> linear sRGB [0,65535].
    std::vector<float> rgb(N * 3);
    for (size_t i = 0; i < N; ++i) {
        const float r = (float)rgb16[i * 3 + 0] * wb[0];
        const float g = (float)rgb16[i * 3 + 1] * wb[1];
        const float b = (float)rgb16[i * 3 + 2] * wb[2];
        rgb[i * 3 + 0] = sat16(ccm[0] * r + ccm[1] * g + ccm[2] * b);
        rgb[i * 3 + 1] = sat16(ccm[3] * r + ccm[4] * g + ccm[5] * b);
        rgb[i * 3 + 2] = sat16(ccm[6] * r + ccm[7] * g + ccm[8] * b);
    }

    // Stage 3: HDR+ tone map.
    tone_map(rgb, W, H, opts.compression, opts.gain);

    // Stage 4: gamma correction.
    for (size_t i = 0; i < N * 3; ++i) rgb[i] = gamma_correct(rgb[i]);

    // Stage 5: global contrast (precompute the curve constants once).
    {
        const float strength = std::max(1e-3f, opts.contrast_strength);
        const float scale = 0.8f + 0.3f / std::min(1.f, strength);
        const float inner_constant = 3.141592f / (2.f * scale);
        const float sin_constant = std::sin(inner_constant);
        const float slope = kMax / (2.f * sin_constant);
        const float constant = slope * sin_constant;
        const float factor = 3.141592f / (scale * kMax);
        const float white_scale = kMax / (kMax - opts.black_level);
        for (size_t i = 0; i < N * 3; ++i)
            rgb[i] = contrast_curve(rgb[i], slope, inner_constant, constant,
                                    factor, opts.black_level, white_scale);
    }

    // Stage 6: 8-bit (HDR+ u8bit_interleaved: v / 256).
    out_rgb8.resize(N * 3);
    for (size_t i = 0; i < N * 3; ++i)
        out_rgb8[i] = (uint8_t)clampf(rgb[i] / 256.f, 0.f, 255.f);
}

}  // namespace hhsr
