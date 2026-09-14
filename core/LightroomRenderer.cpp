#include "LightroomRenderer.h"
#include "LightroomRendererData.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hhsr {
namespace {

using lr::kNH; using lr::kNS; using lr::kToneN;

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float luma(float r, float g, float b) { return 0.2126f*r + 0.7152f*g + 0.0722f*b; }

inline float srgb_oetf(float v) {
    v = clampf(v, 0.f, 1.f);
    return v <= 0.0031308f ? 12.92f*v : 1.055f*std::pow(v, 1.f/2.4f) - 0.055f;
}

// 1-D tone LUT on display luma, 256 uniform samples, linear interp.
inline float tone(float y) {
    y = clampf(y, 0.f, 1.f);
    const float p = y * (float)(kToneN - 1);
    const int i = (int)p;
    if (i >= kToneN - 1) return lr::kToneLut[kToneN - 1];
    return lr::kToneLut[i] + (lr::kToneLut[i+1] - lr::kToneLut[i]) * (p - (float)i);
}

// HSV, hue in [0,1]. Matches scratchpad/stage3.py rgb2hsv / hsv2rgb exactly.
inline void rgb2hsv(float r, float g, float b, float& h, float& s, float& v) {
    const float mx = std::max(r, std::max(g, b));
    const float mn = std::min(r, std::min(g, b));
    const float d = mx - mn;
    v = mx;
    s = mx > 1e-6f ? d / mx : 0.f;
    if (d <= 1e-6f) { h = 0.f; return; }
    float hh;
    if (mx == r)      hh = std::fmod((g - b) / d, 6.f);
    else if (mx == g) hh = ((b - r) / d) + 2.f;
    else              hh = ((r - g) / d) + 4.f;
    if (hh < 0.f) hh += 6.f;
    h = hh / 6.f;
}
inline void hsv2rgb(float h, float s, float v, float& r, float& g, float& b) {
    float hh = h * 6.f;
    int i = (int)std::floor(hh);
    const float f = hh - (float)i;
    i = ((i % 6) + 6) % 6;
    const float p = v * (1.f - s);
    const float q = v * (1.f - s*f);
    const float t = v * (1.f - s*(1.f - f));
    switch (i) {
        case 0: r=v; g=t; b=p; break;
        case 1: r=q; g=v; b=p; break;
        case 2: r=p; g=v; b=t; break;
        case 3: r=p; g=q; b=v; break;
        case 4: r=t; g=p; b=v; break;
        default:r=v; g=p; b=q; break;
    }
    r = clampf(r,0.f,1.f); g = clampf(g,0.f,1.f); b = clampf(b,0.f,1.f);
}

// Bilinear sample of an [kNH x kNS] map at fractional (hue*NH-0.5, sat*NS-0.5),
// hue wrapping, sat clamped. Matches stage3.py bil().
inline float bil_hs(const float* A, float h, float s) {
    const float hf = h*(float)kNH - 0.5f;
    const float sf = s*(float)kNS - 0.5f;
    int h0 = (int)std::floor(hf), s0 = (int)std::floor(sf);
    const float fh = hf - (float)h0, fs = sf - (float)s0;
    auto wrap = [](int a){ return ((a % kNH) + kNH) % kNH; };
    auto cl   = [](int c){ return c < 0 ? 0 : (c > kNS-1 ? kNS-1 : c); };
    const int h0w = wrap(h0), h1w = wrap(h0+1), s0c = cl(s0), s1c = cl(s0+1);
    const float v00 = A[h0w*kNS + s0c], v10 = A[h1w*kNS + s0c];
    const float v01 = A[h0w*kNS + s1c], v11 = A[h1w*kNS + s1c];
    return (v00*(1.f-fh) + v10*fh)*(1.f-fs) + (v01*(1.f-fh) + v11*fh)*fs;
}

// Separable running-sum box blur, clamped borders, divide by actual window
// count. Matches stage45.py boxblur(). src/dst are H*W float planes.
void box_blur(const std::vector<float>& src, std::vector<float>& dst,
              int W, int H, int r) {
    std::vector<float> tmp((size_t)W*H);
    // horizontal
    for (int y = 0; y < H; ++y) {
        const float* s = &src[(size_t)y*W];
        float* t = &tmp[(size_t)y*W];
        double acc = 0.0;
        int lo = 0, hi = -1;
        for (int x = 0; x < W; ++x) {
            const int nlo = std::max(0, x - r), nhi = std::min(W-1, x + r);
            while (hi < nhi) { ++hi; acc += s[hi]; }
            while (lo < nlo) { acc -= s[lo]; ++lo; }
            t[x] = (float)(acc / (double)(nhi - nlo + 1));
        }
    }
    // vertical
    dst.assign((size_t)W*H, 0.f);
    for (int x = 0; x < W; ++x) {
        double acc = 0.0;
        int lo = 0, hi = -1;
        for (int y = 0; y < H; ++y) {
            const int nlo = std::max(0, y - r), nhi = std::min(H-1, y + r);
            while (hi < nhi) { ++hi; acc += tmp[(size_t)hi*W + x]; }
            while (lo < nlo) { acc -= tmp[(size_t)lo*W + x]; ++lo; }
            dst[(size_t)y*W + x] = (float)(acc / (double)(nhi - nlo + 1));
        }
    }
}

}  // namespace

void lightroom_render(const uint16_t* rgb16, int W, int H,
                      const LightroomRenderOpts& opts,
                      std::vector<uint8_t>& out_rgb8) {
    out_rgb8.clear();
    if (!rgb16 || W <= 0 || H <= 0) return;
    const size_t N = (size_t)W * (size_t)H;
    const float* wb = opts.wb ? opts.wb : lr::kRefWB;
    const float* M = lr::kWbCamToSrgb;   // applied to WB'd linear

    // Stages 1-5 -> display RGB float buffer.
    std::vector<float> disp(N * 3);
    const float inv = 1.f / 65535.f;
    for (size_t i = 0; i < N; ++i) {
        // 1. white balance (camera-native linear -> WB'd linear)
        float r = (float)rgb16[i*3+0] * inv * wb[0];
        float g = (float)rgb16[i*3+1] * inv * wb[1];
        float b = (float)rgb16[i*3+2] * inv * wb[2];
        // 2. camera -> sRGB linear (row-major: out_i = sum_j M[i*3+j]*in_j)
        float lrr = M[0]*r + M[1]*g + M[2]*b;
        float lgg = M[3]*r + M[4]*g + M[5]*b;
        float lbb = M[6]*r + M[7]*g + M[8]*b;
        lrr = clampf(lrr,0.f,1.f); lgg = clampf(lgg,0.f,1.f); lbb = clampf(lbb,0.f,1.f);
        // 3. sRGB OETF -> display
        float dr = srgb_oetf(lrr), dg = srgb_oetf(lgg), db = srgb_oetf(lbb);
        // 4. tone curve on display luma, hue-preserving ratio
        float yi = luma(dr, dg, db);
        if (yi > 1e-4f) {
            const float k = tone(yi) / yi;
            dr = clampf(dr*k,0.f,1.f); dg = clampf(dg*k,0.f,1.f); db = clampf(db*k,0.f,1.f);
        }
        // 5. HueSatMap look (Adobe Color + Vibrance). All three maps are sampled
        // at the ORIGINAL (h,s) -- the correction is a function of the input
        // hue/sat, not of the already-shifted hue.
        float h, s, v;
        rgb2hsv(dr, dg, db, h, s, v);
        const float dh = bil_hs(lr::kHueShift, h, s);
        const float ms = bil_hs(lr::kSatMul,   h, s);
        const float mv = bil_hs(lr::kValMul,   h, s);
        float ho = h + dh; ho = ho - std::floor(ho);  // wrap to [0,1)
        s = clampf(s * ms, 0.f, 1.f);
        v = clampf(v * mv, 0.f, 1.f);
        hsv2rgb(ho, s, v, dr, dg, db);
        disp[i*3+0] = dr; disp[i*3+1] = dg; disp[i*3+2] = db;
    }

    // 6. Clarity: midtone-weighted local contrast on luma (spatial).
    if (opts.clarity && lr::kClarityAmount > 0.f) {
        std::vector<float> Y(N), base;
        for (size_t i = 0; i < N; ++i) Y[i] = luma(disp[i*3+0], disp[i*3+1], disp[i*3+2]);
        const int rad = std::max(1, (int)std::lround(std::min(W,H) * lr::kClarityRadiusFrac));
        box_blur(Y, base, W, H, rad);
        const float amt = lr::kClarityAmount;
        for (size_t i = 0; i < N; ++i) {
            const float y = Y[i];
            const float detail = y - base[i];
            float m = clampf((y - 0.5f)/0.5f, -1.f, 1.f);
            m = 1.f - m*m;                            // peak at mid grey
            const float yc = clampf(y + amt*detail*m, 1e-4f, 1.f);
            const float k = yc / std::max(y, 1e-4f);
            disp[i*3+0] = clampf(disp[i*3+0]*k, 0.f, 1.f);
            disp[i*3+1] = clampf(disp[i*3+1]*k, 0.f, 1.f);
            disp[i*3+2] = clampf(disp[i*3+2]*k, 0.f, 1.f);
        }
    }

    // 7. Color NR: chroma-domain smoothing, luma untouched (spatial).
    if (opts.color_nr && lr::kColorNrAmount > 0.f) {
        std::vector<float> dr(N), db(N), br, bb;
        std::vector<float> Yl(N);
        for (size_t i = 0; i < N; ++i) {
            const float r = disp[i*3+0], g = disp[i*3+1], b = disp[i*3+2];
            const float y = luma(r,g,b);
            Yl[i] = y; dr[i] = r - y; db[i] = b - y;
        }
        const int crad = std::max(1, (int)std::lround(std::min(W,H) * lr::kColorNrRadiusFrac));
        box_blur(dr, br, W, H, crad);
        box_blur(db, bb, W, H, crad);
        const float amt = lr::kColorNrAmount;
        for (size_t i = 0; i < N; ++i) {
            const float dr2 = dr[i] + (br[i] - dr[i]) * amt;
            const float db2 = db[i] + (bb[i] - db[i]) * amt;
            const float dg2 = -(0.2126f*dr2 + 0.0722f*db2) / 0.7152f;
            disp[i*3+0] = clampf(Yl[i] + dr2, 0.f, 1.f);
            disp[i*3+1] = clampf(Yl[i] + dg2, 0.f, 1.f);
            disp[i*3+2] = clampf(Yl[i] + db2, 0.f, 1.f);
        }
    }

    // 8. quantise
    out_rgb8.resize(N * 3);
    for (size_t i = 0; i < N*3; ++i)
        out_rgb8[i] = (uint8_t)std::lround(clampf(disp[i], 0.f, 1.f) * 255.f);
}

}  // namespace hhsr
