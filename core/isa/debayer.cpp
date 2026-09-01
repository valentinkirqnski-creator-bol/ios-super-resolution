#include "debayer.h"
#include "structure_tensor.h"
#include "pre_alignment.h"  // fourier_filter_apply (tracking-path high-pass)
#include "parallel_for.h"

#include <cmath>
#include <algorithm>
#include <functional>

namespace isacpu {

namespace {

enum BayerColor { Red = 0, Green = 1, Blue = 2 };

inline float raw_at(const std::vector<uint16_t>& raw, int width, int x, int y) {
    return (float)raw[(size_t)y * width + x];
}

}  // namespace

void debayer_green(const std::vector<uint16_t>& raw, int width, int height,
                   const int cfa[2][2], const float black[3], const float scale[3],
                   std::vector<Rgbf>& out) {
    out.assign((size_t)width * height, Rgbf{});

    auto rawr = [&](int x, int y) { return (raw_at(raw, width, x, y) - black[0]) * scale[0]; };
    auto rawg = [&](int x, int y) { return (raw_at(raw, width, x, y) - black[1]) * scale[1]; };
    auto rawb = [&](int x, int y) { return (raw_at(raw, width, x, y) - black[2]) * scale[2]; };

    parallel_for(height - 4, [&](int rowIdx) {
        int y = rowIdx + 2;
        for (int x = 2; x < width - 2; ++x) {
            int thisPixel = cfa[y % 2][x % 2];
            float g = 0.f;

            if (thisPixel == Green) {
                g = rawg(x, y);
            } else if (thisPixel == Red || thisPixel == Blue) {
                auto same = (thisPixel == Red) ? std::function<float(int, int)>(rawr) : std::function<float(int, int)>(rawb);
                float p = same(x, y);
                float xMinus2 = same(x - 2, y);
                float xMinus1 = rawg(x - 1, y);
                float xPlus1 = rawg(x + 1, y);
                float xPlus2 = same(x + 2, y);

                float yMinus2 = same(x, y - 2);
                float yMinus1 = rawg(x, y - 1);
                float yPlus1 = rawg(x, y + 1);
                float yPlus2 = same(x, y + 2);

                float gradientX = 0.5f * std::fabs(xPlus1 - xMinus1);
                float gradientY = 0.5f * std::fabs(yPlus1 - yMinus1);

                float laplaceX = 0.25f * std::fabs(2.0f * p - xMinus2 - xPlus2);
                float laplaceY = 0.25f * std::fabs(2.0f * p - yMinus2 - yPlus2);

                float interpolX = 0.125f * (-xMinus2 + 4.0f * xMinus1 + 2.0f * p + 4.0f * xPlus1 - xPlus2);
                float interpolY = 0.125f * (-yMinus2 + 4.0f * yMinus1 + 2.0f * p + 4.0f * yPlus1 - yPlus2);

                float weight = (gradientY + laplaceY) / (gradientX + gradientY + laplaceX + laplaceY + 0.000000001f);

                g = weight * interpolX + (1.0f - weight) * interpolY;
            }
            out[(size_t)y * width + x].g = g;
        }
    });
}

void debayer_red_blue(const std::vector<uint16_t>& raw, int width, int height,
                      const int cfa[2][2], const float black[3], const float scale[3],
                      std::vector<Rgbf>& inout) {
    auto rawr = [&](int x, int y) { return (raw_at(raw, width, x, y) - black[0]) * scale[0]; };
    auto rawb = [&](int x, int y) { return (raw_at(raw, width, x, y) - black[2]) * scale[2]; };
    auto green = [&](int x, int y) { return inout[(size_t)y * width + x].g; };

    parallel_for(height - 4, [&](int rowIdx) {
        int y = rowIdx + 2;
        for (int x = 2; x < width - 2; ++x) {
            int thisPixel = cfa[y % 2][x % 2];
            int thisRow = cfa[y % 2][(x + 1) % 2];
            float r, b;
            float g = green(x, y);

            if (thisPixel == Green) {
                if (thisRow == Red) {
                    float xMinus1r = rawr(x - 1, y), xPlus1r = rawr(x + 1, y);
                    float xMinus1g = green(x - 1, y), xPlus1g = green(x + 1, y);
                    r = g + 0.5f * ((xMinus1r - xMinus1g) + (xPlus1r - xPlus1g));

                    float yMinus1b = rawb(x, y - 1), yPlus1b = rawb(x, y + 1);
                    float yMinus1g = green(x, y - 1), yPlus1g = green(x, y + 1);
                    b = g + 0.5f * ((yMinus1b - yMinus1g) + (yPlus1b - yPlus1g));
                } else {
                    float xMinus1b = rawb(x - 1, y), xPlus1b = rawb(x + 1, y);
                    float xMinus1g = green(x - 1, y), xPlus1g = green(x + 1, y);
                    b = g + 0.5f * ((xMinus1b - xMinus1g) + (xPlus1b - xPlus1g));

                    float yMinus1r = rawr(x, y - 1), yPlus1r = rawr(x, y + 1);
                    float yMinus1g = green(x, y - 1), yPlus1g = green(x, y + 1);
                    r = g + 0.5f * ((yMinus1r - yMinus1g) + (yPlus1r - yPlus1g));
                }
            } else if (thisPixel == Red) {
                r = rawr(x, y);
                float mmB = rawb(x - 1, y - 1), pmB = rawb(x + 1, y - 1);
                float ppB = rawb(x + 1, y + 1), mpB = rawb(x - 1, y + 1);
                float mmG = green(x - 1, y - 1), pmG = green(x + 1, y - 1);
                float ppG = green(x + 1, y + 1), mpG = green(x - 1, y + 1);
                b = g + 0.25f * ((mmB - mmG) + (pmB - pmG) + (ppB - ppG) + (mpB - mpG));
            } else {  // Blue
                b = rawb(x, y);
                float mmR = rawr(x - 1, y - 1), pmR = rawr(x + 1, y - 1);
                float ppR = rawr(x + 1, y + 1), mpR = rawr(x - 1, y + 1);
                float mmG = green(x - 1, y - 1), pmG = green(x + 1, y - 1);
                float ppG = green(x + 1, y + 1), mpG = green(x - 1, y + 1);
                r = g + 0.25f * ((mmR - mmG) + (pmR - pmG) + (ppR - ppG) + (mpR - mpG));
            }
            inout[(size_t)y * width + x].r = r;
            inout[(size_t)y * width + x].b = b;
        }
    });
}

namespace {
float apply_srgb_gamma_1(float v) {
    if (std::isnan(v)) v = 0.f;
    // GammasRGB (kernel.cu:324-326) clamps to [0,1] BEFORE the transfer
    // curve -- demosaic over/undershoot at hard edges saturates instead of
    // leaving the valid range. An earlier revision omitted this clamp;
    // caught by the source-identity audit.
    v = std::min(1.0f, std::max(0.0f, v));
    if (v <= 0.0031308f) return 12.92f * v;
    return (1.0f + 0.055f) * std::pow(v, 1.0f / 2.4f) - 0.055f;
}
}  // namespace

void apply_srgb_gamma(std::vector<Rgbf>& img) {
    for (auto& p : img) {
        p.r = apply_srgb_gamma_1(p.r);
        p.g = apply_srgb_gamma_1(p.g);
        p.b = apply_srgb_gamma_1(p.b);
    }
}

void rgb_to_gray(const std::vector<Rgbf>& img, std::vector<float>& out, bool green_only) {
    out.resize(img.size());
    if (green_only) {
        for (size_t i = 0; i < img.size(); ++i) out[i] = img[i].g;
    } else {
        for (size_t i = 0; i < img.size(); ++i)
            out[i] = 0.2989f * img[i].r + 0.5870f * img[i].g + 0.1141f * img[i].b;
    }
}

void debayer_full_res(const std::vector<uint16_t>& raw, int width, int height,
                      const int cfa[2][2], const float black[3], const float white_level[3],
                      const float camera_white[3], std::vector<Rgbf>& out) {
    float scaling[3] = {1.0f / camera_white[0], 1.0f / camera_white[1], 1.0f / camera_white[2]};

    debayer_green(raw, width, height, cfa, black, scaling, out);
    debayer_red_blue(raw, width, height, cfa, black, scaling, out);

    // Undo the as-shot-WB scaling applied above (kept only for the
    // gradient/laplacian comparisons); the real WB lives in the colour
    // matrix downstream.
    float undo[3] = {
        camera_white[0] / white_level[0],
        camera_white[1] / white_level[1],
        camera_white[2] / white_level[2],
    };
    for (auto& p : out) {
        p.r *= undo[0];
        p.g *= undo[1];
        p.b *= undo[2];
    }
}

void debayer_bw_gauss(const std::vector<uint16_t>& raw, int width, int height,
                      const int cfa[2][2], const float black[3], const float white_level[3],
                      const float camera_white[3], float sigma, bool green_only,
                      std::vector<float>& out,
                      bool tracking_fourier, int clear_axis,
                      float high_pass, float high_pass_sigma) {
    std::vector<Rgbf> rgb;
    debayer_full_res(raw, width, height, cfa, black, white_level, camera_white, rgb);
    apply_srgb_gamma(rgb);

    std::vector<float> gray;
    rgb_to_gray(rgb, gray, green_only);

    // DeBayerBWGaussWB's non-skip path (Controller.cs:2409-2414) runs the
    // Fourier high-pass BETWEEN the grayscale conversion and the Gaussian
    // blur -- not after the blur. The operators commute analytically, but
    // the blur's replicate border and the FFT's periodic one make the order
    // observable near edges, so the tracking image must filter here.
    if (tracking_fourier)
        fourier_filter_apply(gray, width, height, clear_axis, high_pass, high_pass_sigma);

    std::vector<float> kernel1d = gaussian_filter_1d(sigma);
    int radius = (int)kernel1d.size() / 2;
    out.assign((size_t)width * height, 0.f);

    std::vector<float> tmp((size_t)width * height);
    auto clamp_index = [](int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); };

    parallel_for(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.f;
            for (int k = -radius; k <= radius; ++k) {
                int sx = clamp_index(x + k, width);
                acc += gray[(size_t)y * width + sx] * kernel1d[k + radius];
            }
            tmp[(size_t)y * width + x] = acc;
        }
    });
    parallel_for(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.f;
            for (int k = -radius; k <= radius; ++k) {
                int sy = clamp_index(y + k, height);
                acc += tmp[(size_t)sy * width + x] * kernel1d[k + radius];
            }
            out[(size_t)y * width + x] = acc;
        }
    });
}

}  // namespace isacpu
