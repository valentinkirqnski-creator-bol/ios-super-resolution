#include "accumulate.h"
#include "parallel_for.h"

#include <cmath>
#include <algorithm>

namespace isacpu {

namespace {
enum BayerColor { Red = 0, Green = 1, Blue = 2 };
int clamp_index(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }
}  // namespace

void debayer_subsample3(const std::vector<uint16_t>& raw, int width, int height,
                        const int cfa[2][2], float max_val, std::vector<Rgbf>& out_half) {
    int dimX = width / 2, dimY = height / 2;
    out_half.assign((size_t)dimX * dimY, Rgbf{});
    float factor = 1.0f / max_val;

    for (int y = 0; y < dimY; ++y) {
        for (int x = 0; x < dimX; ++x) {
            Rgbf pixel{};
            for (int ix = 0; ix < 2; ++ix) {
                for (int iy = 0; iy < 2; ++iy) {
                    int color = cfa[iy][ix];
                    float v = (float)raw[(size_t)(2 * y + iy) * width + (2 * x + ix)] * factor;
                    if (color == Green) pixel.g += v * 0.5f;
                    else if (color == Red) pixel.r = v;
                    else pixel.b = v;
                }
            }
            out_half[(size_t)y * dimX + x] = pixel;
        }
    }
}

void accumulate_images(const std::vector<uint16_t>& raw, int dimX, int dimY,
                      const int cfa[2][2],
                      std::vector<Rgbf>& pixel, std::vector<Rgbf>& total_weight,
                      const std::vector<Vec4f>& certainty_half, int certainty_stride_x,
                      const std::vector<Vec3f>& kernel_param,
                      const std::vector<Vec2f>& shift,
                      const float white_level[3], const float black_level[3]) {
    // Each (x,y) only ever writes its own pixel[idx]/total_weight[idx], so
    // splitting rows across threads is safe.
    parallel_for(dimY - 2, [&](int yi) {
        int y = yi + 1;
        for (int x = 1; x < dimX - 1; ++x) {
            size_t idx = (size_t)y * dimX + x;
            Rgbf px = pixel[idx];
            Rgbf tw = total_weight[idx];
            const Vec3f& kernel = kernel_param[idx];
            Vec2f sh = shift[idx];

            int sx = (int)std::round(sh.x);
            int sy = (int)std::round(sh.y);

            for (int py = -2; py <= 2; ++py) {
                for (int px2 = -2; px2 <= 2; ++px2) {
                    int ppsx = clamp_index(x + px2 + sx, dimX);
                    int ppsy = clamp_index(y + py + sy, dimY);
                    int ppx = clamp_index(x + px2, dimX);
                    int ppy = clamp_index(y + py, dimY);

                    int thisPixel = cfa[ppsy % 2][ppsx % 2];

                    float w = px2 * px2 * kernel.x + 2 * px2 * py * kernel.z + py * py * kernel.y;
                    w = std::exp(-0.5f * w);
                    if (!std::isfinite(w)) w = (px2 * py == 0) ? 1.f : 0.f;

                    float rawv = (float)raw[(size_t)ppsy * dimX + ppsx];
                    const Vec4f& cert = certainty_half[(size_t)(ppy / 2) * certainty_stride_x + (ppx / 2)];

                    if (thisPixel == Green) {
                        rawv = (rawv - black_level[1]) / white_level[1];
                        float certainty = cert.y;
                        if (!std::isfinite(certainty)) certainty = 0.f;
                        px.g += rawv * w * certainty;
                        tw.g += w * certainty;
                    } else if (thisPixel == Red) {
                        rawv = (rawv - black_level[0]) / white_level[0];
                        float certainty = cert.x;
                        if (!std::isfinite(certainty)) certainty = 0.f;
                        px.r += rawv * w * certainty;
                        tw.r += w * certainty;
                    } else {  // Blue
                        rawv = (rawv - black_level[2]) / white_level[2];
                        float certainty = cert.z;
                        if (!std::isfinite(certainty)) certainty = 0.f;
                        px.b += rawv * w * certainty;
                        tw.b += w * certainty;
                    }
                }
            }

            pixel[idx] = px;
            total_weight[idx] = tw;
        }
    });
}

}  // namespace isacpu
