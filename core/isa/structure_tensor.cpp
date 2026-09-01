#include "structure_tensor.h"
#include "parallel_for.h"

#include <cmath>
#include <algorithm>

namespace isacpu {

namespace {

// CUAddressMode.Mirror: reflects symmetrically about the edge, duplicating
// the edge texel once ("...,1,0,0,1,2,..." going out past index 0).
int mirror_index(int i, int n) {
    if (n == 1) return 0;
    int period = 2 * n;
    i %= period;
    if (i < 0) i += period;
    if (i >= n) i = period - 1 - i;
    return i;
}

int clamp_index(int i, int n) {
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

}  // namespace

void compute_derivatives(const std::vector<float>& img, int width, int height,
                         std::vector<float>& dx, std::vector<float>& dy) {
    dx.assign((size_t)width * height, 0.f);
    dy.assign((size_t)width * height, 0.f);
    auto at = [&](int x, int y) { return img[(size_t)y * width + x]; };

    parallel_for(height, [&](int iy) {
        for (int ix = 0; ix < width; ++ix) {
            int xm2 = mirror_index(ix - 2, width), xm1 = mirror_index(ix - 1, width);
            int xp1 = mirror_index(ix + 1, width), xp2 = mirror_index(ix + 2, width);
            float gx = (-at(xp2, iy) + 8.f * at(xp1, iy) - 8.f * at(xm1, iy) + at(xm2, iy)) / 12.f;

            int ym2 = mirror_index(iy - 2, height), ym1 = mirror_index(iy - 1, height);
            int yp1 = mirror_index(iy + 1, height), yp2 = mirror_index(iy + 2, height);
            float gy = (-at(ix, yp2) + 8.f * at(ix, yp1) - 8.f * at(ix, ym1) + at(ix, ym2)) / 12.f;

            dx[(size_t)iy * width + ix] = gx;
            dy[(size_t)iy * width + ix] = gy;
        }
    });
}

void compute_structure_tensor_raw(const std::vector<float>& dx, const std::vector<float>& dy,
                                  int width, int height, std::vector<Vec3f>& out) {
    out.assign((size_t)width * height, Vec3f{});
    parallel_for(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            size_t i = (size_t)y * width + x;
            float gx = dx[i], gy = dy[i];
            out[i] = Vec3f{gx * gx, gy * gy, gx * gy};
        }
    });
}

std::vector<float> gaussian_filter_1d(float sigma) {
    if (sigma <= 0.f) return {0, 0, 0, 0, 1, 0, 0, 0, 0};

    int radius = (int)(sigma / 0.6f - 0.4f);
    int size = radius * 2 + 1 + 2;
    size = std::min(size, 99);

    std::vector<float> ret(size);
    int center = size / 2;
    for (int i = 0; i < size; ++i) {
        int x = i - center;
        ret[i] = std::exp(-(float)(x * x) / (2.f * sigma * sigma));
    }
    float sum = 0.f;
    for (float v : ret) sum += v;
    for (float& v : ret) v /= sum;
    return ret;
}

void gaussian_blur_tensor(const std::vector<Vec3f>& in, int width, int height,
                          const std::vector<float>& kernel1d, std::vector<Vec3f>& out) {
    int radius = (int)kernel1d.size() / 2;
    std::vector<Vec3f> tmp((size_t)width * height);

    // Row pass.
    parallel_for(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            Vec3f acc{};
            for (int k = -radius; k <= radius; ++k) {
                int sx = clamp_index(x + k, width);
                const Vec3f& v = in[(size_t)y * width + sx];
                float w = kernel1d[k + radius];
                acc.x += v.x * w; acc.y += v.y * w; acc.z += v.z * w;
            }
            tmp[(size_t)y * width + x] = acc;
        }
    });

    // Column pass.
    out.assign((size_t)width * height, Vec3f{});
    parallel_for(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            Vec3f acc{};
            for (int k = -radius; k <= radius; ++k) {
                int sy = clamp_index(y + k, height);
                const Vec3f& v = tmp[(size_t)sy * width + x];
                float w = kernel1d[k + radius];
                acc.x += v.x * w; acc.y += v.y * w; acc.z += v.z * w;
            }
            out[(size_t)y * width + x] = acc;
        }
    });
}

void compute_kernel_param(std::vector<Vec3f>& tensor, int width, int height,
                          float Dth, float Dtr, float kDetail, float kDenoise,
                          float kStretch, float kShrink) {
    parallel_for(height, [&](int y) {
      for (int x = 0; x < width; ++x) {
        size_t i = (size_t)y * width + x;
        float a11 = tensor[i].x, a22 = tensor[i].y, a12 = tensor[i].z;

        float help = std::sqrt((a22 - a11) * (a22 - a11) + 4.f * a12 * a12);
        float c = 2.f * a12;
        float s = a22 - a11 + help;

        float norm = std::sqrt(c * c + s * s);
        if (norm > 0.f) { c /= norm; s /= norm; }
        else { c = 1.f; s = 0.f; }

        float lam1 = (a11 + a22 + help) / 2.f;
        float lam2 = (a11 + a22 - help) / 2.f;

        float A = 1.f + std::sqrt((lam1 - lam2) * (lam1 - lam2) / ((lam1 + lam2) * (lam1 + lam2)));
        float D = 1.f - std::sqrt(lam1) / Dtr + Dth;
        D = std::max(0.f, std::min(1.f, D));

        float k1h = kDetail * kStretch * A;
        float k2h = kDetail / kShrink * A;

        float k1 = (1.f - D) * k1h + D * kDetail * kDenoise;
        float k2 = (1.f - D) * k2h + D * kDetail * kDenoise;
        k1 *= k1;
        k2 *= k2;

        float x2 = c, y2 = s, x1 = s, y1 = -c;

        float b11 = k1 * x1 * x1 + x2 * x2 * k2;
        float b12 = k1 * x1 * y1 + x2 * y2 * k2;
        float b22 = k1 * y1 * y1 + y2 * y2 * k2;

        float det = b11 * b22 - b12 * b12 + 0.0000000001f;

        tensor[i] = Vec3f{b22 / det, b11 / det, -b12 / det};
      }
    });
}

}  // namespace isacpu
