#include "affine_warp.h"

#include <cmath>
#include <algorithm>

namespace isacpu {

Mat3 mat3_identity() {
    return {1, 0, 0,
            0, 1, 0,
            0, 0, 1};
}

Mat3 mat3_mul(const Mat3& a, const Mat3& b) {
    Mat3 res{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            float v = 0.f;
            for (int i = 0; i < 3; ++i) v += a[r * 3 + i] * b[i * 3 + c];
            res[r * 3 + c] = v;
        }
    return res;
}

Mat3 mat3_shift(float x, float y) {
    Mat3 m = mat3_identity();
    m[2] = x;
    m[5] = y;
    return m;
}

Mat3 mat3_rotation_deg(float angle_deg) {
    Mat3 m = mat3_identity();
    float rad = angle_deg / 180.f * 3.14159265358979323846f;
    float s = std::sin(rad), c = std::cos(rad);
    m[0] = c; m[3] = -s;
    m[1] = s; m[4] = c;
    return m;
}

Mat3 mat3_rot_around_center(float angle_deg, float width, float height) {
    Mat3 m = mat3_shift(width / 2.f, height / 2.f);
    m = mat3_mul(m, mat3_rotation_deg(angle_deg));
    m = mat3_mul(m, mat3_shift(-width / 2.f, -height / 2.f));
    return m;
}

Mat3 mat3_invert(const Mat3& m) {
    float a = m[0], b = m[1], tx = m[2];
    float c = m[3], d = m[4], ty = m[5];
    float det = a * d - b * c;
    float invdet = det != 0.f ? 1.f / det : 0.f;

    Mat3 out = mat3_identity();
    out[0] = d * invdet;
    out[1] = -b * invdet;
    out[3] = -c * invdet;
    out[4] = a * invdet;
    out[2] = (-d * tx + b * ty) * invdet;
    out[5] = (c * tx - a * ty) * invdet;
    return out;
}

namespace {
// Keys (1981) cubic convolution kernel, a=-0.5 (NPP's Cubic mode).
float cubic_weight(float t) {
    const float a = -0.5f;
    t = std::fabs(t);
    if (t <= 1.f) return (a + 2.f) * t * t * t - (a + 3.f) * t * t + 1.f;
    if (t < 2.f) return a * t * t * t - 5.f * a * t * t + 8.f * a * t - 4.f * a;
    return 0.f;
}
}  // namespace

bool bicubic_sample(const std::vector<float>& img, int width, int height, float x, float y, float& out) {
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    if (x0 - 1 < 0 || x0 + 2 >= width || y0 - 1 < 0 || y0 + 2 >= height) return false;

    float fx = x - x0, fy = y - y0;
    float wx[4], wy[4];
    for (int i = -1; i <= 2; ++i) {
        wx[i + 1] = cubic_weight(fx - i);
        wy[i + 1] = cubic_weight(fy - i);
    }

    float sum = 0.f;
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            float v = img[(size_t)(y0 + j) * width + (x0 + i)];
            sum += v * wx[i + 1] * wy[j + 1];
        }
    }
    out = sum;
    return true;
}

void warp_affine(const std::vector<float>& src, int width, int height, const Mat3& m,
                 std::vector<float>& dst) {
    Mat3 inv = mat3_invert(m);
    for (int dy = 0; dy < height; ++dy) {
        for (int dx = 0; dx < width; ++dx) {
            float sx = inv[0] * dx + inv[1] * dy + inv[2];
            float sy = inv[3] * dx + inv[4] * dy + inv[5];
            float v;
            if (bicubic_sample(src, width, height, sx, sy, v))
                dst[(size_t)dy * width + dx] = v;
        }
    }
}

}  // namespace isacpu
