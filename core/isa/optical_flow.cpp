#include "optical_flow.h"
#include "parallel_for.h"

#include <cmath>
#include <algorithm>
#include <functional>

namespace isacpu {

namespace {

int mirror_index(int i, int n) {
    if (n == 1) return 0;
    int period = 2 * n;
    i %= period;
    if (i < 0) i += period;
    if (i >= n) i = period - 1 - i;
    return i;
}
int clamp_index(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

// Bilinear sample at texel-space coordinates (tx,ty) -- tx=0 is the center
// of pixel 0, matching CUDA's `texel = u*width - 0.5` normalized-coordinate
// convention once the +0.5/-0.5 offsets used throughout these kernels
// cancel out. `border` maps an out-of-range integer texel index into range
// (mirror_index or clamp_index).
float bilinear_sample(const std::vector<float>& img, int w, int h, float tx, float ty,
                      const std::function<int(int, int)>& border) {
    int x0 = (int)std::floor(tx), y0 = (int)std::floor(ty);
    float fx = tx - x0, fy = ty - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    int bx0 = border(x0, w), bx1 = border(x1, w), by0 = border(y0, h), by1 = border(y1, h);

    float v00 = img[(size_t)by0 * w + bx0], v10 = img[(size_t)by0 * w + bx1];
    float v01 = img[(size_t)by1 * w + bx0], v11 = img[(size_t)by1 * w + bx1];
    float top = v00 + (v10 - v00) * fx;
    float bot = v01 + (v11 - v01) * fx;
    return top + (bot - top) * fy;
}

Vec2f bilinear_sample_vec2(const std::vector<Vec2f>& img, int w, int h, float tx, float ty,
                          const std::function<int(int, int)>& border) {
    int x0 = (int)std::floor(tx), y0 = (int)std::floor(ty);
    float fx = tx - x0, fy = ty - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    int bx0 = border(x0, w), bx1 = border(x1, w), by0 = border(y0, h), by1 = border(y1, h);

    Vec2f v00 = img[(size_t)by0 * w + bx0], v10 = img[(size_t)by0 * w + bx1];
    Vec2f v01 = img[(size_t)by1 * w + bx0], v11 = img[(size_t)by1 * w + bx1];
    Vec2f top{v00.x + (v10.x - v00.x) * fx, v00.y + (v10.y - v00.y) * fx};
    Vec2f bot{v01.x + (v11.x - v01.x) * fx, v01.y + (v11.y - v01.y) * fx};
    return Vec2f{top.x + (bot.x - top.x) * fy, top.y + (bot.y - top.y) * fy};
}

}  // namespace

void warp_image(const std::vector<float>& source, int width, int height,
                const std::vector<Vec2f>& flow, std::vector<float>& out) {
    out.assign((size_t)width * height, 0.f);
    parallel_for(height, [&](int iy) {
        for (int ix = 0; ix < width; ++ix) {
            Vec2f shift = flow[(size_t)iy * width + ix];  // Clamp+Point at matching resolution: direct index
            float tx = ix + shift.x;
            float ty = iy + shift.y;
            out[(size_t)iy * width + ix] = bilinear_sample(source, width, height, tx, ty, mirror_index);
        }
    });
}

void compute_derivatives_3way(const std::vector<float>& warped_source, const std::vector<float>& target,
                              int width, int height,
                              std::vector<float>& Ix, std::vector<float>& Iy, std::vector<float>& Iz) {
    Ix.assign((size_t)width * height, 0.f);
    Iy.assign((size_t)width * height, 0.f);
    Iz.assign((size_t)width * height, 0.f);

    auto at_s = [&](int x, int y) { return warped_source[(size_t)mirror_index(y, height) * width + mirror_index(x, width)]; };
    auto at_t = [&](int x, int y) { return target[(size_t)mirror_index(y, height) * width + mirror_index(x, width)]; };

    parallel_for(height, [&](int iy) {
        for (int ix = 0; ix < width; ++ix) {
            float t0 = -at_s(ix + 2, iy) + 8.f * at_s(ix + 1, iy) - 8.f * at_s(ix - 1, iy) + at_s(ix - 2, iy);
            t0 /= 12.f;
            float t1 = -at_t(ix + 2, iy) + 8.f * at_t(ix + 1, iy) - 8.f * at_t(ix - 1, iy) + at_t(ix - 2, iy);
            t1 /= 12.f;
            Ix[(size_t)iy * width + ix] = (t0 + t1) * 0.5f;

            Iz[(size_t)iy * width + ix] = at_s(ix, iy) - at_t(ix, iy);

            t0 = -at_s(ix, iy + 2) + 8.f * at_s(ix, iy + 1) - 8.f * at_s(ix, iy - 1) + at_s(ix, iy - 2);
            t0 /= 12.f;
            t1 = -at_t(ix, iy + 2) + 8.f * at_t(ix, iy + 1) - 8.f * at_t(ix, iy - 1) + at_t(ix, iy - 2);
            t1 /= 12.f;
            Iy[(size_t)iy * width + ix] = (t0 + t1) * 0.5f;
        }
    });
}

void lucas_kanade_update(std::vector<Vec2f>& shifts, const std::vector<float>& Ix, const std::vector<float>& Iy,
                        const std::vector<float>& Iz, int width, int height, int half_window, float min_det) {
    // Ix/Iy/Iz are read-only here; each pxY only ever writes shifts for its
    // own row, so splitting rows across threads is safe.
    parallel_for(height - 2 * half_window, [&](int rowIdx) {
        int pxY = rowIdx + half_window;
        for (int pxX = half_window; pxX < width - half_window; ++pxX) {
            float a11 = 0, a12 = 0, a22 = 0;
            for (int y = -half_window; y <= half_window; ++y) {
                for (int x = -half_window; x <= half_window; ++x) {
                    float dx = Ix[(size_t)(pxY + y) * width + (pxX + x)];
                    float dy = Iy[(size_t)(pxY + y) * width + (pxX + x)];
                    a11 += dx * dx;
                    a12 += dx * dy;
                    a22 += dy * dy;
                }
            }
            float a = a11, b = a12, c = a12, d = a22;

            float theta = 0.5f * std::atan2(2.f * a * c + 2.f * b * d, a * a + b * b - c * c - d * d);
            float ct = std::cos(theta), st = std::sin(theta);
            float UT[4] = {ct, st, -st, ct};  // row-major 2x2, transposed U

            float S1 = a * a + b * b + c * c + d * d;
            float S2 = std::sqrt((a * a + b * b - c * c - d * d) * (a * a + b * b - c * c - d * d) +
                                 4.f * (a * c + b * d) * (a * c + b * d));
            float sigma1 = std::sqrt((S1 + S2) / 2.f);
            float sigma2 = std::sqrt((S1 - S2) / 2.f);

            // ISA-VERBATIM, bug included: the source
            // (Kernels/opticalFlow.cu:255) does `fminf(sigma1, sigma1)` --
            // comparing the largest singular value against itself, never
            // inspecting sigma2, the direction the 2x2 solve is actually
            // ill-conditioned in. Reproduced literally per the
            // identical-to-master requirement. Recorded consequence from
            // desktop testing of the min(sigma1, sigma2) alternative: with
            // the literal check, tiles with a strong single-direction
            // gradient (sigma2 ~= 0) pass, the near-singular matrix is
            // inverted, and the UV correction can explode (observed +-180px
            // flow on a burst whose true motion was ~30px). That IS the
            // shipped original's behavior; if it must be tamed, do it as an
            // explicitly-labeled option, not silently here.
            float smin = std::min(sigma1, sigma1);
            (void)sigma2;
            if (smin < min_det) continue;

            sigma1 = sigma1 != 0.f ? 1.0f / sigma1 : 0.f;
            sigma2 = sigma2 != 0.f ? 1.0f / sigma2 : 0.f;

            float S[4] = {sigma1, 0, 0, sigma2};

            float epsilon = 0.5f * std::atan2(2.f * a * b + 2.f * c * d, a * a - b * b + c * c - d * d);
            float ce = std::cos(epsilon), se = std::sin(epsilon);

            float s11 = (a * ct + c * st) * ce + (b * ct + d * st) * se;
            float s22 = (a * st - c * ct) * se + (-b * st + d * ct) * ce;
            s11 = s11 > 0.f ? 1.f : s11 < 0.f ? -1.f : 0.f;
            s22 = s22 > 0.f ? 1.f : s22 < 0.f ? -1.f : 0.f;

            float V[4] = {s11 * ce, -s22 * se, s11 * se, s22 * ce};

            float m0 = S[0] * UT[0] + S[1] * UT[2];
            float m1 = S[0] * UT[1] + S[1] * UT[3];
            float m2 = S[2] * UT[0] + S[3] * UT[2];
            float m3 = S[2] * UT[1] + S[3] * UT[3];

            float inv0 = V[0] * m0 + V[1] * m2;
            float inv1 = V[0] * m1 + V[1] * m3;
            float inv2 = V[2] * m0 + V[3] * m2;
            float inv3 = V[2] * m1 + V[3] * m3;

            float uv0 = 0.f, uv1 = 0.f;
            for (int y = -half_window; y <= half_window; ++y) {
                for (int x = -half_window; x <= half_window; ++x) {
                    float dx = Ix[(size_t)(pxY + y) * width + (pxX + x)];
                    float dy = Iy[(size_t)(pxY + y) * width + (pxX + x)];
                    float dt = Iz[(size_t)(pxY + y) * width + (pxX + x)];
                    uv0 += (inv0 * dx + inv1 * dy) * dt;
                    uv1 += (inv2 * dx + inv3 * dy) * dt;
                }
            }
            if (std::isnan(uv0)) uv0 = 0.f;
            if (std::isnan(uv1)) uv1 = 0.f;

            // ISA ships with the per-iteration +-2 clamp written out but
            // COMMENTED OUT (opticalFlow.cu:318-319: `/*UV[0] =
            // fmaxf(fminf(2.0f, UV[0]), -2.0f); ...*/`), so no clamp is
            // applied here either -- identical-to-master. (Desktop testing
            // once re-enabled it to stop per-iteration corrections
            // exploding past +-100px on large-motion bursts; that is a
            // tamed variant, not the shipped original.)

            // SIGN CONVENTION, settled by source-identity audit: this
            // port's 5-tap derivative stencil in compute_derivatives_3way
            // computes the STANDARD +dI/dx, whereas the CUDA kernel's
            // stencil (opticalFlow.cu:116-120) computes its NEGATION.
            // CUDA therefore pairs its negated gradients with `shift +=
            // UV`; this port pairs conventional gradients with `shift -=
            // UV`. The two compositions are numerically IDENTICAL (M is
            // quadratic in the gradients; the RHS vector flips sign with
            // them, so UV flips, and += of the flipped UV equals -= of
            // ours). Neither line may be changed without the other.
            Vec2f& s = shifts[(size_t)pxY * width + pxX];
            s.x -= uv0;
            s.y -= uv1;
        }
    });
}

void create_flow_field_from_tiles(const std::vector<Vec2f>& tiled_flow, int tileCountX, int tileCountY,
                                  int /*tileSize*/, int width, int height,
                                  Vec2f base_shift, float base_rotation, std::vector<Vec2f>& out) {
    out.assign((size_t)width * height, Vec2f{});
    float cosr = std::cos(base_rotation), sinr = std::sin(base_rotation);
    int half_w = width / 2, half_h = height / 2;

    parallel_for(height, [&](int pxY) {
        for (int pxX = 0; pxX < width; ++pxX) {
            Vec2f shift;
            shift.x = cosr * -base_shift.x - sinr * -base_shift.y;
            shift.y = sinr * -base_shift.x + cosr * -base_shift.y;

            float patchCenterX = (float)(pxX - half_w);
            float patchCenterY = (float)(pxY - half_h);

            shift.x += cosr * patchCenterX - sinr * patchCenterY - patchCenterX;
            shift.y += sinr * patchCenterX + cosr * patchCenterY - patchCenterY;

            float u = (pxX + 0.5f) / (float)width;
            float v = (pxY + 0.5f) / (float)height;
            float tx = u * tileCountX - 0.5f;
            float ty = v * tileCountY - 0.5f;
            Vec2f shiftPatch = bilinear_sample_vec2(tiled_flow, tileCountX, tileCountY, tx, ty, clamp_index);

            shift.x += shiftPatch.x;
            shift.y += shiftPatch.y;

            out[(size_t)pxY * width + pxX] = shift;
        }
    });
}

void lucas_kanade_refine(const std::vector<float>& source, const std::vector<float>& target,
                         int width, int height, const std::vector<Vec2f>& tiled_flow,
                         int tileSize, int tileCountX, int tileCountY, int iterations,
                         Vec2f base_shift, float base_rotation, float min_det, int window_size,
                         std::vector<Vec2f>& out_flow) {
    create_flow_field_from_tiles(tiled_flow, tileCountX, tileCountY, tileSize, width, height,
                                 base_shift, base_rotation, out_flow);

    std::vector<float> warped, Ix, Iy, Iz;
    for (int iter = 0; iter < iterations; ++iter) {
        warp_image(source, width, height, out_flow, warped);
        compute_derivatives_3way(warped, target, width, height, Ix, Iy, Iz);
        lucas_kanade_update(out_flow, Ix, Iy, Iz, width, height, window_size / 2, min_det);
    }
}

}  // namespace isacpu
