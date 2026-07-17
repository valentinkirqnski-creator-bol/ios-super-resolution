#pragma once
//
// 2x2 linear-algebra helpers — port of handheld_super_resolution/linalg.py
//
#include "types.h"
#include <cmath>

namespace hhsr {

// Matches utils.EPSILON_DIV
static constexpr f32 EPSILON_DIV = 1e-10f;

// solve_2x2: A * X = B (analytical).
inline void solve_2x2(f32 a00, f32 a01, f32 a10, f32 a11,
                      f32 b0, f32 b1, f32 x[2]) {
    f32 det_a = a00 * a11 - a01 * a10;
    x[0] = (a11 * b0 - a01 * b1) / det_a;
    x[1] = (a00 * b1 - a10 * b0) / det_a;
}

// invert_2x2 — on singular (|det| <= EPSILON_DIV) writes identity.
inline void invert_2x2(f32 m00, f32 m01, f32 m10, f32 m11,
                       f32 out[2][2]) {
    f32 det = m00 * m11 - m01 * m10;
    if (std::fabs(det) > EPSILON_DIV) {
        f32 det_i = 1.f / det;
        out[0][0] =  m11 * det_i;
        out[0][1] = -m01 * det_i;
        out[1][0] = -m10 * det_i;
        out[1][1] =  m00 * det_i;
    } else {
        out[0][0] = 1.f; out[0][1] = 0.f;
        out[1][0] = 0.f; out[1][1] = 1.f;
    }
}

// Symmetric convenience: M = [[xx,xy],[xy,yy]] -> inverse packed as (ixx, ixy, iyy).
// Matches invert_2x2 (identity fallback on singular).
inline void invert_sym_2x2(f32 xx, f32 xy, f32 yy, f32& ixx, f32& ixy, f32& iyy) {
    f32 out[2][2];
    invert_2x2(xx, xy, xy, yy, out);
    ixx = out[0][0];
    ixy = out[0][1];
    iyy = out[1][1];
}

// quad_mat_prod: X^T A X with X = [x1, x2]
inline f32 quad_mat_prod(f32 a00, f32 a01, f32 a10, f32 a11, f32 x1, f32 x2) {
    return a00 * x1 * x1 + x1 * x2 * (a01 + a10) + a11 * x2 * x2;
}

inline f32 quad_mat_prod_sym(f32 ixx, f32 ixy, f32 iyy, f32 x1, f32 x2) {
    return quad_mat_prod(ixx, ixy, ixy, iyy, x1, x2);
}

// get_real_polyroots_2 — roots[0] has largest |·|
inline void real_polyroots_2(f32 a, f32 b, f32 c, f32 roots[2]) {
    f32 delta = std::max(b * b - 4.f * a * c, 0.f);
    f32 r1 = (-b + std::sqrt(delta)) / (2.f * a);
    f32 r2 = (-b - std::sqrt(delta)) / (2.f * a);
    if (std::fabs(r1) >= std::fabs(r2)) { roots[0] = r1; roots[1] = r2; }
    else                                { roots[0] = r2; roots[1] = r1; }
}

// get_eigen_val_2x2
inline void eigen_val_2x2(f32 m00, f32 m01, f32 m10, f32 m11, f32 l[2]) {
    f32 b = -(m00 + m11);
    f32 c = m00 * m11 - m01 * m10;
    real_polyroots_2(1.f, b, c, l);
}

// get_eigen_vect_2x2 — M.e1 = l[0].e1 style; uses l[1] as in Python
inline void eigen_vect_2x2(f32 m00, f32 m01, f32 m10, f32 m11,
                           const f32 l[2], f32 e1[2], f32 e2[2]) {
    // Python: if M[0,1] == 0 and M[0,0] == M[1,1]
    if (m01 == 0.f && m00 == m11) {
        e1[0] = 1.f; e1[1] = 0.f;
        e2[0] = 0.f; e2[1] = 1.f;
        return;
    }
    // averaging 2 for increased reliability
    e1[0] = m00 + m01 - l[1];
    e1[1] = m10 + m11 - l[1];

    if (e1[0] == 0.f) {
        e1[1] = 1.f;
        e2[0] = 1.f;
        e2[1] = 0.f;
    } else if (e1[1] == 0.f) {
        e1[0] = 1.f;
        e2[0] = 0.f;
        e2[1] = 1.f;
    } else {
        f32 norm_ = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1]);
        e1[0] /= norm_;
        e1[1] /= norm_;
        f32 sign = std::copysign(1.f, e1[0]);
        e2[1] = std::fabs(e1[0]);
        e2[0] = -e1[1] * sign;
    }
}

// get_eigen_elmts_2x2
inline void eigen_elmts_2x2(f32 m00, f32 m01, f32 m10, f32 m11,
                            f32 l[2], f32 e1[2], f32 e2[2]) {
    eigen_val_2x2(m00, m01, m10, m11, l);
    eigen_vect_2x2(m00, m01, m10, m11, l, e1, e2);
}

// interpolate_cov — bilinear over 4 neighbor covs [2][2][2][2]
// covs[yi][xi][i][j], center_pos = {y, x}; frac from modf as in Python.
inline void interpolate_cov(const f32 covs[2][2][2][2], f32 center_y, f32 center_x,
                            f32 out[2][2]) {
    f32 reframed_posx = center_x - std::floor(center_x); // matches math.modf for >= 0
    f32 reframed_posy = center_y - std::floor(center_y);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            out[i][j] =
                covs[0][0][i][j] * (1.f - reframed_posx) * (1.f - reframed_posy) +
                covs[0][1][i][j] * (reframed_posx)       * (1.f - reframed_posy) +
                covs[1][0][i][j] * (1.f - reframed_posx) * (reframed_posy) +
                covs[1][1][i][j] * (reframed_posx)       * (reframed_posy);
        }
    }
}

// bilinear_interpolation — values[2][2], pos = {y, x} in [0,1]^2
inline f32 bilinear_interpolation(const f32 values[2][2], f32 posy, f32 posx) {
    return values[0][0] * (1.f - posx) * (1.f - posy) +
           values[0][1] * (posx)       * (1.f - posy) +
           values[1][0] * (1.f - posx) * (posy) +
           values[1][1] * (posx)       * (posy);
}

} // namespace hhsr
