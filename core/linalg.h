#pragma once
//
// 2x2 linear-algebra helpers, ported from linalg.py (get_eigen_elmts_2x2,
// invert_2x2, etc.). Symmetric-matrix eigendecomposition follows the
// Wikipedia 2x2 algorithm referenced in the original code.
//
#include "types.h"

namespace hhsr {

// Roots of a*x^2 + b*x + c = 0 (assumed real). roots[0] has the largest module.
inline void real_polyroots_2(f32 a, f32 b, f32 c, f32 roots[2]) {
    f32 delta = std::max(b * b - 4.f * a * c, 0.f);
    f32 s = std::sqrt(delta);
    f32 r1 = (-b + s) / (2.f * a);
    f32 r2 = (-b - s) / (2.f * a);
    if (std::fabs(r1) >= std::fabs(r2)) { roots[0] = r1; roots[1] = r2; }
    else                                { roots[0] = r2; roots[1] = r1; }
}

// Eigenvalues of a symmetric 2x2 matrix M = [[m00,m01],[m10,m11]].
inline void eigen_val_2x2(f32 m00, f32 m01, f32 m10, f32 m11, f32 l[2]) {
    f32 b = -(m00 + m11);
    f32 c = m00 * m11 - m01 * m10;
    real_polyroots_2(1.f, b, c, l);
}

// Full eigendecomposition (values l[2] + orthonormal vectors e1[2], e2[2]).
inline void eigen_elmts_2x2(f32 m00, f32 m01, f32 m10, f32 m11,
                            f32 l[2], f32 e1[2], f32 e2[2]) {
    eigen_val_2x2(m00, m01, m10, m11, l);

    if (m01 == 0.f && m00 == m11) {
        e1[0] = 1.f; e1[1] = 0.f;
        e2[0] = 0.f; e2[1] = 1.f;
        return;
    }
    e1[0] = m00 + m01 - l[1];
    e1[1] = m10 + m11 - l[1];

    if (e1[0] == 0.f) {
        e1[1] = 1.f; e2[0] = 1.f; e2[1] = 0.f;
    } else if (e1[1] == 0.f) {
        e1[0] = 1.f; e2[0] = 0.f; e2[1] = 1.f;
    } else {
        f32 norm = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1]);
        e1[0] /= norm; e1[1] /= norm;
        f32 sign = std::copysign(1.f, e1[0]);
        e2[1] = std::fabs(e1[0]);
        e2[0] = -e1[1] * sign;
    }
}

// Invert a symmetric 2x2 [xx,xy,yy] into [xx,xy,yy]; falls back to identity if singular.
inline void invert_sym_2x2(f32 xx, f32 xy, f32 yy, f32& ixx, f32& ixy, f32& iyy) {
    f32 det = xx * yy - xy * xy;
    if (std::fabs(det) > 1e-10f) {
        f32 inv = 1.f / det;
        ixx =  inv * yy;
        ixy = -inv * xy;
        iyy =  inv * xx;
    } else {
        ixx = 1.f;
        ixy = 0.f;
        iyy = 1.f;
    }
}

} // namespace hhsr
