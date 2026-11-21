// 16-gradient structure tensor for the merge-kernel covariance (Config::kernel_tensor16).
//
// GCam's ComputeStructureTensorFromGreen: per Bayer quad, the green samples of the 3x3
// neighbouring quads (edge-clamped) give, at each of 2x2 positions, four gradients -- one
// at the blue site, one at the red site (both interpolated from the surrounding greens)
// and one along each green diagonal -- so 4 x 4 = 16 gradients, summed and divided by 16,
// then rotated by 45 degrees (the green gradients are taken along the diagonals).
//
// Differences from GCam's kernel (which this does NOT use): the greens are taken in the
// GAT domain (the noise-stabilised domain the standard estimator's thresholds D_th / D_tr
// are tuned in) instead of sqrt space, and the result is expressed in the standard
// estimator's units and axes -- (xx, xy, yy) with y pointing down and the same magnitude
// as its sum of four 2x2-cell gradients on a ramp -- so it drops straight into the
// standard compute_k / covariance construction. GCam's ConstructGaussianCovariance and
// merge kernel are not involved.
//
// One text for the CPU (core/kernels.cpp) and Metal (kernel_estimate_cov16 in
// HHSRKernels.metal).
#pragma once

#if defined(__METAL_VERSION__)
#define T16_DEVICE device
#define T16_THREAD thread
#define T16_INLINE inline
#define T16_SQRT(v) sqrt(v)
#else
#include <cmath>
#define T16_DEVICE
#define T16_THREAD
#define T16_INLINE static inline
#define T16_SQRT(v) std::sqrt(v)
#endif

// Generalized Anscombe transform, identical to the standard estimator's apply_gat.
T16_INLINE float t16_gat(float v, float alpha, float beta) {
    const float c = 0.375f * alpha * alpha + beta;
    const float t = alpha * v + c;
    return (2.f / alpha) * T16_SQRT(t > 0.f ? t : 0.f);
}

// CFA description, from the 2x2 pattern: R at (ry, rx), B at (by, bx).
//   blue_top = (by == 0): BGGR / GBRG -> green_0 = green on B's row, green_1 = on R's row.
//   diag     = (pattern[0][0] is green): GBRG / GRBG use the diagonal gradient formulas.
// raw: Bayer plane, row stride raw_w; gh x gw = quads (raw / 2). Output S = (xx, xy, yy).
T16_INLINE void t16_structure(T16_DEVICE const float* raw, int raw_w, int gh, int gw,
                              int ry, int rx, int by, int bx, int blue_top, int diag,
                              float alpha, float beta, int y, int x, T16_THREAD float* S) {
    float g0[9], g1[9];
    for (int dy = -1; dy <= 1; ++dy) {
        int qy = y + dy;
        qy = qy < 0 ? 0 : (qy > gh - 1 ? gh - 1 : qy);
        for (int dx = -1; dx <= 1; ++dx) {
            int qx = x + dx;
            qx = qx < 0 ? 0 : (qx > gw - 1 ? gw - 1 : qx);
            const float vGr = raw[(2 * qy + ry) * raw_w + 2 * qx + (1 - rx)];   // green on R's row
            const float vGb = raw[(2 * qy + by) * raw_w + 2 * qx + (1 - bx)];   // green on B's row
            const float Gr = t16_gat(vGr, alpha, beta);
            const float Gb = t16_gat(vGb, alpha, beta);
            const int idx = (dy + 1) * 3 + (dx + 1);
            g0[idx] = blue_top ? Gb : Gr;
            g1[idx] = blue_top ? Gr : Gb;
        }
    }
    float sxx = 0.f, syy = 0.f, sxy = 0.f;
    for (int sy = 0; sy < 2; ++sy) {
        for (int sx = 0; sx < 2; ++sx) {
            const float G00 = g0[sy * 3 + sx],       G01 = g0[sy * 3 + sx + 1];
            const float G10 = g1[sy * 3 + sx],       G11 = g1[sy * 3 + sx + 1];
            const float G20 = g0[(sy + 1) * 3 + sx], G21 = g0[(sy + 1) * 3 + sx + 1];
            const float G30 = g1[(sy + 1) * 3 + sx], G31 = g1[(sy + 1) * 3 + sx + 1];
            float b_dx, b_dy, r_dx, r_dy;
            if (diag) {
                b_dx = 0.5f * ((G11 - G01) + (G21 - G10)); b_dy = 0.5f * ((G01 - G10) + (G11 - G21));
                r_dx = 0.5f * ((G21 - G10) + (G30 - G20)); r_dy = 0.5f * ((G21 - G30) + (G10 - G20));
            } else {
                b_dx = 0.5f * ((G11 - G00) + (G20 - G10)); b_dy = 0.5f * ((G00 - G10) + (G11 - G20));
                r_dx = 0.5f * ((G21 - G11) + (G31 - G20)); r_dy = 0.5f * ((G21 - G31) + (G11 - G20));
            }
            const float g1_dx = 0.5f * (G21 - G00), g1_dy = 0.5f * (G01 - G20);
            const float g2_dx = 0.5f * (G31 - G10), g2_dy = 0.5f * (G11 - G30);
            sxx += b_dx * b_dx + r_dx * r_dx + g1_dx * g1_dx + g2_dx * g2_dx;
            syy += b_dy * b_dy + r_dy * r_dy + g1_dy * g1_dy + g2_dy * g2_dy;
            sxy += b_dx * b_dy + r_dx * r_dy + g1_dx * g1_dy + g2_dx * g2_dy;
        }
    }
    sxx /= 16.f; syy /= 16.f; sxy /= 16.f;
    // RotateStructureTensor (45 degrees).
    const float c0 = 0.5f * (sxx + syy), c1 = 0.5f * (syy - sxx);
    const float nx = c0 + sxy, ny = c0 - sxy, nz = c1;
    // Standard-estimator units and axes. Calibrated on linear ramps in every direction
    // and all four CFA layouts: the standard tensor (sum of four 2x2-cell gradients) is
    // exactly 2x this rotated average, and GCam's rotated frame has y pointing up, so the
    // cross term flips sign.
    S[0] = 2.f * nx;
    S[1] = -2.f * nz;
    S[2] = 2.f * ny;
}
