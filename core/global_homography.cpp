#include "global_homography.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace hhsr {
namespace {

// Bilinear sample of a single-channel image, clamped to edge. Returns the value
// and whether the (unclamped) sample was inside the image.
inline f32 sample_clamp(const Image& im, f32 x, f32 y) {
    if (im.w <= 0 || im.h <= 0) return 0.f;
    f32 cx = std::min(std::max(x, 0.f), (f32)(im.w - 1));
    f32 cy = std::min(std::max(y, 0.f), (f32)(im.h - 1));
    int x0 = (int)std::floor(cx), y0 = (int)std::floor(cy);
    int x1 = std::min(x0 + 1, im.w - 1), y1 = std::min(y0 + 1, im.h - 1);
    f32 ax = cx - (f32)x0, ay = cy - (f32)y0;
    f32 top = im.at(y0, x0) + (im.at(y0, x1) - im.at(y0, x0)) * ax;
    f32 bot = im.at(y1, x0) + (im.at(y1, x1) - im.at(y1, x0)) * ax;
    return top + (bot - top) * ay;
}

// 2x box-average downsample (single channel).
Image downsample2x(const Image& in) {
    int oh = std::max(1, in.h / 2), ow = std::max(1, in.w / 2);
    Image out(oh, ow, 1);
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            int y0 = 2 * y, x0 = 2 * x;
            int y1 = std::min(y0 + 1, in.h - 1), x1 = std::min(x0 + 1, in.w - 1);
            out.at(y, x) = 0.25f * (in.at(y0, x0) + in.at(y0, x1) +
                                    in.at(y1, x0) + in.at(y1, x1));
        }
    }
    return out;
}

// Downsample (by halving) until max(h,w) <= max_dim. Returns the image and the
// linear scale factor small/full applied.
Image downsample_to(const Image& in, int max_dim, f32& scale_out) {
    Image cur = in;
    f32 scale = 1.f;
    while (std::max(cur.h, cur.w) > max_dim && cur.h > 8 && cur.w > 8) {
        cur = downsample2x(cur);
        scale *= 0.5f;
    }
    scale_out = scale;
    return cur;
}

inline void mat3_mul(const f32 A[9], const f32 B[9], f32 C[9]) {
    f32 t[9];
    for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col)
            t[r * 3 + col] = A[r * 3 + 0] * B[0 * 3 + col] +
                             A[r * 3 + 1] * B[1 * 3 + col] +
                             A[r * 3 + 2] * B[2 * 3 + col];
    for (int i = 0; i < 9; ++i) C[i] = t[i];
}

// Mean absolute error of comp warped by H against ref, over pixels whose warped
// position lands inside comp. Returns a large value if too few land inside.
f32 warp_error(const Image& ref, const Image& comp, const f32 H[9]) {
    double acc = 0.0; long cnt = 0;
    for (int y = 0; y < ref.h; y += 2) {
        for (int x = 0; x < ref.w; x += 2) {
            f32 ox, oy;
            apply_homography(H, (f32)x, (f32)y, ox, oy);
            if (ox < 0.f || ox > (f32)(comp.w - 1) || oy < 0.f || oy > (f32)(comp.h - 1))
                continue;
            acc += std::fabs((double)(sample_clamp(comp, ox, oy) - ref.at(y, x)));
            ++cnt;
        }
    }
    if (cnt < (long)(ref.h * ref.w) / 32) return 1e30f;  // too little overlap
    return (f32)(acc / (double)cnt);
}

// Coarse rotation (about the image centre) + integer shift search, maximising
// negative SSD (minimising error). Builds an affine homography seed. Handles
// large roll, which Lucas-Kanade alone cannot converge to from identity.
void coarse_seed(const Image& ref, const Image& comp, f32 H[9]) {
    const f32 cx = 0.5f * (f32)(ref.w - 1), cy = 0.5f * (f32)(ref.h - 1);
    const f32 range_deg = 12.f, step_deg = 1.f;
    const int shift_max = 6, shift_step = 2;
    f32 best_err = 1e30f;
    f32 best[9] = {1,0,0, 0,1,0, 0,0,1};
    for (f32 deg = -range_deg; deg <= range_deg + 1e-3f; deg += step_deg) {
        const f32 th = deg * 3.14159265358979323846f / 180.f;
        const f32 ct = std::cos(th), st = std::sin(th);
        // rotation about centre: p -> R(p-c)+c
        const f32 base_tx = cx - (ct * cx - st * cy);
        const f32 base_ty = cy - (st * cx + ct * cy);
        for (int sy = -shift_max; sy <= shift_max; sy += shift_step) {
            for (int sx = -shift_max; sx <= shift_max; sx += shift_step) {
                f32 Hc[9] = { ct, -st, base_tx + (f32)sx,
                              st,  ct, base_ty + (f32)sy,
                              0.f, 0.f, 1.f };
                f32 e = warp_error(ref, comp, Hc);
                if (e < best_err) { best_err = e; for (int i = 0; i < 9; ++i) best[i] = Hc[i]; }
            }
        }
    }
    for (int i = 0; i < 9; ++i) H[i] = best[i];
}

// Solve an 8x8 linear system A x = b in place (Gaussian elimination, partial
// pivot). Returns false if singular.
bool solve8(f32 A[8][8], f32 b[8], f32 x[8]) {
    const int n = 8;
    for (int col = 0; col < n; ++col) {
        int piv = col; f32 best = std::fabs(A[col][col]);
        for (int r = col + 1; r < n; ++r) {
            f32 v = std::fabs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-12f) return false;
        if (piv != col) {
            for (int k = 0; k < n; ++k) std::swap(A[col][k], A[piv][k]);
            std::swap(b[col], b[piv]);
        }
        const f32 inv = 1.f / A[col][col];
        for (int r = col + 1; r < n; ++r) {
            f32 f = A[r][col] * inv;
            if (f == 0.f) continue;
            for (int k = col; k < n; ++k) A[r][k] -= f * A[col][k];
            b[r] -= f * b[col];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        f32 s = b[r];
        for (int k = r + 1; k < n; ++k) s -= A[r][k] * x[k];
        x[r] = s / A[r][r];
    }
    return true;
}

// Forward-additive Lucas-Kanade homography refinement: minimise
// sum [comp(H*p) - ref(p)]^2 over the 8 homography parameters (h8 fixed = 1).
// Levenberg damping; reverts an iteration that increases the error.
void lk_refine(const Image& ref, const Image& comp, f32 H[9], int iters) {
    f32 cur_err = warp_error(ref, comp, H);
    for (int it = 0; it < iters; ++it) {
        f32 A[8][8]; f32 b[8];
        for (int i = 0; i < 8; ++i) { b[i] = 0.f; for (int j = 0; j < 8; ++j) A[i][j] = 0.f; }
        for (int y = 1; y < ref.h - 1; ++y) {
            for (int x = 1; x < ref.w - 1; ++x) {
                const f32 fx = (f32)x, fy = (f32)y;
                const f32 D = H[6] * fx + H[7] * fy + H[8];
                if (std::fabs(D) < 1e-6f) continue;
                const f32 iD = 1.f / D;
                const f32 u = (H[0] * fx + H[1] * fy + H[2]) * iD;
                const f32 v = (H[3] * fx + H[4] * fy + H[5]) * iD;
                if (u < 1.f || u > (f32)(comp.w - 2) || v < 1.f || v > (f32)(comp.h - 2))
                    continue;
                const f32 err = sample_clamp(comp, u, v) - ref.at(y, x);
                const f32 gx = 0.5f * (sample_clamp(comp, u + 1.f, v) - sample_clamp(comp, u - 1.f, v));
                const f32 gy = 0.5f * (sample_clamp(comp, u, v + 1.f) - sample_clamp(comp, u, v - 1.f));
                // d(u,v)/d(h0..h7)
                f32 J[8];
                J[0] = gx * (fx * iD);           J[1] = gx * (fy * iD);           J[2] = gx * iD;
                J[3] = gy * (fx * iD);           J[4] = gy * (fy * iD);           J[5] = gy * iD;
                J[6] = -(gx * u + gy * v) * (fx * iD);
                J[7] = -(gx * u + gy * v) * (fy * iD);
                for (int i = 0; i < 8; ++i) {
                    b[i] -= J[i] * err;
                    for (int j = 0; j < 8; ++j) A[i][j] += J[i] * J[j];
                }
            }
        }
        // Levenberg damping on the diagonal.
        f32 tr = 0.f; for (int i = 0; i < 8; ++i) tr += A[i][i];
        const f32 lam = 1e-3f * (tr / 8.f + 1e-6f);
        for (int i = 0; i < 8; ++i) A[i][i] += lam;
        f32 d[8];
        if (!solve8(A, b, d)) break;
        f32 Hn[9];
        for (int i = 0; i < 8; ++i) Hn[i] = H[i] + d[i];
        Hn[8] = 1.f;
        f32 e = warp_error(ref, comp, Hn);
        if (!(e < cur_err)) break;     // no improvement (or NaN) -> stop
        for (int i = 0; i < 9; ++i) H[i] = Hn[i];
        cur_err = e;
    }
}

} // namespace

Image warp_grey_by_homography(const Image& comp_grey, const f32 H[9]) {
    Image out(comp_grey.h, comp_grey.w, 1);
    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            f32 ox, oy;
            apply_homography(H, (f32)x, (f32)y, ox, oy);
            if (ox < 0.f || ox > (f32)(comp_grey.w - 1) ||
                oy < 0.f || oy > (f32)(comp_grey.h - 1)) {
                out.at(y, x) = 0.f;
            } else {
                out.at(y, x) = sample_clamp(comp_grey, ox, oy);
            }
        }
    }
    return out;
}

void estimate_global_homography(const Image& ref_grey, const Image& comp_grey,
                                const Config& cfg, f32 H_out[9]) {
    (void)cfg;
    const f32 identity[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i = 0; i < 9; ++i) H_out[i] = identity[i];
    if (ref_grey.h <= 0 || ref_grey.w <= 0 ||
        ref_grey.h != comp_grey.h || ref_grey.w != comp_grey.w) return;
    // Require host pixels: on the device the grey can be GPU-resident with an
    // empty host buffer (header only). Reading it would segfault; the caller
    // forces host materialisation, but guard here regardless.
    if (ref_grey.data.size() != (size_t)ref_grey.h * ref_grey.w * ref_grey.c ||
        comp_grey.data.size() != (size_t)comp_grey.h * comp_grey.w * comp_grey.c) return;

    // Estimate at a reduced resolution (warp-then-refine tolerates an
    // imperfect H -- the per-tile align cleans up the residual), then scale the
    // result back to the full grey resolution.
    f32 lk_scale = 1.f, search_scale = 1.f;
    Image lk_ref = downsample_to(ref_grey, 384, lk_scale);
    Image lk_comp = downsample_to(comp_grey, 384, lk_scale);
    Image s_ref = downsample_to(lk_ref, 128, search_scale);   // relative to lk_*
    Image s_comp = downsample_to(lk_comp, 128, search_scale);

    // Coarse rotation+shift seed at the smallest level, in lk-level pixels.
    f32 H[9];
    coarse_seed(s_ref, s_comp, H);
    // Lift the seed from search-level pixels to lk-level pixels: D maps lk->search.
    {
        const f32 f = search_scale;
        const f32 D[9]   = { f, 0, 0,  0, f, 0,  0, 0, 1 };
        const f32 Dinv[9]= { 1.f/f, 0, 0,  0, 1.f/f, 0,  0, 0, 1 };
        f32 tmp[9]; mat3_mul(H, D, tmp); mat3_mul(Dinv, tmp, H);
    }

    lk_refine(lk_ref, lk_comp, H, /*iters=*/40);

    // Reject if the result is not meaningfully better than identity, or is
    // degenerate -- warp-then-refine must never make alignment worse.
    const f32 e_id = warp_error(lk_ref, lk_comp, identity);
    const f32 e_h  = warp_error(lk_ref, lk_comp, H);
    const f32 det2 = H[0] * H[4] - H[1] * H[3];
    const bool degenerate = !(std::fabs(det2) > 0.25f && std::fabs(det2) < 4.f) ||
                            !std::isfinite(e_h);
    if (degenerate || !(e_h < 0.98f * e_id)) return;  // keep identity

    // Scale H from lk-level pixels to full grey pixels: D maps full->lk.
    const f32 f = lk_scale;
    const f32 D[9]    = { f, 0, 0,  0, f, 0,  0, 0, 1 };
    const f32 Dinv[9] = { 1.f/f, 0, 0,  0, 1.f/f, 0,  0, 0, 1 };
    f32 tmp[9]; mat3_mul(H, D, tmp); mat3_mul(Dinv, tmp, H_out);
}

} // namespace hhsr
