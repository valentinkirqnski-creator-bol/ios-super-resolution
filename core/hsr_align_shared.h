#ifndef HHSR_HSR_ALIGN_SHARED_H
#define HHSR_HSR_ALIGN_SHARED_H
//
// hsr_align (the Handheld-SR package aligner) per-pixel / per-tile arithmetic, in a form both compilers accept:
// core/hsr_align.cpp (CPU, parallel over rows) and core/HsrAlignKernels.metal (GPU, one thread per pixel / tile) run
// the SAME text. Semantics are the package's Halide kernels (see hsr_align.h): repeated edges, round half to even,
// strict-less-than first minimum, every per-tile sum accumulated serially with x fastest.
//
// Portability (MSL is C++14-based): no templates, no std::, address spaces through HSRA_DEVICE / HSRA_THREAD, plain
// float / int, fixed-size arrays. Planes are row-major float, flows are separate dx / dy planes (nx x ny).

#ifdef __METAL_VERSION__
#define HSRA_DEVICE device
#define HSRA_THREAD thread
#define HSRA_INLINE static inline
#define hsra_floor(x) floor(x)
#define hsra_fabs(x) fabs(x)
#define hsra_rint(x) rint(x)              // round half to even
#else
#include <math.h>
#define HSRA_DEVICE
#define HSRA_THREAD
#define HSRA_INLINE static inline
#define hsra_floor(x) floorf(x)
#define hsra_fabs(x) fabsf(x)
#define hsra_rint(x) nearbyintf(x)        // default rounding mode: half to even (Halide::round)
#endif

#define HSRA_FLT_EPSILON 1.1920928955078125e-07f
#define HSRA_BIG 3.0e38f                  // "infinity" for the first-minimum searches (finite: fast-math safe)

HSRA_INLINE int hsra_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
HSRA_INLINE float hsra_clampf1(float v) { return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v); }
HSRA_INLINE float hsra_minf(float a, float b) { return b < a ? b : a; }

// BoundaryConditions::repeat_edge
HSRA_INLINE float hsra_edge(const HSRA_DEVICE float* p, int w, int h, int x, int y) {
    return p[hsra_clampi(y, 0, h - 1) * w + hsra_clampi(x, 0, w - 1)];
}

// ---- grey (utils_image.compute_grey_images, DECIMATING) ----------------------------------------------------------
// Pass 1: 2x2 Bayer mean of the un-whitened sites. g00..g11 = per-site gains [sy][sx]; rw = raw width.
HSRA_INLINE float hsra_grey_quad(const HSRA_DEVICE float* raw, int rw, float g00, float g01, float g10, float g11,
                                 int x, int y) {
    const int i0 = (2 * y) * rw + 2 * x, i1 = i0 + rw;
    return (((raw[i0] * g00 + raw[i0 + 1] * g01) + raw[i1] * g10) + raw[i1 + 1] * g11) * 0.25f;
}
// Pass 2: [0.25 0.5 0.25] along y (nearest edges).
HSRA_INLINE float hsra_blur_v(const HSRA_DEVICE float* q, int w, int h, int x, int y) {
    return 0.25f * hsra_edge(q, w, h, x, y - 1) + 0.5f * q[y * w + x] + 0.25f * hsra_edge(q, w, h, x, y + 1);
}
// Pass 3: [0.25 0.5 0.25] along x, clipped to [0, 1].
HSRA_INLINE float hsra_blur_h_clip(const HSRA_DEVICE float* t, int w, int h, int x, int y) {
    const float v = 0.25f * hsra_edge(t, w, h, x - 1, y) + 0.5f * t[y * w + x] + 0.25f * hsra_edge(t, w, h, x + 1, y);
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}

// ---- downsample (one factor-2 round: [1 2 1]/4 then decimate, x then y; w2 = w/2 + w%2) ---------------------------
HSRA_INLINE float hsra_down_h(const HSRA_DEVICE float* in, int w, int h, int x, int y) {
    return (hsra_edge(in, w, h, 2 * x - 1, y) + 2.0f * hsra_edge(in, w, h, 2 * x, y) + hsra_edge(in, w, h, 2 * x + 1, y)) * 0.25f;
}
HSRA_INLINE float hsra_down_v(const HSRA_DEVICE float* hz, int w2, int h, int x, int y) {
    return (hsra_edge(hz, w2, h, x, 2 * y - 1) + 2.0f * hsra_edge(hz, w2, h, x, 2 * y) + hsra_edge(hz, w2, h, x, 2 * y + 1)) * 0.25f;
}

// ---- gradients (5-tap central difference / 12) and per-tile structure sums ------------------------------------------
HSRA_INLINE float hsra_grad_x(const HSRA_DEVICE float* in, int w, int h, int x, int y) {
    return (hsra_edge(in, w, h, x - 2, y) - 8.0f * hsra_edge(in, w, h, x - 1, y) + 8.0f * hsra_edge(in, w, h, x + 1, y) -
            hsra_edge(in, w, h, x + 2, y)) / 12.0f;
}
HSRA_INLINE float hsra_grad_y(const HSRA_DEVICE float* in, int w, int h, int x, int y) {
    return (hsra_edge(in, w, h, x, y - 2) - 8.0f * hsra_edge(in, w, h, x, y - 1) + 8.0f * hsra_edge(in, w, h, x, y + 1) -
            hsra_edge(in, w, h, x, y + 2)) / 12.0f;
}
// s[0] = sum gx^2, s[1] = sum gx*gy, s[2] = sum gy^2 over tile (tx, ty) of size ts.
HSRA_INLINE void hsra_tile_structure(const HSRA_DEVICE float* gx, const HSRA_DEVICE float* gy, int w, int ts, int tx, int ty,
                                     HSRA_THREAD float* s) {
    float a = 0.0f, b = 0.0f, c = 0.0f;
    for (int py = 0; py < ts; ++py)
        for (int px = 0; px < ts; ++px) {
            const int i = (ty * ts + py) * w + tx * ts + px;
            const float u = gx[i], v = gy[i];
            a += u * u; b += u * v; c += v * v;
        }
    s[0] = a; s[1] = b; s[2] = c;
}

// ---- block matching: tile (tx, ty) of size ts, search +-radius around the rounded init vector ----------------------
// init / output vectors indexed ty * cols + tx. l2: 0 = L1, 1 = L2.
HSRA_INLINE void hsra_block_match_tile(const HSRA_DEVICE float* ref, const HSRA_DEVICE float* mov, int w, int h,
                                       float init_dx, float init_dy, int ts, int radius, int l2, int tx, int ty,
                                       HSRA_THREAD float* out) {
    const int cx = (int)hsra_rint(init_dx), cy = (int)hsra_rint(init_dy);
    int bx = 0, by = 0;
    float bc = HSRA_BIG;
    for (int sy = -radius; sy <= radius; ++sy)
        for (int sx = -radius; sx <= radius; ++sx) {
            float cost = 0.0f;
            for (int py = 0; py < ts; ++py) {
                const int y = ty * ts + py;
                for (int px = 0; px < ts; ++px) {
                    const int x = tx * ts + px;
                    const float d = ref[y * w + x] - hsra_edge(mov, w, h, x + cx + sx, y + cy + sy);
                    cost += l2 ? d * d : hsra_fabs(d);
                }
            }
            if (cost < bc) { bx = sx; by = sy; }
            bc = hsra_minf(cost, bc);
        }
    out[0] = (float)(cx + bx);
    out[1] = (float)(cy + by);
}

// ---- one ICA (inverse compositional) iteration for tile (tx, ty) ----------------------------------------------------
// s3 = the tile's (sxx, sxy, syy). out = updated (dx, dy).
HSRA_INLINE void hsra_ica_tile(const HSRA_DEVICE float* ref, const HSRA_DEVICE float* gx, const HSRA_DEVICE float* gy,
                               const HSRA_DEVICE float* mov, int w, int h, int ts, int tx, int ty, float d0x, float d0y,
                               float sa, float sb, float sc, HSRA_THREAD float* out) {
    float r0 = 0.0f, r1 = 0.0f;
    for (int py = 0; py < ts; ++py) {
        const int y = ty * ts + py;
        for (int px = 0; px < ts; ++px) {
            const int x = tx * ts + px;
            const float wx = (float)x + d0x, wy = (float)y + d0y;
            const int ix = (int)hsra_floor(wx), iy = (int)hsra_floor(wy);
            const float fx = wx - (float)ix, fy = wy - (float)iy;
            const float a = hsra_edge(mov, w, h, ix, iy), b = hsra_edge(mov, w, h, ix + 1, iy);
            const float c = hsra_edge(mov, w, h, ix, iy + 1), d = hsra_edge(mov, w, h, ix + 1, iy + 1);
            const float top = a + fx * (b - a);
            const float bottom = c + fx * (d - c);
            const float warped = top + fy * (bottom - top);
            const float res = ref[y * w + x] - warped;
            r0 = r0 + gx[y * w + x] * res;
            r1 = r1 + gy[y * w + x] * res;
        }
    }
    const float det = sa * sc - sb * sb + HSRA_FLT_EPSILON;
    const float ux = (sc * r0 - sb * r1) / det, uy = (sa * r1 - sb * r0) / det;
    out[0] = d0x + hsra_clampf1(ux);
    out[1] = d0y + hsra_clampf1(uy);
}

// ---- kernel argument blocks shared with the Metal host (core/hsr_align_metal.mm) ------------------------------------
struct HsraGreyParams { int rw, w, h, pad; float g00, g01, g10, g11; };   // rw = raw width; w, h = grey size
struct HsraBmParams { int w, h, ts, radius, l2, cols, rows, pad; };       // block matching / ICA

// ---- flow transfer to a new grid: nearest + one horizontal + one vertical source candidate, L1 at the rounded
// shift over the target tile, strict-< first minimum ----------------------------------------------------------------
struct HsraTransferParams {
    int w, h;                 // target level plane
    int tt, stride;           // target tile / stride
    int snx, sny;             // source flow grid
    float spacing;            // grid_scale_ratio * stride
    float flow_scale, off_x, off_y;
};
HSRA_INLINE void hsra_transfer_tile(const HSRA_DEVICE float* sdx, const HSRA_DEVICE float* sdy,
                                    const HSRA_DEVICE float* ref, const HSRA_DEVICE float* mov, HsraTransferParams P,
                                    int tx, int ty, HSRA_THREAD float* out) {
    const float u = ((float)(tx * P.stride) - P.off_x) / P.spacing;
    const float v = ((float)(ty * P.stride) - P.off_y) / P.spacing;
    const int left = (int)hsra_floor(u), top = (int)hsra_floor(v);
    const int nx = (int)hsra_floor(u + 0.5f), ny = (int)hsra_floor(v + 0.5f);
    const int ox = nx == left ? left + 1 : left, oy = ny == top ? top + 1 : top;
    int cxs[3], cys[3];
    cxs[0] = nx; cxs[1] = ox; cxs[2] = nx;
    cys[0] = ny; cys[1] = ny; cys[2] = oy;
    float cdx[3], cdy[3];
    int best = 0;
    float bc = HSRA_BIG;
    for (int c = 0; c < 3; ++c) {
        cdx[c] = hsra_edge(sdx, P.snx, P.sny, cxs[c], cys[c]) * P.flow_scale;
        cdy[c] = hsra_edge(sdy, P.snx, P.sny, cxs[c], cys[c]) * P.flow_scale;
        const int sdxi = (int)hsra_rint(cdx[c]), sdyi = (int)hsra_rint(cdy[c]);
        float cost = 0.0f;
        for (int py = 0; py < P.tt; ++py) {
            const int y = ty * P.stride + py;
            for (int px = 0; px < P.tt; ++px) {
                const int x = tx * P.stride + px;
                cost += hsra_fabs(hsra_edge(ref, P.w, P.h, x, y) - hsra_edge(mov, P.w, P.h, x + sdxi, y + sdyi));
            }
        }
        if (cost < bc) best = c;
        bc = hsra_minf(cost, bc);
    }
    out[0] = cdx[best];
    out[1] = cdy[best];
}

// ---- 3x3 median of one flow component (repeated edges) -------------------------------------------------------------
HSRA_INLINE float hsra_median9(const HSRA_DEVICE float* f, int nx, int ny, int x, int y) {
    float v[9];
    int k = 0;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) v[k++] = hsra_edge(f, nx, ny, x + i, y + j);
    for (int a = 1; a < 9; ++a) {   // insertion sort: the 5th value is the median whatever the order of equal keys
        const float t = v[a];
        int b = a - 1;
        while (b >= 0 && v[b] > t) { v[b + 1] = v[b]; --b; }
        v[b + 1] = t;
    }
    return v[4];
}

#endif
