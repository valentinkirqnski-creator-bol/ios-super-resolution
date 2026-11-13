#ifndef HHSR_SABRE61_SHARED_H
#define HHSR_SABRE61_SHARED_H
//
// GCam 6.1.021 SABRE, per-pixel arithmetic, in a form both compilers accept.
//
// Source of truth: Documents/super/SABRE_6.1_COMPLETE_REFERENCE.md (verbatim GLSL + decompiled native).
// Every function below is a line-for-line transcription of one GLSL function or entry point; the GL
// samplers are emulated explicitly so the CPU (core/sabre61.cpp) and the GPU (core/Sabre61Kernels.metal)
// run the SAME text:
//
//   LINEAR   = 4-texel bilinear at (uv*size - 0.5), CLAMP_TO_EDGE      guides, kernel covariance,
//                                                                      accumulated weights, noise LUT,
//                                                                      merged planes (denoiser)
//   NEAREST  = texel floor(uv*size), clamped                          flow (both passes), rejection mask
//   gather   = the 2x2 quad at floor(texel - 0.5), each texel clamped Bayer, dilate
//
// Sampler modes are the device-measured ones (glGetTexParameteriv at draw time on a live 2x shot +
// Ghidra Texture::Create): flow NEAREST, rejection NEAREST, kernel covariance LINEAR, guides LINEAR.
//
// Texture storage is emulated at each boundary: RGBA16F textures (guide, kernel covariance, merged
// planes) are written through s61_q16 (fp16 round-to-nearest-even); R8 textures (rejection, dilated
// mask, accumulated frame weights) are stored as 8-bit and read back as k/255.
//
// Data layout: our raw planes are ALREADY in SABRE's stage-0 normalized space S2 (the loader delivers
// (raw - black)/(white - black) * wb[c]/wb[G], exposure-matched), so `gains` = 1 and
// `black_levels_times_gains` = 0 here: rggb*gains + black == rggb.
//
// Portability (MSL is C++14-based): no templates, no std::, no references in signatures, address
// spaces through S61_DEVICE / S61_THREAD, plain float/int, POD structs, fixed-size arrays.

#ifdef __METAL_VERSION__
#define S61_DEVICE device
#define S61_CONST constant
#define S61_THREAD thread
#define S61_INLINE static inline
#define s61_floor(x) floor(x)
#define s61_sqrt(x) sqrt(x)
#define s61_exp2(x) exp2(x)
#define s61_fabs(x) fabs(x)
#define s61_log(x) log(x)
static inline float s61_q16(float x) { return float(half(x)); }
#else
#include <math.h>
#include <stdint.h>
#include <string.h>
#define S61_DEVICE
#define S61_CONST
#define S61_THREAD
#define S61_INLINE static inline
#define s61_floor(x) floorf(x)
#define s61_sqrt(x) sqrtf(x)
#define s61_exp2(x) exp2f(x)
#define s61_fabs(x) fabsf(x)
#define s61_log(x) logf(x)
// fp32 -> fp16 -> fp32, round-to-nearest-even, exactly what storing into an RGBA16F texture does
// (portable: no _Float16 / __fp16 dependence).
S61_INLINE uint16_t s61_f2h_bits(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return (uint16_t)(sign | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u));
    if (a >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);   // rounds to >= 65520 -> inf
    uint32_t h;
    if (a < 0x38800000u) {                                // fp16 subnormal range (< 2^-14)
        const int e = (int)(a >> 23);
        if (e < 102) { h = 0; }
        else {
            const uint32_t m = (a & 0x7fffffu) | 0x800000u;
            const int shift = 126 - e;
            const uint32_t q = m >> shift, rem = m & ((1u << shift) - 1u), hf = 1u << (shift - 1);
            h = q + ((rem > hf || (rem == hf && (q & 1u))) ? 1u : 0u);
        }
    } else {
        const uint32_t m = a & 0x1fffu;
        h = ((a - 0x38000000u) >> 13);
        if (m > 0x1000u || (m == 0x1000u && (h & 1u))) h += 1u;
    }
    return (uint16_t)(sign | h);
}
S61_INLINE float s61_h2f(uint16_t hb) {
    const uint32_t he = (hb >> 10) & 0x1fu, hm = hb & 0x3ffu, sign = (uint32_t)(hb & 0x8000u) << 16;
    uint32_t r;
    if (he == 0) {
        if (hm == 0) r = sign;
        else { float v = (float)hm * (1.0f / 16777216.0f); memcpy(&r, &v, 4); r |= sign; }
    } else if (he == 31) r = sign | 0x7f800000u | (hm << 13);
    else r = sign | ((he + 112u) << 23) | (hm << 13);
    float o; memcpy(&o, &r, 4); return o;
}
S61_INLINE float s61_q16_ref(float f) { return s61_h2f(s61_f2h_bits(f)); }
S61_INLINE float s61_q16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    const uint32_t sign = u & 0x80000000u;
    uint32_t a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return f;                       // inf / nan unchanged
    if (a >= 0x477ff000u) {                               // rounds to >= 65520 -> inf
        uint32_t r = sign | 0x7f800000u; float o; memcpy(&o, &r, 4); return o;
    }
    uint32_t h;
    if (a < 0x38800000u) {                                // fp16 subnormal range (< 2^-14)
        const int e = (int)(a >> 23);
        if (e < 102) { h = 0; }                           // < 2^-25: rounds to zero
        else {
            const uint32_t m = (a & 0x7fffffu) | 0x800000u;
            const int shift = 126 - e;                    // 14..24
            const uint32_t q = m >> shift, rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
            h = q + ((rem > half || (rem == half && (q & 1u))) ? 1u : 0u);
        }
    } else {
        const uint32_t m = a & 0x1fffu;                   // 13 dropped mantissa bits
        h = ((a - 0x38000000u) >> 13);
        if (m > 0x1000u || (m == 0x1000u && (h & 1u))) h += 1u;
    }
    // back to fp32
    uint32_t r;
    const uint32_t he = (h >> 10) & 0x1fu, hm = h & 0x3ffu;
    if (he == 0) {
        if (hm == 0) r = sign;
        else { float v = (float)hm * (1.0f / 16777216.0f); memcpy(&r, &v, 4); r |= sign; }
    } else if (he == 31) r = sign | 0x7f800000u | (hm << 13);
    else r = sign | ((he + 112u) << 23) | (hm << 13);
    float o; memcpy(&o, &r, 4); return o;
}
#endif

S61_INLINE float s61_min(float a, float b) { return a < b ? a : b; }
S61_INLINE float s61_max(float a, float b) { return a > b ? a : b; }
S61_INLINE float s61_clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
S61_INLINE int s61_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
S61_INLINE float s61_mix(float a, float b, float t) { return a + (b - a) * t; }
S61_INLINE float s61_sign(float v) { return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); }
S61_INLINE float s61_smoothstep(float e0, float e1, float x) {
    const float t = s61_clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

struct S61F4 { float x, y, z, w; };
struct S61F3 { float x, y, z; };

S61_INLINE S61F4 s61_f4(float x, float y, float z, float w) { S61F4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }
S61_INLINE S61F3 s61_f3(float x, float y, float z) { S61F3 r; r.x = x; r.y = y; r.z = z; return r; }

// ------------------------------------------------------------------------------------------------ constants
// 6.1 tuning (ELF .rodata, reference PART 1). Covariance "sigmas" are PRECISIONS (reciprocals).
#define S61_GREEN_DIFF_IMPACT 10.0f       // guide_color_parameters.x
#define S61_BASE_SIGMA 4.0f               // covariance_parameters1.w
#define S61_SHRINKED_SIGMA 8.0f           // covariance_parameters1.x
#define S61_STRETCHED_SIGMA 1.0f          // covariance_parameters1.y
#define S61_GRADIENT_CLIP 0.001f          // covariance_parameters1.z
#define S61_BLURRED_SIGMA 1.0f            // covariance_parameters2.x
#define S61_GRADIENT_TRANSITION 142.857f  // covariance_parameters2.y = 1/0.007
#define S61_DISTANCE_MULT 1.0f            // rejection_params.x
#define S61_REJECTION_THRESHOLD 0.0f      // rejection_params.y
#define S61_BOOST 6.0f                    // extra_motion_robustness_boost
#define S61_BOOST_STD_THRESHOLD 5.0f      // motion_robustness_boost_std_dev_threshold
#define S61_MOTION_THR_PX 0.2016f         // extra_motion_robustness_motion_threshold = (2016/W_guide)*1e-4  [uv]
#define S61_UNBLOCK_THR_PX 0.20158f       // unblocker_reduction_threshold = (2016/W_guide)*9.999e-5  [uv]
#define S61_ACC_WEIGHT_SCALE 25.0f        // accumulated_weight_scale
#define S61_KB_THRESHOLD 4.0f             // kernel_blurring_params.x
#define S61_KB_MULT 0.3f                  // kernel_blurring_params.y
#define S61_KEPSILON 0.00005f             // merge kEpsilon
#define S61_DENOISING_FACTOR 0.0625f
#define S61_OUTLIER_SIGMA_MULT 0.125f     // 1 / 8
#define S61_REJECT_LUMA_REVERT 0.4f
#define S61_GLOBAL_LUMA_REVERT 1.0f
#define S61_WEIGHT_TEX_BIAS 1.0f

// ------------------------------------------------------------------------------------------------ parameters
// CFA: positions (row, col) inside each 2x2 quad of the R site, the two green sites (g1 = top row) and B.
struct S61Cfa { int ry, rx, g1y, g1x, g2y, g2x, by, bx; };

struct S61GuideParams {
    int Wr, Hr;             // raw (Bayer) size
    int row0, nrows;        // raw rows held in the plane passed in (slice)
    S61Cfa cfa;
    float nsb[4];           // noise_texture_scale_bias (0.9, 0.5, 0.05, 0.25) for a 10x2 LUT
};

struct S61RejParams {
    int Wg, Hg;             // guide size (rejection runs at guide resolution)
    int fnx, fny;           // flow texture size (tiles)
    float fso[4];           // flow_scale_offset
    float mot_thr, ub_thr;  // in flow.z (uv) units
    float nsb[4];
};

struct S61MergeParams {
    int Wo, Ho;             // output size
    int Wr, Hr;             // raw size (merge tex_size)
    int Wg, Hg;             // guide size (cov / mask / accumulated weights)
    int fnx, fny;
    int raw_row0, raw_nrows;   // raw slice
    int cov_row0, cov_nrows;   // cov slice (guide rows)
    int out_row0;           // global output row of local row 0 of the accumulators
    int is_base;
    int use_rejection;      // frame_info.w != 0
    float fso[4];
    float fbp[4];           // frame_border_padded
    S61Cfa cfa;
};

struct S61DenoiseParams {
    int Wo, Ho, Wg, Hg;
    int level;              // dilation.z
    int in_row0, in_nrows;  // rows held by the input plane (band-local)
    int out_row0;           // global row of local row 0 of the output plane
    float yuv_noise[4];     // yuv_noise_model
    float fb[4];            // frame_border
};

// ------------------------------------------------------------------------------------------------ samplers
// 10x2 RGBA32F noise LUT, row-major [row][col][4], LINEAR + CLAMP_TO_EDGE.
S61_INLINE S61F4 s61_lut_linear(const S61_DEVICE float* lut, float u, float v) {
    const float x = u * 10.0f - 0.5f, y = v * 2.0f - 0.5f;
    const float fx = s61_floor(x), fy = s61_floor(y);
    const float ax = x - fx, ay = y - fy;
    const int x0 = s61_clampi((int)fx, 0, 9), x1 = s61_clampi((int)fx + 1, 0, 9);
    const int y0 = s61_clampi((int)fy, 0, 1), y1 = s61_clampi((int)fy + 1, 0, 1);
    float o[4];
    for (int c = 0; c < 4; ++c) {
        const float a = lut[(y0 * 10 + x0) * 4 + c] * (1.0f - ax) + lut[(y0 * 10 + x1) * 4 + c] * ax;
        const float b = lut[(y1 * 10 + x0) * 4 + c] * (1.0f - ax) + lut[(y1 * 10 + x1) * 4 + c] * ax;
        o[c] = a * (1.0f - ay) + b * ay;
    }
    return s61_f4(o[0], o[1], o[2], o[3]);
}

// LINEAR fetch of an interleaved float plane (C channels, rows [row0, row0+nrows) held), clamp to the
// GLOBAL image (W x H) first, then to the held rows.
S61_INLINE void s61_linear_plane(const S61_DEVICE float* p, int W, int H, int C, int row0, int nrows,
                                 float u, float v, S61_THREAD float* out) {
    const float x = u * (float)W - 0.5f, y = v * (float)H - 0.5f;
    const float fx = s61_floor(x), fy = s61_floor(y);
    const float ax = x - fx, ay = y - fy;
    const int x0 = s61_clampi((int)fx, 0, W - 1), x1 = s61_clampi((int)fx + 1, 0, W - 1);
    const int y0 = s61_clampi(s61_clampi((int)fy, 0, H - 1) - row0, 0, nrows - 1);
    const int y1 = s61_clampi(s61_clampi((int)fy + 1, 0, H - 1) - row0, 0, nrows - 1);
    for (int c = 0; c < C; ++c) {
        const float a = p[((long)y0 * W + x0) * C + c] * (1.0f - ax) + p[((long)y0 * W + x1) * C + c] * ax;
        const float b = p[((long)y1 * W + x0) * C + c] * (1.0f - ax) + p[((long)y1 * W + x1) * C + c] * ax;
        out[c] = a * (1.0f - ay) + b * ay;
    }
}
// LINEAR fetch of an 8-bit unorm plane (whole image held).
S61_INLINE float s61_linear_u8(const S61_DEVICE uint8_t* p, int W, int H, float u, float v) {
    const float x = u * (float)W - 0.5f, y = v * (float)H - 0.5f;
    const float fx = s61_floor(x), fy = s61_floor(y);
    const float ax = x - fx, ay = y - fy;
    const int x0 = s61_clampi((int)fx, 0, W - 1), x1 = s61_clampi((int)fx + 1, 0, W - 1);
    const int y0 = s61_clampi((int)fy, 0, H - 1), y1 = s61_clampi((int)fy + 1, 0, H - 1);
    const float k = 1.0f / 255.0f;
    const float a = (float)p[y0 * W + x0] * k * (1.0f - ax) + (float)p[y0 * W + x1] * k * ax;
    const float b = (float)p[y1 * W + x0] * k * (1.0f - ax) + (float)p[y1 * W + x1] * k * ax;
    return a * (1.0f - ay) + b * ay;
}
S61_INLINE float s61_nearest_u8(const S61_DEVICE uint8_t* p, int W, int H, float u, float v) {
    const int x = s61_clampi((int)s61_floor(u * (float)W), 0, W - 1);
    const int y = s61_clampi((int)s61_floor(v * (float)H), 0, H - 1);
    return (float)p[y * W + x] * (1.0f / 255.0f);
}
// NEAREST flow texture (fnx x fny, 4 floats: uv offset xy, local variation z).
S61_INLINE S61F4 s61_flow_nearest(const S61_DEVICE float* flow, int nx, int ny, float u, float v) {
    const int x = s61_clampi((int)s61_floor(u * (float)nx), 0, nx - 1);
    const int y = s61_clampi((int)s61_floor(v * (float)ny), 0, ny - 1);
    const S61_DEVICE float* f = flow + ((long)y * nx + x) * 4;
    return s61_f4(f[0], f[1], f[2], f[3]);
}

// kFunctionSampleBicubic (6.1: NO abs) over a 4-channel guide plane, with LINEAR taps.
S61_INLINE S61F4 s61_sample_bicubic(const S61_DEVICE float* tex, int W, int H, float u, float v) {
    float uvx = u * (float)W + 0.5f, uvy = v * (float)H + 0.5f;
    const float ix = s61_floor(uvx), iy = s61_floor(uvy);
    const float ax = uvx - ix, ay = uvy - iy;
    const float w0x = (1.0f / 6.0f) * (ax * (ax * (-ax + 3.0f) - 3.0f) + 1.0f);
    const float w1x = (1.0f / 6.0f) * (ax * ax * (3.0f * ax - 6.0f) + 4.0f);
    const float w2x = (1.0f / 6.0f) * (ax * (ax * (-3.0f * ax + 3.0f) + 3.0f) + 1.0f);
    const float w3x = (1.0f / 6.0f) * (ax * ax * ax);
    const float w0y = (1.0f / 6.0f) * (ay * (ay * (-ay + 3.0f) - 3.0f) + 1.0f);
    const float w1y = (1.0f / 6.0f) * (ay * ay * (3.0f * ay - 6.0f) + 4.0f);
    const float w2y = (1.0f / 6.0f) * (ay * (ay * (-3.0f * ay + 3.0f) + 3.0f) + 1.0f);
    const float w3y = (1.0f / 6.0f) * (ay * ay * ay);
    const float g0x = w0x + w1x, g1x = w2x + w3x, g0y = w0y + w1y, g1y = w2y + w3y;
    const float h0x = -1.0f + w1x / g0x, h1x = 1.0f + w3x / g1x;
    const float h0y = -1.0f + w1y / g0y, h1y = 1.0f + w3y / g1y;
    const float p00x = (ix + h0x - 0.5f) / (float)W, p00y = (iy + h0y - 0.5f) / (float)H;
    const float p11x = (ix + h1x - 0.5f) / (float)W, p11y = (iy + h1y - 0.5f) / (float)H;
    float t0[4], t1[4], t2[4], t3[4];
    s61_linear_plane(tex, W, H, 4, 0, H, p00x, p00y, t0);
    s61_linear_plane(tex, W, H, 4, 0, H, p11x, p00y, t1);
    s61_linear_plane(tex, W, H, 4, 0, H, p00x, p11y, t2);
    s61_linear_plane(tex, W, H, 4, 0, H, p11x, p11y, t3);
    float o[4];
    for (int c = 0; c < 4; ++c) o[c] = g0y * (g0x * t0[c] + g1x * t1[c]) + g1y * (g0x * t2[c] + g1x * t3[c]);
    return s61_f4(o[0], o[1], o[2], o[3]);
}

// ------------------------------------------------------------------------------------------------ STAGE 1
// Raw fetch with CLAMP_TO_EDGE in the global image, then into the held slice.
S61_INLINE float s61_raw(const S61_DEVICE float* raw, int Wr, int Hr, int row0, int nrows, int x, int y) {
    const int xx = s61_clampi(x, 0, Wr - 1);
    const int yy = s61_clampi(s61_clampi(y, 0, Hr - 1) - row0, 0, nrows - 1);
    return raw[(long)yy * Wr + xx];
}

S61_INLINE S61F3 s61_rgb_to_ycocg(S61F3 rgb) {   // greens already summed
    return s61_f3(rgb.x * 0.25f + rgb.y * 0.25f + rgb.z * 0.25f,
                  rgb.x * -0.25f + rgb.y * 0.25f + rgb.z * -0.25f,
                  rgb.x * 0.5f + rgb.z * -0.5f);
}
S61_INLINE float s61_weight_green(int x) { return x == 0 ? 0.5f : ((x == 1 || x == -1) ? 0.25f : 0.0f); }

// Guide + kernel shader main() for guide pixel (gx, gy). Outputs:
//   ref4 = o_FragColor = (reference_color.xyz [S1 YCoCg], std_dev)
//   cov3 = o_gauss_cov = (cxx, cyy, cxy) precision
//   dbg  = optional statistics for the noise-LUT simulation (may be 0): [green_differences (raw),
//          rotated cov xx, rotated cov yy, eigenvalue1 (raw), |G1-G2| of the centre quad]
S61_INLINE void s61_guide_pixel(const S61_DEVICE float* raw, S61GuideParams P, const S61_DEVICE float* lut,
                                int gx, int gy, S61_THREAD float* ref4, S61_THREAD float* cov3,
                                S61_THREAD float* dbg) {
    float m1r = 0.0f, m1g = 0.0f, m1b = 0.0f, m0r = 0.0f, m0g = 0.0f, m0b = 0.0f;
    float avr = 0.0f, avg = 0.0f, avb = 0.0f;
    float sums[9];
    float green_differences = 0.0f;
    float centre_gdiff = 0.0f;
    for (int y = -1; y <= 1; y += 1) {
        for (int x = -1; x <= 1; x += 1) {
            const int qx = 2 * (gx + x), qy = 2 * (gy + y);
            float r = s61_raw(raw, P.Wr, P.Hr, P.row0, P.nrows, qx + P.cfa.rx, qy + P.cfa.ry);
            float g1 = s61_raw(raw, P.Wr, P.Hr, P.row0, P.nrows, qx + P.cfa.g1x, qy + P.cfa.g1y);
            float g2 = s61_raw(raw, P.Wr, P.Hr, P.row0, P.nrows, qx + P.cfa.g2x, qy + P.cfa.g2y);
            float b = s61_raw(raw, P.Wr, P.Hr, P.row0, P.nrows, qx + P.cfa.bx, qy + P.cfa.by);
            // rggb*gains + black_levels_times_gains: identity on our normalized planes.
            r = s61_sqrt(s61_max(0.0f, r)); g1 = s61_sqrt(s61_max(0.0f, g1));
            g2 = s61_sqrt(s61_max(0.0f, g2)); b = s61_sqrt(s61_max(0.0f, b));
            const S61F3 rgb = s61_f3(r, g1 + g2, b);
            sums[(y + 1) * 3 + x + 1] = s61_rgb_to_ycocg(rgb).x;
            const float w = s61_weight_green(x) * s61_weight_green(y);
            avr += rgb.x * w; avg += rgb.y * w; avb += rgb.z * w;
            m0r += rgb.x; m0g += rgb.y; m0b += rgb.z;
            m1r += rgb.x * rgb.x; m1g += rgb.y * rgb.y; m1b += rgb.z * rgb.z;
            green_differences += s61_fabs(g1 - g2) * w;
            if (x == 0 && y == 0) centre_gdiff = s61_fabs(g1 - g2);
        }
    }
    m0r /= 9.0f; m0g /= 9.0f; m0b /= 9.0f;
    m1r /= 9.0f; m1g /= 9.0f; m1b /= 9.0f;
    const S61F3 mean = s61_rgb_to_ycocg(s61_f3(avr, avg, avb));
    const float mlx = 0.5f * avg;   // mean_luma = (0.5*average_rgb.y, 0, 0)
    const float vr = m1r - m0r * m0r, vg = m1g - m0g * m0g, vb = m1b - m0b * m0b;
    float std_dev = s61_sqrt(s61_max(vr * (1.0f / 9.0f) + vg * (1.0f / 9.0f) + vb * (1.0f / 9.0f), 0.0f));
    std_dev = s61_max(std_dev, green_differences);
    const float impact = s61_min(S61_GREEN_DIFF_IMPACT * green_differences, 1.0f);
    ref4[0] = s61_mix(mean.x, mlx, impact);
    ref4[1] = s61_mix(mean.y, 0.0f, impact);
    ref4[2] = s61_mix(mean.z, 0.0f, impact);
    ref4[3] = std_dev;

    // ---- kernel (REF_COLOR_ONLY not defined) ----
    const S61F4 noise = s61_lut_linear(lut, mean.x * P.nsb[0] + P.nsb[2], 1.0f * P.nsb[1] + P.nsb[3]);
    const float gd_raw = green_differences;
    {
        const float den = green_differences + noise.x;
        green_differences = den != 0.0f ? green_differences * (green_differences / den) : 0.0f;
    }
    float dxx = 0.0f, dyy = 0.0f, dxy = 0.0f;
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
            const float dx = sums[(y + 1) * 3 + (x + 1)] - sums[y * 3 + x];
            const float dy = sums[y * 3 + (x + 1)] - sums[(y + 1) * 3 + x];
            dxx += dx * dx; dyy += dy * dy; dxy += dx * dy;
        }
    // RotateCovariance((dxx,dyy,dxy)/8)
    const float cx = dxx / 8.0f, cy = dyy / 8.0f, cz = dxy / 8.0f;
    const float c0 = 0.5f * (cx + cy), c1 = 0.5f * (cy - cx);
    const float sum_xx = c0 + cz, sum_yy = c0 - cz, sum_xy = c1;
    // AnalyzeStructureTensor(covariance, noise.y)
    const float trace = sum_xx + sum_yy, diff = sum_xx - sum_yy;
    const float sqrt_d = s61_sqrt(s61_max(diff * diff + 4.0f * sum_xy * sum_xy, 0.0f));
    float eig1 = (trace + sqrt_d) * 0.5f;
    const float eig2 = (trace - sqrt_d) * 0.5f;
    float e1x = 1.0f, e1y = 0.0f, e2x = 0.0f, e2y = 1.0f;
    if (s61_fabs(sum_xy) > 0.0001f) {
        const float vx = sum_xy, vy = eig1 - sum_xx;
        const float len = s61_sqrt(vx * vx + vy * vy);
        const float sg = -s61_sign(sum_xy);
        e1x = (len > 0.0f ? vx / len : 0.0f) * sg;
        e1y = (len > 0.0f ? vy / len : 0.0f) * sg;
        e2x = -e1y; e2y = e1x;
    } else if (sum_xx < sum_yy) {
        e1x = 0.0f; e1y = 1.0f; e2x = 1.0f; e2y = 0.0f;
    }
    const float sv1 = s61_sqrt(eig1), sv2 = s61_sqrt(s61_max(eig2, 0.0f));
    const float eig1_raw = eig1;
    {
        const float den = eig1 + noise.y;
        eig1 = den != 0.0f ? eig1 * (eig1 / den) : 0.0f;
    }
    const float strength = s61_sqrt(eig1);
    const float coherence = (sv1 - sv2) / (sv1 + sv2 + 1.0e-6f);
    // ConstructGaussianCovariance
    const float dominant = s61_max(strength, green_differences) - S61_GRADIENT_CLIP;
    const float gradient_blurring = s61_clampf(1.0f - dominant * S61_GRADIENT_TRANSITION, 0.0f, 1.0f);
    const float aniso = s61_mix(S61_BASE_SIGMA, S61_SHRINKED_SIGMA, s61_min(coherence, strength * 5.0f));
    const float s1 = s61_mix(aniso, S61_BLURRED_SIGMA, gradient_blurring);
    const float s2 = s61_mix(s61_mix(S61_BASE_SIGMA, S61_STRETCHED_SIGMA, coherence), S61_BLURRED_SIGMA,
                             gradient_blurring);
    const float l1 = s1 * s1, l2 = s2 * s2;
    // rr = transpose(R) * diag(l1, l2) * R, R = mat2(ev1, ev2) (columns); return (rr[0].x, rr[1].y, rr[0].y)
    cov3[0] = e1x * e1x * l1 + e1y * e1y * l2;
    cov3[1] = e2x * e2x * l1 + e2y * e2y * l2;
    cov3[2] = e1x * e2x * l1 + e1y * e2y * l2;
    if (dbg) {
        dbg[0] = gd_raw; dbg[1] = sum_xx; dbg[2] = sum_yy; dbg[3] = eig1_raw; dbg[4] = centre_gdiff;
    }
}

// ------------------------------------------------------------------------------------------------ STAGE 2
// Rejection main() + ComputeFrameWeight for guide pixel (gx, gy). Returns 1 - weight (R8 rejection).
// unblocker = null -> the unblocker texture reads 0 (unblocker off).
S61_INLINE float s61_rejection_pixel(const S61_DEVICE float* base_guide, const S61_DEVICE float* alt_guide,
                                     const S61_DEVICE float* flow, const S61_DEVICE float* lut,
                                     const S61_DEVICE uint8_t* unblocker, int ub_w, int ub_h,
                                     S61RejParams P, int gx, int gy) {
    const float tcx = ((float)gx + 0.5f) / (float)P.Wg, tcy = ((float)gy + 0.5f) / (float)P.Hg;
    const S61F4 fl = s61_flow_nearest(flow, P.fnx, P.fny, tcx * P.fso[0] + P.fso[2], tcy * P.fso[1] + P.fso[3]);
    float wx = tcx + fl.x, wy = tcy + fl.y;
    wx = wx > 1.0f ? 2.0f - wx : wx;
    wy = wy > 1.0f ? 2.0f - wy : wy;
    wx = wx < 0.0f ? -wx : wx;
    wy = wy < 0.0f ? -wy : wy;
    float unb = unblocker ? s61_nearest_u8(unblocker, ub_w, ub_h, tcx, tcy) : 0.0f;
    const float lfv = fl.z;
    bool prior = false;
    if (lfv < P.ub_thr) unb = 0.0f;
    if (lfv > P.mot_thr) prior = true;

    // ComputeFrameWeight(tex_coord, warped_uv, prior)
    const S61F4 ref = s61_sample_bicubic(base_guide, P.Wg, P.Hg, tcx, tcy);
    const S61F4 cur = s61_sample_bicubic(alt_guide, P.Wg, P.Hg, wx, wy);
    S61F4 noise = s61_lut_linear(lut, ref.x * P.nsb[0] + P.nsb[2], 0.0f * P.nsb[1] + P.nsb[3]);
    float std_dev = s61_min(ref.w, cur.w);
    const float min_std_dev = noise.w;
    float boost = 1.0f;
    if (std_dev > S61_BOOST_STD_THRESHOLD * min_std_dev && prior) boost = S61_BOOST;
    std_dev = s61_max(std_dev, min_std_dev);
    noise.x *= 2.0f; noise.y *= 2.0f; noise.z *= 2.0f;
    float d0 = (cur.x - ref.x) * (cur.x - ref.x);
    float d1 = (cur.y - ref.y) * (cur.y - ref.y);
    float d2 = (cur.z - ref.z) * (cur.z - ref.z);
    // d *= d / (d + noise); 0/0 (no difference, no noise) is no contribution
    d0 = (d0 + noise.x) != 0.0f ? d0 * (d0 / (d0 + noise.x)) : 0.0f;
    d1 = (d1 + noise.y) != 0.0f ? d1 * (d1 / (d1 + noise.y)) : 0.0f;
    d2 = (d2 + noise.z) != 0.0f ? d2 * (d2 / (d2 + noise.z)) : 0.0f;
    const float num = d0 + d1 + d2;
    const float sd2 = std_dev * std_dev;
    float distance = sd2 > 0.0f ? num / sd2 : (num > 0.0f ? 3.0e38f : 0.0f);
    distance *= S61_DISTANCE_MULT * boost;
    const float w = s61_exp2(s61_min(-distance + S61_REJECTION_THRESHOLD, 0.0f));
    const float weight = s61_min(1.0f - unb, w);
    return 1.0f - weight;
}

// Dilate main(): 9 textureGathers of the R8 rejection around the guide pixel (5x5, each texel once),
// sum / 3, output 1 - rejection (keep). Sum order = shader order.
S61_INLINE float s61_dilate_pixel(const S61_DEVICE uint8_t* rej, int W, int H, int gx, int gy) {
    const float k = 1.0f / 255.0f;
#define S61_T(ix, iy) ((float)rej[s61_clampi(gy - 2 + (iy), 0, H - 1) * W + s61_clampi(gx - 2 + (ix), 0, W - 1)] * k)
#define S61_Q(gxx, gyy) (((S61_T(gxx, gyy + 1) + S61_T(gxx + 1, gyy + 1)) + S61_T(gxx + 1, gyy)) + S61_T(gxx, gyy))
    float r = 0.0f;
    r += S61_Q(0, 0);
    r += S61_Q(2, 0);
    r += S61_T(4, 0); r += S61_T(4, 1);
    r += S61_Q(0, 2);
    r += S61_Q(2, 2);
    r += S61_T(4, 2); r += S61_T(4, 3);
    r += S61_T(0, 4); r += S61_T(1, 4);
    r += S61_T(2, 4); r += S61_T(3, 4);
    r += S61_T(4, 4);
#undef S61_Q
#undef S61_T
    r /= 3.0f;
    return 1.0f - r;
}

// unorm8 store (round to nearest, clamp)
S61_INLINE uint8_t s61_to_u8(float v) {
    const float c = s61_clampf(v, 0.0f, 1.0f) * 255.0f;
    return (uint8_t)(int)s61_floor(c + 0.5f);
}
// copy_mask with GL_BLEND(ONE, ONE) into the R8 accumulated weights: dst = unorm8(src/25 + dst)
S61_INLINE uint8_t s61_copy_mask_accumulate(uint8_t acc, uint8_t keep) {
    const float src = ((float)keep * (1.0f / 255.0f)) / S61_ACC_WEIGHT_SCALE;
    return s61_to_u8(src + (float)acc * (1.0f / 255.0f));
}

// ------------------------------------------------------------------------------------------------ STAGE 3
// Merge main() (non-zero-copy path) for output pixel (ox, oy). Writes this frame's contribution:
//   alt frames: color3 / weight3 (already multiplied by frame_weight) -> added by the caller
//   base frame: same, and the caller blends final = (acc_color + color) / (acc_weight + weight)
S61_INLINE void s61_merge_pixel(const S61_DEVICE float* raw, const S61_DEVICE float* cov,
                                const S61_DEVICE float* flow, const S61_DEVICE uint8_t* mask,
                                const S61_DEVICE uint8_t* acc_w, S61MergeParams P, int ox, int oy,
                                S61_THREAD float* color3, S61_THREAD float* weight3) {
    const float tcx = ((float)ox + 0.5f) / (float)P.Wo, tcy = ((float)oy + 0.5f) / (float)P.Ho;
    // mvp_matrix_tex = identity at our output scale: zoomed_tc = tex_coord.
    const float ftx = tcx * P.fso[0] + P.fso[2], fty = tcy * P.fso[1] + P.fso[3];
    float frame_weight = 1.0f;
    if (!P.is_base) {
        frame_weight = s61_nearest_u8(mask, P.Wg, P.Hg, tcx, tcy);
        frame_weight = P.use_rejection ? frame_weight : 1.0f;
    }
    const S61F4 fl = s61_flow_nearest(flow, P.fnx, P.fny, ftx, fty);
    float sx = tcx + fl.x, sy = tcy + fl.y;
    if (sx <= P.fbp[0]) sx = 2.0f * P.fbp[0] - sx;
    if (sy <= P.fbp[1]) sy = 2.0f * P.fbp[1] - sy;
    if (sx > P.fbp[2]) sx = 2.0f * P.fbp[2] - sx;
    if (sy > P.fbp[3]) sy = 2.0f * P.fbp[3] - sy;
    float c[3];
    s61_linear_plane(cov, P.Wg, P.Hg, 3, P.cov_row0, P.cov_nrows, sx, sy, c);
    const float normalization = 1.0f;
    if (P.is_base) {
        const float weight_so_far = s61_linear_u8(acc_w, P.Wg, P.Hg, tcx, tcy) * S61_ACC_WEIGHT_SCALE;
        if (weight_so_far < S61_KB_THRESHOLD) { c[0] *= S61_KB_MULT; c[1] *= S61_KB_MULT; c[2] *= S61_KB_MULT; }
    }
    // SampleNeighborhoodRBF (non zero-copy)
    float csx = sx * (float)P.Wr - 0.5f, csy = sy * (float)P.Hr - 0.5f;
    const float gxc = s61_floor((csx - 0.5f) / 2.0f) * 2.0f, gyc = s61_floor((csy - 0.5f) / 2.0f) * 2.0f;
    const float offx = gxc - csx, offy = gyc - csy;   // round(gather_coordinate) - coord_scaled
    const int gix = (int)gxc, giy = (int)gyc;
    float ai0 = 0.0f, ai1 = 0.0f, ai2 = 0.0f, aw0 = 0.0f, aw1 = 0.0f, aw2 = 0.0f;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            const int qx = gix + 2 * x, qy = giy + 2 * y;      // quad top-left (texel index)
            const float bx = offx + (float)(2 * x), by = offy + (float)(2 * y);
            // ProcessBayerQuad: per site, d = subpixel_offset + site offset
            float vals[4], dxs[4], dys[4];
            int col[4];
            vals[0] = s61_raw(raw, P.Wr, P.Hr, P.raw_row0, P.raw_nrows, qx + P.cfa.rx, qy + P.cfa.ry);
            dxs[0] = bx + (float)P.cfa.rx; dys[0] = by + (float)P.cfa.ry; col[0] = 0;
            vals[1] = s61_raw(raw, P.Wr, P.Hr, P.raw_row0, P.raw_nrows, qx + P.cfa.g1x, qy + P.cfa.g1y);
            dxs[1] = bx + (float)P.cfa.g1x; dys[1] = by + (float)P.cfa.g1y; col[1] = 1;
            vals[2] = s61_raw(raw, P.Wr, P.Hr, P.raw_row0, P.raw_nrows, qx + P.cfa.g2x, qy + P.cfa.g2y);
            dxs[2] = bx + (float)P.cfa.g2x; dys[2] = by + (float)P.cfa.g2y; col[2] = 1;
            vals[3] = s61_raw(raw, P.Wr, P.Hr, P.raw_row0, P.raw_nrows, qx + P.cfa.bx, qy + P.cfa.by);
            dxs[3] = bx + (float)P.cfa.bx; dys[3] = by + (float)P.cfa.by; col[3] = 2;
            for (int s = 0; s < 4; ++s) {
                const float dx = dxs[s], dy = dys[s];
                const float distance = dx * dx * c[0] + dy * dy * c[1] + 2.0f * c[2] * dx * dy;
                const float w = s61_max(s61_fabs(dx), s61_fabs(dy)) <= 1.5f
                                    ? s61_exp2(-0.5f * distance) * normalization + S61_KEPSILON : 0.0f;
                if (col[s] == 0) { ai0 += w * vals[s]; aw0 += w; }
                else if (col[s] == 1) { ai1 += w * vals[s]; aw1 += w; }
                else { ai2 += w * vals[s]; aw2 += w; }
            }
        }
    }
    color3[0] = ai0 * frame_weight; color3[1] = ai1 * frame_weight; color3[2] = ai2 * frame_weight;
    weight3[0] = aw0 * frame_weight; weight3[1] = aw1 * frame_weight; weight3[2] = aw2 * frame_weight;
}

// ------------------------------------------------------------------------------------------------ STAGE 4
S61_INLINE S61F3 s61_to_gcam_yuv(S61F3 c) {      // kLinearRgbToFunkyYuv * col
    return s61_f3(0.2126390f * c.x + 0.7151688f * c.y + 0.0721923f * c.z,
                  -0.7873610f * c.x + 0.7151688f * c.y + 0.0721923f * c.z,
                  0.2126390f * c.x + 0.7151688f * c.y - 0.9278077f * c.z);
}
S61_INLINE S61F3 s61_from_gcam_yuv(S61F3 c) {    // kFunkyYuvToLinearRgb * col
    return s61_f3(c.x - c.y, c.x + 0.297327f * c.y + 0.100944f * c.z, c.x - c.z);
}
S61_INLINE float s61_dn_weight(int x) { return x == 0 ? 0.5f : ((x == 1 || x == -1) ? 0.25f : 0.0f); }

// Denoiser main() for output pixel (ox, oy) at level P.level. `in` is an RGB float plane (3 ch) holding
// output rows [in_row0, in_row0+in_nrows). FINAL_OUTPUT's 14-bit transport is not emulated: the last
// level returns linear S2 RGB.
S61_INLINE S61F3 s61_denoise_pixel(const S61_DEVICE float* in, const S61_DEVICE uint8_t* acc_w,
                                   S61DenoiseParams P, int ox, int oy) {
    const float tcx = ((float)ox + 0.5f) / (float)P.Wo, tcy = ((float)oy + 0.5f) / (float)P.Ho;
    // input_texture_scale = 1, offset = 0 (whole frame); mvp = identity.
    const float num_merged = s61_linear_u8(acc_w, P.Wg, P.Hg, tcx, tcy) * S61_ACC_WEIGHT_SCALE + S61_WEIGHT_TEX_BIAS;
    const float blurry = s61_clampf(1.25f - num_merged * 0.25f, 0.0f, 1.0f);
    const float dil = (float)(1 << P.level);
    const float dx = dil / (float)P.Wo, dy = dil / (float)P.Ho;
    float minx = tcx - dx, miny = tcy - dy, maxx = tcx + dx, maxy = tcy + dy;
    minx = minx < P.fb[0] ? 2.0f * P.fb[0] - minx : minx;
    miny = miny < P.fb[1] ? 2.0f * P.fb[1] - miny : miny;
    maxx = maxx > P.fb[2] ? 2.0f * P.fb[2] - maxx : maxx;
    maxy = maxy > P.fb[3] ? 2.0f * P.fb[3] - maxy : maxy;
    S61F3 s[9];
    const float us[3] = {minx, tcx, maxx};
    const float vs[3] = {miny, tcy, maxy};
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            float t[3];
            s61_linear_plane(in, P.Wo, P.Ho, 3, P.in_row0, P.in_nrows, us[i], vs[j], t);
            s[j * 3 + i] = s61_f3(t[0], t[1], t[2]);
        }
    if (P.level == 0)
        for (int i = 0; i < 9; ++i) s[i] = s61_to_gcam_yuv(s[i]);
    const S61F3 col = s[4];
    float key = col.x;
    const float noise_model_var = P.yuv_noise[0] * s61_fabs(col.x) + P.yuv_noise[1] * s61_fabs(col.y) +
                                  P.yuv_noise[2] * s61_fabs(col.z) + P.yuv_noise[3];
    float bilateral_sigma = noise_model_var;
    // vignette_strength_boost = 0 in 6.1: (1 + 0 * vignette) = 1
    bilateral_sigma *= S61_DENOISING_FACTOR;
    const float prefilter_weight = blurry * (P.level == 0 ? 1.0f : 0.0f);
    const float prefiltered_key = (s[0].x + s[2].x + s[4].x + s[6].x + s[8].x) / 5.0f;
    key = s61_mix(key, prefiltered_key, prefilter_weight);
    float inv_sigma = bilateral_sigma > 0.0f ? 1.0f / bilateral_sigma : 3.0e38f;
    inv_sigma *= num_merged;
    inv_sigma *= 0.5f / s61_log(2.0f);
    const float inv_outlier_sigma = S61_OUTLIER_SIGMA_MULT * inv_sigma;
    float a0 = col.x * 0.25f, a1 = col.y * 0.25f, a2 = col.z * 0.25f, aw = 0.25f;
    float b0 = a0, b1 = a1, b2 = a2, bw = aw;
    const int px[4] = {-1, 0, 1, 1};
    const int py[4] = {-1, -1, -1, 0};
    for (int k = 0; k < 4; ++k) {
        const int x = px[k], y = py[k];
        const S61F3 samp = s[(y + 1) * 3 + x + 1];
        const S61F3 opp = s[(-y + 1) * 3 - x + 1];
        const float diff = (samp.x + opp.x) * 0.5f - key;
        const float diff_sq = diff * diff;
        const float ww = s61_dn_weight(x) * s61_dn_weight(y);
        const float weight = ww * s61_exp2(-diff_sq * inv_sigma);
        const float weight_wider = ww * s61_exp2(-diff_sq * inv_outlier_sigma);
        a0 += (samp.x + opp.x) * weight; a1 += (samp.y + opp.y) * weight; a2 += (samp.z + opp.z) * weight;
        aw += 2.0f * weight;
        b0 += (samp.x + opp.x) * weight_wider; b1 += (samp.y + opp.y) * weight_wider;
        b2 += (samp.z + opp.z) * weight_wider; bw += 2.0f * weight_wider;
    }
    S61F3 den = s61_f3(a0 / aw, a1 / aw, a2 / aw);
    const float outlier_weight = (1.0f - s61_smoothstep(0.25f, 0.5f, aw)) * blurry;
    den.x = s61_mix(den.x, b0 / bw, outlier_weight);
    den.y = s61_mix(den.y, b1 / bw, outlier_weight);
    den.z = s61_mix(den.z, b2 / bw, outlier_weight);
    const float lrf = s61_mix(S61_GLOBAL_LUMA_REVERT, S61_REJECT_LUMA_REVERT, blurry);
    den.x = den.x * (1.0f - lrf) + col.x * lrf;
    // grain_strength = 0 in 6.1: no grain
    if (P.level == 2) den = s61_from_gcam_yuv(den);
    return den;
}

#endif  // HHSR_SABRE61_SHARED_H
