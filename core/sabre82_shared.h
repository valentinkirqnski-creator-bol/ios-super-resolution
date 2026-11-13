#ifndef HHSR_SABRE82_SHARED_H
#define HHSR_SABRE82_SHARED_H
//
// GCam 8.2.300 SABRE, per-pixel arithmetic, in a form both compilers accept.
//
// Source of truth: Documents/super/SABRE_8.2_COMPLETE_REFERENCE.md (verbatim GLSL from libgcastartup.so).
// Every function below transcribes one GLSL function / entry point; GL samplers and texture formats are
// emulated explicitly so the CPU (core/sabre82.cpp) and the GPU (core/Sabre82Kernels.metal) run the SAME text:
//
//   LINEAR   = 4-texel bilinear at (uv*size - 0.5), CLAMP_TO_EDGE      guides, base reference colour,
//                                                                      kernel covariance, noise LUT
//   NEAREST  = texel floor(uv*size), clamped                          extracted Bayer, flow, rejection mask
//   gather   = the 2x2 quad at floor(uv*size - 0.5), each texel clamped (dilate)
//
// Storage emulated at each boundary: RGBA16F (guide, base reference colour) -> fp16 round; RGB10_A2 kernel
// covariance -> 10-bit unorm of the PACK_COV-remapped value; R8 (rejection, dilated mask) -> 8-bit unorm.
//
// Our raw planes are already GCam's normalized space (the loader delivers (raw - black)/(white - black) *
// wb[c]/wb[G], exposure-matched), so `gains` = 1 and `black_levels_times_gains` = 0: x*gains + black == x.
// The ExtractBayerTexture pass (RGGB-ordered quads per texel) is the identity on such a plane: quads are read
// straight from it through the CFA site table. Its `+ vec4(1e-4)` is truncated away by the guide's uvec4()
// read and kept by the merge's float read (S82MergeParams::eps).
//
// Portability (MSL is C++14-based): no templates, no std::, no references in signatures, address spaces
// through S82_DEVICE / S82_THREAD, plain float/int, POD structs, fixed-size arrays.

#ifdef __METAL_VERSION__
#define S82_DEVICE device
#define S82_THREAD thread
#define S82_INLINE static inline
#define s82_floor(x) floor(x)
#define s82_sqrt(x) sqrt(x)
#define s82_exp2(x) exp2(x)
#define s82_fabs(x) fabs(x)
static inline float s82_q16(float x) { return float(half(x)); }
#else
#include <math.h>
#include <stdint.h>
#include <string.h>
#define S82_DEVICE
#define S82_THREAD
#define S82_INLINE static inline
#define s82_floor(x) floorf(x)
#define s82_sqrt(x) sqrtf(x)
#define s82_exp2(x) exp2f(x)
#define s82_fabs(x) fabsf(x)
// fp32 <-> fp16 bits, round-to-nearest-even (what storing into an RGBA16F texture does), portable.
S82_INLINE uint16_t s82_f2h_bits(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return (uint16_t)(sign | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u));
    if (a >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);
    uint32_t h;
    if (a < 0x38800000u) {
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
S82_INLINE float s82_h2f(uint16_t hb) {
    const uint32_t he = (hb >> 10) & 0x1fu, hm = hb & 0x3ffu, sign = (uint32_t)(hb & 0x8000u) << 16;
    uint32_t r;
    if (he == 0) {
        if (hm == 0) r = sign;
        else { float v = (float)hm * (1.0f / 16777216.0f); memcpy(&r, &v, 4); r |= sign; }
    } else if (he == 31) r = sign | 0x7f800000u | (hm << 13);
    else r = sign | ((he + 112u) << 23) | (hm << 13);
    float o; memcpy(&o, &r, 4); return o;
}
S82_INLINE float s82_q16(float f) { return s82_h2f(s82_f2h_bits(f)); }
#endif

S82_INLINE float s82_min(float a, float b) { return a < b ? a : b; }
S82_INLINE float s82_max(float a, float b) { return a > b ? a : b; }
S82_INLINE float s82_clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
S82_INLINE int s82_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
S82_INLINE float s82_mix(float a, float b, float t) { return a + (b - a) * t; }
S82_INLINE float s82_sign(float v) { return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); }

struct S82F4 { float x, y, z, w; };
S82_INLINE S82F4 s82_f4(float x, float y, float z, float w) { S82F4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

// ------------------------------------------------------------------------------------------------ uniforms
// Values not fixed by the 8.2 GLSL are per-shot uniforms; defaults are those captured from the same shader
// family on device (captures/fresh/dbgpull, 2x daylight): see S82Defaults in sabre82.cpp.
#define S82_KFVS_RGB 0.0898866f          // kFilterVarianceScale, bicubic path, RGB
#define S82_KFVS_GREEN 0.211665f         // kFilterVarianceScale, bicubic path, green only
#define S82_KEPSILON 0.00005f            // GetKernelWeight kEpsilon
#define S82_DEHOMOG_EPS 0.0000001f       // SabreDehomogenize kEpsilon
#define S82_FORCE_REJECT_COLOR 10000.0f  // kForceRejectReferenceColor
#define S82_RS_CLIP_LUMA 9000.0f         // residual-shift test: luma at/above = a force-rejected texel

// R, G1 (RGGB-order y), G2 (z), B site positions in each 2x2 quad. st_gbrg selects the GRBG/GBRG structure-
// tensor formula; GCam's guard is `#if defined(BAYER_GBRG) || defined(BAYER_GBRG)` (sic), so only GBRG takes it.
struct S82Cfa { int ry, rx, g1y, g1x, g2y, g2x, by, bx; int st_gbrg; };

struct S82GuideParams {
    int Wr, Hr;               // raw (Bayer) plane
    int Wg, Hg;               // guide / extracted-Bayer size (= tex_size)
    int row0, nrows;          // raw rows held in the plane passed in
    S82Cfa cfa;
    int force_rgb;            // force_reference_color_rgb.x != 0
    float green_clip;         // green_clipping_point
    float fbp[4];             // frame_border_padded (guide pass)
    float nsb[4];             // noise_texture_scale_bias
    float cov1[4];            // covariance_parameters1: x shrinked, y stretched, z gradient clip, w base
    float cov2[2];            // covariance_parameters2: x blurred, y gradient transition
    float pack_rg[4];         // cov_range_rg_factors
    float pack_b[2];          // cov_range_b_factor
};

struct S82RejParams {
    int Wg, Hg;               // guide size (tex_size)
    int fnx, fny;             // flow texture size
    float fso[4];             // flow_scale_offset
    float fbp[4];             // frame_border_padded (rejection pass)
    float nsb[4];
    float cdm[2];             // color_difference_multiplier (RGB, green only)
    float boost;              // extra_motion_robustness_boost
    float boost_var_thr;      // motion_robustness_boost_variance_threshold
    float mot_thr, ub_thr;    // flow.z thresholds (uv)
    float noise_scale;        // 1 = the LUT as filled (exposed as a tuning knob)
    int unx, uny;             // unblocker texture (R8, one texel per 8x8 guide pixels)
    float uscale[2];          // unblocker_scale
};

struct S82MergeParams {
    int Wo, Ho;               // output
    int Wr, Hr;               // raw plane
    int Wg, Hg;               // guide (kernel covariance texture; tex_size of the merge)
    int Wm, Hm;               // dilated rejection mask
    int fnx, fny;
    int raw_row0, raw_nrows;
    int cov_row0, cov_nrows;
    int out_row0;
    int use_mask;             // 0 for the base frame (frame_info.w == 0), 1 otherwise
    float fso[4];
    float fbp[4];             // frame_border_padded (merge pass)
    float unpack_rg[4];       // cov_range_rg
    float unpack_b[2];        // cov_range_b
    float eps[4];             // ExtractBayer's +1e-4 DN in our normalized units, per site (y&1)*2+(x&1)
    S82Cfa cfa;
};

// ------------------------------------------------------------------------------------------------ samplers
S82_INLINE S82F4 s82_lut_linear(const S82_DEVICE float* lut, float u, float v) {   // 10x2 RGBA32F, LINEAR
    const float x = u * 10.0f - 0.5f, y = v * 2.0f - 0.5f;
    const float fx = s82_floor(x), fy = s82_floor(y);
    const float ax = x - fx, ay = y - fy;
    const int x0 = s82_clampi((int)fx, 0, 9), x1 = s82_clampi((int)fx + 1, 0, 9);
    const int y0 = s82_clampi((int)fy, 0, 1), y1 = s82_clampi((int)fy + 1, 0, 1);
    float o[4];
    for (int c = 0; c < 4; ++c) {
        const float a = lut[(y0 * 10 + x0) * 4 + c] * (1.0f - ax) + lut[(y0 * 10 + x1) * 4 + c] * ax;
        const float b = lut[(y1 * 10 + x0) * 4 + c] * (1.0f - ax) + lut[(y1 * 10 + x1) * 4 + c] * ax;
        o[c] = a * (1.0f - ay) + b * ay;
    }
    return s82_f4(o[0], o[1], o[2], o[3]);
}
// LINEAR fetch of an interleaved float plane (C channels; rows [row0, row0+nrows) held).
S82_INLINE void s82_linear_plane(const S82_DEVICE float* p, int W, int H, int C, int row0, int nrows,
                                 float u, float v, S82_THREAD float* out) {
    const float x = u * (float)W - 0.5f, y = v * (float)H - 0.5f;
    const float fx = s82_floor(x), fy = s82_floor(y);
    const float ax = x - fx, ay = y - fy;
    const int x0 = s82_clampi((int)fx, 0, W - 1), x1 = s82_clampi((int)fx + 1, 0, W - 1);
    const int y0 = s82_clampi(s82_clampi((int)fy, 0, H - 1) - row0, 0, nrows - 1);
    const int y1 = s82_clampi(s82_clampi((int)fy + 1, 0, H - 1) - row0, 0, nrows - 1);
    for (int c = 0; c < C; ++c) {
        const float a = p[((long)y0 * W + x0) * C + c] * (1.0f - ax) + p[((long)y0 * W + x1) * C + c] * ax;
        const float b = p[((long)y1 * W + x0) * C + c] * (1.0f - ax) + p[((long)y1 * W + x1) * C + c] * ax;
        out[c] = a * (1.0f - ay) + b * ay;
    }
}
S82_INLINE float s82_nearest_u8(const S82_DEVICE uint8_t* p, int W, int H, float u, float v) {
    const int x = s82_clampi((int)s82_floor(u * (float)W), 0, W - 1);
    const int y = s82_clampi((int)s82_floor(v * (float)H), 0, H - 1);
    return (float)p[y * W + x] * (1.0f / 255.0f);
}
S82_INLINE S82F4 s82_flow_nearest(const S82_DEVICE float* flow, int nx, int ny, float u, float v) {
    const int x = s82_clampi((int)s82_floor(u * (float)nx), 0, nx - 1);
    const int y = s82_clampi((int)s82_floor(v * (float)ny), 0, ny - 1);
    const S82_DEVICE float* f = flow + ((long)y * nx + x) * 4;
    return s82_f4(f[0], f[1], f[2], f[3]);
}
// kFunctionSampleBicubic: ComputeBicubicParameters + SampleBicubicAbsolute (abs after each LINEAR tap).
S82_INLINE S82F4 s82_sample_bicubic_abs(const S82_DEVICE float* tex, int W, int H, float u, float v) {
    const float uvx = u * (float)W + 0.5f, uvy = v * (float)H + 0.5f;
    const float ix = s82_floor(uvx), iy = s82_floor(uvy);
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
    s82_linear_plane(tex, W, H, 4, 0, H, p00x, p00y, t0);
    s82_linear_plane(tex, W, H, 4, 0, H, p11x, p00y, t1);
    s82_linear_plane(tex, W, H, 4, 0, H, p00x, p11y, t2);
    s82_linear_plane(tex, W, H, 4, 0, H, p11x, p11y, t3);
    float o[4];
    for (int c = 0; c < 4; ++c)
        o[c] = g0y * (g0x * s82_fabs(t0[c]) + g1x * s82_fabs(t1[c])) + g1y * (g0x * s82_fabs(t2[c]) + g1x * s82_fabs(t3[c]));
    return s82_f4(o[0], o[1], o[2], o[3]);
}
// kFunctionMirrorUVs
S82_INLINE void s82_mirror(S82_THREAD float* uv, const S82_THREAD float* fbp) {
    if (uv[0] <= fbp[0]) uv[0] = 2.0f * fbp[0] - uv[0];
    if (uv[1] <= fbp[1]) uv[1] = 2.0f * fbp[1] - uv[1];
    if (uv[0] > fbp[2]) uv[0] = 2.0f * fbp[2] - uv[0];
    if (uv[1] > fbp[3]) uv[1] = 2.0f * fbp[3] - uv[1];
}
S82_INLINE float s82_raw(const S82_DEVICE float* raw, int Wr, int Hr, int row0, int nrows, int x, int y) {
    const int xx = s82_clampi(x, 0, Wr - 1);
    const int yy = s82_clampi(s82_clampi(y, 0, Hr - 1) - row0, 0, nrows - 1);
    return raw[(long)yy * Wr + xx];
}
S82_INLINE uint8_t s82_to_u8(float v) {
    return (uint8_t)(int)s82_floor(s82_clampf(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}
S82_INLINE int s82_unorm10_code(float v) {   // RGB10_A2 store: 10-bit unorm code
    return (int)s82_floor(s82_clampf(v, 0.0f, 1.0f) * 1023.0f + 0.5f);
}

// ------------------------------------------------------------------------------------------------ guide
S82_INLINE float s82_weight(int x) { return x == 0 ? 0.5f : ((x == 1 || x == -1) ? 0.25f : 0.0f); }

// kGuideImageEntryPoint (COMPUTE_GUIDE_COLOR, SQRT_COLOR_SPACE, Bayer path) for guide pixel (gx, gy).
//   ref4 = o_frag_color  (RGB reference colour, signed variance; caller stores RGBA16F)
//   cov3 = o_gauss_cov   (PACK_COV-remapped covariance in [0,1]; caller stores RGB10_A2)
S82_INLINE void s82_guide_pixel(const S82_DEVICE float* raw, S82GuideParams P, const S82_DEVICE float* lut,
                                int gx, int gy, S82_THREAD float* ref4, S82_THREAD float* cov3) {
    float tcm[2];
    tcm[0] = ((float)gx + 0.5f) / (float)P.Wg;
    tcm[1] = ((float)gy + 0.5f) / (float)P.Hg;
    s82_mirror(tcm, P.fbp);
    const float tsz = 1.0f / (float)P.Wg, tsw = 1.0f / (float)P.Hg;
    float m1r = 0.0f, m1g = 0.0f, m1b = 0.0f, m0r = 0.0f, m0g = 0.0f, m0b = 0.0f;
    float m0_green = 0.0f, m1_green = 0.0f;
    float avr = 0.0f, avg = 0.0f, avb = 0.0f;
    float g0v[9], g1v[9];
    float center_green = 0.0f;
    for (int y = -1; y <= 1; y += 1) {
        // GetThreeBayerQuadFromExtractedBayer(tc_mirror + (0, tex_size.w*y)): NEAREST texels at x -1, 0, +1
        const float pv = tcm[1] + tsw * (float)y;
        const int qy = s82_clampi((int)s82_floor(pv * (float)P.Hg), 0, P.Hg - 1);
        for (int x = -1; x <= 1; x += 1) {
            const float pu = tcm[0] + tsz * (float)x;
            const int qx = s82_clampi((int)s82_floor(pu * (float)P.Wg), 0, P.Wg - 1);
            const int bx = 2 * qx, by = 2 * qy;
            float r = s82_raw(raw, P.Wr, P.Hr, P.row0, P.nrows, bx + P.cfa.rx, by + P.cfa.ry);
            float g1 = s82_raw(raw, P.Wr, P.Hr, P.row0, P.nrows, bx + P.cfa.g1x, by + P.cfa.g1y);
            float g2 = s82_raw(raw, P.Wr, P.Hr, P.row0, P.nrows, bx + P.cfa.g2x, by + P.cfa.g2y);
            float b = s82_raw(raw, P.Wr, P.Hr, P.row0, P.nrows, bx + P.cfa.bx, by + P.cfa.by);
            r = s82_sqrt(s82_max(0.0f, r)); g1 = s82_sqrt(s82_max(0.0f, g1));
            g2 = s82_sqrt(s82_max(0.0f, g2)); b = s82_sqrt(s82_max(0.0f, b));
            const float cr = r, cg = 0.5f * (g1 + g2), cb = b;
            g0v[(y + 1) * 3 + x + 1] = g1;
            g1v[(y + 1) * 3 + x + 1] = g2;
            const float w = s82_weight(x) * s82_weight(y);
            avr += cr * w; avg += cg * w; avb += cb * w;
            if (x == 0 && y == 0) center_green = cg;
            m0r += cr; m0g += cg; m0b += cb;
            m1r += cr * cr; m1g += cg * cg; m1b += cb * cb;
            m0_green += g1 + g2;
            m1_green += g1 * g1 + g2 * g2;
        }
    }
    m0r /= 9.0f; m0g /= 9.0f; m0b /= 9.0f; m1r /= 9.0f; m1g /= 9.0f; m1b /= 9.0f;
    const float vr = s82_max(0.0f, m1r - m0r * m0r), vg = s82_max(0.0f, m1g - m0g * m0g), vb = s82_max(0.0f, m1b - m0b * m0b);
    m0_green /= 18.0f; m1_green /= 18.0f;
    const float green_variance = s82_max(0.0f, m1_green - m0_green * m0_green);
    const float average_luma = avr * 0.25f + avg * 0.5f + avb * 0.25f;
    const float green_variance_noise = 2.0f * s82_lut_linear(lut, average_luma * P.nsb[0] + P.nsb[2], 1.0f * P.nsb[1] + P.nsb[3]).y;
    float rc0, rc1, rc2, rvar;
    if (green_variance > 3.0f * green_variance_noise && P.force_rgb == 0) {
        rc0 = avr; rc1 = center_green; rc2 = avb;
        rvar = -s82_max(vg, green_variance);
    } else {
        rc0 = avr; rc1 = avg; rc2 = avb;
        rvar = vr * (1.0f / 3.0f) + vg * (1.0f / 3.0f) + vb * (1.0f / 3.0f);
    }
    if (center_green >= P.green_clip) { rc0 = S82_FORCE_REJECT_COLOR; rc1 = S82_FORCE_REJECT_COLOR; rc2 = S82_FORCE_REJECT_COLOR; }
    ref4[0] = rc0; ref4[1] = rc1; ref4[2] = rc2; ref4[3] = rvar;

    // ComputeStructureTensorFromGreen
    float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            const float g00 = g0v[y * 3 + x], g01 = g0v[y * 3 + x + 1];
            const float g10 = g1v[y * 3 + x], g11 = g1v[y * 3 + x + 1];
            const float g20 = g0v[(y + 1) * 3 + x], g21 = g0v[(y + 1) * 3 + x + 1];
            const float g30 = g1v[(y + 1) * 3 + x], g31 = g1v[(y + 1) * 3 + x + 1];
            float bdx, bdy, rdx, rdy;
            if (P.cfa.st_gbrg) {
                bdx = 0.5f * ((g11 - g01) + (g21 - g10));
                bdy = 0.5f * ((g01 - g10) + (g11 - g21));
                rdx = 0.5f * ((g21 - g10) + (g30 - g20));
                rdy = 0.5f * ((g21 - g30) + (g10 - g20));
            } else {
                bdx = 0.5f * ((g11 - g00) + (g20 - g10));
                bdy = 0.5f * ((g00 - g10) + (g11 - g20));
                rdx = 0.5f * ((g21 - g11) + (g31 - g20));
                rdy = 0.5f * ((g21 - g31) + (g11 - g20));
            }
            const float g1dx = 0.5f * (g21 - g00), g1dy = 0.5f * (g01 - g20);
            const float g2dx = 0.5f * (g31 - g10), g2dy = 0.5f * (g11 - g30);
            sxx += bdx * bdx; syy += bdy * bdy; sxy += bdx * bdy;
            sxx += rdx * rdx; syy += rdy * rdy; sxy += rdx * rdy;
            sxx += g1dx * g1dx; syy += g1dy * g1dy; sxy += g1dx * g1dy;
            sxx += g2dx * g2dx; syy += g2dy * g2dy; sxy += g2dx * g2dy;
        }
    }
    sxx /= 16.0f; syy /= 16.0f; sxy /= 16.0f;
    const float stw = 3.0f / 4.0f;
    // RotateStructureTensor
    const float c0 = 0.5f * (sxx + syy), c1 = 0.5f * (syy - sxx);
    const float sum_xx = c0 + sxy, sum_yy = c0 - sxy, sum_xy = c1;
    // AnalyzeStructureTensor(st, green_variance_noise)
    const float trace = sum_xx + sum_yy, diff = sum_xx - sum_yy;
    const float sqrt_d = s82_sqrt(s82_max(diff * diff + 4.0f * sum_xy * sum_xy, 0.0f));
    float eig1 = (trace + sqrt_d) * 0.5f;
    const float eig2 = (trace - sqrt_d) * 0.5f;
    float e1x = 1.0f, e1y = 0.0f, e2x = 0.0f, e2y = 1.0f;
    if (s82_fabs(sum_xy) > 0.0001f) {
        const float vx = sum_xy, vy = eig1 - sum_xx;
        const float len = s82_sqrt(vx * vx + vy * vy);
        const float sg = -s82_sign(sum_xy);
        e1x = (len > 0.0f ? vx / len : 0.0f) * sg;
        e1y = (len > 0.0f ? vy / len : 0.0f) * sg;
        e2x = -e1y; e2y = e1x;
    } else if (sum_xx < sum_yy) {
        e1x = 0.0f; e1y = 1.0f; e2x = 1.0f; e2y = 0.0f;
    }
    const float sv1 = s82_sqrt(eig1), sv2 = s82_sqrt(s82_max(eig2, 0.0f));
    const float nct = stw * green_variance_noise;
    {
        const float den = eig1 + nct;
        eig1 = den != 0.0f ? eig1 * (eig1 / den) : 0.0f;
    }
    const float strength = s82_sqrt(eig1);
    const float coherence = (sv1 - sv2) / (sv1 + sv2 + 1.0e-6f);
    const float gden = green_variance + green_variance_noise;
    const float ncg_std = gden > 0.0f ? s82_sqrt(green_variance * green_variance / gden) : 0.0f;
    // ConstructGaussianCovariance
    const float dominant = s82_max(strength, ncg_std) - P.cov1[2];
    const float gradient_blurring = s82_clampf(1.0f - dominant * P.cov2[1], 0.0f, 1.0f);
    const float aniso = s82_mix(P.cov1[3], P.cov1[0], s82_min(coherence, strength * 5.0f));
    const float s1 = s82_mix(aniso, P.cov2[0], gradient_blurring);
    const float s2 = s82_mix(s82_mix(P.cov1[3], P.cov1[1], coherence), P.cov2[0], gradient_blurring);
    const float l1 = s1 * s1, l2 = s2 * s2;
    const float rr00 = e1x * e1x * l1 + e1y * e1y * l2;   // transpose(R) * diag * R, R = mat2(ev1, ev2)
    const float rr11 = e2x * e2x * l1 + e2y * e2y * l2;
    const float rr01 = e1x * e2x * l1 + e1y * e2y * l2;
    cov3[0] = s82_clampf(rr00 * P.pack_rg[1] + P.pack_rg[0], 0.0f, 1.0f);   // PACK_COV_RG
    cov3[1] = s82_clampf(rr11 * P.pack_rg[3] + P.pack_rg[2], 0.0f, 1.0f);
    cov3[2] = s82_clampf(rr01 * P.pack_b[1] + P.pack_b[0], 0.0f, 1.0f);      // PACK_COV_B
}

// kBaseFrameRefColorEntryPoint (bicubic path) for guide pixel (gx, gy) of the base guide.
S82_INLINE S82F4 s82_base_ref_pixel(const S82_DEVICE float* guide, int Wg, int Hg, int gx, int gy) {
    const float u = ((float)gx + 0.5f) / (float)Wg, v = ((float)gy + 0.5f) / (float)Hg;
    S82F4 r = s82_sample_bicubic_abs(guide, Wg, Hg, u, v);
    float c[4];
    s82_linear_plane(guide, Wg, Hg, 4, 0, Hg, u, v, c);
    if (c[3] < 0.0f) r.w = -r.w;
    return r;
}

// ------------------------------------------------------------------------------------------------ rejection
// kRejectionEntryPoint + ComputeFrameWeight for guide pixel (gx, gy). Returns 1 - weight. `unb` = the burst's
// unblocker texture (R8, P.unx x P.uny, NEAREST at tex_coord * unblocker_scale). *it_out = luma(warped alt) -
// luma(base reference) for the residual-shift test (hhsr extension; 0 where either is force-rejected).
S82_INLINE float s82_rejection_pixel(const S82_DEVICE float* base_ref, const S82_DEVICE float* alt_guide,
                                     const S82_DEVICE float* flow, const S82_DEVICE float* lut,
                                     const S82_DEVICE uint8_t* unb, S82RejParams P, int gx, int gy,
                                     S82_THREAD float* it_out) {
    const float tcx = ((float)gx + 0.5f) / (float)P.Wg, tcy = ((float)gy + 0.5f) / (float)P.Hg;
    const S82F4 fl = s82_flow_nearest(flow, P.fnx, P.fny, tcx * P.fso[0] + P.fso[2], tcy * P.fso[1] + P.fso[3]);
    float wuv[2];
    wuv[0] = tcx + fl.x; wuv[1] = tcy + fl.y;
    s82_mirror(wuv, P.fbp);
    float unblocker = s82_nearest_u8(unb, P.unx, P.uny, tcx * P.uscale[0], tcy * P.uscale[1]);
    const float lfv = fl.z;
    bool prior = false;
    if (lfv < P.ub_thr) unblocker = 0.0f;
    if (lfv > P.mot_thr) prior = true;
    // ComputeFrameWeight
    float rv[4];
    s82_linear_plane(base_ref, P.Wg, P.Hg, 4, 0, P.Hg, tcx, tcy, rv);
    const bool green_only = rv[3] < 0.0f;
    float ref_w = s82_fabs(rv[3]);
    const S82F4 cur = s82_sample_bicubic_abs(alt_guide, P.Wg, P.Hg, wuv[0], wuv[1]);
    float cur_w = cur.w;
    {
        const float lc = (cur.x + cur.y + cur.z) * (1.0f / 3.0f), lr = (rv[0] + rv[1] + rv[2]) * (1.0f / 3.0f);
        *it_out = (lc < S82_RS_CLIP_LUMA && lr < S82_RS_CLIP_LUMA) ? lc - lr : 0.0f;
    }
    const float luma = green_only ? rv[1] : (rv[0] * (1.0f / 3.0f) + rv[1] * (1.0f / 3.0f) + rv[2] * (1.0f / 3.0f));
    const S82F4 rn4 = s82_lut_linear(lut, luma * P.nsb[0] + P.nsb[2], 0.0f * P.nsb[1] + P.nsb[3]);
    const S82F4 cn4 = s82_lut_linear(lut, luma * P.nsb[0] + P.nsb[2], 1.0f * P.nsb[1] + P.nsb[3]);
    const float kf = green_only ? S82_KFVS_GREEN : S82_KFVS_RGB;
    const float ns = P.noise_scale * kf;
    const float rn0 = rn4.x * ns, rn1 = rn4.y * ns, rn2 = rn4.z * ns;
    const float cn0 = cn4.x * ns, cn1 = cn4.y * ns, cn2 = cn4.z * ns;
    ref_w *= kf; cur_w *= kf;
    float pixel_variance = s82_min(ref_w, cur_w);
    const float min_variance = green_only ? rn1 : (rn0 * (1.0f / 3.0f) + rn1 * (1.0f / 3.0f) + rn2 * (1.0f / 3.0f));
    float boost = 1.0f;
    if (ref_w > P.boost_var_thr * min_variance && prior) boost = P.boost;
    const float nv0 = rn0 + cn0, nv1 = rn1 + cn1, nv2 = rn2 + cn2;
    pixel_variance *= 2.0f;
    float d0 = (cur.x - rv[0]) * (cur.x - rv[0]);
    float d1 = (cur.y - rv[1]) * (cur.y - rv[1]);
    float d2 = (cur.z - rv[2]) * (cur.z - rv[2]);
    d0 = s82_max(0.0f, d0 - nv0); d1 = s82_max(0.0f, d1 - nv1); d2 = s82_max(0.0f, d2 - nv2);
    const float v0 = s82_max(pixel_variance, nv0), v1 = s82_max(pixel_variance, nv1), v2 = s82_max(pixel_variance, nv2);
    d0 = v0 > 0.0f ? d0 / v0 : (d0 > 0.0f ? 3.0e38f : 0.0f);
    d1 = v1 > 0.0f ? d1 / v1 : (d1 > 0.0f ? 3.0e38f : 0.0f);
    d2 = v2 > 0.0f ? d2 / v2 : (d2 > 0.0f ? 3.0e38f : 0.0f);
    float distance = green_only ? P.cdm[1] * d1
                                : P.cdm[0] * (d0 * (1.0f / 3.0f) + d1 * (1.0f / 3.0f) + d2 * (1.0f / 3.0f));
    distance *= boost;
    const float w = s82_exp2(s82_min(-distance, 0.0f));
    const float weight = s82_min(1.0f - unblocker, w);
    return 1.0f - weight;
}

// kDilateMaskEntryPoint for output pixel (ox, oy) of an OW x OH target reading the W x H R8 rejection
// (tex_size = W x H): 9 textureGathers, (sum - 0.2) / 2, output 1 - rejection.
// The gather points (tc + off/size)*size - 0.5 land EXACTLY on texel boundaries, so the literal floor() is
// decided by float rounding noise (flips texel to texel). The device resolves them to the fixed 5x5 window
// x in [2o-2, 2o+2], y in [2o-1, 2o+3] (W = 2*OW): measured on c2_GT2 ROBUST_0 -> MASKOP_0 (96.9% exact,
// 100% +-1 = fp16 ties) and 3 dbgpull pairs; the literal reading scored 85.7% / max 204 (a texel = 2 raw px
// to the side). Gather origins below are those integer texels; tap groupings / summation order unchanged.
S82_INLINE float s82_dilate_pixel(const S82_DEVICE uint8_t* rej, int W, int H, int OW, int OH, int ox, int oy) {
    (void)OW; (void)OH;
    const float k = 1.0f / 255.0f;
    float g[4];
#define S82_GATHER(dx, dy)                                                                         \
    {                                                                                              \
        const int i0 = 2 * ox + (dx), j0 = 2 * oy + (dy);                                          \
        const int xa = s82_clampi(i0, 0, W - 1), xb = s82_clampi(i0 + 1, 0, W - 1);               \
        const int ya = s82_clampi(j0, 0, H - 1), yb = s82_clampi(j0 + 1, 0, H - 1);               \
        g[0] = (float)rej[yb * W + xa] * k; g[1] = (float)rej[yb * W + xb] * k;                   \
        g[2] = (float)rej[ya * W + xb] * k; g[3] = (float)rej[ya * W + xa] * k;                   \
    }
#define S82_DOT() (((g[0] + g[1]) + g[2]) + g[3])
    float r = 0.0f;
    S82_GATHER(-2, -1) r += S82_DOT();          // GLSL offset (-1.5, -1.5)
    S82_GATHER(0, -1)  r += S82_DOT();          // ( 0.5, -1.5)
    S82_GATHER(1, -1)  r += g[2]; r += g[1];    // ( 1.5, -1.5)
    S82_GATHER(-2, 1)  r += S82_DOT();          // (-1.5,  0.5)
    S82_GATHER(0, 1)   r += S82_DOT();          // ( 0.5,  0.5)
    S82_GATHER(1, 1)   r += g[2]; r += g[1];    // ( 1.5,  0.5)
    S82_GATHER(-2, 2)  r += g[0]; r += g[1];    // (-1.5,  1.5)
    S82_GATHER(0, 2)   r += g[0]; r += g[1];    // ( 0.5,  1.5)
    S82_GATHER(1, 2)   r += g[1];               // ( 1.5,  1.5)
#undef S82_DOT
#undef S82_GATHER
    r -= 0.2f;
    r /= 2.0f;
    return 1.0f - r;
}

// ------------------------------------------------------------------------------------ residual-shift test
// hhsr EXTENSION, not GCam (Config::sabre_resid_shift_enabled). SABRE's rejection divides the colour difference by
// the local variance, which at a strong edge is ~100x the difference a 0.3-1 px misalignment makes, so it keeps
// misaligned edge samples (burst12 house: weight 0.92-0.94). This measures the misalignment directly: the
// across-edge Lucas-Kanade shift of the warped alt against the base reference over 7x7,
//   s = sum(It * |grad L|) / sum(|grad L|^2)       (guide px; It, L on the same bicubic-B-spline-filtered data)
// and, on strong edges only (|grad L| above the burst's 90th percentile), multiplies the frame's dilated keep by
//   f = exp(-((max(2|s| * cal - t0, 0)) / width)^2)   (2|s|*cal = raw px; cal: the 7x7 sum under-reads a thin edge)
// minimised over the guide pixels a keep texel covers.
struct S82ResidParams {
    int Wg, Hg, Wm, Hm;
    float thr;                // strong-edge |grad L| threshold
    float cal;                // raw px per 2|s|
    float t0, width;          // tolerance and fall-off, raw px
};
// |grad L| of the base reference luma: Sobel / 8, clamp to edge; 0 near force-rejected texels.
S82_INLINE float s82_base_grad_pixel(const S82_DEVICE float* base_ref, int Wg, int Hg, int x, int y) {
    float l[9];
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            const int xx = s82_clampi(x + i, 0, Wg - 1), yy = s82_clampi(y + j, 0, Hg - 1);
            const S82_DEVICE float* p = base_ref + ((long)yy * Wg + xx) * 4;
            const float lu = (p[0] + p[1] + p[2]) * (1.0f / 3.0f);
            if (lu >= S82_RS_CLIP_LUMA) return 0.0f;
            l[(j + 1) * 3 + i + 1] = lu;
        }
    const float gx = ((l[2] - l[0]) + 2.0f * (l[5] - l[3]) + (l[8] - l[6])) * 0.125f;
    const float gy = ((l[6] - l[0]) + 2.0f * (l[7] - l[1]) + (l[8] - l[2])) * 0.125f;
    return s82_sqrt(gx * gx + gy * gy);
}
// R1: the per-guide-pixel factor. itg = It * |grad L| (rejection pass x base gradient), gm = |grad L|.
S82_INLINE float s82_resid_factor_pixel(const S82_DEVICE float* itg, const S82_DEVICE float* gm, S82ResidParams P,
                                        int x, int y) {
    if (!(gm[(long)y * P.Wg + x] > P.thr)) return 1.0f;
    float num = 0.0f, den = 0.0f;
    for (int j = -3; j <= 3; ++j) {
        const int yy = s82_clampi(y + j, 0, P.Hg - 1);
        for (int i = -3; i <= 3; ++i) {
            const long k = (long)yy * P.Wg + s82_clampi(x + i, 0, P.Wg - 1);
            num += itg[k]; den += gm[k] * gm[k];
        }
    }
    if (!(den > 0.0f)) return 1.0f;
    const float r = 2.0f * s82_fabs(num / den) * P.cal;
    const float t = s82_max(r - P.t0, 0.0f) / P.width;
    return s82_exp2(-1.4426950408889634f * t * t);
}
// R2: keep texel (ox, oy) of the Wm x Hm mask times the minimum factor over guide pixels [2o-1, 2o+2]^2.
S82_INLINE float s82_resid_apply_pixel(const S82_DEVICE uint8_t* keep, const S82_DEVICE float* f, S82ResidParams P,
                                       int ox, int oy) {
    float m = 1.0f;
    for (int j = -1; j <= 2; ++j) {
        const int yy = s82_clampi(2 * oy + j, 0, P.Hg - 1);
        for (int i = -1; i <= 2; ++i) m = s82_min(m, f[(long)yy * P.Wg + s82_clampi(2 * ox + i, 0, P.Wg - 1)]);
    }
    return (float)keep[(long)oy * P.Wm + ox] * (1.0f / 255.0f) * m;
}

// ------------------------------------------------------------------------------------------------ merge
// kMergeEntryPoint (ANDROID: zoomed_tc = tex_coord) for output pixel (ox, oy) of one frame. Returns the
// frame's contribution (colour and weight per channel, already multiplied by frame_weight).
S82_INLINE void s82_merge_pixel(const S82_DEVICE float* raw, const S82_DEVICE float* cov,
                                const S82_DEVICE float* flow, const S82_DEVICE uint8_t* mask,
                                S82MergeParams P, int ox, int oy,
                                S82_THREAD float* color3, S82_THREAD float* weight3) {
    const float tcx = ((float)ox + 0.5f) / (float)P.Wo, tcy = ((float)oy + 0.5f) / (float)P.Ho;
    float frame_weight = 1.0f;
    if (P.use_mask) frame_weight = s82_nearest_u8(mask, P.Wm, P.Hm, tcx, tcy);
    const S82F4 fl = s82_flow_nearest(flow, P.fnx, P.fny, tcx * P.fso[0] + P.fso[2], tcy * P.fso[1] + P.fso[3]);
    float s[2];
    s[0] = tcx + fl.x; s[1] = tcy + fl.y;
    s82_mirror(s, P.fbp);
    float c[3];
    s82_linear_plane(cov, P.Wg, P.Hg, 3, P.cov_row0, P.cov_nrows, s[0], s[1], c);
    c[0] = c[0] * P.unpack_rg[1] + P.unpack_rg[0];   // UNPACK_COV_RG
    c[1] = c[1] * P.unpack_rg[3] + P.unpack_rg[2];
    c[2] = c[2] * P.unpack_b[1] + P.unpack_b[0];     // UNPACK_COV_B
    // SampleNeighborhoodRBF (Bayer path): coord in raw pixels, 3x3 around the containing pixel
    const float csx = s[0] * ((float)P.Wg * 2.0f), csy = s[1] * ((float)P.Hg * 2.0f);
    const int px = (int)csx, py = (int)csy;
    const float offx = s82_floor(csx) + 0.5f - csx, offy = s82_floor(csy) + 0.5f - csy;
    float ai0 = 0.0f, ai1 = 0.0f, ai2 = 0.0f, aw0 = 0.0f, aw1 = 0.0f, aw2 = 0.0f;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            const float dx = offx + (float)i, dy = offy + (float)j;
            const float kd = dx * dx * c[0] + dy * dy * c[1] + dx * dy * c[2] * 2.0f;
            const float w = s82_exp2(-0.5f * kd) + S82_KEPSILON;
            const int x = s82_clampi(px + i, 0, P.Wr - 1), y = s82_clampi(py + j, 0, P.Hr - 1);
            const int sy = y & 1, sx = x & 1;
            // Get3x3FromExtractedBayer reads the extracted texel as float, so ExtractBayer's +1e-4 survives here
            // (the guide's uvec4() conversion truncates it away there).
            const float v = s82_raw(raw, P.Wr, P.Hr, P.raw_row0, P.raw_nrows, x, y) + P.eps[sy * 2 + sx];
            if (sy == P.cfa.ry && sx == P.cfa.rx) { ai0 += v * w; aw0 += w; }
            else if (sy == P.cfa.by && sx == P.cfa.bx) { ai2 += v * w; aw2 += w; }
            else { ai1 += v * w; aw1 += w; }
        }
    }
    color3[0] = ai0 * frame_weight; color3[1] = ai1 * frame_weight; color3[2] = ai2 * frame_weight;
    weight3[0] = aw0 * frame_weight; weight3[1] = aw1 * frame_weight; weight3[2] = aw2 * frame_weight;
}

#endif  // HHSR_SABRE82_SHARED_H
