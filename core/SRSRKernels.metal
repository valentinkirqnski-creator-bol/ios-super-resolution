//===----------------------------------------------------------------------===//
// SRSRKernels.metal -- the SRSR pipeline (GCam 6.1 SABRE after Halide-reference
// alignment), one file. Ported from Documents/srsr/Sources/SRSRShaders.
//
// Written in the common subset of the Metal Shading Language and C++: no
// swizzles beyond .x/.y/.z/.w (xy()/xyz() helpers instead), no address-space
// tricks. On the device Xcode compiles it into default.metallib; on the PC the
// test harness compiles the SAME source as C++ against srsr_cpu_shim.h and runs
// it (srsr_cpu.cpp), so the kernels that ship are the kernels that were tested.
// The one structural exception is block matching, whose threadgroup form has a
// serial twin below with identical arithmetic and tie-breaking.
//
// Stage map (SABRE section -> kernel):
//   STAGE 0    srsr_prepare              float raw -> RGGB quad codes + S2 grey
//   ALIGNMENT  srsr_downsample, srsr_gradients, srsr_gradient_tile_sums,
//              srsr_block_matching, srsr_ica_iteration, srsr_flow_transfer,
//              srsr_flow_median, srsr_flow_pack, srsr_clear_flow
//   STAGE 1    srsr_guide                guide (YCoCg sqrt) + merge-kernel covariance
//   STAGE 2f   srsr_unblocker            reconstructed (off by default)
//   STAGE 2    srsr_rejection            ComputeFrameWeight, inverted
//   STAGE 2d   srsr_dilate_horizontal/vertical   5x5 box, 1 - sum/3
//   STAGE 2e   srsr_accumulate_weights   COPY_MASK
//   STAGE 3    srsr_merge                all frames fused, one output band
//   STAGE 4    srsr_denoise              3-level symmetric bilateral, band + halo
//===----------------------------------------------------------------------===//
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#endif
#include "srsr_types.h"

namespace srsr {

#ifdef __METAL_VERSION__
static inline float2 xy(float4 v) { return v.xy; }
static inline float3 xyz(float4 v) { return v.xyz; }
#endif

constexpr sampler kLinearClamp(coord::normalized, address::clamp_to_edge, filter::linear);

// STAGE 0: rggb = q * gains + black_levels_times_gains.
static inline float4 normalizeQuad(uint4 raw, SRSRFrameNormalization n) {
    const float4 gains = float4(n.gain0, n.gain1, n.gain2, n.gain3);
    const float4 blacks = float4(n.blackTimesGain0, n.blackTimesGain1,
                                 n.blackTimesGain2, n.blackTimesGain3);
    return float4(raw) * gains + blacks;
}

// S1 YCoCg over (R, G1+G2, B): greens are summed, so the green weights halve.
static inline float3 RGBToYCoCg(float3 rgb) {
    return float3(rgb.x *  0.25f + rgb.y * 0.25f + rgb.z *  0.25f,
                  rgb.x * -0.25f + rgb.y * 0.25f + rgb.z * -0.25f,
                  rgb.x *  0.5f                 + rgb.z * -0.5f);
}

static inline float WeightGreen(int x) {
    if (x == 0) return 0.5f;
    if (abs(x) == 1) return 0.25f;
    return 0.0f;
}

static inline float3 RotateCovariance(float3 cov_m) {
    const float c0 = 0.5f * (cov_m.x + cov_m.y);
    const float c1 = 0.5f * (cov_m.y - cov_m.x);
    return float3(c0 + cov_m.z, c0 - cov_m.z, c1);
}

static inline void AnalyzeStructureTensor(float3 cov_m, float variance_noise,
                                          thread float2 &eigenvector1,
                                          thread float2 &eigenvector2,
                                          thread float &strength,
                                          thread float &coherence) {
    const float sum_xx = cov_m.x, sum_yy = cov_m.y, sum_xy = cov_m.z;
    const float trace = sum_xx + sum_yy;
    const float diff = sum_xx - sum_yy;
    const float sqrt_d = sqrt(max(diff * diff + 4.0f * sum_xy * sum_xy, 0.0f));
    float eigenvalue1 = (trace + sqrt_d) * 0.5f;
    const float eigenvalue2 = (trace - sqrt_d) * 0.5f;

    eigenvector1 = float2(1.0f, 0.0f);
    eigenvector2 = float2(0.0f, 1.0f);
    if (abs(sum_xy) > 0.0001f) {
        eigenvector1 = normalize(float2(sum_xy, eigenvalue1 - sum_xx)) * -sign(sum_xy);
        eigenvector2 = float2(-eigenvector1.y, eigenvector1.x);
    } else if (sum_xx < sum_yy) {
        eigenvector1 = float2(0.0f, 1.0f);
        eigenvector2 = float2(1.0f, 0.0f);
    }
    const float singular_value1 = sqrt(max(eigenvalue1, 0.0f));
    const float singular_value2 = sqrt(max(eigenvalue2, 0.0f));
    // Wiener-denoise the strength (variance_noise > 0 always: LUT floor).
    eigenvalue1 *= eigenvalue1 / (eigenvalue1 + variance_noise);
    const float singular_value1_denoised = sqrt(max(eigenvalue1, 0.0f));
    coherence = (singular_value1 - singular_value2) /
                (singular_value1 + singular_value2 + 1.0e-6f);
    strength = singular_value1_denoised;
}

static inline float ComputeGradientBlurring(float strength, float green_standard_deviation,
                                            float gradient_clip, float gradient_transition) {
    const float dominant_feature = max(strength, green_standard_deviation) - gradient_clip;
    return clamp(1.0f - dominant_feature * gradient_transition, 0.0f, 1.0f);
}

static inline float3 ConstructGaussianCovariance(float2 eigenvector1, float2 eigenvector2,
                                                 float strength, float coherence,
                                                 float green_differences,
                                                 constant SRSRGuideUniforms &u) {
    const float gradient_blurring =
        ComputeGradientBlurring(strength, green_differences, u.gradientClip, u.gradientTransition);
    const float anisotropic_shrinking =
        mix(u.baseSigma, u.shrinkedSigma, min(coherence, strength * 5.0f));
    const float singular_value1 = mix(anisotropic_shrinking, u.blurredSigma, gradient_blurring);
    const float singular_value2 = mix(mix(u.baseSigma, u.stretchedSigma, coherence),
                                      u.blurredSigma, gradient_blurring);
    const float new_eigenvalue1 = singular_value1 * singular_value1;
    const float new_eigenvalue2 = singular_value2 * singular_value2;
    // GCam ships rr = transpose(R) * D * R with R's columns = (e1, e2). Not the
    // textbook R D R^T -- see memory note gcam-cfa-stencil-and-rr01.
    const float2x2 R = float2x2(eigenvector1, eigenvector2);
    const float2x2 D = float2x2(float2(new_eigenvalue1, 0.0f), float2(0.0f, new_eigenvalue2));
    const float2x2 rr = transpose(R) * D * R;
    return float3(rr[0].x, rr[1].y, rr[0].y);
}

// SABRE 2b: GPU Gems 2 cubic B-spline from four bilinear taps. texelSize = (W, H, 1/W, 1/H).
static inline float4 SampleBicubic(texture2d<float> tex, float2 uv, float4 texelSize) {
    uv = uv * float2(texelSize.x, texelSize.y) + 0.5f;
    const float2 iuv = floor(uv);
    const float2 a = fract(uv);
    const float2 w0 = (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
    const float2 w1 = (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
    const float2 w2 = (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
    const float2 w3 = (1.0f / 6.0f) * (a * a * a);
    const float2 g0 = w0 + w1;
    const float2 g1 = w2 + w3;
    const float2 h0 = -1.0f + w1 / g0;
    const float2 h1 = 1.0f + w3 / g1;
    const float2 tz = float2(texelSize.z, texelSize.w);
    const float2 p00 = (iuv + h0 - 0.5f) * tz;
    const float2 p11 = (iuv + h1 - 0.5f) * tz;
    const float2 p0 = float2(p00.x, p00.y), p1 = float2(p11.x, p00.y);
    const float2 p2 = float2(p00.x, p11.y), p3 = float2(p11.x, p11.y);
    return g0.y * (g0.x * tex.sample(kLinearClamp, p0) + g1.x * tex.sample(kLinearClamp, p1)) +
           g1.y * (g0.x * tex.sample(kLinearClamp, p2) + g1.x * tex.sample(kLinearClamp, p3));
}

// Noise LUT: column i <-> luma i/9, linear between columns (SABRE scale/bias at W=10).
static inline float4 sampleNoiseLUTRow0(constant SRSRNoiseLUT &lut, float luma) {
    const float t = clamp(luma, 0.0f, 1.0f) * float(kSRSRNoiseLUTColumns - 1);
    const int i0 = int(floor(t));
    const int i1 = min(i0 + 1, kSRSRNoiseLUTColumns - 1);
    const float f = t - float(i0);
    const float4 a = float4(lut.row0[i0][0], lut.row0[i0][1], lut.row0[i0][2], lut.row0[i0][3]);
    const float4 b = float4(lut.row0[i1][0], lut.row0[i1][1], lut.row0[i1][2], lut.row0[i1][3]);
    return mix(a, b, f);
}

static inline float4 sampleNoiseLUTRow1(constant SRSRNoiseLUT &lut, float luma) {
    const float t = clamp(luma, 0.0f, 1.0f) * float(kSRSRNoiseLUTColumns - 1);
    const int i0 = int(floor(t));
    const int i1 = min(i0 + 1, kSRSRNoiseLUTColumns - 1);
    const float f = t - float(i0);
    const float4 a = float4(lut.row1[i0][0], lut.row1[i0][1], lut.row1[i0][2], lut.row1[i0][3]);
    const float4 b = float4(lut.row1[i1][0], lut.row1[i1][1], lut.row1[i1][2], lut.row1[i1][3]);
    return mix(a, b, f);
}

// SABRE 4b funky YUV.
static inline float3 ToGcamYUV(float3 c) {
    return float3( 0.2126390f * c.x + 0.7151688f * c.y + 0.0721923f * c.z,
                  -0.7873610f * c.x + 0.7151688f * c.y + 0.0721923f * c.z,
                   0.2126390f * c.x + 0.7151688f * c.y - 0.9278077f * c.z);
}
static inline float3 FromGcamYUV(float3 c) {
    return float3(c.x - c.y,
                  c.x + 0.297327f * c.y + 0.100944f * c.z,
                  c.x - c.z);
}

static inline float DenoiseWeight(int x) {
    if (x == 0) return 0.5f;
    if (abs(x) == 1) return 0.25f;
    return 0.0f;
}

// Halide BoundaryConditions::repeat_edge == clamp of the integer coordinate.
static inline int2 repeatEdge(int2 p, int2 extent) {
    return int2(clamp(p.x, 0, extent.x - 1), clamp(p.y, 0, extent.y - 1));
}

static inline float readClamped(texture2d<float, access::read> tex, int2 p, int2 extent) {
    return tex.read(uint2(repeatEdge(p, extent))).x;
}

// Half-sample symmetric mirror at the image border: the denoiser's
// "2 * border - uv" on texel-centred normalized coordinates, in integers.
static inline int mirrorIndex(int p, int n) {
    if (p < 0) p = -p - 1;
    if (p >= n) p = 2 * n - p - 1;
    return clamp(p, 0, n - 1);
}

} // namespace srsr

using namespace srsr;

//===----------------------------------------------------------------------===//
// PREPARE: loader raw (float, >= 0, pre-whitened, exposure-matched) -> quad
// codes in canonical R G1 G2 B, plus the S2 quad mean as the alignment grey.
//===----------------------------------------------------------------------===//
static inline float srsrRawAt(device const float* src, int x, int y, constant SRSRPrepareUniforms &u) {
    const int cx = clamp(x, 0, u.rawWidth - 1);
    const int cy = clamp(y, 0, u.rawHeight - 1);
    return src[uint(cy) * uint(u.rawWidth) + uint(cx)];
}

kernel void srsr_prepare(device const float* src                   [[buffer(1)]],
                         texture2d<uint,  access::write> quad      [[texture(1)]],
                         texture2d<float, access::write> grey      [[texture(2)]],
                         constant SRSRPrepareUniforms &u           [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.quadWidth) || gid.y >= uint(u.quadHeight)) return;
    const int x0 = u.cropOriginX + 2 * int(gid.x);
    const int y0 = u.cropOriginY + 2 * int(gid.y);
    float v[4];
    v[0] = srsrRawAt(src, x0,     y0,     u);
    v[1] = srsrRawAt(src, x0 + 1, y0,     u);
    v[2] = srsrRawAt(src, x0,     y0 + 1, u);
    v[3] = srsrRawAt(src, x0 + 1, y0 + 1, u);
    uint q[4];
    for (int i = 0; i < 4; ++i)
        q[i] = uint(clamp(v[i] * u.quadScale + 0.5f, 0.0f, 65535.0f));
    const uint4 rggb = uint4(q[u.perm0], q[u.perm1], q[u.perm2], q[u.perm3]);
    quad.write(rggb, gid);
    const float4 s2 = normalizeQuad(rggb, u.norm);
    grey.write(float4(dot(s2, float4(0.25f))), gid);
}

//===----------------------------------------------------------------------===//
// UNBLOCKER (SABRE 2f) -- RECONSTRUCTED (Documents/srsr D5). Green two-scale
// variance loss per 8x8 raw block. Off by default: on the measured device the
// real map was zero.
//===----------------------------------------------------------------------===//
kernel void srsr_unblocker(texture2d<uint,  access::read>  quad  [[texture(0)]],
                           texture2d<float, access::write> outU  [[texture(1)]],
                           constant SRSRUnblockerUniforms &u     [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.outWidth) || gid.y >= uint(u.outHeight)) return;
    const int2 extent = int2(u.quadWidth, u.quadHeight);
    const int q0x = 4 * int(gid.x);
    const int q0y = 4 * int(gid.y);
    float fine[16];
    float sum1 = 0.0f, sumSq1 = 0.0f;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            const uint4 rggb = quad.read(uint2(repeatEdge(int2(q0x + i, q0y + j), extent)));
            const float4 s2 = min(normalizeQuad(rggb, u.norm), float4(u.maxWhite));
            const float green = 0.5f * (s2.y + s2.z);
            fine[j * 4 + i] = green;
            sum1 += green;
            sumSq1 += green * green;
        }
    }
    const float mean1 = sum1 * (1.0f / 16.0f);
    const float v1 = max(sumSq1 * (1.0f / 16.0f) - mean1 * mean1, 0.0f);
    float sum2 = 0.0f, sumSq2 = 0.0f;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            const float coarse = 0.25f * (fine[(2 * j) * 4 + 2 * i] + fine[(2 * j) * 4 + 2 * i + 1] +
                                          fine[(2 * j + 1) * 4 + 2 * i] + fine[(2 * j + 1) * 4 + 2 * i + 1]);
            sum2 += coarse;
            sumSq2 += coarse * coarse;
        }
    }
    const float mean2 = sum2 * 0.25f;
    const float v2 = max(sumSq2 * 0.25f - mean2 * mean2, 0.0f);
    const float loss = max(sqrt(v1) - sqrt(v2), 0.0f);
    outU.write(float4(saturate((loss - u.bias) / u.varianceScale)), gid);
}

//===----------------------------------------------------------------------===//
// ALIGNMENT (Halide reference, Downloads/alignment). Reductions stay serial and
// row-major so every sum is the reference's (Documents/srsr D7).
//===----------------------------------------------------------------------===//
kernel void srsr_clear_flow(texture2d<float, access::write> outFlow  [[texture(0)]],
                            constant SRSRFlowMedianUniforms &u       [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.tilesX) || gid.y >= uint(u.tilesY)) return;
    outFlow.write(float4(0.0f), gid);
}

// One 2x step of [1 2 1] x [1 2 1] / 16, repeat-edge; extent ceil(in / 2).
kernel void srsr_downsample(texture2d<float, access::read>  input   [[texture(0)]],
                            texture2d<float, access::write> output  [[texture(1)]],
                            constant SRSRDownsampleUniforms &u      [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.outWidth) || gid.y >= uint(u.outHeight)) return;
    const int2 extent = int2(u.inWidth, u.inHeight);
    const int cx = 2 * int(gid.x);
    const int cy = 2 * int(gid.y);
    float acc = 0.0f;
    const float wx[3] = { 1.0f, 2.0f, 1.0f };
    for (int j = -1; j <= 1; ++j) {
        float row = 0.0f;
        for (int i = -1; i <= 1; ++i)
            row += wx[i + 1] * readClamped(input, int2(cx + i, cy + j), extent);
        acc += wx[j + 1] * row;
    }
    output.write(float4(acc * (1.0f / 16.0f)), gid);
}

kernel void srsr_gradients(texture2d<float, access::read>  input  [[texture(0)]],
                           texture2d<float, access::write> gx     [[texture(1)]],
                           texture2d<float, access::write> gy     [[texture(2)]],
                           constant SRSRGradientUniforms &u       [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.width) || gid.y >= uint(u.height)) return;
    const int2 extent = int2(u.width, u.height);
    const int x = int(gid.x), y = int(gid.y);
    const float dx = (readClamped(input, int2(x - 2, y), extent)
                      - 8.0f * readClamped(input, int2(x - 1, y), extent)
                      + 8.0f * readClamped(input, int2(x + 1, y), extent)
                      - readClamped(input, int2(x + 2, y), extent)) / 12.0f;
    const float dy = (readClamped(input, int2(x, y - 2), extent)
                      - 8.0f * readClamped(input, int2(x, y - 1), extent)
                      + 8.0f * readClamped(input, int2(x, y + 1), extent)
                      - readClamped(input, int2(x, y + 2), extent)) / 12.0f;
    gx.write(float4(dx), gid);
    gy.write(float4(dy), gid);
}

kernel void srsr_gradient_tile_sums(texture2d<float, access::read>  gx    [[texture(0)]],
                                    texture2d<float, access::read>  gy    [[texture(1)]],
                                    texture2d<float, access::write> hess  [[texture(2)]],
                                    constant SRSRGradientUniforms &u      [[buffer(0)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.tilesX) || gid.y >= uint(u.tilesY)) return;
    const int ts = u.tileSize;
    const int st = u.tileStride > 0 ? u.tileStride : ts;
    const int originX = int(gid.x) * st;
    const int originY = int(gid.y) * st;
    float sum_gxx = 0.0f, sum_gxy = 0.0f, sum_gyy = 0.0f;
    for (int ry = 0; ry < ts; ++ry) {
        for (int rx = 0; rx < ts; ++rx) {
            const uint2 p = uint2(uint(originX + rx), uint(originY + ry));
            const float a = gx.read(p).x;
            const float b = gy.read(p).x;
            sum_gxx += a * a;
            sum_gxy += a * b;
            sum_gyy += b * b;
        }
    }
    hess.write(float4(sum_gxx, sum_gxy, sum_gyy, 0.0f), gid);
}

// Cost of one block-matching candidate over one tile, row-major (shared by the
// threadgroup kernel and its serial twin, so both sum identically).
static inline float srsrBlockCost(texture2d<float, access::read> moving,
                                  int originX, int originY, int ts, int2 extent,
                                  int ox, int oy, int useL2,
                                  thread const float* refTile) {
    float cost = 0.0f;
    for (int py = 0; py < ts; ++py) {
        for (int px = 0; px < ts; ++px) {
            const float ref = refTile[py * ts + px];
            const float mov = readClamped(moving, int2(originX + px + ox, originY + py + oy), extent);
            const float difference = ref - mov;
            cost += (useL2 != 0) ? (difference * difference) : abs(difference);
        }
    }
    return cost;
}

#ifdef __METAL_VERSION__
// Metal threadgroup memory is a different address space; the reference tile is
// copied into thread memory per candidate row instead of casting pointers.
static inline float srsrBlockCostTG(texture2d<float, access::read> moving,
                                    int originX, int originY, int ts, int2 extent,
                                    int ox, int oy, int useL2,
                                    threadgroup const float* refTile) {
    float cost = 0.0f;
    for (int py = 0; py < ts; ++py) {
        for (int px = 0; px < ts; ++px) {
            const float ref = refTile[py * ts + px];
            const float mov = readClamped(moving, int2(originX + px + ox, originY + py + oy), extent);
            const float difference = ref - mov;
            cost += (useL2 != 0) ? (difference * difference) : abs(difference);
        }
    }
    return cost;
}

// One threadgroup per tile, one thread per candidate; thread 0 scans in the
// Halide order (search x innermost from (-r, -r)) with a strict compare.
kernel void srsr_block_matching(texture2d<float, access::read>  reference  [[texture(0)]],
                                texture2d<float, access::read>  moving     [[texture(1)]],
                                texture2d<float, access::read>  initial    [[texture(2)]],
                                texture2d<float, access::write> outFlow    [[texture(3)]],
                                constant SRSRBlockMatchingUniforms &u      [[buffer(0)]],
                                threadgroup float *costs                   [[threadgroup(0)]],
                                threadgroup float *referenceTile           [[threadgroup(1)]],
                                uint2 tileID      [[threadgroup_position_in_grid]],
                                uint  lane        [[thread_index_in_threadgroup]],
                                uint2 tgSize      [[threads_per_threadgroup]]) {
    const int ts = u.tileSize;
    const int radius = u.searchRadius;
    const int span = 2 * radius + 1;
    const int2 extent = int2(u.width, u.height);
    const float2 seed = xy(initial.read(tileID));
    const int centerX = int(rint(seed.x));
    const int centerY = int(rint(seed.y));
    const int st = u.tileStride > 0 ? u.tileStride : ts;
    const int originX = int(tileID.x) * st;
    const int originY = int(tileID.y) * st;
    const int threadCount = int(tgSize.x * tgSize.y);
    for (int i = int(lane); i < ts * ts; i += threadCount)
        referenceTile[i] = reference.read(uint2(uint(originX + i % ts), uint(originY + i / ts))).x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(lane) < u.candidateCount) {
        const int k = int(lane);
        costs[k] = srsrBlockCostTG(moving, originX, originY, ts, extent,
                                   centerX + (k % span) - radius, centerY + (k / span) - radius,
                                   u.useL2, referenceTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        int bestX = 0, bestY = 0;
        float bestCost = INFINITY;
        for (int k = 0; k < u.candidateCount; ++k) {
            const float candidate = costs[k];
            if (candidate < bestCost) {
                bestCost = candidate;
                bestX = (k % span) - radius;
                bestY = (k / span) - radius;
            }
        }
        outFlow.write(float4(float(centerX + bestX), float(centerY + bestY), 0.0f, 0.0f), tileID);
    }
}
#else
// Serial twin for the CPU harness: same per-candidate sums, same scan order.
static inline void srsr_block_matching_tile(texture2d<float, access::read>  reference,
                                            texture2d<float, access::read>  moving,
                                            texture2d<float, access::read>  initial,
                                            texture2d<float, access::write> outFlow,
                                            constant SRSRBlockMatchingUniforms &u,
                                            uint2 tileID, float* refTile) {
    const int ts = u.tileSize;
    const int radius = u.searchRadius;
    const int span = 2 * radius + 1;
    const int2 extent = int2(u.width, u.height);
    const float2 seed = xy(initial.read(tileID));
    const int centerX = int(rint(seed.x));
    const int centerY = int(rint(seed.y));
    const int st = u.tileStride > 0 ? u.tileStride : ts;
    const int originX = int(tileID.x) * st;
    const int originY = int(tileID.y) * st;
    for (int i = 0; i < ts * ts; ++i)
        refTile[i] = reference.read(uint2(uint(originX + i % ts), uint(originY + i / ts))).x;
    int bestX = 0, bestY = 0;
    float bestCost = INFINITY;
    for (int k = 0; k < u.candidateCount; ++k) {
        const float candidate = srsrBlockCost(moving, originX, originY, ts, extent,
                                              centerX + (k % span) - radius, centerY + (k / span) - radius,
                                              u.useL2, refTile);
        if (candidate < bestCost) {
            bestCost = candidate;
            bestX = (k % span) - radius;
            bestY = (k / span) - radius;
        }
    }
    outFlow.write(float4(float(centerX + bestX), float(centerY + bestY), 0.0f, 0.0f), tileID);
}
#endif

// One inverse-compositional iteration per tile: bilinear warp, determinant +
// FLT_EPSILON, update clamped to [-1, 1] per component.
kernel void srsr_ica_iteration(texture2d<float, access::read>  reference  [[texture(0)]],
                               texture2d<float, access::read>  moving     [[texture(1)]],
                               texture2d<float, access::read>  gx         [[texture(2)]],
                               texture2d<float, access::read>  gy         [[texture(3)]],
                               texture2d<float, access::read>  hess       [[texture(4)]],
                               texture2d<float, access::read>  initial    [[texture(5)]],
                               texture2d<float, access::write> outFlow    [[texture(6)]],
                               constant SRSRICAUniforms &u                [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.tilesX) || gid.y >= uint(u.tilesY)) return;
    const int ts = u.tileSize;
    const int2 extent = int2(u.width, u.height);
    const float2 flow = xy(initial.read(gid));
    const int st = u.tileStride > 0 ? u.tileStride : ts;
    const int originX = int(gid.x) * st;
    const int originY = int(gid.y) * st;
    float rhsX = 0.0f, rhsY = 0.0f;
    for (int py = 0; py < ts; ++py) {
        for (int px = 0; px < ts; ++px) {
            const int x = originX + px;
            const int y = originY + py;
            const float wx = float(x) + flow.x;
            const float wy = float(y) + flow.y;
            const int ix = int(floor(wx));
            const int iy = int(floor(wy));
            const float fx = wx - float(ix);
            const float fy = wy - float(iy);
            const float m00 = readClamped(moving, int2(ix,     iy),     extent);
            const float m10 = readClamped(moving, int2(ix + 1, iy),     extent);
            const float m01 = readClamped(moving, int2(ix,     iy + 1), extent);
            const float m11 = readClamped(moving, int2(ix + 1, iy + 1), extent);
            const float top = m00 + fx * (m10 - m00);
            const float bottom = m01 + fx * (m11 - m01);
            const float warped = top + fy * (bottom - top);
            const uint2 p = uint2(uint(x), uint(y));
            const float residual = reference.read(p).x - warped;
            rhsX += gx.read(p).x * residual;
            rhsY += gy.read(p).x * residual;
        }
    }
    const float4 h = hess.read(gid);
    const float a = h.x, b = h.y, c = h.z;
    const float determinant = a * c - b * b + 1.19209290e-07f;
    const float updateX = (c * rhsX - b * rhsY) / determinant;
    const float updateY = (a * rhsY - b * rhsX) / determinant;
    outFlow.write(float4(flow.x + clamp(updateX, -1.0f, 1.0f),
                         flow.y + clamp(updateY, -1.0f, 1.0f), 0.0f, 0.0f), gid);
}

// Flow transfer to a finer / denser grid: nearest, horizontal and vertical
// candidates scored by integer-sampled L1; ties prefer nearest, then h, then v.
kernel void srsr_flow_transfer(texture2d<float, access::read>  sourceFlow  [[texture(0)]],
                               texture2d<float, access::read>  reference   [[texture(1)]],
                               texture2d<float, access::read>  moving      [[texture(2)]],
                               texture2d<float, access::write> outFlow     [[texture(3)]],
                               constant SRSRFlowTransferUniforms &u        [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.targetTilesX) || gid.y >= uint(u.targetTilesY)) return;
    const int ts = u.targetTileSize;
    const int2 imageExtent = int2(u.width, u.height);
    const int2 sourceExtent = int2(u.sourceTilesX, u.sourceTilesY);
    const float sourceSpacingPx = u.gridScaleRatio * float(u.targetStride);
    const float uu = (float(int(gid.x) * u.targetStride) - u.offsetXPx) / sourceSpacingPx;
    const float vv = (float(int(gid.y) * u.targetStride) - u.offsetYPx) / sourceSpacingPx;
    const int left = int(floor(uu));
    const int top = int(floor(vv));
    const int nearX = int(floor(uu + 0.5f));
    const int nearY = int(floor(vv + 0.5f));
    const int otherX = (nearX == left) ? (left + 1) : left;
    const int otherY = (nearY == top) ? (top + 1) : top;
    const int originX = int(gid.x) * u.targetStride;
    const int originY = int(gid.y) * u.targetStride;
    float2 candidates[3];
    int sampleDX[3];
    int sampleDY[3];
    float costs[3] = { 0.0f, 0.0f, 0.0f };
    for (int c = 0; c < 3; ++c) {
        const int cx = (c == 1) ? otherX : nearX;
        const int cy = (c == 2) ? otherY : nearY;
        const float2 raw = xy(sourceFlow.read(uint2(repeatEdge(int2(cx, cy), sourceExtent))));
        candidates[c] = raw * u.flowScale;
        sampleDX[c] = int(rint(candidates[c].x));
        sampleDY[c] = int(rint(candidates[c].y));
    }
    for (int py = 0; py < ts; ++py) {
        for (int px = 0; px < ts; ++px) {
            const int x = originX + px;
            const int y = originY + py;
            const float ref = readClamped(reference, int2(x, y), imageExtent);
            for (int c = 0; c < 3; ++c) {
                const float mov = readClamped(moving, int2(x + sampleDX[c], y + sampleDY[c]), imageExtent);
                costs[c] += abs(ref - mov);
            }
        }
    }
    float bestCost = INFINITY;
    float2 bestFlow = float2(0.0f);
    for (int c = 0; c < 3; ++c) {
        if (costs[c] < bestCost) {
            bestCost = costs[c];
            bestFlow = candidates[c];
        }
    }
    outFlow.write(float4(bestFlow.x, bestFlow.y, 0.0f, 0.0f), gid);
}

static inline float srsrMedian9(thread float *values) {
    for (int i = 1; i < 9; ++i) {
        for (int j = i; j > 0; --j) {
            const float lo = min(values[j - 1], values[j]);
            const float hi = max(values[j - 1], values[j]);
            values[j - 1] = lo;
            values[j] = hi;
        }
    }
    return values[4];
}

kernel void srsr_flow_median(texture2d<float, access::read>  sourceFlow  [[texture(0)]],
                             texture2d<float, access::write> outFlow     [[texture(1)]],
                             constant SRSRFlowMedianUniforms &u          [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.tilesX) || gid.y >= uint(u.tilesY)) return;
    const int2 extent = int2(u.tilesX, u.tilesY);
    float xs[9];
    float ys[9];
    int index = 0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            const float4 v = sourceFlow.read(uint2(repeatEdge(int2(int(gid.x) + i, int(gid.y) + j), extent)));
            xs[index] = v.x;
            ys[index] = v.y;
            ++index;
        }
    }
    outFlow.write(float4(srsrMedian9(xs), srsrMedian9(ys), 0.0f, 0.0f), gid);
}

// Tile flow in grey (quad) pixels -> SABRE flow texture: .xy normalized UV,
// .z the length of the 3x3 min-max extent (local motion variation).
kernel void srsr_flow_pack(texture2d<float, access::read>  sourceFlow  [[texture(0)]],
                           texture2d<float, access::write> outFlow     [[texture(1)]],
                           constant SRSRFlowPackUniforms &u            [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.tilesX) || gid.y >= uint(u.tilesY)) return;
    const int2 extent = int2(u.tilesX, u.tilesY);
    const float2 here = xy(sourceFlow.read(gid));
    float2 lo = here;
    float2 hi = here;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            const float2 v = xy(sourceFlow.read(uint2(repeatEdge(int2(int(gid.x) + i, int(gid.y) + j), extent))));
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    const float2 inv = float2(u.invGreyWidth, u.invGreyHeight);
    const float2 uvFlow = here * inv;
    outFlow.write(float4(uvFlow.x, uvFlow.y, length((hi - lo) * inv), 0.0f), gid);
}

//===----------------------------------------------------------------------===//
// STAGE 1 -- GUIDE: per quad, the S1 (sqrt, YCoCg) reference colour + std_dev,
// and the merge kernel's inverse covariance {cxx, cyy, cxy}.
//===----------------------------------------------------------------------===//
kernel void srsr_guide(texture2d<uint,  access::read>  quad   [[texture(0)]],
                       texture2d<float, access::write> guide  [[texture(1)]],
                       texture2d<float, access::write> cov    [[texture(2)]],
                       constant SRSRGuideUniforms &u          [[buffer(0)]],
                       constant SRSRNoiseLUT &lut             [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.quadWidth) || gid.y >= uint(u.quadHeight)) return;
    const int2 extent = int2(u.quadWidth, u.quadHeight);
    float3 m1_rgb = float3(0.0f);
    float3 m0_rgb = float3(0.0f);
    float3 average_rgb = float3(0.0f);
    float sums[9];
    float green_differences = 0.0f;
    for (int y = -1; y <= 1; y += 1) {
        for (int x = -1; x <= 1; x += 1) {
            const uint4 raw = quad.read(uint2(repeatEdge(int2(int(gid.x) + x, int(gid.y) + y), extent)));
            float4 rggb = normalizeQuad(raw, u.norm);
            rggb = sqrt(max(float4(0.0f), rggb));
            const float3 rgb = float3(rggb.x, rggb.y + rggb.z, rggb.w);
            sums[(y + 1) * 3 + x + 1] = RGBToYCoCg(rgb).x;
            const float w = WeightGreen(x) * WeightGreen(y);
            average_rgb += rgb * w;
            m0_rgb += rgb;
            m1_rgb += rgb * rgb;
            green_differences += abs(rggb.y - rggb.z) * w;
        }
    }
    m0_rgb /= 9.0f;
    m1_rgb /= 9.0f;
    const float3 mean = RGBToYCoCg(average_rgb);
    const float3 m0_rgb_squared = m0_rgb * m0_rgb;
    const float3 mean_luma = float3(0.5f * average_rgb.y, 0.0f, 0.0f);
    float std_dev = sqrt(max(dot(m1_rgb - m0_rgb_squared, float3(1.0f / 9.0f)), 0.0f));
    std_dev = max(std_dev, green_differences);
    const float sub_quad_variance_impact = min(u.greenDifferenceImpact * green_differences, 1.0f);
    const float3 reference_color = mix(mean, mean_luma, sub_quad_variance_impact);
    guide.write(float4(reference_color, std_dev), gid);

    const float4 noise_params = sampleNoiseLUTRow1(lut, mean.x);
    green_differences *= green_differences / (green_differences + noise_params.x);

    float dxx = 0.0f, dyy = 0.0f, dxy = 0.0f;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            const float dx = sums[(y + 1) * 3 + (x + 1)] - sums[y * 3 + x];
            const float dy = sums[y * 3 + (x + 1)] - sums[(y + 1) * 3 + x];
            dxx += dx * dx;
            dyy += dy * dy;
            dxy += dx * dy;
        }
    }
    float3 covariance = float3(dxx, dyy, dxy) / 8.0f;
    covariance = RotateCovariance(covariance);
    float2 eigenvector1, eigenvector2;
    float strength, coherence;
    AnalyzeStructureTensor(covariance, noise_params.y, eigenvector1, eigenvector2, strength, coherence);
    const float3 gauss_covariance = ConstructGaussianCovariance(
        eigenvector1, eigenvector2, strength, coherence, green_differences, u);
    cov.write(float4(gauss_covariance, 0.0f), gid);
}

//===----------------------------------------------------------------------===//
// STAGE 2 -- REJECTION (SABRE 2c ComputeFrameWeight), written inverted.
//===----------------------------------------------------------------------===//
static inline float4 sampleFlowBilinear(texture2d<float, access::read> flow, float2 uv, int2 tiles) {
    const float2 t = uv * float2(tiles) - 0.5f;
    const float2 base = floor(t);
    const float2 f = t - base;
    const int2 i0 = int2(int(base.x), int(base.y));
    const float4 c00 = flow.read(uint2(repeatEdge(i0 + int2(0, 0), tiles)));
    const float4 c10 = flow.read(uint2(repeatEdge(i0 + int2(1, 0), tiles)));
    const float4 c01 = flow.read(uint2(repeatEdge(i0 + int2(0, 1), tiles)));
    const float4 c11 = flow.read(uint2(repeatEdge(i0 + int2(1, 1), tiles)));
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

static inline float ComputeFrameWeight(texture2d<float> baseGuide, texture2d<float> altGuide,
                                       float2 reference_uv, float2 current_frame_uv,
                                       bool rejection_motion_prior_region_detected,
                                       float4 texSize,
                                       constant SRSRRejectionUniforms &u,
                                       constant SRSRNoiseLUT &lut) {
    const float4 reference_frame_values = SampleBicubic(baseGuide, reference_uv, texSize);
    const float4 current_frame = SampleBicubic(altGuide, current_frame_uv, texSize);
    float4 noise_params = sampleNoiseLUTRow0(lut, reference_frame_values.x);
    float std_dev = min(reference_frame_values.w, current_frame.w);
    const float min_std_dev = noise_params.w;
    float robustness_boost = 1.0f;
    if (std_dev > u.motionRobustnessBoostStdDevThreshold * min_std_dev &&
        rejection_motion_prior_region_detected)
        robustness_boost = u.extraMotionRobustnessBoost;
    std_dev = max(std_dev, min_std_dev);
    const float3 n = 2.0f * xyz(noise_params);       // Var(a - b) = 2 Var
    const float3 d = xyz(current_frame) - xyz(reference_frame_values);
    float3 difference_squared = d * d;
    difference_squared *= difference_squared / (difference_squared + n);   // Wiener
    float distance = difference_squared.x + difference_squared.y + difference_squared.z;
    distance = distance / (std_dev * std_dev);
    distance *= u.distanceMultiplier * robustness_boost;
    return exp2(min(-distance + u.rejectionThreshold, 0.0f));
}

kernel void srsr_rejection(texture2d<float> baseGuide                     [[texture(0)]],
                           texture2d<float> altGuide                      [[texture(1)]],
                           texture2d<float, access::read> altFlow         [[texture(2)]],
                           texture2d<float> unblockerTexture              [[texture(3)]],
                           texture2d<float, access::write> outRejection   [[texture(4)]],
                           constant SRSRRejectionUniforms &u              [[buffer(0)]],
                           constant SRSRNoiseLUT &lut                     [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.quadWidth) || gid.y >= uint(u.quadHeight)) return;
    const float4 texSize = float4(float(u.quadWidth), float(u.quadHeight),
                                  1.0f / float(u.quadWidth), 1.0f / float(u.quadHeight));
    const float2 tex_coord = (float2(gid) + 0.5f) * float2(texSize.z, texSize.w);
    const float2 flow_tc = tex_coord * float2(u.flowScaleX, u.flowScaleY) + float2(u.flowOffsetX, u.flowOffsetY);
    const float4 flow = sampleFlowBilinear(altFlow, flow_tc, int2(u.flowTilesX, u.flowTilesY));
    float2 warped_uv = tex_coord + xy(flow);
    warped_uv.x = warped_uv.x > 1.0f ? 2.0f - warped_uv.x : warped_uv.x;
    warped_uv.y = warped_uv.y > 1.0f ? 2.0f - warped_uv.y : warped_uv.y;
    warped_uv.x = warped_uv.x < 0.0f ? -warped_uv.x : warped_uv.x;
    warped_uv.y = warped_uv.y < 0.0f ? -warped_uv.y : warped_uv.y;
    float unblocker = 0.0f;
    if (u.unblockerEnabled != 0) {
        const float2 unblocker_tc = tex_coord * float2(u.unblockerScaleX, u.unblockerScaleY);
        unblocker = unblockerTexture.sample(kLinearClamp, unblocker_tc).x;
    }
    const float local_flow_variation = flow.z;
    if (local_flow_variation < u.unblockerReductionThreshold) unblocker = 0.0f;
    const bool motion_prior = local_flow_variation > u.extraMotionRobustnessMotionThreshold;
    const float frame_weight = ComputeFrameWeight(baseGuide, altGuide, tex_coord, warped_uv,
                                                  motion_prior, texSize, u, lut);
    const float weight = min(1.0f - unblocker, frame_weight);
    outRejection.write(float4(saturate(1.0f - weight)), gid);
}

// SABRE 2d DILATE: unweighted 5x5 box of the inverted mask, separable; then
// keep = 1 - sum / 3 (three fully rejecting texels reject the centre).
kernel void srsr_dilate_horizontal(texture2d<float, access::read>  inRejection  [[texture(0)]],
                                   texture2d<float, access::write> outPartial   [[texture(1)]],
                                   constant SRSRRejectionUniforms &u            [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.quadWidth) || gid.y >= uint(u.quadHeight)) return;
    const int2 extent = int2(u.quadWidth, u.quadHeight);
    float sum = 0.0f;
    for (int i = -2; i <= 2; ++i)
        sum += readClamped(inRejection, int2(int(gid.x) + i, int(gid.y)), extent);
    outPartial.write(float4(sum), gid);
}

kernel void srsr_dilate_vertical(texture2d<float, access::read>  inPartial     [[texture(0)]],
                                 texture2d<float, access::write> outRejection  [[texture(1)]],
                                 constant SRSRRejectionUniforms &u             [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.quadWidth) || gid.y >= uint(u.quadHeight)) return;
    const int2 extent = int2(u.quadWidth, u.quadHeight);
    float rejection = 0.0f;
    for (int j = -2; j <= 2; ++j)
        rejection += readClamped(inPartial, int2(int(gid.x), int(gid.y) + j), extent);
    rejection /= 3.0f;
    outRejection.write(float4(saturate(1.0f - rejection)), gid);
}

// SABRE 2e COPY_MASK: sum of alt keep weights / 25, quantized once to R8.
kernel void srsr_accumulate_weights(array<texture2d<float, access::read>, kSRSRMaxFrames> dilated [[texture(0)]],
                                    texture2d<float, access::write> outWeights [[texture(16)]],
                                    constant SRSRAccumulateUniforms &u         [[buffer(0)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.quadWidth) || gid.y >= uint(u.quadHeight)) return;
    float total = 0.0f;
    for (int f = 0; f < u.frameCount; ++f)
        total += dilated[f].read(gid).x;
    outWeights.write(float4(saturate(total / u.accumulatedWeightScale)), gid);
}

// Robustness-mask export (Config::robustness_save_mask): the MEAN over the
// comparison frames of each frame's dilated keep, 8-bit, into a host-visible
// buffer -- the same quantity the standard pipeline saves as _robustness.pgm
// (white = every frame merged, black = every frame rejected).
kernel void srsr_mask_export(array<texture2d<float, access::read>, kSRSRMaxFrames> dilated [[texture(0)]],
                             device uchar* outMask                     [[buffer(1)]],
                             constant SRSRAccumulateUniforms &u        [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(u.quadWidth) || gid.y >= uint(u.quadHeight)) return;
    float total = 0.0f;
    for (int f = 0; f < u.frameCount; ++f)
        total += dilated[f].read(gid).x;
    const float mean = saturate(total / float(max(1, u.frameCount)));
    outMask[gid.y * uint(u.quadWidth) + gid.x] = uchar(mean * 255.0f + 0.5f);
}

//===----------------------------------------------------------------------===//
// STAGE 3 + 3.5 -- MERGE (fused over frames: alts in burst order, base last),
// dehomogenized in place, one output band. Distances are in Bayer pixels at any
// output scale: the output grid only decides WHERE the kernel is evaluated.
//===----------------------------------------------------------------------===//
static inline void ProcessBayerQuad(texture2d<uint, access::read> quadTexture,
                                    int2 quadIndex, int2 quadExtent,
                                    float2 subpixel_offset,
                                    float3 covariance, float normalization,
                                    SRSRFrameNormalization norm,
                                    float kEpsilon, float footprintLimit,
                                    thread float3 &accum_intensities,
                                    thread float3 &accum_weights) {
    const uint4 raw = quadTexture.read(uint2(repeatEdge(quadIndex, quadExtent)));
    const float4 rggb = normalizeQuad(raw, norm);
    const float2 difference_r = subpixel_offset;
    const float2 difference_g1 = subpixel_offset + float2(1.0f, 0.0f);
    const float2 difference_g2 = subpixel_offset + float2(0.0f, 1.0f);
    const float2 difference_b = subpixel_offset + float2(1.0f, 1.0f);
    const float distance_r = difference_r.x * difference_r.x * covariance.x
                           + difference_r.y * difference_r.y * covariance.y
                           + 2.0f * covariance.z * difference_r.x * difference_r.y;
    const float distance_g1 = difference_g1.x * difference_g1.x * covariance.x
                            + difference_g1.y * difference_g1.y * covariance.y
                            + 2.0f * covariance.z * difference_g1.x * difference_g1.y;
    const float distance_g2 = difference_g2.x * difference_g2.x * covariance.x
                            + difference_g2.y * difference_g2.y * covariance.y
                            + 2.0f * covariance.z * difference_g2.x * difference_g2.y;
    const float distance_b = difference_b.x * difference_b.x * covariance.x
                           + difference_b.y * difference_b.y * covariance.y
                           + 2.0f * covariance.z * difference_b.x * difference_b.y;
    float4 weight;
    weight.x = max(abs(difference_r.x), abs(difference_r.y)) <= footprintLimit
             ? exp2(-0.5f * distance_r) * normalization + kEpsilon : 0.0f;
    weight.y = max(abs(difference_g1.x), abs(difference_g1.y)) <= footprintLimit
             ? exp2(-0.5f * distance_g1) * normalization + kEpsilon : 0.0f;
    weight.z = max(abs(difference_g2.x), abs(difference_g2.y)) <= footprintLimit
             ? exp2(-0.5f * distance_g2) * normalization + kEpsilon : 0.0f;
    weight.w = max(abs(difference_b.x), abs(difference_b.y)) <= footprintLimit
             ? exp2(-0.5f * distance_b) * normalization + kEpsilon : 0.0f;
    accum_intensities += float3(weight.x * rggb.x, weight.y * rggb.y + weight.z * rggb.z, weight.w * rggb.w);
    accum_weights += float3(weight.x, weight.y + weight.z, weight.w);
}

// coord_scaled = uv * Bayer extent - 0.5; gather = floor((coord - 0.5) / 2) * 2;
// the 2x2 quads from there cover the 4x4 Bayer footprint (Documents/srsr checks).
static inline void SampleNeighborhoodRBF(texture2d<uint, access::read> quadTexture,
                                         int2 quadExtent, float2 sampl_uv, int2 bayerExtent,
                                         float3 covariance, float normalization,
                                         SRSRFrameNormalization norm,
                                         float kEpsilon, float footprintLimit,
                                         thread float3 &accum_intensities,
                                         thread float3 &accum_weights) {
    accum_intensities = float3(0.0f);
    accum_weights = float3(0.0f);
    const float2 coord_scaled = sampl_uv * float2(bayerExtent) - 0.5f;
    const float2 gather_coordinate = floor((coord_scaled - 0.5f) * 0.5f) * 2.0f;
    const float2 subpixel_offset = gather_coordinate - coord_scaled;
    const int2 baseQuad = int2(int(floor(gather_coordinate.x * 0.5f)), int(floor(gather_coordinate.y * 0.5f)));
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
            ProcessBayerQuad(quadTexture, baseQuad + int2(x, y), quadExtent,
                             subpixel_offset + float2(float(2 * x), float(2 * y)),
                             covariance, normalization, norm, kEpsilon, footprintLimit,
                             accum_intensities, accum_weights);
}

kernel void srsr_merge(texture2d<float, access::write> outMerged                    [[texture(0)]],
                       texture2d<float> accumulatedFrameWeights                     [[texture(1)]],
                       array<texture2d<uint,  access::read>, kSRSRMaxFrames> quads  [[texture(8)]],
                       array<texture2d<float>, kSRSRMaxFrames> covariances          [[texture(24)]],
                       array<texture2d<float, access::read>, kSRSRMaxFrames> flows  [[texture(40)]],
                       array<texture2d<float>, kSRSRMaxFrames> rejections           [[texture(56)]],
                       constant SRSRMergeUniforms &u                                [[buffer(0)]],
                       constant SRSRFrameNormalization *norms                       [[buffer(1)]],
                       constant SRSRFrameGeometry *geometry                         [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const int gy = int(gid.y) + u.bandY0;
    if (gid.x >= uint(u.outWidth) || int(gid.y) >= u.bandRows || gy >= u.outHeight) return;
    const int2 quadExtent = int2(u.quadWidth, u.quadHeight);
    const int2 bayerExtent = int2(u.bayerWidth, u.bayerHeight);
    const float2 zoomed_tc = (float2(float(gid.x), float(gy)) + 0.5f) /
                             float2(float(u.outWidth), float(u.outHeight));
    float3 accum_color = float3(0.0f);
    float3 accum_weight = float3(0.0f);
    for (int pass = 0; pass < 2; ++pass) {
        for (int f = 0; f < u.frameCount; ++f) {
            const bool isBase = (geometry[f].isBase != 0);
            if ((pass == 0) == isBase) continue;
            float frame_weight = 1.0f;
            float2 sampl_uv = zoomed_tc;
            float3 covariance;
            if (!isBase) {
                frame_weight = (u.applyRejectionWeight != 0)
                             ? rejections[f].sample(kLinearClamp, zoomed_tc).x : 1.0f;
                if (frame_weight <= 0.0f) continue;
                const float2 flow_tc = zoomed_tc * float2(geometry[f].flowScaleX, geometry[f].flowScaleY)
                                     + float2(geometry[f].flowOffsetX, geometry[f].flowOffsetY);
                const int2 flowExtent = int2(int(flows[f].get_width()), int(flows[f].get_height()));
                const int2 flowIndex = int2(int(floor(flow_tc.x * float(flowExtent.x))),
                                            int(floor(flow_tc.y * float(flowExtent.y))));
                const float4 flow = flows[f].read(uint2(repeatEdge(flowIndex, flowExtent)));
                sampl_uv = zoomed_tc + xy(flow);
                if (sampl_uv.x <= u.borderMinX) sampl_uv.x = 2.0f * u.borderMinX - sampl_uv.x;
                if (sampl_uv.y <= u.borderMinY) sampl_uv.y = 2.0f * u.borderMinY - sampl_uv.y;
                if (sampl_uv.x > u.borderMaxX) sampl_uv.x = 2.0f * u.borderMaxX - sampl_uv.x;
                if (sampl_uv.y > u.borderMaxY) sampl_uv.y = 2.0f * u.borderMaxY - sampl_uv.y;
                covariance = xyz(covariances[f].sample(kLinearClamp, sampl_uv));
            } else {
                covariance = xyz(covariances[f].sample(kLinearClamp, sampl_uv));
                const float weight_so_far =
                    accumulatedFrameWeights.sample(kLinearClamp, zoomed_tc).x * u.accumulatedWeightScale;
                if (weight_so_far < u.coverageThreshold)
                    covariance *= u.covarianceMultiplier;
            }
            float3 frame_color, frame_weights;
            SampleNeighborhoodRBF(quads[f], quadExtent, sampl_uv, bayerExtent, covariance,
                                  u.normalization, norms[f], u.kernelEpsilon, u.footprintLimit,
                                  frame_color, frame_weights);
            accum_color += frame_color * frame_weight;
            accum_weight += frame_weights * frame_weight;
        }
    }
    const float3 merged = accum_color / max(accum_weight, float3(u.dehomogenizeEpsilon));
    outMerged.write(float4(merged, 0.0f), gid);
}

//===----------------------------------------------------------------------===//
// STAGE 4 -- DENOISE (SABRE 4a..4c): symmetric-pair bilateral at dilation
// 1, 2, 4 (x the output scale), level 0 converting S2 -> funky YUV (S3), the
// last level converting back and encoding the DNG band (uint16, linear,
// un-whitened by outGain, x65535).
//===----------------------------------------------------------------------===//
struct SRSRAccumulatedColor {
    float4 accum_color;
    float4 accum_color_wider;
};

static inline SRSRAccumulatedColor ProcessSample(thread const float3 *samples, int x, int y, float key,
                                                 float bilateral_sigma, float outlier_sigma) {
    const float3 samp = samples[(y + 1) * 3 + x + 1];
    const float3 samp_opposite = samples[(-y + 1) * 3 - x + 1];
    const float diff = (samp.x + samp_opposite.x) * 0.5f - key;
    const float diff_sq = diff * diff;
    const float weight = DenoiseWeight(x) * DenoiseWeight(y) * exp2(-diff_sq * bilateral_sigma);
    const float weight_wider = DenoiseWeight(x) * DenoiseWeight(y) * exp2(-diff_sq * outlier_sigma);
    SRSRAccumulatedColor o;
    o.accum_color = float4((samp + samp_opposite) * weight, 2.0f * weight);
    o.accum_color_wider = float4((samp + samp_opposite) * weight_wider, 2.0f * weight_wider);
    return o;
}

static inline void srsrWriteDng(device ushort* outFinal, uint index, float3 v, constant SRSRDenoiseUniforms &u) {
    const float3 g = float3(u.outGainR, u.outGainG, u.outGainB);
    float3 c = clamp(v * g, float3(0.0f), float3(1.0f));
    // NaN guard without isnan (fast math may fold it): a NaN fails every compare.
    if (!(c.x >= 0.0f)) c.x = 0.0f;
    if (!(c.y >= 0.0f)) c.y = 0.0f;
    if (!(c.z >= 0.0f)) c.z = 0.0f;
    outFinal[index * 3u + 0u] = ushort(uint(c.x * 65535.0f + 0.5f));
    outFinal[index * 3u + 1u] = ushort(uint(c.y * 65535.0f + 0.5f));
    outFinal[index * 3u + 2u] = ushort(uint(c.z * 65535.0f + 0.5f));
}

kernel void srsr_denoise(texture2d<float, access::read> inBand              [[texture(0)]],
                         texture2d<float> accumulatedFrameWeights           [[texture(1)]],
                         texture2d<float, access::write> outBand            [[texture(2)]],
                         device ushort* outFinal                            [[buffer(1)]],
                         constant SRSRDenoiseUniforms &u                    [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const int gx = int(gid.x);
    const int gyy = int(gid.y) + u.outY0;           // global output row
    if (gid.x >= uint(u.width) || int(gid.y) >= u.outRows || gyy >= u.height) return;
    const int ly = gyy - u.texY0;                   // row in the band textures
    const float2 tex_coord = (float2(float(gx), float(gyy)) + 0.5f) /
                             float2(float(u.width), float(u.height));

    if (u.passthrough != 0) {
        const float3 c = xyz(inBand.read(uint2(uint(gx), uint(ly))));
        srsrWriteDng(outFinal, uint(int(gid.y) * u.width + gx), c, u);
        return;
    }

    const float num_merged_frames =
        accumulatedFrameWeights.sample(kLinearClamp, tex_coord).x * u.weightTextureScale + u.weightTextureBias;
    const float blurry_weight = clamp(1.25f - num_merged_frames * 0.25f, 0.0f, 1.0f);

    // Fetch3x3Samples with the frame_border mirror, as integer texel reads (each
    // dilated tap lands on a texel centre, so the GL bilinear fetch is exact).
    const int d = u.dilation;
    const int xm = mirrorIndex(gx - d, u.width), xp = mirrorIndex(gx + d, u.width);
    const int ym = clamp(mirrorIndex(gyy - d, u.height) - u.texY0, 0, u.texRows - 1);
    const int yp = clamp(mirrorIndex(gyy + d, u.height) - u.texY0, 0, u.texRows - 1);
    const int yc = clamp(ly, 0, u.texRows - 1);
    float3 samples[9];
    samples[0] = xyz(inBand.read(uint2(uint(xm), uint(ym))));
    samples[1] = xyz(inBand.read(uint2(uint(gx), uint(ym))));
    samples[2] = xyz(inBand.read(uint2(uint(xp), uint(ym))));
    samples[3] = xyz(inBand.read(uint2(uint(xm), uint(yc))));
    samples[4] = xyz(inBand.read(uint2(uint(gx), uint(yc))));
    samples[5] = xyz(inBand.read(uint2(uint(xp), uint(yc))));
    samples[6] = xyz(inBand.read(uint2(uint(xm), uint(yp))));
    samples[7] = xyz(inBand.read(uint2(uint(gx), uint(yp))));
    samples[8] = xyz(inBand.read(uint2(uint(xp), uint(yp))));
    if (u.level == 0)
        for (int i = 0; i < 9; ++i) samples[i] = ToGcamYUV(samples[i]);

    const float3 col = samples[4];
    float key = col.x;
    const float noise_model_var =
        dot(float3(u.yuvNoiseModelX, u.yuvNoiseModelY, u.yuvNoiseModelZ) * abs(col), float3(1.0f))
        + u.yuvNoiseModelW;
    float bilateral_sigma = noise_model_var;
    const float2 vc = (tex_coord - float2(0.5f)) * u.vignetteSpatialScaleFactor;
    float vignette = max(min(dot(vc, vc) / 0.25f - 0.25f, 1.0f), 0.0f);
    vignette *= vignette * vignette;
    bilateral_sigma *= u.denoisingFactor * (1.0f + u.vignetteStrengthBoost * vignette);
    bilateral_sigma = max(bilateral_sigma, 1.0e-12f);

    const float prefilter_weight = blurry_weight * ((u.level == 0) ? 1.0f : 0.0f);
    const float prefiltered_key = (samples[0].x + samples[2].x + samples[4].x + samples[6].x + samples[8].x) / 5.0f;
    key = mix(key, prefiltered_key, prefilter_weight);

    const float inv_sigma = (1.0f / bilateral_sigma) * num_merged_frames * (0.5f / log(2.0f));
    const float inv_outlier_sigma = u.outlierSigmaMultiplier * inv_sigma;

    float4 accum_color = float4(col, 1.0f) * DenoiseWeight(0) * DenoiseWeight(0);
    float4 accum_color_wider = accum_color;
    SRSRAccumulatedColor s;
    s = ProcessSample(samples, -1, -1, key, inv_sigma, inv_outlier_sigma);
    accum_color += s.accum_color; accum_color_wider += s.accum_color_wider;
    s = ProcessSample(samples, 0, -1, key, inv_sigma, inv_outlier_sigma);
    accum_color += s.accum_color; accum_color_wider += s.accum_color_wider;
    s = ProcessSample(samples, 1, -1, key, inv_sigma, inv_outlier_sigma);
    accum_color += s.accum_color; accum_color_wider += s.accum_color_wider;
    s = ProcessSample(samples, 1, 0, key, inv_sigma, inv_outlier_sigma);
    accum_color += s.accum_color; accum_color_wider += s.accum_color_wider;

    float3 denoised = xyz(accum_color) / accum_color.w;
    const float outlier_weight = (1.0f - smoothstep(0.25f, 0.5f, accum_color.w)) * blurry_weight;
    const float3 denoised_wider = xyz(accum_color_wider) / accum_color_wider.w;
    denoised = mix(denoised, denoised_wider, outlier_weight);
    const float luma_revert_factor = mix(u.globalLumaRevertFactor, u.rejectLumaRevertFactor, blurry_weight);
    denoised.x = denoised.x * (1.0f - luma_revert_factor) + col.x * luma_revert_factor;

    if (u.isLastLevel != 0) {
        denoised = FromGcamYUV(denoised);   // grain_strength is 0 in 6.1
        srsrWriteDng(outFinal, uint(int(gid.y) * u.width + gx), denoised, u);
        return;
    }
    outBand.write(float4(denoised, 0.0f), uint2(uint(gx), uint(ly)));
}
