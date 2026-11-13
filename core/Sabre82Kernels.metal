// GCam 8.2.300 SABRE on the GPU. The arithmetic is core/sabre82_shared.h -- the SAME text the CPU path
// (core/sabre82.cpp) runs; these kernels map threads to pixels and emulate the texture formats / blends.
// Host: core/sabre82_metal.mm.
#include <metal_stdlib>
using namespace metal;
#include "sabre82_shared.h"

// kGuideImageEntryPoint: ref4 = RGBA16F reference colour + signed variance; cov = RGB10_A2 codes (0..1023).
kernel void s82_guide_kernel(device const float* raw        [[buffer(0)]],
                             device const float* lut        [[buffer(1)]],
                             constant S82GuideParams& P     [[buffer(2)]],
                             device float* ref4             [[buffer(3)]],
                             device ushort* cov             [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x, gy = (int)gid.y;
    if (gx >= P.Wg || gy >= P.Hg) return;
    const S82GuideParams p = P;
    float r4[4], c3[3];
    s82_guide_pixel(raw, p, lut, gx, gy, r4, c3);
    const int i = gy * p.Wg + gx;
    for (int k = 0; k < 4; ++k) ref4[i * 4 + k] = s82_q16(r4[k]);
    for (int k = 0; k < 3; ++k) cov[i * 3 + k] = (ushort)s82_unorm10_code(c3[k]);
}

// kBaseFrameRefColorEntryPoint (bicubic path) -> RGBA16F.
kernel void s82_base_ref_kernel(device const float* guide [[buffer(0)]],
                                device float* out         [[buffer(1)]],
                                constant int2& size       [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x, gy = (int)gid.y;
    if (gx >= size.x || gy >= size.y) return;
    const S82F4 r = s82_base_ref_pixel(guide, size.x, size.y, gx, gy);
    const int i = (gy * size.x + gx) * 4;
    out[i] = s82_q16(r.x); out[i + 1] = s82_q16(r.y); out[i + 2] = s82_q16(r.z); out[i + 3] = s82_q16(r.w);
}

// kRejectionEntryPoint -> R8 (1 - weight).
kernel void s82_rejection_kernel(device const float* base_ref  [[buffer(0)]],
                                 device const float* alt_guide [[buffer(1)]],
                                 device const float* flow      [[buffer(2)]],
                                 device const float* lut       [[buffer(3)]],
                                 constant S82RejParams& P      [[buffer(4)]],
                                 device uint8_t* rej           [[buffer(5)]],
                                 device const uint8_t* unb     [[buffer(6)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x, gy = (int)gid.y;
    if (gx >= P.Wg || gy >= P.Hg) return;
    const S82RejParams p = P;
    rej[gy * p.Wg + gx] = s82_to_u8(s82_rejection_pixel(base_ref, alt_guide, flow, lut, unb, p, gx, gy));
}

// kDilateMaskEntryPoint to the half-guide grid -> R8 keep. size = (W, H, OW, OH).
kernel void s82_dilate_kernel(device const uint8_t* rej [[buffer(0)]],
                              device uint8_t* keep      [[buffer(1)]],
                              constant int4& size       [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]])
{
    const int ox = (int)gid.x, oy = (int)gid.y;
    if (ox >= size.z || oy >= size.w) return;
    keep[oy * size.z + ox] = s82_to_u8(s82_dilate_pixel(rej, size.x, size.y, size.z, size.w, ox, oy));
}

// kMergeEntryPoint, one frame into the band accumulators (additive blend).
kernel void s82_merge_kernel(device const float* raw     [[buffer(0)]],
                             device const float* cov     [[buffer(1)]],
                             device const float* flow    [[buffer(2)]],
                             device const uint8_t* mask  [[buffer(3)]],
                             constant S82MergeParams& P  [[buffer(4)]],
                             device float* accC          [[buffer(5)]],
                             device float* accW          [[buffer(6)]],
                             constant int& nrows         [[buffer(7)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const int ox = (int)gid.x, ly = (int)gid.y;
    if (ox >= P.Wo || ly >= nrows) return;
    const S82MergeParams p = P;
    float c3[3], w3[3];
    s82_merge_pixel(raw, cov, flow, mask, p, ox, p.out_row0 + ly, c3, w3);
    const int i = (ly * p.Wo + ox) * 3;
    for (int k = 0; k < 3; ++k) { accC[i + k] += c3[k]; accW[i + k] += w3[k]; }
}

// SabreDehomogenize: colour * 1 / max(weight, 1e-7).
kernel void s82_dehomog_kernel(device float* accC        [[buffer(0)]],
                               device const float* accW  [[buffer(1)]],
                               constant int& n           [[buffer(2)]],
                               uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= n) return;
    accC[gid] = accC[gid] * (1.0f / s82_max(accW[gid], S82_DEHOMOG_EPS));
}
