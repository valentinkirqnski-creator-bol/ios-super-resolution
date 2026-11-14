// GCam 8.2.300 SABRE bit-exact to a Mali-G71 (Config::sabre_mali_exact) on the GPU: guide, base reference,
// rejection, dilate. The arithmetic is the s82m_* text of core/sabre82_shared.h, the same the CPU model runs.
//
// NOT part of default.metallib: this file and core/sabre82_shared.h ship as bundle resources and
// core/sabre82_metal.mm compiles them at run time with fast math OFF, S82M_CONTRACT_PRAGMA (no FMA contraction)
// and S82_Q16_SOFT (fp16 stores rounded in software), which the app's fast-math library can't hold to the bit.
// The #include below is replaced with the header text before compiling. Buffer layouts are those of the
// portable kernels in core/Sabre82Kernels.metal, so the host swaps pipelines only.
#include <metal_stdlib>
using namespace metal;
#include "sabre82_shared.h"

kernel void s82m_guide_kernel(device const float* raw        [[buffer(0)]],
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
    s82m_guide_pixel(raw, p, lut, gx, gy, r4, c3);
    const int i = gy * p.Wg + gx;
    for (int k = 0; k < 4; ++k) ref4[i * 4 + k] = s82_q16(r4[k]);
    for (int k = 0; k < 3; ++k) cov[i * 3 + k] = (ushort)s82_unorm10_code(c3[k]);
}

kernel void s82m_base_ref_kernel(device const float* guide [[buffer(0)]],
                                 device float* out         [[buffer(1)]],
                                 constant int2& size       [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x, gy = (int)gid.y;
    if (gx >= size.x || gy >= size.y) return;
    const S82F4 r = s82m_base_ref_pixel(guide, size.x, size.y, gx, gy);
    const int i = (gy * size.x + gx) * 4;
    out[i] = s82_q16(r.x); out[i + 1] = s82_q16(r.y); out[i + 2] = s82_q16(r.z); out[i + 3] = s82_q16(r.w);
}

kernel void s82m_rejection_kernel(device const float* base_ref  [[buffer(0)]],
                                  device const float* alt_guide [[buffer(1)]],
                                  device const float* flow      [[buffer(2)]],
                                  device const float* lut       [[buffer(3)]],
                                  constant S82RejParams& P      [[buffer(4)]],
                                  device uint8_t* rej           [[buffer(5)]],
                                  device const uint8_t* unb     [[buffer(6)]],
                                  device const float* gm        [[buffer(7)]],
                                  device float* itg             [[buffer(8)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x, gy = (int)gid.y;
    if (gx >= P.Wg || gy >= P.Hg) return;
    const S82RejParams p = P;
    float it;
    const int i = gy * p.Wg + gx;
    rej[i] = s82m_to_u8(s82m_rejection_pixel(base_ref, alt_guide, flow, lut, unb, p, gx, gy, &it));
    itg[i] = it * gm[i];
}

// size = (W, H, OW, OH)
kernel void s82m_dilate_kernel(device const uint8_t* rej [[buffer(0)]],
                               device uint8_t* keep      [[buffer(1)]],
                               constant int4& size       [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]])
{
    const int ox = (int)gid.x, oy = (int)gid.y;
    if (ox >= size.z || oy >= size.w) return;
    keep[oy * size.z + ox] = s82m_to_u8(s82m_dilate_pixel(rej, size.x, size.y, ox, oy));
}
