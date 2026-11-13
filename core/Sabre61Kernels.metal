// GCam 6.1.021 SABRE merge (STAGE 3) + post-merge denoiser (STAGE 4) on the GPU.
// The arithmetic is core/sabre61_shared.h, the SAME text the CPU path (core/sabre61.cpp) runs; these
// kernels only map threads to pixels and do the blending the GL pipeline did with its blend state.
// Host: core/sabre61_metal.mm.
#include <metal_stdlib>
using namespace metal;
#include "sabre61_shared.h"

// ---------------------------------------------------------------------------------------------- per frame
// STAGE 1: guide + kernel for one guide pixel. ref4 = RGBA16F reference colour + std_dev (fp16-rounded,
// kept as float for the rejection pass); cov = o_gauss_cov stored as half (RGBA16F .xyz).
kernel void s61_guide_kernel(device const float* raw          [[buffer(0)]],
                             device const float* lut          [[buffer(1)]],
                             constant S61GuideParams& P       [[buffer(2)]],
                             device float* ref4               [[buffer(3)]],
                             device half* cov3                [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x, gy = (int)gid.y, Wg = P.Wr / 2, Hg = P.Hr / 2;
    if (gx >= Wg || gy >= Hg) return;
    const S61GuideParams p = P;
    float r4[4], c3[3];
    s61_guide_pixel(raw, p, lut, gx, gy, r4, c3, nullptr);
    const int i = gy * Wg + gx;
    for (int k = 0; k < 4; ++k) ref4[i * 4 + k] = s61_q16(r4[k]);
    for (int k = 0; k < 3; ++k) cov3[i * 3 + k] = half(c3[k]);
}

// STAGE 2: rejection for one guide pixel -> R8 (1 - weight). Unblocker off (texture reads 0).
kernel void s61_rejection_kernel(device const float* base_guide [[buffer(0)]],
                                 device const float* alt_guide  [[buffer(1)]],
                                 device const float* flow       [[buffer(2)]],
                                 device const float* lut        [[buffer(3)]],
                                 constant S61RejParams& P       [[buffer(4)]],
                                 device uint8_t* rej            [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x, gy = (int)gid.y;
    if (gx >= P.Wg || gy >= P.Hg) return;
    const S61RejParams p = P;
    const float r = s61_rejection_pixel(base_guide, alt_guide, flow, lut, nullptr, 0, 0, p, gx, gy);
    rej[gy * p.Wg + gx] = s61_to_u8(r);
}

// Dilate (R8 keep) and copy_mask (additive R8 blend of keep / 25 into the accumulated weights).
kernel void s61_dilate_accumulate_kernel(device const uint8_t* rej [[buffer(0)]],
                                         device uint8_t* keep      [[buffer(1)]],
                                         device uint8_t* acc       [[buffer(2)]],
                                         constant int2& size       [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]])
{
    const int gx = (int)gid.x, gy = (int)gid.y, W = size.x, H = size.y;
    if (gx >= W || gy >= H) return;
    const uint8_t k = s61_to_u8(s61_dilate_pixel(rej, W, H, gx, gy));
    const int i = gy * W + gx;
    keep[i] = k;
    acc[i] = s61_copy_mask_accumulate(acc[i], k);
}

// ---------------------------------------------------------------------------------------------- per band
// One output pixel of one frame. Alt frames: additive blend (GL_ONE, GL_ONE) into accC/accW.
// Base frame (processed last): the (SRC_ALPHA, SRC_COLOR) blend of (1/total_weight, .., color) ->
// accC = (accC + color) / (accW + weight), stored as R16F.
kernel void s61_merge_kernel(device const float* raw         [[buffer(0)]],
                             device const float* cov         [[buffer(1)]],
                             device const float* flow        [[buffer(2)]],
                             device const uint8_t* mask      [[buffer(3)]],
                             device const uint8_t* acc_w     [[buffer(4)]],
                             constant S61MergeParams& P      [[buffer(5)]],
                             device float* accC              [[buffer(6)]],
                             device float* accW              [[buffer(7)]],
                             constant int& nrows             [[buffer(8)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const int ox = (int)gid.x, ly = (int)gid.y;
    if (ox >= P.Wo || ly >= nrows) return;
    const S61MergeParams p = P;
    float c3[3], w3[3];
    s61_merge_pixel(raw, cov, flow, mask, acc_w, p, ox, p.out_row0 + ly, c3, w3);
    const int i = (ly * p.Wo + ox) * 3;
    if (p.is_base == 0) {
        for (int k = 0; k < 3; ++k) { accC[i + k] += c3[k]; accW[i + k] += w3[k]; }
    } else {
        for (int k = 0; k < 3; ++k) {
            const float inv = 1.0f / (accW[i + k] + w3[k]);
            accC[i + k] = s61_q16(c3[k] * inv + accC[i + k] * inv);
        }
    }
}

// One output pixel of one denoiser level. Levels 0 and 1 store R16F; level 2 is the final linear RGB.
kernel void s61_denoise_kernel(device const float* in        [[buffer(0)]],
                               device const uint8_t* acc_w   [[buffer(1)]],
                               constant S61DenoiseParams& P  [[buffer(2)]],
                               device float* out             [[buffer(3)]],
                               constant int& nrows           [[buffer(4)]],
                               uint2 gid [[thread_position_in_grid]])
{
    const int ox = (int)gid.x, ly = (int)gid.y;
    if (ox >= P.Wo || ly >= nrows) return;
    const S61DenoiseParams p = P;
    const S61F3 r = s61_denoise_pixel(in, acc_w, p, ox, p.out_row0 + ly);
    const int i = (ly * p.Wo + ox) * 3;
    if (p.level < 2) { out[i] = s61_q16(r.x); out[i + 1] = s61_q16(r.y); out[i + 2] = s61_q16(r.z); }
    else { out[i] = r.x; out[i + 1] = r.y; out[i + 2] = r.z; }
}
