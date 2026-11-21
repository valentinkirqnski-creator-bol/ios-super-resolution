// hsr_align (Handheld-SR package aligner) on the GPU. The arithmetic is core/hsr_align_shared.h -- the SAME text the
// CPU path (core/hsr_align.cpp) runs; these kernels map one thread to one pixel or one tile. Host: core/hsr_align_metal.mm.
#include <metal_stdlib>
using namespace metal;
#include "hsr_align_shared.h"

kernel void hsra_grey_quad_kernel(device const float* raw     [[buffer(0)]],
                                  constant HsraGreyParams& P  [[buffer(1)]],
                                  device float* q             [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    const int x = (int)gid.x, y = (int)gid.y;
    if (x >= P.w || y >= P.h) return;
    q[y * P.w + x] = hsra_grey_quad(raw, P.rw, P.g00, P.g01, P.g10, P.g11, x, y);
}

// size = (w, h, 0, 0)
kernel void hsra_blur_v_kernel(device const float* q  [[buffer(0)]],
                               constant int4& size    [[buffer(1)]],
                               device float* t        [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]])
{
    const int x = (int)gid.x, y = (int)gid.y;
    if (x >= size.x || y >= size.y) return;
    t[y * size.x + x] = hsra_blur_v(q, size.x, size.y, x, y);
}

kernel void hsra_blur_h_clip_kernel(device const float* t  [[buffer(0)]],
                                    constant int4& size    [[buffer(1)]],
                                    device float* out      [[buffer(2)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    const int x = (int)gid.x, y = (int)gid.y;
    if (x >= size.x || y >= size.y) return;
    out[y * size.x + x] = hsra_blur_h_clip(t, size.x, size.y, x, y);
}

// One factor-2 round. size = (w, h, w2, h2): in is w x h, hz is w2 x h, out is w2 x h2.
kernel void hsra_down_h_kernel(device const float* in  [[buffer(0)]],
                               constant int4& size     [[buffer(1)]],
                               device float* hz        [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]])
{
    const int x = (int)gid.x, y = (int)gid.y;
    if (x >= size.z || y >= size.y) return;
    hz[y * size.z + x] = hsra_down_h(in, size.x, size.y, x, y);
}

kernel void hsra_down_v_kernel(device const float* hz  [[buffer(0)]],
                               constant int4& size     [[buffer(1)]],
                               device float* out       [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]])
{
    const int x = (int)gid.x, y = (int)gid.y;
    if (x >= size.z || y >= size.w) return;
    out[y * size.z + x] = hsra_down_v(hz, size.z, size.y, x, y);
}

// size = (w, h, 0, 0)
kernel void hsra_grad_kernel(device const float* in  [[buffer(0)]],
                             constant int4& size      [[buffer(1)]],
                             device float* gx         [[buffer(2)]],
                             device float* gy         [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const int x = (int)gid.x, y = (int)gid.y;
    if (x >= size.x || y >= size.y) return;
    gx[y * size.x + x] = hsra_grad_x(in, size.x, size.y, x, y);
    gy[y * size.x + x] = hsra_grad_y(in, size.x, size.y, x, y);
}

// size = (w, ts, cols, rows); s3 = (sxx, sxy, syy) per tile, interleaved.
kernel void hsra_tile_structure_kernel(device const float* gx  [[buffer(0)]],
                                       device const float* gy  [[buffer(1)]],
                                       constant int4& size     [[buffer(2)]],
                                       device float* s3        [[buffer(3)]],
                                       uint2 gid [[thread_position_in_grid]])
{
    const int tx = (int)gid.x, ty = (int)gid.y;
    if (tx >= size.z || ty >= size.w) return;
    float s[3];
    hsra_tile_structure(gx, gy, size.x, size.y, tx, ty, s);
    const int i = (ty * size.z + tx) * 3;
    s3[i] = s[0]; s3[i + 1] = s[1]; s3[i + 2] = s[2];
}

kernel void hsra_block_match_kernel(device const float* ref   [[buffer(0)]],
                                    device const float* mov   [[buffer(1)]],
                                    device const float* idx   [[buffer(2)]],
                                    device const float* idy   [[buffer(3)]],
                                    constant HsraBmParams& P  [[buffer(4)]],
                                    device float* odx         [[buffer(5)]],
                                    device float* ody         [[buffer(6)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    const int tx = (int)gid.x, ty = (int)gid.y;
    if (tx >= P.cols || ty >= P.rows) return;
    const int i = ty * P.cols + tx;
    float o[2];
    hsra_block_match_tile(ref, mov, P.w, P.h, idx[i], idy[i], P.ts, P.radius, P.l2, tx, ty, o);
    odx[i] = o[0];
    ody[i] = o[1];
}

// Same layout as HsraBmParams (radius / l2 unused).
kernel void hsra_ica_kernel(device const float* ref   [[buffer(0)]],
                            device const float* gx    [[buffer(1)]],
                            device const float* gy    [[buffer(2)]],
                            device const float* s3    [[buffer(3)]],
                            device const float* mov   [[buffer(4)]],
                            device const float* idx   [[buffer(5)]],
                            device const float* idy   [[buffer(6)]],
                            constant HsraBmParams& P  [[buffer(7)]],
                            device float* odx         [[buffer(8)]],
                            device float* ody         [[buffer(9)]],
                            uint2 gid [[thread_position_in_grid]])
{
    const int tx = (int)gid.x, ty = (int)gid.y;
    if (tx >= P.cols || ty >= P.rows) return;
    const int i = ty * P.cols + tx;
    float o[2];
    hsra_ica_tile(ref, gx, gy, mov, P.w, P.h, P.ts, tx, ty, idx[i], idy[i], s3[i * 3], s3[i * 3 + 1], s3[i * 3 + 2], o);
    odx[i] = o[0];
    ody[i] = o[1];
}

// grid = (cols, rows, 0, 0) of the target grid.
kernel void hsra_transfer_kernel(device const float* sdx          [[buffer(0)]],
                                 device const float* sdy          [[buffer(1)]],
                                 device const float* ref          [[buffer(2)]],
                                 device const float* mov          [[buffer(3)]],
                                 constant HsraTransferParams& P   [[buffer(4)]],
                                 constant int4& grid              [[buffer(5)]],
                                 device float* odx                [[buffer(6)]],
                                 device float* ody                [[buffer(7)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    const int tx = (int)gid.x, ty = (int)gid.y;
    if (tx >= grid.x || ty >= grid.y) return;
    const HsraTransferParams p = P;
    float o[2];
    hsra_transfer_tile(sdx, sdy, ref, mov, p, tx, ty, o);
    odx[ty * grid.x + tx] = o[0];
    ody[ty * grid.x + tx] = o[1];
}

// grid = (nx, ny, 0, 0)
kernel void hsra_median_kernel(device const float* sdx  [[buffer(0)]],
                               device const float* sdy  [[buffer(1)]],
                               constant int4& grid      [[buffer(2)]],
                               device float* odx        [[buffer(3)]],
                               device float* ody        [[buffer(4)]],
                               uint2 gid [[thread_position_in_grid]])
{
    const int x = (int)gid.x, y = (int)gid.y;
    if (x >= grid.x || y >= grid.y) return;
    odx[y * grid.x + x] = hsra_median9(sdx, grid.x, grid.y, x, y);
    ody[y * grid.x + x] = hsra_median9(sdy, grid.x, grid.y, x, y);
}
