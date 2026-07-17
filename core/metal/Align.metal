#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------------
// compute_sobel_gradx / grady
// --------------------------------------------------------------------------------
kernel void kernel_compute_sobel(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> gradxTex [[texture(1)]],
    texture2d<float, access::write> gradyTex [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) return;
    
    int w = inTexture.get_width();
    int h = inTexture.get_height();
    
    // Sobel X: [-1, 0, 1]
    float mx = (gid.x > 0) ? inTexture.read(uint2(gid.x - 1, gid.y)).r : 0.0f;
    float px = ((int)gid.x + 1 < w) ? inTexture.read(uint2(gid.x + 1, gid.y)).r : 0.0f;
    float gx = px - mx;
    
    // Sobel Y: [-1, 0, 1]^T
    float my = (gid.y > 0) ? inTexture.read(uint2(gid.x, gid.y - 1)).r : 0.0f;
    float py = ((int)gid.y + 1 < h) ? inTexture.read(uint2(gid.x, gid.y + 1)).r : 0.0f;
    float gy = py - my;
    
    gradxTex.write(float4(gx, 0, 0, 1), gid);
    gradyTex.write(float4(gy, 0, 0, 1), gid);
}

// --------------------------------------------------------------------------------
// upscale_flow (Nearest Neighbor matching Python flow.repeat_interleave)
// --------------------------------------------------------------------------------
struct UpscaleParams {
    int repeat_factor;
    int upsample_factor;
    int up_ny;
    int up_nx;
};

kernel void kernel_upscale_flow(
    texture2d<float, access::read> inFlow [[texture(0)]],
    texture2d<float, access::write> outFlow [[texture(1)]],
    constant UpscaleParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outFlow.get_width() || gid.y >= outFlow.get_height()) return;

    float2 out_v = float2(0.0f, 0.0f);
    if ((int)gid.y < params.up_ny && (int)gid.x < params.up_nx) {
        int sy = min((int)gid.y / params.repeat_factor, (int)inFlow.get_height() - 1);
        int sx = min((int)gid.x / params.repeat_factor, (int)inFlow.get_width() - 1);
        float2 f = inFlow.read(uint2(sx, sy)).rg;
        out_v = float2(f.x * (float)params.upsample_factor, f.y * (float)params.upsample_factor);
    }
    outFlow.write(float4(out_v.x, out_v.y, 0, 1), gid);
}

// --------------------------------------------------------------------------------
// block_match_L1
// --------------------------------------------------------------------------------
struct BlockMatchParams {
    int ts;
    int search_radius;
};

kernel void kernel_block_match_L1(
    texture2d<float, access::read> refTex [[texture(0)]],
    texture2d<float, access::read> movTex [[texture(1)]],
    texture2d<float, access::read> inFlow [[texture(2)]],
    texture2d<float, access::write> outFlow [[texture(3)]],
    constant BlockMatchParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]] // Thread per patch (ty, tx)
) {
    if (gid.x >= outFlow.get_width() || gid.y >= outFlow.get_height()) return;
    
    int tx = gid.x;
    int ty = gid.y;
    int ts = params.ts;
    int R = params.search_radius;
    
    float2 initial_flow = inFlow.read(gid).rg;
    int flow_dx = (int)round(initial_flow.x);
    int flow_dy = (int)round(initial_flow.y);
    
    int ox = tx * ts;
    int oy = ty * ts;
    
    float best_dist = 1e38f;
    int best_dy = 0;
    int best_dx = 0;
    
    for (int s_dy = -R; s_dy <= R; ++s_dy) {
        for (int s_dx = -R; s_dx <= R; ++s_dx) {
            float dist = 0.0f;
            for (int i = 0; i < ts; ++i) {
                for (int j = 0; j < ts; ++j) {
                    int ry = oy + i;
                    int rx = ox + j;
                    float r_val = (ry < (int)refTex.get_height() && rx < (int)refTex.get_width()) ? 
                                  refTex.read(uint2(rx, ry)).r : 0.0f;
                    
                    int my = oy + flow_dy + s_dy + i;
                    int mx = ox + flow_dx + s_dx + j;
                    my = clamp(my, 0, (int)movTex.get_height() - 1);
                    mx = clamp(mx, 0, (int)movTex.get_width() - 1);
                    float m_val = movTex.read(uint2(mx, my)).r;
                    
                    dist += abs(r_val - m_val);
                }
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_dy = s_dy;
                best_dx = s_dx;
            }
        }
    }
    
    float final_dx = initial_flow.x + (float)best_dx;
    float final_dy = initial_flow.y + (float)best_dy;
    outFlow.write(float4(final_dx, final_dy, 0, 1), gid);
}

// --------------------------------------------------------------------------------
// block_match_L2 (Spatial fallback since writing custom FFT shaders is slow)
// Mathematical parity with FFT L2 distance: ||R - M||^2
// --------------------------------------------------------------------------------
kernel void kernel_block_match_L2(
    texture2d<float, access::read> refTex [[texture(0)]],
    texture2d<float, access::read> movTex [[texture(1)]],
    texture2d<float, access::read> inFlow [[texture(2)]],
    texture2d<float, access::write> outFlow [[texture(3)]],
    constant BlockMatchParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outFlow.get_width() || gid.y >= outFlow.get_height()) return;
    
    int tx = gid.x;
    int ty = gid.y;
    int ts = params.ts;
    int R = params.search_radius;
    
    float2 initial_flow = inFlow.read(gid).rg;
    int flow_dx = (int)round(initial_flow.x);
    int flow_dy = (int)round(initial_flow.y);
    
    int ox = tx * ts;
    int oy = ty * ts;
    
    float best_dist = 1e38f;
    int best_dy = 0;
    int best_dx = 0;
    
    for (int s_dy = -R; s_dy <= R; ++s_dy) {
        for (int s_dx = -R; s_dx <= R; ++s_dx) {
            float dist = 0.0f;
            for (int i = 0; i < ts; ++i) {
                for (int j = 0; j < ts; ++j) {
                    int ry = oy + i;
                    int rx = ox + j;
                    float r_val = (ry < (int)refTex.get_height() && rx < (int)refTex.get_width()) ? 
                                  refTex.read(uint2(rx, ry)).r : 0.0f;
                    
                    int my = oy + flow_dy + s_dy + i;
                    int mx = ox + flow_dx + s_dx + j;
                    my = clamp(my, 0, (int)movTex.get_height() - 1);
                    mx = clamp(mx, 0, (int)movTex.get_width() - 1);
                    float m_val = movTex.read(uint2(mx, my)).r;
                    
                    float d = r_val - m_val;
                    dist += d * d;
                }
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_dy = s_dy;
                best_dx = s_dx;
            }
        }
    }
    
    float final_dx = initial_flow.x + (float)best_dx;
    float final_dy = initial_flow.y + (float)best_dy;
    outFlow.write(float4(final_dx, final_dy, 0, 1), gid);
}

// --------------------------------------------------------------------------------
// ica_refine
// --------------------------------------------------------------------------------
struct IcaParams {
    int ts;
    int n_iter;
};

inline float sample_oob_zero(texture2d<float, access::read> img, int y, int x) {
    if (y >= 0 && y < (int)img.get_height() && x >= 0 && x < (int)img.get_width()) {
        return img.read(uint2(x, y)).r;
    }
    return 0.0f;
}

inline float sample_clamp_edge(texture2d<float, access::read> img, int y, int x) {
    y = clamp(y, 0, (int)img.get_height() - 1);
    x = clamp(x, 0, (int)img.get_width() - 1);
    return img.read(uint2(x, y)).r;
}

inline float bilinear_oob_zero(texture2d<float, access::read> img, int pixel_y, int pixel_x,
                               int floor_off_y, int floor_off_x, float frac_x, float frac_y) {
    int floor_y = pixel_y + floor_off_y;
    int floor_x = pixel_x + floor_off_x;
    
    float m00 = sample_oob_zero(img, floor_y, floor_x);
    float m01 = sample_oob_zero(img, floor_y, floor_x + 1);
    float m10 = sample_oob_zero(img, floor_y + 1, floor_x);
    float m11 = sample_oob_zero(img, floor_y + 1, floor_x + 1);
    
    float lerpx_top = m00 + (m01 - m00) * frac_x;
    float lerpx_bot = m10 + (m11 - m10) * frac_x;
    return lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
}

inline float bilinear_clamp_edge(texture2d<float, access::read> img, int pixel_y, int pixel_x,
                                 int floor_off_y, int floor_off_x, float frac_x, float frac_y) {
    int floor_y = pixel_y + floor_off_y;
    int floor_x = pixel_x + floor_off_x;
    
    float m00 = sample_clamp_edge(img, floor_y, floor_x);
    float m01 = sample_clamp_edge(img, floor_y, floor_x + 1);
    float m10 = sample_clamp_edge(img, floor_y + 1, floor_x);
    float m11 = sample_clamp_edge(img, floor_y + 1, floor_x + 1);
    
    float lerpx_top = m00 + (m01 - m00) * frac_x;
    float lerpx_bot = m10 + (m11 - m10) * frac_x;
    return lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
}

kernel void kernel_ica_refine(
    texture2d<float, access::read> refTex [[texture(0)]],
    texture2d<float, access::read> gxTex [[texture(1)]],
    texture2d<float, access::read> gyTex [[texture(2)]],
    texture2d<float, access::read> movTex [[texture(3)]],
    texture2d<float, access::read> inFlow [[texture(4)]],
    texture2d<float, access::write> outFlow [[texture(5)]],
    constant IcaParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outFlow.get_width() || gid.y >= outFlow.get_height()) return;
    
    int tx = gid.x;
    int ty = gid.y;
    int ts = params.ts;
    
    float2 flow = inFlow.read(gid).rg;
    
    int ox = tx * ts;
    int oy = ty * ts;
    
    // Precompute H matrix
    float h11 = 0.0f, h12 = 0.0f, h22 = 0.0f;
    for (int i = 0; i < ts; ++i) {
        for (int j = 0; j < ts; ++j) {
            int ry = oy + i;
            int rx = ox + j;
            if (ry < (int)refTex.get_height() && rx < (int)refTex.get_width()) {
                float gx = gxTex.read(uint2(rx, ry)).r;
                float gy = gyTex.read(uint2(rx, ry)).r;
                h11 += gx * gx;
                h12 += gx * gy;
                h22 += gy * gy;
            }
        }
    }
    
    float det = h11 * h22 - h12 * h12;
    if (abs(det) < 1e-6f) {
        outFlow.write(float4(flow.x, flow.y, 0, 1), gid);
        return;
    }
    
    float ih11 = h22 / det;
    float ih12 = -h12 / det;
    float ih22 = h11 / det;
    
    for (int iter = 0; iter < params.n_iter; ++iter) {
        float f_dx = floor(flow.x);
        float frac_x = flow.x - f_dx;
        int floor_off_x = (int)f_dx;
        
        float f_dy = floor(flow.y);
        float frac_y = flow.y - f_dy;
        int floor_off_y = (int)f_dy;
        
        float v1 = 0.0f;
        float v2 = 0.0f;
        
        for (int i = 0; i < ts; ++i) {
            for (int j = 0; j < ts; ++j) {
                int ry = oy + i;
                int rx = ox + j;
                if (ry < (int)refTex.get_height() && rx < (int)refTex.get_width()) {
                    float r_val = refTex.read(uint2(rx, ry)).r;
                    float m_val = (ts == 8)
                        ? bilinear_clamp_edge(movTex, ry, rx, floor_off_y, floor_off_x, frac_x, frac_y)
                        : bilinear_oob_zero(movTex, ry, rx, floor_off_y, floor_off_x, frac_x, frac_y);
                    float err = m_val - r_val;
                    
                    float gx = gxTex.read(uint2(rx, ry)).r;
                    float gy = gyTex.read(uint2(rx, ry)).r;
                    
                    v1 += err * gx;
                    v2 += err * gy;
                }
            }
        }
        
        float delta_x = ih11 * v1 + ih12 * v2;
        float delta_y = ih12 * v1 + ih22 * v2;
        
        flow.x -= delta_x;
        flow.y -= delta_y;
    }
    
    outFlow.write(float4(flow.x, flow.y, 0, 1), gid);
}
 
 // --------------------------------------------------------------------------------
// kernel_block_match_L2_FFT
// Custom 24-point 2D DFT native Metal implementation matching PyTorch's irfft2
// --------------------------------------------------------------------------------
kernel void kernel_block_match_L2_FFT(
    texture2d<float, access::read> refTex [[texture(0)]],
    texture2d<float, access::read> movTex [[texture(1)]],
    texture2d<float, access::read> inFlow [[texture(2)]],
    texture2d<float, access::write> outFlow [[texture(3)]],
    constant BlockMatchParams& params [[buffer(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    int tx = tgid.x;
    int ty = tgid.y;
    int ts = params.ts;
    int R = params.search_radius; // Must be 4
    
    if (tx >= (int)outFlow.get_width() || ty >= (int)outFlow.get_height()) return;
    
    float2 initial_flow = inFlow.read(uint2(tx, ty)).rg;
    int flow_dx = (int)round(initial_flow.x);
    int flow_dy = (int)round(initial_flow.y);
    
    int ox = tx * ts;
    int oy = ty * ts;
    
    threadgroup float ref_spatial[576];
    threadgroup float mov_spatial[576];
    threadgroup float2 corrs_freq[576];
    threadgroup float corrs_spatial[576];
    
    int py = tid / 24;
    int px = tid % 24;
    
    // Load ref_spatial (16x16 zero-padded to 24x24 by placing at center R, R)
    float r_val = 0.0f;
    if (py >= R && py < R + ts && px >= R && px < R + ts) {
        int ry = oy + (py - R);
        int rx = ox + (px - R);
        if (ry < (int)refTex.get_height() && rx < (int)refTex.get_width()) {
            r_val = refTex.read(uint2(rx, ry)).r;
        }
    }
    ref_spatial[tid] = r_val;
    
    // Load mov_spatial (24x24)
    int my = oy + flow_dy - R + py;
    int mx = ox + flow_dx - R + px;
    my = clamp(my, 0, (int)movTex.get_height() - 1);
    mx = clamp(mx, 0, (int)movTex.get_width() - 1);
    mov_spatial[tid] = movTex.read(uint2(mx, my)).r;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Forward 2D DFT (size 24x24) for both patches
    float2 R_k = float2(0.0f, 0.0f);
    float2 M_k = float2(0.0f, 0.0f);
    for (int n = 0; n < 576; ++n) {
        int ny = n / 24;
        int nx = n % 24;
        float angle = -2.0f * M_PI_F * ((float)(py * ny) + (float)(px * nx)) / 24.0f;
        float2 tw = float2(cos(angle), sin(angle));
        
        float r_n = ref_spatial[n];
        R_k += float2(r_n * tw.x, r_n * tw.y);
        
        float m_n = mov_spatial[n];
        M_k += float2(m_n * tw.x, m_n * tw.y);
    }
    
    // Complex multiply: conj(R_k) * M_k
    // conj(R_k) = (R_k.x, -R_k.y)
    // C_k = conj(R_k) * M_k = (R_k.x * M_k.x + R_k.y * M_k.y, R_k.x * M_k.y - R_k.y * M_k.x)
    corrs_freq[tid] = float2(R_k.x * M_k.x + R_k.y * M_k.y, R_k.x * M_k.y - R_k.y * M_k.x);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Inverse 2D DFT
    float2 c_spatial = float2(0.0f, 0.0f);
    for (int k = 0; k < 576; ++k) {
        int ky = k / 24;
        int kx = k % 24;
        float angle = 2.0f * M_PI_F * ((float)(py * ky) + (float)(px * kx)) / 24.0f;
        float2 tw = float2(cos(angle), sin(angle));
        
        float2 C = corrs_freq[k];
        c_spatial += float2(C.x * tw.x - C.y * tw.y, C.x * tw.y + C.y * tw.x);
    }
    corrs_spatial[tid] = c_spatial.x / 576.0f;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute L2 Error = sum(mov^2) - 2 * corrs
    // And find argmax (min L2) using threadgroup reduction
    
    int s_dy = py - 12;
    int s_dx = px - 12;
    
    float l2_err = 1e38f;
    
    if (abs(s_dy) <= R && abs(s_dx) <= R) {
        float mov_sq_sum = 0.0f;
        for (int i = 0; i < ts; ++i) {
            for (int j = 0; j < ts; ++j) {
                int sm_y = i + s_dy + R;
                int sm_x = j + s_dx + R;
                float m = mov_spatial[sm_y * 24 + sm_x];
                mov_sq_sum += m * m;
            }
        }
        
        // PyTorch fftshift centers the zero-frequency at 12, 12
        // So corrs_spatial[py * 24 + px] corresponds to shift (s_dy, s_dx)
        float corr = corrs_spatial[tid];
        l2_err = mov_sq_sum - 2.0f * corr;
    }
    
    // Shared memory for reduction
    threadgroup float best_errs[576];
    threadgroup int best_idxs[576];
    
    best_errs[tid] = l2_err;
    best_idxs[tid] = tid;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction to find minimum L2 error
    for (int stride = 256; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < 576) {
            if (best_errs[tid + stride] < best_errs[tid]) {
                best_errs[tid] = best_errs[tid + stride];
                best_idxs[tid] = best_idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Handle the remaining 64 elements (576 = 512 + 64)
    if (tid == 0) {
        for (int i = 1; i < 576; ++i) {
            if (best_errs[i] < best_errs[0]) {
                best_errs[0] = best_errs[i];
                best_idxs[0] = best_idxs[i];
            }
        }
        
        int best_tid = best_idxs[0];
        int best_py = best_tid / 24;
        int best_px = best_tid % 24;
        
        float final_dx = initial_flow.x + (float)(best_px - 12);
        float final_dy = initial_flow.y + (float)(best_py - 12);
        
        outFlow.write(float4(final_dx, final_dy, 0.0f, 1.0f), uint2(tx, ty));
    }
}
