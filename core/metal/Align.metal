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
    int factor;
    int prev_ts;
    int ts;
};

kernel void kernel_upscale_flow(
    texture2d<float, access::read> inFlow [[texture(0)]],
    texture2d<float, access::write> outFlow [[texture(1)]],
    constant UpscaleParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outFlow.get_width() || gid.y >= outFlow.get_height()) return;
    
    // Nearest neighbor sampling matching Python's repeat_interleave
    // scale_factor = prev_ts / (ts * factor)
    float scale = (float)params.prev_ts / (float)(params.ts * params.factor);
    
    uint2 in_gid = uint2(gid.x / params.factor, gid.y / params.factor);
    in_gid.x = min(in_gid.x, (uint)(inFlow.get_width() - 1));
    in_gid.y = min(in_gid.y, (uint)(inFlow.get_height() - 1));
    
    float2 f = inFlow.read(in_gid).rg;
    outFlow.write(float4(f.x * scale, f.y * scale, 0, 1), gid);
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
                    float m_val = bilinear_oob_zero(movTex, ry, rx, floor_off_y, floor_off_x, frac_x, frac_y);
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
