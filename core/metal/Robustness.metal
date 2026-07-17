#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------------
// Dogson Interpolation
// --------------------------------------------------------------------------------
inline float dogson_quadratic(float x) {
    float ax = abs(x);
    if (ax <= 0.5f) return -2.0f * ax * ax + 1.0f;
    if (ax <= 1.5f) return ax * ax - 2.5f * ax + 1.5f;
    return 0.0f;
}

// Extract the Bayer quad into RGB guide image (matches Python's simple extract, wait, Dogson is for warping)
kernel void kernel_extract_guide(
    texture2d<float, access::read> rawTex [[texture(0)]],
    texture2d<float, access::write> guideTex [[texture(1)]],
    constant float3& white_balance [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= guideTex.get_width() || gid.y >= guideTex.get_height()) return;
    
    // Simplification for Bayer RGGB (since we're locked to RGGB in config)
    float r = rawTex.read(uint2(gid.x * 2, gid.y * 2)).r / white_balance.r;
    float gr = rawTex.read(uint2(gid.x * 2 + 1, gid.y * 2)).r / white_balance.g;
    float gb = rawTex.read(uint2(gid.x * 2, gid.y * 2 + 1)).r / white_balance.g;
    float b = rawTex.read(uint2(gid.x * 2 + 1, gid.y * 2 + 1)).r / white_balance.b;
    
    float g = 0.5f * (gr + gb);
    
    // Store as RGB in RGBA texture
    guideTex.write(float4(r, g, b, 1.0f), gid);
}

// --------------------------------------------------------------------------------
// local_stats_3x3
// --------------------------------------------------------------------------------
kernel void kernel_local_stats_3x3(
    texture2d<float, access::read> guideTex [[texture(0)]],
    texture2d<float, access::write> meansTex [[texture(1)]],
    texture2d<float, access::write> varsTex [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = guideTex.get_width();
    int h = guideTex.get_height();
    if (gid.x >= w || gid.y >= h) return;
    
    float3 s = float3(0.0f);
    float3 s2 = float3(0.0f);
    
    for (int i = -1; i <= 1; ++i) {
        int yy = clamp((int)gid.y + i, 0, h - 1);
        for (int j = -1; j <= 1; ++j) {
            int xx = clamp((int)gid.x + j, 0, w - 1);
            float3 v = guideTex.read(uint2(xx, yy)).rgb;
            s += v;
            s2 += v * v;
        }
    }
    
    float3 m = s / 9.0f;
    float3 v = s2 / 9.0f - m * m;
    
    meansTex.write(float4(m, 1.0f), gid);
    varsTex.write(float4(v, 1.0f), gid);
}

// --------------------------------------------------------------------------------
// warp_dogson_diff
// Matches Python: upscaled flow, dogson sample from mov guide, difference from ref guide
// --------------------------------------------------------------------------------
struct RobustnessParams {
    int tile_size;
};

inline float3 sample_dogson_rgb(texture2d<float, access::read> tex, float LR_y, float LR_x) {
    if (LR_y < 0.0f || LR_y >= tex.get_height() || LR_x < 0.0f || LR_x >= tex.get_width()) {
        return float3(1e30f); // OOB
    }
    
    int center_y = (int)round(LR_y);
    int center_x = (int)round(LR_x);
    
    float3 buf = float3(0.0f);
    float w_acc = 0.0f;
    
    for (int i = -1; i <= 1; ++i) {
        int y_ = clamp(center_y + i, 0, (int)tex.get_height() - 1);
        float dy = (float)y_ - LR_y;
        float wy = dogson_quadratic(dy);
        
        for (int j = -1; j <= 1; ++j) {
            int x_ = clamp(center_x + j, 0, (int)tex.get_width() - 1);
            float dx = (float)x_ - LR_x;
            float w = wy * dogson_quadratic(dx);
            
            buf += tex.read(uint2(x_, y_)).rgb * w;
            w_acc += w;
        }
    }
    return (w_acc > 1e-12f) ? buf / w_acc : float3(1e30f);
}

kernel void kernel_warp_dogson_diff(
    texture2d<float, access::read> refGuideTex [[texture(0)]],
    texture2d<float, access::read> movGuideTex [[texture(1)]],
    texture2d<float, access::read> flowTex [[texture(2)]],
    texture2d<float, access::write> d_p_Tex [[texture(3)]],
    constant RobustnessParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= d_p_Tex.get_width() || gid.y >= d_p_Tex.get_height()) return;
    
    int ts = params.tile_size;
    // Wait, the flow we have is in guide coordinates because we ran alignment on guide pyramids.
    // The flow scale is fine as-is.
    int patch_idy = min((int)gid.y / ts, (int)flowTex.get_height() - 1);
    int patch_idx = min((int)gid.x / ts, (int)flowTex.get_width() - 1);
    
    float2 flow = flowTex.read(uint2(patch_idx, patch_idy)).rg;
    
    float LR_y = (float)gid.y + flow.y;
    float LR_x = (float)gid.x + flow.x;
    
    float3 ref_val = refGuideTex.read(gid).rgb;
    float3 mov_val = sample_dogson_rgb(movGuideTex, LR_y, LR_x);
    
    float3 d_p = mov_val - ref_val;
    d_p_Tex.write(float4(d_p, 1.0f), gid);
}

// --------------------------------------------------------------------------------
// apply_noise_model
// --------------------------------------------------------------------------------
kernel void kernel_apply_noise_model(
    texture2d<float, access::read> d_p_Tex [[texture(0)]],
    texture2d<float, access::read> refMeansTex [[texture(1)]],
    texture2d<float, access::read> refVarsTex [[texture(2)]],
    constant float* std_curve [[buffer(0)]],
    constant float* diff_curve [[buffer(1)]],
    texture2d<float, access::write> d_sq_Tex [[texture(3)]],
    texture2d<float, access::write> sigma_sq_Tex [[texture(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= d_sq_Tex.get_width() || gid.y >= d_sq_Tex.get_height()) return;
    
    float3 d_p = d_p_Tex.read(gid).rgb;
    float3 ref_means = refMeansTex.read(gid).rgb;
    float3 ref_vars = refVarsTex.read(gid).rgb;
    
    float d_sq_acc = 0.0f;
    float sigma_sq_acc = 0.0f;
    
    for (int c = 0; c < 3; ++c) {
        float brightness = ref_means[c];
        int id_noise = clamp((int)round(1000.0f * brightness), 0, 1000);
        
        float sigma_t = std_curve[id_noise];
        float d_t = diff_curve[id_noise];
        float sigma_p_sq = ref_vars[c];
        
        sigma_sq_acc += max(sigma_p_sq, sigma_t * sigma_t);
        
        float d_p_c = d_p[c];
        float d_p_sq = d_p_c * d_p_c;
        float shrink = d_p_sq / (d_p_sq + d_t * d_t);
        d_sq_acc += d_p_sq * shrink * shrink;
    }
    
    d_sq_Tex.write(float4(d_sq_acc, 0, 0, 1), gid);
    sigma_sq_Tex.write(float4(sigma_sq_acc, 0, 0, 1), gid);
}

// --------------------------------------------------------------------------------
// local_min_5x5 and threshold
// --------------------------------------------------------------------------------
struct ThresholdParams {
    float t;
    float s1;
    float s2;
    int tile_size;
    float Mt;
};

kernel void kernel_robustness_threshold(
    texture2d<float, access::read> d_sq_Tex [[texture(0)]],
    texture2d<float, access::read> sigma_sq_Tex [[texture(1)]],
    texture2d<float, access::read> flowTex [[texture(2)]],
    texture2d<float, access::write> R_Tex [[texture(3)]],
    constant ThresholdParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = d_sq_Tex.get_width();
    int h = d_sq_Tex.get_height();
    if (gid.x >= w || gid.y >= h) return;
    
    float min_d_sq = 1e38f;
    float min_sigma_sq = 1e38f;
    
    // 5x5 quad window = radius 2
    for (int i = -2; i <= 2; ++i) {
        int yy = clamp((int)gid.y + i, 0, h - 1);
        for (int j = -2; j <= 2; ++j) {
            int xx = clamp((int)gid.x + j, 0, w - 1);
            
            float d2 = d_sq_Tex.read(uint2(xx, yy)).r;
            float s2 = sigma_sq_Tex.read(uint2(xx, yy)).r;
            
            min_d_sq = min(min_d_sq, d2);
            min_sigma_sq = min(min_sigma_sq, s2);
        }
    }
    
    float R_val = clamp(params.s1 * exp(-min_d_sq / min_sigma_sq) - params.t, 0.0f, 1.0f);
    
    // Scale by S multiplier
    int patch_idy = min((int)gid.y / params.tile_size, (int)flowTex.get_height() - 1);
    int patch_idx = min((int)gid.x / params.tile_size, (int)flowTex.get_width() - 1);
    
    float s_multiplier = params.s2;
    
    // S computation (variance of flow around the patch)
    float2 flow_c = flowTex.read(uint2(patch_idx, patch_idy)).rg;
    float mnx = flow_c.x, mny = flow_c.y;
    float mxx = flow_c.x, mxy = flow_c.y;
    
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int yy = clamp(patch_idy + i, 0, (int)flowTex.get_height() - 1);
            int xx = clamp(patch_idx + j, 0, (int)flowTex.get_width() - 1);
            float2 f = flowTex.read(uint2(xx, yy)).rg;
            mnx = min(mnx, f.x); mny = min(mny, f.y);
            mxx = max(mxx, f.x); mxy = max(mxy, f.y);
        }
    }
    
    float diff = (mxx - mnx) * (mxx - mnx) + (mxy - mny) * (mxy - mny);
    if (diff > params.Mt) {
        s_multiplier = 1.0f; // matches python s1 implementation implicitly if exceeded
        // Wait, Python is S = s1 if diff > Mt else s2. So s_multiplier = s1.
        s_multiplier = params.s1; // Ah, the Python S is just the multiplier on the exp() part.
        // Wait, S is the multiplier for R. It's either 1.0 or s2! Let's check python.
        // Actually, Python is R = clamp(S * exp(...) - t). 
        // So S is either s1 or s2.
        // Wait, if diff > Mt, S = s1. Else S = s2.
    }
    
    // Recalculate R with the correct S multiplier
    R_val = clamp(s_multiplier * exp(-min_d_sq / min_sigma_sq) - params.t, 0.0f, 1.0f);
    
    R_Tex.write(float4(R_val, 0, 0, 1), gid);
}
