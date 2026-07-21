#include <metal_stdlib>
using namespace metal;

inline float dogson_quadratic(float x) {
    float ax = abs(x);
    if (ax <= 0.5f) return -2.0f * ax * ax + 1.0f;
    if (ax <= 1.5f) return ax * ax - 2.5f * ax + 1.5f;
    return 0.0f;
}

struct GuideParams {
    float wb_r;
    float wb_g;
    float wb_b;
    int cfa00;
    int cfa01;
    int cfa10;
    int cfa11;
    int bayer_mode;
};

kernel void kernel_extract_guide(
    texture2d<float, access::read> rawTex [[texture(0)]],
    texture2d<float, access::write> guideTex [[texture(1)]],
    constant GuideParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= guideTex.get_width() || gid.y >= guideTex.get_height()) return;

    float3 white_balance = float3(params.wb_r, params.wb_g, params.wb_b);
    float3 out_rgb = float3(0.0f);
    float gsum = 0.0f;

    int cfa[4] = { params.cfa00, params.cfa01, params.cfa10, params.cfa11 };

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int c = cfa[i * 2 + j];
            float v = rawTex.read(uint2(gid.x * 2 + j, gid.y * 2 + i)).r / white_balance[c];
            if (c == 1) {
                gsum += v;
            } else {
                out_rgb[c] = v;
            }
        }
    }
    out_rgb[1] = 0.5f * gsum;
    guideTex.write(float4(out_rgb, 1.0f), gid);
}

kernel void kernel_local_stats_3x3(
    texture2d<float, access::read> guideTex [[texture(0)]],
    texture2d<float, access::write> meansTex [[texture(1)]],
    texture2d<float, access::write> varsTex [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= meansTex.get_width() || gid.y >= meansTex.get_height()) return;

    float3 s = float3(0.0f);
    float3 s2 = float3(0.0f);

    for (int i = -1; i <= 1; ++i) {
        int yy = clamp((int)gid.y + i, 0, (int)guideTex.get_height() - 1);
        for (int j = -1; j <= 1; ++j) {
            int xx = clamp((int)gid.x + j, 0, (int)guideTex.get_width() - 1);
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

inline float sample_dogson_ch(texture2d<float, access::read> tex, float LR_y, float LR_x, int ch) {
    if (LR_y < 0.0f || LR_y >= (float)tex.get_height() || LR_x < 0.0f || LR_x >= (float)tex.get_width())
        return 1.0f / 0.0f;

    int center_y = (int)round(LR_y);
    int center_x = (int)round(LR_x);

    float buf = 0.0f;
    float w_acc = 0.0f;

    for (int i = -1; i <= 1; ++i) {
        int y_ = clamp(center_y + i, 0, (int)tex.get_height() - 1);
        float dy = (float)y_ - LR_y;
        float wy = dogson_quadratic(dy);

        for (int j = -1; j <= 1; ++j) {
            int x_ = clamp(center_x + j, 0, (int)tex.get_width() - 1);
            float dx = (float)x_ - LR_x;
            float w = wy * dogson_quadratic(dx);
            float v = tex.read(uint2(x_, y_))[ch];
            buf += v * w;
            w_acc += w;
        }
    }
    return (w_acc > 1e-12f) ? buf / w_acc : (1.0f / 0.0f);
}

struct WarpStatsParams {
    int raw_h;
    int raw_w;
    int upscale;
    int tile_size;
    int is_ref;
};

kernel void kernel_upscale_warp_stats(
    texture2d<float, access::read> statsTex [[texture(0)]],
    texture2d<float, access::read> flowTex [[texture(1)]],
    texture2d<float, access::write> outTex [[texture(2)]],
    constant WarpStatsParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= (uint)params.raw_w || gid.y >= (uint)params.raw_h) return;

    int y = (int)gid.y;
    int x = (int)gid.x;

    float flow_x = 0.0f;
    float flow_y = 0.0f;
    if (params.is_ref == 0) {
        int patch_idy = y / params.tile_size;
        int patch_idx = x / params.tile_size;
        float2 flow = flowTex.read(uint2(patch_idx, patch_idy)).rg;
        flow_x = flow.x;
        flow_y = flow.y;
    }

    float LR_y = (y + flow_y + 0.5f) / (float)params.upscale - 0.5f;
    float LR_x = (x + flow_x + 0.5f) / (float)params.upscale - 0.5f;

    if (!(LR_y >= 0.0f && LR_y < (float)statsTex.get_height() &&
          LR_x >= 0.0f && LR_x < (float)statsTex.get_width())) {
        outTex.write(float4(1.0f / 0.0f, 1.0f / 0.0f, 1.0f / 0.0f, 1.0f), gid);
        return;
    }

    float3 out_v = float3(
        sample_dogson_ch(statsTex, LR_y, LR_x, 0),
        sample_dogson_ch(statsTex, LR_y, LR_x, 1),
        sample_dogson_ch(statsTex, LR_y, LR_x, 2)
    );
    outTex.write(float4(out_v, 1.0f), gid);
}

struct DpGuideParams {
    int guide_h;
    int guide_w;
    int bayer_mode;
};

kernel void kernel_build_dp_guide(
    texture2d<float, access::read> refMeansRaw [[texture(0)]],
    texture2d<float, access::read> compMeansRaw [[texture(1)]],
    texture2d<float, access::write> d_pTex [[texture(2)]],
    constant DpGuideParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= (uint)params.guide_w || gid.y >= (uint)params.guide_h) return;

    float3 ref_v = refMeansRaw.read(gid).rgb;
    float3 comp_v = compMeansRaw.read(gid).rgb;
    float3 d_p = abs(ref_v - comp_v);
    d_pTex.write(float4(d_p, 1.0f), gid);
}

kernel void kernel_apply_noise_model(
    texture2d<float, access::read> d_pTex [[texture(0)]],
    texture2d<float, access::read> refMeansTex [[texture(1)]],
    texture2d<float, access::read> refVarsTex [[texture(2)]],
    constant float* std_curve [[buffer(0)]],
    constant float* diff_curve [[buffer(1)]],
    texture2d<float, access::write> d_sq_Tex [[texture(3)]],
    texture2d<float, access::write> sigma_sq_Tex [[texture(4)]],
    constant int& num_channels [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= d_sq_Tex.get_width() || gid.y >= d_sq_Tex.get_height()) return;

    float3 d_p = d_pTex.read(gid).rgb;
    float3 ref_means = refMeansTex.read(gid).rgb;
    float3 ref_vars = refVarsTex.read(gid).rgb;

    float d_sq_acc = 0.0f;
    float sigma_sq_acc = 0.0f;

    int nc = max(num_channels, 1);
    for (int c = 0; c < nc; ++c) {
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

struct FlowSParams {
    float Mt;
    float s1;
    float s2;
};

kernel void kernel_compute_flow_S(
    texture2d<float, access::read> flowTex [[texture(0)]],
    texture2d<float, access::write> sTex [[texture(1)]],
    constant FlowSParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= flowTex.get_width() || gid.y >= flowTex.get_height()) return;

    float mnx = INFINITY, mny = INFINITY, mxx = -INFINITY, mxy = -INFINITY;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int yy = (int)gid.y + i;
            int xx = (int)gid.x + j;
            if (yy < 0 || yy >= (int)flowTex.get_height() || xx < 0 || xx >= (int)flowTex.get_width())
                continue;
            float2 f = flowTex.read(uint2(xx, yy)).rg;
            mnx = min(mnx, f.x);
            mxx = max(mxx, f.x);
            mny = min(mny, f.y);
            mxy = max(mxy, f.y);
        }
    }

    float d0 = mxx - mnx;
    float d1 = mxy - mny;
    float Mt2 = params.Mt * params.Mt;
    float s = (d0 * d0 + d1 * d1 > Mt2) ? params.s1 : params.s2;
    sTex.write(float4(s, 0, 0, 1), gid);
}

struct ThresholdParams {
    float t;
    int tile_size;
    int bayer_mode;
};

kernel void kernel_robustness_threshold(
    texture2d<float, access::read> d_sq_Tex [[texture(0)]],
    texture2d<float, access::read> sigma_sq_Tex [[texture(1)]],
    texture2d<float, access::read> sTex [[texture(2)]],
    texture2d<float, access::write> R_Tex [[texture(3)]],
    constant ThresholdParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= R_Tex.get_width() || gid.y >= R_Tex.get_height()) return;

    float d_sq = d_sq_Tex.read(gid).r;
    float sigma_sq = sigma_sq_Tex.read(gid).r;

    int patch_idy = (int)gid.y / params.tile_size;
    int patch_idx = (int)gid.x / params.tile_size;

    float s = sTex.read(uint2(patch_idx, patch_idy)).r;
    float R_val = clamp(s * exp(-d_sq / sigma_sq) - params.t, 0.0f, 1.0f);
    R_Tex.write(float4(R_val, 0, 0, 1), gid);
}

kernel void kernel_local_min_5x5(
    texture2d<float, access::read> inTex [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;

    float mn = INFINITY;
    for (int i = -2; i <= 2; ++i) {
        int yy = clamp((int)gid.y + i, 0, (int)inTex.get_height() - 1);
        for (int j = -2; j <= 2; ++j) {
            int xx = clamp((int)gid.x + j, 0, (int)inTex.get_width() - 1);
            mn = min(mn, inTex.read(uint2(xx, yy)).r);
        }
    }
    outTex.write(float4(mn, 0, 0, 1), gid);
}
