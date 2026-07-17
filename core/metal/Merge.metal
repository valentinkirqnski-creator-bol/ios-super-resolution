#include <metal_stdlib>
using namespace metal;

struct MergeParams {
    int tile_size;
    float scale;
    int bayer_mode;
    int iso_kernel;
    int rad_max;
    float max_multiplier;
    float max_frame_count;
    int acc_rob_enabled;
    int y0;
    int band_h;
    int cfa00;
    int cfa01;
    int cfa10;
    int cfa11;
};

inline void interp_inv_cov(texture2d<float, access::read> covTex, float kmap_i, float kmap_j, thread float& ixx, thread float& ixy, thread float& iyy) {
    float frac_x = kmap_j - floor(kmap_j);
    float frac_y = kmap_i - floor(kmap_i);
    int fx = max((int)kmap_j, 0);
    int fy = max((int)kmap_i, 0);
    int cx = min(fx + 1, (int)covTex.get_width() - 1);
    int cy = min(fy + 1, (int)covTex.get_height() - 1);
    
    float3 tl = covTex.read(uint2(fx, fy)).xyz;
    float3 tr = covTex.read(uint2(cx, fy)).xyz;
    float3 bl = covTex.read(uint2(fx, cy)).xyz;
    float3 br = covTex.read(uint2(cx, cy)).xyz;
    
    float3 top = tl + frac_x * (tr - tl);
    float3 bot = bl + frac_x * (br - bl);
    float3 c = top + frac_y * (bot - top);
    
    float xx = c.x, xy = c.y, yy = c.z;
    float det = xx * yy - xy * xy;
    
    if (abs(det) > 1e-10f) {
        float inv = 1.0f / det;
        ixx = inv * yy;
        ixy = -inv * xy;
        iyy = inv * xx;
    } else {
        ixx = 1.0f;
        ixy = 0.0f;
        iyy = 1.0f;
    }
}

// --------------------------------------------------------------------------------
// accumulate_comp
// --------------------------------------------------------------------------------
kernel void kernel_accumulate_comp(
    texture2d<float, access::read> imgTex [[texture(0)]],
    texture2d<float, access::read> flowTex [[texture(1)]],
    texture2d<float, access::read> covTex [[texture(2)]],
    texture2d<float, access::read> robTex [[texture(3)]],
    texture2d<float, access::read_write> numTex [[texture(4)]],
    texture2d<float, access::read_write> denTex [[texture(5)]],
    constant MergeParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]] // (hr_j, hr_i)
) {
    if (gid.x >= numTex.get_width() || gid.y >= numTex.get_height()) return;
    
    int hr_j = gid.x;
    int hr_i = gid.y;
    
    float lr_x = ((float)hr_j + 0.5f) / params.scale;
    float lr_y = ((float)hr_i + 0.5f) / params.scale;
    
    int px = (int)(lr_x / (float)params.tile_size);
    int py = (int)(lr_y / (float)params.tile_size);
    int tpy = min(py, (int)flowTex.get_height() - 1);
    int tpx = min(px, (int)flowTex.get_width() - 1);
    
    float2 flow = flowTex.read(uint2(tpx, tpy)).rg;
    float flowx = flow.x, flowy = flow.y;
    
    float rob_scale = (params.bayer_mode == 1) ? 0.5f : 1.0f;
    int i_r = min((int)(lr_y * rob_scale), (int)robTex.get_height() - 1);
    int j_r = min((int)(lr_x * rob_scale), (int)robTex.get_width() - 1);
    float local_r = robTex.read(uint2(j_r, i_r)).r;
    
    float lr_mov_x = lr_x + flowx;
    float lr_mov_y = lr_y + flowy;
    
    int lr_w = imgTex.get_width();
    int lr_h = imgTex.get_height();
    
    if (!(lr_mov_x >= 0.0f && lr_mov_x < (float)lr_w && lr_mov_y >= 0.0f && lr_mov_y < (float)lr_h)) return;
    
    float ixx = 0.0f, ixy = 0.0f, iyy = 0.0f;
    if (params.iso_kernel == 0) {
        float kmap_j, kmap_i;
        if (params.bayer_mode == 1) {
            kmap_j = lr_mov_x / 2.0f - 0.5f;
            kmap_i = lr_mov_y / 2.0f - 0.5f;
        } else {
            kmap_j = lr_mov_x - 0.5f;
            kmap_i = lr_mov_y - 0.5f;
        }
        interp_inv_cov(covTex, kmap_i, kmap_j, ixx, ixy, iyy);
    }
    
    int center_j = (int)lr_mov_x;
    int center_i = (int)lr_mov_y;
    float lr_mov_j = lr_mov_x - 0.5f;
    float lr_mov_i = lr_mov_y - 0.5f;
    
    float3 val = float3(0.0f);
    float3 acc = float3(0.0f);

    int cfa00 = params.cfa00, cfa01 = params.cfa01;
    int cfa10 = params.cfa10, cfa11 = params.cfa11;

    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            int j = center_j + dj;
            int i = center_i + di;

            if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

            int channel = 0;
            if (params.bayer_mode == 1) {
                int pi = i & 1, pj = j & 1;
                channel = (pi == 0) ? ((pj == 0) ? cfa00 : cfa01) : ((pj == 0) ? cfa10 : cfa11);
            }
            float c = imgTex.read(uint2(j, i)).r;
            
            float dist_x = (float)j - lr_mov_j;
            float dist_y = (float)i - lr_mov_i;
            
            float z;
            if (params.iso_kernel == 1) {
                z = 2.0f * (dist_x * dist_x + dist_y * dist_y);
            } else {
                z = ixx * dist_x * dist_x + 2.0f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
            }
            z = max(0.0f, z);
            float w = exp(-0.5f * z);
            if (w == 0.0f && di == 0 && dj == 0) w = 1.0f;
            
            val[channel] += w * local_r * c;
            acc[channel] += w * local_r;
        }
    }
    
    float4 prev_num = numTex.read(gid);
    float4 prev_den = denTex.read(gid);
    
    // We only support up to 3 channels (RGB)
    numTex.write(prev_num + float4(val.x, val.y, val.z, 0), gid);
    denTex.write(prev_den + float4(acc.x, acc.y, acc.z, 0), gid);
}

// --------------------------------------------------------------------------------
// accumulate_ref
// --------------------------------------------------------------------------------
inline int python_round(float x) {
    float floor_x = floor(x);
    float frac = x - floor_x;
    if (frac > 0.5f) return (int)floor_x + 1;
    if (frac < 0.5f) return (int)floor_x;
    int a = (int)floor_x;
    return (a % 2 == 0) ? a : a + 1;
}

kernel void kernel_accumulate_ref(
    texture2d<float, access::read> imgTex [[texture(0)]],
    texture2d<float, access::read> covTex [[texture(1)]],
    texture2d<float, access::read> accRobTex [[texture(2)]], // might be empty/1x1 if not enabled
    texture2d<float, access::read_write> numTex [[texture(3)]],
    texture2d<float, access::read_write> denTex [[texture(4)]],
    constant MergeParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= numTex.get_width() || gid.y >= numTex.get_height()) return;
    
    int hr_j = gid.x;
    int hr_i = gid.y;
    
    float coarse_x = (float)hr_j / params.scale;
    float coarse_y = (float)hr_i / params.scale;
    
    int lr_w = imgTex.get_width();
    int lr_h = imgTex.get_height();
    
    float additional_denoise_power = 1.0f;
    int rad = 1;
    
    if (params.acc_rob_enabled == 1) {
        float rob_scale = (params.bayer_mode == 1) ? 0.5f : 1.0f;
        int ay = min((int)round(coarse_y * rob_scale), (int)accRobTex.get_height() - 1);
        int ax = min((int)round(coarse_x * rob_scale), (int)accRobTex.get_width() - 1);
        
        float local_acc_r = accRobTex.read(uint2(max(0, ax), max(0, ay))).r;
        if (local_acc_r <= params.max_frame_count) {
            additional_denoise_power = params.max_multiplier;
            rad = params.rad_max;
        }
    }
    
    float ixx = 0.0f, ixy = 0.0f, iyy = 0.0f;
    if (params.iso_kernel == 0) {
        float kmap_j, kmap_i;
        if (params.bayer_mode == 1) {
            kmap_j = coarse_x / 2.0f - 0.5f;
            kmap_i = coarse_y / 2.0f - 0.5f;
        } else {
            kmap_j = coarse_x - 0.5f;
            kmap_i = coarse_y - 0.5f;
        }
        interp_inv_cov(covTex, kmap_i, kmap_j, ixx, ixy, iyy);
    }
    
    int center_j = python_round(coarse_x);
    int center_i = python_round(coarse_y);
    float coarse_j = coarse_x - 0.5f;
    float coarse_i = coarse_y - 0.5f;
    
    float3 val = float3(0.0f);
    float3 acc = float3(0.0f);

    int cfa00 = params.cfa00, cfa01 = params.cfa01;
    int cfa10 = params.cfa10, cfa11 = params.cfa11;

    for (int di = -rad; di <= rad; ++di) {
        for (int dj = -rad; dj <= rad; ++dj) {
            int j = center_j + dj;
            int i = center_i + di;

            if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

            int channel = 0;
            if (params.bayer_mode == 1) {
                int pi = i & 1, pj = j & 1;
                channel = (pi == 0) ? ((pj == 0) ? cfa00 : cfa01) : ((pj == 0) ? cfa10 : cfa11);
            }
            float c = imgTex.read(uint2(j, i)).r;
            
            float dist_x = (float)j - coarse_j;
            float dist_y = (float)i - coarse_i;
            
            float z;
            if (params.iso_kernel == 1) {
                z = 2.0f * (dist_x * dist_x + dist_y * dist_y);
            } else {
                z = ixx * dist_x * dist_x + 2.0f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
            }
            z = max(0.0f, z);
            float w = exp(-0.5f * z / additional_denoise_power);
            if (w == 0.0f && di == 0 && dj == 0) w = 1.0f;
            
            val[channel] += w * c;
            acc[channel] += w;
        }
    }
    
    float4 prev_num = numTex.read(gid);
    float4 prev_den = denTex.read(gid);
    
    numTex.write(prev_num + float4(val.x, val.y, val.z, 0), gid);
    denTex.write(prev_den + float4(acc.x, acc.y, acc.z, 0), gid);
}

// Normalize and Bayer Demosaic (Simple fallback or exactly matching CPU?)
// The CPU uses a simple 3x3 demosaic in cli_main or output.
// Actually, `accumulate` outputs Num and Den. The final step is num / den.
kernel void kernel_normalize(
    texture2d<float, access::read> numTex [[texture(0)]],
    texture2d<float, access::read> denTex [[texture(1)]],
    texture2d<float, access::write> outTex [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;
    
    float3 n = numTex.read(gid).rgb;
    float3 d = denTex.read(gid).rgb;
    
    float3 out = float3(0.0f);
    for (int i = 0; i < 3; ++i) {
        out[i] = (d[i] > 1e-12f) ? n[i] / d[i] : 0.0f;
    }
    
    outTex.write(float4(out, 1.0f), gid);
}

kernel void kernel_accumulate_comp_band(
    texture2d<float, access::read> imgTex [[texture(0)]],
    texture2d<float, access::read> flowTex [[texture(1)]],
    texture2d<float, access::read> covTex [[texture(2)]],
    texture2d<float, access::read> robTex [[texture(3)]],
    texture2d<float, access::read_write> numTex [[texture(4)]],
    texture2d<float, access::read_write> denTex [[texture(5)]],
    constant MergeParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.y >= (uint)params.band_h || gid.x >= numTex.get_width()) return;

    int hr_j = (int)gid.x;
    int hr_i = params.y0 + (int)gid.y;

    float lr_x = ((float)hr_j + 0.5f) / params.scale;
    float lr_y = ((float)hr_i + 0.5f) / params.scale;

    int px = (int)(lr_x / (float)params.tile_size);
    int py = (int)(lr_y / (float)params.tile_size);
    int tpy = min(py, (int)flowTex.get_height() - 1);
    int tpx = min(px, (int)flowTex.get_width() - 1);

    float2 flow = flowTex.read(uint2(tpx, tpy)).rg;
    float flowx = flow.x, flowy = flow.y;

    float rob_scale = (params.bayer_mode == 1) ? 0.5f : 1.0f;
    int i_r = min((int)(lr_y * rob_scale), (int)robTex.get_height() - 1);
    int j_r = min((int)(lr_x * rob_scale), (int)robTex.get_width() - 1);
    float local_r = robTex.read(uint2(j_r, i_r)).r;

    float lr_mov_x = lr_x + flowx;
    float lr_mov_y = lr_y + flowy;

    int lr_w = imgTex.get_width();
    int lr_h = imgTex.get_height();

    if (!(lr_mov_x >= 0.0f && lr_mov_x < (float)lr_w && lr_mov_y >= 0.0f && lr_mov_y < (float)lr_h)) return;

    float ixx = 0.0f, ixy = 0.0f, iyy = 0.0f;
    if (params.iso_kernel == 0) {
        float kmap_j, kmap_i;
        if (params.bayer_mode == 1) {
            kmap_j = lr_mov_x / 2.0f - 0.5f;
            kmap_i = lr_mov_y / 2.0f - 0.5f;
        } else {
            kmap_j = lr_mov_x - 0.5f;
            kmap_i = lr_mov_y - 0.5f;
        }
        interp_inv_cov(covTex, kmap_i, kmap_j, ixx, ixy, iyy);
    }

    int center_j = (int)lr_mov_x;
    int center_i = (int)lr_mov_y;
    float lr_mov_j = lr_mov_x - 0.5f;
    float lr_mov_i = lr_mov_y - 0.5f;

    float3 val = float3(0.0f);
    float3 acc = float3(0.0f);

    int cfa00 = params.cfa00, cfa01 = params.cfa01;
    int cfa10 = params.cfa10, cfa11 = params.cfa11;

    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            int j = center_j + dj;
            int i = center_i + di;
            if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

            int channel = 0;
            if (params.bayer_mode == 1) {
                int pi = i & 1, pj = j & 1;
                channel = (pi == 0) ? ((pj == 0) ? cfa00 : cfa01) : ((pj == 0) ? cfa10 : cfa11);
            }
            float c = imgTex.read(uint2(j, i)).r;

            float dist_x = (float)j - lr_mov_j;
            float dist_y = (float)i - lr_mov_i;

            float z;
            if (params.iso_kernel == 1) {
                z = 2.0f * (dist_x * dist_x + dist_y * dist_y);
            } else {
                z = ixx * dist_x * dist_x + 2.0f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
            }
            z = max(0.0f, z);
            float w = exp(-0.5f * z);
            if (w == 0.0f && di == 0 && dj == 0) w = 1.0f;

            val[channel] += w * local_r * c;
            acc[channel] += w * local_r;
        }
    }

    float4 prev_num = numTex.read(gid);
    float4 prev_den = denTex.read(gid);
    numTex.write(prev_num + float4(val.x, val.y, val.z, 0), gid);
    denTex.write(prev_den + float4(acc.x, acc.y, acc.z, 0), gid);
}

kernel void kernel_accumulate_ref_band(
    texture2d<float, access::read> imgTex [[texture(0)]],
    texture2d<float, access::read> covTex [[texture(1)]],
    texture2d<float, access::read> accRobTex [[texture(2)]],
    texture2d<float, access::read_write> numTex [[texture(3)]],
    texture2d<float, access::read_write> denTex [[texture(4)]],
    constant MergeParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.y >= (uint)params.band_h || gid.x >= numTex.get_width()) return;

    int hr_j = (int)gid.x;
    int hr_i = params.y0 + (int)gid.y;

    float coarse_x = (float)hr_j / params.scale;
    float coarse_y = (float)hr_i / params.scale;

    int lr_w = imgTex.get_width();
    int lr_h = imgTex.get_height();

    float additional_denoise_power = 1.0f;
    int rad = 1;

    if (params.acc_rob_enabled == 1 && accRobTex.get_width() > 1) {
        float rob_scale = (params.bayer_mode == 1) ? 0.5f : 1.0f;
        int ay = min((int)round(coarse_y * rob_scale), (int)accRobTex.get_height() - 1);
        int ax = min((int)round(coarse_x * rob_scale), (int)accRobTex.get_width() - 1);

        float local_acc_r = accRobTex.read(uint2(max(0, ax), max(0, ay))).r;
        if (local_acc_r <= params.max_frame_count) {
            additional_denoise_power = params.max_multiplier;
            rad = params.rad_max;
        }
    }

    float ixx = 0.0f, ixy = 0.0f, iyy = 0.0f;
    if (params.iso_kernel == 0) {
        float kmap_j, kmap_i;
        if (params.bayer_mode == 1) {
            kmap_j = coarse_x / 2.0f - 0.5f;
            kmap_i = coarse_y / 2.0f - 0.5f;
        } else {
            kmap_j = coarse_x - 0.5f;
            kmap_i = coarse_y - 0.5f;
        }
        interp_inv_cov(covTex, kmap_i, kmap_j, ixx, ixy, iyy);
    }

    int center_j = python_round(coarse_x);
    int center_i = python_round(coarse_y);
    float coarse_j = coarse_x - 0.5f;
    float coarse_i = coarse_y - 0.5f;

    float3 val = float3(0.0f);
    float3 acc = float3(0.0f);

    int cfa00 = params.cfa00, cfa01 = params.cfa01;
    int cfa10 = params.cfa10, cfa11 = params.cfa11;

    for (int di = -rad; di <= rad; ++di) {
        for (int dj = -rad; dj <= rad; ++dj) {
            int j = center_j + dj;
            int i = center_i + di;
            if (!(j >= 0 && j < lr_w && i >= 0 && i < lr_h)) continue;

            int channel = 0;
            if (params.bayer_mode == 1) {
                int pi = i & 1, pj = j & 1;
                channel = (pi == 0) ? ((pj == 0) ? cfa00 : cfa01) : ((pj == 0) ? cfa10 : cfa11);
            }
            float c = imgTex.read(uint2(j, i)).r;

            float dist_x = (float)j - coarse_j;
            float dist_y = (float)i - coarse_i;

            float z;
            if (params.iso_kernel == 1) {
                z = 2.0f * (dist_x * dist_x + dist_y * dist_y);
            } else {
                z = ixx * dist_x * dist_x + 2.0f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
            }
            z = max(0.0f, z);
            float w = exp(-0.5f * z / additional_denoise_power);
            if (w == 0.0f && di == 0 && dj == 0) w = 1.0f;

            val[channel] += w * c;
            acc[channel] += w;
        }
    }

    float4 prev_num = numTex.read(gid);
    float4 prev_den = denTex.read(gid);
    numTex.write(prev_num + float4(val.x, val.y, val.z, 0), gid);
    denTex.write(prev_den + float4(acc.x, acc.y, acc.z, 0), gid);
}
