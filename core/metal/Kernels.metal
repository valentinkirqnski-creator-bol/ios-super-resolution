#include <metal_stdlib>
using namespace metal;

struct KernelParams {
    float alpha;
    float beta;
    float k_detail;
    float k_denoise;
    float D_tr;
    float D_th;
    float k_shrink;
    float k_stretch;
    int selection_law; // 0: Linear, 1: HardThreshold
};

inline void eigen_elmts_2x2(float a, float b, float c, float d, thread float2& l, thread float2& e1, thread float2& e2) {
    float tr = a + d;
    float det = a * d - b * c;
    float delta = sqrt(max(tr * tr - 4.0f * det, 0.0f));
    
    l.x = 0.5f * (tr + delta);
    l.y = 0.5f * (tr - delta);
    
    if (abs(b) > 1e-6f) {
        float norm1 = sqrt(b * b + (l.x - a) * (l.x - a));
        e1 = float2(b, l.x - a) / norm1;
        float norm2 = sqrt(b * b + (l.y - a) * (l.y - a));
        e2 = float2(b, l.y - a) / norm2;
    } else if (abs(c) > 1e-6f) {
        float norm1 = sqrt((l.x - d) * (l.x - d) + c * c);
        e1 = float2(l.x - d, c) / norm1;
        float norm2 = sqrt((l.y - d) * (l.y - d) + c * c);
        e2 = float2(l.y - d, c) / norm2;
    } else {
        e1 = float2(1.0f, 0.0f);
        e2 = float2(0.0f, 1.0f);
    }
}

kernel void kernel_compute_covariances(
    texture2d<float, access::read> rawTex [[texture(0)]],
    texture2d<float, access::write> covTex [[texture(1)]],
    constant KernelParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = covTex.get_width();
    int h = covTex.get_height();
    if (gid.x >= w || gid.y >= h) return;
    
    // generalized anscombe VST inline for the 4x4 raw neighborhood
    auto vst = [&](int rx, int ry) -> float {
        rx = clamp(rx, 0, (int)rawTex.get_width() - 1);
        ry = clamp(ry, 0, (int)rawTex.get_height() - 1);
        float v = rawTex.read(uint2(rx, ry)).r;
        float c = 0.375f * params.alpha * params.alpha + params.beta;
        return (2.0f / params.alpha) * sqrt(max(0.0f, params.alpha * v + c));
    };
    
    // Grey decimate locally for a 3x3 grey window to compute gradients
    float grey[3][3];
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int gx = (int)gid.x + j;
            int gy = (int)gid.y + i;
            
            float p00 = vst(gx * 2, gy * 2);
            float p01 = vst(gx * 2 + 1, gy * 2);
            float p10 = vst(gx * 2, gy * 2 + 1);
            float p11 = vst(gx * 2 + 1, gy * 2 + 1);
            
            grey[i+1][j+1] = 0.25f * (p00 + p01 + p10 + p11);
        }
    }
    
    // Gradients
    float grad_x[2][2];
    float grad_y[2][2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            grad_x[i][j] = grey[i+1][j+1] - grey[i+1][j];
            grad_y[i][j] = grey[i+1][j+1] - grey[i][j+1];
        }
    }
    
    // Structure tensor
    float s00 = 0.0f, s01 = 0.0f, s11 = 0.0f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float gx = grad_x[i][j];
            float gy = grad_y[i][j];
            s00 += gx * gx;
            s01 += gx * gy;
            s11 += gy * gy;
        }
    }
    
    // Eigen
    float2 l, e1, e2;
    eigen_elmts_2x2(s00, s01, s01, s11, l, e1, e2);
    
    // compute k
    float sum = l.x + l.y;
    float A = 1.0f + sqrt(max((l.x - l.y) / (sum == 0.0f ? 1e-12f : sum), 0.0f));
    float D = clamp(1.0f - sqrt(max(l.x, 0.0f)) / params.D_tr + params.D_th, 0.0f, 1.0f);
    
    float kk1, kk2;
    if (params.selection_law == 1) { // HardThreshold
        if (A > 1.95f) { kk1 = 1.0f / params.k_shrink; kk2 = params.k_stretch; }
        else { kk1 = 1.0f; kk2 = 1.0f; }
    } else { // Linear
        kk1 = 1.0f + A / 2.0f * (1.0f / params.k_shrink - 1.0f);
        kk2 = 1.0f + A / 2.0f * (params.k_stretch - 1.0f);
    }
    
    float k1 = params.k_detail * ((1.0f - D) * kk1 + D * params.k_denoise);
    float k2 = params.k_detail * ((1.0f - D) * kk2 + D * params.k_denoise);
    
    float k1s = k1 * k1;
    float k2s = k2 * k2;
    
    float c00 = k1s * e1.x * e1.x + k2s * e2.x * e2.x;
    float c01 = k1s * e1.x * e1.y + k2s * e2.x * e2.y;
    float c11 = k1s * e1.y * e1.y + k2s * e2.y * e2.y;
    
    covTex.write(float4(c00, c01, c11, 1.0f), gid);
}
