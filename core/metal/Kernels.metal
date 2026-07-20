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
    int selection_law;
    int iso_kernel;
};

// Matches core/linalg.h and Python get_eigen_elmts_2x2.
inline void real_polyroots_2(float a, float b, float c, thread float roots[2]) {
    float delta = max(b * b - 4.0f * a * c, 0.0f);
    float s = sqrt(delta);
    float r1 = (-b + s) / (2.0f * a);
    float r2 = (-b - s) / (2.0f * a);
    if (abs(r1) >= abs(r2)) {
        roots[0] = r1;
        roots[1] = r2;
    } else {
        roots[0] = r2;
        roots[1] = r1;
    }
}

inline void eigen_val_2x2(float m00, float m01, float m10, float m11, thread float l[2]) {
    float b = -(m00 + m11);
    float c = m00 * m11 - m01 * m10;
    real_polyroots_2(1.0f, b, c, l);
}

inline void eigen_elmts_2x2(float m00, float m01, float m10, float m11,
                            thread float l[2], thread float2& e1, thread float2& e2) {
    eigen_val_2x2(m00, m01, m10, m11, l);

    if (m01 == 0.0f && m00 == m11) {
        e1 = float2(1.0f, 0.0f);
        e2 = float2(0.0f, 1.0f);
        return;
    }

    float v1x = m00 + m01 - l[1];
    float v1y = m10 + m11 - l[1];

    if (v1x == 0.0f) {
        e1 = float2(0.0f, 1.0f);
        e2 = float2(1.0f, 0.0f);
    } else if (v1y == 0.0f) {
        e1 = float2(1.0f, 0.0f);
        e2 = float2(0.0f, 1.0f);
    } else {
        float norm = sqrt(v1x * v1x + v1y * v1y);
        e1 = float2(v1x / norm, v1y / norm);
        float sign = copysign(1.0f, e1.x);
        e2 = float2(-e1.y * sign, abs(e1.x));
    }
}

kernel void kernel_compute_covariances(
    texture2d<float, access::read> greyTex [[texture(0)]],
    texture2d<float, access::read> gradTex [[texture(1)]],
    texture2d<float, access::write> covTex [[texture(2)]],
    constant KernelParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= covTex.get_width() || gid.y >= covTex.get_height()) return;

    int y = (int)gid.y;
    int x = (int)gid.x;

    float s00 = 0.0f, s01 = 0.0f, s11 = 0.0f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int gy = y - 1 + i;
            int gx = x - 1 + j;
            if (gy < 0 || gy >= (int)gradTex.get_height() ||
                gx < 0 || gx >= (int)gradTex.get_width())
                continue;
            float2 g = gradTex.read(uint2(gx, gy)).rg;
            s00 += g.x * g.x;
            s01 += g.x * g.y;
            s11 += g.y * g.y;
        }
    }

    float l[2];
    float2 e1, e2;
    eigen_elmts_2x2(s00, s01, s01, s11, l, e1, e2);

    float k1, k2;
    if (params.iso_kernel == 1) {
        k1 = params.k_detail;
        k2 = params.k_detail;
    } else {
        float l1 = l[0];
        float l2 = l[1];
        float sum = l1 + l2;
        float A = 1.0f + sqrt(max((l1 - l2) / (sum == 0.0f ? 1e-12f : sum), 0.0f));
        float D = clamp(1.0f - sqrt(max(l1, 0.0f)) / params.D_tr + params.D_th, 0.0f, 1.0f);

        float kk1, kk2;
        if (params.selection_law == 1) {
            if (A > 1.95f) { kk1 = 1.0f / params.k_shrink; kk2 = params.k_stretch; }
            else { kk1 = 1.0f; kk2 = 1.0f; }
        } else {
            kk1 = 1.0f + A / 2.0f * (1.0f / params.k_shrink - 1.0f);
            kk2 = 1.0f + A / 2.0f * (params.k_stretch - 1.0f);
        }
        k1 = params.k_detail * ((1.0f - D) * kk1 + D * params.k_denoise);
        k2 = params.k_detail * ((1.0f - D) * kk2 + D * params.k_denoise);
    }

    float k1s = k1 * k1;
    float k2s = k2 * k2;

    float c00 = k1s * e1.x * e1.x + k2s * e2.x * e2.x;
    float c01 = k1s * e1.x * e1.y + k2s * e2.x * e2.y;
    float c11 = k1s * e1.y * e1.y + k2s * e2.y * e2.y;

    covTex.write(float4(c00, c01, c11, 1.0f), gid);
}
