#include <metal_stdlib>
using namespace metal;

struct GatParams {
    float alpha;
    float beta;
};

kernel void kernel_apply_gat(
    texture2d<float, access::read> rawTex [[texture(0)]],
    texture2d<float, access::write> vstTex [[texture(1)]],
    constant GatParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= vstTex.get_width() || gid.y >= vstTex.get_height()) return;
    float v = rawTex.read(gid).r;
    float c = 0.375f * params.alpha * params.alpha + params.beta;
    float out_v = (2.0f / params.alpha) * sqrt(max(0.0f, params.alpha * v + c));
    vstTex.write(float4(out_v, 0, 0, 1), gid);
}

kernel void kernel_grey_decimate(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    uint2 in_gid = gid * 2;

    float tl = inTexture.read(in_gid + uint2(0, 0)).r;
    float tr = inTexture.read(in_gid + uint2(1, 0)).r;
    float bl = inTexture.read(in_gid + uint2(0, 1)).r;
    float br = inTexture.read(in_gid + uint2(1, 1)).r;

    float out_val = 0.25f * (tl + tr + bl + br);
    outTexture.write(float4(out_val, 0.0, 0.0, 1.0), gid);
}

kernel void kernel_compute_gradients(
    texture2d<float, access::read> greyTex [[texture(0)]],
    texture2d<float, access::write> gradTex [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= gradTex.get_width() || gid.y >= gradTex.get_height()) return;

    int y = (int)gid.y;
    int x = (int)gid.x;

    float tl = greyTex.read(uint2(x, y)).r;
    float tr = greyTex.read(uint2(x + 1, y)).r;
    float bl = greyTex.read(uint2(x, y + 1)).r;
    float br = greyTex.read(uint2(x + 1, y + 1)).r;

    float gx = 0.25f * ((tr - tl) + (br - bl));
    float gy = 0.25f * ((bl - tl) + (br - tr));
    gradTex.write(float4(gx, gy, 0, 1), gid);
}

struct DownsampleParams {
    int factor;
    int ksize;
    int radius;
};

kernel void kernel_downsample(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant float* weights1D [[buffer(0)]],
    constant DownsampleParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int factor = params.factor;
    int radius = params.radius;

    int in_center_x = (int)gid.x * factor + radius;
    int in_center_y = (int)gid.y * factor + radius;

    float acc = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        float wy = weights1D[dy + radius];
        int sy = in_center_y + dy;

        for (int dx = -radius; dx <= radius; ++dx) {
            float wx = weights1D[dx + radius];
            int sx = in_center_x + dx;

            float val = inTexture.read(uint2(sx, sy)).r;
            acc += wy * wx * val;
        }
    }

    outTexture.write(float4(acc, 0.0, 0.0, 1.0), gid);
}
