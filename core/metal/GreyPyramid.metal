#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------------
// compute_grey_decimate
// --------------------------------------------------------------------------------
kernel void kernel_grey_decimate(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    uint2 in_gid = gid * 2;
    
    float tl = inTexture.read(in_gid + uint2(0, 0)).r;
    float tr = inTexture.read(in_gid + uint2(1, 0)).r;
    float bl = inTexture.read(in_gid + uint2(0, 1)).r;
    float br = inTexture.read(in_gid + uint2(1, 1)).r;

    float out_val = 0.25f * (tl + tr + bl + br);
    outTexture.write(float4(out_val, 0.0, 0.0, 1.0), gid);
}

// --------------------------------------------------------------------------------
// downsample_by (Gaussian blur + subsample)
// --------------------------------------------------------------------------------
struct DownsampleParams {
    int factor;
    int ksize;
    int radius;
};

// A 2D convolution that computes the result directly at the subsampled locations.
// Because the radius can be dynamic (based on factor), we pass the kernel weights as a buffer.
// The weights buffer holds a 1D Gaussian kernel. We apply it separably in 2D mathematically
// by doing a double loop: weight_2d = kernel1d[dy] * kernel1d[dx]
kernel void kernel_downsample(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant float* weights1D [[buffer(0)]],
    constant DownsampleParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    int factor = params.factor;
    int radius = params.radius;

    // The output pixel (gid.x, gid.y) corresponds to the input pixel at:
    // in_x = gid.x * factor + radius
    // in_y = gid.y * factor + radius
    // (Because the VALID convolution shifts the coordinate frame by 'radius')
    
    int in_center_x = gid.x * factor + radius;
    int in_center_y = gid.y * factor + radius;

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
