#include "MetalContext.h"
#include "../stages.h"
#include <iostream>

namespace hhsr {

struct NoiseCurves {
    std::vector<f32> std_curve;
    std::vector<f32> diff_curve;
};
NoiseCurves make_noise_curves_cpu(f32 alpha, f32 beta);

#ifdef __OBJC__

struct GPUFrame {
    id<MTLTexture> raw;
    id<MTLTexture> grey;
    std::vector<id<MTLTexture>> pyr;
    id<MTLTexture> guide;
    id<MTLTexture> means;
    id<MTLTexture> vars;
    id<MTLTexture> covs;
};

// ... copy helpers from before ...
static id<MTLTexture> compute_grey_metal_tex(id<MTLTexture> in_raw, bool bayer_mode, MetalContext& ctx) {
    if (!bayer_mode) return in_raw;
    id<MTLComputePipelineState> pso = ctx.get_pipeline_state("kernel_grey_decimate");
    int gw = in_raw.width / 2;
    int gh = in_raw.height / 2;
    id<MTLTexture> out_tex = ctx.create_empty_texture(gw, gh, 1);
    id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setTexture:in_raw atIndex:0];
    [enc setTexture:out_tex atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((gw+15)/16, (gh+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out_tex;
}

static id<MTLTexture> compute_downsample_metal_tex(id<MTLTexture> src, int factor, MetalContext& ctx) {
    if (factor <= 1) return src;
    id<MTLComputePipelineState> pso = ctx.get_pipeline_state("kernel_downsample");
    float sigma = factor * 0.5f;
    int radius = (int)(4.0f * sigma + 0.5f);
    int ksize = 2 * radius + 1;
    std::vector<f32> kernel1d(ksize);
    f32 ksum = 0.f;
    for (int i = -radius; i <= radius; ++i) {
        kernel1d[i + radius] = std::exp(-0.5f * (float)(i * i) / (sigma * sigma));
        ksum += kernel1d[i + radius];
    }
    for (auto& v : kernel1d) v /= ksum;
    int out_h = (src.height - (ksize - 1)) / factor;
    int out_w = (src.width - (ksize - 1)) / factor;
    id<MTLTexture> out_tex = ctx.create_empty_texture(out_w, out_h, 1);
    struct { int factor; int ksize; int radius; } params = { factor, ksize, radius };
    id<MTLBuffer> w_buf = [ctx.device() newBufferWithBytes:kernel1d.data() length:kernel1d.size()*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> p_buf = [ctx.device() newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setTexture:src atIndex:0];
    [enc setTexture:out_tex atIndex:1];
    [enc setBuffer:w_buf offset:0 atIndex:0];
    [enc setBuffer:p_buf offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((out_w+15)/16, (out_h+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return out_tex;
}

static std::vector<id<MTLTexture>> build_pyramid_metal_tex(id<MTLTexture> grey, const std::vector<int>& factors, MetalContext& ctx) {
    std::vector<id<MTLTexture>> pyr;
    id<MTLTexture> cur = grey;
    for (size_t i = 0; i < factors.size(); ++i) {
        int f = factors[i];
        if (!(f == 1 && i == 0)) {
            cur = compute_downsample_metal_tex(cur, f, ctx);
        }
        pyr.push_back(cur);
    }
    return pyr;
}

static id<MTLTexture> align_metal_tex(const std::vector<id<MTLTexture>>& ref_pyr,
                                      const std::vector<id<MTLTexture>>& mov_pyr,
                                      const Config& cfg, int tile_size, MetalContext& ctx) {
    int nlev = (int)ref_pyr.size();
    id<MTLTexture> flow = nil;
    for (int lvl = nlev - 1; lvl >= 0; --lvl) {
        id<MTLTexture> r = ref_pyr[lvl];
        id<MTLTexture> m = mov_pyr[lvl];
        int ts = (lvl < cfg.bm_tile_sizes.size()) ? std::max(4, cfg.bm_tile_sizes[lvl]) : std::max(4, tile_size);
        int radius = (lvl < cfg.bm_search_radii.size()) ? cfg.bm_search_radii[lvl] : 2;
        int ny = std::max(1, (int)r.height / ts);
        int nx = std::max(1, (int)r.width / ts);
        
        id<MTLTexture> up_flow = ctx.create_empty_texture(nx, ny, 2);
        if (flow == nil) {
            std::vector<float> zeros(nx * ny * 2, 0.0f);
            [up_flow replaceRegion:MTLRegionMake2D(0, 0, nx, ny) mipmapLevel:0 withBytes:zeros.data() bytesPerRow:nx*2*sizeof(float)];
        } else {
            int upsample_factor = ((lvl + 1) < cfg.bm_factors.size()) ? cfg.bm_factors[lvl + 1] : 1;
            int prev_ts = ((lvl + 1) < cfg.bm_tile_sizes.size()) ? cfg.bm_tile_sizes[lvl + 1] : ts;
            struct { int factor; int prev_ts; int ts; } p = { upsample_factor, prev_ts, ts };
            id<MTLBuffer> p_buf = [ctx.device() newBufferWithBytes:&p length:sizeof(p) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx.get_pipeline_state("kernel_upscale_flow")];
            [enc setTexture:flow atIndex:0];
            [enc setTexture:up_flow atIndex:1];
            [enc setBuffer:p_buf offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((nx+15)/16, (ny+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        flow = up_flow;
        
        std::string metric = (lvl < cfg.bm_metrics.size()) ? cfg.bm_metrics[lvl] : "L2";
        std::string k_name = "kernel_block_match_" + metric;
        id<MTLComputePipelineState> pso_bm = ctx.get_pipeline_state(k_name.c_str());
        id<MTLTexture> out_bm_flow = ctx.create_empty_texture(nx, ny, 2);
        struct { int ts; int search_radius; } p_bm = { ts, radius };
        id<MTLBuffer> bm_buf = [ctx.device() newBufferWithBytes:&p_bm length:sizeof(p_bm) options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_bm];
            [enc setTexture:r atIndex:0];
            [enc setTexture:m atIndex:1];
            [enc setTexture:flow atIndex:2];
            [enc setTexture:out_bm_flow atIndex:3];
            [enc setBuffer:bm_buf offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((nx+15)/16, (ny+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        flow = out_bm_flow;
        
        id<MTLTexture> gx = ctx.create_empty_texture(r.width, r.height, 1);
        id<MTLTexture> gy = ctx.create_empty_texture(r.width, r.height, 1);
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx.get_pipeline_state("kernel_compute_sobel")];
            [enc setTexture:r atIndex:0];
            [enc setTexture:gx atIndex:1];
            [enc setTexture:gy atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((r.width+15)/16, (r.height+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        
        id<MTLTexture> out_ica_flow = ctx.create_empty_texture(nx, ny, 2);
        struct { int ts; int n_iter; } p_ica = { ts, cfg.ica_n_iter };
        id<MTLBuffer> ica_buf = [ctx.device() newBufferWithBytes:&p_ica length:sizeof(p_ica) options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx.get_pipeline_state("kernel_ica_refine")];
            [enc setTexture:r atIndex:0];
            [enc setTexture:gx atIndex:1];
            [enc setTexture:gy atIndex:2];
            [enc setTexture:m atIndex:3];
            [enc setTexture:flow atIndex:4];
            [enc setTexture:out_ica_flow atIndex:5];
            [enc setBuffer:ica_buf offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((nx+15)/16, (ny+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        flow = out_ica_flow;
    }
    return flow;
}

#endif // __OBJC__

Image process_burst_metal(const std::vector<Image>& burst, const Config& cfg,
                          const std::string& dng_path, const ProgressFn& progress) {
    (void)burst;
    (void)cfg;
    (void)dng_path;
    (void)progress;
    // In-memory Metal path defers to disk-backed try_process_burst_paths_metal on iOS.
    return Image();
}

} // namespace hhsr
