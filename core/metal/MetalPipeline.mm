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
    MetalContext::instance().init();
    if (!MetalContext::instance().is_available() || burst.empty()) {
        std::cerr << "Metal is not available or burst empty." << std::endl;
        return Image();
    }
    std::cout << "Starting Metal pipeline processing..." << std::endl;
    
#ifdef __OBJC__
    auto& ctx = MetalContext::instance();
    
    // 1. Upload reference frame
    GPUFrame ref;
    ref.raw = ctx.create_texture(burst[0]);
    ref.grey = compute_grey_metal_tex(ref.raw, cfg.bayer_mode, ctx);
    ref.pyr = build_pyramid_metal_tex(ref.grey, cfg.bm_factors, ctx);
    
    // 2. Guide, Means, Vars for Robustness
    ref.guide = ctx.create_empty_texture(ref.raw.width / 2, ref.raw.height / 2, 4); // RGBA for 3 channels
    struct { float r; float g; float b; } wb = { cfg.white_balance[0], cfg.white_balance[1], cfg.white_balance[2] };
    id<MTLBuffer> wb_buf = [ctx.device() newBufferWithBytes:&wb length:sizeof(wb) options:MTLResourceStorageModeShared];
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx.get_pipeline_state("kernel_extract_guide")];
        [enc setTexture:ref.raw atIndex:0];
        [enc setTexture:ref.guide atIndex:1];
        [enc setBuffer:wb_buf offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((ref.guide.width+15)/16, (ref.guide.height+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    
    ref.means = ctx.create_empty_texture(ref.guide.width, ref.guide.height, 4);
    ref.vars = ctx.create_empty_texture(ref.guide.width, ref.guide.height, 4);
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx.get_pipeline_state("kernel_local_stats_3x3")];
        [enc setTexture:ref.guide atIndex:0];
        [enc setTexture:ref.means atIndex:1];
        [enc setTexture:ref.vars atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((ref.guide.width+15)/16, (ref.guide.height+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    
    // 3. Noise curves (CPU -> GPU)
    NoiseCurves nc = make_noise_curves_cpu(cfg.alpha, cfg.beta);
    id<MTLBuffer> std_curve_buf = [ctx.device() newBufferWithBytes:nc.std_curve.data() length:nc.std_curve.size()*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> diff_curve_buf = [ctx.device() newBufferWithBytes:nc.diff_curve.data() length:nc.diff_curve.size()*sizeof(float) options:MTLResourceStorageModeShared];
    
    // 4. Covariances
    ref.covs = ctx.create_empty_texture(ref.grey.width, ref.grey.height, 4);
    struct KernelParams {
        float alpha; float beta; float k_detail; float k_denoise; float D_tr; float D_th; float k_shrink; float k_stretch; int selection_law;
    } k_params = { cfg.alpha, cfg.beta, cfg.k_detail, cfg.k_denoise, cfg.D_tr, cfg.D_th, cfg.k_shrink, cfg.k_stretch, (cfg.selection == SelectionLaw::Linear) ? 0 : 1 };
    id<MTLBuffer> k_params_buf = [ctx.device() newBufferWithBytes:&k_params length:sizeof(k_params) options:MTLResourceStorageModeShared];
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx.get_pipeline_state("kernel_compute_covariances")];
        [enc setTexture:ref.raw atIndex:0];
        [enc setTexture:ref.covs atIndex:1];
        [enc setBuffer:k_params_buf offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((ref.covs.width+15)/16, (ref.covs.height+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    
    // 5. Output accumulation textures
    int out_w = burst[0].w * cfg.scale;
    int out_h = burst[0].h * cfg.scale;
    int channels = cfg.bayer_mode ? 3 : 1;
    id<MTLTexture> num_tex = ctx.create_empty_texture(out_w, out_h, 4);
    id<MTLTexture> den_tex = ctx.create_empty_texture(out_w, out_h, 4);
    {
        std::vector<float> zeros(out_w * out_h * 4, 0.0f);
        [num_tex replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0 withBytes:zeros.data() bytesPerRow:out_w*4*sizeof(float)];
        [den_tex replaceRegion:MTLRegionMake2D(0, 0, out_w, out_h) mipmapLevel:0 withBytes:zeros.data() bytesPerRow:out_w*4*sizeof(float)];
    }
    
    struct MergeParams {
        int tile_size; float scale; int bayer_mode; int iso_kernel; int rad_max; float max_multiplier; float max_frame_count; int acc_rob_enabled;
    } m_params = { 16, cfg.scale, cfg.bayer_mode ? 1 : 0, (cfg.kernel == KernelShape::Iso) ? 1 : 0, (int)cfg.acc_rob_rad_max, cfg.acc_rob_max_multiplier, cfg.acc_rob_max_frame_count, cfg.accumulated_robustness_denoiser_enabled ? 1 : 0 };
    id<MTLBuffer> m_params_buf = [ctx.device() newBufferWithBytes:&m_params length:sizeof(m_params) options:MTLResourceStorageModeShared];
    
    // 6. Accumulate Ref
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx.get_pipeline_state("kernel_accumulate_ref")];
        [enc setTexture:ref.raw atIndex:0];
        [enc setTexture:ref.covs atIndex:1];
        [enc setTexture:nil atIndex:2]; // acc_rob
        [enc setTexture:num_tex atIndex:3];
        [enc setTexture:den_tex atIndex:4];
        [enc setBuffer:m_params_buf offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((out_w+15)/16, (out_h+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    
    // 7. Loop over comps
    for (size_t i = 1; i < burst.size(); ++i) {
        if (progress) progress(0.1f + 0.8f * (float)i / burst.size());
        
        GPUFrame comp;
        comp.raw = ctx.create_texture(burst[i]);
        comp.grey = compute_grey_metal_tex(comp.raw, cfg.bayer_mode, ctx);
        comp.pyr = build_pyramid_metal_tex(comp.grey, cfg.bm_factors, ctx);
        
        id<MTLTexture> flow = align_metal_tex(ref.pyr, comp.pyr, cfg, 16, ctx);
        
        comp.guide = ctx.create_empty_texture(comp.raw.width / 2, comp.raw.height / 2, 4);
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx.get_pipeline_state("kernel_extract_guide")];
            [enc setTexture:comp.raw atIndex:0];
            [enc setTexture:comp.guide atIndex:1];
            [enc setBuffer:wb_buf offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((comp.guide.width+15)/16, (comp.guide.height+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        
        // Robustness
        id<MTLTexture> d_p = ctx.create_empty_texture(ref.guide.width, ref.guide.height, 4);
        struct { int tile_size; } r_params = { 16 };
        id<MTLBuffer> r_params_buf = [ctx.device() newBufferWithBytes:&r_params length:sizeof(r_params) options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx.get_pipeline_state("kernel_warp_dogson_diff")];
            [enc setTexture:ref.guide atIndex:0];
            [enc setTexture:comp.guide atIndex:1];
            [enc setTexture:flow atIndex:2];
            [enc setTexture:d_p atIndex:3];
            [enc setBuffer:r_params_buf offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((d_p.width+15)/16, (d_p.height+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        
        id<MTLTexture> d_sq = ctx.create_empty_texture(ref.guide.width, ref.guide.height, 1);
        id<MTLTexture> sigma_sq = ctx.create_empty_texture(ref.guide.width, ref.guide.height, 1);
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx.get_pipeline_state("kernel_apply_noise_model")];
            [enc setTexture:d_p atIndex:0];
            [enc setTexture:ref.means atIndex:1];
            [enc setTexture:ref.vars atIndex:2];
            [enc setTexture:d_sq atIndex:3];
            [enc setTexture:sigma_sq atIndex:4];
            [enc setBuffer:std_curve_buf offset:0 atIndex:0];
            [enc setBuffer:diff_curve_buf offset:0 atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake((d_sq.width+15)/16, (d_sq.height+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        
        id<MTLTexture> R_tex = ctx.create_empty_texture(ref.guide.width, ref.guide.height, 1);
        struct { float t; float s1; float s2; int tile_size; float Mt; } t_params = { cfg.t, cfg.s1, cfg.s2, 16, cfg.Mt };
        id<MTLBuffer> t_params_buf = [ctx.device() newBufferWithBytes:&t_params length:sizeof(t_params) options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx.get_pipeline_state("kernel_robustness_threshold")];
            [enc setTexture:d_sq atIndex:0];
            [enc setTexture:sigma_sq atIndex:1];
            [enc setTexture:flow atIndex:2];
            [enc setTexture:R_tex atIndex:3];
            [enc setBuffer:t_params_buf offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((R_tex.width+15)/16, (R_tex.height+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        
        // Covariances
        comp.covs = ctx.create_empty_texture(comp.grey.width, comp.grey.height, 4);
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx.get_pipeline_state("kernel_compute_covariances")];
            [enc setTexture:comp.raw atIndex:0];
            [enc setTexture:comp.covs atIndex:1];
            [enc setBuffer:k_params_buf offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((comp.covs.width+15)/16, (comp.covs.height+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        
        // Accumulate Comp
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx.get_pipeline_state("kernel_accumulate_comp")];
            [enc setTexture:comp.raw atIndex:0];
            [enc setTexture:flow atIndex:1];
            [enc setTexture:comp.covs atIndex:2];
            [enc setTexture:R_tex atIndex:3];
            [enc setTexture:num_tex atIndex:4];
            [enc setTexture:den_tex atIndex:5];
            [enc setBuffer:m_params_buf offset:0 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((out_w+15)/16, (out_h+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
    
    // 8. Normalize
    id<MTLTexture> final_tex = ctx.create_empty_texture(out_w, out_h, 4);
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx.get_pipeline_state("kernel_normalize")];
        [enc setTexture:num_tex atIndex:0];
        [enc setTexture:den_tex atIndex:1];
        [enc setTexture:final_tex atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((out_w+15)/16, (out_h+15)/16, 1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    
    // 9. Read back to CPU Image
    Image final_img(out_h, out_w, channels);
    ctx.read_texture(final_tex, final_img);
    
    if (progress) progress(1.0f);
    
    return final_img;
#else
    return Image();
#endif
}

} // namespace hhsr
