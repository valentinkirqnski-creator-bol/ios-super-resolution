// Disk-backed Metal pipeline: align, robustness, merge on GPU with CPU fallback.
#include "MetalContext.h"
#include "../stages.h"
#include "../pipeline.h"
#include "../dng_writer.h"
#include "../snr_tuning.h"
#include "../raw_io.h"
#include "../parallel.h"
#include <vector>
#include <array>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iostream>

namespace fs = std::filesystem;

namespace hhsr {

struct NoiseCurves {
    std::vector<f32> std_curve;
    std::vector<f32> diff_curve;
};
NoiseCurves make_noise_curves_cpu(f32 alpha, f32 beta);

#ifdef __OBJC__

namespace {

static const std::vector<std::string> kRequiredKernels = {
    "kernel_grey_decimate", "kernel_downsample", "kernel_extract_guide",
    "kernel_local_stats_3x3", "kernel_upscale_warp_stats", "kernel_build_dp_guide",
    "kernel_apply_noise_model", "kernel_compute_flow_S", "kernel_robustness_threshold",
    "kernel_local_min_5x5", "kernel_compute_covariances",
    "kernel_block_match_L1", "kernel_block_match_L2", "kernel_block_match_L2_FFT", "kernel_compute_sobel",
    "kernel_ica_refine", "kernel_accumulate_comp_band", "kernel_accumulate_ref_band"
};

static bool save_flow(const fs::path& p, const FlowField& f) {
    int32_t hdr[2] = {f.ny, f.nx};
    std::ofstream out(p, std::ios::binary);
    if (!out) return false;
    out.write((const char*)hdr, sizeof(hdr));
    out.write((const char*)f.flow.data(), (std::streamsize)(f.flow.size() * sizeof(f32)));
    return out.good();
}

static bool load_flow(const fs::path& p, FlowField& f) {
    int32_t hdr[2];
    std::ifstream in(p, std::ios::binary);
    if (!in || !in.read((char*)hdr, sizeof(hdr))) return false;
    f = FlowField(hdr[0], hdr[1]);
    return (bool)in.read((char*)f.flow.data(), (std::streamsize)(f.flow.size() * sizeof(f32)));
}

static bool save_covs(const fs::path& p, const CovField& c) {
    int32_t hdr[2] = {c.h, c.w};
    std::ofstream out(p, std::ios::binary);
    if (!out) return false;
    out.write((const char*)hdr, sizeof(hdr));
    out.write((const char*)c.cov.data(), (std::streamsize)(c.cov.size() * sizeof(f32)));
    return out.good();
}

static bool load_covs(const fs::path& p, CovField& c) {
    int32_t hdr[2];
    std::ifstream in(p, std::ios::binary);
    if (!in || !in.read((char*)hdr, sizeof(hdr))) return false;
    c = CovField(hdr[0], hdr[1]);
    return (bool)in.read((char*)c.cov.data(), (std::streamsize)(c.cov.size() * sizeof(f32)));
}

static bool save_image(const fs::path& p, const Image& im) {
    int32_t hdr[3] = {im.h, im.w, im.c};
    std::ofstream out(p, std::ios::binary);
    if (!out) return false;
    out.write((const char*)hdr, sizeof(hdr));
    out.write((const char*)im.data.data(), (std::streamsize)(im.data.size() * sizeof(f32)));
    return out.good();
}

static bool load_image(const fs::path& p, Image& im) {
    int32_t hdr[3];
    std::ifstream in(p, std::ios::binary);
    if (!in || !in.read((char*)hdr, sizeof(hdr))) return false;
    im = Image(hdr[0], hdr[1], hdr[2]);
    return (bool)in.read((char*)im.data.data(), (std::streamsize)(im.data.size() * sizeof(f32)));
}

struct CachedCompMeta {
    FlowField flow;
    Image rob;
    CovField covs;
    int index = 0;
};

static bool load_cached_comp_meta(const fs::path& cache, int k, CachedCompMeta& out) {
    std::string idx = std::to_string(k);
    fs::path fflow = cache / ("f" + idx + ".flow");
    if (!fs::exists(fflow)) return false;
    out.index = k;
    return load_flow(fflow, out.flow) &&
           load_image(cache / ("f" + idx + ".rob"), out.rob) &&
           load_covs(cache / ("f" + idx + ".cov"), out.covs);
}

static bool load_cached_comp_raw(const fs::path& cache, int k, Image& comp) {
    return load_image(cache / ("f" + std::to_string(k) + ".raw"), comp) && comp.h > 0;
}

static void absorb_robustness_sum(Image& acc_rob, const Image& rob, bool& have) {
    if (!have) {
        acc_rob = Image(rob.h, rob.w, 1);
        acc_rob.data = rob.data;
        have = true;
        return;
    }
    for (size_t i = 0; i < rob.data.size(); ++i)
        acc_rob.data[i] += rob.data[i];
}

static void encode_band_rows(const Image& num_band, const Image& den_band, int y0, int bh,
                             const Config& work, int nch, Image& preview, float pscale,
                             int ph, int pw, int Ws, std::vector<uint16_t>& row16) {
    auto to_srgb = [](f32 v) {
        v = clampf(v, 0.f, 1.f);
        return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
    };
    const int x_step = std::max(1, (int)std::ceil(1.f / std::max(pscale, 1e-6f)));
    parallel_rows(bh, work.num_threads, [&](int i) {
        int gy = y0 + i;
        int py = std::min(ph - 1, (int)(gy * pscale));
        for (int x = 0; x < Ws; ++x) {
            size_t base = ((size_t)i * Ws + x) * 3;
            f32 cn[3] = {0, 0, 0};
            for (int ch = 0; ch < nch; ++ch) {
                f32 d = den_band.at(i, x, ch);
                cn[ch] = (d > 0.f) ? num_band.at(i, x, ch) / d : 0.f;
            }
            f32 outc[3];
            if (work.bake_srgb && nch >= 3) {
                f32 wr = cn[0] * work.white_balance[0];
                f32 wg = cn[1] * work.white_balance[1];
                f32 wb = cn[2] * work.white_balance[2];
                const f32* m = work.cam_to_srgb;
                outc[0] = m[0] * wr + m[1] * wg + m[2] * wb;
                outc[1] = m[3] * wr + m[4] * wg + m[5] * wb;
                outc[2] = m[6] * wr + m[7] * wg + m[8] * wb;
            } else if (nch >= 3) {
                outc[0] = cn[0]; outc[1] = cn[1]; outc[2] = cn[2];
            } else {
                outc[0] = outc[1] = outc[2] = cn[0];
            }
            for (int k = 0; k < 3; ++k) {
                f32 v = work.bake_srgb ? to_srgb(outc[k]) : clampf(outc[k], 0.f, 1.f);
                row16[base + k] = (uint16_t)(v * 65535.f + 0.5f);
            }
            if ((x % x_step) == 0) {
                f32 preview_lin[3] = {outc[0], outc[1], outc[2]};
                if (!work.bake_srgb && nch >= 3) {
                    f32 wr = cn[0] * work.white_balance[0];
                    f32 wg = cn[1] * work.white_balance[1];
                    f32 wb = cn[2] * work.white_balance[2];
                    if (work.has_cam_to_srgb) {
                        const f32* m = work.cam_to_srgb;
                        preview_lin[0] = m[0] * wr + m[1] * wg + m[2] * wb;
                        preview_lin[1] = m[3] * wr + m[4] * wg + m[5] * wb;
                        preview_lin[2] = m[6] * wr + m[7] * wg + m[8] * wb;
                    } else {
                        preview_lin[0] = wr; preview_lin[1] = wg; preview_lin[2] = wb;
                    }
                }
                int px = std::min(pw - 1, (int)(x * pscale));
                for (int k = 0; k < 3; ++k)
                    preview.at(py, px, k) = to_srgb(clampf(preview_lin[k], 0.f, 1.f));
            }
        }
    });
}

struct RefGpuState {
    id<MTLTexture> raw;
    id<MTLTexture> grey;
    std::vector<id<MTLTexture>> pyr;
    id<MTLTexture> guide;
    id<MTLTexture> means_guide;
    id<MTLTexture> vars_guide;
    id<MTLTexture> means_raw;
    id<MTLTexture> vars_raw;
    id<MTLTexture> covs;
    id<MTLTexture> dummy_flow;
};

struct GuideParamsBuf {
    float wb_r;
    float wb_g;
    float wb_b;
    int cfa00;
    int cfa01;
    int cfa10;
    int cfa11;
    int bayer_mode;
};

static GuideParamsBuf make_guide_params(const Config& cfg) {
    GuideParamsBuf p{};
    p.wb_r = cfg.white_balance[0];
    p.wb_g = cfg.white_balance[1];
    p.wb_b = cfg.white_balance[2];
    p.cfa00 = cfg.cfa.p[0][0];
    p.cfa01 = cfg.cfa.p[0][1];
    p.cfa10 = cfg.cfa.p[1][0];
    p.cfa11 = cfg.cfa.p[1][1];
    p.bayer_mode = cfg.bayer_mode ? 1 : 0;
    return p;
}

static bool run_cmd(id<MTLCommandBuffer> cmd) {
    if (!cmd) return false;
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
}

static bool begin_kernel(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
    if (!pso || !enc) return false;
    [enc setComputePipelineState:pso];
    return true;
}

static void dispatch_kernel(id<MTLComputeCommandEncoder> enc, int w, int h, int tg = 16) {
    [enc dispatchThreadgroups:MTLSizeMake((w + tg - 1) / tg, (h + tg - 1) / tg, 1)
         threadsPerThreadgroup:MTLSizeMake(tg, tg, 1)];
}

static id<MTLTexture> compute_grey_metal(id<MTLTexture> in_raw, bool bayer_mode, MetalContext& ctx) {
    if (!bayer_mode) return in_raw;
    id<MTLComputePipelineState> pso = ctx.get_pipeline_state("kernel_grey_decimate");
    int gw = (int)in_raw.width / 2;
    int gh = (int)in_raw.height / 2;
    id<MTLTexture> out_tex = ctx.create_empty_texture(gw, gh, 1, true);
    if (!pso || !out_tex) return nil;
    id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!begin_kernel(enc, pso)) return nil;
    [enc setTexture:in_raw atIndex:0];
    [enc setTexture:out_tex atIndex:1];
    dispatch_kernel(enc, gw, gh);
    [enc endEncoding];
    return run_cmd(cmd) ? out_tex : nil;
}

static id<MTLTexture> compute_downsample_metal(id<MTLTexture> src, int factor, MetalContext& ctx) {
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
    int out_h = ((int)src.height - (ksize - 1)) / factor;
    int out_w = ((int)src.width - (ksize - 1)) / factor;
    id<MTLTexture> out_tex = ctx.create_empty_texture(out_w, out_h, 1, true);
    if (!pso || !out_tex) return nil;
    struct { int factor; int ksize; int radius; } params = { factor, ksize, radius };
    id<MTLBuffer> w_buf = [ctx.device() newBufferWithBytes:kernel1d.data()
                                                    length:kernel1d.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> p_buf = [ctx.device() newBufferWithBytes:&params length:sizeof(params)
                                                   options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!begin_kernel(enc, pso)) return nil;
    [enc setTexture:src atIndex:0];
    [enc setTexture:out_tex atIndex:1];
    [enc setBuffer:w_buf offset:0 atIndex:0];
    [enc setBuffer:p_buf offset:0 atIndex:1];
    dispatch_kernel(enc, out_w, out_h);
    [enc endEncoding];
    return run_cmd(cmd) ? out_tex : nil;
}

static std::vector<id<MTLTexture>> build_pyramid_metal(id<MTLTexture> grey,
                                                         const std::vector<int>& factors,
                                                         MetalContext& ctx) {
    std::vector<id<MTLTexture>> pyr;
    id<MTLTexture> cur = grey;
    for (size_t i = 0; i < factors.size(); ++i) {
        int f = factors[i];
        if (!(f == 1 && i == 0))
            cur = compute_downsample_metal(cur, f, ctx);
        if (!cur) return {};
        pyr.push_back(cur);
    }
    return pyr;
}

static FlowField align_metal_cpu_upscale(const std::vector<id<MTLTexture>>& ref_pyr,
                                         const std::vector<id<MTLTexture>>& mov_pyr,
                                         const Config& cfg, int tile_size, MetalContext& ctx) {
    int nlev = (int)ref_pyr.size();
    FlowField flow_cpu;
    for (int lvl = nlev - 1; lvl >= 0; --lvl) {
        id<MTLTexture> r = ref_pyr[lvl];
        id<MTLTexture> m = mov_pyr[lvl];
        int ts = (lvl < (int)cfg.bm_tile_sizes.size()) ? std::max(4, cfg.bm_tile_sizes[lvl])
                                                       : std::max(4, tile_size);
        int radius = (lvl < (int)cfg.bm_search_radii.size()) ? cfg.bm_search_radii[lvl] : 2;
        int ny = std::max(1, (int)r.height / ts);
        int nx = std::max(1, (int)r.width / ts);

        if (flow_cpu.nx == 0) {
            flow_cpu = FlowField(ny, nx);
        } else {
            int upsample_factor = ((lvl + 1) < (int)cfg.bm_factors.size()) ? cfg.bm_factors[lvl + 1] : 1;
            int prev_ts = ((lvl + 1) < (int)cfg.bm_tile_sizes.size()) ? cfg.bm_tile_sizes[lvl + 1] : ts;
            flow_cpu = upscale_alignment_flow(flow_cpu, ny, nx, upsample_factor, ts, prev_ts);
        }

        id<MTLTexture> flow = ctx.create_texture_from_flow(flow_cpu);
        if (!flow) return FlowField();

        std::string metric = (lvl < (int)cfg.bm_metrics.size()) ? cfg.bm_metrics[lvl] : "L2";
        std::string kernel_name = (metric == "L2") ? "kernel_block_match_L2_FFT" : "kernel_block_match_" + metric;
        id<MTLComputePipelineState> pso_bm = ctx.get_pipeline_state(kernel_name);
        id<MTLTexture> out_bm = ctx.create_empty_texture(nx, ny, 2, true);
        struct { int ts; int search_radius; } p_bm = { ts, radius };
        id<MTLBuffer> bm_buf = [ctx.device() newBufferWithBytes:&p_bm length:sizeof(p_bm)
                                                        options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (!begin_kernel(enc, pso_bm)) return FlowField();
            [enc setTexture:r atIndex:0];
            [enc setTexture:m atIndex:1];
            [enc setTexture:flow atIndex:2];
            [enc setTexture:out_bm atIndex:3];
            [enc setBuffer:bm_buf offset:0 atIndex:0];
            
            if (metric == "L2") {
                [enc dispatchThreadgroups:MTLSizeMake(nx, ny, 1)
                    threadsPerThreadgroup:MTLSizeMake(576, 1, 1)];
            } else {
                dispatch_kernel(enc, nx, ny);
            }
            
            [enc endEncoding];
            if (!run_cmd(cmd)) return FlowField();
        }
        flow = out_bm;

        id<MTLTexture> gx = ctx.create_empty_texture((int)r.width, (int)r.height, 1, true);
        id<MTLTexture> gy = ctx.create_empty_texture((int)r.width, (int)r.height, 1, true);
        id<MTLComputePipelineState> pso_sobel = ctx.get_pipeline_state("kernel_compute_sobel");
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (!begin_kernel(enc, pso_sobel)) return FlowField();
            [enc setTexture:r atIndex:0];
            [enc setTexture:gx atIndex:1];
            [enc setTexture:gy atIndex:2];
            dispatch_kernel(enc, (int)r.width, (int)r.height);
            [enc endEncoding];
            if (!run_cmd(cmd)) return FlowField();
        }

        id<MTLTexture> out_ica = ctx.create_empty_texture(nx, ny, 2, false);
        struct { int ts; int n_iter; } p_ica = { ts, cfg.ica_n_iter };
        id<MTLBuffer> ica_buf = [ctx.device() newBufferWithBytes:&p_ica length:sizeof(p_ica)
                                                         options:MTLResourceStorageModeShared];
        id<MTLComputePipelineState> pso_ica = ctx.get_pipeline_state("kernel_ica_refine");
        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (!begin_kernel(enc, pso_ica)) return FlowField();
            [enc setTexture:r atIndex:0];
            [enc setTexture:gx atIndex:1];
            [enc setTexture:gy atIndex:2];
            [enc setTexture:m atIndex:3];
            [enc setTexture:flow atIndex:4];
            [enc setTexture:out_ica atIndex:5];
            [enc setBuffer:ica_buf offset:0 atIndex:0];
            dispatch_kernel(enc, nx, ny);
            [enc endEncoding];
            if (!run_cmd(cmd)) return FlowField();
        }
        ctx.read_flow_texture(out_ica, flow_cpu);
    }
    return flow_cpu;
}

static bool init_ref_gpu(const Image& ref_raw, const Config& cfg, RefGpuState& ref,
                         id<MTLBuffer> guide_buf, id<MTLBuffer> k_params_buf, MetalContext& ctx) {
    ref.raw = ctx.create_texture(ref_raw);
    if (!ref.raw) return false;
    ref.grey = compute_grey_metal(ref.raw, cfg.bayer_mode, ctx);
    if (!ref.grey) return false;
    ref.pyr = build_pyramid_metal(ref.grey, cfg.bm_factors, ctx);
    if (ref.pyr.empty()) return false;

    int gh = cfg.bayer_mode ? ref_raw.h / 2 : ref_raw.h;
    int gw = cfg.bayer_mode ? ref_raw.w / 2 : ref_raw.w;
    ref.guide = ctx.create_empty_texture(gw, gh, 4, true);
    ref.means_guide = ctx.create_empty_texture(gw, gh, 4, true);
    ref.vars_guide = ctx.create_empty_texture(gw, gh, 4, true);
    ref.means_raw = ctx.create_empty_texture(ref_raw.w, ref_raw.h, 4, true);
    ref.vars_raw = ctx.create_empty_texture(ref_raw.w, ref_raw.h, 4, true);
    ref.covs = ctx.create_empty_texture((int)ref.grey.width, (int)ref.grey.height, 4, true);
    ref.dummy_flow = ctx.create_empty_texture(1, 1, 2, true);
    if (!ref.guide || !ref.means_guide || !ref.vars_guide || !ref.means_raw || !ref.vars_raw || !ref.covs || !ref.dummy_flow)
        return false;

    id<MTLComputePipelineState> pso_guide = ctx.get_pipeline_state("kernel_extract_guide");
    id<MTLComputePipelineState> pso_stats = ctx.get_pipeline_state("kernel_local_stats_3x3");
    id<MTLComputePipelineState> pso_warp = ctx.get_pipeline_state("kernel_upscale_warp_stats");
    id<MTLComputePipelineState> pso_cov = ctx.get_pipeline_state("kernel_compute_covariances");

    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_guide)) return false;
        [enc setTexture:ref.raw atIndex:0];
        [enc setTexture:ref.guide atIndex:1];
        [enc setBuffer:guide_buf offset:0 atIndex:0];
        dispatch_kernel(enc, gw, gh);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_stats)) return false;
        [enc setTexture:ref.guide atIndex:0];
        [enc setTexture:ref.means_guide atIndex:1];
        [enc setTexture:ref.vars_guide atIndex:2];
        dispatch_kernel(enc, gw, gh);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        struct { int raw_h; int raw_w; int upscale; int tile_size; int is_ref; } wp = {
            ref_raw.h, ref_raw.w, cfg.bayer_mode ? 2 : 1, 0, 1
        };
        id<MTLBuffer> wp_buf = [ctx.device() newBufferWithBytes:&wp length:sizeof(wp)
                                                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_warp)) return false;
        [enc setTexture:ref.means_guide atIndex:0];
        [enc setTexture:ref.dummy_flow atIndex:1];
        [enc setTexture:ref.means_raw atIndex:2];
        [enc setBuffer:wp_buf offset:0 atIndex:0];
        dispatch_kernel(enc, ref_raw.w, ref_raw.h);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        struct { int raw_h; int raw_w; int upscale; int tile_size; int is_ref; } wp = {
            ref_raw.h, ref_raw.w, cfg.bayer_mode ? 2 : 1, 0, 1
        };
        id<MTLBuffer> wp_buf = [ctx.device() newBufferWithBytes:&wp length:sizeof(wp)
                                                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_warp)) return false;
        [enc setTexture:ref.vars_guide atIndex:0];
        [enc setTexture:ref.dummy_flow atIndex:1];
        [enc setTexture:ref.vars_raw atIndex:2];
        [enc setBuffer:wp_buf offset:0 atIndex:0];
        dispatch_kernel(enc, ref_raw.w, ref_raw.h);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_cov)) return false;
        [enc setTexture:ref.raw atIndex:0];
        [enc setTexture:ref.covs atIndex:1];
        [enc setBuffer:k_params_buf offset:0 atIndex:0];
        dispatch_kernel(enc, (int)ref.covs.width, (int)ref.covs.height);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    return true;
}

static bool compute_robustness_metal(id<MTLTexture> comp_raw, id<MTLTexture> flow_tex,
                                     const RefGpuState& ref, int tile_size, const Config& cfg,
                                     id<MTLBuffer> guide_buf, id<MTLBuffer> std_buf,
                                     id<MTLBuffer> diff_buf, Image& out_rob, MetalContext& ctx) {
    int gh = cfg.bayer_mode ? (int)comp_raw.height / 2 : (int)comp_raw.height;
    int gw = cfg.bayer_mode ? (int)comp_raw.width / 2 : (int)comp_raw.width;
    int raw_h = (int)comp_raw.height;
    int raw_w = (int)comp_raw.width;

    id<MTLTexture> guide = ctx.create_empty_texture(gw, gh, 4, true);
    id<MTLTexture> comp_means = ctx.create_empty_texture(gw, gh, 4, true);
    id<MTLTexture> comp_vars = ctx.create_empty_texture(gw, gh, 4, true);
    id<MTLTexture> comp_means_raw = ctx.create_empty_texture(raw_w, raw_h, 4, true);
    id<MTLTexture> d_p = ctx.create_empty_texture(raw_w, raw_h, 4, true);
    id<MTLTexture> d_sq = ctx.create_empty_texture(raw_w, raw_h, 1, true);
    id<MTLTexture> sigma_sq = ctx.create_empty_texture(raw_w, raw_h, 1, true);
    id<MTLTexture> s_tex = ctx.create_empty_texture((int)flow_tex.width, (int)flow_tex.height, 1, true);
    id<MTLTexture> r_pre = ctx.create_empty_texture(raw_w, raw_h, 1, true);
    id<MTLTexture> r_out = ctx.create_empty_texture(raw_w, raw_h, 1, false);
    if (!guide || !comp_means || !comp_means_raw || !d_p || !d_sq || !sigma_sq || !s_tex || !r_out)
        return false;

    id<MTLComputePipelineState> pso_guide = ctx.get_pipeline_state("kernel_extract_guide");
    id<MTLComputePipelineState> pso_stats = ctx.get_pipeline_state("kernel_local_stats_3x3");
    id<MTLComputePipelineState> pso_warp = ctx.get_pipeline_state("kernel_upscale_warp_stats");
    id<MTLComputePipelineState> pso_dp = ctx.get_pipeline_state("kernel_build_dp_guide");
    id<MTLComputePipelineState> pso_noise = ctx.get_pipeline_state("kernel_apply_noise_model");
    id<MTLComputePipelineState> pso_s = ctx.get_pipeline_state("kernel_compute_flow_S");
    id<MTLComputePipelineState> pso_thr = ctx.get_pipeline_state("kernel_robustness_threshold");
    id<MTLComputePipelineState> pso_min = ctx.get_pipeline_state("kernel_local_min_5x5");

    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_guide)) return false;
        [enc setTexture:comp_raw atIndex:0];
        [enc setTexture:guide atIndex:1];
        [enc setBuffer:guide_buf offset:0 atIndex:0];
        dispatch_kernel(enc, gw, gh);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_stats)) return false;
        [enc setTexture:guide atIndex:0];
        [enc setTexture:comp_means atIndex:1];
        [enc setTexture:comp_vars atIndex:2];
        dispatch_kernel(enc, gw, gh);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        struct { int raw_h; int raw_w; int upscale; int tile_size; int is_ref; } wp = {
            raw_h, raw_w, cfg.bayer_mode ? 2 : 1, tile_size, 0
        };
        id<MTLBuffer> wp_buf = [ctx.device() newBufferWithBytes:&wp length:sizeof(wp)
                                                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_warp)) return false;
        [enc setTexture:comp_means atIndex:0];
        [enc setTexture:flow_tex atIndex:1];
        [enc setTexture:comp_means_raw atIndex:2];
        [enc setBuffer:wp_buf offset:0 atIndex:0];
        dispatch_kernel(enc, raw_w, raw_h);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        struct { int guide_h; int guide_w; int bayer_mode; } dp = { raw_h, raw_w, cfg.bayer_mode ? 1 : 0 };
        id<MTLBuffer> dp_buf = [ctx.device() newBufferWithBytes:&dp length:sizeof(dp)
                                                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_dp)) return false;
        [enc setTexture:ref.means_raw atIndex:0];
        [enc setTexture:comp_means_raw atIndex:1];
        [enc setTexture:d_p atIndex:2];
        [enc setBuffer:dp_buf offset:0 atIndex:0];
        dispatch_kernel(enc, raw_w, raw_h);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_noise)) return false;
        [enc setTexture:d_p atIndex:0];
        [enc setTexture:ref.means_raw atIndex:1];
        [enc setTexture:ref.vars_raw atIndex:2];
        [enc setTexture:d_sq atIndex:3];
        [enc setTexture:sigma_sq atIndex:4];
        [enc setBuffer:std_buf offset:0 atIndex:0];
        [enc setBuffer:diff_buf offset:0 atIndex:1];
        dispatch_kernel(enc, raw_w, raw_h);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        struct { float Mt; float s1; float s2; } sp = { cfg.r_Mt, cfg.r_s1, cfg.r_s2 };
        id<MTLBuffer> sp_buf = [ctx.device() newBufferWithBytes:&sp length:sizeof(sp)
                                                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_s)) return false;
        [enc setTexture:flow_tex atIndex:0];
        [enc setTexture:s_tex atIndex:1];
        [enc setBuffer:sp_buf offset:0 atIndex:0];
        dispatch_kernel(enc, (int)flow_tex.width, (int)flow_tex.height);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        struct { int bayer_mode; int tile_size; float t; } tp = {
            cfg.bayer_mode ? 1 : 0, tile_size, cfg.r_t
        };
        id<MTLBuffer> tp_buf = [ctx.device() newBufferWithBytes:&tp length:sizeof(tp)
                                                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_thr)) return false;
        [enc setTexture:d_sq atIndex:0];
        [enc setTexture:sigma_sq atIndex:1];
        [enc setTexture:s_tex atIndex:2];
        [enc setTexture:r_pre atIndex:3];
        [enc setBuffer:tp_buf offset:0 atIndex:0];
        dispatch_kernel(enc, raw_w, raw_h);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }
    {
        id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!begin_kernel(enc, pso_min)) return false;
        [enc setTexture:r_pre atIndex:0];
        [enc setTexture:r_out atIndex:1];
        dispatch_kernel(enc, raw_w, raw_h);
        [enc endEncoding];
        if (!run_cmd(cmd)) return false;
    }

    out_rob = Image(raw_h, raw_w, 1);
    ctx.read_texture(r_out, out_rob);
    return out_rob.h > 0;
}

static bool estimate_covs_metal(id<MTLTexture> raw, bool bayer_mode, __strong id<MTLTexture>& covs_out,
                                id<MTLBuffer> k_params_buf, MetalContext& ctx) {
    int cov_w = bayer_mode ? (int)raw.width / 2 : (int)raw.width;
    int cov_h = bayer_mode ? (int)raw.height / 2 : (int)raw.height;
    covs_out = ctx.create_empty_texture(cov_w, cov_h, 4, false);
    if (!covs_out) return false;
    id<MTLComputePipelineState> pso = ctx.get_pipeline_state("kernel_compute_covariances");
    id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!begin_kernel(enc, pso)) return false;
    [enc setTexture:raw atIndex:0];
    [enc setTexture:covs_out atIndex:1];
    [enc setBuffer:k_params_buf offset:0 atIndex:0];
    dispatch_kernel(enc, cov_w, cov_h);
    [enc endEncoding];
    return run_cmd(cmd);
}

} // namespace

bool try_process_burst_paths_metal(const std::vector<std::string>& paths, const Config& cfg,
                                   const std::string& dng_path, const ProgressFn& progress,
                                   int maxPreviewDim, Image& out_preview) {
    out_preview = Image();
    if (!cfg.use_metal || paths.size() < 2) return false;

    MetalContext& ctx = MetalContext::instance();
    ctx.init();
    if (!ctx.is_available() || !ctx.validate_pipelines(kRequiredKernels)) {
        std::cerr << "Metal unavailable or missing kernels; falling back to CPU." << std::endl;
        return false;
    }

    Config work = cfg;
    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };

    report("Loading reference frame (Metal)", 0.02f);
    Image ref = load_raw_frame(paths[0], work, true);
    if (ref.h <= 0 || ref.w <= 0) return false;
    tune_config_snr(ref, work);

    const int ref_h = ref.h, ref_w = ref.w;
    const int n = (int)paths.size();
    const int tile_size = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    const int nch = work.bayer_mode ? 3 : 1;

    GuideParamsBuf gp = make_guide_params(work);
    id<MTLBuffer> guide_buf = [ctx.device() newBufferWithBytes:&gp length:sizeof(gp)
                                                         options:MTLResourceStorageModeShared];

    struct KernelParams {
        float alpha; float beta; float k_detail; float k_denoise;
        float D_tr; float D_th; float k_shrink; float k_stretch; int selection_law;
    } k_params = { work.alpha, work.beta, work.k_detail, work.k_denoise,
                   work.D_tr, work.D_th, work.k_shrink, work.k_stretch,
                   (work.selection == SelectionLaw::Linear) ? 0 : 1 };
    id<MTLBuffer> k_params_buf = [ctx.device() newBufferWithBytes:&k_params length:sizeof(k_params)
                                                            options:MTLResourceStorageModeShared];

    NoiseCurves nc = make_noise_curves_cpu(work.alpha, work.beta);
    id<MTLBuffer> std_buf = [ctx.device() newBufferWithBytes:nc.std_curve.data()
                                                      length:nc.std_curve.size() * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> diff_buf = [ctx.device() newBufferWithBytes:nc.diff_curve.data()
                                                       length:nc.diff_curve.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];

    RefGpuState ref_gpu;
    report("Reference: GPU grey + pyramid + robustness", 0.05f);
    if (!init_ref_gpu(ref, work, ref_gpu, guide_buf, k_params_buf, ctx)) {
        std::cerr << "Metal ref init failed; falling back to CPU." << std::endl;
        return false;
    }
    ref = Image();

    fs::path cache = fs::path(dng_path).parent_path() /
                     (fs::path(dng_path).stem().string() + "_cache");
    std::error_code ec;
    fs::remove_all(cache, ec);
    fs::create_directories(cache, ec);

    int n_comp_ok = 0;
    for (int k = 1; k < n; ++k) {
        report("Frame " + std::to_string(k + 1) + ": GPU analyze",
               0.08f + 0.35f * (float)(k - 1) / std::max(1, n - 1));

        Image comp = load_raw_frame(paths[k], work, false, ref_h, ref_w);
        if (comp.h <= 0) continue;

        id<MTLTexture> comp_raw = ctx.create_texture(comp);
        if (!comp_raw) continue;
        id<MTLTexture> comp_grey = compute_grey_metal(comp_raw, work.bayer_mode, ctx);
        if (!comp_grey) continue;
        std::vector<id<MTLTexture>> comp_pyr = build_pyramid_metal(comp_grey, work.bm_factors, ctx);
        if (comp_pyr.empty()) continue;

        FlowField flow = align_metal_cpu_upscale(ref_gpu.pyr, comp_pyr, work, tile_size, ctx);
        if (flow.nx <= 0) continue;
        id<MTLTexture> flow_tex = ctx.create_texture_from_flow(flow);
        if (!flow_tex) continue;

        Image rob;
        if (!work.robustness_enabled) {
            int gh = work.bayer_mode ? ref_h / 2 : ref_h;
            int gw = work.bayer_mode ? ref_w / 2 : ref_w;
            rob = Image(gh, gw, 1);
            std::fill(rob.data.begin(), rob.data.end(), 1.f);
        } else if (!compute_robustness_metal(comp_raw, flow_tex, ref_gpu, tile_size, work,
                                             guide_buf, std_buf, diff_buf, rob, ctx)) {
            continue;
        }

        id<MTLTexture> comp_covs = nil;
        if (!estimate_covs_metal(comp_raw, work.bayer_mode, comp_covs, k_params_buf, ctx)) continue;
        CovField covs;
        Image cov_img;
        ctx.read_texture(comp_covs, cov_img);
        covs = CovField(cov_img.h, cov_img.w);
        for (int y = 0; y < covs.h; ++y) {
            for (int x = 0; x < covs.w; ++x) {
                covs.at(y, x)[0] = cov_img.at(y, x, 0);
                covs.at(y, x)[1] = cov_img.at(y, x, 1);
                covs.at(y, x)[2] = 0.f;
                covs.at(y, x)[3] = cov_img.at(y, x, 2);
            }
        }

        std::string idx = std::to_string(k);
        if (!save_flow(cache / ("f" + idx + ".flow"), flow) ||
            !save_image(cache / ("f" + idx + ".rob"), rob) ||
            !save_covs(cache / ("f" + idx + ".cov"), covs) ||
            !save_image(cache / ("f" + idx + ".raw"), comp)) {
            continue;
        }
        n_comp_ok++;
    }

    ref_gpu.pyr.clear();

    if (n_comp_ok < 1) {
        fs::remove_all(cache, ec);
        report("Error: Metal analyze failed", 1.f);
        return false;
    }

    report("Loading frames for GPU merge", 0.44f);
    ref = load_raw_frame(paths[0], work, false, ref_h, ref_w);
    if (ref.h <= 0) {
        fs::remove_all(cache, ec);
        return false;
    }
    id<MTLTexture> ref_raw_merge = ctx.create_texture(ref);
    if (!ref_raw_merge) return false;

    std::vector<CachedCompMeta> cached_meta;
    cached_meta.reserve(n - 1);
    for (int k = 1; k < n; ++k) {
        CachedCompMeta meta;
        if (load_cached_comp_meta(cache, k, meta))
            cached_meta.push_back(std::move(meta));
    }
    if (cached_meta.empty()) {
        fs::remove_all(cache, ec);
        return false;
    }

    const bool stream_comp_raw = (n - 1) > 4;
    Image acc_rob;
    bool have_acc_rob = false;
    for (const CachedCompMeta& meta : cached_meta)
        absorb_robustness_sum(acc_rob, meta.rob, have_acc_rob);
    id<MTLTexture> acc_rob_tex = nil;
    const bool use_acc_rob = work.accumulated_robustness_denoiser_enabled && have_acc_rob;
    if (use_acc_rob)
        acc_rob_tex = ctx.create_texture(acc_rob);

    const int Hs = (int)std::nearbyint(work.scale * ref.h);
    const int Ws = (int)std::nearbyint(work.scale * ref.w);

    DngStreamWriter writer;
    const std::string& model = work.camera_model.empty() ? std::string("HandheldSR-x2") : work.camera_model;
    const std::string& make = work.camera_make.empty() ? std::string("HandheldSR") : work.camera_make;
    if (!writer.open(dng_path, Ws, Hs, model, work.orientation,
                     work.has_color_matrix ? work.color_matrix : nullptr,
                     work.bayer_mode ? work.white_balance : nullptr,
                     work.bake_srgb, make)) {
        fs::remove_all(cache, ec);
        return false;
    }

    const float pscale = std::min(1.f, (float)maxPreviewDim / (float)std::max(Hs, Ws));
    const int ph = std::max(1, (int)(Hs * pscale));
    const int pw = std::max(1, (int)(Ws * pscale));
    out_preview = Image(ph, pw, 3);

    const size_t band_budget = 64u * 1024u * 1024u;
    const size_t bytes_per_row = (size_t)Ws * nch * 4 * 2;
    int band_rows = (int)std::max<size_t>(4, band_budget / std::max<size_t>(1, bytes_per_row));
    band_rows = std::min(band_rows, Hs);
    std::vector<uint16_t> row16((size_t)band_rows * Ws * 3);

    id<MTLComputePipelineState> pso_comp = ctx.get_pipeline_state("kernel_accumulate_comp_band");
    id<MTLComputePipelineState> pso_ref = ctx.get_pipeline_state("kernel_accumulate_ref_band");
    if (!pso_comp || !pso_ref) {
        writer.close();
        fs::remove_all(cache, ec);
        return false;
    }

    for (int y0 = 0; y0 < Hs; y0 += band_rows) {
        const int bh = std::min(band_rows, Hs - y0);
        id<MTLTexture> num_tex = ctx.create_empty_texture(Ws, bh, 4, false);
        id<MTLTexture> den_tex = ctx.create_empty_texture(Ws, bh, 4, false);
        if (!num_tex || !den_tex) {
            writer.close();
            fs::remove_all(cache, ec);
            return false;
        }
        {
            std::vector<float> zeros((size_t)Ws * bh * 4, 0.f);
            [num_tex replaceRegion:MTLRegionMake2D(0, 0, Ws, bh) mipmapLevel:0
                          withBytes:zeros.data() bytesPerRow:Ws * 4 * sizeof(float)];
            [den_tex replaceRegion:MTLRegionMake2D(0, 0, Ws, bh) mipmapLevel:0
                          withBytes:zeros.data() bytesPerRow:Ws * 4 * sizeof(float)];
        }

        struct MergeParamsBuf {
            int tile_size; float scale; int bayer_mode; int iso_kernel;
            int rad_max; float max_multiplier; float max_frame_count;
            int acc_rob_enabled; int y0; int band_h;
            int cfa00; int cfa01; int cfa10; int cfa11;
        } mp = { tile_size, work.scale, work.bayer_mode ? 1 : 0,
                 (work.kernel == KernelShape::Iso) ? 1 : 0,
                 (int)work.acc_rob_rad_max, work.acc_rob_max_multiplier,
                 work.acc_rob_max_frame_count, use_acc_rob ? 1 : 0,
                 y0, bh,
                 (int)work.cfa.p[0][0], (int)work.cfa.p[0][1],
                 (int)work.cfa.p[1][0], (int)work.cfa.p[1][1] };
        id<MTLBuffer> mp_buf = [ctx.device() newBufferWithBytes:&mp length:sizeof(mp)
                                                        options:MTLResourceStorageModeShared];

        for (const CachedCompMeta& meta : cached_meta) {
            Image comp;
            if (stream_comp_raw) {
                if (!load_cached_comp_raw(cache, meta.index, comp)) continue;
            } else {
                if (!load_image(cache / ("f" + std::to_string(meta.index) + ".raw"), comp))
                    continue;
            }
            id<MTLTexture> comp_tex = ctx.create_texture(comp);
            id<MTLTexture> flow_tex = ctx.create_texture_from_flow(meta.flow);
            id<MTLTexture> rob_tex = ctx.create_texture(meta.rob);
            id<MTLTexture> cov_tex = nil;
            {
                Image cov_img(meta.covs.h, meta.covs.w, 4);
                for (int y = 0; y < meta.covs.h; ++y) {
                    for (int x = 0; x < meta.covs.w; ++x) {
                        cov_img.at(y, x, 0) = meta.covs.at(y, x)[0];
                        cov_img.at(y, x, 1) = meta.covs.at(y, x)[1];
                        cov_img.at(y, x, 2) = meta.covs.at(y, x)[3];
                        cov_img.at(y, x, 3) = 1.f;
                    }
                }
                cov_tex = ctx.create_texture(cov_img);
            }
            if (!comp_tex || !flow_tex || !rob_tex || !cov_tex) {
                writer.close();
                fs::remove_all(cache, ec);
                return false;
            }
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (!begin_kernel(enc, pso_comp)) {
                writer.close();
                fs::remove_all(cache, ec);
                return false;
            }
            [enc setTexture:comp_tex atIndex:0];
            [enc setTexture:flow_tex atIndex:1];
            [enc setTexture:cov_tex atIndex:2];
            [enc setTexture:rob_tex atIndex:3];
            [enc setTexture:num_tex atIndex:4];
            [enc setTexture:den_tex atIndex:5];
            [enc setBuffer:mp_buf offset:0 atIndex:0];
            dispatch_kernel(enc, Ws, bh);
            [enc endEncoding];
            if (!run_cmd(cmd)) {
                writer.close();
                fs::remove_all(cache, ec);
                return false;
            }
        }

        {
            id<MTLCommandBuffer> cmd = [ctx.command_queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (!begin_kernel(enc, pso_ref)) {
                writer.close();
                fs::remove_all(cache, ec);
                return false;
            }
            [enc setTexture:ref_raw_merge atIndex:0];
            [enc setTexture:ref_gpu.covs atIndex:1];
            [enc setTexture:use_acc_rob ? acc_rob_tex : nil atIndex:2];
            [enc setTexture:num_tex atIndex:3];
            [enc setTexture:den_tex atIndex:4];
            [enc setBuffer:mp_buf offset:0 atIndex:0];
            dispatch_kernel(enc, Ws, bh);
            [enc endEncoding];
            if (!run_cmd(cmd)) {
                writer.close();
                fs::remove_all(cache, ec);
                return false;
            }
        }

        Image num_band(bh, Ws, nch), den_band(bh, Ws, nch);
        ctx.read_texture(num_tex, num_band);
        ctx.read_texture(den_tex, den_band);
        if (num_band.c == 4) {
            Image n3(bh, Ws, nch), d3(bh, Ws, nch);
            for (int y = 0; y < bh; ++y) {
                for (int x = 0; x < Ws; ++x) {
                    for (int ch = 0; ch < nch; ++ch) {
                        n3.at(y, x, ch) = num_band.at(y, x, ch);
                        d3.at(y, x, ch) = den_band.at(y, x, ch);
                    }
                }
            }
            num_band = std::move(n3);
            den_band = std::move(d3);
        }

        encode_band_rows(num_band, den_band, y0, bh, work, nch, out_preview, pscale, ph, pw, Ws, row16);
        writer.write_rows(row16.data(), bh);
        report("GPU merging output", 0.48f + 0.50f * (float)(y0 + bh) / Hs);
    }

    writer.close();
    fs::remove_all(cache, ec);
    report("Done (Metal)", 1.f);
    return out_preview.w > 0;
}

#else

bool try_process_burst_paths_metal(const std::vector<std::string>&, const Config&,
                                   const std::string&, const ProgressFn&, int, Image&) {
    return false;
}

#endif

} // namespace hhsr

