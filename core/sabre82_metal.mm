// GCam 8.2 SABRE on Metal (kernels: core/Sabre82Kernels.metal): per-frame guide/kernel, base reference
// colour, rejection, dilate; per-band merge + dehomogenize. Same contracts as the CPU versions in
// core/sabre82.cpp; every entry returns false (the caller falls back to the CPU) whenever the device, library,
// a pipeline or a buffer is unavailable or a command buffer fails.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "sabre82.h"
#include <algorithm>
#include <cstring>
#include <mutex>

namespace hhsr {
namespace sabre82 {

namespace {
struct S82Gpu {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> guide = nil, base_ref = nil, reject = nil, dilate = nil, merge = nil, dehomog = nil;
    bool ok = false;
    // burst state: the base reference colour every rejection pass reads
    id<MTLBuffer> base_ref_buf = nil;
    int Wg = 0, Hg = 0;
};
S82Gpu& gpu() {
    static S82Gpu g;
    static std::once_flag once;
    std::call_once(once, [] {
        g.device = MTLCreateSystemDefaultDevice();
        if (!g.device) return;
        g.queue = [g.device newCommandQueue];
        if (!g.queue) return;
        NSError* err = nil;
        id<MTLLibrary> lib = nil;
        NSString* path = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
        if (path) lib = [g.device newLibraryWithFile:path error:&err];
        if (!lib) lib = [g.device newDefaultLibrary];
        if (!lib) return;
        auto make = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> f = [lib newFunctionWithName:name];
            if (!f) return nil;
            NSError* e2 = nil;
            return [g.device newComputePipelineStateWithFunction:f error:&e2];
        };
        g.guide = make(@"s82_guide_kernel");
        g.base_ref = make(@"s82_base_ref_kernel");
        g.reject = make(@"s82_rejection_kernel");
        g.dilate = make(@"s82_dilate_kernel");
        g.merge = make(@"s82_merge_kernel");
        g.dehomog = make(@"s82_dehomog_kernel");
        g.ok = g.guide && g.base_ref && g.reject && g.dilate && g.merge && g.dehomog;
    });
    return g;
}
id<MTLBuffer> buf(id<MTLDevice> d, const void* src, size_t bytes) {
    if (bytes == 0) bytes = 16;
    return src ? [d newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared]
               : [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}
void dispatch2d(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, int w, int h) {
    const NSUInteger tw = std::min<NSUInteger>(16, p.threadExecutionWidth);
    const NSUInteger th = std::max<NSUInteger>(1, std::min<NSUInteger>(16, p.maxTotalThreadsPerThreadgroup / tw));
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)w + tw - 1) / tw, ((NSUInteger)h + th - 1) / th, 1)
      threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}
void dispatch1d(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, int n) {
    const NSUInteger tw = std::min<NSUInteger>(256, p.maxTotalThreadsPerThreadgroup);
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + tw - 1) / tw, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
}
bool run(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted;
}
void encode_guide(id<MTLComputeCommandEncoder> e, S82Gpu& g, id<MTLBuffer> raw, id<MTLBuffer> lut,
                  const S82GuideParams& P, id<MTLBuffer> r4, id<MTLBuffer> cov) {
    [e setComputePipelineState:g.guide];
    [e setBuffer:raw offset:0 atIndex:0];
    [e setBuffer:lut offset:0 atIndex:1];
    [e setBytes:&P length:sizeof(P) atIndex:2];
    [e setBuffer:r4 offset:0 atIndex:3];
    [e setBuffer:cov offset:0 atIndex:4];
    dispatch2d(e, g.guide, P.Wg, P.Hg);
}
}  // namespace

bool metal_base_stages(const Image& ref, const S82Cfa& cfa, const float lut[80], Guide& guide,
                       std::vector<float>& base_ref) {
    @autoreleasepool {
        S82Gpu& g = gpu();
        if (!g.ok || ref.data.size() < (size_t)ref.w * ref.h) return false;
        id<MTLDevice> d = g.device;
        const S82GuideParams P = guide_params(ref.w, ref.h, cfa);
        const size_t n = (size_t)P.Wg * P.Hg;
        id<MTLBuffer> raw = buf(d, ref.data.data(), (size_t)ref.w * ref.h * 4);
        id<MTLBuffer> lb = buf(d, lut, 80 * 4);
        id<MTLBuffer> r4 = buf(d, nullptr, n * 16), cov = buf(d, nullptr, n * 6), br = buf(d, nullptr, n * 16);
        if (!raw || !lb || !r4 || !cov || !br) return false;
        const int size[2] = {P.Wg, P.Hg};
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encode_guide(e, g, raw, lb, P, r4, cov);
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:g.base_ref];
        [e setBuffer:r4 offset:0 atIndex:0];
        [e setBuffer:br offset:0 atIndex:1];
        [e setBytes:size length:sizeof(size) atIndex:2];
        dispatch2d(e, g.base_ref, P.Wg, P.Hg);
        [e endEncoding];
        if (!run(cb)) return false;
        guide.w = P.Wg; guide.h = P.Hg;
        guide.ref4.assign((const float*)r4.contents, (const float*)r4.contents + n * 4);
        guide.cov.assign((const uint16_t*)cov.contents, (const uint16_t*)cov.contents + n * 3);
        base_ref.assign((const float*)br.contents, (const float*)br.contents + n * 4);
        g.base_ref_buf = br; g.Wg = P.Wg; g.Hg = P.Hg;
        return true;
    }
}

bool metal_frame_stages(const Image& comp, const FlowTex& flow, const S82Cfa& cfa, const float lut[80],
                        float noise_scale, std::vector<uint16_t>& cov_out, std::vector<uint8_t>& keep) {
    @autoreleasepool {
        S82Gpu& g = gpu();
        if (!g.ok || !g.base_ref_buf || comp.w / 2 != g.Wg || comp.h / 2 != g.Hg ||
            comp.data.size() < (size_t)comp.w * comp.h)
            return false;
        id<MTLDevice> d = g.device;
        const S82GuideParams GP = guide_params(comp.w, comp.h, cfa);
        const S82RejParams RP = rej_params(g.Wg, g.Hg, flow, noise_scale);
        const int Wm = std::max(1, g.Wg / 2), Hm = std::max(1, g.Hg / 2);
        const size_t n = (size_t)g.Wg * g.Hg, nm = (size_t)Wm * Hm;
        id<MTLBuffer> raw = buf(d, comp.data.data(), (size_t)comp.w * comp.h * 4);
        id<MTLBuffer> lb = buf(d, lut, 80 * 4);
        id<MTLBuffer> fl = buf(d, flow.data.data(), flow.data.size() * 4);
        id<MTLBuffer> r4 = buf(d, nullptr, n * 16), cov = buf(d, nullptr, n * 6);
        id<MTLBuffer> rej = buf(d, nullptr, n), kp = buf(d, nullptr, nm);
        if (!raw || !lb || !fl || !r4 || !cov || !rej || !kp) return false;
        const int size4[4] = {g.Wg, g.Hg, Wm, Hm};
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encode_guide(e, g, raw, lb, GP, r4, cov);
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:g.reject];
        [e setBuffer:g.base_ref_buf offset:0 atIndex:0];
        [e setBuffer:r4 offset:0 atIndex:1];
        [e setBuffer:fl offset:0 atIndex:2];
        [e setBuffer:lb offset:0 atIndex:3];
        [e setBytes:&RP length:sizeof(RP) atIndex:4];
        [e setBuffer:rej offset:0 atIndex:5];
        dispatch2d(e, g.reject, g.Wg, g.Hg);
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:g.dilate];
        [e setBuffer:rej offset:0 atIndex:0];
        [e setBuffer:kp offset:0 atIndex:1];
        [e setBytes:size4 length:sizeof(size4) atIndex:2];
        dispatch2d(e, g.dilate, Wm, Hm);
        [e endEncoding];
        if (!run(cb)) return false;
        cov_out.assign((const uint16_t*)cov.contents, (const uint16_t*)cov.contents + n * 3);
        keep.assign((const uint8_t*)kp.contents, (const uint8_t*)kp.contents + nm);
        return true;
    }
}

void metal_end_burst() {
    S82Gpu& g = gpu();
    g.base_ref_buf = nil; g.Wg = g.Hg = 0;
}

bool merge_band_metal(const std::vector<const MergeFrame*>& frames, const BandContext& ctx, int y0, int y1,
                      std::vector<float>& out_rgb) {
    @autoreleasepool {
        S82Gpu& g = gpu();
        if (!g.ok || frames.empty()) return false;
        id<MTLDevice> d = g.device;
        const int bh = y1 - y0, Wo = ctx.Wo;
        const int nfloat = bh * Wo * 3;
        id<MTLBuffer> accC = buf(d, nullptr, (size_t)nfloat * 4), accW = buf(d, nullptr, (size_t)nfloat * 4);
        if (!accC || !accW) return false;
        std::memset(accC.contents, 0, (size_t)nfloat * 4);
        std::memset(accW.contents, 0, (size_t)nfloat * 4);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        std::vector<id<MTLBuffer>> keep_alive;
        for (size_t fi = 0; fi < frames.size(); ++fi) {
            const MergeFrame& f = *frames[fi];
            const bool is_base = (fi + 1 == frames.size());
            int r0, r1, g0, g1;
            band_rows(f, ctx, y0, y1, r0, r1, g0, g1);
            id<MTLBuffer> raw = buf(d, f.raw->data.data() + (size_t)r0 * ctx.Wr, (size_t)(r1 - r0) * ctx.Wr * 4);
            id<MTLBuffer> cov = buf(d, nullptr, (size_t)(g1 - g0) * ctx.Wg * 3 * 4);
            id<MTLBuffer> fl = buf(d, f.flow.data.data(), f.flow.data.size() * 4);
            id<MTLBuffer> mk = is_base ? buf(d, nullptr, 16) : buf(d, f.keep.data(), f.keep.size());
            if (!raw || !cov || !fl || !mk) { [e endEncoding]; return false; }
            float* cp = (float*)cov.contents;
            const uint16_t* src = f.cov.data() + (size_t)g0 * ctx.Wg * 3;
            const size_t ncov = (size_t)(g1 - g0) * ctx.Wg * 3;
            for (size_t i = 0; i < ncov; ++i) cp[i] = (float)src[i] * (1.0f / 1023.0f);
            const S82MergeParams P = merge_params(f, ctx, is_base, r0, r1, g0, g1, y0);
            [e setComputePipelineState:g.merge];
            [e setBuffer:raw offset:0 atIndex:0];
            [e setBuffer:cov offset:0 atIndex:1];
            [e setBuffer:fl offset:0 atIndex:2];
            [e setBuffer:mk offset:0 atIndex:3];
            [e setBytes:&P length:sizeof(P) atIndex:4];
            [e setBuffer:accC offset:0 atIndex:5];
            [e setBuffer:accW offset:0 atIndex:6];
            [e setBytes:&bh length:sizeof(int) atIndex:7];
            dispatch2d(e, g.merge, Wo, bh);
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            keep_alive.push_back(raw); keep_alive.push_back(cov); keep_alive.push_back(fl); keep_alive.push_back(mk);
        }
        [e setComputePipelineState:g.dehomog];
        [e setBuffer:accC offset:0 atIndex:0];
        [e setBuffer:accW offset:0 atIndex:1];
        [e setBytes:&nfloat length:sizeof(int) atIndex:2];
        dispatch1d(e, g.dehomog, nfloat);
        [e endEncoding];
        if (!run(cb)) return false;
        out_rgb.resize((size_t)nfloat);
        std::memcpy(out_rgb.data(), accC.contents, (size_t)nfloat * 4);
        return true;
    }
}

}  // namespace sabre82
}  // namespace hhsr
