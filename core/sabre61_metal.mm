// GCam 6.1 SABRE on Metal (kernels: core/Sabre61Kernels.metal): the per-frame stages (guide + kernel,
// rejection, dilate, accumulated weights) and the per-band merge + denoiser. Same contracts as the CPU
// versions in core/sabre61.cpp; every entry returns false (the caller falls back to the CPU) whenever the
// device, library, a pipeline or a buffer is unavailable or a command buffer fails.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "sabre61.h"
#include <algorithm>
#include <cstring>
#include <mutex>

namespace hhsr {
namespace sabre61 {

namespace {
struct S61Gpu {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> merge = nil;
    id<MTLComputePipelineState> denoise = nil;
    id<MTLComputePipelineState> guide = nil;
    id<MTLComputePipelineState> reject = nil;
    id<MTLComputePipelineState> dilate = nil;
    bool ok = false;
    // Burst state: the base frame's guide (read by every rejection pass) and the accumulated frame weights
    // stay resident between frames.
    id<MTLBuffer> base_ref4 = nil;
    id<MTLBuffer> acc = nil;
    int Wg = 0, Hg = 0;
};
S61Gpu& gpu() {
    static S61Gpu g;
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
        g.merge = make(@"s61_merge_kernel");
        g.denoise = make(@"s61_denoise_kernel");
        g.guide = make(@"s61_guide_kernel");
        g.reject = make(@"s61_rejection_kernel");
        g.dilate = make(@"s61_dilate_accumulate_kernel");
        g.ok = g.merge && g.denoise && g.guide && g.reject && g.dilate;
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
}  // namespace

static S61GuideParams guide_params(const Image& raw, const S61Cfa& cfa) {
    S61GuideParams P{};
    P.Wr = raw.w; P.Hr = raw.h; P.row0 = 0; P.nrows = raw.h; P.cfa = cfa;
    P.nsb[0] = 0.9f; P.nsb[1] = 0.5f; P.nsb[2] = 0.05f; P.nsb[3] = 0.25f;
    return P;
}

bool metal_base_guide(const Image& ref, const S61Cfa& cfa, const float lut[80], Guide& out,
                      std::vector<uint16_t>& cov_h) {
    @autoreleasepool {
        S61Gpu& g = gpu();
        if (!g.ok || ref.data.size() < (size_t)ref.w * ref.h) return false;
        id<MTLDevice> d = g.device;
        const int Wg = ref.w / 2, Hg = ref.h / 2;
        const size_t n = (size_t)Wg * Hg;
        id<MTLBuffer> raw = buf(d, ref.data.data(), (size_t)ref.w * ref.h * 4);
        id<MTLBuffer> lb = buf(d, lut, 80 * 4);
        id<MTLBuffer> r4 = buf(d, nullptr, n * 16), c3 = buf(d, nullptr, n * 6);
        id<MTLBuffer> acc = buf(d, nullptr, n);
        if (!raw || !lb || !r4 || !c3 || !acc) return false;
        std::memset(acc.contents, 0, n);
        const S61GuideParams P = guide_params(ref, cfa);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.guide];
        [e setBuffer:raw offset:0 atIndex:0];
        [e setBuffer:lb offset:0 atIndex:1];
        [e setBytes:&P length:sizeof(P) atIndex:2];
        [e setBuffer:r4 offset:0 atIndex:3];
        [e setBuffer:c3 offset:0 atIndex:4];
        dispatch2d(e, g.guide, Wg, Hg);
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return false;
        out.w = Wg; out.h = Hg;
        out.ref4.assign((const float*)r4.contents, (const float*)r4.contents + n * 4);   // CPU-fallback copy
        out.cov3.clear();
        cov_h.assign((const uint16_t*)c3.contents, (const uint16_t*)c3.contents + n * 3);
        g.base_ref4 = r4; g.acc = acc; g.Wg = Wg; g.Hg = Hg;
        return true;
    }
}

bool metal_frame_stages(const Image& comp, const FlowTex& flow, const S61Cfa& cfa, const float lut[80],
                        std::vector<uint16_t>& cov_h, std::vector<uint8_t>& keep,
                        std::vector<uint8_t>& acc_host) {
    @autoreleasepool {
        S61Gpu& g = gpu();
        if (!g.ok || !g.base_ref4 || !g.acc || comp.w / 2 != g.Wg || comp.h / 2 != g.Hg ||
            comp.data.size() < (size_t)comp.w * comp.h)
            return false;
        id<MTLDevice> d = g.device;
        const int Wg = g.Wg, Hg = g.Hg;
        const size_t n = (size_t)Wg * Hg;
        // The accumulated weights may have been advanced on the CPU (fallback frame): resync first.
        if (acc_host.size() == n) std::memcpy(g.acc.contents, acc_host.data(), n);
        id<MTLBuffer> raw = buf(d, comp.data.data(), (size_t)comp.w * comp.h * 4);
        id<MTLBuffer> lb = buf(d, lut, 80 * 4);
        id<MTLBuffer> fl = buf(d, flow.data.data(), flow.data.size() * 4);
        id<MTLBuffer> r4 = buf(d, nullptr, n * 16), c3 = buf(d, nullptr, n * 6);
        id<MTLBuffer> rej = buf(d, nullptr, n), kp = buf(d, nullptr, n);
        if (!raw || !lb || !fl || !r4 || !c3 || !rej || !kp) return false;
        const S61GuideParams GP = guide_params(comp, cfa);
        S61RejParams RP{};
        RP.Wg = Wg; RP.Hg = Hg; RP.fnx = flow.nx; RP.fny = flow.ny;
        for (int k = 0; k < 4; ++k) RP.fso[k] = flow.fso[k];
        RP.mot_thr = S61_MOTION_THR_PX / (float)Wg; RP.ub_thr = S61_UNBLOCK_THR_PX / (float)Wg;
        RP.nsb[0] = 0.9f; RP.nsb[1] = 0.5f; RP.nsb[2] = 0.05f; RP.nsb[3] = 0.25f;
        const int size[2] = {Wg, Hg};
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.guide];
        [e setBuffer:raw offset:0 atIndex:0];
        [e setBuffer:lb offset:0 atIndex:1];
        [e setBytes:&GP length:sizeof(GP) atIndex:2];
        [e setBuffer:r4 offset:0 atIndex:3];
        [e setBuffer:c3 offset:0 atIndex:4];
        dispatch2d(e, g.guide, Wg, Hg);
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:g.reject];
        [e setBuffer:g.base_ref4 offset:0 atIndex:0];
        [e setBuffer:r4 offset:0 atIndex:1];
        [e setBuffer:fl offset:0 atIndex:2];
        [e setBuffer:lb offset:0 atIndex:3];
        [e setBytes:&RP length:sizeof(RP) atIndex:4];
        [e setBuffer:rej offset:0 atIndex:5];
        dispatch2d(e, g.reject, Wg, Hg);
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:g.dilate];
        [e setBuffer:rej offset:0 atIndex:0];
        [e setBuffer:kp offset:0 atIndex:1];
        [e setBuffer:g.acc offset:0 atIndex:2];
        [e setBytes:size length:sizeof(size) atIndex:3];
        dispatch2d(e, g.dilate, Wg, Hg);
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return false;
        cov_h.assign((const uint16_t*)c3.contents, (const uint16_t*)c3.contents + n * 3);
        keep.assign((const uint8_t*)kp.contents, (const uint8_t*)kp.contents + n);
        acc_host.assign((const uint8_t*)g.acc.contents, (const uint8_t*)g.acc.contents + n);
        return true;
    }
}

void metal_end_burst() {
    S61Gpu& g = gpu();
    g.base_ref4 = nil; g.acc = nil; g.Wg = g.Hg = 0;
}

bool merge_denoise_band_metal(const std::vector<const MergeFrame*>& alts, const MergeFrame& base,
                              const BandContext& ctx, int y0, int y1, std::vector<float>& out_rgb) {
    @autoreleasepool {
        S61Gpu& g = gpu();
        if (!g.ok || !ctx.acc) return false;
        id<MTLDevice> d = g.device;
        const int m0 = std::max(0, y0 - kDenoiseHalo), m1 = std::min(ctx.Ho, y1 + kDenoiseHalo);
        const int mh = m1 - m0, Wo = ctx.Wo;
        const size_t band_floats = (size_t)mh * Wo * 3;
        id<MTLBuffer> accC = buf(d, nullptr, band_floats * 4), accW = buf(d, nullptr, band_floats * 4);
        id<MTLBuffer> accWeights = buf(d, ctx.acc->data(), ctx.acc->size());
        if (!accC || !accW || !accWeights) return false;
        std::memset(accC.contents, 0, band_floats * 4);
        std::memset(accW.contents, 0, band_floats * 4);

        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];   // serial: dispatches run in order
        std::vector<id<MTLBuffer>> keep_alive;
        auto merge_frame = [&](const MergeFrame& f, bool is_base) -> bool {
            int r0, r1, g0, g1;
            band_rows(f, ctx, m0, m1, r0, r1, g0, g1);
            id<MTLBuffer> raw = buf(d, f.raw->data.data() + (size_t)r0 * ctx.Wr, (size_t)(r1 - r0) * ctx.Wr * 4);
            id<MTLBuffer> cov = buf(d, nullptr, (size_t)(g1 - g0) * ctx.Wg * 3 * 4);
            id<MTLBuffer> flow = buf(d, f.flow.data.data(), f.flow.data.size() * 4);
            id<MTLBuffer> mask = is_base ? accWeights : buf(d, f.keep.data(), f.keep.size());
            if (!raw || !cov || !flow || !mask) return false;
            float* cp = (float*)cov.contents;
            const uint16_t* src = f.cov_h.data() + (size_t)g0 * ctx.Wg * 3;
            const size_t ncov = (size_t)(g1 - g0) * ctx.Wg * 3;
            for (size_t i = 0; i < ncov; ++i) cp[i] = s61_h2f(src[i]);
            S61MergeParams P{};
            P.Wo = ctx.Wo; P.Ho = ctx.Ho; P.Wr = ctx.Wr; P.Hr = ctx.Hr; P.Wg = ctx.Wg; P.Hg = ctx.Hg;
            P.fnx = f.flow.nx; P.fny = f.flow.ny;
            P.raw_row0 = r0; P.raw_nrows = r1 - r0;
            P.cov_row0 = g0; P.cov_nrows = g1 - g0;
            P.out_row0 = m0; P.is_base = is_base ? 1 : 0; P.use_rejection = ctx.use_rejection ? 1 : 0;
            for (int k = 0; k < 4; ++k) P.fso[k] = f.flow.fso[k];
            P.fbp[0] = 1.5f / ctx.Wr; P.fbp[1] = 1.5f / ctx.Hr; P.fbp[2] = 1.f - 1.5f / ctx.Wr; P.fbp[3] = 1.f - 1.5f / ctx.Hr;
            P.cfa = ctx.cfa;
            [e setComputePipelineState:g.merge];
            [e setBuffer:raw offset:0 atIndex:0];
            [e setBuffer:cov offset:0 atIndex:1];
            [e setBuffer:flow offset:0 atIndex:2];
            [e setBuffer:mask offset:0 atIndex:3];
            [e setBuffer:accWeights offset:0 atIndex:4];
            [e setBytes:&P length:sizeof(P) atIndex:5];
            [e setBuffer:accC offset:0 atIndex:6];
            [e setBuffer:accW offset:0 atIndex:7];
            [e setBytes:&mh length:sizeof(int) atIndex:8];
            dispatch2d(e, g.merge, Wo, mh);
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            keep_alive.push_back(raw); keep_alive.push_back(cov); keep_alive.push_back(flow); keep_alive.push_back(mask);
            return true;
        };
        bool ok = true;
        for (const MergeFrame* f : alts) ok = ok && merge_frame(*f, false);
        ok = ok && merge_frame(base, true);
        if (!ok) { [e endEncoding]; return false; }

        // denoiser, 3 levels: rows computed shrink toward [y0, y1)
        S61DenoiseParams D{};
        D.Wo = ctx.Wo; D.Ho = ctx.Ho; D.Wg = ctx.Wg; D.Hg = ctx.Hg;
        for (int k = 0; k < 4; ++k) D.yuv_noise[k] = ctx.yuv_noise[k];
        D.fb[0] = 1.5f / ctx.Wo; D.fb[1] = 1.5f / ctx.Ho; D.fb[2] = (ctx.Wo - 1.5f) / ctx.Wo; D.fb[3] = (ctx.Ho - 1.5f) / ctx.Ho;
        id<MTLBuffer> cur = accC;
        int cur0 = m0, cur1 = m1;
        id<MTLBuffer> outs[3] = {nil, nil, nil};
        int o0s[3] = {0, 0, 0}, o1s[3] = {0, 0, 0};
        for (int level = 0; level < 3; ++level) {
            const int reach = (level == 0) ? 9 : (level == 1 ? 6 : 0);
            const int o0 = std::max(0, y0 - reach), o1 = std::min(ctx.Ho, y1 + reach), oh = o1 - o0;
            outs[level] = (level == 1) ? accW : buf(d, nullptr, (size_t)oh * Wo * 3 * 4);   // reuse accW
            if (!outs[level]) { [e endEncoding]; return false; }
            D.level = level; D.in_row0 = cur0; D.in_nrows = cur1 - cur0; D.out_row0 = o0;
            [e setComputePipelineState:g.denoise];
            [e setBuffer:cur offset:0 atIndex:0];
            [e setBuffer:accWeights offset:0 atIndex:1];
            [e setBytes:&D length:sizeof(D) atIndex:2];
            [e setBuffer:outs[level] offset:0 atIndex:3];
            [e setBytes:&oh length:sizeof(int) atIndex:4];
            dispatch2d(e, g.denoise, Wo, oh);
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            cur = outs[level]; cur0 = o0; cur1 = o1;
            o0s[level] = o0; o1s[level] = o1;
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return false;
        const size_t n = (size_t)(y1 - y0) * Wo * 3;
        out_rgb.resize(n);
        std::memcpy(out_rgb.data(), outs[2].contents, n * 4);
        return true;
    }
}

}  // namespace sabre61
}  // namespace hhsr
