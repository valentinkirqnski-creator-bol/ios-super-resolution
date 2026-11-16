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
#include <string>

namespace hhsr {
namespace sabre82 {

namespace {
struct S82Gpu {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> guide = nil, base_ref = nil, reject = nil, dilate = nil, merge = nil, dehomog = nil;
    id<MTLComputePipelineState> resid_factor = nil, resid_apply = nil;   // residual-shift test (optional)
    bool ok = false;
    // burst state: the base reference colour every rejection pass reads
    id<MTLBuffer> base_ref_buf = nil;
    int Wg = 0, Hg = 0;
    // burst state: |grad L| of the base reference (residual-shift test; the rejection pass always reads it)
    id<MTLBuffer> gm_buf = nil;
    const float* gm_src = nullptr;
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
        g.resid_factor = make(@"s82_resid_factor_kernel");
        g.resid_apply = make(@"s82_resid_apply_kernel");
        g.ok = g.guide && g.base_ref && g.reject && g.dilate && g.merge && g.dehomog;
    });
    return g;
}
id<MTLBuffer> buf(id<MTLDevice> d, const void* src, size_t bytes) {
    if (bytes == 0) bytes = 16;
    return src ? [d newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared]
               : [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}
// Mali-G71 bit-exact stages (Config::sabre_mali_exact): core/Sabre82MaliKernels.msl + core/sabre82_shared.h ship
// as bundle resources and are compiled here once, with fast math off, no FMA contraction (S82M_CONTRACT_PRAGMA)
// and software fp16 stores (S82_Q16_SOFT). If the toolchain rejects a variant the next, looser one is tried;
// `status` names the variant that compiled or why none did (the pipeline then runs these stages on the CPU).
struct S82MaliGpu {
    id<MTLComputePipelineState> guide = nil, base_ref = nil, reject = nil, dilate = nil;
    bool ok = false;
    std::string status = "not compiled";
};
S82MaliGpu& mali_gpu() {
    static S82MaliGpu m;
    static std::once_flag once;
    std::call_once(once, [] {
        S82Gpu& g = gpu();
        if (!g.device) { m.status = "no Metal device"; return; }
        NSBundle* b = [NSBundle mainBundle];
        NSString* kp = [b pathForResource:@"Sabre82MaliKernels" ofType:@"msl"];
        NSString* hp = [b pathForResource:@"sabre82_shared" ofType:@"h"];
        if (!kp || !hp) { m.status = "Mali kernel source missing from the app bundle"; return; }
        NSString* ks = [NSString stringWithContentsOfFile:kp encoding:NSUTF8StringEncoding error:nil];
        NSString* hs = [NSString stringWithContentsOfFile:hp encoding:NSUTF8StringEncoding error:nil];
        if (!ks || !hs) { m.status = "cannot read the Mali kernel source"; return; }
        NSString* src = [ks stringByReplacingOccurrencesOfString:@"#include \"sabre82_shared.h\"" withString:hs];
        static const char* kVariant[3] = {"precise math, no contraction", "plain math, no contraction", "plain math"};
        for (int v = 0; v < 3 && !m.ok; ++v) {
            MTLCompileOptions* o = [[MTLCompileOptions alloc] init];
            if (@available(iOS 18.0, macOS 15.0, *)) {   // fastMathEnabled's replacement
                o.mathMode = MTLMathModeSafe;
                o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            } else {
                o.fastMathEnabled = NO;
            }
            NSMutableDictionary* mac = [NSMutableDictionary dictionary];
            mac[@"S82M_GPU"] = @1;
            mac[@"S82_Q16_SOFT"] = @1;
            if (v < 2) mac[@"S82M_CONTRACT_PRAGMA"] = @1;
            if (v >= 1) mac[@"S82M_PLAIN_MATH"] = @1;
            o.preprocessorMacros = mac;
            NSError* err = nil;
            id<MTLLibrary> lib = [g.device newLibraryWithSource:src options:o error:&err];
            if (!lib) {
                std::string e = err ? std::string(err.localizedDescription.UTF8String) : std::string("unknown error");
                if (e.size() > 400) e.resize(400);
                m.status = "compile failed: " + e;
                NSLog(@"SABRE Mali kernels (%s): %s", kVariant[v], e.c_str());
                continue;
            }
            auto make = [&](NSString* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> f = [lib newFunctionWithName:name];
                if (!f) return nil;
                NSError* e2 = nil;
                return [g.device newComputePipelineStateWithFunction:f error:&e2];
            };
            m.guide = make(@"s82m_guide_kernel");
            m.base_ref = make(@"s82m_base_ref_kernel");
            m.reject = make(@"s82m_rejection_kernel");
            m.dilate = make(@"s82m_dilate_kernel");
            m.ok = m.guide && m.base_ref && m.reject && m.dilate;
            m.status = m.ok ? std::string("GPU, ") + kVariant[v] : std::string("pipeline creation failed");
        }
    });
    return m;
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
void encode_guide(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pso, id<MTLBuffer> raw,
                  id<MTLBuffer> lut, const S82GuideParams& P, id<MTLBuffer> r4, id<MTLBuffer> cov) {
    [e setComputePipelineState:pso];
    [e setBuffer:raw offset:0 atIndex:0];
    [e setBuffer:lut offset:0 atIndex:1];
    [e setBytes:&P length:sizeof(P) atIndex:2];
    [e setBuffer:r4 offset:0 atIndex:3];
    [e setBuffer:cov offset:0 atIndex:4];
    dispatch2d(e, pso, P.Wg, P.Hg);
}
}  // namespace

bool metal_mali_available(std::string* status) {
    S82MaliGpu& m = mali_gpu();
    if (status) *status = m.status;
    return m.ok && gpu().ok;
}

bool metal_base_stages(const Image& ref, const S82Cfa& cfa, const float lut[80], const GuideTuning& t,
                       Guide& guide, std::vector<float>& base_ref, bool mali) {
    @autoreleasepool {
        S82Gpu& g = gpu();
        if (!g.ok || ref.data.size() < (size_t)ref.w * ref.h) return false;
        id<MTLComputePipelineState> p_guide = g.guide, p_base = g.base_ref;
        if (mali) {
            S82MaliGpu& m = mali_gpu();
            if (!m.ok) return false;
            p_guide = m.guide; p_base = m.base_ref;
        }
        id<MTLDevice> d = g.device;
        const S82GuideParams P = guide_params(ref.w, ref.h, cfa, t);
        const size_t n = (size_t)P.Wg * P.Hg;
        id<MTLBuffer> raw = buf(d, ref.data.data(), (size_t)ref.w * ref.h * 4);
        id<MTLBuffer> lb = buf(d, lut, 80 * 4);
        id<MTLBuffer> r4 = buf(d, nullptr, n * 16), cov = buf(d, nullptr, n * 6), br = buf(d, nullptr, n * 16);
        if (!raw || !lb || !r4 || !cov || !br) return false;
        const int size[4] = {P.Wg, P.Hg, P.rejection_only, 0};
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encode_guide(e, p_guide, raw, lb, P, r4, cov);
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:p_base];
        [e setBuffer:r4 offset:0 atIndex:0];
        [e setBuffer:br offset:0 atIndex:1];
        [e setBytes:size length:sizeof(size) atIndex:2];
        dispatch2d(e, p_base, P.Wg, P.Hg);
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
                        const GuideTuning& t, float noise_scale, const Unblocker& unb, const ResidShift& rs,
                        std::vector<uint16_t>& cov_out, std::vector<uint8_t>& keep, bool mali) {
    @autoreleasepool {
        S82Gpu& g = gpu();
        if (!g.ok || !g.base_ref_buf || comp.w / 2 != g.Wg || comp.h / 2 != g.Hg ||
            comp.data.size() < (size_t)comp.w * comp.h)
            return false;
        id<MTLComputePipelineState> p_guide = g.guide, p_rej = g.reject, p_dil = g.dilate;
        if (mali) {
            S82MaliGpu& m = mali_gpu();
            if (!m.ok) return false;
            p_guide = m.guide; p_rej = m.reject; p_dil = m.dilate;
        }
        id<MTLDevice> d = g.device;
        const S82GuideParams GP = guide_params(comp.w, comp.h, cfa, t);
        const S82RejParams RP = rej_params(g.Wg, g.Hg, flow, noise_scale, unb, t.rejection_only);
        const int Wm = std::max(1, g.Wg / 2), Hm = std::max(1, g.Hg / 2);
        const size_t n = (size_t)g.Wg * g.Hg, nm = (size_t)Wm * Hm;
        if (unb.data.size() < (size_t)unb.w * unb.h || rs.gm.size() != n) return false;
        if (rs.enabled && (!g.resid_factor || !g.resid_apply)) return false;
        if (!g.gm_buf || g.gm_src != rs.gm.data()) {
            g.gm_buf = buf(d, rs.gm.data(), n * 4);
            g.gm_src = rs.gm.data();
            if (!g.gm_buf) return false;
        }
        id<MTLBuffer> raw = buf(d, comp.data.data(), (size_t)comp.w * comp.h * 4);
        id<MTLBuffer> lb = buf(d, lut, 80 * 4);
        id<MTLBuffer> fl = buf(d, flow.data.data(), flow.data.size() * 4);
        id<MTLBuffer> ub = buf(d, unb.data.data(), unb.data.size());
        id<MTLBuffer> r4 = buf(d, nullptr, n * 16), cov = buf(d, nullptr, n * 6);
        id<MTLBuffer> rej = buf(d, nullptr, n), kp = buf(d, nullptr, nm);
        id<MTLBuffer> itg = buf(d, nullptr, n * 4);
        id<MTLBuffer> rf = rs.enabled ? buf(d, nullptr, n * 4) : nil, kp2 = rs.enabled ? buf(d, nullptr, nm) : nil;
        if (!raw || !lb || !fl || !ub || !r4 || !cov || !rej || !kp || !itg || (rs.enabled && (!rf || !kp2))) return false;
        const int size4[4] = {g.Wg, g.Hg, Wm, Hm};
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encode_guide(e, p_guide, raw, lb, GP, r4, cov);
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:p_rej];
        [e setBuffer:g.base_ref_buf offset:0 atIndex:0];
        [e setBuffer:r4 offset:0 atIndex:1];
        [e setBuffer:fl offset:0 atIndex:2];
        [e setBuffer:lb offset:0 atIndex:3];
        [e setBytes:&RP length:sizeof(RP) atIndex:4];
        [e setBuffer:rej offset:0 atIndex:5];
        [e setBuffer:ub offset:0 atIndex:6];
        [e setBuffer:g.gm_buf offset:0 atIndex:7];
        [e setBuffer:itg offset:0 atIndex:8];
        dispatch2d(e, p_rej, g.Wg, g.Hg);
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:p_dil];
        [e setBuffer:rej offset:0 atIndex:0];
        [e setBuffer:kp offset:0 atIndex:1];
        [e setBytes:size4 length:sizeof(size4) atIndex:2];
        dispatch2d(e, p_dil, Wm, Hm);
        if (rs.enabled) {
            const S82ResidParams RS = rs.P;
            [e setComputePipelineState:g.resid_factor];
            [e setBuffer:itg offset:0 atIndex:0];
            [e setBuffer:g.gm_buf offset:0 atIndex:1];
            [e setBytes:&RS length:sizeof(RS) atIndex:2];
            [e setBuffer:rf offset:0 atIndex:3];
            dispatch2d(e, g.resid_factor, g.Wg, g.Hg);
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [e setComputePipelineState:g.resid_apply];
            [e setBuffer:kp offset:0 atIndex:0];
            [e setBuffer:rf offset:0 atIndex:1];
            [e setBytes:&RS length:sizeof(RS) atIndex:2];
            [e setBuffer:kp2 offset:0 atIndex:3];
            dispatch2d(e, g.resid_apply, Wm, Hm);
        }
        [e endEncoding];
        if (!run(cb)) return false;
        cov_out.assign((const uint16_t*)cov.contents, (const uint16_t*)cov.contents + n * 3);
        id<MTLBuffer> kout = rs.enabled ? kp2 : kp;
        keep.assign((const uint8_t*)kout.contents, (const uint8_t*)kout.contents + nm);
        return true;
    }
}

void metal_end_burst() {
    S82Gpu& g = gpu();
    g.base_ref_buf = nil; g.Wg = g.Hg = 0;
    g.gm_buf = nil; g.gm_src = nullptr;
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
            id<MTLBuffer> mk = (is_base && f.keep.empty()) ? buf(d, nullptr, 16) : buf(d, f.keep.data(), f.keep.size());
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
