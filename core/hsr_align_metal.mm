// hsr_align on Metal (kernels: core/HsrAlignKernels.metal, arithmetic: core/hsr_align_shared.h, shared with the CPU
// path in core/hsr_align.cpp). The reference pyramid (grey, levels, gradients, per-tile structure sums) stays resident
// for the burst; each comparison frame is uploaded once and grey + pyramid + coarse-to-fine block matching / ICA /
// flow transfer + the final 16/8 grid + 3x3 median run in ONE command buffer; only the final flow (and, on request,
// the full-resolution grey) comes back. Every entry returns false whenever the device, library, a pipeline, a buffer
// or a command buffer fails -- the caller then runs the CPU aligner.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "hsr_align.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

namespace hhsr {
namespace hsra {

namespace {
struct HsraGpu {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> grey = nil, blur_v = nil, blur_h = nil, down_h = nil, down_v = nil, grad = nil,
                                tstruct = nil, bm = nil, ica = nil, transfer = nil, median = nil;
    bool ok = false;
};
HsraGpu& gpu() {
    static HsraGpu g;
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
        g.grey = make(@"hsra_grey_quad_kernel");
        g.blur_v = make(@"hsra_blur_v_kernel");
        g.blur_h = make(@"hsra_blur_h_clip_kernel");
        g.down_h = make(@"hsra_down_h_kernel");
        g.down_v = make(@"hsra_down_v_kernel");
        g.grad = make(@"hsra_grad_kernel");
        g.tstruct = make(@"hsra_tile_structure_kernel");
        g.bm = make(@"hsra_block_match_kernel");
        g.ica = make(@"hsra_ica_kernel");
        g.transfer = make(@"hsra_transfer_kernel");
        g.median = make(@"hsra_median_kernel");
        g.ok = g.grey && g.blur_v && g.blur_h && g.down_h && g.down_v && g.grad && g.tstruct && g.bm && g.ica &&
               g.transfer && g.median;
    });
    return g;
}

// Burst state: sizes, the resident reference pyramid, and scratch reused by every frame.
struct Burst {
    bool ready = false;
    Options opt;
    int rw = 0, rh = 0, gw = 0, gh = 0;
    std::vector<int> ws, hs;
    id<MTLBuffer> raw = nil, q = nil, t = nil;                 // raw upload, grey / downsample scratch
    std::vector<id<MTLBuffer>> ref_img, ref_gx, ref_gy, ref_s3, mov_img;
    id<MTLBuffer> fa_dx = nil, fa_dy = nil, fb_dx = nil, fb_dy = nil;   // flow ping-pong
    size_t flow_cap = 0;
};
Burst& burst() { static Burst b; return b; }
std::mutex& mtx() { static std::mutex m; return m; }

id<MTLBuffer> newbuf(id<MTLDevice> d, size_t bytes) {
    return [d newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
}
void dispatch2d(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, int w, int h) {
    const NSUInteger tw = std::min<NSUInteger>(16, p.threadExecutionWidth);
    const NSUInteger th = std::max<NSUInteger>(1, std::min<NSUInteger>(16, p.maxTotalThreadsPerThreadgroup / tw));
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)w + tw - 1) / tw, ((NSUInteger)h + th - 1) / th, 1)
      threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
bool run(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted;
}

// grey (into levels[0]) + the downsampled levels. b.raw holds the frame.
void encode_pyramid(id<MTLComputeCommandEncoder> e, const HsraGpu& g, Burst& b, const float gains[4],
                    std::vector<id<MTLBuffer>>& levels) {
    HsraGreyParams GP;
    GP.rw = b.rw; GP.w = b.gw; GP.h = b.gh; GP.pad = 0;
    GP.g00 = gains[0]; GP.g01 = gains[1]; GP.g10 = gains[2]; GP.g11 = gains[3];
    const int gsize[4] = {b.gw, b.gh, 0, 0};
    [e setComputePipelineState:g.grey];
    [e setBuffer:b.raw offset:0 atIndex:0];
    [e setBytes:&GP length:sizeof(GP) atIndex:1];
    [e setBuffer:b.q offset:0 atIndex:2];
    dispatch2d(e, g.grey, b.gw, b.gh);
    [e setComputePipelineState:g.blur_v];
    [e setBuffer:b.q offset:0 atIndex:0];
    [e setBytes:gsize length:sizeof(gsize) atIndex:1];
    [e setBuffer:b.t offset:0 atIndex:2];
    dispatch2d(e, g.blur_v, b.gw, b.gh);
    [e setComputePipelineState:g.blur_h];
    [e setBuffer:b.t offset:0 atIndex:0];
    [e setBytes:gsize length:sizeof(gsize) atIndex:1];
    [e setBuffer:levels[0] offset:0 atIndex:2];
    dispatch2d(e, g.blur_h, b.gw, b.gh);
    for (size_t i = 1; i < levels.size(); ++i) {
        int w = b.ws[i - 1], h = b.hs[i - 1];
        id<MTLBuffer> src = levels[i - 1];
        for (int rem = b.opt.factors[i - 1]; rem > 1; rem >>= 1) {
            const int w2 = w / 2 + w % 2, h2 = h / 2 + h % 2;
            const bool last = (rem >> 1) <= 1;
            id<MTLBuffer> dst = last ? levels[i] : b.t;   // t: the intermediate of a factor-4 level
            const int size[4] = {w, h, w2, h2};
            [e setComputePipelineState:g.down_h];
            [e setBuffer:src offset:0 atIndex:0];
            [e setBytes:size length:sizeof(size) atIndex:1];
            [e setBuffer:b.q offset:0 atIndex:2];
            dispatch2d(e, g.down_h, w2, h);
            [e setComputePipelineState:g.down_v];
            [e setBuffer:b.q offset:0 atIndex:0];
            [e setBytes:size length:sizeof(size) atIndex:1];
            [e setBuffer:dst offset:0 atIndex:2];
            dispatch2d(e, g.down_v, w2, h2);
            src = dst; w = w2; h = h2;
        }
    }
}

bool upload(Burst& b, const Image& raw) {
    if (raw.w != b.rw || raw.h != b.rh || raw.data.size() < (size_t)raw.w * raw.h) return false;
    std::memcpy(b.raw.contents, raw.data.data(), (size_t)raw.w * raw.h * 4);
    return true;
}

void read_grey(const Burst& b, id<MTLBuffer> l0, Plane* grey_out) {
    if (!grey_out) return;
    *grey_out = Plane(b.gw, b.gh);
    std::memcpy(grey_out->d.data(), l0.contents, (size_t)b.gw * b.gh * 4);
}
}  // namespace

bool metal_set_reference(const Image& raw, const Config& cfg, const Options& opt, Plane* grey_out) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lk(mtx());
        HsraGpu& g = gpu();
        Burst& b = burst();
        b = Burst();
        if (!g.ok || raw.w < 32 || raw.h < 32 || raw.data.size() < (size_t)raw.w * raw.h) return false;
        const size_t n = opt.factors.size() + 1;
        if (opt.tiles.size() != n || opt.radii.size() != n || opt.l2.size() != n) return false;
        b.opt = opt;
        b.rw = raw.w; b.rh = raw.h; b.gw = raw.w / 2; b.gh = raw.h / 2;
        pyramid_dims(b.gw, b.gh, opt, b.ws, b.hs);
        if (b.gw < 16 || b.gh < 16) return false;
        for (size_t i = 0; i < n; ++i)
            if (b.ws[i] < opt.tiles[i] || b.hs[i] < opt.tiles[i]) return false;   // CPU align() refuses these too
        id<MTLDevice> d = g.device;
        const size_t gn = (size_t)b.gw * b.gh;
        b.raw = newbuf(d, (size_t)raw.w * raw.h * 4);
        b.q = newbuf(d, gn * 4);
        b.t = newbuf(d, gn * 4);
        if (!b.raw || !b.q || !b.t) return false;
        size_t cap = 0;
        for (size_t i = 0; i < n; ++i) {
            const size_t ln = (size_t)b.ws[i] * b.hs[i];
            const int cols = b.ws[i] / opt.tiles[i], rows = b.hs[i] / opt.tiles[i];
            b.ref_img.push_back(newbuf(d, ln * 4));
            b.ref_gx.push_back(newbuf(d, ln * 4));
            b.ref_gy.push_back(newbuf(d, ln * 4));
            b.ref_s3.push_back(newbuf(d, (size_t)cols * rows * 3 * 4));
            b.mov_img.push_back(newbuf(d, ln * 4));
            if (!b.ref_img.back() || !b.ref_gx.back() || !b.ref_gy.back() || !b.ref_s3.back() || !b.mov_img.back())
                return false;
            cap = std::max(cap, (size_t)cols * rows);
        }
        cap = std::max(cap, (size_t)((b.gw - 16) / 8 + 1) * (size_t)((b.gh - 16) / 8 + 1));
        b.flow_cap = cap;
        b.fa_dx = newbuf(d, cap * 4); b.fa_dy = newbuf(d, cap * 4);
        b.fb_dx = newbuf(d, cap * 4); b.fb_dy = newbuf(d, cap * 4);
        if (!b.fa_dx || !b.fa_dy || !b.fb_dx || !b.fb_dy) return false;
        if (!upload(b, raw)) return false;
        float gains[4];
        grey_gains(cfg, gains);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encode_pyramid(e, g, b, gains, b.ref_img);
        for (size_t i = 0; i < n; ++i) {
            const int size[4] = {b.ws[i], b.hs[i], 0, 0};
            [e setComputePipelineState:g.grad];
            [e setBuffer:b.ref_img[i] offset:0 atIndex:0];
            [e setBytes:size length:sizeof(size) atIndex:1];
            [e setBuffer:b.ref_gx[i] offset:0 atIndex:2];
            [e setBuffer:b.ref_gy[i] offset:0 atIndex:3];
            dispatch2d(e, g.grad, b.ws[i], b.hs[i]);
            const int ts = opt.tiles[i], cols = b.ws[i] / ts, rows = b.hs[i] / ts;
            const int ssize[4] = {b.ws[i], ts, cols, rows};
            [e setComputePipelineState:g.tstruct];
            [e setBuffer:b.ref_gx[i] offset:0 atIndex:0];
            [e setBuffer:b.ref_gy[i] offset:0 atIndex:1];
            [e setBytes:ssize length:sizeof(ssize) atIndex:2];
            [e setBuffer:b.ref_s3[i] offset:0 atIndex:3];
            dispatch2d(e, g.tstruct, cols, rows);
        }
        [e endEncoding];
        if (!run(cb)) { b = Burst(); return false; }
        read_grey(b, b.ref_img[0], grey_out);
        b.ready = true;
        return true;
    }
}

bool metal_align(const Image& raw, const Config& cfg, Flow& out, Plane* grey_out) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lk(mtx());
        HsraGpu& g = gpu();
        Burst& b = burst();
        if (!g.ok || !b.ready || !upload(b, raw)) return false;
        const Options& o = b.opt;
        const size_t n = b.ws.size(), coarsest = n - 1;
        float gains[4];
        grey_gains(cfg, gains);
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        // coarsest-level initial flow = 0
        int cols = b.ws[coarsest] / o.tiles[coarsest], rows = b.hs[coarsest] / o.tiles[coarsest];
        {
            id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
            [bl fillBuffer:b.fa_dx range:NSMakeRange(0, (NSUInteger)cols * rows * 4) value:0];
            [bl fillBuffer:b.fa_dy range:NSMakeRange(0, (NSUInteger)cols * rows * 4) value:0];
            [bl endEncoding];
        }
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encode_pyramid(e, g, b, gains, b.mov_img);
        id<MTLBuffer> cdx = b.fa_dx, cdy = b.fa_dy, ndx = b.fb_dx, ndy = b.fb_dy;   // current / next
        auto flip = [&]() { std::swap(cdx, ndx); std::swap(cdy, ndy); };
        for (size_t level = coarsest;; --level) {
            HsraBmParams BP;
            BP.w = b.ws[level]; BP.h = b.hs[level]; BP.ts = o.tiles[level]; BP.radius = o.radii[level];
            BP.l2 = o.l2[level] != 0 ? 1 : 0; BP.cols = cols; BP.rows = rows; BP.pad = 0;
            [e setComputePipelineState:g.bm];
            [e setBuffer:b.ref_img[level] offset:0 atIndex:0];
            [e setBuffer:b.mov_img[level] offset:0 atIndex:1];
            [e setBuffer:cdx offset:0 atIndex:2];
            [e setBuffer:cdy offset:0 atIndex:3];
            [e setBytes:&BP length:sizeof(BP) atIndex:4];
            [e setBuffer:ndx offset:0 atIndex:5];
            [e setBuffer:ndy offset:0 atIndex:6];
            dispatch2d(e, g.bm, cols, rows);
            flip();
            for (int it = 0; it < o.ica_iterations; ++it) {
                [e setComputePipelineState:g.ica];
                [e setBuffer:b.ref_img[level] offset:0 atIndex:0];
                [e setBuffer:b.ref_gx[level] offset:0 atIndex:1];
                [e setBuffer:b.ref_gy[level] offset:0 atIndex:2];
                [e setBuffer:b.ref_s3[level] offset:0 atIndex:3];
                [e setBuffer:b.mov_img[level] offset:0 atIndex:4];
                [e setBuffer:cdx offset:0 atIndex:5];
                [e setBuffer:cdy offset:0 atIndex:6];
                [e setBytes:&BP length:sizeof(BP) atIndex:7];
                [e setBuffer:ndx offset:0 atIndex:8];
                [e setBuffer:ndy offset:0 atIndex:9];
                dispatch2d(e, g.ica, cols, rows);
                flip();
            }
            const HsraTransferParams TP = level == 0
                ? final_transfer_params(o, cols, rows, b.ws[0], b.hs[0])
                : level_transfer_params(o, level, cols, rows, b.ws[level - 1], b.hs[level - 1]);
            const int tcols = transfer_cols(TP), trows = transfer_rows(TP);
            if (tcols <= 0 || trows <= 0 || (size_t)tcols * trows > b.flow_cap) { [e endEncoding]; return false; }
            const size_t target = level == 0 ? 0 : level - 1;
            const int grid[4] = {tcols, trows, 0, 0};
            [e setComputePipelineState:g.transfer];
            [e setBuffer:cdx offset:0 atIndex:0];
            [e setBuffer:cdy offset:0 atIndex:1];
            [e setBuffer:b.ref_img[target] offset:0 atIndex:2];
            [e setBuffer:b.mov_img[target] offset:0 atIndex:3];
            [e setBytes:&TP length:sizeof(TP) atIndex:4];
            [e setBytes:grid length:sizeof(grid) atIndex:5];
            [e setBuffer:ndx offset:0 atIndex:6];
            [e setBuffer:ndy offset:0 atIndex:7];
            dispatch2d(e, g.transfer, tcols, trows);
            flip();
            cols = tcols; rows = trows;
            if (level == 0) break;
        }
        const int grid[4] = {cols, rows, 0, 0};
        [e setComputePipelineState:g.median];
        [e setBuffer:cdx offset:0 atIndex:0];
        [e setBuffer:cdy offset:0 atIndex:1];
        [e setBytes:grid length:sizeof(grid) atIndex:2];
        [e setBuffer:ndx offset:0 atIndex:3];
        [e setBuffer:ndy offset:0 atIndex:4];
        dispatch2d(e, g.median, cols, rows);
        flip();
        [e endEncoding];
        if (!run(cb)) return false;
        out = Flow(cols, rows, 16, 8);
        std::memcpy(out.dx.data(), cdx.contents, (size_t)cols * rows * 4);
        std::memcpy(out.dy.data(), cdy.contents, (size_t)cols * rows * 4);
        read_grey(b, b.mov_img[0], grey_out);
        return true;
    }
}

void metal_release() {
    std::lock_guard<std::mutex> lk(mtx());
    burst() = Burst();
}

}  // namespace hsra
}  // namespace hhsr
