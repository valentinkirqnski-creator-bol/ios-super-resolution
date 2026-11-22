// srsr_metal.mm -- the SRSR backend on the device GPU (see srsr_backend.h).
// Kernels: SRSRKernels.metal, compiled by Xcode into the app's default.metallib.
// Textures are private (GPU-only), buffers shared (the staging raw and the DNG
// output rows are read/written by the CPU in place, no blits). One command
// buffer per batch; GPU time goes to the profile under the batch label.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "srsr_backend.h"
#include "prof.h"
#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>

namespace hhsr {
namespace srsr {
namespace {

struct MCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> ps[(int)K::count] = {};
    bool nonuniform = false;   // dispatchThreads needs Apple4+ (A11 and later)
    bool ok = false;
};

static MCtx& mctx() {
    static MCtx c;
    static std::once_flag once;
    std::call_once(once, [] {
        @autoreleasepool {
            c.dev = MTLCreateSystemDefaultDevice();
            if (!c.dev) return;
            c.queue = [c.dev newCommandQueue];
            if (!c.queue) return;
            if (@available(iOS 13.0, macOS 10.15, *)) c.nonuniform = [c.dev supportsFamily:MTLGPUFamilyApple4];
            NSError* err = nil;
            id<MTLLibrary> lib = nil;
            NSString* path = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
            if (path) lib = [c.dev newLibraryWithFile:path error:&err];
            if (!lib) lib = [c.dev newDefaultLibrary];
            if (!lib) return;
            for (int k = 0; k < (int)K::count; ++k) {
                id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kernel_name((K)k)]];
                if (!fn) {
                    NSLog(@"SRSR: kernel %s missing from the Metal library", kernel_name((K)k));
                    return;
                }
                err = nil;
                c.ps[k] = [c.dev newComputePipelineStateWithFunction:fn error:&err];
                if (!c.ps[k]) {
                    NSLog(@"SRSR: pipeline %s failed: %@", kernel_name((K)k), err);
                    return;
                }
            }
            c.ok = true;
        }
    });
    return c;
}

static MTLPixelFormat pixel_format(Fmt f) {
    switch (f) {
        case Fmt::R8Unorm: return MTLPixelFormatR8Unorm;
        case Fmt::R32Float: return MTLPixelFormatR32Float;
        case Fmt::RGBA16Float: return MTLPixelFormatRGBA16Float;
        case Fmt::RGBA16Uint: return MTLPixelFormatRGBA16Uint;
        case Fmt::RGBA32Float: return MTLPixelFormatRGBA32Float;
    }
    return MTLPixelFormatR32Float;
}

struct MTex : Tex { id<MTLTexture> t = nil; };
struct MBuf : Buf {
    id<MTLBuffer> b = nil;
    void* contents() override { return b ? [b contents] : nullptr; }
};

class MetalBackend : public Backend {
public:
    MetalBackend() : c_(mctx()) {}
    const char* name() const override { return "Metal"; }

    std::unique_ptr<Tex> make_tex(Fmt f, int w, int h, const char* label) override {
        @autoreleasepool {
            MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format(f)
                                                                                         width:(NSUInteger)w
                                                                                        height:(NSUInteger)h
                                                                                     mipmapped:NO];
            d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            d.storageMode = MTLStorageModePrivate;
            id<MTLTexture> t = [c_.dev newTextureWithDescriptor:d];
            if (!t) return nullptr;
            if (label) t.label = [NSString stringWithUTF8String:label];
            auto r = std::make_unique<MTex>();
            r->t = t; r->w = w; r->h = h; r->fmt = f;
            return r;
        }
    }

    std::unique_ptr<Buf> make_buf(size_t bytes, const char* label) override {
        @autoreleasepool {
            id<MTLBuffer> b = [c_.dev newBufferWithLength:(NSUInteger)std::max<size_t>(bytes, 16)
                                                  options:MTLResourceStorageModeShared];
            if (!b) return nullptr;
            if (label) b.label = [NSString stringWithUTF8String:label];
            auto r = std::make_unique<MBuf>();
            r->b = b; r->size = bytes;
            return r;
        }
    }

    bool begin() override {
        @autoreleasepool {
            cb_ = [c_.queue commandBuffer];
            enc_ = cb_ ? [cb_ computeCommandEncoder] : nil;
        }
        return cb_ != nil && enc_ != nil;
    }

    void tex(int slot, Tex* t) override {
        if (enc_ && t) [enc_ setTexture:static_cast<MTex*>(t)->t atIndex:(NSUInteger)slot];
    }
    void bytes(int slot, const void* p, size_t n) override {
        if (enc_ && p && n) [enc_ setBytes:p length:(NSUInteger)n atIndex:(NSUInteger)slot];
    }
    void buf(int slot, Buf* b) override {
        if (enc_ && b) [enc_ setBuffer:static_cast<MBuf*>(b)->b offset:0 atIndex:(NSUInteger)slot];
    }

    void dispatch(K k, int w, int h) override {
        if (!enc_ || w <= 0 || h <= 0) return;
        id<MTLComputePipelineState> ps = c_.ps[(int)k];
        [enc_ setComputePipelineState:ps];
        const NSUInteger tw = std::max<NSUInteger>(1, std::min<NSUInteger>(ps.threadExecutionWidth, (NSUInteger)w));
        const NSUInteger th = std::max<NSUInteger>(1, std::min<NSUInteger>(ps.maxTotalThreadsPerThreadgroup / tw, (NSUInteger)h));
        if (c_.nonuniform) {
            [enc_ dispatchThreads:MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1)
            threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
        } else {   // every SRSR kernel bounds-checks its grid
            [enc_ dispatchThreadgroups:MTLSizeMake(((NSUInteger)w + tw - 1) / tw, ((NSUInteger)h + th - 1) / th, 1)
                 threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
        }
    }

    void dispatch_bm(int tx, int ty, int threads, size_t tg0, size_t tg1) override {
        if (!enc_ || tx <= 0 || ty <= 0) return;
        id<MTLComputePipelineState> ps = c_.ps[(int)K::block_matching];
        if ((NSUInteger)threads > ps.maxTotalThreadsPerThreadgroup || tg0 + tg1 > c_.dev.maxThreadgroupMemoryLength) {
            failed_->store(true);
            return;
        }
        [enc_ setComputePipelineState:ps];
        [enc_ setThreadgroupMemoryLength:(NSUInteger)tg0 atIndex:0];
        [enc_ setThreadgroupMemoryLength:(NSUInteger)tg1 atIndex:1];
        [enc_ dispatchThreadgroups:MTLSizeMake((NSUInteger)tx, (NSUInteger)ty, 1)
             threadsPerThreadgroup:MTLSizeMake((NSUInteger)threads, 1, 1)];
    }

    uint64_t commit(const char* label) override {
        if (!cb_ || !enc_) return 0;
        @autoreleasepool {
            [enc_ endEncoding];
            enc_ = nil;
            // By value: a completion handler can still be running after waitUntilCompleted
            // returns, so it must not point into this object.
            std::shared_ptr<std::atomic<bool>> failed = failed_;
            const bool prof = prof_enabled();
            const char* name = label ? label : "srsr";
            [cb_ addCompletedHandler:^(id<MTLCommandBuffer> done) {
                if (done.status == MTLCommandBufferStatusError) failed->store(true);
                if (prof) prof_add_gpu(name, (done.GPUEndTime - done.GPUStartTime) * 1000.0);
            }];
            [cb_ commit];
            const uint64_t t = ++ticket_;
            inflight_[t] = cb_;
            cb_ = nil;
            return t;
        }
    }

    bool wait(uint64_t ticket) override {
        auto it = inflight_.find(ticket);
        if (it != inflight_.end()) {
            id<MTLCommandBuffer> cb = it->second;
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) failed_->store(true);
            inflight_.erase(it);
        }
        return !failed_->load();
    }

    bool wait_all() override {
        while (!inflight_.empty()) wait(inflight_.begin()->first);
        return !failed_->load();
    }

    bool failed() const override { return failed_->load(); }

    ~MetalBackend() override {
        if (enc_) [enc_ endEncoding];
        wait_all();
    }

private:
    MCtx& c_;
    id<MTLCommandBuffer> cb_ = nil;
    id<MTLComputeCommandEncoder> enc_ = nil;
    std::map<uint64_t, id<MTLCommandBuffer>> inflight_;
    uint64_t ticket_ = 0;
    std::shared_ptr<std::atomic<bool>> failed_ = std::make_shared<std::atomic<bool>>(false);
};

} // namespace

std::unique_ptr<Backend> make_backend() {
    if (!mctx().ok) return nullptr;
    return std::make_unique<MetalBackend>();
}

} // namespace srsr
} // namespace hhsr
