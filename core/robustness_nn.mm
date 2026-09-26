#include "robustness_nn.h"

#ifdef __APPLE__
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <os/proc.h>
#include <vector>

namespace hhsr {
namespace {

// Headroom below which a learned mask declines to run and the caller falls
// back to the analytic one. A mask that degrades is recoverable; a jetsam kill
// loses the whole capture, so this errs toward giving up the feature.
constexpr uint64_t kMinAvailableBytes = 220ull * 1024ull * 1024ull;

// One loaded Core ML model plus its cached input buffer. Two of these exist:
// the mask that REPLACES Wronski Eq. 5-9, and the refinement that multiplies
// it down. They differ only in resource name, channel count and output name,
// so everything below -- the lazy load, the headroom check, the stride-safe
// fills, the fail-closed behaviour -- is written once.
struct CoreMLMask {
    const char* resource;     // .mlmodelc base name in the app bundle
    const char* input_name;   // graph input feature
    const char* output_name;  // graph output feature
    const char* tag;          // log prefix
    int channels;

    MLModel* model = nil;     // stays nil if the one load attempt fails
    bool tried = false;

    // The input buffer is the same shape for every strip of every frame, so
    // it is allocated once and refilled rather than reallocated ~50 times per
    // burst. Churning 20 MB allocations against a pipeline already near the
    // footprint limit is exactly how a run dies on the second or third frame.
    MLMultiArray* input = nil;
    NSInteger input_h = 0, input_w = 0;

    MLModel* load() {
        if (tried) return model;
        tried = true;
        @autoreleasepool {
            // Xcode compiles a bundled .mlmodel/.mlpackage into <name>.mlmodelc
            // at build time and places it in the app bundle under the same
            // base name.
            NSURL* url = [[NSBundle mainBundle]
                URLForResource:[NSString stringWithUTF8String:resource]
                 withExtension:@"mlmodelc"];
            if (!url) {
                NSLog(@"[%s] %s.mlmodelc not found in app bundle", tag, resource);
                return nil;
            }
            MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
            // Runs once per comparison frame on a plane the size of the guide
            // image; let Core ML place it on the ANE or GPU as it sees fit.
            config.computeUnits = MLComputeUnitsAll;
            NSError* err = nil;
            model = [MLModel modelWithContentsOfURL:url configuration:config error:&err];
            if (!model) NSLog(@"[%s] model load failed: %@", tag, err);
        }
        return model;
    }

    void release_buffers() {
        input = nil;
        input_h = input_w = 0;
    }

    bool infer(const Image& feat, Image& out) {
        MLModel* m = load();
        if (!m) return false;
        if (feat.h <= 0 || feat.w <= 0 || feat.c != channels) return false;

        // os_proc_available_memory reports what this process may still
        // allocate before the per-process limit, which is the number that
        // decides a jetsam kill -- not free system RAM.
        const size_t avail = os_proc_available_memory();
        if (avail != 0 && avail < kMinAvailableBytes) {
            NSLog(@"[%s] only %.0f MB headroom, using analytic mask",
                  tag, (double)avail / (1024.0 * 1024.0));
            release_buffers();
            return false;
        }

        @autoreleasepool {
            NSError* err = nil;
            const NSInteger C = channels, H = feat.h, W = feat.w;
            // The graph takes NCHW; Image stores interleaved, so this
            // transposes on the way in. Done here rather than in the feature
            // builder so the portable side keeps the same interleaved layout
            // everything else in the pipeline uses.
            if (!input || input_h != H || input_w != W) {
                input = [[MLMultiArray alloc]
                    initWithShape:@[@1, @(C), @(H), @(W)]
                         dataType:MLMultiArrayDataTypeFloat32
                            error:&err];
                input_h = H;
                input_w = W;
            }
            MLMultiArray* in = input;
            if (!in) {
                NSLog(@"[%s] input allocation failed: %@", tag, err);
                release_buffers();
                return false;
            }
            // MLMultiArray is NOT guaranteed contiguous: it carries
            // per-dimension strides, and ANE-backed buffers are routinely
            // row-padded for alignment. Writing C*H*W floats linearly into a
            // padded allocation overruns it and corrupts the heap -- which
            // does not fault here, it crashes a frame or two later with no
            // jetsam report. Index through the strides, and use the handler
            // API so the pointer is guaranteed valid (and large enough) for
            // the duration of the write.
            const size_t plane = (size_t)H * (size_t)W;
            const f32* src = feat.data.data();
            __block bool wrote = false;
            // The handler passes the strides in, so there is no need to read
            // .strides separately -- these are authoritative for this buffer.
            [in getMutableBytesWithHandler:^(void* ptr, NSInteger len,
                                             NSArray<NSNumber*>* strides) {
                if (strides.count != 4) return;
                const NSInteger isC = strides[1].integerValue;
                const NSInteger isH = strides[2].integerValue;
                const NSInteger isW = strides[3].integerValue;
                // len is in bytes; refuse rather than trust the arithmetic.
                const NSInteger need =
                    ((NSInteger)C - 1) * isC + ((NSInteger)H - 1) * isH +
                    ((NSInteger)W - 1) * isW + 1;
                if (len < need * (NSInteger)sizeof(float)) return;
                float* dst = (float*)ptr;
                for (NSInteger c = 0; c < (NSInteger)C; ++c)
                    for (NSInteger y = 0; y < (NSInteger)H; ++y) {
                        float* row = dst + c * isC + y * isH;
                        const f32* sp = src + ((size_t)y * (size_t)W) * (size_t)C + (size_t)c;
                        for (NSInteger x = 0; x < (NSInteger)W; ++x)
                            row[x * isW] = sp[(size_t)x * (size_t)C];
                    }
                wrote = true;
            }];
            if (!wrote) {
                NSLog(@"[%s] could not fill input (rank/size mismatch)", tag);
                release_buffers();
                return false;
            }

            MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{[NSString stringWithUTF8String:input_name]:
                                         [MLFeatureValue featureValueWithMultiArray:in]}
                             error:&err];
            if (!provider) {
                NSLog(@"[%s] input provider failed: %@", tag, err);
                return false;
            }

            id<MLFeatureProvider> res = [m predictionFromFeatures:provider error:&err];
            if (!res) {
                NSLog(@"[%s] prediction failed: %@", tag, err);
                release_buffers();
                return false;
            }
            if (!logged_once) {
                logged_once = true;
                // One line per burst, not per strip: enough to see the real
                // footprint on device without flooding the log.
                NSLog(@"[%s] strip %ldx%ld, headroom after first prediction: %.0f MB",
                      tag, (long)H, (long)W,
                      (double)os_proc_available_memory() / (1024.0 * 1024.0));
            }
            MLFeatureValue* fv =
                [res featureValueForName:[NSString stringWithUTF8String:output_name]];
            MLMultiArray* outArr = fv ? fv.multiArrayValue : nil;
            if (!outArr) {
                NSLog(@"[%s] output '%s' missing", tag, output_name);
                return false;
            }
            // Expect (1,1,H,W); accept any shape whose element count matches,
            // since a leading batch/channel of 1 may or may not be reported.
            NSInteger n = 1;
            for (NSNumber* d in outArr.shape) n *= d.integerValue;
            if (n != (NSInteger)plane) {
                NSLog(@"[%s] output shape %@ != %ld pixels", tag, outArr.shape, (long)plane);
                return false;
            }
            // Same stride caveat on the way out, and this buffer is Core ML's,
            // not ours -- reading it flat is how a padded ANE output walks off
            // the end of the allocation.
            NSArray<NSNumber*>* ostr = outArr.strides;
            const NSInteger ond = (NSInteger)ostr.count;
            const NSInteger osH = (ond >= 2) ? ostr[ond - 2].integerValue : (NSInteger)W;
            const NSInteger osW = (ond >= 1) ? ostr[ond - 1].integerValue : 1;
            Image r((int)H, (int)W, 1);
            // Captured C++ objects are const inside a block, so `r.data[...]`
            // would yield a const reference. Take the pointer out here; the
            // pointer is captured by value and its pointee stays writable.
            f32* rdata = r.data.data();
            __block bool read_ok = false;
            [outArr getBytesWithHandler:^(const void* ptr, NSInteger len) {
                const NSInteger need =
                    ((NSInteger)H - 1) * osH + ((NSInteger)W - 1) * osW + 1;
                if (len < need * (NSInteger)sizeof(float)) return;
                const float* op = (const float*)ptr;
                for (NSInteger y = 0; y < (NSInteger)H; ++y) {
                    const float* row = op + y * osH;
                    f32* dr = rdata + (size_t)y * (size_t)W;
                    for (NSInteger x = 0; x < (NSInteger)W; ++x) {
                        float v = row[x * osW];
                        // The graph ends in a sigmoid, so this only guards
                        // non-finite values from a malformed model -- but this
                        // number multiplies every merge accumulator, and one
                        // NaN would poison the whole output pixel.
                        if (!(v > 0.f)) v = 0.f;
                        if (v > 1.f) v = 1.f;
                        dr[x] = v;
                    }
                }
                read_ok = true;
            }];
            if (!read_ok) {
                NSLog(@"[%s] output buffer smaller than its own strides imply", tag);
                return false;
            }
            out = std::move(r);
        }
        return true;
    }

    bool logged_once = false;
};

CoreMLMask g_replace = {"RobustnessNet", "features", "robustness",
                        "robustness_nn", kRobustnessNnChannels};
CoreMLMask g_refine  = {"RobustnessRefineNet", "features", "keep",
                        "robustness_refine", kRobustnessRefineChannels};

} // namespace

bool robustness_nn_available() { return g_replace.load() != nil; }
bool robustness_nn_infer(const Image& feat, Image& out) { return g_replace.infer(feat, out); }
void robustness_nn_release_buffers() { g_replace.release_buffers(); }

bool robustness_refine_available() { return g_refine.load() != nil; }
bool robustness_refine_infer(const Image& feat, Image& out) { return g_refine.infer(feat, out); }
void robustness_refine_release_buffers() { g_refine.release_buffers(); }

} // namespace hhsr

#else   // !__APPLE__

namespace hhsr {
bool robustness_nn_available() { return false; }
bool robustness_nn_infer(const Image&, Image&) { return false; }
void robustness_nn_release_buffers() {}
bool robustness_refine_available() { return false; }
bool robustness_refine_infer(const Image&, Image&) { return false; }
void robustness_refine_release_buffers() {}
} // namespace hhsr

#endif
