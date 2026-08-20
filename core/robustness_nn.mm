#include "robustness_nn.h"

#ifdef __APPLE__
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <os/proc.h>
#include <vector>
#include <cstring>

namespace hhsr {
namespace {

MLModel* g_model = nil;      // stays nil if the one load attempt below fails
bool g_tried = false;

// The input buffer is the same shape for every strip of every frame, so it is
// allocated once and refilled rather than reallocated ~50 times per burst.
// Churning 20 MB allocations against a pipeline that is already near the
// footprint limit is exactly how a run dies on the second or third frame.
MLMultiArray* g_input = nil;
NSInteger g_input_h = 0, g_input_w = 0;

// Headroom below which the learned mask declines to run and the caller falls
// back to the analytic one. A mask that degrades is recoverable; a jetsam kill
// loses the whole capture, so this errs toward giving up the feature.
constexpr uint64_t kMinAvailableBytes = 220ull * 1024ull * 1024ull;

MLModel* load_model() {
    if (g_tried) return g_model;
    g_tried = true;
    @autoreleasepool {
        // Xcode compiles a bundled .mlmodel/.mlpackage into <name>.mlmodelc at
        // build time and places it in the app bundle under the same base name.
        NSURL* url = [[NSBundle mainBundle] URLForResource:@"RobustnessNet"
                                            withExtension:@"mlmodelc"];
        if (!url) {
            NSLog(@"[robustness_nn] RobustnessNet.mlmodelc not found in app bundle");
            return nil;
        }
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        // The mask runs once per comparison frame on a plane the size of the
        // guide image; let Core ML place it on the ANE or GPU as it sees fit.
        config.computeUnits = MLComputeUnitsAll;
        NSError* err = nil;
        g_model = [MLModel modelWithContentsOfURL:url configuration:config error:&err];
        if (!g_model)
            NSLog(@"[robustness_nn] model load failed: %@", err);
    }
    return g_model;
}

} // namespace

bool robustness_nn_available() { return load_model() != nil; }

void robustness_nn_release_buffers() {
    g_input = nil;
    g_input_h = g_input_w = 0;
}

bool robustness_nn_infer(const Image& feat, Image& out) {
    MLModel* model = load_model();
    if (!model) return false;
    if (feat.h <= 0 || feat.w <= 0 || feat.c != kRobustnessNnChannels) {
        // This was a bare `return false`, and it is the path that fires
        // whenever the feature layout has moved on but the bundled .mlmodelc
        // has not. Silently falling back to the analytic mask there is
        // indistinguishable, from the outside, from the learned mask running
        // and simply not helping -- which cost real debugging time. Say it.
        NSLog(@"[robustness_nn] feature/model mismatch: built %d channels at "
              @"%dx%d, model expects %d. Using the ANALYTIC mask -- the "
              @"bundled model predates the current feature set and must be "
              @"retrained and re-exported.",
              feat.c, feat.w, feat.h, kRobustnessNnChannels);
        return false;
    }

    // os_proc_available_memory reports what this process may still allocate
    // before the per-process limit, which is the number that decides a jetsam
    // kill -- not free system RAM.
    const size_t avail = os_proc_available_memory();
    if (avail != 0 && avail < kMinAvailableBytes) {
        NSLog(@"[robustness_nn] only %.0f MB headroom, using analytic mask",
              (double)avail / (1024.0 * 1024.0));
        robustness_nn_release_buffers();
        return false;
    }

    @autoreleasepool {
        NSError* err = nil;
        const NSInteger C = kRobustnessNnChannels, H = feat.h, W = feat.w;
        // The graph takes NCHW; Image stores interleaved, so this transposes
        // on the way in. Done here rather than in the feature builder so the
        // portable side keeps the same interleaved layout everything else in
        // the pipeline uses.
        if (!g_input || g_input_h != H || g_input_w != W) {
            g_input = [[MLMultiArray alloc]
                initWithShape:@[@1, @(C), @(H), @(W)]
                     dataType:MLMultiArrayDataTypeFloat32
                        error:&err];
            g_input_h = H;
            g_input_w = W;
        }
        MLMultiArray* in = g_input;
        if (!in) {
            NSLog(@"[robustness_nn] input allocation failed: %@", err);
            robustness_nn_release_buffers();
            return false;
        }
        // The feature builder now writes NCHW directly (planar=true), which is
        // the order this wants, so what used to be a transpose over 18 planes
        // is a row copy. Measured at guide resolution: interleaved build 22.2
        // ms + transpose 28.4 ms per strip against 23.2 ms for the planar
        // build, i.e. ~27 ms saved per strip and ~218 ms per comparison frame,
        // with output verified bit-identical.
        //
        // MLMultiArray is NOT guaranteed contiguous: it carries per-dimension
        // strides, and ANE-backed buffers are routinely row-padded for
        // alignment. Writing C*H*W floats linearly into a padded allocation
        // overruns it and corrupts the heap -- which does not fault here, it
        // crashes a frame or two later with no jetsam report. Index through
        // the strides, and use the handler API so the pointer is guaranteed
        // valid (and large enough) for the duration of the write.
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
                    // Source is planar: channel c, row y, contiguous in x.
                    const f32* sp = src + ((size_t)c * (size_t)H + (size_t)y) * (size_t)W;
                    if (isW == 1) {
                        std::memcpy(row, sp, (size_t)W * sizeof(float));
                    } else {
                        for (NSInteger x = 0; x < (NSInteger)W; ++x)
                            row[x * isW] = sp[(size_t)x];
                    }
                }
            wrote = true;
        }];
        if (!wrote) {
            NSLog(@"[robustness_nn] could not fill input (rank/size mismatch)");
            robustness_nn_release_buffers();
            return false;
        }

        MLDictionaryFeatureProvider* input = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"features": [MLFeatureValue featureValueWithMultiArray:in]}
                         error:&err];
        if (!input) {
            NSLog(@"[robustness_nn] input provider failed: %@", err);
            return false;
        }

        id<MLFeatureProvider> res = [model predictionFromFeatures:input error:&err];
        if (!res) {
            NSLog(@"[robustness_nn] prediction failed: %@", err);
            robustness_nn_release_buffers();
            return false;
        }
        static bool logged_once = false;
        if (!logged_once) {
            logged_once = true;
            // One line per burst, not per strip: enough to see the real
            // footprint on device without flooding the log.
            NSLog(@"[robustness_nn] strip %ldx%ld, headroom after first "
                  @"prediction: %.0f MB", (long)H, (long)W,
                  (double)os_proc_available_memory() / (1024.0 * 1024.0));
        }
        MLFeatureValue* fv = [res featureValueForName:@"robustness"];
        MLMultiArray* outArr = fv ? fv.multiArrayValue : nil;
        if (!outArr) {
            NSLog(@"[robustness_nn] output 'robustness' missing");
            return false;
        }
        // Expect (1,1,H,W); accept any shape whose element count matches, since
        // a leading batch/channel of 1 may or may not be reported.
        NSInteger n = 1;
        for (NSNumber* d in outArr.shape) n *= d.integerValue;
        if (n != (NSInteger)plane) {
            NSLog(@"[robustness_nn] output shape %@ != %ld pixels", outArr.shape, (long)plane);
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
                    // non-finite values from a malformed model -- but R
                    // multiplies every merge accumulator, and one NaN would
                    // poison the whole output pixel.
                    if (!(v > 0.f)) v = 0.f;
                    if (v > 1.f) v = 1.f;
                    dr[x] = v;
                }
            }
            read_ok = true;
        }];
        if (!read_ok) {
            NSLog(@"[robustness_nn] output buffer smaller than its own strides imply");
            return false;
        }
        out = std::move(r);
    }
    return true;
}

} // namespace hhsr

#else   // !__APPLE__

namespace hhsr {
bool robustness_nn_available() { return false; }
bool robustness_nn_infer(const Image&, Image&) { return false; }
void robustness_nn_release_buffers() {}
} // namespace hhsr

#endif
