#include "robustness_nn.h"

#ifdef __APPLE__
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <os/proc.h>
#include <vector>

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
    if (feat.h <= 0 || feat.w <= 0 || feat.c != kRobustnessNnChannels) return false;

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
        float* dst = (float*)in.dataPointer;
        const f32* src = feat.data.data();
        const size_t plane = (size_t)H * (size_t)W;
        for (size_t c = 0; c < (size_t)C; ++c) {
            float* dp = dst + c * plane;
            const f32* sp = src + c;
            for (size_t p = 0; p < plane; ++p) dp[p] = sp[p * (size_t)C];
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
        Image r((int)H, (int)W, 1);
        const float* op = (const float*)outArr.dataPointer;
        for (size_t p = 0; p < plane; ++p) {
            float v = op[p];
            // The graph ends in a sigmoid, so this only guards non-finite
            // values from a malformed model -- but R multiplies every merge
            // accumulator, and one NaN would poison the whole output pixel.
            if (!(v > 0.f)) v = 0.f;
            if (v > 1.f) v = 1.f;
            r.data[p] = v;
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
