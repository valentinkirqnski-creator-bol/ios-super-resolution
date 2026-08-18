#include "robustness_nn.h"

#ifdef __APPLE__
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <vector>

namespace hhsr {
namespace {

MLModel* g_model = nil;      // stays nil if the one load attempt below fails
bool g_tried = false;

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

bool robustness_nn_infer(const Image& feat, Image& out) {
    MLModel* model = load_model();
    if (!model) return false;
    if (feat.h <= 0 || feat.w <= 0 || feat.c != kRobustnessNnChannels) return false;

    @autoreleasepool {
        NSError* err = nil;
        const NSInteger C = kRobustnessNnChannels, H = feat.h, W = feat.w;
        // The graph takes NCHW; Image stores interleaved, so this transposes
        // on the way in. Done here rather than in the feature builder so the
        // portable side keeps the same interleaved layout everything else in
        // the pipeline uses.
        MLMultiArray* in = [[MLMultiArray alloc]
            initWithShape:@[@1, @(C), @(H), @(W)]
                 dataType:MLMultiArrayDataTypeFloat32
                    error:&err];
        if (!in) {
            NSLog(@"[robustness_nn] input allocation failed: %@", err);
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
            return false;
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
} // namespace hhsr

#endif
