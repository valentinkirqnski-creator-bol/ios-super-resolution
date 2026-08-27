// See neural_flow.h for the contract and the "untested on device" note --
// written and reasoned through without an Xcode/Core ML runtime available to
// compile or run it against.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include "neural_flow.h"
#include <mutex>

namespace hhsr {
namespace {

// Fixed at Core ML conversion time (scratchpad/convert_coreml.py: H, W).
// Kept in sync manually -- neural_flow_estimate() rejects any other size
// rather than silently resizing into a shape the model wasn't built for.
constexpr int kModelH = 1512;
constexpr int kModelW = 2016;

std::once_flag g_load_once;
MLModel* g_model = nil;   // stays nil if the load attempt (below, run once) fails

MLModel* load_model() {
    std::call_once(g_load_once, [] {
        // Xcode compiles a bundled .mlpackage into <name>.mlmodelc at build
        // time and places it in the app's resource bundle under the same
        // base name -- this expects "PWCNetFlow.mlpackage" to have been
        // added to the app target.
        NSURL* url = [[NSBundle mainBundle] URLForResource:@"PWCNetFlow" withExtension:@"mlmodelc"];
        if (url == nil) {
            NSLog(@"[neural_flow] PWCNetFlow.mlmodelc not found in app bundle");
            return;
        }
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll; // GPU/ANE preferred, matches ct.ComputeUnit.ALL at conversion time
        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (model == nil) {
            NSLog(@"[neural_flow] MLModel load failed: %@", error.localizedDescription);
            return;
        }
        g_model = model;
    });
    return g_model;
}

// Fills a freshly-allocated (1,3,H,W) float32 MLMultiArray from an
// interleaved-RGB Image (compute_guide()'s layout), respecting the array's
// own strides rather than assuming a particular packing.
bool fill_input_array(MLMultiArray* arr, const Image& guide) {
    if (arr.dataType != MLMultiArrayDataTypeFloat32) return false;
    NSArray<NSNumber*>* strides = arr.strides;
    if (strides.count != 4) return false;
    const long s_n = strides[0].longValue;
    const long s_c = strides[1].longValue;
    const long s_y = strides[2].longValue;
    const long s_x = strides[3].longValue;
    float* base = (float*)arr.dataPointer;
    if (base == nullptr) return false;

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < guide.h; ++y) {
            const long row_off = 0 * s_n + (long)c * s_c + (long)y * s_y;
            for (int x = 0; x < guide.w; ++x) {
                base[row_off + (long)x * s_x] = guide.at(y, x, c);
            }
        }
    }
    return true;
}

// Reads a (1,2,H,W) float32 MLMultiArray into dx-plane-then-dy-plane order
// (flow_from_dense_guide()'s expected layout), respecting strides.
bool read_flow_output(MLMultiArray* arr, int h, int w, std::vector<f32>& out) {
    if (arr.dataType != MLMultiArrayDataTypeFloat32) return false;
    NSArray<NSNumber*>* shape = arr.shape;
    if (shape.count != 4 || shape[1].intValue != 2 ||
        shape[2].intValue != h || shape[3].intValue != w)
        return false;
    NSArray<NSNumber*>* strides = arr.strides;
    if (strides.count != 4) return false;
    const long s_n = strides[0].longValue;
    const long s_c = strides[1].longValue;
    const long s_y = strides[2].longValue;
    const long s_x = strides[3].longValue;
    const float* base = (const float*)arr.dataPointer;
    if (base == nullptr) return false;

    out.assign((size_t)2 * h * w, 0.f);
    for (int c = 0; c < 2; ++c) {
        f32* plane = out.data() + (size_t)c * h * w;
        for (int y = 0; y < h; ++y) {
            const long row_off = 0 * s_n + (long)c * s_c + (long)y * s_y;
            for (int x = 0; x < w; ++x) {
                plane[(size_t)y * w + x] = base[row_off + (long)x * s_x];
            }
        }
    }
    return true;
}

} // namespace

bool neural_flow_available() {
    return load_model() != nil;
}

bool neural_flow_estimate(const Image& ref_guide, const Image& comp_guide,
                          std::vector<f32>& dense_flow_out) {
    dense_flow_out.clear();

    if (ref_guide.h != kModelH || ref_guide.w != kModelW || ref_guide.c != 3 ||
        comp_guide.h != kModelH || comp_guide.w != kModelW || comp_guide.c != 3) {
        NSLog(@"[neural_flow] guide size %dx%d (want %dx%d) -- refusing rather than "
              "silently resizing into a shape the model wasn't converted for",
              ref_guide.w, ref_guide.h, kModelW, kModelH);
        return false;
    }

    MLModel* model = load_model();
    if (model == nil) return false;

    @autoreleasepool {
        NSError* error = nil;
        NSArray<NSNumber*>* shape = @[@1, @3, @(kModelH), @(kModelW)];
        MLMultiArray* refArr = [[MLMultiArray alloc] initWithShape:shape
                                                           dataType:MLMultiArrayDataTypeFloat32
                                                              error:&error];
        if (refArr == nil) {
            NSLog(@"[neural_flow] ref MLMultiArray alloc failed: %@", error.localizedDescription);
            return false;
        }
        MLMultiArray* compArr = [[MLMultiArray alloc] initWithShape:shape
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                               error:&error];
        if (compArr == nil) {
            NSLog(@"[neural_flow] comp MLMultiArray alloc failed: %@", error.localizedDescription);
            return false;
        }

        if (!fill_input_array(refArr, ref_guide) || !fill_input_array(compArr, comp_guide)) {
            NSLog(@"[neural_flow] failed to fill input arrays");
            return false;
        }

        NSDictionary<NSString*, MLFeatureValue*>* inputDict = @{
            @"ref_frame": [MLFeatureValue featureValueWithMultiArray:refArr],
            @"comp_frame": [MLFeatureValue featureValueWithMultiArray:compArr],
        };
        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict error:&error];
        if (provider == nil) {
            NSLog(@"[neural_flow] feature provider init failed: %@", error.localizedDescription);
            return false;
        }

        id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&error];
        if (result == nil) {
            NSLog(@"[neural_flow] prediction failed: %@", error.localizedDescription);
            return false;
        }

        MLFeatureValue* flowVal = [result featureValueForName:@"flow"];
        MLMultiArray* flowArr = flowVal.multiArrayValue;
        if (flowArr == nil) {
            NSLog(@"[neural_flow] no 'flow' output in prediction result");
            return false;
        }

        if (!read_flow_output(flowArr, kModelH, kModelW, dense_flow_out)) {
            NSLog(@"[neural_flow] failed to read flow output (shape/dtype mismatch?)");
            dense_flow_out.clear();
            return false;
        }
    }

    return true;
}

} // namespace hhsr
