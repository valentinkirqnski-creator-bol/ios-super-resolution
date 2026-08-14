#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#if __has_include(<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>)
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#define HHSR_HAVE_MPSGRAPH 1
#else
#define HHSR_HAVE_MPSGRAPH 0
#endif

#include "mps_fft.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace hhsr {

bool mps_fft_enabled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    bool on = true;
    if (const char* v = std::getenv("HHSR_MPSGRAPH_FFT")) {
        on = !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' ||
               v[0] == 'f' || v[0] == 'F');
    }
#if HHSR_HAVE_MPSGRAPH
    if (on) {
        if (@available(iOS 16.0, macOS 13.0, *)) {
            // supported
        } else {
            on = false;
        }
    }
#else
    on = false;
#endif
    cached = on ? 1 : 0;
    return on;
}

#if HHSR_HAVE_MPSGRAPH

namespace {

// Does the Stockham path zero this spectrum entry? Replicates
// zero_fft_borders_natural, which in turn replicates the composed
// fftshift_swap_x / fftshift_swap_y / zero_fft_borders / unshift sequence. The
// swap index maps are reproduced verbatim rather than simplified to
// (x + w/2) % w: for odd sizes the swap leaves the final element in place and
// the modular form disagrees. Checked elementwise over 1225 dimension pairs
// when zero_fft_borders_natural landed.
static bool spectrum_zeroed(int y, int x, int h, int w) {
    const int half_w = w / 2, half_h = h / 2;
    const int xs = (x < half_w) ? (x + half_w)
                                : ((x < 2 * half_w) ? (x - half_w) : x);
    const int ys = (y < half_h) ? (y + half_h)
                                : ((y < 2 * half_h) ? (y - half_h) : y);
    const int y0 = h / 4, x0 = w / 4;
    return (ys < y0 || ys >= h - y0 || xs < x0 || xs >= w - x0);
}

// Mask weight for the Hermitean-domain multiply. Not 0/1 -- 0.5 appears, and
// that is not an approximation but the exact equivalent.
//
// The zeroing above is NOT conjugate-symmetric: it zeroes w/4 <= x < 3w/4,
// whose reflection is w/4 < x <= 3w/4, so the two disagree at exactly x = w/4
// and x = 3w/4 (and likewise for y). The masked spectrum is therefore not
// Hermitean, its inverse transform is not purely real, and extract_real in the
// Stockham path silently discards a nonzero imaginary component.
//
// HermiteanToRealFFT instead assumes a Hermitean input, which is equivalent to
// taking the real part -- and Re(IFFT(m . X)) = IFFT(Herm-part(m . X)). Since X
// is Hermitean and m is real, Herm-part(m . X)(y,x) = ((m(y,x) + m(-y,-x))/2)
// . X(y,x). So the exact Hermitean-domain mask is the average of the mask and
// its reflection, which is 1 or 0 everywhere except those boundary rows and
// columns, where it is 0.5.
//
// This matters: a naive 0/1 mask was measured at 51% relative error against the
// Stockham output, while the averaged mask matches to 4.9e-16.
static float spectrum_weight(int y, int x, int h, int w) {
    const float a = spectrum_zeroed(y, x, h, w) ? 0.f : 1.f;
    const int ry = (h - y) % h;
    const int rx = (w - x) % w;
    const float b = spectrum_zeroed(ry, rx, h, w) ? 0.f : 1.f;
    return 0.5f * (a + b);
}

struct Plan {
    void* graph = nullptr;   // MPSGraph*
    void* in = nullptr;      // MPSGraphTensor*
    void* out = nullptr;     // MPSGraphTensor*
    int h = 0, w = 0;
};

Plan g_plan;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
// Pooled input/output. The first version allocated a fresh input buffer per
// frame via newBufferWithBytes and let MPSGraph allocate result storage
// internally, which cost ~324MB of peak footprint across a burst. These are
// sized once and reused; both are exactly n floats, so a burst holds 2 x 48.8MB
// steady instead of churning allocations.
id<MTLBuffer> g_in_buf = nil;
id<MTLBuffer> g_out_buf = nil;
size_t g_buf_elems = 0;
std::mutex g_mutex;

API_AVAILABLE(ios(16.0), macos(13.0))
bool build_plan(int h, int w) {
    if (g_plan.graph && g_plan.h == h && g_plan.w == w) return true;

    MPSGraph* graph = [[MPSGraph alloc] init];
    if (!graph) return false;

    MPSGraphTensor* in = [graph placeholderWithShape:@[ @(h), @(w) ]
                                            dataType:MPSDataTypeFloat32
                                                name:nil];

    MPSGraphFFTDescriptor* fwd = [MPSGraphFFTDescriptor descriptor];
    fwd.inverse = NO;
    fwd.scalingMode = MPSGraphFFTScalingModeNone;
    fwd.roundToOddHermitean = NO;
    MPSGraphTensor* spec = [graph realToHermiteanFFTWithTensor:in
                                                          axes:@[ @0, @1 ]
                                                    descriptor:fwd
                                                          name:nil];

    // The border zeroing becomes a multiply by the weight mask above, so the
    // whole low-pass stays inside one graph: one buffer in, one buffer out, no
    // intermediate readback. The mask is complex with zero imaginary part so it
    // scales the spectrum elementwise; (a+bi)(m+0i) = ma + mbi.
    const int hw = w / 2 + 1;
    std::vector<float> mask((size_t)h * hw * 2, 0.f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < hw; ++x) {
            mask[((size_t)y * hw + x) * 2 + 0] = spectrum_weight(y, x, h, w);
        }
    }
    NSData* maskData = [NSData dataWithBytes:mask.data()
                                      length:mask.size() * sizeof(float)];
    MPSGraphTensor* maskT = [graph constantWithData:maskData
                                              shape:@[ @(h), @(hw) ]
                                           dataType:MPSDataTypeComplexFloat32];
    MPSGraphTensor* masked = [graph multiplicationWithPrimaryTensor:spec
                                                    secondaryTensor:maskT
                                                               name:nil];

    // Size scaling on the inverse only reproduces the Stockham path, where
    // fft_scale_inv divides by n once per 1D inverse -- w on the rows and h on
    // the columns, so 1/(h*w) overall, and nothing on the forward.
    MPSGraphFFTDescriptor* inv = [MPSGraphFFTDescriptor descriptor];
    inv.inverse = YES;
    inv.scalingMode = MPSGraphFFTScalingModeSize;
    inv.roundToOddHermitean = NO;
    MPSGraphTensor* out = [graph HermiteanToRealFFTWithTensor:masked
                                                         axes:@[ @0, @1 ]
                                                   descriptor:inv
                                                         name:nil];
    if (!out) return false;

    g_plan.graph = (__bridge_retained void*)graph;
    g_plan.in = (__bridge_retained void*)in;
    g_plan.out = (__bridge_retained void*)out;
    g_plan.h = h;
    g_plan.w = w;
    return true;
}

API_AVAILABLE(ios(16.0), macos(13.0))
bool ensure_buffers(size_t n, bool caller_supplies_output) {
    const bool have_out = caller_supplies_output || g_out_buf != nil;
    if (g_in_buf && have_out && g_buf_elems >= n) return true;
    const size_t bytes = n * sizeof(float);
    if (!g_in_buf || g_buf_elems < n)
        g_in_buf = [g_device newBufferWithLength:bytes
                                         options:MTLResourceStorageModeShared];
    if (!caller_supplies_output && (!g_out_buf || g_buf_elems < n))
        g_out_buf = [g_device newBufferWithLength:bytes
                                          options:MTLResourceStorageModeShared];
    if (!g_in_buf || (!caller_supplies_output && !g_out_buf)) {
        g_in_buf = nil;
        g_out_buf = nil;
        g_buf_elems = 0;
        return false;
    }
    g_buf_elems = n;
    return true;
}

API_AVAILABLE(ios(16.0), macos(13.0))
bool run_plan(const float* in, float* out, int h, int w, id<MTLBuffer> dst) {
    if (!build_plan(h, w)) return false;

    MPSGraph* graph = (__bridge MPSGraph*)g_plan.graph;
    MPSGraphTensor* inT = (__bridge MPSGraphTensor*)g_plan.in;
    MPSGraphTensor* outT = (__bridge MPSGraphTensor*)g_plan.out;

    const size_t n = (size_t)h * (size_t)w;
    if (!ensure_buffers(n, dst != nil)) return false;
    memcpy([g_in_buf contents], in, n * sizeof(float));
    // Write into the caller's buffer when it supplied one; g_out_buf is then
    // never allocated at all.
    id<MTLBuffer> outBuf = dst ? dst : g_out_buf;
    if (!outBuf || [outBuf length] < n * sizeof(float)) return false;

    MPSGraphTensorData* inData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:g_in_buf
                                                shape:@[ @(h), @(w) ]
                                             dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* outData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:@[ @(h), @(w) ]
                                             dataType:MPSDataTypeFloat32];
    if (!inData || !outData) return false;

    // Supplying the results dictionary makes the graph write into g_out_buf
    // rather than allocating its own storage, and lets the result be read
    // straight from the buffer instead of through readBytes.
    [graph runWithMTLCommandQueue:g_queue
                            feeds:@{ inT : inData }
                 targetOperations:nil
                resultsDictionary:@{ outT : outData }];

    memcpy(out, [outBuf contents], n * sizeof(float));
    return true;
}

}  // namespace

void mps_fft_prewarm(int h, int w) {
    if (h <= 0 || w <= 0 || (h & 1) || (w & 1)) return;
    if (!mps_fft_enabled()) return;
    if (@available(iOS 16.0, macOS 13.0, *)) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_device) {
            g_device = MTLCreateSystemDefaultDevice();
            if (!g_device) return;
            g_queue = [g_device newCommandQueue];
            if (!g_queue) return;
        }
        @autoreleasepool {
            (void)build_plan(h, w);
            (void)ensure_buffers((size_t)h * (size_t)w, false);
        }
    }
}

void mps_fft_release_buffers() {
    if (@available(iOS 16.0, macOS 13.0, *)) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_in_buf = nil;
        g_out_buf = nil;
        g_buf_elems = 0;
    }
}

void mps_fft_release_all() {
    if (@available(iOS 16.0, macOS 13.0, *)) {
        std::lock_guard<std::mutex> lock(g_mutex);
        @autoreleasepool {
            // __bridge_transfer hands the retain from the raw pointers back to
            // ARC, which releases them as these locals go out of scope.
            if (g_plan.graph) {
                MPSGraph* g = (__bridge_transfer MPSGraph*)g_plan.graph;
                (void)g;
                g_plan.graph = nullptr;
            }
            if (g_plan.in) {
                MPSGraphTensor* t = (__bridge_transfer MPSGraphTensor*)g_plan.in;
                (void)t;
                g_plan.in = nullptr;
            }
            if (g_plan.out) {
                MPSGraphTensor* t = (__bridge_transfer MPSGraphTensor*)g_plan.out;
                (void)t;
                g_plan.out = nullptr;
            }
        }
        g_plan.h = 0;
        g_plan.w = 0;
        g_in_buf = nil;
        g_out_buf = nil;
        g_buf_elems = 0;
    }
}

bool mps_grey_lowpass(const float* in, float* out, int h, int w,
                      void* out_mtl_buffer) {
    if (!in || !out || h <= 0 || w <= 0) return false;
    // Even dimensions only, matching the Stockham caller: the shift the mask is
    // derived from assumes them, and Bayer RAW is always even.
    if ((h & 1) || (w & 1)) return false;
    if (!mps_fft_enabled()) return false;

    if (@available(iOS 16.0, macOS 13.0, *)) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_device) {
            g_device = MTLCreateSystemDefaultDevice();
            if (!g_device) return false;
            g_queue = [g_device newCommandQueue];
            if (!g_queue) return false;
        }
        @autoreleasepool {
            return run_plan(in, out, h, w, (__bridge id<MTLBuffer>)out_mtl_buffer);
        }
    }
    return false;
}

#else  // !HHSR_HAVE_MPSGRAPH

void mps_fft_prewarm(int, int) {}
void mps_fft_release_all() {}
void mps_fft_release_buffers() {}
bool mps_grey_lowpass(const float*, float*, int, int, void*) { return false; }

#endif

}  // namespace hhsr
