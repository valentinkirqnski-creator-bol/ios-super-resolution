#include "MetalContext.h"
#include "../stages.h"
#include "../parallel.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace hhsr;

// Expose internal functions from grey_pyramid.cpp for testing
namespace hhsr {
    Image compute_grey_decimate(const Image& raw, bool bayer_mode);
    Image compute_grey_metal(const Image& raw, bool bayer_mode);
}

void test_grey_decimate() {
    std::cout << "Testing Metal kernel_grey_decimate..." << std::endl;
    
    // Create a mock Bayer raw image (must be even dimensions)
    int h = 128, w = 128;
    Image raw(h, w, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            raw.at(y, x) = (y * w + x) / (f32)(h * w); // dummy values
        }
    }
    
    // CPU version
    Image cpu_grey = compute_grey_decimate(raw, true);
    
    // Metal version
    MetalContext::instance().init();
    if (!MetalContext::instance().is_available()) {
        std::cerr << "Metal not available for testing. Please run on macOS/iOS." << std::endl;
        return;
    }
    
#ifdef __OBJC__
    Image mtl_grey = compute_grey_metal(raw, true);
    
    if (cpu_grey.h != mtl_grey.h || cpu_grey.w != mtl_grey.w) {
        std::cerr << "Dimension mismatch!" << std::endl;
        return;
    }
    
    f32 max_diff = 0;
    for (size_t i = 0; i < cpu_grey.data.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(cpu_grey.data[i] - mtl_grey.data[i]));
    }
    
    std::cout << "Max diff between CPU and Metal grey_decimate: " << max_diff << std::endl;
    if (max_diff < 1e-5) {
        std::cout << "SUCCESS: Parity verified for grey_decimate!" << std::endl;
    } else {
        std::cerr << "FAILED: Parity mismatch in grey_decimate!" << std::endl;
    }
#endif
}

namespace hhsr {
    Image downsample_metal_for_testing(const Image& src, int factor);
}

void test_downsample() {
    std::cout << "\nTesting Metal kernel_downsample..." << std::endl;
    int h = 128, w = 128;
    Image grey(h, w, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            grey.at(y, x) = (y * w + x) / (f32)(h * w);
        }
    }
    
    // Test factor 2
    Image cpu_down = hhsr::build_pyramid(grey, {1, 2}).levels[1];
    
#ifdef __OBJC__
    Image mtl_down = hhsr::compute_downsample_metal(grey, 2);
    
    if (cpu_down.h != mtl_down.h || cpu_down.w != mtl_down.w) {
        std::cerr << "Dimension mismatch!" << std::endl;
        return;
    }
    
    f32 max_diff = 0;
    for (size_t i = 0; i < cpu_down.data.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(cpu_down.data[i] - mtl_down.data[i]));
    }
    
    std::cout << "Max diff between CPU and Metal downsample: " << max_diff << std::endl;
    if (max_diff < 1e-5) {
        std::cout << "SUCCESS: Parity verified for downsample!" << std::endl;
    } else {
        std::cerr << "FAILED: Parity mismatch in downsample!" << std::endl;
    }
#endif
}

int main() {
    test_grey_decimate();
    test_downsample();
    return 0;
}
