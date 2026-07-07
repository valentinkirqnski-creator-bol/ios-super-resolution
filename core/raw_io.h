#pragma once
//
// RAW input/output. Two paths are provided:
//   1. A synthetic burst generator, so the full pipeline is runnable and
//      testable on-device immediately without any RAW files or extra deps.
//   2. A libraw-backed DNG/RAW loader (compiled only when HAVE_LIBRAW is set
//      and a prebuilt arm64 libraw is linked; see CMakeLists.txt).
//
#include "types.h"
#include <vector>
#include <string>

namespace hhsr {

// Generates `n` slightly shifted/noisy frames of a synthetic Bayer scene.
// Useful for validating the pipeline end-to-end on a device.
std::vector<Image> synth_burst(int h, int w, int n, unsigned seed = 1234);

// Loads a burst of DNG/RAW files (first path == reference). Fills `cfg` fields
// read from metadata (CFA, white balance, alpha/beta) when available.
std::vector<Image> load_raw_burst(const std::vector<std::string>& paths, Config& cfg);

// Loads a single DNG/RAW frame. When `is_reference`, populates `cfg` metadata.
// If crop_h/crop_w > 0, crops to that even size (top-left) for burst consistency.
Image load_raw_frame(const std::string& path, Config& cfg, bool is_reference,
                     int crop_h = 0, int crop_w = 0);

// Convert a normalized RGB float image [H,W,3] (0..1) to packed 8-bit RGBA
// (for handing back to Android Bitmap). Applies sRGB gamma.
std::vector<uint8_t> to_rgba8(const Image& rgb);

} // namespace hhsr
