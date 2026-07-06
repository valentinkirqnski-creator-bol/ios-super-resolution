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
// Returns empty vector if libraw is not compiled in.
std::vector<Image> load_raw_burst(const std::vector<std::string>& paths, Config& cfg);

// Convert a normalized RGB float image [H,W,3] (0..1) to packed 8-bit RGBA
// (for handing back to Android Bitmap). Applies sRGB gamma.
std::vector<uint8_t> to_rgba8(const Image& rgb);

} // namespace hhsr
