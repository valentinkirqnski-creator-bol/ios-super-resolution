#pragma once
//
// Minimal, dependency-free DNG writer. Produces an uncompressed 16-bit
// "Linear Raw" DNG (SamplesPerPixel = 3) from a demosaicked RGB image.
// Readable by Adobe Camera Raw, darktable and RawTherapee.
//
// Two APIs:
//   * write_linear_dng()  — one-shot, from a full in-memory Image.
//   * DngStreamWriter     — streams 16-bit rows incrementally so the full
//                           (up to 48 MP) output never has to live in RAM.
//
#include "types.h"
#include <string>
#include <cstdio>
#include <cstdint>

namespace hhsr {

bool write_linear_dng(const std::string& path, const Image& rgb,
                      const std::string& camera_model = "HandheldSR-x2");

class DngStreamWriter {
public:
    // Opens `path` and writes the DNG header/IFD for a W x H, 3-sample image.
    //   orientation : EXIF orientation (1 = normal, 6/8 = 90 deg rotations).
    //   colorMatrixXYZtoCam : 9 floats (ColorMatrix1, XYZ->camera) or nullptr
    //                         to fall back to identity.
    //   wbGainsGreenNorm    : 3 green-normalized camera WB gains (r,g,b) used to
    //                         emit AsShotNeutral, or nullptr to omit it.
    //   bakedSrgb : when true, writes a plain fully-developed 16-bit RGB image
    //               (PhotometricInterpretation = RGB) that displays correctly in
    //               any viewer. When false, writes a camera-native LinearRaw DNG.
    bool open(const std::string& path, int W, int H,
              const std::string& camera_model = "HandheldSR-x2",
              int orientation = 1,
              const float* colorMatrixXYZtoCam = nullptr,
              const float* wbGainsGreenNorm = nullptr,
              bool bakedSrgb = false);

    // Appends `nrows` full-width rows of interleaved 16-bit RGB (row-major,
    // W*3 samples per row). Rows must be supplied top-to-bottom in order.
    bool write_rows(const uint16_t* rgb16, int nrows);

    bool close();
    ~DngStreamWriter();

private:
    FILE* f_ = nullptr;
    int W_ = 0, H_ = 0;
    long rows_written_ = 0;
};

} // namespace hhsr
