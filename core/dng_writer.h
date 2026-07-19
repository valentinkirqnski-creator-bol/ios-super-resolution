#pragma once
#include "types.h"
#include <string>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace hhsr {

bool write_linear_dng(const std::string& path, const Image& rgb,
                      const std::string& camera_model = "HandheldSR-x2");

// Decode a HandheldSR LinearRaw Deflate DNG (Compression=8, Predictor=2) to planar RGB16.
bool load_linear_dng_rgb16(const std::string& path, std::vector<uint16_t>& rgb,
                           int& W, int& H);

// Same as load_linear_dng_rgb16, plus WB gains (green-normalized) and cam→sRGB 3×3
// when written by DngStreamWriter (private tags). Falls back to identity / 1,1,1.
bool load_linear_dng_rgb16_color(const std::string& path, std::vector<uint16_t>& rgb,
                                 int& W, int& H, float wb[3], float cam_to_srgb[9],
                                 bool& has_color);

// Streaming LinearRaw RGB DNG with lossless Deflate (ZIP) + horizontal predictor.
class DngStreamWriter {
public:
    // colorMatrixXYZtoCam: 9 floats row-major (optional).
    // wbGainsGreenNorm: RGB gains, G≈1 (optional).
    // camToSrgb: 9 floats LibRaw rgb_cam (optional; used by JPEG export).
    bool open(const std::string& path, int W, int H,
              const std::string& camera_model = "HandheldSR-x2",
              int orientation = 1,
              const float* colorMatrixXYZtoCam = nullptr,
              const float* wbGainsGreenNorm = nullptr,
              bool bakedSrgb = false,
              const std::string& camera_make = "HandheldSR",
              const float* camToSrgb = nullptr);

    bool write_rows(const uint16_t* rgb16, int nrows);
    bool close();
    ~DngStreamWriter();

private:
    FILE* f_ = nullptr;
    int W_ = 0, H_ = 0;
    long rows_written_ = 0;
    uint32_t strip_byte_counts_offset_ = 0; // file offset of StripByteCounts LONG
    uint32_t compressed_bytes_ = 0;
    void* z_stream_ = nullptr;             // z_stream*
    std::vector<uint8_t> z_out_;
    std::vector<uint16_t> pred_row_;       // predictor scratch (one row)
    bool deflate_ok_ = false;
};

} // namespace hhsr
