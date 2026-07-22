#pragma once
#include "types.h"
#include <string>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace hhsr {

bool write_linear_dng(const std::string& path, const Image& rgb,
                      const std::string& camera_model = "HandheldSR-x2");

// Decode a HandheldSR LinearRaw Deflate DNG (Compression=8, Predictor=1 or 2) to planar RGB16.
bool load_linear_dng_rgb16(const std::string& path, std::vector<uint16_t>& rgb,
                           int& W, int& H);

// Embed a complete JPEG bitstream as DNG SubIFD preview (NewSubfileType=1,
// Compression=7). Keeps IFD0 LinearRaw for Lightroom; Photos/ImageIO use the
// preview for thumbnail / display when they cannot decode Deflate LinearRaw.
bool embed_dng_jpeg_preview(const std::string& path,
                            const uint8_t* jpeg, size_t jpeg_len,
                            int jpeg_w, int jpeg_h);

// Same as load_linear_dng_rgb16, plus WB gains (green-normalized) and cam→sRGB 3×3
// when written by DngStreamWriter (private tags). Falls back to identity / 1,1,1.
bool load_linear_dng_rgb16_color(const std::string& path, std::vector<uint16_t>& rgb,
                                 int& W, int& H, float wb[3], float cam_to_srgb[9],
                                 bool& has_color);

// Streaming LinearRaw RGB DNG with fast lossless Deflate (ZIP), no predictor.
// Same decoded pixels as before; write path optimized for merge latency.
class DngStreamWriter {
public:
    // colorMatrixXYZtoCam: 9 floats row-major (optional).
    // wbGainsGreenNorm: RGB gains, G≈1 (optional).
    // camToSrgb: 9 floats LibRaw rgb_cam (optional; used by JPEG export).
    // pixelsPrewhitened: merge RGB already has WB baked (Python utils_dng order).
    //   → AsShotNeutral=1,1,1 + AnalogBalance=gains (tag 50727); private WB=1,1,1.
    //   Otherwise AsShotNeutral=1/gains and private tag stores gains for JPEG.
    bool open(const std::string& path, int W, int H,
              const std::string& camera_model = "HandheldSR-x2",
              int orientation = 1,
              const float* colorMatrixXYZtoCam = nullptr,
              const float* wbGainsGreenNorm = nullptr,
              bool bakedSrgb = false,
              const std::string& camera_make = "HandheldSR",
              const float* camToSrgb = nullptr,
              bool pixelsPrewhitened = false);

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
    bool deflate_ok_ = false;
};

} // namespace hhsr
