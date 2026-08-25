#pragma once
#include "types.h"
#include <string>
#include <cstdio>
#include <cstdint>
#include <future>
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
// Color-rendering metadata only (wb, matrix) from a linear DNG we wrote --
// no pixel inflate, reads a small file prefix. For when the pixels are
// already in memory (Rgb16Sink).
bool load_linear_dng_color_meta(const std::string& path, float wb[3],
                                float cam_to_srgb[9], bool& has_color);
bool load_linear_dng_rgb16_color(const std::string& path, std::vector<uint16_t>& rgb,
                                 int& W, int& H, float wb[3], float cam_to_srgb[9],
                                 bool& has_color);

// Streaming LinearRaw RGB DNG with fast lossless Deflate (ZIP), no predictor.
// Same decoded pixels as before; write path optimized for merge latency.
// Option A highlight headroom (Config::dng_store_unwhitened): whether the
// encoder should divide the stored rows by the WB gains, and those gains.
// One definition so the writer's AsShotNeutral branch and every encoder make
// the same decision. Active only for the linear (non-baked) RGB DNG of a
// pre-whitened merge with valid gains.
inline bool dng_unwhiten_active(const Config& cfg, int nch) {
    if (!cfg.dng_store_unwhitened || !cfg.raw_prewhitened || cfg.bake_srgb ||
        nch < 3)
        return false;
    for (int i = 0; i < 3; ++i)
        if (!(cfg.white_balance[i] > 1e-6f) ||
            !std::isfinite(cfg.white_balance[i]))
            return false;
    return true;
}
// Per-channel multipliers applied to the stored rows: 1/gain, G-normalised.
inline void dng_unwhiten_gains(const Config& cfg, int nch, float g[3]) {
    const bool on = dng_unwhiten_active(cfg, nch);
    g[0] = on ? cfg.white_balance[1] / cfg.white_balance[0] : 1.f;
    g[1] = 1.f;
    g[2] = on ? cfg.white_balance[1] / cfg.white_balance[2] : 1.f;
}

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
              bool pixelsPrewhitened = false,
              bool losslessJpeg = false);

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

    // Lossless-JPEG tiled mode (Config::dng_lossless_jpeg): rows accumulate
    // into one tile-row band; full bands encode their tiles in parallel and
    // stream out. Tile offset/count arrays are patched at close.
    bool lossless_ = false;
    int tile_w_ = 0, tile_l_ = 0, ntx_ = 0, nty_ = 0;
    int rows_in_band_ = 0;
    std::vector<uint16_t> band_buf_;
    // One band in flight: the filled band encodes + writes on a worker while
    // the caller fills the other buffer, so tile encoding hides behind the
    // next band's normalize/readback instead of extending the tail.
    std::vector<uint16_t> band_back_;
    std::future<bool> band_fut_;
    std::vector<uint32_t> tile_offsets_, tile_counts_;
    uint32_t tile_off_arr_pos_ = 0, tile_cnt_arr_pos_ = 0;
    bool flush_band();
    bool encode_and_write_band(std::vector<uint16_t>& band, int valid_rows);
};

} // namespace hhsr
