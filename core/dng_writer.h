#pragma once
#include "types.h"
#include <string>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace hhsr {

// Capture-time EXIF the writer may embed in a real Exif sub-IFD (tag 34665).
// Mirrors Config's capture_* fields (see types.h). 0 / empty = not available --
// the matching tag is omitted, never written as a false zero.
struct CaptureExif {
    float iso = 0.f;
    float exposure_seconds = 0.f;
    float f_number = 0.f;
    float focal_length_mm = 0.f;
    std::string lens_model;
    std::string datetime;  // "YYYY:MM:DD HH:MM:SS"; empty = use file time
};

bool write_linear_dng(const std::string& path, const Image& rgb,
                      const std::string& camera_model = "HandheldSR-x2");

// Decode a HandheldSR LinearRaw DNG to interleaved RGB16. Handles every codec
// the writer emits: uncompressed (1), lossless JPEG (7), Deflate (8, Predictor
// 1 or 2), single- or multi-strip.
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

// Everything a renderer needs to get colour right, read back verbatim rather
// than pre-combined. load_linear_dng_rgb16_color hands out one already-derived
// camera→sRGB matrix, and which space that matrix expects depends on how the
// file was written -- LibRaw's rgb_cam wants white-balanced input, the
// ColorMatrix-derived fallback does not. A renderer that white-balances itself
// cannot tell the two apart from the matrix alone, and picking wrong is a
// systematic cast. This hands back the ingredients instead, so the caller can
// build the matrix for the space it actually renders in.
struct LinearDngColorInfo {
    float wb[3] = {1.f, 1.f, 1.f};      // gains the renderer still has to apply
    bool  has_wb = false;
    float cam_to_srgb[9] = {1,0,0, 0,1,0, 0,0,1};
    bool  has_cam_to_srgb = false;
    float color_matrix[9] = {0};        // ColorMatrix1: XYZ(D65) -> camera
    bool  has_color_matrix = false;
    float analog_balance[3] = {1.f, 1.f, 1.f};  // gains ALREADY baked into pixels
    bool  has_analog_balance = false;
    // AsShotNeutral: the camera neutral expressed in the STORED pixels' space.
    // The authoritative white balance -- it is the one a reader can check the
    // pixels against, and it is what every other DNG reader uses. The private
    // tag's gains agree with it when both are present, and it is present in
    // files written before that tag existed.
    float as_shot_neutral[3] = {1.f, 1.f, 1.f};
    bool  has_as_shot_neutral = false;
    int   orientation = 1;              // TIFF tag 274
};

bool load_linear_dng_rgb16_info(const std::string& path, std::vector<uint16_t>& rgb,
                                int& W, int& H, LinearDngColorInfo& info);

// Streaming LinearRaw RGB DNG. The image strip is written with one of the
// lossless codecs in Config::DngCodec; all of them decode to the same uint16
// samples, so the choice is size/latency only.
//
// Highlight headroom (Config::dng_store_unwhitened): whether the encoder should
// divide the stored rows by the WB gains, and those gains. One definition so
// the writer's AsShotNeutral branch and every encoder make the same decision.
// Active only for the linear (non-baked) RGB DNG of a pre-whitened merge with
// valid gains.
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
              int codec = Config::DNG_CODEC_LJPEG,
              const CaptureExif* exif = nullptr,
              int numThreads = 0);

    // Build a CaptureExif from a Config's capture_* fields.
    static CaptureExif exif_from_config(const Config& cfg) {
        CaptureExif e;
        e.iso = cfg.capture_iso;
        e.exposure_seconds = cfg.capture_exposure_seconds;
        e.f_number = cfg.capture_f_number;
        e.focal_length_mm = cfg.capture_focal_length_mm;
        e.lens_model = cfg.capture_lens_model;
        e.datetime = cfg.capture_datetime;
        return e;
    }

    bool write_rows(const uint16_t* rgb16, int nrows);
    bool close();
    ~DngStreamWriter();

    // Rows per strip in the lossless-JPEG path. Every strip is an independent
    // SOI..EOI stream, so this is the parallel encode granularity as well as
    // the TIFF one. 64 keeps the per-thread scratch at ~2.9MB on a 48MP frame
    // while still giving a 480-row merge band 7-8 strips to spread over cores;
    // measured size is flat to within 0.06% anywhere from 30 to 252 rows, and
    // every multi-strip layout beats a single strip (per-strip Huffman tables
    // adapt to local statistics).
    static constexpr int kLjpegStripRows = 64;

private:
    bool flush_ljpeg_strip(const uint16_t* rows16, int nrows);
    bool encode_band_ljpeg(const uint16_t* rgb16, int nrows);
    bool join_async();

    FILE* f_ = nullptr;
    int W_ = 0, H_ = 0;
    long rows_written_ = 0;
    uint32_t strip_byte_counts_pos_ = 0;   // file offset of the StripByteCounts array
    uint32_t strip_offsets_pos_ = 0;       // file offset of the StripOffsets array
    uint32_t compressed_bytes_ = 0;
    int codec_ = Config::DNG_CODEC_NONE;
    int num_threads_ = 0;
    void* z_stream_ = nullptr;             // z_stream*
    std::vector<uint8_t> z_out_;
    bool deflate_ok_ = false;

    // Lossless-JPEG state: one entry per strip, patched into the two LONG
    // arrays at close(), plus the carry buffer holding the rows of a strip that
    // a merge band ended part-way through.
    std::vector<uint32_t> strip_offsets_;
    std::vector<uint32_t> strip_sizes_;
    std::vector<uint16_t> pending_;
    int pending_rows_ = 0;
    int encoded_rows_ = 0;                 // rows actually through the encoder
    uint32_t next_strip_offset_ = 0;

    // The encode runs on a worker so it overlaps whatever produces the next
    // band -- the merge, on the paths that still have GPU work in flight.
    // write_rows takes a COPY of the caller's rows rather than borrowing them:
    // the online path hands back a single pooled GPU buffer that the next band
    // overwrites, and the banded/fused paths each have their own double-
    // buffering scheme. Copying (~2ms for a 480-row band, against tens of ms of
    // encode) keeps write_rows' contract exactly as synchronous as it was, so
    // no call site has to reason about the worker's lifetime.
    std::vector<uint16_t> async_buf_;
    std::vector<std::vector<uint8_t>> enc_scratch_;   // one per strip slot, reused
    int async_rows_ = 0;
    bool async_ok_ = true;
    void* async_thread_ = nullptr;         // std::thread*, kept out of the header
};

} // namespace hhsr
