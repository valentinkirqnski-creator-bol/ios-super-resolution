#include "dng_writer.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace hhsr {

// --- Little-endian TIFF/DNG writer -------------------------------------
enum : uint16_t { T_BYTE = 1, T_ASCII = 2, T_SHORT = 3, T_LONG = 4,
                  T_RATIONAL = 5, T_SRATIONAL = 10 };

namespace {

static inline uint32_t type_size(uint16_t t) {
    switch (t) {
        case T_BYTE: case T_ASCII: return 1;
        case T_SHORT: return 2;
        case T_LONG: return 4;
        case T_RATIONAL: case T_SRATIONAL: return 8;
        default: return 1;
    }
}

// One IFD entry. If `payload` is non-empty it is written to the heap and the
// entry stores its offset; otherwise `inlineval` holds the (<=4 byte) value.
struct Entry {
    uint16_t tag = 0;
    uint16_t type = 0;
    uint32_t count = 0;
    uint32_t inlineval = 0;
    std::vector<uint8_t> payload; // external data (empty => inline)
};

static void w16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(v & 0xFF); b.push_back(v >> 8); }
static void w32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(v & 0xFF); b.push_back((v >> 8) & 0xFF);
    b.push_back((v >> 16) & 0xFF); b.push_back((v >> 24) & 0xFF);
}

struct IFD {
    std::vector<Entry> e;

    void shortv(uint16_t tag, uint16_t v) { e.push_back({tag, T_SHORT, 1, (uint32_t)v, {}}); }
    void longv(uint16_t tag, uint32_t v)  { e.push_back({tag, T_LONG, 1, v, {}}); }

    void shorts(uint16_t tag, std::vector<uint16_t> vals) {
        if (vals.size() == 1) { shortv(tag, vals[0]); return; }
        if (vals.size() == 2) {
            e.push_back({tag, T_SHORT, 2, (uint32_t)vals[0] | ((uint32_t)vals[1] << 16), {}});
            return;
        }
        std::vector<uint8_t> p;
        for (uint16_t v : vals) w16(p, v);
        e.push_back({tag, T_SHORT, (uint32_t)vals.size(), 0, p});
    }
    void bytes4(uint16_t tag, uint8_t a, uint8_t b, uint8_t c, uint8_t dd) {
        uint32_t v = (uint32_t)a | ((uint32_t)b << 8) | ((uint32_t)c << 16) | ((uint32_t)dd << 24);
        e.push_back({tag, T_BYTE, 4, v, {}});
    }
    void ascii(uint16_t tag, const std::string& s) {
        std::string z = s; z.push_back('\0');
        if (z.size() <= 4) {
            uint32_t v = 0; memcpy(&v, z.data(), z.size());
            e.push_back({tag, T_ASCII, (uint32_t)z.size(), v, {}});
        } else {
            std::vector<uint8_t> p(z.begin(), z.end());
            e.push_back({tag, T_ASCII, (uint32_t)z.size(), 0, p});
        }
    }
    void srational(uint16_t tag, std::vector<int32_t> nd) {
        std::vector<uint8_t> p;
        for (int32_t v : nd) w32(p, (uint32_t)v);
        e.push_back({tag, T_SRATIONAL, (uint32_t)(nd.size() / 2), 0, p});
    }
    void rational(uint16_t tag, std::vector<uint32_t> nd) {
        std::vector<uint8_t> p;
        for (uint32_t v : nd) w32(p, v);
        e.push_back({tag, T_RATIONAL, (uint32_t)(nd.size() / 2), 0, p});
    }
    void longs(uint16_t tag, std::vector<uint32_t> vals) {
        if (vals.size() == 1) { longv(tag, vals[0]); return; }
        // Multi-value LONG tags (crop size, active area) need external storage
        // because image dimensions exceed the 4-byte inline limit.
        std::vector<uint8_t> p;
        for (uint32_t v : vals) w32(p, v);
        e.push_back({tag, T_LONG, (uint32_t)vals.size(), 0, p});
    }
};

} // namespace

// Builds the DNG header + IFD + heap bytes for a W x H, 3-sample image and
// returns them plus the file offset at which the strip data begins.
static std::vector<uint8_t> build_dng_prefix(int W, int H,
                                             const std::string& camera_make,
                                             const std::string& camera_model,
                                             int orientation,
                                             const float* cm /*9, XYZ->cam*/,
                                             const float* wb /*3, green-norm, WB in pixels*/,
                                             bool baked_srgb,
                                             uint32_t& strip_offset_out) {
    const uint32_t strip_bytes = (uint32_t)W * H * 3 * 2;

    IFD ifd;
    ifd.longv(254, 0);                 // NewSubfileType
    ifd.ascii(271, camera_make.empty() ? "HandheldSR" : camera_make); // Make
    ifd.ascii(272, camera_model.empty() ? "HandheldSR-x2" : camera_model); // Model
    ifd.longv(256, (uint32_t)W);       // ImageWidth
    ifd.longv(257, (uint32_t)H);       // ImageLength
    ifd.shorts(258, {16, 16, 16});     // BitsPerSample
    ifd.shortv(259, 1);                // Compression = none
    if (baked_srgb)
        ifd.shortv(262, 2);            // PhotometricInterpretation = RGB (sRGB baked)
    else
        ifd.shortv(262, 34892);        // PhotometricInterpretation = LinearRaw
    ifd.longv(273, 0);                 // StripOffsets (patched after layout)
    if (orientation >= 1 && orientation <= 8)
        ifd.shortv(274, (uint16_t)orientation); // Orientation
    ifd.shortv(277, 3);                // SamplesPerPixel
    ifd.longv(278, (uint32_t)H);       // RowsPerStrip
    ifd.longv(279, strip_bytes);       // StripByteCounts
    ifd.shortv(284, 1);                // PlanarConfiguration = chunky
    ifd.shorts(339, {1, 1, 1});        // SampleFormat = unsigned int

    // Crop / active area — required by iOS Photos to show non-zero dimensions.
    ifd.longs(50719, {0, 0});                      // DefaultCropOrigin
    ifd.longs(50720, {(uint32_t)W, (uint32_t)H});  // DefaultCropSize
    ifd.longs(50829, {0, 0, (uint32_t)H, (uint32_t)W}); // ActiveArea

    // DNG identity tags — always written so viewers (incl. iOS Photos) recognize
    // the file as DNG, not a generic RGB TIFF.
    ifd.bytes4(50706, 1, 4, 0, 0);     // DNGVersion 1.4.0.0
    ifd.bytes4(50707, 1, 3, 0, 0);     // DNGBackwardVersion 1.3.0.0
    ifd.ascii(50708, camera_model.empty() ? "HandheldSR-x2" : camera_model); // UniqueCameraModel
    ifd.shorts(50714, {0, 0, 0});      // BlackLevel (per channel)
    ifd.longs(50717, {65535, 65535, 65535}); // WhiteLevel (per channel)

    if (!baked_srgb) {
        ifd.shortv(50778, 21);             // CalibrationIlluminant1 = D65
        // Identity ColorMatrix1. Merged pixels already include camera WB; writing
        // the real matrix plus AsShotNeutral/AnalogBalance makes iOS and Lightroom
        // Mobile re-develop color and produces a magenta cast. This matches the
        // first working build (Lightroom import OK).
        ifd.srational(50721, {1,1, 0,1, 0,1,  0,1, 1,1, 0,1,  0,1, 0,1, 1,1});
        // Do NOT write AsShotNeutral or AnalogBalance — WB is in the pixel data.
    }

    // IFD entries must be sorted by tag.
    std::sort(ifd.e.begin(), ifd.e.end(), [](const Entry& a, const Entry& b) { return a.tag < b.tag; });

    const uint32_t n = (uint32_t)ifd.e.size();
    const uint32_t ifd_offset = 8;
    const uint32_t ifd_size = 2 + n * 12 + 4;
    const uint32_t heap_base = ifd_offset + ifd_size;

    // Assign heap offsets, then the strip after the heap (2-byte aligned).
    std::vector<uint8_t> heap;
    for (auto& e : ifd.e) {
        if (!e.payload.empty()) {
            if (heap.size() & 1) heap.push_back(0);
            e.inlineval = heap_base + (uint32_t)heap.size();
            heap.insert(heap.end(), e.payload.begin(), e.payload.end());
        }
    }
    uint32_t strip_offset = heap_base + (uint32_t)heap.size();
    if (strip_offset & 1) strip_offset += 1;
    for (auto& e : ifd.e) if (e.tag == 273) e.inlineval = strip_offset;

    // Serialize.
    std::vector<uint8_t> out;
    out.push_back('I'); out.push_back('I');
    w16(out, 42);
    w32(out, ifd_offset);
    w16(out, (uint16_t)n);
    for (const auto& e : ifd.e) {
        w16(out, e.tag);
        w16(out, e.type);
        w32(out, e.count);
        w32(out, e.inlineval);
    }
    w32(out, 0); // no next IFD
    out.insert(out.end(), heap.begin(), heap.end());
    while (out.size() < strip_offset) out.push_back(0);

    strip_offset_out = strip_offset;
    return out;
}

bool write_linear_dng(const std::string& path, const Image& rgb, const std::string& camera_model) {
    if (rgb.h <= 0 || rgb.w <= 0) return false;
    DngStreamWriter w;
    if (!w.open(path, rgb.w, rgb.h, camera_model)) return false;
    std::vector<uint16_t> row((size_t)rgb.w * 3);
    for (int y = 0; y < rgb.h; ++y) {
        for (int x = 0; x < rgb.w; ++x)
            for (int c = 0; c < 3; ++c) {
                f32 v = clampf(rgb.c >= 3 ? rgb.at(y, x, c) : rgb.at(y, x, 0), 0.f, 1.f);
                row[(size_t)x * 3 + c] = (uint16_t)(v * 65535.f + 0.5f);
            }
        if (!w.write_rows(row.data(), 1)) return false;
    }
    return w.close();
}

// --- Streaming writer ---
bool DngStreamWriter::open(const std::string& path, int W, int H, const std::string& camera_model,
                           int orientation, const float* colorMatrixXYZtoCam,
                           const float* wbGainsGreenNorm, bool bakedSrgb,
                           const std::string& camera_make) {
    if (W <= 0 || H <= 0) return false;
    W_ = W; H_ = H; rows_written_ = 0;
    uint32_t strip_offset = 0;
    std::vector<uint8_t> prefix = build_dng_prefix(W, H, camera_make, camera_model, orientation,
                                                   colorMatrixXYZtoCam, wbGainsGreenNorm,
                                                   bakedSrgb, strip_offset);
    f_ = fopen(path.c_str(), "wb");
    if (!f_) return false;
    return fwrite(prefix.data(), 1, prefix.size(), f_) == prefix.size();
}

bool DngStreamWriter::write_rows(const uint16_t* rgb16, int nrows) {
    if (!f_ || nrows <= 0) return false;
    if (rows_written_ + nrows > H_) nrows = H_ - (int)rows_written_;
    if (nrows <= 0) return true;
    // uint16 samples are little-endian on all supported ABIs (arm64/x86); write raw.
    size_t n = (size_t)nrows * W_ * 3;
    bool ok = fwrite(rgb16, sizeof(uint16_t), n, f_) == n;
    rows_written_ += nrows;
    return ok;
}

bool DngStreamWriter::close() {
    if (!f_) return false;
    fclose(f_);
    f_ = nullptr;
    return rows_written_ == H_;
}

DngStreamWriter::~DngStreamWriter() { if (f_) fclose(f_); }

} // namespace hhsr
