#include "dng_writer.h"
#include "lj92_enc.h"
#include "parallel.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <utility>
#include <zlib.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#if defined(__APPLE__)
#include <fcntl.h>
#endif

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

struct Entry {
    uint16_t tag = 0;
    uint16_t type = 0;
    uint32_t count = 0;
    uint32_t inlineval = 0;
    std::vector<uint8_t> payload;
};

static void w16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(v & 0xFF); b.push_back(v >> 8); }
static void w32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(v & 0xFF); b.push_back((v >> 8) & 0xFF);
    b.push_back((v >> 16) & 0xFF); b.push_back((v >> 24) & 0xFF);
}

static uint16_t r16(const uint8_t* p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static uint32_t r32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
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
        std::vector<uint8_t> p;
        for (uint32_t v : vals) w32(p, v);
        e.push_back({tag, T_LONG, (uint32_t)vals.size(), 0, p});
    }
};

static std::string now_tiff_datetime() {
    char buf[20];
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::strftime(buf, sizeof(buf), "%Y:%m:%d %H:%M:%S", &tm);
    return std::string(buf);
}

// TIFF Predictor=2 horizontal differencing (chunky RGB16), in-place, right→left.
static void apply_hdiff_rgb16(uint16_t* row, int W) {
    for (int x = W - 1; x >= 1; --x) {
        row[x * 3 + 0] = (uint16_t)(row[x * 3 + 0] - row[(x - 1) * 3 + 0]);
        row[x * 3 + 1] = (uint16_t)(row[x * 3 + 1] - row[(x - 1) * 3 + 1]);
        row[x * 3 + 2] = (uint16_t)(row[x * 3 + 2] - row[(x - 1) * 3 + 2]);
    }
}

static void undo_hdiff_rgb16(uint16_t* row, int W) {
    for (int x = 1; x < W; ++x) {
        row[x * 3 + 0] = (uint16_t)(row[x * 3 + 0] + row[(x - 1) * 3 + 0]);
        row[x * 3 + 1] = (uint16_t)(row[x * 3 + 1] + row[(x - 1) * 3 + 1]);
        row[x * 3 + 2] = (uint16_t)(row[x * 3 + 2] + row[(x - 1) * 3 + 2]);
    }
}

static void append_f32_le(std::vector<uint8_t>& p, float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    w32(p, u);
}

static bool invert_3x3(const float* m, float* inv) {
    const float a = m[0], b = m[1], c = m[2];
    const float d = m[3], e = m[4], f = m[5];
    const float g = m[6], h = m[7], i = m[8];
    const float A = e * i - f * h;
    const float B = c * h - b * i;
    const float C = b * f - c * e;
    const float D = f * g - d * i;
    const float E = a * i - c * g;
    const float F = c * d - a * f;
    const float G = d * h - e * g;
    const float H = b * g - a * h;
    const float I = a * e - b * d;
    const float det = a * A + b * D + c * G;
    if (!(std::fabs(det) > 1e-8f) || !std::isfinite(det)) return false;
    const float s = 1.f / det;
    inv[0] = A * s; inv[1] = B * s; inv[2] = C * s;
    inv[3] = D * s; inv[4] = E * s; inv[5] = F * s;
    inv[6] = G * s; inv[7] = H * s; inv[8] = I * s;
    for (int k = 0; k < 9; ++k)
        if (!std::isfinite(inv[k])) return false;
    return true;
}

// Scale every row to sum to this. The matrix is applied AFTER automatic
// exposure, so its overall gain feeds straight into the tone curve -- which was
// tuned against a matrix whose rows summed to about 0.612. Equal sums also mean
// a neutral camera value renders neutral, which unequal ones quietly do not.
static constexpr float kNeutralRowSum = 0.6124f;

static bool derive_cam_to_srgb_from_color_matrix(const float* color_matrix,
                                                 const float* analog_balance,
                                                 float* out) {
    // DNG ColorMatrix is XYZ -> camera. Invert to camera -> XYZ, then convert XYZ to sRGB.
    float cam_to_xyz[9];
    if (!color_matrix || !invert_3x3(color_matrix, cam_to_xyz)) return false;
    constexpr float xyz_to_srgb[9] = {
         3.2406f, -1.5372f, -0.4986f,
        -0.9689f,  1.8758f,  0.0415f,
         0.0557f, -0.2040f,  1.0570f
    };
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) {
            out[r * 3 + col] =
                xyz_to_srgb[r * 3 + 0] * cam_to_xyz[0 * 3 + col] +
                xyz_to_srgb[r * 3 + 1] * cam_to_xyz[1 * 3 + col] +
                xyz_to_srgb[r * 3 + 2] * cam_to_xyz[2 * 3 + col];
        }
    }
    // The stored pixels are pre-white-balanced, so what sits in the file is
    // AnalogBalance . ColorMatrix . XYZ, not ColorMatrix . XYZ. Undoing only the
    // colour matrix therefore leaves the WB gains baked into the transform.
    // Dividing column c by AnalogBalance[c] is the missing inverse, and it is
    // what put red 17% low in blues (rendered hue 206 against a reference 217,
    // read as teal) and 25% high in warm areas (hue 22 against 31).
    //
    // It also means the correct matrix differs per shot, because AnalogBalance
    // is that shot's white balance -- one hardcoded matrix cannot be right for
    // every capture.
    if (analog_balance) {
        for (int c = 0; c < 3; ++c) {
            const float ab = analog_balance[c];
            if (!(std::fabs(ab) > 1e-6f) || !std::isfinite(ab)) return false;
            for (int r = 0; r < 3; ++r) out[r * 3 + c] /= ab;
        }
    }

    for (int r = 0; r < 3; ++r) {
        const float sum = out[r * 3 + 0] + out[r * 3 + 1] + out[r * 3 + 2];
        if (!(std::fabs(sum) > 1e-6f) || !std::isfinite(sum)) return false;
        const float k = kNeutralRowSum / sum;
        for (int c = 0; c < 3; ++c) out[r * 3 + c] *= k;
    }

    for (int k = 0; k < 9; ++k)
        if (!std::isfinite(out[k])) return false;
    return true;
}

static bool is_identity_3x3(const float* m) {
    if (!m) return false;
    for (int k = 0; k < 9; ++k) {
        const float target = (k % 4 == 0) ? 1.f : 0.f;
        if (std::fabs(m[k] - target) > 1e-5f) return false;
    }
    return true;
}

} // namespace

// Deflate of a 48MP LinearRaw strip measured ~8.6s of single-threaded zlib —
// the single largest item in a burst, and unparallelizable because a zlib
// stream is inherently serial. On 16-bit linear photographic data it only buys
// ~1.2-1.5x, so storing uncompressed trades ~90MB of file size for those 8.6s.
// Decoded pixels are identical either way, and load_linear_dng_rgb16 already
// handles Compression=1. Flip to true to restore Deflate (the zlib path below
// is kept intact); a multi-strip parallel Deflate is the eventual middle ground.
static constexpr bool kDngCompress = false;

// Builds DNG header. StripByteCounts left as 0 — patched once the strip is done.
// Private tag 65000: 12×f32 LE = wb[3] + cam_to_srgb[9] for JPEG export.
static std::vector<uint8_t> build_dng_prefix(int W, int H,
                                             const std::string& camera_make,
                                             const std::string& camera_model,
                                             int orientation,
                                             const float* cm,
                                             const float* wb,
                                             bool baked_srgb,
                                             const float* cam_to_srgb,
                                             bool pixels_prewhitened,
                                             uint32_t& strip_offset_out,
                                             uint32_t& strip_byte_counts_offset_out,
                                             bool lossless = false,
                                             int tile_w = 0, int tile_l = 0,
                                             uint32_t* tile_off_arr_pos = nullptr,
                                             uint32_t* tile_cnt_arr_pos = nullptr) {
    float derived_cam_to_srgb[9];
    const float* jpeg_cam_to_srgb = cam_to_srgb;
    // When the pixels are pre-white-balanced the gains go out as AnalogBalance
    // below, so the derivation has to undo them here too -- otherwise the matrix
    // cached in the private tag disagrees with the one the loader reconstructs
    // from the file, and the two render paths drift apart.
    const float* ab = (pixels_prewhitened && wb) ? wb : nullptr;
    if (!jpeg_cam_to_srgb && cm &&
        derive_cam_to_srgb_from_color_matrix(cm, ab, derived_cam_to_srgb)) {
        jpeg_cam_to_srgb = derived_cam_to_srgb;
    }

    IFD ifd;
    ifd.longv(254, 0);                 // NewSubfileType
    ifd.ascii(271, camera_make.empty() ? "HandheldSR" : camera_make);
    ifd.ascii(272, camera_model.empty() ? "HandheldSR-x2" : camera_model);
    ifd.longv(256, (uint32_t)W);
    ifd.longv(257, (uint32_t)H);
    ifd.shorts(258, {16, 16, 16});
    // 7 = lossless JPEG tiles, 8 = Adobe Deflate, 1 = uncompressed. Same
    // decoded pixels in every case.
    ifd.shortv(259, lossless ? 7 : (kDngCompress ? 8 : 1));
    if (baked_srgb)
        ifd.shortv(262, 2);            // RGB
    else
        ifd.shortv(262, 34892);        // LinearRaw
    const int ntx = (lossless && tile_w > 0) ? (W + tile_w - 1) / tile_w : 0;
    const int nty = (lossless && tile_l > 0) ? (H + tile_l - 1) / tile_l : 0;
    const uint32_t ntiles = (uint32_t)ntx * (uint32_t)nty;
    if (!lossless)
        ifd.longv(273, 0);             // StripOffsets (patched)
    // SubIFDs, reserved empty. Adding this tag later would grow IFD0 by 12
    // bytes and push the image strip along with it, which is why embedding a
    // preview used to rebuild the entire file -- a 292MB read plus a 292MB
    // write plus both buffers resident, just to attach a thumbnail. Reserved
    // here, embedding becomes an append and a four-byte patch.
    ifd.longv(330, 0);                 // SubIFDs (patched by embed_dng_jpeg_preview)
    if (orientation >= 1 && orientation <= 8)
        ifd.shortv(274, (uint16_t)orientation);
    ifd.shortv(277, 3);                // SamplesPerPixel
    if (!lossless) {
        ifd.longv(278, (uint32_t)H);   // RowsPerStrip
        ifd.longv(279, 0);             // StripByteCounts (patched after compress)
    } else {
        ifd.longv(322, (uint32_t)tile_w);
        ifd.longv(323, (uint32_t)tile_l);
        ifd.longs(324, std::vector<uint32_t>(ntiles, 0)); // TileOffsets (patched)
        ifd.longs(325, std::vector<uint32_t>(ntiles, 0)); // TileByteCounts (patched)
    }
    ifd.shortv(284, 1);                // PlanarConfiguration = chunky
    ifd.ascii(305, "HandheldSR");      // Software
    ifd.ascii(306, now_tiff_datetime()); // DateTime
    if (!lossless)
        ifd.shortv(317, 1);            // Predictor = none (unused by lossless JPEG)
    ifd.shorts(339, {1, 1, 1});        // SampleFormat = unsigned

    ifd.longs(50719, {0, 0});
    ifd.longs(50720, {(uint32_t)W, (uint32_t)H});
    ifd.longs(50829, {0, 0, (uint32_t)H, (uint32_t)W});

    ifd.bytes4(50706, 1, 4, 0, 0);     // DNGVersion 1.4.0.0
    ifd.bytes4(50707, 1, 3, 0, 0);     // DNGBackwardVersion 1.3.0.0
    ifd.ascii(50708, camera_model.empty() ? "HandheldSR-x2" : camera_model);
    ifd.shorts(50714, {0, 0, 0});
    ifd.longs(50717, {65535, 65535, 65535});

    if (!baked_srgb) {
        ifd.shortv(50778, 21);         // CalibrationIlluminant1 = D65
        if (cm) {
            std::vector<int32_t> nd;
            nd.reserve(18);
            for (int i = 0; i < 9; ++i) {
                nd.push_back((int32_t)std::lround(cm[i] * 10000.f));
                nd.push_back(10000);
            }
            ifd.srational(50721, std::move(nd));
        } else {
            ifd.srational(50721, {1,1, 0,1, 0,1,  0,1, 1,1, 0,1,  0,1, 0,1, 1,1});
        }
        if (wb) {
            auto to_rat = [](float g) -> std::pair<uint32_t, uint32_t> {
                float n = (g > 1e-6f) ? (1.f / g) : 1.f;
                return {(uint32_t)std::lround(n * 10000.f), 10000u};
            };
            auto gain_rat = [](float g) -> std::pair<uint32_t, uint32_t> {
                float v = (g > 1e-6f) ? g : 1.f;
                return {(uint32_t)std::lround(v * 10000.f), 10000u};
            };
            if (pixels_prewhitened) {
                // Python utils_dng: pixels already × WB → AsShotNeutral=1 + AnalogBalance=gains.
                ifd.rational(50728, {10000, 10000, 10000, 10000, 10000, 10000});
                auto ar = gain_rat(wb[0]), ag = gain_rat(wb[1]), ab = gain_rat(wb[2]);
                ifd.rational(50727, {ar.first, ar.second, ag.first, ag.second, ab.first, ab.second});
            } else {
                auto r = to_rat(wb[0]), g = to_rat(wb[1]), b = to_rat(wb[2]);
                ifd.rational(50728, {r.first, r.second, g.first, g.second, b.first, b.second});
            }
        }
        ifd.shortv(50831, 1);          // ColorimetricReference = scene referred
    }

    if (wb || jpeg_cam_to_srgb) {
        std::vector<uint8_t> blob;
        blob.reserve(48);
        // JPEG/preview must not apply WB again when pixels are already pre-whitened.
        for (int i = 0; i < 3; ++i)
            append_f32_le(blob, (wb && !pixels_prewhitened) ? wb[i] : 1.f);
        for (int i = 0; i < 9; ++i) {
            float v = 0.f;
            if (jpeg_cam_to_srgb) v = jpeg_cam_to_srgb[i];
            else if (i == 0 || i == 4 || i == 8) v = 1.f;
            append_f32_le(blob, v);
        }
        ifd.e.push_back({65000, T_BYTE, (uint32_t)blob.size(), 0, std::move(blob)});
    }

    std::sort(ifd.e.begin(), ifd.e.end(), [](const Entry& a, const Entry& b) {
        return a.tag < b.tag;
    });

    const uint32_t n = (uint32_t)ifd.e.size();
    const uint32_t ifd_offset = 8;
    const uint32_t ifd_size = 2 + n * 12 + 4;
    const uint32_t heap_base = ifd_offset + ifd_size;

    std::vector<uint8_t> heap;
    int strip_off_entry = -1;
    for (int i = 0; i < (int)ifd.e.size(); ++i) {
        auto& e = ifd.e[(size_t)i];
        if (e.tag == 273) strip_off_entry = i;
        if (!e.payload.empty()) {
            if (heap.size() & 1) heap.push_back(0);
            e.inlineval = heap_base + (uint32_t)heap.size();
            heap.insert(heap.end(), e.payload.begin(), e.payload.end());
            if (e.tag == 324 && tile_off_arr_pos) *tile_off_arr_pos = e.inlineval;
            if (e.tag == 325 && tile_cnt_arr_pos) *tile_cnt_arr_pos = e.inlineval;
        }
    }
    uint32_t strip_offset = heap_base + (uint32_t)heap.size();
    if (strip_offset & 1) strip_offset += 1;
    if (strip_off_entry >= 0) ifd.e[(size_t)strip_off_entry].inlineval = strip_offset;

    std::vector<uint8_t> out;
    out.push_back('I'); out.push_back('I');
    w16(out, 42);
    w32(out, ifd_offset);
    w16(out, (uint16_t)n);
    for (int i = 0; i < (int)ifd.e.size(); ++i) {
        const auto& e = ifd.e[(size_t)i];
        if (e.tag == 279) {
            strip_byte_counts_offset_out = (uint32_t)out.size() + 8;
        }
        // Single-tile images store the one-element arrays inline in the IFD.
        if (e.tag == 324 && e.payload.empty() && tile_off_arr_pos)
            *tile_off_arr_pos = (uint32_t)out.size() + 8;
        if (e.tag == 325 && e.payload.empty() && tile_cnt_arr_pos)
            *tile_cnt_arr_pos = (uint32_t)out.size() + 8;
        w16(out, e.tag);
        w16(out, e.type);
        w32(out, e.count);
        w32(out, e.inlineval);
    }
    w32(out, 0);
    out.insert(out.end(), heap.begin(), heap.end());
    while (out.size() < strip_offset) out.push_back(0);

    strip_offset_out = strip_offset;
    (void)type_size;
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

bool DngStreamWriter::open(const std::string& path, int W, int H, const std::string& camera_model,
                           int orientation, const float* colorMatrixXYZtoCam,
                           const float* wbGainsGreenNorm, bool bakedSrgb,
                           const std::string& camera_make, const float* camToSrgb,
                           bool pixelsPrewhitened, bool losslessJpeg) {
    if (W <= 0 || H <= 0) return false;
    W_ = W; H_ = H; rows_written_ = 0;
    compressed_bytes_ = 0;
    strip_byte_counts_offset_ = 0;
    deflate_ok_ = false;
    lossless_ = losslessJpeg;
    tile_off_arr_pos_ = tile_cnt_arr_pos_ = 0;
    tile_offsets_.clear();
    tile_counts_.clear();
    rows_in_band_ = 0;
    if (lossless_) {
        // Tile dims must be multiples of 16 (TIFF); 256x256 balances
        // parallelism (one task per tile) against per-tile header overhead.
        tile_w_ = std::min(256, ((W + 15) / 16) * 16);
        tile_l_ = 256;
        ntx_ = (W + tile_w_ - 1) / tile_w_;
        nty_ = (H + tile_l_ - 1) / tile_l_;
        band_buf_.assign((size_t)tile_l_ * (size_t)W * 3u, 0);
        tile_offsets_.reserve((size_t)ntx_ * (size_t)nty_);
        tile_counts_.reserve((size_t)ntx_ * (size_t)nty_);
    }

    uint32_t strip_offset = 0;
    std::vector<uint8_t> prefix = build_dng_prefix(W, H, camera_make, camera_model, orientation,
                                                   colorMatrixXYZtoCam, wbGainsGreenNorm,
                                                   bakedSrgb, camToSrgb, pixelsPrewhitened,
                                                   strip_offset, strip_byte_counts_offset_,
                                                   lossless_, tile_w_, tile_l_,
                                                   &tile_off_arr_pos_, &tile_cnt_arr_pos_);
    if (lossless_ && (!tile_off_arr_pos_ || !tile_cnt_arr_pos_)) return false;
    f_ = fopen(path.c_str(), "wb+");
    if (!f_) return false;
    // Large stdio buffer — fewer syscalls during streaming writes.
    setvbuf(f_, nullptr, _IOFBF, 1u << 20);
#if defined(__APPLE__)
    // Uncompressed strips push ~290MB through the unified buffer cache, and iOS
    // charges those dirty pages to phys_footprint — the metric jetsam enforces.
    // Measured +167MB peak / -167MB headroom after switching off Deflate. The
    // file is written once and not read back here, so keep it out of the cache.
    (void)fcntl(fileno(f_), F_NOCACHE, 1);
#endif
    if (fwrite(prefix.data(), 1, prefix.size(), f_) != prefix.size()) {
        fclose(f_); f_ = nullptr;
        return false;
    }

    if (lossless_ || !kDngCompress) {
        // Lossless-JPEG tiles or uncompressed: no zlib state at all.
        deflate_ok_ = true;
        return true;
    }

    auto* zs = new z_stream();
    std::memset(zs, 0, sizeof(z_stream));
    // Fastest lossless zlib level — same decoded RGB16, much less CPU than Z_BEST/default.
    if (deflateInit(zs, Z_BEST_SPEED) != Z_OK) {
        delete zs;
        fclose(f_); f_ = nullptr;
        return false;
    }
    z_stream_ = zs;
    z_out_.resize(1u << 20);
    deflate_ok_ = true;
    return true;
}

bool DngStreamWriter::write_rows(const uint16_t* rgb16, int nrows) {
    if (!f_ || !deflate_ok_ || !rgb16 || nrows <= 0) return false;
    if (rows_written_ + nrows > H_) nrows = H_ - (int)rows_written_;
    if (nrows <= 0) return true;

    if (lossless_) {
        const size_t row_elems = (size_t)W_ * 3u;
        int done = 0;
        while (done < nrows) {
            const int take = std::min(nrows - done, tile_l_ - rows_in_band_);
            std::memcpy(band_buf_.data() + (size_t)rows_in_band_ * row_elems,
                        rgb16 + (size_t)done * row_elems,
                        (size_t)take * row_elems * sizeof(uint16_t));
            rows_in_band_ += take;
            done += take;
            rows_written_ += take;
            if (rows_in_band_ == tile_l_ && !flush_band()) return false;
        }
        return true;
    }

    if (!kDngCompress) {
        const size_t nbytes = (size_t)nrows * (size_t)W_ * 3u * sizeof(uint16_t);
        if (fwrite(rgb16, 1, nbytes, f_) != nbytes) return false;
        compressed_bytes_ += (uint32_t)nbytes;
        rows_written_ += nrows;
        return true;
    }

    if (!z_stream_) return false;
    auto* zs = static_cast<z_stream*>(z_stream_);
    // Bulk feed (no per-row copy / predictor) — same pixels, far less overhead.
    const size_t nbytes = (size_t)nrows * (size_t)W_ * 3u * sizeof(uint16_t);
    zs->next_in = reinterpret_cast<Bytef*>(const_cast<uint16_t*>(rgb16));
    zs->avail_in = 0;
    size_t remaining = nbytes;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(rgb16);
    while (remaining > 0) {
        const uInt chunk = (uInt)std::min(remaining, (size_t)0x80000000u);
        zs->next_in = reinterpret_cast<Bytef*>(const_cast<uint8_t*>(src));
        zs->avail_in = chunk;
        src += chunk;
        remaining -= chunk;
        while (zs->avail_in > 0) {
            zs->next_out = z_out_.data();
            zs->avail_out = (uInt)z_out_.size();
            int ret = deflate(zs, Z_NO_FLUSH);
            if (ret != Z_OK) return false;
            size_t produced = z_out_.size() - zs->avail_out;
            if (produced) {
                if (fwrite(z_out_.data(), 1, produced, f_) != produced) return false;
                compressed_bytes_ += (uint32_t)produced;
            }
        }
    }
    rows_written_ += nrows;
    return true;
}

// Encode one tile-row band (valid_rows valid rows; short bottom bands are
// padded by repeating the last row -- decoders discard tile padding) and
// stream the tiles out in order. Tiles encode in parallel: each is an
// independent complete lossless-JPEG stream. Runs on the flush worker, which
// is the only thread that touches f_ between open's prefix and close's
// patches, so file position stays coherent without locking.
bool DngStreamWriter::encode_and_write_band(std::vector<uint16_t>& band,
                                            int valid_rows) {
    const size_t row_elems = (size_t)W_ * 3u;
    for (int y = valid_rows; y < tile_l_; ++y)
        std::memcpy(band.data() + (size_t)y * row_elems,
                    band.data() + (size_t)(valid_rows - 1) * row_elems,
                    row_elems * sizeof(uint16_t));

    std::vector<std::vector<uint8_t>> outs((size_t)ntx_);
    std::vector<uint8_t> okf((size_t)ntx_, 1);
    parallel_rows(ntx_, 0, [&](int tx) {
        std::vector<uint16_t> t((size_t)tile_w_ * (size_t)tile_l_ * 3u);
        const int x0 = tx * tile_w_;
        const int nx = std::min(tile_w_, W_ - x0);
        for (int y = 0; y < tile_l_; ++y) {
            const uint16_t* src = band.data() + (size_t)y * row_elems +
                                  (size_t)x0 * 3u;
            uint16_t* dst = t.data() + (size_t)y * (size_t)tile_w_ * 3u;
            std::memcpy(dst, src, (size_t)nx * 3u * sizeof(uint16_t));
            for (int x = nx; x < tile_w_; ++x)
                std::memcpy(dst + (size_t)x * 3u, dst + (size_t)(nx - 1) * 3u,
                            3u * sizeof(uint16_t));
        }
        if (!lj92_encode_strip(t.data(), tile_w_, tile_l_, 3, outs[(size_t)tx]))
            okf[(size_t)tx] = 0;
    });
    for (int tx = 0; tx < ntx_; ++tx) {
        if (!okf[(size_t)tx]) return false;
        long posl = ftell(f_);
        if (posl < 0) return false;
        if (posl & 1) {  // keep tile offsets word-aligned
            const uint8_t pad = 0;
            if (fwrite(&pad, 1, 1, f_) != 1) return false;
            ++posl;
        }
        const auto& o = outs[(size_t)tx];
        if (fwrite(o.data(), 1, o.size(), f_) != o.size()) return false;
        tile_offsets_.push_back((uint32_t)posl);
        tile_counts_.push_back((uint32_t)o.size());
    }
    return true;
}

// Hand the filled band to the worker and swap in the other buffer. At most
// one band is in flight; a second flush first collects the previous result.
bool DngStreamWriter::flush_band() {
    if (rows_in_band_ <= 0) return true;
    if (band_fut_.valid() && !band_fut_.get()) return false;
    if (band_back_.size() != band_buf_.size())
        band_back_.assign(band_buf_.size(), 0);
    band_buf_.swap(band_back_);
    const int valid = rows_in_band_;
    rows_in_band_ = 0;
    band_fut_ = std::async(std::launch::async, [this, valid]() {
        return encode_and_write_band(band_back_, valid);
    });
    return true;
}

bool DngStreamWriter::close() {
    if (!f_) return false;
    bool ok = rows_written_ == H_ && deflate_ok_ &&
              (lossless_ || !kDngCompress || z_stream_);

    // The flush worker owns f_ while a band is in flight: join it before
    // anything below touches or closes the file, on EVERY path.
    if (lossless_ && band_fut_.valid() && !band_fut_.get()) ok = false;

    if (ok && lossless_) {
        if (rows_in_band_ > 0 && !flush_band()) ok = false;
        if (band_fut_.valid() && !band_fut_.get()) ok = false;
        const size_t want = (size_t)ntx_ * (size_t)nty_;
        if (ok && (tile_offsets_.size() != want || tile_counts_.size() != want))
            ok = false;
        if (ok) {
            auto patch_longs = [&](uint32_t at, const std::vector<uint32_t>& v) -> bool {
                if (fseek(f_, (long)at, SEEK_SET) != 0) return false;
                std::vector<uint8_t> le(v.size() * 4u);
                for (size_t i = 0; i < v.size(); ++i) {
                    le[i * 4 + 0] = (uint8_t)(v[i] & 0xFF);
                    le[i * 4 + 1] = (uint8_t)((v[i] >> 8) & 0xFF);
                    le[i * 4 + 2] = (uint8_t)((v[i] >> 16) & 0xFF);
                    le[i * 4 + 3] = (uint8_t)((v[i] >> 24) & 0xFF);
                }
                return fwrite(le.data(), 1, le.size(), f_) == le.size();
            };
            if (!patch_longs(tile_off_arr_pos_, tile_offsets_) ||
                !patch_longs(tile_cnt_arr_pos_, tile_counts_))
                ok = false;
        }
    }

    if (ok && !lossless_ && kDngCompress) {
        auto* zs = static_cast<z_stream*>(z_stream_);
        int ret;
        do {
            zs->next_out = z_out_.data();
            zs->avail_out = (uInt)z_out_.size();
            ret = deflate(zs, Z_FINISH);
            if (ret != Z_OK && ret != Z_STREAM_END) { ok = false; break; }
            size_t produced = z_out_.size() - zs->avail_out;
            if (produced) {
                if (fwrite(z_out_.data(), 1, produced, f_) != produced) { ok = false; break; }
                compressed_bytes_ += (uint32_t)produced;
            }
        } while (ret != Z_STREAM_END);
    }

    // Patch StripByteCounts for both strip paths: Deflate accumulates the
    // compressed size above, uncompressed the raw size in write_rows.
    if (ok && !lossless_ && strip_byte_counts_offset_ > 0) {
        if (fseek(f_, (long)strip_byte_counts_offset_, SEEK_SET) == 0) {
            uint8_t le[4] = {
                (uint8_t)(compressed_bytes_ & 0xFF),
                (uint8_t)((compressed_bytes_ >> 8) & 0xFF),
                (uint8_t)((compressed_bytes_ >> 16) & 0xFF),
                (uint8_t)((compressed_bytes_ >> 24) & 0xFF),
            };
            if (fwrite(le, 1, 4, f_) != 4) ok = false;
        } else {
            ok = false;
        }
    }

    if (z_stream_) {
        deflateEnd(static_cast<z_stream*>(z_stream_));
        delete static_cast<z_stream*>(z_stream_);
        z_stream_ = nullptr;
    }
    fclose(f_);
    f_ = nullptr;
    deflate_ok_ = false;
    return ok;
}

DngStreamWriter::~DngStreamWriter() {
    if (z_stream_) {
        deflateEnd(static_cast<z_stream*>(z_stream_));
        delete static_cast<z_stream*>(z_stream_);
        z_stream_ = nullptr;
    }
    if (f_) fclose(f_);
}

// Build the preview SubIFD. Every value fits inline, so it needs no heap.
static std::vector<uint8_t> build_preview_ifd(uint32_t jpeg_off, size_t jpeg_len,
                                              int jpeg_w, int jpeg_h) {
    IFD prev;
    prev.longv(254, 1);               // NewSubfileType = reduced resolution
    prev.longv(256, (uint32_t)jpeg_w);
    prev.longv(257, (uint32_t)jpeg_h);
    prev.shorts(258, {8, 8, 8});
    prev.shortv(259, 7);              // JPEG
    prev.shortv(262, 6);              // YCbCr
    prev.longv(273, jpeg_off);        // StripOffsets
    prev.shortv(277, 3);
    prev.longv(278, (uint32_t)jpeg_h);
    prev.longv(279, (uint32_t)jpeg_len);
    prev.shortv(284, 1);
    prev.shorts(530, {2, 2});
    prev.shortv(531, 1);
    std::sort(prev.e.begin(), prev.e.end(),
              [](const Entry& a, const Entry& b) { return a.tag < b.tag; });
    std::vector<uint8_t> out;
    w16(out, (uint16_t)prev.e.size());
    for (const auto& e : prev.e) {
        w16(out, e.tag);
        w16(out, e.type);
        w32(out, e.count);
        w32(out, e.inlineval);
    }
    w32(out, 0);                      // next IFD
    return out;
}

// Append the preview and point the reserved SubIFDs tag at it. Returns false if
// the tag is absent, so the caller can fall back to rebuilding the file.
static bool embed_preview_append(const std::string& path,
                                 const uint8_t* jpeg, size_t jpeg_len,
                                 int jpeg_w, int jpeg_h) {
    FILE* f = fopen(path.c_str(), "r+b");
    if (!f) return false;
    uint8_t hdr[8];
    if (fread(hdr, 1, 8, f) != 8 || hdr[0] != 'I' || hdr[1] != 'I' || r16(hdr + 2) != 42) {
        fclose(f);
        return false;
    }
    const uint32_t ifd0 = r32(hdr + 4);
    if (fseek(f, (long)ifd0, SEEK_SET) != 0) { fclose(f); return false; }
    uint8_t cnt[2];
    if (fread(cnt, 1, 2, f) != 2) { fclose(f); return false; }
    const uint16_t nent = r16(cnt);
    if (nent == 0 || nent > 512) { fclose(f); return false; }
    std::vector<uint8_t> entries((size_t)nent * 12u);
    if (fread(entries.data(), 1, entries.size(), f) != entries.size()) {
        fclose(f);
        return false;
    }

    long subifd_value_off = -1;
    uint32_t existing = 0;
    for (uint16_t i = 0; i < nent; ++i) {
        const uint8_t* e = entries.data() + (size_t)i * 12u;
        if (r16(e) == 330) {
            subifd_value_off = (long)ifd0 + 2 + (long)i * 12 + 8;
            existing = r32(e + 8);
            break;
        }
    }
    if (subifd_value_off < 0) { fclose(f); return false; }   // old file, rebuild
    if (existing != 0) { fclose(f); return true; }            // already embedded

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long end = ftell(f);
    if (end < 16) { fclose(f); return false; }
    if (end & 1) { const uint8_t pad = 0; fwrite(&pad, 1, 1, f); ++end; }
    const uint32_t jpeg_off = (uint32_t)end;
    if (fwrite(jpeg, 1, jpeg_len, f) != jpeg_len) { fclose(f); return false; }

    long after = jpeg_off + (long)jpeg_len;
    if (after & 1) { const uint8_t pad = 0; fwrite(&pad, 1, 1, f); ++after; }
    const uint32_t ifd1_off = (uint32_t)after;
    const std::vector<uint8_t> ifd1 =
        build_preview_ifd(jpeg_off, jpeg_len, jpeg_w, jpeg_h);
    if (fwrite(ifd1.data(), 1, ifd1.size(), f) != ifd1.size()) { fclose(f); return false; }

    // Patch last: until this lands the appended bytes are unreferenced, so a
    // failure part-way leaves a DNG that still reads correctly, just without a
    // preview.
    if (fseek(f, subifd_value_off, SEEK_SET) != 0) { fclose(f); return false; }
    uint8_t v[4] = {(uint8_t)(ifd1_off & 0xFF), (uint8_t)((ifd1_off >> 8) & 0xFF),
                    (uint8_t)((ifd1_off >> 16) & 0xFF), (uint8_t)((ifd1_off >> 24) & 0xFF)};
    const bool ok = fwrite(v, 1, 4, f) == 4;
    fclose(f);
    return ok;
}

bool embed_dng_jpeg_preview(const std::string& path,
                            const uint8_t* jpeg, size_t jpeg_len,
                            int jpeg_w, int jpeg_h) {
    if (!jpeg || jpeg_len < 4 || jpeg_w <= 0 || jpeg_h <= 0) return false;
    // SOI marker
    if (jpeg[0] != 0xFF || jpeg[1] != 0xD8) return false;

    // Fast path: the writer reserves an empty SubIFDs tag, so the preview can be
    // appended and the tag patched in place. Touches a few kilobytes instead of
    // reading and rewriting the whole file. Falls back to the rebuild below for
    // DNGs written before the slot existed.
    if (embed_preview_append(path, jpeg, jpeg_len, jpeg_w, jpeg_h)) return true;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long fsz = ftell(f);
    if (fsz < 16) { fclose(f); return false; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
    std::vector<uint8_t> file((size_t)fsz);
    if (fread(file.data(), 1, file.size(), f) != file.size()) { fclose(f); return false; }
    fclose(f);

    if (file[0] != 'I' || file[1] != 'I' || r16(file.data() + 2) != 42) return false;
    uint32_t ifd0 = r32(file.data() + 4);
    if (ifd0 + 2 > file.size()) return false;
    uint16_t nent = r16(file.data() + ifd0);
    if (ifd0 + 2u + (uint32_t)nent * 12u + 4u > file.size()) return false;

    // Already has a preview — leave alone (idempotent). The value must be
    // checked, not just the tag: the writer now reserves an EMPTY SubIFDs entry
    // so the fast path above can patch it, and testing the tag alone would treat
    // every freshly written DNG as already done and silently embed nothing.
    for (uint16_t i = 0; i < nent; ++i) {
        const uint8_t* e = file.data() + ifd0 + 2 + i * 12;
        if (r16(e) == 330 && r32(e + 8) != 0) return true;
    }

    uint32_t strip_off = 0, strip_bc = 0;
    IFD ifd;
    for (uint16_t i = 0; i < nent; ++i) {
        const uint8_t* e = file.data() + ifd0 + 2 + i * 12;
        uint16_t tag = r16(e), type = r16(e + 2);
        uint32_t count = r32(e + 4), val = r32(e + 8);
        if (tag == 330) continue; // replace below
        Entry ent;
        ent.tag = tag;
        ent.type = type;
        ent.count = count;
        uint32_t nbytes = count * type_size(type);
        if (nbytes <= 4) {
            ent.inlineval = val;
        } else {
            if (val + nbytes > file.size()) return false;
            ent.payload.assign(file.begin() + val, file.begin() + val + nbytes);
            ent.inlineval = 0;
        }
        ifd.e.push_back(std::move(ent));
        if (tag == 273 && type == T_LONG && count == 1) strip_off = val;
        if (tag == 279 && type == T_LONG && count == 1) strip_bc = val;
        if (tag == 273 && type == T_SHORT && count == 1) strip_off = val & 0xFFFF;
        if (tag == 279 && type == T_SHORT && count == 1) strip_bc = val & 0xFFFF;
    }
    if (strip_off == 0 || strip_bc == 0 || strip_off + strip_bc > file.size()) return false;

    // Placeholder SubIFDs — patched after layout.
    ifd.longv(330, 0);

    std::sort(ifd.e.begin(), ifd.e.end(), [](const Entry& a, const Entry& b) {
        return a.tag < b.tag;
    });

    const uint32_t n = (uint32_t)ifd.e.size();
    const uint32_t ifd_offset = 8;
    const uint32_t ifd_size = 2 + n * 12 + 4;
    const uint32_t heap_base = ifd_offset + ifd_size;

    std::vector<uint8_t> heap;
    int strip_off_entry = -1;
    int subifd_entry = -1;
    for (int i = 0; i < (int)ifd.e.size(); ++i) {
        auto& e = ifd.e[(size_t)i];
        if (e.tag == 273) strip_off_entry = i;
        if (e.tag == 330) subifd_entry = i;
        if (!e.payload.empty()) {
            if (heap.size() & 1) heap.push_back(0);
            e.inlineval = heap_base + (uint32_t)heap.size();
            heap.insert(heap.end(), e.payload.begin(), e.payload.end());
        }
    }
    uint32_t new_strip = heap_base + (uint32_t)heap.size();
    if (new_strip & 1) new_strip += 1;
    if (strip_off_entry >= 0) ifd.e[(size_t)strip_off_entry].inlineval = new_strip;

    uint32_t jpeg_off = new_strip + strip_bc;
    if (jpeg_off & 1) jpeg_off += 1;

    // IFD1 (JPEG preview) after JPEG payload
    IFD prev;
    prev.longv(254, 1); // NewSubfileType = Reduced resolution
    prev.longv(256, (uint32_t)jpeg_w);
    prev.longv(257, (uint32_t)jpeg_h);
    prev.shorts(258, {8, 8, 8});
    prev.shortv(259, 7);              // JPEG
    prev.shortv(262, 6);              // YCbCr
    prev.longv(273, jpeg_off);        // StripOffsets
    prev.shortv(277, 3);
    prev.longv(278, (uint32_t)jpeg_h);
    prev.longv(279, (uint32_t)jpeg_len);
    prev.shortv(284, 1);
    prev.shorts(530, {2, 2});         // YCbCrSubSampling 4:2:0-ish (common)
    prev.shortv(531, 1);              // YCbCrPositioning = centered
    std::sort(prev.e.begin(), prev.e.end(), [](const Entry& a, const Entry& b) {
        return a.tag < b.tag;
    });

    const uint32_t ifd1_offset = jpeg_off + (uint32_t)jpeg_len;
    const uint32_t ifd1_aligned = (ifd1_offset + 1u) & ~1u;
    if (subifd_entry >= 0) ifd.e[(size_t)subifd_entry].inlineval = ifd1_aligned;

    std::vector<uint8_t> out;
    out.reserve((size_t)new_strip + strip_bc + jpeg_len + 512);
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
    w32(out, 0); // next IFD
    out.insert(out.end(), heap.begin(), heap.end());
    while (out.size() < new_strip) out.push_back(0);
    out.insert(out.end(), file.begin() + strip_off, file.begin() + strip_off + strip_bc);
    while (out.size() < jpeg_off) out.push_back(0);
    out.insert(out.end(), jpeg, jpeg + jpeg_len);
    while (out.size() < ifd1_aligned) out.push_back(0);

    // Write IFD1 (all values inline — no heap for this small IFD)
    w16(out, (uint16_t)prev.e.size());
    for (const auto& e : prev.e) {
        w16(out, e.tag);
        w16(out, e.type);
        w32(out, e.count);
        w32(out, e.inlineval);
    }
    w32(out, 0);

    std::string tmp = path + ".preview.tmp";
    FILE* fo = fopen(tmp.c_str(), "wb");
    if (!fo) return false;
    if (fwrite(out.data(), 1, out.size(), fo) != out.size()) {
        fclose(fo);
        std::remove(tmp.c_str());
        return false;
    }
    fclose(fo);
#if defined(_WIN32)
    if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        std::remove(tmp.c_str());
        return false;
    }
#else
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(path.c_str());
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            std::remove(tmp.c_str());
            return false;
        }
    }
#endif
    return true;
}

bool load_linear_dng_rgb16(const std::string& path, std::vector<uint16_t>& rgb, int& W, int& H) {
    rgb.clear();
    W = H = 0;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long fsz = ftell(f);
    if (fsz < 16) { fclose(f); return false; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }

    std::vector<uint8_t> file((size_t)fsz);
    if (fread(file.data(), 1, file.size(), f) != file.size()) { fclose(f); return false; }
    fclose(f);

    if (file[0] != 'I' || file[1] != 'I' || r16(file.data() + 2) != 42) return false;
    uint32_t ifd = r32(file.data() + 4);
    if (ifd + 2 > file.size()) return false;
    uint16_t nent = r16(file.data() + ifd);
    if (ifd + 2u + (uint32_t)nent * 12u + 4u > file.size()) return false;

    uint32_t width = 0, height = 0, strip_off = 0, strip_bc = 0, rows_per_strip = 0;
    uint32_t tile_w = 0, tile_l = 0, tile_off_at = 0, tile_cnt_at = 0, ntiles = 0;
    uint16_t compression = 1, predictor = 1, spp = 0;
    for (uint16_t i = 0; i < nent; ++i) {
        const uint8_t* e = file.data() + ifd + 2 + i * 12;
        uint16_t tag = r16(e), type = r16(e + 2);
        uint32_t count = r32(e + 4), val = r32(e + 8);
        auto as_long = [&](uint32_t fallback) -> uint32_t {
            if (type == T_LONG && count == 1) return val;
            if (type == T_SHORT && count == 1) return val & 0xFFFF;
            return fallback;
        };
        switch (tag) {
            case 256: width = as_long(width); break;
            case 257: height = as_long(height); break;
            case 259: compression = (uint16_t)as_long(compression); break;
            case 273: strip_off = as_long(strip_off); break;
            case 277: spp = (uint16_t)as_long(spp); break;
            case 278: rows_per_strip = as_long(rows_per_strip); break;
            case 279: strip_bc = as_long(strip_bc); break;
            case 317: predictor = (uint16_t)as_long(predictor); break;
            case 322: tile_w = as_long(tile_w); break;
            case 323: tile_l = as_long(tile_l); break;
            // count 1 = inline value; else offset of the LONG array. The
            // entry's file offset of the value slot works for both reads.
            case 324: ntiles = count; tile_off_at = (count == 1) ? (uint32_t)(ifd + 2 + i * 12 + 8) : val; break;
            case 325: tile_cnt_at = (count == 1) ? (uint32_t)(ifd + 2 + i * 12 + 8) : val; break;
            default: break;
        }
    }
    if (width == 0 || height == 0 || spp != 3) return false;
    if (rows_per_strip == 0) rows_per_strip = height;
    if (compression == 7) {
        if (tile_w == 0 || tile_l == 0 || ntiles == 0 ||
            !tile_off_at || !tile_cnt_at)
            return false;
        const uint32_t ntx = (width + tile_w - 1) / tile_w;
        const uint32_t nty = (height + tile_l - 1) / tile_l;
        if (ntiles != ntx * nty) return false;
        if ((size_t)tile_off_at + 4u * ntiles > file.size() ||
            (size_t)tile_cnt_at + 4u * ntiles > file.size())
            return false;
        rgb.resize((size_t)width * height * 3);
        std::vector<uint16_t> tbuf((size_t)tile_w * tile_l * 3);
        for (uint32_t t = 0; t < ntiles; ++t) {
            const uint32_t off = r32(file.data() + tile_off_at + 4u * t);
            const uint32_t len = r32(file.data() + tile_cnt_at + 4u * t);
            if (!len || (size_t)off + len > file.size()) { rgb.clear(); return false; }
            if (!lj92_decode_strip(file.data() + off, len,
                                   (int)tile_w, (int)tile_l, 3, tbuf.data())) {
                rgb.clear();
                return false;
            }
            const uint32_t ty = t / ntx, tx = t % ntx;
            const uint32_t y0 = ty * tile_l, x0 = tx * tile_w;
            const uint32_t ny = std::min(tile_l, height - y0);
            const uint32_t nx = std::min(tile_w, width - x0);
            for (uint32_t y = 0; y < ny; ++y)
                std::memcpy(rgb.data() + ((size_t)(y0 + y) * width + x0) * 3u,
                            tbuf.data() + (size_t)y * tile_w * 3u,
                            (size_t)nx * 3u * sizeof(uint16_t));
        }
        W = (int)width;
        H = (int)height;
        return true;
    }
    if (strip_off == 0) return false;
    if (compression != 8 && compression != 1) return false;
    if (strip_off >= file.size()) return false;

    const size_t raw_bytes = (size_t)width * height * 3 * sizeof(uint16_t);
    rgb.resize((size_t)width * height * 3);

    if (compression == 1) {
        if (strip_off + raw_bytes > file.size()) { rgb.clear(); return false; }
        std::memcpy(rgb.data(), file.data() + strip_off, raw_bytes);
    } else {
        if (strip_bc == 0 || strip_off + strip_bc > file.size()) { rgb.clear(); return false; }
        z_stream zs{};
        if (inflateInit(&zs) != Z_OK) { rgb.clear(); return false; }
        zs.next_in = file.data() + strip_off;
        zs.avail_in = (uInt)strip_bc;
        zs.next_out = reinterpret_cast<Bytef*>(rgb.data());
        zs.avail_out = (uInt)raw_bytes;
        int ret = inflate(&zs, Z_FINISH);
        inflateEnd(&zs);
        if (ret != Z_STREAM_END || zs.total_out != raw_bytes) { rgb.clear(); return false; }
    }

    if (predictor == 2) {
        for (uint32_t y = 0; y < height; ++y)
            undo_hdiff_rgb16(rgb.data() + (size_t)y * width * 3, (int)width);
    }

    W = (int)width;
    H = (int)height;
    return true;
}

// Color-rendering tags only (private WB+matrix tag 65000, ColorMatrix1,
// AnalogBalance), parsed from an in-memory prefix of the file. Split out of
// load_linear_dng_rgb16_color so the render path can get wb/matrix WITHOUT
// inflating the pixel strips -- our writer lays every tag and small value
// block ahead of the pixel data, so a modest prefix always contains them.
static void parse_linear_dng_color_tags(const std::vector<uint8_t>& file,
                                        float wb[3], float cam_to_srgb[9],
                                        bool& has_color) {
    if (file.size() < 16 || file[0] != 'I' || file[1] != 'I') return;
    uint32_t ifd = r32(file.data() + 4);
    if (ifd + 2 > file.size()) return;
    uint16_t nent = r16(file.data() + ifd);
    bool private_color = false;
    bool has_color_matrix = false;
    float color_matrix[9] = {0};
    bool has_analog_balance = false;
    float analog_balance[3] = {1.f, 1.f, 1.f};
    for (uint16_t i = 0; i < nent; ++i) {
        const uint8_t* e = file.data() + ifd + 2 + i * 12;
        uint16_t tag = r16(e), type = r16(e + 2);
        uint32_t count = r32(e + 4), val = r32(e + 8);
        if (tag == 65000 && type == T_BYTE && count >= 48) {
            uint32_t off = (count <= 4) ? (uint32_t)(e + 8 - file.data()) : val;
            if (off + 48 <= file.size()) {
                auto read_f = [&](uint32_t o) -> float {
                    uint32_t u = r32(file.data() + o);
                    float v = 0.f;
                    std::memcpy(&v, &u, sizeof(v));
                    return v;
                };
                for (int k = 0; k < 3; ++k) wb[k] = read_f(off + (uint32_t)k * 4);
                for (int k = 0; k < 9; ++k) cam_to_srgb[k] = read_f(off + 12 + (uint32_t)k * 4);
                private_color = true;
                has_color = true;
            }
            continue;
        }
        if (tag == 50721 && (type == T_SRATIONAL || type == T_RATIONAL) && count >= 9) {
            const uint32_t bytes = count * type_size(type);
            uint32_t off = (bytes <= 4) ? (uint32_t)(e + 8 - file.data()) : val;
            if (off + 9u * 8u <= file.size()) {
                bool ok = true;
                for (int k = 0; k < 9; ++k) {
                    const uint8_t* p = file.data() + off + (uint32_t)k * 8u;
                    const int32_t num = (type == T_SRATIONAL)
                        ? (int32_t)r32(p) : (int32_t)(uint32_t)r32(p);
                    const int32_t den = (int32_t)r32(p + 4);
                    if (den == 0) { ok = false; break; }
                    color_matrix[k] = (float)num / (float)den;
                    ok = ok && std::isfinite(color_matrix[k]);
                }
                has_color_matrix = ok;
            }
        }
        if (tag == 50727 && (type == T_RATIONAL || type == T_SRATIONAL) && count >= 3) {
            const uint32_t bytes = count * type_size(type);
            uint32_t off = (bytes <= 4) ? (uint32_t)(e + 8 - file.data()) : val;
            if (off + 3u * 8u <= file.size()) {
                bool ok = true;
                for (int k = 0; k < 3; ++k) {
                    const uint8_t* p = file.data() + off + (uint32_t)k * 8u;
                    const int32_t num = (type == T_SRATIONAL)
                        ? (int32_t)r32(p) : (int32_t)(uint32_t)r32(p);
                    const int32_t den = (int32_t)r32(p + 4);
                    if (den == 0) { ok = false; break; }
                    analog_balance[k] = (float)num / (float)den;
                    ok = ok && std::isfinite(analog_balance[k]) &&
                         std::fabs(analog_balance[k]) > 1e-6f;
                }
                has_analog_balance = ok;
            }
        }
    }
    if ((!private_color || is_identity_3x3(cam_to_srgb)) && has_color_matrix) {
        float derived[9];
        if (derive_cam_to_srgb_from_color_matrix(
                color_matrix, has_analog_balance ? analog_balance : nullptr, derived)) {
            for (int k = 0; k < 9; ++k) cam_to_srgb[k] = derived[k];
            has_color = true;
        }
    }
}

static void reset_color_out(float wb[3], float cam_to_srgb[9], bool& has_color) {
    has_color = false;
    wb[0] = wb[1] = wb[2] = 1.f;
    for (int i = 0; i < 9; ++i) cam_to_srgb[i] = (i % 4 == 0) ? 1.f : 0.f;
}

bool load_linear_dng_color_meta(const std::string& path, float wb[3],
                                float cam_to_srgb[9], bool& has_color) {
    reset_color_out(wb, cam_to_srgb, has_color);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    // 4MB prefix: our writer keeps the IFD and every referenced value block
    // ahead of the pixel strips, orders of magnitude inside this. A tag whose
    // data lay beyond the prefix simply fails its bounds check and is skipped
    // -- the same silent fallback the full parse used for a truncated file.
    std::vector<uint8_t> file(4u << 20);
    const size_t got = fread(file.data(), 1, file.size(), f);
    fclose(f);
    file.resize(got);
    parse_linear_dng_color_tags(file, wb, cam_to_srgb, has_color);
    return got >= 16;
}

bool load_linear_dng_rgb16_color(const std::string& path, std::vector<uint16_t>& rgb,
                                 int& W, int& H, float wb[3], float cam_to_srgb[9],
                                 bool& has_color) {
    reset_color_out(wb, cam_to_srgb, has_color);
    if (!load_linear_dng_rgb16(path, rgb, W, H)) return false;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return true;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return true; }
    long fsz = ftell(f);
    if (fsz < 16) { fclose(f); return true; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return true; }
    std::vector<uint8_t> file((size_t)fsz);
    if (fread(file.data(), 1, file.size(), f) != file.size()) { fclose(f); return true; }
    fclose(f);

    parse_linear_dng_color_tags(file, wb, cam_to_srgb, has_color);
    return true;
}

} // namespace hhsr
