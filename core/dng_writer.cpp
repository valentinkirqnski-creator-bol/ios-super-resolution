#include "dng_writer.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <ctime>
#include <zlib.h>

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

} // namespace

// Builds DNG header. StripByteCounts left as 0 — patched after Deflate finishes.
// Returns prefix bytes; strip_offset_out / strip_byte_counts_offset_out are file offsets.
static std::vector<uint8_t> build_dng_prefix(int W, int H,
                                             const std::string& camera_make,
                                             const std::string& camera_model,
                                             int orientation,
                                             const float* /*cm*/,
                                             const float* /*wb*/,
                                             bool baked_srgb,
                                             uint32_t& strip_offset_out,
                                             uint32_t& strip_byte_counts_offset_out) {
    IFD ifd;
    ifd.longv(254, 0);                 // NewSubfileType
    ifd.ascii(271, camera_make.empty() ? "HandheldSR" : camera_make);
    ifd.ascii(272, camera_model.empty() ? "HandheldSR-x2" : camera_model);
    ifd.longv(256, (uint32_t)W);
    ifd.longv(257, (uint32_t)H);
    ifd.shorts(258, {16, 16, 16});
    ifd.shortv(259, 8);                // Compression = Adobe Deflate (lossless ZIP)
    if (baked_srgb)
        ifd.shortv(262, 2);            // RGB
    else
        ifd.shortv(262, 34892);        // LinearRaw
    ifd.longv(273, 0);                 // StripOffsets (patched)
    if (orientation >= 1 && orientation <= 8)
        ifd.shortv(274, (uint16_t)orientation);
    ifd.shortv(277, 3);                // SamplesPerPixel
    ifd.longv(278, (uint32_t)H);       // RowsPerStrip
    ifd.longv(279, 0);                 // StripByteCounts (patched after compress)
    ifd.shortv(284, 1);                // PlanarConfiguration = chunky
    ifd.ascii(305, "HandheldSR");      // Software
    ifd.ascii(306, now_tiff_datetime()); // DateTime
    ifd.shorts(339, {1, 1, 1});        // SampleFormat = unsigned

    // Photos / DNG readers need crop + active area for non-zero dimensions.
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
        // Identity ColorMatrix1 — WB already in pixels; real matrix + AsShotNeutral
        // re-develops color and casts magenta on iOS Photos / LR Mobile.
        ifd.srational(50721, {1,1, 0,1, 0,1,  0,1, 1,1, 0,1,  0,1, 0,1, 1,1});
        // Scene-referred linear data (helps Photos treat LinearRaw correctly).
        ifd.shortv(50831, 1);          // ColorimetricReference = scene referred
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
    int strip_bc_entry = -1;
    for (int i = 0; i < (int)ifd.e.size(); ++i) {
        auto& e = ifd.e[(size_t)i];
        if (e.tag == 273) strip_off_entry = i;
        if (e.tag == 279) strip_bc_entry = i;
        if (!e.payload.empty()) {
            if (heap.size() & 1) heap.push_back(0);
            e.inlineval = heap_base + (uint32_t)heap.size();
            heap.insert(heap.end(), e.payload.begin(), e.payload.end());
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
            // Value field starts 8 bytes into the 12-byte IFD entry.
            strip_byte_counts_offset_out = (uint32_t)out.size() + 8;
        }
        w16(out, e.tag);
        w16(out, e.type);
        w32(out, e.count);
        w32(out, e.inlineval);
    }
    w32(out, 0);
    out.insert(out.end(), heap.begin(), heap.end());
    while (out.size() < strip_offset) out.push_back(0);

    strip_offset_out = strip_offset;
    (void)strip_bc_entry;
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
                           const std::string& camera_make) {
    if (W <= 0 || H <= 0) return false;
    W_ = W; H_ = H; rows_written_ = 0;
    compressed_bytes_ = 0;
    strip_byte_counts_offset_ = 0;
    deflate_ok_ = false;

    uint32_t strip_offset = 0;
    std::vector<uint8_t> prefix = build_dng_prefix(W, H, camera_make, camera_model, orientation,
                                                   colorMatrixXYZtoCam, wbGainsGreenNorm,
                                                   bakedSrgb, strip_offset,
                                                   strip_byte_counts_offset_);
    f_ = fopen(path.c_str(), "wb+");
    if (!f_) return false;
    if (fwrite(prefix.data(), 1, prefix.size(), f_) != prefix.size()) {
        fclose(f_); f_ = nullptr;
        return false;
    }

    auto* zs = new z_stream();
    std::memset(zs, 0, sizeof(z_stream));
    if (deflateInit(zs, Z_DEFAULT_COMPRESSION) != Z_OK) {
        delete zs;
        fclose(f_); f_ = nullptr;
        return false;
    }
    z_stream_ = zs;
    z_out_.resize(256 * 1024);
    deflate_ok_ = true;
    return true;
}

bool DngStreamWriter::write_rows(const uint16_t* rgb16, int nrows) {
    if (!f_ || !deflate_ok_ || !z_stream_ || nrows <= 0) return false;
    if (rows_written_ + nrows > H_) nrows = H_ - (int)rows_written_;
    if (nrows <= 0) return true;

    auto* zs = static_cast<z_stream*>(z_stream_);
    zs->next_in = reinterpret_cast<Bytef*>(const_cast<uint16_t*>(rgb16));
    zs->avail_in = (uInt)((size_t)nrows * W_ * 3 * sizeof(uint16_t));

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
    rows_written_ += nrows;
    return true;
}

bool DngStreamWriter::close() {
    if (!f_) return false;
    bool ok = rows_written_ == H_ && deflate_ok_ && z_stream_;

    if (ok) {
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

        if (ok && strip_byte_counts_offset_ > 0) {
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

} // namespace hhsr
