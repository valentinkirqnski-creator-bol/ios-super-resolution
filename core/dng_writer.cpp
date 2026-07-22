#include "dng_writer.h"
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

} // namespace

// Builds DNG header. StripByteCounts left as 0 — patched after Deflate finishes.
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
    ifd.shortv(317, 1);                // Predictor = none (faster write; same pixels)
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

    if (wb || cam_to_srgb) {
        std::vector<uint8_t> blob;
        blob.reserve(48);
        // JPEG/preview must not apply WB again when pixels are already pre-whitened.
        for (int i = 0; i < 3; ++i)
            append_f32_le(blob, (wb && !pixels_prewhitened) ? wb[i] : 1.f);
        for (int i = 0; i < 9; ++i) {
            float v = 0.f;
            if (cam_to_srgb) v = cam_to_srgb[i];
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
                           bool pixelsPrewhitened) {
    if (W <= 0 || H <= 0) return false;
    W_ = W; H_ = H; rows_written_ = 0;
    compressed_bytes_ = 0;
    strip_byte_counts_offset_ = 0;
    deflate_ok_ = false;

    uint32_t strip_offset = 0;
    std::vector<uint8_t> prefix = build_dng_prefix(W, H, camera_make, camera_model, orientation,
                                                   colorMatrixXYZtoCam, wbGainsGreenNorm,
                                                   bakedSrgb, camToSrgb, pixelsPrewhitened,
                                                   strip_offset, strip_byte_counts_offset_);
    f_ = fopen(path.c_str(), "wb+");
    if (!f_) return false;
    // Large stdio buffer — fewer syscalls during streaming Deflate.
    setvbuf(f_, nullptr, _IOFBF, 1u << 20);
    if (fwrite(prefix.data(), 1, prefix.size(), f_) != prefix.size()) {
        fclose(f_); f_ = nullptr;
        return false;
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
    if (!f_ || !deflate_ok_ || !z_stream_ || !rgb16 || nrows <= 0) return false;
    if (rows_written_ + nrows > H_) nrows = H_ - (int)rows_written_;
    if (nrows <= 0) return true;

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

bool embed_dng_jpeg_preview(const std::string& path,
                            const uint8_t* jpeg, size_t jpeg_len,
                            int jpeg_w, int jpeg_h) {
    if (!jpeg || jpeg_len < 4 || jpeg_w <= 0 || jpeg_h <= 0) return false;
    // SOI marker
    if (jpeg[0] != 0xFF || jpeg[1] != 0xD8) return false;

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

    // Already has SubIFDs — leave alone (idempotent).
    for (uint16_t i = 0; i < nent; ++i) {
        const uint8_t* e = file.data() + ifd0 + 2 + i * 12;
        if (r16(e) == 330) return true;
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
            default: break;
        }
    }
    if (width == 0 || height == 0 || spp != 3 || strip_off == 0) return false;
    if (rows_per_strip == 0) rows_per_strip = height;
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

bool load_linear_dng_rgb16_color(const std::string& path, std::vector<uint16_t>& rgb,
                                 int& W, int& H, float wb[3], float cam_to_srgb[9],
                                 bool& has_color) {
    has_color = false;
    wb[0] = wb[1] = wb[2] = 1.f;
    for (int i = 0; i < 9; ++i) cam_to_srgb[i] = (i % 4 == 0) ? 1.f : 0.f;

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

    if (file[0] != 'I' || file[1] != 'I') return true;
    uint32_t ifd = r32(file.data() + 4);
    if (ifd + 2 > file.size()) return true;
    uint16_t nent = r16(file.data() + ifd);
    for (uint16_t i = 0; i < nent; ++i) {
        const uint8_t* e = file.data() + ifd + 2 + i * 12;
        uint16_t tag = r16(e), type = r16(e + 2);
        uint32_t count = r32(e + 4), val = r32(e + 8);
        if (tag != 65000 || type != T_BYTE || count < 48) continue;
        uint32_t off = (count <= 4) ? (uint32_t)(e + 8 - file.data()) : val;
        if (off + 48 > file.size()) continue;
        auto read_f = [&](uint32_t o) -> float {
            uint32_t u = r32(file.data() + o);
            float v = 0.f;
            std::memcpy(&v, &u, sizeof(v));
            return v;
        };
        for (int k = 0; k < 3; ++k) wb[k] = read_f(off + (uint32_t)k * 4);
        for (int k = 0; k < 9; ++k) cam_to_srgb[k] = read_f(off + 12 + (uint32_t)k * 4);
        has_color = true;
        break;
    }
    return true;
}

} // namespace hhsr
