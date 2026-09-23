#include "dng_writer.h"
#include "ljpeg.h"
#include "parallel.h"
#include <cstdio>
#include <thread>
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

// TIFF Predictor=2, undo side only. The writer never emits Predictor=2 -- the
// encode half sat here unused, since the Deflate path always wrote Predictor=1
// and lossless JPEG does its own prediction -- but files from other tools can
// carry it, so the reader still needs this.
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

// Why the default codec is lossless JPEG (Compression=7) and not Deflate.
//
// Measured on a real 48MP merge (8064x6048x3, 292.6MB of samples):
//   uncompressed      292.6MB
//   Deflate, zlib-1   294.6MB   <- LARGER than raw, for ~8.6s of serial CPU
//   Deflate, zlib-6   256.8MB
//   lossless JPEG     172.6MB   (-41.0%)
//
// zlib cannot model 16-bit photographic data: the low bits are sensor noise
// (the odd/even code split measures 0.50, i.e. the LSB is a coin flip), and a
// 32KB LZ77 window finds no matches in it. Lossless JPEG predicts each sample
// from its left neighbour and entropy-codes the residual, which is exactly the
// structure this data has. A zlib stream is also inherently serial, whereas
// each JPEG strip here is a self-contained SOI..EOI bitstream, so encoding and
// decoding both run on every core.
//
// load_linear_dng_rgb16 reads all three (Compression = 1, 7 and 8).

// Serialize a standalone IFD (count + 12-byte entries + next-pointer=0 + heap
// for out-of-line payloads) as it will sit at `base_offset` in the final file.
// Used for the Exif sub-IFD: same shape as IFD0's own tail below, factored out
// so a nested directory can be built without duplicating the offset arithmetic.
static std::vector<uint8_t> serialize_ifd_at(IFD& ifd, uint32_t base_offset) {
    std::sort(ifd.e.begin(), ifd.e.end(), [](const Entry& a, const Entry& b) {
        return a.tag < b.tag;
    });
    const uint32_t n = (uint32_t)ifd.e.size();
    const uint32_t dir_size = 2 + n * 12 + 4;
    std::vector<uint8_t> heap;
    for (auto& en : ifd.e) {
        if (en.payload.empty()) continue;
        if (heap.size() & 1) heap.push_back(0);
        en.inlineval = base_offset + dir_size + (uint32_t)heap.size();
        heap.insert(heap.end(), en.payload.begin(), en.payload.end());
    }
    std::vector<uint8_t> out;
    w16(out, (uint16_t)n);
    for (auto& en : ifd.e) {
        w16(out, en.tag);
        w16(out, en.type);
        w32(out, en.count);
        w32(out, en.inlineval);
    }
    w32(out, 0);  // next IFD
    out.insert(out.end(), heap.begin(), heap.end());
    return out;
}

// EXIF RATIONAL from a float (exposure/aperture/focal length): denominator by
// magnitude so short shutter speeds (1/8000) and long ones keep precision.
static void push_rational_from_float(std::vector<uint32_t>& nd, float v) {
    if (!(v > 0.f) || !std::isfinite(v)) { nd.push_back(0); nd.push_back(1); return; }
    uint32_t den = (v < 1.f) ? 1000000u : 1000u;
    nd.push_back((uint32_t)std::lround((double)v * den));
    nd.push_back(den);
}

// Builds DNG header. StripOffsets and StripByteCounts are reserved as
// `nstrips`-long LONG arrays of zeros and patched once each strip is on disk;
// the two *_pos_out values are the file offsets of those arrays (for nstrips==1
// the single LONG lives inline in the IFD entry, which is exactly where the
// patch has always gone).
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
                                             int codec,
                                             int nstrips,
                                             int rows_per_strip,
                                             uint32_t& strip_offset_out,
                                             uint32_t& strip_offsets_pos_out,
                                             uint32_t& strip_byte_counts_pos_out,
                                             const CaptureExif* exif = nullptr) {
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
    // OUR identity, never the source camera's -- see UniqueCameraModel below.
    ifd.ascii(271, "HandheldSR");
    ifd.ascii(272, "HandheldSR-x2");
    ifd.longv(256, (uint32_t)W);
    ifd.longv(257, (uint32_t)H);
    ifd.shorts(258, {16, 16, 16});
    // 7 = lossless JPEG (SOF3), 8 = Adobe Deflate (ZIP), 1 = uncompressed.
    // All three decode to the same uint16 samples.
    ifd.shortv(259, codec == Config::DNG_CODEC_LJPEG    ? 7 :
                    codec == Config::DNG_CODEC_DEFLATE  ? 8 : 1);
    if (baked_srgb)
        ifd.shortv(262, 2);            // RGB
    else
        ifd.shortv(262, 34892);        // LinearRaw
    ifd.longs(273, std::vector<uint32_t>((size_t)nstrips, 0));  // StripOffsets (patched)
    // SubIFDs, reserved empty. Adding this tag later would grow IFD0 by 12
    // bytes and push the image strip along with it, which is why embedding a
    // preview used to rebuild the entire file -- a 292MB read plus a 292MB
    // write plus both buffers resident, just to attach a thumbnail. Reserved
    // here, embedding becomes an append and a four-byte patch.
    ifd.longv(330, 0);                 // SubIFDs (patched by embed_dng_jpeg_preview)
    if (orientation >= 1 && orientation <= 8)
        ifd.shortv(274, (uint16_t)orientation);
    ifd.shortv(277, 3);                // SamplesPerPixel
    ifd.longv(278, (uint32_t)rows_per_strip);
    ifd.longs(279, std::vector<uint32_t>((size_t)nstrips, 0));  // StripByteCounts (patched)
    ifd.shortv(284, 1);                // PlanarConfiguration = chunky
    // Provenance lives here. Software is descriptive and is never used to
    // match a camera profile, so naming the source camera in it is safe.
    ifd.ascii(305, camera_model.empty()
                       ? std::string("HandheldSR")
                       : ("HandheldSR (from " + camera_make +
                          (camera_make.empty() ? "" : " ") + camera_model + ")"));
    ifd.ascii(306, now_tiff_datetime()); // DateTime (file write time)
    // ExifIFD reserved here so it grows IFD0 by exactly one entry regardless of
    // which fields exif carries; the offset is patched once the Exif sub-IFD's
    // own bytes are laid into the heap, below.
    const bool has_exif = exif && (exif->iso > 0.f || exif->exposure_seconds > 0.f ||
                                   exif->f_number > 0.f || exif->focal_length_mm > 0.f ||
                                   !exif->lens_model.empty() || !exif->datetime.empty());
    if (has_exif) ifd.longv(34665, 0);  // ExifIFD (patched below)
    // Predictor is defined for LZW/Deflate only; lossless JPEG carries its own
    // prediction inside the scan, so the tag is omitted rather than written as
    // a meaningless 1.
    if (codec != Config::DNG_CODEC_LJPEG)
        ifd.shortv(317, 1);            // Predictor = none (faster write; same pixels)
    ifd.shorts(339, {1, 1, 1});        // SampleFormat = unsigned

    ifd.longs(50719, {0, 0});
    ifd.longs(50720, {(uint32_t)W, (uint32_t)H});
    ifd.longs(50829, {0, 0, (uint32_t)H, (uint32_t)W});

    ifd.bytes4(50706, 1, 4, 0, 0);     // DNGVersion 1.4.0.0
    ifd.bytes4(50707, 1, 3, 0, 0);     // DNGBackwardVersion 1.3.0.0
    // UniqueCameraModel is the tag a RAW pipeline keys its BUILT-IN camera
    // profile on, with Make/Model matched alongside it. This used to copy the
    // source camera's string, so the file announced itself as an iPhone raw
    // while actually being 3-channel LinearRaw at up to twice the sensor's
    // resolution, with BlackLevel 0 / WhiteLevel 65535 and none of the lens
    // opcodes an iPhone raw carries.
    //
    // Announcing an identity we do not have also makes a reader PREFER its own
    // profile over the ColorMatrix1 and AsShotNeutral in the file, which is
    // backwards for a synthetic DNG: those tags are the only correct
    // description of this data. A name no vendor profile matches forces the
    // reader to use them.
    //
    // On its own this did NOT fix the black render in Photos -- 10fae28 was
    // written with this already in the tree -- so it is correctness, not the
    // cure. Restored because 52f3822 reverted it with the rest of the tree.
    ifd.ascii(50708, "HandheldSR-x2");
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
        // No tag here. 50831 is AsShotICCProfile, not ColorimetricReference
        // (that is 50879) -- this wrote ColorimetricReference's value into
        // AsShotICCProfile's number, so every DNG we have written announced an
        // embedded ICC profile that was one SHORT long.
        //
        // AsShotICCProfile, where a reader honours it, REPLACES ColorMatrix1
        // and AsShotNeutral as the camera-space-to-PCS transform. Adobe-derived
        // readers ignore the tag, which is why Lightroom and LibRaw render
        // these files; a ColorSync-based pipeline does not, and two bytes
        // cannot build a transform. That is exactly the split in the symptom:
        // Photos shows the JPEG SubIFD first, because a preview needs no camera
        // profile, then the RAW render lands and is black -- at any resolution
        // and under any of the three codecs, which is what ruled out size,
        // compression and UniqueCameraModel in turn.
        //
        // Nothing replaces it: ColorimetricReference defaults to 0 = scene
        // referred, which is what this data is and what the line meant to say.
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
    strip_offsets_pos_out = 0;
    strip_byte_counts_pos_out = 0;
    for (int i = 0; i < (int)ifd.e.size(); ++i) {
        auto& e = ifd.e[(size_t)i];
        if (e.tag == 273) strip_off_entry = i;
        if (!e.payload.empty()) {
            if (heap.size() & 1) heap.push_back(0);
            e.inlineval = heap_base + (uint32_t)heap.size();
            // Multi-strip: the two LONG arrays live out of line, and their heap
            // address is where close() writes the real offsets and sizes.
            if (e.tag == 273) strip_offsets_pos_out = e.inlineval;
            if (e.tag == 279) strip_byte_counts_pos_out = e.inlineval;
            heap.insert(heap.end(), e.payload.begin(), e.payload.end());
        }
    }
    if (has_exif) {
        IFD exif_ifd;
        if (exif->iso > 0.f)
            exif_ifd.shorts(34855, {(uint16_t)clampf(exif->iso, 0.f, 65535.f)});
        if (exif->exposure_seconds > 0.f) {
            std::vector<uint32_t> nd; push_rational_from_float(nd, exif->exposure_seconds);
            exif_ifd.rational(33434, nd);   // ExposureTime
        }
        if (exif->f_number > 0.f) {
            std::vector<uint32_t> nd; push_rational_from_float(nd, exif->f_number);
            exif_ifd.rational(33437, nd);   // FNumber
        }
        if (exif->focal_length_mm > 0.f) {
            std::vector<uint32_t> nd; push_rational_from_float(nd, exif->focal_length_mm);
            exif_ifd.rational(37386, nd);   // FocalLength
        }
        if (!exif->lens_model.empty()) exif_ifd.ascii(42036, exif->lens_model);   // LensModel
        exif_ifd.ascii(36867, exif->datetime.empty() ? now_tiff_datetime()
                                                     : exif->datetime);           // DateTimeOriginal
        if (heap.size() & 1) heap.push_back(0);
        const uint32_t exif_offset = heap_base + (uint32_t)heap.size();
        std::vector<uint8_t> exif_bytes = serialize_ifd_at(exif_ifd, exif_offset);
        heap.insert(heap.end(), exif_bytes.begin(), exif_bytes.end());
        for (auto& e : ifd.e)
            if (e.tag == 34665) e.inlineval = exif_offset;
    }
    uint32_t strip_offset = heap_base + (uint32_t)heap.size();
    if (strip_offset & 1) strip_offset += 1;
    // Single strip: the offset is the entry's own inline value, as before.
    if (nstrips == 1 && strip_off_entry >= 0)
        ifd.e[(size_t)strip_off_entry].inlineval = strip_offset;

    std::vector<uint8_t> out;
    out.push_back('I'); out.push_back('I');
    w16(out, 42);
    w32(out, ifd_offset);
    w16(out, (uint16_t)n);
    for (int i = 0; i < (int)ifd.e.size(); ++i) {
        const auto& e = ifd.e[(size_t)i];
        // Single strip: both values sit inline in the IFD entry, so that is
        // where the patch goes.
        if (nstrips == 1 && e.tag == 273) strip_offsets_pos_out = (uint32_t)out.size() + 8;
        if (nstrips == 1 && e.tag == 279) strip_byte_counts_pos_out = (uint32_t)out.size() + 8;
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
                           bool pixelsPrewhitened, int codec,
                           const CaptureExif* exif, int numThreads) {
    if (W <= 0 || H <= 0) return false;
    join_async();   // a reused writer must not leave a worker behind
    W_ = W; H_ = H; rows_written_ = 0;
    compressed_bytes_ = 0;
    codec_ = (codec == Config::DNG_CODEC_LJPEG || codec == Config::DNG_CODEC_DEFLATE)
                 ? codec : Config::DNG_CODEC_NONE;
    num_threads_ = numThreads;
    strip_byte_counts_pos_ = 0;
    strip_offsets_pos_ = 0;
    deflate_ok_ = false;
    strip_offsets_.clear();
    strip_sizes_.clear();
    pending_.clear();
    enc_scratch_.clear();
    pending_rows_ = 0;
    encoded_rows_ = 0;
    async_rows_ = 0;
    async_ok_ = true;

    const bool ljpeg = codec_ == Config::DNG_CODEC_LJPEG;
    const int rows_per_strip = ljpeg ? std::min(kLjpegStripRows, H) : H;
    const int nstrips = ljpeg ? ((H + rows_per_strip - 1) / rows_per_strip) : 1;

    uint32_t strip_offset = 0;
    std::vector<uint8_t> prefix = build_dng_prefix(W, H, camera_make, camera_model, orientation,
                                                   colorMatrixXYZtoCam, wbGainsGreenNorm,
                                                   bakedSrgb, camToSrgb, pixelsPrewhitened,
                                                   codec_, nstrips, rows_per_strip,
                                                   strip_offset, strip_offsets_pos_,
                                                   strip_byte_counts_pos_,
                                                   exif);
    if (ljpeg) {
        strip_offsets_.reserve((size_t)nstrips);
        strip_sizes_.reserve((size_t)nstrips);
        pending_.resize((size_t)rows_per_strip * (size_t)W * 3u);
    }
    next_strip_offset_ = strip_offset;
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

    if (codec_ != Config::DNG_CODEC_DEFLATE) {
        // Uncompressed and lossless JPEG both write strips straight out; no
        // zlib state to carry at all.
        deflate_ok_ = true;
        return true;
    }

    auto* zs = new z_stream();
    std::memset(zs, 0, sizeof(z_stream));
    // Level 6, not Z_BEST_SPEED. Measured on a 48MP merge, level 1 emits
    // 294.6MB against 292.6MB uncompressed -- it cannot model 16-bit sensor
    // noise, so it spends ~8.6s of unparallelizable CPU to make the file
    // *larger*. Level 6 reaches 256.8MB. Lossless JPEG beats both at 172.6MB
    // and is the default; this path is kept for comparison.
    if (deflateInit(zs, 6) != Z_OK) {
        delete zs;
        fclose(f_); f_ = nullptr;
        return false;
    }
    z_stream_ = zs;
    z_out_.resize(1u << 20);
    deflate_ok_ = true;
    return true;
}

// Encode one complete strip and append it. Callers hand over whole strips only
// (the last one may be short); the offset/size pair is recorded for close().
bool DngStreamWriter::flush_ljpeg_strip(const uint16_t* rows16, int nrows) {
    std::vector<uint8_t> enc;
    if (!ljpeg_encode(rows16, W_, nrows, 3, enc)) return false;
    if (fwrite(enc.data(), 1, enc.size(), f_) != enc.size()) return false;
    strip_offsets_.push_back(next_strip_offset_);
    strip_sizes_.push_back((uint32_t)enc.size());
    next_strip_offset_ += (uint32_t)enc.size();
    compressed_bytes_ += (uint32_t)enc.size();
    return true;
}

// Wait for the worker started by the previous write_rows, if any, and take its
// result. Returns false once anything in the encode has failed.
bool DngStreamWriter::join_async() {
    if (async_thread_) {
        auto* t = static_cast<std::thread*>(async_thread_);
        if (t->joinable()) t->join();
        delete t;
        async_thread_ = nullptr;
    }
    return async_ok_;
}

// Runs on the worker thread. Carries any part-strip from the previous band,
// encodes every whole strip in parallel, then appends them in row order.
bool DngStreamWriter::encode_band_ljpeg(const uint16_t* rgb16, int nrows) {
    {
        const size_t row_samples = (size_t)W_ * 3u;
        const int strip_rows = std::min(kLjpegStripRows, H_);
        int consumed = 0;

        // Top up a strip a previous band ended part-way through. Merge bands
        // are not multiples of the strip height, so this carry is the only copy
        // on the path -- at most 63 rows per band.
        if (pending_rows_ > 0) {
            const int need = std::min(strip_rows - pending_rows_, nrows);
            std::memcpy(pending_.data() + (size_t)pending_rows_ * row_samples,
                        rgb16, (size_t)need * row_samples * sizeof(uint16_t));
            pending_rows_ += need;
            consumed = need;
            // A short final strip is left to the end-of-image flush below.
            if (pending_rows_ == strip_rows) {
                if (!flush_ljpeg_strip(pending_.data(), pending_rows_)) return false;
                pending_rows_ = 0;
            }
        }

        // Whole strips encode straight out of the band buffer, in parallel:
        // each is a self-contained bitstream, so there is no cross-strip state
        // the way a single Deflate stream would have.
        const int rest = nrows - consumed;
        const int nfull = rest / strip_rows;
        if (nfull > 0) {
            // One scratch buffer per strip slot, kept across bands: after the
            // first band these are already the right size, so the encode does
            // no allocation at all.
            if (enc_scratch_.size() < (size_t)nfull) enc_scratch_.resize((size_t)nfull);
            std::vector<size_t> len((size_t)nfull, 0);
            std::vector<char> ok((size_t)nfull, 0);
            const uint16_t* base = rgb16 + (size_t)consumed * row_samples;
            parallel_rows(nfull, num_threads_, [&](int k) {
                ok[(size_t)k] = ljpeg_encode(base + (size_t)k * (size_t)strip_rows * row_samples,
                                             W_, strip_rows, 3,
                                             enc_scratch_[(size_t)k], len[(size_t)k]) ? 1 : 0;
            });
            for (int k = 0; k < nfull; ++k) {
                if (!ok[(size_t)k]) return false;
                const size_t n = len[(size_t)k];
                if (fwrite(enc_scratch_[(size_t)k].data(), 1, n, f_) != n) return false;
                strip_offsets_.push_back(next_strip_offset_);
                strip_sizes_.push_back((uint32_t)n);
                next_strip_offset_ += (uint32_t)n;
                compressed_bytes_ += (uint32_t)n;
            }
            consumed += nfull * strip_rows;
        }

        // Carry the tail. When it is the tail of the image it becomes the short
        // final strip.
        const int tail = nrows - consumed;
        if (tail > 0) {
            std::memcpy(pending_.data() + (size_t)pending_rows_ * row_samples,
                        rgb16 + (size_t)consumed * row_samples,
                        (size_t)tail * row_samples * sizeof(uint16_t));
            pending_rows_ += tail;
        }
        encoded_rows_ += nrows;
        if (pending_rows_ > 0 && encoded_rows_ == H_) {
            if (!flush_ljpeg_strip(pending_.data(), pending_rows_)) return false;
            pending_rows_ = 0;
        }
        return true;
    }
}

bool DngStreamWriter::write_rows(const uint16_t* rgb16, int nrows) {
    if (!f_ || !deflate_ok_ || !rgb16 || nrows <= 0) return false;
    if (rows_written_ + nrows > H_) nrows = H_ - (int)rows_written_;
    if (nrows <= 0) return true;

    if (codec_ == Config::DNG_CODEC_LJPEG) {
        // Hand the band to the worker and return: the caller gets on with
        // producing the next one while this encodes.
        if (!join_async()) return false;
        const size_t n = (size_t)nrows * (size_t)W_ * 3u;
        if (async_buf_.size() < n) async_buf_.resize(n);
        std::memcpy(async_buf_.data(), rgb16, n * sizeof(uint16_t));
        async_rows_ = nrows;
        rows_written_ += nrows;
        async_thread_ = new std::thread([this]() {
            if (!encode_band_ljpeg(async_buf_.data(), async_rows_)) async_ok_ = false;
        });
        return true;
    }

    if (codec_ != Config::DNG_CODEC_DEFLATE) {
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

bool DngStreamWriter::close() {
    if (!f_) return false;
    const bool use_ljpeg = codec_ == Config::DNG_CODEC_LJPEG;
    const bool use_deflate = codec_ == Config::DNG_CODEC_DEFLATE;
    // The last band may still be encoding; everything below needs it finished.
    const bool async_ok = join_async();
    bool ok = rows_written_ == H_ && deflate_ok_ && async_ok &&
              (!use_deflate || z_stream_);

    if (ok && use_ljpeg) {
        // Any rows still carried (a merge band that ended mid-strip, with the
        // image ending there too) go out as the short final strip.
        if (pending_rows_ > 0) {
            ok = flush_ljpeg_strip(pending_.data(), pending_rows_);
            pending_rows_ = 0;
        }
        const int strip_rows = std::min(kLjpegStripRows, H_);
        const size_t want = (size_t)((H_ + strip_rows - 1) / strip_rows);
        if (ok && strip_offsets_.size() != want) ok = false;   // strip count must match RowsPerStrip

        // Patch both LONG arrays now that every strip's place is known.
        if (ok && strip_offsets_pos_ > 0 && strip_byte_counts_pos_ > 0) {
            std::vector<uint8_t> buf;
            buf.reserve(strip_offsets_.size() * 4u);
            for (uint32_t v : strip_offsets_) w32(buf, v);
            if (fseek(f_, (long)strip_offsets_pos_, SEEK_SET) != 0 ||
                fwrite(buf.data(), 1, buf.size(), f_) != buf.size()) ok = false;
            buf.clear();
            for (uint32_t v : strip_sizes_) w32(buf, v);
            if (ok && (fseek(f_, (long)strip_byte_counts_pos_, SEEK_SET) != 0 ||
                       fwrite(buf.data(), 1, buf.size(), f_) != buf.size())) ok = false;
        } else if (ok) {
            ok = false;
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

    if (ok && use_deflate) {
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

    // Patch StripByteCounts for the two single-strip paths: Deflate accumulates
    // the compressed size above, uncompressed accumulates the raw size in
    // write_rows. StripOffsets is already correct inline, so only the one LONG
    // needs writing here.
    if (ok && strip_byte_counts_pos_ > 0) {
        if (fseek(f_, (long)strip_byte_counts_pos_, SEEK_SET) == 0) {
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
    join_async();   // never outlive a worker still reading async_buf_ / f_
    if (z_stream_) {
        deflateEnd(static_cast<z_stream*>(z_stream_));
        delete static_cast<z_stream*>(z_stream_);
        z_stream_ = nullptr;
    }
    if (f_) fclose(f_);
}

// Build the preview SubIFD, laid out for the file offset it will occupy.
//
// BitsPerSample {8,8,8} is three SHORTs = 6 bytes, so it does NOT fit in an
// entry's 4-byte value field and needs a heap slot. Serializing the entries
// alone left its offset at 0, and readers dutifully fetched BitsPerSample from
// the start of the file -- decoding the TIFF header as {18761, 42, 8}, i.e.
// 'II' and the magic 42. serialize_ifd_at places out-of-line payloads properly,
// which is why this goes through it rather than emitting the entries by hand.
static std::vector<uint8_t> build_preview_ifd(uint32_t ifd_off, uint32_t jpeg_off,
                                              size_t jpeg_len, int jpeg_w, int jpeg_h) {
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
    prev.shorts(530, {2, 2});         // YCbCrSubSampling (two SHORTs: fits inline)
    prev.shortv(531, 1);              // YCbCrPositioning = centered
    return serialize_ifd_at(prev, ifd_off);
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
        build_preview_ifd(ifd1_off, jpeg_off, jpeg_len, jpeg_w, jpeg_h);
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

    const uint32_t ifd1_offset = jpeg_off + (uint32_t)jpeg_len;
    const uint32_t ifd1_aligned = (ifd1_offset + 1u) & ~1u;
    // IFD1 (JPEG preview) after the JPEG payload, built the same way as the
    // append path so BitsPerSample gets a real heap slot rather than offset 0.
    const std::vector<uint8_t> prev_bytes =
        build_preview_ifd(ifd1_aligned, jpeg_off, jpeg_len, jpeg_w, jpeg_h);
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
    out.insert(out.end(), prev_bytes.begin(), prev_bytes.end());

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
    std::vector<uint32_t> strip_offs, strip_bcs;
    // Pull a LONG/SHORT array out of an entry, inline or from the heap. A
    // lossless-JPEG DNG carries one entry per strip, so StripOffsets and
    // StripByteCounts are no longer single values.
    auto read_array = [&](uint16_t type, uint32_t count, uint32_t val,
                          std::vector<uint32_t>& dst) {
        dst.clear();
        const uint32_t esz = (type == T_LONG) ? 4u : (type == T_SHORT) ? 2u : 0u;
        if (esz == 0 || count == 0) return;
        if (esz * count <= 4u) {
            for (uint32_t k = 0; k < count; ++k)
                dst.push_back(esz == 4 ? val : ((val >> (16 * k)) & 0xFFFFu));
            return;
        }
        if ((size_t)val + (size_t)esz * count > file.size()) return;
        const uint8_t* p = file.data() + val;
        for (uint32_t k = 0; k < count; ++k)
            dst.push_back(esz == 4 ? r32(p + 4 * k) : (uint32_t)r16(p + 2 * k));
    };
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
            case 273: read_array(type, count, val, strip_offs); break;
            case 277: spp = (uint16_t)as_long(spp); break;
            case 278: rows_per_strip = as_long(rows_per_strip); break;
            case 279: read_array(type, count, val, strip_bcs); break;
            case 317: predictor = (uint16_t)as_long(predictor); break;
            default: break;
        }
    }
    if (width == 0 || height == 0 || spp != 3) return false;
    if (strip_offs.empty()) return false;
    if (rows_per_strip == 0) rows_per_strip = height;
    if (compression != 8 && compression != 1 && compression != 7) return false;
    strip_off = strip_offs[0];
    strip_bc = strip_bcs.empty() ? 0 : strip_bcs[0];
    if (strip_off >= file.size()) return false;

    const size_t raw_bytes = (size_t)width * height * 3 * sizeof(uint16_t);
    rgb.resize((size_t)width * height * 3);

    if (compression == 7) {
        // Lossless JPEG: every strip is an independent stream, so they decode
        // on all cores the same way they were written.
        const uint32_t nstrips = (height + rows_per_strip - 1) / rows_per_strip;
        if (strip_offs.size() != nstrips || strip_bcs.size() != nstrips) { rgb.clear(); return false; }
        std::vector<char> ok(nstrips, 0);
        parallel_rows((int)nstrips, 0, [&](int k) {
            const uint32_t y0 = (uint32_t)k * rows_per_strip;
            const uint32_t rows = std::min(rows_per_strip, height - y0);
            const size_t off = strip_offs[(size_t)k], bc = strip_bcs[(size_t)k];
            if (off + bc > file.size() || bc == 0) return;
            ok[(size_t)k] = ljpeg_decode(file.data() + off, bc,
                                         rgb.data() + (size_t)y0 * width * 3,
                                         (int)width, (int)rows, 3) ? 1 : 0;
        });
        for (uint32_t k = 0; k < nstrips; ++k)
            if (!ok[k]) { rgb.clear(); return false; }
    } else if (compression == 1) {
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

bool load_linear_dng_rgb16_info(const std::string& path, std::vector<uint16_t>& rgb,
                                int& W, int& H, LinearDngColorInfo& info) {
    info = LinearDngColorInfo{};

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
        if (tag == 274 && type == T_SHORT && count >= 1) {
            const uint16_t o = r16(e + 8);
            if (o >= 1 && o <= 8) info.orientation = (int)o;
            continue;
        }
        if (tag == 65000 && type == T_BYTE && count >= 48) {
            uint32_t off = (count <= 4) ? (uint32_t)(e + 8 - file.data()) : val;
            if (off + 48 <= file.size()) {
                auto read_f = [&](uint32_t o) -> float {
                    uint32_t u = r32(file.data() + o);
                    float v = 0.f;
                    std::memcpy(&v, &u, sizeof(v));
                    return v;
                };
                for (int k = 0; k < 3; ++k) info.wb[k] = read_f(off + (uint32_t)k * 4);
                for (int k = 0; k < 9; ++k)
                    info.cam_to_srgb[k] = read_f(off + 12 + (uint32_t)k * 4);
                bool wb_ok = true;
                for (int k = 0; k < 3; ++k)
                    wb_ok = wb_ok && std::isfinite(info.wb[k]) && info.wb[k] > 1e-6f;
                if (!wb_ok) { info.wb[0] = info.wb[1] = info.wb[2] = 1.f; }
                info.has_wb = wb_ok;
                info.has_cam_to_srgb = true;
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
                    info.color_matrix[k] = (float)num / (float)den;
                    ok = ok && std::isfinite(info.color_matrix[k]);
                }
                info.has_color_matrix = ok;
            }
        }
        if (tag == 50728 && (type == T_RATIONAL || type == T_SRATIONAL) && count >= 3) {
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
                    info.as_shot_neutral[k] = (float)num / (float)den;
                    ok = ok && std::isfinite(info.as_shot_neutral[k]) &&
                         info.as_shot_neutral[k] > 1e-6f;
                }
                info.has_as_shot_neutral = ok;
            }
            continue;
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
                    info.analog_balance[k] = (float)num / (float)den;
                    ok = ok && std::isfinite(info.analog_balance[k]) &&
                         std::fabs(info.analog_balance[k]) > 1e-6f;
                }
                info.has_analog_balance = ok;
            }
        }
    }
    return true;
}

bool load_linear_dng_rgb16_color(const std::string& path, std::vector<uint16_t>& rgb,
                                 int& W, int& H, float wb[3], float cam_to_srgb[9],
                                 bool& has_color) {
    has_color = false;
    wb[0] = wb[1] = wb[2] = 1.f;
    for (int i = 0; i < 9; ++i) cam_to_srgb[i] = (i % 4 == 0) ? 1.f : 0.f;

    LinearDngColorInfo info;
    if (!load_linear_dng_rgb16_info(path, rgb, W, H, info)) return false;

    if (info.has_cam_to_srgb) {
        for (int k = 0; k < 3; ++k) wb[k] = info.wb[k];
        for (int k = 0; k < 9; ++k) cam_to_srgb[k] = info.cam_to_srgb[k];
        has_color = true;
    }
    if ((!info.has_cam_to_srgb || is_identity_3x3(cam_to_srgb)) && info.has_color_matrix) {
        float derived[9];
        if (derive_cam_to_srgb_from_color_matrix(
                info.color_matrix,
                info.has_analog_balance ? info.analog_balance : nullptr, derived)) {
            for (int k = 0; k < 9; ++k) cam_to_srgb[k] = derived[k];
            has_color = true;
        }
    }
    return true;
}

} // namespace hhsr
