#include "raw_io.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>

#ifdef HAVE_LIBRAW
// LibRaw pulls in <windows.h> on Windows, which defines min/max macros that
// clash with std::min/std::max. Suppress them before the include.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <libraw/libraw.h>
#endif

namespace hhsr {

std::vector<Image> synth_burst(int h, int w, int n, unsigned seed) {
    // Ensure even dimensions for the Bayer quads.
    h &= ~1; w &= ~1;
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.f, 0.01f);
    std::uniform_real_distribution<float> jitter(-1.5f, 1.5f);

    // Build a clean high-frequency "scene" (slanted edges + rings) at 0..1.
    auto scene = [&](float y, float x) {
        float v = 0.5f + 0.4f * std::sin(0.20f * (x + y)) * std::cos(0.13f * (x - y));
        v += 0.15f * ((int)(x * 0.25f) % 2 == 0 ? 1.f : -1.f);
        return clampf(v, 0.f, 1.f);
    };

    // Default RGGB CFA.
    auto cfa_gain = [](int y, int x) {
        // Slight per-channel gain to make demosaicking visible.
        int c = (y & 1) == 0 ? ((x & 1) == 0 ? 0 : 1) : ((x & 1) == 0 ? 1 : 2);
        return c == 0 ? 0.9f : (c == 2 ? 0.8f : 1.0f);
    };

    std::vector<Image> burst;
    burst.reserve(n);
    for (int k = 0; k < n; ++k) {
        float sy = (k == 0) ? 0.f : jitter(rng);
        float sx = (k == 0) ? 0.f : jitter(rng);
        Image img(h, w, 1);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                img.at(y, x) = clampf(scene(y + sy, x + sx) * cfa_gain(y, x) + noise(rng), 0.f, 1.f);
        burst.push_back(std::move(img));
    }
    return burst;
}

#ifdef HAVE_LIBRAW

// DNG NoiseProfile tag 0xC761 (DOUBLE). Python: tags['Image Tag 0xC761'].
// Walks IFD0 + SubIFDs (330) + IFD chain. Supports II (LE) and MM (BE) — Apple
// ProRAW/DNG is often MM; old LE-only path silently fell back to Pixel α/β.
static bool try_read_dng_noise_profile(const std::string& path, float& alpha, float& beta) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t hdr[8];
    if (std::fread(hdr, 1, 8, f) != 8) { std::fclose(f); return false; }
    const bool le = (hdr[0] == 'I' && hdr[1] == 'I');
    const bool be = (hdr[0] == 'M' && hdr[1] == 'M');
    if (!le && !be) { std::fclose(f); return false; }

    auto u16 = [&](const uint8_t* p) -> uint16_t {
        return le ? (uint16_t)(p[0] | (p[1] << 8))
                  : (uint16_t)((p[0] << 8) | p[1]);
    };
    auto u32 = [&](const uint8_t* p) -> uint32_t {
        return le ? (uint32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24))
                  : (uint32_t)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
    };
    auto rd16 = [&](uint16_t& v) -> bool {
        uint8_t b[2];
        if (std::fread(b, 1, 2, f) != 2) return false;
        v = u16(b);
        return true;
    };
    auto rd32 = [&](uint32_t& v) -> bool {
        uint8_t b[4];
        if (std::fread(b, 1, 4, f) != 4) return false;
        v = u32(b);
        return true;
    };

    uint32_t ifd0 = u32(hdr + 4);

    auto parse_noise_at = [&](uint32_t data_off, uint32_t cnt, float& a_out, float& b_out) -> bool {
        if (cnt < 2 || (cnt % 2u) != 0) return false;
        std::vector<uint8_t> bytes((size_t)cnt * 8u);
        if (std::fseek(f, (long)data_off, SEEK_SET) != 0) return false;
        if (std::fread(bytes.data(), 8, cnt, f) != cnt) return false;
        double sa = 0, sb = 0;
        const uint32_t nplanes = cnt / 2u;
        // Python super_resolution.py (bayer): always sum(::2)/3 and sum(1::2)/3 —
        // even when the tag has only one plane. Dividing by nplanes alone made α≈3×
        // too large on Apple DNGs → whitened robustness masks.
        const uint32_t use = (nplanes >= 3) ? 3u : nplanes;
        for (uint32_t pi = 0; pi < use; ++pi) {
            for (int k = 0; k < 2; ++k) {
                uint8_t raw8[8];
                const uint8_t* src = bytes.data() + (size_t)(2 * pi + (uint32_t)k) * 8u;
                if (le) {
                    std::memcpy(raw8, src, 8);
                } else {
                    for (int i = 0; i < 8; ++i) raw8[i] = src[7 - i];
                }
                double d;
                std::memcpy(&d, raw8, 8);
                if (k == 0) sa += d; else sb += d;
            }
        }
        a_out = (float)(sa / 3.0);
        b_out = (float)(sb / 3.0);
        return a_out > 0.f && std::isfinite(a_out) && std::isfinite(b_out);
    };

    auto scan_ifd = [&](uint32_t off, float& a_out, float& b_out,
                        std::vector<uint32_t>& subifds, uint32_t& next_ifd) -> bool {
        next_ifd = 0;
        if (off == 0) return false;
        if (std::fseek(f, (long)off, SEEK_SET) != 0) return false;
        uint16_t nent = 0;
        if (!rd16(nent) || nent == 0 || nent > 512) return false;
        bool found = false;
        for (uint16_t i = 0; i < nent; ++i) {
            uint8_t e[12];
            if (std::fread(e, 1, 12, f) != 12) return false;
            const uint16_t tag = u16(e + 0);
            const uint16_t typ = u16(e + 2);
            const uint32_t cnt = u32(e + 4);
            const uint32_t val = u32(e + 8);
            const long ent_end = (long)off + 2 + (long)(i + 1) * 12;

            if (tag == 330 && cnt >= 1 && typ == 4) {
                if (cnt == 1) {
                    subifds.push_back(val);
                } else {
                    std::vector<uint8_t> blob((size_t)cnt * 4u);
                    if (std::fseek(f, (long)val, SEEK_SET) == 0 &&
                        std::fread(blob.data(), 4, cnt, f) == cnt) {
                        for (uint32_t j = 0; j < cnt; ++j)
                            subifds.push_back(u32(blob.data() + (size_t)j * 4u));
                    }
                    if (std::fseek(f, ent_end, SEEK_SET) != 0) return found;
                }
            }
            // NoiseProfile = 0xC761 only (DOUBLE).
            if (!found && tag == 0xC761 && cnt >= 2 && typ == 12) {
                const uint32_t data_off = (cnt * 8u > 4u)
                    ? val
                    : (uint32_t)(off + 2u + (uint32_t)i * 12u + 8u);
                float aa = 0, bb = 0;
                if (parse_noise_at(data_off, cnt, aa, bb)) {
                    a_out = aa; b_out = bb; found = true;
                }
                if (std::fseek(f, ent_end, SEEK_SET) != 0) return found;
            }
        }
        if (!rd32(next_ifd)) next_ifd = 0;
        return found;
    };

    float a = 0, b = 0;
    std::vector<uint32_t> queue;
    queue.push_back(ifd0);
    std::vector<uint32_t> visited;
    bool ok = false;
    for (size_t qi = 0; qi < queue.size() && qi < 32; ++qi) {
        uint32_t cur = queue[qi];
        bool seen = false;
        for (uint32_t v : visited) if (v == cur) { seen = true; break; }
        if (seen || cur == 0) continue;
        visited.push_back(cur);
        std::vector<uint32_t> subs;
        uint32_t next = 0;
        if (scan_ifd(cur, a, b, subs, next)) ok = true;
        for (uint32_t s : subs) queue.push_back(s);
        if (next) queue.push_back(next);
        if (ok) break;
    }
    std::fclose(f);
    if (!ok) return false;
    alpha = a;
    beta = b;
    return true;
}
static Image decode_raw_file(LibRaw& raw, Config& cfg, bool is_reference,
                             int crop_h, int crop_w, const std::string& path) {
    Image img;
    if (raw.imgdata.rawdata.raw_image == nullptr) return img;

    auto& S = raw.imgdata.sizes;
    auto& C = raw.imgdata.color;
    int stride = S.raw_width;
    // Python utils_dng.py uses raw.raw_image.copy(), not raw_image_visible.
    // Use LibRaw's full raw buffer so the algorithm sees the same raw geometry.
    int top = 0, left = 0;
    int vw = S.raw_width & ~1, vh = S.raw_height & ~1;
    img = Image(vh, vw, 1);

    // Metadata first so CFA/WB exist for this frame (and for comps after ref).
    if (is_reference) {
        if (raw.imgdata.idata.make[0])
            cfg.camera_make = raw.imgdata.idata.make;
        if (raw.imgdata.idata.model[0])
            cfg.camera_model = raw.imgdata.idata.model;
        // Python utils_dng: store raw camera_whitebalance (not /green).
        // Load multiplies by (wb[c]/wb[G]); later stages consume those values directly.
        for (int i = 0; i < 3; ++i)
            if (C.cam_mul[i] > 0) cfg.white_balance[i] = C.cam_mul[i];
        if (!(cfg.white_balance[1] > 0.f)) cfg.white_balance[1] = 1.f;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                int c = raw.COLOR(i, j);
                cfg.cfa.p[i][j] = (c == 3) ? 1 : (uint8_t)c;
            }
        bool any = false;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (C.cam_xyz[i][j] != 0.f) any = true;
        if (any) {
            cfg.has_color_matrix = true;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    cfg.color_matrix[i * 3 + j] = C.cam_xyz[i][j];
        }
        bool any_rc = false;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (C.rgb_cam[i][j] != 0.f) any_rc = true;
        if (any_rc) {
            cfg.has_cam_to_srgb = true;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    cfg.cam_to_srgb[i * 3 + j] = C.rgb_cam[i][j];
        }
        switch (S.flip) {
            case 3:  cfg.orientation = 3; break;
            case 5:  cfg.orientation = 8; break;
            case 6:  cfg.orientation = 6; break;
            default: cfg.orientation = 1; break;
        }
        float na = 0.f, nb = 0.f;
        if (try_read_dng_noise_profile(path, na, nb)) {
            cfg.alpha = na;
            cfg.beta = nb;
            cfg.has_noise_profile = true;
        } else {
            cfg.has_noise_profile = false;
        }

        // Python: black_level_per_channel[R,G,B,(G2)]; index by CFA color.
        float black_ch[4];
        bool have_cblack = false;
        for (int i = 0; i < 4; ++i) {
            if (C.cblack[i] != 0) have_cblack = true;
            black_ch[i] = (float)C.black + (float)C.cblack[i];
        }
        if (!have_cblack) {
            for (int i = 0; i < 4; ++i) black_ch[i] = (float)C.black;
        }
        cfg.black_levels[0] = black_ch[0];
        cfg.black_levels[1] = black_ch[1];
        cfg.black_levels[2] = black_ch[2];
        cfg.white_level = (float)C.maximum > 0 ? (float)C.maximum : 65535.f;
        cfg.has_black_levels = true;
    }

    // Python utils_dng: per-channel black, (v-black)/(white-black), then * (wb[c]/wb[G]).
    float maxv = cfg.has_black_levels && cfg.white_level > 0.f
        ? cfg.white_level
        : ((float)C.maximum > 0 ? (float)C.maximum : 65535.f);
    float bl_rgb[3];
    if (cfg.has_black_levels) {
        bl_rgb[0] = cfg.black_levels[0];
        bl_rgb[1] = cfg.black_levels[1];
        bl_rgb[2] = cfg.black_levels[2];
    } else {
        float black_ch[4];
        bool have_cblack = false;
        for (int i = 0; i < 4; ++i) {
            if (C.cblack[i] != 0) have_cblack = true;
            black_ch[i] = (float)C.black + (float)C.cblack[i];
        }
        if (!have_cblack) {
            for (int i = 0; i < 4; ++i) black_ch[i] = (float)C.black;
        }
        bl_rgb[0] = black_ch[0];
        bl_rgb[1] = black_ch[1];
        bl_rgb[2] = black_ch[2];
    }
    float site_black[2][2];
    float site_wb[2][2];
    const float wb_g = (cfg.white_balance[1] > 0.f) ? cfg.white_balance[1] : 1.f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int c = (int)cfg.cfa.p[i][j];
            if (c < 0) c = 0;
            if (c > 2) c = 1;
            site_black[i][j] = bl_rgb[c];
            // Python: k = white_balance[channel] / white_balance[1]
            float w = cfg.white_balance[c] / wb_g;
            if (!(w > 0.f) || !std::isfinite(w)) w = 1.f;
            site_wb[i][j] = w;
        }
    }

    for (int y = 0; y < img.h; ++y) {
        const int fi = y & 1;
        for (int x = 0; x < img.w; ++x) {
            const int fj = x & 1;
            float bl = site_black[fi][fj];
            float denom = std::max(1.f, maxv - bl);
            float v = ((float)raw.imgdata.rawdata.raw_image[(top + y) * stride + (left + x)] - bl) / denom;
            v *= site_wb[fi][fj];
            if (!std::isfinite(v) || v < 0.f) v = 0.f;
            else if (v > 1.f) v = 1.f;
            img.at(y, x) = v;
        }
    }
    cfg.raw_prewhitened = true;

    if (cfg.input_crop_factor > 1) {
        const int factor = cfg.input_crop_factor;
        int ch = (img.h / factor) & ~1;
        int cw = (img.w / factor) & ~1;
        int y0 = ((img.h - ch) / 2) & ~1;
        int x0 = ((img.w - cw) / 2) & ~1;
        if (ch > 0 && cw > 0) {
            Image cropped(ch, cw, 1);
            for (int y = 0; y < ch; ++y)
                for (int x = 0; x < cw; ++x)
                    cropped.at(y, x) = img.at(y0 + y, x0 + x);
            img = std::move(cropped);
        }
    }

    if (crop_h > 0 && crop_w > 0 && (img.h > crop_h || img.w > crop_w)) {
        int mh = std::min(img.h, crop_h) & ~1;
        int mw = std::min(img.w, crop_w) & ~1;
        Image c(mh, mw, 1);
        for (int y = 0; y < mh; ++y)
            for (int x = 0; x < mw; ++x)
                c.at(y, x) = img.at(y, x);
        img = std::move(c);
    }
    return img;
}
#endif

Image load_raw_frame(const std::string& path, Config& cfg, bool is_reference,
                     int crop_h, int crop_w) {
#ifdef HAVE_LIBRAW
    // LibRaw embeds a multi-MB libraw_data_t — must not live on the small GCD
    // worker stack (~512 KB) or iOS kills the thread (SIGBUS / stack guard).
    std::unique_ptr<LibRaw> raw(new LibRaw());
    if (raw->open_file(path.c_str()) != LIBRAW_SUCCESS) return Image();
    if (raw->unpack() != LIBRAW_SUCCESS) { raw->recycle(); return Image(); }
    Image img = decode_raw_file(*raw, cfg, is_reference, crop_h, crop_w, path);
    raw->recycle();
    return img;
#else
    (void)path; (void)cfg; (void)is_reference; (void)crop_h; (void)crop_w;
    return Image();
#endif
}

std::vector<Image> load_raw_burst(const std::vector<std::string>& paths, Config& cfg) {
    std::vector<Image> burst;
#ifdef HAVE_LIBRAW
    int crop_h = 0, crop_w = 0;
    for (const auto& p : paths) {
        bool is_ref = burst.empty();
        Image img = load_raw_frame(p, cfg, is_ref, crop_h, crop_w);
        if (img.h <= 0 || img.w <= 0) continue;
        if (is_ref) { crop_h = img.h; crop_w = img.w; }
        burst.push_back(std::move(img));
    }
#else
    (void)paths; (void)cfg;
#endif
    return burst;
}

std::vector<uint8_t> to_rgba8(const Image& rgb) {
    std::vector<uint8_t> out((size_t)rgb.h * rgb.w * 4);
    auto srgb = [](f32 v) {
        v = clampf(v, 0.f, 1.f);
        v = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
        return (uint8_t)std::lround(clampf(v, 0.f, 1.f) * 255.f);
    };
    for (int y = 0; y < rgb.h; ++y) {
        for (int x = 0; x < rgb.w; ++x) {
            size_t o = ((size_t)y * rgb.w + x) * 4;
            if (rgb.c >= 3) {
                out[o + 0] = srgb(rgb.at(y, x, 0));
                out[o + 1] = srgb(rgb.at(y, x, 1));
                out[o + 2] = srgb(rgb.at(y, x, 2));
            } else {
                uint8_t g = srgb(rgb.at(y, x, 0));
                out[o + 0] = out[o + 1] = out[o + 2] = g;
            }
            out[o + 3] = 255;
        }
    }
    return out;
}

} // namespace hhsr
