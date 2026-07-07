#include "raw_io.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>

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
static Image decode_raw_file(LibRaw& raw, Config& cfg, bool is_reference,
                             int crop_h, int crop_w) {
    Image img;
    if (raw.imgdata.rawdata.raw_image == nullptr) return img;

    auto& S = raw.imgdata.sizes;
    auto& C = raw.imgdata.color;
    int stride = S.raw_width;
    int top = S.top_margin & ~1, left = S.left_margin & ~1;
    int vw = S.width & ~1, vh = S.height & ~1;
    img = Image(vh, vw, 1);

    float maxv = (float)C.maximum > 0 ? (float)C.maximum : 65535.f;
    float black = (float)C.black;
    float denom = std::max(1.f, maxv - black);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x)
            img.at(y, x) = clampf(
                ((float)raw.imgdata.rawdata.raw_image[(top + y) * stride + (left + x)] - black) / denom,
                0.f, 1.f);

    if (is_reference) {
        if (raw.imgdata.idata.make[0])
            cfg.camera_make = raw.imgdata.idata.make;
        if (raw.imgdata.idata.model[0])
            cfg.camera_model = raw.imgdata.idata.model;
        for (int i = 0; i < 3; ++i)
            if (C.cam_mul[i] > 0) cfg.white_balance[i] = C.cam_mul[i] / C.cam_mul[1];
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
    Image img = decode_raw_file(*raw, cfg, is_reference, crop_h, crop_w);
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
