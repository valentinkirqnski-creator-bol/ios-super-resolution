#include "raw_io.h"
#include <random>
#include <cmath>
#include <algorithm>

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

std::vector<Image> load_raw_burst(const std::vector<std::string>& paths, Config& cfg) {
    std::vector<Image> burst;
#ifdef HAVE_LIBRAW
    for (const auto& p : paths) {
        LibRaw raw;
        if (raw.open_file(p.c_str()) != LIBRAW_SUCCESS) continue;
        if (raw.unpack() != LIBRAW_SUCCESS) { raw.recycle(); continue; }
        auto& S = raw.imgdata.sizes;
        auto& C = raw.imgdata.color;
        if (raw.imgdata.rawdata.raw_image == nullptr) { raw.recycle(); continue; } // not a bayer raw

        int stride = S.raw_width;               // full row stride incl. margins
        int top = S.top_margin, left = S.left_margin;
        int vw = S.width, vh = S.height;        // visible (active) area
        // Keep CFA phase consistent: align crop to even pixels.
        top &= ~1; left &= ~1;
        Image img(vh & ~1, vw & ~1, 1);
        float maxv = (float)C.maximum > 0 ? (float)C.maximum : 65535.f;
        float black = (float)C.black;
        float denom = std::max(1.f, maxv - black);
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x)
                img.at(y, x) = clampf(((float)raw.imgdata.rawdata.raw_image[(top + y) * stride + (left + x)] - black) / denom,
                                      0.f, 1.f);
        burst.push_back(std::move(img));

        if (&p == &paths.front()) {
            // Populate metadata-driven config from the reference frame.
            for (int i = 0; i < 3; ++i)
                if (C.cam_mul[i] > 0) cfg.white_balance[i] = C.cam_mul[i] / C.cam_mul[1];
            // libraw CFA index -> our 2x2 (0=R,1=G,2=B).
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) {
                    int c = raw.COLOR(i, j);
                    cfg.cfa.p[i][j] = (c == 3) ? 1 : (uint8_t)c; // second green -> green
                }
            // ColorMatrix1 (XYZ D65 -> camera RGB) straight from the file.
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
            // camera-RGB -> linear sRGB (D65), used to bake display-ready color.
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
            // Sensor orientation -> EXIF orientation for the output DNG.
            switch (S.flip) {
                case 3:  cfg.orientation = 3; break; // 180
                case 5:  cfg.orientation = 8; break; // 90 CCW
                case 6:  cfg.orientation = 6; break; // 90 CW
                default: cfg.orientation = 1; break; // none
            }
        }
        raw.recycle();
    }
#else
    (void)paths; (void)cfg;
#endif

    // The pipeline requires all frames to share the reference dimensions.
    // Crop every frame to the common (min) even size, preserving CFA phase.
    if (burst.size() > 1) {
        int mh = burst[0].h, mw = burst[0].w;
        for (const auto& im : burst) { mh = std::min(mh, im.h); mw = std::min(mw, im.w); }
        mh &= ~1; mw &= ~1;
        for (auto& im : burst) {
            if (im.h == mh && im.w == mw) continue;
            Image c(mh, mw, 1);
            for (int y = 0; y < mh; ++y)
                for (int x = 0; x < mw; ++x)
                    c.at(y, x) = im.at(y, x);
            im = std::move(c);
        }
    }
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
