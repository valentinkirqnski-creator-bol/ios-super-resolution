// Structure tensors of burst frames (see structure_map.h).
#include "structure_map.h"
#include "parallel.h"
#if defined(__APPLE__)
#include "metal_gpu.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace hhsr {

static void sm_hsv_to_rgb8(float h, float s, float v, unsigned char* rgb) {
    const float hh = (h - std::floor(h)) * 6.f;
    const int i = (int)hh;
    const float f = hh - (float)i;
    const float p = v * (1.f - s), q = v * (1.f - s * f), t = v * (1.f - s * (1.f - f));
    float r, g, b;
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
    auto q8 = [](float x) { return (unsigned char)(std::min(1.f, std::max(0.f, x)) * 255.f + 0.5f); };
    rgb[0] = q8(r); rgb[1] = q8(g); rgb[2] = q8(b);
}

// Separable Gaussian blur of channel `ch` of a (h, w, 4) image, in place (tmp: h*w scratch).
static void sm_blur(Image& a, int ch, std::vector<float>& tmp, const std::vector<float>& k, int nt) {
    const int h = a.h, w = a.w, r = (int)k.size() - 1;
    parallel_rows(h, nt, [&](int y) {
        for (int x = 0; x < w; ++x) {
            float s = k[0] * a.at(y, x, ch);
            for (int i = 1; i <= r; ++i)
                s += k[(size_t)i] * (a.at(y, std::max(0, x - i), ch) + a.at(y, std::min(w - 1, x + i), ch));
            tmp[(size_t)y * w + x] = s;
        }
    });
    parallel_rows(h, nt, [&](int y) {
        for (int x = 0; x < w; ++x) {
            float s = k[0] * tmp[(size_t)y * w + x];
            for (int i = 1; i <= r; ++i)
                s += k[(size_t)i] * (tmp[(size_t)std::max(0, y - i) * w + x] + tmp[(size_t)std::min(h - 1, y + i) * w + x]);
            a.at(y, x, ch) = s;
        }
    });
}

Image structure_tensor_half(const Image& raw, const Config& cfg, float sigma) {
    if (raw.h < 4 || raw.w < 4 || raw.data.size() < (size_t)raw.h * raw.w * raw.c) return Image();
    const int gh = raw.h / 2, gw = raw.w / 2;
    const size_t plane = (size_t)gh * gw;
    const int nt = cfg.num_threads;
    // DNG noise profile of one raw sample at level m: alpha m + beta (green channel's).
    const float na = std::max(0.f, cfg.alpha_dng[1]), nb = std::max(0.f, cfg.beta_dng[1]);

    // Half-resolution grey: sqrt of the mean of each 2x2 quad (all CFA colours).
    std::vector<float> g(plane);
    parallel_rows(gh, nt, [&](int y) {
        for (int x = 0; x < gw; ++x) {
            float s = 0.f;
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    for (int c = 0; c < raw.c; ++c) s += raw.at(2 * y + i, 2 * x + j, c);
            g[(size_t)y * gw + x] = std::sqrt(std::max(0.f, s / (float)(4 * raw.c)));
        }
    });

    // Tensor components from central differences, plus the per-axis noise variance of
    // that gradient: quad mean of 4 samples -> var (alpha m + beta) / 4; sqrt -> divided
    // by 4m (delta method); central difference of two independent samples -> / 2.
    Image J(gh, gw, 4);
    parallel_rows(gh, nt, [&](int y) {
        const int ym = std::max(0, y - 1), yp = std::min(gh - 1, y + 1);
        for (int x = 0; x < gw; ++x) {
            const int xm = std::max(0, x - 1), xp = std::min(gw - 1, x + 1);
            const float gx = 0.5f * (g[(size_t)y * gw + xp] - g[(size_t)y * gw + xm]);
            const float gy = 0.5f * (g[(size_t)yp * gw + x] - g[(size_t)ym * gw + x]);
            const float v = g[(size_t)y * gw + x];
            const float m = std::max(1e-5f, v * v);
            J.at(y, x, 0) = gx * gx;
            J.at(y, x, 1) = gx * gy;
            J.at(y, x, 2) = gy * gy;
            J.at(y, x, 3) = (na * m + nb) / (32.f * m);
        }
    });

    const float s = std::max(0.3f, sigma);
    const int kr = std::max(1, (int)std::ceil(3.f * s));
    std::vector<float> k((size_t)kr + 1);
    float ks = 0.f;
    for (int i = 0; i <= kr; ++i) {
        k[(size_t)i] = std::exp(-0.5f * (float)(i * i) / (s * s));
        ks += (i == 0) ? k[0] : 2.f * k[(size_t)i];
    }
    for (float& v : k) v /= ks;
    for (int ch = 0; ch < 4; ++ch) sm_blur(J, ch, g, k, nt);   // g reused as scratch
    return J;
}

bool write_structure_map(const Image& raw_in, const Config& cfg, const std::string& dng_path,
                         int frame_index, bool is_reference, float& scale_io) {
    const Image* rawp = &raw_in;
#if defined(__APPLE__)
    // Capture path: the frame was decoded straight into its GPU slot and raw_in is a
    // header-only shell; read the ACTIVE slot back (the caller has switched to it).
    Image raw_host;
    if (raw_in.data.size() < (size_t)raw_in.h * (size_t)raw_in.w * (size_t)std::max(1, raw_in.c) &&
        metal_fetch_host_raw(raw_host) && raw_host.h == raw_in.h && raw_host.w == raw_in.w)
        rawp = &raw_host;
    // Tensor on the GPU (structure_tensor_metal); CPU twin when unavailable.
    Image J = structure_tensor_metal(*rawp, cfg, cfg.structure_map_sigma);
    if (J.h <= 0 || J.w <= 0) J = structure_tensor_half(*rawp, cfg, cfg.structure_map_sigma);
#else
    Image J = structure_tensor_half(*rawp, cfg, cfg.structure_map_sigma);
#endif
    if (J.h <= 0 || J.w <= 0) return false;
    const int gh = J.h, gw = J.w;
    const size_t plane = (size_t)gh * gw;
    const int nt = cfg.num_threads;

    std::vector<float> strength(plane);
    parallel_rows(gh, nt, [&](int y) {
        for (int x = 0; x < gw; ++x) {
            const float a = J.at(y, x, 0), b = J.at(y, x, 1), d = J.at(y, x, 2);
            const float rt = std::sqrt(0.25f * (a - d) * (a - d) + b * b);
            strength[(size_t)y * gw + x] = std::sqrt(std::max(0.f, 0.5f * (a + d) + rt));
        }
    });
    if (!(scale_io > 0.f)) {
        std::vector<float> smp;
        smp.reserve(plane / 16 + 1);
        for (size_t i = 0; i < plane; i += 16) smp.push_back(strength[i]);
        const size_t q = (size_t)(0.995 * (double)(smp.size() - 1));
        std::nth_element(smp.begin(), smp.begin() + (std::ptrdiff_t)q, smp.end());
        scale_io = std::max(1e-6f, smp[q]);
    }
    const float inv_scale = 1.f / scale_io;

    std::vector<unsigned char> rgb(plane * 3);
    parallel_rows(gh, nt, [&](int y) {
        for (int x = 0; x < gw; ++x) {
            const size_t o = (size_t)y * gw + x;
            const float a = J.at(y, x, 0), b = J.at(y, x, 1), d = J.at(y, x, 2);
            const float rt = std::sqrt(0.25f * (a - d) * (a - d) + b * b);
            const float l1 = 0.5f * (a + d) + rt, l2 = std::max(0.f, 0.5f * (a + d) - rt);
            const float coh = (l1 + l2 > 1e-20f) ? (l1 - l2) / (l1 + l2) : 0.f;
            // Dominant gradient direction theta; the edge runs perpendicular to it.
            const float theta = 0.5f * std::atan2(2.f * b, a - d);
            const float hue = (theta + 1.5707963f) / 3.14159265f;   // [0, 1] (wraps)
            sm_hsv_to_rgb8(hue, coh, strength[o] * inv_scale, &rgb[o * 3]);
        }
    });

    char tag[48];
    std::snprintf(tag, sizeof(tag), "_structure_f%02d%s.ppm", frame_index, is_reference ? "_ref" : "");
    std::string path = dng_path;
    const std::string suf = ".dng";
    if (path.size() >= suf.size() && path.compare(path.size() - suf.size(), suf.size(), suf) == 0)
        path.replace(path.size() - suf.size(), suf.size(), tag);
    else
        path += tag;
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << gw << " " << gh << "\n255\n";
    out.write((const char*)rgb.data(), (std::streamsize)rgb.size());
    return (bool)out;
}

}  // namespace hhsr
