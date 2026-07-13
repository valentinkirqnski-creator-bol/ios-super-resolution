#include "stages.h"
#include "parallel.h"
#include <complex>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace hhsr {

namespace {

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

} // namespace

// Exposed FFT helpers (used by both grey_pyramid.cpp and align.cpp).
void fft1d(std::vector<std::complex<f32>>& a, bool inverse) {
    const int n = (int)a.size();
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        f32 ang = (inverse ? -2.f : 2.f) * (f32)M_PI / (f32)len;
        std::complex<f32> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<f32> w(1.f, 0.f);
            for (int k = 0; k < len / 2; ++k) {
                std::complex<f32> u = a[i + k];
                std::complex<f32> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (auto& v : a) v /= (f32)n;
    }
}

void fft2d(std::vector<std::complex<f32>>& data, int h, int w, bool inverse) {
    std::vector<std::complex<f32>> row((size_t)w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) row[(size_t)x] = data[(size_t)y * w + x];
        fft1d(row, inverse);
        for (int x = 0; x < w; ++x) data[(size_t)y * w + x] = row[(size_t)x];
    }
    row.assign((size_t)h, {0.f, 0.f});
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) row[(size_t)y] = data[(size_t)y * w + x];
        fft1d(row, inverse);
        for (int y = 0; y < h; ++y) data[(size_t)y * w + x] = row[(size_t)y];
    }
}

void fftshift2d(std::vector<std::complex<f32>>& data, int h, int w) {
    auto swap_quadrant = [&](int y0, int x0, int y1, int x1, int hh, int ww) {
        for (int y = 0; y < hh; ++y) {
            for (int x = 0; x < ww; ++x) {
                std::swap(data[(size_t)(y0 + y) * w + (x0 + x)],
                          data[(size_t)(y1 + y) * w + (x1 + x)]);
            }
        }
    };
    swap_quadrant(0, 0, h / 2, w / 2, h / 2, w / 2);
    swap_quadrant(0, w / 2, h / 2, 0, h / 2, w - w / 2);
}

// Alg. 3 FFT grey — matches utils_image.compute_grey_images(method="FFT").
Image compute_grey_fft(const Image& raw) {
    const int h = raw.h, w = raw.w;
    const int ph = next_pow2(h), pw = next_pow2(w);
    std::vector<std::complex<f32>> buf((size_t)ph * pw, {0.f, 0.f});
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            buf[(size_t)y * pw + x] = {raw.at(y, x), 0.f};

    fft2d(buf, ph, pw, false);
    fftshift2d(buf, ph, pw);

    const int y0 = ph / 4, x0 = pw / 4;
    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            if (y < y0 || y >= ph - y0 || x < x0 || x >= pw - x0)
                buf[(size_t)y * pw + x] = {0.f, 0.f};
        }
    }

    fftshift2d(buf, ph, pw);
    fft2d(buf, ph, pw, true);

    Image grey(h, w, 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            grey.at(y, x) = buf[(size_t)y * pw + x].real();
    return grey;
}

Image compute_grey(const Image& raw, bool bayer_mode, GreyMethod method) {
    if (!bayer_mode) return raw;
    if (method == GreyMethod::FFT) return compute_grey_fft(raw);
    return compute_grey_decimate(raw, true);
}

Image compute_grey_decimate(const Image& raw, bool bayer_mode) {
    if (!bayer_mode) return raw;
    int gh = raw.h / 2, gw = raw.w / 2;
    Image grey(gh, gw, 1);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            f32 s = raw.at(2 * y, 2 * x) + raw.at(2 * y, 2 * x + 1) +
                    raw.at(2 * y + 1, 2 * x) + raw.at(2 * y + 1, 2 * x + 1);
            grey.at(y, x) = 0.25f * s;
        }
    }
    return grey;
}

// 5-tap binomial kernel (1 4 6 4 1)/16, applied separably. Matches the
// Gaussian pyramid downsampling used by the reference (cuda_downsample).
static Image downsample_by(const Image& src, int factor) {
    if (factor <= 1) return src;
    // Blur then subsample. Use repeated binomial blur proportional to factor.
    Image blurred = gaussian_blur(src, 0.5f * factor);
    int dh = src.h / factor, dw = src.w / factor;
    Image out(dh, dw, 1);
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x)
            out.at(y, x) = blurred.at(y * factor, x * factor);
    return out;
}

Pyramid build_pyramid(const Image& grey, const std::vector<int>& factors) {
    Pyramid pyr;
    // factors are defined fine-to-coarse (e.g. {1,2,4,4}); build absolute
    // downsample factors and the images at each level.
    Image cur = grey;
    int abs_factor = 1;
    for (size_t i = 0; i < factors.size(); ++i) {
        int f = factors[i];
        cur = (f == 1 && i == 0) ? grey : downsample_by(cur, f);
        abs_factor *= f;
        pyr.levels.push_back(cur);
        pyr.abs_factors.push_back(abs_factor);
    }
    return pyr;
}

Image compute_gradients(const Image& grey) {
    // Matches kernels.py: separable [-0.5,0.5] pair producing a 2-channel
    // gradient (gx, gy), reducing the image by 1 pixel in each direction.
    int gh = grey.h - 1, gw = grey.w - 1;
    if (gh < 1 || gw < 1) return Image(0, 0, 2);
    Image grad(gh, gw, 2);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            f32 tl = grey.at(y, x),     tr = grey.at(y, x + 1);
            f32 bl = grey.at(y + 1, x), br = grey.at(y + 1, x + 1);
            // gx = horizontal diff averaged over rows; gy = vertical diff.
            grad.at(y, x, 0) = 0.5f * ((tr - tl) + (br - bl));
            grad.at(y, x, 1) = 0.5f * ((bl - tl) + (br - tr));
        }
    }
    return grad;
}

Image gaussian_blur(const Image& src, float sigma) {
    if (sigma <= 0.f) return src;
    int radius = std::max(1, (int)std::ceil(3.f * sigma));
    std::vector<f32> k(2 * radius + 1);
    f32 sum = 0.f;
    for (int i = -radius; i <= radius; ++i) {
        f32 v = std::exp(-0.5f * (i * i) / (sigma * sigma));
        k[i + radius] = v; sum += v;
    }
    for (auto& v : k) v /= sum;

    Image tmp(src.h, src.w, src.c);
    // Horizontal pass (clamp-to-edge borders).
    for (int y = 0; y < src.h; ++y) {
        for (int x = 0; x < src.w; ++x) {
            for (int ch = 0; ch < src.c; ++ch) {
                f32 acc = 0.f;
                for (int i = -radius; i <= radius; ++i) {
                    int xx = clampf((f32)(x + i), 0.f, (f32)(src.w - 1));
                    acc += k[i + radius] * src.at(y, xx, ch);
                }
                tmp.at(y, x, ch) = acc;
            }
        }
    }
    Image out(src.h, src.w, src.c);
    // Vertical pass.
    for (int y = 0; y < src.h; ++y) {
        for (int x = 0; x < src.w; ++x) {
            for (int ch = 0; ch < src.c; ++ch) {
                f32 acc = 0.f;
                for (int i = -radius; i <= radius; ++i) {
                    int yy = clampf((f32)(y + i), 0.f, (f32)(src.h - 1));
                    acc += k[i + radius] * tmp.at(yy, x, ch);
                }
                out.at(y, x, ch) = acc;
            }
        }
    }
    return out;
}

} // namespace hhsr
