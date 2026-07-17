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

void fft1d(std::vector<std::complex<f32>>& a, bool inverse, std::vector<std::complex<f32>>* dft_buf) {
    const int n = (int)a.size();
    if ((n & (n - 1)) != 0) {
        // Slow exact DFT for non-power-of-2
        std::vector<std::complex<f32>> local_buf;
        std::vector<std::complex<f32>>& out = dft_buf ? *dft_buf : local_buf;
        if (out.size() < (size_t)n) out.resize(n);
        std::fill(out.begin(), out.begin() + n, std::complex<f32>{0.f, 0.f});

        f32 dir = inverse ? 1.f : -1.f;
        for (int k = 0; k < n; ++k) {
            for (int t = 0; t < n; ++t) {
                f32 ang = dir * 2.f * (f32)M_PI * k * t / n;
                out[k] += a[t] * std::complex<f32>(std::cos(ang), std::sin(ang));
            }
        }
        for (int i = 0; i < n; ++i) {
            a[i] = inverse ? out[i] / (f32)n : out[i];
        }
        return;
    }

    // Fast Radix-2 FFT
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

void fft2d(std::vector<std::complex<f32>>& data, int h, int w, bool inverse,
           std::vector<std::complex<f32>>* row_buf,
           std::vector<std::complex<f32>>* dft_buf) {
    std::vector<std::complex<f32>> local_row;
    std::vector<std::complex<f32>>& row = row_buf ? *row_buf : local_row;
    int max_dim = std::max(h, w);
    if (row.size() < (size_t)max_dim) row.resize(max_dim);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) row[(size_t)x] = data[(size_t)y * w + x];
        std::vector<std::complex<f32>> slice(row.begin(), row.begin() + w);
        fft1d(slice, inverse, dft_buf);
        for (int x = 0; x < w; ++x) data[(size_t)y * w + x] = slice[(size_t)x];
    }
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) row[(size_t)y] = data[(size_t)y * w + x];
        std::vector<std::complex<f32>> slice(row.begin(), row.begin() + h);
        fft1d(slice, inverse, dft_buf);
        for (int y = 0; y < h; ++y) data[(size_t)y * w + x] = slice[(size_t)y];
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

// scipy.ndimage._filters._gaussian_kernel1d (order=0) + cuda_downsample:
//   sigma = factor * 0.5, radius = int(4 * sigma + 0.5)
//   valid (no-pad) separable conv, then stride subsample.
static std::vector<f32> scipy_gaussian_kernel1d(f32 sigma, int radius) {
    std::vector<f32> k(2 * radius + 1);
    f32 sigma2 = sigma * sigma;
    f32 sum = 0.f;
    for (int i = -radius; i <= radius; ++i) {
        f32 v = std::exp(-0.5f / sigma2 * (f32)(i * i));
        k[i + radius] = v;
        sum += v;
    }
    for (f32& v : k) v /= sum;
    return k; // order-0 is symmetric; Python's [::-1] for correlate is a no-op
}

static Image downsample_by(const Image& src, int factor) {
    // Matches utils_image.cuda_downsample(..., kernel='gaussian', factor)
    if (factor <= 1) return src;

    f32 sigma = 0.5f * (f32)factor;
    int radius = (int)(4.f * sigma + 0.5f); // int(4*factor*0.5 + 0.5)
    std::vector<f32> ker = scipy_gaussian_kernel1d(sigma, radius);
    const int klen = (int)ker.size(); // 2*radius+1

    // Valid vertical conv: out_h = src.h - klen + 1
    int tmp_h = src.h - klen + 1;
    int tmp_w = src.w;
    if (tmp_h < 1 || tmp_w < 1) return Image(0, 0, 1);
    Image tmp(tmp_h, tmp_w, 1);
    for (int y = 0; y < tmp_h; ++y) {
        for (int x = 0; x < tmp_w; ++x) {
            f32 acc = 0.f;
            for (int i = 0; i < klen; ++i)
                acc += ker[i] * src.at(y + i, x);
            tmp.at(y, x) = acc;
        }
    }

    // Valid horizontal conv
    int filt_h = tmp_h;
    int filt_w = tmp_w - klen + 1;
    if (filt_w < 1) return Image(0, 0, 1);
    Image filtered(filt_h, filt_w, 1);
    for (int y = 0; y < filt_h; ++y) {
        for (int x = 0; x < filt_w; ++x) {
            f32 acc = 0.f;
            for (int j = 0; j < klen; ++j)
                acc += ker[j] * tmp.at(y, x + j);
            filtered.at(y, x) = acc;
        }
    }

    // h2, w2 = floor(shape / factor); return filtered[:, :, :h2*factor:factor, :w2*factor:factor]
    int h2 = filt_h / factor;
    int w2 = filt_w / factor;
    Image out(h2, w2, 1);
    for (int y = 0; y < h2; ++y)
        for (int x = 0; x < w2; ++x)
            out.at(y, x) = filtered.at(y * factor, x * factor);
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
    // Matches kernels.py separable convs:
    //   tmp0 = [-0.5, 0.5] * I,  tmp1 = [0.5, 0.5] * I  (horizontal)
    //   gx = [0.5, 0.5]^T * tmp0, gy = [-0.5, 0.5]^T * tmp1  (vertical, groups=2)
    // => gx = 0.25*((tr-tl)+(br-bl)), gy = 0.25*((bl+br)-(tl+tr))
    // Output is reduced by 1 pixel in each direction.
    int gh = grey.h - 1, gw = grey.w - 1;
    if (gh < 1 || gw < 1) return Image(0, 0, 2);
    Image grad(gh, gw, 2);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            f32 tl = grey.at(y, x),     tr = grey.at(y, x + 1);
            f32 bl = grey.at(y + 1, x), br = grey.at(y + 1, x + 1);
            grad.at(y, x, 0) = 0.25f * ((tr - tl) + (br - bl));
            grad.at(y, x, 1) = 0.25f * ((bl - tl) + (br - tr));
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
