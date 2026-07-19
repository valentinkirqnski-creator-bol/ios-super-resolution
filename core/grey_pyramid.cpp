#include "stages.h"
#include "parallel.h"
#include <complex>
#include <cmath>
#include <vector>

#ifdef __APPLE__
#include "metal_gpu.h"
#endif

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

// Radix-2 only; n must be power of 2. Does not divide by n on inverse.
static void fft1d_pow2_inplace_ref(std::vector<std::complex<f32>>& a, bool inverse) {
    const int n = (int)a.size();
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        f32 ang = (inverse ? 2.f : -2.f) * (f32)M_PI / (f32)len;
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
}

// Use the portable Cooley–Tukey reference (same as Metal fft1d_pow2_cpp).
// Previously Apple used vDSP here; that diverged from the GPU path.
static void fft1d_pow2_inplace(std::vector<std::complex<f32>>& a, bool inverse) {
    fft1d_pow2_inplace_ref(a, inverse);
}

// Bluestein's algorithm — arbitrary-n DFT via padded radix-2 convolution.
static void fft1d_bluestein(std::vector<std::complex<f32>>& a, bool inverse) {
    const int n = (int)a.size();
    if (n <= 1) return;
    const int m = next_pow2(2 * n - 1);
    const f32 dir = inverse ? 1.f : -1.f;

    std::vector<std::complex<f32>> chirp(n);
    for (int i = 0; i < n; ++i) {
        f32 ang = dir * (f32)M_PI * (f32)i * (f32)i / (f32)n;
        chirp[i] = {std::cos(ang), std::sin(ang)};
    }

    std::vector<std::complex<f32>> A((size_t)m, {0.f, 0.f});
    std::vector<std::complex<f32>> B((size_t)m, {0.f, 0.f});
    for (int i = 0; i < n; ++i) {
        A[i] = a[i] * chirp[i];
        B[i] = std::conj(chirp[i]);
        if (i > 0) B[m - i] = B[i];
    }

    fft1d_pow2_inplace(A, false);
    fft1d_pow2_inplace(B, false);
    for (int i = 0; i < m; ++i) A[i] *= B[i];
    fft1d_pow2_inplace(A, true);
    for (int i = 0; i < m; ++i) A[i] /= (f32)m;

    for (int i = 0; i < n; ++i)
        a[i] = A[i] * chirp[i];
}

static void roll_axis0(std::vector<std::complex<f32>>& data, int h, int w, int shift) {
    if (h <= 1 || shift % h == 0) return;
    shift %= h;
    if (shift < 0) shift += h;
    std::vector<std::complex<f32>> tmp(data);
    for (int y = 0; y < h; ++y) {
        int ny = (y + shift) % h;
        for (int x = 0; x < w; ++x)
            data[(size_t)ny * w + x] = tmp[(size_t)y * w + x];
    }
}

static void roll_axis1(std::vector<std::complex<f32>>& data, int h, int w, int shift) {
    if (w <= 1 || shift % w == 0) return;
    shift %= w;
    if (shift < 0) shift += w;
    std::vector<std::complex<f32>> row(w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) row[x] = data[(size_t)y * w + x];
        for (int x = 0; x < w; ++x)
            data[(size_t)y * w + (x + shift) % w] = row[x];
    }
}

} // namespace

void fft1d(std::vector<std::complex<f32>>& a, bool inverse, std::vector<std::complex<f32>>* /*dft_buf*/) {
    const int n = (int)a.size();
    if (n <= 1) return;
    if ((n & (n - 1)) == 0) {
        fft1d_pow2_inplace(a, inverse);
        if (inverse) for (auto& v : a) v /= (f32)n;
        return;
    }
    // Same as Metal: Bluestein (no vDSP DFT — keeps CPU/GPU FFT identical).
    fft1d_bluestein(a, inverse);
    if (inverse) for (auto& v : a) v /= (f32)n;
}

void fft2d(std::vector<std::complex<f32>>& data, int h, int w, bool inverse,
           std::vector<std::complex<f32>>* row_buf,
           std::vector<std::complex<f32>>* dft_buf) {
    (void)dft_buf;
    std::vector<std::complex<f32>> local_work;
    std::vector<std::complex<f32>>& work = row_buf ? *row_buf : local_work;

    for (int y = 0; y < h; ++y) {
        work.resize((size_t)w);
        for (int x = 0; x < w; ++x) work[(size_t)x] = data[(size_t)y * w + x];
        fft1d(work, inverse, nullptr);
        for (int x = 0; x < w; ++x) data[(size_t)y * w + x] = work[(size_t)x];
    }
    for (int x = 0; x < w; ++x) {
        work.resize((size_t)h);
        for (int y = 0; y < h; ++y) work[(size_t)y] = data[(size_t)y * w + x];
        fft1d(work, inverse, nullptr);
        for (int y = 0; y < h; ++y) data[(size_t)y * w + x] = work[(size_t)y];
    }
}

void fftshift2d(std::vector<std::complex<f32>>& data, int h, int w) {
    // numpy.fft.fftshift: roll by +n//2 on each axis
    roll_axis0(data, h, w, h / 2);
    roll_axis1(data, h, w, w / 2);
}

static void ifftshift2d(std::vector<std::complex<f32>>& data, int h, int w) {
    // numpy.fft.ifftshift: roll by -n//2 on each axis
    roll_axis0(data, h, w, -(h / 2));
    roll_axis1(data, h, w, -(w / 2));
}

// Torch torch.fft.rfft2 / irfft2 packing via row–column 1D FFTs (vDSP-backed fft1d).
void rfft2(const f32* in, int h, int w, std::vector<std::complex<f32>>& out) {
    const int wh = w / 2 + 1;
    out.assign((size_t)h * wh, {0.f, 0.f});
    if (h <= 0 || w <= 0) return;

    std::vector<std::complex<f32>> row((size_t)w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x)
            row[(size_t)x] = {in[(size_t)y * w + x], 0.f};
        fft1d(row, false, nullptr);
        for (int x = 0; x < wh; ++x)
            out[(size_t)y * wh + x] = row[(size_t)x];
    }

    std::vector<std::complex<f32>> col((size_t)h);
    for (int x = 0; x < wh; ++x) {
        for (int y = 0; y < h; ++y)
            col[(size_t)y] = out[(size_t)y * wh + x];
        fft1d(col, false, nullptr);
        for (int y = 0; y < h; ++y)
            out[(size_t)y * wh + x] = col[(size_t)y];
    }
}

void irfft2(const std::vector<std::complex<f32>>& in, int h, int w, std::vector<f32>& out) {
    const int wh = w / 2 + 1;
    out.assign((size_t)h * w, 0.f);
    if (h <= 0 || w <= 0 || (int)in.size() < h * wh) return;

    std::vector<std::complex<f32>> work = in;
    std::vector<std::complex<f32>> col((size_t)h);
    for (int x = 0; x < wh; ++x) {
        for (int y = 0; y < h; ++y)
            col[(size_t)y] = work[(size_t)y * wh + x];
        fft1d(col, true, nullptr);
        for (int y = 0; y < h; ++y)
            work[(size_t)y * wh + x] = col[(size_t)y];
    }

    std::vector<std::complex<f32>> row((size_t)w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < wh; ++x)
            row[(size_t)x] = work[(size_t)y * wh + x];
        // Hermitian completion (numpy/torch irfft2)
        for (int x = wh; x < w; ++x) {
            int k = w - x;
            row[(size_t)x] = std::conj(work[(size_t)y * wh + k]);
        }
        fft1d(row, true, nullptr);
        for (int x = 0; x < w; ++x)
            out[(size_t)y * w + x] = row[(size_t)x].real();
    }
}

void fftshift2d_real(std::vector<f32>& data, int h, int w) {
    // torch.fft.fftshift: out[i] = in[(i + n//2) % n]
    if (h <= 0 || w <= 0) return;
    std::vector<f32> tmp = data;
    int shy = h / 2, shx = w / 2;
    for (int y = 0; y < h; ++y) {
        int sy = (y + shy) % h;
        for (int x = 0; x < w; ++x) {
            int sx = (x + shx) % w;
            data[(size_t)y * w + x] = tmp[(size_t)sy * w + sx];
        }
    }
}

// Alg. 3 FFT grey — matches utils_image.compute_grey_images(method="FFT")
// at native (h,w); no power-of-2 padding.
// On Apple: Metal GPU only (same math). No CPU fallback.
Image compute_grey_fft(const Image& raw) {
#ifdef __APPLE__
    Image grey = compute_grey_fft_metal(raw);
    if (grey.h > 0 && grey.w > 0) return grey;
    return Image();
#else
    const int h = raw.h, w = raw.w;
    if (h <= 0 || w <= 0) return Image();
    std::vector<std::complex<f32>> buf((size_t)h * w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            buf[(size_t)y * w + x] = {raw.at(y, x), 0.f};

    fft2d(buf, h, w, false);
    fftshift2d(buf, h, w);

    // Python: zero [:h//4,:], [:,:w//4], [-h//4:,:], [:,-w//4:]
    const int y0 = h / 4, x0 = w / 4;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (y < y0 || y >= h - y0 || x < x0 || x >= w - x0)
                buf[(size_t)y * w + x] = {0.f, 0.f};
        }
    }

    ifftshift2d(buf, h, w);
    fft2d(buf, h, w, true);

    Image grey(h, w, 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            grey.at(y, x) = buf[(size_t)y * w + x].real();
    return grey;
#endif
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
    parallel_rows(tmp_h, 0, [&](int y) {
        for (int x = 0; x < tmp_w; ++x) {
            f32 acc = 0.f;
            for (int i = 0; i < klen; ++i)
                acc += ker[i] * src.at(y + i, x);
            tmp.at(y, x) = acc;
        }
    });

    // Valid horizontal conv
    int filt_h = tmp_h;
    int filt_w = tmp_w - klen + 1;
    if (filt_w < 1) return Image(0, 0, 1);
    Image filtered(filt_h, filt_w, 1);
    parallel_rows(filt_h, 0, [&](int y) {
        for (int x = 0; x < filt_w; ++x) {
            f32 acc = 0.f;
            for (int j = 0; j < klen; ++j)
                acc += ker[j] * tmp.at(y, x + j);
            filtered.at(y, x) = acc;
        }
    });

    // h2, w2 = floor(shape / factor); return filtered[:, :, :h2*factor:factor, :w2*factor:factor]
    int h2 = filt_h / factor;
    int w2 = filt_w / factor;
    Image out(h2, w2, 1);
    parallel_rows(h2, 0, [&](int y) {
        for (int x = 0; x < w2; ++x)
            out.at(y, x) = filtered.at(y * factor, x * factor);
    });
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
        if (f == 1 && i == 0) {
            cur = grey;
        } else {
#ifdef __APPLE__
            // Same math as downsample_by / Python cuda_downsample.
            Image gpu_out;
            if (downsample_by_metal(cur, f, gpu_out) && gpu_out.h > 0)
                cur = std::move(gpu_out);
            else
                cur = downsample_by(cur, f);
#else
            cur = downsample_by(cur, f);
#endif
        }
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
