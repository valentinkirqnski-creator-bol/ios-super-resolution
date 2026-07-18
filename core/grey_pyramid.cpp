#include "stages.h"
#include "parallel.h"
#include <complex>
#include <cmath>
#include <vector>
#include <unordered_map>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
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

static unsigned log2_pow2(int n) {
    unsigned l = 0;
    while ((1u << l) < (unsigned)n) ++l;
    return l;
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

#ifdef __APPLE__
// Cached vDSP radix-2 setups (thread-local; execute is thread-safe per setup use).
static FFTSetup vdsp_fftsetup(unsigned log2n) {
    static thread_local FFTSetup setups[32] = {};
    if (log2n >= 32) return nullptr;
    if (!setups[log2n])
        setups[log2n] = vDSP_create_fftsetup(log2n, kFFTRadix2);
    return setups[log2n];
}

static void fft1d_pow2_inplace(std::vector<std::complex<f32>>& a, bool inverse) {
    const int n = (int)a.size();
    if (n <= 1) return;
    unsigned log2n = log2_pow2(n);
    FFTSetup setup = vdsp_fftsetup(log2n);
    if (!setup) {
        fft1d_pow2_inplace_ref(a, inverse);
        return;
    }
    // Split-complex scratch (thread-local, grow as needed).
    static thread_local std::vector<f32> re, im;
    if (re.size() < (size_t)n) {
        re.resize((size_t)n);
        im.resize((size_t)n);
    }
    for (int i = 0; i < n; ++i) {
        re[(size_t)i] = a[(size_t)i].real();
        im[(size_t)i] = a[(size_t)i].imag();
    }
    DSPSplitComplex split{re.data(), im.data()};
    vDSP_fft_zip(setup, &split, 1, log2n, inverse ? FFT_INVERSE : FFT_FORWARD);
    for (int i = 0; i < n; ++i)
        a[(size_t)i] = {re[(size_t)i], im[(size_t)i]};
}

// vDSP DFT for lengths f*2^n (f in {1,3,5,15}, n>=3). Returns false if unsupported.
static bool fft1d_vdsp_dft(std::vector<std::complex<f32>>& a, bool inverse) {
    const int n = (int)a.size();
    if (n <= 1) return true;

    struct Key {
        int n;
        bool inv;
        bool operator==(const Key& o) const { return n == o.n && inv == o.inv; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return (size_t)k.n * 2u + (k.inv ? 1u : 0u);
        }
    };
    static thread_local std::unordered_map<Key, vDSP_DFT_Setup, KeyHash> cache;

    Key key{n, inverse};
    vDSP_DFT_Setup setup = nullptr;
    auto it = cache.find(key);
    if (it != cache.end()) {
        setup = it->second;
    } else {
        setup = vDSP_DFT_zop_CreateSetup(
            nullptr, (vDSP_Length)n,
            inverse ? vDSP_DFT_INVERSE : vDSP_DFT_FORWARD);
        if (!setup) return false;
        cache.emplace(key, setup);
    }

    static thread_local std::vector<f32> re, im;
    if (re.size() < (size_t)n) {
        re.resize((size_t)n);
        im.resize((size_t)n);
    }
    for (int i = 0; i < n; ++i) {
        re[(size_t)i] = a[(size_t)i].real();
        im[(size_t)i] = a[(size_t)i].imag();
    }
    vDSP_DFT_Execute(setup, re.data(), im.data(), re.data(), im.data());
    for (int i = 0; i < n; ++i)
        a[(size_t)i] = {re[(size_t)i], im[(size_t)i]};
    return true;
}
#else
static void fft1d_pow2_inplace(std::vector<std::complex<f32>>& a, bool inverse) {
    fft1d_pow2_inplace_ref(a, inverse);
}
#endif

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
#ifdef __APPLE__
    // Prefer Accelerate DFT when length is supported (f*2^n).
    if (fft1d_vdsp_dft(a, inverse)) {
        if (inverse) for (auto& v : a) v /= (f32)n;
        return;
    }
#endif
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

// Alg. 3 FFT grey — matches utils_image.compute_grey_images(method="FFT")
// at native (h,w); no power-of-2 padding.
Image compute_grey_fft(const Image& raw) {
    const int h = raw.h, w = raw.w;
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
