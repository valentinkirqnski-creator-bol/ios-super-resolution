#include "fft.h"

#include <cmath>

namespace isacpu {

namespace {

constexpr float kPi = 3.14159265358979323846f;

int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Iterative radix-2 Cooley-Tukey, in place. `a.size()` must be a power of 2.
// Unnormalized in both directions (matches cuFFT's convention).
void fft_pow2(std::vector<Cplx>& a, bool inverse) {
    size_t n = a.size();
    if (n <= 1) return;

    // Bit-reversal permutation.
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        float ang = (inverse ? 2.f : -2.f) * kPi / (float)len;
        Cplx wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            Cplx w(1.f, 0.f);
            for (size_t k = 0; k < len / 2; ++k) {
                Cplx u = a[i + k];
                Cplx v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Bluestein's algorithm: arbitrary-size FFT built on the radix-2 core.
// Unnormalized (same convention as fft_pow2).
void fft_bluestein(std::vector<Cplx>& a, bool inverse) {
    size_t n = a.size();
    int m = next_pow2((int)(2 * n - 1));
    float sign = inverse ? 1.f : -1.f;

    // Chirp: w[k] = exp(sign * i * pi * k^2 / n).
    std::vector<Cplx> w(n);
    for (size_t k = 0; k < n; ++k) {
        // k*k can overflow a 32-bit int for large n; reduce mod 2n first
        // since the phase only depends on (k*k) mod (2n).
        size_t kk = (k * k) % (2 * n);
        float ang = sign * kPi * (float)kk / (float)n;
        w[k] = Cplx(std::cos(ang), std::sin(ang));
    }

    std::vector<Cplx> A(m, Cplx(0.f, 0.f)), B(m, Cplx(0.f, 0.f));
    for (size_t k = 0; k < n; ++k) A[k] = a[k] * w[k];
    B[0] = std::conj(w[0]);
    for (size_t k = 1; k < n; ++k) {
        Cplx wk = std::conj(w[k]);
        B[k] = wk;
        B[m - k] = wk;
    }

    fft_pow2(A, false);
    fft_pow2(B, false);
    for (int i = 0; i < m; ++i) A[i] *= B[i];
    fft_pow2(A, true);  // unnormalized inverse: divide by m ourselves below

    float invm = 1.f / (float)m;
    for (size_t k = 0; k < n; ++k) a[k] = w[k] * (A[k] * invm);
}

}  // namespace

void fft1d(std::vector<Cplx>& a, bool inverse) {
    size_t n = a.size();
    if (n <= 1) return;
    if ((n & (n - 1)) == 0) {
        fft_pow2(a, inverse);
    } else {
        fft_bluestein(a, inverse);
    }
}

void fft2d(std::vector<Cplx>& img, int width, int height, bool inverse) {
    std::vector<Cplx> row(width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) row[x] = img[(size_t)y * width + x];
        fft1d(row, inverse);
        for (int x = 0; x < width; ++x) img[(size_t)y * width + x] = row[x];
    }
    std::vector<Cplx> col(height);
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) col[y] = img[(size_t)y * width + x];
        fft1d(col, inverse);
        for (int y = 0; y < height; ++y) img[(size_t)y * width + x] = col[y];
    }
}

std::vector<Cplx> rfft2d(const std::vector<float>& img, int width, int height) {
    std::vector<Cplx> spectrum((size_t)width * height);
    for (size_t i = 0; i < spectrum.size(); ++i) spectrum[i] = Cplx(img[i], 0.f);
    fft2d(spectrum, width, height, false);
    return spectrum;
}

std::vector<float> irfft2d(const std::vector<Cplx>& spectrum, int width, int height) {
    std::vector<Cplx> tmp = spectrum;
    fft2d(tmp, width, height, true);
    std::vector<float> out((size_t)width * height);
    for (size_t i = 0; i < out.size(); ++i) out[i] = tmp[i].real();
    return out;
}

}  // namespace isacpu
