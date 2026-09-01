#pragma once
// Self-contained arbitrary-size complex FFT for ISA-CPU's FFT-based
// pre-alignment (phase/cross-correlation, Fourier high-pass filtering).
// ISA itself uses cuFFT's real-to-complex plans; this port uses a plain
// full complex-to-complex FFT throughout instead (simpler to implement
// correctly, and correctness/clarity matter more than the 2x memory a
// packed R2C spectrum would save for a CPU tool at these image sizes).
//
// Arbitrary N is handled via Bluestein's algorithm (chirp z-transform) on
// top of an iterative radix-2 Cooley-Tukey core, so image dimensions do
// not need to be powers of two.
#include <vector>
#include <complex>

namespace isacpu {

using Cplx = std::complex<float>;

// In-place 1D FFT/IFFT of arbitrary size (any N >= 1). Matches cuFFT's own
// convention: NEITHER direction is normalized (forward then inverse scales
// the signal by N) -- callers divide by width*height themselves, exactly
// mirroring ISA's own explicit `.Div(width*height)` after every inverse-FFT
// call, so that convention lines up 1:1 with the C# orchestration this
// will be ported alongside.
void fft1d(std::vector<Cplx>& a, bool inverse);

// 2D FFT/IFFT (row pass then column pass) of a width x height complex
// image, in place, row-major. Unnormalized in both directions, same as
// fft1d above.
void fft2d(std::vector<Cplx>& img, int width, int height, bool inverse);

// Convenience: real image -> complex spectrum (forward FFT2D, unnormalized).
std::vector<Cplx> rfft2d(const std::vector<float>& img, int width, int height);

// Convenience: complex spectrum -> real image (inverse FFT2D, discarding
// the (should-be-negligible) imaginary part -- matching cuFFT's C2R, which
// assumes Hermitian-symmetric input and returns a plain real image
// directly). Unnormalized; callers divide by width*height themselves.
std::vector<float> irfft2d(const std::vector<Cplx>& spectrum, int width, int height);

}  // namespace isacpu
