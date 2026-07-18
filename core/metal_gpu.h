#pragma once
//
// Metal GPU backend for grey-FFT and L2 block-matching.
// FFT matches grey_pyramid.cpp (fft1d_pow2_inplace_ref + Bluestein, same
// twiddle recurrence and scaling). L2 matches Torch rfft2/irfft2 path.
// No CPU fallback on Apple: failure returns empty / false.
//
#include "types.h"
#include <complex>
#include <vector>

namespace hhsr {

// Returns false if MTL device / pipelines could not be created.
bool metal_gpu_init();

// Alg. 3 FFT grey on GPU. Empty image on failure.
Image compute_grey_fft_metal(const Image& raw);

// L2 block-match one pyramid level on GPU (updates flow in place).
// Returns false on failure (caller must not fall back to CPU).
bool block_match_level_L2_metal(const Image& ref, const Image& moving,
                                int tile_size, int search_radius,
                                FlowField& flow);

} // namespace hhsr
