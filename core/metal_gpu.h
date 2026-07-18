#pragma once
//
// Metal GPU backend for grey-FFT and L2 block-matching.
// Same mathematical pipeline as the CPU path (row/col 1D DFT via radix-2 +
// Bluestein, Torch-style rfft2/irfft2 L2). No CPU/vDSP fallback on Apple:
// if Metal cannot run, these entry points return failure / empty outputs.
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
