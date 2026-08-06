#pragma once
//
// MPSGraph-backed replacement for the grey low-pass FFT.
//
// The hand-written Stockham path performs four full-size passes over a 97MB
// complex buffer: forward rows, forward columns, inverse rows, inverse columns.
// The input to that transform is real and the output is real, so half of every
// row pass and half of every column pass is spent transforming zeros.
//
// MPSGraph exposes real-to-Hermitean and Hermitean-to-real transforms that
// exploit exactly that, in an implementation Apple tunes per GPU. The spectrum
// is stored as h x (w/2+1) complex instead of h x w, which also roughly halves
// the working set.
//
// NOT bit-identical to the Stockham path: a different radix decomposition and a
// different summation order give results differing by a few ULP. Note the
// existing Metal path is already not bit-identical to the CPU reference -- see
// the comment on block_match_level_L2 in align.cpp -- so this is a change in
// degree rather than in kind. It still wants validating with tools/compare_dng.py
// against a reference DNG rather than by eye.
//
// Availability: the FFT operations on MPSGraph are iOS 16+, while this project
// deploys to iOS 15, so every entry point is guarded and falls back.
#include "types.h"

namespace hhsr {

// True when MPSGraph FFT can be used: OS is new enough, the device supports it,
// and HHSR_MPSGRAPH_FFT is not set to 0.
bool mps_fft_enabled();

// Ideal low-pass matching compute_grey_fft_metal: forward 2D FFT, keep only the
// four corner blocks of the spectrum (the low frequencies), inverse 2D FFT.
// Input and output are both h*w real, row-major.
//
// Returns false if unavailable or if anything fails, in which case the caller
// must fall back to the Stockham path. Never partially writes `out` on failure.
bool mps_grey_lowpass(const float* in, float* out, int h, int w);

}  // namespace hhsr
