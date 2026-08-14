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
// out_mtl_buffer, when non-null, is an id<MTLBuffer> of at least h*w floats
// that the graph writes into directly. The caller usually needs the result in a
// Metal buffer anyway (align reuses it as the pinned moving grey), so supplying
// it here avoids a second full-frame staging buffer and one memcpy per frame.
bool mps_grey_lowpass(const float* in, float* out, int h, int w,
                      void* out_mtl_buffer = nullptr);

// Build and cache the graph for these dimensions ahead of time.
//
// MPSGraph compiles on first use, which measured ~1100ms for a 12MP frame and
// landed entirely on the reference frame's grey. Calling this before the burst
// (ideally when the camera configures, since it only needs the sensor
// dimensions) moves that off the shutter path. Safe to call repeatedly and from
// any thread; a no-op once the plan for these dimensions exists.
void mps_fft_prewarm(int h, int w);

// Release everything: the compiled plan, its tensors, and the pooled buffers.
//
// The grey FFT only runs during frame analysis, but peak footprint is reached
// later at merge:band, so anything MPSGraph holds is dead weight exactly where
// headroom is scarcest -- and a compiled FFT plan retains considerably more
// than the staging buffers alone (twiddle tables and intermediate tensors for
// both transform axes).
//
// Pair with mps_fft_prewarm once the burst is finished and memory is low, so
// the next shot still finds a warm plan. Rebuilding costs ~1100ms but happens
// off the shutter path.
void mps_fft_release_all();

// Drop only the staging buffers, keeping the compiled graph.
//
// The buffers are the memory -- two Shared allocations of h*w floats, ~96MB
// at 12MP and ~384MB at 48MP -- and they are what has to go before the merge
// peak. The compiled graph is small by comparison but costs ~1100ms at 12MP
// to rebuild, and releasing it meant the next burst paid that compile on its
// reference frame whenever the background prewarm had not finished. Same
// memory recovered, without the recompile.
void mps_fft_release_buffers();

}  // namespace hhsr
