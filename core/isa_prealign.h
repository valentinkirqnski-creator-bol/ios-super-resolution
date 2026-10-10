#pragma once
// Fresh, self-contained reimplementation of ImageStackAlignator's global
// pre-alignment (Michael Kunz, github.com/kunzmi/ImageStackAlignator:
// ComputePreAlignment -> PreAlignment.ScanAngles), MINUS the gyro roll seed.
//
// ISA seeds its rotation search with the camera's measured roll
// (rollTrack - rollReference from the Pentax maker-notes). iOS bursts carry no
// per-frame attitude, so the search here is centred on 0 instead -- the only
// intended deviation from ISA. Everything else mirrors ScanAngles:
//
//   For each candidate rotation angle:
//     - rotate the comparison grey about the image centre,
//     - forward 2D FFT,
//     - multiply by the CONJUGATE reference spectrum (cross power spectrum),
//     - inverse 2D FFT -> a circular cross-correlation map,
//     - take the peak: its WRAPPED location is the translation, its height
//       scores the angle.
//   A coarse pass (step 5*incr over +/-range), then a fine pass (step incr over
//   +/-10*incr around the best angle). Returns (-peakX, -peakY, bestAngle).
//
// This file does NOT read or reuse the port's existing global_prealignment /
// global_homography code; it carries its own FFT, rotation warp and peak search.
#include "types.h"

namespace hhsr {

// Estimate a global (translation + in-plane rotation) between a reference and a
// comparison GREY image (single channel, same size). On success returns true and
// fills, in the make_global_initial_flow convention (grey pixels, rotation about
// the grey image centre):
//   comp( R(rot) * (p - c) + c + (dx, dy) ) ~= ref(p)
// so (out_dx, out_dy, out_rot_rad) can be passed straight to align() as the
// initial seed. Returns false when the correlation peak is unreliable.
//
// The FFT runs on an isotropically downscaled, zero-padded power-of-two copy of
// the greys (cap: Config::isa_prealign_fft_max_dim); the recovered shift is
// scaled back to grey pixels. Rotation is scale-invariant.
bool estimate_isa_prealign(const Image& ref_grey, const Image& comp_grey,
                           const Config& cfg,
                           float& out_dx, float& out_dy, float& out_rot_rad);

}  // namespace hhsr
