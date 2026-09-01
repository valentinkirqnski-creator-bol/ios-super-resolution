#pragma once
// CPU port of ImageStackAlignator's robustness/certainty mask:
// Kernels/RobustnessModell.cu:ComputeRobustnessMask.
#include <vector>
#include "debayer.h"
#include "accumulate.h"

namespace isacpu {

// `ref_half`/`moved_half` are half-resolution box-subsampled debayer images
// (debayer_subsample3 output), both width x height. `shift_half` is the
// optical-flow displacement in RAW-resolution pixel units (matching the
// source's own `shiftf`, which it internally halves via `shift.x =
// round(shiftf.x*0.5)` before indexing the half-res `moved_half`), sampled
// at the SAME half-resolution grid as `ref_half`/`moved_half`.
//
// NOTE: the real source samples the flow field through a texture at
// arbitrary resolution via normalized UV coordinates (`of.LastFlow`, whose
// true pixel dimensions are set by the optical-flow stage, not yet ported
// -- see Stage 5-7). This port takes `shift_half` pre-resampled to the
// half-res grid; when the real flow field is wired in, resample it onto
// this grid the same way UpSampleShifts does before calling this function.
//
// Only interior pixels (1 <= x < width-1, 1 <= y < height-1) are written,
// matching the source's own border skip -- `out` is not otherwise touched,
// so the caller decides the border's starting value (ISA resets the whole
// mask to {1,1,1,1} once per accumulation run, before any frame's mask is
// computed over it).
void compute_robustness_mask(const std::vector<Rgbf>& ref_half, const std::vector<Rgbf>& moved_half,
                             const std::vector<Vec2f>& shift_half, int width, int height,
                             float alpha, float beta, float threshold_m,
                             std::vector<Vec4f>& out);

}  // namespace isacpu
