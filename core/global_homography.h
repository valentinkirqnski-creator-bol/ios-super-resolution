#pragma once
// Global homography warp-then-refine for large camera roll / global perspective.
// Fresh implementation (not the existing global_prealignment code): estimates one
// 3x3 homography between the reference and comparison GREY images, used to warp
// the comparison into the reference frame before align(). See Config::
// global_homography_enabled and the FlowField::global_h composition in merge /
// robustness.
#include "types.h"

namespace hhsr {

// Estimate H (row-major 3x3) mapping a REFERENCE-grey pixel to the corresponding
// COMPARISON-grey pixel, i.e. warped(p) = comp(H*p) aligns comp onto ref. Returns
// identity if estimation is unreliable (degenerate / worse than identity). Both
// greys must be single channel and the same size. Coarse rotation+shift seed then
// Lucas-Kanade homography refinement, coarse-to-fine.
void estimate_global_homography(const Image& ref_grey, const Image& comp_grey,
                                const Config& cfg, f32 H_out[9]);

// Lucas-Kanade refine of an EXISTING homography seed (row-major 3x3, full grey
// pixels) into a full 8-DOF homography -- used to upgrade the ISA pre-align's
// robust rigid (rotation+translation) estimate to one that also absorbs global
// scale, shear and perspective ("any camera motion"). H_inout is refined in
// place; the result is kept only if it lowers the warp error vs the seed and is
// non-degenerate, so it never makes the seed worse. Needs host grey pixels.
void refine_global_homography_seed(const Image& ref_grey, const Image& comp_grey,
                                   const Config& cfg, f32 H_inout[9]);

// warped(y,x) = comp_grey(H*(x,y)); out-of-bounds -> 0. Single channel.
Image warp_grey_by_homography(const Image& comp_grey, const f32 H[9]);

// Normalized cross-correlation (Pearson, in [-1,1]) between the reference grey
// and a pre-aligned (warped) comparison grey, over their valid overlap. A frame
// the global pre-align could register scores near 1; one it could not (extreme
// motion, motion blur, scene change, tiny overlap) scores near 0, and
// essentially no overlap returns -1. Used as a frame-level gate to reject
// unfittable frames before merge. Returns 1 when it cannot be assessed (dims
// mismatch / empty), so the caller does not reject on a non-measurement.
f32 prealign_fit_ncc(const Image& ref_grey, const Image& warped_comp_grey);

} // namespace hhsr
