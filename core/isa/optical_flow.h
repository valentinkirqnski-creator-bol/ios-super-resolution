#pragma once
// CPU port of ImageStackAlignator's dense optical-flow refinement:
// Kernels/opticalFlow.cu (WarpingKernel, ComputeDerivativesKernel,
// lucasKanadeOptim, CreateFlowFieldFromTiles) + PEFStudioDX/OpticalFlow.cs's
// LucasKanade() driver loop.
//
// This is the per-pixel refinement pass that runs AFTER patch tracking
// (Stage 7, not yet built) produces a per-tile initial shift estimate;
// algorithmically it is self-contained, so it is built and verified here
// against a synthetic starting flow (including all-zero) per the plan.
#include <vector>
#include "accumulate.h"  // Vec2f

namespace isacpu {

// WarpingKernel: samples `source` (mirror-boundary, bilinear) at
// (x+0.5+flow.x, y+0.5+flow.y) for each pixel -- `flow` is read directly
// (same resolution as source/target throughout the LK loop, so the
// source's own Clamp+Point flow-texture sampling reduces to a direct index).
void warp_image(const std::vector<float>& source, int width, int height,
                const std::vector<Vec2f>& flow, std::vector<float>& out);

// ComputeDerivativesKernel: Ix/Iy are the average of `warped_source` and
// `target`'s 5-tap central-difference spatial derivatives (mirror
// boundary); Iz is warped_source - target (the temporal/data term).
void compute_derivatives_3way(const std::vector<float>& warped_source, const std::vector<float>& target,
                              int width, int height,
                              std::vector<float>& Ix, std::vector<float>& Iy, std::vector<float>& Iz);

// lucasKanadeOptim: per-pixel windowed 2x2 structure-matrix solve,
// SUBTRACTING the computed correction from `shifts` in place. The `-=`
// here is NOT a deviation: this port's derivative stencil computes the
// standard +dI/dx while the CUDA kernel's stencil computes its negation
// (opticalFlow.cu:116-120), so CUDA's `shift += UV` composed with its
// negated gradients equals this port's `shift -= UV` composed with
// conventional ones -- numerically identical end to end (see the comment
// at the subtraction site). Neither side may change without the other.
//
// Reproduced verbatim, bugs included, per the identical-to-master
// requirement: the source's `smin = fminf(sigma1, sigma1)` reject
// threshold (never inspects sigma2) and the ABSENCE of the per-iteration
// +-2 clamp (present in the source only as commented-out code).
void lucas_kanade_update(std::vector<Vec2f>& shifts, const std::vector<float>& Ix, const std::vector<float>& Iy,
                        const std::vector<float>& Iz, int width, int height, int half_window, float min_det);

// CreateFlowFieldFromTiles: upsamples a tileCountX x tileCountY tiled shift
// field to a dense width x height flow field (bilinear, clamp boundary),
// adding the rigid pre-alignment baseShift/baseRotation about the image
// center (matching the source's own sign conventions exactly).
void create_flow_field_from_tiles(const std::vector<Vec2f>& tiled_flow, int tileCountX, int tileCountY,
                                  int tileSize, int width, int height,
                                  Vec2f base_shift, float base_rotation, std::vector<Vec2f>& out);

// OpticalFlow.LucasKanade: the full iterative-warping driver. `tiled_flow`
// is the Stage-7 patch-tracking initial estimate (pass all-zero tiles with
// tileCountX=tileCountY=1 for a synthetic from-zero test).
void lucas_kanade_refine(const std::vector<float>& source, const std::vector<float>& target,
                         int width, int height, const std::vector<Vec2f>& tiled_flow,
                         int tileSize, int tileCountX, int tileCountY, int iterations,
                         Vec2f base_shift, float base_rotation, float min_det, int window_size,
                         std::vector<Vec2f>& out_flow);

}  // namespace isacpu
