#pragma once
// CPU port of ImageStackAlignator's per-tile patch tracker:
// PEFStudioDX/PatchTracker.cs's Track() plus the kernel.cu tile/NCC/
// sub-pixel-refine kernels it drives (convertToTilesOverlapBorder,
// convertToTilesOverlapPreShift, squaredSum, boxFilterWithBorderX/Y,
// normalizedCC, findMinimum). Despite the name, normalizedCC actually
// produces a per-shift-candidate SSD (sum-of-squared-differences) surface
// via the polynomial expansion ||a-b||^2 = ||a||^2+||b||^2-2·a·b, computing
// the cross term via FFT phase correlation -- findMinimum then finds its
// MINIMUM (best match), not a maximum.
#include <vector>
#include "accumulate.h"  // Vec2f

namespace isacpu {

// Converts `img` into tileCountX*tileCountY consecutive tiles, each
// (tileSize+2*maxShift) square, applying only the rigid baseShift/
// baseRotation about the image center (matching convertToTilesOverlapBorder
// -- used for the REFERENCE/template image, which has no per-tile
// pre-shift).
std::vector<float> convert_to_tiles_overlap_border(const std::vector<float>& img, int width, int height,
                                                   int tileSize, int maxShift, int tileCountX, int tileCountY,
                                                   Vec2f base_shift, float base_rotation);

// Same, but also applies a per-tile pre-shift (matching
// convertToTilesOverlapPreShift -- used for the TRACKED image, seeded by
// the previous pyramid level's result via preShift).
std::vector<float> convert_to_tiles_overlap_pre_shift(const std::vector<float>& img, int width, int height,
                                                      const std::vector<Vec2f>& pre_shift,
                                                      int tileSize, int maxShift, int tileCountX, int tileCountY,
                                                      Vec2f base_shift, float base_rotation);

// squaredSum: sum of squares over each tile's CENTER tileSize x tileSize
// region (excluding the maxShift border), one value per tile.
std::vector<float> squared_sum(const std::vector<float>& tiles, int maxShift, int tileSize, int tileCount);

// boxFilterWithBorderX/Y (separable): boxFilterX squares while summing
// along X; boxFilterY sums (no further squaring) along Y. Applied X-then-Y
// to the TRACKED tiles, this gives the local sum-of-squares of the tracked
// image over a tileSize x tileSize window centered at every candidate
// shift position within the border.
std::vector<float> box_filter_with_border_x(const std::vector<float>& tiles, int maxShift, int tileSize, int tileCount);
std::vector<float> box_filter_with_border_y(const std::vector<float>& tiles, int maxShift, int tileSize, int tileCount);

// normalizedCC: shiftImage[shift] = squaredTemplate[tile] + boxFiltered[shift] - 2*ccImage[shift]
// (an SSD surface, per tile, over all (2*maxShift+1)^2 integer shift
// candidates). `cc_image` is the UNNORMALIZED-then-divided real cross-
// correlation (IFFT(conj(FFT(ref_tiles))*FFT(track_tiles)) / blockSize^2).
std::vector<float> normalized_cc(const std::vector<float>& cc_image, const std::vector<float>& squared_template,
                                 const std::vector<float>& box_filtered_image, int maxShift, int tileSize, int tileCount);

// findMinimum: locates the SSD surface's minimum per tile and refines it to
// sub-pixel precision via a fixed 3x3 quadratic fit (matches the source's
// FA11/FA22/FA12/Fb1/Fb2 stencils exactly). Returns {0,0} for a tile whose
// minimum sits on the search border (can't fit a 3x3 neighborhood) or whose
// peak isn't at least `threshold` below the surface's own maximum.
std::vector<Vec2f> find_minimum(const std::vector<float>& shift_image, int maxShift, int tileCountX, int tileCountY,
                                float threshold);

// PatchTracker.Track: the full per-level driver. `pre_shift` is
// tileCountX*tileCountY, updated IN PLACE (this frame pair's found
// correction is ADDED onto it, matching the source's own
// `preShiftFloat.Add(patchShiftFloat)`).
void track(const std::vector<float>& img_to_track, const std::vector<float>& img_ref, int width, int height,
          std::vector<Vec2f>& pre_shift, int tileSize, int maxShift, int tileCountX, int tileCountY,
          Vec2f base_shift_ref, float base_rotation_ref, Vec2f base_shift_to_track, float base_rotation_to_track,
          float threshold);

}  // namespace isacpu
