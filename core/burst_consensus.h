#pragma once
// Burst consensus check on the motion vectors (Config::fb_consensus_enabled): a tile's
// shift in one comparison frame is trusted only if the OTHER frames of the burst agree
// with it.
//
// Every comparison frame k is aligned to the reference, giving one vector F_k(t) per
// tile t. The camera's own motion differs per frame, so it is removed first: a robust
// (IRLS, Huber) affine fit G_k over the frame's structured tiles; what is left,
// r_k(t) = F_k(t) - G_k(t), is the tile's LOCAL motion -- parallax, a moving object, or
// an alignment error. Real local motion is consistent across the burst: it follows a
// smooth path in time (the reference sits at r = 0 at its own time). So for every
// frame k and tile t, a robust straight line through the OTHER frames' residuals
// (r_j(t) at their times, j != k, plus the reference's 0) predicts where frame k's
// residual should be; the disagreement d = r_k(t) - prediction is that frame's error
// at that tile -- "do the other frames agree with this shift?".
//
// As in the forward-backward check, d is measured only ACROSS the tile's local edge
// (weighted by the reference tile's structure tensor / lambda1: a slide along a
// straight edge is invisible and unmeasurable), the error expected from noise alone is
// subtracted, flat tiles get 0, and the result becomes C = exp(-d^2 / (2 tau^2))
// multiplied into that frame's mask (fb_apply_confidence_raw). Uses only the vectors
// already computed by the alignment: no extra alignment, no extra images.
#include "types.h"
#include <vector>

namespace hhsr {

// Per tile of the (ny x nx, tile_size raw px) merge grid, from the reference grey:
// {wxx, wxy, wyy, noise_e2}: the structure tensor normalised by its largest eigenvalue
// (weights for the across-edge error) and the noise-only expected squared error (raw
// px^2); all 0 for flat tiles.
std::vector<float> consensus_tile_structure(const Image& ref_grey, int raw_h, int raw_w,
                                            int ny, int nx, int tile_size, const Config& cfg);

// flows[i]: frame i's tile vectors (ny*nx*2, raw px, the merge flow); times[i]: its
// position in the burst (frame index); ref_time: the reference's. Returns per frame the
// per-tile disagreement (raw px, weighted + noise-corrected), or empty vectors when
// fewer than 3 frames are available.
std::vector<std::vector<float>> burst_consensus_error(
    const std::vector<std::vector<float>>& flows, const std::vector<int>& times,
    int ref_time, int ny, int nx, int tile_size, const std::vector<float>& tile_struct,
    const Config& cfg);

}  // namespace hhsr
