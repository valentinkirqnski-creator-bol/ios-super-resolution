#pragma once
// Coarse gradient-overlay frame gate (Config::frame_drop_enabled).
//
// At the COARSEST pyramid level only (the alignment's own pyramid, 32x below the grey):
//   1. the reference and the comparison frame are turned into gradient images;
//   2. the comparison is aligned to the reference with the pipeline's own block matching
//      (block_match_coarse: the coarsest level's L2 search, here with radius +-9 by default);
//   3. the aligned comparison gradient image is laid over the reference's, tile by tile.
// If every structured tile coincides -- no edge left shifted or rotated -- the frame keeps
// its place in the merge. If edges still do not line up after the alignment, the frame holds
// motion that per-tile translation cannot explain (rotation or parallax) and is dropped.
//
// "Coincide" per tile = normalized gradient correlation sum(gr . gc) / sqrt(sum|gr|^2 sum|gc|^2)
// (1 = identical edges; exposure-invariant) of at least tile_min_score. Tiles without real
// structure (gradient energy under 9x the reference's flattest tiles) are not judged.

#include "types.h"
#include <vector>

namespace hhsr {

struct FrameGateResult {
    int judged_tiles = 0;     // tiles with structure
    int bad_tiles = 0;        // structured tiles whose edges do not line up after alignment
    f32 bad_fraction = 0.f;   // bad / judged
    f32 worst_score = 1.f;    // lowest tile correlation among judged tiles
    f32 median_score = 1.f;
    int tiles_x = 0, tiles_y = 0;      // tile grid
    std::vector<int> bad_tile_ids;     // ty * tiles_x + tx of each bad tile
};

// ref_coarse / comp_coarse: the coarsest pyramid level of each frame's alignment grey (same
// size). tile_size: the coarsest level's block-matching tile (grey px of that level).
FrameGateResult frame_gate_coarse(const Image& ref_coarse, const Image& comp_coarse,
                                  int tile_size, int search_radius, f32 tile_min_score,
                                  const Config& cfg);

} // namespace hhsr
