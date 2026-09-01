#pragma once
// CPU port of ImageStackAlignator's global shift consistency solve:
// PEFStudioDX/ShiftCollection.cs's MinimizeCUBLAS + the small kernels in
// Kernels/ShiftMinimizerKernels.cu (checkForOutliers, getOptimalShifts).
//
// ISA solves this per-tile, batched across all tiles via cuBLAS (GemmBatched
// + MatinvBatched/Getrf+Getri). Each per-tile system is tiny (n1 = frameCount
// - 1 unknowns, e.g. 7 for an 8-frame burst) -- exactly the kind of problem
// the project plan flagged as "very likely replaceable with a small per-tile
// direct solve, no CUBLAS needed". This port does that: a plain Gauss-Jordan
// solve per tile, looped in ordinary CPU code instead of batched on a GPU.
//
// TrackingStrategy.Full (all-pairs, ISA's default) and OnlyOnReference are
// both implemented below; the drivers use Full. (An earlier revision of
// this comment claimed only Full existed.)
#include <vector>
#include "accumulate.h"  // Vec2f

namespace isacpu {

struct ShiftPair { int reference, to_track; };

// Every (reference < toTrack) pair, ordered to match shift_matrix_full's
// rows directly (grouped by frame distance, then start frame -- ISA's own
// ShiftCollection keeps two different orderings, FillShiftPairs
// (reference-grouped, only used to decide which pairs to run Track() on)
// and FillIndexTable (distance-grouped, the actual storage/design-matrix
// order) reconciled via an index table; this port collapses that into one
// canonical ordering used everywhere, which is mathematically equivalent
// and removes the need for a separate lookup table).
std::vector<ShiftPair> shift_pairs_full(int frame_count);

// CreateShiftMatrix (Full strategy): m x n1 design matrix, row-major, m =
// shift_pairs_full(frame_count).size(), n1 = frame_count-1. Row for pair
// (ref,track) has 1s in columns [ref, track) -- the measured pairwise shift
// equals the sum of the intervening frame-to-frame increments.
std::vector<float> shift_matrix_full(int frame_count);

// TrackingStrategy.OnlyOnReference: only (referenceIndex, f) pairs -- n1
// measurements for n1 unknowns (exactly determined, no redundancy for
// outlier rejection), vs. Full's C(frameCount,2) overdetermined system.
// Real ISA strategy (not a shortcut invented here), much cheaper since
// patch tracking cost scales with pair count: O(frameCount) instead of
// O(frameCount^2). Used as this port's default given the priority on
// wall-clock speed over Full's extra redundancy.
std::vector<ShiftPair> shift_pairs_only_reference(int frame_count, int reference_index);
std::vector<float> shift_matrix_only_reference(int frame_count, int reference_index);

// MinimizeCUBLAS, for ONE tile: `measured` holds one Vec2f per shift pair
// (same order as shift_pairs_full), `design` is shift_matrix_full's output.
// Returns the n1 solved frame-to-frame incremental shifts (x in the least-
// squares solve), with up to 10 rounds of single-worst-outlier rejection
// exactly matching checkForOutliers (residual threshold: squared distance
// > 1.0 pixel).
std::vector<Vec2f> minimize_shifts_for_tile(const std::vector<Vec2f>& measured, const std::vector<float>& design,
                                           int frame_count);

// getOptimalShifts: partial cumulative sum of the per-tile incremental
// shifts between referenceIndex and imageToTrack (matching the source's own
// sign convention: sums forward if reference < track, subtracts if
// reference > track).
Vec2f optimal_shift_from_increments(const std::vector<Vec2f>& increments, int reference_index, int image_to_track);

}  // namespace isacpu
