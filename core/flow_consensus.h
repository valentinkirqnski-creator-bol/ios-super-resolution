#pragma once
//
// Multi-frame trajectory consensus for the per-tile displacement field
// (Cheng et al., "Electron counting and beam-induced motion correction...",
// doi:10.1038/nmeth.2472 -- the cryo-EM patch-trajectory idea, adapted).
//
// The reference-based aligner measures each comparison frame's shift to the
// reference INDEPENDENTLY per tile. That throws away a redundancy: the shift
// ref->f, measured directly, must also equal the chained shift ref->(f-1) plus
// the adjacent-frame shift (f-1)->f. Measuring BOTH the direct shifts and the
// adjacent-frame ("chain") shifts over-determines each tile's trajectory, and a
// robust least-squares solve over that redundancy pins the trajectory while
// down-weighting any single measurement that contradicts it -- i.e. a block-
// match BLUNDER (a repeated-pattern mis-lock, an occlusion), without touching
// genuine parallax (which is consistent across the direct and chained paths and
// so survives the solve unchanged).
//
// This is the key difference from a plain multi-frame consensus (which rejects a
// shift merely for disagreeing with its neighbours, and so discards real
// parallax): here rejection is on COMPOSITION inconsistency, not disagreement.
//
// Purely CPU, self-contained, and consumed on the host FlowField before it
// reaches robustness/merge -- so both the CPU and the Metal paths pick up the
// correction with no kernel change. Off by default (Config::flow_consensus_*).
//
#include "types.h"
#include <vector>
#include <cstdint>

namespace hhsr {

// One coarse per-tile shift measurement plus a per-tile validity flag.
struct CoarseFlow {
    FlowField flow;              // ny*nx, displacement that samples B at A_pos+flow
    std::vector<uint8_t> valid;  // ny*nx, 1 = confident measurement, 0 = ambiguous/flat
};

// Per-tile coarse block match: the displacement that maps `a_grey` onto
// `b_grey` (b sampled at a_pos + flow), on a tile_size grid, searched over
// +/- radius grey px around (seed_x, seed_y) with integer steps and a parabolic
// sub-pixel refine. A tile is marked invalid when its reference window is near
// flat (no texture to match) or the match is ambiguous (the best and the next
// local minimum are within a small margin). Grid is floor(h/tile)x floor(w/tile).
CoarseFlow coarse_block_match(const Image& a_grey, const Image& b_grey,
                              int tile_size, int radius,
                              f32 seed_x, f32 seed_y, int num_threads);

struct TrajectoryConsensus {
    std::vector<FlowField> flow;            // per frame: consensus reference->f
    std::vector<std::vector<uint8_t>> trusted; // per frame, ny*nx: 1 = redundant
};

// Robust trajectory consensus. `direct[f]` is the measured reference->f
// displacement for every frame f in [0,n) (the entry for f==ref_index is
// ignored and may be empty). `chain[f]` is the measured f->(f+1) displacement
// for f in [0,n-1) (entry n-1 unused). All share one ny*nx grid. Returns, per
// frame, the consensus reference->f displacement per tile (ref_index all-zero),
// plus a `trusted` mask that is 1 only where that frame's tile is backed by at
// least two valid measurements (so the trajectory could actually be checked --
// a lone direct measurement is NOT trusted, since there is nothing to out-vote
// a blunder with). robust_c is the Cauchy scale in grey px; `iters` IRLS passes.
TrajectoryConsensus solve_trajectory_consensus(
    const std::vector<CoarseFlow>& direct,
    const std::vector<CoarseFlow>& chain,
    int ref_index, f32 robust_c, int iters, int num_threads);

// Replace blunder tiles in `align_flow` with the consensus where (a) the tile is
// trusted (redundantly measured) and (b) the two disagree by more than reject_px
// (grey px, Euclidean). Clean tiles, and tiles the pre-pass could not verify, are
// left exactly as the refined aligner produced them -- so sub-pixel detail (the
// super-resolution signal) is preserved and an unreliable measurement can never
// corrupt a good tile. `align_flow`, `consensus` and `trusted` must share the
// grid. Returns the number of tiles replaced.
int apply_consensus_correction(FlowField& align_flow, const FlowField& consensus,
                               const std::vector<uint8_t>& trusted, f32 reject_px);

} // namespace hhsr
