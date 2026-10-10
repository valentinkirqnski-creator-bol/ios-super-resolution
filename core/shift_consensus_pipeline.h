#pragma once
//
// Pipeline glue for the ISA multi-frame shift-consistency consensus
// (core/shift_consensus.*). This measures the redundant pairwise per-tile
// tracks a burst needs, runs the consensus solve, and returns each frame's
// cleaned flow ready for the merge/robustness consumers -- an opt-in
// replacement for the plain per-frame frame->reference track
// (Config::shift_consensus_enabled).
//
#include "types.h"
#include <functional>
#include <vector>

namespace hhsr {

// Compute per-frame consensus flows for a burst.
//
//   ref_grey      : full-resolution FFT grey of the reference frame.
//   ref_index     : absolute index of the reference within the burst.
//   frame_count   : total frames in the burst.
//   grey_of_frame : returns frame f's full-resolution grey (same dims as
//                   ref_grey). grey_of_frame(ref_index) may hand back ref_grey.
//   cfg, tile_size: alignment config and block-match tile size.
//
// Returns a vector indexed by ABSOLUTE frame index (size frame_count). Each
// usable comparison frame's entry is a raw-grid FlowField with global_h set
// (comp_pos = global_h * (lr + flow)); the reference entry and any frame that
// could not be solved have ny == 0. Returns an EMPTY vector when disabled or
// when preconditions are not met (needs full-res grey), so the caller falls
// back to its normal per-frame align.
std::vector<FlowField> compute_consensus_flows(
    const Image& ref_grey, int ref_index, int frame_count,
    const std::function<Image(int)>& grey_of_frame,
    const Config& cfg, int tile_size);

} // namespace hhsr
