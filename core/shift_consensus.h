#pragma once
//
// Multi-frame shift-consistency outlier rejection -- a faithful CPU port of
// ImageStackAlignator's ShiftCollection (PEFStudioDX/ShiftCollection.cs) and
// the ShiftMinimizerKernels (Kernels/ShiftMinimizerKernels.cu) by Michael
// Kunz (github.com/kunzmi/ImageStackAlignator, LGPL-3.0).
//
// The idea (README section d, citing Cheng et al. cryo-EM motion correction):
// instead of measuring only each comparison frame's shift to the reference
// (N-1 measurements, no redundancy), measure MANY frame-pair shifts. Every
// measured pair (i->j) must equal the sum of the unknown sequential one-to-one
// shifts it spans:  shift(i->j) = x_i + x_{i+1} + ... + x_{j-1}. Stacking all m
// measurements gives an over-determined linear system  A x = b  per tile, with
// A an m x (N-1) 0/1 incidence matrix. Solving it by least squares and
// iteratively discarding the single worst-residual measurement (ISA: up to 10
// passes, reject when |measured - A x|^2 > 1 px^2) rejects mis-tracks and
// yields an outlier-free consensus displacement field.
//
// This module is the shift-vector math ONLY -- it does not touch images. The
// caller measures the pairwise per-tile shifts (by block-matching frame j
// against frame i, exactly as our align() already does for frame->reference)
// and hands the measurements here; the result is, per non-reference frame, a
// cleaned per-tile shift to the reference. It is self-contained (depends only
// on <vector>/<cstdint> + f32) so it builds in the CPU harness and can be
// mirrored to Metal later.
//
#include <cstdint>
#include <vector>

namespace hhsr {

using f32 = float;

// Which frame pairs to measure -- mirrors ShiftCollection.TrackingStrategy.
// Only Full and Blocks give the redundancy outlier rejection needs; the other
// two are kept for parity / cost control.
enum class TrackingStrategy {
    Full,            // every pair (i,j), i<j  -> N(N-1)/2 measurements
    OnlyOnReference, // frame->reference only (the plain Google way, no redundancy)
    OnReferenceBlock,// full pairs inside a block around ref, singletons outside
    Blocks           // all pairs within blockSize of each other (sliding window)
};

// A single measured shift: block-match of frame `toTrack` against frame
// `reference` (same ordering convention as ISA's ShiftPair).
struct ShiftPair {
    int reference;
    int toTrack;
};

// A per-tile 2D displacement grid for one measured pair. Length ny*nx*2,
// interleaved (dx,dy) row-major -- same layout as FlowField::flow, so the
// caller can hand an align() result straight in.
struct ShiftGrid {
    int ny = 0, nx = 0;
    std::vector<f32> v; // ny*nx*2
    f32 dx(int t) const { return v[(size_t)t * 2 + 0]; }
    f32 dy(int t) const { return v[(size_t)t * 2 + 1]; }
};

class ShiftConsensus {
public:
    // frameCount including the reference; referenceIndex in [0,frameCount).
    // blockSize is clamped to frameCount-1 (ISA does the same).
    ShiftConsensus(int frameCount, int referenceIndex,
                   TrackingStrategy strategy, int blockSize);

    // The frame pairs the caller must measure, in the exact order the solver
    // expects (measurement index == position in this list). Mirrors
    // ShiftCollection.FillShiftPairs / GetShiftPairs.
    const std::vector<ShiftPair>& pairs() const { return pairs_; }
    int shift_count() const { return (int)pairs_.size(); }
    int frame_count() const { return frameCount_; }
    int reference_index() const { return referenceIndex_; }

    // Outlier threshold in px^2 on |measured - reprojected|^2 (ISA hard-codes
    // 1.0). Exposed so the integration can tune it; <=0 keeps 1.0.
    void set_reject_threshold_px2(f32 t) { rejectPx2_ = (t > 0.f) ? t : 1.f; }
    void set_max_iterations(int it) { maxIters_ = (it > 0) ? it : 1; }

    // Solve the per-tile consistency system for every tile.
    //  measured[k]  = the ShiftGrid for pairs()[k], all sharing ny*nx.
    //  out[f]       = cleaned per-tile shift of frame f to the reference,
    //                 for every f != referenceIndex (out[referenceIndex] is
    //                 left zero). Vector length frameCount, each ny*nx*2.
    // Returns false if the grids are inconsistent / empty.
    bool minimize(const std::vector<ShiftGrid>& measured,
                  std::vector<ShiftGrid>& out) const;

    // Exposed for testing: the m x (N-1) incidence matrix A, column-major with
    // leading dimension m (A(meas,unk) == data[unk*m + meas]) -- the exact
    // layout ShiftCollection.CreateShiftMatrix produces.
    std::vector<f32> incidence_matrix() const { return CreateShiftMatrix(); }

private:
    int frameCount_;
    int referenceIndex_;
    TrackingStrategy strategy_;
    int blockSize_;
    f32 rejectPx2_ = 1.f;
    int maxIters_ = 10;

    std::vector<ShiftPair> pairs_;         // FillShiftPairs
    std::vector<int> indices_;             // FillIndexTable, frameCount*frameCount, -1 = none

    int GetShiftCount() const;
    std::vector<f32> CreateShiftMatrix() const;
    void FillShiftPairs();
    void FillIndexTable();
};

} // namespace hhsr
