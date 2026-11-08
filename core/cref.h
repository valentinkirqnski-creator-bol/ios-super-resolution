#pragma once
// Consensus refiner (Config::cref_enabled): a small CNN that REFINES the existing
// robustness mask (Wronski / Alg. 6 or GCam) of one comparison frame using the
// multi-frame consensus.
//
// Why consensus: at the finest detail a correctly aligned frame does not match the
// reference (each frame samples the scene at its own sub-pixel phase -- aliasing, the
// SR signal itself), and a sub-pixel slip produces the same reference-vs-frame
// difference. The other frames break the tie: aliasing is consistent with what they
// imply about the scene, a slip is not.
//
// Inputs (all on the reference's grey grid, the FFT grey = raw resolution), per pixel:
//   0  R                          the existing mask, resampled to the grey grid
//   1  c((comp - ref)  / sd)      comparison frame (warped by its flow) vs reference
//   2  c((comp - loo)  / sd)      vs the leave-one-out consensus of the OTHER frames
//   3  c((ref  - loo)  / sd)      the reference's own deviation (its aliasing)
//   4  c(hp(ref)  / s1)           the reference's fine structure (3x3 high-pass)
//   5  c(hp(comp) / s1)           the comparison frame's fine structure
// with sd = sqrt(2 (alpha m + beta)), s1 = sqrt(alpha m + beta), m = max(ref, 0) (the
// DNG noise profile), c(x) = clamp(x, -8, 8) / 8, loo = (n cons - comp) / (n - 1).
//
// Network: 4 x [3x3 conv, 8 ch, ReLU] with dilations 1, 2, 4, 1 (17 x 17 px receptive
// field, replicate padding) + 1x1 conv -> delta;
//   R' = sigmoid(logit(clamp(R, 0.02, 0.98)) + delta)
// so it can lower the mask (slips the analytic mask kept) or raise it (over-rejection).
// ~2.2k weights (core/cref_weights.h, exported by tools/cref/export_weights.py).
#include "types.h"
#include <vector>

namespace hhsr {

constexpr int kCrefIn = 6;
constexpr int kCrefCh = 8;
constexpr int kCrefLayers = 4;          // 3x3 layers
constexpr int kCrefHalo = 8;            // receptive-field radius (1+2+4+1)
extern const int kCrefDil[kCrefLayers];

// True when trained weights are compiled in (core/cref_weights.h); otherwise the
// refine is a no-op.
bool cref_available();

// Feature planes [kCrefIn][h*w] (plane-major) for rows [y0, y1) of the grey grid,
// with `halo` extra rows above/below (edge-clamped): out has (y1-y0+2*halo) rows.
// R_grey: the mask already on the grey grid (h x w). n_frames: comparison frames in
// the consensus (cons = their mean, comp included).
void cref_features(const Image& ref_grey, const Image& comp_aligned, const Image& cons,
                   int n_frames, const Image& R_grey, const Config& cfg,
                   int y0, int y1, int halo, std::vector<float>& out);

// Refine R in place (any resolution; it is brought to the grey grid and back).
// Returns false (R untouched) if a precondition fails or no weights are compiled in.
bool cref_refine(Image& R, const Image& ref_grey, const Image& comp_aligned,
                 const Image& cons, int n_frames, const Config& cfg);

}  // namespace hhsr
