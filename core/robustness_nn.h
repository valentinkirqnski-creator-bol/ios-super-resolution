#pragma once
//
// Learned robustness mask: a small convolutional network that predicts, per
// guide pixel, whether merging this comparison frame is safe -- the job
// Wronski Eq. 5-9 does analytically in robustness.cpp.
//
// Why a network at all. The analytic mask decides from d^2/sigma^2, a colour
// difference between 3x3 means of the half-res guide. That statistic is
// structurally blind to the failure this port actually suffers from: a tile
// whose flow is badly wrong but whose wrongly-fetched content resembles the
// right content (flat shadow onto flat shadow), or whose error lives in
// structure finer than the 6x6-raw support of those means (thin rods). No
// tuning of s/t recovers information the statistic never carried. Measured
// against ground truth on synthetic bursts built from this camera's own raws
// (tools/rob_nn), the analytic mask separates harmful from harmless pixels
// with AUC 0.638; the network reaches 0.926 from the same inputs plus the
// flow field and a wider receptive field.
//
// The network is NOT a different photometric test. Its advantage is that it
// also sees the estimated flow, the local spread of that flow, and a ~30
// raw-pixel neighbourhood -- so it can reject a tile because its motion is
// implausible or disagrees with its surroundings, which is exactly the
// evidence available when the photometry is degenerate.
//
// Feature planes are assembled by build_robustness_nn_features (stages.h) so
// the layout lives in portable C++ next to the code that must match the
// training generator, rather than here.
//
// UNTESTED ON DEVICE: this file, its .mm implementation and the bundled
// RobustnessNet.mlmodel were produced on a non-Apple machine with no Core ML
// runtime, so load and numerical behaviour on-device have not been
// exercised. Every entry point fails closed -- robustness_nn_available()
// returns false and compute_robustness falls back to the analytic mask --
// so a broken or missing model degrades to current behaviour rather than
// producing a wrong mask.
//
#include "types.h"

namespace hhsr {

// True once RobustnessNet has been located and loaded (lazy, cached). False
// if the resource is missing or MLModel construction fails, in which case
// robustness_nn_infer() also returns false and the caller must fall back to
// the analytic mask.
bool robustness_nn_available();

// Runs the model on assembled feature planes.
//
// feat: guide resolution, kRobustnessNnChannels interleaved channels, in the
//       order documented on build_robustness_nn_features.
// out:  resized to feat.h x feat.w x 1, values in [0,1] -- 1 merge, 0 reject.
//
// Returns false on any failure (model unavailable, allocation failure,
// prediction error, unexpected output shape); `out` is left untouched.
bool robustness_nn_infer(const Image& feat, Image& out);

// Drops the cached input buffer. Called automatically whenever inference
// fails or headroom runs short; exposed so a caller that knows the burst is
// over can return the memory without waiting for the next failure.
void robustness_nn_release_buffers();

// ---- refinement network (RobustnessRefineNet, tools/rob_refine) ----------
//
// A SECOND, independent model with the opposite relationship to the analytic
// mask. robustness_nn_infer above produces a mask that REPLACES Wronski
// Eq. 5-9. These produce a keep-confidence q that MULTIPLIES it:
//
//     R_final = R * (1 - kappa * (1 - q))
//
// so the analytic mask stays authoritative and the network can only take
// weight away, never add it. The two are independently toggled and can run
// together, in which case the refinement refines whichever mask the
// replacement stage left behind.
//
// The target is the failure Eq. 5-9 is structurally unable to see: with one
// flow vector per tile, rotation and parallax leave a fraction-of-a-pixel
// misalignment that grows toward the tile edges, doubling or thickening thin
// edges while d^2/sigma^2 reads as MORE confident rather than less. See
// types.h (kRobustnessRefineChannels) and tools/rob_refine/README.md.
bool robustness_refine_available();

// feat: kRobustnessRefineChannels interleaved planes in the order
//       build_robustness_refine_features documents (stages.h).
// out:  resized to feat.h x feat.w x 1, q in [0,1] -- 1 keep, 0 reject.
//
// Returns false on any failure, leaving `out` untouched; the caller then
// leaves R exactly as the analytic mask produced it, which is the reason
// enabling this can never make the pipeline worse than not enabling it.
bool robustness_refine_infer(const Image& feat, Image& out);

void robustness_refine_release_buffers();

} // namespace hhsr
