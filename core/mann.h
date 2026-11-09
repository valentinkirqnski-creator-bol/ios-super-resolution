#pragma once
// Misalignment-artifact NN (Config::mann_enabled): a very small CNN that runs ONLY on
// tiles whose flow is irregular (the 3x3 tile-flow spread M of compute_flow_irregularity
// above the GCam motion threshold, the same flow.z test the SABRE boost uses) and decides
// there whether this comparison frame would leave a visible artifact: an edge shifted
// across itself, a doubled / ghosted edge, or a stair of slightly different tile
// offsets along a long edge (the aperture problem). Variance loss under low-pass
// filtering is measured on those tiles (reference, comparison, their difference, and
// the pair) and fed to the network with the local structure. No multi-frame consensus,
// so no pre-pass and no second alignment: it uses this frame's warped grey from the
// main loop.
//
// Inputs (grey grid = raw resolution), per pixel:
//   0  R                      the existing mask (Wronski / GCam), on the grey grid
//   1  c((comp - ref) / sd)   flow-warped comparison vs reference
//   2  c(hp(ref)  / s1)       reference fine structure (3x3 high-pass)
//   3  c(hp(comp) / s1)       comparison fine structure
//   4  VL(ref)                variance loss of the reference under the 5x5 binomial low-pass
//   5  VL(comp)               same on the warped comparison
//   6  VL(comp - ref)         same on the difference (noise 2x): HF-only disagreement
//   7  pair HF disagreement   sum(hp_r - hp_c)^2 / (sum hp_r^2 + sum hp_c^2), noise-corrected, /2
//   8  min(M / thresh, 4) / 4 the gate signal (bilinear over tile centres)
//   9  min(|f_bil - f_near|, 2) / 2   seam: how far this pixel's tile vector is from the
//                             smooth field (large at a step between neighbouring tiles)
//  10  min(D, 4) / 4          tile flow disagreement with its 3x3 median
// (sd = sqrt(2 nv), s1 = sqrt(nv), nv = alpha*max(mean5,0) + beta of the reference;
//  c(x) = clamp(x, -8, 8) / 8. VL = 1 - var(L) / var, noise-corrected, 0 below SNR 4.)
// Net: 4 x [3x3 conv, 8 ch, ReLU], dilations 1, 2, 4, 1 (radius 8) + 1x1 head;
// R_nn = sigmoid(logit(clamp(R)) + delta - strictness).
// Output: R' = R + g (R_nn - R), g = smoothstep(0.75 T, 1.25 T, M), T = motion thresh x
// mann_gate, so outside irregular tiles the mask is untouched (bit-identical where g == 0)
// and only bands / column spans holding gated tiles are computed. ~2.6k weights
// (core/mann_weights.h).
#include "types.h"
#include <vector>

namespace hhsr {

constexpr int kMannIn = 11;
constexpr int kMannCh = 8;
constexpr int kMannLayers = 4;
constexpr int kMannHalo = 8;            // 1 + 2 + 4 + 1
extern const int kMannDil[kMannLayers];

bool mann_available();

// The GCam flow.z threshold in raw px for this grid (gcam_sabre_motion_thresh x tile x 4128/16).
float mann_motion_thresh(int tile_size, const Config& cfg);

// Per-tile helpers shared by the app and the dataset tool.
std::vector<float> mann_tile_irregularity(const FlowField& flow);    // M (raw px)

// Feature planes [kMannIn][(wy1-wy0) x (wx1-wx0)] for the grey rect [wy0,wy1)x[wx0,wx1),
// which must lie inside the image (callers clamp their halo to the image, so the conv's
// edge clamp is the image edge = the training's replicate padding). Also the gate g for
// the same rect in gate_out. M: mann_tile_irregularity(flow); tdis: cref_flow_disagreement.
void mann_features(const Image& ref_grey, const Image& comp_aligned, const Image& R_grey,
                   const FlowField& flow, const std::vector<float>& M,
                   const std::vector<float>& tdis, int tile_size, const Config& cfg,
                   int wy0, int wy1, int wx0, int wx1,
                   std::vector<float>& out, std::vector<float>& gate_out);

// Refine R in place (any resolution; brought to the grey grid and back). flow /
// tile_size: the flow the merge uses. False (R untouched) if no weights / bad input.
bool mann_refine(Image& R, const Image& ref_grey, const Image& comp_aligned,
                 const FlowField& flow, int tile_size, const Config& cfg);

}  // namespace hhsr
