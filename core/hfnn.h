#pragma once
// HF-artifact NN (Config::hfnn_enabled): a very small CNN that replaces the HF
// artifact rejection rule (paper S.3) -- it decides, ONLY in high-frequency /
// repeated-pattern areas, whether the comparison frame is misaligned there (wrong
// period lock, sliding along lines, tile-shaped steps) or correctly aligned fine
// detail. No multi-frame consensus, so no pre-pass and no second alignment: it uses
// the comparison frame's own aligned grey from the main loop.
//
// Inputs (grey grid = raw resolution), per pixel:
//   0  R                       the existing mask (Wronski / GCam), on the grey grid
//   1  c((comp - ref) / sd)    flow-warped comparison vs reference
//   2  c(hp(ref)  / s1)        reference fine structure (3x3 high-pass)
//   3  c(hp(comp) / s1)        comparison fine structure
//   4  min(D, 4) / 4           tile flow disagreement with its 3x3 median (cref_aux_rows)
//   5  HF variance loss        of the reference (cref_aux_rows): where it applies
// (sd, s1, c() as in cref.h.) Net: 3 x [3x3 conv, 8 ch, ReLU], dilations 1, 2, 4
// (radius 7) + 1x1 head; R_nn = sigmoid(logit(clamp(R)) + delta).
// Output: R' = R + w (R_nn - R), w = smoothstep(gate - 0.1, gate + 0.1, HF loss), so
// outside HF areas the mask is untouched (bit-identical where w == 0); bands with no
// HF pixel are skipped entirely. ~1.4k weights (core/hfnn_weights.h).
#include "types.h"
#include <vector>

namespace hhsr {

constexpr int kHfnnIn = 6;
constexpr int kHfnnCh = 8;
constexpr int kHfnnLayers = 3;
constexpr int kHfnnHalo = 7;            // 1 + 2 + 4
extern const int kHfnnDil[kHfnnLayers];

bool hfnn_available();

// Feature planes [kHfnnIn][rows*w] for grey rows [y0, y1) with `halo` rows each side
// (edge-clamped). Also returns the HF-loss plane for those rows in hf_out (rows*w).
void hfnn_features(const Image& ref_grey, const Image& comp_aligned, const Image& R_grey,
                   const std::vector<float>& tdis, int ny, int nx, int tile_size,
                   const Config& cfg, int y0, int y1, int halo,
                   std::vector<float>& out, std::vector<float>& hf_out);

// Refine R in place (any resolution; brought to the grey grid and back). flow /
// tile_size: the flow the merge uses. False (R untouched) if no weights / bad input.
bool hfnn_refine(Image& R, const Image& ref_grey, const Image& comp_aligned,
                 const FlowField& flow, int tile_size, const Config& cfg);

}  // namespace hhsr
