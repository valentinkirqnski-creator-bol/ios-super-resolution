#pragma once
// High-frequency misalignment rejection (Wronski et al. S.3), implemented fresh
// and self-contained. Two signals, both required to reject a region:
//   Test A -- high-frequency content: a large fraction of local image variance is
//             destroyed by a low-pass filter (repetitive fine texture).
//   Test B -- alignment instability: the per-tile displacement varies sharply
//             across the local tile neighbourhood.
// Where both hold the comparison frame's robustness is forced to zero (H=1),
// R' = R*(1-H); the reference frame remains the fallback. All on the half-res
// guide grid. Nothing here reads or depends on any other robustness code.
#include "types.h"
#include <vector>

namespace hhsr {

// Test A. Per guide-pixel high-frequency variance-loss L in [0,1], on a luminance
// signal of the guide: L = (Vbefore - Vafter) / (Vbefore + eps), where Vbefore and
// Vafter are the local variance of the guide luminance before and after a 3x3
// low-pass, over a (win x win) clamp-to-edge window. L is forced to 0 where the
// local variance is below min_var (flat/near-uniform regions never flag).
Image hf_variance_loss(const Image& guide, int win, f32 min_var);

// Test B. Per-tile local variation of the displacement field: the mean Euclidean
// distance from a tile's vector to each of its (up to 8) neighbours. Returned as
// flow.ny * flow.nx, row-major. High = the estimated motion is locally inconsistent.
std::vector<f32> hf_flow_variation(const FlowField& flow);

// Combine the two tests into a guide-resolution rejection mask H (1 = reject),
// H(p) = (L(p) > tl) AND (D(tile of p) > td). guide_tile_size is the tile size in
// GUIDE pixels (how many guide pixels one flow tile spans).
Image hf_build_mask(const Image& Lmap, const std::vector<f32>& D,
                    int flow_ny, int flow_nx, int guide_tile_size, f32 tl, f32 td);

// Apply R' = R * (1 - H) in place. H is guide-resolution; R may be any resolution
// (each R pixel maps to its covering guide pixel by nearest scaling), so this is
// correct whether R is guide-res or Dodgson-upscaled raw-res. Rejected pixels are
// zeroed on every channel. Call BEFORE the 5x5 minimum filter.
void hf_apply_mask(Image& R, const Image& H);

// One-shot convenience: build the mask from the guide + flow and apply it to R.
void hf_reject_apply(Image& R, const Image& guide, const FlowField& flow,
                     int guide_tile_size, const Config& cfg);

}  // namespace hhsr
