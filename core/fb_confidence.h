#pragma once
// Forward-backward flow confidence (Config::fb_conf_enabled).
//
// Forward displacement F: reference -> comparison (the merge's tile flow). Backward
// displacement B: comparison -> reference, measured from the comparison side. A correct
// correspondence round-trips: x + F + B(x + F) ~ x. The round-trip error e becomes a
// per-tile confidence C = exp(-e^2 / (2 tau^2)) that MULTIPLIES the robustness mask
// (after every other mask stage), so a suspicious tile is down-weighted and a clearly
// inconsistent one rejected. With tau = 0.75 px: e 0.05 -> 1.00, 0.3 -> 0.92,
// 1.5 -> 0.14, 4 -> 0.
//
// Backward modes:
//   local (default, fb_conf_full = false): per tile, the comparison patch at the forward
//     destination is the template and B is solved by a few inverse-compositional
//     Lucas-Kanade steps against the reference, starting from -F. Full resolution, no
//     pyramid, no extra images: memory = one float per tile; cost ~ one ICA level. It
//     catches tiles whose vector the comparison side disagrees with (occlusion, two
//     motions in a tile, edges placed differently, sub-pixel compromises) but not a wrong
//     lock that is consistent both ways (e.g. a whole-period slip on a repeated pattern).
//   full (fb_conf_full = true): an independent coarse-to-fine align() comparison ->
//     reference (the same backend as the forward pass, GPU on device); B is sampled at
//     the forward destination. Catches consistent-looking blunders too; ~2x alignment
//     time and one transient comparison pyramid. Falls back to local when a global
//     homography warped the frame.
//
// e is measured in raw px and (a) weighted by the comparison patch's structure tensor
// normalised to its largest eigenvalue, so along a straight edge only the ACROSS-edge
// part counts (a slide along an edge is unobservable and invisible -- the aperture
// problem), and (b) noise-corrected: the error expected from sensor noise alone
// (~4 sigma^2 / lambda1) is subtracted (x fb_conf_noise_k), so dim flat tiles are not
// rejected. Flat tiles (no structure) get e = 0 (misalignment there is invisible).
#include "types.h"
#include <vector>

namespace hhsr {

// Per-patch round-trip error (raw px, weighted + noise-corrected) on the patch grid
// (flow.ny * fb_conf_subdiv) x (flow.nx * fb_conf_subdiv), for the merge flow
// `flow` (raw px, raw tile grid of tile_size). ref_grey / comp_grey: the alignment greys
// (raw or half resolution; scale inferred from raw_w). Empty on bad input.
std::vector<float> fb_tile_error(const Image& ref_grey, const Image& comp_grey,
                                 const FlowField& flow, int raw_h, int raw_w,
                                 int tile_size, const Config& cfg);

// Multiply the mask by C(e) = exp(-e^2 / (2 tau^2)) per tile: R (any resolution
// that maps onto the raw frame: raw, guide = raw/2, or merge = raw*scale) and/or the
// 8-bit merge-res mask. Nearest tile (the merge's int(lr / tile) fetch), bilinear
// between tile centres when the merge uses the bilinear flow. Returns the mean C.
double fb_apply_confidence(Image* R, Mask8* m8, const std::vector<float>& err,
                           const FlowField& flow, int raw_h, int raw_w,
                           int tile_size, const Config& cfg);

// The same multiply on raw buffers (a fp32 mask [h][w][c] and/or an 8-bit mask [h8][w8];
// either may be null): err on the (tiles_ny*subdiv) x (tiles_nx*subdiv) patch grid,
// confidence exp(-e^2 / (2 tau^2)). Used for masks that live in GPU shared memory.
double fb_apply_confidence_raw(float* data, int h, int w, int c, uint8_t* data8, int h8, int w8,
                               const std::vector<float>& err, int tiles_ny, int tiles_nx,
                               int raw_h, int raw_w, int tile_size, int subdiv, float tau_px,
                               const Config& cfg);

}  // namespace hhsr
