// sr_gate -- a learned per-frame robustness mask (Config::sr_gate_enabled).
//
// WHAT IT REPLACES
//
// Eq. 5-9 in one go: the exponential of d^2/sigma^2, the s1/s2 motion prior,
// the r_t offset, the geometry rejection and the 5x5 minimum. The network emits
// the final per-pixel decision, so none of those is applied on top of it. The
// 5x5 minimum in particular exists to spread rejection outward from a pointwise
// test with no spatial context; this has a 13x13 receptive field and was trained
// to produce the finished mask, so dilating it again would double-count.
//
// WHY IT EXISTS
//
// Two measured facts about the shipping configuration.
//
//  1. The mask now lives on the FULL-RESOLUTION FFT guide
//     (robustness_fft_guide_active). Eq. 9's 5x5 minimum was designed for the
//     half-resolution guide, where each sample already averaged a 2x2 Bayer
//     quad. On the full-resolution guide it takes the worst of 25 much noisier
//     per-pixel tests, and measures 0.95 -> 0.61 mean R at ZERO flow error --
//     it rejects most of the burst before there is anything to reject.
//
//  2. A sub-pixel offset between frames is the signal an SR merge feeds on, not
//     damage. R = s*exp(-d^2/sigma^2) scores it as damage, because d^2 is
//     derived from "what we fetched versus what we should have fetched" and
//     cannot represent the merge kernel's use of the offset as information. The
//     project measured the consequence directly: below ~1.6 px of per-tile flow
//     error, rejection is NET HARMFUL.
//
// So the gate is not trained to reproduce a better R. It is trained against the
// MERGED IMAGE -- specifically against what the merge would have produced with
// perfect alignment and nothing rejected -- which is the only target that can
// price a sub-pixel offset as a gain and a ghost as a loss on the same scale.
// tools/sr_gate/ holds the simulator, the training and the measurements.
//
// SHAPE
//
//   conv 3x3 dilation 1   8 -> 8   ReLU
//   conv 3x3 dilation 2   8 -> 8   ReLU
//   conv 3x3 dilation 3   8 -> 8   ReLU
//   conv 1x1              8 -> 1   sigmoid
//
// 1761 parameters, ~1.8k multiply-adds per pixel, replicate padding at the
// border (the same index clamping every other window in robustness.cpp uses).
#pragma once

#include "types.h"
#include "sr_gate_shared.h"

namespace hhsr {

// Shape and the feature compressions live in sr_gate_shared.h, which
// core/HHSRKernels.metal includes too -- there is one copy of those numbers
// because they are part of the trained function, not a display choice.
static constexpr int kSrGateFeatures = SRG_FEATURES;
static constexpr int kSrGateWidth = SRG_WIDTH;
static constexpr int kSrGateLayers = SRG_LAYERS;
// Rows of context each side of an output row: the sum of the dilations.
static constexpr int kSrGateHalo = SRG_HALO;

// Feature channel order. Interleaved in the returned Image, so channel i of
// pixel (y, x) is feat.at(y, x, i). See sr_gate_shared.h for what each one is.
enum SrGateFeature {
    kSrGateFeatExpA = SRG_F_EXP_A,
    kSrGateFeatLogA = SRG_F_LOG_A,
    kSrGateFeatSnr = SRG_F_SNR,
    kSrGateFeatSubpix = SRG_F_SUBPIX,
    kSrGateFeatSpan = SRG_F_SPAN,
    kSrGateFeatEmag = SRG_F_EMAG,
    kSrGateFeatGrad = SRG_F_GRAD,
    kSrGateFeatDirE = SRG_F_DIR_E,
};

// True when the weights are present and the shapes agree with this header.
bool sr_gate_available();

// The 8-channel input plane for ONE comparison frame, on the same lattice as
// ref_means (the raw lattice in the shipping configuration).
//
// d_sq / sigma_sq are Eq. 6's outputs, which compute_robustness has already
// built -- the gate deliberately consumes them rather than recomputing, so it
// scores the same correspondence and the same noise model the analytic mask
// does.
Image build_sr_gate_features(const Image& ref_means, const Image& ref_vars,
                             const Image& d_sq, const Image& sigma_sq,
                             const FlowField& flow, int tile_size,
                             const Config& cfg);

// CPU inference over a whole feature plane. This is the golden reference the
// Metal path is checked against, and the only path on non-Apple builds.
// Returns an empty Image if the feature plane is malformed.
Image sr_gate_infer_cpu(const Image& feat);

// Features + inference in one call. Returns an empty Image when the gate is
// disabled or unavailable, which every caller treats as "fall back to the
// analytic mask" rather than as an error.
// ---- the artifact U-Net (Config::sr_gate_unet_enabled) -------------------
// See core/sr_gate_unet.cpp. Predicts log1p(artifact / noise sigma); the mask
// comes from sr_gate_unet_mask() in sr_gate_shared.h, whose tolerance is
// Config::sr_gate_tau.
bool sr_gate_unet_available();
// The flat weight blob, so metal_gpu.mm can upload it without including the
// generated header. Same arrangement as sr_gate_weights() above.
const f32* sr_gate_unet_weights(int* count);
Image build_sr_gate_unet_features(const Image& ref_means, const Image& ref_vars,
                                  const Image& comp_means, const Image& d_sq,
                                  const Image& sigma_sq, const FlowField& flow,
                                  int tile_size, const Config& cfg);
// tau/beta explicit so the parity harness can sweep the policy without a Config.
Image sr_gate_unet_infer_cpu(const Image& feat, f32 tau, f32 beta);
Image sr_gate_unet_mask_image(const Image& ref_means, const Image& ref_vars,
                              const Image& comp_means, const Image& d_sq,
                              const Image& sigma_sq, const FlowField& flow,
                              int tile_size, const Config& cfg);

Image sr_gate_mask(const Image& ref_means, const Image& ref_vars,
                   const Image& d_sq, const Image& sigma_sq,
                   const FlowField& flow, int tile_size, const Config& cfg);

// The flat weight blob, for the GPU upload. Layout is documented in
// core/sr_gate_weights.h and is what the Metal kernel indexes.
const f32* sr_gate_weights(int* count);

}  // namespace hhsr
