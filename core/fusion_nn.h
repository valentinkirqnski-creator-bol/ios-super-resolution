#pragma once
//
// Fusion NN (Config::fusion_nn_enabled, tools/fusion_nn, model r1): a small network that sets
// each comparison frame's merge weight per pixel instead of the analytic robustness
// (Wronski Eq. 5-9). It was trained end to end through this app's exact merge
// (core/merge.cpp, reproduced in tools/fusion_nn/fusion.py) on the final image only -- no mask
// labels -- with the app's own flow and kernels.
//
// What it predicts. One logit l_f per frame (reference included) and guide pixel; the frames'
// normalised contribution weights are softmax_f(l_f). The merge divides by the summed kernel
// weights, so the softmax denominator cancels and the merge only needs each comparison frame's
// weight RELATIVE to the reference: R_f = exp(l_f - l_ref). That is what fusion_nn_mask returns,
// in the shape of the analytic mask it replaces, so every merge path consumes it unchanged.
// R_f is NOT capped at 1 (the model was trained without a cap): a comparison frame can count
// more than the reference where the reference is the outlier.
//
// Inputs (per frame, guide = half resolution, quad RGB: R, mean G, B), variance-stabilised
// with the DNG noise profile (GAT, 1 unit = 1 noise std):
//   is-reference, reference brightness, (frame - reference) / 4, (frame - burst mean) / 4
// where every comparison guide is sampled where the merge's own tile flow puts the reference
// pixel, and the burst mean is over all frames (reference included) -- a pre-pass, see
// compute_fusion_nn_consensus in pipeline_paths.cpp.
//
// Network (core/fusion_nn_weights.h, 1,993 parameters): 3x3 conv 8->12, 4 residual
// depthwise-separable blocks (3x3 depthwise dilations 1/2/4/8, 1x1 12->12, ReLU), 1x1 -> logit.
// Receptive field +-16 guide px. CPU, fp32, banded (64 rows + 16-row halo), so the working set
// is a few MB regardless of image size; ~1.9 kMAC per guide pixel.
//
// Fails closed: any unsupported input (no noise profile, dense/affine/homography flow, shape
// mismatch, low memory) leaves the analytic mask in place.
//
#include "types.h"

namespace hhsr {

// Quad RGB guide of a Bayer frame: [h/2, w/2, 3] (R, mean of the two G, B).
Image fusion_nn_guide(const Image& raw, const Config& cfg);

// True when the merge reads plain per-tile vectors from `flow` (what the network was trained on).
bool fusion_nn_flow_supported(const FlowField& flow);

// `guide` sampled (bilinear, border clamp) where the merge's tile flow puts each reference guide
// pixel: the tile of raw position 2g+0.5, displacement in raw px / 2.
Image fusion_nn_warp_guide(const Image& guide, const FlowField& flow, int tile_size, int threads);

// Generalised Anscombe transform in place (unit noise std): (2/a) sqrt(max(a x + 3/8 a^2 + b, 0)).
void fusion_nn_gat(Image& g, float a, float b, int threads);

// The noise model the network was trained with (DNG noise profile, green). False if absent.
bool fusion_nn_noise(const Config& cfg, float& a, float& b);

// One frame's logits [gh, gw, 1]. ref_gat / frame_gat / mean_gat: GAT guides [gh, gw, 3];
// frame_gat already warped to the reference. is_ref: the reference's own pass.
Image fusion_nn_logits(const Image& ref_gat, const Image& frame_gat, const Image& mean_gat,
                       bool is_ref, float a, float b, int threads);

// R = exp(l_f - l_ref) at the analytic mask's shape (h, w, c): guide resolution samples the
// logits directly, any other size bilinearly in guide coordinates ((y + 0.5) * gh / h - 0.5).
Image fusion_nn_mask(const Image& l_f, const Image& l_ref, int h, int w, int c, int threads);

// Packed weights for the GPU path (fusion_nn_band_metal): stem [9][8][12] + bias [12],
// per block depthwise [9][12] + bias, pointwise [in 12][out 12] + bias, head [12] + bias.
struct FusionNnWeights {
    const float* stem = nullptr;
    const float* stem_b = nullptr;
    const float* dw[4] = {};
    const float* dw_b[4] = {};
    const float* pw[4] = {};
    const float* pw_b[4] = {};
    const float* head = nullptr;
    int dil[4] = {};
};
FusionNnWeights fusion_nn_weights();

// Upper bound of the transient host memory one comparison frame needs (bytes).
size_t fusion_nn_frame_bytes(int raw_h, int raw_w, int mask_h, int mask_w, int mask_c);

}  // namespace hhsr
