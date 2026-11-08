#pragma once
// Structure tensors of burst frames:
//  - structure_tensor_half: the tensor itself (shared by the Structure Map export and
//    the GCam structure-match robustness term, Config::gcam_structure_match).
//  - write_structure_map: per-frame colour visualisation (Config::structure_map_save),
//    written beside the DNG as <dng stem>_structure_fNN[_ref].ppm; the app saves them to
//    Photos.
#include "types.h"
#include <string>

namespace hhsr {

// Structure tensor J = G_sigma * (grad I grad I^T) of the frame's half-resolution grey
// I = sqrt(mean of each 2x2 Bayer quad) (sqrt: shadows keep their structure, shot noise
// becomes roughly level-independent). Returns Image(h/2, w/2, 4) holding, per pixel,
//   0 Jxx, 1 Jxy, 2 Jyy  -- the smoothed tensor (sigma in half-res px)
//   3 nv                  -- the per-axis gradient noise variance E[g_x^2] of pure noise
//                            at that brightness, from the DNG noise profile (alpha, beta)
//                            and the same smoothing, so the noise part of the tensor is
//                            nv * I and the signal part is J - nv * I.
// Empty Image on an empty / too small frame.
Image structure_tensor_half(const Image& raw, const Config& cfg, float sigma);

// Colour map of the tensor, l1 >= l2 its eigenvalues:
//   hue        = EDGE orientation (perpendicular to the dominant gradient), 180 deg = full circle
//   saturation = coherence (l1 - l2) / (l1 + l2): 1 on a clean edge, 0 on corners / isotropic texture
//   brightness = structure strength sqrt(l1), normalised by `scale_io`
// Flat areas are black, edges vivid colour, corners and texture grey-white.
// scale_io: strength normaliser shared across the burst; pass <= 0 for the first frame
// (the reference), which sets it to its own 99.5th percentile, so every frame's map uses
// the same brightness scale and the frames are directly comparable.
// Returns false (and writes nothing) on an empty frame or an I/O error.
bool write_structure_map(const Image& raw, const Config& cfg, const std::string& dng_path,
                         int frame_index, bool is_reference, float& scale_io);

}  // namespace hhsr
