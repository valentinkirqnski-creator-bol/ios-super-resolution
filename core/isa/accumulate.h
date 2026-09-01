#pragma once
// CPU port of ImageStackAlignator's merge/accumulate kernels:
// Kernels/DeBayerKernels.cu (accumulateImages, deBayersSubSample3).
#include <vector>
#include <cstdint>
#include "structure_tensor.h"
#include "debayer.h"

namespace isacpu {

struct Vec2f { float x = 0, y = 0; };
struct Vec4f { float x = 0, y = 0, z = 0, w = 0; };

// deBayersSubSample3: half-resolution box-subsample debayer (2x2 -> 1),
// used to build the half-res robustness/certainty-mask input images.
void debayer_subsample3(const std::vector<uint16_t>& raw, int width, int height,
                        const int cfa[2][2], float max_val, std::vector<Rgbf>& out_half);

// accumulateImages: raw-resolution merge. Adds one frame's contribution
// into `pixel`/`total_weight` (both dimX*dimY, same layout as `raw`).
// `kernel_param` is the reference frame's per-pixel merge-kernel precision
// matrix (Vec3f{XX,YY,XY}, from compute_kernel_param), `shift` is this
// frame's per-pixel optical-flow displacement (raw-resolution, rounded to
// the nearest integer pixel exactly as the source does), `certainty` is the
// half-resolution per-channel robustness mask (Vec4f, .x=R .y=G .z=B).
void accumulate_images(const std::vector<uint16_t>& raw, int dimX, int dimY,
                      const int cfa[2][2],
                      std::vector<Rgbf>& pixel, std::vector<Rgbf>& total_weight,
                      const std::vector<Vec4f>& certainty_half, int certainty_stride_x,
                      const std::vector<Vec3f>& kernel_param,
                      const std::vector<Vec2f>& shift,
                      const float white_level[3], const float black_level[3]);

}  // namespace isacpu
