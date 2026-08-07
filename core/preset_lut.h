#pragma once
// Fitted look-up table reproducing the user's fixed external preset.
//
// Maps the merged linear RGB this pipeline produces straight to final display
// sRGB, absorbing white balance, the colour matrix, gamma and the preset's tone
// and colour grading in one step.
//
// Accuracy on the reference pair (linear DNG + the target JPEG rendered from
// it), after the uint8 quantisation and through the real trilinear path:
// 3.29 LSB mean over the whole frame, 4.15 LSB over a fifth of the frame held
// out of the fit. It is deliberately NOT exact -- the preset was measured to
// contain content-adaptive processing, so a colour-only mapping cannot reach
// zero.
//
// See preset_lut.cpp for how it is fitted, and for why a pair of JPEGs without
// the DNG is not enough to fit it.
//
// Applies to the preview/JPEG only. The DNG is written from the unmodified
// merge and must stay that way.
#include "types.h"

namespace hhsr {
constexpr int kPresetLutN = 33;
extern const unsigned char kPresetLut[kPresetLutN * kPresetLutN * kPresetLutN * 3];
bool preset_lut_enabled();
// lin: linear RGB as written to the DNG. Returns display sRGB in [0,1].
void preset_lut_apply(const f32 lin[3], f32 out[3]);
}
