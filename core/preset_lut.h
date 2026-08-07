#pragma once
// Fitted look-up table reproducing the user's fixed external preset.
//
// Maps the merged linear RGB this pipeline produces straight to final display
// sRGB, absorbing white balance, the colour matrix, gamma and the preset's tone
// and colour grading in one step.
//
// Accuracy on the reference pair, measured after the uint8 quantisation:
// 14.0 LSB mean over the whole frame, 9.4 LSB over smooth regions, where
// registration between the two handheld frames cannot bias the number. It is
// deliberately NOT exact -- the preset was measured to contain content-adaptive
// processing, so a colour-only mapping cannot reach zero.
//
// See preset_lut.cpp for how the table was repaired and what is still open;
// the accuracy figures the previous version of this comment quoted were
// measured on training pixels and understated the real error by 6x.
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
