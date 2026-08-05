#pragma once
// Fitted look-up table reproducing the user's fixed external preset.
//
// Fitted from one DNG/JPEG pair of the same burst (handheld_sr_x2.dng /
// IMG_6039.JPG). Maps the merged linear RGB this pipeline produces straight to
// final display sRGB, absorbing white balance, the colour matrix, gamma and the
// preset's tone and colour grading in one step.
//
// Accuracy against that pair: mean 6.06 LSB, RMS 8.23, p95 16.1, p99 26.7,
// 97.2% of variance explained. It is deliberately NOT exact -- the preset was
// measured to contain content-adaptive processing, so the residual in flat
// regions stays near 6 LSB no matter how the fit is sliced. Smooth skies and
// gradients are where that shows.
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
