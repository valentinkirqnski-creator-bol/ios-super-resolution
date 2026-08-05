#pragma once
#include "types.h"

namespace hhsr {

// out_brightness, when non-null, receives the mean that was computed here.
// The caller's status line needs the same value, and at 12MP a second scan of
// the reference costs as much as this whole function. Handing the result back
// keeps it bit-identical -- it is the same sum, not a recomputation.
void tune_config_snr(const Image& ref_raw, Config& cfg, f32* out_brightness = nullptr);

} // namespace hhsr
