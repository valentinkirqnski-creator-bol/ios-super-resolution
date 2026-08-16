#pragma once
//
// Optional neural alignment path: PWCNet run via Core ML, as an alternative
// flow source to the classical block-matching align() in align.cpp. Feeds
// the same downstream consumer (flow_from_dense_guide -> compute_robustness
// -> merge) rather than replacing any of the robustness/merge math -- see
// the Settings "Use Neural Flow" toggle.
//
// UNTESTED ON DEVICE: this file, its .mm implementation, and the bundled
// PWCNetFlow.mlpackage were built and traced/converted on a non-Apple
// machine (no Xcode/Core ML runtime available to compile or run against).
// The graph conversion was verified to complete cleanly through every op and
// MIL pass; the .mlpackage weight blob itself must still be produced by
// running scratchpad/convert_coreml.py on a Mac. Build, load, and numerical
// correctness on-device have not been exercised.
//
#include "types.h"
#include <vector>

namespace hhsr {

// True once the bundled PWCNetFlow model has been located and loaded
// successfully (lazy, cached — the first call pays the load cost). False if
// the model resource is missing or MLModel construction fails, in which
// case neural_flow_estimate() also returns false and the caller should fall
// back to the classical align() path.
bool neural_flow_available();

// Runs the bundled PWCNet Core ML model on a reference/comparison guide
// image pair. ref_guide/comp_guide must be exactly the resolution the model
// was converted for (1512x2016, 3-channel, values in the same 0..1 range
// compute_guide() produces) -- there is no internal resize, a mismatched
// size fails the call rather than silently distorting the input.
//
// dense_flow_out is resized and filled with the dx plane (guide_h*guide_w
// floats) followed by the dy plane (guide_h*guide_w floats), in guide-pixel
// units -- the exact layout flow_from_dense_guide() (stages.h) expects.
//
// Returns false on any failure (model unavailable, size mismatch, Core ML
// prediction error); dense_flow_out is left empty in that case.
bool neural_flow_estimate(const Image& ref_guide, const Image& comp_guide,
                          std::vector<f32>& dense_flow_out);

} // namespace hhsr
