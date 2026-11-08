#pragma once
// Consensus refiner weights (core/cref.h). PLACEHOLDER: no trained model yet --
// kCrefTrained = false makes cref_refine a no-op. Overwritten by
// tools/cref/export_weights.py after training. Layout: PyTorch [cout][cin][3][3].
namespace hhsr {
constexpr bool kCrefTrained = false;
inline const float kCrefW0[8 * 6 * 9] = {};
inline const float kCrefB0[8] = {};
inline const float kCrefW1[8 * 8 * 9] = {};
inline const float kCrefB1[8] = {};
inline const float kCrefW2[8 * 8 * 9] = {};
inline const float kCrefB2[8] = {};
inline const float kCrefW3[8 * 8 * 9] = {};
inline const float kCrefB3[8] = {};
inline const float kCrefWO[8] = {};
inline const float kCrefBO = 0.f;
}  // namespace hhsr
