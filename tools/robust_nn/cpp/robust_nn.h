// robust_nn.h — CPU reference of the robustness refinement net (same per-pixel
// code as the Metal kernels; see robust_nn_shared.h for the input contract).
#pragma once
#include <vector>

#include "robust_nn_shared.h"

namespace robust_nn {

// Refines one comparison frame's analytic robustness r0 -> r_out (all grey res).
// ref, cmp, r0, r_out: width*height floats; flow_xy: 2*width*height floats.
// Optional scratch avoids reallocation across frames. feat_flow_xy (optional): the flow the flow-consistency
// features read instead of flow_xy (e.g. the residual tile flow without a global homography); the warp always
// uses flow_xy.
struct Scratch {
    std::vector<float> J, feat;
};

void refine(const float* ref, const float* cmp, const float* flow_xy, const float* r0, const RNNParams& P,
            float* r_out, Scratch* scratch = nullptr, float* feat_out = nullptr,
            const float* feat_flow_xy = nullptr);

}  // namespace robust_nn
