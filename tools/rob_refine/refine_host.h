#pragma once
#include "types.h"

// Test-harness replacement for Core ML.
//
// robustness_nn.mm reaches Core ML directly, which exists only on Apple
// platforms. To exercise the SHIPPED refinement path -- the feature builder,
// the strip loop, the bounded multiply -- anywhere else, the harness links
// host_stubs.cpp instead of robustness_nn.mm and installs an evaluator here.
// Nothing in the app knows this exists.
namespace hhsr {

using RefineHostFn = bool (*)(const Image& feat, Image& out);

RefineHostFn host_refine_fn();
void robustness_refine_set_host(RefineHostFn fn);

}  // namespace hhsr
