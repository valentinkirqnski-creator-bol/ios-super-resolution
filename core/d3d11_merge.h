// Direct3D 11 compute-shader merge backend (Windows / AMD / any D3D11 GPU).
//
// First increment of a GPU backend for non-Apple machines: it offloads the
// comparison-frame merge accumulation (Alg. 4, accumulate_comp) -- the bulk of
// the merge cost (N frames x passes) -- to a D3D11 compute shader. The single
// reference pass stays on the CPU. Results are close to, but NOT bit-identical
// to, the CPU/Metal paths (GPU float ordering, exp, round differ).
//
// Only the common configuration is supported; merge_comp_band_d3d11 returns
// false (caller falls back to the CPU accumulate) when the frame uses affine or
// dense flow, per-channel robustness, or has no resident raw pixels.
#pragma once
#ifdef _WIN32
#include "types.h"

namespace hhsr {

// True if a D3D11 compute device is available (lazily created on first call).
bool d3d11_merge_available();

// GPU twin of accumulate_comp: adds this comparison frame's contribution to the
// num/den band in place. Returns false if the configuration is unsupported (no
// GPU work done -- the caller must run the CPU path instead) or on any GPU error.
bool merge_comp_band_d3d11(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                           const Image& robustness, int tile_size,
                           Image& num_band, Image& den_band, int y0, const Config& cfg,
                           int frame_id);

// Drop cached per-frame GPU buffers (call between independent merges).
void d3d11_merge_reset();

}  // namespace hhsr
#endif  // _WIN32
