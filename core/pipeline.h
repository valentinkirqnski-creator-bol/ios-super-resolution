#pragma once
//
// Top-level pipeline entry point, mirroring super_resolution.py:main()/process().
//
#include "types.h"
#include <vector>
#include <functional>

namespace hhsr {

// Progress callback: (stageName, fraction 0..1).
using ProgressFn = std::function<void(const std::string&, float)>;

// Per-comparison-frame data precomputed once and kept resident while the output
// is accumulated band-by-band. Shared with the GPU merge backend.
struct FrameData {
    FlowField flow;
    CovField  covs;
    Image     robustness; // grey resolution
};

// Runs the full Handheld MFSR pipeline on a burst of raw Bayer frames.
//   burst[0]  is the reference frame J_1 (all frames same size, normalized 0..1).
// Returns an RGB image [Hs, Ws, 3] (or [Hs, Ws, 1] in grey mode), NO post-processing.
// Holds the full output in memory — use for small/synthetic inputs.
Image process_burst(const std::vector<Image>& burst, const Config& cfg,
                    const ProgressFn& progress = nullptr);

// Memory-safe variant for full-resolution (up to 48 MP) output: accumulates the
// output in horizontal row-bands and streams them straight to a DNG file, so the
// full output never resides in RAM. Returns a downscaled sRGB-linear preview
// (longest side <= maxPreviewDim) for on-screen display.
Image process_burst_to_dng(const std::vector<Image>& burst, const Config& cfg,
                           const std::string& dng_path, const ProgressFn& progress,
                           int maxPreviewDim = 1536);

} // namespace hhsr
