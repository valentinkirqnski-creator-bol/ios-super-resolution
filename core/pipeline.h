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
using RawFrameLoaderFn = std::function<Image(int index, Config& cfg,
                                             bool is_reference,
                                             int crop_h, int crop_w)>;

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

// Low-memory mobile path: reads frames from disk one at a time, caches per-frame
// analysis to temp files, streams the output DNG. Peak RAM ≈ 1 ref + 1 comp frame.
Image process_burst_paths_to_dng(const std::vector<std::string>& paths, const Config& cfg,
                                 const std::string& dng_path, const ProgressFn& progress,
                                 int maxPreviewDim = 512);

// Same low-memory mobile path, but frames are supplied by a caller-owned loader.
// This lets iOS feed RAW pixel buffers directly while preserving the exact same
// alignment, robustness, merge, and DNG output stages.
Image process_burst_loader_to_dng(int frame_count, const RawFrameLoaderFn& loader,
                                  const Config& cfg, const std::string& dng_path,
                                  const ProgressFn& progress, int maxPreviewDim = 512);

// Write accumulated robustness as 8-bit PGM (mean R over comps → gray).
// Path: replace ".dng" with "_robustness<name_suffix>.pgm", or append that.
// name_suffix is "" for the combined mask, "_s1"/"_s2" for the split ones.
// target_h/target_w: when > 0, the mask is NEAREST-upsampled to that size
// before writing (guide-resolution R -> raw resolution, matching 1.4's
// cli.py cv2.resize(..., INTER_NEAREST) to the output). 0 = write native.
bool write_robustness_mask_pgm(const Image& acc_rob, int n_comp_frames,
                               const std::string& dng_path,
                               const char* name_suffix = "",
                               int target_h = 0, int target_w = 0);

// Write a per-tile optical-flow field as a colour PPM for diagnosis, next to
// the DNG ("<dng>_flow.ppm"). Middlebury-style colour wheel: hue = flow
// DIRECTION, brightness = MAGNITUDE (self-scaled to the field's max). A smooth
// hue gradient means smooth flow (e.g. an aperture slide); hard colour bands/
// blocks mean the flow field itself is broken. Nearest-upsampled to
// target_h/target_w when given (so it overlays the mask), else native tile grid.
bool write_flow_ppm(const FlowField& flow, const std::string& dng_path,
                    int target_h = 0, int target_w = 0);

} // namespace hhsr
