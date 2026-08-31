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

// Optional sink for the final RGB16 rows as they are encoded into the DNG.
// The app's JPEG export / DNG preview embed used to re-open the DNG it had
// just written and inflate all of it back (seconds at 48MP) to get pixels
// that existed in memory moments earlier. When non-null, the online encode
// loop appends each band here (w*h*3 uint16, ~292MB at 48MP -- transient,
// freed by the caller after export) so the export renders straight from
// memory. Left empty on failure or on the non-online path.
struct Rgb16Sink {
    int w = 0, h = 0;
    std::vector<uint16_t> rgb;
};

// Low-memory mobile path: reads frames from disk one at a time, caches per-frame
// analysis to temp files, streams the output DNG. Peak RAM ≈ 1 ref + 1 comp frame.
Image process_burst_paths_to_dng(const std::vector<std::string>& paths, const Config& cfg,
                                 const std::string& dng_path, const ProgressFn& progress,
                                 int maxPreviewDim = 512,
                                 Rgb16Sink* rgb16_sink = nullptr);

// Same low-memory mobile path, but frames are supplied by a caller-owned loader.
// This lets iOS feed RAW pixel buffers directly while preserving the exact same
// alignment, robustness, merge, and DNG output stages.
Image process_burst_loader_to_dng(int frame_count, const RawFrameLoaderFn& loader,
                                  const Config& cfg, const std::string& dng_path,
                                  const ProgressFn& progress, int maxPreviewDim = 512,
                                  Rgb16Sink* rgb16_sink = nullptr);

// HDR+ mode (Config::hdrplus_mode, hdrplus_mode.cpp): burst align + merge
// from hdr-plus-master instead of the super-resolution pipeline. Output is
// at INPUT resolution (denoise, not upscale), demosaicked bilinearly.
// process_burst and process_burst_to_dng / the loader path branch into
// these when the toggle is on -- callers keep their usual entry points.
Image process_burst_hdrplus(const std::vector<Image>& burst, const Config& cfg,
                            const ProgressFn& progress = nullptr);
Image process_burst_loader_to_dng_hdrplus(int frame_count, const RawFrameLoaderFn& loader,
                                          const Config& cfg, const std::string& dng_path,
                                          const ProgressFn& progress, int maxPreviewDim = 512,
                                          Rgb16Sink* rgb16_sink = nullptr);
Image hdrplus_demosaic_bilinear(const Image& mosaic, const Config& cfg,
                                int num_threads);

// Write accumulated robustness as 8-bit PGM (mean R over comps → gray).
// Path: replace ".dng" with "_robustness<name_suffix>.pgm", or append that.
// name_suffix is "" for the combined mask, "_s1"/"_s2" for the split ones.
bool write_robustness_mask_pgm(const Image& acc_rob, int n_comp_frames,
                               const std::string& dng_path,
                               const char* name_suffix = "");

} // namespace hhsr
