#pragma once
// SRSR whole-pipeline entry (Config::srsr_enabled). See srsr_pipeline.cpp.
#include "pipeline.h"
#include "srsr_types.h"
#include <string>

namespace hhsr {

// Runs the GCam 6.1 SABRE pipeline (SRSRKernels.metal) on the burst the loader
// supplies and streams the merged result into a LinearRaw DNG at `dng_path`.
// Returns the preview on success; on failure an empty Image, with the reason in
// *why (the caller then runs the normal pipeline).
Image process_burst_srsr_to_dng(int frame_count, const RawFrameLoaderFn& loader,
                                const Config& cfg, const std::string& dng_path,
                                const ProgressFn& progress, int maxPreviewDim,
                                std::string* why);

namespace srsr {

// Raw noise of one frame as the loader delivers it: Var(x) = alpha*x + beta on the
// sensor's [0,1] signal (DNG NoiseProfile, per CFA colour R,G,B), then x is
// multiplied by wb[c] (pre-whitening, wb[c]/wb[G]) and by the exposure-match gain.
struct FrameNoise {
    float alpha[3], beta[3], wb[3];
    float gain;
};
FrameNoise frame_noise(const Config& cfg, float gain);

// Monte-Carlo 10x2 SABRE noise LUT for frame `cur` (row 1: its own guide) compared
// against the base frame (row 0: rejection). See srsr_types.h for the definitions.
void build_noise_lut(const FrameNoise& base, const FrameNoise& cur,
                     float green_difference_impact, SRSRNoiseLUT& out);

// SABRE yuv_noise_model (STAGE 4d) of the base frame: x, y, z, w.
void yuv_noise_model(const FrameNoise& base, float out[4]);

} // namespace srsr
} // namespace hhsr
