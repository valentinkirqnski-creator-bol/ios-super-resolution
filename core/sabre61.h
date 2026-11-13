#pragma once
// GCam 6.1.021 SABRE (Config::robustness_gcam_enabled): the full multi-frame pipeline after alignment --
// guide + structure tensor + kernel, rejection, dilate, accumulated weights, merge, post-merge denoiser.
// Per-pixel math: core/sabre61_shared.h (shared with core/Sabre61Kernels.metal). Orchestration:
// process_burst_sabre61 in core/pipeline_paths.cpp. Reference: Documents/super/SABRE_6.1_COMPLETE_REFERENCE.md.
#include "sabre61_shared.h"
#include "types.h"
#include <cstdint>
#include <vector>

namespace hhsr {
namespace sabre61 {

// Site positions of R, G1 (top row), G2, B inside a quad, from our CFA.
S61Cfa cfa_from(const CFA& cfa);

// Per-channel noise of our normalized (S2) planes: var_c(x) = slope[c]*x + offset[c]. The DNG
// NoiseProfile describes the sensor before the loader's wb[c]/wb[G] gain g_c, so slope = g_c*alpha,
// offset = g_c^2*beta.
void noise_model_s2(const Config& cfg, float slope[3], float offset[3]);

// The 10x2 RGBA noise LUT ([row][col][4]), filled by simulation as GCam's fill defines it: the
// statistics the guide shader itself produces on a flat noisy patch at perceptual luma L = i/9
// (linear signal L^2 on every channel) with our raw-level noise (our SABRE input is the full-res Bayer,
// so GCam's ComputeDownsampledRgbNoiseModels reduction is the identity here).
//   row 0: .xyz = variance of reference_color (S1 YCoCg), .w = expected std_dev (noise floor)
//   row 1: .x = var(|G1 - G2|) of a quad in perceptual space, .y = expected squared gradient of the
//          rotated structure tensor ((cxx + cyy) / 2), .zw = 0
void build_noise_lut(const float slope[3], const float offset[3], const S61Cfa& cfa, float lut[80]);

// Denoiser yuv_noise_model: per-channel variance propagated through the funky-YUV luma row.
void yuv_noise_model(const float slope[3], const float offset[3], float out[4]);

// GCam flow texture from our raw tile flow (3x3 median-filtered, as ConvertAlignmentHalide): one RGBA vector
// per tile (uv offset xy, 3x3 min-max
// extent z), NEAREST, with flow_scale_offset mapping tex_coord onto the tile grid.
struct FlowTex {
    int nx = 0, ny = 0;
    float fso[4] = {1.f, 1.f, 0.f, 0.f};
    std::vector<float> data;          // ny*nx*4
    float min_dy_px = 0.f, max_dy_px = 0.f;   // raw-pixel vertical displacement range (merge slicing)
};
FlowTex make_flow_tex(const FlowField& flow, int tile_size, int Wr, int Hr, const Config& cfg);
// Base frame: zero flow on the same grid.
FlowTex zero_flow_tex(int nx, int ny, int tile_size, int Wr, int Hr);

// Guide pass output at guide resolution (Wr/2 x Hr/2): ref4 (RGBA16F) and cov3 (RGBA16F .xyz),
// both fp16-quantized.
struct Guide {
    int w = 0, h = 0;
    std::vector<float> ref4;
    std::vector<float> cov3;
};
Guide compute_guide(const Image& raw, const S61Cfa& cfa, const float lut[80], int threads);

void rejection(const Guide& base, const Guide& alt, const FlowTex& flow, const float lut[80],
               std::vector<uint8_t>& rej, int threads);
void dilate(const std::vector<uint8_t>& rej, int w, int h, std::vector<uint8_t>& keep, int threads);
void copy_mask_accumulate(const std::vector<uint8_t>& keep, std::vector<uint8_t>& acc);

// Everything the merge needs from one frame.
struct MergeFrame {
    const Image* raw = nullptr;       // normalized Bayer (Wr x Hr)
    std::vector<uint16_t> cov_h;      // fp16 bits, guide res x 3
    FlowTex flow;
    std::vector<uint8_t> keep;        // dilated mask (empty for the base)
};

struct BandContext {
    int Wo = 0, Ho = 0, Wr = 0, Hr = 0, Wg = 0, Hg = 0;
    S61Cfa cfa{};
    float yuv_noise[4] = {0.f, 0.f, 0.f, 0.f};
    const std::vector<uint8_t>* acc = nullptr;   // accumulated frame weights (guide res, R8)
    int threads = 0;
    bool use_rejection = true;
};
// Merge every frame (alts in order, base last) and run the 3-level denoiser for output rows [y0, y1).
// Writes linear S2 RGB (3 floats per pixel) for exactly those rows. Uses Metal when available and
// allowed; falls back to the CPU on any GPU failure.
bool merge_denoise_band(const std::vector<const MergeFrame*>& alts, const MergeFrame& base,
                        const BandContext& ctx, int y0, int y1, bool allow_gpu,
                        std::vector<float>& out_rgb);

// Halo rows merged above/below a band for the denoiser: its levels reach 1 + 2 + 4 rows; each level keeps
// one spare row more than the next one reads (9 / 6 / 0), so the bilinear taps that land a hair off a texel
// centre never fall outside the held rows.
constexpr int kDenoiseHalo = 10;

#if defined(__APPLE__)
// GPU per-frame stages (core/sabre61_metal.mm). false = not run; the caller uses the CPU stages.
// metal_base_guide: guide + kernel of the base frame; keeps the base guide and a zeroed accumulated-weights
// plane resident for the burst; returns the guide (host copy, for a CPU fallback) and the kernel as fp16.
bool metal_base_guide(const Image& ref, const S61Cfa& cfa, const float lut[80], Guide& out,
                      std::vector<uint16_t>& cov_h);
// metal_frame_stages: guide + kernel of one comparison frame, rejection against the resident base guide,
// dilate, copy_mask into the accumulated weights (acc is synced from and back to the host copy).
bool metal_frame_stages(const Image& comp, const FlowTex& flow, const S61Cfa& cfa, const float lut[80],
                        std::vector<uint16_t>& cov_h, std::vector<uint8_t>& keep, std::vector<uint8_t>& acc);
void metal_end_burst();
#endif

// Raw rows [r0, r1) and guide rows [g0, g1) a merge band [m0, m1) can read through this frame's flow.
void band_rows(const MergeFrame& f, const BandContext& ctx, int m0, int m1, int& r0, int& r1, int& g0, int& g1);

}  // namespace sabre61
}  // namespace hhsr
