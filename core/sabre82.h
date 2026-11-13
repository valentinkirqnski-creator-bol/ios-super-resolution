#pragma once
// GCam 8.2.300 SABRE (Config::robustness_gcam_enabled): everything after alignment -- guide + structure
// tensor + kernel (RGB10_A2-packed), base reference colour, rejection, dilate, merge, dehomogenize.
// Per-pixel math: core/sabre82_shared.h (shared with core/Sabre82Kernels.metal). Orchestration:
// process_burst_sabre in core/pipeline_paths.cpp. Reference: Documents/super/SABRE_8.2_COMPLETE_REFERENCE.md.
#include "sabre82_shared.h"
#include "types.h"
#include <cstdint>
#include <vector>

namespace hhsr {
namespace sabre82 {

S82Cfa cfa_from(const CFA& cfa);

// Per-channel noise of our normalized planes: var_c(x) = slope[c]*x + offset[c] (DNG NoiseProfile scaled by
// the loader's wb[c]/wb[G] gain g: slope = g*alpha, offset = g^2*beta).
void noise_model_s2(const Config& cfg, float slope[3], float offset[3]);

// 8.2 noise LUT (10 x 2 RGBA, [row][col][4]): per-channel noise variance of the guide's per-quad colour in
// sqrt space -- .x R, .y averaged green 0.5*(G1+G2), .z B, .w 0 -- at perceptual luma L = i/9 (linear L^2),
// before any guide filtering (the rejection's kFilterVarianceScale accounts for that). Row 0 = reference
// frame, row 1 = current frame: one noise model per burst, so the rows are equal. Filled by simulation from
// the raw-level noise model (GCam's fill is behind a virtual dispatch; not recovered).
void build_noise_lut(const float slope[3], const float offset[3], float lut[80]);

// Uniforms of each pass (per-shot values from the captured 8.x-family device run where the GLSL is silent).
S82GuideParams guide_params(int Wr, int Hr, const S82Cfa& cfa);

// GCam flow texture from our raw tile flow: one RGBA16F vector per tile (uv xy, 3x3 min-max extent z),
// 3x3 per-component median first (MedianFilter3x3Halide), NEAREST, flow_scale_offset onto the tile grid.
struct FlowTex {
    int nx = 0, ny = 0;
    float fso[4] = {1.f, 1.f, 0.f, 0.f};
    std::vector<float> data;          // ny*nx*4
    float min_dy_px = 0.f, max_dy_px = 0.f;
};
FlowTex make_flow_tex(const FlowField& flow, int tile_size, int Wr, int Hr, const Config& cfg);
FlowTex zero_flow_tex(int nx, int ny, int tile_size, int Wr, int Hr);
S82RejParams rej_params(int Wg, int Hg, const FlowTex& flow, float noise_scale);

// Guide pass output at guide resolution (Wr/2 x Hr/2): ref4 = RGBA16F (fp16-rounded floats), cov = the
// RGB10_A2 kernel covariance as 10-bit codes (3 per pixel).
struct Guide {
    int w = 0, h = 0;
    std::vector<float> ref4;
    std::vector<uint16_t> cov;
};
Guide compute_guide(const Image& raw, const S82Cfa& cfa, const float lut[80], int threads);
// BaseFrameRefColor pass: bicubic-filtered base guide, sign kept (RGBA16F).
std::vector<float> base_reference(const Guide& base, int threads);
void rejection(const std::vector<float>& base_ref, const Guide& alt, const FlowTex& flow, const float lut[80],
               float noise_scale, std::vector<uint8_t>& rej, int threads);
// DilateMask to the half-guide grid (Wg/2 x Hg/2).
void dilate(const std::vector<uint8_t>& rej, int w, int h, std::vector<uint8_t>& keep, int threads);

struct MergeFrame {
    const Image* raw = nullptr;       // normalized Bayer (Wr x Hr)
    std::vector<uint16_t> cov;        // 10-bit codes, guide res x 3
    FlowTex flow;
    std::vector<uint8_t> keep;        // dilated mask (half-guide res); empty for the base
};
struct BandContext {
    int Wo = 0, Ho = 0, Wr = 0, Hr = 0, Wg = 0, Hg = 0, Wm = 0, Hm = 0;
    S82Cfa cfa{};
    int threads = 0;
};
// Merge every frame (alts, then the base with frame weight 1) for output rows [y0, y1) and dehomogenize:
// out = sum(colour) / max(sum(weight), 1e-7), linear RGB (3 floats per pixel). Metal when available and
// allowed; CPU otherwise or on any GPU failure.
bool merge_band(const std::vector<const MergeFrame*>& frames_alts_then_base, const BandContext& ctx,
                int y0, int y1, bool allow_gpu, std::vector<float>& out_rgb);
// Raw rows [r0, r1) and guide rows [g0, g1) output rows [y0, y1) can read through this frame's flow.
void band_rows(const MergeFrame& f, const BandContext& ctx, int y0, int y1, int& r0, int& r1, int& g0, int& g1);
S82MergeParams merge_params(const MergeFrame& f, const BandContext& ctx, bool is_base, int r0, int r1,
                            int g0, int g1, int out_row0);

#if defined(__APPLE__)
// GPU per-frame stages (core/sabre82_metal.mm); false = not run, the caller uses the CPU stages.
// metal_base_stages: guide + kernel of the base frame and its BaseFrameRefColor, kept resident for the burst;
// returns the guide (host copy) and the base reference colour (host copy) for a CPU fallback.
bool metal_base_stages(const Image& ref, const S82Cfa& cfa, const float lut[80], Guide& guide,
                       std::vector<float>& base_ref);
// metal_frame_stages: guide + kernel of one comparison frame, rejection against the resident base reference,
// dilate. Returns the kernel covariance codes and the dilated mask.
bool metal_frame_stages(const Image& comp, const FlowTex& flow, const S82Cfa& cfa, const float lut[80],
                        float noise_scale, std::vector<uint16_t>& cov, std::vector<uint8_t>& keep);
void metal_end_burst();
bool merge_band_metal(const std::vector<const MergeFrame*>& frames, const BandContext& ctx, int y0, int y1,
                      std::vector<float>& out_rgb);
#endif

}  // namespace sabre82
}  // namespace hhsr
