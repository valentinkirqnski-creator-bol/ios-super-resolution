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
// frame, row 1 = current frame: one noise model per burst, so the rows are equal. Filled by quadrature of
// var(sqrt(max(0, x))) under the raw-level noise model (GCam's fill is behind a virtual dispatch; not
// recovered). Checked against a captured device LUT (dbgpull texid11): rows equal and R / avg-G / B
// semantics confirmed, within ~5% from L = 3/9 up; the three darkest rows differ (12% rms overall).
void build_noise_lut(const float slope[3], const float offset[3], float lut[80]);

// ---- per-capture covariance tuning (sabre_tuning.cc / frame_metadata.cc / sabre_merge.cc) ----
// EvalCurve (FUN_01410b64): piecewise-linear (x, y) knots, clamped to the end knots, upper-bound search,
// y1*t + y0*(1-t). n = number of knots.
float eval_curve(const float* xy, int n, float K);
// EstimateSnr (FUN_02202b64), green channel: signal / sqrt(read + shot*signal); signal <= 0 -> 0.18.
float estimate_snr(float signal, float shot_noise_factor, float read_noise_var);
// The 6 MergeOptions curves (FUN_0141068c) at K = sqrt(num_frames/12)*snr, uploaded (FUN_012c1998) as
// covariance_parameters1 = (1/(p0/p5), 1/(p0*p4), p2, 1/p0), covariance_parameters2 = (1/(p0*p1), 1/p3).
struct CovTuning {
    float p[6];
    float cov1[4];
    float cov2[2];
    float snr = 0.f, K = 0.f;
};
CovTuning cov_tuning(float snr, int num_frames);
// green_clipping_point (FUN_012c1998): FLT_MAX when K*wb_G1 >= 1 (the base frame, and any frame not brighter
// than it), else 0.5*K*(wb_G1 + wb_G2)*(white - 3)/white -- compared against the guide's sqrt-space green.
// K = exposure gain the loader applied to this frame (1 for the base); wb_G = 1 in our green-normalized space.
float green_clip_point(float K, float white_level);
// Per-site ExtractBayer epsilon (1e-4 DN) in our normalized units for a frame scaled by K.
void extract_eps(const Config& cfg, const CFA& cfa, float K, float eps[4]);

// Per-frame guide uniforms.
struct GuideTuning {
    float cov1[4], cov2[2];
    float green_clip;
};
GuideTuning guide_tuning(const CovTuning& cov, float green_clip);
GuideTuning default_guide_tuning();   // the captured device values (K ~ 6.22), clip off

// Uniforms of each pass (per-shot values from the captured 8.x-family device run where the GLSL is silent).
S82GuideParams guide_params(int Wr, int Hr, const S82Cfa& cfa, const GuideTuning& t);

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

// Unblocker (unblocker.cc -> UnblockerRaw16Halide): one R8 texel per 16x16 Bayer tile (= 8x8 guide pixels),
// computed once per burst from the base frame and read by every rejection pass, NEAREST at
// tex_coord * unblocker_scale. All-zero = no detection (also what an absent texture means here).
struct Unblocker {
    int w = 1, h = 1;
    float scale[2] = {1.f, 1.f};
    std::vector<uint8_t> data = std::vector<uint8_t>(1, 0);
};
Unblocker zero_unblocker(int Wr, int Hr);
// UnblockerRaw16Halide on a u16 DN Bayer plane (W x H). Per 8x8 half-res tile (one per 16x16 raw): the two
// green phases' rounded mean / rounded population variance V_raw, the rounded variance V_blur of a 3x3
// halving-add blur of their rounded average, signal = max(mean - black_green, 0),
// m = sqrt(max(V_raw - 4*(read + shot*signal), 0) / (V_blur * 128/9)), weight = trunc(255*clamp(0.2*(m - 0.45),
// 0, 1)) on a 1-tile-apron grid (Halide mirror_image boundary), output = the same 3x3 halving-add blur of the
// weights. green_on_even: greens at Bayer indices 0 and 3 (GRBG / GBRG). Rounding of the halving adds:
// gcam82_rounding = false is the device build that produced the captured map (LMC 8.4: uhadd/urhadd mix,
// fused noise madd) -- bit-exact on all 12969 texels; true = the GCam 8.2.300 decompile (every add urhadd).
Unblocker unblocker_dn(const uint16_t* raw, int W, int H, bool green_on_even, float shot_noise_factor_dn,
                       float read_noise_var_dn2, float black_level_green, bool gcam82_rounding = false);
// The burst's unblocker from our normalized base frame: green sites back to DN with the loader's black /
// white, noise model (slope_g, offset_g in normalized units) scaled to DN (shot*range, read*range^2).
// Returns zero_unblocker when the DNG black / white levels are unknown.
Unblocker compute_unblocker(const Image& base, const Config& cfg, float slope_g, float offset_g);
S82RejParams rej_params(int Wg, int Hg, const FlowTex& flow, float noise_scale, const Unblocker& unb);

// Guide pass output at guide resolution (Wr/2 x Hr/2): ref4 = RGBA16F (fp16-rounded floats), cov = the
// RGB10_A2 kernel covariance as 10-bit codes (3 per pixel).
struct Guide {
    int w = 0, h = 0;
    std::vector<float> ref4;
    std::vector<uint16_t> cov;
};
Guide compute_guide(const Image& raw, const S82Cfa& cfa, const float lut[80], const GuideTuning& t, int threads);
// BaseFrameRefColor pass: bicubic-filtered base guide, sign kept (RGBA16F).
std::vector<float> base_reference(const Guide& base, int threads);
// Residual-shift test (hhsr extension, Config::sabre_resid_shift_enabled; see sabre82_shared.h). Per burst:
// |grad L| of the base reference and the strong-edge threshold (Config::sabre_resid_shift_edge_pct percentile).
// gm is always filled (the rejection pass reads it); `enabled` gates the extra passes.
struct ResidShift {
    bool enabled = false;
    std::vector<float> gm;
    S82ResidParams P{};
};
ResidShift resid_shift_setup(const std::vector<float>& base_ref, int Wg, int Hg, const Config& cfg, int threads);
// R1 + R2 on the CPU: keep (Wg/2 x Hg/2) *= min factor; itg = It * |grad L| from rejection().
void resid_shift_apply(const ResidShift& rs, const std::vector<float>& itg, std::vector<uint8_t>& keep, int threads);

// rejection: also fills itg (Wg x Hg, It * |grad L|) for the residual-shift test when rs.enabled.
void rejection(const std::vector<float>& base_ref, const Guide& alt, const FlowTex& flow, const float lut[80],
               float noise_scale, const Unblocker& unb, const ResidShift& rs, std::vector<uint8_t>& rej,
               std::vector<float>& itg, int threads);
// DilateMask to the half-guide grid (Wg/2 x Hg/2).
void dilate(const std::vector<uint8_t>& rej, int w, int h, std::vector<uint8_t>& keep, int threads);

struct MergeFrame {
    const Image* raw = nullptr;       // normalized Bayer (Wr x Hr)
    std::vector<uint16_t> cov;        // 10-bit codes, guide res x 3
    FlowTex flow;
    std::vector<uint8_t> keep;        // dilated mask (half-guide res); empty for the base
    float eps[4] = {0.f, 0.f, 0.f, 0.f};   // extract_eps of this frame
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
bool metal_base_stages(const Image& ref, const S82Cfa& cfa, const float lut[80], const GuideTuning& t,
                       Guide& guide, std::vector<float>& base_ref);
// metal_frame_stages: guide + kernel of one comparison frame, rejection against the resident base reference,
// dilate. Returns the kernel covariance codes and the dilated mask.
bool metal_frame_stages(const Image& comp, const FlowTex& flow, const S82Cfa& cfa, const float lut[80],
                        const GuideTuning& t, float noise_scale, const Unblocker& unb, const ResidShift& rs,
                        std::vector<uint16_t>& cov, std::vector<uint8_t>& keep);
void metal_end_burst();
bool merge_band_metal(const std::vector<const MergeFrame*>& frames, const BandContext& ctx, int y0, int y1,
                      std::vector<float>& out_rgb);
#endif

}  // namespace sabre82
}  // namespace hhsr
