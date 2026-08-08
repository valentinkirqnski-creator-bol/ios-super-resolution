#pragma once
//
// Core data types for the Handheld Multi-Frame Super-Resolution port.
//
// This is a faithful C++ CPU re-implementation of the algorithm from
// Wronski et al. (SIGGRAPH 2019), ported from the Python/Numba reference in
// this repository. The types are intentionally simple (flat row-major float
// buffers) so the same data can later be uploaded to a GPU (Vulkan SSBOs)
// without changing the algorithm code.
//
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace hhsr {

using f32 = float;

// Row-major image / tensor with an arbitrary number of interleaved channels.
struct Image {
    int h = 0;
    int w = 0;
    int c = 1;
    std::vector<f32> data;

    Image() = default;
    Image(int h_, int w_, int c_ = 1) : h(h_), w(w_), c(c_), data((size_t)h_ * w_ * c_, 0.f) {}

    inline f32&       at(int y, int x, int ch = 0)       { return data[((size_t)y * w + x) * c + ch]; }
    inline const f32& at(int y, int x, int ch = 0) const { return data[((size_t)y * w + x) * c + ch]; }

    inline bool inside(int y, int x) const { return y >= 0 && y < h && x >= 0 && x < w; }
    inline size_t size() const { return data.size(); }
};

// Per-tile optical flow field: shape [nTilesY, nTilesX, 2] (dx, dy).
struct FlowField {
    int ny = 0;
    int nx = 0;
    std::vector<f32> flow; // ny*nx*2

    FlowField() = default;
    FlowField(int ny_, int nx_) : ny(ny_), nx(nx_), flow((size_t)ny_ * nx_ * 2, 0.f) {}

    inline f32& dx(int ty, int tx) { return flow[((size_t)ty * nx + tx) * 2 + 0]; }
    inline f32& dy(int ty, int tx) { return flow[((size_t)ty * nx + tx) * 2 + 1]; }
    inline f32 dx(int ty, int tx) const { return flow[((size_t)ty * nx + tx) * 2 + 0]; }
    inline f32 dy(int ty, int tx) const { return flow[((size_t)ty * nx + tx) * 2 + 1]; }
};

// Per-grey-pixel 2x2 covariance field (steerable kernels): [h, w, 4] = xx,xy,yx,yy.
struct CovField {
    int h = 0;
    int w = 0;
    std::vector<f32> cov;

    CovField() = default;
    CovField(int h_, int w_) : h(h_), w(w_), cov((size_t)h_ * w_ * 4, 0.f) {}

    inline f32* at(int y, int x) { return &cov[((size_t)y * w + x) * 4]; }
    inline const f32* at(int y, int x) const { return &cov[((size_t)y * w + x) * 4]; }
};

enum class GreyMethod { FFT, Decimate };
enum class KernelShape { Iso, Steerable };
enum class SelectionLaw { HardThreshold, Linear };

// Block-matching tile sizes are expressed in GREY pixels, and the two grey
// methods do not share a resolution: the FFT grey is full raw size, the 2x2
// decimate grey is half. So tile 16 spans 16 raw pixels on the FFT path and 32
// on the decimate path -- four times the scene area, averaging away that much
// more of any motion that varies inside one tile (parallax, depth steps, a
// moving subject). A robustness mask computed at half resolution cannot see the
// resulting sub-pixel error, but the merge can, because sample placement is
// exactly what super-resolution depends on.
//
// Halving the tile on the decimate path makes both cover the same scene
// footprint. Applied ONLY to alignment: the merge's tile size comes from
// bm_tile_sizes[0] read in pipeline.cpp / pipeline_paths.cpp and is deliberately
// left alone. The FFT path is untouched because its divisor is 1 by definition.
//
// Floored at 8: align_metal accepts only 8/16/32/64 and returns false for
// anything else, which would silently drop the whole burst onto the CPU
// aligner. The coarsest level is already 8, so it stays 8.
inline int align_tile_scaled(int ts, bool match_scene, GreyMethod gm) {
    if (!match_scene || gm != GreyMethod::Decimate) return ts;
    const int t = ts / 2;
    return t < 8 ? 8 : t;
}

// 2x2 Bayer CFA pattern (indices into {R=0,G=1,B=2}).
struct CFA {
    uint8_t p[2][2] = {{0, 1}, {1, 2}}; // default RGGB
};

// Full pipeline configuration, mirroring configs/default.yaml.
struct Config {
    float scale = 1.0f;
    // Centre-crop the Bayer input by this factor before SR, which is how zoom
    // is realised: at 2.0 a 12 MP sensor frame becomes a 3 MP crop that SR takes
    // back to 12 MP. Continuous rather than integral, so any magnification works
    // on the same principle. 1.0 disables it.
    float input_crop_zoom = 1.f;
    bool  bayer_mode = true;
    GreyMethod grey_method = GreyMethod::FFT;

    // Noise model: sigma^2 = alpha * I + beta   (already scaled for ISO).
    // Defaults are Pixel-ish fallbacks; overwritten from DNG NoiseProfile (0xC761).
    float alpha = 1.80710882e-4f;
    float beta  = 3.1937599182128e-6f;
    bool  has_noise_profile = false;

    // Alignment (coarse-to-fine handled internally).
    std::vector<int> bm_factors      = {1, 2, 4, 4};
    std::vector<int> bm_tile_sizes   = {16, 16, 16, 8}; // filled by SNR when tile_size=SNR_based
    // Halve alignment tiles on the half-resolution decimate grey so they cover
    // the same scene area as on the FFT grey. See align_tile_scaled above.
    // Changes alignment output, so it ships off.
    bool  align_tile_match_scene = false;
    std::vector<f32> bm_tile_size_factors = {1.f, 1.f, 1.f, 0.5f};
    std::vector<int> bm_search_radii = {1, 4, 4, 4};
    std::vector<std::string> bm_metrics = {"L1", "L2", "L2", "L2"};
    int  ica_n_iter = 3;
    int  alignment_tile_size = 0; // 0 = SNR auto; otherwise force 8/16/32/64.
    // Off: alignment matches d5215ec, which had no thumbnail pre-alignment pass.
    // With this false the plan stays empty, so every frame enters align() with a
    // zero initial transform and frame 0 stays the reference.
    bool global_prealignment_enabled = false;
    bool global_prealignment_choose_reference = true;
    float global_prealignment_rotation_range_deg = 0.0f;
    float global_prealignment_rotation_step_deg = 0.25f;
    int   global_prealignment_max_shift = 24;       // thumbnail pixels
    int   global_prealignment_thumb_max_dim = 320;

    // Robustness (Eq. 5: R = s·exp(-d²/σ²) - t). Match configs/default.yaml.
    bool  robustness_enabled = true;
    bool  robustness_save_mask = true;
    float r_t  = 0.12f;
    float r_s1 = 2.0f;
    float r_s2 = 12.0f;
    float r_Mt = 0.8f;
    // Scales the estimated sensor noise variance subtracted before the HF
    // loss ratio. >1 assumes more noise, so less of the local variance counts
    // as signal and fewer areas are flagged as high-frequency detail; <1 is
    // more aggressive. 1.0 leaves the estimate as measured.
    // Minimum texture SNR: signal variance must exceed this multiple of the
    // noise floor before a pixel is treated as real high-frequency detail.
    // loss landed, then a hardcoded kMinTextureSnr = 4; configurable again.
    // High-frequency artifact rejection (Wronski et al., "High Frequency
    // Artifacts Removal"). Block matching cannot resolve repetitive fine
    // texture -- the aperture problem -- so it locks onto the wrong period and
    // produces blocky artifacts. Rejected only where BOTH hold:
    //
    //   1. the patch is almost entirely high-frequency: most of its local
    //      variance disappears under a low-pass filter, and
    //   2. the alignment vector field varies a lot locally -- the same r_Mt
    //      test that drives the motion prior, as the paper specifies.
    //
    // Both are needed. Hair and fur are high-frequency but track cleanly, so
    // condition 2 spares them. A flat noisy wall has unstable flow but no real
    // high-frequency signal, so condition 1 spares it.
    //
    // Off by default: it changes which pixels merge, so enable it deliberately.
    bool  hf_artifact_removal_enabled = false;
    // loss = 1 - variance_lowpass / variance_original, both noise-corrected.
    // Above this the patch counts as high-frequency. Reference points measured
    // on typical content: flat wall ~0.04, face ~0.19, brick ~0.83,
    // checkerboard ~0.92.
    float hf_variance_loss_threshold = 0.75f;
    // The patch is skipped unless its signal variance exceeds this multiple of
    // the estimated sensor noise variance. This is what stops ISO 6400 noise
    // reading as high-frequency detail, and it scales with brightness through
    // the noise model rather than being a fixed number.
    float hf_min_texture_snr = 4.0f;
    bool  motion_edge_rejection_enabled = true;
    float motion_edge_threshold = 0.025f;
    float motion_edge_residual_threshold = 2.5f;
    float motion_edge_noise_floor_multiplier = 1.0f;
    int   motion_edge_neighborhood_radius = 1;

    // accumulated_robustness_denoiser.merge — on in 460-main params.py
    bool  accumulated_robustness_denoiser_enabled = true;
    float acc_rob_rad_max = 2.0f;
    float acc_rob_max_multiplier = 8.0f;
    // How the burst is merged. 0 = pick by working-set size, 1 = force banded
    // (accumulate a band at a time, every frame resident across the merge),
    // 2 = force online (one full-size accumulator, each frame released as soon
    // as it is merged, so the working set is flat in frame count).
    //
    // Not simply "online is better": the online accumulator scales with OUTPUT
    // pixels, so at 2x it costs four times what it does at 1x and loses to
    // banding until the burst is long. 0 compares the two and picks.
    int   merge_arch = 0;
    // Adapt the reference-kernel enlargement continuously to the accumulated
    // robustness, instead of the reference implementation's step. Off gives the
    // reference behaviour exactly, including the accumulator overwrite, which is
    // what tools/compare_dng.py needs to line up with the Python.
    bool  acc_rob_adaptive = true;
    // Only consulted when acc_rob_adaptive is false. 2 is the reference's own
    // value for the merge variant; its median and gauss blocks say 8, and taking
    // 8 from those is what made the step fire on every pixel of every burst.
    float acc_rob_max_frame_count = 2.0f;
    // Total frames in the burst, filled in by the pipeline rather than tuned.
    // The reference-kernel enlargement is derived from this and the accumulated
    // robustness, so there is no threshold to set. 0 means unknown, which
    // disables the enlargement rather than guessing.
    int   burst_frame_count = 0;

    // Merge / steerable kernels.
    KernelShape  kernel = KernelShape::Steerable;
    SelectionLaw selection = SelectionLaw::HardThreshold;
    bool  snr_auto_tune = true; // Python always runs update_snr_config
    float k_detail  = 0.17f;  // SNR lerp [0.33, 0.25] when snr_auto_tune
    float k_denoise = 0.0f;   // SNR lerp [5.0, 3.0] when snr_auto_tune
    float D_th      = 0.76f;  // overwritten by SNR lerp [0.81, 0.71]
    float D_tr      = 1.12f;  // overwritten by SNR lerp [1.24, 1.0]
    float k_stretch = 4.0f;
    float k_shrink  = 2.0f;

    CFA cfa;
    // Raw AsShot-style gains (cam_mul), same as Python camera_whitebalance.
    // Load applies k = wb[c]/wb[G]; robustness guide undoes with wb[c] (raw).
    float white_balance[3] = {1.f, 1.f, 1.f};
    // True after load_raw applies per-channel black + WB like Python utils_dng.
    // Merge output is then already white-balanced — preview/bake/DNG AsShot must
    // not apply those gains again (use 1,1,1 for display / private tag).
    bool  raw_prewhitened = false;
    // Ref-frame blacks (per color R,G,B) + white; reused for all burst frames.
    bool  has_black_levels = false;
    float black_levels[3] = {0.f, 0.f, 0.f};
    float white_level = 0.f;

    int   orientation = 1;
    bool  has_color_matrix = false;
    float color_matrix[9] = {1,0,0, 0,1,0, 0,0,1};
    bool  has_cam_to_srgb = false;
    float cam_to_srgb[9] = {1,0,0, 0,1,0, 0,0,1};
    bool  bake_srgb = false;
    // Direct RAW app path: full-res streaming can reload the captured uint16 RAW
    // file instead of spilling/reloading an extra normalized-float cache copy.
    bool  stream_comp_raw_from_loader = false;

    std::string camera_make;
    std::string camera_model;

    int num_threads = 0;      // 0 => hardware_concurrency
    bool use_gpu = false;     // opt-in Vulkan compute merge (experimental)
};

inline f32 clampf(f32 v, f32 lo, f32 hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline f32 smoothstepf(f32 edge0, f32 edge1, f32 x) {
    if (edge1 <= edge0) return x >= edge1 ? 1.f : 0.f;
    f32 t = clampf((x - edge0) / (edge1 - edge0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

} // namespace hhsr
