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
    std::vector<uint32_t> aperture_limited; // ny*nx, 1 = Hessian says 1D/aperture-limited

    FlowField() = default;
    FlowField(int ny_, int nx_) : ny(ny_), nx(nx_),
        flow((size_t)ny_ * nx_ * 2, 0.f),
        aperture_limited((size_t)ny_ * nx_, 0u) {}

    inline f32& dx(int ty, int tx) { return flow[((size_t)ty * nx + tx) * 2 + 0]; }
    inline f32& dy(int ty, int tx) { return flow[((size_t)ty * nx + tx) * 2 + 1]; }
    inline f32 dx(int ty, int tx) const { return flow[((size_t)ty * nx + tx) * 2 + 0]; }
    inline f32 dy(int ty, int tx) const { return flow[((size_t)ty * nx + tx) * 2 + 1]; }
    inline uint32_t& aperture(int ty, int tx) { return aperture_limited[(size_t)ty * nx + tx]; }
    inline uint32_t aperture(int ty, int tx) const { return aperture_limited[(size_t)ty * nx + tx]; }
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

struct IspParams {
    // Manual trim on top of the automatic exposure, in stops.
    float exposure_ev = 0.0f;
    // Where highlight recovery starts, as a fraction of full scale. 1.0 off.
    //
    // The merge writes the DNG pre-white-balanced, so the WB gains are applied
    // BEFORE the 16-bit ceiling. On this sensor those are R x2.06, G x1.0,
    // B x1.84, so a highlight that clipped evenly in raw arrives with R and B
    // pinned at 1.0 and G far below -- hue 300-312, magenta -- and where only
    // green clipped, the complementary green patches. Measured on a raw clip of
    // 0.55: stored (1.000, 0.550, 1.000), saturation 0.45.
    //
    // Anything above the knee is pulled toward neutral at its own peak, so a
    // blown highlight renders white. The per-channel output curve used to hide
    // this by compressing the largest channel hardest; moving the curve onto
    // luminance was correct but removed that accidental cover.
    float highlight_knee = 0.88f;
    // How much of the local (as opposed to global) tone mapping to apply.
    // 0 disables it and leaves a purely global render.
    float local_strength = 0.75f;
    // Highlight compression and shadow lift, applied to the blurred base only.
    float highlight_rolloff = 0.65f;
    float shadow_lift = 0.28f;
    // Display-space black point, applied after the output curve. The log-space
    // shadow lift raises the whole lower range uniformly, which leaves no true
    // black -- measured p1 sat at 22/255 against a reference of 15 until this
    // was added. Renormalised so display white stays at 1.
    float black_point = 0.065f;
    // Red trim on the camera RGB before the matrix. The measured matrix runs
    // about 6% short on red against the reference renders, which reads as a
    // cool, flat image. Red only: also cutting blue overshot and left B at 92
    // against a reference 98.
    float warmth = 0.05f;
    // How much of the measured camera->sRGB matrix to apply, blended toward
    // identity with the neutral response held constant.
    //
    // This was reduced to 0.60 and then 0.75 while chasing teal blues and brown
    // reds, before the cause was found: the output curve running per channel.
    // With that curve moved onto luminance, hue is stable across this parameter
    // -- probing the full chain, sky blue lands at 214 degrees at 0.75 and 212
    // at 1.0, foliage at 122 and 125 -- and only saturation moves (sky 0.85 to
    // 0.95). So the reduction was never fixing the cast, only draining colour,
    // which is what made the render look flat. Back to the full measured matrix.
    float colour_strength = 1.0f;
    // S-curve in display space, applied to luminance so hue is preserved.
    float contrast = 0.55f;
    // Saturation-dependent boost: muted colours gain, already-saturated ones
    // barely move. This is what separates vibrance from RGB *= k.
    float vibrance = 0.50f;
    // Flat multiplier applied after vibrance.
    float saturation = 1.0f;
    // Re-adds the detail layer above unity for local micro-contrast. Distinct
    // from sharpening: no high-pass, no halos, no noise amplification.
    float local_contrast = 0.20f;
    // Hold saturation and hue in the skin band. Without a face detector this is
    // the cheap substitute, and it is what stops strong tone mapping plus
    // vibrance turning skin orange.
    bool skin_protect = true;
    bool enabled = true;
};

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
    std::vector<f32> bm_tile_size_factors = {1.f, 1.f, 1.f, 0.5f};
    std::vector<int> bm_search_radii = {1, 4, 4, 4};
    std::vector<std::string> bm_metrics = {"L1", "L2", "L2", "L2"};
    int  ica_n_iter = 3;
    // Run ICA after block matching on EVERY pyramid level, not only the finest.
    //
    // The reference implementation (alignment.py, align_lvl) does block matching
    // then align_lvl_ica at each level. This port refined only once, at the
    // finest, so every coarse level emits integer-only flow with up to half a
    // pixel of residual, which upscale_flow then multiplies by the upsampling
    // factor.
    //
    // That amplification is not unchecked -- each level re-searches, so it only
    // matters if the arriving error exceeds that level's search radius. With
    // factors {1,2,4,4} and radii {1,4,4,4} the arriving error is 0.5*factor:
    // 2px into levels 2 and 1 against a radius of 4, comfortable; but 1px into
    // level 0 against a radius of 1. The finest level runs with exactly zero
    // slack, so anything else that adds error -- aliasing from the box-filter
    // decimation, motion varying inside a tile, the multi-hypothesis step
    // taking a neighbour's vector -- lands outside what it can recover.
    //
    // Refining per level makes what arrives sub-pixel and restores that margin.
    // It also hands the finest-level ICA a far better starting point, which
    // matters because ICA is a local gradient method with a finite basin of
    // attraction. It does NOT improve final sub-pixel precision in raw terms:
    // the last refinement still happens on a half-resolution grey.
    //
    // Restricted to the decimate grey. It is half raw resolution, so both the
    // accumulated error and that +/-1 budget are twice as coarse in scene terms
    // as on the full-res FFT grey -- and keeping FFT on the single finest-level
    // refinement leaves its output bit-identical to before.
    bool align_ica_per_level = true;

    // Extend per-level ICA to the full-res FFT grey as well.
    //
    // The restriction above was for bit-compatibility, not for any technical
    // reason, and it leaves the FFT path in the worse of the two positions:
    // levels 3..1 emit integer-only flow, level 1 runs at half scale so its
    // residual is up to 0.5px there, upscale_flow multiplies that by 2, and
    // exactly 1.0px arrives at a finest level whose search radius is 1. The
    // budget is spent before level 0 begins, so anything else that adds error
    // -- upscale_flow_460 taking a neighbour's vector, aliasing from the
    // decimation, motion varying inside a tile -- lands outside +/-1 and is
    // never corrected. That is what a tile-shaped displacement in the output
    // looks like.
    //
    // Costs memory rather than time: the reference gradient/Hessian cache goes
    // resident for the burst, and the FFT grey has 4x the pixels of the
    // decimate grey, so roughly +120MB at 12MP. It saves compute, since those
    // gradients are currently rebuilt per frame.
    bool align_ica_per_level_fft = false;

    // True when ICA should run on every pyramid level rather than only the
    // finest.
    bool ica_every_level() const {
        return align_ica_per_level &&
               (grey_method == GreyMethod::Decimate || align_ica_per_level_fft);
    }

    // On the FFT grey, run per-level ICA on the COARSE levels only and leave
    // the finest to the single pass that already exists.
    //
    // The per-level cache holds ref + gradx + grady as full-resolution float
    // buffers for each level. On the decimate grey that is about 180MB across
    // four levels at 48MP; on the FFT grey, which has four times the pixels,
    // level 0 alone is about 576MB and the set is about 730MB. Allocation then
    // fails, prep_level_ica_gpu returns false, align_metal returns false, and
    // pipeline_paths skips the frame -- every frame, so nothing merges at all.
    //
    // Skipping level 0 in the loop costs nothing, because the finest level is
    // refined either way: the existing single pass allocates those same
    // buffers. Levels 1..3 add only about a quarter of full resolution, and
    // they are where the benefit is -- integer-only flow originates on the
    // coarse levels, and that is what arrives at level 0 with its budget spent.
    bool ica_per_level_coarse_only() const {
        return align_ica_per_level_fft && grey_method == GreyMethod::FFT;
    }

    // Override the finest level's block-match search radius. 0 keeps
    // bm_search_radii[0].
    //
    // The default of 1 assumes the flow arriving from level 1 is already
    // sub-pixel, which is only true when ICA has run there. Raising it buys
    // margin directly, at (2r+1)^2 / 9 times the finest level's search cost:
    // 2.8x at radius 2, 5.4x at radius 3. The finest level is the most
    // expensive one, so this is not free.
    int align_fine_search_radius = 0;
    // Median-filter a tile's displacement component when it disagrees with its
    // neighbours by more than flow_regularize_threshold pixels.
    //
    // The aperture problem is local, not global: a tile holding one long edge
    // cannot see its along-edge motion, but that motion is shared with the
    // neighbours and one tile over there is usually a corner or an end that
    // determines it. Applied per component, so on a horizontal edge the
    // unobservable dx is repaired and the well-determined dy is left alone.
    //
    // Aimed at static scenes, where the true field is one rigid transform and
    // any tile-to-tile disagreement is error. Across a genuine motion boundary
    // the neighbours' median belongs to neither side, so leave it off when the
    // subject moves.
    bool  flow_regularize_enabled = false;
    float flow_regularize_threshold = 1.0f;
    // A tile is considered aperture-limited when lambda2/lambda1 is below this
    // ratio. Higher catches more edge-like tiles; lower limits repair to very
    // one-dimensional tiles.
    float flow_regularize_aperture_ratio = 0.15f;
    // Test switch: force merge robustness to zero for every tile that passes
    // the same Hessian aperture-limited test, instead of repairing the flow.
    bool  flow_reject_1d_enabled = false;

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
    // Experimental: when the alignment grey is the full-resolution FFT image,
    // warp/upscale the half-res guide statistics back to the raw grid and build
    // the robustness mask at raw resolution. The 2x2 decimated grey path keeps
    // the Google/Wronski half-res guide mask.
    bool  robustness_full_res_fft = false;

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

    // JPEG / preview rendering. Never affects the linear DNG, which is always
    // written from the unmodified merge.
    IspParams isp;
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
