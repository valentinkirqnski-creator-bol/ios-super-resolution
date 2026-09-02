#pragma once
//
// Declarations for each pipeline stage, mirroring the modules of the Python
// reference (grey/pyramid, alignment, robustness, kernels, merge).
//
#include "types.h"
#include <vector>
#include <complex>
#include <string>

namespace hhsr {

// ---- grey_pyramid.cpp (FFT helpers, exposed for align.cpp) ---------------
void fft1d(std::vector<std::complex<f32>>& a, bool inverse, std::vector<std::complex<f32>>* dft_buf = nullptr);
void fft2d(std::vector<std::complex<f32>>& data, int h, int w, bool inverse,
           std::vector<std::complex<f32>>* row_buf = nullptr,
           std::vector<std::complex<f32>>* dft_buf = nullptr);
void fftshift2d(std::vector<std::complex<f32>>& data, int h, int w);

// Torch-style real 2D FFT packing: out is [h, w/2+1] complex (row-major).
// Built on the same vDSP-backed fft1d as fft2d.
void rfft2(const f32* in, int h, int w, std::vector<std::complex<f32>>& out);
void irfft2(const std::vector<std::complex<f32>>& in, int h, int w, std::vector<f32>& out);
void fftshift2d_real(std::vector<f32>& data, int h, int w);

// ---- grey_pyramid.cpp ---------------------------------------------------
Image compute_grey_decimate(const Image& raw, bool bayer_mode,
                            bool lowpass = false);
Image compute_grey_fft(const Image& raw);
Image compute_grey(const Image& raw, bool bayer_mode, GreyMethod method,
                   bool decimate_lowpass = false);

// Re-express a flow field estimated on the grey onto a raw-resolution tile
// grid, scaling displacements by the resolution ratio.
//
// Alignment runs on the grey, which is half resolution when the Bayer quad
// average is used. Robustness and merge both index the flow as
// raw_coordinate / tile_size and add the displacement in raw pixels, so the
// field has to be converted before either sees it. A no-op when the grey is
// already full resolution.
// r_Mt is the motion-prior threshold, in RAW pixels. The prior is measured
// here rather than carried, because only this function knows the grey-to-raw
// scale needed to express the span in those units. num_threads as in Config.
FlowField flow_to_raw_tile_grid(const FlowField& flow, int raw_h, int raw_w,
                                int grey_h, int grey_w, int tile_size,
                                f32 r_Mt, int num_threads,
                                int guide_tile_size = 0);

// Boundary-selected half-pitch refinement of a raw-grid flow field: fine cells
// whose four surrounding tile vectors agree keep the bilinear blend
// (first-order faithful smooth-region behaviour -- see FlowField::fine_flow
// for bounds); cells at a motion disagreement get whichever
// single tile vector best explains the alignment grey there. Fills
// FlowField::fine_*; sample_bilinear and the GPU hosts consume it
// transparently. Run after flow_to_raw_tile_grid, while the greys are alive.
// Full-res ICA polish of a raw-grid flow field on band-limited full-res greys
// (Config::align_fullres_polish). Metal on device, CPU on host. Returns false
// (flow untouched) when it cannot run. Call after flow_to_raw_tile_grid,
// before flow_densify_boundary_select.
bool flow_fullres_ica_polish(const Image& ref_grey_full, const Image& mov_grey_full,
                             FlowField& flow, int tile_size, const Config& cfg);
// Dense per-cell Lucas-Kanade refinement of the flow field, at the finest
// lattice pitch that fits Config::flow_dense_lk_max_mb. Dispatched from
// flow_densify_boundary_select when Config::flow_dense_lk_enabled.
// Lattice the dense refinement will fill: the finest pitch = tile_size/div
// whose field fits Config::flow_dense_lk_max_mb. False when none fits, in
// which case the coarse field is left alone. Shared by the CPU body and the
// Metal twin so they cannot fill different lattices.
bool flow_dense_lk_lattice(int raw_h, int raw_w, int tile_size,
                           const Config& cfg,
                           int& div, int& fny, int& fnx, f32& pitch);

void flow_densify_lucas_kanade(FlowField& flow,
                               const Image& ref_grey, const Image& mov_grey,
                               int raw_h, int raw_w, int tile_size,
                               const Config& cfg);

void flow_densify_boundary_select(FlowField& flow,
                                  const Image& ref_grey, const Image& mov_grey,
                                  int raw_h, int raw_w, int tile_size,
                                  const Config& cfg);

// 1 where the flow field is irregular over the 3x3 tile neighbourhood -- the
// r_Mt test (Wronski et al. Eq. 7/8, literally: raw max-min span, no
// detrending). sx/sy scale the stored displacements into r_Mt's units.
std::vector<uint32_t> compute_motion_irregular(const FlowField& flow, f32 Mt,
                                               f32 sx, f32 sy, int num_threads);

struct Pyramid { std::vector<Image> levels; std::vector<int> abs_factors; };
Pyramid build_pyramid(const Image& grey, const std::vector<int>& factors);

Image compute_gradients(const Image& grey);
Image gaussian_blur(const Image& src, float sigma);

// Circular pad so height/width are multiples of tile_size (alignment.init_alignment).
Image pad_image_circular(const Image& img, int tile_size);

// ---- align.cpp ----------------------------------------------------------
// Global rigid (rotation + translation) model of one comparison frame
// relative to the reference, in alignment-grey pixels, by an FFT correlation
// sweep over candidate angles. ImageStackAlignator's PreAlignment.ScanAngles.
// Returns RigidModel::valid == false when it could not run.
RigidModel estimate_global_rigid(const Image& ref_grey, const Image& moving_grey,
                                 const Config& cfg);

// Cumulative downscale of pyramid level `lvl` relative to the alignment grey
// (the running product of bm_factors[0..lvl]).
f32 pyramid_level_scale(const Config& cfg, int lvl);

// Fill a coarsest-level flow field with the rigid model evaluated at each
// tile centre, expressed in that level's own pixel units. No-op when the
// model is not valid. Both align() and align_metal_impl call this on the
// field they build for the coarsest level, so the two stay twins.
void seed_flow_from_rigid(FlowField& flow, const RigidModel& model,
                          int level_tile_size, f32 level_scale);

FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg,
                int tile_size);
// Free cached ref Sobel/Hessian (call when reference pyramid is released).
void clear_align_ref_ica_cache();

// ---- robustness.cpp -----------------------------------------------------
// Noise-floor autoscale (Config::r_noise_floor_autoscale) as a per-channel,
// per-brightness-band multiplier on sigma_t/d_t. Banded rather than one
// global scalar because the model error is brightness-dependent (PRNU grows
// with signal, read-noise floors dominate shadows): a single mid-tone scale
// left bright flat regions -- the sky -- under-floored and reading as
// misalignment. Bands are quantiles of the guide brightness axis;
// at() interpolates piecewise-linearly between band centres so sigma_t has
// no seams at band boundaries.
struct RobSigmaScale {
    static constexpr int kBands = 8;
    f32 s[3][kBands] = {{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                        {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                        {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}};
    f32 at(int ch, f32 b) const {
        if (ch < 0 || ch > 2) return 1.f;
        f32 t = b * (f32)kBands - 0.5f;
        if (!(t > 0.f)) return s[ch][0];
        if (t >= (f32)(kBands - 1)) return s[ch][kBands - 1];
        const int i = (int)t;
        const f32 f = t - (f32)i;
        return s[ch][i] + (s[ch][i + 1] - s[ch][i]) * f;
    }
};

struct RefStats {
    Image means;
    Image stds;
    // Dodgson-quadratic upscale of means/stds to raw resolution (Algorithm
    // 6's literal R <- Zeros(H,W) path -- see Config::
    // robustness_raw_resolution_enabled). Computed once per burst here
    // rather than once per comparison frame, since the reference doesn't
    // change. Empty unless that toggle is on.
    Image means_hires;
    Image stds_hires;
    // The reference GUIDE IMAGE itself (not its 3x3 stats) at guide
    // resolution, for the fine-scale robustness term (Config::
    // robustness_fine_term): the box-mean d cannot see edge misalignments
    // below ~3 raw px (the mean averages them away), so a second distance is
    // measured on the raw guide samples. CPU path only -- the Metal host
    // pins its own GPU-resident copy instead of a host image.
    Image guide;
    // Per-channel, per-brightness-band multiplier on sigma_t/d_t, measured
    // from the reference frame (Config::r_noise_floor_autoscale).
    // 1 = trust the model as-is.
    RobSigmaScale sigma_scale;
}; // guide resolution [h/2, w/2, ch] for Bayer (means_hires/stds_hires: raw [h, w, ch])
RefStats init_robustness(const Image& ref_raw, const Config& cfg);

// Directory for the Monte Carlo noise-curve disk cache. Bit-identical to a
// rebuild -- unitary_MC is deterministic, seeded per brightness bin -- so this
// only saves the measured 2-2.8 s/channel cold build across app launches.
// Unset (host tools, tests) = no disk cache, prior behaviour exactly.
void robustness_set_noise_cache_dir(const std::string& dir);
// Build/cache-load every noise curve the burst's robustness calls will use,
// through the same accessors (same keys, same values). Thread-safe against
// concurrent curve requests; run it on a worker right after SNR tuning so a
// curve-cache MISS builds during ref analysis instead of stalling the first
// comparison frame.
void robustness_prewarm_noise_curves(const Config& cfg);
// Eq. 9 for the raw-resolution robustness path: 2x2 min-reduce to the guide
// lattice, 5x5 min there (Wronski's 10x10-raw footprint on Wronski's grid),
// nearest-upsample back to raw. Shared by the CPU path and the Metal host
// (which runs the mask kernel on GPU and this stage on CPU so both paths
// stay bit-identical without a new shader).
Image robustness_local_min_on_guide(const Image& R);

// Bayer quad -> guide-resolution RGB (or the raw plane itself outside Bayer mode).
Image compute_guide(const Image& raw, const Config& cfg);

// Measure the per-channel, per-brightness-band noise-floor scale
// (Config::r_noise_floor_autoscale) from the reference raw: builds the
// guide, samples local 3x3 stats, and takes a robust low quantile of
// measured-sigma / model-sigma_t per brightness band (bands without enough
// samples inherit the nearest measured band's scale). All 1 when disabled
// or unmeasurable. Shared by CPU init_robustness and the Metal host.
void robustness_noise_floor_scale(const Image& ref_raw, const Config& cfg,
                                  RobSigmaScale& out_scale);

// MC noise std at brightness in [0,1]: std_curve[round(1000*b)] (fast_monte_carlo).
f32 noise_std_at_brightness(f32 brightness, f32 alpha, f32 beta);
f32 noise_std_at_brightness(f32 brightness, const Config& cfg);

// Full noise curves (1001 bins) for GPU upload — same cache as CPU robustness.
void fetch_noise_curves(f32 alpha, f32 beta,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve);
void fetch_noise_curves(const Config& cfg,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve);

// Per-guide-channel counterpart: ch's curve is built from Config::
// noise_alpha_ch(ch)/noise_beta_ch(ch) rather than the cross-channel mean.
void fetch_noise_curves_channel(const Config& cfg, int ch,
                                std::vector<f32>& std_curve, std::vector<f32>& diff_curve);

// Robustness mask r at raw resolution [h, w, 1].
Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg);

// ---- kernels.cpp --------------------------------------------------------
CovField estimate_kernels(const Image& raw, const Config& cfg);

// ---- merge.cpp ----------------------------------------------------------
// frame_id: optional stable id for GPU buffer reuse when raw is streamed via scratch.
// Returns false when the frame did not reach the accumulator. The banded
// callers ignore it -- a band a frame cannot contribute to is normal -- but the
// online path merges each frame exactly once, so false there means the frame is
// simply absent from the output.
bool merge_comp_band(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                     const Image& robustness, int tile_size,
                     Image& num_band, Image& den_band, int y0, const Config& cfg,
                     int frame_id = -1);

void merge_ref_band(const Image& ref_raw, const CovField& covs,
                    Image& num_band, Image& den_band, int y0, const Config& cfg,
                    const Image* acc_rob = nullptr);

void merge_comp(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                const Image& robustness, int tile_size,
                Image& num, Image& den, const Config& cfg);
void merge_ref(const Image& ref_raw, const CovField& covs,
               Image& num, Image& den, const Config& cfg,
               const Image* acc_rob = nullptr);

// Accumulator health before num/den (for green/black speckle debugging).
struct AccumDiag {
    size_t pixels = 0;
    size_t den_zero[3] = {0, 0, 0};
    size_t den_tiny[3] = {0, 0, 0};      // 0 < d < 1e-12
    size_t den_nonfinite[3] = {0, 0, 0};
    size_t num_nonfinite[3] = {0, 0, 0};
    size_t only_green = 0;               // G>0, R==0, B==0
    size_t rgb_all_zero = 0;
};
void accumulate_diag(const Image& num, const Image& den, AccumDiag& d);
void accumulate_diag_ptr(const f32* nump, const f32* denp, size_t n_pixels,
                         int c, AccumDiag& d);
std::string format_accum_diag(const AccumDiag& d);

} // namespace hhsr
