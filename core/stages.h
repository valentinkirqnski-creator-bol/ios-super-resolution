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
Image compute_grey_decimate(const Image& raw, bool bayer_mode);
Image compute_grey_fft(const Image& raw);
Image compute_grey(const Image& raw, bool bayer_mode, GreyMethod method);

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
                                f32 r_Mt, int num_threads);

// 1 where the flow field is irregular over the 3x3 tile neighbourhood -- the
// r_Mt test (Wronski et al. Eq. 7/8, literally: raw max-min span, no
// detrending). sx/sy scale the stored displacements into r_Mt's units.
std::vector<uint32_t> compute_motion_irregular(const FlowField& flow, f32 Mt,
                                               f32 sx, f32 sy, int num_threads);

// Builds a raw-pixel tile-grid FlowField from a dense per-guide-pixel flow
// field produced by an external neural flow estimator (PWCNet), as a
// drop-in alternative to align()+flow_to_raw_tile_grid for any downstream
// consumer. dense_flow: dx plane (guide_h*guide_w floats) followed by dy
// plane (guide_h*guide_w floats), values in GUIDE-pixel units -- the layout
// a Core ML (1,2,guide_h,guide_w) MLMultiArray output has. See align.cpp
// for what's intentionally left unset (aperture_limited, match_ambiguous,
// motion_irregular) and why.
FlowField flow_from_dense_guide(const f32* dense_flow, int guide_h, int guide_w,
                                int raw_h, int raw_w, int tile_size,
                                f32 r_Mt, int num_threads);

struct Pyramid { std::vector<Image> levels; std::vector<int> abs_factors; };
Pyramid build_pyramid(const Image& grey, const std::vector<int>& factors);

Image compute_gradients(const Image& grey);
Image gaussian_blur(const Image& src, float sigma);

// Circular pad so height/width are multiples of tile_size (alignment.init_alignment).
Image pad_image_circular(const Image& img, int tile_size);

// ---- align.cpp ----------------------------------------------------------
FlowField make_global_initial_flow(int ny, int nx, int tile_size, int abs_factor,
                                   int finest_h, int finest_w,
                                   f32 initial_dx, f32 initial_dy,
                                   f32 initial_rotation_rad);
FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg,
                int tile_size,
                f32 initial_dx = 0.f, f32 initial_dy = 0.f,
                f32 initial_rotation_rad = 0.f);
// Free cached ref Sobel/Hessian (call when reference pyramid is released).
void clear_align_ref_ica_cache();

// ---- robustness.cpp -----------------------------------------------------
struct RefStats {
    Image means;
    Image stds;
    Image hf_loss;   // 1-channel high-frequency variance-loss map
}; // guide resolution [h/2, w/2, ch] for Bayer
RefStats init_robustness(const Image& ref_raw, const Config& cfg);

// Bayer quad -> guide-resolution RGB (or the raw plane itself outside Bayer
// mode). Exposed so callers besides robustness.cpp (neural_flow's caller)
// can build the exact same guide image the classical path scores against.
Image compute_guide(const Image& raw, const Config& cfg);

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
// Falls back to the shared bundled table when debug_pixel4a_noise_profile
// is set (no per-channel data exists for it).
void fetch_noise_curves_channel(const Config& cfg, int ch,
                                std::vector<f32>& std_curve, std::vector<f32>& diff_curve);

// Robustness mask r at raw resolution [h, w, 1].
//
// s_select_out, when non-null, also receives a per-pixel record of which motion
// prior each pixel was scored with: 1 where the strict s1 applied (sharp local
// variation in the flow field, or an aperture-limited tile), 0 where the
// permissive s2 did. Used to split the accumulated mask for inspection.
Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg,
                         Image* s_select_out = nullptr);

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
