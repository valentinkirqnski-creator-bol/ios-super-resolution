#pragma once
//
// Declarations for each pipeline stage, mirroring the modules of the Python
// reference (grey/pyramid, alignment, robustness, kernels, merge).
//
#include "types.h"
#include <vector>
#include <complex>

namespace hhsr {

// ---- grey_pyramid.cpp (FFT helpers, exposed for align.cpp) ---------------
void fft1d(std::vector<std::complex<f32>>& a, bool inverse, std::vector<std::complex<f32>>* dft_buf = nullptr);
void fft2d(std::vector<std::complex<f32>>& data, int h, int w, bool inverse,
           std::vector<std::complex<f32>>* row_buf = nullptr,
           std::vector<std::complex<f32>>* dft_buf = nullptr);
void fftshift2d(std::vector<std::complex<f32>>& data, int h, int w);

// ---- grey_pyramid.cpp ---------------------------------------------------
Image compute_grey_decimate(const Image& raw, bool bayer_mode);
Image compute_grey_fft(const Image& raw);
Image compute_grey(const Image& raw, bool bayer_mode, GreyMethod method);

struct Pyramid { std::vector<Image> levels; std::vector<int> abs_factors; };
Pyramid build_pyramid(const Image& grey, const std::vector<int>& factors);

Image compute_gradients(const Image& grey);
Image gaussian_blur(const Image& src, float sigma);

// Circular pad grey to tile boundary (matches Python init_alignment F.pad circular).
Image pad_grey_circular(const Image& grey, int tile_size);

// ---- align.cpp ----------------------------------------------------------
Image compute_sobel_gradx(const Image& img);
Image compute_sobel_grady(const Image& img);

void block_match_level_L1(const Image& ref, const Image& moving,
                          int tile_size, int search_radius,
                          FlowField& flow, int num_threads);

void block_match_level_L2(const Image& ref, const Image& moving,
                          int tile_size, int search_radius,
                          FlowField& flow, int num_threads);

// Per-tile inverse Hessian for ICA: [ny*nx*4] = ih11, ih12, ih22, valid.
void compute_hessian_inverse(const Image& ref, const Image& gradx, const Image& grady,
                             int tile_size, int ny, int nx,
                             std::vector<f32>& out_ih);

void ica_refine_level(const Image& ref, const Image& gradx, const Image& grady,
                      const Image& moving, FlowField& flow, int tile_size,
                      int n_iter, int num_threads);

FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg,
                int tile_size);

FlowField upscale_alignment_flow(const FlowField& in, int target_ny, int target_nx,
                                 int upsample_factor, int new_tile_size,
                                 int prev_tile_size);

// ---- robustness.cpp -----------------------------------------------------
struct RefStats { Image means; Image stds; }; // raw resolution [h, w, ch]
RefStats init_robustness(const Image& ref_raw, const Config& cfg);

// Robustness mask r at raw resolution [h, w, 1].
Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg);

// ---- kernels.cpp --------------------------------------------------------
CovField estimate_kernels(const Image& raw, const Config& cfg);

// ---- merge.cpp ----------------------------------------------------------
void merge_comp_band(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                     const Image& robustness, int tile_size,
                     Image& num_band, Image& den_band, int y0, const Config& cfg);

void merge_ref_band(const Image& ref_raw, const CovField& covs,
                    Image& num_band, Image& den_band, int y0, const Config& cfg,
                    const Image* acc_rob = nullptr);

void merge_comp(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                const Image& robustness, int tile_size,
                Image& num, Image& den, const Config& cfg);
void merge_ref(const Image& ref_raw, const CovField& covs,
               Image& num, Image& den, const Config& cfg,
               const Image* acc_rob = nullptr);

} // namespace hhsr
