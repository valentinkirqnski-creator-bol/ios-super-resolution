#pragma once
//
// Declarations for each pipeline stage, mirroring the modules of the Python
// reference (grey/pyramid, alignment, robustness, kernels, merge).
//
#include "types.h"
#include <vector>

namespace hhsr {

// ---- grey_pyramid.cpp ---------------------------------------------------
// Bayer -> grey by averaging each 2x2 quad (the "decimate" method). Output is
// half-resolution in each axis. For grey_mode the input is returned as-is.
Image compute_grey_decimate(const Image& raw, bool bayer_mode);

// Coarse-to-fine Gaussian pyramid. levels[0] is the finest (== input).
struct Pyramid { std::vector<Image> levels; std::vector<int> abs_factors; };
Pyramid build_pyramid(const Image& grey, const std::vector<int>& factors);

// Central-difference gradients of a grey image -> [h-? , w-?, 2] packed image.
Image compute_gradients(const Image& grey);

// Separable Gaussian blur (used for robustness guide + optional smoothing).
Image gaussian_blur(const Image& src, float sigma);

// ---- align.cpp ----------------------------------------------------------
// Full Alg. 2 registration: multi-scale block matching + ICA refinement.
// Returns per-tile flow at the finest tile grid (tile_size = tile_sizes[0]).
FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg,
                int tile_size);

// ---- robustness.cpp -----------------------------------------------------
struct RefStats { Image means; Image stds; }; // both [gh, gw, ch]
RefStats init_robustness(const Image& ref_raw, const Config& cfg);

// Robustness mask R, one value per raw pixel of the reference. [h, w, 1].
Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg);

// ---- kernels.cpp --------------------------------------------------------
// Alg. 5: per-grey-pixel steerable covariance matrices for image `raw`.
CovField estimate_kernels(const Image& raw, const Config& cfg);

// ---- merge.cpp ----------------------------------------------------------
// Alg. 4 / Alg. 11 accumulation. num_band/den_band cover output rows
// [y0, y0 + num_band.h) at the full output width; local row i maps to global
// output row (y0 + i). This lets the pipeline process the (huge) output in
// bounded-memory horizontal bands.
void merge_comp_band(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                     const Image& robustness, int tile_size,
                     Image& num_band, Image& den_band, int y0, const Config& cfg);

void merge_ref_band(const Image& ref_raw, const CovField& covs,
                    Image& num_band, Image& den_band, int y0, const Config& cfg);

// Whole-image convenience wrappers (num/den are full [Hs, Ws, nch]).
void merge_comp(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                const Image& robustness, int tile_size,
                Image& num, Image& den, const Config& cfg);
void merge_ref(const Image& ref_raw, const CovField& covs,
               Image& num, Image& den, const Config& cfg);

} // namespace hhsr
