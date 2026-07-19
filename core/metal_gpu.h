#pragma once
//
// Metal GPU backend for grey-FFT, L2 block-matching, kernel covariance, and merge.
// FFT matches grey_pyramid.cpp (fft1d_pow2_inplace_ref + Bluestein, same
// twiddle recurrence and scaling). L2 matches Torch rfft2/irfft2 path.
// Kernels match kernels.cpp estimate_kernels (GAT + decimate + grads + cov).
// Merge matches merge.cpp accumulate_comp / accumulate_ref (incl. robustness).
// No CPU fallback on Apple: failure returns empty / false.
//
#include "types.h"
#include <complex>
#include <vector>

namespace hhsr {

// Returns false if MTL device / pipelines could not be created.
bool metal_gpu_init();

// Alg. 3 FFT grey on GPU. Empty image on failure.
Image compute_grey_fft_metal(const Image& raw);

// L2 block-match one pyramid level on GPU (updates flow in place).
// Returns false on failure (caller must not fall back to CPU).
bool block_match_level_L2_metal(const Image& ref, const Image& moving,
                                int tile_size, int search_radius,
                                FlowField& flow);

// Alg. 5 kernel covariance on GPU. Empty CovField on failure.
CovField estimate_kernels_metal(const Image& raw, const Config& cfg);

// Alg. 4 / 11 band merge on GPU. Accumulates into num_band/den_band.
// Same math as merge_comp_band / merge_ref_band (robustness unchanged).
// No CPU fallback. Host caches per-frame GPU buffers across bands and batches
// all comps+ref for a band into one command buffer + one compute encoder.
// frame_id >= 0: stable cache key (needed when CPU streams into one scratch Image).
// When metal_merge_has_frame(frame_id), comp_raw may be empty (skip disk reload).
bool merge_comp_band_metal(const Image& comp_raw, const FlowField& flow,
                           const CovField& covs, const Image& robustness,
                           int tile_size, Image& num_band, Image& den_band,
                           int y0, const Config& cfg, int frame_id = -1);
bool merge_ref_band_metal(const Image& ref_raw, const CovField& covs,
                          Image& num_band, Image& den_band, int y0,
                          const Config& cfg, const Image* acc_rob);

// True if this comparison frame's RAW/flow/cov/R already reside on the GPU.
bool metal_merge_has_frame(int frame_id);

// Upload one comparison frame into the GPU merge cache (no accumulate).
// Call before the band loop so band 0 is not stalled on PCIe copies.
bool metal_merge_prefetch_frame(const Image& comp_raw, const FlowField& flow,
                                const CovField& covs, const Image& robustness,
                                int frame_id);

// Drop previous burst's GPU merge cache (call once before prefetching a new shot).
void metal_merge_begin_burst();

// When true, reuse one GPU num/den slot (wait each band). Cuts peak RAM ~2× so
// full-res 1× can use larger bands without jetsam. Default false (2× double-buffer).
void metal_merge_set_single_acc_slot(bool enabled);

// merge_ref_band_metal commits asynchronously and resolves any *previous* in-flight
// band into its host images (so encode can overlap the next GPU band). Call this
// to wait + readback the latest band before using its num/den. No-op if idle.
bool metal_merge_wait_inflight();

} // namespace hhsr
