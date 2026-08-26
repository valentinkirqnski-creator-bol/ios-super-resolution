#pragma once
//
// Metal GPU backend for grey-FFT, L2 BM, kernel covariance, robustness, and merge.
// FFT matches grey_pyramid.cpp (fft1d_pow2_inplace_ref + Bluestein).
// L2 BM matches Torch rfft2/irfft2/fftshift math; Metal FFT ≠ Torch float stream.
// Prefer HHSR_L2_CPU=1 / HHSR_ALIGN_CPU=1 for closer CPU/vDSP parity on dumps.
// Kernels match kernels.cpp estimate_kernels (GAT + decimate + grads + cov).
// Merge matches merge.cpp accumulate_comp / accumulate_ref (incl. robustness).
//
#include "types.h"
#include "stages.h"
#include <complex>
#include <vector>

namespace hhsr {

// Returns false if MTL device / pipelines could not be created.
bool metal_gpu_init();

// Direct RAW app path: uint16 Bayer -> normalized float Bayer with the same
// black/WB/clamp math as DecodeRawFrameDictionary's CPU fallback.
bool metal_decode_raw16_to_float(const void* raw_data, size_t raw_bytes,
                                 int h, int w, int bytes_per_row,
                                 const float site_black[4],
                                 const float site_denom[4],
                                 const float site_wb[4],
                                 Image& out);

// Alg. 3 FFT grey on GPU. Empty image on failure.
Image compute_grey_fft_metal(const Image& raw);

// L2 block-match one pyramid level on GPU (updates flow in place).
// Returns false on failure (caller must not fall back to CPU).
bool block_match_level_L2_metal(const Image& ref, const Image& moving,
                                int tile_size, int search_radius,
                                FlowField& flow,
                                float ambiguity_ratio = 1.10f,
                                bool write_ambiguity = false,
                                bool fallback_on_ambiguous = false,
                                bool subpixel = false);

// L1 BM for ts==16 (default finest level). Same warp-reduce + broken argmin
// as align.cpp. Returns false if unsupported (ts!=16 or R>1) or GPU fail.
bool block_match_level_L1_metal(const Image& ref, const Image& moving,
                                int tile_size, int search_radius,
                                FlowField& flow,
                                float ambiguity_ratio = 1.10f,
                                bool write_ambiguity = false,
                                bool fallback_on_ambiguous = false,
                                bool subpixel = false);

// ICA refine one pyramid level (ICA.py ica_kernel_8/16). Same bilinear rules,
// modf/trunc, butterfly reduce order, and Ax=B update as align.cpp / Python.
// hess: packed [ny*nx*4] = 00,01,10,11. Returns false if ts not in {8,16}.
// damp_ratio: Levenberg-Marquardt damping toward that eigenvalue ratio, 0 off.
// max_step: per-iteration displacement bound in pixels, 0 off.
bool ica_refine_level_metal(const Image& ref, const Image& gradx, const Image& grady,
                            const std::vector<float>& hess_packed,
                            const Image& moving, FlowField& flow,
                            int tile_size, int n_iter,
                            float damp_ratio = 0.f, float max_step = 0.f);

// Exact cuda_downsample / grey_pyramid.cpp downsample_by (valid gauss + stride).
bool downsample_by_metal(const Image& src, int factor, Image& out);

// GPU-resident moving pyramid + per-level Sobel/Hessian + BM→ICA + flow upscale
// (same math as align()). Sobel/Hess are computed one pyramid level at a time
// (no all-level sticky cache — that jetsams at 1×). Uses sticky grey from
// compute_grey_fft_metal when dims match. Downloads final flow only.
bool align_metal(const Pyramid& ref_pyr, const Image& ref_grey,
                 const Image& moving_grey,
                 const Config& cfg, int tile_size, FlowField& flow_out,
                 f32 initial_dx = 0.f, f32 initial_dy = 0.f,
                 f32 initial_rotation_rad = 0.f);

// Clear GPU-resident reference ICA buffers reused across comparison frames.
void metal_clear_ref_ica_cache();

// num/den → packed RGB16 (same math as encode_band_rows DNG path). Preview
// sampling stays on the host. Returns false → caller uses CPU encode.
// Pointer form, for the online merge: it holds one accumulator for the whole
// output and offsets a row into it rather than materialising a band image.
bool metal_normalize_band_rgb16_ptr(const float* num_p, const float* den_p,
                                    int bh, int Ws, int nch,
                                    const Config& cfg, std::vector<uint16_t>& row16);
bool metal_normalize_band_rgb16(const Image& num_band, const Image& den_band,
                                const Config& cfg, std::vector<uint16_t>& row16);

// Alg. 5 kernel covariance on GPU. Empty CovField on failure.
CovField estimate_kernels_metal(const Image& raw, const Config& cfg);

// Robustness hot path on GPU (1:1 with robustness.cpp). Noise curves stay on CPU.
// Empty RefStats / Image on failure.
// init pins ref means/vars on GPU; after init, host RefStats pixel buffers may be
// cleared (keep h/w/c) — compute_robustness_metal uses the pinned GPU copy.
RefStats init_robustness_metal(const Image& ref_raw, const Config& cfg);
void metal_release_host_ref_stats(RefStats& ref_stats); // free host pixels; keep dims

// Copies the pinned reference means/variances back from their Metal buffers
// into ref_stats.means/.stds on the host.
//
// init_robustness_metal deliberately returns a RefStats carrying only
// DIMENSIONS -- the pixels live in GPU buffers, because every consumer on
// this path is itself a kernel and a readback would be a pure waste. Anything
// that needs to touch those statistics from C++ (the learned robustness mask
// builds its feature planes from them) must call this first, or it will index
// an Image whose h/w/c look valid and whose data vector is empty.
//
// Returns false if the buffers are missing or the dimensions disagree.
bool metal_fetch_host_ref_stats(RefStats& ref_stats);
// s_select_out, when non-null, also receives a per-pixel record of which motion
// prior was applied: 1 where the strict s1 was used, 0 where s2 was.
Image compute_robustness_metal(const Image& comp_raw, const RefStats& ref_stats,
                               const FlowField& flow, int tile_size, const Config& cfg,
                               Image* s_select_out = nullptr);

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
// Opens the merge frame table for a new burst.
//
// trim_analyze_scratch also drops the reference robustness statistics
// (clear_rob_ref_gpu). The host copy is released right after the reference is
// analyzed, so the GPU holds the only one -- pass false when calling this
// before the comparison frames have been scored, or every mask comes back
// empty and the merge falls back to the reference alone.
void metal_merge_begin_burst(bool trim_analyze_scratch = true);

// Free grow-only L2 / Alg. 5 scratch (call before merge prefetch / new burst).
void metal_trim_analyze_scratch();

// Drop the pinned moving grey so align_metal re-uploads instead of reusing it.
void metal_invalidate_sticky_grey();

// When true, reuse one GPU num/den slot (wait each band). Cuts peak RAM ~2× so
// full-res 1× can use larger bands without jetsam. Default false (2× double-buffer).
void metal_merge_set_single_acc_slot(bool enabled);

// merge_ref_band_metal commits asynchronously and resolves any *previous* in-flight
// band into its host images (so encode can overlap the next GPU band). Call this
// to wait + readback the latest band before using its num/den. No-op if idle.
bool metal_merge_wait_inflight();

// Online merge. One accumulator sized to the whole output, persisting across
// command buffers, so a frame can be merged and released instead of staying
// resident until the last band. Working set stops growing with frame count.
//
// begin_online -> (merge_comp_band + flush_online) per frame -> merge_ref_band
// -> finish_online -> end_online. flush_online waits, because the point of
// committing per frame is to let that frame's GPU buffers go.
// Drop one frame's cached GPU upload. Online merges a frame once, so its
// buffers are dead the moment its command buffer completes.
void metal_merge_release_frame(int frame_id);
void metal_merge_begin_online(int out_h, int out_w, int nch);
// Wait for the merge and hand back the accumulator where it already lives.
// Shared storage means it is CPU addressable in place, so the caller never
// needs a full-size host copy of it.
bool metal_merge_map_online(const float** num, const float** den, size_t* nelem);
void metal_merge_end_online();
bool metal_merge_flush_online();

} // namespace hhsr
