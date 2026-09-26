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

// ---------------------------------------------------------------------------
// GPU-resident comparison frames.
//
// Every analysis stage and then the merge consume the same 12 MP planes, and
// each one used to take its input from a host Image and put its output back into
// one: nine crossings of the host/device boundary per frame, for buffers the CPU
// never reads. Measured on an 8-frame 12 MP burst: 211 ms of GPU inside a
// 1544 ms frame, the rest allocate-and-touch at about 1.24 GB/s.
//
// A burst opens one allocation per resource -- images, flows, covariances,
// robustness masks -- divided into per-frame slices. Each producing kernel is
// bound at its slice's offset, so nothing moves and no kernel changed; the merge
// then binds four buffers however long the burst is and reaches each frame by
// offset, which is what lets one dispatch cover all of them.
//
// The stage entry points keep their Image / CovField signatures so the portable
// core and the CPU reference path are untouched. While residency is on they
// return a DIMENSIONS-ONLY Image or CovField (h/w/c set, storage empty), the same
// convention metal_release_host_ref_stats already uses.
// ---------------------------------------------------------------------------

// Size the burst's slice buffers. slot indices are the caller's frame indices,
// 0..n_frames-1. Returns false if the allocation does not fit, in which case the
// caller must run the host path.
bool metal_frames_begin(int n_frames, int raw_h, int raw_w, int tile_size,
                        const Config& cfg);
void metal_frames_end();

// True when the last metal_frames_begin refused because a buffer would not
// allocate, as opposed to refusing on policy (the raw-resolution robustness
// mask, whose slice size is not knowable up front). The distinction matters to
// the caller: a policy refusal says nothing about how much memory is free,
// whereas a failed allocation is direct evidence that the device is tight --
// and the path the caller would otherwise escalate to wants far more.
bool metal_frames_alloc_refused();

// Which slot the stage calls below read from and write into. -1 disables
// residency for the next call (the host path runs unchanged).
void metal_set_active_frame(int slot);
// Whether stage outputs stay resident. Off for the reference frame, whose host
// pixels tune_config_snr and the pyramid still need.
void metal_set_gpu_resident(bool on);
bool metal_frames_active();

// Copy one frame's flow field into its slice. This is the only per-frame
// analysis result the host still owns -- it is ~0.4 MB and the CPU genuinely
// reads it (compute_motion_irregular, rob_compute_s).
bool metal_frame_set_flow(int slot, const FlowField& flow);

// Per-row non-zero flags of a resident robustness mask, so the band-skip test
// does not need the 12 MB mask on the host. Twin of robustness_row_activity.
bool metal_frame_rob_rows(int slot, std::vector<uint8_t>& rows, bool& any);

// Seed a slot from a host plane. Used for the reference, whose decode keeps its
// host pixels because tune_config_snr sums them and the pyramid is built from
// them, but whose plane the merge still reads from the GPU like any other.
bool metal_frame_put_raw(int slot, const Image& img);

// True once this slot holds everything the merge needs.
bool metal_frame_merge_ready(int slot);

// Whether this slot's raw plane is already in its slice. The capture path fills
// it during decode (metal_decode_raw16_to_float writes straight into the slot);
// a CPU-decoded frame does not, and the caller has to seed it from the host
// plane. Lets that seeding skip a 48.8MB memcpy it does not need.
bool metal_frame_has_raw(int slot);

// Direct RAW app path: uint16 Bayer -> normalized float Bayer with the same
// black/WB/clamp math as DecodeRawFrameDictionary's CPU fallback.
// src_y0/src_x0/out_h/out_w crop inside the source plane (origins must be even,
// so the CFA phase is preserved). gpu_slot >= 0 writes the result into that
// frame's resident slice; keep_host = false then returns a dimensions-only Image.
bool metal_decode_raw16_to_float(const void* raw_data, size_t raw_bytes,
                                 int h, int w, int bytes_per_row,
                                 const float site_black[4],
                                 const float site_denom[4],
                                 const float site_wb[4],
                                 int src_y0, int src_x0, int out_h, int out_w,
                                 int gpu_slot, bool keep_host,
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
                                bool fallback_on_ambiguous = false);

// L1 BM for ts==16 (default finest level). Same warp-reduce + broken argmin
// as align.cpp. Returns false if unsupported (ts!=16 or R>1) or GPU fail.
bool block_match_level_L1_metal(const Image& ref, const Image& moving,
                                int tile_size, int search_radius,
                                FlowField& flow,
                                float ambiguity_ratio = 1.10f,
                                bool write_ambiguity = false,
                                bool fallback_on_ambiguous = false);

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

// Normalize rows [y0, y0+bh) of the ONLINE accumulator with no staging copies.
//
// The pointer form above has to be handed host pointers, so for the online
// merge it allocated a fresh 46.4MB buffer for num, another for den and an
// output buffer, then memcpyd the accumulator into them -- out of one
// MTLResourceStorageModeShared buffer and into another, once per band. This
// binds the accumulator at a byte offset where it already lives and hands back
// the 16-bit rows in a pooled GPU buffer the DNG writer reads in place.
//
// *out_rows stays valid until the next call or until the online merge ends.
// Returns false (with *out_rows = nullptr) when the online accumulator is not
// the source, so the caller falls back to the pointer form.
bool metal_normalize_online_rows_rgb16(int y0, int bh, int Ws, int nch,
                                       const Config& cfg, const uint16_t** out_rows);

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
//
// refined_out, when non-null, reports whether the rob_refine_mask kernel ran
// (Config::robustness_refine_nn_enabled). compute_robustness must know: it
// applies the refinement itself for the CPU paths, and running it again on a
// mask the GPU has already refined would apply the reduction twice.
Image compute_robustness_metal(const Image& comp_raw, const RefStats& ref_stats,
                               const FlowField& flow, int tile_size, const Config& cfg,
                               Image* s_select_out = nullptr,
                               bool* refined_out = nullptr);

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

// ---------------------------------------------------------------------------
// Fused band merge.
//
// One dispatch per band covering every resident comparison frame and then the
// reference, accumulating num/den in REGISTERS and writing the band's 16-bit
// rows directly. The accumulators never reach memory, so the full-output pair
// (1.17 GB at 48 MP) stops existing and the per-band read-modify-write traffic
// (2.34 GB per pass) goes away with it.
//
// Bit-identical to the per-dispatch form: comparison frames in index order then
// the reference, the same order banded and online both used, and the same
// merge_comp_contrib / merge_ref_contrib / normalise code. An f32 stored to
// memory and reloaded is exact, which is why dropping the round trips cannot
// move a value.
//
// *out_rows points into a pooled GPU buffer, valid until the next call.
// prev_step / prev_h / prev_w / prev_scale describe the host thumbnail's sample
// grid. The fused kernel has no num/den for the host to sample, so it writes the
// sampled pixels' NORMALIZED camera RGB and the host applies the same colour
// transform, curve and LUT it always did. prev_step 0 skips it.
// out_slot alternates 0/1 between bands, so the DNG writer can read one band's
// rows in place while the GPU produces the next.
// Why the last metal_merge_band_fused returned false. Every early exit in it is
// a different bug with the same symptom, and the caller could only report one
// generic message -- which named memory, the one thing it is usually not.
const char* metal_merge_fused_refusal();

bool metal_merge_band_fused(const int* comp_slots, int n_comp, int ref_slot,
                            int y0, int bh, int Hs, int Ws, int nch,
                            int tile_size, const Config& cfg,
                            int prev_step, int prev_h, int prev_w, float prev_scale,
                            int out_slot, const uint16_t** out_rows);
// [prev_h][prev_w][3] normalized camera RGB, filled across the band loop.
const float* metal_merge_fused_preview();
// Zero the accumulator-health counters before the band loop, and read them back
// after it. `pixels` is the output size; the kernel only counts anomalies.
void metal_merge_fused_reset_diag();
bool metal_merge_fused_read_diag(AccumDiag& diag, size_t pixels);

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
