#pragma once
// Handheld-Multi-Frame-Super-Resolution-1.4 ("bran") pipeline pieces that the rest of
// the port did not already have, ported exactly (Config::bran14_pipeline):
//
//   utils_image.cfa_to_rggb / rggb_to_cfa   -- the burst is flipped to RGGB and back
//   utils_image.estimate_image_snr          -- SNR in dB from the RGBG noise profile
//   params.update_snr_config                -- tile size up to 64 + the four lerps
//   robustness.py                           -- guide, local stats, Dodgson warp,
//                                              d / sigma with the Monte Carlo LUT
//                                              (noise_lut14.h), S, threshold, 5x5 min
//
// Float typing follows what Numba compiles the CUDA kernels to: wherever a kernel mixes
// an int-initialised accumulator, a Python float or an int coordinate with float32 data,
// Numba promotes to float64, and those steps are computed in double here and stored
// back to float32 where 1.4 stores to a float32 array. Python's builtin max/min lower to
// "v > acc ? v : acc" (Numba do_minmax), which is what turns the out-of-frame inf / NaN
// into R = 0 -- reproduced literally.
#include "types.h"
#include <vector>

namespace hhsr {

// cfa_to_rggb: 0 = RGGB (as is), 1 = BGGR (flip both axes = 180 deg), 2 = GBRG (flip x),
// 3 = GRBG (flip y), -1 = not a Bayer pattern 1.4 supports.
int bran14_flip_mode(const CFA& cfa);
// Applies the flip in place to a [h, w, c] image. The flip is its own inverse
// (rggb_to_cfa == cfa_to_rggb), so the same call deflips the output.
void bran14_flip(Image& img, int mode);

// utils_dng.noise_profile_from_tags: the NoiseProfile tag's float64 values as
// (alpha, beta) pairs, expanded to R, G1, B, G2 (1 plane -> x4, 3 -> (a0, a1, a2, a1),
// 4 -> as is). Fills cfg.noise14_* and returns noise14_valid (false: odd count, another
// plane count, or a non-finite / non-positive alpha -- 1.4's gat asserts alpha > 0).
bool bran14_set_noise_profile(Config& cfg, const double* values, int count);

// Per-site black for rawpy black_level_per_channel (R, G1, B, G2) as 1.4 indexes it:
// after the cfa_to_rggb flip, position (0,0) uses [0], (0,1) [1], (1,0) [3], (1,1) [2].
// Written back in UNFLIPPED sensor site order into cfg.black14_site.
void bran14_set_black_per_channel(Config& cfg, const float black_ch[4]);

// estimate_image_snr on the (RGGB, normalised, un-white-balanced) reference, alpha /
// beta in R, G1, B, G2 order. Returns dB.
double bran14_estimate_snr(const Image& ref_rggb, const double alpha[4], const double beta[4]);

// update_snr_config with every SNR_based field left at SNR_based (1.4's defaults):
// Ts = 64 / 32 / 16 at <= 14 / <= 22 / else dB (no 32 cap), tile_sizes = int(Ts * [1, 1,
// 1, 0.5]), and k_detail / k_denoise / D_th / D_tr lerped over [6, 30] dB. Returns Ts.
// k_detail_override still wins when the user has set it.
int bran14_update_snr_config(Config& cfg, double snr_db);

// Reference local statistics of the 3-channel guide, planar [3][gh][gw].
struct RefStats14 {
    int gh = 0, gw = 0;
    std::vector<float> mean;
    std::vector<float> var;
};
RefStats14 bran14_init_robustness(const Image& ref_rggb, int num_threads);

struct Robustness14Params {
    int tile_size = 16;              // config.alignment.tile_size (raw pixels)
    bool noise_correction = true;
    double t = 0.12, s1 = 2.0, s2 = 12.0, Mt = 0.8;
    double sigma_mult = 1.0;         // Config::bran14_sigma_mult (1 = 1.4 exactly)
    const float* sigma_sq_curve = nullptr;   // LUT, n_curve entries
    const float* d_sq_curve = nullptr;
    int n_curve = 0;
    int num_threads = 0;
};

// compute_robustness: returns r = local_min(R) at guide resolution [gh, gw, 1].
// flow is 1.4's patch-wise flow (flow[..., 0] = x, flow[..., 1] = y, raw pixels) on the
// tile grid align() produced.
Image bran14_compute_robustness(const Image& comp_rggb, const RefStats14& ref,
                                const FlowField& flow, const Robustness14Params& p);

// Intermediate buffers, for the oracle test only.
struct Robustness14Debug {
    std::vector<float> comp_means, warped, d_sq, sigma_sq, S, R;
};
Image bran14_compute_robustness_debug(const Image& comp_rggb, const RefStats14& ref,
                                      const FlowField& flow, const Robustness14Params& p,
                                      Robustness14Debug* dbg);

}  // namespace hhsr
