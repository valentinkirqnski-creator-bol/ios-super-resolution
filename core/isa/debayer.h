#pragma once
// CPU port of ImageStackAlignator's debayer pipeline:
// Kernels/DeBayerKernels.cu (deBayerGreenKernel, deBayerRedBlueKernel),
// Kernels/kernel.cu (GammasRGB/applysRGBGamma), and the C# orchestration in
// ImageStackAlignatorController.cs (DeBayerFullRes, DeBayerBWGaussWB).
#include <vector>
#include <cstdint>

namespace isacpu {

struct Rgbf { float r = 0, g = 0, b = 0; };

// cfa[y%2][x%2] in {0=Red,1=Green,2=Blue}, matching DngRaw::cfa.
// black/scale are per-channel (R,G,B), applied as (raw - black[c]) * scale[c]
// before demosaicing (scale is normally 1/as-shot-white-balance-gain, used
// only to make the gradient/laplacian comparisons channel-comparable).
// Pixels within 2px of the border are left at {0,0,0} (unwritten), matching
// the source kernels' own `x>=width-2||x<2` / `y>=height-2||y<2` early-outs.
void debayer_green(const std::vector<uint16_t>& raw, int width, int height,
                   const int cfa[2][2], const float black[3], const float scale[3],
                   std::vector<Rgbf>& out);
void debayer_red_blue(const std::vector<uint16_t>& raw, int width, int height,
                      const int cfa[2][2], const float black[3], const float scale[3],
                      std::vector<Rgbf>& inout);

// sRGB gamma, matching kernel.cu's applysRGBGamma (NaN -> 0 first).
void apply_srgb_gamma(std::vector<Rgbf>& img);

// Rec.601-weighted luma (or green-only, matching GreenChannelOnly).
void rgb_to_gray(const std::vector<Rgbf>& img, std::vector<float>& out, bool green_only);

// Full DeBayerFullRes: green+red/blue demosaic under an as-shot-WB scaling
// (for better gradient comparisons), then that scaling undone so the output
// is black/white-level normalized (raw-black)/whiteLevel with no WB baked
// in -- matching the "remove again white balance" step in DeBayerFullRes.
// black/white_level are the REFERENCE frame's (used for all frames, matching
// `forColor = _pefFiles[0]`); camera_white is the as-shot white balance
// (DngRaw::as_shot_neutral).
void debayer_full_res(const std::vector<uint16_t>& raw, int width, int height,
                      const int cfa[2][2], const float black[3], const float white_level[3],
                      const float camera_white[3], std::vector<Rgbf>& out);

// DeBayerBWGaussWB's non-test-mode path: debayer_full_res -> sRGB gamma ->
// grayscale -> [Fourier high-pass, tracking path only] -> separable
// Gaussian blur (replicate border, sigma via gaussian_filter_1d from
// structure_tensor.h). tracking_fourier=false matches the
// `skipFourierFilter=true` call used ahead of accumulation prep;
// tracking_fourier=true matches the tracking path (Controller.cs:2409-
// 2414), which filters BETWEEN grayscale and blur.
void debayer_bw_gauss(const std::vector<uint16_t>& raw, int width, int height,
                      const int cfa[2][2], const float black[3], const float white_level[3],
                      const float camera_white[3], float sigma, bool green_only,
                      std::vector<float>& out,
                      bool tracking_fourier = false, int clear_axis = 0,
                      float high_pass = 0.01f, float high_pass_sigma = 0.0025f);

}  // namespace isacpu
