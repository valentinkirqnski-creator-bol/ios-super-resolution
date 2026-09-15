#pragma once
//
// Faithful port of hdrplus-python finishing.py (the _final.jpg path), adapted
// to run on this pipeline's merged camera-native linear RGB and to stay within
// a bounded memory budget at 48 MP.
//
// Pipeline (1:1 with finishing.py, params from package/algorithm/params.py):
//   0. WB + camera->sRGB matrix -> linear sRGB [0,1]  (== rawpy.postprocess,
//      colour-wise; the demosaic is already done by the merge)
//   1. localToneMap: auto-gain -> synthetic long -> sRGB gamma -> Mertens
//      exposure fusion (exposure weight only) -> un-gamma -> per-pixel channel
//      scaling by fusedGray/shortGray
//   2. enhanceContrast: x - gtmContrast*sin(2*pi*x)   (gtmContrast = 0.075)
//   3. sRGB gamma compress
//   4. sharpenTriple: unsharp mask, sigmas 1/2/4, amounts 1/0.5/0.5,
//      thresholds 0.02/0.04/0.06, averaged
//   5. 8-bit (round-half-up)
//
// MEMORY: the tone map (step 1) is computed on a downscaled grayscale
// (ltm_downsample) producing a smooth full-res gain map -- a low-frequency map,
// so the look matches, but that one step is not native-resolution. Everything
// else runs full-res, streamed in horizontal bands, so peak stays well under
// ~150 MB at 48 MP. NOT byte-identical to the Python (different demosaic, no
// OpenCV Mertens/Gaussian internals, different JPEG encoder) -- same steps and
// look. See scratchpad/py_finish.py for the validated reference.
//
#include <cstdint>
#include <functional>
#include <vector>

namespace hhsr {

struct HdrPlusPyParams {
    float wb[3]  = {1.f, 1.f, 1.f};                 // green-normalised WB gains
    float ccm[9] = {1,0,0, 0,1,0, 0,0,1};           // WB'd cam RGB -> sRGB (row-major)
    float gtm_contrast = 0.075f;
    float sharpen_sigma[3]     = {1.f, 2.f, 4.f};
    float sharpen_amount[3]    = {1.f, 0.5f, 0.5f};
    float sharpen_threshold[3] = {0.02f, 0.04f, 0.06f};
    int   ltm_downsample = 8;   // grayscale reduction factor for the tone map
};

// Fills `out` (rows [y0, y0+bh) of camera-native LINEAR RGB, interleaved,
// 65535 == 1.0 normalised to float [0,1)), for the finisher to consume. bh rows
// x W x 3 floats. Rows outside [0,H) should be clamped (edge-replicated).
using LinearBandReader = std::function<void(int y0, int bh, float* out)>;

// Receives one finished band of 8-bit sRGB (bh rows x W x 3 uint8), top to
// bottom, in order.
using U8BandWriter = std::function<void(int y0, int bh, const uint8_t* rgb8)>;

// W,H are the merged image dimensions (landscape, as stored). Streams two passes
// over read_band and emits finished bands via write_band. band_rows bounds the
// working set; 0 picks a default.
void hdrplus_py_finish(int W, int H, const HdrPlusPyParams& p,
                       const LinearBandReader& read_band,
                       const U8BandWriter& write_band,
                       int band_rows = 0);

}  // namespace hhsr
