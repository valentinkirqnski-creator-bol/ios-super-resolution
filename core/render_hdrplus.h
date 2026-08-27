#pragma once
//
// HDR+ finishing pipeline (Monod / Delon / Veit, IPOL, Section 5.2), ported
// verbatim from code/package/algorithm/finishing.py in this repo's reference
// tree. This replaces render_isp.* on the JPEG path.
//
// finish() there, in order:
//   1. demosaic + WB + CCM to linear [0,1] RGB      (rawpy postprocess; the
//      caller supplies this -- see hdrplus_finish's contract below)
//   2. local tone map: exposure fusion of a synthetic short/long pair
//   3. global tone map: x -= c * sin(2*pi*x)          (in LINEAR, as they do)
//   4. sRGB gamma compression
//   5. triple-scale thresholded unsharp mask
//   6. 8-bit
//
// Faithful to the source including the parts I would not have chosen:
//   - applyScaling_ clips each channel to [0,1] independently after the luma
//     ratio, with no gamut/desaturation step, so a saturated highlight pushed
//     over 1 shifts hue as one channel clips before the others. render_isp's
//     gamut step existed to prevent exactly that; it is deliberately NOT
//     carried over here.
//   - enhanceContrast runs BEFORE the gamma curve, so the S-curve pivots at
//     linear 0.5 (~0.74 in display terms). It is a highlight rearrangement
//     more than a midtone contrast curve. Also deliberate: it is what the
//     reference does.
//
// One deviation, for memory rather than taste, and it is a parameter:
// fusion_max_dim. A full-resolution Mertens pyramid at 2x output (48 MP)
// needs roughly 640 MB of pyramid buffers, which does not fit on device. The
// fusion runs on a downscaled luminance and the resulting per-pixel RATIO is
// bilinearly upsampled to full resolution. That is sound rather than a fudge:
// both fusion inputs are derived from one grayscale image (short, and gain *
// short), so their high-frequency content differs only by a smooth gain and
// the ratio fused/short carries almost no detail. Set fusion_max_dim = 0 to
// fuse at full resolution and match the reference bit-for-bit.
//
#include "types.h"

namespace hhsr {

struct HdrPlusFinishParams {
    // params.py 'finishing'/'tuning'. ltm_gain < 0 selects the automatic
    // search (their 'auto'), which is localToneMap's gain loop.
    float ltm_gain          = -1.f;
    float gtm_contrast      = 0.075f;
    float sharpen_sigma[3]     = {1.f, 2.f, 4.f};
    float sharpen_amount[3]    = {1.f, 0.5f, 0.5f};
    float sharpen_threshold[3] = {0.02f, 0.04f, 0.06f};

    // 0 = fuse at full resolution (reference-exact, memory-hungry). Otherwise
    // the longest edge the exposure fusion runs at; the ratio map is upsampled
    // from there. See the note above.
    int   fusion_max_dim    = 1024;

    bool  enable_ltm        = true;
    bool  enable_gtm        = true;
    bool  enable_sharpen    = true;
};

// sRGB transfer function, matching gammasRGB() exactly -- including that the
// reference clips to [0,1] inside the transfer itself.
f32 hdrplus_gamma_compress(f32 x);
f32 hdrplus_gamma_decompress(f32 x);

// The gain localToneMap() would pick for this image, exposed so callers can
// reuse one gain across frames (finish() does this for the reference image).
// grey must be the mean_() luminance, already linear.
f32 hdrplus_auto_gain(const std::vector<f32>& grey, int h, int w);

// In-place finishing. rgb must be interleaved 3-channel LINEAR [0,1] RGB --
// demosaiced, white-balanced and colour-corrected, i.e. the state rawpy's
// postprocess() hands finish(). On return it holds display-referred sRGB,
// still float [0,1]; the caller quantises to 8-bit.
void hdrplus_finish(Image& rgb, const HdrPlusFinishParams& p);

} // namespace hhsr
