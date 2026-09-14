#pragma once
//
// HDR+ finishing pipeline (a C++ reimplementation of the finish() stage from
// timothybrooks/hdr-plus, src/finish.cpp), adapted to run on this pipeline's
// already-demosaicked camera-native linear RGB rather than a Bayer mosaic.
//
// A SEPARATE JPEG/preview finishing option -- it does not touch the HHSR merge
// or the DNG. HDR+'s own Bayer-only stages (black/white level, white_balance,
// demosaic) are dropped because our input is already demosaicked and normalised
// to [0,65535]; its chroma-denoise and sharpen stages are omitted because the
// original finish() computes but never uses them. What remains, in order and
// math, matches HDR+:
//   1. white balance (per-capture gains)      -- our data is stored pre-WB
//   2. sRGB colour-correction matrix (CCM)     -- HDR+ "srgb" stage
//   3. tone_map (exposure-fusion, comp/gain)   -- HDR+'s local tone map
//   4. gamma correction (IEC sRGB, 16-bit)
//   5. global contrast (scaled-cosine S-curve + black subtract)
//   6. 8-bit
//
// Everything runs in the [0,65535] domain HDR+ uses, so its constants
// (gamma toe/cutoffs, contrast slope, black_level=2000) carry over unchanged.
//
#include <cstdint>
#include <vector>

namespace hhsr {

struct HdrPlusFinishOpts {
    // 3 green-normalised WB gains applied before the CCM. null -> the reference
    // capture's gains.
    const float* wb = nullptr;
    // Camera-linear(WB'd) -> sRGB-linear 3x3, row-major so out_i = sum_j
    // ccm[i*3+j]*in_j. null -> the reference capture's matrix.
    const float* ccm = nullptr;
    // HDR+ defaults (bin/HDRPlus.cpp): compression 3.8, gain 1.1.
    float compression = 3.8f;
    float gain = 1.1f;
    // finish() defaults: contrast_strength 5.0, black_level 2000 (of 65535).
    float contrast_strength = 5.0f;
    float black_level = 2000.f;
};

// rgb16: interleaved camera-native LINEAR RGB, 65535 == 1.0 (the linear DNG
// data, pre-white-balance). W*H*3 samples. Writes W*H*3 uint8 sRGB.
void hdrplus_finish(const uint16_t* rgb16, int W, int H,
                    const HdrPlusFinishOpts& opts,
                    std::vector<uint8_t>& out_rgb8);

}  // namespace hhsr
