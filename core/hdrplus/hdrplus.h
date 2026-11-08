#pragma once
//
// HDR+ pipeline (Timothy Brooks / Simon Gardling), ported from the Halide
// reference in `hdr-plus-master/src` to plain C++ so it runs without Halide.
// This is the faithful CPU twin; a Metal device path mirrors it kernel-for-
// kernel (see core/HHSRKernels.metal + core/metal_gpu.mm). It is a *different*
// algorithm from the Wronski MFSR pipeline in the rest of core/ — the app's
// HDR toggle routes here instead.
//
// Numeric domain: the Halide code works in integer space (uint16 samples,
// int32 intermediates, specific truncating / saturating casts). To reproduce
// its results exactly this port keeps that integer domain rather than the
// float `Image` the MFSR core uses. Every cast is matched to the Halide one:
//   u16(x)      -> truncating cast to [0,65535] of a non-negative value
//   u16_sat(x)  -> round-to-zero then clamp to [0,65535]
//   u8_sat(x)   -> clamp to [0,255]
//   i32(x)      -> truncation toward zero
//
// Stages (all constants fixed from the source):
//   align   : 3-level pyramid (box_down2, gauss_down4 x2), 32px tiles / 16px
//             overlap, hierarchical L1 block match, +/-4 search per level.
//   merge   : temporal (L1-distance tile weighting) + spatial (raised-cosine
//             blend of the half-overlapped tiles) -> one denoised Bayer frame.
//   finish  : black/white level -> white balance -> Malvar demosaic -> sRGB
//             CCM -> exposure-fusion tone map -> gamma -> contrast -> 8-bit.
//
// Faithfulness note: the Halide finish() returns u8bit_interleaved(contrast_
// output); its chroma_denoise and sharpen Funcs are computed but NOT used in
// the output, and srgb() consumes demosaic_output (not the denoised one). This
// port therefore omits chroma-denoise / sharpen / bilateral / desaturate /
// rgb<->yuv from the output path ON PURPOSE, to match results exactly. Do not
// "restore" them without changing the reference too.
//
#include <cstdint>
#include <vector>
#include <string>

namespace hhsr {
namespace hdrplus {

// CFA pattern codes match hdr-plus-master Burst.h CfaPattern.
enum CfaPattern {
    CFA_UNKNOWN = 0,
    CFA_RGGB    = 1,
    CFA_GRBG    = 2,
    CFA_BGGR    = 3,
    CFA_GBRG    = 4,
};

struct Params {
    uint16_t black_point = 0;      // bp
    uint16_t white_point = 65535;  // wp
    // White balance, green-normalised (g0 == g1 == 1 as in InputSource.cpp).
    float wb_r  = 1.f;
    float wb_g0 = 1.f;
    float wb_g1 = 1.f;
    float wb_b  = 1.f;
    int   cfa_pattern = CFA_RGGB;
    // Color correction matrix, row-major ccm[r*3+c] == srgb_matrix(r,c); the
    // sRGB stage computes out_c = sum_r ccm[r*3+c] * in_r (finish.cpp:634).
    // Fill from LibRaw as GetColorCorrectionMatrix: ccm(i,j) = rgb_cam[j][i].
    float ccm[9] = {1,0,0, 0,1,0, 0,0,1};
    // Tone-map dynamic-range compression and gain. HDRPlus.cpp ships 3.8/1.1,
    // but the app shoots at normal exposure, so a lower gain avoids over-bright
    // output and lower compression keeps more global contrast (app default).
    float compression = 2.4f;
    float gain = 0.69f;
};

// Align + temporally/spatially merge a burst of raw Bayer frames into one
// denoised Bayer frame (uint16, w*h). frames[0] is the reference; every frame
// is w*h uint16, row-major. Reproduces align()+merge() from the Halide source.
void align_and_merge(const std::vector<const uint16_t*>& frames, int w, int h,
                     std::vector<uint16_t>& merged_out);

// Finish a merged Bayer frame to an 8-bit interleaved RGB image (w*h*3 uint8,
// row-major, channel-interleaved). Reproduces finish() from the Halide source.
void finish(const uint16_t* merged, int w, int h, const Params& p,
            std::vector<uint8_t>& rgb_out);

// GPU (Metal) implementation of the full pipeline, same contract as process().
// Defined in core/hdrplus_metal.mm (Apple only). Returns false if Metal is
// unavailable, the kernels fail to compile, or anything goes wrong — the caller
// then falls back to the CPU path. The kernels are runtime-compiled MSL, so a
// compile error degrades gracefully here instead of breaking the app build.
bool process_gpu(const std::vector<const uint16_t*>& frames, int w, int h,
                 const Params& p,
                 std::vector<uint16_t>& merged_out,
                 std::vector<uint8_t>& rgb_out);

// Full pipeline: align_and_merge then finish. `merged_out` (Bayer u16) is kept
// so the caller can write the pre-finish linear DNG; `rgb_out` (u8) is the
// tone-mapped render for the JPEG. Returns false only on degenerate input.
// `allow_gpu` tries process_gpu first (Apple), falling back to the CPU twin.
bool process(const std::vector<const uint16_t*>& frames, int w, int h,
             const Params& p,
             std::vector<uint16_t>& merged_out,
             std::vector<uint8_t>& rgb_out,
             bool allow_gpu = true);

} // namespace hdrplus
} // namespace hhsr
