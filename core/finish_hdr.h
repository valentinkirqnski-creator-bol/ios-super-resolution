#pragma once
//
// HDR finishing: merged LinearRaw DNG -> display sRGB, for the JPG export.
//
// Written as a second, independent path rather than another knob on render_isp
// because the two disagree about where the dynamic range lives. render_isp is
// handed a buffer that ReapplyWhiteBalanceIfStored has already rolled off at a
// fixed knee and requantised to 16 bits, so the scene peak reaching its tone
// curve is 1.0 no matter what the sensor saw. This path reads the DNG's own
// camera-space samples, where every channel still saturates at full scale and
// the white-balance gains (R x2.06, B x1.84 on this sensor) are headroom rather
// than something already spent -- so the shoulder is positioned against the
// real peak and the compression is the only place the range is reduced.
//
// Five properties this path guarantees, in the order they are established:
//
//   * BLACK comes from the image, not a constant, and is set twice because
//     there are two different problems. A low percentile of the per-pixel
//     minimum channel is subtracted, which is where a veiling pedestal lives;
//     one scalar for all three channels, since a per-channel black would tint
//     the shadows -- measured and subtracted once the white-balance gains are
//     on, because that is the only space where one scalar IS neutral. An
//     un-white-balanced file stores a neutral grey as (g/2.06, g, g/1.84), so
//     the per-pixel minimum there is simply the red channel, and taking it out
//     of all three sends red to zero and the shadows green. Then, after the
//     render, the level that puts
//     the darkest content on zero is measured from the rendered image itself
//     and taken out -- without that the shadow lift below leaves the frame
//     sitting on a grey floor, which reads as haze.
//   * CLIPPING is detected in CAMERA space, before the gains, where all three
//     channels saturate at the same value. A pixel the sensor blew is pulled to
//     neutral at its own peak, so it renders white. The pink is what you get
//     detecting it afterwards: the gains push red and blue through full scale
//     first, and whichever channel is left behind decides the hue.
//   * The MATRIX is built for the space it is applied in. inv(ColorMatrix1) is
//     defined on camera-native values and the cached tag-65000 matrix is
//     defined on white-balanced ones; they differ by a per-channel column
//     scale, and no row normalisation stands in for it, because rows scale the
//     output and the error is on the input. This derives the matrix from
//     ColorMatrix1 and divides out every gain the samples actually carry, so
//     there is no residual cast in either container -- falling back to the
//     cached matrix only when the file has no real ColorMatrix1.
//   * TONE MAPPING is one scalar gain on R, G and B together, from a
//     guided-filter base layer -- shadows lifted and highlights compressed
//     around middle grey -- followed by a shoulder that sends the measured peak
//     to display white. Hue cannot move, because no channel is curved alone.
//   * SHARPENING does not exist here. There is no unsharp mask, no
//     micro-contrast, no high-pass of any kind, by construction rather than by
//     a parameter set to zero.
//
#include "types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hhsr {

struct FinishHdrParams {
    // ---- exposure -------------------------------------------------------
    // Where the scene's log-average lands in linear. The log-average is a
    // geometric mean, so it sits below the median on any scene with large dark
    // regions; 0.20 renders those at a sensible mid tone rather than dark.
    float auto_key = 0.20f;
    float exposure_ev = 0.f;          // manual trim, in stops

    // ---- tone mapping ---------------------------------------------------
    // Both act on the blurred base layer only, in log2 around middle grey, so
    // detail is carried through untouched and only the range moves.
    // 0 = no change, 1 = the strongest compression this allows (half the
    // distance to middle grey).
    float shadow_lift = 0.55f;
    float highlight_rolloff = 0.75f;
    // How much of that base compression to apply. Below 1 the render keeps some
    // of the original global contrast.
    float local_strength = 0.85f;
    // An S in display space, as a luminance ratio -- applied to luma and
    // re-applied to RGB as one factor, so it cannot rotate hue. Still modest on
    // purpose: the point of the base compression above is that the whole range
    // stays visible, and a strong contrast curve throws the ends of it away
    // again. Swept on the colour chart below, 0.10 -> 0.34 raises luma RMS
    // 37.5 -> 40.4 and clips nothing extra at any point, so this is a look
    // choice rather than a limit; 0.26 gives RMS 39.4 and widens the 5th/95th
    // luma percentiles from 51/158 to 49/161.
    float contrast = 0.26f;
    // Display black anchor, MEASURED rather than assumed. The base compression
    // above lifts the shadows on purpose, which leaves the darkest content
    // sitting at a grey the whole image then reads as haze through; a constant
    // subtraction is the usual answer and is wrong in both directions, crushing
    // a low-contrast scene and failing to reach black on a contrasty one. So
    // the render is sampled on a stride, this fraction of the samples is
    // whatever should land on 0, and the level that puts them there becomes the
    // black point -- capped, so a scene that genuinely has no black content
    // keeps its shadows.
    float display_black_percentile = 0.002f;
    float display_black_max = 0.16f;

    // ---- highlights -----------------------------------------------------
    // Camera-space value at and above which a sample counts as sensor-clipped.
    // Under full scale because the merge averages several frames, so a site
    // that clipped in all of them still lands a little short of the ceiling.
    float clip_threshold = 0.995f;
    // Where the pull toward neutral starts ramping in. The gap to
    // clip_threshold is the smoothstep's width, so recovery has no visible edge.
    float clip_soft = 0.90f;

    // ---- black ----------------------------------------------------------
    // Fraction of pixels that land on true black.
    float black_percentile = 0.0015f;
    // Ceiling on the subtraction, as a fraction of full scale. A low-key scene
    // that genuinely has no black content must not have its shadows crushed to
    // manufacture one.
    float black_max = 0.03f;

    // ---- colour ---------------------------------------------------------
    // 1 = the matrix's own saturation. Applied uniformly, so this is the part of
    // the boost that reaches colours vibrance deliberately will not touch --
    // kept small for exactly that reason.
    float saturation = 1.06f;
    // Boost that falls away as a colour approaches saturation, and is faded out
    // entirely in the brightest tones -- a near-white highlight must never be
    // re-saturated, because what gets amplified there is the residual channel
    // imbalance, i.e. the pink.
    //
    // This carries most of the vibrancy, precisely because of that falloff: the
    // weight is (1 - sat)^2, so a muted colour gets nearly all of it and an
    // already-vivid one almost none, which lifts the picture without driving
    // anything further out of gamut.
    //
    // Measured on a chart of muted patches at three exposure levels plus a
    // neutral wedge and a blown corner: together with saturation 1.06 and
    // contrast 0.26 this moves mean chroma 0.185 -> 0.211 and its 90th
    // percentile 0.367 -> 0.413, while the fraction of clipped samples stays at
    // 0.25%, mean saturation above luma 200 stays at 0.0001, and the worst
    // channel spread on the neutral wedge stays at 2. Those last two are the
    // ones that matter: they are what says the highlight fade is still holding
    // and that no tint is being invented.
    float vibrance = 0.40f;
    // Colour-difference-only smoothing. Luminance is preserved exactly, so this
    // removes shadow blotching without touching detail. It is here because the
    // shadow lift above is strong enough to make chroma noise visible.
    float chroma_denoise = 0.40f;
    float chroma_denoise_radius = 12.f;   // in full-resolution pixels
};

// Reads the DNG, renders it, and returns interleaved 8-bit sRGB (W*H*3).
// `orientation_out` receives the file's TIFF orientation so the encoder can tag
// the JPEG with it. Returns false if the DNG cannot be decoded.
bool finish_hdr_from_dng(const std::string& dng_path, const FinishHdrParams& p,
                         std::vector<uint8_t>& out_rgb8, int& W, int& H,
                         int& orientation_out);

}  // namespace hhsr
