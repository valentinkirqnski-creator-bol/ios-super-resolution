#pragma once
//
// Scene-referred linear RGB -> display sRGB, the way a phone ISP does it.
//
// This replaces matching a fixed external grade with a LUT. A 3D LUT is a
// per-pixel function of colour, and the look being aimed at is not: the whole
// point of an HDR render is that a bright sky and a dark doorway holding the
// same value get treated differently. Measured on two reference pairs, an ideal
// per-pixel gain cut the residual from 4.59 to 1.77 LSB and from 3.73 to 1.68 --
// roughly 60% of the error a colour-only mapping cannot reach, and about
// +/-0.6 EV of local adjustment.
//
// Order matters and is not negotiable in two places:
//
//   * Dynamic-range compression happens in LINEAR, before the output curve.
//     Once the shoulder has mapped a value to white the detail is gone, and no
//     amount of work downstream brings it back.
//   * The local gain multiplies R, G and B by the SAME factor. Bending the
//     three channels independently through a tone curve shifts hue, which is
//     what makes naive tone mapping look plastic.
//
// The gain map is built once per image at 1/8 resolution (~3MB at 48MP) with a
// self-guided edge-preserving filter, then sampled bilinearly per pixel. It is
// smooth by construction, so multiplying by it preserves all detail: the ratio
// Y/base is untouched, which is the multiplicative form of base+detail
// recombination.
//
// The DNG is unaffected. It is still written from the unmodified merge, and
// must stay that way -- this is the JPEG/preview render only.
//
#include "types.h"
#include <cstdint>

namespace hhsr {


// Everything derived from one pass over the image: the automatic exposure, the
// white point the output curve rolls off to, and the two low-resolution maps.
struct IspState {
    int W = 0, H = 0;
    int gw = 0, gh = 0;              // gain/base map dimensions
    int shift = 3;                   // downsample factor, 1 << shift
    std::vector<f32> gain;           // mapped_base / base, per low-res pixel
    std::vector<f32> base;           // exposed linear luminance, blurred
    f32 exposure = 1.f;              // linear multiplier applied before anything
    f32 white = 4.f;                 // output curve maps this to display 1.0
    f32 m[9] = {1,0,0, 0,1,0, 0,0,1};// camera linear -> sRGB linear, already
                                     // blended by colour_strength
    // std::pow dominated the render: srgb_oetf called it three times per pixel
    // and the local-contrast term once more, about 190M calls at 48MP on one
    // thread. Both are fixed-shape curves, so they are tabulated once here.
    std::vector<f32> oetf;           // [0,1] -> display, kOetfN entries
    std::vector<f32> lcurve;         // detail ratio ^ local_contrast, kLcN entries
    IspParams p;
    bool valid = false;
};

// One pass over the whole image. rgb16 is interleaved RGB, 65535 = full scale,
// scene-referred and already white balanced (what this pipeline writes to the
// linear DNG). cam_to_srgb may be null, in which case a measured default for
// this sensor is used.
bool isp_analyse(const uint16_t* rgb16, int W, int H,
                 const float* cam_to_srgb, const IspParams& p, IspState& st);

// Per-pixel render. x and y locate the pixel in the full-resolution image so the
// gain map can be sampled; r/g/b are linear in [0,1]. Outputs display sRGB in
// [0,1], ready to quantise.
void isp_render(const IspState& st, f32 r, f32 g, f32 b, int x, int y,
                f32& sr, f32& sg, f32& sb);

}  // namespace hhsr
