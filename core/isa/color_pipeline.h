#pragma once
// CPU port of ImageStackAlignator's colour pipeline: PefFile/DNGColorSpec.cs
// (Adobe DNG SDK's camera-profile interpolation: dual-illuminant colour
// matrix blending by correlated colour temperature, Bradford white-point
// adaptation, iterative neutral-to-white-point convergence),
// PefFile/DNGColorSpace.cs (fixed ProPhoto/sRGB-D50 primaries), and
// ImageStackAlignatorController.cs's SaveAs16BitTiff (the actual per-pixel
// chain: clamp -> camera-to-ProPhoto -> clamp -> tone curve -> ProPhoto-to-
// sRGB -> exposure -> gamma).
//
// Deliberate scope cut: ForwardMatrix1/2, CameraCalibration1/2,
// AnalogBalance, and ReductionMatrix1/2 are not read by this port's DNG
// reader and are treated as absent (identity/empty), matching ISA's own
// fallback behaviour when those tags are missing. When a DNG DOES carry a
// ForwardMatrix (as the real test file used this session does), ISA
// prefers that branch for its CameraToPCS matrix; this port always uses
// the "invert PCStoCamera" branch instead. Both are DNG-spec-valid ways to
// render camera colour; this is a real, disclosed accuracy trade-off, not
// an oversight.
//
// The tone curve IS reproduced: the WPF curve editor's constructor seeds
// a pronounced non-identity S-curve (LUTControl.xaml.cs:60-71) and fires
// its LUT evaluation during startup layout, so ISA's untouched default
// applies that curve, not an identity line (an earlier revision of this
// header claimed otherwise -- disproved by the source-identity audit).
// See DefaultToneLut in the .cpp: LUTControl's dng_spline solver verbatim,
// 256 samples at i/255 (float-quantized like C#'s defaultLUT), cubic
// interpolation modelling NPP's closed-source LUTCubic.
//
// Additional undisclosed-until-now scope cut, now disclosed: the
// AsShotWhiteXY branch of DNGColorSpec (cs:363-390) is not ported -- this
// port's inputs always carry AsShotNeutral, never AsShotWhiteXY, so the
// NeutralToXY path below is the one ISA itself would take for them.
#include "debayer.h"
#include "dng_reader.h"

namespace isacpu {

struct Mat3d { double m[9]; };  // row-major 3x3

// A * B (row-major 3x3 multiply), exposed for callers composing the fixed
// colour-space matrices (e.g. prophoto_from_pcs() * camera_to_pcs_matrix()).
Mat3d mat3_mul_public(const Mat3d& A, const Mat3d& B);

// camera_to_pcs_matrix: computes the camera-RGB -> XYZ(D50) matrix for
// `raw`'s ColorMatrix1/2 + CalibrationIlluminant1/2 + AsShotNeutral, plus
// the re-derived "camera white" (fCameraWhite in the source -- NOT
// necessarily identical to the raw AsShotNeutral tag, since it's round-
// tripped through the whole white-point solve) used for the highlight
// clamp in render_pixel.
struct CameraProfile {
    Mat3d camera_to_pcs;
    float camera_white[3];
};
CameraProfile camera_to_pcs_matrix(const DngRaw& raw);

// Fixed, standard colour-space matrices (Adobe DNG SDK's hardcoded
// primaries, each rescaled so toPCS*(1,1,1) reaches the D50 PCS white
// point exactly -- matching DNGColorSpace's constructor).
Mat3d prophoto_from_pcs();
Mat3d prophoto_to_pcs();
Mat3d srgb50_from_pcs();

// SaveAs16BitTiff's per-pixel chain, given a pixel already in "camera-
// native, black/white-level normalized" units (accumulate_images' own
// output convention) and the two precomputed matrices
// camera_to_prophoto = prophoto_from_pcs() * camera_to_pcs_matrix(raw).camera_to_pcs
// prophoto_to_srgb   = srgb50_from_pcs() * prophoto_to_pcs()
// Returns sRGB, gamma-encoded, in [0,1] (multiply by 65535 and round for
// 16-bit TIFF output).
Rgbf render_pixel(Rgbf camera_rgb, const Mat3d& camera_to_prophoto, const Mat3d& prophoto_to_srgb,
                  const float camera_white[3], float exposure_stops);

}  // namespace isacpu
