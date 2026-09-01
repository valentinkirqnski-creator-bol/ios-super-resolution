#pragma once
// CPU port of ImageStackAlignator's whole-image FFT-based rotation/shift
// pre-alignment: PEFStudioDX/PreAlignment.cs (ScanAngles, FourierFilter)
// plus the kernel.cu math they drive (conjugateComplexMulKernel,
// fourierFilter). Uses a full complex spectrum throughout instead of
// cuFFT's packed R2C layout (see fft.h) -- the fourierFilter port below is
// adapted accordingly (applying the same frequency-domain filter formula
// symmetrically over the whole spectrum rather than just its packed half).
//
// NOT ported here (belongs to Stage 7, PatchTracker): squaredSum,
// boxFilterWithBorderX/Y, normalizedCC, findMinimum -- despite being
// declared as PreAlignment.cs fields, they are never actually called
// anywhere in that file; they operate on per-tile data and are only used
// by the patch tracker.
#include <vector>
#include "fft.h"

namespace isacpu {

struct RotationResult {
    float shift_x, shift_y;  // matches ScanAngles' own `-maxX, -maxY`
    float best_angle_deg;
    float best_value;
};

// conjugateComplexMulKernel: b[i] = conj(a[i]) * b[i], in place.
void conjugate_complex_mul(const std::vector<Cplx>& a, std::vector<Cplx>& b);

// fourierFilter, generalized from cuFFT's packed R2C half-spectrum to the
// full complex spectrum this port uses (see file comment above). lp/lps
// are fixed at 1 by every real call site in ISA (PreAlignment.cs always
// passes those two literal `1`s) but are exposed for completeness.
void fourier_filter(std::vector<Cplx>& spectrum, int width, int height,
                    int clear_axis, float lp, float hp, float lps, float hps);

// PreAlignment.ScanAngles: coarse-then-fine rotation search via FFT
// cross-correlation. `ref_img`/`tracked_img` are both width x height.
// `zero_deg` is the search center (usually 0), `range`/`incr` in degrees.
RotationResult scan_angles(const std::vector<float>& ref_img, const std::vector<float>& tracked_img,
                          int width, int height, float incr, float range, float zero_deg);

// PreAlignment.FourierFilter: FFT -> fourier_filter -> IFFT -> clamp to
// [0,1] (NPP ThresholdLTGT(0,0,1,1): <0 -> 0, >1 -> 1, values in [0,1]
// pass through UNCHANGED), in place. An earlier revision binarized here
// and mis-documented that as the original's behavior.
void fourier_filter_apply(std::vector<float>& img, int width, int height,
                          int clear_axis, float high_pass, float high_pass_sigma);

}  // namespace isacpu
