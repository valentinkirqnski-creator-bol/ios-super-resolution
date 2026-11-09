#pragma once
// Demosaiced-reference fallback (Config::ref_demosaic_fallback).
//
// Where every comparison frame is rejected, the merge output is the reference
// alone, and accumulate_ref builds each colour only from that colour's own Bayer
// sites in a 3x3 window: R and B come from 1-4 off-centre samples, so edges get
// colour fringes and zipper. With the toggle on, the reference is demosaiced once
// per burst (edge-directed, colour-difference interpolation with soft V/H and
// diagonal direction weighting) and the reference step of the merge blends its
// contribution toward the kernel-weighted demosaiced RGB, by how much of the
// pixel's merge weight the reference supplies alone. Pixels where comparison
// frames contributed keep the plain merge (bit-identical below the blend start).
#include "types.h"

namespace hhsr {

// Full-resolution RGB (H x W x 3) in the same units as the input Bayer plane.
// Empty if the input is not a single-plane Bayer frame.
Image demosaic_ref_rgb(const Image& raw, const Config& cfg);

// Merge-side state, one burst at a time. merge.cpp reads it in accumulate_ref
// (CPU); on Apple the plane is uploaded as half RGB for the Metal ref merge.
void merge_set_ref_demosaic(Image&& rgb);
void merge_clear_ref_demosaic();
const Image* merge_ref_demosaic();

// Blend weight from the reference's share of the pixel's total merge weight:
// smoothstep(lo, hi, share). Shared by CPU and (mirrored) Metal.
inline float ref_fallback_t(float share, float lo, float hi) {
    if (!(hi > lo)) return share >= lo ? 1.f : 0.f;
    float t = (share - lo) / (hi - lo);
    t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
    return t * t * (3.f - 2.f * t);
}

}  // namespace hhsr
