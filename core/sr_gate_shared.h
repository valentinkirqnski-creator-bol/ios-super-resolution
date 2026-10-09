#ifndef HHSR_SR_GATE_SHARED_H
#define HHSR_SR_GATE_SHARED_H
//
// sr_gate's feature definitions, in a form BOTH compilers accept.
//
// Included by core/HHSRKernels.metal (the sr_gate_features kernel) and by
// core/sr_gate.cpp (build_sr_gate_features). That is the point of it: the eight
// compressions below -- a log1p with a particular divisor, a saturation point, a
// noise-relative normalisation, a distance to the nearest integer sample -- are
// not a display choice. They are part of the trained function. If the CPU and
// the GPU disagreed on any one of them by a factor of two, the network would
// still emit a plausible-looking mask, just not the one it was fitted to, and
// nothing would report an error. Two hand-written copies would agree on the day
// they were written and not a month later, so there is one copy.
//
// tools/sr_gate/srsim.py build_features is the third copy, and it is the one
// the weights were actually fitted against. tools/sr_gate/parity.py checks all
// of them agree on a real frame, to 6e-8.
//
// Portability rules for anything added below, since this compiles as Metal
// Shading Language (C++14-based) and as ordinary C++17:
//   - no templates, no std:: anything, no dynamic allocation, no references in
//     signatures (Metal would need an address space on them);
//   - address spaces go through SRG_THREAD, never bare;
//   - only the math functions both languages spell the same way, and min/max/
//     clamp through the macros (C++ has no free clamp, Metal has no std::min).

#ifdef __METAL_VERSION__
#define SRG_THREAD thread
#else
#define SRG_THREAD
#include <cmath>
#endif

#define SRG_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SRG_MAX(a, b) ((a) > (b) ? (a) : (b))
#define SRG_CLAMP01(v) SRG_MIN(SRG_MAX((v), 0.0f), 1.0f)

// ---- shape ---------------------------------------------------------------
#define SRG_FEATURES 8
#define SRG_WIDTH 8
#define SRG_LAYERS 3
#define SRG_DIL0 1
#define SRG_DIL1 2
#define SRG_DIL2 3
// Rows of context each side of an output row: the sum of the dilations.
#define SRG_HALO (SRG_DIL0 + SRG_DIL1 + SRG_DIL2)

// Flat weight-blob offsets. Written out rather than looped so a mismatch with
// the generated core/sr_gate_weights.h is a compile-time failure.
#define SRG_OFF_W0 0
#define SRG_OFF_B0 (SRG_OFF_W0 + SRG_WIDTH * SRG_FEATURES * 9)
#define SRG_OFF_W1 (SRG_OFF_B0 + SRG_WIDTH)
#define SRG_OFF_B1 (SRG_OFF_W1 + SRG_WIDTH * SRG_WIDTH * 9)
#define SRG_OFF_W2 (SRG_OFF_B1 + SRG_WIDTH)
#define SRG_OFF_B2 (SRG_OFF_W2 + SRG_WIDTH * SRG_WIDTH * 9)
#define SRG_OFF_HW (SRG_OFF_B2 + SRG_WIDTH)
#define SRG_OFF_HB (SRG_OFF_HW + SRG_WIDTH)
#define SRG_WEIGHTS_N (SRG_OFF_HB + 1)

// ---- feature compressions ------------------------------------------------
#define SRG_LOG_A_SCALE 0.125f          // 1/8
#define SRG_SNR_SCALE 0.125f            // 1/8
#define SRG_SPAN_SCALE 2.0f             // of span/tile_size
#define SRG_EMAG_SCALE 0.5f             // |E| in raw px; 2 px saturates
#define SRG_GRAD_SCALE 0.16666667f      // 1/6
#define SRG_DIRE_SCALE 0.16666667f      // 1/6
#define SRG_SUBPIX_SCALE 1.41421356f    // 1/sqrt(1/2), the largest |frac| dist

// Channel order in the interleaved feature plane.
#define SRG_F_EXP_A 0
#define SRG_F_LOG_A 1
#define SRG_F_SNR 2
#define SRG_F_SUBPIX 3
#define SRG_F_SPAN 4
#define SRG_F_EMAG 5
#define SRG_F_GRAD 6
#define SRG_F_DIR_E 7

// Everything a pixel's features are built from. Each backend fills this its own
// way -- a CPU thread walks rows and reads planes it already has, a GPU thread
// gathers its own window from device memory -- and then both call the one
// function below.
struct SrGateInputs {
    // Eq. 6 at this pixel, aggregated across guide channels exactly as
    // apply_noise_model aggregates them.
    float d_sq;
    float sigma_sq;
    // Summed local variance and its summed noise floor, across guide channels.
    float var_sum;
    float nvar_sum;
    // This pixel's tile displacement in RAW pixels -- raw, not guide, because
    // the sample lattice the phase matters on is the raw one.
    float raw_fx;
    float raw_fy;
    // Magnitude of the 3x3 tile neighbourhood's flow span, raw pixels.
    float span;
    // Within-tile geometric residual E, raw pixels.
    float ex;
    float ey;
    // Reference guide gradient at this pixel, already divided by the
    // guide-to-raw scale, and the noise sigma in those same units.
    float gix;
    float giy;
    float nsig;
    float tile_size;
};

// The eight channels, all in [0, 1] so the network sees one scale independent of
// exposure, ISO and sensor.
//
//   0 exp_a  exp(-d^2/sigma^2)                 Wronski's exponential, unscaled
//   1 log_a  log1p(d^2/sigma^2) / 8            the tail exp() has saturated
//   2 snr    log1p(signal var / noise var)/8   detail worth merging for
//   3 subpix |frac(raw flow)| / sqrt(1/2)      the NEW SAMPLE PHASE
//   4 span   3x3 tile flow span / ts * 2       motion irregularity
//   5 emag   |E| / 2                           within-tile translation error
//   6 grad   log1p(|grad g| / sigma_n) / 6     is there an edge to smear
//   7 dir_e  log1p(|grad g . E| / sigma_n)/6   predicted error ACROSS the edge
//
// 3 is the channel that makes this different from any mask derived from the
// residual alone. A half-pixel offset maximises d^2 AND is exactly the sample
// placement super-resolution needs; 0 and 1 cannot tell those apart, and 3 can.
// 7 is the converse: a coherent per-tile misalignment produces a residual the
// noise model explains away, so 0 and 1 miss it, while the flow field predicts
// it outright.
inline void sr_gate_features_from(SRG_THREAD const struct SrGateInputs* in,
                                 SRG_THREAD float* out) {
    // d^2/sigma^2, saturated rather than infinite.
    //
    // Range tests, not isfinite(): Metal compiles with fast-math on by default,
    // under which NaN and Inf handling is not something to rely on. A plain
    // comparison is false for NaN on the hardware either way, so both the
    // out-of-bounds fetch (d^2 = inf on the CPU, NaN once the noise model's
    // inf/inf shrinkage has run) and a zero sigma land on the same saturated
    // value, which reads as fully rejected. No statistics means no evidence.
    float a;
    const float sg = in->sigma_sq;
    const float ds = in->d_sq;
    if (sg > 0.0f && sg < 1.0e30f && ds >= 0.0f && ds < 1.0e30f) a = ds / sg;
    else a = 1.0e30f;
    if (!(a >= 0.0f)) a = 1.0e30f;

    out[SRG_F_EXP_A] = exp(-SRG_MIN(a, 60.0f));
    out[SRG_F_LOG_A] = SRG_CLAMP01(log(1.0f + SRG_MIN(a, 1.0e12f)) * SRG_LOG_A_SCALE);

    const float nvar_safe = SRG_MAX(in->nvar_sum, 1.0e-20f);
    const float sig_var = SRG_MAX(in->var_sum - in->nvar_sum, 0.0f);
    out[SRG_F_SNR] = SRG_CLAMP01(log(1.0f + sig_var / nvar_safe) * SRG_SNR_SCALE);

    const float u = fabs(in->raw_fx - rint(in->raw_fx));
    const float v = fabs(in->raw_fy - rint(in->raw_fy));
    out[SRG_F_SUBPIX] = SRG_CLAMP01(sqrt(u * u + v * v) * SRG_SUBPIX_SCALE);

    const float ts = (in->tile_size > 0.0f) ? in->tile_size : 1.0f;
    out[SRG_F_SPAN] = SRG_CLAMP01(in->span / ts * SRG_SPAN_SCALE);

    const float emag = sqrt(in->ex * in->ex + in->ey * in->ey);
    out[SRG_F_EMAG] = SRG_CLAMP01(emag * SRG_EMAG_SCALE);

    const float nsig = SRG_MAX(in->nsig, 1.0e-20f);
    const float gmag = sqrt(in->gix * in->gix + in->giy * in->giy);
    out[SRG_F_GRAD] = SRG_CLAMP01(log(1.0f + gmag / nsig) * SRG_GRAD_SCALE);
    const float dproj = fabs(in->gix * in->ex + in->giy * in->ey);
    out[SRG_F_DIR_E] = SRG_CLAMP01(log(1.0f + dproj / nsig) * SRG_DIRE_SCALE);
}

// ==========================================================================
// The artifact U-Net (Config::sr_gate_unet_enabled)
// ==========================================================================
//
// A second, larger gate that does NOT emit a mask. It predicts
// log1p(artifact / noise_sigma) -- how much damage merging this frame at this
// pixel would do -- and the mask comes afterwards from a policy whose tolerance
// is Config::sr_gate_tau. The eight statistics above are not enough for that:
// they summarise the residual and cannot carry its spatial structure, and a
// displacement across an edge leaves a signed +/- residual pair that |d|
// reduces to "large d beside a large gradient". So twelve image-domain channels
// are appended -- the reference guide, the WARPED comparison guide, their
// signed difference, both luminance gradient magnitudes and their dot product.
//
// Appended, never interleaved: channels 0..7 are byte-identical to the small
// gate's, so both read the same feature builder for that part.
//
// Shape constants are emitted by tools/sr_gate/export_unet.py; if they drift
// from the generated weight blob the static_assert there fails to compile.
#define SRGU_IN 20
#define SRGU_IMG 12
#define SRGU_BASE 16
#define SRGU_MID 28
#define SRGU_DIL0 2
#define SRGU_DIL1 4

// Receptive field, in MASK pixels, by the rf += (k-1)*dilation*jump recursion
// over e1, e2, pool, d1, b1, b2, (upsample), u1. Rows of context a band needs
// on each side is half of it, rounded up.
#define SRGU_RF 36
#define SRGU_HALO 18

// Flat weight-blob offsets, written out so a mismatch with the generated
// core/sr_gate_unet_weights.h is a compile-time failure.
#define SRGU_OFF_E1W 0
#define SRGU_OFF_E1B (SRGU_OFF_E1W + SRGU_BASE * SRGU_IN * 9)
#define SRGU_OFF_E2W (SRGU_OFF_E1B + SRGU_BASE)
#define SRGU_OFF_E2B (SRGU_OFF_E2W + SRGU_BASE * SRGU_BASE * 9)
#define SRGU_OFF_D1W (SRGU_OFF_E2B + SRGU_BASE)
#define SRGU_OFF_D1B (SRGU_OFF_D1W + SRGU_MID * SRGU_BASE * 9)
#define SRGU_OFF_B1W (SRGU_OFF_D1B + SRGU_MID)
#define SRGU_OFF_B1B (SRGU_OFF_B1W + SRGU_MID * SRGU_MID * 9)
#define SRGU_OFF_B2W (SRGU_OFF_B1B + SRGU_MID)
#define SRGU_OFF_B2B (SRGU_OFF_B2W + SRGU_MID * SRGU_MID * 9)
#define SRGU_OFF_U1W (SRGU_OFF_B2B + SRGU_MID)
#define SRGU_OFF_U1B (SRGU_OFF_U1W + SRGU_BASE * (SRGU_MID + SRGU_BASE) * 9)
#define SRGU_OFF_HW  (SRGU_OFF_U1B + SRGU_BASE)
#define SRGU_OFF_HB  (SRGU_OFF_HW + SRGU_BASE)
#define SRGU_WEIGHTS_N (SRGU_OFF_HB + 1)

// Per-pixel image-domain inputs. The gradients are plain central differences of
// the CHANNEL MEAN, deliberately NOT divided by the guide-to-raw scale the way
// SrGateInputs::gix is -- the network was fitted on the unscaled ones
// (tools/sr_gate/build.py _lum_grad), and halving them here would feed it
// inputs it never saw.
struct SrGateUNetInputs {
    float ref[3];      // reference guide local means
    float warp[3];     // comparison guide local means, warped by the flow
    float rgx, rgy;    // d/dx, d/dy of mean(ref)
    float wgx, wgy;    // d/dx, d/dy of mean(warp)
};

// Channels 8..19 of the U-Net's input vector. Order matches
// tools/sr_gate/build.py image_features exactly.
inline void sr_gate_unet_image_features(
        SRG_THREAD const struct SrGateUNetInputs* in, SRG_THREAD float* out) {
    out[0] = in->ref[0];
    out[1] = in->ref[1];
    out[2] = in->ref[2];
    out[3] = in->warp[0];
    out[4] = in->warp[1];
    out[5] = in->warp[2];
    // Signed, not absolute: the sign pair either side of an edge is the whole
    // signature of a sub-pixel displacement, and |.| would erase it.
    out[6] = in->warp[0] - in->ref[0];
    out[7] = in->warp[1] - in->ref[1];
    out[8] = in->warp[2] - in->ref[2];
    out[9] = sqrt(in->rgx * in->rgx + in->rgy * in->rgy);
    out[10] = sqrt(in->wgx * in->wgx + in->wgy * in->wgy);
    out[11] = in->rgx * in->wgx + in->rgy * in->wgy;
}

// log1p(artifact) -> R. tau is the tolerated artifact in noise sigma, beta the
// softness around it. The ONLY place the accept/reject preference lives; the
// weights above contain no notion of it.
inline float sr_gate_unet_mask(float pred_log1p, float tau, float beta) {
    const float a = exp(SRG_MIN(SRG_MAX(pred_log1p, -8.0f), 8.0f)) - 1.0f;
    return 1.0f / (1.0f + exp(-(tau - a) / SRG_MAX(beta, 1e-4f)));
}

#endif  // HHSR_SR_GATE_SHARED_H
