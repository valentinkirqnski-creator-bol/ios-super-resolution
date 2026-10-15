#ifndef HHSR_ROB_MASK_NN_H
#define HHSR_ROB_MASK_NN_H
//
// The build-105 "new" robustness-mask refiner (tools/nn_mask_refine): a tiny
// darken-only net trained on synthetic bursts with deliberately corrupted
// alignment vectors. Measured +6.4% merged-MSE over the analytic mask on
// held-out synthetic data. Distinct from robustness_refine_shared.h (the older
// 47-feature refiner); this is the net the user asked to ship.
//
// Nine features, computed by the CALLER to match tools/nn_mask_refine/common.py
// exactly (that parity is checked in the harness). This header holds only the
// part that MUST match the trained weights bit-for-bit: per-feature
// normalisation + the 9->10->8->1 MLP + the darken-only multiply. Shared by
// core/robustness.cpp (CPU) and core/HHSRKernels.metal (GPU) so the two paths
// cannot drift.
//
// Feature order (see common.py build_burst / flow_tile_features):
//   0 R            analytic mask value being refined
//   1 d_over_sig   |ref - comp_aligned| / sqrt(alpha*ref + beta)
//   2 gref         |grad ref|, 0.5*central-difference luma
//   3 gradv        per-tile flow-field gradient magnitude
//   4 div          flow divergence (per tile, px across one tile)
//   5 curl         flow curl
//   6 dist         within-tile distance from the tile centre (0..~0.7)
//   7 gradv*dist   the physical magnitude of a single-vector error
//   8 cons_dev     |comp_aligned - multi_frame_mean| / sigma  (the consensus
//                  outlier signal the analytic mask lacks; needs the pre-pass
//                  mean of all aligned comparison frames)

#ifdef __METAL_VERSION__
#define RMN_THREAD thread
#define RMN_DEVICE device
#else
#include <math.h>
#define RMN_THREAD
#define RMN_DEVICE
#endif

#include "rob_mask_nn_weights.h"   // ROB_NN_FEAT, ROB_NN_FMU/FSD, W0/B0/W1/B1/W2/B2

// q in [0,1]: how much of R to KEEP. Darken-only -- the caller multiplies.
inline float rob_mask_nn_gate(const RMN_THREAD float* feat) {
    float x[9];
    for (int c = 0; c < ROB_NN_FEAT; ++c)
        x[c] = (feat[c] - ROB_NN_FMU[c]) / ROB_NN_FSD[c];
    float h0[10];
    for (int o = 0; o < 10; ++o) {
        float s = ROB_NN_B0[o];
        for (int c = 0; c < ROB_NN_FEAT; ++c) s += ROB_NN_W0[o * ROB_NN_FEAT + c] * x[c];
        h0[o] = s > 0.f ? s : 0.f;
    }
    float h1[8];
    for (int o = 0; o < 8; ++o) {
        float s = ROB_NN_B1[o];
        for (int c = 0; c < 10; ++c) s += ROB_NN_W1[o * 10 + c] * h0[c];
        h1[o] = s > 0.f ? s : 0.f;
    }
    float s = ROB_NN_B2[0];
    for (int c = 0; c < 8; ++c) s += ROB_NN_W2[c] * h1[c];
    return 1.f / (1.f + exp(-s));
}

// Assemble the 9 features (f7 is derived here so the caller cannot forget it)
// and return the KEEP gate. All other features are passed exactly as common.py
// computes them.
inline float rob_mask_nn_gate_from(float R, float d_over_sig, float gref,
                                   float gradv, float div, float curl,
                                   float dist, float cons_dev) {
    float f[9];
    f[0] = R; f[1] = d_over_sig; f[2] = gref; f[3] = gradv;
    f[4] = div; f[5] = curl; f[6] = dist; f[7] = gradv * dist; f[8] = cons_dev;
    for (int c = 0; c < 9; ++c) if (!isfinite(f[c])) f[c] = 0.f;
    return rob_mask_nn_gate(f);
}

#endif  // HHSR_ROB_MASK_NN_H
