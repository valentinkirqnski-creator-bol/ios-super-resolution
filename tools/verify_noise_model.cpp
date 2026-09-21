// Standalone check of Config::noise_alpha/noise_beta.
//   g++ -std=gnu++17 -I core tools/verify_noise_model.cpp -o verify_noise_model
// Guards the derivation that two independent loaders depend on: a stored copy
// of this was filled by only one of them and pinned the device to the fallback.

#include "types.h"
#include <cstdio>
#include <cmath>
using namespace hhsr;
static int fails = 0;
static void chk(const char* n, float got, float want) {
    bool ok = std::fabs(got - want) <= 1e-9f + 1e-4f * std::fabs(want);
    if (!ok) ++fails;
    std::printf("%-46s got %.8g  want %.8g  %s\n", n, got, want, ok ? "OK" : "FAIL");
}
int main() {
    // The Sony RX100VII ISO-4000 profile from the report.
    Config c;
    c.alpha_dng[0] = 0.003751f; c.alpha_dng[1] = 0.002057f; c.alpha_dng[2] = 0.003751f;
    c.beta_dng[0]  = 0.0000782f; c.beta_dng[1] = 0.0000213f; c.beta_dng[2] = 0.0000782f;
    c.white_balance[0] = 2.40f; c.white_balance[1] = 1.0f; c.white_balance[2] = 2.09f;

    // The guide's green channel averages the two green SITES of an RGGB tile, and
    // averaging n independent samples divides the variance by n -- so green's
    // alpha and beta each carry a 1/2 that red and blue do not. That factor used
    // to live at the point of use, in guide_noise_var, and moved into these
    // accessors when the geometry gradient became a second consumer. This
    // verifier was written before the move and still expected the unweighted
    // mean, so it reported a failure against correct code; the weight belongs in
    // the expectation now. GW below is noise_guide_weight for RGGB.
    const float GW_R = 1.f, GW_G = 0.5f, GW_B = 1.f;

    // Before any loader has touched the pixels, the profile must stand unscaled --
    // but still guide-weighted, which is a property of the guide, not the loader.
    c.raw_prewhitened = false;
    chk("not prewhitened: alpha = weighted mean",
        c.noise_alpha(),
        (0.003751f*GW_R + 0.002057f*GW_G + 0.003751f*GW_B) / 3.f);
    chk("not prewhitened: beta  = weighted mean",
        c.noise_beta(),
        (0.0000782f*GW_R + 0.0000213f*GW_G + 0.0000782f*GW_B) / 3.f);

    // After the loader applied wb[c]/wb[G], alpha scales by g and beta by g^2.
    c.raw_prewhitened = true;
    chk("prewhitened: alpha",
        c.noise_alpha(),
        (0.003751f*2.40f*GW_R + 0.002057f*1.0f*GW_G + 0.003751f*2.09f*GW_B) / 3.f);
    chk("prewhitened: beta",
        c.noise_beta(),
        (0.0000782f*2.40f*2.40f*GW_R + 0.0000213f*1.0f*GW_G +
         0.0000782f*2.09f*2.09f*GW_B) / 3.f);

    // The per-channel form 1.4 actually evaluates: alpha[c]*x + beta[c], no mean.
    chk("per-channel alpha R", c.noise_channel_alpha(0), 0.003751f*2.40f*GW_R);
    chk("per-channel alpha G", c.noise_channel_alpha(1), 0.002057f*1.0f*GW_G);
    chk("per-channel alpha B", c.noise_channel_alpha(2), 0.003751f*2.09f*GW_B);
    chk("per-channel beta  G", c.noise_channel_beta(1),  0.0000213f*1.0f*GW_G);

    // The invariant that broke once and would break silently: green is halved
    // EXACTLY once. Double-halving it -- accessor plus a leftover halving at the
    // point of use -- quarters green's variance, which quietly inflates every
    // green SNR and makes the geometry test over-trust noise as gradient.
    chk("green halved exactly once",
        c.noise_channel_alpha(1) / (0.002057f * 1.0f), 0.5f);

    // The mean is the three per-channel values and nothing else, so a consumer
    // that wants one number and one that names a channel cannot drift apart.
    chk("mean == mean of per-channel", c.noise_alpha(),
        (c.noise_channel_alpha(0) + c.noise_channel_alpha(1) +
         c.noise_channel_alpha(2)) / 3.f);

    // This is the whole point: the value must go UP versus the old code, which
    // averaged the profile and never scaled it at all.
    float old_way = (0.003751f + 0.002057f + 0.003751f) / 3.f;  // unweighted, unscaled
    std::printf("\nold (averaged, unscaled) alpha = %.8g\n", old_way);
    std::printf("new (per-channel, scaled) alpha = %.8g   ratio %.3fx\n",
                c.noise_alpha(), c.noise_alpha() / old_way);
    if (!(c.noise_alpha() > old_way)) { std::printf("FAIL not larger\n"); ++fails; }

    // Degenerate metadata must not produce a NaN that poisons every threshold.
    Config d; d.raw_prewhitened = true;
    d.white_balance[0] = 0.f; d.white_balance[1] = 0.f; d.white_balance[2] = 0.f;
    bool finite = std::isfinite(d.noise_alpha()) && std::isfinite(d.noise_beta());
    std::printf("zero white balance stays finite: %s\n", finite ? "OK" : "FAIL");
    if (!finite) ++fails;

    std::printf("\n%s\n", fails ? "FAILURES" : "all checks passed");
    return fails ? 1 : 0;
}
