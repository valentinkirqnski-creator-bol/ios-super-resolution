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

    // Before any loader has touched the pixels, the profile must stand unscaled.
    c.raw_prewhitened = false;
    chk("not prewhitened: alpha = plain mean",
        c.noise_alpha(), (0.003751f + 0.002057f + 0.003751f) / 3.f);
    chk("not prewhitened: beta  = plain mean",
        c.noise_beta(), (0.0000782f + 0.0000213f + 0.0000782f) / 3.f);

    // After the loader applied wb[c]/wb[G], alpha scales by g and beta by g^2.
    c.raw_prewhitened = true;
    chk("prewhitened: alpha",
        c.noise_alpha(), (0.003751f*2.40f + 0.002057f*1.0f + 0.003751f*2.09f) / 3.f);
    chk("prewhitened: beta",
        c.noise_beta(),
        (0.0000782f*2.40f*2.40f + 0.0000213f*1.0f + 0.0000782f*2.09f*2.09f) / 3.f);

    // This is the whole point: the value must go UP versus the old code, which
    // averaged the profile and never scaled it at all.
    float old_way = (0.003751f + 0.002057f + 0.003751f) / 3.f;
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
