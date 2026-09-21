// Verifies the per-channel noise model against Python 1.4's.
//
//   g++ -std=c++17 -O2 -I core tools/verify_noise_channels.cpp -o verify_noise_channels
//   ./verify_noise_channels      (exit 0 iff every channel matches)
//
// 1.4 evaluates alpha[c]*x + beta[c] per Bayer plane (estimate_image_snr in
// utils_image.py). This pipeline differs upstream in two ways that the model has
// to absorb rather than ignore:
//
//   * the raw is already white-balanced (raw_prewhitened), so for v = g*r the
//     variance becomes (g*alpha)*v + g^2*beta -- the same physical model in the
//     app's coordinates, not a deviation from it;
//   * the guide's green channel averages BOTH green Bayer sites, which halves its
//     variance; 1.4 needs no such term because it never builds that guide.
//
// What WAS a deviation, and what this guards: the per-channel values used to be
// collapsed into one cross-channel mean, so the channel argument only ever
// selected a green halving -- which the averaged scalar had already partly
// applied, making it count twice for green.
#include "types.h"
#include <cstdio>
#include <cmath>
using namespace hhsr;
static int fails = 0;
static void eq(const char* n, double a, double b) {
    const bool ok = std::fabs(a - b) <= 1e-9 + 1e-6 * std::fabs(b);
    if (!ok) ++fails;
    std::printf("  %-52s %.8g vs %.8g  %s\n", n, a, b, ok ? "OK" : "FAIL");
}
int main() {
    Config c;
    c.bayer_mode = true;
    c.raw_prewhitened = true;
    c.white_balance[0] = 1.87622f; c.white_balance[1] = 1.f; c.white_balance[2] = 2.07983f;
    // A deliberately per-channel profile, which is the case the old scalar lost.
    c.alpha_dng[0] = 1e-3f; c.alpha_dng[1] = 2e-3f; c.alpha_dng[2] = 3e-3f;
    c.beta_dng[0]  = 1e-6f; c.beta_dng[1]  = 2e-6f; c.beta_dng[2]  = 3e-6f;

    // 1.4 evaluates alpha[c]*x + beta[c] on the RAW value. For a pre-whitened
    // v = g*r that becomes (g*alpha)*v + g^2*beta, times 1/nsites for a guide
    // channel built by averaging that many Bayer sites.
    for (int ch = 0; ch < 3; ++ch) {
        const double g = c.white_balance[ch] / c.white_balance[1];
        const double w = (ch == 1) ? 0.5 : 1.0;   // two green sites
        char n1[80], n2[80];
        std::snprintf(n1, sizeof(n1), "channel %d alpha = g*alpha_dng*weight", ch);
        std::snprintf(n2, sizeof(n2), "channel %d beta  = g^2*beta_dng*weight", ch);
        eq(n1, c.noise_channel_alpha(ch), c.alpha_dng[ch] * g * w);
        eq(n2, c.noise_channel_beta(ch),  c.beta_dng[ch] * g * g * w);
    }
    // The averaged pair must still be exactly the mean of the three, so the SNR
    // estimate and the Monte Carlo curves see what they always did.
    double sa = 0, sb = 0;
    for (int ch = 0; ch < 3; ++ch) { sa += c.noise_channel_alpha(ch); sb += c.noise_channel_beta(ch); }
    eq("noise_alpha() == mean of the three channels", c.noise_alpha(), sa / 3.0);
    eq("noise_beta()  == mean of the three channels", c.noise_beta(),  sb / 3.0);

    // debug_noise_model_disabled must zero the per-channel form too.
    c.debug_noise_model_disabled = true;
    eq("noise model disabled zeroes channel alpha", c.noise_channel_alpha_robustness(1), 0.0);
    eq("noise model disabled zeroes channel beta",  c.noise_channel_beta_robustness(1), 0.0);
    c.debug_noise_model_disabled = false;

    // Un-prewhitened raw: the gain term must vanish, leaving 1.4's own numbers.
    c.raw_prewhitened = false;
    eq("no prewhitening -> alpha is the DNG's own (x weight)",
       c.noise_channel_alpha(0), c.alpha_dng[0]);
    std::printf("\n%d failed\n", fails);
    return fails ? 1 : 0;
}
