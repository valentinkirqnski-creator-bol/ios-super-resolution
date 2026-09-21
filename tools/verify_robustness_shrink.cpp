// The Wiener shrink in the robustness mask, against Python 1.4's guard.
//
//   g++ -std=c++17 -O2 tools/verify_robustness_shrink.cpp -o verify_robustness_shrink
//   ./verify_robustness_shrink       (exit 0 iff every case matches 1.4)
//
// 1.4 (robustness.py) writes:
//
//     sigma_sq_ = max(sigma_sq_, sigma_noise_sq)
//     if d_sq_ > 0:
//         shrink = d_sq_ / (d_sq_ + d_noise_sq)
//         d_sq_ *= shrink * shrink
//
// The guard is load-bearing in two opposite ways, and an unguarded divide gets
// one of them backwards:
//
//   d = 0, d_noise = 0   the pixel matched EXACTLY and the noise model is off.
//                        1.4 leaves d_sq_ = 0, so R lands at its maximum.
//                        Unguarded this is 0/0 = NaN, which then poisons every
//                        merge accumulator that touches the pixel. Reached on a
//                        blown highlight where both frames sit at 1.0, or a
//                        crushed black where both sit at 0.
//
//   d = +inf             an out-of-bounds Dodgson sample, +inf BY DESIGN so that
//                        R = 0. The guard passes (inf > 0), shrink is inf/inf =
//                        NaN as before, and the downstream !isfinite(r_val) -> 0
//                        still gives the intended rejection. Clamping r_val
//                        instead of guarding the divide would force the first
//                        case to 0 too, which is the opposite of correct.
#include <cmath>
#include <cstdio>
#include <limits>

static int fails = 0;

// The shipped form, transcribed from rob_make_mask / compute_robustness.
static float shrunk(float d_ms_sq, float d_md_sq) {
    if (d_ms_sq > 0.f) {
        const float shrink = d_ms_sq / (d_ms_sq + d_md_sq);
        return d_ms_sq * shrink * shrink;
    }
    return 0.f;
}

// 1.4's, transcribed from robustness.py.
static float ref_14(float d_sq, float d_noise_sq) {
    if (d_sq > 0.f) {
        const float shrink = d_sq / (d_sq + d_noise_sq);
        d_sq *= shrink * shrink;
    }
    return d_sq;
}

static void check(const char* what, float d, float dn) {
    const float got = shrunk(d, dn);
    const float want = ref_14(d, dn);
    const bool both_nan = std::isnan(got) && std::isnan(want);
    const bool ok = both_nan || got == want;
    if (!ok) ++fails;
    std::printf("  %-46s d=%-9.4g dn=%-9.4g -> %-10.6g (1.4: %-10.6g) %s\n",
                what, d, dn, got, want, ok ? "OK" : "FAIL");
}

int main() {
    const float inf = std::numeric_limits<float>::infinity();
    std::printf("Wiener shrink vs Python 1.4's guarded form\n\n");

    std::printf("-- noise model OFF (d_noise = 0), which is where 0/0 lives --\n");
    check("matched exactly: blown highlight / black", 0.f, 0.f);
    check("normal difference, no noise term", 0.04f, 0.f);
    check("tiny difference, no noise term", 1e-12f, 0.f);

    std::printf("\n-- noise model ON --\n");
    check("matched exactly, noise term present", 0.f, 1e-4f);
    check("difference below the noise", 1e-5f, 1e-3f);
    check("difference above the noise", 0.04f, 1e-4f);

    std::printf("\n-- out-of-bounds sample: +inf BY DESIGN, must stay NaN --\n");
    check("inf difference, noise on", inf, 1e-4f);
    check("inf difference, noise off", inf, 0.f);

    std::printf("\n-- the guarded case must not be NaN --\n");
    const bool clean = !std::isnan(shrunk(0.f, 0.f));
    if (!clean) ++fails;
    std::printf("  %-46s %s\n", "0/0 yields a finite 0, not NaN", clean ? "OK" : "FAIL");
    // And it must be the value that makes R maximal: exp(-0/sigma) = 1.
    const float r_factor = std::exp(-shrunk(0.f, 0.f) / 0.01f);
    const bool maximal = std::fabs(r_factor - 1.f) < 1e-6f;
    if (!maximal) ++fails;
    std::printf("  %-46s exp(-d/sigma) = %.6f %s\n",
                "a perfectly matched pixel scores fully trusted", r_factor,
                maximal ? "OK" : "FAIL");

    std::printf("\n%d failed\n", fails);
    return fails ? 1 : 0;
}
