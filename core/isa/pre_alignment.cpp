#include "pre_alignment.h"
#include "affine_warp.h"
#include "parallel_for.h"

#include <cmath>
#include <algorithm>
#include <cfloat>

namespace isacpu {

void conjugate_complex_mul(const std::vector<Cplx>& a, std::vector<Cplx>& b) {
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        Cplx valA = std::conj(a[i]);
        b[i] = valA * b[i];
    }
}

void fourier_filter(std::vector<Cplx>& spectrum, int width, int height,
                    int clear_axis, float lp, float hp, float lps, float hps) {
    lp = lp - lps;
    hp = hp + hps;

    for (int ky = 0; ky < height; ++ky) {
        for (int kx = 0; kx < width; ++kx) {
            float mx = (kx <= width / 2) ? (float)kx : (float)(kx - width);
            float my = (ky > height * 0.5f) ? -(float)(height - ky) : (float)ky;
            mx /= (float)width;
            my /= (float)height;

            float dist = std::sqrt(mx * mx + my * my);
            float fil = 0.f;

            if (lp > 0.f) {
                if (dist <= lp) fil = 1.f;
            } else {
                if (dist <= 1.0f) fil = 1.f;
            }
            if (lps > 0.f) {
                float fil2 = (dist < lp) ? 1.f : 0.f;
                fil2 = (-fil + 1.f) * std::exp(-((dist - lp) * (dist - lp) / (2.f * lps * lps)));
                if (fil2 > 0.001f) fil = fil2;
            }
            if (lps > 0.f && lp == 0.f && hp == 0.f && hps == 0.f) {
                fil = std::exp(-((dist - lp) * (dist - lp) / (2.f * lps * lps)));
            }
            if (hp > 0.f) {
                float fil2 = (dist >= hp) ? 1.f : 0.f;
                fil *= fil2;
                if (hps > 0.f) {
                    float fil3 = (dist < hp) ? 1.f : 0.f;
                    fil3 = (-fil2 + 1.f) * std::exp(-((dist - hp) * (dist - hp) / (2.f * hps * hps)));
                    if (fil3 > 0.001f) fil = fil3;
                }
            }

            Cplx erg = spectrum[(size_t)ky * width + kx];
            erg *= fil;

            int kx_dist = (kx <= width / 2) ? kx : (width - kx);
            if (kx_dist < clear_axis || std::fabs(my) * height < clear_axis) erg = Cplx(0.f, 0.f);

            spectrum[(size_t)ky * width + kx] = erg;
        }
    }
}

RotationResult scan_angles(const std::vector<float>& ref_img, const std::vector<float>& tracked_img,
                          int width, int height, float incr, float range, float zero_deg) {
    auto ref_spectrum = rfft2d(ref_img, width, height);
    float norm = 1.f / (float)((size_t)width * height);

    struct Candidate { float val, ang, bx, by; };

    // Each candidate angle's FFT correlation is independent -- evaluate the
    // whole batch across threads, then reduce sequentially (in the same
    // ascending-angle order the original serial loop used) so tie-breaking
    // matches exactly: first strictly-greater value wins.
    auto eval_angle = [&](float ang) -> Candidate {
        std::vector<float> warped((size_t)width * height, 0.f);
        Mat3 m = mat3_rot_around_center(ang, (float)width, (float)height);
        warp_affine(tracked_img, width, height, m, warped);

        auto spec = rfft2d(warped, width, height);
        conjugate_complex_mul(ref_spectrum, spec);
        auto corr = irfft2d(spec, width, height);

        float v = -FLT_MAX;
        int bx = 0, by = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float val = corr[(size_t)y * width + x] * norm;
                if (val > v) { v = val; bx = x; by = y; }
            }
        }
        return Candidate{v, ang, (float)bx, (float)by};
    };

    auto eval_batch = [&](const std::vector<float>& angles) {
        std::vector<Candidate> results(angles.size());
        parallel_for((int)angles.size(), [&](int i) { results[i] = eval_angle(angles[i]); });
        return results;
    };

    // The angle sweeps accumulate in DOUBLE, exactly like ScanAngles' C#
    // (`double ang`, PreAlignment.cs:176/192/222). Accumulating in float
    // dropped the topmost fine-sweep angle whenever the coarse best equalled
    // `zero` (float 0.9000002 + 0.1 = 1.0000002 > 1.0f skips the endpoint the
    // double loop reaches at 0.9999999999999998) -- 20 candidate angles where
    // the original evaluates 21. maxVal is double for the same parity.
    double maxVal = -1e300;
    float maxAng = 0.f, maxX = 0.f, maxY = 0.f;
    auto reduce = [&](const std::vector<Candidate>& results) {
        for (auto& c : results) if ((double)c.val > maxVal) { maxVal = c.val; maxAng = c.ang; maxX = c.bx; maxY = c.by; }
    };

    // Coarse search.
    std::vector<float> coarse_angles;
    for (double ang = (double)zero_deg - (double)range; ang <= (double)zero_deg + (double)range;
         ang += 5.0 * (double)incr)
        coarse_angles.push_back((float)ang);
    reduce(eval_batch(coarse_angles));

    // Fine search around the coarse best (maxVal is NOT reset -- matches
    // the source, which shares one accumulator across both loops).
    std::vector<float> fine_angles;
    const double zero2 = (double)maxAng, range2 = 10.0 * (double)incr;
    for (double ang = zero2 - range2; ang <= zero2 + range2; ang += (double)incr)
        fine_angles.push_back((float)ang);
    reduce(eval_batch(fine_angles));

    if (maxX > width / 2) maxX -= width;
    if (maxY > height / 2) maxY -= height;

    return RotationResult{-maxX, -maxY, maxAng, (float)maxVal};
}

void fourier_filter_apply(std::vector<float>& img, int width, int height,
                          int clear_axis, float high_pass, float high_pass_sigma) {
    auto spectrum = rfft2d(img, width, height);
    fourier_filter(spectrum, width, height, clear_axis, /*lp=*/1.f, high_pass, /*lps=*/1.f, high_pass_sigma);
    auto filtered = irfft2d(spectrum, width, height);

    float norm = 1.f / (float)((size_t)width * height);
    for (auto& v : filtered) v *= norm;

    // NPP ThresholdLTGT(0, 0, 1, 1) (PreAlignment.cs:280) is a CLAMP: values
    // below 0 become 0, above 1 become 1, and everything in [0,1] passes
    // through UNCHANGED -- a graded, gradient-weighted image, not a binary
    // mask. An earlier revision binarized here (>0 -> 1), which replaced
    // every pre-alignment / tracking correlation input with a mask; caught by
    // the source-identity audit.
    for (size_t i = 0; i < img.size(); ++i)
        img[i] = std::min(1.f, std::max(0.f, filtered[i]));
}

}  // namespace isacpu
