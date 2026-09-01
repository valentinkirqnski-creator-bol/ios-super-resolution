#include "robustness.h"
#include "parallel_for.h"

#include <cmath>
#include <algorithm>

namespace isacpu {

namespace {
int clamp_index(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }
}  // namespace

void compute_robustness_mask(const std::vector<Rgbf>& ref_half, const std::vector<Rgbf>& moved_half,
                             const std::vector<Vec2f>& shift_half, int width, int height,
                             float alpha, float beta, float threshold_m,
                             std::vector<Vec4f>& out) {
    auto shift_at = [&](int x, int y) { return shift_half[(size_t)y * width + x]; };

    parallel_for(height - 2, [&](int rowIdx) {
        int pxY = rowIdx + 1;
        for (int pxX = 1; pxX < width - 1; ++pxX) {
            Vec2f shiftf = shift_at(pxX, pxY);
            Vec2f maxShift = shiftf, minShift = shiftf;

            for (int y = -2; y <= 2; ++y) {
                for (int x = -2; x <= 2; ++x) {
                    Vec2f s = shift_at(clamp_index(pxX + x, width), clamp_index(pxY + y, height));
                    // ISA-VERBATIM, bug included (RobustnessModell.cu:67-70):
                    // the source compares each sample against the CONSTANT
                    // center value and OVERWRITES -- after the loop only the
                    // last sample (+2,+2) and the center survive, not the
                    // true 5x5 range. Reproduced literally per the
                    // identical-to-master requirement (an earlier revision
                    // accumulated the real window range here).
                    maxShift.x = std::max(s.x, shiftf.x);
                    maxShift.y = std::max(s.y, shiftf.y);
                    minShift.x = std::min(s.x, shiftf.x);
                    minShift.y = std::min(s.y, shiftf.y);
                }
            }

            int shx = (int)std::round(shiftf.x * 0.5f);
            int shy = (int)std::round(shiftf.y * 0.5f);

            Rgbf meanRef{}, meanMoved{};
            Rgbf pixelsRef[3][3];
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    Rgbf p = ref_half[(size_t)(pxY + y) * width + (pxX + x)];
                    pixelsRef[y + 1][x + 1] = p;
                    meanRef.r += p.r; meanRef.g += p.g; meanRef.b += p.b;

                    int ppy = clamp_index(pxY + shy + y, height);
                    int ppx = clamp_index(pxX + shx + x, width);
                    Rgbf pm = moved_half[(size_t)ppy * width + ppx];
                    meanMoved.r += pm.r; meanMoved.g += pm.g; meanMoved.b += pm.b;
                }
            }
            meanRef.r /= 9.f; meanRef.g /= 9.f; meanRef.b /= 9.f;
            meanMoved.r /= 9.f; meanMoved.g /= 9.f; meanMoved.b /= 9.f;

            float meandist = (std::fabs(meanRef.r - meanMoved.r) + std::fabs(meanRef.g - meanMoved.g) +
                             std::fabs(meanRef.b - meanMoved.b)) / 3.f;
            maxShift.x *= 0.5f * meandist; maxShift.y *= 0.5f * meandist;
            minShift.x *= 0.5f * meandist; minShift.y *= 0.5f * meandist;

            float M = std::sqrt((maxShift.x - minShift.x) * (maxShift.x - minShift.x) +
                                (maxShift.y - minShift.y) * (maxShift.y - minShift.y));

            Rgbf stdRef{};
            for (int y = 0; y < 3; ++y) {
                for (int x = 0; x < 3; ++x) {
                    const Rgbf& p = pixelsRef[y][x];
                    stdRef.r += (p.r - meanRef.r) * (p.r - meanRef.r);
                    stdRef.g += (p.g - meanRef.g) * (p.g - meanRef.g);
                    stdRef.b += (p.b - meanRef.b) * (p.b - meanRef.b);
                }
            }
            stdRef.r = std::sqrt(stdRef.r / 9.f);
            stdRef.g = std::sqrt(stdRef.g / 9.f);
            stdRef.b = std::sqrt(stdRef.b / 9.f);

            Rgbf sigmaMD;
            sigmaMD.r = std::sqrt(alpha * meanRef.r + beta);
            sigmaMD.g = std::sqrt(alpha * meanRef.g + beta) / std::sqrt(2.f);
            sigmaMD.b = std::sqrt(alpha * meanRef.b + beta);

            Rgbf dist;
            dist.r = std::fabs(meanRef.r - meanMoved.r);
            dist.g = std::fabs(meanRef.g - meanMoved.g);
            dist.b = std::fabs(meanRef.b - meanMoved.b);

            Rgbf sigma;
            sigma.r = std::max(sigmaMD.r, stdRef.r);
            sigma.g = std::max(sigmaMD.g, stdRef.g);
            sigma.b = std::max(sigmaMD.b, stdRef.b);

            dist.r *= stdRef.r * stdRef.r / (stdRef.r * stdRef.r + sigmaMD.r * sigmaMD.r);
            dist.g *= stdRef.g * stdRef.g / (stdRef.g * stdRef.g + sigmaMD.g * sigmaMD.g);
            dist.b *= stdRef.b * stdRef.b / (stdRef.b * stdRef.b + sigmaMD.b * sigmaMD.b);

            float s = (M > threshold_m) ? 0.f : 1.5f;
            const float t = 0.12f;

            Vec4f mask;
            mask.x = std::max(0.f, std::min(s * std::exp(-dist.r * dist.r / (sigma.r * sigma.r)) - t, 1.0f));
            mask.y = std::max(0.f, std::min(s * std::exp(-dist.g * dist.g / (sigma.g * sigma.g)) - t, 1.0f));
            mask.z = std::max(0.f, std::min(s * std::exp(-dist.b * dist.b / (sigma.b * sigma.b)) - t, 1.0f));
            mask.w = M;

            out[(size_t)pxY * width + pxX] = mask;
        }
    });
}

}  // namespace isacpu
