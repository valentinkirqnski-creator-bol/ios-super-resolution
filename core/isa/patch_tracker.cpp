#include "patch_tracker.h"
#include "fft.h"
#include "parallel_for.h"

#include <cmath>
#include <algorithm>
#include <cfloat>

namespace isacpu {

namespace {

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

}  // namespace

std::vector<float> convert_to_tiles_overlap_border(const std::vector<float>& img, int width, int height,
                                                   int tileSize, int maxShift, int tileCountX, int tileCountY,
                                                   Vec2f base_shift, float base_rotation) {
    int blockSize = tileSize + 2 * maxShift;
    int tileCount = tileCountX * tileCountY;
    std::vector<float> out((size_t)blockSize * blockSize * tileCount, 0.f);

    float sf = std::sin(base_rotation), cf = std::cos(base_rotation);

    parallel_for(tileCount, [&](int tileIdx) {
        int tileIdxY = tileIdx / tileCountX;
        int tileIdxX = tileIdx - tileIdxY * tileCountX;

        Vec2f shift;
        shift.x = cf * -base_shift.x - sf * -base_shift.y;
        shift.y = sf * -base_shift.x + cf * -base_shift.y;

        float patchCenterX = (float)(tileIdxX * tileSize + tileSize / 2 - width / 2);
        float patchCenterY = (float)(tileIdxY * tileSize + tileSize / 2 - height / 2);
        shift.x += cf * patchCenterX - sf * patchCenterY - patchCenterX;
        shift.y += sf * patchCenterX + cf * patchCenterY - patchCenterY;

        for (int pxY = 0; pxY < blockSize; ++pxY) {
            for (int pxX = 0; pxX < blockSize; ++pxX) {
                size_t o = (size_t)tileIdx * blockSize * blockSize + (size_t)pxY * blockSize + pxX;
                if (pxX < maxShift || pxY < maxShift || pxX >= tileSize + maxShift || pxY >= tileSize + maxShift)
                    continue;  // already 0

                int pxInImgX = tileIdxX * tileSize + pxX + (int)std::round(shift.x);
                int pxInImgY = tileIdxY * tileSize + pxY + (int)std::round(shift.y);
                pxInImgX = clampi(pxInImgX, 0, width - 1);
                pxInImgY = clampi(pxInImgY, 0, height - 1);
                out[o] = img[(size_t)pxInImgY * width + pxInImgX];
            }
        }
    });
    return out;
}

std::vector<float> convert_to_tiles_overlap_pre_shift(const std::vector<float>& img, int width, int height,
                                                      const std::vector<Vec2f>& pre_shift,
                                                      int tileSize, int maxShift, int tileCountX, int tileCountY,
                                                      Vec2f base_shift, float base_rotation) {
    int blockSize = tileSize + 2 * maxShift;
    int tileCount = tileCountX * tileCountY;
    std::vector<float> out((size_t)blockSize * blockSize * tileCount, 0.f);

    float sf = std::sin(base_rotation), cf = std::cos(base_rotation);

    parallel_for(tileCount, [&](int tileIdx) {
        int tileIdxY = tileIdx / tileCountX;
        int tileIdxX = tileIdx - tileIdxY * tileCountX;

        Vec2f shift = pre_shift[(size_t)tileIdxY * tileCountX + tileIdxX];
        shift.x += cf * -base_shift.x - sf * -base_shift.y;
        shift.y += sf * -base_shift.x + cf * -base_shift.y;

        float patchCenterX = (float)(tileIdxX * tileSize + tileSize / 2 - width / 2);
        float patchCenterY = (float)(tileIdxY * tileSize + tileSize / 2 - height / 2);
        shift.x += cf * patchCenterX - sf * patchCenterY - patchCenterX;
        shift.y += sf * patchCenterX + cf * patchCenterY - patchCenterY;

        int pxInImgX0 = tileIdxX * tileSize + (int)std::round(shift.x);
        int pxInImgY0 = tileIdxY * tileSize + (int)std::round(shift.y);

        for (int pxY = 0; pxY < blockSize; ++pxY) {
            for (int pxX = 0; pxX < blockSize; ++pxX) {
                size_t o = (size_t)tileIdx * blockSize * blockSize + (size_t)pxY * blockSize + pxX;
                int pxInImgX = clampi(pxInImgX0 + pxX, 0, width - 1);
                int pxInImgY = clampi(pxInImgY0 + pxY, 0, height - 1);
                out[o] = img[(size_t)pxInImgY * width + pxInImgX];
            }
        }
    });
    return out;
}

std::vector<float> squared_sum(const std::vector<float>& tiles, int maxShift, int tileSize, int tileCount) {
    int blockSize = tileSize + 2 * maxShift;
    std::vector<float> out(tileCount, 0.f);
    parallel_for(tileCount, [&](int tileIdx) {
        size_t tileArray = (size_t)tileIdx * blockSize * blockSize;
        float sum = 0.f;
        for (int y = 0; y < tileSize; ++y) {
            size_t yShift = (size_t)(y + maxShift) * blockSize;
            for (int x = 0; x < tileSize; ++x) {
                float v = tiles[tileArray + yShift + x + maxShift];
                sum += v * v;
            }
        }
        out[tileIdx] = sum;
    });
    return out;
}

std::vector<float> box_filter_with_border_x(const std::vector<float>& tiles, int maxShift, int tileSize, int tileCount) {
    int blockSize = tileSize + 2 * maxShift;
    std::vector<float> out((size_t)blockSize * blockSize * tileCount, 0.f);
    int lo = tileSize / 2, hi = maxShift * 2 + tileSize / 2;

    parallel_for(tileCount, [&](int tileIdx) {
        size_t base = (size_t)tileIdx * blockSize * blockSize;
        for (int pxY = 0; pxY < blockSize; ++pxY) {
            size_t row = base + (size_t)pxY * blockSize;
            for (int pxX = lo; pxX <= hi; ++pxX) {
                float acc = 0.f;
                for (int shift = -tileSize / 2; shift < tileSize / 2; ++shift) {
                    float v = tiles[row + (pxX + shift)];
                    acc += v * v;
                }
                out[row + pxX] = acc;
            }
        }
    });
    return out;
}

std::vector<float> box_filter_with_border_y(const std::vector<float>& tiles, int maxShift, int tileSize, int tileCount) {
    int blockSize = tileSize + 2 * maxShift;
    std::vector<float> out((size_t)blockSize * blockSize * tileCount, 0.f);
    int lo = tileSize / 2, hi = maxShift * 2 + tileSize / 2;

    parallel_for(tileCount, [&](int tileIdx) {
        size_t base = (size_t)tileIdx * blockSize * blockSize;
        for (int pxX = 0; pxX < blockSize; ++pxX) {
            for (int pxY = lo; pxY <= hi; ++pxY) {
                float acc = 0.f;
                for (int shift = -tileSize / 2; shift < tileSize / 2; ++shift) {
                    acc += tiles[base + (size_t)(pxY + shift) * blockSize + pxX];
                }
                out[base + (size_t)pxY * blockSize + pxX] = acc;
            }
        }
    });
    return out;
}

std::vector<float> normalized_cc(const std::vector<float>& cc_image, const std::vector<float>& squared_template,
                                 const std::vector<float>& box_filtered_image, int maxShift, int tileSize, int tileCount) {
    int blockSize = tileSize + 2 * maxShift;
    int shiftDim = 2 * maxShift + 1;
    std::vector<float> out((size_t)shiftDim * shiftDim * tileCount, 0.f);

    parallel_for(tileCount, [&](int tileIdx) {
        size_t blockBase = (size_t)tileIdx * blockSize * blockSize;
        size_t outBase = (size_t)tileIdx * shiftDim * shiftDim;
        for (int pxY = 0; pxY <= 2 * maxShift; ++pxY) {
            int shiftY = pxY - maxShift;
            int fftShiftY = shiftY < 0 ? blockSize + shiftY : shiftY;
            for (int pxX = 0; pxX <= 2 * maxShift; ++pxX) {
                int shiftX = pxX - maxShift;
                int fftShiftX = shiftX < 0 ? blockSize + shiftX : shiftX;

                size_t pxInCC = blockBase + (size_t)fftShiftY * blockSize + fftShiftX;
                size_t pxInBox = blockBase + (size_t)(blockSize / 2 + shiftY) * blockSize + (blockSize / 2 + shiftX);
                size_t pxOut = outBase + (size_t)pxY * shiftDim + pxX;

                out[pxOut] = squared_template[tileIdx] + box_filtered_image[pxInBox] - 2.f * cc_image[pxInCC];
            }
        }
    });
    return out;
}

namespace {
constexpr float FA11[9] = {1.f / 4, -2.f / 4, 1.f / 4, 2.f / 4, -4.f / 4, 2.f / 4, 1.f / 4, -2.f / 4, 1.f / 4};
constexpr float FA22[9] = {1.f / 4, 2.f / 4, 1.f / 4, -2.f / 4, -4.f / 4, -2.f / 4, 1.f / 4, 2.f / 4, 1.f / 4};
constexpr float FA12[9] = {1.f / 4, 0.f, -1.f / 4, 0.f, 0.f, 0.f, -1.f / 4, 0.f, 1.f / 4};
constexpr float Fb1[9] = {-1.f / 8, 0.f, 1.f / 8, -2.f / 8, 0.f, 2.f / 8, -1.f / 8, 0.f, 1.f / 8};
constexpr float Fb2[9] = {-1.f / 8, -2.f / 8, -1.f / 8, 0.f, 0.f, 0.f, 1.f / 8, 2.f / 8, 1.f / 8};
}  // namespace

std::vector<Vec2f> find_minimum(const std::vector<float>& shift_image, int maxShift, int tileCountX, int tileCountY,
                                float threshold) {
    int tileCount = tileCountX * tileCountY;
    int shiftDim = 2 * maxShift + 1;
    int pixelsInTile = shiftDim * shiftDim;
    std::vector<Vec2f> out(tileCount, Vec2f{0, 0});

    parallel_for(tileCount, [&](int tileIdx) {
        size_t zOffset = (size_t)tileIdx * pixelsInTile;

        float minVal = FLT_MAX, maxVal = -FLT_MAX;
        int minIdx = -1;
        for (int i = 0; i < pixelsInTile; ++i) {
            float v = shift_image[zOffset + i];
            maxVal = std::max(maxVal, v);
            if (v < minVal) { minVal = v; minIdx = i; }
        }

        int cy = minIdx / shiftDim;
        int cx = minIdx - cy * shiftDim;
        Vec2f coord{(float)cx, (float)cy};

        if (coord.x < 1 || coord.y < 1 || coord.x >= 2 * maxShift || coord.y >= 2 * maxShift) {
            coord = Vec2f{0, 0};
        } else {
            float A11 = 0, A22 = 0, A12 = 0, b1 = 0, b2 = 0;
            auto accum = [&](int i, int idx) {
                float img = shift_image[zOffset + idx];
                A11 += FA11[i] * img; A22 += FA22[i] * img; A12 += FA12[i] * img;
                b1 += Fb1[i] * img; b2 += Fb2[i] * img;
            };
            for (int i = 0; i < 3; ++i) accum(i, minIdx + i - 1 - shiftDim);
            for (int i = 3; i < 6; ++i) accum(i, minIdx + i - 4);
            for (int i = 6; i < 9; ++i) accum(i, minIdx + i - 7 + shiftDim);

            A11 = std::max(A11, 0.f);
            A22 = std::max(A22, 0.f);

            float detA = A11 * A22 - A12 * A12;
            if (detA < 0.f) { A12 = 0.f; detA = A11 * A22; }

            if (detA != 0.f) {
                float muX = (A22 * b1 - A12 * b2) / detA;
                float muY = (A11 * b2 - A12 * b1) / detA;
                if (std::fabs(muX) > 1.f) muX = 0.f;
                if (std::fabs(muY) > 1.f) muY = 0.f;
                coord.x -= muX;
                coord.y -= muY;
            }
            coord.x -= maxShift;
            coord.y -= maxShift;
        }

        if (threshold + minVal > maxVal) coord = Vec2f{0, 0};
        out[tileIdx] = coord;
    });
    return out;
}

void track(const std::vector<float>& img_to_track, const std::vector<float>& img_ref, int width, int height,
          std::vector<Vec2f>& pre_shift, int tileSize, int maxShift, int tileCountX, int tileCountY,
          Vec2f base_shift_ref, float base_rotation_ref, Vec2f base_shift_to_track, float base_rotation_to_track,
          float threshold) {
    int blockSize = tileSize + 2 * maxShift;
    int tileCount = tileCountX * tileCountY;

    auto ref_tiles = convert_to_tiles_overlap_border(img_ref, width, height, tileSize, maxShift, tileCountX, tileCountY,
                                                     base_shift_ref, base_rotation_ref);
    auto track_tiles = convert_to_tiles_overlap_pre_shift(img_to_track, width, height, pre_shift, tileSize, maxShift,
                                                          tileCountX, tileCountY, base_shift_to_track, base_rotation_to_track);

    // This per-tile FFT cross-correlation is the dominant cost of patch
    // tracking (thousands of independent small FFTs) and is embarrassingly
    // parallel: each tile only ever reads its own slice of ref_tiles/
    // track_tiles and writes its own disjoint slice of cc_image.
    std::vector<float> cc_image((size_t)blockSize * blockSize * tileCount);
    parallel_for(tileCount, [&](int t) {
        std::vector<float> ref_tile(ref_tiles.begin() + (size_t)t * blockSize * blockSize,
                                    ref_tiles.begin() + (size_t)(t + 1) * blockSize * blockSize);
        std::vector<float> track_tile(track_tiles.begin() + (size_t)t * blockSize * blockSize,
                                      track_tiles.begin() + (size_t)(t + 1) * blockSize * blockSize);

        auto ref_spec = rfft2d(ref_tile, blockSize, blockSize);
        auto track_spec = rfft2d(track_tile, blockSize, blockSize);
        for (size_t i = 0; i < ref_spec.size(); ++i) track_spec[i] = std::conj(ref_spec[i]) * track_spec[i];
        auto corr = irfft2d(track_spec, blockSize, blockSize);

        float norm = 1.f / (float)(blockSize * blockSize);
        for (size_t i = 0; i < corr.size(); ++i)
            cc_image[(size_t)t * blockSize * blockSize + i] = corr[i] * norm;
    });

    auto squared_template = squared_sum(ref_tiles, maxShift, tileSize, tileCount);
    auto boxed_x = box_filter_with_border_x(track_tiles, maxShift, tileSize, tileCount);
    auto boxed_xy = box_filter_with_border_y(boxed_x, maxShift, tileSize, tileCount);

    auto shift_image = normalized_cc(cc_image, squared_template, boxed_xy, maxShift, tileSize, tileCount);
    auto patch_shift = find_minimum(shift_image, maxShift, tileCountX, tileCountY, threshold);

    for (int i = 0; i < tileCount; ++i) {
        pre_shift[i].x += patch_shift[i].x;
        pre_shift[i].y += patch_shift[i].y;
    }
}

}  // namespace isacpu
