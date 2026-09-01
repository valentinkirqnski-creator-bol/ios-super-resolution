#include "shift_minimizer.h"

#include <cmath>
#include <algorithm>

namespace isacpu {

namespace {

void mat_mul(const float* A, int ar, int ac, const float* B, int br, int bc, float* C) {
    (void)br;
    for (int r = 0; r < ar; ++r)
        for (int c = 0; c < bc; ++c) {
            float sum = 0.f;
            for (int k = 0; k < ac; ++k) sum += A[r * ac + k] * B[k * bc + c];
            C[r * bc + c] = sum;
        }
}

void mat_transpose(const float* A, int ar, int ac, float* At) {
    for (int r = 0; r < ar; ++r)
        for (int c = 0; c < ac; ++c) At[c * ar + r] = A[r * ac + c];
}

// Gauss-Jordan inversion of an n x n row-major matrix, in place. Returns
// false (M left unspecified) if singular -- matches ISA's own
// MatinvBatchedS/Getrf+Getri failure signaled via `inversionInfo != 0`.
bool mat_invert(std::vector<float>& M, int n) {
    std::vector<float> aug((size_t)n * 2 * n, 0.f);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) aug[(size_t)r * 2 * n + c] = M[(size_t)r * n + c];
        aug[(size_t)r * 2 * n + n + r] = 1.f;
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        float best = std::fabs(aug[(size_t)col * 2 * n + col]);
        for (int r = col + 1; r < n; ++r) {
            float v = std::fabs(aug[(size_t)r * 2 * n + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-12f) return false;
        if (pivot != col)
            for (int c = 0; c < 2 * n; ++c) std::swap(aug[(size_t)col * 2 * n + c], aug[(size_t)pivot * 2 * n + c]);

        float pv = aug[(size_t)col * 2 * n + col];
        for (int c = 0; c < 2 * n; ++c) aug[(size_t)col * 2 * n + c] /= pv;

        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            float f = aug[(size_t)r * 2 * n + col];
            if (f == 0.f) continue;
            for (int c = 0; c < 2 * n; ++c) aug[(size_t)r * 2 * n + c] -= f * aug[(size_t)col * 2 * n + c];
        }
    }
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) M[(size_t)r * n + c] = aug[(size_t)r * 2 * n + n + c];
    return true;
}

}  // namespace

std::vector<ShiftPair> shift_pairs_full(int frame_count) {
    std::vector<ShiftPair> pairs;
    for (int distance = 1; distance < frame_count; ++distance)
        for (int frame = 0; frame + distance < frame_count; ++frame)
            pairs.push_back(ShiftPair{frame, frame + distance});
    return pairs;
}

std::vector<float> shift_matrix_full(int frame_count) {
    int n1 = frame_count - 1;
    auto pairs = shift_pairs_full(frame_count);
    int m = (int)pairs.size();
    std::vector<float> A((size_t)m * n1, 0.f);
    for (int row = 0; row < m; ++row)
        for (int col = pairs[row].reference; col < pairs[row].to_track; ++col)
            A[(size_t)row * n1 + col] = 1.f;
    return A;
}

std::vector<ShiftPair> shift_pairs_only_reference(int frame_count, int reference_index) {
    std::vector<ShiftPair> pairs;
    for (int f = 0; f < frame_count; ++f)
        if (f != reference_index) pairs.push_back(ShiftPair{reference_index, f});
    return pairs;
}

std::vector<float> shift_matrix_only_reference(int frame_count, int reference_index) {
    int n1 = frame_count - 1;
    auto pairs = shift_pairs_only_reference(frame_count, reference_index);
    int m = (int)pairs.size();
    std::vector<float> A((size_t)m * n1, 0.f);
    for (int row = 0; row < m; ++row) {
        int ref = pairs[row].reference, trk = pairs[row].to_track;
        if (trk > ref) {
            for (int col = ref; col < trk; ++col) A[(size_t)row * n1 + col] = 1.f;
        } else {
            for (int col = trk; col < ref; ++col) A[(size_t)row * n1 + col] = -1.f;
        }
    }
    return A;
}

std::vector<Vec2f> minimize_shifts_for_tile(const std::vector<Vec2f>& measured, const std::vector<float>& design,
                                           int frame_count) {
    int n1 = frame_count - 1;
    int m = (int)measured.size();

    std::vector<float> A = design;      // working copy, rows zeroed as outliers are found
    std::vector<Vec2f> b = measured;    // working copy, same
    std::vector<Vec2f> x(n1, Vec2f{0, 0});

    for (int iter = 0; iter < 10; ++iter) {
        std::vector<float> At((size_t)n1 * m);
        mat_transpose(A.data(), m, n1, At.data());

        std::vector<float> AtA((size_t)n1 * n1);
        mat_mul(At.data(), n1, m, A.data(), m, n1, AtA.data());

        if (!mat_invert(AtA, n1)) break;  // singular: stop, keep the last valid solve

        std::vector<float> solved((size_t)n1 * m);
        mat_mul(AtA.data(), n1, n1, At.data(), n1, m, solved.data());

        std::vector<float> bx(m), by(m);
        for (int i = 0; i < m; ++i) { bx[i] = b[i].x; by[i] = b[i].y; }
        std::vector<float> xx(n1), xy(n1);
        mat_mul(solved.data(), n1, m, bx.data(), m, 1, xx.data());
        mat_mul(solved.data(), n1, m, by.data(), m, 1, xy.data());
        for (int i = 0; i < n1; ++i) x[i] = Vec2f{xx[i], xy[i]};

        std::vector<Vec2f> b_optim(m);
        for (int r = 0; r < m; ++r) {
            float sx = 0.f, sy = 0.f;
            for (int k = 0; k < n1; ++k) { sx += A[(size_t)r * n1 + k] * x[k].x; sy += A[(size_t)r * n1 + k] * x[k].y; }
            b_optim[r] = Vec2f{sx, sy};
        }

        float maxDist = 1.0f;
        int idxMax = -1;
        for (int i = 0; i < m; ++i) {
            float dx = b[i].x - b_optim[i].x, dy = b[i].y - b_optim[i].y;
            float dist = dx * dx + dy * dy;
            if (dist > maxDist) { maxDist = dist; idxMax = i; }
        }
        if (idxMax == -1) break;  // converged: no residual exceeds the 1px^2 threshold

        b[idxMax] = Vec2f{0, 0};
        for (int col = 0; col < n1; ++col) A[(size_t)idxMax * n1 + col] = 0.f;
    }
    return x;
}

Vec2f optimal_shift_from_increments(const std::vector<Vec2f>& increments, int reference_index, int image_to_track) {
    Vec2f total{0, 0};
    if (reference_index < image_to_track) {
        for (int i = reference_index; i < image_to_track; ++i) { total.x += increments[i].x; total.y += increments[i].y; }
    } else if (image_to_track < reference_index) {
        for (int i = image_to_track; i < reference_index; ++i) { total.x -= increments[i].x; total.y -= increments[i].y; }
    }
    return total;
}

}  // namespace isacpu
