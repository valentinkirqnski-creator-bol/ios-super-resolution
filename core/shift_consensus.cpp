// See shift_consensus.h. Faithful CPU port of ImageStackAlignator's
// ShiftCollection + ShiftMinimizerKernels (M. Kunz, LGPL-3.0). Comments cite
// the originating lines so the correspondence stays auditable.
#include "shift_consensus.h"
#include <cmath>

namespace hhsr {

// ---- topology: identical to ShiftCollection.cs -----------------------------

// ShiftCollection.GetShiftCount (lines 115-134).
int ShiftConsensus::GetShiftCount() const {
    switch (strategy_) {
        case TrackingStrategy::Full:
            return (int)((frameCount_ - 1) * ((float)frameCount_ / 2.0f));
        case TrackingStrategy::OnlyOnReference:
            return frameCount_ - 1;
        case TrackingStrategy::OnReferenceBlock: {
            int shiftsInBlock = (int)((blockSize_ - 1) * ((float)blockSize_ / 2.0f));
            int framesOutsideBlock = frameCount_ - blockSize_;
            return shiftsInBlock + blockSize_ * framesOutsideBlock;
        }
        case TrackingStrategy::Blocks:
            return (int)(-0.5 * blockSize_ * (blockSize_ - 2 * frameCount_ + 1));
    }
    return 0;
}

// ShiftCollection.CreateShiftMatrix (lines 136-279). Column-major, lda = m:
// element (measurement `counter`, unknown `l`) lives at matrix[l*m + counter].
std::vector<f32> ShiftConsensus::CreateShiftMatrix() const {
    const int m = GetShiftCount();
    std::vector<f32> matrix((size_t)(frameCount_ - 1) * m, 0.f);
    switch (strategy_) {
        case TrackingStrategy::Full: {
            int counter = 0;
            for (int shifts = 0; shifts < (frameCount_ - 1); shifts++) {
                int count = (frameCount_ - 1) - (shifts);
                for (int line = 0; line < count; line++) {
                    for (int l = line; l <= line + (frameCount_ - 1) - count; l++)
                        matrix[(size_t)l * m + counter] = 1;
                    counter++;
                }
            }
        } break;
        case TrackingStrategy::OnlyOnReference: {
            for (int img = 0; img < frameCount_ - 1; img++) {
                if (img < referenceIndex_) {
                    int dist = referenceIndex_ - img;
                    for (int i = referenceIndex_ - dist; i < referenceIndex_; i++)
                        matrix[(size_t)i * m + img] = -1;
                }
                if (img >= referenceIndex_) {
                    int dist = img - referenceIndex_;
                    for (int i = referenceIndex_; i <= referenceIndex_ + dist; i++)
                        matrix[(size_t)i * m + img] = 1;
                }
            }
        } break;
        case TrackingStrategy::OnReferenceBlock: {
            int firstIndex, lastIndex;
            if (referenceIndex_ < frameCount_ / 2) {
                firstIndex = referenceIndex_ - blockSize_ / 2;
                if (firstIndex < 0) firstIndex = 0;
                lastIndex = firstIndex + blockSize_ - 1;
            } else {
                lastIndex = referenceIndex_ + blockSize_ / 2;
                if (lastIndex >= frameCount_) lastIndex = frameCount_ - 1;
                firstIndex = lastIndex - blockSize_ + 1;
            }
            int counter = 0;
            for (int shifts = 0; shifts < blockSize_ - 1; shifts++) {
                int count = blockSize_ - 1 - (shifts);
                for (int line = 0; line < count; line++) {
                    for (int l = line; l <= line + blockSize_ - 1 - count; l++)
                        matrix[(size_t)(l + firstIndex) * m + (counter)] = 1;
                    counter++;
                }
            }
            for (int i = 0; i < firstIndex; i++)
                for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++) {
                    for (int j = i; j < toBlock; j++) matrix[(size_t)j * m + counter] = -1;
                    counter++;
                }
            for (int i = lastIndex + 1; i < frameCount_; i++)
                for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++) {
                    for (int j = toBlock; j < i; j++) matrix[(size_t)j * m + counter] = 1;
                    counter++;
                }
        } break;
        case TrackingStrategy::Blocks: {
            int counter = 0;
            for (int shifts = 0; shifts < blockSize_; shifts++) {
                int count = (frameCount_ - 1) - (shifts);
                for (int line = 0; line < count; line++) {
                    for (int l = line; l <= line + (frameCount_ - 1) - count; l++)
                        matrix[(size_t)l * m + counter] = 1;
                    counter++;
                }
            }
        } break;
    }
    return matrix;
}

// ShiftCollection.FillShiftPairs (lines 281-383).
void ShiftConsensus::FillShiftPairs() {
    pairs_.clear();
    switch (strategy_) {
        case TrackingStrategy::Full:
            for (int reference = 0; reference < frameCount_; reference++)
                for (int toTrack = reference + 1; toTrack < frameCount_; toTrack++)
                    pairs_.push_back({reference, toTrack});
            break;
        case TrackingStrategy::OnlyOnReference:
            for (int toTrack = 0; toTrack < frameCount_; toTrack++)
                if (toTrack != referenceIndex_)
                    pairs_.push_back({referenceIndex_, toTrack});
            break;
        case TrackingStrategy::OnReferenceBlock: {
            int firstIndex, lastIndex;
            if (referenceIndex_ < frameCount_ / 2) {
                firstIndex = referenceIndex_ - blockSize_ / 2;
                if (firstIndex < 0) firstIndex = 0;
                lastIndex = firstIndex + blockSize_ - 1;
            } else {
                lastIndex = referenceIndex_ + blockSize_ / 2;
                if (lastIndex >= frameCount_) lastIndex = frameCount_ - 1;
                firstIndex = lastIndex - blockSize_ + 1;
            }
            for (int shifts = 0; shifts < blockSize_ - 1; shifts++) {
                int count = blockSize_ - 1 - (shifts);
                for (int line = 0; line < count; line++)
                    pairs_.push_back({line + firstIndex, line + firstIndex + blockSize_ - count});
            }
            for (int i = 0; i < firstIndex; i++)
                for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                    pairs_.push_back({toBlock, i});
            for (int i = lastIndex + 1; i < frameCount_; i++)
                for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                    pairs_.push_back({toBlock, i});
        } break;
        case TrackingStrategy::Blocks:
            for (int shifts = 0; shifts < blockSize_; shifts++) {
                int count = (frameCount_ - 1) - (shifts);
                for (int line = 0; line < count; line++)
                    pairs_.push_back({line, line + frameCount_ - count});
            }
            break;
    }
}

// ShiftCollection.FillIndexTable (lines 385-498).
void ShiftConsensus::FillIndexTable() {
    indices_.assign((size_t)frameCount_ * frameCount_, -1);
    auto IDX = [&](int i, int j) -> int& { return indices_[(size_t)i * frameCount_ + j]; };
    switch (strategy_) {
        case TrackingStrategy::Full: {
            int counter = 0;
            for (int distance = 1; distance < frameCount_; distance++)
                for (int frame = 0; frame + distance < frameCount_; frame++)
                    IDX(frame, frame + distance) = counter++;
        } break;
        case TrackingStrategy::OnlyOnReference: {
            int counter = 0;
            for (int toTrack = 0; toTrack < frameCount_; toTrack++)
                if (toTrack != referenceIndex_) IDX(referenceIndex_, toTrack) = counter++;
        } break;
        case TrackingStrategy::OnReferenceBlock: {
            int firstIndex, lastIndex;
            if (referenceIndex_ < frameCount_ / 2) {
                firstIndex = referenceIndex_ - blockSize_ / 2;
                if (firstIndex < 0) firstIndex = 0;
                lastIndex = firstIndex + blockSize_ - 1;
            } else {
                lastIndex = referenceIndex_ + blockSize_ / 2;
                if (lastIndex >= frameCount_) lastIndex = frameCount_ - 1;
                firstIndex = lastIndex - blockSize_ + 1;
            }
            int counter = 0;
            for (int shifts = 0; shifts < blockSize_ - 1; shifts++) {
                int count = blockSize_ - 1 - (shifts);
                for (int line = 0; line < count; line++)
                    IDX(line + firstIndex, line + firstIndex + blockSize_ - count) = counter++;
            }
            for (int i = 0; i < firstIndex; i++)
                for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                    IDX(toBlock, i) = counter++;
            for (int i = lastIndex + 1; i < frameCount_; i++)
                for (int toBlock = firstIndex; toBlock <= lastIndex; toBlock++)
                    IDX(toBlock, i) = counter++;
        } break;
        case TrackingStrategy::Blocks: {
            int counter = 0;
            for (int distance = 1; distance <= blockSize_; distance++)
                for (int frame = 0; frame + distance < frameCount_; frame++)
                    IDX(frame, frame + distance) = counter++;
        } break;
    }
}

ShiftConsensus::ShiftConsensus(int frameCount, int referenceIndex,
                               TrackingStrategy strategy, int blockSize)
    : frameCount_(frameCount), referenceIndex_(referenceIndex), strategy_(strategy) {
    blockSize_ = (blockSize >= frameCount) ? (frameCount - 1) : blockSize;
    if (blockSize_ < 1) blockSize_ = 1;
    FillShiftPairs();
    FillIndexTable();
}

// ---- per-tile solve: ShiftCollection.MinimizeCUBLAS (lines 740-824) +
//      ShiftMinimizerKernels.checkForOutliers (line 76) + getOptimalShifts
//      (line 184), done densely on the CPU, one small system per tile. --------

// Solve S (n x n, symmetric) * X = R (n x 2) in place via Gaussian elimination
// with partial pivoting. Returns false if singular (ISA: inversionInfo != 0 ->
// dead tile). S/R are row-major, length n*n and n*2.
static bool solve_nx2(f32* S, f32* R, int n) {
    for (int col = 0; col < n; ++col) {
        // partial pivot
        int piv = col; f32 best = std::fabs(S[(size_t)col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            f32 a = std::fabs(S[(size_t)r * n + col]);
            if (a > best) { best = a; piv = r; }
        }
        if (best <= 1e-12f) return false;
        if (piv != col) {
            for (int k = 0; k < n; ++k) std::swap(S[(size_t)col * n + k], S[(size_t)piv * n + k]);
            std::swap(R[(size_t)col * 2 + 0], R[(size_t)piv * 2 + 0]);
            std::swap(R[(size_t)col * 2 + 1], R[(size_t)piv * 2 + 1]);
        }
        f32 d = S[(size_t)col * n + col], inv = 1.f / d;
        for (int k = 0; k < n; ++k) S[(size_t)col * n + k] *= inv;
        R[(size_t)col * 2 + 0] *= inv; R[(size_t)col * 2 + 1] *= inv;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            f32 f = S[(size_t)r * n + col];
            if (f == 0.f) continue;
            for (int k = 0; k < n; ++k) S[(size_t)r * n + k] -= f * S[(size_t)col * n + k];
            R[(size_t)r * 2 + 0] -= f * R[(size_t)col * 2 + 0];
            R[(size_t)r * 2 + 1] -= f * R[(size_t)col * 2 + 1];
        }
    }
    return true;
}

bool ShiftConsensus::minimize(const std::vector<ShiftGrid>& measured,
                              std::vector<ShiftGrid>& out) const {
    const int m = shift_count();
    const int n1 = frameCount_ - 1;
    if ((int)measured.size() != m || m <= 0 || n1 <= 0) return false;
    const int ny = measured[0].ny, nx = measured[0].nx;
    if (ny <= 0 || nx <= 0) return false;
    for (const auto& g : measured)
        if (g.ny != ny || g.nx != nx || (int)g.v.size() != ny * nx * 2) return false;

    const int tileCount = ny * nx;
    const std::vector<f32> A0 = CreateShiftMatrix(); // shared incidence (safe copy)

    // measured[k] is pairs_[k] (reference-grouped order), but A's rows and the
    // index table are distance-grouped: ISA stores each measured shift via the
    // indices[from,to] accessor, so b's row for pair (i,j) is indices_[i,j],
    // NOT k. Precompute that permutation once.
    std::vector<int> row_of_pair(m);
    for (int k = 0; k < m; ++k) {
        int i = pairs_[k].reference, j = pairs_[k].toTrack;
        row_of_pair[k] = indices_[(size_t)i * frameCount_ + j];
    }

    out.assign(frameCount_, ShiftGrid{});
    for (int f = 0; f < frameCount_; ++f) {
        out[f].ny = ny; out[f].nx = nx;
        out[f].v.assign((size_t)tileCount * 2, 0.f);
    }

    // Per-tile scratch (A working copy is mutated by rejection, so each tile
    // starts from A0 afresh -- ISA's copyShiftMatrix + shiftSafeMatrices).
    std::vector<f32> A(A0.size());
    std::vector<f32> bx(m), by(m);           // measured, split x/y
    std::vector<f32> AtA((size_t)n1 * n1), Atb((size_t)n1 * 2);
    std::vector<f32> xseq((size_t)n1 * 2);   // the sequential one-to-one shifts

    for (int t = 0; t < tileCount; ++t) {
        A = A0;
        for (int k = 0; k < m; ++k) { int r = row_of_pair[k]; bx[r] = measured[k].dx(t); by[r] = measured[k].dy(t); }

        bool dead = false;
        for (int iter = 0; iter < maxIters_; ++iter) {
            // Normal equations AtA = A^T A, Atb = A^T b. A is col-major lda=m:
            // A(meas,unk) = A[unk*m + meas].
            std::fill(AtA.begin(), AtA.end(), 0.f);
            std::fill(Atb.begin(), Atb.end(), 0.f);
            for (int u = 0; u < n1; ++u) {
                const f32* Au = &A[(size_t)u * m];
                double ax = 0, ay = 0;
                for (int k = 0; k < m; ++k) { f32 a = Au[k]; if (a != 0.f) { ax += (double)a * bx[k]; ay += (double)a * by[k]; } }
                Atb[(size_t)u * 2 + 0] = (f32)ax; Atb[(size_t)u * 2 + 1] = (f32)ay;
                for (int v = u; v < n1; ++v) {
                    const f32* Av = &A[(size_t)v * m];
                    double s = 0;
                    for (int k = 0; k < m; ++k) s += (double)Au[k] * Av[k];
                    AtA[(size_t)u * n1 + v] = AtA[(size_t)v * n1 + u] = (f32)s;
                }
            }
            // x = (AtA)^-1 Atb
            xseq = Atb;
            std::vector<f32> S = AtA;
            if (!solve_nx2(S.data(), xseq.data(), n1)) { dead = true; break; }

            // reproject optim = A x, find worst residual (checkForOutliers).
            int idxMax = -1; f32 maxd = rejectPx2_;
            for (int k = 0; k < m; ++k) {
                double ox = 0, oy = 0;
                for (int u = 0; u < n1; ++u) {
                    f32 a = A[(size_t)u * m + k];
                    if (a != 0.f) { ox += (double)a * xseq[(size_t)u * 2 + 0]; oy += (double)a * xseq[(size_t)u * 2 + 1]; }
                }
                f32 dxk = bx[k] - (f32)ox, dyk = by[k] - (f32)oy;
                f32 dist = dxk * dxk + dyk * dyk;
                if (dist > maxd) { maxd = dist; idxMax = k; }
            }
            if (idxMax < 0) break; // converged: no measurement exceeds threshold
            // remove the worst outlier: zero its measurement and its A row.
            bx[idxMax] = 0.f; by[idxMax] = 0.f;
            for (int u = 0; u < n1; ++u) A[(size_t)u * m + idxMax] = 0.f;
        }
        if (dead) continue; // leave this tile at zero shift (ISA: status = -1)

        // getOptimalShifts: frame f -> reference = signed sum of sequential x.
        for (int f = 0; f < frameCount_; ++f) {
            if (f == referenceIndex_) continue;
            double sx = 0, sy = 0;
            if (referenceIndex_ < f) {
                for (int i = referenceIndex_; i < f; ++i) { sx += xseq[(size_t)i * 2 + 0]; sy += xseq[(size_t)i * 2 + 1]; }
            } else {
                for (int i = f; i < referenceIndex_; ++i) { sx -= xseq[(size_t)i * 2 + 0]; sy -= xseq[(size_t)i * 2 + 1]; }
            }
            out[f].v[(size_t)t * 2 + 0] = (f32)sx;
            out[f].v[(size_t)t * 2 + 1] = (f32)sy;
        }
    }
    return true;
}

} // namespace hhsr
