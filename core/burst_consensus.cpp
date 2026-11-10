// Burst consensus check on the motion vectors (see burst_consensus.h).
#include "burst_consensus.h"
#include "parallel.h"
#include <algorithm>
#include <cmath>

namespace hhsr {

std::vector<float> consensus_tile_structure(const Image& ref_grey, int raw_h, int raw_w,
                                            int ny, int nx, int tile_size, const Config& cfg) {
    std::vector<float> out((size_t)std::max(0, ny) * std::max(0, nx) * 4, 0.f);
    if (ny <= 0 || nx <= 0 || raw_w <= 0 || raw_h <= 0 || ref_grey.h < 3 || ref_grey.w < 3) return out;
    const int ts = std::max(1, tile_size);
    const float s = (float)ref_grey.w / (float)raw_w;           // grey px per raw px
    const float na = std::max(0.f, cfg.alpha_dng[1]), nb = std::max(1e-12f, cfg.beta_dng[1]);
    parallel_rows(ny, cfg.num_threads, [&](int ty) {
        for (int tx = 0; tx < nx; ++tx) {
            // The tile's footprint on the grey grid (the merge's int(lr / ts) tile).
            const int y0 = std::max(1, (int)std::floor((float)(ty * ts) * s));
            const int y1 = std::min(ref_grey.h - 1, (int)std::floor((float)((ty + 1) * ts) * s));
            const int x0 = std::max(1, (int)std::floor((float)(tx * ts) * s));
            const int x1 = std::min(ref_grey.w - 1, (int)std::floor((float)((tx + 1) * ts) * s));
            if (y1 <= y0 || x1 <= x0) continue;
            double hxx = 0, hxy = 0, hyy = 0, msum = 0;
            int n = 0;
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x) {
                    const float gx = 0.5f * (ref_grey.at(y, x + 1) - ref_grey.at(y, x - 1));
                    const float gy = 0.5f * (ref_grey.at(y + 1, x) - ref_grey.at(y - 1, x));
                    hxx += gx * gx; hxy += gx * gy; hyy += gy * gy; msum += ref_grey.at(y, x); ++n;
                }
            if (!n) continue;
            const double tr = hxx + hyy, dsc = std::sqrt(std::max(0.0, 0.25 * (hxx - hyy) * (hxx - hyy) + hxy * hxy));
            const double l1 = 0.5 * tr + dsc;
            const double nv = na * std::max(0.0, msum / n) + nb;
            if (l1 / n < 2.0 * nv) continue;                    // flat: misalignment invisible
            float* o = &out[((size_t)ty * nx + tx) * 4];
            o[0] = (float)(hxx / l1); o[1] = (float)(hxy / l1); o[2] = (float)(hyy / l1);
            // Noise-only expected d^T W d of a vector difference (both vectors noisy),
            // as in fb_tile_error: ~8 nv / lambda1 in grey px^2 -> raw px^2.
            o[3] = (float)(8.0 * nv / l1 / ((double)s * s));
        }
    });
    return out;
}

namespace {

// Robust (IRLS, Huber delta) affine fit of a frame's tile vectors over structured tiles.
// Coordinates normalised to ~[-1, 1] for conditioning. G(t) written into g (ny*nx*2).
void fit_affine(const float* F, const std::vector<float>& ts4, int ny, int nx, int ts,
                std::vector<float>& g) {
    const size_t nt = (size_t)ny * nx;
    g.assign(nt * 2, 0.f);
    const double cxm = 0.5 * nx * ts, cym = 0.5 * ny * ts;
    std::vector<double> w(nt, 0.0);
    size_t n_ok = 0;
    for (size_t t = 0; t < nt; ++t) {
        const bool ok = ts4[t * 4 + 0] + ts4[t * 4 + 2] > 0.f && std::isfinite(F[2 * t]) && std::isfinite(F[2 * t + 1]);
        w[t] = ok ? 1.0 : 0.0; n_ok += ok;
    }
    double ax[3] = {0, 0, 0}, ay[3] = {0, 0, 0};
    if (n_ok < 12) {   // too few structured tiles: a robust (median) translation
        std::vector<float> vx, vy;
        for (size_t t = 0; t < nt; ++t)
            if (std::isfinite(F[2 * t]) && std::isfinite(F[2 * t + 1])) { vx.push_back(F[2 * t]); vy.push_back(F[2 * t + 1]); }
        if (vx.empty()) return;
        std::nth_element(vx.begin(), vx.begin() + vx.size() / 2, vx.end());
        std::nth_element(vy.begin(), vy.begin() + vy.size() / 2, vy.end());
        for (size_t t = 0; t < nt; ++t) { g[2 * t] = vx[vx.size() / 2]; g[2 * t + 1] = vy[vy.size() / 2]; }
        return;
    }
    const double delta = 1.0;   // raw px
    for (int it = 0; it < 5; ++it) {
        double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, bx[3] = {0, 0, 0}, by[3] = {0, 0, 0};
        for (int ty = 0; ty < ny; ++ty)
            for (int tx = 0; tx < nx; ++tx) {
                const size_t t = (size_t)ty * nx + tx;
                if (w[t] <= 0.0) continue;
                const double u = (((tx + 0.5) * ts) - cxm) / cxm, v = (((ty + 0.5) * ts) - cym) / cym;
                const double p[3] = {1.0, u, v};
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) A[i][j] += w[t] * p[i] * p[j];
                    bx[i] += w[t] * p[i] * F[2 * t]; by[i] += w[t] * p[i] * F[2 * t + 1];
                }
            }
        // 3x3 solve (Cramer).
        auto det3 = [](const double M[3][3]) {
            return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                   M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        };
        const double D = det3(A);
        if (!(std::fabs(D) > 1e-12)) break;
        for (int c = 0; c < 3; ++c) {
            double Mx[3][3], My[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) { Mx[i][j] = (j == c) ? bx[i] : A[i][j]; My[i][j] = (j == c) ? by[i] : A[i][j]; }
            ax[c] = det3(Mx) / D; ay[c] = det3(My) / D;
        }
        // Huber reweighting on the 2-D residual (structured tiles only).
        for (int ty = 0; ty < ny; ++ty)
            for (int tx = 0; tx < nx; ++tx) {
                const size_t t = (size_t)ty * nx + tx;
                if (w[t] <= 0.0) continue;
                const double u = (((tx + 0.5) * ts) - cxm) / cxm, v = (((ty + 0.5) * ts) - cym) / cym;
                const double rx = F[2 * t] - (ax[0] + ax[1] * u + ax[2] * v);
                const double ry = F[2 * t + 1] - (ay[0] + ay[1] * u + ay[2] * v);
                const double r = std::sqrt(rx * rx + ry * ry);
                w[t] = r <= delta ? 1.0 : delta / r;
            }
    }
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            const size_t t = (size_t)ty * nx + tx;
            const double u = (((tx + 0.5) * ts) - cxm) / cxm, v = (((ty + 0.5) * ts) - cym) / cym;
            g[2 * t] = (float)(ax[0] + ax[1] * u + ax[2] * v);
            g[2 * t + 1] = (float)(ay[0] + ay[1] * u + ay[2] * v);
        }
}

}  // namespace

std::vector<std::vector<float>> burst_consensus_error(
    const std::vector<std::vector<float>>& flows, const std::vector<int>& times,
    int ref_time, int ny, int nx, int tile_size, const std::vector<float>& tile_struct,
    const Config& cfg) {
    const int nf = (int)flows.size();
    std::vector<std::vector<float>> err((size_t)std::max(0, nf));
    const size_t nt = (size_t)std::max(0, ny) * std::max(0, nx);
    if (nf < 3 || (int)times.size() != nf || nt == 0 || tile_struct.size() != nt * 4) return err;
    for (const auto& f : flows) if (f.size() < nt * 2) return err;
    const int ts = std::max(1, tile_size);
    // 1. Remove each frame's camera motion: residual r_k = F_k - G_k.
    std::vector<std::vector<float>> R((size_t)nf);
    parallel_rows(nf, cfg.num_threads, [&](int k) {
        std::vector<float> g;
        fit_affine(flows[(size_t)k].data(), tile_struct, ny, nx, ts, g);
        R[(size_t)k].resize(nt * 2);
        for (size_t i = 0; i < nt * 2; ++i) R[(size_t)k][i] = flows[(size_t)k][i] - g[i];
    });
    for (auto& e : err) e.assign(nt, 0.f);
    const float noise_k = cfg.fb_conf_noise_k;
    const double ref_w = 2.0;   // the reference's residual is exactly 0 at its time
    const double hub = 0.5;     // raw px
    // 2. Per tile, per frame: robust line through the OTHER frames (+ reference) in time.
    parallel_rows(ny, cfg.num_threads, [&](int ty) {
        std::vector<double> tt, rx, ry, ww, wb;
        for (int tx = 0; tx < nx; ++tx) {
            const size_t t = (size_t)ty * nx + tx;
            const float* W = &tile_struct[t * 4];
            if (W[0] + W[2] <= 0.f) continue;                  // flat tile
            for (int k = 0; k < nf; ++k) {
                tt.clear(); rx.clear(); ry.clear(); wb.clear();
                for (int j = 0; j < nf; ++j) {
                    if (j == k) continue;
                    const float a = R[(size_t)j][2 * t], b = R[(size_t)j][2 * t + 1];
                    if (!std::isfinite(a) || !std::isfinite(b)) continue;
                    tt.push_back((double)times[(size_t)j]); rx.push_back(a); ry.push_back(b); wb.push_back(1.0);
                }
                tt.push_back((double)ref_time); rx.push_back(0.0); ry.push_back(0.0); wb.push_back(ref_w);
                const size_t m = tt.size();
                ww = wb;
                double px = 0, py = 0, sx = 0, sy = 0;   // prediction line: p + s (tau - tau_k)
                const double tk = (double)times[(size_t)k];
                for (int it = 0; it < 4; ++it) {
                    double S = 0, St = 0, Stt = 0, Sx = 0, Sy = 0, Stx = 0, Sty = 0;
                    for (size_t i = 0; i < m; ++i) {
                        const double d = tt[i] - tk;
                        S += ww[i]; St += ww[i] * d; Stt += ww[i] * d * d;
                        Sx += ww[i] * rx[i]; Sy += ww[i] * ry[i]; Stx += ww[i] * d * rx[i]; Sty += ww[i] * d * ry[i];
                    }
                    const double den = S * Stt - St * St;
                    if (m >= 3 && den > 1e-9 * S * S) {
                        px = (Stt * Sx - St * Stx) / den; sx = (S * Stx - St * Sx) / den;
                        py = (Stt * Sy - St * Sty) / den; sy = (S * Sty - St * Sy) / den;
                    } else {                                     // too few points: constant
                        px = S > 0 ? Sx / S : 0.0; py = S > 0 ? Sy / S : 0.0; sx = sy = 0.0;
                    }
                    for (size_t i = 0; i < m; ++i) {
                        const double d = tt[i] - tk;
                        const double ex = rx[i] - (px + sx * d), ey = ry[i] - (py + sy * d);
                        const double r = std::sqrt(ex * ex + ey * ey);
                        ww[i] = wb[i] * (r <= hub ? 1.0 : hub / r);
                    }
                }
                // Disagreement of frame k with the others' prediction at its time.
                const double dx = R[(size_t)k][2 * t] - px, dy = R[(size_t)k][2 * t + 1] - py;
                if (!std::isfinite(dx) || !std::isfinite(dy)) { err[(size_t)k][t] = 1e3f; continue; }
                const double w2 = cfg.fb_conf_along_edges
                    ? dx * dx + dy * dy                              // slides along edges count too
                    : W[0] * dx * dx + 2.0 * W[1] * dx * dy + W[2] * dy * dy;
                const double e2 = std::max(0.0, w2 - (double)noise_k * W[3]);
                err[(size_t)k][t] = (float)std::sqrt(e2);
            }
        }
    });
    return err;
}

}  // namespace hhsr
