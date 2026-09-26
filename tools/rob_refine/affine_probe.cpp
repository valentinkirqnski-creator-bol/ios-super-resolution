// Is a least-squares local affine fit a better within-tile error predictor than
// the central difference the refinement currently uses?
//
// The refinement's strongest feature by a wide margin is E = -G.(u,v): the flow
// field's gradient times the pixel's offset from its tile centre. G is a CENTRAL
// DIFFERENCE of the immediately neighbouring tile vectors -- a three-tap
// estimate of a field whose every sample is itself a noisy block-match result.
// Measured, |E| reaches only 0.285 rank correlation with the true damage, and
// that number is the ceiling on everything downstream of it.
//
// Under rotation, though, the true flow is locally AFFINE, exactly:
//
//     flow(p) = A (p - c) + t
//
// so fitting A and t by least squares over a WINDOW of tile vectors uses 25 or
// 49 samples instead of 4, and should cut the estimator's noise by the usual
// square root. Two things fall out:
//
//   E_affine  -- the within-tile error from the fitted model, which should
//                predict the real displacement error better than E does;
//   residual  -- how badly the affine model itself fits the local tile vectors,
//                which is a direct measure of NON-affine motion. Rotation is
//                affine and parallax is not, so this separates them, which no
//                current feature does.
//
// This is a measurement, not a feature yet: it prints the correlation of each
// candidate against the true per-pixel displacement error computed from the
// known synthetic warp, so the decision to build it is made on evidence.
//
// Usage: affine_probe ref.dng [rot_deg] [win_radius]
#include "stages.h"
#include "parallel.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
using namespace hhsr;

namespace {

float sample_phase(const Image& raw, float sy, float sx, int oy, int ox) {
    const float fi = (sy - (float)oy) * 0.5f, fj = (sx - (float)ox) * 0.5f;
    const int i0 = (int)std::floor(fi), j0 = (int)std::floor(fj);
    const float ai = fi - (float)i0, aj = fj - (float)j0;
    const int imax = (raw.h - 1 - oy) / 2, jmax = (raw.w - 1 - ox) / 2;
    auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
    const int i1 = cl(i0 + 1, imax), j1 = cl(j0 + 1, jmax);
    const int ic = cl(i0, imax), jc = cl(j0, jmax);
    auto at = [&](int i, int j) { return raw.at(2 * i + oy, 2 * j + ox); };
    const float top = at(ic, jc) + (at(ic, j1) - at(ic, jc)) * aj;
    const float bot = at(i1, jc) + (at(i1, j1) - at(i1, jc)) * aj;
    return top + (bot - top) * ai;
}

// Spearman, via rank transform.
double rank_corr(std::vector<float> a, std::vector<float> b) {
    const size_t n = a.size();
    if (n < 100) return 0.0;
    std::vector<size_t> ia(n), ib(n);
    for (size_t i = 0; i < n; ++i) ia[i] = ib[i] = i;
    std::sort(ia.begin(), ia.end(), [&](size_t i, size_t j) { return a[i] < a[j]; });
    std::sort(ib.begin(), ib.end(), [&](size_t i, size_t j) { return b[i] < b[j]; });
    std::vector<double> ra(n), rb(n);
    for (size_t k = 0; k < n; ++k) { ra[ia[k]] = (double)k; rb[ib[k]] = (double)k; }
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += ra[i]; mb += rb[i]; }
    ma /= (double)n; mb /= (double)n;
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = ra[i] - ma, y = rb[i] - mb;
        num += x * y; da += x * x; db += y * y;
    }
    return (da > 0 && db > 0) ? num / std::sqrt(da * db) : 0.0;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::printf("usage: affine_probe ref.dng [rot_deg] [win_radius]\n");
        return 1;
    }
    const float rot_deg = (argc > 2) ? (float)std::atof(argv[2]) : 0.5f;
    const int WR = (argc > 3) ? std::atoi(argv[3]) : 2;   // tile-window radius

    Config cfg;
    cfg.scale = 2.f; cfg.bayer_mode = true;
    cfg.grey_method = GreyMethod::Decimate;
    cfg.r_t = 0.12f; cfg.r_s1 = 2.f; cfg.r_s2 = 12.f; cfg.r_Mt = 0.8f;
    cfg.num_threads = 0; cfg.alignment_tile_size = 16;

    Image ref = load_raw_frame(argv[1], cfg, true, 0, 0);
    if (ref.w <= 0) { std::printf("decode failed\n"); return 1; }
    Config work = cfg;
    work.burst_frame_count = 2;
    tune_config_snr(ref, work);
    work.r_t = cfg.r_t; work.r_s1 = cfg.r_s1; work.r_s2 = cfg.r_s2;
    const int ts = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    clear_align_ref_ica_cache();

    // Pure rotation about the frame centre: the true flow is exactly affine, so
    // the affine fit has a right answer to find and the comparison is clean.
    const float th = rot_deg * 3.14159265f / 180.f;
    const float ct = std::cos(th), st = std::sin(th);
    const float cy = 0.5f * (float)ref.h, cx = 0.5f * (float)ref.w;
    Image comp(ref.h, ref.w, 1);
    const float alpha = work.noise_alpha(), beta = work.noise_beta();
    parallel_rows(ref.h, work.num_threads, [&](int y) {
        std::mt19937 rng(11u + (uint32_t)y);
        std::normal_distribution<float> g(0.f, 1.f);
        for (int x = 0; x < ref.w; ++x) {
            const float dy = (float)y - cy, dx = (float)x - cx;
            const float sy = ct * dy + st * dx + cy;
            const float sx = -st * dy + ct * dx + cx;
            float v = sample_phase(ref, sy, sx, y & 1, x & 1);
            v += g(rng) * std::sqrt(std::max(alpha * std::max(v, 0.f) + beta, 0.f));
            comp.at(y, x) = std::max(v, 0.f);
        }
    });

    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
    Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
    FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, ts, 0.f, 0.f, 0.f);
    flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w,
                                 ts, work.r_Mt, work.num_threads,
                                 work.grey_tile_size(ts));
    std::printf("%dx%d raw, tiles %dx%d, ts %d, rotation %.3f deg, "
                "affine window %dx%d tiles\n",
                ref.w, ref.h, flow.nx, flow.ny, ts, rot_deg, 2 * WR + 1, 2 * WR + 1);

    // ---- per tile: central-difference gradient, and a least-squares affine ---
    const int NY = flow.ny, NX = flow.nx;
    std::vector<float> a11(NY * NX), a12(NY * NX), a21(NY * NX), a22(NY * NX);
    std::vector<float> t1(NY * NX), t2(NY * NX), fitres(NY * NX);
    parallel_rows(NY, work.num_threads, [&](int ty) {
        for (int tx = 0; tx < NX; ++tx) {
            // Fit dx = a11*u + a12*v + t1, dy = a21*u + a22*v + t2 over the
            // window, with (u,v) the tile offset from this tile in raw px.
            double Sww = 0, Suu = 0, Svv = 0, Suv = 0, Su = 0, Sv = 0;
            double Sx = 0, Sxu = 0, Sxv = 0, Sy = 0, Syu = 0, Syv = 0;
            for (int i = -WR; i <= WR; ++i)
                for (int j = -WR; j <= WR; ++j) {
                    const int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= NY || xx < 0 || xx >= NX) continue;
                    const double u = (double)j * ts, v = (double)i * ts;
                    const double fx = flow.dx(yy, xx), fy = flow.dy(yy, xx);
                    Sww += 1; Suu += u * u; Svv += v * v; Suv += u * v;
                    Su += u; Sv += v;
                    Sx += fx; Sxu += fx * u; Sxv += fx * v;
                    Sy += fy; Syu += fy * u; Syv += fy * v;
                }
            // Normal equations for [a, b, c] against [u, v, 1], solved twice.
            const double M[3][3] = {{Suu, Suv, Su}, {Suv, Svv, Sv}, {Su, Sv, Sww}};
            const double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                             - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                             + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
            const size_t k = (size_t)ty * NX + tx;
            if (std::fabs(det) < 1e-12) {
                a11[k] = a12[k] = a21[k] = a22[k] = 0.f;
                t1[k] = (float)flow.dx(ty, tx); t2[k] = (float)flow.dy(ty, tx);
                fitres[k] = 0.f;
                continue;
            }
            auto solve = [&](double r0, double r1, double r2, double* o) {
                const double inv[3][3] = {
                    {(M[1][1]*M[2][2]-M[1][2]*M[2][1])/det,
                     (M[0][2]*M[2][1]-M[0][1]*M[2][2])/det,
                     (M[0][1]*M[1][2]-M[0][2]*M[1][1])/det},
                    {(M[1][2]*M[2][0]-M[1][0]*M[2][2])/det,
                     (M[0][0]*M[2][2]-M[0][2]*M[2][0])/det,
                     (M[0][2]*M[1][0]-M[0][0]*M[1][2])/det},
                    {(M[1][0]*M[2][1]-M[1][1]*M[2][0])/det,
                     (M[0][1]*M[2][0]-M[0][0]*M[2][1])/det,
                     (M[0][0]*M[1][1]-M[0][1]*M[1][0])/det}};
                o[0] = inv[0][0]*r0 + inv[0][1]*r1 + inv[0][2]*r2;
                o[1] = inv[1][0]*r0 + inv[1][1]*r1 + inv[1][2]*r2;
                o[2] = inv[2][0]*r0 + inv[2][1]*r1 + inv[2][2]*r2;
            };
            double px[3], py[3];
            solve(Sxu, Sxv, Sx, px);
            solve(Syu, Syv, Sy, py);
            a11[k] = (float)px[0]; a12[k] = (float)px[1]; t1[k] = (float)px[2];
            a21[k] = (float)py[0]; a22[k] = (float)py[1]; t2[k] = (float)py[2];
            // How badly the affine model fits: rotation is affine, parallax is
            // not, so this is a direct non-affine-motion measure.
            double ss = 0; int n = 0;
            for (int i = -WR; i <= WR; ++i)
                for (int j = -WR; j <= WR; ++j) {
                    const int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= NY || xx < 0 || xx >= NX) continue;
                    const double u = (double)j * ts, v = (double)i * ts;
                    const double ex = px[0]*u + px[1]*v + px[2] - flow.dx(yy, xx);
                    const double ey = py[0]*u + py[1]*v + py[2] - flow.dy(yy, xx);
                    ss += ex*ex + ey*ey; ++n;
                }
            fitres[k] = (float)std::sqrt(ss / std::max(n, 1));
        }
    });

    // ---- per guide pixel: the candidates, and the truth -------------------
    const int gh = ref.h / 2, gw = ref.w / 2;
    std::vector<float> v_E, v_Eaff, v_res, v_true, v_Mspan;
    v_E.reserve(200000);
    std::mt19937 pick(3);
    const int STRIDE = 7;
    for (int gy = 2; gy < gh - 2; gy += STRIDE) {
        for (int gx = 2; gx < gw - 2; gx += STRIDE) {
            const float rawy = 2.f * gy, rawx = 2.f * gx;
            auto clt = [](int a, int hi) { return a < 0 ? 0 : (a >= hi ? hi - 1 : a); };
            const int pty = clt((int)((rawy + 0.5f) / ts), NY);
            const int ptx = clt((int)((rawx + 0.5f) / ts), NX);
            const size_t k = (size_t)pty * NX + ptx;
            const float u = rawx - ((float)ptx + 0.5f) * ts;
            const float v = rawy - ((float)pty + 0.5f) * ts;

            // current: central difference of immediate neighbours
            const int ptu = clt(pty - 1, NY), ptd = clt(pty + 1, NY);
            const int pxl = clt(ptx - 1, NX), pxr = clt(ptx + 1, NX);
            const float i2 = 1.f / (2.f * ts);
            const float gdxdx = (flow.dx(pty, pxr) - flow.dx(pty, pxl)) * i2;
            const float gdydx = (flow.dy(pty, pxr) - flow.dy(pty, pxl)) * i2;
            const float gdxdy = (flow.dx(ptd, ptx) - flow.dx(ptu, ptx)) * i2;
            const float gdydy = (flow.dy(ptd, ptx) - flow.dy(ptu, ptx)) * i2;
            const float Ex = -(gdxdx * u + gdxdy * v);
            const float Ey = -(gdydx * u + gdydy * v);

            // affine fit: the model's own prediction at this offset, minus what
            // the merge will actually use (the tile's constant vector)
            const float pfx = a11[k] * u + a12[k] * v + t1[k];
            const float pfy = a21[k] * u + a22[k] * v + t2[k];
            const float Eax = flow.dx(pty, ptx) - pfx;
            const float Eay = flow.dy(pty, ptx) - pfy;

            // truth: the tile vector against the real displacement here
            const float dy0 = rawy - cy, dx0 = rawx - cx;
            const float tyv = ct * dy0 - st * dx0 + cy;
            const float txv = st * dy0 + ct * dx0 + cx;
            const float ftx = txv - rawx, fty = tyv - rawy;
            const float tex = flow.dx(pty, ptx) - ftx;
            const float tey = flow.dy(pty, ptx) - fty;

            float mnx = 1e30f, mny = 1e30f, mxx = -1e30f, mxy = -1e30f;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j) {
                    const int yy = pty + i, xx = ptx + j;
                    if (yy < 0 || yy >= NY || xx < 0 || xx >= NX) continue;
                    mnx = std::min(mnx, flow.dx(yy, xx)); mxx = std::max(mxx, flow.dx(yy, xx));
                    mny = std::min(mny, flow.dy(yy, xx)); mxy = std::max(mxy, flow.dy(yy, xx));
                }
            v_E.push_back(std::sqrt(Ex * Ex + Ey * Ey));
            v_Eaff.push_back(std::sqrt(Eax * Eax + Eay * Eay));
            v_res.push_back(fitres[k]);
            v_true.push_back(std::sqrt(tex * tex + tey * tey));
            v_Mspan.push_back(std::sqrt((mxx - mnx) * (mxx - mnx) + (mxy - mny) * (mxy - mny)));
        }
    }
    std::printf("%zu sampled guide pixels\n\n", v_true.size());

    double mt = 0;
    for (float t : v_true) mt += t;
    mt /= (double)v_true.size();
    std::printf("mean TRUE within-tile displacement error: %.4f raw px\n\n", mt);
    std::printf("  %-34s %10s %12s\n", "predictor of the true error",
                "rank corr", "mean value");
    auto line = [&](const char* n, std::vector<float>& v) {
        double m = 0;
        for (float x : v) m += x;
        std::printf("  %-34s %10.3f %12.4f\n", n, rank_corr(v, v_true),
                    m / (double)v.size());
    };
    line("|E|  central difference (current)", v_E);
    line("|E|  least-squares affine (new)", v_Eaff);
    line("affine fit residual (new)", v_res);
    line("Mspan (current, per tile)", v_Mspan);
    return 0;
}
