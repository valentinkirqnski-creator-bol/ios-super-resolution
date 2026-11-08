// Multi-signal alignment-confidence robustness (Config::robustness_multisignal).
//
// Replaces Wronski's R = s exp(-d^2/sigma^2) - t with four independent indicators of
// alignment quality, evaluated at guide resolution (half-res RGB, one value per Bayer quad):
//
//   R_p    photometric consistency of LOW-PASSED guides. Low-passing removes most of the
//          legitimate sub-pixel sampling (aliasing) difference between frames while keeping
//          structure, so a residual means structural disagreement. The residual is
//          normalised by its expected variance
//              V_c = V_noise,c + g_c^T Sigma_u g_c          (g_c = grad of the ref low pass)
//          i.e. sensor noise of the low-passed difference plus the photometric change the
//          alignment UNCERTAINTY alone can explain, and
//              z = mean_c e_c^2 / (tau_p^2 V_c),   R_p = exp(-z / 2).
//   R_fb   forward-backward consistency of the flow: E = || u_0->i + u_i->0(x + u_0->i) ||
//          (needs the backward alignment, pipeline).
//   R_flow local motion-field consistency: residual of a least-squares affine fit
//          u(x + d) = u(x) + J d over the 3x3 neighbouring tiles, so rotation / zoom /
//          perspective (smooth flow) costs nothing and only unexplained deviations do.
//   R_unc  alignment uncertainty from the Lucas-Kanade Hessian of the reference,
//          Sigma_u = sigma_n^2 (H + lambda I)^-1.
//   The three geometric signals are judged by the RECONSTRUCTION DAMAGE their displacement
//   error would do -- a shift delta changes the image by about grad L . delta -- measured
//   in noise variances, D = mean_c grad_c^T E grad_c / V_noise,c, and R = exp(-D / 2 tau^2).
//   Measured on burst12 the raw errors are large wherever the flow is ill-defined (median
//   forward-backward error 0.8 px, p90 21 px: sky, flat walls), where a misplaced sample
//   does no harm; a pixel threshold rejected most of the frame, the damage test only where
//   structure would be doubled.
//
//   R_denoise = R_p^a R_fb^b                      (safe to average for noise)
//   R_SR      = R_denoise R_flow^c R_unc          (safe to contribute fine detail)
//
// The merge consumes one weight per guide pixel, so the two are blended by the
// reference's local FLATNESS: a flat area holds no recoverable detail, so the denoising
// confidence is the right one there; a detailed area uses the stricter SR confidence.
//     R = R_SR + (R_denoise - R_SR) * flat,  then a local min (Config::ms_min_window).
// The merge keeps using the original RAW samples at their true positions -- only the
// decision of whether to trust them comes from the low-passed guides.
#include "stages.h"
#include "parallel.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace hhsr {

void ms_guide(const Image& raw, const Config& cfg, std::vector<float>& g, int& gh, int& gw) {
    gh = raw.h / 2;
    gw = raw.w / 2;
    const size_t plane = (size_t)gh * gw;
    g.assign(3 * plane, 0.f);
    int cnt[3] = {0, 0, 0};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) ++cnt[std::min(2, (int)cfg.cfa.p[i][j])];
    parallel_rows(gh, cfg.num_threads, [&](int y) {
        for (int x = 0; x < gw; ++x) {
            float s[3] = {0.f, 0.f, 0.f};
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    s[std::min(2, (int)cfg.cfa.p[i][j])] += raw.at(2 * y + i, 2 * x + j);
            const size_t o = (size_t)y * gw + x;
            for (int c = 0; c < 3; ++c) g[c * plane + o] = cnt[c] ? s[c] / (float)cnt[c] : 0.f;
        }
    });
}

static std::vector<float> ms_taps(float sigma) {
    if (!(sigma > 0.f)) return {1.f};
    const int r = std::max(1, (int)std::ceil(3.f * sigma));
    std::vector<float> k((size_t)r + 1);
    float sum = 0.f;
    for (int i = 0; i <= r; ++i) {
        k[(size_t)i] = std::exp(-0.5f * (float)(i * i) / (sigma * sigma));
        sum += (i == 0) ? k[0] : 2.f * k[(size_t)i];
    }
    for (float& v : k) v /= sum;
    return k;   // centre first, symmetric
}

void ms_blur(const std::vector<float>& in, int gh, int gw, const std::vector<float>& k,
             std::vector<float>& out, int nt) {
    const size_t plane = (size_t)gh * gw;
    const int r = (int)k.size() - 1;
    std::vector<float> tmp(in.size());
    out.assign(in.size(), 0.f);
    for (int c = 0; c < 3; ++c) {
        const float* src = in.data() + c * plane;
        float* t = tmp.data() + c * plane;
        float* d = out.data() + c * plane;
        parallel_rows(gh, nt, [&](int y) {
            for (int x = 0; x < gw; ++x) {
                float s = k[0] * src[(size_t)y * gw + x];
                for (int i = 1; i <= r; ++i)
                    s += k[(size_t)i] * (src[(size_t)y * gw + std::max(0, x - i)] +
                                         src[(size_t)y * gw + std::min(gw - 1, x + i)]);
                t[(size_t)y * gw + x] = s;
            }
        });
        parallel_rows(gh, nt, [&](int y) {
            for (int x = 0; x < gw; ++x) {
                float s = k[0] * t[(size_t)y * gw + x];
                for (int i = 1; i <= r; ++i)
                    s += k[(size_t)i] * (t[(size_t)std::max(0, y - i) * gw + x] +
                                         t[(size_t)std::min(gh - 1, y + i) * gw + x]);
                d[(size_t)y * gw + x] = s;
            }
        });
    }
}

void ms_init_reference(const Image& ref_raw, const Config& cfg, RefStats& st) {
    RefStats::Ms& m = st.ms;
    std::vector<float> g;
    ms_guide(ref_raw, cfg, g, m.gh, m.gw);
    const int gh = m.gh, gw = m.gw;
    const size_t plane = (size_t)gh * gw;
    m.blur = ms_taps(cfg.ms_lowpass_sigma);
    ms_blur(g, gh, gw, m.blur, m.L, cfg.num_threads);

    // Noise of one guide channel at brightness mu: (alpha_c mu + beta_c) / (sites averaged),
    // with the camera profile scaled into the white-balanced domain the loader produces
    // (alpha' = g alpha, beta' = g^2 beta). Physical noise, NOT 460's alpha/3 robustness
    // quirk: measured on burst12, this makes z of two aligned frames median 1.0 (460's
    // gives 4.0).
    int cnt[3] = {0, 0, 0};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) ++cnt[std::min(2, (int)cfg.cfa.p[i][j])];
    float ach[3], bch[3];
    for (int c = 0; c < 3; ++c) {
        float g = 1.f;
        if (cfg.raw_prewhitened) {
            const float w1 = cfg.white_balance[1], wc = cfg.white_balance[c];
            if (std::isfinite(w1) && std::isfinite(wc) && w1 > 0.f && wc > 0.f) g = wc / w1;
        }
        ach[c] = cfg.alpha_dng[c] * g;
        bch[c] = cfg.beta_dng[c] * g * g;
    }
    float e1 = 0.f;
    for (size_t i = 0; i < m.blur.size(); ++i) e1 += (i == 0 ? 1.f : 2.f) * m.blur[i] * m.blur[i];
    const float e2 = e1 * e1;   // variance gain of the separable 2-D low pass
    // Noise-only expectation of G = |grad L|^2 / V_n for the reference's own low pass: a
    // central difference of the blurred noise has variance sigma^2 e1 (e1 - rho2) / 2 per
    // component (rho2 = sum_i k_i k_(i+2), the lag-2 autocorrelation of the taps), and
    // V_n = 2 sigma^2 e1^2, so g0 = (e1 - rho2) / (2 e1).
    {
        const int r = (int)m.blur.size() - 1;
        auto tap = [&](int i) { return (std::abs(i) <= r) ? m.blur[(size_t)std::abs(i)] : 0.f; };
        float rho2 = 0.f;
        for (int i = -r - 2; i <= r; ++i) rho2 += tap(i) * tap(i + 2);
        m.g0 = (e1 - rho2) / (2.f * e1);
    }
    auto vch = [&](int c, float mu) {
        return std::max(0.f, ach[c] * std::max(mu, 0.f) + bch[c]) / (float)std::max(1, cnt[c]);
    };

    m.gx.assign(3 * plane, 0.f);
    m.gy.assign(3 * plane, 0.f);
    m.vn.assign(3 * plane, 0.f);
    for (int c = 0; c < 3; ++c) {
        const float* L = m.L.data() + c * plane;
        parallel_rows(gh, cfg.num_threads, [&](int y) {
            for (int x = 0; x < gw; ++x) {
                const size_t o = (size_t)y * gw + x;
                m.gx[c * plane + o] = 0.5f * (L[(size_t)y * gw + std::min(gw - 1, x + 1)] -
                                              L[(size_t)y * gw + std::max(0, x - 1)]);
                m.gy[c * plane + o] = 0.5f * (L[(size_t)std::min(gh - 1, y + 1) * gw + x] -
                                              L[(size_t)std::max(0, y - 1) * gw + x]);
                // difference of two independently noisy low-passed frames
                m.vn[c * plane + o] = 2.f * e2 * vch(c, L[o]);
            }
        });
    }

    // Multi-channel Lucas-Kanade information over a 5x5 window of the (unblurred) guide:
    //   H = sum_c sum_window grad G_c grad G_c^T / sigma_n,c^2,   sigma_n,c^2 = 2 v_c,
    // the temporal-residual noise of channel c. Sigma_u = (H + lambda I)^-1, lambda = 1/cap^2
    // so a textureless window saturates at the configured cap. Using every channel (not
    // the luminance alone) keeps a chroma edge from reading as unconstrained -- the damage
    // test below is per channel, so the uncertainty must be too.
    std::vector<float> ggx(3 * plane), ggy(3 * plane);
    for (int c = 0; c < 3; ++c) {
        const float* G = g.data() + c * plane;
        parallel_rows(gh, cfg.num_threads, [&](int y) {
            for (int x = 0; x < gw; ++x) {
                const size_t o = (size_t)y * gw + x;
                ggx[c * plane + o] = 0.5f * (G[(size_t)y * gw + std::min(gw - 1, x + 1)] - G[(size_t)y * gw + std::max(0, x - 1)]);
                ggy[c * plane + o] = 0.5f * (G[(size_t)std::min(gh - 1, y + 1) * gw + x] - G[(size_t)std::max(0, y - 1) * gw + x]);
            }
        });
    }
    const float cap_g = std::max(0.25f, 0.5f * cfg.ms_unc_cap);   // raw px -> guide px
    m.sig.assign(3 * plane, 0.f);
    m.flat.assign(plane, 0.f);
    const float kflat = std::max(1e-3f, cfg.ms_flat_k);
    parallel_rows(gh, cfg.num_threads, [&](int y) {
        for (int x = 0; x < gw; ++x) {
            const size_t o = (size_t)y * gw + x;
            double sxx = 0, sxy = 0, syy = 0;
            for (int c = 0; c < 3; ++c) {
                const double isn = 1.0 / std::max(2.0 * vch(c, m.L[c * plane + o]), 1e-14);
                double cxx = 0, cxy = 0, cyy = 0;
                for (int i = -2; i <= 2; ++i) {
                    const int yy = std::min(gh - 1, std::max(0, y + i));
                    for (int j = -2; j <= 2; ++j) {
                        const int xx = std::min(gw - 1, std::max(0, x + j));
                        const size_t q = c * plane + (size_t)yy * gw + xx;
                        cxx += (double)ggx[q] * ggx[q];
                        cxy += (double)ggx[q] * ggy[q];
                        cyy += (double)ggy[q] * ggy[q];
                    }
                }
                sxx += cxx * isn; sxy += cxy * isn; syy += cyy * isn;
            }
            const double lam = 1.0 / ((double)cap_g * cap_g);
            const double a = sxx + lam, bb = sxy, d = syy + lam;
            const double det = a * d - bb * bb;
            m.sig[o] = (float)(d / det);
            m.sig[plane + o] = (float)(-bb / det);
            m.sig[2 * plane + o] = (float)(a / det);
            // Detail SNR: the information over its pure-noise expectation (a central
            // difference of noise has variance v/2 per component, over sigma_n^2 = 2v:
            // 1/4 x 2 components x 25 px x 3 channels = 37.5); flat = 1 at noise level.
            const double D = (sxx + syy) / 37.5;
            m.flat[o] = (float)std::exp(-std::max(0.0, D - 1.0) / kflat);
        }
    });
}

void ms_fill_fb_error(FlowField& fwd, const FlowField& bwd, int ts) {
    fwd.fb_error.assign((size_t)fwd.ny * fwd.nx, 0.f);
    if (bwd.ny <= 0 || bwd.nx <= 0 || ts <= 0) return;
    auto bwd_at = [&](float ry, float rx, float& by, float& bx) {
        // bilinear over tile centres, clamped
        float ty = ry / ts - 0.5f, tx = rx / ts - 0.5f;
        ty = std::min((float)bwd.ny - 1, std::max(0.f, ty));
        tx = std::min((float)bwd.nx - 1, std::max(0.f, tx));
        const int y0 = (int)ty, x0 = (int)tx;
        const int y1 = std::min(bwd.ny - 1, y0 + 1), x1 = std::min(bwd.nx - 1, x0 + 1);
        const float fy = ty - y0, fx = tx - x0;
        auto lerp2 = [&](bool dy_) {
            auto v = [&](int yy, int xx) { return dy_ ? bwd.dy(yy, xx) : bwd.dx(yy, xx); };
            const float top = v(y0, x0) + (v(y0, x1) - v(y0, x0)) * fx;
            const float bot = v(y1, x0) + (v(y1, x1) - v(y1, x0)) * fx;
            return top + (bot - top) * fy;
        };
        by = lerp2(true);
        bx = lerp2(false);
    };
    for (int ty = 0; ty < fwd.ny; ++ty)
        for (int tx = 0; tx < fwd.nx; ++tx) {
            const float cy = (ty + 0.5f) * ts, cx = (tx + 0.5f) * ts;
            const float fy = fwd.dy(ty, tx), fx = fwd.dx(ty, tx);
            float by = 0.f, bx = 0.f;
            bwd_at(cy + fy, cx + fx, by, bx);
            fwd.fb_error[(size_t)ty * fwd.nx + tx] = std::sqrt((fx + bx) * (fx + bx) + (fy + by) * (fy + by));
        }
}

// Least-squares affine fit of one flow component over the in-grid 3x3 neighbourhood:
// u(d) = a0 + a1 dx + a2 dy. Returns the sum of squared residuals; falls back to a
// constant fit when the design is singular (fewer than 3 non-collinear tiles).
static double affine_residual(const double* u, const int* dx, const int* dy, int n) {
    double m00 = n, m01 = 0, m02 = 0, m11 = 0, m12 = 0, m22 = 0, r0 = 0, r1 = 0, r2 = 0;
    for (int i = 0; i < n; ++i) {
        m01 += dx[i]; m02 += dy[i]; m11 += dx[i] * dx[i]; m12 += dx[i] * dy[i]; m22 += dy[i] * dy[i];
        r0 += u[i]; r1 += u[i] * dx[i]; r2 += u[i] * dy[i];
    }
    const double det = m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02) +
                       m02 * (m01 * m12 - m11 * m02);
    double a0, a1 = 0, a2 = 0;
    if (std::fabs(det) > 1e-9) {
        a0 = (r0 * (m11 * m22 - m12 * m12) - m01 * (r1 * m22 - m12 * r2) + m02 * (r1 * m12 - m11 * r2)) / det;
        a1 = (m00 * (r1 * m22 - m12 * r2) - r0 * (m01 * m22 - m12 * m02) + m02 * (m01 * r2 - r1 * m02)) / det;
        a2 = (m00 * (m11 * r2 - r1 * m12) - m01 * (m01 * r2 - r1 * m02) + r0 * (m01 * m12 - m11 * m02)) / det;
    } else {
        a0 = r0 / n;
    }
    double ss = 0;
    for (int i = 0; i < n; ++i) {
        const double e = u[i] - (a0 + a1 * dx[i] + a2 * dy[i]);
        ss += e * e;
    }
    return ss;
}

void ms_tile_errors(const FlowField& flow, std::vector<float>& efb2, std::vector<float>& eflow) {
    const int ny = flow.ny, nx = flow.nx;
    efb2.assign((size_t)ny * nx, 0.f);
    eflow.assign((size_t)ny * nx, 0.f);
    const bool have_fb = flow.fb_error.size() == (size_t)ny * nx;
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            const size_t o = (size_t)ty * nx + tx;
            if (have_fb) efb2[o] = flow.fb_error[o] * flow.fb_error[o];
            double ux[9], uy[9];
            int dxs[9], dys[9], n = 0;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j) {
                    const int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= ny || xx < 0 || xx >= nx) continue;
                    ux[n] = flow.dx(yy, xx); uy[n] = flow.dy(yy, xx);
                    dxs[n] = j; dys[n] = i; ++n;
                }
            eflow[o] = (float)((affine_residual(ux, dxs, dys, n) + affine_residual(uy, dxs, dys, n)) / n);
        }
}

Image ms_local_min(const Image& R, int window) {
    const int r = std::max(0, window / 2);
    if (r == 0) return R;
    Image tmp(R.h, R.w, 1), out(R.h, R.w, 1);
    parallel_rows(R.h, 0, [&](int y) {
        for (int x = 0; x < R.w; ++x) {
            float m = R.at(y, x);
            for (int j = -r; j <= r; ++j) m = std::min(m, R.at(y, std::min(R.w - 1, std::max(0, x + j))));
            tmp.at(y, x) = m;
        }
    });
    parallel_rows(R.h, 0, [&](int y) {
        for (int x = 0; x < R.w; ++x) {
            float m = tmp.at(y, x);
            for (int i = -r; i <= r; ++i) m = std::min(m, tmp.at(std::min(R.h - 1, std::max(0, y + i)), x));
            out.at(y, x) = m;
        }
    });
    return out;
}

Image ms_compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                            const FlowField& flow, int ts, const Config& cfg) {
    const RefStats::Ms& m = ref_stats.ms;
    if (m.gh <= 0 || flow.ny <= 0 || flow.nx <= 0 || ts <= 0) return Image();
    std::vector<float> g, Lc;
    int gh = 0, gw = 0;
    ms_guide(comp_raw, cfg, g, gh, gw);
    if (gh != m.gh || gw != m.gw) return Image();
    ms_blur(g, gh, gw, m.blur, Lc, cfg.num_threads);
    std::vector<float>().swap(g);
    std::vector<float> efb2, eflow;
    ms_tile_errors(flow, efb2, eflow);

    const size_t plane = (size_t)gh * gw;
    const float itp2 = 1.f / std::max(1e-6f, cfg.ms_tau_p * cfg.ms_tau_p);
    const float sfl = 0.25f * cfg.ms_shift_floor * cfg.ms_shift_floor;   // guide px^2
    // damage exponents: -D / (2 tau^2) * exponent
    const float kfb = cfg.ms_exp_fb / (2.f * std::max(1e-6f, cfg.ms_tau_fb * cfg.ms_tau_fb));
    const float kfl = cfg.ms_exp_flow / (2.f * std::max(1e-6f, cfg.ms_tau_flow * cfg.ms_tau_flow));
    const float ku = 1.f / (2.f * std::max(1e-6f, cfg.ms_tau_u * cfg.ms_tau_u));
    // Pass 1: per-pixel normalised photometric energy z (mean over the 3 channels of
    // e^2 / (tau_p^2 V)); -1 marks a correspondence that leaves the frame.
    std::vector<float> Z(plane, -1.f);
    parallel_rows(gh, cfg.num_threads, [&](int y) {
        const int pty = std::min(flow.ny - 1, (int)((2.f * y + 0.5f) / ts));
        for (int x = 0; x < gw; ++x) {
            const int ptx = std::min(flow.nx - 1, (int)((2.f * x + 0.5f) / ts));
            const size_t to = (size_t)pty * flow.nx + ptx;
            const float sy = y + 0.5f * flow.dy(pty, ptx), sx = x + 0.5f * flow.dx(pty, ptx);
            const size_t o = (size_t)y * gw + x;
            if (!(sy >= 0.f && sy <= (float)(gh - 1) && sx >= 0.f && sx <= (float)(gw - 1)))
                continue;            // the correspondence leaves the frame -> R = 0
            const int y0 = std::min(gh - 2, (int)sy), x0 = std::min(gw - 2, (int)sx);
            const float fy = sy - y0, fx = sx - x0;
            const float s00 = m.sig[o], s01 = m.sig[plane + o], s11 = m.sig[2 * plane + o];
            // Interpolation weights of the bilinear warp: its error on a smooth signal is
            // about 1/2 f(1-f) L'' per axis.
            const float wxi = 0.5f * fx * (1.f - fx), wyi = 0.5f * fy * (1.f - fy);
            const int xm = std::max(0, x - 1), xp = std::min(gw - 1, x + 1);
            const int ym = std::max(0, y - 1), yp = std::min(gh - 1, y + 1);
            float z = 0.f;
            for (int c = 0; c < 3; ++c) {
                const float* L = Lc.data() + c * plane;
                const float* Lr = m.L.data() + c * plane;
                const size_t q = (size_t)y0 * gw + x0;
                const float top = L[q] + (L[q + 1] - L[q]) * fx;
                const float bot = L[q + gw] + (L[q + gw + 1] - L[q + gw]) * fx;
                const float e = Lr[o] - (top + (bot - top) * fy);
                const float gxv = m.gx[c * plane + o], gyv = m.gy[c * plane + o];
                const float dxx = Lr[(size_t)y * gw + xp] + Lr[(size_t)y * gw + xm] - 2.f * Lr[o];
                const float dyy = Lr[(size_t)yp * gw + x] + Lr[(size_t)ym * gw + x] - 2.f * Lr[o];
                const float ei = wxi * dxx + wyi * dyy;
                const float V = m.vn[c * plane + o] +
                                gxv * gxv * (s00 + sfl) + 2.f * gxv * gyv * s01 + gyv * gyv * (s11 + sfl) +
                                ei * ei + 1e-10f;
                z += e * e / V;
            }
            Z[o] = z * itp2 / 3.f;
            (void)to;
        }
    });
    // Pass 2: aggregate the photometric evidence over 3x3 (a per-pixel z is a noisy
    // chi-square statistic; its 3x3 mean is a 27-sample estimate, so the final local
    // minimum responds to misalignment rather than to the noise tail), then combine.
    Image R(gh, gw, 1);
    parallel_rows(gh, cfg.num_threads, [&](int y) {
        const int pty = std::min(flow.ny - 1, (int)((2.f * y + 0.5f) / ts));
        for (int x = 0; x < gw; ++x) {
            const size_t o = (size_t)y * gw + x;
            if (Z[o] < 0.f) { R.at(y, x) = 0.f; continue; }
            const int ptx = std::min(flow.nx - 1, (int)((2.f * x + 0.5f) / ts));
            const size_t to = (size_t)pty * flow.nx + ptx;
            float zs = 0.f;
            int n = 0;
            for (int i = -1; i <= 1; ++i) {
                const int yy = y + i;
                if (yy < 0 || yy >= gh) continue;
                for (int j = -1; j <= 1; ++j) {
                    const int xx = x + j;
                    if (xx < 0 || xx >= gw) continue;
                    const float v = Z[(size_t)yy * gw + xx];
                    if (v >= 0.f) { zs += v; ++n; }
                }
            }
            const float z = zs / (float)n;
            const float s00 = m.sig[o], s01 = m.sig[plane + o], s11 = m.sig[2 * plane + o];
            // Reconstruction damage of each geometric error, in noise variances:
            // G = mean_c |grad L_c|^2 / V_n,c (per guide px^2); an isotropic displacement
            // error of variance e (guide px^2) does G * e; Sigma_u does grad^T Sigma grad.
            float G = 0.f, Du = 0.f;
            for (int c = 0; c < 3; ++c) {
                const float gxv = m.gx[c * plane + o], gyv = m.gy[c * plane + o];
                const float iv = 1.f / (m.vn[c * plane + o] + 1e-10f);
                G += (gxv * gxv + gyv * gyv) * iv;
                Du += (gxv * gxv * s00 + 2.f * gxv * gyv * s01 + gyv * gyv * s11) * iv;
            }
            G /= 3.f;
            Du /= 3.f;
            // Only SIGNAL gradient turns a displacement error into damage: remove the
            // reference's own noise contribution (3x its expectation), so a flat sky with a
            // blundered flow is still averaged for denoising.
            const float Gs = std::max(0.f, G - 3.f * m.g0);
            Du *= (G > 0.f) ? Gs / G : 0.f;
            G = Gs;
            // Per-tile errors interpolated bilinearly between tile centres (a nearest
            // fetch prints the 16 px tile grid into the mask).
            float tyf = (2.f * y + 0.5f) / ts - 0.5f, txf = (2.f * x + 0.5f) / ts - 0.5f;
            tyf = std::min((float)flow.ny - 1.f, std::max(0.f, tyf));
            txf = std::min((float)flow.nx - 1.f, std::max(0.f, txf));
            const int ty0 = (int)tyf, tx0 = (int)txf;
            const int ty1 = std::min(flow.ny - 1, ty0 + 1), tx1 = std::min(flow.nx - 1, tx0 + 1);
            const float wy = tyf - ty0, wx = txf - tx0;
            auto bil = [&](const std::vector<float>& v) {
                const float a = v[(size_t)ty0 * flow.nx + tx0], b = v[(size_t)ty0 * flow.nx + tx1];
                const float c = v[(size_t)ty1 * flow.nx + tx0], d = v[(size_t)ty1 * flow.nx + tx1];
                const float top = a + (b - a) * wx, bot = c + (d - c) * wx;
                return top + (bot - top) * wy;
            };
            (void)to;
            const float Dfb = G * 0.25f * bil(efb2);     // raw px^2 -> guide px^2
            const float Dfl = G * 0.25f * bil(eflow);
            const float Rp = std::exp(-0.5f * z * cfg.ms_exp_p);   // R_p^a
            const float Rfb = std::exp(-Dfb * kfb);                 // R_fb^b
            const float Rfl = std::exp(-Dfl * kfl);                 // R_flow^c
            const float Run = std::exp(-Du * ku);
            const float Rdn = Rp * Rfb;
            const float Rsr = Rdn * Rfl * Run;
            const float f = m.flat[o];
            R.at(y, x) = std::min(1.f, std::max(0.f, Rsr + (Rdn - Rsr) * f));
        }
    });
    return ms_local_min(R, cfg.ms_min_window);
}

}  // namespace hhsr
