// Forward-backward flow confidence (see fb_confidence.h).
#include "fb_confidence.h"
#include "parallel.h"
#include "stages.h"
#if defined(__APPLE__)
#include "metal_gpu.h"
#endif
#include <algorithm>
#include <cmath>

namespace hhsr {

namespace {

inline float px_clamped(const Image& g, int y, int x) {
    return g.at(std::min(g.h - 1, std::max(0, y)), std::min(g.w - 1, std::max(0, x)));
}

// Bilinear sample (floor split, edge-clamped).
inline float bilin(const Image& g, float y, float x) {
    const float fy = std::floor(y), fx = std::floor(x);
    const int y0 = (int)fy, x0 = (int)fx;
    const float ay = y - fy, ax = x - fx;
    const float a = px_clamped(g, y0, x0), b = px_clamped(g, y0, x0 + 1);
    const float c = px_clamped(g, y0 + 1, x0), d = px_clamped(g, y0 + 1, x0 + 1);
    const float t = a + (b - a) * ax, u = c + (d - c) * ax;
    return t + (u - t) * ay;
}

// Bilinear sample of a tile-grid flow at tile coordinate (ty, tx), edge-clamped.
inline void sample_tile_flow(const FlowField& f, float ty, float tx, float& dx, float& dy) {
    ty = std::min((float)(f.ny - 1), std::max(0.f, ty));
    tx = std::min((float)(f.nx - 1), std::max(0.f, tx));
    const int y0 = std::min(f.ny - 1, (int)ty), x0 = std::min(f.nx - 1, (int)tx);
    const int y1 = std::min(f.ny - 1, y0 + 1), x1 = std::min(f.nx - 1, x0 + 1);
    const float ay = ty - (float)y0, ax = tx - (float)x0;
    auto lerp2 = [&](int c) {
        const float a = f.flow[((size_t)y0 * f.nx + x0) * 2 + c], b = f.flow[((size_t)y0 * f.nx + x1) * 2 + c];
        const float d = f.flow[((size_t)y1 * f.nx + x0) * 2 + c], e = f.flow[((size_t)y1 * f.nx + x1) * 2 + c];
        const float t = a + (b - a) * ax, u = d + (e - d) * ax;
        return t + (u - t) * ay;
    };
    dx = lerp2(0); dy = lerp2(1);
}

// Forward destination (raw px) of raw point (y, x) under the merge flow vector (fx, fy):
// comp_pos = H * (p + flow) when a global homography warped the frame.
inline void forward_dest(const FlowField& flow, float y, float x, float fx, float fy, float& oy, float& ox) {
    const float px = x + fx, py = y + fy;
    if (!flow.has_global_h) { oy = py; ox = px; return; }
    const float* H = flow.global_h;
    const float w = H[6] * px + H[7] * py + H[8];
    const float iw = std::fabs(w) > 1e-12f ? 1.f / w : 1.f;
    ox = (H[0] * px + H[1] * py + H[2]) * iw;
    oy = (H[3] * px + H[4] * py + H[5]) * iw;
}

}  // namespace

std::vector<float> fb_tile_error(const Image& ref_grey, const Image& comp_grey,
                                 const FlowField& flow, int raw_h, int raw_w,
                                 int tile_size, const Config& cfg) {
    const int ny = flow.ny, nx = flow.nx, ts = std::max(1, tile_size);
    if (ny <= 0 || nx <= 0 || flow.flow.size() < (size_t)ny * nx * 2 || raw_w <= 0 || raw_h <= 0) return {};
    if (ref_grey.data.empty() || comp_grey.data.empty() || ref_grey.h != comp_grey.h || ref_grey.w != comp_grey.w)
        return {};
    // Grey <-> raw: grey g sits at raw (g + 0.5) / s - 0.5 (s = 1 FFT grey, 0.5 quad grey).
    const float s = (float)ref_grey.w / (float)raw_w;
    auto to_g = [&](float r) { return (r + 0.5f) * s - 0.5f; };
    // Checked in sub-patches of the tile (fb_conf_subdiv^2 per tile), each with the TILE's
    // vector: a whole-tile check agrees with itself wherever the forward fit converged;
    // a sub-patch exposes the parts of the tile the single vector does not fit.
    const int sub = std::max(1, std::min(8, cfg.fb_conf_subdiv));
    const float ps = (float)ts / (float)sub;
    const int pny = ny * sub, pnx = nx * sub;
    const int hw = std::max(2, (int)std::lround(0.5f * ps * s));   // half template (grey px)
    const int n_iter = std::max(1, cfg.fb_conf_iters);
    const float na = std::max(0.f, cfg.alpha_dng[1]), nb = std::max(1e-12f, cfg.beta_dng[1]);

    // Full mode: an independent backward alignment (comparison -> reference).
    FlowField bwd;
    int bgts = 0;
    if (cfg.fb_conf_full && !flow.has_global_h) {
        Config bc = cfg;
        bc.flow_fb_consistency_enabled = false;   // no nested FB inside the backward pass
        const int ats = cfg.bm_tile_sizes.empty() ? 16 : cfg.bm_tile_sizes[0];
        bgts = cfg.grey_tile_size(ats);
        // Seed with the negated median forward flow (grey px).
        std::vector<float> vx, vy;
        vx.reserve((size_t)ny * nx); vy.reserve((size_t)ny * nx);
        for (int t = 0; t < ny * nx; ++t) { vx.push_back(flow.flow[2 * t]); vy.push_back(flow.flow[2 * t + 1]); }
        std::nth_element(vx.begin(), vx.begin() + vx.size() / 2, vx.end());
        std::nth_element(vy.begin(), vy.begin() + vy.size() / 2, vy.end());
        Pyramid comp_pyr;
        {
            Image comp_padded = pad_image_circular(comp_grey, bgts);
            comp_pyr = build_pyramid(comp_padded, bc.bm_factors);
        }
#if defined(__APPLE__)
        metal_invalidate_sticky_grey();
#endif
        bwd = align(comp_pyr, comp_grey, ref_grey, bc, ats, -vx[vx.size() / 2] * s, -vy[vy.size() / 2] * s, 0.f);
#if defined(__APPLE__)
        metal_invalidate_sticky_grey();
        metal_clear_ref_ica_cache();   // the backward pass keyed GPU caches on temporaries
#endif
        clear_align_ref_ica_cache();
        if (bwd.ny <= 0 || bwd.nx <= 0 || bwd.flow.empty()) bgts = 0;
    }

    std::vector<float> err((size_t)pny * pnx, 0.f);
    parallel_rows(pny, cfg.num_threads, [&](int pyi) {
        const int n = 2 * hw;
        std::vector<float> T((size_t)n * n), GX((size_t)n * n), GY((size_t)n * n);
        for (int pxi = 0; pxi < pnx; ++pxi) {
            const size_t t = (size_t)(pyi / sub) * nx + (size_t)(pxi / sub);   // the patch's tile
            const size_t pi = (size_t)pyi * pnx + pxi;
            const float fx = flow.flow[2 * t], fy = flow.flow[2 * t + 1];
            if (!std::isfinite(fx) || !std::isfinite(fy)) { err[pi] = 1e3f; continue; }
            // Patch centre (raw pixel-centre coordinates) and its forward destination.
            const float cy = ((float)pyi + 0.5f) * ps - 0.5f, cx = ((float)pxi + 0.5f) * ps - 0.5f;
            float dy, dx;
            forward_dest(flow, cy, cx, fx, fy, dy, dx);
            const float cgy = to_g(cy), cgx = to_g(cx), dgy = to_g(dy), dgx = to_g(dx);
            // Template: the comparison patch around the destination (integer window).
            const int y0 = (int)std::floor(dgy - (float)hw + 0.5f), x0 = (int)std::floor(dgx - (float)hw + 0.5f);
            if (y0 < 1 || x0 < 1 || y0 + n + 1 > comp_grey.h || x0 + n + 1 > comp_grey.w) continue;   // off frame
            double hxx = 0, hxy = 0, hyy = 0, msum = 0;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const int y = y0 + i, x = x0 + j;
                    const float gx = 0.5f * (comp_grey.at(y, x + 1) - comp_grey.at(y, x - 1));
                    const float gy = 0.5f * (comp_grey.at(y + 1, x) - comp_grey.at(y - 1, x));
                    const size_t q = (size_t)i * n + j;
                    T[q] = comp_grey.at(y, x); GX[q] = gx; GY[q] = gy;
                    hxx += gx * gx; hxy += gx * gy; hyy += gy * gy; msum += T[q];
                }
            const double inv_n = 1.0 / ((double)n * n);
            const double tr = hxx + hyy, dsc = std::sqrt(std::max(0.0, 0.25 * (hxx - hyy) * (hxx - hyy) + hxy * hxy));
            const double l1 = 0.5 * tr + dsc;                     // largest eigenvalue (sum over the patch)
            const double nv = na * std::max(0.0, msum * inv_n) + nb;   // per-pixel noise variance (grey ~ raw units)
            // Flat patch: gradient energy below ~noise level -> no evidence, misalignment invisible.
            // (Central-difference gradient noise variance is nv / 2 per axis.)
            if (l1 * inv_n < 2.0 * nv) continue;
            // Backward displacement B (grey px): ref position of template pixel y is y + B.
            float By, Bx;
            if (bgts > 0) {
                sample_tile_flow(bwd, (dgy + 0.5f) / (float)bgts - 0.5f, (dgx + 0.5f) / (float)bgts - 0.5f, Bx, By);
            } else {
                // Inverse-compositional LK, template = comparison patch, starting from the
                // exact inverse of the forward correspondence at the template centre.
                By = cgy - dgy; Bx = cgx - dgx;
                const double mu = 1e-3 * tr + 1e-12;              // light damping (aperture tiles)
                const double a00 = hxx + mu, a01 = hxy, a11 = hyy + mu;
                const double det = a00 * a11 - a01 * a01;
                if (!(det > 0)) continue;
                const double i00 = a11 / det, i01 = -a01 / det, i11 = a00 / det;
                bool diverged = false;
                for (int it = 0; it < n_iter; ++it) {
                    double bx = 0, by = 0;
                    for (int i = 0; i < n; ++i)
                        for (int j = 0; j < n; ++j) {
                            const size_t q = (size_t)i * n + j;
                            const float r = bilin(ref_grey, (float)(y0 + i) + By, (float)(x0 + j) + Bx) - T[q];
                            bx += GX[q] * r; by += GY[q] * r;
                        }
                    const float ddx = (float)(i00 * bx + i01 * by), ddy = (float)(i01 * bx + i11 * by);
                    Bx -= ddx; By -= ddy;
                    if (std::hypot(Bx - (cgx - dgx), By - (cgy - dgy)) > 8.f * s + 4.f) { diverged = true; break; }
                    if (ddx * ddx + ddy * ddy < 1e-6f) break;
                }
                if (diverged) { err[pi] = 1e3f; continue; }   // no consistent backward match: reject
            }
            // Round-trip error (grey px -> raw px): back-projected position minus the start.
            const float ey = (dgy + By - cgy) / s, ex = (dgx + Bx - cgx) / s;
            // Structure weighting W = Hc / lambda1: only constrained directions count.
            // fb_conf_along_edges: the FULL round-trip distance (slides along an edge count
            // too). Along a straight uniform edge that component is unmeasurable (LK drift),
            // so this also rejects correctly aligned straight edges -- opt-in.
            const double w2 = cfg.fb_conf_along_edges
                ? (double)ex * ex + (double)ey * ey
                : (hxx * ex * ex + 2.0 * hxy * ex * ey + hyy * ey * ey) / l1;
            // Noise floor: E[e^T W e] from noise ~ 4 sigma_r^2 / lambda1 with sigma_r^2 = 2 nv
            // (both frames), in grey px^2 -> raw px^2.
            const double noise_e2 = 8.0 * nv / l1 / ((double)s * s);
            const double e2 = std::max(0.0, w2 - (double)cfg.fb_conf_noise_k * noise_e2);
            err[pi] = (float)std::sqrt(e2);
        }
    });
    return err;
}

double fb_apply_confidence_raw(float* data, int h, int w, int c, uint8_t* data8, int h8, int w8,
                               const std::vector<float>& err, int tiles_ny, int tiles_nx,
                               int raw_h, int raw_w, int tile_size, int subdiv, float tau_px,
                               const Config& cfg) {
    const int sub = std::max(1, std::min(8, subdiv));
    const int ny = tiles_ny * sub, nx = tiles_nx * sub;   // patch grid
    const float ts = (float)std::max(1, tile_size) / (float)sub;   // patch size (raw px)
    if (err.size() != (size_t)ny * nx || ny <= 0 || nx <= 0) return 1.0;
    const float tau = std::max(1e-3f, tau_px);
    std::vector<float> C(err.size());
    double csum = 0;
    for (size_t i = 0; i < err.size(); ++i) {
        const float e = err[i];
        C[i] = std::exp(-(e * e) / (2.f * tau * tau));
        csum += C[i];
    }
    const bool bil = cfg.gcam_flow_bilinear_active();
    // Confidence at raw coordinate (ry, rx): the merge's nearest int(lr / tile) tile,
    // or bilinear between tile centres when the merge fetches a bilinear flow.
    auto conf_at = [&](float ry, float rx) {
        if (!bil) {
            // Mask pixel centre in raw coords (raw y; guide 2g+0.5; merge-res
            // (t+0.5)/s-0.5) -> its patch: the same tile the merge now fetches the flow
            // of for this mask pixel (merge.cpp seam_tile).
            const int ty = std::min(ny - 1, std::max(0, (int)std::floor(ry / ts)));
            const int tx = std::min(nx - 1, std::max(0, (int)std::floor(rx / ts)));
            return C[(size_t)ty * nx + tx];
        }
        float u = (ry + 0.5f) / (float)ts - 0.5f, v = (rx + 0.5f) / (float)ts - 0.5f;
        u = std::min((float)(ny - 1), std::max(0.f, u)); v = std::min((float)(nx - 1), std::max(0.f, v));
        const int y0 = std::min(ny - 1, (int)u), x0 = std::min(nx - 1, (int)v);
        const int y1 = std::min(ny - 1, y0 + 1), x1 = std::min(nx - 1, x0 + 1);
        const float ay = u - (float)y0, ax = v - (float)x0;
        const float a = C[(size_t)y0 * nx + x0], b = C[(size_t)y0 * nx + x1];
        const float cc = C[(size_t)y1 * nx + x0], d = C[(size_t)y1 * nx + x1];
        const float t = a + (b - a) * ax, ww = cc + (d - cc) * ax;
        return t + (ww - t) * ay;
    };
    if (data && h > 0 && w > 0 && c > 0) {
        const float ky = (float)raw_h / (float)h, kx = (float)raw_w / (float)w;
        parallel_rows(h, cfg.num_threads, [&](int y) {
            float* row = data + (size_t)y * w * c;
            for (int x = 0; x < w; ++x) {
                const float cf = conf_at(((float)y + 0.5f) * ky - 0.5f, ((float)x + 0.5f) * kx - 0.5f);
                if (cf >= 1.f) continue;
                for (int ch = 0; ch < c; ++ch) row[(size_t)x * c + ch] *= cf;
            }
        });
    }
    if (data8 && h8 > 0 && w8 > 0) {
        const float ky = (float)raw_h / (float)h8, kx = (float)raw_w / (float)w8;
        parallel_rows(h8, cfg.num_threads, [&](int y) {
            for (int x = 0; x < w8; ++x) {
                const float cf = conf_at(((float)y + 0.5f) * ky - 0.5f, ((float)x + 0.5f) * kx - 0.5f);
                if (cf >= 1.f) continue;
                uint8_t& v = data8[(size_t)y * w8 + x];
                v = (uint8_t)std::lround((float)v * cf);
            }
        });
    }
    return csum / (double)std::max<size_t>(1, C.size());
}

double fb_apply_confidence(Image* R, Mask8* m8, const std::vector<float>& err,
                           const FlowField& flow, int raw_h, int raw_w,
                           int tile_size, const Config& cfg) {
    const bool rok = R && R->h > 0 && R->w > 0 && R->data.size() == (size_t)R->h * R->w * R->c;
    const bool mok = m8 && !m8->empty();
    return fb_apply_confidence_raw(rok ? R->data.data() : nullptr, rok ? R->h : 0, rok ? R->w : 0,
                                   rok ? R->c : 0, mok ? m8->data.data() : nullptr,
                                   mok ? m8->h : 0, mok ? m8->w : 0, err, flow.ny, flow.nx,
                                   raw_h, raw_w, tile_size, cfg.fb_conf_subdiv, cfg.fb_conf_tau_px, cfg);
}

}  // namespace hhsr
