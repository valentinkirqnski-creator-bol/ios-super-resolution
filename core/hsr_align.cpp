// Port of Handheld-Multi-Frame-Super-Resolution/alignment (Halide) -- see hsr_align.h.
#include "hsr_align.h"
#include "hsr_align_shared.h"
#include "parallel.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

namespace hhsr {
namespace hsra {

namespace {
// Undo the loaders' white-balance prewhitening: the package aligns on the raw without white balance.
inline float unwhite_gain(const Config& cfg, int c) {
    if (!cfg.raw_prewhitened || c < 0 || c > 2) return 1.f;
    const float g = cfg.white_balance[c] / cfg.white_balance[1];
    return (std::isfinite(g) && g > 0.f) ? 1.f / g : 1.f;
}
}  // namespace

// ------------------------------------------------------------------------------------------- settings
Options options_for_snr(float snr_db, int force_tile) {
    Options o;
    const float snr = std::min(30.f, std::max(6.f, std::isfinite(snr_db) ? snr_db : 6.f));
    o.snr_db = snr;
    o.base_tile = snr <= 14.f ? 64 : (snr <= 22.f ? 32 : 16);
    if (force_tile == 16 || force_tile == 32 || force_tile == 64) o.base_tile = force_tile;
    o.tiles.assign(1, o.base_tile);
    double cap = 64.0;
    for (size_t level = 1; level < o.radii.size(); ++level) {
        const double t = std::max(8.0, std::min((double)o.tiles.back(), cap));
        o.tiles.push_back((int)t);   // radii {1,3,3,3,3}: always a power of two
        cap /= (double)(o.radii[level] + 1);
    }
    return o;
}

float estimate_image_snr_db(const Image& raw, const Config& cfg) {
    const int H = raw.h & ~1, W = raw.w & ~1;
    if (H <= 0 || W <= 0 || raw.data.size() < (size_t)raw.h * raw.w) return 6.f;
    int site_c[2][2];
    float site_g[2][2];
    for (int sy = 0; sy < 2; ++sy)
        for (int sx = 0; sx < 2; ++sx) {
            int c = cfg.cfa.p[sy][sx];
            if (c < 0 || c > 2) c = 1;
            site_c[sy][sx] = c;
            site_g[sy][sx] = unwhite_gain(cfg, c);
        }
    // Per-row double sums (rows in parallel), added in row order: deterministic.
    std::vector<double> row_top((size_t)H, 0.0), row_bot((size_t)H, 0.0);
    parallel_rows(H, 0, [&](int y) {
        double top = 0.0, bot = 0.0;
        const float* r = raw.data.data() + (size_t)y * raw.w;
        for (int x = 0; x < W; ++x) {
            const int c = site_c[y & 1][x & 1];
            const float v = r[x] * site_g[y & 1][x & 1];
            if (!(v > 0.f && v < 1.f)) continue;
            top += (double)v * v;
            bot += (double)cfg.alpha_dng[c] * v + cfg.beta_dng[c];
        }
        row_top[(size_t)y] = top;
        row_bot[(size_t)y] = bot;
    });
    double top = 0.0, bot = 0.0;
    for (int y = 0; y < H; ++y) { top += row_top[(size_t)y]; bot += row_bot[(size_t)y]; }
    if (!(bot > 0.0) || !(top > 0.0)) return 6.f;
    return (float)(20.0 * std::log10(std::sqrt(top / bot)));
}

void grey_gains(const Config& cfg, float g[4]) {
    for (int sy = 0; sy < 2; ++sy)
        for (int sx = 0; sx < 2; ++sx) g[sy * 2 + sx] = unwhite_gain(cfg, cfg.cfa.p[sy][sx]);
}

Plane grey_from_bayer(const Image& raw, const Config& cfg, int threads) {
    const int w = raw.w / 2, h = raw.h / 2;
    Plane q(w, h), t(w, h), out(w, h);
    float g[4];
    grey_gains(cfg, g);
    const float* r = raw.data.data();
    parallel_rows(h, threads, [&](int y) {
        for (int x = 0; x < w; ++x) q.at(x, y) = hsra_grey_quad(r, raw.w, g[0], g[1], g[2], g[3], x, y);
    });
    // convolve1d([0.25, 0.5, 0.25], axis=0 then axis=1, mode="nearest")
    parallel_rows(h, threads, [&](int y) {
        for (int x = 0; x < w; ++x) t.at(x, y) = hsra_blur_v(q.d.data(), w, h, x, y);
    });
    parallel_rows(h, threads, [&](int y) {
        for (int x = 0; x < w; ++x) out.at(x, y) = hsra_blur_h_clip(t.d.data(), w, h, x, y);
    });
    return out;
}

// ------------------------------------------------------------------------------------------- kernels
Plane downsample(const Plane& in, int factor, int threads) {
    if (factor <= 1) return in;
    Plane cur = in;
    for (int rem = factor; rem > 1; rem >>= 1) {
        const int w2 = cur.w / 2 + cur.w % 2, h2 = cur.h / 2 + cur.h % 2;
        Plane hz(w2, cur.h), out(w2, h2);
        parallel_rows(cur.h, threads, [&](int y) {
            for (int x = 0; x < w2; ++x) hz.at(x, y) = hsra_down_h(cur.d.data(), cur.w, cur.h, x, y);
        });
        parallel_rows(h2, threads, [&](int y) {
            for (int x = 0; x < w2; ++x) out.at(x, y) = hsra_down_v(hz.d.data(), w2, cur.h, x, y);
        });
        cur = std::move(out);
    }
    return cur;
}

void gradients(const Plane& in, int ts, Plane& gx, Plane& gy, Plane& sxx, Plane& sxy, Plane& syy, int threads) {
    const int W = in.w, H = in.h;
    gx = Plane(W, H); gy = Plane(W, H);
    parallel_rows(H, threads, [&](int y) {
        for (int x = 0; x < W; ++x) {
            gx.at(x, y) = hsra_grad_x(in.d.data(), W, H, x, y);
            gy.at(x, y) = hsra_grad_y(in.d.data(), W, H, x, y);
        }
    });
    const int tx_n = W / ts, ty_n = H / ts;
    sxx = Plane(tx_n, ty_n); sxy = Plane(tx_n, ty_n); syy = Plane(tx_n, ty_n);
    parallel_rows(ty_n, threads, [&](int ty) {
        for (int tx = 0; tx < tx_n; ++tx) {
            float s3[3];
            hsra_tile_structure(gx.d.data(), gy.d.data(), W, ts, tx, ty, s3);
            sxx.at(tx, ty) = s3[0]; sxy.at(tx, ty) = s3[1]; syy.at(tx, ty) = s3[2];
        }
    });
}

Flow block_matching(const Plane& ref, const Plane& mov, const Flow& init, int ts, int radius, bool l2, int threads) {
    const int cols = ref.w / ts, rows = ref.h / ts;
    Flow out(cols, rows, ts, ts);
    parallel_rows(rows, threads, [&](int ty) {
        for (int tx = 0; tx < cols; ++tx) {
            const size_t i = (size_t)ty * cols + tx;
            float o[2];
            hsra_block_match_tile(ref.d.data(), mov.d.data(), ref.w, ref.h, init.dx[i], init.dy[i], ts, radius, l2 ? 1 : 0,
                                  tx, ty, o);
            out.dx[i] = o[0];
            out.dy[i] = o[1];
        }
    });
    return out;
}

Flow ica(const Level& ref, const Plane& mov, const Flow& init, int iterations, int threads) {
    Flow cur = init;
    const int ts = init.tile, cols = init.nx, rows = init.ny;
    for (int it = 0; it < iterations; ++it) {
        Flow nxt(cols, rows, ts, ts);
        parallel_rows(rows, threads, [&](int ty) {
            for (int tx = 0; tx < cols; ++tx) {
                const size_t i = (size_t)ty * cols + tx;
                float o[2];
                hsra_ica_tile(ref.img.d.data(), ref.gx.d.data(), ref.gy.d.data(), mov.d.data(), mov.w, mov.h, ts, tx, ty,
                              cur.dx[i], cur.dy[i], ref.sxx.at(tx, ty), ref.sxy.at(tx, ty), ref.syy.at(tx, ty), o);
                nxt.dx[i] = o[0];
                nxt.dy[i] = o[1];
            }
        });
        cur = std::move(nxt);
    }
    return cur;
}

Flow transfer_flow(const Flow& src, const Plane& ref, const Plane& mov, int tt, int stride, float grid_scale_ratio,
                   float flow_scale, float off_x, float off_y, int threads) {
    const int cols = (ref.w - tt) / stride + 1, rows = (ref.h - tt) / stride + 1;
    Flow out(cols, rows, tt, stride);
    HsraTransferParams P;
    P.w = ref.w; P.h = ref.h; P.tt = tt; P.stride = stride; P.snx = src.nx; P.sny = src.ny;
    P.spacing = grid_scale_ratio * (float)stride; P.flow_scale = flow_scale; P.off_x = off_x; P.off_y = off_y;
    parallel_rows(rows, threads, [&](int ty) {
        for (int tx = 0; tx < cols; ++tx) {
            float o[2];
            hsra_transfer_tile(src.dx.data(), src.dy.data(), ref.d.data(), mov.d.data(), P, tx, ty, o);
            const size_t i = (size_t)ty * cols + tx;
            out.dx[i] = o[0];
            out.dy[i] = o[1];
        }
    });
    return out;
}

Flow transfer_flow(const Flow& src, const Plane& ref, const Plane& mov, const HsraTransferParams& P, int threads) {
    return transfer_flow(src, ref, mov, P.tt, P.stride, P.spacing / (float)P.stride, P.flow_scale, P.off_x, P.off_y,
                         threads);
}

HsraTransferParams level_transfer_params(const Options& o, size_t source, int src_nx, int src_ny, int tw, int th) {
    const size_t target = source - 1;
    const int target_tile = o.tiles[target];
    const int target_stride = target_tile;   // non-overlapping tiles between levels
    const int src_tile = o.tiles[source], src_stride = o.tiles[source];
    const float pixel_scale = (float)o.factors[target];
    const float grid_scale_ratio = pixel_scale * (float)src_stride / (float)target_stride;
    const float source_center_px = (float)src_tile * pixel_scale / 2.0f;
    const float target_center_px = (float)target_tile / 2.0f;
    const float offset_px = source_center_px - target_center_px;
    HsraTransferParams P;
    P.w = tw; P.h = th; P.tt = target_tile; P.stride = target_stride; P.snx = src_nx; P.sny = src_ny;
    P.spacing = grid_scale_ratio * (float)target_stride; P.flow_scale = pixel_scale; P.off_x = offset_px; P.off_y = offset_px;
    return P;
}

HsraTransferParams final_transfer_params(const Options& o, int src_nx, int src_ny, int w, int h) {
    // Final overlapping grid: 16x16 tiles, stride 8, first tile corners aligned.
    const int out_tile = 16, out_stride = 8;
    const float grid_scale_ratio = (float)o.tiles[0] / (float)out_stride;
    const float offset_px = (float)o.tiles[0] / 2.0f - (float)out_tile / 2.0f;
    HsraTransferParams P;
    P.w = w; P.h = h; P.tt = out_tile; P.stride = out_stride; P.snx = src_nx; P.sny = src_ny;
    P.spacing = grid_scale_ratio * (float)out_stride; P.flow_scale = 1.0f; P.off_x = offset_px; P.off_y = offset_px;
    return P;
}

void pyramid_dims(int w, int h, const Options& opt, std::vector<int>& ws, std::vector<int>& hs) {
    ws.assign(1, w); hs.assign(1, h);
    for (size_t i = 0; i < opt.factors.size(); ++i) {
        for (int rem = opt.factors[i]; rem > 1; rem >>= 1) { w = w / 2 + w % 2; h = h / 2 + h % 2; }
        ws.push_back(w); hs.push_back(h);
    }
}

Flow median_filter(const Flow& src) {
    Flow out(src.nx, src.ny, src.tile, src.stride);
    for (int y = 0; y < src.ny; ++y)
        for (int x = 0; x < src.nx; ++x) {
            out.dx[(size_t)y * src.nx + x] = hsra_median9(src.dx.data(), src.nx, src.ny, x, y);
            out.dy[(size_t)y * src.nx + x] = hsra_median9(src.dy.data(), src.nx, src.ny, x, y);
        }
    return out;
}

// ------------------------------------------------------------------------------------------- driver
Pyramid build_pyramid(const Plane& grey, const Options& opt, bool reference, int threads) {
    Pyramid p;
    p.reference = reference;
    p.opt = opt;
    const size_t n = opt.factors.size() + 1;
    Plane cur = grey;
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) cur = downsample(cur, opt.factors[i - 1], threads);
        Level lv;
        lv.img = cur;
        if (reference && i < opt.tiles.size() && cur.w >= opt.tiles[i] && cur.h >= opt.tiles[i])
            gradients(cur, opt.tiles[i], lv.gx, lv.gy, lv.sxx, lv.sxy, lv.syy, threads);
        p.levels.push_back(std::move(lv));
    }
    return p;
}

Flow align(const Pyramid& ref, const Pyramid& mov, int threads) {
    const Options& o = ref.opt;
    const size_t n = ref.levels.size();
    if (n == 0 || mov.levels.size() != n || o.tiles.size() != n || o.radii.size() != n || o.l2.size() != n)
        return Flow();
    for (size_t i = 0; i < n; ++i) {
        const Plane& a = ref.levels[i].img;
        if (a.w != mov.levels[i].img.w || a.h != mov.levels[i].img.h) return Flow();
        if (a.w < o.tiles[i] || a.h < o.tiles[i] || ref.levels[i].sxx.d.empty()) return Flow();   // sanitize_config
    }
    if (ref.levels[0].img.w < 16 || ref.levels[0].img.h < 16) return Flow();
    const size_t coarsest = n - 1;
    const int ts0 = o.tiles[coarsest];
    Flow flow(ref.levels[coarsest].img.w / ts0, ref.levels[coarsest].img.h / ts0, ts0, ts0);
    for (size_t level = coarsest;; --level) {
        const Level& R = ref.levels[level];
        const Plane& M = mov.levels[level].img;
        flow = block_matching(R.img, M, flow, o.tiles[level], o.radii[level], o.l2[level] != 0, threads);
        if (o.ica_iterations > 0) flow = ica(R, M, flow, o.ica_iterations, threads);
        if (level == 0) break;
        const size_t target = level - 1;
        flow = transfer_flow(flow, ref.levels[target].img, mov.levels[target].img,
                             level_transfer_params(o, level, flow.nx, flow.ny, ref.levels[target].img.w,
                                                   ref.levels[target].img.h), threads);
    }
    flow = transfer_flow(flow, ref.levels[0].img, mov.levels[0].img,
                         final_transfer_params(o, flow.nx, flow.ny, ref.levels[0].img.w, ref.levels[0].img.h), threads);
    return median_filter(flow);
}

FlowField to_raw_flow_field(const Flow& f, int raw_w, int raw_h) {
    if (f.nx <= 0 || f.ny <= 0) return FlowField();
    const int nx = std::max(f.nx, (raw_w + 15) / 16), ny = std::max(f.ny, (raw_h + 15) / 16);
    FlowField out(ny, nx);
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            const size_t i = (size_t)std::min(y, f.ny - 1) * f.nx + std::min(x, f.nx - 1);
            out.dx(y, x) = 2.0f * f.dx[i];
            out.dy(y, x) = 2.0f * f.dy[i];
        }
    return out;
}

}  // namespace hsra
}  // namespace hhsr
