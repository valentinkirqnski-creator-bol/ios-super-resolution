// Port of Handheld-Multi-Frame-Super-Resolution/alignment (Halide) -- see hsr_align.h.
#include "hsr_align.h"
#include "parallel.h"
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace hhsr {
namespace hsra {

namespace {
inline int round_even(float v) { return (int)std::nearbyint(v); }   // Halide::round: ties to even
inline float clampf1(float v) { return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v); }
inline float flow_edge(const std::vector<float>& f, int nx, int ny, int x, int y) {
    x = x < 0 ? 0 : (x >= nx ? nx - 1 : x);
    y = y < 0 ? 0 : (y >= ny ? ny - 1 : y);
    return f[(size_t)y * nx + x];
}
// Undo the loaders' white-balance prewhitening: the package aligns on the raw without white balance.
inline float unwhite_gain(const Config& cfg, int c) {
    if (!cfg.raw_prewhitened || c < 0 || c > 2) return 1.f;
    const float g = cfg.white_balance[c] / cfg.white_balance[1];
    return (std::isfinite(g) && g > 0.f) ? 1.f / g : 1.f;
}
}  // namespace

// ------------------------------------------------------------------------------------------- settings
Options options_for_snr(float snr_db) {
    Options o;
    const float snr = std::min(30.f, std::max(6.f, std::isfinite(snr_db) ? snr_db : 6.f));
    o.snr_db = snr;
    o.base_tile = snr <= 14.f ? 64 : (snr <= 22.f ? 32 : 16);
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
    double top = 0.0, bot = 0.0;
    const int H = raw.h & ~1, W = raw.w & ~1;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int c = cfg.cfa.p[y & 1][x & 1];
            if (c < 0 || c > 2) c = 1;
            const float v = raw.data[(size_t)y * raw.w + x] * unwhite_gain(cfg, c);
            if (!(v > 0.f && v < 1.f)) continue;
            top += (double)v * v;
            bot += (double)cfg.alpha_dng[c] * v + cfg.beta_dng[c];
        }
    if (!(bot > 0.0) || !(top > 0.0)) return 6.f;
    return (float)(20.0 * std::log10(std::sqrt(top / bot)));
}

Plane grey_from_bayer(const Image& raw, const Config& cfg, int threads) {
    const int w = raw.w / 2, h = raw.h / 2;
    Plane q(w, h), t(w, h), out(w, h);
    float g[2][2];
    for (int sy = 0; sy < 2; ++sy)
        for (int sx = 0; sx < 2; ++sx) g[sy][sx] = unwhite_gain(cfg, cfg.cfa.p[sy][sx]);
    const float* r = raw.data.data();
    parallel_rows(h, threads, [&](int y) {
        const float* r0 = r + (size_t)(2 * y) * raw.w;
        const float* r1 = r0 + raw.w;
        for (int x = 0; x < w; ++x)
            q.at(x, y) = (((r0[2 * x] * g[0][0] + r0[2 * x + 1] * g[0][1]) + r1[2 * x] * g[1][0]) +
                          r1[2 * x + 1] * g[1][1]) * 0.25f;
    });
    // convolve1d([0.25, 0.5, 0.25], axis=0 then axis=1, mode="nearest")
    parallel_rows(h, threads, [&](int y) {
        for (int x = 0; x < w; ++x)
            t.at(x, y) = 0.25f * q.edge(x, y - 1) + 0.5f * q.at(x, y) + 0.25f * q.edge(x, y + 1);
    });
    parallel_rows(h, threads, [&](int y) {
        for (int x = 0; x < w; ++x) {
            const float v = 0.25f * t.edge(x - 1, y) + 0.5f * t.at(x, y) + 0.25f * t.edge(x + 1, y);
            out.at(x, y) = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        }
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
            for (int x = 0; x < w2; ++x)
                hz.at(x, y) = (cur.edge(2 * x - 1, y) + 2.0f * cur.edge(2 * x, y) + cur.edge(2 * x + 1, y)) * 0.25f;
        });
        parallel_rows(h2, threads, [&](int y) {
            for (int x = 0; x < w2; ++x)
                out.at(x, y) = (hz.edge(x, 2 * y - 1) + 2.0f * hz.edge(x, 2 * y) + hz.edge(x, 2 * y + 1)) * 0.25f;
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
            gx.at(x, y) = (in.edge(x - 2, y) - 8.0f * in.edge(x - 1, y) + 8.0f * in.edge(x + 1, y) - in.edge(x + 2, y)) / 12.0f;
            gy.at(x, y) = (in.edge(x, y - 2) - 8.0f * in.edge(x, y - 1) + 8.0f * in.edge(x, y + 1) - in.edge(x, y + 2)) / 12.0f;
        }
    });
    const int tx_n = W / ts, ty_n = H / ts;
    sxx = Plane(tx_n, ty_n); sxy = Plane(tx_n, ty_n); syy = Plane(tx_n, ty_n);
    parallel_rows(ty_n, threads, [&](int ty) {
        for (int tx = 0; tx < tx_n; ++tx) {
            float a = 0.0f, b = 0.0f, c = 0.0f;
            for (int py = 0; py < ts; ++py)
                for (int px = 0; px < ts; ++px) {
                    const float u = gx.at(tx * ts + px, ty * ts + py), v = gy.at(tx * ts + px, ty * ts + py);
                    a += u * u; b += u * v; c += v * v;
                }
            sxx.at(tx, ty) = a; sxy.at(tx, ty) = b; syy.at(tx, ty) = c;
        }
    });
}

Flow block_matching(const Plane& ref, const Plane& mov, const Flow& init, int ts, int radius, bool l2, int threads) {
    const int cols = ref.w / ts, rows = ref.h / ts;
    Flow out(cols, rows, ts, ts);
    parallel_rows(rows, threads, [&](int ty) {
        for (int tx = 0; tx < cols; ++tx) {
            const size_t i = (size_t)ty * cols + tx;
            const int cx = round_even(init.dx[i]), cy = round_even(init.dy[i]);
            int bx = 0, by = 0;
            float bc = INFINITY;
            for (int sy = -radius; sy <= radius; ++sy)
                for (int sx = -radius; sx <= radius; ++sx) {
                    float cost = 0.0f;
                    for (int py = 0; py < ts; ++py) {
                        const int y = ty * ts + py;
                        for (int px = 0; px < ts; ++px) {
                            const int x = tx * ts + px;
                            const float d = ref.at(x, y) - mov.edge(x + cx + sx, y + cy + sy);
                            cost += l2 ? d * d : std::fabs(d);
                        }
                    }
                    if (cost < bc) { bx = sx; by = sy; }
                    bc = std::min(cost, bc);
                }
            out.dx[i] = (float)(cx + bx);
            out.dy[i] = (float)(cy + by);
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
                const float d0x = cur.dx[i], d0y = cur.dy[i];
                float r0 = 0.0f, r1 = 0.0f;
                for (int py = 0; py < ts; ++py) {
                    const int y = ty * ts + py;
                    for (int px = 0; px < ts; ++px) {
                        const int x = tx * ts + px;
                        const float wx = (float)x + d0x, wy = (float)y + d0y;
                        const int ix = (int)std::floor(wx), iy = (int)std::floor(wy);
                        const float fx = wx - (float)ix, fy = wy - (float)iy;
                        const float a = mov.edge(ix, iy), b = mov.edge(ix + 1, iy);
                        const float c = mov.edge(ix, iy + 1), d = mov.edge(ix + 1, iy + 1);
                        const float top = a + fx * (b - a);
                        const float bottom = c + fx * (d - c);
                        const float warped = top + fy * (bottom - top);
                        const float res = ref.img.at(x, y) - warped;
                        r0 = r0 + ref.gx.at(x, y) * res;
                        r1 = r1 + ref.gy.at(x, y) * res;
                    }
                }
                const float a = ref.sxx.at(tx, ty), b = ref.sxy.at(tx, ty), c = ref.syy.at(tx, ty);
                const float det = a * c - b * b + FLT_EPSILON;
                const float ux = (c * r0 - b * r1) / det, uy = (a * r1 - b * r0) / det;
                nxt.dx[i] = d0x + clampf1(ux);
                nxt.dy[i] = d0y + clampf1(uy);
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
    const float spacing = grid_scale_ratio * (float)stride;
    parallel_rows(rows, threads, [&](int ty) {
        for (int tx = 0; tx < cols; ++tx) {
            const float u = ((float)(tx * stride) - off_x) / spacing;
            const float v = ((float)(ty * stride) - off_y) / spacing;
            const int left = (int)std::floor(u), top = (int)std::floor(v);
            const int nx = (int)std::floor(u + 0.5f), ny = (int)std::floor(v + 0.5f);
            const int ox = nx == left ? left + 1 : left, oy = ny == top ? top + 1 : top;
            const int cxs[3] = {nx, ox, nx}, cys[3] = {ny, ny, oy};
            float cdx[3], cdy[3];
            int best = 0;
            float bc = INFINITY;
            for (int c = 0; c < 3; ++c) {
                cdx[c] = flow_edge(src.dx, src.nx, src.ny, cxs[c], cys[c]) * flow_scale;
                cdy[c] = flow_edge(src.dy, src.nx, src.ny, cxs[c], cys[c]) * flow_scale;
                const int sdx = round_even(cdx[c]), sdy = round_even(cdy[c]);
                float cost = 0.0f;
                for (int py = 0; py < tt; ++py) {
                    const int y = ty * stride + py;
                    for (int px = 0; px < tt; ++px) {
                        const int x = tx * stride + px;
                        cost += std::fabs(ref.edge(x, y) - mov.edge(x + sdx, y + sdy));
                    }
                }
                if (cost < bc) best = c;
                bc = std::min(cost, bc);
            }
            const size_t i = (size_t)ty * cols + tx;
            out.dx[i] = cdx[best];
            out.dy[i] = cdy[best];
        }
    });
    return out;
}

Flow median_filter(const Flow& src) {
    Flow out(src.nx, src.ny, src.tile, src.stride);
    for (int y = 0; y < src.ny; ++y)
        for (int x = 0; x < src.nx; ++x) {
            float vx[9], vy[9];
            int k = 0;
            for (int j = -1; j <= 1; ++j)
                for (int i = -1; i <= 1; ++i) {
                    vx[k] = flow_edge(src.dx, src.nx, src.ny, x + i, y + j);
                    vy[k] = flow_edge(src.dy, src.nx, src.ny, x + i, y + j);
                    ++k;
                }
            std::nth_element(vx, vx + 4, vx + 9);
            std::nth_element(vy, vy + 4, vy + 9);
            out.dx[(size_t)y * src.nx + x] = vx[4];
            out.dy[(size_t)y * src.nx + x] = vy[4];
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
        const int target_tile = o.tiles[target];
        const int target_stride = target_tile;   // non-overlapping tiles between levels
        const float pixel_scale = (float)o.factors[target];
        const float grid_scale_ratio = pixel_scale * (float)flow.stride / (float)target_stride;
        const float source_center_px = (float)flow.tile * pixel_scale / 2.0f;
        const float target_center_px = (float)target_tile / 2.0f;
        const float offset_px = source_center_px - target_center_px;
        flow = transfer_flow(flow, ref.levels[target].img, mov.levels[target].img, target_tile, target_stride,
                             grid_scale_ratio, pixel_scale, offset_px, offset_px, threads);
    }
    // Final overlapping grid: 16x16 tiles, stride 8, first tile corners aligned.
    const int out_tile = 16, out_stride = 8;
    const float grid_scale_ratio = (float)flow.stride / (float)out_stride;
    const float offset_px = (float)flow.tile / 2.0f - (float)out_tile / 2.0f;
    flow = transfer_flow(flow, ref.levels[0].img, mov.levels[0].img, out_tile, out_stride, grid_scale_ratio, 1.0f,
                         offset_px, offset_px, threads);
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
