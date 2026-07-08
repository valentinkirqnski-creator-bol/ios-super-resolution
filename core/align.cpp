#include "stages.h"
#include "parallel.h"
#include "linalg.h"

namespace hhsr {

// Bilinear sample with clamp-to-edge.
static inline f32 sample(const Image& img, f32 y, f32 x) {
    y = clampf(y, 0.f, (f32)(img.h - 1));
    x = clampf(x, 0.f, (f32)(img.w - 1));
    int y0 = (int)std::floor(y), x0 = (int)std::floor(x);
    int y1 = std::min(y0 + 1, img.h - 1), x1 = std::min(x0 + 1, img.w - 1);
    f32 fy = y - y0, fx = x - x0;
    f32 top = img.at(y0, x0) * (1 - fx) + img.at(y0, x1) * fx;
    f32 bot = img.at(y1, x0) * (1 - fx) + img.at(y1, x1) * fx;
    return top * (1 - fy) + bot * fy;
}

// Block matching at one pyramid level. `init` provides a per-tile flow guess
// (already scaled to this level's pixels). Returns refined per-tile flow.
static FlowField block_match_level(const Image& ref, const Image& moving,
                                   int tile_size, int search_radius,
                                   const FlowField& init, bool use_l1,
                                   int num_threads) {
    int ny = std::max(1, ref.h / tile_size);
    int nx = std::max(1, ref.w / tile_size);
    FlowField flow(ny, nx);

    parallel_rows(ny, num_threads, [&](int ty) {
        for (int tx = 0; tx < nx; ++tx) {
            f32 gx = init.nx > 0 ? init.dx(std::min(ty, init.ny - 1), std::min(tx, init.nx - 1)) : 0.f;
            f32 gy = init.nx > 0 ? init.dy(std::min(ty, init.ny - 1), std::min(tx, init.nx - 1)) : 0.f;

            int oy = ty * tile_size, ox = tx * tile_size;
            f32 best = 1e30f, bestdx = gx, bestdy = gy;

            for (int sdy = -search_radius; sdy <= search_radius; ++sdy) {
                for (int sdx = -search_radius; sdx <= search_radius; ++sdx) {
                    f32 fdx = gx + sdx, fdy = gy + sdy;
                    f32 cost = 0.f;
                    for (int i = 0; i < tile_size; ++i) {
                        int ry = oy + i;
                        if (ry >= ref.h) break;
                        for (int j = 0; j < tile_size; ++j) {
                            int rx = ox + j;
                            if (rx >= ref.w) break;
                            f32 m = sample(moving, ry + fdy, rx + fdx);
                            f32 d = ref.at(ry, rx) - m;
                            cost += use_l1 ? std::fabs(d) : d * d;
                        }
                    }
                    if (cost < best) { best = cost; bestdx = fdx; bestdy = fdy; }
                }
            }
            flow.dx(ty, tx) = bestdx;
            flow.dy(ty, tx) = bestdy;
        }
    });
    return flow;
}

// Nearest-neighbour resize of a flow field to a new tile grid, scaling the
// vectors by `mul` (the pixel-size ratio between the two levels).
static FlowField upscale_flow(const FlowField& in, int ny, int nx, f32 mul) {
    FlowField out(ny, nx);
    for (int ty = 0; ty < ny; ++ty) {
        for (int tx = 0; tx < nx; ++tx) {
            int sy = in.ny > 0 ? std::min((int)((f32)ty / ny * in.ny), in.ny - 1) : 0;
            int sx = in.nx > 0 ? std::min((int)((f32)tx / nx * in.nx), in.nx - 1) : 0;
            out.dx(ty, tx) = in.dx(sy, sx) * mul;
            out.dy(ty, tx) = in.dy(sy, sx) * mul;
        }
    }
    return out;
}

// Inverse-compositional Lucas-Kanade (ICA) refinement at the finest level.
// Solves a per-tile 2x2 system from the reference gradients (Alg. 3).
static void ica_refine(const Image& ref, const Image& grad, const Image& moving,
                       FlowField& flow, int tile_size, int n_iter, int num_threads) {
    int ny = flow.ny, nx = flow.nx;
    parallel_rows(ny, num_threads, [&](int ty) {
        for (int tx = 0; tx < nx; ++tx) {
            int oy = ty * tile_size, ox = tx * tile_size;

            // Hessian H = sum(J^T J) over the tile (constant across iterations).
            f32 h00 = 0, h01 = 0, h11 = 0;
            for (int i = 0; i < tile_size; ++i) {
                int gy = oy + i;
                if (gy >= grad.h) break;
                for (int j = 0; j < tile_size; ++j) {
                    int gx = ox + j;
                    if (gx >= grad.w) break;
                    f32 dx = grad.at(gy, gx, 0), dy = grad.at(gy, gx, 1);
                    h00 += dx * dx; h01 += dx * dy; h11 += dy * dy;
                }
            }
            f32 ixx, ixy, iyy;
            if (!invert_sym_2x2(h00, h01, h11, ixx, ixy, iyy)) continue;

            f32 fx = flow.dx(ty, tx), fy = flow.dy(ty, tx);
            for (int it = 0; it < n_iter; ++it) {
                f32 b0 = 0, b1 = 0;
                for (int i = 0; i < tile_size; ++i) {
                    int gy = oy + i;
                    if (gy >= grad.h) break;
                    for (int j = 0; j < tile_size; ++j) {
                        int gx = ox + j;
                        if (gx >= grad.w) break;
                        f32 dxg = grad.at(gy, gx, 0), dyg = grad.at(gy, gx, 1);
                        f32 diff = sample(moving, gy + fy, gx + fx) - ref.at(gy, gx);
                        b0 += dxg * diff; b1 += dyg * diff;
                    }
                }
                // delta = H^-1 b
                f32 ddx = ixx * b0 + ixy * b1;
                f32 ddy = ixy * b0 + iyy * b1;
                fx -= ddx; fy -= ddy;
            }
            flow.dx(ty, tx) = fx;
            flow.dy(ty, tx) = fy;
        }
    });
}

FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg, int tile_size) {
    Pyramid mov_pyr = build_pyramid(moving_grey, cfg.bm_factors);

    int nlev = (int)ref_pyr.levels.size();
    FlowField flow; // empty => zero init at coarsest level

    // Coarse (last) -> fine (first).
    for (int lvl = nlev - 1; lvl >= 0; --lvl) {
        const Image& r = ref_pyr.levels[lvl];
        const Image& m = mov_pyr.levels[lvl];
        int ts = (lvl < (int)cfg.bm_tile_sizes.size())
                     ? std::max(4, cfg.bm_tile_sizes[lvl])
                     : std::max(4, tile_size);
        int radius = (lvl < (int)cfg.bm_search_radii.size()) ? cfg.bm_search_radii[lvl] : 2;
        bool use_l1 = (lvl == 0); // reference uses L1 at the finest scale

        int ny = std::max(1, r.h / ts);
        int nx = std::max(1, r.w / ts);

        FlowField init = (flow.nx == 0) ? FlowField(ny, nx)
                                        : upscale_flow(flow, ny, nx,
                                              lvl + 1 < nlev
                                                  ? (f32)ref_pyr.abs_factors[lvl + 1] / ref_pyr.abs_factors[lvl]
                                                  : 1.f);
        flow = block_match_level(r, m, ts, radius, init, use_l1, cfg.num_threads);
    }

    // ICA refinement on the finest grey image.
    Image grad = compute_gradients(ref_grey);
    ica_refine(ref_grey, grad, moving_grey, flow, tile_size, cfg.ica_n_iter, cfg.num_threads);
    return flow;
}

} // namespace hhsr
