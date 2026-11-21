#include "frame_gate.h"
#include "stages.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>

namespace hhsr {
namespace {

// Central-difference gradients (edge clamp) and the gradient-magnitude image.
static void grad_images(const Image& a, Image& gx, Image& gy, Image& mag) {
    const int W = a.w, H = a.h;
    gx = Image(H, W, 1); gy = Image(H, W, 1); mag = Image(H, W, 1);
    for (int y = 0; y < H; ++y) {
        const int yu = std::max(0, y - 1), yd = std::min(H - 1, y + 1);
        for (int x = 0; x < W; ++x) {
            const int xl = std::max(0, x - 1), xr = std::min(W - 1, x + 1);
            const f32 dx = 0.5f * (a.at(y, xr) - a.at(y, xl));
            const f32 dy = 0.5f * (a.at(yd, x) - a.at(yu, x));
            gx.at(y, x) = dx; gy.at(y, x) = dy;
            mag.at(y, x) = std::sqrt(dx * dx + dy * dy);
        }
    }
}

static inline f32 at_clamped(const Image& im, int y, int x) {
    y = std::min(std::max(y, 0), im.h - 1);
    x = std::min(std::max(x, 0), im.w - 1);
    return im.at(y, x);
}

} // namespace

FrameGateResult frame_gate_coarse(const Image& ref_coarse, const Image& comp_coarse,
                                  int tile_size, int search_radius, f32 tile_min_score,
                                  const Config& cfg) {
    FrameGateResult res;
    if (ref_coarse.h < tile_size || ref_coarse.w < tile_size || tile_size <= 0 ||
        comp_coarse.h != ref_coarse.h || comp_coarse.w != ref_coarse.w ||
        ref_coarse.data.size() < (size_t)ref_coarse.h * ref_coarse.w ||
        comp_coarse.data.size() < (size_t)comp_coarse.h * comp_coarse.w)
        return res;

    // 1) gradient images
    Image rgx, rgy, rmag, cgx, cgy, cmag;
    grad_images(ref_coarse, rgx, rgy, rmag);
    grad_images(comp_coarse, cgx, cgy, cmag);

    // 2) the pipeline's coarse block matching, on the gradient images
    const FlowField flow = block_match_coarse(rmag, cmag, tile_size, search_radius, cfg);
    if (flow.ny <= 0 || flow.nx <= 0) return res;

    // 3) overlay: each tile of the comparison shifted by its matched offset (comparison pixel
    //    p + flow shows reference pixel p), correlated with the reference's gradients
    const int nt = flow.ny * flow.nx;
    std::vector<f32> energy((size_t)nt, 0.f), score((size_t)nt, 1.f);
    std::vector<uint8_t> outside((size_t)nt, 0);
    // The frame's global offset = median tile offset. A reference tile whose content lies
    // outside the comparison frame under that offset (content that left the frame) has
    // nothing to compare with -- the block matching then returns some in-frame match --
    // so it is not judged either.
    int gdx = 0, gdy = 0;
    {
        std::vector<f32> vx((size_t)nt), vy((size_t)nt);
        for (int t = 0; t < nt; ++t) { vx[(size_t)t] = flow.flow[(size_t)t * 2]; vy[(size_t)t] = flow.flow[(size_t)t * 2 + 1]; }
        std::nth_element(vx.begin(), vx.begin() + nt / 2, vx.end());
        std::nth_element(vy.begin(), vy.begin() + nt / 2, vy.end());
        gdx = (int)std::lround(vx[(size_t)nt / 2]); gdy = (int)std::lround(vy[(size_t)nt / 2]);
    }
    for (int ty = 0; ty < flow.ny; ++ty)
        for (int tx = 0; tx < flow.nx; ++tx) {
            const int dx = (int)std::lround(flow.dx(ty, tx)), dy = (int)std::lround(flow.dy(ty, tx));
            // No overlap to compare: the matched block lies (partly) outside the comparison
            // frame -- content that entered / left the frame, not rotation or parallax.
            const size_t t0 = (size_t)ty * flow.nx + tx;
            if (tx * tile_size + dx < 0 || (tx + 1) * tile_size + dx > cgx.w ||
                ty * tile_size + dy < 0 || (ty + 1) * tile_size + dy > cgx.h ||
                tx * tile_size + gdx < 0 || (tx + 1) * tile_size + gdx > cgx.w ||
                ty * tile_size + gdy < 0 || (ty + 1) * tile_size + gdy > cgx.h) {
                outside[t0] = 1;
                continue;
            }
            double num = 0, er = 0, ec = 0; int n = 0;
            for (int y = ty * tile_size; y < (ty + 1) * tile_size; ++y)
                for (int x = tx * tile_size; x < (tx + 1) * tile_size; ++x) {
                    const f32 ax = rgx.at(y, x), ay = rgy.at(y, x);
                    const f32 bx = at_clamped(cgx, y + dy, x + dx), by = at_clamped(cgy, y + dy, x + dx);
                    num += (double)ax * bx + (double)ay * by;
                    er += (double)ax * ax + (double)ay * ay;
                    ec += (double)bx * bx + (double)by * by;
                    ++n;
                }
            const size_t t = (size_t)ty * flow.nx + tx;
            energy[t] = n > 0 ? (f32)(er / n) : 0.f;
            const double den = std::sqrt(er * ec);
            score[t] = den > 0.0 ? (f32)(num / den) : 1.f;
        }

    // Judge only structured tiles: energy >= 9x the 10th percentile (the flattest tiles).
    std::vector<f32> e;
    for (int t = 0; t < nt; ++t) if (!outside[(size_t)t]) e.push_back(energy[(size_t)t]);
    if (e.empty()) return res;
    std::sort(e.begin(), e.end());
    const f32 floor_e = std::max(e[(size_t)std::min<size_t>(e.size() - 1, e.size() / 10)], 1e-12f);
    std::vector<f32> judged;
    for (int t = 0; t < nt; ++t) {
        if (outside[(size_t)t] || !(energy[(size_t)t] >= 9.f * floor_e)) continue;
        const f32 s = score[(size_t)t];
        judged.push_back(s);
        res.worst_score = std::min(res.worst_score, s);
        if (s < tile_min_score) {
            ++res.bad_tiles;
            res.bad_tile_ids.push_back(t);
        }
    }
    res.judged_tiles = (int)judged.size();
    res.tiles_x = flow.nx; res.tiles_y = flow.ny;
    if (!judged.empty()) {
        res.bad_fraction = (f32)res.bad_tiles / (f32)res.judged_tiles;
        std::nth_element(judged.begin(), judged.begin() + judged.size() / 2, judged.end());
        res.median_score = judged[judged.size() / 2];
    }
    return res;
}

} // namespace hhsr
