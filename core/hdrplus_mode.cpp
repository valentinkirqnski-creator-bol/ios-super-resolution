//
// HDR+ mode (Config::hdrplus_mode): burst align + merge from hdr-plus-master
// (the Halide implementation, src/align.cpp + src/merge.cpp + src/util.cpp),
// replacing the whole super-resolution pipeline when enabled. CPU golden
// reference here; the Metal twins live in metal_gpu.mm / HHSRKernels.metal
// (hdrplus_metal_begin/add_frame/finish) and this file falls back to the CPU
// body when they are unavailable or fail.
//
// Halide-to-C++ semantics preserved deliberately:
//  - BoundaryConditions::mirror_interior -> hp_mirror() (period 2n-2, the
//    edge pixel is not repeated). Applied at the mosaic AND at each pyramid
//    layer; Halide extends a layer by filtering the mirrored source instead,
//    which differs only in the outermost boundary taps (documented deviation,
//    same one as the desktop port this was verified against).
//  - Halide integer division floors; hp_div_floor() is used wherever an
//    offset can be negative.
//  - argmin over RDom(-4,8,-4,8): r.x innermost, first minimum wins ties ->
//    yi outer / xi inner with strict '<'.
//  - The app's frames are normalized floats (u16 sensor values in the
//    original), so the L1-distance-tuned constants (min_dist 10 / max_dist
//    300, on 12-bit values) are kept meaningful by scaling distances with
//    kHpValueScale = 4095. Alignment needs no scale (argmin is
//    scale-invariant); only the merge weights do.
//  - Constants from align.h: T_SIZE 32, T_SIZE_2 16, DOWNSAMPLE_RATE 4,
//    MIN/MAX_OFFSET -168/126; from merge.cpp: factor 8, min_dist 10,
//    max_dist 300.
//
// The output is a denoised Bayer mosaic at INPUT resolution (HDR+ is a
// denoiser, not an upscaler; Config::scale is ignored here). It is
// demosaicked bilinearly for the app's RGB DNG path -- the Halide repo's
// finish.cpp (its own demosaic + tone map + sharpen) is intentionally not
// ported, because the app's output is a DNG meant for a raw developer.
//
#include "pipeline.h"
#include "stages.h"
#include "dng_writer.h"
#include "parallel.h"
#include "metal_gpu.h"
#include "prof.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace hhsr {

namespace {

constexpr int kHpTSize = 32;       // tile size in the Bayer mosaic
constexpr int kHpTSize2 = 16;      // tile size throughout the alignment pyramid
constexpr int kHpDownsample = 4;   // rate between alignment pyramid layers
constexpr int kHpMinOffset = -168;
constexpr int kHpMaxOffset = 126;
constexpr float kHpValueScale = 4095.f;  // normalized float -> 12-bit range

inline int hp_div_floor(int a, int b) {  // b > 0
    int q = a / b, r = a % b;
    return (r != 0 && (r < 0)) ? q - 1 : q;
}

inline int hp_mirror(int x, int n) {  // mirror_interior: period 2n-2
    if (n <= 1) return 0;
    const int p = 2 * n - 2;
    x = ((x % p) + p) % p;
    return x < n ? x : p - x;
}

struct HpPlane {
    int w = 0, h = 0;
    std::vector<f32> d;
    HpPlane() = default;
    HpPlane(int hh, int ww) : w(ww), h(hh), d((size_t)hh * ww, 0.f) {}
    inline f32 at(int x, int y) const {  // mirrored read, any coordinate
        return d[(size_t)hp_mirror(y, h) * w + hp_mirror(x, w)];
    }
    inline f32 raw(int x, int y) const { return d[(size_t)y * w + x]; }
};

HpPlane hp_from_image(const Image& img) {
    HpPlane p(img.h, img.w);
    // single-channel mosaic expected; channel 0 either way
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x)
            p.d[(size_t)y * img.w + x] = img.at(y, x, 0);
    return p;
}

// util.cpp box_down2 (u16 sum/4 there; float mean here).
HpPlane hp_box_down2(const HpPlane& in, int num_threads) {
    HpPlane out(in.h / 2, in.w / 2);
    parallel_rows(out.h, num_threads, [&](int y) {
        for (int x = 0; x < out.w; ++x)
            out.d[(size_t)y * out.w + x] =
                0.25f * (in.raw(2 * x, 2 * y) + in.raw(2 * x + 1, 2 * y) +
                         in.raw(2 * x, 2 * y + 1) + in.raw(2 * x + 1, 2 * y + 1));
    });
    return out;
}

// util.cpp gauss_down4: 5x5 integer kernel / 159, stride 4.
HpPlane hp_gauss_down4(const HpPlane& in, int num_threads) {
    static const f32 k[5][5] = {{2, 4, 5, 4, 2},
                                {4, 9, 12, 9, 4},
                                {5, 12, 15, 12, 5},
                                {4, 9, 12, 9, 4},
                                {2, 4, 5, 4, 2}};
    HpPlane out(in.h / 4, in.w / 4);
    parallel_rows(out.h, num_threads, [&](int y) {
        for (int x = 0; x < out.w; ++x) {
            f32 s = 0.f;
            for (int j = -2; j <= 2; ++j)
                for (int i = -2; i <= 2; ++i)
                    s += in.at(4 * x + i, 4 * y + j) * k[j + 2][i + 2];
            out.d[(size_t)y * out.w + x] = s / 159.f;
        }
    });
    return out;
}

struct HpAlignGrid {
    int nx = 0, ny = 0;
    std::vector<int> dx, dy;
    inline int idx(int tx, int ty) const {  // repeat_edge
        tx = std::min(std::max(tx, 0), nx - 1);
        ty = std::min(std::max(ty, 0), ny - 1);
        return ty * nx + tx;
    }
};

// align.cpp align_layer: best offset per 16x16 tile (stride 8) given the
// coarser layer's alignment scaled by DOWNSAMPLE_RATE and clamped to
// [prev_min, prev_max].
HpAlignGrid hp_align_layer(const HpPlane& ref, const HpPlane& alt,
                           const HpAlignGrid* prev, int prev_min, int prev_max,
                           int num_threads) {
    HpAlignGrid out;
    out.nx = std::max(1, ref.w / (kHpTSize2 / 2) - 1);
    out.ny = std::max(1, ref.h / (kHpTSize2 / 2) - 1);
    out.dx.assign((size_t)out.nx * out.ny, 0);
    out.dy.assign((size_t)out.nx * out.ny, 0);
    parallel_rows(out.ny, num_threads, [&](int ty) {
        for (int tx = 0; tx < out.nx; ++tx) {
            int pox = 0, poy = 0;
            if (prev) {
                const int ptx = hp_div_floor(tx - 1, kHpDownsample);
                const int pty = hp_div_floor(ty - 1, kHpDownsample);
                const int pi = prev->idx(ptx, pty);
                pox = kHpDownsample * std::min(std::max(prev->dx[pi], prev_min), prev_max);
                poy = kHpDownsample * std::min(std::max(prev->dy[pi], prev_min), prev_max);
            }
            const int x0 = tx * (kHpTSize2 / 2);
            const int y0 = ty * (kHpTSize2 / 2);
            f32 best = std::numeric_limits<f32>::max();
            int bxi = 0, byi = 0;
            for (int yi = -4; yi <= 3; ++yi) {
                for (int xi = -4; xi <= 3; ++xi) {
                    f32 score = 0.f;
                    for (int j = 0; j < kHpTSize2; ++j)
                        for (int i = 0; i < kHpTSize2; ++i)
                            score += std::fabs(ref.at(x0 + i, y0 + j) -
                                               alt.at(x0 + pox + xi + i, y0 + poy + yi + j));
                    if (score < best) { best = score; bxi = xi; byi = yi; }
                }
            }
            out.dx[(size_t)ty * out.nx + tx] = bxi + pox;
            out.dy[(size_t)ty * out.nx + tx] = byi + poy;
        }
    });
    return out;
}

}  // namespace

// Streaming CPU accumulator: init with the reference, add each comparison
// frame as it is decoded, finish into the merged mosaic. Matches the
// per-tile formulation of merge.cpp exactly through a per-pixel identity:
// each output pixel is covered by exactly 4 half-overlapped tiles, one per
// (tx parity, ty parity) class, so 4 parity planes S[p] accumulate
// w_tile * warped comparison values and the per-tile weight sums live in
// Wsum. finish() then evaluates merge_spatial's 4-tile raised-cosine blend
// with S[p]/Wsum[tile_p].
struct HdrPlusCpuState {
    int H = 0, W = 0, num_tx = 0, num_ty = 0, nthreads = 0;
    HpPlane ref;                // reference mosaic (per-pixel term, weight 1)
    HpPlane ref_l0, ref_l1, ref_l2;
    std::vector<f32> S[4];      // H*W each; parity p = (tx&1) | ((ty&1)<<1)
    std::vector<f32> wsum;      // (num_tx+2)*(num_ty+2), tiles -1..num_t*
    f32 window[kHpTSize];

    inline int wsum_idx(int tx, int ty) const {
        return (ty + 1) * (num_tx + 2) + (tx + 1);
    }

    bool init(const Image& ref_img, const Config& cfg) {
        nthreads = cfg.num_threads;
        H = ref_img.h; W = ref_img.w;
        num_tx = W / kHpTSize2 - 1;
        num_ty = H / kHpTSize2 - 1;
        if (num_tx < 1 || num_ty < 1) return false;
        ref = hp_from_image(ref_img);
        ref_l0 = hp_box_down2(ref, nthreads);
        ref_l1 = hp_gauss_down4(ref_l0, nthreads);
        ref_l2 = hp_gauss_down4(ref_l1, nthreads);
        for (int p = 0; p < 4; ++p) S[p].assign((size_t)H * W, 0.f);
        // reference merges with itself at weight 1 in every tile
        parallel_rows(H, nthreads, [&](int y) {
            for (int x = 0; x < W; ++x) {
                const f32 v = ref.raw(x, y);
                for (int p = 0; p < 4; ++p) S[p][(size_t)y * W + x] = v;
            }
        });
        wsum.assign((size_t)(num_tx + 2) * (num_ty + 2), 1.f);
        for (int v = 0; v < kHpTSize; ++v)
            window[v] = 0.5f - 0.5f * std::cos(2.f * 3.141592f * (v + 0.5f) / kHpTSize);
        return true;
    }

    void add_frame(const Image& comp_img) {
        const HpPlane comp = hp_from_image(comp_img);
        const HpPlane l0 = hp_box_down2(comp, nthreads);
        const HpPlane l1 = hp_gauss_down4(l0, nthreads);
        const HpPlane l2 = hp_gauss_down4(l1, nthreads);

        // coarse-to-fine alignment (align.cpp bounds: min/max search -4/3)
        const int min_2 = -4, max_2 = 3;                      // 4*0 + search
        const int min_1 = kHpDownsample * min_2 - 4;          // -20
        const int max_1 = kHpDownsample * max_2 + 3;          //  15
        HpAlignGrid a2 = hp_align_layer(ref_l2, l2, nullptr, 0, 0, nthreads);
        HpAlignGrid a1 = hp_align_layer(ref_l1, l1, &a2, min_2, max_2, nthreads);
        HpAlignGrid a0 = hp_align_layer(ref_l0, l0, &a1, min_1, max_1, nthreads);
        for (size_t i = 0; i < a0.dx.size(); ++i) {  // back to Bayer coords
            a0.dx[i] *= 2;
            a0.dy[i] *= 2;
        }

        // per-tile temporal weights on the half-res layer (merge_temporal)
        const f32 factor = 8.f;
        const int min_dist = 10, max_dist = 300;
        const int wnx = num_tx + 2, wny = num_ty + 2;
        std::vector<f32> wt((size_t)wnx * wny, 0.f);
        parallel_rows(wny, nthreads, [&](int wy) {
            const int ty = wy - 1;
            for (int wx = 0; wx < wnx; ++wx) {
                const int tx = wx - 1;
                const int ai = a0.idx(tx, ty);
                const int offx = std::min(std::max(a0.dx[ai], kHpMinOffset), kHpMaxOffset);
                const int offy = std::min(std::max(a0.dy[ai], kHpMinOffset), kHpMaxOffset);
                const int ox2 = hp_div_floor(offx, 2), oy2 = hp_div_floor(offy, 2);
                const int lx0 = tx * (kHpTSize2 / 2), ly0 = ty * (kHpTSize2 / 2);
                f32 sad = 0.f;
                for (int j = 0; j < kHpTSize2; ++j)
                    for (int i = 0; i < kHpTSize2; ++i)
                        sad += std::fabs(ref_l0.at(lx0 + i, ly0 + j) -
                                         l0.at(lx0 + ox2 + i, ly0 + oy2 + j));
                // Halide: dist = i32(sum)/256 (integer), then float math.
                const int dist = (int)(sad * kHpValueScale / 256.f);
                const f32 norm_dist =
                    std::max(1.f, (f32)dist / factor - (f32)min_dist / factor);
                const f32 w = (norm_dist > (f32)(max_dist - min_dist)) ? 0.f : 1.f / norm_dist;
                wt[(size_t)wy * wnx + wx] = w;
                wsum[(size_t)wy * wnx + wx] += w;
            }
        });

        // accumulate the warped frame into the 4 parity planes (per-pixel:
        // each thread owns its row, no races)
        parallel_rows(H, nthreads, [&](int y) {
            const int ty0 = y / kHpTSize2 - 1, ty1 = y / kHpTSize2;
            for (int x = 0; x < W; ++x) {
                const int tx0 = x / kHpTSize2 - 1, tx1 = x / kHpTSize2;
                const int txs[2] = {tx0, tx1};
                const int tys[2] = {ty0, ty1};
                for (int a = 0; a < 2; ++a) {
                    for (int b = 0; b < 2; ++b) {
                        const int tx = txs[a], ty = tys[b];
                        const f32 w = wt[(size_t)(ty + 1) * wnx + (tx + 1)];
                        if (w == 0.f) continue;
                        const int ai = a0.idx(tx, ty);
                        // merge_temporal's pixel sum uses the UNCLAMPED offset
                        const f32 v = comp.at(x + a0.dx[ai], y + a0.dy[ai]);
                        const int p = (tx & 1) | ((ty & 1) << 1);
                        S[p][(size_t)y * W + x] += w * v;
                    }
                }
            }
        });
    }

    // merge_spatial: 4-tile raised-cosine blend of the normalized tiles.
    Image finish() {
        Image out(H, W, 1);
        parallel_rows(H, nthreads, [&](int y) {
            const int ty0 = y / kHpTSize2 - 1, ty1 = y / kHpTSize2;
            const int iy0 = y % kHpTSize2 + kHpTSize2, iy1 = y % kHpTSize2;
            for (int x = 0; x < W; ++x) {
                const int tx0 = x / kHpTSize2 - 1, tx1 = x / kHpTSize2;
                const int ix0 = x % kHpTSize2 + kHpTSize2, ix1 = x % kHpTSize2;
                const int txs[2] = {tx0, tx1};
                const int ixs[2] = {ix0, ix1};
                const int tys[2] = {ty0, ty1};
                const int iys[2] = {iy0, iy1};
                f32 acc = 0.f;
                for (int b = 0; b < 2; ++b) {
                    for (int a = 0; a < 2; ++a) {
                        const int tx = txs[a], ty = tys[b];
                        const int p = (tx & 1) | ((ty & 1) << 1);
                        const f32 s = S[p][(size_t)y * W + x];
                        const f32 tw = wsum[wsum_idx(tx, ty)];
                        acc += window[ixs[a]] * window[iys[b]] * (s / tw);
                    }
                }
                out.at(y, x, 0) = acc;
            }
        });
        return out;
    }
};

// Bilinear demosaic of the merged mosaic -> RGB (the repo's finish.cpp is
// not ported; this only exists so the app's RGB DNG path has pixels).
Image hdrplus_demosaic_bilinear(const Image& mosaic, const Config& cfg,
                                int num_threads) {
    const int H = mosaic.h, W = mosaic.w;
    Image out(H, W, 3);
    auto cfa_at = [&](int y, int x) { return (int)cfg.cfa.p[y & 1][x & 1]; };
    auto m = [&](int y, int x) {
        return mosaic.at(hp_mirror(y, H), hp_mirror(x, W), 0);
    };
    parallel_rows(H, num_threads, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const int c = cfa_at(y, x);
            f32 rgb[3];
            const f32 v = mosaic.at(y, x, 0);
            if (c == 1) {  // green site: R/B from the axis that carries them
                rgb[1] = v;
                const int ch = cfa_at(y, x + 1);      // horizontal neighbour colour
                const int cv = cfa_at(y + 1, x);      // vertical neighbour colour
                const f32 hmean = 0.5f * (m(y, x - 1) + m(y, x + 1));
                const f32 vmean = 0.5f * (m(y - 1, x) + m(y + 1, x));
                rgb[ch] = hmean;
                rgb[cv] = vmean;
            } else {       // red or blue site
                rgb[c] = v;
                rgb[1] = 0.25f * (m(y - 1, x) + m(y + 1, x) + m(y, x - 1) + m(y, x + 1));
                rgb[2 - c] = 0.25f * (m(y - 1, x - 1) + m(y - 1, x + 1) +
                                      m(y + 1, x - 1) + m(y + 1, x + 1));
            }
            out.at(y, x, 0) = rgb[0];
            out.at(y, x, 1) = rgb[1];
            out.at(y, x, 2) = rgb[2];
        }
    });
    return out;
}

Image process_burst_hdrplus(const std::vector<Image>& burst, const Config& cfg,
                            const ProgressFn& progress) {
    if (burst.size() < 2) return Image();
    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };
    const int n = (int)burst.size();
    report("HDR+: reference pyramid", 0.03f);
    HdrPlusCpuState st;
    if (!st.init(burst[0], cfg)) return Image();
    for (int k = 1; k < n; ++k) {
        report("HDR+: frame " + std::to_string(k + 1), 0.05f + 0.75f * (k - 1) / std::max(1, n - 1));
        st.add_frame(burst[k]);
    }
    report("HDR+: spatial blend", 0.85f);
    Image mosaic = st.finish();
    report("HDR+: demosaic", 0.92f);
    return hdrplus_demosaic_bilinear(mosaic, cfg, cfg.num_threads);
}

Image process_burst_loader_to_dng_hdrplus(int frame_count, const RawFrameLoaderFn& loader,
                                          const Config& cfg, const std::string& dng_path,
                                          const ProgressFn& progress, int maxPreviewDim,
                                          Rgb16Sink* rgb16_sink) {
    if (rgb16_sink) { rgb16_sink->w = 0; rgb16_sink->h = 0; rgb16_sink->rgb.clear(); }
    if (frame_count < 2 || !loader) return Image();
    Config work = cfg;
    work.burst_frame_count = frame_count;
    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };

    report("HDR+: loading reference", 0.02f);
    Image ref = loader(0, work, true, 0, 0);
    if (ref.h <= 0 || ref.w <= 0) return Image();

    // GPU first; any failure falls back to one clean CPU pass (frames are
    // reloaded through the caller's loader, which is re-invokable on both
    // the file-path route and the app's buffer route). Guarded at compile
    // time because this file, unlike pipeline_paths.cpp, is also linked into
    // the desktop harness builds where metal_gpu.mm does not exist.
    Image mosaic;
#if defined(__APPLE__)
    bool gpu = hdrplus_metal_begin(ref, work);
    if (gpu) {
        for (int k = 1; k < frame_count && gpu; ++k) {
            report("HDR+ (GPU): frame " + std::to_string(k + 1),
                   0.05f + 0.70f * (k - 1) / std::max(1, frame_count - 1));
            Image comp = loader(k, work, false, ref.h, ref.w);
            gpu = comp.h == ref.h && comp.w == ref.w && hdrplus_metal_add_frame(comp);
        }
        if (gpu) {
            report("HDR+ (GPU): finalize", 0.80f);
            gpu = hdrplus_metal_finish(mosaic);
        }
        if (!gpu) hdrplus_metal_abort();
    }
#else
    const bool gpu = false;
#endif
    if (!gpu) {
        report("HDR+ (CPU): reference pyramid", 0.04f);
        HdrPlusCpuState st;
        if (!st.init(ref, work)) return Image();
        for (int k = 1; k < frame_count; ++k) {
            report("HDR+ (CPU): frame " + std::to_string(k + 1),
                   0.05f + 0.70f * (k - 1) / std::max(1, frame_count - 1));
            Image comp = loader(k, work, false, ref.h, ref.w);
            if (comp.h != ref.h || comp.w != ref.w) continue;
            st.add_frame(comp);
        }
        report("HDR+ (CPU): spatial blend", 0.80f);
        mosaic = st.finish();
    }
    if (mosaic.h <= 0) return Image();
    ref = Image();  // release the reference before the RGB allocations below

    report("HDR+: demosaic", 0.86f);
    Image rgb = hdrplus_demosaic_bilinear(mosaic, work, work.num_threads);
    mosaic = Image();

    // Encode + stream to DNG: same conventions as process_burst_to_dng
    // (bake_srgb / prewhitened / unwhiten), at INPUT resolution.
    const int H = rgb.h, W = rgb.w, nch = 3;
    DngStreamWriter writer;
    if (!writer.open(dng_path, W, H, "HandheldSR-HDRPlus", work.orientation,
                     work.has_color_matrix ? work.color_matrix : nullptr,
                     work.bayer_mode ? work.white_balance : nullptr,
                     work.bake_srgb, "HandheldSR",
                     work.has_cam_to_srgb ? work.cam_to_srgb : nullptr,
                     work.raw_prewhitened && !dng_unwhiten_active(work, nch),
                     work.dng_lossless_jpeg)) {
        report("Error: cannot open output DNG", 1.0f);
        return Image();
    }

    const float pscale = std::min(1.f, (float)maxPreviewDim / std::max(H, W));
    const int ph = std::max(1, (int)(H * pscale));
    const int pw = std::max(1, (int)(W * pscale));
    Image preview(ph, pw, 3);
    if (rgb16_sink) {
        rgb16_sink->w = W;
        rgb16_sink->h = H;
        rgb16_sink->rgb.reserve((size_t)W * H * 3);
    }

    auto to_srgb = [](f32 v) {
        v = clampf(v, 0.f, 1.f);
        return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
    };
    float sgw[3];
    dng_unwhiten_gains(work, nch, sgw);

    std::vector<uint16_t> row16((size_t)W * 3);
    for (int y = 0; y < H; ++y) {
        const int py = std::min(ph - 1, (int)(y * pscale));
        for (int x = 0; x < W; ++x) {
            f32 outc[3];
            if (work.bake_srgb) {
                const f32 g0 = work.raw_prewhitened ? 1.f : work.white_balance[0];
                const f32 g1 = work.raw_prewhitened ? 1.f : work.white_balance[1];
                const f32 g2 = work.raw_prewhitened ? 1.f : work.white_balance[2];
                const f32 wr = rgb.at(y, x, 0) * g0;
                const f32 wg = rgb.at(y, x, 1) * g1;
                const f32 wb = rgb.at(y, x, 2) * g2;
                const f32* mm = work.cam_to_srgb;
                outc[0] = mm[0] * wr + mm[1] * wg + mm[2] * wb;
                outc[1] = mm[3] * wr + mm[4] * wg + mm[5] * wb;
                outc[2] = mm[6] * wr + mm[7] * wg + mm[8] * wb;
            } else {
                outc[0] = rgb.at(y, x, 0);
                outc[1] = rgb.at(y, x, 1);
                outc[2] = rgb.at(y, x, 2);
            }
            const int px = std::min(pw - 1, (int)(x * pscale));
            for (int k = 0; k < 3; ++k) {
                const f32 v = work.bake_srgb ? to_srgb(outc[k]) : clampf(outc[k], 0.f, 1.f);
                const f32 vs = work.bake_srgb ? v : clampf(outc[k] * sgw[k], 0.f, 1.f);
                row16[(size_t)x * 3 + k] = (uint16_t)(vs * 65535.f + 0.5f);
                preview.at(py, px, k) = v;
            }
        }
        writer.write_rows(row16.data(), 1);
        if (rgb16_sink)
            rgb16_sink->rgb.insert(rgb16_sink->rgb.end(), row16.begin(), row16.end());
        if ((y & 255) == 0)
            report("HDR+: writing DNG", 0.90f + 0.09f * (float)y / H);
    }
    writer.close();
    report("Done", 1.0f);
    return preview;
}

} // namespace hhsr
