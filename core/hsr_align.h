#pragma once
//
// The pipeline's alignment: a plain C++ port of the Handheld-Multi-Frame-Super-Resolution `alignment` module
// (Halide AOT kernels: downsample_generator, gradients, block_matching, ica, flow_transfer, flow_median, and the
// driver alignment.cpp), plus the package's grayscale (utils_image.compute_grey_images, DECIMATING), its SNR
// estimate (utils_image.estimate_image_snr) and its SNR-based level settings (params.update_snr_config /
// alignment_tile_sizes, config.AlignmentConfig defaults).
//
// Kernel semantics follow the Halide definitions exactly: repeated edges, Halide::round = round half to EVEN,
// strict-less-than "first minimum" tie-breaking in the search orders the generators use, and every per-tile sum
// accumulated serially in the RDom order (x fastest).
//
// Output: one vector per 16x16 grey tile with stride 8 (the package's FINAL_TILE_SIZE / FINAL_TILE_STRIDE), 3x3
// median filtered, in grey pixels. The package's merge and robustness read tile floor(grey_x / 8), i.e. raw pixels
// [16k, 16k + 16) use tile k -- exactly a FlowField on a 16-raw-pixel grid (to_raw_flow_field).
//
#include "types.h"
#include "hsr_align_shared.h"
#include <vector>

namespace hhsr {
namespace hsra {

struct Plane {                    // row-major float plane, (x, y) = column, row
    int w = 0, h = 0;
    std::vector<float> d;
    Plane() = default;
    Plane(int w_, int h_) : w(w_), h(h_), d((size_t)w_ * h_, 0.f) {}
    float at(int x, int y) const { return d[(size_t)y * w + x]; }
    float& at(int x, int y) { return d[(size_t)y * w + x]; }
    float edge(int x, int y) const {   // BoundaryConditions::repeat_edge
        x = x < 0 ? 0 : (x >= w ? w - 1 : x);
        y = y < 0 ? 0 : (y >= h ? h - 1 : y);
        return d[(size_t)y * w + x];
    }
};

struct Options {                  // per-level settings, finest to coarsest (config.AlignmentConfig)
    std::vector<int> factors{2, 4, 4, 4};
    std::vector<int> radii{1, 3, 3, 3, 3};
    std::vector<int> l2{0, 1, 1, 1, 1};   // block_matching.metrics: 0 = L1, 1 = L2
    std::vector<int> tiles;               // derived (alignment_tile_sizes)
    int ica_iterations = 3;
    int base_tile = 0;                    // the SNR-selected tile_size (64 / 32 / 16)
    float snr_db = 0.f;
};

// params.update_snr_config (tile_size SNR_BASED) + alignment_tile_sizes. snr_db is clipped to [6, 30].
// force_tile = 16 / 32 / 64 replaces the SNR-selected base tile (Config::sabre_align_tile); 0 = the package's rule.
Options options_for_snr(float snr_db, int force_tile = 0);

// utils_image.estimate_image_snr on the reference raw (normalized, per-CFA-site alpha/beta of the DNG noise
// profile), in dB. Prewhitened raws are un-whitened first (the package reads raw without white balance).
float estimate_image_snr_db(const Image& raw, const Config& cfg);

// utils_image.compute_grey_images(DECIMATING): 2x2 Bayer mean, [0.25 0.5 0.25] separable blur (nearest edges),
// clipped to [0, 1]. Prewhitened raws are un-whitened first.
Plane grey_from_bayer(const Image& raw, const Config& cfg, int threads);
// Per-site gains of the grey's 2x2 mean, [sy * 2 + sx] (1 unless the loader pre-whitened the raw).
void grey_gains(const Config& cfg, float g[4]);

// Plane size of every pyramid level (finest first) for a grey of w x h.
void pyramid_dims(int w, int h, const Options& opt, std::vector<int>& ws, std::vector<int>& hs);
// Flow-transfer parameters of align(): from level `source` (its non-overlapping tile grid src_nx x src_ny) onto
// level source - 1 (plane tw x th), and the final 16/8 grid on level 0 (plane w x h) from level 0's tile grid.
HsraTransferParams level_transfer_params(const Options& opt, size_t source, int src_nx, int src_ny, int tw, int th);
HsraTransferParams final_transfer_params(const Options& opt, int src_nx, int src_ny, int w, int h);
// Grid size a transfer produces.
inline int transfer_cols(const HsraTransferParams& P) { return (P.w - P.tt) / P.stride + 1; }
inline int transfer_rows(const HsraTransferParams& P) { return (P.h - P.tt) / P.stride + 1; }

struct Level {
    Plane img;
    Plane gx, gy, sxx, sxy, syy;  // reference only
};
struct Pyramid {
    bool reference = false;
    Options opt;
    std::vector<Level> levels;
};
Pyramid build_pyramid(const Plane& grey, const Options& opt, bool reference, int threads);

struct Flow {                     // per-tile displacement in current-level pixels
    int nx = 0, ny = 0, tile = 0, stride = 0;
    std::vector<float> dx, dy;
    Flow() = default;
    Flow(int nx_, int ny_, int tile_, int stride_)
        : nx(nx_), ny(ny_), tile(tile_), stride(stride_), dx((size_t)nx_ * ny_, 0.f), dy((size_t)nx_ * ny_, 0.f) {}
};

// alignment.align: coarse-to-fine block matching + ICA, flow transfer, final 16/8 grid, 3x3 median.
// Returns an empty Flow if the pyramids are unusable (too small for their tiles).
Flow align(const Pyramid& ref, const Pyramid& mov, int threads);

// The 16/8 grey grid as a raw-pixel FlowField: one cell per 16 raw px (cell k = raw [16k, 16k + 16), the tile the
// package's merge/robustness read there), displacement x2. Padded to ceil(raw / 16) cells by repeating the edge
// vectors, so every raw pixel has a cell (the package indexes past its grid at the right/bottom border).
FlowField to_raw_flow_field(const Flow& f, int raw_w, int raw_h);

#if defined(__APPLE__)
// The same aligner on Metal (core/hsr_align_metal.mm, kernels core/HsrAlignKernels.metal, arithmetic shared with the
// CPU through core/hsr_align_shared.h). metal_set_reference builds the reference pyramid on the GPU and keeps it for
// the burst; metal_align runs a comparison frame's grey, pyramid and the whole align() on the GPU and returns the
// final 16/8 flow (what align() returns). grey_out (optional) receives the full-resolution grey. false = not run
// (no device / library / memory, or a command buffer failed): the caller uses the CPU functions above.
bool metal_set_reference(const Image& raw, const Config& cfg, const Options& opt, Plane* grey_out);
bool metal_align(const Image& raw, const Config& cfg, Flow& out, Plane* grey_out);
void metal_release();
#endif

// Kernels (exposed for the tests).
Plane downsample(const Plane& in, int factor, int threads);
void gradients(const Plane& in, int tile, Plane& gx, Plane& gy, Plane& sxx, Plane& sxy, Plane& syy, int threads);
Flow block_matching(const Plane& ref, const Plane& mov, const Flow& init, int tile, int radius, bool l2, int threads);
Flow ica(const Level& ref, const Plane& mov, const Flow& init, int iterations, int threads);
Flow transfer_flow(const Flow& src, const Plane& ref, const Plane& mov, int target_tile, int target_stride,
                   float grid_scale_ratio, float flow_scale, float offset_x_px, float offset_y_px, int threads);
Flow transfer_flow(const Flow& src, const Plane& ref, const Plane& mov, const HsraTransferParams& P, int threads);
Flow median_filter(const Flow& src);

}  // namespace hsra
}  // namespace hhsr
