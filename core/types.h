#pragma once
//
// Core data types for the Handheld Multi-Frame Super-Resolution port.
//
// This is a faithful C++ CPU re-implementation of the algorithm from
// Wronski et al. (SIGGRAPH 2019), ported from the Python/Numba reference in
// this repository. The types are intentionally simple (flat row-major float
// buffers) so the same data can later be uploaded to a GPU (Vulkan SSBOs)
// without changing the algorithm code.
//
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace hhsr {

using f32 = float;

// Row-major image / tensor with an arbitrary number of interleaved channels.
struct Image {
    int h = 0;
    int w = 0;
    int c = 1;
    std::vector<f32> data;

    Image() = default;
    Image(int h_, int w_, int c_ = 1) : h(h_), w(w_), c(c_), data((size_t)h_ * w_ * c_, 0.f) {}

    inline f32&       at(int y, int x, int ch = 0)       { return data[((size_t)y * w + x) * c + ch]; }
    inline const f32& at(int y, int x, int ch = 0) const { return data[((size_t)y * w + x) * c + ch]; }

    inline bool inside(int y, int x) const { return y >= 0 && y < h && x >= 0 && x < w; }
    inline size_t size() const { return data.size(); }
};

// Per-tile optical flow field: shape [nTilesY, nTilesX, 2] (dx, dy).
struct FlowField {
    int ny = 0;
    int nx = 0;
    std::vector<f32> flow; // ny*nx*2

    FlowField() = default;
    FlowField(int ny_, int nx_) : ny(ny_), nx(nx_), flow((size_t)ny_ * nx_ * 2, 0.f) {}

    inline f32& dx(int ty, int tx) { return flow[((size_t)ty * nx + tx) * 2 + 0]; }
    inline f32& dy(int ty, int tx) { return flow[((size_t)ty * nx + tx) * 2 + 1]; }
    inline f32 dx(int ty, int tx) const { return flow[((size_t)ty * nx + tx) * 2 + 0]; }
    inline f32 dy(int ty, int tx) const { return flow[((size_t)ty * nx + tx) * 2 + 1]; }
};

// Bilinear sample of a per-tile flow field (reduces blocky merge seams).
inline void sample_flow_bilinear(const FlowField& flow, f32 ty, f32 tx, f32& dx, f32& dy) {
    if (flow.nx <= 0 || flow.ny <= 0) { dx = dy = 0.f; return; }
    tx = clampf(tx, 0.f, (f32)(flow.nx - 1));
    ty = clampf(ty, 0.f, (f32)(flow.ny - 1));
    int x0 = (int)std::floor(tx), y0 = (int)std::floor(ty);
    int x1 = std::min(x0 + 1, flow.nx - 1), y1 = std::min(y0 + 1, flow.ny - 1);
    f32 fx = tx - (f32)x0, fy = ty - (f32)y0;
    f32 w00 = (1.f - fx) * (1.f - fy), w10 = fx * (1.f - fy);
    f32 w01 = (1.f - fx) * fy,         w11 = fx * fy;
    dx = w00 * flow.dx(y0, x0) + w10 * flow.dx(y0, x1)
       + w01 * flow.dx(y1, x0) + w11 * flow.dx(y1, x1);
    dy = w00 * flow.dy(y0, x0) + w10 * flow.dy(y0, x1)
       + w01 * flow.dy(y1, x0) + w11 * flow.dy(y1, x1);
}

// Per-grey-pixel 2x2 covariance field (steerable kernels): [h, w, 4] = xx,xy,yx,yy.
struct CovField {
    int h = 0;
    int w = 0;
    std::vector<f32> cov;

    CovField() = default;
    CovField(int h_, int w_) : h(h_), w(w_), cov((size_t)h_ * w_ * 4, 0.f) {}

    inline f32* at(int y, int x) { return &cov[((size_t)y * w + x) * 4]; }
    inline const f32* at(int y, int x) const { return &cov[((size_t)y * w + x) * 4]; }
};

enum class GreyMethod { Decimate }; // FFT-grey omitted on CPU; decimate is used throughout.
enum class KernelShape { Iso, Steerable };
enum class SelectionLaw { HardThreshold, Linear };

// 2x2 Bayer CFA pattern (indices into {R=0,G=1,B=2}).
struct CFA {
    uint8_t p[2][2] = {{0, 1}, {1, 2}}; // default RGGB
};

// Full pipeline configuration, mirroring configs/default.yaml.
struct Config {
    float scale = 2.0f;
    bool  bayer_mode = true;

    // Noise model: sigma^2 = alpha * I + beta   (already scaled for ISO).
    float alpha = 1.80710882e-4f;
    float beta  = 3.1937599182e-6f;

    // Alignment (coarse-to-fine handled internally).
    std::vector<int> bm_factors      = {1, 2, 4, 4};
    std::vector<int> bm_tile_sizes   = {16, 16, 16, 8};
    std::vector<int> bm_search_radii = {1, 4, 4, 4};
    int  ica_n_iter = 3;

    // Robustness.
    bool  robustness_enabled = true;
    float r_t  = 0.12f;
    float r_s1 = 2.0f;
    float r_s2 = 12.0f;
    float r_Mt = 0.8f;

    // Merge / steerable kernels.
    KernelShape  kernel = KernelShape::Steerable;
    SelectionLaw selection = SelectionLaw::Linear;
    bool  snr_auto_tune = true;
    float k_detail  = 0.42f;
    float k_denoise = 3.5f;
    float D_th      = 0.005f;
    float D_tr      = 0.014f;
    float k_stretch = 4.0f;
    float k_shrink  = 2.0f;
    float aniso_detail_floor = 1.65f;
    float color_saturation = 1.0f;

    CFA cfa;
    float white_balance[3] = {1.f, 1.f, 1.f};

    int   orientation = 1;
    bool  has_color_matrix = false;
    float color_matrix[9] = {1,0,0, 0,1,0, 0,0,1};
    bool  has_cam_to_srgb = false;
    float cam_to_srgb[9] = {1,0,0, 0,1,0, 0,0,1};
    bool  bake_srgb = false;

    std::string camera_make;
    std::string camera_model;

    int num_threads = 0;      // 0 => hardware_concurrency
    bool use_gpu = false;     // opt-in Vulkan compute merge (experimental)
};

inline f32 clampf(f32 v, f32 lo, f32 hi) { return v < lo ? lo : (v > hi ? hi : v); }

} // namespace hhsr
