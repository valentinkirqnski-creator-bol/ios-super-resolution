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
    // ---- Sharpness control ----
    // k_detail is the master "softness" knob: it scales the merge-kernel width.
    //   LARGER k_detail  -> wider kernels -> SOFTER  output (less sharpening)
    //   SMALLER k_detail -> tighter kernels -> SHARPER output (more detail/aliasing)
    // Reference default is 0.33 -> sub-pixel kernels that ALIAS (harsh, crunchy,
    // "over-sharpened"). 0.42 brings kernels to ~1px = correct reconstruction:
    // removes aliasing/halos WITHOUT blurring real merged detail.
    // Detail is only lost above ~0.70. Try 0.55 for softer, 0.36 for crisper.
    float k_detail  = 0.42f;
    float k_denoise = 3.0f;
    float D_th      = 0.005f;
    float D_tr      = 0.014f;
    // k_stretch elongates kernels along edges (edge over-sharpening). Lowering
    // it from 4.0 -> 3.0 reduces harsh edge crispening.
    float k_stretch = 3.0f;
    float k_shrink  = 2.0f;

    CFA cfa;
    float white_balance[3] = {1.f, 1.f, 1.f}; // green-normalized camera WB gains

    // Color / orientation metadata read from the reference DNG.
    int   orientation = 1;                 // EXIF orientation (1=normal,6/8=90deg)
    bool  has_color_matrix = false;        // true once populated from the file
    float color_matrix[9] = {1,0,0, 0,1,0, 0,0,1}; // ColorMatrix1: XYZ(D65) -> camera
    // camera-RGB -> linear sRGB (LibRaw rgb_cam). Used to bake display-ready color.
    bool  has_cam_to_srgb = false;
    float cam_to_srgb[9] = {1,0,0, 0,1,0, 0,0,1};
    // When true the output is a fully-developed sRGB image (looks correct in any
    // viewer). When false a camera-native linear "raw" DNG is written instead.
    bool  bake_srgb = true;

    int num_threads = 0;      // 0 => hardware_concurrency
    bool use_gpu = false;     // opt-in Vulkan compute merge (experimental)
};

inline f32 clampf(f32 v, f32 lo, f32 hi) { return v < lo ? lo : (v > hi ? hi : v); }

} // namespace hhsr
