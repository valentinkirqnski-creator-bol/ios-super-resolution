// Metal runtime for grey-FFT + L2 BM (Apple / iOS only).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_gpu.h"
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hhsr {
namespace {

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

struct MetalCtx {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipes;
    bool ok = false;

    id<MTLComputePipelineState> pipe(const char* name) {
        auto it = pipes.find(name);
        if (it != pipes.end()) return it->second;
        if (!library) return nil;
        NSError* err = nil;
        id<MTLFunction> fn = [library newFunctionWithName:@(name)];
        if (!fn) return nil;
        id<MTLComputePipelineState> p =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!p) return nil;
        pipes[name] = p;
        return p;
    }
};

static MetalCtx& ctx() {
    static MetalCtx c;
    static std::once_flag once;
    std::call_once(once, [] {
        c.device = MTLCreateSystemDefaultDevice();
        if (!c.device) return;
        c.queue = [c.device newCommandQueue];
        if (!c.queue) return;
        NSError* err = nil;
        // Prefer default metallib compiled from HHSRKernels.metal in the app bundle.
        NSString* path = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
        if (path) {
            c.library = [c.device newLibraryWithFile:path error:&err];
        }
        if (!c.library) {
            c.library = [c.device newDefaultLibrary];
        }
        if (!c.library) return;
        // Touch critical pipelines
        const char* need[] = {
            "fft1d_pow2_cpp", "fft_scale_inv", "make_chirp",
            "bluestein_pack_A", "bluestein_clear_B", "bluestein_fill_B", "bluestein_extract",
            "cbuf_mul_broadcast_B", "pack_rows_real", "transpose_c",
            "fftshift2d_c", "zero_fft_borders", "extract_real",
            "l2_pack_tiles", "l2_conj_mul", "l2_argmin", "fftshift2d_real",
            "pack_tile_rows", "take_rfft_half", "write_rfft_cols_from_half",
            "write_half_from_cols", "expand_half_to_full_rows", "extract_real_tiles",
            nullptr};
        for (int i = 0; need[i]; ++i) {
            if (!c.pipe(need[i])) return;
        }
        c.ok = true;
    });
    return c;
}

static id<MTLBuffer> buf(const void* data, size_t bytes, MTLResourceOptions opt =
                             MTLResourceStorageModeShared) {
    auto& c = ctx();
    if (!data) return [c.device newBufferWithLength:bytes options:opt];
    return [c.device newBufferWithBytes:data length:bytes options:opt];
}

static void dispatch2(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                      NSUInteger w, NSUInteger h) {
    if (w == 0 || h == 0) return;
    [enc setComputePipelineState:p];
    MTLSize tg = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(w, h, 1);
    // threadgroup aligned
    NSUInteger tw = p.threadExecutionWidth;
    NSUInteger th = p.maxTotalThreadsPerThreadgroup / tw;
    if (th < 1) th = 1;
    tg = MTLSizeMake(tw, th, 1);
    if (@available(iOS 11.0, *)) {
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    } else {
        MTLSize groups = MTLSizeMake((w + tw - 1) / tw, (h + th - 1) / th, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
    }
}

static void dispatch1(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                      NSUInteger n) {
    if (n == 0) return;
    [enc setComputePipelineState:p];
    NSUInteger tw = p.threadExecutionWidth;
    MTLSize tg = MTLSizeMake(tw, 1, 1);
    if (@available(iOS 11.0, *)) {
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:tg];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake((n + tw - 1) / tw, 1, 1)
            threadsPerThreadgroup:tg];
    }
}

// C++ fft1d_pow2_inplace_ref — unscaled. Optional apply_inv_scale matches fft1d().
static bool fft1d_pow2_gpu(id<MTLBuffer> data, uint32_t n, uint32_t stride,
                           uint32_t batch, bool inverse, bool apply_inv_scale,
                           id<MTLCommandBuffer> cmd) {
    if (n <= 1) return true;
    if ((n & (n - 1)) != 0) return false;
    auto& c = ctx();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    uint32_t n_u = n, str = stride, bat = batch;
    int inv = inverse ? 1 : 0;

    [enc setComputePipelineState:c.pipe("fft1d_pow2_cpp")];
    [enc setBuffer:data offset:0 atIndex:0];
    [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
    [enc setBytes:&str length:sizeof(str) atIndex:2];
    [enc setBytes:&bat length:sizeof(bat) atIndex:3];
    [enc setBytes:&inv length:sizeof(inv) atIndex:4];
    NSUInteger tw = c.pipe("fft1d_pow2_cpp").threadExecutionWidth;
    if (@available(iOS 11.0, *)) {
        [enc dispatchThreads:MTLSizeMake(batch, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake((batch + tw - 1) / tw, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    }

    if (inverse && apply_inv_scale) {
        [enc setBuffer:data offset:0 atIndex:0];
        [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
        [enc setBytes:&str length:sizeof(str) atIndex:2];
        [enc setBytes:&bat length:sizeof(bat) atIndex:3];
        dispatch2(enc, c.pipe("fft_scale_inv"), n, batch);
    }
    [enc endEncoding];
    return true;
}

// C++ fft1d_bluestein exactly: inner pow2 unscaled, then A/=m, outer fft1d /=n if inv.
static bool fft1d_bluestein_gpu(id<MTLBuffer> data, uint32_t n, uint32_t stride,
                                uint32_t batch, bool inverse,
                                id<MTLCommandBuffer> cmd) {
    if (n <= 1) return true;
    if ((n & (n - 1)) == 0)
        return fft1d_pow2_gpu(data, n, stride, batch, inverse, /*scale*/true, cmd);

    auto& c = ctx();
    const uint32_t m = (uint32_t)next_pow2(2 * (int)n - 1);
    const size_t chirp_bytes = sizeof(float) * 2 * n;
    const size_t B_bytes = sizeof(float) * 2 * m;
    const size_t A_bytes = sizeof(float) * 2 * m * batch;

    id<MTLBuffer> chirp = buf(nullptr, chirp_bytes);
    id<MTLBuffer> B = buf(nullptr, B_bytes);
    id<MTLBuffer> A = buf(nullptr, A_bytes);

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t n_u = n, m_u = m, str = stride, bat = batch;
    int inv = inverse ? 1 : 0;

    [enc setBuffer:chirp offset:0 atIndex:0];
    [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
    [enc setBytes:&inv length:sizeof(inv) atIndex:2];
    dispatch1(enc, c.pipe("make_chirp"), n);

    [enc setBuffer:B offset:0 atIndex:0];
    [enc setBytes:&m_u length:sizeof(m_u) atIndex:1];
    dispatch1(enc, c.pipe("bluestein_clear_B"), m);

    [enc setBuffer:B offset:0 atIndex:0];
    [enc setBuffer:chirp offset:0 atIndex:1];
    [enc setBytes:&n_u length:sizeof(n_u) atIndex:2];
    [enc setBytes:&m_u length:sizeof(m_u) atIndex:3];
    dispatch1(enc, c.pipe("bluestein_fill_B"), n);

    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:data offset:0 atIndex:1];
    [enc setBuffer:chirp offset:0 atIndex:2];
    [enc setBytes:&n_u length:sizeof(n_u) atIndex:3];
    [enc setBytes:&m_u length:sizeof(m_u) atIndex:4];
    [enc setBytes:&str length:sizeof(str) atIndex:5];
    [enc setBytes:&bat length:sizeof(bat) atIndex:6];
    dispatch2(enc, c.pipe("bluestein_pack_A"), m, batch);
    [enc endEncoding];

    // Inner pow2: no 1/m scale (matches C++ fft1d_pow2_inplace_ref)
    if (!fft1d_pow2_gpu(B, m, m, 1, false, false, cmd)) return false;
    if (!fft1d_pow2_gpu(A, m, m, batch, false, false, cmd)) return false;

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBytes:&m_u length:sizeof(m_u) atIndex:2];
    [enc setBytes:&bat length:sizeof(bat) atIndex:3];
    dispatch2(enc, c.pipe("cbuf_mul_broadcast_B"), m, batch);
    [enc endEncoding];

    // Inverse pow2 unscaled, then explicit /= m (C++ bluestein)
    if (!fft1d_pow2_gpu(A, m, m, batch, true, false, cmd)) return false;
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBytes:&m_u length:sizeof(m_u) atIndex:1];
    uint32_t a_stride = m_u;
    [enc setBytes:&a_stride length:sizeof(a_stride) atIndex:2];
    [enc setBytes:&bat length:sizeof(bat) atIndex:3];
    dispatch2(enc, c.pipe("fft_scale_inv"), m, batch);

    [enc setBuffer:data offset:0 atIndex:0];
    [enc setBuffer:A offset:0 atIndex:1];
    [enc setBuffer:chirp offset:0 atIndex:2];
    [enc setBytes:&n_u length:sizeof(n_u) atIndex:3];
    [enc setBytes:&m_u length:sizeof(m_u) atIndex:4];
    [enc setBytes:&str length:sizeof(str) atIndex:5];
    [enc setBytes:&bat length:sizeof(bat) atIndex:6];
    dispatch2(enc, c.pipe("bluestein_extract"), n, batch);
    [enc endEncoding];

    // Outer fft1d inverse normalize by n
    if (inverse) {
        enc = [cmd computeCommandEncoder];
        [enc setBuffer:data offset:0 atIndex:0];
        [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
        [enc setBytes:&str length:sizeof(str) atIndex:2];
        [enc setBytes:&bat length:sizeof(bat) atIndex:3];
        dispatch2(enc, c.pipe("fft_scale_inv"), n, batch);
        [enc endEncoding];
    }
    return true;
}

static bool fft1d_gpu(id<MTLBuffer> data, uint32_t n, uint32_t stride, uint32_t batch,
                      bool inverse, id<MTLCommandBuffer> cmd) {
    if ((n & (n - 1)) == 0)
        return fft1d_pow2_gpu(data, n, stride, batch, inverse, /*scale*/true, cmd);
    return fft1d_bluestein_gpu(data, n, stride, batch, inverse, cmd);
}

// Complex HxW buffer: row FFTs then col FFTs (via transpose).
static bool fft2d_gpu(id<MTLBuffer> cbuf, id<MTLBuffer> scratch, uint32_t h, uint32_t w,
                      bool inverse, id<MTLCommandBuffer> cmd) {
    auto& c = ctx();
    // Rows: batch=h, n=w, stride=w
    if (!fft1d_gpu(cbuf, w, w, h, inverse, cmd)) return false;

    // Transpose to scratch [w][h]
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:cbuf offset:0 atIndex:1];
    [enc setBytes:&h length:sizeof(h) atIndex:2];
    [enc setBytes:&w length:sizeof(w) atIndex:3];
    dispatch2(enc, c.pipe("transpose_c"), w, h);
    [enc endEncoding];

    // Cols as rows of transposed: batch=w, n=h, stride=h
    if (!fft1d_gpu(scratch, h, h, w, inverse, cmd)) return false;

    // Transpose back
    enc = [cmd computeCommandEncoder];
    // transpose scratch (w,h) -> cbuf (h,w): swap dims
    uint32_t th = w, tw = h;
    [enc setBuffer:cbuf offset:0 atIndex:0];
    [enc setBuffer:scratch offset:0 atIndex:1];
    [enc setBytes:&th length:sizeof(th) atIndex:2];
    [enc setBytes:&tw length:sizeof(tw) atIndex:3];
    dispatch2(enc, c.pipe("transpose_c"), tw, th);
    [enc endEncoding];
    return true;
}

} // namespace

bool metal_gpu_init() { return ctx().ok; }

Image compute_grey_fft_metal(const Image& raw) {
    if (!metal_gpu_init() || raw.h <= 0 || raw.w <= 0) return Image();
    auto& c = ctx();
    const uint32_t h = (uint32_t)raw.h, w = (uint32_t)raw.w;
    const size_t n = (size_t)h * w;
    const size_t cbytes = n * sizeof(float) * 2;

    id<MTLBuffer> real_in = buf(raw.data.data(), n * sizeof(float));
    id<MTLBuffer> c0 = buf(nullptr, cbytes);
    id<MTLBuffer> c1 = buf(nullptr, cbytes);

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:c0 offset:0 atIndex:0];
    [enc setBuffer:real_in offset:0 atIndex:1];
    [enc setBytes:&h length:sizeof(h) atIndex:2];
    [enc setBytes:&w length:sizeof(w) atIndex:3];
    dispatch2(enc, c.pipe("pack_rows_real"), w, h);
    [enc endEncoding];

    if (!fft2d_gpu(c0, c1, h, w, false, cmd)) return Image();

    // fftshift c0 -> c1
    enc = [cmd computeCommandEncoder];
    int inv = 0;
    [enc setBuffer:c1 offset:0 atIndex:0];
    [enc setBuffer:c0 offset:0 atIndex:1];
    [enc setBytes:&h length:sizeof(h) atIndex:2];
    [enc setBytes:&w length:sizeof(w) atIndex:3];
    [enc setBytes:&inv length:sizeof(inv) atIndex:4];
    dispatch2(enc, c.pipe("fftshift2d_c"), w, h);

    [enc setBuffer:c1 offset:0 atIndex:0];
    [enc setBytes:&h length:sizeof(h) atIndex:1];
    [enc setBytes:&w length:sizeof(w) atIndex:2];
    dispatch2(enc, c.pipe("zero_fft_borders"), w, h);

    inv = 1;
    [enc setBuffer:c0 offset:0 atIndex:0];
    [enc setBuffer:c1 offset:0 atIndex:1];
    [enc setBytes:&h length:sizeof(h) atIndex:2];
    [enc setBytes:&w length:sizeof(w) atIndex:3];
    [enc setBytes:&inv length:sizeof(inv) atIndex:4];
    dispatch2(enc, c.pipe("fftshift2d_c"), w, h);
    [enc endEncoding];

    if (!fft2d_gpu(c0, c1, h, w, true, cmd)) return Image();

    id<MTLBuffer> real_out = buf(nullptr, n * sizeof(float));
    enc = [cmd computeCommandEncoder];
    uint32_t count = (uint32_t)n;
    [enc setBuffer:real_out offset:0 atIndex:0];
    [enc setBuffer:c0 offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:2];
    dispatch1(enc, c.pipe("extract_real"), n);
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return Image();

    Image grey((int)h, (int)w, 1);
    memcpy(grey.data.data(), [real_out contents], n * sizeof(float));
    return grey;
}

bool block_match_level_L2_metal(const Image& ref, const Image& moving,
                                int tile_size, int search_radius,
                                FlowField& flow) {
    if (!metal_gpu_init()) return false;
    auto& c = ctx();
    const int ny = flow.ny, nx = flow.nx;
    if (ny <= 0 || nx <= 0) return true;
    const int ts = tile_size, R = search_radius;
    const int N = 2 * R + ts;
    const int wh = N / 2 + 1;
    const uint32_t ntiles = (uint32_t)(ny * nx);
    const size_t tile_bytes = (size_t)N * N * sizeof(float);
    const size_t all_tiles = tile_bytes * ntiles;
    const size_t half_c = (size_t)ntiles * N * wh * sizeof(float) * 2;
    const size_t full_c = (size_t)ntiles * N * N * sizeof(float) * 2;
    const size_t row_batch = (size_t)ntiles * N; // rows for row-FFT

    struct L2Params {
        uint32_t ny, nx;
        int ts, R, N;
        int ref_h, ref_w, mov_h, mov_w;
    } P{(uint32_t)ny, (uint32_t)nx, ts, R, N, ref.h, ref.w, moving.h, moving.w};

    id<MTLBuffer> ref_img = buf(ref.data.data(), ref.data.size() * sizeof(float));
    id<MTLBuffer> mov_img = buf(moving.data.data(), moving.data.size() * sizeof(float));
    id<MTLBuffer> flow_b = buf(flow.flow.data(), flow.flow.size() * sizeof(float));
    id<MTLBuffer> ref_pad = buf(nullptr, all_tiles);
    id<MTLBuffer> mov_patch = buf(nullptr, all_tiles);

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:ref_pad offset:0 atIndex:0];
    [enc setBuffer:mov_patch offset:0 atIndex:1];
    [enc setBuffer:ref_img offset:0 atIndex:2];
    [enc setBuffer:mov_img offset:0 atIndex:3];
    [enc setBuffer:flow_b offset:0 atIndex:4];
    [enc setBytes:&P length:sizeof(P) atIndex:5];
    dispatch1(enc, c.pipe("l2_pack_tiles"), ntiles);
    [enc endEncoding];

    // --- rfft2 ref tiles ---
    id<MTLBuffer> rows = buf(nullptr, row_batch * N * sizeof(float) * 2);
    enc = [cmd computeCommandEncoder];
    uint32_t Nu = (uint32_t)N, nt = ntiles;
    [enc setBuffer:rows offset:0 atIndex:0];
    [enc setBuffer:ref_pad offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&nt length:sizeof(nt) atIndex:3];
    dispatch2(enc, c.pipe("pack_tile_rows"), N, row_batch);
    [enc endEncoding];

    if (!fft1d_gpu(rows, Nu, Nu, (uint32_t)row_batch, false, cmd)) return false;

    id<MTLBuffer> F_ref = buf(nullptr, half_c);
    enc = [cmd computeCommandEncoder];
    uint32_t wh_u = (uint32_t)wh, nrows = (uint32_t)row_batch;
    [enc setBuffer:F_ref offset:0 atIndex:0];
    [enc setBuffer:rows offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nrows length:sizeof(nrows) atIndex:4];
    dispatch2(enc, c.pipe("take_rfft_half"), wh, row_batch);
    [enc endEncoding];

    id<MTLBuffer> cols = buf(nullptr, (size_t)ntiles * wh * N * sizeof(float) * 2);
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:cols offset:0 atIndex:0];
    [enc setBuffer:F_ref offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_rfft_cols_from_half"), N * wh, ntiles);
    [enc endEncoding];

    if (!fft1d_gpu(cols, Nu, Nu, ntiles * wh_u, false, cmd)) return false;

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:F_ref offset:0 atIndex:0];
    [enc setBuffer:cols offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_half_from_cols"), N * wh, ntiles);
    [enc endEncoding];

    // --- rfft2 mov tiles (reuse rows/cols) ---
    id<MTLBuffer> F_mov = buf(nullptr, half_c);
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:rows offset:0 atIndex:0];
    [enc setBuffer:mov_patch offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&nt length:sizeof(nt) atIndex:3];
    dispatch2(enc, c.pipe("pack_tile_rows"), N, row_batch);
    [enc endEncoding];
    if (!fft1d_gpu(rows, Nu, Nu, (uint32_t)row_batch, false, cmd)) return false;
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:F_mov offset:0 atIndex:0];
    [enc setBuffer:rows offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nrows length:sizeof(nrows) atIndex:4];
    dispatch2(enc, c.pipe("take_rfft_half"), wh, row_batch);
    [enc endEncoding];
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:cols offset:0 atIndex:0];
    [enc setBuffer:F_mov offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_rfft_cols_from_half"), N * wh, ntiles);
    [enc endEncoding];
    if (!fft1d_gpu(cols, Nu, Nu, ntiles * wh_u, false, cmd)) return false;
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:F_mov offset:0 atIndex:0];
    [enc setBuffer:cols offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_half_from_cols"), N * wh, ntiles);
    [enc endEncoding];

    // conj(F_ref) * F_mov
    enc = [cmd computeCommandEncoder];
    uint32_t half_count = ntiles * (uint32_t)N * wh_u;
    [enc setBuffer:F_ref offset:0 atIndex:0];
    [enc setBuffer:F_mov offset:0 atIndex:1];
    [enc setBytes:&half_count length:sizeof(half_count) atIndex:2];
    dispatch1(enc, c.pipe("l2_conj_mul"), half_count);
    [enc endEncoding];

    // --- irfft2: col IFFT then row IFFT ---
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:cols offset:0 atIndex:0];
    [enc setBuffer:F_ref offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_rfft_cols_from_half"), N * wh, ntiles);
    [enc endEncoding];
    if (!fft1d_gpu(cols, Nu, Nu, ntiles * wh_u, true, cmd)) return false;
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:F_ref offset:0 atIndex:0];
    [enc setBuffer:cols offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_half_from_cols"), N * wh, ntiles);
    [enc endEncoding];

    id<MTLBuffer> full = buf(nullptr, full_c);
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:full offset:0 atIndex:0];
    [enc setBuffer:F_ref offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("expand_half_to_full_rows"), N * N, ntiles);
    [enc endEncoding];

    // Row IFFT: treat as batch ntiles*N length-N (data already contiguous as [ntiles*N][N])
    if (!fft1d_gpu(full, Nu, Nu, (uint32_t)row_batch, true, cmd)) return false;

    id<MTLBuffer> corr = buf(nullptr, all_tiles);
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:corr offset:0 atIndex:0];
    [enc setBuffer:full offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&nt length:sizeof(nt) atIndex:3];
    dispatch2(enc, c.pipe("extract_real_tiles"), N * N, ntiles);
    [enc endEncoding];

    id<MTLBuffer> corr_shift = buf(nullptr, all_tiles);
    enc = [cmd computeCommandEncoder];
    uint32_t Nh = (uint32_t)N, Nw = (uint32_t)N;
    [enc setBuffer:corr_shift offset:0 atIndex:0];
    [enc setBuffer:corr offset:0 atIndex:1];
    [enc setBytes:&Nh length:sizeof(Nh) atIndex:2];
    [enc setBytes:&Nw length:sizeof(Nw) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("fftshift2d_real"), N * N, ntiles);

    [enc setBuffer:flow_b offset:0 atIndex:0];
    [enc setBuffer:corr_shift offset:0 atIndex:1];
    [enc setBuffer:mov_patch offset:0 atIndex:2];
    [enc setBytes:&P length:sizeof(P) atIndex:3];
    dispatch1(enc, c.pipe("l2_argmin"), ntiles);
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;

    memcpy(flow.flow.data(), [flow_b contents], flow.flow.size() * sizeof(float));
    return true;
}

} // namespace hhsr
