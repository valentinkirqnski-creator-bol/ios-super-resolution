// Metal runtime for grey-FFT + L2 BM (Apple / iOS only).
// Chunks large batches so full-res (1×) fits in memory / GPU time; 2× crop is smaller.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_gpu.h"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hhsr {
namespace {

// Max 1D FFTs per Bluestein/pow2 dispatch (limits A scratch + GPU watchdog).
static constexpr uint32_t kFftBatchChunk = 48;
// Max L2 tiles processed together (buffers scale with ntiles * N²).
static constexpr uint32_t kL2TileChunk = 256;

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
        NSString* path = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
        if (path) {
            c.library = [c.device newLibraryWithFile:path error:&err];
        }
        if (!c.library) {
            c.library = [c.device newDefaultLibrary];
        }
        if (!c.library) return;
        const char* need[] = {
            "fft1d_pow2_cpp", "fft_scale_inv", "make_chirp",
            "bluestein_pack_A", "bluestein_clear_B", "bluestein_fill_B", "bluestein_extract",
            "cbuf_mul_broadcast_B", "pack_rows_real", "transpose_c",
            "gather_cols", "scatter_cols", "fftshift_swap_x", "fftshift_swap_y",
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

static id<MTLBuffer> buf(const void* data, size_t bytes) {
    auto& c = ctx();
    if (bytes == 0) return nil;
    id<MTLBuffer> b = nil;
    if (!data)
        b = [c.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    else
        b = [c.device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    return b;
}

static void dispatch2(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                      NSUInteger w, NSUInteger h) {
    if (w == 0 || h == 0 || !p) return;
    [enc setComputePipelineState:p];
    NSUInteger tw = p.threadExecutionWidth;
    NSUInteger th = std::max<NSUInteger>(1, p.maxTotalThreadsPerThreadgroup / tw);
    MTLSize tg = MTLSizeMake(tw, th, 1);
    if (@available(iOS 11.0, *)) {
        [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:tg];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake((w + tw - 1) / tw, (h + th - 1) / th, 1)
            threadsPerThreadgroup:tg];
    }
}

static void dispatch1(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                      NSUInteger n) {
    if (n == 0 || !p) return;
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

static bool flush_cmd(__strong id<MTLCommandBuffer>& cmd) {
    auto& c = ctx();
    if (!cmd) return false;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;
    cmd = [c.queue commandBuffer];
    return cmd != nil;
}

static bool fft1d_pow2_gpu_chunk(id<MTLBuffer> data, uint32_t n, uint32_t stride,
                                 uint32_t batch_offset, uint32_t batch,
                                 bool inverse, bool apply_inv_scale,
                                 __strong id<MTLCommandBuffer>& cmd) {
    if (n <= 1 || batch == 0) return true;
    if ((n & (n - 1)) != 0) return false;
    auto& c = ctx();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;

    uint32_t n_u = n, str = stride, bat = batch;
    int inv = inverse ? 1 : 0;
    NSUInteger byte_off = (NSUInteger)batch_offset * stride * sizeof(float) * 2;

    [enc setComputePipelineState:c.pipe("fft1d_pow2_cpp")];
    [enc setBuffer:data offset:byte_off atIndex:0];
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
        [enc setBuffer:data offset:byte_off atIndex:0];
        [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
        [enc setBytes:&str length:sizeof(str) atIndex:2];
        [enc setBytes:&bat length:sizeof(bat) atIndex:3];
        dispatch2(enc, c.pipe("fft_scale_inv"), n, batch);
    }
    [enc endEncoding];
    return flush_cmd(cmd);
}

static bool fft1d_pow2_gpu(id<MTLBuffer> data, uint32_t n, uint32_t stride,
                           uint32_t batch, bool inverse, bool apply_inv_scale,
                           __strong id<MTLCommandBuffer>& cmd) {
    for (uint32_t b0 = 0; b0 < batch; b0 += kFftBatchChunk) {
        uint32_t bc = std::min(kFftBatchChunk, batch - b0);
        if (!fft1d_pow2_gpu_chunk(data, n, stride, b0, bc, inverse, apply_inv_scale, cmd))
            return false;
    }
    return true;
}

static bool fft1d_bluestein_gpu(id<MTLBuffer> data, uint32_t n, uint32_t stride,
                                uint32_t batch, bool inverse,
                                __strong id<MTLCommandBuffer>& cmd) {
    if (n <= 1 || batch == 0) return true;
    if ((n & (n - 1)) == 0)
        return fft1d_pow2_gpu(data, n, stride, batch, inverse, /*scale*/true, cmd);

    auto& c = ctx();
    const uint32_t m = (uint32_t)next_pow2(2 * (int)n - 1);
    id<MTLBuffer> chirp = buf(nullptr, sizeof(float) * 2 * n);
    id<MTLBuffer> B = buf(nullptr, sizeof(float) * 2 * m);
    if (!chirp || !B) return false;

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t n_u = n, m_u = m;
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
    [enc endEncoding];

    if (!fft1d_pow2_gpu(B, m, m, 1, false, false, cmd)) return false;

    id<MTLBuffer> A = buf(nullptr, sizeof(float) * 2 * m * kFftBatchChunk);
    if (!A) return false;

    for (uint32_t b0 = 0; b0 < batch; b0 += kFftBatchChunk) {
        uint32_t bc = std::min(kFftBatchChunk, batch - b0);
        uint32_t str = stride;
        uint32_t bat = bc;
        NSUInteger in_off = (NSUInteger)b0 * stride * sizeof(float) * 2;

        enc = [cmd computeCommandEncoder];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:data offset:in_off atIndex:1];
        [enc setBuffer:chirp offset:0 atIndex:2];
        [enc setBytes:&n_u length:sizeof(n_u) atIndex:3];
        [enc setBytes:&m_u length:sizeof(m_u) atIndex:4];
        [enc setBytes:&str length:sizeof(str) atIndex:5];
        [enc setBytes:&bat length:sizeof(bat) atIndex:6];
        dispatch2(enc, c.pipe("bluestein_pack_A"), m, bc);
        [enc endEncoding];

        if (!fft1d_pow2_gpu(A, m, m, bc, false, false, cmd)) return false;

        enc = [cmd computeCommandEncoder];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:B offset:0 atIndex:1];
        [enc setBytes:&m_u length:sizeof(m_u) atIndex:2];
        [enc setBytes:&bat length:sizeof(bat) atIndex:3];
        dispatch2(enc, c.pipe("cbuf_mul_broadcast_B"), m, bc);
        [enc endEncoding];

        if (!fft1d_pow2_gpu(A, m, m, bc, true, false, cmd)) return false;

        enc = [cmd computeCommandEncoder];
        uint32_t a_stride = m_u;
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBytes:&m_u length:sizeof(m_u) atIndex:1];
        [enc setBytes:&a_stride length:sizeof(a_stride) atIndex:2];
        [enc setBytes:&bat length:sizeof(bat) atIndex:3];
        dispatch2(enc, c.pipe("fft_scale_inv"), m, bc);

        [enc setBuffer:data offset:in_off atIndex:0];
        [enc setBuffer:A offset:0 atIndex:1];
        [enc setBuffer:chirp offset:0 atIndex:2];
        [enc setBytes:&n_u length:sizeof(n_u) atIndex:3];
        [enc setBytes:&m_u length:sizeof(m_u) atIndex:4];
        [enc setBytes:&str length:sizeof(str) atIndex:5];
        [enc setBytes:&bat length:sizeof(bat) atIndex:6];
        dispatch2(enc, c.pipe("bluestein_extract"), n, bc);
        [enc endEncoding];
        if (!flush_cmd(cmd)) return false;
    }

    if (inverse) {
        for (uint32_t b0 = 0; b0 < batch; b0 += kFftBatchChunk) {
            uint32_t bc = std::min(kFftBatchChunk, batch - b0);
            enc = [cmd computeCommandEncoder];
            uint32_t str = stride;
            NSUInteger off = (NSUInteger)b0 * stride * sizeof(float) * 2;
            [enc setBuffer:data offset:off atIndex:0];
            [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
            [enc setBytes:&str length:sizeof(str) atIndex:2];
            [enc setBytes:&bc length:sizeof(bc) atIndex:3];
            dispatch2(enc, c.pipe("fft_scale_inv"), n, bc);
            [enc endEncoding];
            if (!flush_cmd(cmd)) return false;
        }
    }
    return true;
}

static bool fft1d_gpu(id<MTLBuffer> data, uint32_t n, uint32_t stride, uint32_t batch,
                      bool inverse, __strong id<MTLCommandBuffer>& cmd) {
    if ((n & (n - 1)) == 0)
        return fft1d_pow2_gpu(data, n, stride, batch, inverse, /*scale*/true, cmd);
    return fft1d_bluestein_gpu(data, n, stride, batch, inverse, cmd);
}

// 2D FFT with one full-frame complex buffer: row FFTs in place, column FFTs via strips.
static bool fft2d_gpu(id<MTLBuffer> cbuf, id<MTLBuffer> col_scratch, uint32_t h, uint32_t w,
                      bool inverse, __strong id<MTLCommandBuffer>& cmd) {
    auto& c = ctx();
    if (!fft1d_gpu(cbuf, w, w, h, inverse, cmd)) return false;

    for (uint32_t col0 = 0; col0 < w; col0 += kFftBatchChunk) {
        uint32_t ncol = std::min(kFftBatchChunk, w - col0);
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setBuffer:col_scratch offset:0 atIndex:0];
        [enc setBuffer:cbuf offset:0 atIndex:1];
        [enc setBytes:&h length:sizeof(h) atIndex:2];
        [enc setBytes:&w length:sizeof(w) atIndex:3];
        [enc setBytes:&col0 length:sizeof(col0) atIndex:4];
        [enc setBytes:&ncol length:sizeof(ncol) atIndex:5];
        dispatch2(enc, c.pipe("gather_cols"), h, ncol);
        [enc endEncoding];

        if (!fft1d_gpu(col_scratch, h, h, ncol, inverse, cmd)) return false;

        enc = [cmd computeCommandEncoder];
        [enc setBuffer:cbuf offset:0 atIndex:0];
        [enc setBuffer:col_scratch offset:0 atIndex:1];
        [enc setBytes:&h length:sizeof(h) atIndex:2];
        [enc setBytes:&w length:sizeof(w) atIndex:3];
        [enc setBytes:&col0 length:sizeof(col0) atIndex:4];
        [enc setBytes:&ncol length:sizeof(ncol) atIndex:5];
        dispatch2(enc, c.pipe("scatter_cols"), h, ncol);
        [enc endEncoding];
        if (!flush_cmd(cmd)) return false;
    }
    return true;
}

// Run one L2 chunk: tile_base .. tile_base+tile_count-1 (local buffers sized to tile_count).
static bool l2_chunk(id<MTLBuffer> ref_img, id<MTLBuffer> mov_img, id<MTLBuffer> flow_b,
                     int ref_h, int ref_w, int mov_h, int mov_w,
                     int ts, int R, int N, int wh,
                     uint32_t ny, uint32_t nx, uint32_t tile_base, uint32_t tile_count) {
    auto& c = ctx();
    const size_t tile_elems = (size_t)N * N;
    const size_t tile_bytes = tile_elems * sizeof(float);
    const size_t all_tiles = tile_bytes * tile_count;
    const size_t half_c = (size_t)tile_count * N * wh * sizeof(float) * 2;
    const size_t full_c = (size_t)tile_count * tile_elems * sizeof(float) * 2;
    const uint32_t row_batch = tile_count * (uint32_t)N;

    struct L2Params {
        uint32_t ny, nx;
        int ts, R, N;
        int ref_h, ref_w, mov_h, mov_w;
        uint32_t tile_base, tile_count;
    } P{ny, nx, ts, R, N, ref_h, ref_w, mov_h, mov_w, tile_base, tile_count};

    id<MTLBuffer> ref_pad = buf(nullptr, all_tiles);
    id<MTLBuffer> mov_patch = buf(nullptr, all_tiles);
    id<MTLBuffer> rows = buf(nullptr, (size_t)row_batch * N * sizeof(float) * 2);
    id<MTLBuffer> F_ref = buf(nullptr, half_c);
    id<MTLBuffer> F_mov = buf(nullptr, half_c);
    id<MTLBuffer> cols = buf(nullptr, (size_t)tile_count * wh * N * sizeof(float) * 2);
    id<MTLBuffer> full = buf(nullptr, full_c);
    id<MTLBuffer> corr = buf(nullptr, all_tiles);
    id<MTLBuffer> corr_shift = buf(nullptr, all_tiles);
    if (!ref_pad || !mov_patch || !rows || !F_ref || !F_mov || !cols || !full ||
        !corr || !corr_shift)
        return false;

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return false;

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:ref_pad offset:0 atIndex:0];
    [enc setBuffer:mov_patch offset:0 atIndex:1];
    [enc setBuffer:ref_img offset:0 atIndex:2];
    [enc setBuffer:mov_img offset:0 atIndex:3];
    [enc setBuffer:flow_b offset:0 atIndex:4];
    [enc setBytes:&P length:sizeof(P) atIndex:5];
    dispatch1(enc, c.pipe("l2_pack_tiles"), tile_count);
    [enc endEncoding];

    uint32_t Nu = (uint32_t)N, nt = tile_count, wh_u = (uint32_t)wh, nrows = row_batch;

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:rows offset:0 atIndex:0];
    [enc setBuffer:ref_pad offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&nt length:sizeof(nt) atIndex:3];
    dispatch2(enc, c.pipe("pack_tile_rows"), N, row_batch);
    [enc endEncoding];
    if (!fft1d_gpu(rows, Nu, Nu, row_batch, false, cmd)) return false;

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:F_ref offset:0 atIndex:0];
    [enc setBuffer:rows offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nrows length:sizeof(nrows) atIndex:4];
    dispatch2(enc, c.pipe("take_rfft_half"), wh, row_batch);
    [enc endEncoding];

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:cols offset:0 atIndex:0];
    [enc setBuffer:F_ref offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_rfft_cols_from_half"), N * wh, tile_count);
    [enc endEncoding];
    if (!fft1d_gpu(cols, Nu, Nu, tile_count * wh_u, false, cmd)) return false;

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:F_ref offset:0 atIndex:0];
    [enc setBuffer:cols offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_half_from_cols"), N * wh, tile_count);
    [enc endEncoding];

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:rows offset:0 atIndex:0];
    [enc setBuffer:mov_patch offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&nt length:sizeof(nt) atIndex:3];
    dispatch2(enc, c.pipe("pack_tile_rows"), N, row_batch);
    [enc endEncoding];
    if (!fft1d_gpu(rows, Nu, Nu, row_batch, false, cmd)) return false;

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
    dispatch2(enc, c.pipe("write_rfft_cols_from_half"), N * wh, tile_count);
    [enc endEncoding];
    if (!fft1d_gpu(cols, Nu, Nu, tile_count * wh_u, false, cmd)) return false;

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:F_mov offset:0 atIndex:0];
    [enc setBuffer:cols offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_half_from_cols"), N * wh, tile_count);
    [enc endEncoding];

    enc = [cmd computeCommandEncoder];
    uint32_t half_count = tile_count * (uint32_t)N * wh_u;
    [enc setBuffer:F_ref offset:0 atIndex:0];
    [enc setBuffer:F_mov offset:0 atIndex:1];
    [enc setBytes:&half_count length:sizeof(half_count) atIndex:2];
    dispatch1(enc, c.pipe("l2_conj_mul"), half_count);
    [enc endEncoding];

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:cols offset:0 atIndex:0];
    [enc setBuffer:F_ref offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_rfft_cols_from_half"), N * wh, tile_count);
    [enc endEncoding];
    if (!fft1d_gpu(cols, Nu, Nu, tile_count * wh_u, true, cmd)) return false;

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:F_ref offset:0 atIndex:0];
    [enc setBuffer:cols offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("write_half_from_cols"), N * wh, tile_count);
    [enc endEncoding];

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:full offset:0 atIndex:0];
    [enc setBuffer:F_ref offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&wh_u length:sizeof(wh_u) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("expand_half_to_full_rows"), N * N, tile_count);
    [enc endEncoding];
    if (!fft1d_gpu(full, Nu, Nu, row_batch, true, cmd)) return false;

    enc = [cmd computeCommandEncoder];
    [enc setBuffer:corr offset:0 atIndex:0];
    [enc setBuffer:full offset:0 atIndex:1];
    [enc setBytes:&Nu length:sizeof(Nu) atIndex:2];
    [enc setBytes:&nt length:sizeof(nt) atIndex:3];
    dispatch2(enc, c.pipe("extract_real_tiles"), N * N, tile_count);
    [enc endEncoding];

    enc = [cmd computeCommandEncoder];
    uint32_t Nh = (uint32_t)N, Nw = (uint32_t)N;
    [enc setBuffer:corr_shift offset:0 atIndex:0];
    [enc setBuffer:corr offset:0 atIndex:1];
    [enc setBytes:&Nh length:sizeof(Nh) atIndex:2];
    [enc setBytes:&Nw length:sizeof(Nw) atIndex:3];
    [enc setBytes:&nt length:sizeof(nt) atIndex:4];
    dispatch2(enc, c.pipe("fftshift2d_real"), N * N, tile_count);

    [enc setBuffer:flow_b offset:0 atIndex:0];
    [enc setBuffer:corr_shift offset:0 atIndex:1];
    [enc setBuffer:mov_patch offset:0 atIndex:2];
    [enc setBytes:&P length:sizeof(P) atIndex:3];
    dispatch1(enc, c.pipe("l2_argmin"), tile_count);
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
}

} // namespace

bool metal_gpu_init() { return ctx().ok; }

Image compute_grey_fft_metal(const Image& raw) {
    if (!metal_gpu_init() || raw.h <= 0 || raw.w <= 0) return Image();
    auto& c = ctx();
    const uint32_t h = (uint32_t)raw.h, w = (uint32_t)raw.w;
    // In-place fftshift requires even dims (Bayer RAW is even).
    if ((h & 1u) || (w & 1u)) return Image();
    const size_t n = (size_t)h * w;
    const size_t cbytes = n * sizeof(float) * 2;
    const size_t col_bytes = (size_t)kFftBatchChunk * h * sizeof(float) * 2;

    id<MTLBuffer> real_in = buf(raw.data.data(), n * sizeof(float));
    id<MTLBuffer> c0 = buf(nullptr, cbytes);
    id<MTLBuffer> col_scratch = buf(nullptr, col_bytes);
    if (!real_in || !c0 || !col_scratch) return Image();

    // Forward FFT (one full complex + column strip scratch)
    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return Image();
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:c0 offset:0 atIndex:0];
    [enc setBuffer:real_in offset:0 atIndex:1];
    [enc setBytes:&h length:sizeof(h) atIndex:2];
    [enc setBytes:&w length:sizeof(w) atIndex:3];
    dispatch2(enc, c.pipe("pack_rows_real"), w, h);
    [enc endEncoding];
    real_in = nil; // drop early; CB retains until complete
    if (!fft2d_gpu(c0, col_scratch, h, w, false, cmd)) return Image();
    // fft2d flushes internally; cmd is a fresh empty buffer — start shift pass on it.

    // In-place fftshift → zero borders → fftshift (even size: shift is involution)
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:c0 offset:0 atIndex:0];
    [enc setBytes:&h length:sizeof(h) atIndex:1];
    [enc setBytes:&w length:sizeof(w) atIndex:2];
    dispatch2(enc, c.pipe("fftshift_swap_x"), w / 2u, h);
    dispatch2(enc, c.pipe("fftshift_swap_y"), w, h / 2u);

    [enc setBuffer:c0 offset:0 atIndex:0];
    [enc setBytes:&h length:sizeof(h) atIndex:1];
    [enc setBytes:&w length:sizeof(w) atIndex:2];
    dispatch2(enc, c.pipe("zero_fft_borders"), w, h);

    [enc setBuffer:c0 offset:0 atIndex:0];
    [enc setBytes:&h length:sizeof(h) atIndex:1];
    [enc setBytes:&w length:sizeof(w) atIndex:2];
    dispatch2(enc, c.pipe("fftshift_swap_x"), w / 2u, h);
    dispatch2(enc, c.pipe("fftshift_swap_y"), w, h / 2u);
    [enc endEncoding];

    if (!fft2d_gpu(c0, col_scratch, h, w, true, cmd)) return Image();

    id<MTLBuffer> real_out = buf(nullptr, n * sizeof(float));
    if (!real_out) return Image();
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
    const int ny = flow.ny, nx = flow.nx;
    if (ny <= 0 || nx <= 0) return true;
    const int ts = tile_size, R = search_radius;
    const int N = 2 * R + ts;
    const int wh = N / 2 + 1;
    const uint32_t ntiles = (uint32_t)(ny * nx);

    id<MTLBuffer> ref_img = buf(ref.data.data(), ref.data.size() * sizeof(float));
    id<MTLBuffer> mov_img = buf(moving.data.data(), moving.data.size() * sizeof(float));
    id<MTLBuffer> flow_b = buf(flow.flow.data(), flow.flow.size() * sizeof(float));
    if (!ref_img || !mov_img || !flow_b) return false;

    for (uint32_t t0 = 0; t0 < ntiles; t0 += kL2TileChunk) {
        uint32_t tc = std::min(kL2TileChunk, ntiles - t0);
        if (!l2_chunk(ref_img, mov_img, flow_b, ref.h, ref.w, moving.h, moving.w,
                      ts, R, N, wh, (uint32_t)ny, (uint32_t)nx, t0, tc))
            return false;
    }
    memcpy(flow.flow.data(), [flow_b contents], flow.flow.size() * sizeof(float));
    return true;
}

} // namespace hhsr
