// Metal runtime for grey-FFT + L2 BM + merge accumulate (Apple / iOS only).
// Chunks large batches so full-res (1×) fits in memory / GPU time; 2× crop is smaller.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hhsr {
namespace {

// Bluestein A scratch rows per dispatch (~16MB at m=8192 with 128).
static constexpr uint32_t kFftBatchChunk = 256;
// Soft commit (no wait) every N Bluestein groups to bound CB size; dual-A covers hazards.
static constexpr uint32_t kCommitEveryChunks = 16;
// Soft commit every N column strips in 2D FFT (no wait).
static constexpr uint32_t kColCommitEvery = 32;
// Max L2 tiles processed together (buffers scale with ntiles * N²).
static constexpr uint32_t kL2TileChunk = 1024;

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
    // Dual Bluestein A: ping-pong across soft commits without waitUntilCompleted.
    id<MTLBuffer> scratch_A0 = nil;
    id<MTLBuffer> scratch_A1 = nil;
    id<MTLBuffer> scratch_B = nil;
    id<MTLBuffer> scratch_chirp = nil;
    id<MTLBuffer> scratch_cols = nil;
    size_t scratch_A0_bytes = 0;
    size_t scratch_A1_bytes = 0;
    size_t scratch_B_bytes = 0;
    size_t scratch_chirp_bytes = 0;
    size_t scratch_cols_bytes = 0;
    // Cached Bluestein chirp + FFT(B) for (n, inverse).
    uint32_t cache_n = 0;
    int cache_inv = -1;
    id<MTLBuffer> cache_chirp = nil;
    id<MTLBuffer> cache_B = nil;
    // Reused L2 working set (grow-only).
    id<MTLBuffer> l2_ref_pad = nil;
    id<MTLBuffer> l2_mov_patch = nil;
    id<MTLBuffer> l2_rows = nil;
    id<MTLBuffer> l2_F_ref = nil;
    id<MTLBuffer> l2_F_mov = nil;
    id<MTLBuffer> l2_cols = nil;
    id<MTLBuffer> l2_full = nil;
    id<MTLBuffer> l2_corr = nil;
    id<MTLBuffer> l2_corr_shift = nil;
    size_t l2_ref_pad_b = 0, l2_mov_patch_b = 0, l2_rows_b = 0;
    size_t l2_F_ref_b = 0, l2_F_mov_b = 0, l2_cols_b = 0;
    size_t l2_full_b = 0, l2_corr_b = 0, l2_corr_shift_b = 0;
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

    id<MTLBuffer> scratch(__strong id<MTLBuffer>& slot, size_t& slot_bytes, size_t need) {
        if (need == 0) return nil;
        if (slot && slot_bytes >= need) return slot;
        slot = [device newBufferWithLength:need options:MTLResourceStorageModeShared];
        slot_bytes = slot ? need : 0;
        return slot;
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
            "fft1d_pow2_cpp", "fft1d_bitrev", "fft1d_butterfly", "fft_scale_inv", "make_chirp",
            "bluestein_pack_A", "bluestein_clear_B", "bluestein_fill_B", "bluestein_extract",
            "cbuf_mul_broadcast_B", "pack_rows_real", "transpose_c",
            "gather_cols", "scatter_cols", "fftshift_swap_x", "fftshift_swap_y",
            "fftshift2d_c", "zero_fft_borders", "extract_real",
            "l2_pack_tiles", "l2_conj_mul", "l2_argmin", "fftshift2d_real",
            "pack_tile_rows", "take_rfft_half", "write_rfft_cols_from_half",
            "write_half_from_cols", "expand_half_to_full_rows", "extract_real_tiles",
            "merge_accumulate_comp", "merge_accumulate_ref",
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

// Commit and wait — only for final readback / error boundaries.
static bool flush_cmd(__strong id<MTLCommandBuffer>& cmd) {
    auto& c = ctx();
    if (!cmd) return false;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;
    cmd = [c.queue commandBuffer];
    return cmd != nil;
}

// Commit without waiting so the GPU can stay busy; caller must not reuse
// resources still owned by the previous CB (use dual scratch).
static bool commit_cmd(__strong id<MTLCommandBuffer>& cmd) {
    auto& c = ctx();
    if (!cmd) return false;
    [cmd commit];
    cmd = [c.queue commandBuffer];
    return cmd != nil;
}

// Parallel staged pow2 FFT — no CPU sync (encode only).
static bool fft1d_pow2_encode(id<MTLBuffer> data, uint32_t n, uint32_t stride,
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

    [enc setBuffer:data offset:byte_off atIndex:0];
    [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
    [enc setBytes:&str length:sizeof(str) atIndex:2];
    [enc setBytes:&bat length:sizeof(bat) atIndex:3];
    dispatch2(enc, c.pipe("fft1d_bitrev"), n, batch);

    for (uint32_t len = 2; len <= n; len <<= 1) {
        [enc setBuffer:data offset:byte_off atIndex:0];
        [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
        [enc setBytes:&str length:sizeof(str) atIndex:2];
        [enc setBytes:&bat length:sizeof(bat) atIndex:3];
        [enc setBytes:&len length:sizeof(len) atIndex:4];
        [enc setBytes:&inv length:sizeof(inv) atIndex:5];
        dispatch2(enc, c.pipe("fft1d_butterfly"), n / 2, batch);
    }

    if (inverse && apply_inv_scale) {
        [enc setBuffer:data offset:byte_off atIndex:0];
        [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
        [enc setBytes:&str length:sizeof(str) atIndex:2];
        [enc setBytes:&bat length:sizeof(bat) atIndex:3];
        dispatch2(enc, c.pipe("fft_scale_inv"), n, batch);
    }
    [enc endEncoding];
    return true;
}

static bool fft1d_pow2_gpu(id<MTLBuffer> data, uint32_t n, uint32_t stride,
                           uint32_t batch, bool inverse, bool apply_inv_scale,
                           __strong id<MTLCommandBuffer>& cmd) {
    // Encode all batches without mid-flush; caller flushes at phase boundaries.
    for (uint32_t b0 = 0; b0 < batch; b0 += kFftBatchChunk) {
        uint32_t bc = std::min(kFftBatchChunk, batch - b0);
        if (!fft1d_pow2_encode(data, n, stride, b0, bc, inverse, apply_inv_scale, cmd))
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
    const size_t chirp_bytes = sizeof(float) * 2 * n;
    const size_t B_bytes = sizeof(float) * 2 * m;
    const size_t A_bytes = sizeof(float) * 2 * m * kFftBatchChunk;
    id<MTLBuffer> A0 = c.scratch(c.scratch_A0, c.scratch_A0_bytes, A_bytes);
    id<MTLBuffer> A1 = c.scratch(c.scratch_A1, c.scratch_A1_bytes, A_bytes);
    if (!A0 || !A1) return false;

    uint32_t n_u = n, m_u = m;
    int inv = inverse ? 1 : 0;
    id<MTLBuffer> chirp = nil;
    id<MTLBuffer> B = nil;

    if (c.cache_n == n && c.cache_inv == inv && c.cache_chirp && c.cache_B) {
        chirp = c.cache_chirp;
        B = c.cache_B;
    } else {
        chirp = c.scratch(c.scratch_chirp, c.scratch_chirp_bytes, chirp_bytes);
        B = c.scratch(c.scratch_B, c.scratch_B_bytes, B_bytes);
        if (!chirp || !B) return false;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
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

        if (!fft1d_pow2_encode(B, m, m, 0, 1, false, false, cmd)) return false;
        // Hold B/chirp for reuse across column strips / L2 calls with same n.
        c.cache_chirp = chirp;
        c.cache_B = B;
        c.cache_n = n;
        c.cache_inv = inv;
    }

    uint32_t chunks_since_commit = 0;
    int a_sel = 0;
    for (uint32_t b0 = 0; b0 < batch; b0 += kFftBatchChunk) {
        uint32_t bc = std::min(kFftBatchChunk, batch - b0);
        uint32_t str = stride;
        uint32_t bat = bc;
        NSUInteger in_off = (NSUInteger)b0 * stride * sizeof(float) * 2;
        id<MTLBuffer> A = (a_sel == 0) ? A0 : A1;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:data offset:in_off atIndex:1];
        [enc setBuffer:chirp offset:0 atIndex:2];
        [enc setBytes:&n_u length:sizeof(n_u) atIndex:3];
        [enc setBytes:&m_u length:sizeof(m_u) atIndex:4];
        [enc setBytes:&str length:sizeof(str) atIndex:5];
        [enc setBytes:&bat length:sizeof(bat) atIndex:6];
        dispatch2(enc, c.pipe("bluestein_pack_A"), m, bc);
        [enc endEncoding];

        if (!fft1d_pow2_encode(A, m, m, 0, bc, false, false, cmd)) return false;

        enc = [cmd computeCommandEncoder];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:B offset:0 atIndex:1];
        [enc setBytes:&m_u length:sizeof(m_u) atIndex:2];
        [enc setBytes:&bat length:sizeof(bat) atIndex:3];
        dispatch2(enc, c.pipe("cbuf_mul_broadcast_B"), m, bc);
        [enc endEncoding];

        if (!fft1d_pow2_encode(A, m, m, 0, bc, true, false, cmd)) return false;

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

        a_sel ^= 1;
        if (++chunks_since_commit >= kCommitEveryChunks) {
            chunks_since_commit = 0;
            // Soft commit — dual A means the next chunk uses the other buffer.
            if (!commit_cmd(cmd)) return false;
        }
    }

    if (inverse) {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        for (uint32_t b0 = 0; b0 < batch; b0 += kFftBatchChunk) {
            uint32_t bc = std::min(kFftBatchChunk, batch - b0);
            uint32_t str = stride;
            NSUInteger off = (NSUInteger)b0 * stride * sizeof(float) * 2;
            [enc setBuffer:data offset:off atIndex:0];
            [enc setBytes:&n_u length:sizeof(n_u) atIndex:1];
            [enc setBytes:&str length:sizeof(str) atIndex:2];
            [enc setBytes:&bc length:sizeof(bc) atIndex:3];
            dispatch2(enc, c.pipe("fft_scale_inv"), n, bc);
        }
        [enc endEncoding];
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
    // Rows then columns stay on the GPU; Metal tracks buffer hazards in-CB.
    // Soft-commit only to bound encoder size — no waitUntilCompleted mid-FFT.
    if (!fft1d_gpu(cbuf, w, w, h, inverse, cmd)) return false;
    if (!commit_cmd(cmd)) return false;

    uint32_t strips = 0;
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

        if (++strips >= kColCommitEvery) {
            strips = 0;
            if (!commit_cmd(cmd)) return false;
        }
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

    id<MTLBuffer> ref_pad = c.scratch(c.l2_ref_pad, c.l2_ref_pad_b, all_tiles);
    id<MTLBuffer> mov_patch = c.scratch(c.l2_mov_patch, c.l2_mov_patch_b, all_tiles);
    id<MTLBuffer> rows = c.scratch(c.l2_rows, c.l2_rows_b, (size_t)row_batch * N * sizeof(float) * 2);
    id<MTLBuffer> F_ref = c.scratch(c.l2_F_ref, c.l2_F_ref_b, half_c);
    id<MTLBuffer> F_mov = c.scratch(c.l2_F_mov, c.l2_F_mov_b, half_c);
    id<MTLBuffer> cols = c.scratch(c.l2_cols, c.l2_cols_b, (size_t)tile_count * wh * N * sizeof(float) * 2);
    id<MTLBuffer> full = c.scratch(c.l2_full, c.l2_full_b, full_c);
    id<MTLBuffer> corr = c.scratch(c.l2_corr, c.l2_corr_b, all_tiles);
    id<MTLBuffer> corr_shift = c.scratch(c.l2_corr_shift, c.l2_corr_shift_b, all_tiles);
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
    id<MTLBuffer> col_scratch = c.scratch(c.scratch_cols, c.scratch_cols_bytes, col_bytes);
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


namespace {

// Must match HHSRKernels.metal MergeCompParams / MergeRefParams layout
// (including 16-byte padding for constant-buffer setBytes).
struct MergeCompParamsCPU {
    uint32_t band_h, Ws, y0, lr_h, lr_w;
    uint32_t flow_ny, flow_nx, cov_h, cov_w;
    uint32_t nch, bayer, iso, tile_size;
    float scale;
    uint32_t cfa00, cfa01, cfa10, cfa11;
    uint32_t _pad0 = 0, _pad1 = 0;
};
static_assert(sizeof(MergeCompParamsCPU) == 80, "MergeCompParamsCPU layout");

struct MergeRefParamsCPU {
    uint32_t band_h, Ws, y0, lr_h, lr_w;
    uint32_t cov_h, cov_w, acc_h, acc_w;
    uint32_t nch, bayer, iso, robustness_denoise, rad_max;
    float scale;
    float max_multiplier;
    float max_frame_count;
    uint32_t cfa00, cfa01, cfa10, cfa11;
    uint32_t _pad0 = 0, _pad1 = 0, _pad2 = 0;
};
static_assert(sizeof(MergeRefParamsCPU) == 96, "MergeRefParamsCPU layout");

struct MergeAccState {
    id<MTLBuffer> num = nil;
    id<MTLBuffer> den = nil;
    size_t capacity = 0; // allocated bytes
    size_t bytes = 0;    // active band bytes
};

// One GPU-resident copy per comparison frame (keeps buffers across bands).
struct MergeFrameGpu {
    int key = -1; // >=0: frame_id; -1: pointer-keyed
    int lr_h = 0, lr_w = 0;
    int flow_ny = 0, flow_nx = 0;
    int cov_h = 0, cov_w = 0;
    const f32* img = nullptr;
    const f32* flow = nullptr;
    const f32* cov = nullptr;
    const f32* rob = nullptr;
    id<MTLBuffer> b_img = nil;
    id<MTLBuffer> b_flow = nil;
    id<MTLBuffer> b_cov = nil;
    id<MTLBuffer> b_rob = nil;
    size_t img_b = 0, flow_b = 0, cov_b = 0, rob_b = 0;
};

struct MergeRefGpu {
    const f32* img = nullptr;
    const f32* cov = nullptr;
    const f32* acc = nullptr;
    id<MTLBuffer> b_img = nil;
    id<MTLBuffer> b_cov = nil;
    id<MTLBuffer> b_acc = nil;
    size_t img_b = 0, cov_b = 0, acc_b = 0;
};

static MergeAccState g_merge_acc;
static std::vector<MergeFrameGpu> g_merge_frames;
static MergeRefGpu g_merge_ref;
static __strong id<MTLCommandBuffer> g_merge_band_cmd = nil;
static __strong id<MTLComputeCommandEncoder> g_merge_enc = nil;

static void merge_enc_close() {
    if (g_merge_enc) {
        [g_merge_enc endEncoding];
        g_merge_enc = nil;
    }
}

static void merge_band_cmd_reset() {
    merge_enc_close();
    g_merge_band_cmd = nil;
}

static bool merge_band_cmd_ensure() {
    if (g_merge_band_cmd) return true;
    auto& c = ctx();
    g_merge_band_cmd = [c.queue commandBuffer];
    return g_merge_band_cmd != nil;
}

// One compute encoder for all comps + ref in a band (avoids N begin/end pairs).
static bool merge_enc_ensure() {
    if (g_merge_enc) return true;
    if (!merge_band_cmd_ensure()) return false;
    g_merge_enc = [g_merge_band_cmd computeCommandEncoder];
    return g_merge_enc != nil;
}

static bool merge_frame_resident(int frame_id) {
    if (frame_id < 0) return false;
    for (const MergeFrameGpu& e : g_merge_frames) {
        if (e.key == frame_id && e.b_img && e.b_flow && e.b_rob) return true;
    }
    return false;
}

static id<MTLBuffer> ensure_sized(__strong id<MTLBuffer>& slot, size_t nbytes) {
    if (nbytes == 0) return nil;
    if (slot && slot.length >= nbytes) return slot;
    slot = buf(nullptr, nbytes);
    return slot;
}

static bool copy_into(id<MTLBuffer> slot, const f32* data, size_t nbytes) {
    if (!slot || !data || nbytes == 0 || slot.length < nbytes) return false;
    memcpy([slot contents], data, nbytes);
    return true;
}

// Resolve comparison-frame GPU buffers. Prefer frame_key (>=0) so streamed
// scratch Images (same CPU pointer, new pixels) still hit across bands.
// Cache hit does not require CPU img/flow/rob pointers (skip disk reload).
static bool acquire_frame_gpu(const Image& img, const FlowField& flow,
                              const CovField& covs, const Image& rob, int frame_key,
                              __strong id<MTLBuffer>& b_img, __strong id<MTLBuffer>& b_flow,
                              __strong id<MTLBuffer>& b_cov, __strong id<MTLBuffer>& b_rob) {
    const f32* ip = img.data.empty() ? nullptr : img.data.data();
    const f32* fp = flow.flow.empty() ? nullptr : flow.flow.data();
    const f32* cp = covs.cov.empty() ? nullptr : covs.cov.data();
    const f32* rp = rob.data.empty() ? nullptr : rob.data.data();
    const size_t img_b = img.data.size() * sizeof(float);
    const size_t flow_b = flow.flow.size() * sizeof(float);
    const size_t cov_b = covs.cov.size() * sizeof(float);
    const size_t rob_b = rob.data.size() * sizeof(float);

    auto finish = [&](MergeFrameGpu& e) -> bool {
        b_img = e.b_img;
        b_flow = e.b_flow;
        b_cov = e.b_cov;
        b_rob = e.b_rob;
        if (!b_cov) {
            static float kDummyCov[4] = {1.f, 0.f, 0.f, 1.f};
            b_cov = buf(kDummyCov, sizeof(kDummyCov));
            e.b_cov = b_cov;
        }
        return b_img && b_flow && b_rob && b_cov;
    };

    // Stable frame id: upload once, reuse for every band.
    if (frame_key >= 0) {
        for (MergeFrameGpu& e : g_merge_frames) {
            if (e.key != frame_key) continue;
            if (e.b_img && e.b_flow && e.b_rob) return finish(e);
            break;
        }
        // Miss — need host data to upload.
        if (!ip || !fp || !rp || img_b == 0 || flow_b == 0 || rob_b == 0) return false;
        auto upload_into = [&](MergeFrameGpu& e) -> bool {
            e.key = frame_key;
            e.lr_h = img.h;
            e.lr_w = img.w;
            e.flow_ny = flow.ny;
            e.flow_nx = flow.nx;
            e.cov_h = covs.h;
            e.cov_w = covs.w;
            e.img = ip;
            e.flow = fp;
            e.cov = cp;
            e.rob = rp;
            e.img_b = img_b;
            e.flow_b = flow_b;
            e.cov_b = cov_b;
            e.rob_b = rob_b;
            e.b_img = buf(ip, img_b);
            e.b_flow = buf(fp, flow_b);
            e.b_cov = cov_b ? buf(cp, cov_b) : nil;
            e.b_rob = buf(rp, rob_b);
            if (!e.b_img || !e.b_flow || !e.b_rob) return false;
            if (!e.b_cov) {
                static float kDummyCov[4] = {1.f, 0.f, 0.f, 1.f};
                e.b_cov = buf(kDummyCov, sizeof(kDummyCov));
                if (!e.b_cov) return false;
            }
            return finish(e);
        };
        for (MergeFrameGpu& e : g_merge_frames) {
            if (e.key == frame_key) return upload_into(e);
        }
        MergeFrameGpu e;
        g_merge_frames.push_back(e);
        return upload_into(g_merge_frames.back());
    }

    if (!ip || !fp || !rp || img_b == 0 || flow_b == 0 || rob_b == 0) return false;

    for (MergeFrameGpu& e : g_merge_frames) {
        if (e.key >= 0 || e.img != ip) continue;
        const bool same_aux = (e.flow == fp && e.cov == cp && e.rob == rp &&
                               e.img_b == img_b && e.flow_b == flow_b &&
                               e.cov_b == cov_b && e.rob_b == rob_b);
        if (same_aux && e.b_img && e.b_flow && e.b_rob)
            return finish(e);
        // Same img allocation, new contents/aux (legacy streamed path).
        if (!ensure_sized(e.b_img, img_b) || !copy_into(e.b_img, ip, img_b)) return false;
        e.b_flow = buf(fp, flow_b);
        e.b_cov = cov_b ? buf(cp, cov_b) : nil;
        e.b_rob = buf(rp, rob_b);
        e.flow = fp;
        e.cov = cp;
        e.rob = rp;
        e.img_b = img_b;
        e.flow_b = flow_b;
        e.cov_b = cov_b;
        e.rob_b = rob_b;
        if (!e.b_flow || !e.b_rob) return false;
        if (!e.b_cov) {
            static float kDummyCov[4] = {1.f, 0.f, 0.f, 1.f};
            e.b_cov = buf(kDummyCov, sizeof(kDummyCov));
            if (!e.b_cov) return false;
        }
        return finish(e);
    }

    MergeFrameGpu e;
    e.key = -1;
    e.lr_h = img.h;
    e.lr_w = img.w;
    e.flow_ny = flow.ny;
    e.flow_nx = flow.nx;
    e.cov_h = covs.h;
    e.cov_w = covs.w;
    e.img = ip;
    e.flow = fp;
    e.cov = cp;
    e.rob = rp;
    e.img_b = img_b;
    e.flow_b = flow_b;
    e.cov_b = cov_b;
    e.rob_b = rob_b;
    e.b_img = buf(ip, img_b);
    e.b_flow = buf(fp, flow_b);
    e.b_cov = cov_b ? buf(cp, cov_b) : nil;
    e.b_rob = buf(rp, rob_b);
    if (!e.b_img || !e.b_flow || !e.b_rob) return false;
    if (!e.b_cov) {
        static float kDummyCov[4] = {1.f, 0.f, 0.f, 1.f};
        e.b_cov = buf(kDummyCov, sizeof(kDummyCov));
        if (!e.b_cov) return false;
    }
    g_merge_frames.push_back(e);
    return finish(g_merge_frames.back());
}

static bool acquire_ref_gpu(const Image& img, const CovField& covs, const Image* acc_rob,
                            bool denoise, __strong id<MTLBuffer>& b_img,
                            __strong id<MTLBuffer>& b_cov, __strong id<MTLBuffer>& b_acc) {
    const f32* ip = img.data.data();
    const f32* cp = covs.cov.data();
    const f32* ap = (denoise && acc_rob) ? acc_rob->data.data() : nullptr;
    const size_t img_b = img.data.size() * sizeof(float);
    const size_t cov_b = covs.cov.size() * sizeof(float);
    const size_t acc_b = (denoise && acc_rob) ? acc_rob->data.size() * sizeof(float) : sizeof(float);
    if (!ip || img_b == 0) return false;

    if (g_merge_ref.img == ip && g_merge_ref.cov == cp && g_merge_ref.acc == ap &&
        g_merge_ref.img_b == img_b && g_merge_ref.cov_b == cov_b &&
        g_merge_ref.acc_b == acc_b && g_merge_ref.b_img) {
        b_img = g_merge_ref.b_img;
        b_cov = g_merge_ref.b_cov;
        b_acc = g_merge_ref.b_acc;
        if (!b_cov) {
            static float kDummyCov[4] = {1.f, 0.f, 0.f, 1.f};
            b_cov = buf(kDummyCov, sizeof(kDummyCov));
            g_merge_ref.b_cov = b_cov;
        }
        return b_img && b_cov && b_acc;
    }

    g_merge_ref.img = ip;
    g_merge_ref.cov = cp;
    g_merge_ref.acc = ap;
    g_merge_ref.img_b = img_b;
    g_merge_ref.cov_b = cov_b;
    g_merge_ref.acc_b = acc_b;
    g_merge_ref.b_img = buf(ip, img_b);
    g_merge_ref.b_cov = cov_b ? buf(cp, cov_b) : nil;
    if (denoise && ap)
        g_merge_ref.b_acc = buf(ap, acc_b);
    else {
        static float kDummyRob = 0.f;
        g_merge_ref.b_acc = buf(&kDummyRob, sizeof(float));
    }
    if (!g_merge_ref.b_img || !g_merge_ref.b_acc) return false;
    if (!g_merge_ref.b_cov) {
        static float kDummyCov[4] = {1.f, 0.f, 0.f, 1.f};
        g_merge_ref.b_cov = buf(kDummyCov, sizeof(kDummyCov));
        if (!g_merge_ref.b_cov) return false;
    }
    b_img = g_merge_ref.b_img;
    b_cov = g_merge_ref.b_cov;
    b_acc = g_merge_ref.b_acc;
    return true;
}

// Grow-only GPU accumulators. Zero only at the start of a band (when CB is new).
static bool ensure_acc_buffers(size_t nelem, bool zero) {
    const size_t bytes = nelem * sizeof(float);
    if (bytes == 0) return false;
    auto& c = ctx();
    if (!g_merge_acc.num || !g_merge_acc.den || g_merge_acc.capacity < bytes) {
        g_merge_acc.num = [c.device newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared];
        g_merge_acc.den = [c.device newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared];
        g_merge_acc.capacity = (g_merge_acc.num && g_merge_acc.den) ? bytes : 0;
        if (!g_merge_acc.capacity) return false;
        zero = true;
    }
    g_merge_acc.bytes = bytes;
    if (zero) {
        memset([g_merge_acc.num contents], 0, bytes);
        memset([g_merge_acc.den contents], 0, bytes);
    }
    return true;
}

static bool readback_acc(Image& num_band, Image& den_band) {
    if (!g_merge_acc.num || !g_merge_acc.den) return false;
    const size_t bytes = g_merge_acc.bytes;
    if (num_band.data.size() * sizeof(float) < bytes ||
        den_band.data.size() * sizeof(float) < bytes)
        return false;
    memcpy(num_band.data.data(), [g_merge_acc.num contents], bytes);
    memcpy(den_band.data.data(), [g_merge_acc.den contents], bytes);
    return true;
}

} // namespace

bool metal_merge_has_frame(int frame_id) {
    return merge_frame_resident(frame_id);
}

bool merge_comp_band_metal(const Image& comp_raw, const FlowField& flow,
                           const CovField& covs, const Image& robustness,
                           int tile_size, Image& num_band, Image& den_band,
                           int y0, const Config& cfg, int frame_id) {
    if (!metal_gpu_init()) return false;
    if (num_band.h <= 0 || num_band.w <= 0) return false;
    const bool resident = merge_frame_resident(frame_id);
    if (!resident && (comp_raw.h <= 0 || comp_raw.w <= 0)) return false;
    auto& c = ctx();

    // New merge pass (band 0, no open CB yet): drop previous shot's GPU frames.
    const bool start_band = (g_merge_band_cmd == nil);
    if (y0 == 0 && start_band) {
        g_merge_frames.clear();
        g_merge_ref = {};
    }

    if (!ensure_acc_buffers(num_band.data.size(), start_band)) return false;

    id<MTLBuffer> b_img = nil, b_flow = nil, b_cov = nil, b_rob = nil;
    if (!acquire_frame_gpu(comp_raw, flow, covs, robustness, frame_id,
                           b_img, b_flow, b_cov, b_rob))
        return false;

    MergeCompParamsCPU p{};
    p.band_h = (uint32_t)num_band.h;
    p.Ws = (uint32_t)num_band.w;
    p.y0 = (uint32_t)y0;
    p.nch = (uint32_t)num_band.c;
    p.bayer = cfg.bayer_mode ? 1u : 0u;
    p.iso = (cfg.kernel == KernelShape::Iso) ? 1u : 0u;
    p.tile_size = (uint32_t)tile_size;
    p.scale = cfg.scale;
    p.cfa00 = cfg.cfa.p[0][0];
    p.cfa01 = cfg.cfa.p[0][1];
    p.cfa10 = cfg.cfa.p[1][0];
    p.cfa11 = cfg.cfa.p[1][1];

    if (comp_raw.h > 0 && comp_raw.w > 0) {
        p.lr_h = (uint32_t)comp_raw.h;
        p.lr_w = (uint32_t)comp_raw.w;
        p.flow_ny = (uint32_t)flow.ny;
        p.flow_nx = (uint32_t)flow.nx;
        p.cov_h = covs.h > 0 ? (uint32_t)covs.h : 1u;
        p.cov_w = covs.w > 0 ? (uint32_t)covs.w : 1u;
    } else {
        const MergeFrameGpu* hit = nullptr;
        for (const MergeFrameGpu& e : g_merge_frames) {
            if (e.key == frame_id && e.b_img) { hit = &e; break; }
        }
        if (!hit || hit->lr_h <= 0 || hit->lr_w <= 0) return false;
        p.lr_h = (uint32_t)hit->lr_h;
        p.lr_w = (uint32_t)hit->lr_w;
        p.flow_ny = (uint32_t)std::max(1, hit->flow_ny);
        p.flow_nx = (uint32_t)std::max(1, hit->flow_nx);
        p.cov_h = hit->cov_h > 0 ? (uint32_t)hit->cov_h : 1u;
        p.cov_w = hit->cov_w > 0 ? (uint32_t)hit->cov_w : 1u;
    }

    if (!merge_enc_ensure()) {
        merge_band_cmd_reset();
        return false;
    }
    id<MTLComputeCommandEncoder> enc = g_merge_enc;
    [enc setBuffer:g_merge_acc.num offset:0 atIndex:0];
    [enc setBuffer:g_merge_acc.den offset:0 atIndex:1];
    [enc setBuffer:b_img offset:0 atIndex:2];
    [enc setBuffer:b_flow offset:0 atIndex:3];
    [enc setBuffer:b_cov offset:0 atIndex:4];
    [enc setBuffer:b_rob offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    dispatch2(enc, c.pipe("merge_accumulate_comp"), p.Ws, p.band_h);
    // Keep encoder open for remaining comps + ref in this band.
    return true;
}

bool merge_ref_band_metal(const Image& ref_raw, const CovField& covs,
                          Image& num_band, Image& den_band, int y0,
                          const Config& cfg, const Image* acc_rob) {
    if (!metal_gpu_init()) return false;
    if (num_band.h <= 0 || num_band.w <= 0 || ref_raw.h <= 0) return false;
    auto& c = ctx();

    const bool start_band = (g_merge_band_cmd == nil);
    if (y0 == 0 && start_band) {
        g_merge_frames.clear();
        g_merge_ref = {};
    }

    if (!ensure_acc_buffers(num_band.data.size(), start_band)) return false;

    const bool denoise = cfg.accumulated_robustness_denoiser_enabled && acc_rob &&
                         acc_rob->h > 0 && acc_rob->w > 0;
    id<MTLBuffer> b_img = nil, b_cov = nil, b_acc = nil;
    if (!acquire_ref_gpu(ref_raw, covs, acc_rob, denoise, b_img, b_cov, b_acc))
        return false;

    MergeRefParamsCPU p{};
    p.band_h = (uint32_t)num_band.h;
    p.Ws = (uint32_t)num_band.w;
    p.y0 = (uint32_t)y0;
    p.lr_h = (uint32_t)ref_raw.h;
    p.lr_w = (uint32_t)ref_raw.w;
    p.cov_h = covs.h > 0 ? (uint32_t)covs.h : 1u;
    p.cov_w = covs.w > 0 ? (uint32_t)covs.w : 1u;
    p.acc_h = denoise ? (uint32_t)acc_rob->h : 1u;
    p.acc_w = denoise ? (uint32_t)acc_rob->w : 1u;
    p.nch = (uint32_t)num_band.c;
    p.bayer = cfg.bayer_mode ? 1u : 0u;
    p.iso = (cfg.kernel == KernelShape::Iso) ? 1u : 0u;
    p.robustness_denoise = denoise ? 1u : 0u;
    p.rad_max = (uint32_t)std::max(0, (int)cfg.acc_rob_rad_max);
    p.scale = cfg.scale;
    p.max_multiplier = cfg.acc_rob_max_multiplier;
    p.max_frame_count = cfg.acc_rob_max_frame_count;
    p.cfa00 = cfg.cfa.p[0][0];
    p.cfa01 = cfg.cfa.p[0][1];
    p.cfa10 = cfg.cfa.p[1][0];
    p.cfa11 = cfg.cfa.p[1][1];

    if (!merge_enc_ensure()) {
        merge_band_cmd_reset();
        return false;
    }
    id<MTLComputeCommandEncoder> enc = g_merge_enc;
    [enc setBuffer:g_merge_acc.num offset:0 atIndex:0];
    [enc setBuffer:g_merge_acc.den offset:0 atIndex:1];
    [enc setBuffer:b_img offset:0 atIndex:2];
    [enc setBuffer:b_cov offset:0 atIndex:3];
    [enc setBuffer:b_acc offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    dispatch2(enc, c.pipe("merge_accumulate_ref"), p.Ws, p.band_h);
    merge_enc_close();

    [g_merge_band_cmd commit];
    [g_merge_band_cmd waitUntilCompleted];
    const bool ok = (g_merge_band_cmd.status == MTLCommandBufferStatusCompleted);
    merge_band_cmd_reset();
    if (!ok) return false;

    return readback_acc(num_band, den_band);
}

} // namespace hhsr
