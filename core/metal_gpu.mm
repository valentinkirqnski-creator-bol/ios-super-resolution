// Metal runtime for grey-FFT + L2 BM + merge accumulate (Apple / iOS only).
// Chunks large batches so full-res (1×) fits in memory / GPU time; 2× crop is smaller.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_gpu.h"
#include "debug_utils.h"
#include "prof.h"
#include "mps_fft.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace hhsr {
namespace {

// Bluestein A scratch rows per dispatch (~16MB at m=8192 with 128).
static constexpr uint32_t kFftBatchChunk = 320;
// Soft commit (no wait) every N Bluestein groups to bound CB size; dual-A covers hazards.
static constexpr uint32_t kCommitEveryChunks = 24;
// Soft commit every N column strips in 2D FFT (no wait).
static constexpr uint32_t kColCommitEvery = 48;
// Max L2 tiles processed together (buffers scale with ntiles * N²).
// Larger chunks + dual-slot async (below) cut full-res 1× sync thrash.
static constexpr uint32_t kL2TileChunk = 2048;

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
    // Dual L2 working sets (grow-only) so chunk N+1 can encode while N runs.
    struct L2Scratch {
        id<MTLBuffer> ref_pad = nil, mov_patch = nil, rows = nil;
        id<MTLBuffer> F_ref = nil, F_mov = nil, cols = nil, full = nil;
        id<MTLBuffer> corr = nil, corr_shift = nil;
        size_t ref_pad_b = 0, mov_patch_b = 0, rows_b = 0;
        size_t F_ref_b = 0, F_mov_b = 0, cols_b = 0, full_b = 0;
        size_t corr_b = 0, corr_shift_b = 0;
    };
    L2Scratch l2[2];
    // Grow-only Alg. 5 temps — avoid alloc/free per frame (memory pressure slows merge).
    id<MTLBuffer> kern_raw = nil, kern_vst = nil, kern_grey = nil;
    id<MTLBuffer> kern_grad = nil, kern_cov = nil;
    size_t kern_raw_b = 0, kern_vst_b = 0, kern_grey_b = 0;
    size_t kern_grad_b = 0, kern_cov_b = 0;
    // Grow-only grey-FFT temps. Each call otherwise allocated the input, the
    // complex working set and the output fresh — ~190MB at 12MP — and the kernel
    // has to zero those pages before handing them over, which is CPU time
    // charged to compute_grey rather than to the GPU.
    id<MTLBuffer> fft_in = nil, fft_c0 = nil, fft_out = nil;
    size_t fft_in_b = 0, fft_c0_b = 0, fft_out_b = 0;
    // Stockham is autosort but out-of-place, so it needs a ping-pong buffer the
    // size of the largest batch it transforms (the row pass: h*w complex).
    id<MTLBuffer> fft_pp = nil;
    size_t fft_pp_b = 0;
    // Last grey-FFT output (Shared) — align_metal can reuse without re-upload.
    id<MTLBuffer> sticky_grey = nil;
    int sticky_grey_h = 0, sticky_grey_w = 0;
    // Reference Sobel gradients and ICA Hessian, cached per pyramid level.
    // One slot was enough while ICA ran only on the finest level; per-level ICA
    // asks for a different level on every call, so a single slot would miss
    // every time and recompute the gradients once per comparison frame instead
    // of once per burst. The reference does not change within a burst, so all
    // levels stay valid together.
    struct IcaRefSlot {
        const void* key = nullptr;
        id<MTLBuffer> ref = nil;
        id<MTLBuffer> gx = nil;
        id<MTLBuffer> gy = nil;
        id<MTLBuffer> hess = nil;
        int h = 0, w = 0, ts = 0, ny = 0, nx = 0;
    };
    static constexpr int kIcaSlots = 8;   // pyramids here are 4 levels
    IcaRefSlot ica[kIcaSlots];
    int ica_next = 0;
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
            "raw16_to_float_bayer",
            "fft1d_pow2_cpp", "fft1d_bitrev", "fft1d_butterfly", "fft_scale_inv", "make_chirp",
            "fft_stockham_stage", "fft_copy_batch",
            "bluestein_pack_A", "bluestein_clear_B", "bluestein_fill_B", "bluestein_extract",
            "cbuf_mul_broadcast_B", "pack_rows_real", "transpose_c",
            "gather_cols", "scatter_cols", "fftshift_swap_x", "fftshift_swap_y",
            "fftshift2d_c", "zero_fft_borders", "zero_fft_borders_natural", "extract_real",
            "merge_accumulate_comp_x4",
            "align_local_search_460",
            "l2_pack_tiles", "l2_conj_mul", "l2_argmin", "fftshift2d_real",
            "pack_tile_rows", "take_rfft_half", "write_rfft_cols_from_half",
            "write_half_from_cols", "expand_half_to_full_rows", "extract_real_tiles",
            "merge_accumulate_comp", "merge_accumulate_ref",
            "kernel_gat", "kernel_decimate_grey", "kernel_gradients", "kernel_estimate_cov",
            "rob_guide_bayer", "rob_local_stats_3x3", "rob_upscale_dogson",
            "rob_lowpass_gaussian5x5", "rob_hf_loss_adaptive",
            "rob_tile_residual_high", "rob_make_mask", "rob_local_min_5x5",
            "l1_bm_ts16", "l1_bm_ts32", "l1_bm_ts64", "ica_refine_tile",
            "pyr_conv_y", "pyr_conv_x", "pyr_subsample",
            "align_sobel_x", "align_sobel_y", "align_hessian",
            "align_upscale_flow", "align_upscale_flow_460",
            "merge_normalize_rgb16",
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

// Attach a GPU-time probe. Uses the completion handler on work the pipeline
// already waits for, so it adds no synchronization of its own — measuring must
// not create the stalls we are trying to measure.
static void prof_tag_gpu(id<MTLCommandBuffer> cmd, const char* name) {
    if (!prof_enabled() || !cmd) return;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
        prof_add_gpu(name, (done.GPUEndTime - done.GPUStartTime) * 1000.0);
    }];
}

// Label for command buffers that are soft-committed part-way through a stage.
// prof_tag_gpu is otherwise only attached to the last buffer of a multi-commit
// sequence, so every earlier segment's GPU time went unrecorded — the grey FFT
// commits several times per call and was reported as just its tail.
// thread_local: the 2x path runs estimate_kernels on a std::async thread, so a
// shared global would race with the stage set on the pipeline thread.
static thread_local const char* g_prof_stage = nullptr;

struct ProfStageScope {
    const char* prev;
    explicit ProfStageScope(const char* s) : prev(g_prof_stage) { g_prof_stage = s; }
    ~ProfStageScope() { g_prof_stage = prev; }
    ProfStageScope(const ProfStageScope&) = delete;
    ProfStageScope& operator=(const ProfStageScope&) = delete;
};

// Commit and wait — only for final readback / error boundaries.
static bool flush_cmd(__strong id<MTLCommandBuffer>& cmd) {
    auto& c = ctx();
    if (!cmd) return false;
    if (g_prof_stage) prof_tag_gpu(cmd, g_prof_stage);
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
    if (g_prof_stage) prof_tag_gpu(cmd, g_prof_stage);
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

// Set false to fall back to Bluestein for non-power-of-two lengths.
static constexpr bool kUseStockham = true;

// Radices largest-first, matching the verified host prototype. Returns false if
// anything other than 2/3/5/7 remains, in which case Bluestein still handles it.
static bool fft_factorize(uint32_t n, uint32_t* out, int cap, int& count) {
    count = 0;
    uint32_t m = n;
    const uint32_t radices[] = {7u, 5u, 4u, 3u, 2u};
    for (uint32_t r : radices) {
        while (m % r == 0u) {
            if (count >= cap) return false;
            out[count++] = r;
            m /= r;
        }
    }
    return m == 1u && count > 0;
}

// Mixed-radix Stockham. Autosort, so no digit-reversal pass; ping-pongs between
// `data` and `pp`, which must be at least batch*stride complex elements.
static bool fft1d_stockham_gpu(id<MTLBuffer> data, id<MTLBuffer> pp,
                               uint32_t n, uint32_t stride, uint32_t batch,
                               bool inverse, __strong id<MTLCommandBuffer>& cmd) {
    auto& c = ctx();
    if (!data || !pp || n < 2u || batch == 0u || !cmd) return false;

    uint32_t rad[24];
    int nrad = 0;
    if (!fft_factorize(n, rad, 24, nrad)) return false;

    struct StockhamParamsCPU {
        uint32_t N, R, Ns, NR, stride, batch_count, inverse, _pad0;
    };
    static_assert(sizeof(StockhamParamsCPU) == 32, "StockhamParamsCPU layout");

    id<MTLComputePipelineState> pipe = c.pipe("fft_stockham_stage");
    if (!pipe) return false;

    id<MTLBuffer> src = data;
    id<MTLBuffer> dst = pp;
    uint32_t Ns = 1u;
    for (int s = 0; s < nrad; ++s) {
        const uint32_t R = rad[s];
        StockhamParamsCPU p{};
        p.N = n;
        p.R = R;
        p.Ns = Ns;
        p.NR = n / R;
        p.stride = stride;
        p.batch_count = batch;
        p.inverse = inverse ? 1u : 0u;
        p._pad0 = 0u;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return false;
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        dispatch2(enc, pipe, p.NR, batch);
        [enc endEncoding];

        // Result of this stage is in dst; it becomes the next stage's source.
        std::swap(src, dst);
        Ns *= R;
    }

    // An odd number of stages leaves the result in the ping-pong buffer.
    if (src != data) {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return false;
        [enc setBuffer:data offset:0 atIndex:0];
        [enc setBuffer:src offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc setBytes:&stride length:sizeof(stride) atIndex:3];
        [enc setBytes:&batch length:sizeof(batch) atIndex:4];
        dispatch2(enc, c.pipe("fft_copy_batch"), n, batch);
        [enc endEncoding];
    }

    // Same convention as the pow2 and Bluestein paths: the inverse divides by n.
    if (inverse) {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return false;
        [enc setBuffer:data offset:0 atIndex:0];
        [enc setBytes:&n length:sizeof(n) atIndex:1];
        [enc setBytes:&stride length:sizeof(stride) atIndex:2];
        [enc setBytes:&batch length:sizeof(batch) atIndex:3];
        dispatch2(enc, c.pipe("fft_scale_inv"), n, batch);
        [enc endEncoding];
    }
    return true;
}

static bool fft1d_gpu(id<MTLBuffer> data, id<MTLBuffer> pp, uint32_t n, uint32_t stride,
                      uint32_t batch, bool inverse, __strong id<MTLCommandBuffer>& cmd) {
    if ((n & (n - 1)) == 0)
        return fft1d_pow2_gpu(data, n, stride, batch, inverse, /*scale*/true, cmd);
    if (kUseStockham && pp) {
        int nrad = 0;
        uint32_t rad[24];
        if (fft_factorize(n, rad, 24, nrad))
            return fft1d_stockham_gpu(data, pp, n, stride, batch, inverse, cmd);
    }
    return fft1d_bluestein_gpu(data, n, stride, batch, inverse, cmd);
}

// 2D FFT with one full-frame complex buffer: row FFTs in place, column FFTs via strips.
//
// prune_lowpass is for the grey-FFT caller, whose forward transform is followed
// by fftshift -> zero_fft_borders -> fftshift. That zeroing keeps only the
// central half of the shifted spectrum. Shifted position xs holds natural column
// x = (xs + w/2) mod w, and xs is zeroed when xs < w/4 or xs >= 3w/4, so natural
// columns [w/4, 3w/4) are zeroed for every row without exception. The shift is a
// pure permutation, so each untransformed column lands in exactly one position
// that is then overwritten with zero — skipping their column transform is
// equivalent, not an approximation.
//
// Only the forward direction can prune: by the inverse pass the row transform
// has already spread the surviving coefficients across every column.
static bool fft2d_gpu(id<MTLBuffer> cbuf, id<MTLBuffer> col_scratch, id<MTLBuffer> pp,
                      uint32_t h, uint32_t w,
                      bool inverse, __strong id<MTLCommandBuffer>& cmd,
                      bool prune_lowpass = false) {
    auto& c = ctx();
    // Rows then columns stay on the GPU; Metal tracks buffer hazards in-CB.
    // Soft-commit only to bound encoder size — no waitUntilCompleted mid-FFT.
    if (!fft1d_gpu(cbuf, pp, w, w, h, inverse, cmd)) return false;
    if (!commit_cmd(cmd)) return false;

    const bool prune = prune_lowpass && !inverse && w >= 8u;
    const uint32_t keep_lo = w / 4u;        // first discarded column
    const uint32_t keep_hi = w - w / 4u;    // first kept column again

    uint32_t strips = 0;
    for (uint32_t col0 = 0; col0 < w; col0 += kFftBatchChunk) {
        uint32_t ncol = std::min(kFftBatchChunk, w - col0);
        // Skip only strips lying wholly inside the discarded band; partially
        // overlapping strips are transformed in full so the kept columns in
        // them stay exact.
        if (prune && col0 >= keep_lo && col0 + ncol <= keep_hi) continue;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setBuffer:col_scratch offset:0 atIndex:0];
        [enc setBuffer:cbuf offset:0 atIndex:1];
        [enc setBytes:&h length:sizeof(h) atIndex:2];
        [enc setBytes:&w length:sizeof(w) atIndex:3];
        [enc setBytes:&col0 length:sizeof(col0) atIndex:4];
        [enc setBytes:&ncol length:sizeof(ncol) atIndex:5];
        dispatch2(enc, c.pipe("gather_cols"), h, ncol);
        [enc endEncoding];

        if (!fft1d_gpu(col_scratch, pp, h, h, ncol, inverse, cmd)) return false;

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

// Run one L2 chunk: tile_base .. tile_base+tile_count-1 (buffers from scratch slot).
// Commits asynchronously; caller waits before reusing the same slot.
static bool l2_chunk(id<MTLBuffer> ref_img, id<MTLBuffer> mov_img, id<MTLBuffer> flow_b,
                     int ref_h, int ref_w, int mov_h, int mov_w,
                     int ts, int R, int N, int wh,
                     uint32_t ny, uint32_t nx, uint32_t tile_base, uint32_t tile_count,
                     int slot, __strong id<MTLCommandBuffer>& out_cmd) {
    auto& c = ctx();
    if (slot < 0 || slot > 1) return false;
    MetalCtx::L2Scratch& s = c.l2[slot];
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

    id<MTLBuffer> ref_pad = c.scratch(s.ref_pad, s.ref_pad_b, all_tiles);
    id<MTLBuffer> mov_patch = c.scratch(s.mov_patch, s.mov_patch_b, all_tiles);
    id<MTLBuffer> rows = c.scratch(s.rows, s.rows_b, (size_t)row_batch * N * sizeof(float) * 2);
    id<MTLBuffer> F_ref = c.scratch(s.F_ref, s.F_ref_b, half_c);
    id<MTLBuffer> F_mov = c.scratch(s.F_mov, s.F_mov_b, half_c);
    id<MTLBuffer> cols = c.scratch(s.cols, s.cols_b, (size_t)tile_count * wh * N * sizeof(float) * 2);
    id<MTLBuffer> full = c.scratch(s.full, s.full_b, full_c);
    id<MTLBuffer> corr = c.scratch(s.corr, s.corr_b, all_tiles);
    id<MTLBuffer> corr_shift = c.scratch(s.corr_shift, s.corr_shift_b, all_tiles);
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
    if (!fft1d_gpu(rows, nil, Nu, Nu, row_batch, false, cmd)) return false;

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
    if (!fft1d_gpu(cols, nil, Nu, Nu, tile_count * wh_u, false, cmd)) return false;

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
    if (!fft1d_gpu(rows, nil, Nu, Nu, row_batch, false, cmd)) return false;

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
    if (!fft1d_gpu(cols, nil, Nu, Nu, tile_count * wh_u, false, cmd)) return false;

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
    if (!fft1d_gpu(cols, nil, Nu, Nu, tile_count * wh_u, true, cmd)) return false;

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
    if (!fft1d_gpu(full, nil, Nu, Nu, row_batch, true, cmd)) return false;

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
    out_cmd = cmd;
    return true;
}

} // namespace

bool metal_gpu_init() { return ctx().ok; }

bool metal_decode_raw16_to_float(const void* raw_data, size_t raw_bytes,
                                 int h, int w, int bytes_per_row,
                                 const float site_black[4],
                                 const float site_denom[4],
                                 const float site_wb[4],
                                 Image& out) {
    if (!metal_gpu_init() || !raw_data || h <= 0 || w <= 0 ||
        bytes_per_row < w * (int)sizeof(uint16_t) || (bytes_per_row & 1))
        return false;
    const size_t need_raw = (size_t)bytes_per_row * (size_t)h;
    if (raw_bytes < need_raw) return false;
    for (int i = 0; i < 4; ++i) {
        if (!(site_denom[i] > 0.f) || !std::isfinite(site_denom[i]) ||
            !std::isfinite(site_black[i]) || !std::isfinite(site_wb[i]))
            return false;
    }

    auto& c = ctx();
    struct RawDecodeParamsCPU {
        uint32_t h, w, stride_shorts, pad0;
        float black[4];
        float denom[4];
        float wb[4];
    };
    static_assert(sizeof(RawDecodeParamsCPU) == 64, "RawDecodeParamsCPU");

    RawDecodeParamsCPU p{};
    p.h = (uint32_t)h;
    p.w = (uint32_t)w;
    p.stride_shorts = (uint32_t)(bytes_per_row / (int)sizeof(uint16_t));
    for (int i = 0; i < 4; ++i) {
        p.black[i] = site_black[i];
        p.denom[i] = site_denom[i];
        p.wb[i] = site_wb[i];
    }

    const size_t out_b = (size_t)h * (size_t)w * sizeof(float);
    id<MTLBuffer> b_raw = buf(raw_data, need_raw);
    id<MTLBuffer> b_out = buf(nullptr, out_b);
    if (!b_raw || !b_out) return false;

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_raw offset:0 atIndex:0];
    [enc setBuffer:b_out offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    dispatch2(enc, c.pipe("raw16_to_float_bayer"), (NSUInteger)w, (NSUInteger)h);
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;

    Image decoded(h, w, 1);
    memcpy(decoded.data.data(), [b_out contents], out_b);
    out = std::move(decoded);
    return true;
}

static Image compute_grey_fft_metal_impl(const Image& raw) {
    if (!metal_gpu_init() || raw.h <= 0 || raw.w <= 0) return Image();
    // Attributes the soft-committed FFT segments, which previously vanished
    // from the profile: comp:grey measured ~750ms/frame against only ~204ms of
    // tagged GPU, and the gap was never CPU work.
    ProfStageScope prof_stage("grey:fft-segments");
    auto& c = ctx();
    const uint32_t h = (uint32_t)raw.h, w = (uint32_t)raw.w;
    // In-place fftshift requires even dims (Bayer RAW is even).
    if ((h & 1u) || (w & 1u)) return Image();
    const size_t n = (size_t)h * w;
    const size_t cbytes = n * sizeof(float) * 2;
    const size_t col_bytes = (size_t)kFftBatchChunk * h * sizeof(float) * 2;

    // MPSGraph does the same low-pass with real-to-Hermitean and
    // Hermitean-to-real transforms, so it never spends a pass transforming the
    // zero imaginary half, and it stores h x (w/2+1) instead of h x w. Falls
    // through to the Stockham path below on any failure or when unavailable
    // (the ops are iOS 16+, this project deploys to 15).
    if (mps_fft_enabled()) {
        ProfStageScope prof_mps("grey:fft-mpsgraph");
        Image grey;
        grey.h = (int)h;
        grey.w = (int)w;
        grey.c = 1;
        grey.data.resize(n);
        // Hand the graph the buffer align_metal will pin, so the result lands
        // there directly. Saves a second full-frame staging buffer inside
        // mps_fft (~47MB held across the whole analysis) and one memcpy per
        // frame. align_metal reuses sticky_grey when the dimensions match, so
        // it has to be refreshed here either way -- leaving the previous
        // frame's buffer pinned would align against a stale grey.
        id<MTLBuffer> pinned = c.scratch(c.fft_out, c.fft_out_b, n * sizeof(float));
        if (mps_grey_lowpass(raw.data.data(), grey.data.data(), (int)h, (int)w,
                             (__bridge void*)pinned)) {
            if (pinned) {
                c.sticky_grey = pinned;
                c.sticky_grey_h = (int)h;
                c.sticky_grey_w = (int)w;
            } else {
                c.sticky_grey = nil;
                c.sticky_grey_h = 0;
                c.sticky_grey_w = 0;
            }
            return grey;
        }
        c.sticky_grey = nil;
        c.sticky_grey_h = 0;
        c.sticky_grey_w = 0;
    }

    // Pooled rather than freshly allocated: pack_rows_real writes every element
    // of c0 and extract_real writes every element of the output, so neither
    // depends on the buffer arriving zeroed.
    id<MTLBuffer> real_in = c.scratch(c.fft_in, c.fft_in_b, n * sizeof(float));
    id<MTLBuffer> c0 = c.scratch(c.fft_c0, c.fft_c0_b, cbytes);
    id<MTLBuffer> col_scratch = c.scratch(c.scratch_cols, c.scratch_cols_bytes, col_bytes);
    // Ping-pong for Stockham. The row pass is the larger of the two batches
    // (h*w complex vs kFftBatchChunk*h), so sizing to cbytes covers both.
    // nil is tolerated: fft1d_gpu falls back to Bluestein without it.
    id<MTLBuffer> fft_pp = c.scratch(c.fft_pp, c.fft_pp_b, cbytes);
    if (!real_in || !c0 || !col_scratch) return Image();
    memcpy([real_in contents], raw.data.data(), n * sizeof(float));

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
    real_in = nil; // release the local ref; the pool and the CB keep it alive
    // Prune: the border zeroing below discards these columns unconditionally.
    if (!fft2d_gpu(c0, col_scratch, fft_pp, h, w, false, cmd, /*prune_lowpass*/true))
        return Image();
    // fft2d flushes internally; cmd is a fresh empty buffer — start shift pass on it.

    // Was fftshift -> zero borders -> fftshift, five full-buffer passes. The two
    // shifts are permutations that cancel, so the whole sequence collapses to
    // zeroing the preimages of the border in natural order -- one pass, same
    // bytes out. At this size the buffer does not fit in cache, so the four
    // removed passes were four round trips to DRAM.
    enc = [cmd computeCommandEncoder];
    [enc setBuffer:c0 offset:0 atIndex:0];
    [enc setBytes:&h length:sizeof(h) atIndex:1];
    [enc setBytes:&w length:sizeof(w) atIndex:2];
    dispatch2(enc, c.pipe("zero_fft_borders_natural"), w, h);
    [enc endEncoding];

    if (!fft2d_gpu(c0, col_scratch, fft_pp, h, w, true, cmd)) return Image();

    id<MTLBuffer> real_out = c.scratch(c.fft_out, c.fft_out_b, n * sizeof(float));
    if (!real_out) return Image();
    enc = [cmd computeCommandEncoder];
    uint32_t count = (uint32_t)n;
    [enc setBuffer:real_out offset:0 atIndex:0];
    [enc setBuffer:c0 offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:2];
    dispatch1(enc, c.pipe("extract_real"), n);
    [enc endEncoding];

    prof_tag_gpu(cmd, "grey:fft");
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return Image();

    // Pin for align_metal (moving grey) — same pixels as the host Image.
    c.sticky_grey = real_out;
    c.sticky_grey_h = (int)h;
    c.sticky_grey_w = (int)w;

    // assign() allocates and fills in one pass. Image(h,w,c) would value-init
    // the whole vector first, so a 12MP grey wrote ~48MB of zeros that the
    // following copy immediately overwrote.
    Image grey;
    grey.h = (int)h;
    grey.w = (int)w;
    grey.c = 1;
    const float* out_src = (const float*)[real_out contents];
    grey.data.assign(out_src, out_src + n);
    return grey;
}

static CovField estimate_kernels_metal_impl(const Image& raw, const Config& cfg) {
    if (!metal_gpu_init() || raw.h <= 0 || raw.w <= 0) return CovField();
    auto& c = ctx();

    struct KernelEstParamsCPU {
        uint32_t raw_h, raw_w, grey_h, grey_w;
        uint32_t bayer, selection;
        float alpha, beta;
        float k_detail, k_denoise, D_th, D_tr, k_stretch, k_shrink;
        uint32_t _pad0 = 0, _pad1 = 0;
    };
    static_assert(sizeof(KernelEstParamsCPU) == 64, "KernelEstParamsCPU layout");

    const bool bayer = cfg.bayer_mode;
    const int grey_h = bayer ? raw.h / 2 : raw.h;
    const int grey_w = bayer ? raw.w / 2 : raw.w;
    if (grey_h < 2 || grey_w < 2) return CovField();

    KernelEstParamsCPU p{};
    p.raw_h = (uint32_t)raw.h;
    p.raw_w = (uint32_t)raw.w;
    p.grey_h = (uint32_t)grey_h;
    p.grey_w = (uint32_t)grey_w;
    p.bayer = bayer ? 1u : 0u;
    p.selection = 0u; // 460-main kernels.py always uses hard thresholding.
    p.alpha = cfg.noise_alpha();
    p.beta = cfg.noise_beta();
    p.k_detail = cfg.k_detail;
    p.k_denoise = cfg.k_denoise;
    p.D_th = cfg.D_th;
    p.D_tr = cfg.D_tr;
    p.k_stretch = cfg.k_stretch;
    p.k_shrink = cfg.k_shrink;

    const size_t raw_b = raw.data.size() * sizeof(float);
    const size_t grey_b = (size_t)grey_h * (size_t)grey_w * sizeof(float);
    const size_t grad_b = (size_t)(grey_h - 1) * (size_t)(grey_w - 1) * 2u * sizeof(float);
    const size_t cov_b = (size_t)grey_h * (size_t)grey_w * 4u * sizeof(float);

    id<MTLBuffer> b_raw = c.scratch(c.kern_raw, c.kern_raw_b, raw_b);
    id<MTLBuffer> b_vst = c.scratch(c.kern_vst, c.kern_vst_b, grey_b);
    id<MTLBuffer> b_grey = c.scratch(c.kern_grey, c.kern_grey_b, grey_b);
    id<MTLBuffer> b_grad = c.scratch(c.kern_grad, c.kern_grad_b, grad_b);
    id<MTLBuffer> b_cov = c.scratch(c.kern_cov, c.kern_cov_b, cov_b);
    if (!b_raw || !b_vst || !b_grey || !b_grad || !b_cov) return CovField();
    memcpy([b_raw contents], raw.data.data(), raw_b);

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return CovField();

    // One encoder: Metal tracks hazards between dispatches (same math, less overhead).
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return CovField();

    [enc setBuffer:b_grey offset:0 atIndex:0];
    [enc setBuffer:b_raw offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    dispatch2(enc, c.pipe("kernel_decimate_grey"), p.grey_w, p.grey_h);

    [enc setBuffer:b_vst offset:0 atIndex:0];
    [enc setBuffer:b_grey offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    dispatch2(enc, c.pipe("kernel_gat"), p.grey_w, p.grey_h);

    [enc setBuffer:b_grad offset:0 atIndex:0];
    [enc setBuffer:b_vst offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    dispatch2(enc, c.pipe("kernel_gradients"), p.grey_w - 1u, p.grey_h - 1u);

    [enc setBuffer:b_cov offset:0 atIndex:0];
    [enc setBuffer:b_grad offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    dispatch2(enc, c.pipe("kernel_estimate_cov"), p.grey_w, p.grey_h);
    [enc endEncoding];

    prof_tag_gpu(cmd, "kernels:estimate-cov");
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return CovField();

    CovField covs(grey_h, grey_w);
    memcpy(covs.cov.data(), [b_cov contents], cov_b);
    return covs;
}

namespace {

struct RobGuideParamsCPU {
    uint32_t raw_h, raw_w, guide_h, guide_w;
    uint32_t bayer;
    uint32_t cfa00, cfa01, cfa10, cfa11;
    float wb0, wb1, wb2;
    uint32_t _pad0 = 0;
};
static_assert(sizeof(RobGuideParamsCPU) == 52, "RobGuideParamsCPU");

struct RobStatsParamsCPU {
    uint32_t h, w, nch;
    // Min-filter radius in guide pixels for rob_local_min_5x5 (was _pad0).
    // Ignored by the other kernels that share this params struct.
    uint32_t radius = 2;
};
static_assert(sizeof(RobStatsParamsCPU) == 16, "RobStatsParamsCPU");


struct RobDogsonParamsCPU {
    uint32_t in_h, in_w, out_h, out_w, nch;
    uint32_t is_ref, tile_size, flow_ny, flow_nx;
    float s;
    uint32_t _pad0 = 0, _pad1 = 0;
};
static_assert(sizeof(RobDogsonParamsCPU) == 48, "RobDogsonParamsCPU");

struct RobMaskParamsCPU {
    uint32_t h, w, nch, tile_size, flow_ny, flow_nx, curve_n, bayer;
    float r_t;
    uint32_t hf_enabled = 0;
    float hf_variance_loss_threshold = 0.f;
    uint32_t motion_edge_enabled = 0;
    float motion_edge_threshold = 0.f;
    float motion_edge_residual_threshold = 0.f;
    float alpha = 0.f;
    float beta = 0.f;
    float motion_edge_noise_floor_multiplier = 1.f;
    uint32_t motion_edge_neighborhood_radius = 0;
    // Keep in lockstep with RobMaskParams in HHSRKernels.metal: same fields,
    // same order. A mismatch here is silent on the shader side.
    float flow_reject_1d_residual_threshold = 0.f;
    uint32_t aperture_reject_enabled = 0;
    float r_s1 = 0.f;   // motion prior for aperture-limited tiles (was _pad0)
    uint32_t save_s_select = 0;  // 1 = also emit the per-pixel s1/s2 selector
    uint32_t ambiguous_enabled = 0;  // 1 = demote tiles with an ambiguous match
    uint32_t chain_reject_enabled = 0;  // 1 = demote chain-inconsistent tiles (was _pad1)
    float r_s_chain = 0.f;  // motion prior for chain-inconsistent tiles
    uint32_t motion_magnitude_veto_enabled = 0;  // 1 = hard-reject on M veto (was _pad2)
    uint32_t _pad3 = 0;
    uint32_t _pad4 = 0;
};
static_assert(sizeof(RobMaskParamsCPU) == 112, "RobMaskParamsCPU");

struct RobHfLossParamsCPU {
    uint32_t h, w, nch;
    uint32_t _pad0 = 0;
    float alpha = 0.f, beta = 0.f;
    float min_texture_snr = 0.f;
    float _pad1 = 0.f;
};
static_assert(sizeof(RobHfLossParamsCPU) == 32, "RobHfLossParamsCPU");

static bool rob_run_hf_loss(id<MTLBuffer> b_guide, id<MTLBuffer> b_means,
                            id<MTLBuffer> b_vars, __strong id<MTLBuffer>& b_loss,
                            int guide_h, int guide_w, int nch,
                            const Config& cfg, id<MTLCommandBuffer> cmd) {
    auto& c = ctx();
    const size_t guide_b = (size_t)guide_h * (size_t)guide_w * (size_t)nch * sizeof(float);
    const size_t loss_b = (size_t)guide_h * (size_t)guide_w * sizeof(float);
    id<MTLBuffer> b_lp_guide = buf(nullptr, guide_b);
    id<MTLBuffer> b_lp_means = buf(nullptr, guide_b);
    id<MTLBuffer> b_lp_vars = buf(nullptr, guide_b);
    b_loss = buf(nullptr, loss_b);
    if (!b_guide || !b_means || !b_vars || !b_lp_guide ||
        !b_lp_means || !b_lp_vars || !b_loss)
        return false;

    RobStatsParamsCPU sp{};
    sp.h = (uint32_t)guide_h;
    sp.w = (uint32_t)guide_w;
    sp.nch = (uint32_t)nch;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_lp_guide offset:0 atIndex:0];
    [enc setBuffer:b_guide offset:0 atIndex:1];
    [enc setBytes:&sp length:sizeof(sp) atIndex:2];
    dispatch2(enc, c.pipe("rob_lowpass_gaussian5x5"), sp.w, sp.h);
    [enc endEncoding];

    enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_lp_means offset:0 atIndex:0];
    [enc setBuffer:b_lp_vars offset:0 atIndex:1];
    [enc setBuffer:b_lp_guide offset:0 atIndex:2];
    [enc setBytes:&sp length:sizeof(sp) atIndex:3];
    dispatch2(enc, c.pipe("rob_local_stats_3x3"), sp.w, sp.h);
    [enc endEncoding];

    RobHfLossParamsCPU hp{};
    hp.h = (uint32_t)guide_h;
    hp.w = (uint32_t)guide_w;
    hp.nch = (uint32_t)nch;
    hp.alpha = cfg.noise_alpha();
    hp.min_texture_snr = cfg.hf_min_texture_snr;
    hp.beta = cfg.noise_beta();
    enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_loss offset:0 atIndex:0];
    [enc setBuffer:b_means offset:0 atIndex:1];
    [enc setBuffer:b_vars offset:0 atIndex:2];
    [enc setBuffer:b_lp_vars offset:0 atIndex:3];
    [enc setBytes:&hp length:sizeof(hp) atIndex:4];
    dispatch2(enc, c.pipe("rob_hf_loss_adaptive"), hp.w, hp.h);
    [enc endEncoding];
    return true;
}

static std::vector<f32> rob_compute_s(const FlowField& flow, f32 Mt, f32 s1, f32 s2,
                                      std::vector<uint32_t>* irregular_out = nullptr) {
    const f32 inf = std::numeric_limits<f32>::infinity();
    std::vector<f32> S((size_t)flow.ny * (size_t)flow.nx, s2);
    if (irregular_out) irregular_out->assign((size_t)flow.ny * (size_t)flow.nx, 0u);
    // Prefer the measurement taken on the alignment grid -- see compute_s in
    // robustness.cpp, which this mirrors.
    if (flow.has_motion_prior()) {
        for (size_t i = 0; i < S.size(); ++i) {
            const bool irregular = flow.motion_irregular[i] != 0u;
            S[i] = irregular ? s1 : s2;
            if (irregular_out) (*irregular_out)[i] = irregular ? 1u : 0u;
        }
        return S;
    }
    for (int ty = 0; ty < flow.ny; ++ty) {
        for (int tx = 0; tx < flow.nx; ++tx) {
            f32 mnx = inf, mny = inf, mxx = -inf, mxy = -inf;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int yy = ty + i, xx = tx + j;
                    if (yy < 0 || yy >= flow.ny || xx < 0 || xx >= flow.nx) continue;
                    f32 fx = flow.dx(yy, xx), fy = flow.dy(yy, xx);
                    mnx = std::min(mnx, fx);
                    mxx = std::max(mxx, fx);
                    mny = std::min(mny, fy);
                    mxy = std::max(mxy, fy);
                }
            }
            f32 d0 = mxx - mnx, d1 = mxy - mny;
            const bool irregular = d0 * d0 + d1 * d1 > Mt * Mt;
            const size_t idx = (size_t)ty * (size_t)flow.nx + (size_t)tx;
            S[idx] = irregular ? s1 : s2;
            if (irregular_out) (*irregular_out)[idx] = irregular ? 1u : 0u;
        }
    }
    return S;
}

static bool rob_run_guide_stats(const Image& raw, const Config& cfg,
                                __strong id<MTLBuffer>& b_guide,
                                __strong id<MTLBuffer>& b_means,
                                __strong id<MTLBuffer>& b_vars,
                                int& guide_h, int& guide_w, int& nch,
                                id<MTLCommandBuffer> cmd) {
    auto& c = ctx();
    const bool bayer = cfg.bayer_mode;
    guide_h = bayer ? raw.h / 2 : raw.h;
    guide_w = bayer ? raw.w / 2 : raw.w;
    nch = bayer ? 3 : 1;
    if (guide_h < 1 || guide_w < 1) return false;

    const size_t raw_b = raw.data.size() * sizeof(float);
    const size_t guide_b = (size_t)guide_h * (size_t)guide_w * (size_t)nch * sizeof(float);
    id<MTLBuffer> b_raw = buf(raw.data.data(), raw_b);
    b_guide = buf(nullptr, guide_b);
    b_means = buf(nullptr, guide_b);
    b_vars = buf(nullptr, guide_b);
    if (!b_raw || !b_guide || !b_means || !b_vars) return false;

    if (bayer) {
        RobGuideParamsCPU gp{};
        gp.raw_h = (uint32_t)raw.h;
        gp.raw_w = (uint32_t)raw.w;
        gp.guide_h = (uint32_t)guide_h;
        gp.guide_w = (uint32_t)guide_w;
        gp.bayer = 1u;
        gp.cfa00 = cfg.cfa.p[0][0];
        gp.cfa01 = cfg.cfa.p[0][1];
        gp.cfa10 = cfg.cfa.p[1][0];
        gp.cfa11 = cfg.cfa.p[1][1];
        gp.wb0 = cfg.white_balance[0];
        gp.wb1 = cfg.white_balance[1];
        gp.wb2 = cfg.white_balance[2];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return false;
        [enc setBuffer:b_guide offset:0 atIndex:0];
        [enc setBuffer:b_raw offset:0 atIndex:1];
        [enc setBytes:&gp length:sizeof(gp) atIndex:2];
        dispatch2(enc, c.pipe("rob_guide_bayer"), gp.guide_w, gp.guide_h);
        [enc endEncoding];
    } else {
        memcpy([b_guide contents], raw.data.data(), guide_b);
    }

    RobStatsParamsCPU sp{};
    sp.h = (uint32_t)guide_h;
    sp.w = (uint32_t)guide_w;
    sp.nch = (uint32_t)nch;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_means offset:0 atIndex:0];
    [enc setBuffer:b_vars offset:0 atIndex:1];
    [enc setBuffer:b_guide offset:0 atIndex:2];
    [enc setBytes:&sp length:sizeof(sp) atIndex:3];
    dispatch2(enc, c.pipe("rob_local_stats_3x3"), sp.w, sp.h);
    [enc endEncoding];
    return true;
}

static bool rob_dogson(id<MTLBuffer> b_in, __strong id<MTLBuffer>& b_out,
                       int in_h, int in_w, int nch, bool is_ref,
                       const FlowField* flow, int tile_size,
                       int& out_h, int& out_w, id<MTLCommandBuffer> cmd) {
    auto& c = ctx();
    out_h = (nch == 3) ? in_h * 2 : in_h;
    out_w = (nch == 3) ? in_w * 2 : in_w;
    const size_t out_b = (size_t)out_h * (size_t)out_w * (size_t)nch * sizeof(float);
    b_out = buf(nullptr, out_b);
    if (!b_out) return false;

    RobDogsonParamsCPU dp{};
    dp.in_h = (uint32_t)in_h;
    dp.in_w = (uint32_t)in_w;
    dp.out_h = (uint32_t)out_h;
    dp.out_w = (uint32_t)out_w;
    dp.nch = (uint32_t)nch;
    dp.is_ref = is_ref ? 1u : 0u;
    dp.tile_size = (uint32_t)std::max(0, tile_size);
    dp.flow_ny = (!is_ref && flow) ? (uint32_t)flow->ny : 0u;
    dp.flow_nx = (!is_ref && flow) ? (uint32_t)flow->nx : 0u;
    dp.s = 2.f;

    id<MTLBuffer> b_flow = nil;
    if (!is_ref && flow && !flow->flow.empty()) {
        b_flow = buf(flow->flow.data(), flow->flow.size() * sizeof(float));
        if (!b_flow) return false;
    } else {
        static float kDummyFlow[2] = {0.f, 0.f};
        b_flow = buf(kDummyFlow, sizeof(kDummyFlow));
        if (!b_flow) return false;
    }

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_out offset:0 atIndex:0];
    [enc setBuffer:b_in offset:0 atIndex:1];
    [enc setBuffer:b_flow offset:0 atIndex:2];
    [enc setBytes:&dp length:sizeof(dp) atIndex:3];
    dispatch2(enc, c.pipe("rob_upscale_dogson"), dp.out_w, dp.out_h);
    [enc endEncoding];
    return true;
}

} // namespace

// Sticky ref means/vars for compute_robustness_metal (avoid re-upload + free host).
static id<MTLBuffer> g_rob_ref_hf = nil;
static size_t g_rob_ref_hf_bytes = 0;
static id<MTLBuffer> g_rob_ref_m = nil;
static id<MTLBuffer> g_rob_ref_v = nil;
static int g_rob_ref_h = 0, g_rob_ref_w = 0, g_rob_ref_c = 0;
static size_t g_rob_ref_bytes = 0;
// std/diff curve buffers hold up to 3 CONCATENATED per-channel curves
// (each g_rob_curve_n entries; channel ch's curve starts at ch*g_rob_curve_n)
// rather than one curve shared by every guide channel -- see
// Config::noise_alpha_ch/noise_beta_ch and CPU's make_noise_curves_channel,
// which this mirrors. rob_make_mask indexes std_curve[ch*p.curve_n + id].
static id<MTLBuffer> g_rob_std_curve = nil;
static id<MTLBuffer> g_rob_diff_curve = nil;
static size_t g_rob_curve_n = 0;
static float g_rob_curve_alpha[3] = {std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN()};
static float g_rob_curve_beta[3]  = {std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN()};
static bool g_rob_curve_pixel4a = false;
static int g_rob_curve_pixel4a_iso = 0;

static void clear_rob_ref_gpu() {
    g_rob_ref_hf = nil;
    g_rob_ref_hf_bytes = 0;
    g_rob_ref_m = nil;
    g_rob_ref_v = nil;
    g_rob_ref_h = g_rob_ref_w = g_rob_ref_c = 0;
    g_rob_ref_bytes = 0;
    g_rob_std_curve = nil;
    g_rob_diff_curve = nil;
    g_rob_curve_n = 0;
    for (int i = 0; i < 3; ++i) {
        g_rob_curve_alpha[i] = std::numeric_limits<float>::quiet_NaN();
        g_rob_curve_beta[i]  = std::numeric_limits<float>::quiet_NaN();
    }
    g_rob_curve_pixel4a = false;
    g_rob_curve_pixel4a_iso = 0;
}

static RefStats init_robustness_metal_impl(const Image& ref_raw, const Config& cfg) {
    if (!metal_gpu_init() || ref_raw.h <= 0 || ref_raw.w <= 0) return RefStats();
    auto& c = ctx();
    clear_rob_ref_gpu();
    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return RefStats();

    id<MTLBuffer> b_guide = nil, b_means = nil, b_vars = nil;
    int gh = 0, gw = 0, nch = 0;
    if (!rob_run_guide_stats(ref_raw, cfg, b_guide, b_means, b_vars, gh, gw, nch, cmd))
        return RefStats();

    // Must be encoded before the commit below: a committed MTLCommandBuffer
    // cannot accept further encoders, and doing so is a hard error rather than
    // a silent no-op. This block previously sat after the commit, which crashed
    // as soon as high-frequency rejection was switched on.
    id<MTLBuffer> b_ref_hf_loss = nil;
    if (cfg.hf_artifact_removal_enabled &&
        !rob_run_hf_loss(b_guide, b_means, b_vars, b_ref_hf_loss, gh, gw, nch, cfg, cmd))
        return RefStats();

    // No CPU readback follows. Commit without waiting; subsequent work on the
    // same queue observes these pinned buffers after this command completes.
    [cmd commit];

    g_rob_ref_hf = b_ref_hf_loss;
    g_rob_ref_hf_bytes = b_ref_hf_loss
        ? (size_t)gh * (size_t)gw * sizeof(float)
        : 0;

    // Pin guide-grid local stats for all comparison frames (460-main robustness).
    g_rob_ref_m = b_means;
    g_rob_ref_v = b_vars;
    g_rob_ref_h = gh;
    g_rob_ref_w = gw;
    g_rob_ref_c = nch;
    g_rob_ref_bytes = (size_t)gh * (size_t)gw * (size_t)nch * sizeof(float);

    RefStats st;
    // Keep only dimensions on the CPU. The actual reference means/vars/HF loss
    // stay resident in pinned Metal buffers above; copying them back here is an
    // avoidable readback and peak-RAM spike on full-res bursts.
    st.means.h = gh;
    st.means.w = gw;
    st.means.c = nch;
    st.stds.h = gh;
    st.stds.w = gw;
    st.stds.c = nch;
    return st;
}

void metal_release_host_ref_stats(RefStats& ref_stats) {
    // Keep h/w/c for dimension checks; drop ~2× full-res 3ch float host copies.
    const int mh = ref_stats.means.h, mw = ref_stats.means.w, mc = ref_stats.means.c;
    const int sh = ref_stats.stds.h, sw = ref_stats.stds.w, sc = ref_stats.stds.c;
    ref_stats.means = Image();
    ref_stats.stds = Image();
    ref_stats.means.h = mh;
    ref_stats.means.w = mw;
    ref_stats.means.c = mc;
    ref_stats.stds.h = sh;
    ref_stats.stds.w = sw;
    ref_stats.stds.c = sc;
}

static Image compute_robustness_metal_impl(const Image& comp_raw, const RefStats& ref_stats,
                                           const FlowField& flow, int tile_size, const Config& cfg,
                                           Image* s_select_out) {
    if (!metal_gpu_init() || comp_raw.h <= 0 || comp_raw.w <= 0) return Image();
    if (ref_stats.means.h <= 0 || ref_stats.means.w <= 0) return Image();
    auto& c = ctx();

    std::vector<uint32_t> motion_irregular;
    std::vector<f32> S = rob_compute_s(flow, cfg.r_Mt, cfg.r_s1, cfg.r_s2,
                                       (cfg.motion_edge_rejection_enabled ||
                                        cfg.hf_artifact_removal_enabled)
                                           ? &motion_irregular
                                           : nullptr);
    if (!cfg.motion_edge_rejection_enabled && !cfg.hf_artifact_removal_enabled)
        motion_irregular.assign(S.size(), 0u);

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return Image();

    id<MTLBuffer> b_guide = nil, b_gmeans = nil, b_gvars = nil;
    int gh = 0, gw = 0, nch = 0;
    if (!rob_run_guide_stats(comp_raw, cfg, b_guide, b_gmeans, b_gvars, gh, gw, nch, cmd))
        return Image();

    if (gh != ref_stats.means.h || gw != ref_stats.means.w || nch != ref_stats.means.c)
        return Image();

    const size_t ref_b = (size_t)gh * (size_t)gw * (size_t)nch * sizeof(float);
    const size_t mask_b = (size_t)gh * (size_t)gw * sizeof(float);
    id<MTLBuffer> b_ref_m = nil;
    id<MTLBuffer> b_ref_v = nil;
    if (g_rob_ref_m && g_rob_ref_v && g_rob_ref_bytes == ref_b &&
        g_rob_ref_h == gh && g_rob_ref_w == gw && g_rob_ref_c == nch) {
        b_ref_m = g_rob_ref_m;
        b_ref_v = g_rob_ref_v;
    } else if (!ref_stats.means.data.empty() && !ref_stats.stds.data.empty()) {
        b_ref_m = buf(ref_stats.means.data.data(), ref_b);
        b_ref_v = buf(ref_stats.stds.data.data(), ref_b);
    } else {
        return Image();
    }

    // nch guide channels -> nch curves fetched (3 for Bayer, 1 otherwise),
    // each channel scored against its own instead of all sharing the
    // cross-channel-mean curve. Cache key is per-channel alpha/beta so a
    // change in any one channel's WB-derived value rebuilds all of them
    // (curves are concatenated into one buffer, so a partial rebuild isn't
    // meaningfully cheaper).
    const int curve_nch = std::max(1, std::min(3, nch));
    bool curves_stale = !g_rob_std_curve || !g_rob_diff_curve ||
        g_rob_curve_pixel4a != cfg.debug_pixel4a_noise_profile ||
        g_rob_curve_pixel4a_iso != cfg.debug_pixel4a_noise_curve_iso;
    for (int ch = 0; ch < curve_nch && !curves_stale; ++ch) {
        const f32 a = (curve_nch == 3) ? cfg.noise_alpha_ch(ch) : cfg.noise_alpha();
        const f32 b = (curve_nch == 3) ? cfg.noise_beta_ch(ch) : cfg.noise_beta();
        if (g_rob_curve_alpha[ch] != a || g_rob_curve_beta[ch] != b) curves_stale = true;
    }
    if (curves_stale) {
        std::vector<f32> std_all, diff_all;
        size_t n = 0;
        for (int ch = 0; ch < curve_nch; ++ch) {
            std::vector<f32> std_curve, diff_curve;
            if (curve_nch == 3)
                fetch_noise_curves_channel(cfg, ch, std_curve, diff_curve);
            else
                fetch_noise_curves(cfg, std_curve, diff_curve);
            if (std_curve.empty() || diff_curve.empty()) return Image();
            if (ch == 0) {
                n = std_curve.size();
                std_all.resize(n * (size_t)curve_nch);
                diff_all.resize(n * (size_t)curve_nch);
            } else if (std_curve.size() != n || diff_curve.size() != n) {
                return Image();
            }
            std::copy(std_curve.begin(), std_curve.end(), std_all.begin() + (size_t)ch * n);
            std::copy(diff_curve.begin(), diff_curve.end(), diff_all.begin() + (size_t)ch * n);
            g_rob_curve_alpha[ch] = (curve_nch == 3) ? cfg.noise_alpha_ch(ch) : cfg.noise_alpha();
            g_rob_curve_beta[ch] = (curve_nch == 3) ? cfg.noise_beta_ch(ch) : cfg.noise_beta();
        }
        g_rob_std_curve = buf(std_all.data(), std_all.size() * sizeof(float));
        g_rob_diff_curve = buf(diff_all.data(), diff_all.size() * sizeof(float));
        g_rob_curve_n = n;
        g_rob_curve_pixel4a = cfg.debug_pixel4a_noise_profile;
        g_rob_curve_pixel4a_iso = cfg.debug_pixel4a_noise_curve_iso;
    }
    if (g_rob_curve_n == 0) return Image();
    id<MTLBuffer> b_std = g_rob_std_curve;
    id<MTLBuffer> b_diff = g_rob_diff_curve;
    id<MTLBuffer> b_S = buf(S.data(), S.size() * sizeof(float));
    id<MTLBuffer> b_motion = buf(motion_irregular.data(), motion_irregular.size() * sizeof(uint32_t));
    const size_t n_tiles = (size_t)std::max(0, flow.ny) * (size_t)std::max(0, flow.nx);
    const bool aperture_reject_on =
        cfg.flow_reject_1d_enabled &&
        flow.aperture_limited.size() == n_tiles;
    id<MTLBuffer> b_aperture = aperture_reject_on
        ? buf(flow.aperture_limited.data(), flow.aperture_limited.size() * sizeof(uint32_t))
        : b_motion;
    id<MTLBuffer> b_tile_residual_high = aperture_reject_on
        ? buf(nullptr, n_tiles * sizeof(uint32_t))
        : b_motion;

    const size_t hf_b = (size_t)gh * (size_t)gw * sizeof(float);
    id<MTLBuffer> b_ref_hf = b_ref_v;
    if (cfg.hf_artifact_removal_enabled) {
        if (!g_rob_ref_hf || g_rob_ref_hf_bytes != hf_b) return Image();
        b_ref_hf = g_rob_ref_hf;
    }
    id<MTLBuffer> b_flow = buf(flow.flow.data(), flow.flow.size() * sizeof(float));
    id<MTLBuffer> b_R = buf(nullptr, mask_b);
    id<MTLBuffer> b_out = buf(nullptr, mask_b);
    // Bound unconditionally because the kernel declares it; sized for real only
    // when asked, so the off case costs 4 bytes rather than a full mask plane.
    const bool want_s_select = (s_select_out != nullptr);
    id<MTLBuffer> b_s_select = buf(nullptr, want_s_select ? mask_b : sizeof(float));
    if (!b_ref_m || !b_ref_v || !b_std || !b_diff ||
        !b_S || !b_motion || !b_aperture || !b_tile_residual_high || !b_flow ||
        !b_R || !b_out || !b_s_select)
        return Image();

    RobMaskParamsCPU mp{};
    mp.h = (uint32_t)gh;
    mp.w = (uint32_t)gw;
    mp.nch = (uint32_t)nch;
    mp.tile_size = (uint32_t)tile_size;
    mp.flow_ny = (uint32_t)flow.ny;
    mp.flow_nx = (uint32_t)flow.nx;
    mp.curve_n = (uint32_t)g_rob_curve_n;
    mp.bayer = cfg.bayer_mode ? 1u : 0u;
    mp.r_t = cfg.r_t;
    mp.hf_enabled = cfg.hf_artifact_removal_enabled ? 1u : 0u;
    mp.hf_variance_loss_threshold = cfg.hf_variance_loss_threshold;
    mp.motion_edge_enabled = cfg.motion_edge_rejection_enabled ? 1u : 0u;
    mp.motion_edge_threshold = cfg.motion_edge_threshold;
    mp.motion_edge_residual_threshold = cfg.motion_edge_residual_threshold;
    mp.alpha = cfg.noise_alpha();
    mp.beta = cfg.noise_beta();
    mp.motion_edge_noise_floor_multiplier = cfg.motion_edge_noise_floor_multiplier;
    mp.motion_edge_neighborhood_radius =
        (uint32_t)std::max(0, std::min(2, cfg.motion_edge_neighborhood_radius));
    mp.flow_reject_1d_residual_threshold = cfg.flow_reject_1d_residual_threshold;
    mp.aperture_reject_enabled = aperture_reject_on ? 1u : 0u;
    mp.r_s1 = cfg.r_s1;
    mp.save_s_select = want_s_select ? 1u : 0u;
    const bool amb_on = cfg.flow_reject_ambiguous_enabled &&
                        flow.match_ambiguous.size() == n_tiles;
    mp.ambiguous_enabled = amb_on ? 1u : 0u;
    id<MTLBuffer> b_match_amb = amb_on
        ? buf(flow.match_ambiguous.data(), flow.match_ambiguous.size() * sizeof(uint32_t))
        : b_motion;
    if (!b_match_amb) return Image();

    const bool chain_on = cfg.chain_consistency_enabled &&
                          flow.chain_inconsistent.size() == n_tiles;
    mp.chain_reject_enabled = chain_on ? 1u : 0u;
    mp.r_s_chain = cfg.r_s_chain;
    id<MTLBuffer> b_chain = chain_on
        ? buf(flow.chain_inconsistent.data(), flow.chain_inconsistent.size() * sizeof(uint32_t))
        : b_motion;
    if (!b_chain) return Image();

    const bool magnitude_veto_on = cfg.motion_magnitude_veto_enabled &&
                                   flow.motion_magnitude_reject.size() == n_tiles;
    mp.motion_magnitude_veto_enabled = magnitude_veto_on ? 1u : 0u;
    id<MTLBuffer> b_magnitude = magnitude_veto_on
        ? buf(flow.motion_magnitude_reject.data(),
              flow.motion_magnitude_reject.size() * sizeof(uint32_t))
        : b_motion;
    if (!b_magnitude) return Image();

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return Image();
    if (aperture_reject_on) {
        [enc setBuffer:b_tile_residual_high offset:0 atIndex:0];
        [enc setBuffer:b_gmeans offset:0 atIndex:1];
        [enc setBuffer:b_ref_m offset:0 atIndex:2];
        [enc setBuffer:b_ref_v offset:0 atIndex:3];
        [enc setBuffer:b_std offset:0 atIndex:4];
        [enc setBuffer:b_diff offset:0 atIndex:5];
        [enc setBuffer:b_flow offset:0 atIndex:6];
        [enc setBytes:&mp length:sizeof(mp) atIndex:7];
        dispatch2(enc, c.pipe("rob_tile_residual_high"), mp.flow_nx, mp.flow_ny);
        [enc endEncoding];

        enc = [cmd computeCommandEncoder];
        if (!enc) return Image();
    }
    [enc setBuffer:b_R offset:0 atIndex:0];
    [enc setBuffer:b_gmeans offset:0 atIndex:1];
    [enc setBuffer:b_ref_m offset:0 atIndex:3];
    [enc setBuffer:b_ref_v offset:0 atIndex:4];
    [enc setBuffer:b_std offset:0 atIndex:6];
    [enc setBuffer:b_diff offset:0 atIndex:7];
    [enc setBuffer:b_S offset:0 atIndex:8];
    [enc setBuffer:b_motion offset:0 atIndex:9];
    [enc setBuffer:b_ref_hf offset:0 atIndex:5];
    [enc setBuffer:b_flow offset:0 atIndex:10];
    [enc setBytes:&mp length:sizeof(mp) atIndex:11];
    [enc setBuffer:b_aperture offset:0 atIndex:12];
    [enc setBuffer:b_tile_residual_high offset:0 atIndex:13];
    [enc setBuffer:b_s_select offset:0 atIndex:14];
    [enc setBuffer:b_match_amb offset:0 atIndex:15];
    [enc setBuffer:b_chain offset:0 atIndex:16];
    [enc setBuffer:b_magnitude offset:0 atIndex:17];
    dispatch2(enc, c.pipe("rob_make_mask"), mp.w, mp.h);
    [enc endEncoding];

    RobStatsParamsCPU sp{};
    sp.h = (uint32_t)gh;
    sp.w = (uint32_t)gw;
    sp.nch = 1u;
    sp.radius = (uint32_t)std::max(0, cfg.robustness_min_pool_radius);
    enc = [cmd computeCommandEncoder];
    if (!enc) return Image();
    [enc setBuffer:b_out offset:0 atIndex:0];
    [enc setBuffer:b_R offset:0 atIndex:1];
    [enc setBytes:&sp length:sizeof(sp) atIndex:2];
    dispatch2(enc, c.pipe("rob_local_min_5x5"), sp.w, sp.h);
    [enc endEncoding];

    prof_tag_gpu(cmd, "robustness:all");
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return Image();

    Image r(gh, gw, 1);
    memcpy(r.data.data(), [b_out contents], mask_b);
    // Taken from rob_make_mask's output, not rob_local_min_5x5's: the selector
    // is a per-pixel record of which prior was applied, and eroding it would
    // smear the boundary between the two regions.
    if (want_s_select) {
        *s_select_out = Image(gh, gw, 1);
        memcpy(s_select_out->data.data(), [b_s_select contents], mask_b);
    }
    return r;
}

namespace {

struct PyrDownParamsCPU {
    uint32_t in_h, in_w, out_h, out_w, klen, factor;
    uint32_t _pad0 = 0, _pad1 = 0;
};
static_assert(sizeof(PyrDownParamsCPU) == 32, "PyrDownParamsCPU");

static std::vector<float> scipy_gaussian_kernel1d_metal(float sigma, int radius) {
    std::vector<float> k(2 * radius + 1);
    float sigma2 = sigma * sigma;
    float sum = 0.f;
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-0.5f / sigma2 * (float)(i * i));
        k[(size_t)(i + radius)] = v;
        sum += v;
    }
    for (float& v : k) v /= sum;
    return k;
}

// GPU valid-gauss + stride. Same math as grey_pyramid.cpp downsample_by.
// __strong out-param: ARC requires it for id& (same as merge robustness helpers).
// Align stages commit without stalling the CPU. Command buffers on one queue
// execute in commit order, so waiting on the most recently committed one implies
// every earlier stage finished. Each stage previously did commit +
// waitUntilCompleted: about 14 blocking round trips per comparison frame.
//
// Safe only because nothing in the align path shares a buffer across stages any
// more. Metal tracks hazards within a command buffer but not across them, and an
// earlier attempt at this raced against pooled downsample scratch.
static __strong id<MTLCommandBuffer> g_align_last_cmd = nil;

static void align_commit(id<MTLCommandBuffer> cmd) {
    [cmd commit];
    g_align_last_cmd = cmd;
}

// Wait for all committed align work. Required before any CPU read of a buffer
// the align stages wrote.
static bool align_drain() {
    if (!g_align_last_cmd) return true;
    [g_align_last_cmd waitUntilCompleted];
    const bool ok = (g_align_last_cmd.status == MTLCommandBufferStatusCompleted);
    g_align_last_cmd = nil;
    return ok;
}

static bool gpu_downsample_buf(id<MTLBuffer> src, int sh, int sw, int factor,
                               __strong id<MTLBuffer>& dst, int& dh, int& dw) {
    if (factor <= 1) {
        dst = src;
        dh = sh;
        dw = sw;
        return src != nil;
    }
    auto& c = ctx();
    float sigma = 0.5f * (float)factor;
    int radius = (int)(4.f * sigma + 0.5f);
    std::vector<float> ker = scipy_gaussian_kernel1d_metal(sigma, radius);
    const int klen = (int)ker.size();
    int tmp_h = sh - klen + 1;
    int tmp_w = sw;
    if (tmp_h < 1 || tmp_w < 1) return false;
    int filt_h = tmp_h;
    int filt_w = tmp_w - klen + 1;
    if (filt_w < 1) return false;
    dh = filt_h / factor;
    dw = filt_w / factor;
    if (dh < 1 || dw < 1) return false;

    id<MTLBuffer> b_ker = buf(ker.data(), ker.size() * sizeof(float));
    id<MTLBuffer> b_tmp = buf(nullptr, (size_t)tmp_h * tmp_w * sizeof(float));
    id<MTLBuffer> b_filt = buf(nullptr, (size_t)filt_h * filt_w * sizeof(float));
    id<MTLBuffer> b_out = buf(nullptr, (size_t)dh * dw * sizeof(float));
    if (!b_ker || !b_tmp || !b_filt || !b_out) return false;

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;

    PyrDownParamsCPU py{};
    py.in_h = (uint32_t)sh;
    py.in_w = (uint32_t)sw;
    py.out_h = (uint32_t)tmp_h;
    py.out_w = (uint32_t)tmp_w;
    py.klen = (uint32_t)klen;
    py.factor = (uint32_t)factor;
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:b_tmp offset:0 atIndex:1];
    [enc setBuffer:b_ker offset:0 atIndex:2];
    [enc setBytes:&py length:sizeof(py) atIndex:3];
    dispatch2(enc, c.pipe("pyr_conv_y"), (NSUInteger)tmp_w, (NSUInteger)tmp_h);

    PyrDownParamsCPU px = py;
    px.in_h = (uint32_t)tmp_h;
    px.in_w = (uint32_t)tmp_w;
    px.out_h = (uint32_t)filt_h;
    px.out_w = (uint32_t)filt_w;
    [enc setBuffer:b_tmp offset:0 atIndex:0];
    [enc setBuffer:b_filt offset:0 atIndex:1];
    [enc setBuffer:b_ker offset:0 atIndex:2];
    [enc setBytes:&px length:sizeof(px) atIndex:3];
    dispatch2(enc, c.pipe("pyr_conv_x"), (NSUInteger)filt_w, (NSUInteger)filt_h);

    PyrDownParamsCPU ps{};
    ps.in_h = (uint32_t)filt_h;
    ps.in_w = (uint32_t)filt_w;
    ps.out_h = (uint32_t)dh;
    ps.out_w = (uint32_t)dw;
    ps.klen = (uint32_t)klen;
    ps.factor = (uint32_t)factor;
    [enc setBuffer:b_filt offset:0 atIndex:0];
    [enc setBuffer:b_out offset:0 atIndex:1];
    [enc setBytes:&ps length:sizeof(ps) atIndex:3];
    dispatch2(enc, c.pipe("pyr_subsample"), (NSUInteger)dw, (NSUInteger)dh);
    [enc endEncoding];
    prof_tag_gpu(cmd, "align:downsample");
    align_commit(cmd);
    dst = b_out;
    return true;
}

static bool l2_bufs(id<MTLBuffer> ref_img, id<MTLBuffer> mov_img, id<MTLBuffer> flow_b,
                    int ref_h, int ref_w, int mov_h, int mov_w,
                    int tile_size, int search_radius, int ny, int nx) {
    if (ny <= 0 || nx <= 0) return true;
    const int ts = tile_size, R = search_radius;
    const int N = 2 * R + ts;
    const int wh = N / 2 + 1;
    const uint32_t ntiles = (uint32_t)(ny * nx);
    const bool bluestein = (N & (N - 1)) != 0;
    __strong id<MTLCommandBuffer> pending[2] = {nil, nil};
    int slot = 0;
    auto wait_slot = [&](int s) -> bool {
        if (!pending[s]) return true;
        [pending[s] waitUntilCompleted];
        const bool ok = (pending[s].status == MTLCommandBufferStatusCompleted);
        pending[s] = nil;
        return ok;
    };
    for (uint32_t t0 = 0; t0 < ntiles; t0 += kL2TileChunk) {
        uint32_t tc = std::min(kL2TileChunk, ntiles - t0);
        if (bluestein) {
            if (!wait_slot(0) || !wait_slot(1)) return false;
        } else if (!wait_slot(slot)) {
            return false;
        }
        if (!l2_chunk(ref_img, mov_img, flow_b, ref_h, ref_w, mov_h, mov_w,
                      ts, R, N, wh, (uint32_t)ny, (uint32_t)nx, t0, tc,
                      slot, pending[slot]))
            return false;
        slot ^= 1;
    }
    return wait_slot(0) && wait_slot(1);
}

static bool local_search_460_bufs(id<MTLBuffer> b_ref, id<MTLBuffer> b_mov,
                                  id<MTLBuffer> b_flow,
                                  int ref_h, int ref_w, int mov_h, int mov_w,
                                  int tile_size, int search_radius,
                                  int ny, int nx, bool l1,
                                  id<MTLBuffer> b_ambiguity = nil,
                                  float ambiguity_ratio = 1.10f,
                                  bool write_ambiguity = false,
                                  bool fallback_on_ambiguous = false) {
    if (search_radius < 0 || tile_size <= 0) return false;
    if (ny <= 0 || nx <= 0) return true;
    auto& c = ctx();
    struct AlignLocalSearch460ParamsCPU {
        uint32_t ny, nx;
        int32_t ts, R;
        int32_t ref_h, ref_w, mov_h, mov_w;
        uint32_t l1;
        float ambiguity_ratio;
        uint32_t write_ambiguity;
        uint32_t fallback_on_ambiguous = 0;  // was _pad0
    };
    static_assert(sizeof(AlignLocalSearch460ParamsCPU) == 48,
                  "AlignLocalSearch460ParamsCPU layout");
    AlignLocalSearch460ParamsCPU p{};
    p.ny = (uint32_t)ny;
    p.nx = (uint32_t)nx;
    p.ts = (int32_t)tile_size;
    p.R = (int32_t)search_radius;
    p.ref_h = (int32_t)ref_h;
    p.ref_w = (int32_t)ref_w;
    p.mov_h = (int32_t)mov_h;
    p.mov_w = (int32_t)mov_w;
    p.l1 = l1 ? 1u : 0u;
    if (!std::isfinite(ambiguity_ratio)) ambiguity_ratio = 1.10f;
    p.ambiguity_ratio = std::max(ambiguity_ratio, 1.0f);
    p.write_ambiguity = (write_ambiguity && b_ambiguity) ? 1u : 0u;
    p.fallback_on_ambiguous = fallback_on_ambiguous ? 1u : 0u;

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    id<MTLComputePipelineState> pipe = c.pipe("align_local_search_460");
    if (!pipe) return false;
    [enc setBuffer:b_ref offset:0 atIndex:0];
    [enc setBuffer:b_mov offset:0 atIndex:1];
    [enc setBuffer:b_flow offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc setBuffer:(b_ambiguity ? b_ambiguity : b_flow) offset:0 atIndex:4];

    // One threadgroup per tile: lanes split the candidate shifts and the kernel
    // stages the reference tile on-chip. `lanes` must be a power of two for the
    // tree reduction, and is capped so the tile plus the two reduction arrays
    // fit the device's threadgroup memory budget.
    // Threadgroup allocations must be 16-byte aligned.
    auto align16 = [](NSUInteger n) -> NSUInteger { return (n + 15u) & ~(NSUInteger)15u; };
    const NSUInteger tile_bytes =
        align16((NSUInteger)tile_size * (NSUInteger)tile_size * sizeof(float));
    const NSUInteger tg_limit = c.device.maxThreadgroupMemoryLength;
    NSUInteger lanes = 64;
    while (lanes > 1 && lanes > pipe.maxTotalThreadsPerThreadgroup) lanes >>= 1;
    auto red_bytes = [&](NSUInteger n) -> NSUInteger {
        return align16(n * sizeof(float)) + align16(n * sizeof(int)) +
               align16(n * sizeof(float)) + align16(n * sizeof(int));
    };
    while (lanes > 1 && tile_bytes + red_bytes(lanes) > tg_limit) lanes >>= 1;
    if (tile_bytes + red_bytes(lanes) > tg_limit) return false;

    [enc setComputePipelineState:pipe];
    [enc setThreadgroupMemoryLength:tile_bytes atIndex:0];
    [enc setThreadgroupMemoryLength:align16(lanes * sizeof(float)) atIndex:1];
    [enc setThreadgroupMemoryLength:align16(lanes * sizeof(int)) atIndex:2];
    [enc setThreadgroupMemoryLength:align16(lanes * sizeof(float)) atIndex:3];
    [enc setThreadgroupMemoryLength:align16(lanes * sizeof(int)) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)ny * (NSUInteger)nx, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
    [enc endEncoding];
    prof_tag_gpu(cmd, l1 ? "align:local-search-L1" : "align:local-search-L2");
    align_commit(cmd);
    return true;
}

static bool l1_bufs(id<MTLBuffer> b_ref, id<MTLBuffer> b_mov, id<MTLBuffer> b_flow,
                    int ref_h, int ref_w, int mov_h, int mov_w,
                    int tile_size, int search_radius, int ny, int nx) {
    if (search_radius < 0) return false;
    if (tile_size == 16) {
        if (2 * search_radius + 16 > 32) return false;
    } else if (tile_size == 32 || tile_size == 64) {
        if (2 * search_radius > 16) return false;
    } else {
        return false;
    }
    if (ny <= 0 || nx <= 0) return true;
    auto& c = ctx();
    struct L1BmParamsCPU {
        uint32_t ref_h, ref_w, mov_h, mov_w;
        uint32_t ny, nx, ts, R;
    };
    static_assert(sizeof(L1BmParamsCPU) == 32, "L1BmParamsCPU layout");
    L1BmParamsCPU p{};
    p.ref_h = (uint32_t)ref_h;
    p.ref_w = (uint32_t)ref_w;
    p.mov_h = (uint32_t)mov_h;
    p.mov_w = (uint32_t)mov_w;
    p.ny = (uint32_t)ny;
    p.nx = (uint32_t)nx;
    p.ts = (uint32_t)tile_size;
    p.R = (uint32_t)search_radius;
    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_ref offset:0 atIndex:0];
    [enc setBuffer:b_mov offset:0 atIndex:1];
    [enc setBuffer:b_flow offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    const char* pipe_name = tile_size == 16 ? "l1_bm_ts16" :
                            tile_size == 32 ? "l1_bm_ts32" : "l1_bm_ts64";
    id<MTLComputePipelineState> pipe = c.pipe(pipe_name);
    if (!pipe) return false;
    [enc setComputePipelineState:pipe];
    const NSUInteger ntiles = (NSUInteger)ny * (NSUInteger)nx;
    NSUInteger tg = std::min(ntiles, (NSUInteger)pipe.maxTotalThreadsPerThreadgroup);
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(ntiles, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    prof_tag_gpu(cmd, "align:l1-block-match");
    align_commit(cmd);
    return true;
}

// Mirrors ica_damp_ratio / ica_max_step in align.cpp; duplicated because those
// are file-local there and this translation unit cannot see them.
static float ica_damp_ratio_metal(const Config& cfg) {
    if (!cfg.ica_regularize_enabled) return 0.f;
    const float r = cfg.flow_regularize_aperture_ratio;
    if (!std::isfinite(r) || r <= 0.f) return 0.f;
    return std::min(r, 1.f);
}
static float ica_max_step_metal(const Config& cfg, int search_radius) {
    if (!cfg.ica_regularize_enabled) return 0.f;
    return (float)std::max(1, search_radius);
}

static bool ica_bufs(id<MTLBuffer> b_ref, id<MTLBuffer> b_gx, id<MTLBuffer> b_gy,
                     id<MTLBuffer> b_hess, id<MTLBuffer> b_mov, id<MTLBuffer> b_flow,
                     int ref_h, int ref_w, int mov_h, int mov_w,
                     int ny, int nx, int tile_size, int n_iter,
                     float damp_ratio, float max_step) {
    if (tile_size != 8 && tile_size != 16 && tile_size != 32 && tile_size != 64)
        return false;
    if (n_iter < 0) return false;
    if (ny <= 0 || nx <= 0) return true;
    auto& c = ctx();
    struct IcaParamsCPU {
        uint32_t ref_h, ref_w, mov_h, mov_w;
        uint32_t ny, nx, ts, n_iter;
        uint32_t clamp_edge;
        float damp_ratio = 0.f;   // LM damping toward this eigenvalue ratio
        float max_step = 0.f;     // per-iteration displacement bound, px
        uint32_t _pad0 = 0;
    };
    static_assert(sizeof(IcaParamsCPU) == 48, "IcaParamsCPU layout");
    IcaParamsCPU p{};
    p.ref_h = (uint32_t)ref_h;
    p.ref_w = (uint32_t)ref_w;
    p.mov_h = (uint32_t)mov_h;
    p.mov_w = (uint32_t)mov_w;
    p.ny = (uint32_t)ny;
    p.nx = (uint32_t)nx;
    p.ts = (uint32_t)tile_size;
    p.n_iter = (uint32_t)n_iter;
    p.clamp_edge = (tile_size == 8) ? 1u : 0u;
    p.damp_ratio = damp_ratio;
    p.max_step = max_step;
    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_ref offset:0 atIndex:0];
    [enc setBuffer:b_gx offset:0 atIndex:1];
    [enc setBuffer:b_gy offset:0 atIndex:2];
    [enc setBuffer:b_hess offset:0 atIndex:3];
    [enc setBuffer:b_mov offset:0 atIndex:4];
    [enc setBuffer:b_flow offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    id<MTLComputePipelineState> pipe = c.pipe("ica_refine_tile");
    if (!pipe) return false;
    [enc setComputePipelineState:pipe];
    // One threadgroup per tile; lanes cover the tile's pixels and drive the
    // butterfly reduction. n_pix (64 or 256 for the ts<=16 path) is a power of
    // two, which the reduction pairing requires.
    const NSUInteger n = (NSUInteger)ny * (NSUInteger)nx;
    const NSUInteger n_pix = (NSUInteger)tile_size * (NSUInteger)tile_size;
    auto align16 = [](NSUInteger v) -> NSUInteger { return (v + 15u) & ~(NSUInteger)15u; };
    const NSUInteger stage_bytes = align16(n_pix * sizeof(float));
    NSUInteger lanes = n_pix;
    while (lanes > 1 && lanes > pipe.maxTotalThreadsPerThreadgroup) lanes >>= 1;
    if (2 * stage_bytes > c.device.maxThreadgroupMemoryLength) return false;

    [enc setThreadgroupMemoryLength:stage_bytes atIndex:0];
    [enc setThreadgroupMemoryLength:stage_bytes atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
    [enc endEncoding];
    prof_tag_gpu(cmd, "align:ica-refine");
    align_commit(cmd);
    return true;
}

} // namespace

bool block_match_level_L2_metal(const Image& ref, const Image& moving,
                                int tile_size, int search_radius,
                                FlowField& flow,
                                float ambiguity_ratio,
                                bool write_ambiguity,
                                bool fallback_on_ambiguous) {
    if (!metal_gpu_init()) return false;
    const int ny = flow.ny, nx = flow.nx;
    if (ny <= 0 || nx <= 0) return true;
    id<MTLBuffer> ref_img = buf(ref.data.data(), ref.data.size() * sizeof(float));
    id<MTLBuffer> mov_img = buf(moving.data.data(), moving.data.size() * sizeof(float));
    id<MTLBuffer> flow_b = buf(flow.flow.data(), flow.flow.size() * sizeof(float));
    if (write_ambiguity)
        flow.match_ambiguous.assign((size_t)ny * (size_t)nx, 0u);
    id<MTLBuffer> b_ambiguity = write_ambiguity
        ? buf(nullptr, flow.match_ambiguous.size() * sizeof(uint32_t))
        : nil;
    if (!ref_img || !mov_img || !flow_b || (write_ambiguity && !b_ambiguity)) return false;
    if (!local_search_460_bufs(ref_img, mov_img, flow_b, ref.h, ref.w,
                               moving.h, moving.w, tile_size, search_radius,
                               ny, nx, false, b_ambiguity,
                               ambiguity_ratio, write_ambiguity,
                               fallback_on_ambiguous))
        return false;
    // Single-stage entry point: the helper above now commits without waiting,
    // so drain before reading the result back to the host.
    if (!align_drain()) return false;
    memcpy(flow.flow.data(), [flow_b contents], flow.flow.size() * sizeof(float));
    if (write_ambiguity)
        memcpy(flow.match_ambiguous.data(), [b_ambiguity contents],
               flow.match_ambiguous.size() * sizeof(uint32_t));
    return true;
}

bool block_match_level_L1_metal(const Image& ref, const Image& moving,
                                int tile_size, int search_radius,
                                FlowField& flow,
                                float ambiguity_ratio,
                                bool write_ambiguity,
                                bool fallback_on_ambiguous) {
    if (!metal_gpu_init()) return false;
    const int ny = flow.ny, nx = flow.nx;
    if (ny <= 0 || nx <= 0) return true;
    id<MTLBuffer> b_ref = buf(ref.data.data(), ref.data.size() * sizeof(float));
    id<MTLBuffer> b_mov = buf(moving.data.data(), moving.data.size() * sizeof(float));
    id<MTLBuffer> b_flow = buf(flow.flow.data(), flow.flow.size() * sizeof(float));
    if (write_ambiguity)
        flow.match_ambiguous.assign((size_t)ny * (size_t)nx, 0u);
    id<MTLBuffer> b_ambiguity = write_ambiguity
        ? buf(nullptr, flow.match_ambiguous.size() * sizeof(uint32_t))
        : nil;
    if (!b_ref || !b_mov || !b_flow || (write_ambiguity && !b_ambiguity)) return false;
    if (!local_search_460_bufs(b_ref, b_mov, b_flow, ref.h, ref.w,
                               moving.h, moving.w, tile_size, search_radius,
                               ny, nx, true, b_ambiguity,
                               ambiguity_ratio, write_ambiguity,
                               fallback_on_ambiguous))
        return false;
    // Single-stage entry point: the helper above now commits without waiting,
    // so drain before reading the result back to the host.
    if (!align_drain()) return false;
    memcpy(flow.flow.data(), [b_flow contents], flow.flow.size() * sizeof(float));
    if (write_ambiguity)
        memcpy(flow.match_ambiguous.data(), [b_ambiguity contents],
               flow.match_ambiguous.size() * sizeof(uint32_t));
    return true;
}

bool ica_refine_level_metal(const Image& ref, const Image& gradx, const Image& grady,
                            const std::vector<float>& hess_packed,
                            const Image& moving, FlowField& flow,
                            int tile_size, int n_iter,
                            float damp_ratio, float max_step) {
    if (!metal_gpu_init()) return false;
    if (tile_size != 8 && tile_size != 16 && tile_size != 32 && tile_size != 64)
        return false;
    if (n_iter < 0) return false;
    const int ny = flow.ny, nx = flow.nx;
    if (ny <= 0 || nx <= 0) return true;
    const size_t ntiles = (size_t)ny * (size_t)nx;
    if (hess_packed.size() < ntiles * 4) return false;
    if (gradx.h != ref.h || gradx.w != ref.w || grady.h != ref.h || grady.w != ref.w)
        return false;
    id<MTLBuffer> b_ref = buf(ref.data.data(), ref.data.size() * sizeof(float));
    id<MTLBuffer> b_gx = buf(gradx.data.data(), gradx.data.size() * sizeof(float));
    id<MTLBuffer> b_gy = buf(grady.data.data(), grady.data.size() * sizeof(float));
    id<MTLBuffer> b_hess = buf(hess_packed.data(), hess_packed.size() * sizeof(float));
    id<MTLBuffer> b_mov = buf(moving.data.data(), moving.data.size() * sizeof(float));
    id<MTLBuffer> b_flow = buf(flow.flow.data(), flow.flow.size() * sizeof(float));
    if (!b_ref || !b_gx || !b_gy || !b_hess || !b_mov || !b_flow) return false;
    if (!ica_bufs(b_ref, b_gx, b_gy, b_hess, b_mov, b_flow,
                  ref.h, ref.w, moving.h, moving.w, ny, nx, tile_size, n_iter,
                  damp_ratio, max_step))
        return false;
    // Single-stage entry point: the helper above now commits without waiting,
    // so drain before reading the result back to the host.
    if (!align_drain()) return false;
    memcpy(flow.flow.data(), [b_flow contents], flow.flow.size() * sizeof(float));
    return true;
}

bool downsample_by_metal(const Image& src, int factor, Image& out) {
    if (!metal_gpu_init() || src.h <= 0 || src.w <= 0) return false;
    if (factor <= 1) {
        out = src;
        return true;
    }
    id<MTLBuffer> b_src = buf(src.data.data(), src.data.size() * sizeof(float));
    if (!b_src) return false;
    id<MTLBuffer> b_dst = nil;
    int dh = 0, dw = 0;
    if (!gpu_downsample_buf(b_src, src.h, src.w, factor, b_dst, dh, dw) || !b_dst)
        return false;
    out = Image(dh, dw, 1);
    // Single-stage entry point: the helper above now commits without waiting,
    // so drain before reading the result back to the host.
    if (!align_drain()) return false;
    memcpy(out.data.data(), [b_dst contents], (size_t)dh * dw * sizeof(float));
    return true;
}

namespace {

struct AlignImgParamsCPU {
    uint32_t h, w, _pad0 = 0, _pad1 = 0;
};
static_assert(sizeof(AlignImgParamsCPU) == 16, "AlignImgParamsCPU");

struct AlignHessParamsCPU {
    uint32_t h, w, ny, nx, ts;
    uint32_t _pad0 = 0, _pad1 = 0, _pad2 = 0;
};
static_assert(sizeof(AlignHessParamsCPU) == 32, "AlignHessParamsCPU");

struct AlignUpscaleParamsCPU {
    uint32_t in_ny, in_nx, target_ny, target_nx;
    uint32_t upsample_factor, repeat_factor, up_ny, up_nx;
    int32_t ts;
    int32_t ref_h, ref_w, mov_h, mov_w;
    uint32_t carry_ambiguity = 0;   // 1 = propagate the block-matching ambiguity flag
};
static_assert(sizeof(AlignUpscaleParamsCPU) == 56, "AlignUpscaleParamsCPU");

// One pyramid level: upload ref, Sobel gx/gy, Hessian. Temps only — do not
// retain all levels at once (full-res sticky cache jetsams on 1×).
static bool prep_level_ica_gpu(const Image& r, int ts,
                               __strong id<MTLBuffer>& b_ref,
                               __strong id<MTLBuffer>& b_gx,
                               __strong id<MTLBuffer>& b_gy,
                               __strong id<MTLBuffer>& b_hess,
                               int& ny, int& nx) {
    auto& c = ctx();
    ny = (r.h + ts - 1) / ts;
    nx = (r.w + ts - 1) / ts;
    if (ny <= 0 || nx <= 0 || r.h <= 0 || r.w <= 0) return false;
    const void* key = r.data.empty() ? nullptr : (const void*)r.data.data();
    // Slots are few, so a linear scan is cheaper than any indexing scheme, and
    // it keeps this function's signature free of a level number it would
    // otherwise only use as a cache key.
    if (key) {
        for (int i = 0; i < MetalCtx::kIcaSlots; ++i) {
            const auto& sl = c.ica[i];
            if (sl.key == key && sl.ref && sl.gx && sl.gy && sl.hess &&
                sl.h == r.h && sl.w == r.w && sl.ts == ts &&
                sl.ny == ny && sl.nx == nx) {
                b_ref = sl.ref;
                b_gx = sl.gx;
                b_gy = sl.gy;
                b_hess = sl.hess;
                return true;
            }
        }
    }
    b_ref = buf(r.data.data(), r.data.size() * sizeof(float));
    const size_t pix_b = (size_t)r.h * (size_t)r.w * sizeof(float);
    const size_t hess_b = (size_t)ny * (size_t)nx * 4u * sizeof(float);
    b_gx = buf(nullptr, pix_b);
    b_gy = buf(nullptr, pix_b);
    b_hess = buf(nullptr, hess_b);
    if (!b_ref || !b_gx || !b_gy || !b_hess) return false;

    id<MTLComputePipelineState> px = c.pipe("align_sobel_x");
    id<MTLComputePipelineState> py = c.pipe("align_sobel_y");
    id<MTLComputePipelineState> ph = c.pipe("align_hessian");
    if (!px || !py || !ph) return false;

    AlignImgParamsCPU ip{};
    ip.h = (uint32_t)r.h;
    ip.w = (uint32_t)r.w;
    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_ref offset:0 atIndex:0];
    [enc setBuffer:b_gx offset:0 atIndex:1];
    [enc setBytes:&ip length:sizeof(ip) atIndex:2];
    dispatch2(enc, px, (NSUInteger)r.w, (NSUInteger)r.h);
    [enc setBuffer:b_ref offset:0 atIndex:0];
    [enc setBuffer:b_gy offset:0 atIndex:1];
    [enc setBytes:&ip length:sizeof(ip) atIndex:2];
    dispatch2(enc, py, (NSUInteger)r.w, (NSUInteger)r.h);

    AlignHessParamsCPU hp{};
    hp.h = (uint32_t)r.h;
    hp.w = (uint32_t)r.w;
    hp.ny = (uint32_t)ny;
    hp.nx = (uint32_t)nx;
    hp.ts = (uint32_t)ts;
    [enc setBuffer:b_gx offset:0 atIndex:0];
    [enc setBuffer:b_gy offset:0 atIndex:1];
    [enc setBuffer:b_hess offset:0 atIndex:2];
    [enc setBytes:&hp length:sizeof(hp) atIndex:3];
    [enc setComputePipelineState:ph];
    const NSUInteger ntiles = (NSUInteger)ny * (NSUInteger)nx;
    NSUInteger tg = std::min(ntiles, (NSUInteger)ph.maxTotalThreadsPerThreadgroup);
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(ntiles, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    prof_tag_gpu(cmd, "align:prep-sobel-hessian");
    align_commit(cmd);
    const bool ok = true;
    if (ok && key) {
        // Reuse the slot already holding this level, else take the next one.
        int slot = -1;
        for (int i = 0; i < MetalCtx::kIcaSlots && slot < 0; ++i)
            if (c.ica[i].key == key && c.ica[i].h == r.h &&
                c.ica[i].w == r.w && c.ica[i].ts == ts)
                slot = i;
        if (slot < 0) {
            slot = c.ica_next;
            c.ica_next = (c.ica_next + 1) % MetalCtx::kIcaSlots;
        }
        auto& sl = c.ica[slot];
        sl.key = key;
        sl.ref = b_ref;
        sl.gx = b_gx;
        sl.gy = b_gy;
        sl.hess = b_hess;
        sl.h = r.h;
        sl.w = r.w;
        sl.ts = ts;
        sl.ny = ny;
        sl.nx = nx;
    }
    return ok;
}

// __strong out-param: ARC requires it for id& (same as gpu_downsample_buf).
static bool upscale_flow_460_bufs(id<MTLBuffer> b_in, int in_ny, int in_nx,
                                  id<MTLBuffer> b_ref, id<MTLBuffer> b_mov,
                                  __strong id<MTLBuffer>& b_out,
                                  int target_ny, int target_nx,
                                  int upsample_factor, int new_tile_size,
                                  int prev_tile_size,
                                  int ref_h, int ref_w, int mov_h, int mov_w,
                                  id<MTLBuffer> b_amb_in = nil,
                                  __strong id<MTLBuffer>* b_amb_out = nullptr) {
    auto& c = ctx();
    int tile_ratio = new_tile_size / std::max(1, prev_tile_size);
    int repeat_factor = upsample_factor / std::max(1, tile_ratio);
    if (repeat_factor < 1) repeat_factor = 1;
    int up_ny = in_ny * repeat_factor;
    int up_nx = in_nx * repeat_factor;
    const size_t out_b = (size_t)target_ny * (size_t)target_nx * 2u * sizeof(float);
    b_out = buf(nullptr, out_b);
    if (!b_out) return false;
    // The grid changes size between levels, so the ambiguity flag needs its own
    // resized buffer here, filled by the kernel from whichever coarse tile
    // supplied the winning candidate.
    const bool carry_amb = (b_amb_in != nil && b_amb_out != nullptr);
    if (carry_amb) {
        *b_amb_out = buf(nullptr, (size_t)target_ny * (size_t)target_nx * sizeof(uint32_t));
        if (!*b_amb_out) return false;
    }

    AlignUpscaleParamsCPU p{};
    p.in_ny = (uint32_t)in_ny;
    p.in_nx = (uint32_t)in_nx;
    p.target_ny = (uint32_t)target_ny;
    p.target_nx = (uint32_t)target_nx;
    p.upsample_factor = (uint32_t)upsample_factor;
    p.repeat_factor = (uint32_t)repeat_factor;
    p.up_ny = (uint32_t)up_ny;
    p.up_nx = (uint32_t)up_nx;
    p.ts = (int32_t)new_tile_size;
    p.ref_h = (int32_t)ref_h;
    p.ref_w = (int32_t)ref_w;
    p.mov_h = (int32_t)mov_h;
    p.mov_w = (int32_t)mov_w;
    p.carry_ambiguity = carry_amb ? 1u : 0u;

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    id<MTLComputePipelineState> pipe = c.pipe("align_upscale_flow_460");
    if (!pipe) return false;
    [enc setBuffer:b_in offset:0 atIndex:0];
    [enc setBuffer:b_ref offset:0 atIndex:1];
    [enc setBuffer:b_mov offset:0 atIndex:2];
    [enc setBuffer:b_out offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    // Bound unconditionally because the kernel declares them; harmless aliases
    // when carry_amb is false, since the kernel guards on p.carry_ambiguity.
    [enc setBuffer:(carry_amb ? b_amb_in : b_in) offset:0 atIndex:5];
    [enc setBuffer:(carry_amb ? *b_amb_out : b_out) offset:0 atIndex:6];
    dispatch2(enc, pipe, (NSUInteger)target_nx, (NSUInteger)target_ny);
    [enc endEncoding];
    prof_tag_gpu(cmd, "align:upscale-flow");
    align_commit(cmd);
    return true;
}

} // namespace

static bool g_dumped_ref_grads = false;

static bool align_metal_impl(const Pyramid& ref_pyr, const Image& ref_grey,
                 const Image& moving_grey,
                 const Config& cfg, int tile_size, FlowField& flow_out,
                 f32 initial_dx, f32 initial_dy, f32 initial_rotation_rad) {
    if (!metal_gpu_init()) return false;
    // Same reason as the grey FFT: the L2 path soft-commits mid-stage.
    ProfStageScope prof_stage("align:soft-segments");
    const int nlev = (int)ref_pyr.levels.size();
    if (nlev <= 0) return false;
    if (ref_grey.h <= 0 || ref_grey.w <= 0 ||
        moving_grey.h <= 0 || moving_grey.w <= 0) return false;
    // Fine-first levels[]: params arrays are fine→coarse, so index with lvl.
    for (int lvl = 0; lvl < nlev; ++lvl) {
        int ts = (lvl < (int)cfg.bm_tile_sizes.size()) ? cfg.bm_tile_sizes[lvl] : tile_size;
        if (ts != 8 && ts != 16 && ts != 32 && ts != 64) return false;
        std::string metric = "L2";
        if (lvl < (int)cfg.bm_metrics.size()) metric = cfg.bm_metrics[lvl];
        int radius = (lvl < (int)cfg.bm_search_radii.size()) ? cfg.bm_search_radii[lvl] : 2;
        if (metric != "L1" && metric != "L2") return false;
        if (radius < 0) return false;
    }

    // Drain any work left in flight by an early return from a previous call.
    (void)align_drain();
    auto& c = ctx();
    Image moving_padded_grey = pad_image_circular(moving_grey, tile_size);
    id<MTLBuffer> mov0 = buf(moving_padded_grey.data.data(),
                             moving_padded_grey.data.size() * sizeof(float));
    if (!mov0) return false;

    struct Lev {
        id<MTLBuffer> img = nil;
        int h = 0, w = 0;
    };
    std::vector<Lev> mov_pyr((size_t)nlev);
    mov_pyr[0] = {mov0, moving_padded_grey.h, moving_padded_grey.w};
    for (int i = 0; i < nlev; ++i) {
        int f = (i < (int)cfg.bm_factors.size()) ? cfg.bm_factors[i] : 1;
        if (i == 0 && f == 1) continue;
        if (i == 0) {
            id<MTLBuffer> dst = nil;
            int dh = 0, dw = 0;
            if (!gpu_downsample_buf(mov0, moving_padded_grey.h, moving_padded_grey.w, f, dst, dh, dw))
                return false;
            mov_pyr[0] = {dst, dh, dw};
        } else {
            id<MTLBuffer> dst = nil;
            int dh = 0, dw = 0;
            const Lev& prev = mov_pyr[(size_t)i - 1];
            if (!gpu_downsample_buf(prev.img, prev.h, prev.w, f, dst, dh, dw))
                return false;
            mov_pyr[(size_t)i] = {dst, dh, dw};
        }
    }
    for (int i = 1; i < nlev; ++i) {
        if (!mov_pyr[(size_t)i].img) return false;
    }

    id<MTLBuffer> b_flow = nil;
    id<MTLBuffer> b_ambiguity = nil;
    int flow_ny = 0, flow_nx = 0;
    int ambiguity_ny = 0, ambiguity_nx = 0;
    const bool want_amb = cfg.flow_reject_ambiguous_enabled;
    const bool want_fallback = cfg.align_ambiguous_fallback_enabled;
    const bool ica_all = cfg.ica_every_level();

    for (int lvl = nlev - 1; lvl >= 0; --lvl) {
        const Image& r = ref_pyr.levels[(size_t)lvl];
        const Lev& m = mov_pyr[(size_t)lvl];
        int ts = (lvl < (int)cfg.bm_tile_sizes.size()) ? cfg.bm_tile_sizes[lvl] : tile_size;
        int radius = (lvl < (int)cfg.bm_search_radii.size()) ? cfg.bm_search_radii[lvl] : 2;

        id<MTLBuffer> b_ref = buf(r.data.data(), r.data.size() * sizeof(float));
        if (!b_ref) return false;
        int ny = r.h / ts;
        int nx = r.w / ts;
        if (ny <= 0 || nx <= 0) return false;

        if (!b_flow) {
            flow_ny = ny;
            flow_nx = nx;
            const int abs_factor = (lvl < (int)ref_pyr.abs_factors.size())
                                   ? ref_pyr.abs_factors[(size_t)lvl] : 1;
            FlowField initial = make_global_initial_flow(ny, nx, ts, abs_factor,
                                                         ref_grey.h, ref_grey.w,
                                                         initial_dx, initial_dy,
                                                         initial_rotation_rad);
            b_flow = buf(initial.flow.data(),
                         (size_t)ny * (size_t)nx * 2u * sizeof(float));
            if (!b_flow) return false;
        } else {
            int upsample_factor = ((lvl + 1) < (int)cfg.bm_factors.size())
                                  ? cfg.bm_factors[lvl + 1] : 1;
            int prev_ts = ((lvl + 1) < (int)cfg.bm_tile_sizes.size())
                          ? cfg.bm_tile_sizes[lvl + 1] : ts;
            id<MTLBuffer> b_up = nil;
            id<MTLBuffer> b_amb_up = nil;
            if (!upscale_flow_460_bufs(b_flow, flow_ny, flow_nx, b_ref, m.img,
                                       b_up, ny, nx, upsample_factor, ts, prev_ts,
                                       r.h, r.w, m.h, m.w,
                                       want_amb ? b_ambiguity : nil,
                                       want_amb ? &b_amb_up : nullptr) || !b_up)
                return false;
            b_flow = b_up;
            if (want_amb && b_amb_up) {
                b_ambiguity = b_amb_up;
                ambiguity_ny = ny;
                ambiguity_nx = nx;
            }
            flow_ny = ny;
            flow_nx = nx;
        }

        // Not reset per level: the flag accumulates from the coarsest level,
        // where a wrong match is 8px x 32 abs factor = 256 raw px of error, down
        // to the finest. The upscale above resized and propagated it, and
        // local_search ORs its own finding into it.
        if (want_amb && (!b_ambiguity || ambiguity_ny != ny || ambiguity_nx != nx)) {
            b_ambiguity = buf(nullptr, (size_t)ny * (size_t)nx * sizeof(uint32_t));
            if (!b_ambiguity) return false;
            ambiguity_ny = ny;
            ambiguity_nx = nx;
        }

        std::string metric = "L2";
        if (lvl < (int)cfg.bm_metrics.size()) metric = cfg.bm_metrics[lvl];
        bool bm_ok = false;
        if (metric == "L1")
            bm_ok = local_search_460_bufs(b_ref, m.img, b_flow, r.h, r.w,
                                          m.h, m.w, ts, radius, ny, nx, true,
                                          b_ambiguity,
                                          cfg.flow_reject_1d_ambiguity_ratio,
                                          want_amb, want_fallback);
        else
            bm_ok = local_search_460_bufs(b_ref, m.img, b_flow, r.h, r.w,
                                          m.h, m.w, ts, radius, ny, nx, false,
                                          b_ambiguity,
                                          cfg.flow_reject_1d_ambiguity_ratio,
                                          want_amb, want_fallback);
        if (!bm_ok) return false;

        // alignment.py align_lvl: block matching, then ICA, on this level --
        // before upscale_flow multiplies whatever error is left by the
        // upsampling factor on the way to the next one.
        if (ica_all && !(lvl == 0 && cfg.ica_per_level_coarse_only())) {
            id<MTLBuffer> l_ref = nil, l_gx = nil, l_gy = nil, l_hess = nil;
            int l_ny = 0, l_nx = 0;
            // Not fatal. A refinement that cannot be prepared costs this level
            // its sub-pixel correction; failing the whole call loses the frame,
            // because pipeline_paths treats a false return as "Metal alignment
            // failed" and skips it. Losing precision beats losing the frame.
            if (!prep_level_ica_gpu(r, ts, l_ref, l_gx, l_gy, l_hess, l_ny, l_nx))
                continue;
            // prep tiles with ceil, block matching with floor; they agree for
            // every shipped pyramid, but skip rather than dispatch a mismatched
            // grid if a future tile size makes them differ.
            if (l_ny == ny && l_nx == nx &&
                !ica_bufs(l_ref, l_gx, l_gy, l_hess, m.img, b_flow,
                          r.h, r.w, m.h, m.w, ny, nx, ts, cfg.ica_n_iter,
                          ica_damp_ratio_metal(cfg), ica_max_step_metal(cfg, radius)))
                return false;
        }
    }

    // No drain here: nothing below reads GPU memory on the CPU until the flow
    // readback, and the ICA stages are ordered behind the pyramid work by the
    // queue. Draining here would stall out most of the async benefit.
    // Full-res FFT grey keeps the single finest-level refinement, unchanged.
    // With per-level ICA the loop already refined level 0 and repeating it here
    // would run ICA twice on the finest scale.
    if (!b_flow || flow_ny <= 0 || flow_nx <= 0) return false;
    // Coarse-only leaves the finest level to this pass, so it must still run.
    if (!ica_all || cfg.ica_per_level_coarse_only()) {
    id<MTLBuffer> b_ref_native = nil, b_gx = nil, b_gy = nil, b_hess = nil;
    int ica_ny = 0, ica_nx = 0;
    if (!prep_level_ica_gpu(ref_grey, tile_size, b_ref_native, b_gx, b_gy, b_hess,
                            ica_ny, ica_nx))
        return false;
    if (ica_ny != flow_ny || ica_nx != flow_nx) return false;
    if (!g_dumped_ref_grads && b_gx && b_gy) {
        debug_dump_bin("cpp_gradx_ica", (const float*)[b_gx contents],
                       (size_t)ref_grey.h * (size_t)ref_grey.w);
        debug_dump_bin("cpp_grady_ica", (const float*)[b_gy contents],
                       (size_t)ref_grey.h * (size_t)ref_grey.w);
        g_dumped_ref_grads = true;
    }
    // Deliberately not reusing c.sticky_grey here. At this commit the grey FFT
    // output buffer is pooled, and handing a pooled buffer to a later stage is
    // the pattern that produced the corrupted pyramid earlier on this branch.
    // A fresh upload costs one copy per frame and removes the hazard entirely.
    id<MTLBuffer> b_mov_native = buf(moving_grey.data.data(),
                                     moving_grey.data.size() * sizeof(float));
    if (!b_mov_native) return false;
    if (!ica_bufs(b_ref_native, b_gx, b_gy, b_hess, b_mov_native, b_flow,
                  ref_grey.h, ref_grey.w, moving_grey.h, moving_grey.w,
                  flow_ny, flow_nx, tile_size, cfg.ica_n_iter,
                  ica_damp_ratio_metal(cfg),
                  ica_max_step_metal(cfg, cfg.bm_search_radii.empty()
                                              ? 1 : cfg.bm_search_radii[0])))
        return false;
    }

    // Single sync point for the whole align before the CPU reads the flow.
    if (!align_drain()) return false;
    flow_out = FlowField(flow_ny, flow_nx);
    memcpy(flow_out.flow.data(), [b_flow contents],
           flow_out.flow.size() * sizeof(float));
    if (want_amb && b_ambiguity &&
        ambiguity_ny == flow_ny && ambiguity_nx == flow_nx) {
        flow_out.match_ambiguous.assign((size_t)flow_ny * (size_t)flow_nx, 0u);
        memcpy(flow_out.match_ambiguous.data(), [b_ambiguity contents],
               flow_out.match_ambiguous.size() * sizeof(uint32_t));
    }
    return true;
}

void metal_clear_ref_ica_cache() {
    auto& c = ctx();
    for (int i = 0; i < MetalCtx::kIcaSlots; ++i) c.ica[i] = {};
    c.ica_next = 0;
    g_dumped_ref_grads = false;
}

bool metal_normalize_band_rgb16(const Image& num_band, const Image& den_band,
                                const Config& cfg, std::vector<uint16_t>& row16) {
    const int bh = num_band.h, Ws = num_band.w, nch = num_band.c;
    if (bh <= 0 || Ws <= 0 || nch < 1) return false;
    if (den_band.h != bh || den_band.w != Ws || den_band.c != nch) return false;
    const size_t n = (size_t)bh * (size_t)Ws * (size_t)nch;
    if (num_band.data.size() < n || den_band.data.size() < n) return false;
    return metal_normalize_band_rgb16_ptr(num_band.data.data(), den_band.data.data(),
                                          bh, Ws, nch, cfg, row16);
}

bool metal_normalize_band_rgb16_ptr(const float* num_p, const float* den_p,
                                    int bh, int Ws, int nch,
                                    const Config& cfg, std::vector<uint16_t>& row16) {
    if (!metal_gpu_init()) return false;
    if (!num_p || !den_p || bh <= 0 || Ws <= 0 || nch < 1) return false;
    const size_t n = (size_t)bh * (size_t)Ws * (size_t)nch;

    row16.resize((size_t)bh * (size_t)Ws * 3u);
    auto& c = ctx();
    struct MergeNormParamsCPU {
        uint32_t bh, Ws, nch, bake;
        float wb0, wb1, wb2;
        float m00, m01, m02, m10, m11, m12, m20, m21, m22;
    };
    static_assert(sizeof(MergeNormParamsCPU) == 64, "MergeNormParamsCPU");
    MergeNormParamsCPU p{};
    p.bh = (uint32_t)bh;
    p.Ws = (uint32_t)Ws;
    p.nch = (uint32_t)nch;
    p.bake = (cfg.bake_srgb && nch >= 3) ? 1u : 0u;
    p.wb0 = cfg.raw_prewhitened ? 1.f : cfg.white_balance[0];
    p.wb1 = cfg.raw_prewhitened ? 1.f : cfg.white_balance[1];
    p.wb2 = cfg.raw_prewhitened ? 1.f : cfg.white_balance[2];
    const float* m = cfg.cam_to_srgb;
    p.m00 = m[0]; p.m01 = m[1]; p.m02 = m[2];
    p.m10 = m[3]; p.m11 = m[4]; p.m12 = m[5];
    p.m20 = m[6]; p.m21 = m[7]; p.m22 = m[8];

    id<MTLBuffer> b_num = buf(num_p, n * sizeof(float));
    id<MTLBuffer> b_den = buf(den_p, n * sizeof(float));
    id<MTLBuffer> b_out = buf(nullptr, row16.size() * sizeof(uint16_t));
    if (!b_num || !b_den || !b_out) return false;

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    if (!cmd) return false;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return false;
    [enc setBuffer:b_num offset:0 atIndex:0];
    [enc setBuffer:b_den offset:0 atIndex:1];
    [enc setBuffer:b_out offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    dispatch2(enc, c.pipe("merge_normalize_rgb16"), (NSUInteger)Ws, (NSUInteger)bh);
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;
    memcpy(row16.data(), [b_out contents], row16.size() * sizeof(uint16_t));
    return true;
}


namespace {

// Must match HHSRKernels.metal MergeCompParams / MergeRefParams layout
// (including 16-byte padding for constant-buffer setBytes).
struct MergeCompParamsCPU {
    uint32_t band_h, Ws, y0, lr_h, lr_w;
    uint32_t rob_h, rob_w;
    uint32_t flow_ny, flow_nx, cov_h, cov_w;
    uint32_t nch, bayer, iso, tile_size;
    float scale;
    uint32_t cfa00, cfa01, cfa10, cfa11;
    uint32_t _pad0 = 0, _pad1 = 0, _pad2 = 0, _pad3 = 0;
};
static_assert(sizeof(MergeCompParamsCPU) == 96, "MergeCompParamsCPU layout");

struct MergeRefParamsCPU {
    uint32_t band_h, Ws, y0, lr_h, lr_w;
    uint32_t cov_h, cov_w, acc_h, acc_w;
    uint32_t nch, bayer, iso, robustness_denoise, rad_max;
    float scale;
    float max_multiplier;
    float burst_frames;
    uint32_t cfa00, cfa01, cfa10, cfa11;
    uint32_t adaptive;
    float max_frame_count;
    uint32_t _pad0 = 0;
};
static_assert(sizeof(MergeRefParamsCPU) == 96, "MergeRefParamsCPU layout");

// Double-buffered GPU accumulators so band N+1 can run while CPU encodes band N.
struct MergeAccSlot {
    id<MTLBuffer> num = nil;
    id<MTLBuffer> den = nil;
    size_t capacity = 0;
    size_t bytes = 0;
};

struct MergeInflight {
    __strong id<MTLCommandBuffer> cmd = nil;
    int slot = -1;
    Image* num = nullptr;
    Image* den = nullptr;
};

// One GPU-resident copy per comparison frame (keeps buffers across bands).
struct MergeFrameGpu {
    int key = -1; // >=0: frame_id; -1: pointer-keyed
    int lr_h = 0, lr_w = 0;
    int rob_h = 0, rob_w = 0;
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

static MergeAccSlot g_merge_acc[2];
static int g_merge_write_slot = 0;
static bool g_merge_need_zero = false;
static bool g_merge_single_slot = false;
// Online merge: one accumulator that outlives its command buffer, so each frame
// can be committed and its GPU inputs released instead of every frame staying
// resident until the last band. Banded mode ties "new command buffer" to "zero
// the accumulator"; online has to separate the two.
static bool g_merge_online = false;
static bool g_merge_online_zeroed = false;
// Online carries its own output geometry, so the caller does not have to hold a
// full-size host image just to describe the accumulator's shape.
static int g_online_h = 0, g_online_w = 0, g_online_nch = 0;
static MergeInflight g_merge_inflight;
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

// Comparison frames are batched so a group shares one read-modify-write of the
// band accumulators instead of one each. See merge_accumulate_comp_x4: the
// addition order is unchanged, so this is bit-identical, it just stops pushing
// the accumulators through DRAM once per frame.
//
// 4 keeps the binding count at 20, well inside the limit, and already removes
// most of the traffic: 7 frames go from 7 accumulator round trips to 2.
static constexpr uint32_t kMergeFuseMax = 4u;
struct PendingMergeComp {
    id<MTLBuffer> img = nil, flow = nil, cov = nil, rob = nil;
    MergeCompParamsCPU p{};
};
static std::vector<PendingMergeComp> g_merge_pending;

static void merge_band_cmd_reset() {
    g_merge_pending.clear();
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
// Accumulators are zeroed with a blit on this CB (not a CPU memset).
static bool merge_enc_ensure() {
    if (g_merge_enc) return true;
    if (!merge_band_cmd_ensure()) return false;
    if (g_merge_need_zero) {
        MergeAccSlot& slot = g_merge_acc[g_merge_write_slot];
        id<MTLBlitCommandEncoder> blit = [g_merge_band_cmd blitCommandEncoder];
        if (!blit) return false;
        [blit fillBuffer:slot.num range:NSMakeRange(0, slot.bytes) value:0];
        [blit fillBuffer:slot.den range:NSMakeRange(0, slot.bytes) value:0];
        [blit endEncoding];
        g_merge_need_zero = false;
    }
    g_merge_enc = [g_merge_band_cmd computeCommandEncoder];
    return g_merge_enc != nil;
}

// Comparison frames are batched so a group shares one read-modify-write of the
// band accumulators instead of one each. See merge_accumulate_comp_x4: the
// addition order is unchanged, so this is bit-identical, it just stops pushing
// the accumulators through DRAM once per frame.
static bool merge_flush_pending() {
    if (g_merge_pending.empty()) return true;
    auto& c = ctx();
    if (!merge_enc_ensure()) {
        g_merge_pending.clear();
        merge_band_cmd_reset();
        return false;
    }
    MergeAccSlot& slot = g_merge_acc[g_merge_write_slot];
    id<MTLComputeCommandEncoder> enc = g_merge_enc;
    const uint32_t nf = (uint32_t)g_merge_pending.size();

    // Metal requires every declared binding to be set even when the loop will
    // not read it, so unused slots repeat the last real frame. ps[0] also has
    // to be valid because the kernel takes the band geometry from it.
    MergeCompParamsCPU ps[kMergeFuseMax];
    for (uint32_t i = 0; i < kMergeFuseMax; ++i)
        ps[i] = g_merge_pending[std::min<uint32_t>(i, nf - 1u)].p;

    [enc setBuffer:slot.num offset:0 atIndex:0];
    [enc setBuffer:slot.den offset:0 atIndex:1];
    [enc setBytes:ps length:sizeof(ps) atIndex:2];
    [enc setBytes:&nf length:sizeof(nf) atIndex:3];
    for (uint32_t i = 0; i < kMergeFuseMax; ++i) {
        const PendingMergeComp& e = g_merge_pending[std::min<uint32_t>(i, nf - 1u)];
        [enc setBuffer:e.img  offset:0 atIndex:4u + i];
        [enc setBuffer:e.flow offset:0 atIndex:8u + i];
        [enc setBuffer:e.cov  offset:0 atIndex:12u + i];
        [enc setBuffer:e.rob  offset:0 atIndex:16u + i];
    }
    dispatch2(enc, c.pipe("merge_accumulate_comp_x4"), ps[0].Ws, ps[0].band_h);
    g_merge_pending.clear();
    return true;
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
            e.rob_h = rob.h;
            e.rob_w = rob.w;
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
        e.rob_h = rob.h;
        e.rob_w = rob.w;
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
    e.rob_h = rob.h;
    e.rob_w = rob.w;
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

static bool readback_slot(int slot_i, Image& num_band, Image& den_band);
static bool metal_merge_wait_inflight_impl();

// Grow-only per-slot GPU accumulators. Zero via blit when the band CB opens.
static bool ensure_acc_buffers(size_t nelem, bool start_band) {
    const size_t bytes = nelem * sizeof(float);
    if (bytes == 0) return false;
    auto& c = ctx();
    if (start_band) {
        if (g_merge_online) {
            // Same slot for the whole burst, zeroed only on the first open.
            // No wait here: the caller flushes explicitly after each frame, and
            // waiting through metal_merge_wait_inflight_impl would read the
            // whole accumulator back once per frame.
            g_merge_write_slot = 0;
            if (!g_merge_online_zeroed) {
                g_merge_need_zero = true;
                g_merge_online_zeroed = true;
            }
        } else if (g_merge_single_slot) {
            // One slot only: always wait previous before reusing (saves ~2× GPU RAM).
            if (g_merge_inflight.cmd) {
                if (!metal_merge_wait_inflight_impl()) return false;
            }
            g_merge_write_slot = 0;
        } else {
            // Ping-pong: free slot may still be in flight for encode overlap.
            g_merge_write_slot ^= 1;
            if (g_merge_inflight.cmd && g_merge_inflight.slot == g_merge_write_slot) {
                if (!metal_merge_wait_inflight_impl()) return false;
            }
        }
        // Banded opens a command buffer per band and accumulates every frame
        // into it, so "new command buffer" and "zero the accumulator" mean the
        // same thing there. Online commits after each frame -- it has to, or the
        // frame's GPU buffers cannot be freed -- so start_band is true on every
        // single frame, and zeroing here wiped everything merged so far. Each
        // frame erased its predecessors and only the reference, merged last,
        // survived into the output. The online branch above zeroes once.
        if (!g_merge_online) g_merge_need_zero = true;
    }
    MergeAccSlot& slot = g_merge_acc[g_merge_write_slot];
    if (!slot.num || !slot.den || slot.capacity < bytes) {
        slot.num = [c.device newBufferWithLength:bytes
                                         options:MTLResourceStorageModeShared];
        slot.den = [c.device newBufferWithLength:bytes
                                         options:MTLResourceStorageModeShared];
        slot.capacity = (slot.num && slot.den) ? bytes : 0;
        if (!slot.capacity) return false;
        g_merge_need_zero = true;
    }
    slot.bytes = bytes;
    return true;
}

static bool readback_slot(int slot_i, Image& num_band, Image& den_band) {
    if (slot_i < 0 || slot_i > 1) return false;
    MergeAccSlot& slot = g_merge_acc[slot_i];
    if (!slot.num || !slot.den) return false;
    const size_t bytes = slot.bytes;
    if (num_band.data.size() * sizeof(float) < bytes ||
        den_band.data.size() * sizeof(float) < bytes)
        return false;
    memcpy(num_band.data.data(), [slot.num contents], bytes);
    memcpy(den_band.data.data(), [slot.den contents], bytes);
    return true;
}

// Wait + readback the in-flight band into the Image* captured at commit time.
static bool metal_merge_wait_inflight_impl() {
    if (!g_merge_inflight.cmd) return true;
    [g_merge_inflight.cmd waitUntilCompleted];
    const bool ok = (g_merge_inflight.cmd.status == MTLCommandBufferStatusCompleted);
    Image* num = g_merge_inflight.num;
    Image* den = g_merge_inflight.den;
    const int slot = g_merge_inflight.slot;
    g_merge_inflight = {};
    if (!ok || !num || !den) return false;
    return readback_slot(slot, *num, *den);
}

} // namespace

bool metal_merge_has_frame(int frame_id) {
    return merge_frame_resident(frame_id);
}

bool metal_merge_wait_inflight() {
    return metal_merge_wait_inflight_impl();
}

void metal_merge_release_frame(int frame_id) {
    // acquire_frame_gpu caches uploads by frame id so every band can reuse
    // them. Online uses each frame exactly once, so without this the cache
    // would keep growing and the working set would still scale with the burst,
    // which is the whole thing online mode exists to avoid.
    for (size_t i = 0; i < g_merge_frames.size(); ++i) {
        if (g_merge_frames[i].key != frame_id) continue;
        g_merge_frames.erase(g_merge_frames.begin() + (long)i);
        return;
    }
}

void metal_merge_begin_online(int out_h, int out_w, int nch) {
    g_online_h = out_h;
    g_online_w = out_w;
    g_online_nch = nch;
    // Resolve anything banded still in flight before the accumulator changes
    // meaning, or its readback would land in a band image that has gone.
    (void)metal_merge_wait_inflight_impl();
    merge_band_cmd_reset();
    g_merge_online = true;
    g_merge_online_zeroed = false;
    g_merge_write_slot = 0;
    g_merge_need_zero = false;
}

// Accumulator slots are grow-only and shared with the banded path, so they are
// dropped rather than resized. Assigning {} nils the strong buffer refs, which
// under ARC releases them.
static void merge_release_acc_slots() {
    g_merge_acc[0] = {};
    g_merge_acc[1] = {};
}

void metal_merge_end_online() {
    // Release the accumulator with the burst instead of leaving it for the next
    // begin_burst to clear. Online sizes it to the whole output -- 22MB per
    // output megapixel, ~1.1GB at 48MP -- so holding it until the following
    // capture meant the app idled at over a gigabyte, and every shot after the
    // first began with that much of the jetsam budget already spent. Measured:
    // burst:begin 1247.7MB against a ~150MB app, and the matching -1150MB drop
    // did not appear until begin_burst, one whole burst later.
    //
    // Safe at this point: flush_online commits and waits, so nothing is queued
    // against these buffers, and every caller has finished reading the mapping
    // from metal_merge_map_online -- the output path calls this only after the
    // encode loop, the rest are error exits that return immediately. The wait
    // and reset mirror begin_burst, which frees the same slots, and cover the
    // error paths that arrive with a band command buffer still open.
    (void)metal_merge_wait_inflight_impl();
    merge_band_cmd_reset();
    merge_release_acc_slots();
    g_merge_online = false;
    g_merge_online_zeroed = false;
    g_online_h = g_online_w = g_online_nch = 0;
}

bool metal_merge_map_online(const float** num, const float** den, size_t* nelem) {
    if (!g_merge_online || !num || !den) return false;
    if (!metal_merge_flush_online()) return false;
    if (!g_merge_online_zeroed) return false;      // nothing ever accumulated
    MergeAccSlot& slot = g_merge_acc[0];
    if (!slot.num || !slot.den) return false;
    // MTLResourceStorageModeShared: the contents are already CPU addressable, so
    // handing back the pointer instead of memcpying into a host image saves a
    // second copy of the largest allocation in the pipeline. Same bytes.
    *num = (const float*)[slot.num contents];
    *den = (const float*)[slot.den contents];
    if (nelem) *nelem = (size_t)g_online_h * (size_t)g_online_w * (size_t)g_online_nch;
    return *num != nullptr && *den != nullptr;
}

bool metal_merge_flush_online() {
    if (!g_merge_online) return false;
    if (!g_merge_band_cmd) return true;
    if (!merge_flush_pending()) { merge_band_cmd_reset(); return false; }
    merge_enc_close();
    prof_tag_gpu(g_merge_band_cmd, "merge:online-cb");
    id<MTLCommandBuffer> cmd = g_merge_band_cmd;
    [cmd commit];
    // Wait rather than pipeline: the point of committing per frame is to free
    // that frame's GPU buffers, which cannot happen while its work is queued.
    [cmd waitUntilCompleted];
    const bool ok = (cmd.status == MTLCommandBufferStatusCompleted);
    merge_band_cmd_reset();
    return ok;
}


void metal_merge_set_single_acc_slot(bool enabled) {
    g_merge_single_slot = enabled;
}

void metal_invalidate_sticky_grey() {
    if (!metal_gpu_init()) return;
    auto& c = ctx();
    c.sticky_grey = nil;
    c.sticky_grey_h = c.sticky_grey_w = 0;
}

void metal_trim_analyze_scratch() {
    auto& c = ctx();
    c.kern_raw = nil; c.kern_vst = nil; c.kern_grey = nil;
    c.kern_grad = nil; c.kern_cov = nil;
    c.kern_raw_b = c.kern_vst_b = c.kern_grey_b = 0;
    c.kern_grad_b = c.kern_cov_b = 0;
    c.l2[0] = {};
    c.l2[1] = {};
    // Grey-FFT pools are analyze-only (~190MB at 12MP); merge must not inherit them.
    c.fft_in = nil; c.fft_c0 = nil; c.fft_out = nil;
    c.fft_in_b = c.fft_c0_b = c.fft_out_b = 0;
    c.fft_pp = nil;
    c.fft_pp_b = 0;
    c.scratch_cols = nil;
    c.scratch_cols_bytes = 0;
    c.sticky_grey = nil;
    c.sticky_grey_h = c.sticky_grey_w = 0;
    for (int i = 0; i < MetalCtx::kIcaSlots; ++i) c.ica[i] = {};
    c.ica_next = 0;
    clear_rob_ref_gpu();
}

void metal_merge_begin_burst(bool trim_analyze_scratch) {
    (void)metal_merge_wait_inflight_impl();
    merge_band_cmd_reset();
    // Clear online state here rather than trusting every exit path to do it.
    // Leaking it would make the next banded merge take its dispatch geometry
    // from the previous burst's output size, which is silent corruption.
    g_merge_online = false;
    g_merge_online_zeroed = false;
    g_online_h = g_online_w = g_online_nch = 0;
    g_merge_frames.clear();
    g_merge_ref = {};
    // First band XORs to slot 0 when double-buffered.
    g_merge_write_slot = g_merge_single_slot ? 0 : 1;
    g_merge_need_zero = false;
    // Drop previous burst's grow-only acc slots (1× bands can be hundreds of MB;
    // keeping them across shots causes jetsam on the next full-res merge).
    // end_online now frees them as soon as the online output is encoded, so for
    // that path this is only a backstop; the banded path still has no end hook
    // and relies on this, one burst late.
    merge_release_acc_slots();
    // Drop analyze scratch so merge prefetch is not fighting L2/Alg. 5 temps.
    // Skipped when the caller opens the frame table before analysis: this also
    // clears the reference robustness statistics, whose host copy has already
    // been released by then.
    if (trim_analyze_scratch) metal_trim_analyze_scratch();
}

bool metal_merge_prefetch_frame(const Image& comp_raw, const FlowField& flow,
                                const CovField& covs, const Image& robustness,
                                int frame_id) {
    if (!metal_gpu_init() || frame_id < 0) return false;
    if (comp_raw.h <= 0 || comp_raw.w <= 0) return false;
    id<MTLBuffer> b_img = nil, b_flow = nil, b_cov = nil, b_rob = nil;
    return acquire_frame_gpu(comp_raw, flow, covs, robustness, frame_id,
                             b_img, b_flow, b_cov, b_rob);
}

bool merge_comp_band_metal(const Image& comp_raw, const FlowField& flow,
                           const CovField& covs, const Image& robustness,
                           int tile_size, Image& num_band, Image& den_band,
                           int y0, const Config& cfg, int frame_id) {
    if (!metal_gpu_init()) return false;
    const int acc_h = g_merge_online ? g_online_h : num_band.h;
    const int acc_w = g_merge_online ? g_online_w : num_band.w;
    const int acc_c = g_merge_online ? g_online_nch : num_band.c;
    if (acc_h <= 0 || acc_w <= 0 || acc_c <= 0) return false;
    const bool resident = merge_frame_resident(frame_id);
    if (!resident && (comp_raw.h <= 0 || comp_raw.w <= 0)) return false;
    auto& c = ctx();

    // New band when no CB is open (previous band finished / burst begin).
    const bool start_band = (g_merge_band_cmd == nil);

    if (!ensure_acc_buffers((size_t)acc_h * acc_w * acc_c, start_band)) return false;

    id<MTLBuffer> b_img = nil, b_flow = nil, b_cov = nil, b_rob = nil;
    if (!acquire_frame_gpu(comp_raw, flow, covs, robustness, frame_id,
                           b_img, b_flow, b_cov, b_rob))
        return false;

    MergeCompParamsCPU p{};
    p.band_h = (uint32_t)acc_h;
    p.Ws = (uint32_t)acc_w;
    p.y0 = (uint32_t)y0;
    p.nch = (uint32_t)acc_c;
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
        p.rob_h = (uint32_t)robustness.h;
        p.rob_w = (uint32_t)robustness.w;
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
        p.rob_h = (uint32_t)std::max(1, hit->rob_h);
        p.rob_w = (uint32_t)std::max(1, hit->rob_w);
        p.flow_ny = (uint32_t)std::max(1, hit->flow_ny);
        p.flow_nx = (uint32_t)std::max(1, hit->flow_nx);
        p.cov_h = hit->cov_h > 0 ? (uint32_t)hit->cov_h : 1u;
        p.cov_w = hit->cov_w > 0 ? (uint32_t)hit->cov_w : 1u;
    }

    MergeAccSlot& slot = g_merge_acc[g_merge_write_slot];
    if (!merge_enc_ensure()) {
        merge_band_cmd_reset();
        return false;
    }
    (void)slot;
    PendingMergeComp pend;
    pend.img = b_img; pend.flow = b_flow; pend.cov = b_cov; pend.rob = b_rob;
    pend.p = p;
    g_merge_pending.push_back(pend);
    if (g_merge_pending.size() >= (size_t)kMergeFuseMax && !merge_flush_pending())
        return false;
    // Keep encoder open for remaining comps + ref in this band. Any tail group
    // is flushed by merge_ref_band_metal, which always runs after the comps.
    return true;
}

bool merge_ref_band_metal(const Image& ref_raw, const CovField& covs,
                          Image& num_band, Image& den_band, int y0,
                          const Config& cfg, const Image* acc_rob) {
    // Comps first, then ref -- same order the separate dispatches ran in.
    if (!merge_flush_pending()) return false;
    if (!metal_gpu_init()) return false;
    const int acc_h = g_merge_online ? g_online_h : num_band.h;
    const int acc_w = g_merge_online ? g_online_w : num_band.w;
    const int acc_c = g_merge_online ? g_online_nch : num_band.c;
    if (acc_h <= 0 || acc_w <= 0 || acc_c <= 0 || ref_raw.h <= 0) return false;
    auto& c = ctx();

    const bool start_band = (g_merge_band_cmd == nil);

    if (!ensure_acc_buffers((size_t)acc_h * acc_w * acc_c, start_band)) return false;

    const bool denoise = cfg.accumulated_robustness_denoiser_enabled && acc_rob &&
                         acc_rob->h > 0 && acc_rob->w > 0;
    id<MTLBuffer> b_img = nil, b_cov = nil, b_acc = nil;
    if (!acquire_ref_gpu(ref_raw, covs, acc_rob, denoise, b_img, b_cov, b_acc))
        return false;

    MergeRefParamsCPU p{};
    p.band_h = (uint32_t)acc_h;
    p.Ws = (uint32_t)acc_w;
    p.y0 = (uint32_t)y0;
    p.lr_h = (uint32_t)ref_raw.h;
    p.lr_w = (uint32_t)ref_raw.w;
    p.cov_h = covs.h > 0 ? (uint32_t)covs.h : 1u;
    p.cov_w = covs.w > 0 ? (uint32_t)covs.w : 1u;
    p.acc_h = denoise ? (uint32_t)acc_rob->h : 1u;
    p.acc_w = denoise ? (uint32_t)acc_rob->w : 1u;
    p.nch = (uint32_t)acc_c;
    p.bayer = cfg.bayer_mode ? 1u : 0u;
    p.iso = (cfg.kernel == KernelShape::Iso) ? 1u : 0u;
    p.robustness_denoise = denoise ? 1u : 0u;
    p.rad_max = (uint32_t)std::max(0, (int)cfg.acc_rob_rad_max);
    p.scale = cfg.scale;
    p.max_multiplier = cfg.acc_rob_max_multiplier;
    p.burst_frames = (float)cfg.burst_frame_count;
    p.adaptive = cfg.acc_rob_adaptive ? 1u : 0u;
    p.max_frame_count = cfg.acc_rob_max_frame_count;
    p.cfa00 = cfg.cfa.p[0][0];
    p.cfa01 = cfg.cfa.p[0][1];
    p.cfa10 = cfg.cfa.p[1][0];
    p.cfa11 = cfg.cfa.p[1][1];

    MergeAccSlot& slot = g_merge_acc[g_merge_write_slot];
    if (!merge_enc_ensure()) {
        merge_band_cmd_reset();
        return false;
    }
    id<MTLComputeCommandEncoder> enc = g_merge_enc;
    [enc setBuffer:slot.num offset:0 atIndex:0];
    [enc setBuffer:slot.den offset:0 atIndex:1];
    [enc setBuffer:b_img offset:0 atIndex:2];
    [enc setBuffer:b_cov offset:0 atIndex:3];
    [enc setBuffer:b_acc offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    dispatch2(enc, c.pipe("merge_accumulate_ref"), p.Ws, p.band_h);
    merge_enc_close();

    // Online: the reference is just the last contribution to a long-lived
    // accumulator, so leave the command buffer for metal_merge_map_online to
    // commit and read back exactly once.
    if (g_merge_online) return true;

    // One CB per band, covering every comp dispatch + the ref dispatch.
    prof_tag_gpu(g_merge_band_cmd, "merge:band-cb");
    [g_merge_band_cmd commit];
    // Resolve any previous in-flight band now (GPU current keeps running).
    if (g_merge_inflight.cmd) {
        if (!metal_merge_wait_inflight_impl()) {
            merge_band_cmd_reset();
            return false;
        }
    }
    g_merge_inflight.cmd = g_merge_band_cmd;
    g_merge_inflight.slot = g_merge_write_slot;
    g_merge_inflight.num = &num_band;
    g_merge_inflight.den = &den_band;
    merge_band_cmd_reset();
    // Caller must metal_merge_wait_inflight() before reading this band's host images,
    // or start the next band (which resolves this one when the following ref commits).
    return true;
}

// --- Autorelease boundaries -------------------------------------------------
//
// MTLCommandBuffer and MTLComputeCommandEncoder come back autoreleased, and a
// command buffer retains every resource it references until it is deallocated.
// align_metal alone opens ~14 command buffers per comparison frame, each holding
// multi-megabyte grey/flow/gradient buffers, so with no pool on the calling
// thread a burst accumulated every frame's Metal temporaries before anything was
// released. Measured peak footprint reached 2784MB with only 287MB of jetsam
// headroom left, with the peak landing in the analyze loop.
//
// These wrappers bound that to one call's worth. The bodies are unchanged; the
// C++ return values are not autoreleased objects, so constructing them inside
// the pool and returning after it drains is safe.

Image compute_grey_fft_metal(const Image& raw) {
    @autoreleasepool { return compute_grey_fft_metal_impl(raw); }
}

CovField estimate_kernels_metal(const Image& raw, const Config& cfg) {
    @autoreleasepool { return estimate_kernels_metal_impl(raw, cfg); }
}

RefStats init_robustness_metal(const Image& ref_raw, const Config& cfg) {
    @autoreleasepool { return init_robustness_metal_impl(ref_raw, cfg); }
}

Image compute_robustness_metal(const Image& comp_raw, const RefStats& ref_stats,
                               const FlowField& flow, int tile_size, const Config& cfg,
                               Image* s_select_out) {
    @autoreleasepool {
        return compute_robustness_metal_impl(comp_raw, ref_stats, flow, tile_size, cfg,
                                             s_select_out);
    }
}

bool align_metal(const Pyramid& ref_pyr, const Image& ref_grey,
                 const Image& moving_grey,
                 const Config& cfg, int tile_size, FlowField& flow_out,
                 f32 initial_dx, f32 initial_dy, f32 initial_rotation_rad) {
    @autoreleasepool {
        return align_metal_impl(ref_pyr, ref_grey, moving_grey, cfg, tile_size,
                                flow_out, initial_dx, initial_dy, initial_rotation_rad);
    }
}

} // namespace hhsr
