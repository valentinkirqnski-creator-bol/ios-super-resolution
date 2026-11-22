#pragma once
// SRSR compute backend: the one interface srsr_pipeline.cpp encodes against.
// Metal (srsr_metal.mm) on the device; the CPU emulation (srsr_cpu.cpp) in the
// PC test harness runs the same SRSRKernels.metal source.
//
// Semantics follow a Metal compute encoder: bindings persist until replaced; a
// batch is one command buffer; batches execute in commit order.
#include <cstddef>
#include <cstdint>
#include <memory>

namespace hhsr {
namespace srsr {

enum class Fmt { R8Unorm, R32Float, RGBA16Float, RGBA16Uint, RGBA32Float };

inline size_t fmt_bytes(Fmt f) {
    switch (f) {
        case Fmt::R8Unorm: return 1;
        case Fmt::R32Float: return 4;
        case Fmt::RGBA16Float: case Fmt::RGBA16Uint: return 8;
        case Fmt::RGBA32Float: return 16;
    }
    return 4;
}

enum class K {
    prepare, unblocker, clear_flow, downsample, gradients, gradient_tile_sums,
    block_matching, ica_iteration, flow_transfer, flow_median, flow_pack,
    guide, rejection, dilate_h, dilate_v, accumulate_weights, merge, denoise, count
};

inline const char* kernel_name(K k) {
    static const char* names[] = {
        "srsr_prepare", "srsr_unblocker", "srsr_clear_flow", "srsr_downsample", "srsr_gradients",
        "srsr_gradient_tile_sums", "srsr_block_matching", "srsr_ica_iteration", "srsr_flow_transfer",
        "srsr_flow_median", "srsr_flow_pack", "srsr_guide", "srsr_rejection", "srsr_dilate_horizontal",
        "srsr_dilate_vertical", "srsr_accumulate_weights", "srsr_merge", "srsr_denoise"};
    return names[(int)k];
}

struct Tex {
    virtual ~Tex() = default;
    int w = 0, h = 0;
    Fmt fmt = Fmt::R32Float;
    size_t bytes() const { return (size_t)w * (size_t)h * fmt_bytes(fmt); }
};

// Host-visible buffer (shared storage on the device: no copies either way).
struct Buf {
    virtual ~Buf() = default;
    size_t size = 0;
    virtual void* contents() = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    virtual std::unique_ptr<Tex> make_tex(Fmt f, int w, int h, const char* label) = 0;
    virtual std::unique_ptr<Buf> make_buf(size_t bytes, const char* label) = 0;

    virtual bool begin() = 0;                                  // open a batch
    virtual void tex(int slot, Tex* t) = 0;
    virtual void bytes(int slot, const void* p, size_t n) = 0; // <= 4 KB
    virtual void buf(int slot, Buf* b) = 0;
    virtual void dispatch(K k, int w, int h) = 0;              // one thread per element
    // srsr_block_matching: one threadgroup per tile, `threads` lanes, two
    // threadgroup allocations (bytes, multiples of 16).
    virtual void dispatch_bm(int tiles_x, int tiles_y, int threads, size_t tg0, size_t tg1) = 0;
    virtual uint64_t commit(const char* label) = 0;            // close + submit; returns a ticket
    virtual bool wait(uint64_t ticket) = 0;                    // false: the batch failed
    virtual bool wait_all() = 0;
    virtual bool failed() const = 0;
};

// Defined by exactly one of srsr_metal.mm (Apple) / srsr_cpu.cpp (PC harness).
// Returns null when no backend can run.
std::unique_ptr<Backend> make_backend();

} // namespace srsr
} // namespace hhsr
