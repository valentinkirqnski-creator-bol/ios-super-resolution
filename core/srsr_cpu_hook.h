#pragma once
// srsr_cpu_hook.h -- PC test harness only (srsr_cpu.cpp; NOT in the iOS target).
// Lets a tool observe every dispatch of the CPU backend after it ran, with the
// bound slots exactly as the kernel saw them. tools/srsr_nn uses it to dump the
// merge inputs (quads, covariances, flows, masks, uniforms) and the merged band.
#include "srsr_backend.h"
#include <cstdint>
#include <functional>
#include <vector>

namespace hhsr {
namespace srsr {

struct CpuSlotView {
    bool bound = false;
    Fmt fmt = Fmt::R32Float;
    int w = 0, h = 0, nc = 0;          // channels as stored (1 or 4)
    const float* f = nullptr;          // float formats (R8Unorm / RGBA16Float already quantised)
    const uint16_t* u = nullptr;       // RGBA16Uint
};

// Bindings persist across dispatches (as in Metal), so a slot this kernel does not
// read may still point at a texture freed since: ask only for the slots the
// kernel uses.
struct CpuDispatchView {
    K kernel = K::count;
    int w = 0, h = 0;                  // the dispatch grid
    std::function<CpuSlotView(int)> tex;
    const std::vector<uint8_t>* bytes = nullptr;   // 8 setBytes slots
    std::function<void*(int)> buf;                 // a buffer slot's contents (nullptr if unbound)
};

// Called after every dispatch when set. Single-threaded (the backend is synchronous).
extern void (*g_cpu_dispatch_hook)(const CpuDispatchView&);

} // namespace srsr
} // namespace hhsr
