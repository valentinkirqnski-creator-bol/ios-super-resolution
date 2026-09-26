// Host-side stubs so the portable core links into a Windows/Linux test harness.
// Everything here is either an Apple-only facility (profiling, Core ML) or
// lives in a translation unit the harness deliberately does not pull in
// (pipeline.cpp, which drags the whole app in with it).
#include "types.h"
#include "prof.h"
#include "debug_utils.h"
#include "robustness_nn.h"
#include "refine_host.h"   // tools/rob_refine
#include <chrono>

namespace hhsr {

// A pure-C++ stand-in for Core ML, so the whole refinement stage -- feature
// builder, strip loop, the bounded multiply -- can be exercised and measured
// off-device with the same weights the exporter will ship.
static RefineHostFn g_host_refine = nullptr;
RefineHostFn host_refine_fn() { return g_host_refine; }
void robustness_refine_set_host(RefineHostFn fn) { g_host_refine = fn; }

// --- pipeline.cpp: verbatim, the only two symbols align.cpp needs from it ---
int pad_image_circular_amount(const Image& img, int tile_size) {
    if (tile_size <= 0) return 0;
    const int pad_h = (tile_size - img.h % tile_size) % tile_size;
    const int pad_w = (tile_size - img.w % tile_size) % tile_size;
    return pad_h | pad_w;
}
Image pad_image_circular(const Image& img, int tile_size) {
    int pad_h = (tile_size - img.h % tile_size) % tile_size;
    int pad_w = (tile_size - img.w % tile_size) % tile_size;
    if (pad_h == 0 && pad_w == 0) return img;
    Image padded(img.h + pad_h, img.w + pad_w, img.c);
    for (int y = 0; y < padded.h; ++y) {
        int src_y = y < img.h ? y : (y - img.h);
        for (int x = 0; x < padded.w; ++x) {
            int src_x = x < img.w ? x : (x - img.w);
            for (int ch = 0; ch < img.c; ++ch)
                padded.at(y, x, ch) = img.at(src_y, src_x, ch);
        }
    }
    return padded;
}

// --- prof.mm / debug_utils.mm: inert ---
bool prof_enabled() { return false; }
double prof_now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}
void prof_add_cpu(const char*, double) {}
void prof_add_gpu(const char*, double) {}
uint64_t prof_footprint_bytes() { return 0; }
uint64_t prof_available_bytes() { return 0; }
void prof_mark_memory(const char*) {}
uint64_t prof_peak_footprint_bytes() { return 0; }
uint64_t prof_min_available_bytes() { return 0; }
std::string prof_report() { return {}; }
std::string prof_save_report(const std::string&) { return {}; }
void prof_reset() {}
ProfScope::ProfScope(const char* n) : name_(n), t0_(0.0) {}
ProfScope::~ProfScope() {}

bool debug_dumps_enabled() { return false; }
void debug_ensure_dir() {}
void debug_dump_bin(const std::string&, const float*, size_t) {}
void debug_dump_text(const std::string&, const std::string&) {}

// --- robustness_nn.mm: Core ML is Apple-only, so the learned replacement
// mask is simply unavailable here and robustness.cpp takes its analytic path.
bool robustness_nn_available() { return false; }
bool robustness_nn_infer(const Image&, Image&) { return false; }
void robustness_nn_release_buffers() {}

// The refinement model is loaded by the harness itself (refine_dataset /
// refine_eval install a pure-C++ evaluator through robustness_refine_set_host
// below) rather than by Core ML, so these forward to whatever it installed.
bool robustness_refine_available() { return host_refine_fn() != nullptr; }
bool robustness_refine_infer(const Image& feat, Image& out) {
    auto fn = host_refine_fn();
    return fn ? fn(feat, out) : false;
}
void robustness_refine_release_buffers() {}

} // namespace hhsr
