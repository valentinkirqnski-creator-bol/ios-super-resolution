// Platform-specific stubs for desktop builds
// These provide minimal implementations for functions that are normally
// provided by iOS frameworks or debug/profiling code.

#include "types.h"
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define SR_MKDIR(p) _mkdir(p)
#else
#define SR_MKDIR(p) mkdir(p, 0777)
#endif

namespace hhsr {

// Debug dumping for the desktop harness. Opt-in via HHSR_ENABLE_DEBUG_DUMPS=1,
// output dir via HHSR_DEBUG_DIR (default ./debug_dumps). Mirrors the iOS
// debug_utils.mm behaviour: only the small per-frame flow/mask fields are
// written unless HHSR_DEBUG_ALL=1.
static bool sr_debug_enabled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    const char* f = std::getenv("HHSR_ENABLE_DEBUG_DUMPS");
    cached = (f && (f[0]=='1'||f[0]=='y'||f[0]=='Y'||f[0]=='t'||f[0]=='T')) ? 1 : 0;
    return cached != 0;
}
static bool sr_debug_all() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    const char* f = std::getenv("HHSR_DEBUG_ALL");
    cached = (f && (f[0]=='1'||f[0]=='y'||f[0]=='Y'||f[0]=='t'||f[0]=='T')) ? 1 : 0;
    return cached != 0;
}
static std::string sr_dump_dir() {
    if (const char* d = std::getenv("HHSR_DEBUG_DIR")) return std::string(d);
    return "debug_dumps";
}
static bool sr_dump_wanted(const std::string& name) {
    if (sr_debug_all()) return true;
    return name.rfind("cpp_flow_", 0) == 0 || name.rfind("cpp_mask_", 0) == 0 ||
           name.rfind("cpp_ref_grey", 0) == 0 || name.rfind("cpp_acc_rob", 0) == 0;
}

bool debug_dumps_enabled() { return sr_debug_enabled(); }

void debug_ensure_dir() {
    if (!sr_debug_enabled()) return;
    SR_MKDIR(sr_dump_dir().c_str());
}

void debug_dump_bin(const std::string& filename, const float* data, size_t count) {
    if (!sr_debug_enabled() || !sr_dump_wanted(filename) || !data || count == 0) return;
    debug_ensure_dir();
    const std::string path = sr_dump_dir() + "/" + filename + ".bin";
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(data, sizeof(float), count, f);
        std::fclose(f);
        std::printf("[DEBUG] Saved %s (%zu floats)\n", path.c_str(), count);
    }
}

void debug_dump_text(const std::string& filename, const std::string& text) {
    if (!sr_debug_enabled()) return;
    debug_ensure_dir();
    const std::string path = sr_dump_dir() + "/" + filename + ".txt";
    if (FILE* f = std::fopen(path.c_str(), "w")) {
        std::fprintf(f, "%s\n", text.c_str());
        std::fclose(f);
    }
}

// Profiling stubs
double prof_now_ms() {
    static auto start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    return (double)elapsed.count();
}

void prof_add_cpu(const char* label, double ms) {
    // No-op on desktop - could log if needed
    (void)label;
    (void)ms;
}

bool prof_enabled() {
    return false;  // Profiling disabled on desktop
}

// Neural network stubs (disabled for desktop)
bool robustness_nn_available() {
    return false;  // Not available without Metal
}

bool robustness_nn_infer(const Image& input, Image& output) {
    (void)input;
    (void)output;
    return false;  // Not available
}

bool robustness_refine_available() {
    return false;  // Not available without Metal
}

bool robustness_refine_infer(const Image& input, Image& output) {
    (void)input;
    (void)output;
    return false;  // Not available
}

}  // namespace hhsr

// Windows socket library dependency
// Note: htonl, ntohl, htons, ntohs are provided by Windows socket libraries
// which are linked via LibRaw and other system libraries.
