#include "debug_utils.h"
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <string>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0777)
#endif

namespace hhsr {

// ON by default on this diagnostic branch: a sideloaded IPA has no Xcode scheme
// to inject environment variables, so env-only gating produced no data at all.
// HHSR_ENABLE_DEBUG_DUMPS=0 disables. FLIP BACK TO OPT-IN BEFORE RELEASE.
bool debug_dumps_enabled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    bool enabled = true;
    if (const char* flag = std::getenv("HHSR_ENABLE_DEBUG_DUMPS")) {
        enabled = !(flag[0] == '0' || flag[0] == 'n' || flag[0] == 'N' ||
                    flag[0] == 'f' || flag[0] == 'F');
    }
    cached = enabled ? 1 : 0;
    return enabled;
}

// Whether to write every dump or only the small diagnostic ones.
//
// A full burst dumps the reference grey, every comparison RAW, every moving
// grey, the ICA gradients and the merge bands — several GB per shot at 12MP,
// which would fill the device and dominate the runtime. Flow and mask are what
// distinguish an alignment difference from a robustness difference, and both
// are small: flow is per-tile, mask is guide resolution.
//
// Set HHSR_DEBUG_ALL=1 for everything.
static bool debug_dump_all() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    const char* flag = std::getenv("HHSR_DEBUG_ALL");
    cached = (flag && (flag[0] == '1' || flag[0] == 'y' || flag[0] == 'Y' ||
                       flag[0] == 't' || flag[0] == 'T')) ? 1 : 0;
    return cached != 0;
}

static bool debug_dump_wanted(const std::string& name) {
    if (debug_dump_all()) return true;
    return name.rfind("cpp_flow_", 0) == 0 || name.rfind("cpp_mask_", 0) == 0;
}

static std::string get_dump_dir() {
    if (const char* env_dir = std::getenv("HHSR_DEBUG_DIR")) {
        return std::string(env_dir);
    }
#ifdef __APPLE__
    // iOS sandbox: $HOME/Documents is the app Documents folder (Files app).
    if (const char* home = std::getenv("HOME")) {
        return std::string(home) + "/Documents/debug_dumps";
    }
#endif
    return "debug_dumps";
}

void debug_ensure_dir() {
    if (!debug_dumps_enabled()) return;
    MKDIR(get_dump_dir().c_str());
}

void debug_dump_bin(const std::string& name, const float* data, size_t size) {
    if (!debug_dumps_enabled()) return;
    if (!debug_dump_wanted(name)) return;
    if (!data || size == 0) return;
    debug_ensure_dir();
    std::string path = get_dump_dir() + "/" + name + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data, sizeof(float), size, f);
        fclose(f);
        printf("[DEBUG] Saved %s (%zu floats)\n", path.c_str(), size);
    }
}

void debug_dump_text(const std::string& name, const std::string& text) {
    if (!debug_dumps_enabled()) return;
    debug_ensure_dir();
    std::string path = get_dump_dir() + "/" + name + ".txt";
    FILE* f = fopen(path.c_str(), "w");
    if (f) {
        fprintf(f, "%s\n", text.c_str());
        fclose(f);
    }
}

} // namespace hhsr
