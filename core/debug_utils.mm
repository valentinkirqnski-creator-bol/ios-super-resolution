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

bool debug_dumps_enabled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    bool enabled = false;
    if (const char* dir = std::getenv("HHSR_DEBUG_DIR")) {
        if (dir[0] != '\0') enabled = true;
    }
    if (!enabled) {
        if (const char* flag = std::getenv("HHSR_ENABLE_DEBUG_DUMPS")) {
            enabled = flag[0] == '1' || flag[0] == 'y' || flag[0] == 'Y' ||
                      flag[0] == 't' || flag[0] == 'T';
        }
    }
    cached = enabled ? 1 : 0;
    return enabled;
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
