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
    MKDIR(get_dump_dir().c_str());
}

void debug_dump_bin(const std::string& name, const float* data, size_t size) {
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
    debug_ensure_dir();
    std::string path = get_dump_dir() + "/" + name + ".txt";
    FILE* f = fopen(path.c_str(), "w");
    if (f) {
        fprintf(f, "%s\n", text.c_str());
        fclose(f);
    }
}

} // namespace hhsr
