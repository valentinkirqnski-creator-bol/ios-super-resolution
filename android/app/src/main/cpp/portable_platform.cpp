// Android implementations of the two platform services the portable core calls
// without an __APPLE__ guard: profiling and debug dumps.
//
// Every metal_* call in the core is guarded, so nothing else from the .mm files
// is needed. These two are not, so without this the link fails on undefined
// symbols -- prof_add_cpu, prof_now_ms, debug_dump_bin and the rest.
//
// The profiler is worth having rather than stubbing: the per-stage timing and
// the memory-by-stage table are exactly what is needed to find out whether this
// is usable on a 3GB phone, and both port cleanly. Footprint comes from
// /proc/self/statm instead of phys_footprint.
#include "core/prof.h"
#include "core/debug_utils.h"
#include "core/mps_fft.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <string>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <vector>

namespace hhsr {
namespace {

struct Bucket {
    std::string name;
    double ms = 0.0;
    uint64_t calls = 0;
};

struct MemMark {
    std::string name;
    uint64_t last = 0, peak = 0;
    int64_t net = 0;
    uint64_t hits = 0;
};

struct State {
    std::mutex m;
    std::vector<Bucket> cpu, gpu;
    std::vector<MemMark> marks;
    uint64_t last_fp = 0, peak_fp = 0, min_avail = UINT64_MAX;
    bool have_last = false;
};

State& state() {
    static State s;
    return s;
}

Bucket& bucket(std::vector<Bucket>& v, const char* name) {
    const std::string n = name ? name : "";
    for (Bucket& b : v)
        if (b.name == n) return b;
    v.push_back(Bucket{n, 0.0, 0});
    return v.back();
}

std::string fmt_mb(uint64_t b) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.1f MB", (double)b / (1024.0 * 1024.0));
    return buf;
}

std::string dump_dir() {
    const char* d = std::getenv("HHSR_DEBUG_DIR");
    return d && *d ? std::string(d) : std::string();
}

}  // namespace

bool prof_enabled() {
    static const int on = [] {
        const char* v = std::getenv("HHSR_PROF");
        if (v) return (v[0] != '0') ? 1 : 0;
#ifdef NDEBUG
        return 0;
#else
        return 1;
#endif
    }();
    return on != 0;
}

double prof_now_ms() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

void prof_add_cpu(const char* name, double ms) {
    if (!prof_enabled()) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    Bucket& b = bucket(s.cpu, name);
    b.ms += ms;
    b.calls++;
}

void prof_add_gpu(const char* name, double ms) {
    if (!prof_enabled()) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    Bucket& b = bucket(s.gpu, name);
    b.ms += ms;
    b.calls++;
}

// Resident set from /proc/self/statm, field 2, in pages. The nearest analogue to
// phys_footprint: what the kernel would count against this process.
uint64_t prof_footprint_bytes() {
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long total = 0, rss = 0;
    const int got = std::fscanf(f, "%lu %lu", &total, &rss);
    std::fclose(f);
    if (got != 2) return 0;
    return (uint64_t)rss * (uint64_t)sysconf(_SC_PAGESIZE);
}

// Android has no os_proc_available_memory. Free system memory is the closest
// honest proxy, and it is NOT the same thing: the per-app limit is lower, so
// treat this as an upper bound on the headroom rather than the headroom.
uint64_t prof_available_bytes() {
    struct sysinfo si {};
    if (sysinfo(&si) != 0) return 0;
    return (uint64_t)si.freeram * (uint64_t)si.mem_unit;
}

void prof_mark_memory(const char* label) {
    if (!prof_enabled()) return;
    const uint64_t fp = prof_footprint_bytes();
    const uint64_t av = prof_available_bytes();
    State& s = state();
    std::lock_guard<std::mutex> lk(s.m);

    const std::string name = label ? label : "";
    MemMark* mk = nullptr;
    for (MemMark& e : s.marks)
        if (e.name == name) { mk = &e; break; }
    if (!mk) {
        MemMark e;
        e.name = name;
        s.marks.push_back(std::move(e));
        mk = &s.marks.back();
    }
    mk->last = fp;
    if (fp > mk->peak) mk->peak = fp;
    if (s.have_last) mk->net += (int64_t)fp - (int64_t)s.last_fp;
    mk->hits++;
    s.last_fp = fp;
    s.have_last = true;

    if (fp > s.peak_fp) s.peak_fp = fp;
    if (av > 0 && av < s.min_avail) s.min_avail = av;
}

uint64_t prof_peak_footprint_bytes() { return state().peak_fp; }
uint64_t prof_min_available_bytes() {
    const uint64_t v = state().min_avail;
    return v == UINT64_MAX ? 0 : v;
}

std::string prof_report() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    std::string out = "=== HHSR profile (Android) ===\n";
    char line[256];

    double cpu_total = 0.0;
    for (const Bucket& b : s.cpu) cpu_total += b.ms;
    std::snprintf(line, sizeof(line), "%-30s %10s %8s %10s\n",
                  "CPU stage", "total ms", "calls", "ms/call");
    out += line;
    for (const Bucket& b : s.cpu) {
        std::snprintf(line, sizeof(line), "  %-28s %10.1f %8llu %10.2f\n",
                      b.name.c_str(), b.ms, (unsigned long long)b.calls,
                      b.calls ? b.ms / (double)b.calls : 0.0);
        out += line;
    }
    std::snprintf(line, sizeof(line), "  %-28s %10.1f\n", "(sum)", cpu_total);
    out += line;

    if (!s.marks.empty()) {
        std::snprintf(line, sizeof(line), "\n%-28s %10s %10s %10s %6s\n",
                      "Memory by stage", "footprint", "peak", "net +/-", "hits");
        out += line;
        for (const MemMark& mk : s.marks) {
            std::snprintf(line, sizeof(line), "  %-26s %10s %10s %+9.1f %6llu\n",
                          mk.name.c_str(), fmt_mb(mk.last).c_str(),
                          fmt_mb(mk.peak).c_str(),
                          (double)mk.net / (1024.0 * 1024.0),
                          (unsigned long long)mk.hits);
            out += line;
        }
    }
    std::snprintf(line, sizeof(line), "\n  peak RSS %s\n", fmt_mb(s.peak_fp).c_str());
    out += line;
    out += "  (free system RAM is an upper bound on headroom, not the per-app limit)\n";
    return out;
}

std::string prof_save_report(const std::string& text) {
    const std::string dir = dump_dir();
    if (dir.empty()) return {};
    const std::string path = dir + "/hhsr_profile.txt";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return {};
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
    return path;
}

void prof_reset() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    s.cpu.clear();
    s.gpu.clear();
    s.marks.clear();
    s.last_fp = 0;
    s.peak_fp = 0;
    s.min_avail = UINT64_MAX;
    s.have_last = false;
}

ProfScope::ProfScope(const char* name)
    : name_(name), t0_(prof_enabled() ? prof_now_ms() : 0.0) {}
ProfScope::~ProfScope() {
    if (prof_enabled()) prof_add_cpu(name_, prof_now_ms() - t0_);
}

// ---------------------------------------------------------------- dumps
bool debug_dumps_enabled() {
    static const int on = [] {
        const char* v = std::getenv("HHSR_ENABLE_DEBUG_DUMPS");
        if (v && v[0] != '0') return 1;
        return std::getenv("HHSR_DEBUG_DIR") ? 1 : 0;
    }();
    return on != 0;
}

void debug_ensure_dir() {}

void debug_dump_bin(const std::string& name, const float* data, size_t size) {
    if (!debug_dumps_enabled() || !data) return;
    const std::string dir = dump_dir();
    if (dir.empty()) return;
    FILE* f = std::fopen((dir + "/" + name + ".bin").c_str(), "wb");
    if (!f) return;
    std::fwrite(data, sizeof(float), size, f);
    std::fclose(f);
}

// ---------------------------------------------------------------- MPSGraph
// pipeline_paths.cpp calls the prewarm and release entry points unguarded --
// deliberately, since mps_fft.mm already defines them as no-ops when MPSGraph
// is missing. There is no MPSGraph at all here, so they are no-ops for the same
// reason. mps_fft_enabled() returning false keeps every other caller on the
// portable path.
bool mps_fft_enabled() { return false; }
void mps_fft_prewarm(int, int) {}
void mps_fft_release_all() {}
bool mps_grey_lowpass(const float*, float*, int, int, void*) { return false; }

void debug_dump_text(const std::string& name, const std::string& text) {
    if (!debug_dumps_enabled()) return;
    const std::string dir = dump_dir();
    if (dir.empty()) return;
    FILE* f = std::fopen((dir + "/" + name + ".txt").c_str(), "wb");
    if (!f) return;
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
}

}  // namespace hhsr
