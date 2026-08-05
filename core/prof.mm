#include "prof.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <os/proc.h>
#else
#include <chrono>
#endif

namespace hhsr {
namespace {

struct Bucket {
    std::string name;
    double total_ms = 0.0;
    uint64_t calls = 0;
};

struct ProfState {
    std::mutex m;
    std::vector<Bucket> cpu;
    std::vector<Bucket> gpu;
    uint64_t peak_footprint = 0;
    std::string peak_label;
    uint64_t min_available = UINT64_MAX;
    std::string min_label;
    double t_origin = 0.0;
};

ProfState& state() {
    static ProfState s;
    return s;
}

// Linear scan keeps first-seen (pipeline) order in the report; the bucket count
// is small and this only runs when profiling is on.
void accumulate(std::vector<Bucket>& v, const char* name, double ms) {
    for (Bucket& b : v) {
        if (b.name == name) {
            b.total_ms += ms;
            b.calls++;
            return;
        }
    }
    Bucket b;
    b.name = name;
    b.total_ms = ms;
    b.calls = 1;
    v.push_back(std::move(b));
}

std::string fmt_mb(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MB", (double)bytes / (1024.0 * 1024.0));
    return std::string(buf);
}

void append_table(std::ostringstream& ss, const char* title, const char* unit_col,
                  const std::vector<Bucket>& v) {
    if (v.empty()) return;
    double grand = 0.0;
    for (const Bucket& b : v) grand += b.total_ms;

    char line[256];
    std::snprintf(line, sizeof(line), "%-28s %10s %9s %10s %7s\n", title, "total ms",
                  unit_col, "ms/call", "%");
    ss << line;
    for (const Bucket& b : v) {
        const double per = b.calls ? b.total_ms / (double)b.calls : 0.0;
        const double pct = grand > 0.0 ? 100.0 * b.total_ms / grand : 0.0;
        std::snprintf(line, sizeof(line), "  %-26s %10.1f %9llu %10.2f %6.1f%%\n",
                      b.name.c_str(), b.total_ms,
                      (unsigned long long)b.calls, per, pct);
        ss << line;
    }
    std::snprintf(line, sizeof(line), "  %-26s %10.1f\n", "(sum)", grand);
    ss << line << "\n";
}

}  // namespace

// State when HHSR_PROF is unset. On unconditionally: the build that matters
// here is a sideloaded IPA, which has no Xcode scheme to inject environment
// variables, so anything opt-in produces no data at all on the only hardware
// the timings are being measured on.
//
// This used to key off NDEBUG and default off for release. That was working by
// accident rather than by design: Xcode does not define NDEBUG for C/C++ in its
// release configuration the way CMake does, so release builds were profiling
// anyway. Making it explicit so the reports do not silently stop the day a
// build setting adds NDEBUG.
//
// The cost is negligible and deliberately so. CPU buckets are timestamp reads,
// GPU timings come from GPUEndTime/GPUStartTime inside an existing completion
// handler and never add a synchronization point, and the memory marks are a
// couple of dozen task_info calls per burst.
//
// HHSR_PROF=0 disables. This is separate from HHSR_ENABLE_DEBUG_DUMPS, which
// gates the far more expensive per-image dumps and statistics passes and is
// off by default.
static constexpr bool kProfDefaultOn = true;

bool prof_enabled() {
    static const bool on = []() {
        const char* v = std::getenv("HHSR_PROF");
        if (!v || !v[0]) return kProfDefaultOn;
        return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' ||
                 v[0] == 'f' || v[0] == 'F');
    }();
    return on;
}

std::string prof_save_report(const std::string& text) {
    std::string path;
#if defined(__APPLE__)
    if (const char* home = std::getenv("HOME")) {
        path = std::string(home) + "/Documents/hhsr_profile.txt";
    }
#endif
    if (path.empty()) path = "hhsr_profile.txt";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return std::string();
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
    return path;
}

double prof_now_ms() {
#if defined(__APPLE__)
    static const double scale = []() {
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        // ticks -> ns -> ms
        return (double)tb.numer / (double)tb.denom / 1.0e6;
    }();
    return (double)mach_absolute_time() * scale;
#else
    using clock = std::chrono::steady_clock;
    const auto d = clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(d).count();
#endif
}

void prof_add_cpu(const char* name, double ms) {
    if (!prof_enabled() || !name) return;
    ProfState& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    accumulate(s.cpu, name, ms);
}

void prof_add_gpu(const char* name, double ms) {
    if (!prof_enabled() || !name) return;
    // Completion handlers can fire with a zero/negative delta if the command
    // buffer never ran on the GPU (error or empty encode); don't pollute stats.
    if (!(ms > 0.0)) return;
    ProfState& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    accumulate(s.gpu, name, ms);
}

uint64_t prof_footprint_bytes() {
#if defined(__APPLE__)
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const kern_return_t kr =
        task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) return 0;
    return (uint64_t)info.phys_footprint;
#else
    return 0;
#endif
}

uint64_t prof_available_bytes() {
#if defined(__APPLE__)
    if (@available(iOS 13.0, macOS 10.15, *)) {
        return (uint64_t)os_proc_available_memory();
    }
    return 0;
#else
    return 0;
#endif
}

void prof_mark_memory(const char* label) {
    if (!prof_enabled()) return;
    const uint64_t fp = prof_footprint_bytes();
    const uint64_t avail = prof_available_bytes();
    ProfState& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    if (fp > s.peak_footprint) {
        s.peak_footprint = fp;
        s.peak_label = label ? label : "";
    }
    // available==0 means the API is unavailable, not "no headroom left".
    if (avail > 0 && avail < s.min_available) {
        s.min_available = avail;
        s.min_label = label ? label : "";
    }
}

uint64_t prof_peak_footprint_bytes() {
    ProfState& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    return s.peak_footprint;
}

uint64_t prof_min_available_bytes() {
    ProfState& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    return s.min_available == UINT64_MAX ? 0 : s.min_available;
}

std::string prof_report() {
    if (!prof_enabled()) return std::string();
    ProfState& s = state();
    std::lock_guard<std::mutex> lk(s.m);

    std::ostringstream ss;
    ss << "\n=== HHSR profile ===\n";
    append_table(ss, "CPU stage", "calls", s.cpu);
    append_table(ss, "GPU kernel", "dispatch", s.gpu);

    if (s.peak_footprint > 0) {
        ss << "Memory\n"
           << "  peak footprint             " << fmt_mb(s.peak_footprint)
           << "  @ " << (s.peak_label.empty() ? "?" : s.peak_label) << "\n";
    }
    if (s.min_available != UINT64_MAX) {
        ss << "  min headroom to jetsam     " << fmt_mb(s.min_available)
           << "  @ " << (s.min_label.empty() ? "?" : s.min_label) << "\n";
    }
    ss << "====================\n";
    return ss.str();
}

void prof_reset() {
    ProfState& s = state();
    std::lock_guard<std::mutex> lk(s.m);
    s.cpu.clear();
    s.gpu.clear();
    s.peak_footprint = 0;
    s.peak_label.clear();
    s.min_available = UINT64_MAX;
    s.min_label.clear();
    s.t_origin = 0.0;
}

ProfScope::ProfScope(const char* name)
    : name_(name), t0_(prof_enabled() ? prof_now_ms() : 0.0) {}

ProfScope::~ProfScope() {
    if (!prof_enabled()) return;
    prof_add_cpu(name_, prof_now_ms() - t0_);
}

}  // namespace hhsr
