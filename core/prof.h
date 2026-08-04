#pragma once

#include <cstdint>
#include <string>

// Per-stage wall/GPU timing + jetsam-relevant memory watermarks.
//
// Defaults ON for non-release builds (a sideloaded build has no Xcode scheme to
// inject environment variables, so opt-in gating would silently yield no data)
// and OFF when NDEBUG is defined. HHSR_PROF=1/0 overrides either way.
//
// Deliberately does NOT add GPU synchronization: GPU time is harvested from
// MTLCommandBuffer completion handlers (GPUStartTime/GPUEndTime), which fire on
// work the pipeline already waits for. Inserting waitUntilCompleted to measure
// would change the very stalls we are trying to measure.

namespace hhsr {

bool prof_enabled();

// Monotonic milliseconds (mach_absolute_time on Apple, steady_clock elsewhere).
double prof_now_ms();

// Accumulate into a named CPU bucket (total ms + call count).
void prof_add_cpu(const char* name, double ms);

// Accumulate into a named GPU bucket. Call from a completion handler using
// (GPUEndTime - GPUStartTime) * 1000.0.
void prof_add_gpu(const char* name, double ms);

// Bytes charged to this process by jetsam (TASK_VM_INFO phys_footprint).
uint64_t prof_footprint_bytes();

// Bytes still available before this process hits its jetsam limit
// (os_proc_available_memory). Returns 0 when unavailable.
uint64_t prof_available_bytes();

// Sample footprint/available now and fold into the high/low watermarks.
// `label` is retained for whichever sample set the peak footprint.
void prof_mark_memory(const char* label);

// Highest footprint / lowest jetsam headroom seen since the last reset.
uint64_t prof_peak_footprint_bytes();
uint64_t prof_min_available_bytes();

// Human-readable table of everything accumulated since the last reset.
std::string prof_report();

// Writes `text` to <app Documents>/hhsr_profile.txt so a sideloaded build can
// surface results with no debugger attached (the app sets UIFileSharingEnabled,
// so the file is reachable from the Files app). Returns the path written, or an
// empty string on failure. Independent of the debug-dump env gate.
std::string prof_save_report(const std::string& text);

void prof_reset();

// Scoped CPU timer. Zero-cost when profiling is disabled.
class ProfScope {
public:
    explicit ProfScope(const char* name);
    ~ProfScope();
    ProfScope(const ProfScope&) = delete;
    ProfScope& operator=(const ProfScope&) = delete;

private:
    const char* name_;
    double t0_;
};

}  // namespace hhsr

#define HHSR_PROF_CONCAT_(a, b) a##b
#define HHSR_PROF_CONCAT(a, b) HHSR_PROF_CONCAT_(a, b)
#define HHSR_PROF_SCOPE(name) \
    ::hhsr::ProfScope HHSR_PROF_CONCAT(hhsr_prof_scope_, __LINE__)(name)
