#include <cstdint>
#include <string>
namespace hhsr {
uint64_t prof_available_bytes() { return 0; }
uint64_t prof_footprint_bytes() { return 0; }
void prof_mark_memory(const char*) {}
uint64_t prof_peak_footprint_bytes() { return 0; }
uint64_t prof_min_available_bytes() { return 0; }
std::string prof_report() { return std::string(); }
std::string prof_save_report(const std::string&) { return std::string(); }
void prof_reset() {}
void mps_fft_prewarm(int, int) {}
void mps_fft_release_buffers() {}
}
