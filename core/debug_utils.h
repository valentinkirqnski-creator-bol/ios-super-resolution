#pragma once

#include <string>
#include <cstddef>

namespace hhsr {

// Writes float32 dumps under Documents/debug_dumps (iOS) or $HHSR_DEBUG_DIR / debug_dumps.
void debug_ensure_dir();
void debug_dump_bin(const std::string& name, const float* data, size_t size);

// Optional one-line sidecar (e.g. reference path / frame index).
void debug_dump_text(const std::string& name, const std::string& text);

} // namespace hhsr
