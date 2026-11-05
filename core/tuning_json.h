// Shared "app settings -> Config" applier for the desktop harness.
//
// The iOS app serialises its TuningParams to a flat JSON object (the same keys
// SRBridge.mm's ApplyTuningParams consumes). This header lets the desktop exe
// load that exact JSON and build an identical Config, so sr_merge runs the same
// algorithm with the same options as the phone. The key->Config mapping here
// MIRRORS SRBridge.mm ApplyTuningParams field-for-field (cfg.* only; the iOS-only
// HDR/ISP/JPEG finishing globals are not part of the merge->DNG+mask output and
// are skipped). Keep the two in sync when adding a tuning parameter.
#pragma once
#include <string>
#include <map>
#include "types.h"

namespace hhsr {

// Apply a parsed {key: number} map to cfg (bools as 0/1, ints rounded).
void apply_tuning_kv(Config& cfg, const std::map<std::string, double>& kv);

// Parse a flat JSON object file ({"key": number|true|false, ...}) and apply it.
// Returns false (and sets *err) if the file can't be read or parsed; unknown
// keys are ignored. Nested objects/arrays and string values are skipped.
bool apply_tuning_json_file(Config& cfg, const std::string& path, std::string* err);

}  // namespace hhsr
