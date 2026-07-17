#pragma once
#include "types.h"
#include <string>
#include <cstdio>
#include <cstdint>

namespace hhsr {

bool write_linear_dng(const std::string& path, const Image& rgb,
                      const std::string& camera_model = "HandheldSR-x2");

class DngStreamWriter {
public:
    bool open(const std::string& path, int W, int H,
              const std::string& camera_model = "HandheldSR-x2",
              int orientation = 1,
              const float* colorMatrixXYZtoCam = nullptr,
              const float* wbGainsGreenNorm = nullptr,
              bool bakedSrgb = false,
              const std::string& camera_make = "HandheldSR");

    bool write_rows(const uint16_t* rgb16, int nrows);
    bool close();
    ~DngStreamWriter();

private:
    FILE* f_ = nullptr;
    int W_ = 0, H_ = 0;
    long rows_written_ = 0;
};

} // namespace hhsr
