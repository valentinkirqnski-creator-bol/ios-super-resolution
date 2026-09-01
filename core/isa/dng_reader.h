#pragma once
// Targeted DNG reader for the ISA-CPU reimplementation. Reads exactly the
// tags ImageStackAlignator's own DNGFile.cs reads (see the project plan,
// Stage 1) -- not a general TIFF/EXIF library. Handles both uncompressed
// strips and Compression=7 (lossless JPEG) tiled raw data, the latter via
// the vendored third_party/lj92 decoder.
#include <cstdint>
#include <string>
#include <vector>

namespace isacpu {

struct DngRaw {
    int width = 0;
    int height = 0;
    std::vector<uint16_t> pixels;  // width*height, row-major, one sample/pixel (Bayer mosaic)

    // CFA colour at (row&1, col&1). 0=R, 1=G, 2=B.
    int cfa[2][2] = {{0, 1}, {1, 2}};

    // Per-CFA-colour black level (R, G, B) -- DNG's BlackLevel is usually one
    // value shared by all four sites, but some cameras give up to 4; we keep
    // R/G/B (green sites always share one value in that case).
    float black_level[3] = {0.f, 0.f, 0.f};
    float white_level = 65535.f;

    // XYZ-to-camera colour matrices (ColorMatrix1/2), row-major 3x3, if
    // present -- ColorMatrix1/CalibrationIlluminant1 is always present when
    // any is; ColorMatrix2/CalibrationIlluminant2 only for dual-illuminant
    // profiles. Illuminant values are the standard EXIF LightSource codes
    // (17=StandardLightA, 21=D65, etc.), 0 if absent/unknown.
    bool has_color_matrix1 = false;
    float color_matrix1[9] = {0};
    int calibration_illuminant1 = 0;
    bool has_color_matrix2 = false;
    float color_matrix2[9] = {0};
    int calibration_illuminant2 = 0;

    // As-shot white balance, camera-native gains (R,G,B), if present.
    bool has_as_shot_neutral = false;
    float as_shot_neutral[3] = {1.f, 1.f, 1.f};

    // DNG NoiseProfile (0xC761): per-channel (alpha, beta), if present.
    bool has_noise_profile = false;
    float noise_alpha[3] = {0.f, 0.f, 0.f};
    float noise_beta[3] = {0.f, 0.f, 0.f};

    int orientation = 1;  // EXIF orientation value (1 = normal)
    int iso = 0;
};

// Reads one DNG file's raw Bayer image + the metadata above. Returns false
// on any file-format problem (missing tags, unsupported compression, I/O
// error) -- check DngRaw::width > 0 to confirm success on true, too.
bool read_dng_raw(const std::string& path, DngRaw& out);

}  // namespace isacpu
