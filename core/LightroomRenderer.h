#pragma once
//
// Lightroom-match JPEG renderer: merged camera-native linear RGB -> display sRGB
// calibrated to reproduce Lightroom Mobile's render of this pipeline's DNG
// (Adobe Color / PV15.4, the fixed slider set the reference pair was shot with).
//
// This is a SEPARATE finishing path from render_isp.cpp's HDR "phone ISP" look
// and from render_match_python14. It does NOT touch the HHSR merge or the DNG:
// it consumes the same camera-native linear RGB the linear DNG stores and emits
// 8-bit sRGB ready for JPEG. Calibrated offline against the handheld_sr_x2
// reference pair; see scratchpad/stage*.py and core/LightroomRendererData.h.
//
// Pipeline, in order (each stage was measured against the reference JPEG):
//   1. white balance      (per-capture gains; default the reference capture's)
//   2. camera->sRGB 3x3   (kWbCamToSrgb, applied to WB'd linear)
//   3. sRGB OETF          (to display-referred)
//   4. tone curve         (1-D on display luma, hue-preserving ratio)
//   5. HueSatMap look     (Adobe Color + Vibrance, 2-D hue x sat correction)
//   6. Clarity            (midtone-weighted local contrast on luma)  [spatial]
//   7. Color NR           (chroma-domain smoothing, luma untouched)  [spatial]
//   8. clamp + 8-bit
//
// The matrix bakes the reference capture's WB when opts.wb is null; pass the
// DNG's own WB (1/AsShotNeutral, green-normalised) to track a different capture.
//
#include <cstdint>
#include <vector>

namespace hhsr {

struct LightroomRenderOpts {
    // 3 green-normalised WB gains applied before the matrix. null -> the
    // reference capture's gains (lr::kRefWB). Pass the DNG's own gains for a
    // different capture so the matrix's neutral assumption holds.
    const float* wb = nullptr;
    bool clarity = true;    // stage 6
    bool color_nr = true;   // stage 7
};

// rgb16: interleaved camera-native LINEAR RGB, 65535 == 1.0 -- exactly what the
// linear DNG stores (pre-white-balance). W*H*3 samples. Writes W*H*3 uint8 sRGB.
// Whole-image because Clarity and Color NR are spatial.
void lightroom_render(const uint16_t* rgb16, int W, int H,
                      const LightroomRenderOpts& opts,
                      std::vector<uint8_t>& out_rgb8);

}  // namespace hhsr
