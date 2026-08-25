#include "snr_tuning.h"
#include "stages.h"
#include <algorithm>
#include <cmath>

namespace hhsr {

static f32 lerpf(f32 x, f32 x0, f32 x1, f32 y0, f32 y1) {
    // Matches params.lerp: clip t to [0,1]
    if (x1 <= x0) return y0;
    f32 t = (x - x0) / (x1 - x0);
    t = clampf(t, 0.f, 1.f);
    return y0 + (y1 - y0) * t;
}

static int sanitize_alignment_tile_size(int tile_size) {
    switch (tile_size) {
        case 8:
        case 16:
        case 32:
        case 64:
            return tile_size;
        default:
            return 0;
    }
}

static void set_alignment_tile_sizes(Config& cfg, int base_tile_size) {
    base_tile_size = sanitize_alignment_tile_size(base_tile_size);
    if (base_tile_size <= 0) return;
    cfg.bm_tile_sizes.clear();
    cfg.bm_tile_sizes.reserve(cfg.bm_tile_size_factors.size());
    for (f32 f : cfg.bm_tile_size_factors) {
        int ts = (int)(base_tile_size * f);
        // Keep the Metal alignment path resident for manual 8px tiles: the
        // coarsest level would otherwise become 4px from the 0.5 factor.
        ts = std::max(8, ts);
        cfg.bm_tile_sizes.push_back(ts);
    }
}

void tune_config_snr(const Image& ref_raw, Config& cfg, f32* out_brightness) {
    const int manual_tile_size = sanitize_alignment_tile_size(cfg.alignment_tile_size);
    if (out_brightness) *out_brightness = 0.f;
    if (!cfg.snr_auto_tune || ref_raw.data.empty()) {
        if (manual_tile_size > 0)
            set_alignment_tile_sizes(cfg, manual_tile_size);
        return;
    }

    // Python super_resolution.py: brightness = mean(ref); SNR = brightness / std_curve[round(1000*b)]
    f32 sum = 0.f;
    for (f32 v : ref_raw.data) sum += v;
    f32 brightness = sum / (f32)ref_raw.data.size();
    if (out_brightness) *out_brightness = brightness;
    f32 sigma = noise_std_at_brightness(brightness, cfg);
    f32 snr = (sigma > 1e-8f) ? brightness / sigma : 15.f;
    snr = clampf(snr, 6.f, 30.f);

    // params.update_snr_config
    cfg.k_detail = lerpf(snr, 6.f, 30.f, 0.33f, 0.25f);
    cfg.k_denoise = lerpf(snr, 6.f, 30.f, 5.0f, 3.0f);
    if (!cfg.d_thresh_manual) {
        if (cfg.kernel_google_s1) {
            // Supplement S.2: units of gradient magnitude of the [0,1] image.
            cfg.D_th = lerpf(snr, 6.f, 30.f, 0.010f, 0.001f);
            cfg.D_tr = lerpf(snr, 6.f, 30.f, 0.020f, 0.006f);
        } else {
            cfg.D_th = lerpf(snr, 6.f, 30.f, 0.81f, 0.71f);
            cfg.D_tr = lerpf(snr, 6.f, 30.f, 1.24f, 1.0f);
        }
    }

    int Ts = (snr <= 14.f) ? 64 : (snr <= 22.f) ? 32 : 16;
    // 460-main falls back to 32 because its block matching kernels do not
    // support tiles larger than that.
    if (Ts > 32) Ts = 32;
    set_alignment_tile_sizes(cfg, manual_tile_size > 0 ? manual_tile_size : Ts);
}

} // namespace hhsr
