#include "snr_tuning.h"

namespace hhsr {

static f32 lerpf(f32 x, f32 x0, f32 x1, f32 y0, f32 y1) {
    if (x1 <= x0) return y0;
    x = clampf(x, x0, x1);
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

void tune_config_snr(const Image& ref_raw, Config& cfg) {
    if (!cfg.snr_auto_tune || ref_raw.data.empty()) return;

    f32 sum = 0.f;
    for (f32 v : ref_raw.data) sum += v;
    f32 brightness = sum / (f32)ref_raw.data.size();
    f32 sigma = std::sqrt(std::max(0.f, cfg.alpha * brightness + cfg.beta));
    f32 snr = (sigma > 1e-8f) ? brightness / sigma : 15.f;
    snr = clampf(snr, 6.f, 30.f);

    cfg.k_detail = lerpf(snr, 6.f, 30.f, 0.30f, 0.29f);
    cfg.k_denoise = lerpf(snr, 6.f, 30.f, 5.0f, 3.0f);
    cfg.D_th = lerpf(snr, 6.f, 30.f, 0.81f, 0.71f);
    cfg.D_tr = lerpf(snr, 6.f, 30.f, 1.24f, 1.0f);

    int Ts = (snr <= 14.f) ? 64 : (snr <= 22.f) ? 32 : 16;
    cfg.bm_tile_sizes = {Ts, Ts, Ts, std::max(4, Ts / 2)};
}

} // namespace hhsr
