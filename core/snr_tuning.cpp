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
    // align_fine_finest_tile: move the small (0.5) factor from the coarsest to
    // the FINEST level -- {1,1,1,0.5} -> {0.5,1,1,1}, i.e. {16,16,16,8} ->
    // {8,16,16,16}. Finest/merge tile halves (less within-tile rotation error);
    // coarse levels stay large for a robust global estimate. See Config.
    std::vector<f32> factors = cfg.bm_tile_size_factors;
    if (cfg.align_fine_finest_tile && factors.size() >= 2)
        std::swap(factors.front(), factors.back());
    for (f32 f : factors) {
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

    // IPOL main estimate_image_snr (utils_image.py): per-CFA-channel SNR in dB,
    //   SNR = 20*log10( sqrt( sum(v^2) / sum(alpha_c*v + beta_c) ) )
    // over pixels with 0 < v < 1. The tile-size thresholds and the k_detail/
    // k_denoise/D_th/D_tr lerps below already match the reference; only the SNR
    // value changes (was mean-brightness / noise-std).
    double top = 0.0, bot = 0.0;
    const int rw = ref_raw.w, rh = ref_raw.h;
    for (int y = 0; y < rh; ++y) {
        for (int x = 0; x < rw; ++x) {
            const f32 v = ref_raw.data[(size_t)y * (size_t)rw + (size_t)x];
            if (!(v > 0.f && v < 1.f)) continue;
            const int c = cfg.bayer_mode ? (int)cfg.cfa.p[y & 1][x & 1] : 0;
            const f32 a = cfg.noise_alpha_ch(c), b = cfg.noise_beta_ch(c);
            top += (double)v * (double)v;
            bot += (double)a * (double)v + (double)b;
        }
    }
    if (out_brightness) {
        double s = 0.0; for (f32 v : ref_raw.data) s += (double)v;
        *out_brightness = (f32)(s / (double)std::max<size_t>(1, ref_raw.data.size()));
    }
    f32 snr;
    if (bot > 0.0 && top > 0.0) {
        const f32 snr_lin = (f32)std::sqrt(top / bot);
        snr = 20.f * std::log10(std::max(1e-6f, snr_lin));
    } else {
        snr = 15.f;
    }
    snr = clampf(snr, 6.f, 30.f);

    // params.update_snr_config
    cfg.k_detail = lerpf(snr, 6.f, 30.f, 0.33f, 0.25f);
    cfg.k_denoise = lerpf(snr, 6.f, 30.f, 5.0f, 3.0f);
    cfg.D_th = lerpf(snr, 6.f, 30.f, 0.81f, 0.71f);
    cfg.D_tr = lerpf(snr, 6.f, 30.f, 1.24f, 1.0f);

    int Ts = (snr <= 14.f) ? 64 : (snr <= 22.f) ? 32 : 16;
    // IPOL main match: the newer reference supports tile 64 (dedicated BM/ICA
    // kernels), and so does this port (l1_bm_ts64, the ts==64 ICA branch), so
    // the old "clamp to 32" 460 fallback is removed -- low-SNR bursts use 64.
    set_alignment_tile_sizes(cfg, manual_tile_size > 0 ? manual_tile_size : Ts);
}

} // namespace hhsr
