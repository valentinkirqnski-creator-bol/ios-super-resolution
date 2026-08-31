// Burst processing for memory-constrained mobile targets (iOS).
// Full-res (1×) and large bursts: spill comparison Bayer to disk after analyze.
// Lighter 2× crops keep RAW+analysis in RAM. Apple: GPU merge prefetch after analyze.
#include "pipeline.h"
#include "stages.h"
#include "dng_writer.h"
#include "snr_tuning.h"
#include "raw_io.h"
#include "parallel.h"
#include "debug_utils.h"
#include "prof.h"
#include "mps_fft.h"
#include "preset_lut.h"
#if defined(__APPLE__)
#include "metal_gpu.h"
#endif
#include <vector>
#include <array>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <future>
#include <deque>
#include <mutex>
#include <memory>
#include <limits>
#include <sstream>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif
#if defined(__APPLE__)
#include <thread>
#endif

namespace fs = std::filesystem;

namespace hhsr {

// std::async and std::thread create raw pthreads. On Apple platforms those
// start at the default QoS instead of inheriting the dispatch queue that
// spawned them, so although the burst runs on a .userInitiated queue, the
// decode prefetch, the Bayer spill, kernel estimation and the DNG encode were
// all running a tier below the work that blocks on them. A lower QoS is also a
// scheduling hint toward efficiency cores, which have both lower clocks and
// less memory bandwidth -- and every one of these workers is bandwidth-bound.
//
// Purely a scheduling change: no computation is affected.
static inline void worker_qos() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
}

// Prefetch decoders run a tier LOWER still. At USER_INITIATED, three
// concurrent LibRaw decodes competed with parallel_rows (same QoS) for the
// P-cores and the whole analyze stage inflated: align 42->111ms, robustness
// 37->380ms per frame -- the decode stall was eliminated but re-exported as
// compute contention, and the burst got slower overall. UTILITY steers the
// decoders to spare E-core cycles (it is not the throttled BACKGROUND tier),
// leaving the P-cores to the pipeline that consumes their output. A decode
// takes longer on an E-core, which is exactly what the depth-3 queue is for.
static inline void decode_qos() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
}


namespace {

static bool save_image(const fs::path& p, const Image& im) {
    int32_t hdr[3] = {im.h, im.w, im.c};
    std::ofstream out(p, std::ios::binary);
    if (!out) return false;
    out.write((const char*)hdr, sizeof(hdr));
    out.write((const char*)im.data.data(), (std::streamsize)(im.data.size() * sizeof(f32)));
    return out.good();
}

static bool load_image(const fs::path& p, Image& im) {
    int32_t hdr[3];
    std::ifstream in(p, std::ios::binary);
    if (!in || !in.read((char*)hdr, sizeof(hdr))) return false;
    // Reuse storage when shape matches — avoids alloc+zero on every reload.
    if (im.h != hdr[0] || im.w != hdr[1] || im.c != hdr[2])
        im = Image(hdr[0], hdr[1], hdr[2]);
    return (bool)in.read((char*)im.data.data(), (std::streamsize)(im.data.size() * sizeof(f32)));
}

struct CachedCompFrame {
    FlowField flow;
    Image rob;
    CovField covs;
    Image comp;
    std::vector<uint8_t> rob_rows_nonzero;
    bool rob_has_nonzero = true;
    int index = 0;
};

struct CachedCompMeta {
    FlowField flow;
    Image rob;
    CovField covs;
    std::vector<uint8_t> rob_rows_nonzero;
    bool rob_has_nonzero = true;
    int index = 0;
};

static bool load_cached_comp_raw(const fs::path& cache, int k, Image& comp) {
    return load_image(cache / ("f" + std::to_string(k) + ".raw"), comp) && comp.h > 0;
}

static bool robustness_row_activity(const Image& rob, std::vector<uint8_t>& rows) {
    rows.assign((size_t)std::max(0, rob.h), 0u);
    if (rob.h <= 0 || rob.w <= 0 || rob.c <= 0 || rob.data.empty())
        return false;
    bool any = false;
    for (int y = 0; y < rob.h; ++y) {
        bool row_any = false;
        const size_t base = (size_t)y * (size_t)rob.w * (size_t)std::max(1, rob.c);
        const size_t count = (size_t)rob.w * (size_t)std::max(1, rob.c);
        for (size_t i = 0; i < count; ++i) {
            if (rob.data[base + i] != 0.f) {
                row_any = true;
                break;
            }
        }
        rows[(size_t)y] = row_any ? 1u : 0u;
        any = any || row_any;
    }
    return any;
}

static Image compute_robustness_and_activity(const Image& comp,
                                             const RefStats& ref_stats,
                                             const FlowField& flow,
                                             int tile_size,
                                             const Config& work,
                                             std::vector<uint8_t>& rows,
                                             bool& has_nonzero) {
    Image rob = compute_robustness(comp, ref_stats, flow, tile_size, work);
    has_nonzero = robustness_row_activity(rob, rows);
    return rob;
}

static inline int round_away_to_int(f32 x) {
    return (int)std::lround(x);
}

static bool robustness_band_can_contribute(const std::vector<uint8_t>& rows,
                                           int y0, int bh, float scale,
                                           bool bayer_mode) {
    if (rows.empty()) return true;
    const int rh = (int)rows.size();
    for (int local_y = 0; local_y < bh; ++local_y) {
        const f32 lr_y = (f32)(y0 + local_y) / scale;
        int ry;
        if (bayer_mode)
            ry = std::min(std::max(round_away_to_int((lr_y - 0.5f) / 2.f), 0), rh - 1);
        else
            ry = std::min(std::max(round_away_to_int(lr_y), 0), rh - 1);
        if (rows[(size_t)ry]) return true;
    }
    return false;
}

struct DebugImageStats {
    f32 min_v = std::numeric_limits<f32>::infinity();
    f32 max_v = -std::numeric_limits<f32>::infinity();
    double mean = 0.0;
    size_t nonfinite = 0;
};

static DebugImageStats image_stats(const Image& im) {
    DebugImageStats s;
    if (im.data.empty()) {
        s.min_v = 0.f;
        s.max_v = 0.f;
        return s;
    }
    double sum = 0.0;
    size_t finite_n = 0;
    for (f32 v : im.data) {
        if (!std::isfinite(v)) {
            ++s.nonfinite;
            continue;
        }
        s.min_v = std::min(s.min_v, v);
        s.max_v = std::max(s.max_v, v);
        sum += v;
        ++finite_n;
    }
    if (finite_n == 0) {
        s.min_v = 0.f;
        s.max_v = 0.f;
    }
    s.mean = finite_n ? sum / (double)finite_n : 0.0;
    return s;
}

struct DebugCovStats {
    f32 min_det = std::numeric_limits<f32>::infinity();
    f32 max_det = -std::numeric_limits<f32>::infinity();
    size_t det_nonfinite = 0;
    size_t det_nonpositive = 0;
    size_t det_tiny = 0;
};

static DebugCovStats cov_stats(const CovField& covs) {
    DebugCovStats s;
    if (covs.cov.empty()) {
        s.min_det = 0.f;
        s.max_det = 0.f;
        return s;
    }
    const size_t n = (size_t)covs.h * (size_t)covs.w;
    for (size_t i = 0; i < n; ++i) {
        const f32* c = &covs.cov[i * 4u];
        f32 det = c[0] * c[3] - c[1] * c[1];
        if (!std::isfinite(det)) {
            ++s.det_nonfinite;
            continue;
        }
        s.min_det = std::min(s.min_det, det);
        s.max_det = std::max(s.max_det, det);
        if (det <= 0.f) ++s.det_nonpositive;
        else if (det < 1e-12f) ++s.det_tiny;
    }
    if (!std::isfinite(s.min_det)) s.min_det = 0.f;
    if (!std::isfinite(s.max_det)) s.max_det = 0.f;
    return s;
}

struct MergeDebugStats {
    size_t pixels = 0;
    size_t channel_zero[3] = {0, 0, 0};
    size_t channel_tiny[3] = {0, 0, 0};
    size_t channel_nonfinite[3] = {0, 0, 0};
    size_t all_zero = 0;
    size_t only_green = 0;
    f32 min_den[3] = {
        std::numeric_limits<f32>::infinity(),
        std::numeric_limits<f32>::infinity(),
        std::numeric_limits<f32>::infinity()
    };
    f32 max_den[3] = {
        -std::numeric_limits<f32>::infinity(),
        -std::numeric_limits<f32>::infinity(),
        -std::numeric_limits<f32>::infinity()
    };
    std::string samples;
    size_t sample_count = 0;
};

static bool analyze_merge_band(const Image& num, const Image& den, int y0,
                               MergeDebugStats& total) {
    constexpr f32 tiny = 1e-12f;
    bool suspicious_band = false;
    const int nch = std::min(3, den.c);
    for (int y = 0; y < den.h; ++y) {
        for (int x = 0; x < den.w; ++x) {
            ++total.pixels;
            f32 d[3] = {0.f, 0.f, 0.f};
            f32 n[3] = {0.f, 0.f, 0.f};
            bool bad_pixel = false;
            for (int ch = 0; ch < nch; ++ch) {
                d[ch] = den.at(y, x, ch);
                n[ch] = num.at(y, x, ch);
                if (std::isfinite(d[ch])) {
                    total.min_den[ch] = std::min(total.min_den[ch], d[ch]);
                    total.max_den[ch] = std::max(total.max_den[ch], d[ch]);
                    if (d[ch] == 0.f) {
                        ++total.channel_zero[ch];
                        bad_pixel = true;
                    } else if (d[ch] > 0.f && d[ch] < tiny) {
                        ++total.channel_tiny[ch];
                        bad_pixel = true;
                    }
                } else {
                    ++total.channel_nonfinite[ch];
                    bad_pixel = true;
                }
            }
            const bool r_bad = nch > 0 && (!std::isfinite(d[0]) || d[0] <= tiny);
            const bool g_ok = nch > 1 && std::isfinite(d[1]) && d[1] > tiny;
            const bool b_bad = nch > 2 && (!std::isfinite(d[2]) || d[2] <= tiny);
            const bool all_bad = (nch == 1)
                ? r_bad
                : (r_bad && (!g_ok) && b_bad);
            const bool only_g = (nch >= 3 && r_bad && g_ok && b_bad);
            if (all_bad) {
                ++total.all_zero;
                bad_pixel = true;
            }
            if (only_g) {
                ++total.only_green;
                bad_pixel = true;
            }
            if (bad_pixel) {
                suspicious_band = true;
                if (total.sample_count < 2000) {
                    char line[320];
                    std::snprintf(line, sizeof(line),
                        "y=%d x=%d den=(%.9g,%.9g,%.9g) num=(%.9g,%.9g,%.9g)\n",
                        y0 + y, x, d[0], d[1], d[2], n[0], n[1], n[2]);
                    total.samples += line;
                    ++total.sample_count;
                }
            }
        }
    }
    return suspicious_band;
}

static void append_image_summary(std::ostringstream& ss, const char* name, const Image& im) {
    DebugImageStats s = image_stats(im);
    ss << name << " shape=" << im.h << "x" << im.w << "x" << im.c
       << " min=" << s.min_v << " max=" << s.max_v
       << " mean=" << s.mean << " nonfinite=" << s.nonfinite << "\n";
}

static void append_cov_summary(std::ostringstream& ss, const char* name, const CovField& covs) {
    DebugCovStats s = cov_stats(covs);
    ss << name << " shape=" << covs.h << "x" << covs.w
       << " det_min=" << s.min_det << " det_max=" << s.max_det
       << " det_nonfinite=" << s.det_nonfinite
       << " det_nonpositive=" << s.det_nonpositive
       << " det_tiny=" << s.det_tiny << "\n";
}

static void append_merge_summary(std::ostringstream& ss, const MergeDebugStats& s) {
    ss << "merge_pixels=" << s.pixels << "\n";
    const char* channel_name[3] = {"R", "G", "B"};
    for (int ch = 0; ch < 3; ++ch) {
        const f32 min_den = std::isfinite(s.min_den[ch]) ? s.min_den[ch] : 0.f;
        const f32 max_den = std::isfinite(s.max_den[ch]) ? s.max_den[ch] : 0.f;
        ss << "den_ch" << ch << "_" << channel_name[ch]
           << " min=" << min_den
           << " max=" << max_den
           << " zero=" << s.channel_zero[ch]
           << " tiny=" << s.channel_tiny[ch]
           << " nonfinite=" << s.channel_nonfinite[ch] << "\n";
    }
    ss << "den_all_zero=" << s.all_zero << "\n";
    ss << "den_only_green=" << s.only_green << "\n";
    ss << "den_sample_count=" << s.sample_count << "\n";
    if (!s.samples.empty()) {
        ss << "den_samples\n";
        ss << s.samples;
    }
}

static void absorb_robustness_sum(Image& acc_rob, const Image& rob, bool& have) {
    if (!have) {
        acc_rob = Image(rob.h, rob.w, 1);
        acc_rob.data = rob.data;
        have = true;
        return;
    }
    for (size_t i = 0; i < rob.data.size(); ++i)
        acc_rob.data[i] += rob.data[i];
}

static void build_robustness_sum(const std::vector<CachedCompFrame>& cached,
                                 const std::vector<CachedCompMeta>& cached_meta,
                                 bool stream_comp_raw,
                                 Image& acc_rob, bool& have) {
    if (stream_comp_raw) {
        for (const CachedCompMeta& meta : cached_meta)
            absorb_robustness_sum(acc_rob, meta.rob, have);
    } else {
        for (const CachedCompFrame& fc : cached)
            absorb_robustness_sum(acc_rob, fc.rob, have);
    }
}

static void encode_band_rows_ptr(const f32* nump, const f32* denp, int y0, int bh,
                                 const Config& work, int nch, Image& preview, float pscale,
                                 int ph, int pw, int Ws, std::vector<uint16_t>& row16) {
    // Same num/den → RGB16 math as before; pointer loops + sparse preview only.
    auto to_srgb = [](f32 v) {
        if (!std::isfinite(v)) return 0.f;
        v = clampf(v, 0.f, 1.f);
        return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
    };
    auto safe_div = [](f32 n, f32 d) {
        if (!(d > 0.f) || !std::isfinite(n) || !std::isfinite(d)) return 0.f;
        const f32 v = n / d;
        return std::isfinite(v) ? v : 0.f;
    };
    const int x_step = std::max(1, (int)std::ceil(1.f / std::max(pscale, 1e-6f)));
    // Preview is UI-only; sample a bit more sparsely (DNG pixels unchanged).
    const int y_step = std::max(1, x_step);
    const bool bake = work.bake_srgb && nch >= 3;
    const f32* m = work.cam_to_srgb;
    // Pre-whitened RAW already has WB baked in (Python utils_dng order).
    const f32 wb0 = work.raw_prewhitened ? 1.f : work.white_balance[0];
    const f32 wb1 = work.raw_prewhitened ? 1.f : work.white_balance[1];
    const f32 wb2 = work.raw_prewhitened ? 1.f : work.white_balance[2];
    const bool prev_color = !bake && nch >= 3 && work.has_cam_to_srgb;
    // Un-white-balance the STORED rows only (see Config::dng_store_unwhitened);
    // the preview keeps the white-balanced values it always showed.
    float sg[3];
    dng_unwhiten_gains(work, nch, sg);

#if defined(__APPLE__)
    // Dense DNG band on GPU (1:1); sparse preview stays on CPU below.
    const bool gpu_rgb = metal_normalize_band_rgb16_ptr(nump, denp, bh, Ws, nch, work, row16);
#else
    const bool gpu_rgb = false;
#endif
    uint16_t* outp = row16.data();
    if (!gpu_rgb) {
        row16.resize((size_t)bh * (size_t)Ws * 3u);
        outp = row16.data();
    }

    parallel_rows(bh, work.num_threads, [&](int i) {
        const int gy = y0 + i;
        const bool do_prev_row = ((gy % y_step) == 0);
        // When the GPU produced the dense rows, this loop exists ONLY for the
        // sparse preview -- stride straight to the sampled positions instead
        // of iterating all 48M pixels to skip them (measured: a large slice
        // of the encode tail doing nothing).
        if (gpu_rgb && !do_prev_row) return;
        const int px_step = gpu_rgb ? x_step : 1;
        const int py = std::min(ph - 1, (int)(gy * pscale));
        const size_t row_off = (size_t)i * (size_t)Ws * (size_t)nch;
        for (int x = 0; x < Ws; x += px_step) {
            const bool need_prev = do_prev_row && (x % x_step) == 0;
            if (gpu_rgb && !need_prev) continue;

            const size_t pi = row_off + (size_t)x * (size_t)nch;
            f32 cn0 = safe_div(nump[pi], denp[pi]);
            f32 cn1 = 0.f, cn2 = 0.f;
            if (nch >= 2) {
                cn1 = safe_div(nump[pi + 1], denp[pi + 1]);
            }
            if (nch >= 3) {
                cn2 = safe_div(nump[pi + 2], denp[pi + 2]);
            }
            f32 lin0, lin1, lin2;
            if (bake) {
                f32 wr = cn0 * wb0, wg = cn1 * wb1, wb = cn2 * wb2;
                lin0 = m[0] * wr + m[1] * wg + m[2] * wb;
                lin1 = m[3] * wr + m[4] * wg + m[5] * wb;
                lin2 = m[6] * wr + m[7] * wg + m[8] * wb;
            } else if (nch >= 3) {
                lin0 = cn0; lin1 = cn1; lin2 = cn2;
            } else {
                lin0 = lin1 = lin2 = cn0;
            }
            if (!gpu_rgb) {
                const f32 v0 = bake ? to_srgb(lin0) : clampf(lin0 * sg[0], 0.f, 1.f);
                const f32 v1 = bake ? to_srgb(lin1) : clampf(lin1 * sg[1], 0.f, 1.f);
                const f32 v2 = bake ? to_srgb(lin2) : clampf(lin2 * sg[2], 0.f, 1.f);
                const size_t base = ((size_t)i * (size_t)Ws + (size_t)x) * 3u;
                outp[base + 0] = (uint16_t)(v0 * 65535.f + 0.5f);
                outp[base + 1] = (uint16_t)(v1 * 65535.f + 0.5f);
                outp[base + 2] = (uint16_t)(v2 * 65535.f + 0.5f);
            }
            if (need_prev) {
                f32 preview_lin[3] = {lin0, lin1, lin2};
                if (prev_color) {
                    f32 wr = cn0 * wb0, wg = cn1 * wb1, wb = cn2 * wb2;
                    preview_lin[0] = m[0] * wr + m[1] * wg + m[2] * wb;
                    preview_lin[1] = m[3] * wr + m[4] * wg + m[5] * wb;
                    preview_lin[2] = m[6] * wr + m[7] * wg + m[8] * wb;
                } else if (!bake && nch >= 3) {
                    preview_lin[0] = cn0 * wb0;
                    preview_lin[1] = cn1 * wb1;
                    preview_lin[2] = cn2 * wb2;
                }
                const int px = std::min(pw - 1, (int)(x * pscale));
                if (preset_lut_enabled()) {
                    // The LUT was fitted from the DNG's linear values straight
                    // to final sRGB, so it already contains white balance, the
                    // colour matrix and gamma. Feed it the unmodified merge
                    // rather than the matrixed preview_lin above, or those
                    // steps get applied twice.
                    const f32 dng_lin[3] = {lin0, lin1, lin2};
                    f32 srgb[3];
                    preset_lut_apply(dng_lin, srgb);
                    for (int k = 0; k < 3; ++k)
                        preview.at(py, px, k) = clampf(srgb[k], 0.f, 1.f);
                } else {
                    for (int k = 0; k < 3; ++k)
                        preview.at(py, px, k) = to_srgb(clampf(preview_lin[k], 0.f, 1.f));
                }
            }
        }
    });
}

// Banded callers hold a band-shaped image; the online caller holds the whole
// accumulator and offsets a row into it.
static void encode_band_rows(const Image& num_band, const Image& den_band, int y0, int bh,
                             const Config& work, int nch, Image& preview, float pscale,
                             int ph, int pw, int Ws, std::vector<uint16_t>& row16) {
    encode_band_rows_ptr(num_band.data.data(), den_band.data.data(), y0, bh,
                         work, nch, preview, pscale, ph, pw, Ws, row16);
}

// Working set of each merge architecture, so the cheaper one can be picked
// rather than assumed. Banded keeps every frame's flow/cov/robustness (and the
// Bayer too unless it is spilled) alive across the whole merge; online keeps
// one accumulator sized to the output plus the single frame in flight.
//
// "Flat in frame count" is not automatically "smaller": the online accumulator
// scales with OUTPUT pixels, so at 2x it costs 4x what it does at 1x and loses
// to banding until the burst is long.
// Frames committed per command buffer.
//
// One. Committing several together lets merge_flush_pending reach the fusion
// threshold in metal_gpu.mm, so a group shares one read-modify-write of the
// accumulator instead of one each -- worth roughly 700ms across a 15-frame
// burst at 48 MP. But each queued frame's GPU upload stays alive until the
// commit, and the online accumulator is already resident alongside the whole
// analysis working set, so those frames land directly on the peak. Three extra
// frames are 313MB on a ceiling that decides whether online is usable at all.
//
// Memory wins: the ceiling is what does not move with burst length, and the
// traffic cost is spread across a burst already running for seconds.
static constexpr int kOnlineFuse = 1;


// Burst length at or above which the merge goes online.
static constexpr int kOnlineMinFrames = 6;

static bool choose_online_merge(int mode, int Hs, int Ws, int nch, int n,
                                bool spill, int band_rows, int raw_h, int raw_w,
                                bool fft_grey) {
    if (mode == 1) return false;              // forced banded
    if (mode == 2) return true;               // forced online

    if (n < kOnlineMinFrames) return false;

    const size_t f = sizeof(f32);
    const size_t raw = (size_t)raw_h * (size_t)raw_w * f;
    const size_t half = (size_t)(raw_h / 2) * (size_t)(raw_w / 2);
    const size_t covs = half * 4 * f;
    const size_t rob = half * f;
    // What one comparison frame costs across the whole banded merge.
    //
    // Not reduced by spilling. Spilling writes the host Bayer to disk, but the
    // frame has already been uploaded to the GPU by metal_merge_prefetch_frame
    // and stays there so every band can reuse it -- and on unified memory that
    // is the same RAM. Charging only covs+rob here understated a 13-frame burst
    // by 528MB, which was enough to pick banding and then peak at 2144MB.
    (void)spill;
    const size_t frame = raw + covs + rob;
    const size_t in_flight = frame;

    // Reference-side state that survives into the merge.
    const size_t fixed = raw + covs + rob;
    // Reference-side state released only when the analysis loop ends: the grey,
    // its pyramid, the local statistics, the FFT scratch, and the frame being
    // decoded plus the one prefetched behind it.
    //
    // This term is why the comparison below is between peaks and not between
    // merge working sets. Online allocates its accumulator on the first frame,
    // which is inside the analysis loop, so the accumulator and all of this are
    // resident at the same time -- they never were under banding, where the
    // accumulator is band-sized and only appears after this has been freed.
    // Ignoring the overlap is what put a 48 MP burst at 2240 MB.
    const size_t temp = half * f                 // grey
                      + half * f * 4 / 3         // pyramid
                      + 2 * half * 3 * f         // means + vars, 3 channels
                      // The FFT grey needs a full-res complex scratch; the 2x2
                      // Bayer quad grey does not, and that 93MB at 12MP sits
                      // directly on the online peak because it is alive while
                      // the accumulator is.
                      + (fft_grey ? raw * 2 : 0)
                      + raw * 2;                 // current + prefetched frame

    const size_t acc_full = (size_t)Hs * Ws * nch * f * 2;
    // Bands exist twice over: the double-buffered host images the encoder reads
    // and the matching pair of GPU accumulator slots, plus the two staging
    // buffers of 16-bit output rows.
    const size_t band_host = (size_t)band_rows * Ws * nch * f * 2 * 2;
    const size_t band_acc = band_host * 2 + (size_t)band_rows * Ws * 3 * 2 * 2;
    const size_t frames = (size_t)std::max(0, n - 1) * frame;

    const size_t banded_peak = std::max(fixed + temp + frames, fixed + band_acc + frames);
    const size_t online_peak = fixed + temp + acc_full
                             + (size_t)kOnlineFuse * in_flight;
    if (online_peak >= banded_peak) return false;

    // Hard fallback: if the peak will not fit at all, banding is the only thing
    // that runs. 0 means the query is unavailable, i.e. no known constraint.
    const uint64_t avail = prof_available_bytes();
    constexpr uint64_t kOnlineHeadroom = 400ull * 1024ull * 1024ull;
    if (avail != 0 && (uint64_t)online_peak + kOnlineHeadroom > avail) return false;
    return true;
}

} // namespace

Image process_burst_loader_to_dng(int frame_count, const RawFrameLoaderFn& loader,
                                  const Config& cfg, const std::string& dng_path,
                                  const ProgressFn& progress, int maxPreviewDim,
                                  Rgb16Sink* rgb16_sink) {
    if (rgb16_sink) { rgb16_sink->w = 0; rgb16_sink->h = 0; rgb16_sink->rgb.clear(); }
    if (frame_count < 2 || !loader) return Image();
    if (cfg.hdrplus_mode)
        return process_burst_loader_to_dng_hdrplus(frame_count, loader, cfg, dng_path,
                                                   progress, maxPreviewDim, rgb16_sink);

    prof_reset();
    prof_mark_memory("burst:begin");
    const double t_burst = prof_now_ms();

    Config work = cfg;
    // Drives the reference-kernel enlargement; not a tuning knob.
    work.burst_frame_count = frame_count;
    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };
    const bool debug = debug_dumps_enabled();
    std::ostringstream debug_summary;
    if (debug) debug_summary << "cpp_debug_summary\n";

    report("Loading first frame", 0.02f);
    const double t_first = prof_now_ms();
    Image first_ref = loader(0, work, true, 0, 0);
    prof_add_cpu("ref:decode", prof_now_ms() - t_first);
    if (first_ref.h <= 0 || first_ref.w <= 0) return Image();

    Image ref = std::move(first_ref);
    if (ref.h <= 0 || ref.w <= 0) return Image();

    const int ref_index = 0;

    // MPSGraph compiles its FFT plan on first use (~1100ms at 12MP), and that
    // landed squarely on the reference grey. Start it here, as soon as the
    // dimensions are known, so it overlaps everything up to that point. Only a
    // partial hide -- the compile is longer than the work ahead of it -- so the
    // real fix is calling mps_fft_prewarm when the camera configures, well
    // before the shutter. Harmless when MPSGraph is unavailable.
    std::future<void> mps_warm = std::async(std::launch::async, [&]() {
        worker_qos();
        mps_fft_prewarm(ref.h, ref.w);
    });

    clear_align_ref_ica_cache();
    debug_dump_bin("cpp_raw_ref", ref.data.data(), ref.data.size());
    if (debug) append_image_summary(debug_summary, "raw_ref", ref);
    if (debug) {
        std::string listing = "ref_index=" + std::to_string(ref_index) + "\n";
        for (int i = 0; i < frame_count; ++i)
            listing += std::to_string(i) + " provider_frame\n";
        debug_dump_text("cpp_burst_paths", listing);
    }
    // Prefetch state for the comparison decodes: an in-order queue of decode
    // futures, seeded below once SNR tuning has been joined and topped back up
    // each loop iteration. Depth 1 was not enough once the per-frame pipeline
    // work dropped to ~130ms: a DNG decode takes ~390ms, so a single prefetch
    // launched one frame ahead left comp:decode-wait stalling ~265ms/frame --
    // the consumer outran the producer. Depth 3 keeps three decodes in flight
    // (390 / 130 rounds to 3), which is enough for the queue's front to be
    // ready when the loop reaches it. Scheduling only: the same frames are
    // decoded in the same comp_indices order and produce the same values;
    // decode-order independence holds because a comparison decode's only
    // Config write is re-storing raw_prewhitened = true (all metadata writes
    // are is_reference-gated, and the reference decoded long ago).
    //
    // Costs up to two more resident Bayer frames (~49MB each) than depth 1.
    // Peak footprint is at merge:band, not during analyze; headroom to jetsam
    // was 1527MB on the profiled run.
    struct PrefetchedDecode {
        int k = -1;
        std::future<Image> fut;
    };
    std::deque<PrefetchedDecode> pref_q;
    const int kDecodePrefetchDepth = 3;
    // Crop dims by value: the decode thread may still be running after the
    // reference image's pixels have been stashed and freed.
    const int pref_ref_h = ref.h, pref_ref_w = ref.w;
    int pref_seq = 0;
    auto launch_prefetch = [&, pref_ref_h, pref_ref_w](int frame) {
        // The first two decodes get the responsive tier: their only cover is
        // the brief reference stage, and UTILITY there was the measured 1-2s
        // "analyzing frame 2" stall variant. Later decodes stay on the
        // efficiency tier so they cannot steal P-cores from the analyze loop.
        const bool eager = pref_seq++ < 2;
        pref_q.push_back({frame, std::async(std::launch::async,
                                            [&, frame, pref_ref_h, pref_ref_w, eager]() {
            if (eager) worker_qos(); else decode_qos();
            return loader(frame, work, false, pref_ref_h, pref_ref_w);
        })});
    };

    // Accumulated robustness. Summed inside the loop on the streamed path so
    // each frame's mask can be released as soon as it is on the GPU, rather
    // than every mask being held until after the loop.
    Image acc_rob;
    bool have_acc_rob = false;

    const double t_snr = prof_now_ms();
    f32 ref_brightness = 0.f;
    // The SNR scan is a serial sum over 12.2M floats. It cannot be vectorized
    // (float addition is not associative) and must not be parallelized: its
    // result picks the alignment tile size and k_detail/k_denoise/D_th/D_tr, so
    // the summation order is load-bearing. What it can do is run while the GPU
    // computes the reference grey.
    //
    // Safe because the two touch disjoint state: compute_grey reads only
    // work.bayer_mode and work.grey_method, while tune_config_snr writes only
    // k_detail, k_denoise, D_th, D_tr and bm_tile_sizes. Distinct non-bitfield
    // members are separate memory locations, so there is no race, and the sum
    // itself is byte-for-byte the same sum in the same order.
    std::future<void> snr_fut = std::async(std::launch::async, [&]() {
        worker_qos();
        tune_config_snr(ref, work, &ref_brightness);
    });

    // Reference grey on the GPU, overlapping the scan above.
    report("Reference: grey + pyramid", 0.07f);
    prof_mark_memory("ref:start");
    const double t_ref_grey = prof_now_ms();
    // 460-main block matching circular-pads the reference before pyramid construction.
    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method,
                                  work.grey_decimate_lowpass);
    debug_dump_bin("cpp_ref_grey", ref_grey.data.data(), ref_grey.data.size());

    prof_add_cpu("ref:grey(gpu)", prof_now_ms() - t_ref_grey);

    // Everything below needs the tuned tile size, so join here. Timed on its
    // own: this is the residual wait after the grey, not the scan's whole
    // duration. Previously this bucket and ref:grey+pyramid both spanned the
    // grey, so the two double-counted it and the report read as a regression.
    const double t_snr_join = prof_now_ms();
    snr_fut.get();
    prof_add_cpu("setup:snr-join(residual)", prof_now_ms() - t_snr_join);

    // Prewarm the robustness noise curves on a worker. On a curve-cache MISS
    // (every new ISO changes alpha/beta and with them the cache key) the
    // Monte-Carlo builds otherwise run synchronously inside the FIRST
    // comparison frame's robustness call -- the "analyzing frame 2" freeze.
    // The curves' cache is mutex-guarded, so if the frame catches up it waits
    // for the in-flight build rather than duplicating it. Same keys, same
    // values; only where the build runs changes. After snr_fut.get() because
    // the accessors and tune_config_snr share the noise model.
    std::future<void> noise_warm = std::async(std::launch::async, [&]() {
        worker_qos();
        robustness_prewarm_noise_curves(work);
    });

    // Seed the decode queue now, up to its full depth. These used to run
    // synchronously at the top of the comparison loop (comp:decode(sync),
    // ~187ms with nothing overlapping it); the pyramid build plus robustness
    // and kernel estimation now cover the first, and by the time the loop
    // consumes frame 1 the queue is refilling behind it.
    //
    // Deliberately after snr_fut.get() rather than before the reference stage:
    // the loader takes Config by non-const reference and may write to it, while
    // tune_config_snr reads cfg.alpha and cfg.beta. Overlapping those two could
    // race on the noise model, changing the SNR and therefore the alignment
    // tile size -- an algorithm change, not an optimization.
    //
    // Same enumeration as comp_indices below (every frame but the reference,
    // in order), so the queue's front always matches the loop's next k.
    {
        int seeded = 0;
        for (int i = 0; i < frame_count && seeded < kDecodePrefetchDepth; ++i) {
            if (i == ref_index) continue;
            launch_prefetch(i);
            ++seeded;
        }
    }

    // Band-limited full-res grey of the reference for the full-res ICA
    // polish; built lazily at the first comparison frame, held for the burst
    // (~48MB). Empty when the polish is off or failed to build.
    Image ref_grey_polish;

    // Same UI status line as "Frame N: analyze" (CameraModel.statusText).
    {
        // Same value tune_config_snr just computed. It used to be summed again
        // here, and the two scans together were the single largest CPU stage in
        // the burst -- 325ms of serial float adds over 12.2M samples, which the
        // compiler cannot vectorize because float addition is not associative.
        // Reusing the result is bit-identical; a parallel reduction would not be.
        const f32 brightness = ref_brightness;
        const f32 sigma = noise_std_at_brightness(brightness, work);
        const f32 snr = (sigma > 1e-8f) ? brightness / sigma : 0.f;
        char buf[352];
        const int t0 = work.bm_tile_sizes.size() > 0 ? work.bm_tile_sizes[0] : -1;
        std::snprintf(buf, sizeof(buf),
            "Noise %s α=%.3g β=%.3g  b=%.2f σ=%.2e SNR=%.1f  T=%d  r_t=%.2f",
            work.has_noise_profile ? "OK" : "FALLBACK",
            work.noise_alpha(), work.noise_beta(), brightness, sigma, snr, t0,
            work.r_t);
        report(buf, 0.065f);
    }
    (void)t_snr;

    const int ref_h = ref.h, ref_w = ref.w;
    const int n = frame_count;
    const int tile_size = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    // Output dimensions are known as soon as the reference is, and the online
    // merge needs its accumulator before the first frame is analyzed.
    const int out_h = (int)std::lround(work.scale * (f32)ref_h);
    const int out_w = (int)std::lround(work.scale * (f32)ref_w);
    const int out_nch = work.bayer_mode ? 3 : 1;
    const int nch = work.bayer_mode ? 3 : 1;
    std::vector<int> comp_indices;
    comp_indices.reserve((size_t)std::max(0, n - 1));
    for (int i = 0; i < n; ++i) {
        if (i != ref_index) comp_indices.push_back(i);
    }

    const double t_pyr = prof_now_ms();
    Image ref_grey_padded = pad_image_circular(ref_grey, work.grey_tile_size(tile_size));
    Pyramid ref_pyr = build_pyramid(ref_grey_padded, work.bm_factors);
    prof_add_cpu("ref:pad+pyramid", prof_now_ms() - t_pyr);
    if (debug && !ref_pyr.levels.empty()) {
        // Match Python py_pyramid_0: first after pyramid[::-1] = coarsest.
        const Image& coarse = ref_pyr.levels.back();
        debug_dump_bin("cpp_pyramid_0", coarse.data.data(), coarse.data.size());
    }
    // Full-res (~12MP Bayer) cannot hold every comparison RAW + dual Metal peaks.
    const bool full_res =
        ((size_t)ref_h * (size_t)ref_w) >= 8ull * 1000ull * 1000ull;

    // Same math; sequential on Apple Metal (rob + kernels both GPU — overlap
    // doubles peak and races shared scratch). Light crops may still overlap.
    RefStats ref_stats;
    CovField ref_covs;
    const double t_ref_analyze = prof_now_ms();
    if (full_res) {
        ref_stats = init_robustness(ref, work);
        ref_covs = estimate_kernels(ref, work);
    } else {
        std::future<CovField> ref_cov_fut =
            std::async(std::launch::async, [&]() { worker_qos(); return estimate_kernels(ref, work); });
        ref_stats = init_robustness(ref, work);
        ref_covs = ref_cov_fut.get();
    }
    prof_add_cpu("ref:robustness+kernels", prof_now_ms() - t_ref_analyze);
    prof_mark_memory("ref:analyzed");
    if (debug) append_cov_summary(debug_summary, "cov_ref", ref_covs);
#if defined(__APPLE__)
    // GPU already holds ref means/vars; drop host copies to cut peak RAM.
    metal_release_host_ref_stats(ref_stats);
#endif

    // Keep ref Bayer in RAM for merge (avoids a second LibRaw decode).
    // Peak during analyze ≈ ref + one comparison (+ optional prefetch on 2×).

    fs::path cache = fs::path(dng_path).parent_path() /
                     (fs::path(dng_path).stem().string() + "_cache");
    std::error_code ec;
    fs::remove_all(cache, ec);

    // Stream Bayer on full-res (any burst size) or when comps > 8. Keeps flow/R/cov.
    std::vector<CachedCompFrame> cached;
    std::vector<CachedCompMeta> cached_meta;
    const bool stream_comp_raw = full_res || (n - 1) > 8;
    const bool cache_streamed_comp_raw =
        stream_comp_raw && !work.stream_comp_raw_from_loader;
    auto load_streamed_comp_raw = [&](int frame_index, Image& comp) -> bool {
        if (cache_streamed_comp_raw)
            return load_cached_comp_raw(cache, frame_index, comp);
        comp = loader(frame_index, work, false, ref_h, ref_w);
        return comp.h > 0 && comp.w > 0;
    };
    if (cache_streamed_comp_raw)
        fs::create_directories(cache, ec);

    // Online merges each frame into one full-size accumulator and drops it
    // immediately, so the working set stops growing with the burst. It is not
    // free: that accumulator scales with OUTPUT pixels, so at 2x it costs four
    // times what it does at 1x and loses to banding until the burst is long.
    // choose_online_merge compares the two rather than assuming.
    bool use_online = false;
#if defined(__APPLE__)
    use_online = choose_online_merge(work.merge_arch, out_h, out_w, out_nch, n,
                                     cache_streamed_comp_raw, 480, ref_h, ref_w,
                                     work.grey_method == GreyMethod::FFT);
#endif
    // No host accumulator: the GPU buffer is shared storage, so it is read in
    // place. Allocating a matching host image would double the largest
    // allocation in the pipeline to hold the same bytes twice.
    Image num_sink, den_sink;
    std::vector<int> online_pending;   // frames queued but not yet committed
    int online_skip_rejected = 0, online_skip_nodata = 0, online_skip_gpu = 0;
    const int online_fuse = kOnlineFuse;

    cached.reserve(use_online ? 0 : (size_t)std::max(0, n - 1));
    cached_meta.reserve(use_online ? 0 : (size_t)std::max(0, n - 1));

    // pref_q is declared above: the first kDecodePrefetchDepth comparison
    // decodes are already in flight, launched before the reference analysis.
    std::future<bool> spill_fut;
    bool spill_pending = false;
    int n_comp_ok = 0;
#if defined(__APPLE__)
    // Opened here, not at merge time, so each comparison frame can be uploaded
    // as soon as it is analyzed. set_single_acc_slot must precede begin_burst,
    // which derives the initial write slot from it.
    metal_merge_set_single_acc_slot(false);
    // false: the reference robustness statistics live only on the GPU by now
    // (metal_release_host_ref_stats above), and the analyze trim would clear
    // them. The trim still runs later, once analysis is finished.
    metal_merge_begin_burst(/*trim_analyze_scratch*/ false);
    // After begin_burst, which clears online state along with the rest of the
    // merge globals.
    if (use_online)
        metal_merge_begin_online(out_h, out_w, out_nch,
                                 work.merge_fp16_accumulator);
    prof_mark_memory(use_online ? "merge:online-armed" : "merge:banded-armed");
#endif

    auto drain_spill = [&]() {
        if (!spill_fut.valid()) return;
        const bool ok = spill_fut.get();
        if (!ok && spill_pending && !cached_meta.empty()) {
            cached_meta.pop_back();
            n_comp_ok = std::max(0, n_comp_ok - 1);
        }
        spill_pending = false;
    };
    for (int pos = 0; pos < (int)comp_indices.size(); ++pos) {
        const double t_frame_total = prof_now_ms();
        const int k = comp_indices[(size_t)pos];
        const double t_rep = prof_now_ms();
        report("Frame " + std::to_string(k + 1) + ": analyze",
               0.08f + 0.35f * (float)pos / std::max(1, n - 1));
        prof_add_cpu("comp:progress-report", prof_now_ms() - t_rep);
        const double t_spill = prof_now_ms();
        drain_spill();
        prof_add_cpu("comp:spill-drain", prof_now_ms() - t_spill);

        Image comp;
        const double t_decode = prof_now_ms();
        if (!pref_q.empty() && pref_q.front().k == k && pref_q.front().fut.valid()) {
            comp = pref_q.front().fut.get();
            pref_q.pop_front();
            // Stall only. Near-zero here means the prefetch fully hid the decode.
            prof_add_cpu("comp:decode-wait(prefetched)", prof_now_ms() - t_decode);
        } else {
            comp = loader(k, work, false, ref_h, ref_w);
            prof_add_cpu("comp:decode(sync)", prof_now_ms() - t_decode);
        }
        if (comp.h <= 0) continue;
        debug_dump_bin("cpp_raw_comp_" + std::to_string(pos),
                       comp.data.data(), comp.data.size());
        if (debug) {
            const std::string raw_name = "raw_comp_" + std::to_string(pos);
            append_image_summary(debug_summary, raw_name.c_str(), comp);
        }

        // Top the decode queue back up here on both paths, immediately after
        // this frame's decode is consumed, so the new decode overlaps grey +
        // align + robustness + kernels of this frame AND the queued decodes
        // ahead of it. After popping the front, the queue holds positions
        // pos+1 .. pos+size, so the next frame to launch is comp_indices at
        // pos + 1 + size. Scheduling only -- the same frames are decoded in
        // the same order and produce the same values.
        {
            const double t_pf = prof_now_ms();
            while ((int)pref_q.size() < kDecodePrefetchDepth &&
                   pos + 1 + (int)pref_q.size() < (int)comp_indices.size()) {
                launch_prefetch(comp_indices[(size_t)(pos + 1 + (int)pref_q.size())]);
            }
            prof_add_cpu("comp:prefetch-launch", prof_now_ms() - t_pf);
        }

        const double t_comp_grey = prof_now_ms();
        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method,
                                       work.grey_decimate_lowpass);
        prof_add_cpu("comp:grey", prof_now_ms() - t_comp_grey);
        debug_dump_bin("cpp_mov_grey_" + std::to_string(pos),
                       comp_grey.data.data(), comp_grey.data.size());
        const double t_align = prof_now_ms();
        FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, tile_size);
        // Alignment ran on the grey. With the Bayer quad average that is
        // half resolution, so the flow is on a half-res tile grid with
        // half-res displacements, while robustness and merge both index
        // it as raw_coordinate / tile_size and add raw pixels. Convert
        // here, before anything downstream sees it. No-op when the grey
        // is full resolution, so the FFT path is unaffected.
        flow = flow_to_raw_tile_grid(flow, comp.h, comp.w,
                                     comp_grey.h, comp_grey.w, tile_size,
                                     work.r_Mt, work.num_threads,
                                     work.grey_tile_size(tile_size));
        flow.sample_bicubic = work.flow_bicubic_sampling &&
                              work.flow_bilinear_sampling;
        // Full-res ICA polish (decimate only): re-measure the finished flow's
        // sub-pixel part at RAW resolution on the band-limited FFT grey --
        // the image the FFT path measures on. The half-res estimate seeds it,
        // already sub-pixel, so this is last-mile refinement inside ICA's
        // basin. The boundary densify below then measures on the SAME
        // full-res greys, doubling its precision too.
        const bool polish = work.align_fullres_polish &&
                            work.bayer_mode &&
                            work.grey_method == GreyMethod::Decimate;
        if (polish) {
            const double t_pol = prof_now_ms();
            if (ref_grey_polish.h <= 0)   // once per burst, first comp frame
                ref_grey_polish =
                    pad_image_circular(compute_grey_fft(ref), tile_size);
            Image comp_grey_polish =
                pad_image_circular(compute_grey_fft(comp), tile_size);
            if (ref_grey_polish.h > 0 && comp_grey_polish.h > 0) {
                flow_fullres_ica_polish(ref_grey_polish, comp_grey_polish,
                                        flow, tile_size, work);
                flow_densify_boundary_select(flow, ref_grey_polish,
                                             comp_grey_polish,
                                             comp.h, comp.w, tile_size, work);
            } else {
                flow_densify_boundary_select(flow, ref_grey, comp_grey,
                                             comp.h, comp.w, tile_size, work);
            }
            prof_add_cpu("comp:fullres-polish", prof_now_ms() - t_pol);
        } else {
            // Boundary-selected half-pitch refinement (fine grid); must run
            // here, while both greys are still alive. No-op when off.
            flow_densify_boundary_select(flow, ref_grey, comp_grey,
                                         comp.h, comp.w, tile_size, work);
        }
        prof_add_cpu("comp:align", prof_now_ms() - t_align);
        prof_mark_memory("analyze:after-align");
        debug_dump_bin("cpp_flow_" + std::to_string(pos),
                       flow.flow.data(), flow.flow.size());
        const double t_freeg = prof_now_ms();
        comp_grey = Image(); // free before robustness/kernels peak
        prof_add_cpu("comp:free-grey", prof_now_ms() - t_freeg);
        if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty()) {
            report("Frame " + std::to_string(k + 1) + ": Metal alignment failed",
                   0.08f + 0.35f * (float)(pos + 1) / std::max(1, n - 1));
            continue;
        }

        Image rob;
        CovField covs;
        std::vector<uint8_t> rob_rows_nonzero;
        bool rob_has_nonzero = true;
        if (full_res) {
            // Keep rob ∥ kernels serialized — dual Metal peaks jetsam on 1×.
            const double t_rob = prof_now_ms();
            rob = compute_robustness_and_activity(comp, ref_stats, flow,
                                                  tile_size, work,
                                                  rob_rows_nonzero,
                                                  rob_has_nonzero);
            prof_add_cpu("comp:robustness", prof_now_ms() - t_rob);
            // robustness_row_activity is a pure CPU pass over the finished mask,
            // while estimate_kernels is GPU, so these two can run together. This
            // is NOT the case the comment above warns about: only one Metal
            // workload is in flight, so there is no dual Metal peak.
            //
            // Kernels are now computed unconditionally instead of behind
            // rob_has_nonzero. That is not a behaviour change: covs is already
            // discarded downstream when the mask is empty
            // (meta.covs = rob_has_nonzero ? std::move(covs) : CovField()), so
            // the result was never used in that case. It costs a wasted GPU pass
            // only when a comparison frame's mask is entirely zero, which means
            // the frame contributes nothing at all -- rare, and it does not
            // raise the peak because the allocation happens either way here.
            const double t_kern = prof_now_ms();
            std::future<CovField> cov_fut =
                std::async(std::launch::async, [&]() { worker_qos(); return estimate_kernels(comp, work); });
            covs = cov_fut.get();
            prof_add_cpu("comp:kernels", prof_now_ms() - t_kern);
        } else {
            // Same math; overlap only when peak RAM is affordable (2× crop).
            std::future<CovField> cov_fut =
                std::async(std::launch::async, [&]() { worker_qos(); return estimate_kernels(comp, work); });
            rob = compute_robustness_and_activity(comp, ref_stats, flow,
                                                  tile_size, work,
                                                  rob_rows_nonzero,
                                                  rob_has_nonzero);
            covs = cov_fut.get();
        }
        debug_dump_bin("cpp_mask_" + std::to_string(pos),
                       rob.data.data(), rob.data.size());
        if (debug) {
            const std::string mask_name = "mask_" + std::to_string(pos);
            const std::string cov_name = "cov_comp_" + std::to_string(pos);
            append_image_summary(debug_summary, mask_name.c_str(), rob);
            append_cov_summary(debug_summary, cov_name.c_str(), covs);
        }
        const double t_stash = prof_now_ms();
        if (use_online) {
#if defined(__APPLE__)
            // Same frames in the same order the banded path walks, so the
            // robustness sum is float-for-float what it produces.
            absorb_robustness_sum(acc_rob, rob, have_acc_rob);
            bool merged = true;
            bool contributed = false;
            if (!rob_has_nonzero) {
                online_skip_rejected++;
            } else if (comp.h <= 0 || comp.w <= 0) {
                online_skip_nodata++;
            } else if (!merge_comp_band(comp, flow, covs, rob, tile_size,
                                        num_sink, den_sink, 0, work, k)) {
                // The GPU refused the frame -- geometry, an allocation, or the
                // upload. Dropping that silently is how a burst ends up as the
                // reference frame alone with nothing reported anywhere.
                online_skip_gpu++;
            } else {
                contributed = true;
                online_pending.push_back(k);
                // Commit in groups, not per frame: the accumulator fusion needs
                // several frames queued together, and a frame's GPU buffers
                // cannot be released until its work has run.
                if ((int)online_pending.size() >= online_fuse) {
                    // Non-blocking: commit and continue into the next frame's
                    // analysis while this merge runs. The frame's uploads are
                    // released when the NEXT commit (or the final blocking
                    // flush) retires this one -- so at most one extra frame
                    // (~110-146 MB) is ever resident, independent of burst
                    // length. The old blocking wait here cost ~160 ms/frame of
                    // pure CPU idle against a ~90 ms GPU merge.
                    merged = metal_merge_commit_online(online_pending);
                    online_pending.clear();
                }
            }
            // Everything this frame owned goes here. Nothing is carried to the
            // merge phase, which is what makes the working set flat in n.
            comp = Image();
            rob = Image();
            covs = CovField();
            flow = FlowField();
            rob_rows_nonzero.clear();
            rob_rows_nonzero.shrink_to_fit();
            if (!merged) {
                report("Error: GPU merge failed (memory?)", 1.f);
                metal_merge_end_online();
                if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
                return Image();
            }
            // Only frames that reached the accumulator count. Counting every
            // analyzed frame meant a burst where none of them merged still
            // looked fully analysed, and the output -- the reference alone --
            // came out with no error at all.
            if (contributed) n_comp_ok++;
            prof_mark_memory("merge:online-frame");
#endif
        } else if (stream_comp_raw) {
            // Keep flow/R/cov in RAM. For DNG-file input, spill normalized Bayer
            // async; for direct RAW, reload the original uint16 frame later.
            // Grow-only L2/Alg.5 scratch stays until merge.
            // Hand the frame to the GPU now. It has to go there before the
            // merge either way, and doing it here replaces a 48.8MB write to
            // disk plus the matching read back (comp:spill-drain plus most of
            // merge:gpu-upload) with the upload that was going to happen anyway.
            //
            // Peak footprint is unaffected: these buffers are resident during
            // merge:band regardless, which is where the peak occurs. This only
            // moves the allocation earlier, into a phase whose working set is
            // well below that peak.
            bool uploaded_to_gpu = false;
#if defined(__APPLE__)
            // Uploading each frame as it is analyzed removes the disk round
            // trip, but residency grows with the burst -- roughly 110MB per
            // comparison frame at 12MP once image, covariance and mask are
            // counted. A 15-frame burst would add about 770MB over an 8-frame
            // one, which is more than the headroom allows.
            //
            // So stop when headroom gets tight and fall back to spilling. A long
            // burst then degrades in speed rather than being killed, and short
            // bursts are unaffected. prof_available_bytes returns 0 when the
            // query is unavailable, which is treated as no constraint.
            constexpr uint64_t kMinHeadroomForUpload = 700ull * 1024ull * 1024ull;
            const uint64_t avail = prof_available_bytes();
            const bool headroom_ok = (avail == 0) || (avail > kMinHeadroomForUpload);
            if (rob_has_nonzero && comp.h > 0 && comp.w > 0 && headroom_ok)
                uploaded_to_gpu = metal_merge_prefetch_frame(comp, flow, covs, rob, k);
#endif
            if (uploaded_to_gpu) {
                comp = Image();
            } else if (cache_streamed_comp_raw && rob_has_nonzero) {
                const int sk = k;
                auto spill_img = std::make_shared<Image>(std::move(comp));
                const fs::path spill_path = cache / ("f" + std::to_string(sk) + ".raw");
                spill_fut = std::async(std::launch::async, [spill_path, spill_img]() {
                    worker_qos();
                    return save_image(spill_path, *spill_img);
                });
                spill_pending = true;
            } else {
                comp = Image();
            }
            // Same frames in the same order build_robustness_sum walked, so
            // the sum is float-for-float what it produced.
            absorb_robustness_sum(acc_rob, rob, have_acc_rob);

            CachedCompMeta meta;
            meta.index = k;
            if (uploaded_to_gpu) {
                // The GPU holds img, flow, cov and rob for this frame, and
                // acquire_frame_gpu serves them from that cache for every band,
                // so the host copies are dead weight from here to the merge --
                // about 61MB per frame, 427MB across a burst. Peak footprint
                // moved to analyze:frame-end once frames began uploading during
                // analysis, which is exactly where this was accumulating.
                meta.flow = FlowField();
                meta.rob = Image();
                meta.covs = CovField();
            } else {
                meta.flow = rob_has_nonzero ? std::move(flow) : FlowField();
                meta.rob = std::move(rob);
                meta.covs = rob_has_nonzero ? std::move(covs) : CovField();
            }
            meta.rob_rows_nonzero = std::move(rob_rows_nonzero);
            meta.rob_has_nonzero = rob_has_nonzero;
            cached_meta.push_back(std::move(meta));
            if (rob_has_nonzero) n_comp_ok++;
        } else {
            CachedCompFrame fc;
            fc.index = k;
            fc.flow = rob_has_nonzero ? std::move(flow) : FlowField();
            fc.rob = std::move(rob);
            fc.covs = rob_has_nonzero ? std::move(covs) : CovField();
            fc.comp = rob_has_nonzero ? std::move(comp) : Image();
            fc.rob_rows_nonzero = std::move(rob_rows_nonzero);
            fc.rob_has_nonzero = rob_has_nonzero;
            cached.push_back(std::move(fc));
            if (rob_has_nonzero) n_comp_ok++;
        }
        prof_add_cpu("comp:stash+free-raw", prof_now_ms() - t_stash);
        prof_add_cpu("comp:frame(total)", prof_now_ms() - t_frame_total);
        prof_mark_memory("analyze:frame-end");
    }
    drain_spill();
    if (mps_warm.valid()) mps_warm.get();
    if (noise_warm.valid()) noise_warm.get();
    for (auto& p : pref_q)                      // drain unused prefetches
        if (p.fut.valid()) (void)p.fut.get();
    pref_q.clear();

    if (n_comp_ok < 1) {
        if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
        char why[160];
        std::snprintf(why, sizeof(why),
                      "Error: no comparison frame merged (rejected %d, no data %d, gpu %d)",
                      online_skip_rejected, online_skip_nodata, online_skip_gpu);
        report(why, 1.f);
#if defined(__APPLE__)
        metal_merge_end_online();
#endif
        return Image();
    }

    // Release reference-side helpers not needed during merge.
    // The grey FFT is finished for this burst, so everything MPSGraph holds --
    // the compiled plan as well as the staging buffers -- is dead weight through
    // the merge, which is where peak footprint occurs. Rebuilt after the burst
    // below, off the shutter path.
    // Buffers only. Releasing the compiled graph as well meant the next burst
    // recompiled it (~1100ms at 12MP) on its reference frame whenever the
    // detached prewarm had not finished -- which is the intermittent slow FFT.
    // The buffers are the memory that has to go before the merge peak.
    mps_fft_release_buffers();
    clear_align_ref_ica_cache();
    ref_grey = Image();
    ref_pyr = Pyramid();
    ref_stats = RefStats();

    if (ref.h <= 0) {
        if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
        report("Error: reference frame missing for merge", 1.f);
#if defined(__APPLE__)
        metal_merge_end_online();
#endif
        return Image();
    }

    const int Hs = (int)std::lround(work.scale * ref.h);
    const int Ws = (int)std::lround(work.scale * ref.w);

    const bool accumulate_r =
        work.accumulated_robustness_denoiser_enabled || work.robustness_save_mask;
    const double t_accrob = prof_now_ms();
    // Streamed frames were summed in the loop above, as each was released.
    if (!stream_comp_raw)
        build_robustness_sum(cached, cached_meta, stream_comp_raw, acc_rob, have_acc_rob);
    prof_add_cpu("merge:acc-rob-sum", prof_now_ms() - t_accrob);
    const Image* acc_rob_ptr = (accumulate_r && have_acc_rob) ? &acc_rob : nullptr;
    if (have_acc_rob) {
        debug_dump_bin("cpp_acc_rob", acc_rob.data.data(), acc_rob.data.size());
        if (debug) append_image_summary(debug_summary, "acc_rob", acc_rob);
    }

    const double t_open = prof_now_ms();
    DngStreamWriter writer;
    const std::string& model = work.camera_model.empty() ? std::string("HandheldSR-x2") : work.camera_model;
    const std::string& make = work.camera_make.empty() ? std::string("HandheldSR") : work.camera_make;
    if (!writer.open(dng_path, Ws, Hs, model, work.orientation,
                     work.has_color_matrix ? work.color_matrix : nullptr,
                     work.white_balance,
                     work.bake_srgb, make,
                     work.has_cam_to_srgb ? work.cam_to_srgb : nullptr,
                     // Unwhitened rows are true camera-space raw again: the
                     // writer then emits AsShotNeutral = 1/gain and stores
                     // the real gains in the private tag for the app render.
                     work.raw_prewhitened && !dng_unwhiten_active(work, nch),
                     work.dng_lossless_jpeg)) {
        if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
        report("Error: cannot open output DNG", 1.f);
#if defined(__APPLE__)
        metal_merge_end_online();
#endif
        return Image();
    }

    const float pscale = std::min(1.f, (float)maxPreviewDim / (float)std::max(Hs, Ws));
    const int ph = std::max(1, (int)(Hs * pscale));
    const int pw = std::max(1, (int)(Ws * pscale));
    Image preview(ph, pw, 3);

    // Band size: both 1× and 2× double-buffer GPU+host.
    //
    // Accumulator RAM is (band_budget x slot_count), so a half-size band with
    // two slots costs exactly what one full-size slot cost before, while
    // letting band N+1 encode against slot B while band N is still executing
    // in slot A. Single-slot 1× had to waitUntilCompleted before every band
    // (ensure_acc_buffers), which serialized the whole merge.
    //
    // Host side shrinks outright: num/den dual bands are 4 x band bytes, so
    // halving the band halves that too (~370MB -> ~185MB at full-res scale-2).
    // Same math: band height never affects a pixel's accumulation order.
    const size_t out_px = (size_t)Hs * (size_t)Ws;
    const bool heavy_1x = out_px >= 28ull * 1000ull * 1000ull; // ~full-res scale-2
#if defined(__APPLE__)
    const size_t band_budget = heavy_1x ? (96u * 1024u * 1024u) : (384u * 1024u * 1024u);
#else
    const size_t band_budget = 64u * 1024u * 1024u;
#endif
    const size_t bytes_per_row = (size_t)Ws * nch * 4 * 2; // num+den float row
    int band_rows = (int)std::max<size_t>(4, band_budget / std::max<size_t>(1, bytes_per_row));
    band_rows = std::min(band_rows, Hs);
    // ~480 rows fits the 96MB per-slot budget at full-res scale-2 (2 slots).
    if (heavy_1x) band_rows = std::min(band_rows, 480);
    std::vector<uint16_t> row16((size_t)band_rows * (size_t)Ws * 3u);
    prof_add_cpu("merge:open+alloc", prof_now_ms() - t_open);

    Image comp_scratch;
#if defined(__APPLE__)
    // Upload comparison frames to GPU before the band loop so merge isn't
    // stalled on the first band's PCIe copies. On full-res, drop host R/cov
    // after each upload so we never hold host+GPU copies of every frame.
    report("Preparing GPU merge", 0.46f);
    const double t_prefetch = prof_now_ms();
    // Double-buffered on 1× too: the band was halved above to keep peak
    // accumulator RAM identical while restoring cross-band GPU overlap.
    // begin_burst already ran before the analysis loop and would clear the
    // frames uploaded there. All that is still wanted here is the scratch trim
    // it used to do, so merge is not fighting the analyze temporaries.
    metal_trim_analyze_scratch();
    if (stream_comp_raw) {
        for (CachedCompMeta& meta : cached_meta) {
            if (!meta.rob_has_nonzero) continue;
            if (metal_merge_has_frame(meta.index)) {
                // Uploaded during analysis; nothing to reload.
                if (heavy_1x) {
                    meta.rob = Image();
                    meta.covs = CovField();
                    meta.flow = FlowField();
                }
                continue;
            }
            if (!load_streamed_comp_raw(meta.index, comp_scratch)) continue;
            if (metal_merge_prefetch_frame(comp_scratch, meta.flow, meta.covs, meta.rob,
                                           meta.index)) {
                meta.rob = Image();
                meta.covs = CovField();
                meta.flow = FlowField();
            }
            if (heavy_1x) comp_scratch = Image();
        }
        comp_scratch = Image();
    } else {
        for (CachedCompFrame& fc : cached) {
            if (!fc.rob_has_nonzero) continue;
            if (metal_merge_prefetch_frame(fc.comp, fc.flow, fc.covs, fc.rob, fc.index)) {
                fc.comp = Image();
                if (heavy_1x) {
                    fc.rob = Image();
                    fc.covs = CovField();
                    fc.flow = FlowField();
                }
            }
        }
    }
    // Includes the disk reload / re-decode of every comparison frame on 1×.
    prof_add_cpu("merge:gpu-upload", prof_now_ms() - t_prefetch);
    prof_mark_memory("merge:frames-resident");
#endif

    AccumDiag diag;
    MergeDebugStats merge_debug;
#if defined(__APPLE__)
    // Dual host bands + async Deflate for both 1× and 2×. 1× still uses a
    // single GPU acc slot (ensure_acc waits/readbacks before reuse).
    Image num_bands[2], den_bands[2];
    std::vector<uint16_t> row16_async[2];
    row16_async[0].resize(row16.size());
    row16_async[1].resize(row16.size());
    if (use_online) {
        // The reference is the last contribution, and it needs the accumulated
        // robustness, which only exists once every frame has been merged.
        report("Merging output", 0.48f);
        const double t_ref_online = prof_now_ms();
        if (!online_pending.empty()) {
            if (!metal_merge_flush_online()) {
                report("Error: GPU merge failed (memory?)", 1.f);
                metal_merge_end_online();
                writer.close();
                if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
                return Image();
            }
            for (int fk : online_pending) metal_merge_release_frame(fk);
            online_pending.clear();
        }
        merge_ref_band(ref, ref_covs, num_sink, den_sink, 0, work, acc_rob_ptr);
        prof_add_cpu("merge:online-ref+readback", prof_now_ms() - t_ref_online);
        prof_mark_memory("merge:online");
        // Rows come out of one finished accumulator, so encoding is a plain
        // top-to-bottom sweep. Every band is fetched through
        // metal_merge_read_band_float: in fp32 mode that is the same zero-copy
        // pointer the old whole-image map handed out, in fp16 mode a GPU pass
        // widens just this band into reused staging -- so the encoder below
        // and the diagnostics never know which storage the accumulator used.
        bool got = true;
        const double t_encode = prof_now_ms();
        // Ping-pong row16_async so the serial Deflate + fwrite of band N runs
        // on a worker while the CPU encodes band N+1. Exactly one write is in
        // flight at a time (the writer's z_stream is stateful and rows must
        // land in order); the join before each launch enforces that and keeps
        // the buffer being written untouched, since the encoder only ever
        // fills the OTHER buffer. Same rows, same order, same bytes -- only
        // the overlap is new.
        std::future<void> write_fut;
        int enc_cur = 0;
        for (int y0 = 0; y0 < Hs; y0 += band_rows) {
            const int bh = std::min(band_rows, Hs - y0);
            const f32* nump = nullptr;
            const f32* denp = nullptr;
            if (!metal_merge_read_band_float(y0, bh, &nump, &denp)) {
                got = false;
                break;
            }
            accumulate_diag_ptr(nump, denp, (size_t)bh * (size_t)Ws, nch, diag);
            std::vector<uint16_t>& r16 = row16_async[enc_cur];
            if (r16.size() < (size_t)bh * (size_t)Ws * 3u)
                r16.resize((size_t)bh * (size_t)Ws * 3u);
            encode_band_rows_ptr(nump, denp, y0, bh, work, nch, preview, pscale,
                                 ph, pw, Ws, r16);
            // Keep the finished rows for the in-memory export (see Rgb16Sink).
            // Plain copy of bytes the encoder just produced -- the DNG on disk
            // and this buffer are the same values by construction.
            if (rgb16_sink) {
                if (rgb16_sink->rgb.empty()) {
                    rgb16_sink->w = Ws;
                    rgb16_sink->h = Hs;
                    rgb16_sink->rgb.assign((size_t)Hs * (size_t)Ws * 3u, 0);
                }
                std::copy(r16.begin(),
                          r16.begin() + (size_t)bh * (size_t)Ws * 3u,
                          rgb16_sink->rgb.begin() + (size_t)y0 * (size_t)Ws * 3u);
            }
            if (write_fut.valid()) write_fut.get();
            write_fut = std::async(std::launch::async, [&writer, &r16, bh]() {
                worker_qos();
                writer.write_rows(r16.data(), bh);
            });
            enc_cur ^= 1;
            report("Merging output", 0.48f + 0.50f * (float)(y0 + bh) / Hs);
        }
        if (write_fut.valid()) write_fut.get();
        // The 48 MP divide + ISP + 16-bit pack + DNG write was invisible in
        // every profile so far -- roughly 2 s of wall with no bucket. Named so
        // the next profile shows it instead of leaving it as unexplained wall.
        prof_add_cpu("out:encode-bands", prof_now_ms() - t_encode);
        if (!got) {
            report("Error: GPU merge failed (memory?)", 1.f);
            metal_merge_end_online();
            writer.close();
            if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
            return Image();
        }
        // Accumulator released with the burst, not before: the band pointers
        // above aliased it (fp32 mode) until the loop finished.
        metal_merge_end_online();
    } else {
    int cur = 0;
    bool have_ready = false;
    int ready = 0, ready_y0 = 0, ready_bh = 0;
    std::thread encode_thr;
    auto join_encode = [&]() {
        if (encode_thr.joinable()) encode_thr.join();
    };
    report("Merging output", 0.48f);
    for (int y0 = 0; y0 < Hs; y0 += band_rows) {
        const double t_band = prof_now_ms();
        const int bh = std::min(band_rows, Hs - y0);
        cur ^= 1;
        const double t_join = prof_now_ms();
        join_encode();
        prof_add_cpu("merge:wait-dng-encode", prof_now_ms() - t_join);
        Image& num_band = num_bands[cur];
        Image& den_band = den_bands[cur];
        if (num_band.h != bh || num_band.w != Ws || num_band.c != nch) {
            try {
                num_band = Image(bh, Ws, nch);
                den_band = Image(bh, Ws, nch);
            } catch (...) {
                report("Error: out of memory during merge", 1.f);
                join_encode();
                metal_merge_set_single_acc_slot(false);
                writer.close();
                if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
                return Image();
            }
        }

        // On 1× (single acc slot) the previous band's waitUntilCompleted +
        // readback happens inside the first ensure_acc_buffers below, so this
        // bucket is "GPU stall + encode", not encode alone.
        const double t_comp_bands = prof_now_ms();
        if (stream_comp_raw) {
            for (const CachedCompMeta& meta : cached_meta) {
                if (!meta.rob_has_nonzero ||
                    !robustness_band_can_contribute(meta.rob_rows_nonzero, y0, bh,
                                                    work.scale, work.bayer_mode))
                    continue;
                if (metal_merge_has_frame(meta.index)) {
                    Image empty;
                    merge_comp_band(empty, meta.flow, meta.covs, meta.rob, tile_size,
                                    num_band, den_band, y0, work, meta.index);
                    continue;
                }
                if (!load_streamed_comp_raw(meta.index, comp_scratch)) continue;
                merge_comp_band(comp_scratch, meta.flow, meta.covs, meta.rob, tile_size,
                                num_band, den_band, y0, work, meta.index);
            }
        } else {
            for (const CachedCompFrame& fc : cached) {
                if (!fc.rob_has_nonzero ||
                    !robustness_band_can_contribute(fc.rob_rows_nonzero, y0, bh,
                                                    work.scale, work.bayer_mode))
                    continue;
                if (metal_merge_has_frame(fc.index)) {
                    Image empty;
                    merge_comp_band(empty, fc.flow, fc.covs, fc.rob, tile_size,
                                    num_band, den_band, y0, work, fc.index);
                    continue;
                }
                if (fc.comp.h <= 0) continue;
                merge_comp_band(fc.comp, fc.flow, fc.covs, fc.rob, tile_size,
                                num_band, den_band, y0, work, fc.index);
            }
        }

        prof_add_cpu("merge:comp-encode+gpu-stall", prof_now_ms() - t_comp_bands);

        const double t_ref_band = prof_now_ms();
        const bool ref_band_ok =
            merge_ref_band_metal(ref, ref_covs, num_band, den_band, y0, work, acc_rob_ptr);
        prof_add_cpu("merge:ref-encode+commit", prof_now_ms() - t_ref_band);
        if (!ref_band_ok) {
            report("Error: GPU merge failed (memory?)", 1.f);
            join_encode();
            metal_merge_set_single_acc_slot(false);
            writer.close();
            if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
            return Image();
        }

        // Previous band is resident on host once ensure_acc waited (1×) or
        // ping-pong resolved (2×). Encode it while this band's GPU runs.
        if (have_ready) {
            const int er = ready, ey0 = ready_y0, ebh = ready_bh;
            if (debug && analyze_merge_band(num_bands[er], den_bands[er], ey0, merge_debug)) {
                const std::string ys = std::to_string(ey0);
                debug_dump_bin("cpp_num_bad_band_y" + ys,
                               num_bands[er].data.data(), num_bands[er].data.size());
                debug_dump_bin("cpp_den_bad_band_y" + ys,
                               den_bands[er].data.data(), den_bands[er].data.size());
            }
            std::vector<uint16_t>& out16 = row16_async[er];
            if (out16.size() < (size_t)ebh * (size_t)Ws * 3u)
                out16.resize((size_t)ebh * (size_t)Ws * 3u);
            encode_thr = std::thread([&, er, ey0, ebh]() {
                worker_qos();
                encode_band_rows(num_bands[er], den_bands[er], ey0, ebh, work, nch,
                                 preview, pscale, ph, pw, Ws, out16);
                writer.write_rows(out16.data(), ebh);
            });
            report("Merging output", 0.48f + 0.50f * (float)(ey0 + ebh) / Hs);
        }
        ready = cur;
        ready_y0 = y0;
        ready_bh = bh;
        have_ready = true;
        prof_add_cpu("merge:band(total)", prof_now_ms() - t_band);
        prof_mark_memory("merge:band");
    }
    join_encode();
    if (have_ready && metal_merge_wait_inflight()) {
        Image& rn = num_bands[ready];
        Image& rd = den_bands[ready];
        accumulate_diag(rn, rd, diag);
        if (debug && analyze_merge_band(rn, rd, ready_y0, merge_debug)) {
            const std::string ys = std::to_string(ready_y0);
            debug_dump_bin("cpp_num_bad_band_y" + ys, rn.data.data(), rn.data.size());
            debug_dump_bin("cpp_den_bad_band_y" + ys, rd.data.data(), rd.data.size());
        }
        if (row16.size() < (size_t)ready_bh * (size_t)Ws * 3u)
            row16.resize((size_t)ready_bh * (size_t)Ws * 3u);
        encode_band_rows(rn, rd, ready_y0, ready_bh, work, nch, preview, pscale, ph, pw, Ws, row16);
        writer.write_rows(row16.data(), ready_bh);
        report("Merging output", 0.48f + 0.50f * (float)(ready_y0 + ready_bh) / Hs);
    }
    metal_merge_set_single_acc_slot(false);
    }   // else (banded)
#else
    Image num_band, den_band;
    report("Merging output", 0.48f);
    for (int y0 = 0; y0 < Hs; y0 += band_rows) {
        const int bh = std::min(band_rows, Hs - y0);
        if (num_band.h != bh || num_band.w != Ws || num_band.c != nch) {
            num_band = Image(bh, Ws, nch);
            den_band = Image(bh, Ws, nch);
        } else {
            std::fill(num_band.data.begin(), num_band.data.end(), 0.f);
            std::fill(den_band.data.begin(), den_band.data.end(), 0.f);
        }

        if (stream_comp_raw) {
            for (const CachedCompMeta& meta : cached_meta) {
                if (!meta.rob_has_nonzero ||
                    !robustness_band_can_contribute(meta.rob_rows_nonzero, y0, bh,
                                                    work.scale, work.bayer_mode))
                    continue;
                if (!load_streamed_comp_raw(meta.index, comp_scratch)) continue;
                merge_comp_band(comp_scratch, meta.flow, meta.covs, meta.rob, tile_size,
                                num_band, den_band, y0, work, meta.index);
            }
        } else {
            for (const CachedCompFrame& fc : cached) {
                if (!fc.rob_has_nonzero ||
                    !robustness_band_can_contribute(fc.rob_rows_nonzero, y0, bh,
                                                    work.scale, work.bayer_mode))
                    continue;
                merge_comp_band(fc.comp, fc.flow, fc.covs, fc.rob, tile_size,
                                num_band, den_band, y0, work, fc.index);
            }
        }

        merge_ref_band(ref, ref_covs, num_band, den_band, y0, work, acc_rob_ptr);
        if (y0 + bh >= Hs)
            accumulate_diag(num_band, den_band, diag);
        if (debug && analyze_merge_band(num_band, den_band, y0, merge_debug)) {
            const std::string ys = std::to_string(y0);
            debug_dump_bin("cpp_num_bad_band_y" + ys,
                           num_band.data.data(), num_band.data.size());
            debug_dump_bin("cpp_den_bad_band_y" + ys,
                           den_band.data.data(), den_band.data.size());
        }

        encode_band_rows(num_band, den_band, y0, bh, work, nch, preview, pscale, ph, pw, Ws, row16);
        writer.write_rows(row16.data(), bh);
        report("Merging output", 0.48f + 0.50f * (float)(y0 + bh) / Hs);
    }
#endif

    const double t_close = prof_now_ms();
    writer.close();
    prof_add_cpu("out:dng-close(flush)", prof_now_ms() - t_close);
    const double t_tail = prof_now_ms();
    if (work.robustness_save_mask && have_acc_rob) {
        if (write_robustness_mask_pgm(acc_rob, n - 1, dng_path))
            report("Wrote robustness mask", 0.985f);
    }
    cached.clear();
    cached_meta.clear();
    ref = Image();
    ref_covs = CovField();
    if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
    prof_add_cpu("out:mask-pgm+teardown", prof_now_ms() - t_tail);
    report(format_accum_diag(diag), 0.99f);
    if (debug) {
        append_merge_summary(debug_summary, merge_debug);
        debug_dump_text("cpp_debug_summary", debug_summary.str());
    }
    // Rebuild the FFT plan now that the burst is done and memory is at its
    // lowest, so the next shot finds it warm. Detached and by-value so it can
    // outlive this frame; std::async would block in its future destructor and
    // defeat the point.
    if (ref_h > 0 && ref_w > 0) {
        const int pw_h = ref_h, pw_w = ref_w;
        std::thread([pw_h, pw_w]() {
            worker_qos();
            mps_fft_prewarm(pw_h, pw_w);
        }).detach();
    }

    prof_mark_memory("burst:end");
    if (prof_enabled()) {
        // Wall time reported separately: the buckets above overlap (async
        // decode/encode threads) and must not be summed against it.
        const double wall = prof_now_ms() - t_burst;
        char hdr[192];
        std::snprintf(hdr, sizeof(hdr),
                      "\nburst wall %.1f ms over %d frames (%.1f ms/frame)\n",
                      wall, n, wall / std::max(1, n));
        std::string prof = prof_report() + hdr;
        if (!cfg.debug_string_capture.empty()) {
            prof += "\n=== Metadata ===\n" + cfg.debug_string_capture + "\n";
        }
        std::printf("%s", prof.c_str());
        std::fflush(stdout);
        prof_save_report(prof);
        // Also surface on-device: a sideloaded build has no attached console.
        char one_line[192];
        std::snprintf(one_line, sizeof(one_line),
                      "%.1fs total, %.0f ms/frame - peak %.0f MB, headroom %.0f MB",
                      wall / 1000.0, wall / std::max(1, n),
                      (double)prof_peak_footprint_bytes() / (1024.0 * 1024.0),
                      (double)prof_min_available_bytes() / (1024.0 * 1024.0));
        report(one_line, 0.995f);
    }
    report("Done", 1.f);
    return preview;
}

Image process_burst_paths_to_dng(const std::vector<std::string>& paths, const Config& cfg,
                                 const std::string& dng_path, const ProgressFn& progress,
                                 int maxPreviewDim, Rgb16Sink* rgb16_sink) {
    return process_burst_loader_to_dng(
        (int)paths.size(),
        [&](int index, Config& work, bool is_reference, int crop_h, int crop_w) {
            if (index < 0 || index >= (int)paths.size()) return Image();
            return load_raw_frame(paths[(size_t)index], work, is_reference, crop_h, crop_w);
        },
        cfg, dng_path, progress, maxPreviewDim, rgb16_sink);
}

} // namespace hhsr
