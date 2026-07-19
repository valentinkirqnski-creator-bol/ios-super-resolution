// Burst processing for memory-constrained mobile targets (iOS).
// ≤8 comps: keep RAW+analysis in RAM. Larger bursts: spill Bayer to disk only.
// On Apple, frames are prefetched to the GPU before the merge band loop.
#include "pipeline.h"
#include "stages.h"
#include "dng_writer.h"
#include "snr_tuning.h"
#include "raw_io.h"
#include "parallel.h"
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

namespace fs = std::filesystem;

namespace hhsr {

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
    int index = 0;
};

struct CachedCompMeta {
    FlowField flow;
    Image rob;
    CovField covs;
    int index = 0;
};

static bool load_cached_comp_raw(const fs::path& cache, int k, Image& comp) {
    return load_image(cache / ("f" + std::to_string(k) + ".raw"), comp) && comp.h > 0;
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

static void encode_band_rows(const Image& num_band, const Image& den_band, int y0, int bh,
                             const Config& work, int nch, Image& preview, float pscale,
                             int ph, int pw, int Ws, std::vector<uint16_t>& row16) {
    auto to_srgb = [](f32 v) {
        v = clampf(v, 0.f, 1.f);
        return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
    };
    const int x_step = std::max(1, (int)std::ceil(1.f / std::max(pscale, 1e-6f)));
    parallel_rows(bh, work.num_threads, [&](int i) {
        int gy = y0 + i;
        int py = std::min(ph - 1, (int)(gy * pscale));
        for (int x = 0; x < Ws; ++x) {
            size_t base = ((size_t)i * Ws + x) * 3;
            f32 cn[3] = {0, 0, 0};
            for (int ch = 0; ch < nch; ++ch) {
                f32 d = den_band.at(i, x, ch);
                cn[ch] = (d > 0.f) ? num_band.at(i, x, ch) / d : 0.f;
            }
            f32 outc[3];
            if (work.bake_srgb && nch >= 3) {
                f32 wr = cn[0] * work.white_balance[0];
                f32 wg = cn[1] * work.white_balance[1];
                f32 wb = cn[2] * work.white_balance[2];
                const f32* m = work.cam_to_srgb;
                outc[0] = m[0] * wr + m[1] * wg + m[2] * wb;
                outc[1] = m[3] * wr + m[4] * wg + m[5] * wb;
                outc[2] = m[6] * wr + m[7] * wg + m[8] * wb;
            } else if (nch >= 3) {
                outc[0] = cn[0]; outc[1] = cn[1]; outc[2] = cn[2];
            } else {
                outc[0] = outc[1] = outc[2] = cn[0];
            }
            for (int k = 0; k < 3; ++k) {
                f32 v = work.bake_srgb ? to_srgb(outc[k]) : clampf(outc[k], 0.f, 1.f);
                row16[base + k] = (uint16_t)(v * 65535.f + 0.5f);
            }
            if ((x % x_step) == 0) {
                f32 preview_lin[3] = {outc[0], outc[1], outc[2]};
                if (!work.bake_srgb && nch >= 3) {
                    f32 wr = cn[0] * work.white_balance[0];
                    f32 wg = cn[1] * work.white_balance[1];
                    f32 wb = cn[2] * work.white_balance[2];
                    if (work.has_cam_to_srgb) {
                        const f32* m = work.cam_to_srgb;
                        preview_lin[0] = m[0] * wr + m[1] * wg + m[2] * wb;
                        preview_lin[1] = m[3] * wr + m[4] * wg + m[5] * wb;
                        preview_lin[2] = m[6] * wr + m[7] * wg + m[8] * wb;
                    } else {
                        preview_lin[0] = wr; preview_lin[1] = wg; preview_lin[2] = wb;
                    }
                }
                int px = std::min(pw - 1, (int)(x * pscale));
                for (int k = 0; k < 3; ++k)
                    preview.at(py, px, k) = to_srgb(clampf(preview_lin[k], 0.f, 1.f));
            }
        }
    });
}

} // namespace

Image process_burst_paths_to_dng(const std::vector<std::string>& paths, const Config& cfg,
                                 const std::string& dng_path, const ProgressFn& progress,
                                 int maxPreviewDim) {
    if (paths.size() < 2) return Image();

    Config work = cfg;
    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };

    report("Loading reference frame", 0.02f);
    Image ref = load_raw_frame(paths[0], work, true);
    if (ref.h <= 0 || ref.w <= 0) return Image();
    tune_config_snr(ref, work);

    const int ref_h = ref.h, ref_w = ref.w;
    const int n = (int)paths.size();
    const int tile_size = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    const int nch = work.bayer_mode ? 3 : 1;

    report("Reference: grey + pyramid", 0.05f);
    // Python init_alignment: circular-pad REF only, then pyramid. Moving stays unpadded.
    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    ref_grey = pad_image_circular(ref_grey, tile_size);
    Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
    RefStats ref_stats = init_robustness(ref, work);
    CovField ref_covs = estimate_kernels(ref, work);

    // Keep ref Bayer in RAM for merge (avoids a second LibRaw decode).
    // Peak during analyze ≈ ref + one comparison frame.

    fs::path cache = fs::path(dng_path).parent_path() /
                     (fs::path(dng_path).stem().string() + "_cache");
    std::error_code ec;
    fs::remove_all(cache, ec);

    // Keep analysis in RAM. Only spill RAW to disk when the burst is large.
    // (Old path wrote everything to disk then re-read it — that was "Loading frames".)
    std::vector<CachedCompFrame> cached;
    std::vector<CachedCompMeta> cached_meta;
    const bool stream_comp_raw = (n - 1) > 8; // ≤8 frames: full RAM path
    if (stream_comp_raw)
        fs::create_directories(cache, ec);

    cached.reserve((size_t)std::max(0, n - 1));
    cached_meta.reserve((size_t)std::max(0, n - 1));

    int n_comp_ok = 0;
    for (int k = 1; k < n; ++k) {
        report("Frame " + std::to_string(k + 1) + ": analyze",
               0.08f + 0.35f * (float)(k - 1) / std::max(1, n - 1));

        Image comp = load_raw_frame(paths[k], work, false, ref_h, ref_w);
        if (comp.h <= 0) continue;

        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
        FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, tile_size);
        Image rob = compute_robustness(comp, ref_stats, flow, tile_size, work);
        CovField covs = estimate_kernels(comp, work);
        comp_grey = Image();

        if (stream_comp_raw) {
            // Spill Bayer only; keep flow/R/cov in RAM (small vs RAW).
            if (!save_image(cache / ("f" + std::to_string(k) + ".raw"), comp)) {
                continue;
            }
            CachedCompMeta meta;
            meta.index = k;
            meta.flow = std::move(flow);
            meta.rob = std::move(rob);
            meta.covs = std::move(covs);
            cached_meta.push_back(std::move(meta));
            n_comp_ok++;
        } else {
            CachedCompFrame fc;
            fc.index = k;
            fc.flow = std::move(flow);
            fc.rob = std::move(rob);
            fc.covs = std::move(covs);
            fc.comp = std::move(comp);
            cached.push_back(std::move(fc));
            n_comp_ok++;
        }
    }

    if (n_comp_ok < 1) {
        if (stream_comp_raw) fs::remove_all(cache, ec);
        report("Error: could not analyze comparison frames", 1.f);
        return Image();
    }

    // Release reference-side helpers not needed during merge.
    ref_grey = Image();
    ref_pyr = Pyramid();
    ref_stats = RefStats();

    if (ref.h <= 0) {
        if (stream_comp_raw) fs::remove_all(cache, ec);
        report("Error: reference frame missing for merge", 1.f);
        return Image();
    }

    const int Hs = (int)std::lround(work.scale * ref.h);
    const int Ws = (int)std::lround(work.scale * ref.w);

    const bool accumulate_r =
        work.accumulated_robustness_denoiser_enabled || work.robustness_save_mask;
    Image acc_rob;
    bool have_acc_rob = false;
    build_robustness_sum(cached, cached_meta, stream_comp_raw, acc_rob, have_acc_rob);
    const Image* acc_rob_ptr = (accumulate_r && have_acc_rob) ? &acc_rob : nullptr;

    DngStreamWriter writer;
    const std::string& model = work.camera_model.empty() ? std::string("HandheldSR-x2") : work.camera_model;
    const std::string& make = work.camera_make.empty() ? std::string("HandheldSR") : work.camera_make;
    if (!writer.open(dng_path, Ws, Hs, model, work.orientation,
                     work.has_color_matrix ? work.color_matrix : nullptr,
                     work.white_balance,
                     work.bake_srgb, make,
                     work.has_cam_to_srgb ? work.cam_to_srgb : nullptr)) {
        if (stream_comp_raw) fs::remove_all(cache, ec);
        report("Error: cannot open output DNG", 1.f);
        return Image();
    }

    const float pscale = std::min(1.f, (float)maxPreviewDim / (float)std::max(Hs, Ws));
    const int ph = std::max(1, (int)(Hs * pscale));
    const int pw = std::max(1, (int)(Ws * pscale));
    Image preview(ph, pw, 3);

    // Larger bands on Apple → fewer GPU sync/readback round-trips.
    // 1× (~8k wide) needs more budget than 2× crop to stay at ~2–3 bands.
#if defined(__APPLE__)
    const size_t band_budget = 256u * 1024u * 1024u;
#else
    const size_t band_budget = 64u * 1024u * 1024u;
#endif
    const size_t bytes_per_row = (size_t)Ws * nch * 4 * 2;
    int band_rows = (int)std::max<size_t>(4, band_budget / std::max<size_t>(1, bytes_per_row));
    band_rows = std::min(band_rows, Hs);
    std::vector<uint16_t> row16((size_t)band_rows * Ws * 3);

    Image comp_scratch;
#if defined(__APPLE__)
    // Upload all comparison frames to GPU before the band loop so merge isn't
    // stalled on the first band's PCIe copies.
    report("Preparing GPU merge", 0.46f);
    metal_merge_begin_burst();
    if (stream_comp_raw) {
        for (const CachedCompMeta& meta : cached_meta) {
            if (!load_cached_comp_raw(cache, meta.index, comp_scratch)) continue;
            metal_merge_prefetch_frame(comp_scratch, meta.flow, meta.covs, meta.rob,
                                       meta.index);
        }
        comp_scratch = Image(); // free host RAW; GPU holds copies
    } else {
        for (CachedCompFrame& fc : cached) {
            if (metal_merge_prefetch_frame(fc.comp, fc.flow, fc.covs, fc.rob, fc.index))
                fc.comp = Image(); // host RAW freed; GPU holds the copy
        }
    }
#endif

    AccumDiag diag;
#if defined(__APPLE__)
    // Double-buffer host bands: while GPU runs band N, CPU encodes band N-1.
    Image num_bands[2], den_bands[2];
    int cur = 0;
    bool have_ready = false;
    int ready = 0, ready_y0 = 0, ready_bh = 0;
    report("Merging output", 0.48f);
    for (int y0 = 0; y0 < Hs; y0 += band_rows) {
        const int bh = std::min(band_rows, Hs - y0);
        cur ^= 1;
        Image& num_band = num_bands[cur];
        Image& den_band = den_bands[cur];
        if (num_band.h != bh || num_band.w != Ws || num_band.c != nch) {
            num_band = Image(bh, Ws, nch);
            den_band = Image(bh, Ws, nch);
        }

        if (stream_comp_raw) {
            for (const CachedCompMeta& meta : cached_meta) {
                if (metal_merge_has_frame(meta.index)) {
                    Image empty;
                    merge_comp_band(empty, meta.flow, meta.covs, meta.rob, tile_size,
                                    num_band, den_band, y0, work, meta.index);
                    continue;
                }
                if (!load_cached_comp_raw(cache, meta.index, comp_scratch)) continue;
                merge_comp_band(comp_scratch, meta.flow, meta.covs, meta.rob, tile_size,
                                num_band, den_band, y0, work, meta.index);
            }
        } else {
            for (const CachedCompFrame& fc : cached) {
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

        // Async commit; resolves previous in-flight band into its host images.
        if (!merge_ref_band_metal(ref, ref_covs, num_band, den_band, y0, work, acc_rob_ptr))
            continue;

        if (have_ready) {
            Image& rn = num_bands[ready];
            Image& rd = den_bands[ready];
            encode_band_rows(rn, rd, ready_y0, ready_bh, work, nch, preview, pscale, ph, pw, Ws, row16);
            writer.write_rows(row16.data(), ready_bh);
            report("Merging output", 0.48f + 0.50f * (float)(ready_y0 + ready_bh) / Hs);
        }
        ready = cur;
        ready_y0 = y0;
        ready_bh = bh;
        have_ready = true;
    }
    if (have_ready && metal_merge_wait_inflight()) {
        Image& rn = num_bands[ready];
        Image& rd = den_bands[ready];
        accumulate_diag(rn, rd, diag);
        encode_band_rows(rn, rd, ready_y0, ready_bh, work, nch, preview, pscale, ph, pw, Ws, row16);
        writer.write_rows(row16.data(), ready_bh);
        report("Merging output", 0.48f + 0.50f * (float)(ready_y0 + ready_bh) / Hs);
    }
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
                if (!load_cached_comp_raw(cache, meta.index, comp_scratch)) continue;
                merge_comp_band(comp_scratch, meta.flow, meta.covs, meta.rob, tile_size,
                                num_band, den_band, y0, work, meta.index);
            }
        } else {
            for (const CachedCompFrame& fc : cached)
                merge_comp_band(fc.comp, fc.flow, fc.covs, fc.rob, tile_size,
                                num_band, den_band, y0, work, fc.index);
        }

        merge_ref_band(ref, ref_covs, num_band, den_band, y0, work, acc_rob_ptr);
        if (y0 + bh >= Hs)
            accumulate_diag(num_band, den_band, diag);

        encode_band_rows(num_band, den_band, y0, bh, work, nch, preview, pscale, ph, pw, Ws, row16);
        writer.write_rows(row16.data(), bh);
        report("Merging output", 0.48f + 0.50f * (float)(y0 + bh) / Hs);
    }
#endif

    writer.close();
    if (work.robustness_save_mask && have_acc_rob) {
        if (write_robustness_mask_pgm(acc_rob, n - 1, dng_path))
            report("Wrote robustness mask", 0.985f);
    }
    cached.clear();
    cached_meta.clear();
    ref = Image();
    ref_covs = CovField();
    if (stream_comp_raw) fs::remove_all(cache, ec);
    report(format_accum_diag(diag), 0.99f);
    report("Done", 1.f);
    return preview;
}

} // namespace hhsr
