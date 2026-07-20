// Burst processing for memory-constrained mobile targets (iOS).
// Full-res (1×) and large bursts: spill comparison Bayer to disk after analyze.
// Lighter 2× crops keep RAW+analysis in RAM. Apple: GPU merge prefetch after analyze.
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
#include <future>
#include <memory>
#if defined(__APPLE__)
#include <thread>
#endif

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
    // Same num/den → RGB16 math as before; pointer loops + sparse preview only.
    auto to_srgb = [](f32 v) {
        v = clampf(v, 0.f, 1.f);
        return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
    };
    const int x_step = std::max(1, (int)std::ceil(1.f / std::max(pscale, 1e-6f)));
    // Preview is UI-only; sample a bit more sparsely (DNG pixels unchanged).
    const int y_step = std::max(1, x_step);
    const f32* nump = num_band.data.data();
    const f32* denp = den_band.data.data();
    const bool bake = work.bake_srgb && nch >= 3;
    const f32* m = work.cam_to_srgb;
    const f32 wb0 = work.white_balance[0], wb1 = work.white_balance[1], wb2 = work.white_balance[2];
    const bool prev_color = !bake && nch >= 3 && work.has_cam_to_srgb;

#if defined(__APPLE__)
    // Dense DNG band on GPU (1:1); sparse preview stays on CPU below.
    const bool gpu_rgb = metal_normalize_band_rgb16(num_band, den_band, work, row16);
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
        const int py = std::min(ph - 1, (int)(gy * pscale));
        const size_t row_off = (size_t)i * (size_t)Ws * (size_t)nch;
        for (int x = 0; x < Ws; ++x) {
            const bool need_prev = do_prev_row && (x % x_step) == 0;
            if (gpu_rgb && !need_prev) continue;

            const size_t pi = row_off + (size_t)x * (size_t)nch;
            f32 d0 = denp[pi];
            f32 cn0 = (d0 > 0.f) ? nump[pi] / d0 : 0.f;
            f32 cn1 = 0.f, cn2 = 0.f;
            if (nch >= 2) {
                f32 d1 = denp[pi + 1];
                cn1 = (d1 > 0.f) ? nump[pi + 1] / d1 : 0.f;
            }
            if (nch >= 3) {
                f32 d2 = denp[pi + 2];
                cn2 = (d2 > 0.f) ? nump[pi + 2] / d2 : 0.f;
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
                const f32 v0 = bake ? to_srgb(lin0) : clampf(lin0, 0.f, 1.f);
                const f32 v1 = bake ? to_srgb(lin1) : clampf(lin1, 0.f, 1.f);
                const f32 v2 = bake ? to_srgb(lin2) : clampf(lin2, 0.f, 1.f);
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
    ref_grey = Image(); // align uses pyramid only

    // Full-res (~12MP Bayer) cannot hold every comparison RAW + dual Metal peaks.
    const bool full_res =
        ((size_t)ref_h * (size_t)ref_w) >= 8ull * 1000ull * 1000ull;

    // Same math; sequential on Apple Metal (rob + kernels both GPU — overlap
    // doubles peak and races shared scratch). Light crops may still overlap.
    RefStats ref_stats;
    CovField ref_covs;
    if (full_res) {
        ref_stats = init_robustness(ref, work);
        ref_covs = estimate_kernels(ref, work);
    } else {
        std::future<CovField> ref_cov_fut =
            std::async(std::launch::async, [&]() { return estimate_kernels(ref, work); });
        ref_stats = init_robustness(ref, work);
        ref_covs = ref_cov_fut.get();
    }
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

    // Spill Bayer on full-res (any burst size) or when comps > 8. Keeps flow/R/cov.
    std::vector<CachedCompFrame> cached;
    std::vector<CachedCompMeta> cached_meta;
    const bool stream_comp_raw = full_res || (n - 1) > 8;
    if (stream_comp_raw)
        fs::create_directories(cache, ec);

    cached.reserve((size_t)std::max(0, n - 1));
    cached_meta.reserve((size_t)std::max(0, n - 1));

    // Prefetch next LibRaw decode: 2× during whole analyze; 1× only after grey
    // is freed (overlaps rob/kernels, +1 Bayer peak briefly).
    int pref_k = -1;
    std::future<Image> pref_fut;
    std::future<bool> spill_fut;
    bool spill_pending = false;
    int n_comp_ok = 0;
    auto drain_spill = [&]() {
        if (!spill_fut.valid()) return;
        const bool ok = spill_fut.get();
        if (!ok && spill_pending && !cached_meta.empty()) {
            cached_meta.pop_back();
            n_comp_ok = std::max(0, n_comp_ok - 1);
        }
        spill_pending = false;
    };
    for (int k = 1; k < n; ++k) {
        report("Frame " + std::to_string(k + 1) + ": analyze",
               0.08f + 0.35f * (float)(k - 1) / std::max(1, n - 1));
        drain_spill();

        Image comp;
        if (pref_k == k && pref_fut.valid()) {
            comp = pref_fut.get();
            pref_k = -1;
        } else {
            comp = load_raw_frame(paths[k], work, false, ref_h, ref_w);
        }
        if (comp.h <= 0) continue;

        // 2×: decode next during align. 1×: wait until after grey (lower peak).
        if (!full_res && k + 1 < n) {
            const int nk = k + 1;
            pref_k = nk;
            pref_fut = std::async(std::launch::async, [&, nk]() {
                return load_raw_frame(paths[nk], work, false, ref_h, ref_w);
            });
        }

        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
        FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, tile_size);
        comp_grey = Image(); // free before robustness/kernels peak

        // Full-res: decode next while Metal rob/kernels run (grey already freed).
        if (full_res && k + 1 < n) {
            const int nk = k + 1;
            pref_k = nk;
            pref_fut = std::async(std::launch::async, [&, nk]() {
                return load_raw_frame(paths[nk], work, false, ref_h, ref_w);
            });
        }

        Image rob;
        CovField covs;
        if (full_res) {
            // Keep rob ∥ kernels serialized — dual Metal peaks jetsam on 1×.
            rob = compute_robustness(comp, ref_stats, flow, tile_size, work);
            covs = estimate_kernels(comp, work);
        } else {
            // Same math; overlap only when peak RAM is affordable (2× crop).
            std::future<CovField> cov_fut =
                std::async(std::launch::async, [&]() { return estimate_kernels(comp, work); });
            rob = compute_robustness(comp, ref_stats, flow, tile_size, work);
            covs = cov_fut.get();
        }

        if (stream_comp_raw) {
            // Spill Bayer async (overlaps next decode already in flight). Keep
            // flow/R/cov in RAM. Grow-only L2/Alg.5 scratch stays until merge.
            const int sk = k;
            auto spill_img = std::make_shared<Image>(std::move(comp));
            const fs::path spill_path = cache / ("f" + std::to_string(sk) + ".raw");
            spill_fut = std::async(std::launch::async, [spill_path, spill_img]() {
                return save_image(spill_path, *spill_img);
            });
            spill_pending = true;
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
    drain_spill();
    if (pref_fut.valid()) (void)pref_fut.get(); // drain unused prefetch

    if (n_comp_ok < 1) {
        if (stream_comp_raw) fs::remove_all(cache, ec);
        report("Error: could not analyze comparison frames", 1.f);
        return Image();
    }

    // Release reference-side helpers not needed during merge.
    clear_align_ref_ica_cache();
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

    // Band size: 2× double-buffers GPU+host. Full 1× keeps one GPU acc slot
    // (jetsam-safe) but dual host bands so Deflate can overlap the next GPU band.
    const size_t out_px = (size_t)Hs * (size_t)Ws;
    const bool heavy_1x = out_px >= 28ull * 1000ull * 1000ull; // ~full-res scale-2
#if defined(__APPLE__)
    const size_t band_budget = heavy_1x ? (192u * 1024u * 1024u) : (384u * 1024u * 1024u);
#else
    const size_t band_budget = 64u * 1024u * 1024u;
#endif
    const size_t bytes_per_row = (size_t)Ws * nch * 4 * 2; // num+den float row
    int band_rows = (int)std::max<size_t>(4, band_budget / std::max<size_t>(1, bytes_per_row));
    band_rows = std::min(band_rows, Hs);
    // ~960 rows fits the 192MB single-slot budget at full-res scale-2.
    if (heavy_1x) band_rows = std::min(band_rows, 960);
    std::vector<uint16_t> row16((size_t)band_rows * (size_t)Ws * 3u);

    Image comp_scratch;
#if defined(__APPLE__)
    // Upload comparison frames to GPU before the band loop so merge isn't
    // stalled on the first band's PCIe copies. On full-res, drop host R/cov
    // after each upload so we never hold host+GPU copies of every frame.
    report("Preparing GPU merge", 0.46f);
    metal_merge_set_single_acc_slot(heavy_1x);
    metal_merge_begin_burst();
    if (stream_comp_raw) {
        for (CachedCompMeta& meta : cached_meta) {
            if (!load_cached_comp_raw(cache, meta.index, comp_scratch)) continue;
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
#endif

    AccumDiag diag;
#if defined(__APPLE__)
    // Dual host bands + async Deflate for both 1× and 2×. 1× still uses a
    // single GPU acc slot (ensure_acc waits/readbacks before reuse).
    Image num_bands[2], den_bands[2];
    std::vector<uint16_t> row16_async[2];
    row16_async[0].resize(row16.size());
    row16_async[1].resize(row16.size());
    int cur = 0;
    bool have_ready = false;
    int ready = 0, ready_y0 = 0, ready_bh = 0;
    std::thread encode_thr;
    auto join_encode = [&]() {
        if (encode_thr.joinable()) encode_thr.join();
    };
    report("Merging output", 0.48f);
    for (int y0 = 0; y0 < Hs; y0 += band_rows) {
        const int bh = std::min(band_rows, Hs - y0);
        cur ^= 1;
        join_encode();
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
                if (stream_comp_raw) fs::remove_all(cache, ec);
                return Image();
            }
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

        if (!merge_ref_band_metal(ref, ref_covs, num_band, den_band, y0, work, acc_rob_ptr)) {
            report("Error: GPU merge failed (memory?)", 1.f);
            join_encode();
            metal_merge_set_single_acc_slot(false);
            writer.close();
            if (stream_comp_raw) fs::remove_all(cache, ec);
            return Image();
        }

        // Previous band is resident on host once ensure_acc waited (1×) or
        // ping-pong resolved (2×). Encode it while this band's GPU runs.
        if (have_ready) {
            const int er = ready, ey0 = ready_y0, ebh = ready_bh;
            std::vector<uint16_t>& out16 = row16_async[er];
            if (out16.size() < (size_t)ebh * (size_t)Ws * 3u)
                out16.resize((size_t)ebh * (size_t)Ws * 3u);
            encode_thr = std::thread([&, er, ey0, ebh]() {
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
    }
    join_encode();
    if (have_ready && metal_merge_wait_inflight()) {
        Image& rn = num_bands[ready];
        Image& rd = den_bands[ready];
        accumulate_diag(rn, rd, diag);
        if (row16.size() < (size_t)ready_bh * (size_t)Ws * 3u)
            row16.resize((size_t)ready_bh * (size_t)Ws * 3u);
        encode_band_rows(rn, rd, ready_y0, ready_bh, work, nch, preview, pscale, ph, pw, Ws, row16);
        writer.write_rows(row16.data(), ready_bh);
        report("Merging output", 0.48f + 0.50f * (float)(ready_y0 + ready_bh) / Hs);
    }
    metal_merge_set_single_acc_slot(false);
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
