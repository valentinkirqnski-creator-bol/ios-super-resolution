// Disk-backed burst processing for memory-constrained mobile targets (iOS).
// Peak RAM ≈ 1 reference frame + 1 comparison frame + 1 frame's analysis + band buffers.
#include "pipeline.h"
#include "stages.h"
#include "dng_writer.h"
#include "snr_tuning.h"
#include "raw_io.h"
#include "parallel.h"
#include <vector>
#include <array>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cmath>

namespace fs = std::filesystem;

namespace hhsr {

namespace {

static bool save_flow(const fs::path& p, const FlowField& f) {
    int32_t hdr[2] = {f.ny, f.nx};
    std::ofstream out(p, std::ios::binary);
    if (!out) return false;
    out.write((const char*)hdr, sizeof(hdr));
    out.write((const char*)f.flow.data(), (std::streamsize)(f.flow.size() * sizeof(f32)));
    return out.good();
}

static bool load_flow(const fs::path& p, FlowField& f) {
    int32_t hdr[2];
    std::ifstream in(p, std::ios::binary);
    if (!in || !in.read((char*)hdr, sizeof(hdr))) return false;
    f = FlowField(hdr[0], hdr[1]);
    return (bool)in.read((char*)f.flow.data(), (std::streamsize)(f.flow.size() * sizeof(f32)));
}

static bool save_covs(const fs::path& p, const CovField& c) {
    int32_t hdr[2] = {c.h, c.w};
    std::vector<f32> payload;
    payload.reserve(2 + c.cov.size());
    // write header then data in one file
    std::ofstream out(p, std::ios::binary);
    if (!out) return false;
    out.write((const char*)hdr, sizeof(hdr));
    out.write((const char*)c.cov.data(), (std::streamsize)(c.cov.size() * sizeof(f32)));
    return out.good();
}

static bool load_covs(const fs::path& p, CovField& c) {
    int32_t hdr[2];
    std::ifstream in(p, std::ios::binary);
    if (!in || !in.read((char*)hdr, sizeof(hdr))) return false;
    c = CovField(hdr[0], hdr[1]);
    return (bool)in.read((char*)c.cov.data(), (std::streamsize)(c.cov.size() * sizeof(f32)));
}

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
    im = Image(hdr[0], hdr[1], hdr[2]);
    return (bool)in.read((char*)im.data.data(), (std::streamsize)(im.data.size() * sizeof(f32)));
}

struct CachedCompFrame {
    FlowField flow;
    Image rob;
    CovField covs;
    Image comp;
};

struct CachedCompMeta {
    FlowField flow;
    Image rob;
    CovField covs;
    int index = 0;
};

static bool load_cached_comp_meta(const fs::path& cache, int k, CachedCompMeta& out) {
    std::string idx = std::to_string(k);
    fs::path fflow = cache / ("f" + idx + ".flow");
    if (!fs::exists(fflow)) return false;
    out.index = k;
    return load_flow(fflow, out.flow) &&
           load_image(cache / ("f" + idx + ".rob"), out.rob) &&
           load_covs(cache / ("f" + idx + ".cov"), out.covs);
}

static bool load_cached_comp_raw(const fs::path& cache, int k, Image& comp) {
    return load_image(cache / ("f" + std::to_string(k) + ".raw"), comp) && comp.h > 0;
}

static bool load_cached_comp(const fs::path& cache, int k, CachedCompFrame& out) {
    std::string idx = std::to_string(k);
    fs::path fflow = cache / ("f" + idx + ".flow");
    if (!fs::exists(fflow)) return false;
    return load_flow(fflow, out.flow) &&
           load_image(cache / ("f" + idx + ".rob"), out.rob) &&
           load_covs(cache / ("f" + idx + ".cov"), out.covs) &&
           load_image(cache / ("f" + idx + ".raw"), out.comp) &&
           out.comp.h > 0;
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
                // Python utils.divide: bare num/den
                cn[ch] = num_band.at(i, x, ch) / den_band.at(i, x, ch);
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

    // Reference Bayer not needed during comparison analysis — reload for merge pass.
    ref = Image();

    fs::path cache = fs::path(dng_path).parent_path() /
                     (fs::path(dng_path).stem().string() + "_cache");
    std::error_code ec;
    fs::remove_all(cache, ec);
    fs::create_directories(cache, ec);

    // Pass 1 — analyze one comparison frame at a time; persist analysis to disk.
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

        std::string idx = std::to_string(k);
        if (!save_flow(cache / ("f" + idx + ".flow"), flow) ||
            !save_image(cache / ("f" + idx + ".rob"), rob) ||
            !save_covs(cache / ("f" + idx + ".cov"), covs) ||
            !save_image(cache / ("f" + idx + ".raw"), comp)) {
            comp = Image();
            continue;
        }
        n_comp_ok++;
        comp = Image();
        comp_grey = Image();
        flow = FlowField();
        rob = Image();
        covs = CovField();
    }

    if (n_comp_ok < 1) {
        fs::remove_all(cache, ec);
        report("Error: could not analyze comparison frames", 1.f);
        return Image();
    }

    // Release reference-side helpers not needed during merge.
    ref_grey = Image();
    ref_pyr = Pyramid();
    ref_stats = RefStats();

    report("Loading frames for merge", 0.44f);
    ref = load_raw_frame(paths[0], work, false, ref_h, ref_w);
    if (ref.h <= 0) {
        fs::remove_all(cache, ec);
        report("Error: could not reload reference frame", 1.f);
        return Image();
    }

    std::vector<CachedCompFrame> cached;
    std::vector<CachedCompMeta> cached_meta;
    const bool stream_comp_raw = (n - 1) > 4; // 6+ frames: stream; ≤5 uses fast preload

    if (stream_comp_raw) {
        cached_meta.reserve(n - 1);
        for (int k = 1; k < n; ++k) {
            CachedCompMeta meta;
            if (load_cached_comp_meta(cache, k, meta))
                cached_meta.push_back(std::move(meta));
        }
        if (cached_meta.empty()) {
            fs::remove_all(cache, ec);
            report("Error: could not load cached frames", 1.f);
            return Image();
        }
    } else {
        cached.reserve(n - 1);
        for (int k = 1; k < n; ++k) {
            CachedCompFrame fc;
            if (load_cached_comp(cache, k, fc))
                cached.push_back(std::move(fc));
        }
        if (cached.empty()) {
            fs::remove_all(cache, ec);
            report("Error: could not load cached frames", 1.f);
            return Image();
        }
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
    if (!writer.open(dng_path, Ws, Hs, model, work.orientation, nullptr, nullptr,
                     work.bake_srgb, make)) {
        fs::remove_all(cache, ec);
        report("Error: cannot open output DNG", 1.f);
        return Image();
    }

    const float pscale = std::min(1.f, (float)maxPreviewDim / (float)std::max(Hs, Ws));
    const int ph = std::max(1, (int)(Hs * pscale));
    const int pw = std::max(1, (int)(Ws * pscale));
    Image preview(ph, pw, 3);

    // ~64 MB band budget.
    const size_t band_budget = 64u * 1024u * 1024u;
    const size_t bytes_per_row = (size_t)Ws * nch * 4 * 2;
    int band_rows = (int)std::max<size_t>(4, band_budget / std::max<size_t>(1, bytes_per_row));
    band_rows = std::min(band_rows, Hs);
    std::vector<uint16_t> row16((size_t)band_rows * Ws * 3);

    for (int y0 = 0; y0 < Hs; y0 += band_rows) {
        const int bh = std::min(band_rows, Hs - y0);
        Image num_band(bh, Ws, nch), den_band(bh, Ws, nch);

        if (stream_comp_raw) {
            for (const CachedCompMeta& meta : cached_meta) {
                Image comp;
                if (!load_cached_comp_raw(cache, meta.index, comp)) continue;
                merge_comp_band(comp, meta.flow, meta.covs, meta.rob, tile_size,
                                num_band, den_band, y0, work);
                comp = Image();
            }
        } else {
            for (const CachedCompFrame& fc : cached)
                merge_comp_band(fc.comp, fc.flow, fc.covs, fc.rob, tile_size,
                                num_band, den_band, y0, work);
        }

        merge_ref_band(ref, ref_covs, num_band, den_band, y0, work, acc_rob_ptr);

        encode_band_rows(num_band, den_band, y0, bh, work, nch, preview, pscale, ph, pw, Ws, row16);
        writer.write_rows(row16.data(), bh);
        report("Merging output", 0.48f + 0.50f * (float)(y0 + bh) / Hs);
    }

    writer.close();
    cached.clear();
    cached_meta.clear();
    ref = Image();
    ref_covs = CovField();
    fs::remove_all(cache, ec);
    report("Done", 1.f);
    return preview;
}

} // namespace hhsr
