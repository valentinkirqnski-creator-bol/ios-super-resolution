#include "pipeline.h"
#include "stages.h"
#include "dng_writer.h"
#include "snr_tuning.h"
#include <vector>

#ifdef HHSR_USE_VULKAN
#include "gpu/gpu_merge.h"
#endif

namespace hhsr {

Image pad_image_circular(const Image& img, int tile_size) {
    int pad_h = (tile_size - img.h % tile_size) % tile_size;
    int pad_w = (tile_size - img.w % tile_size) % tile_size;
    if (pad_h == 0 && pad_w == 0) return img;
    Image padded(img.h + pad_h, img.w + pad_w, img.c);
    for (int y = 0; y < padded.h; ++y) {
        int src_y = y < img.h ? y : (y - img.h);
        for (int x = 0; x < padded.w; ++x) {
            int src_x = x < img.w ? x : (x - img.w);
            for (int ch = 0; ch < img.c; ++ch) {
                padded.at(y, x, ch) = img.at(src_y, src_x, ch);
            }
        }
    }
    return padded;
}

Image process_burst(const std::vector<Image>& burst, const Config& cfg,
                    const ProgressFn& progress) {
    if (burst.empty()) return Image();
    Config work = cfg;
    tune_config_snr(burst[0], work);
    const Image& ref = burst[0];
    int n = (int)burst.size();

    int tile_size = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];

    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };

    report("Reference: grey + pyramid", 0.02f);
    // Python init_alignment: circular-pad REF only, then pyramid. Moving stays unpadded.
    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    ref_grey = pad_image_circular(ref_grey, tile_size);
    Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);

    report("Reference: local stats", 0.05f);
    RefStats ref_stats = init_robustness(ref, work);
    CovField ref_covs = estimate_kernels(ref, work);

    const bool accumulate_r =
        work.accumulated_robustness_denoiser_enabled || work.robustness_save_mask;
    Image acc_rob;
    bool have_acc_rob = false;

    int Hs = (int)std::lround(work.scale * ref.h);
    int Ws = (int)std::lround(work.scale * ref.w);
    int nch = work.bayer_mode ? 3 : 1;
    Image num(Hs, Ws, nch), den(Hs, Ws, nch);

    for (int k = 1; k < n; ++k) {
        float base = 0.05f + 0.85f * (float)(k - 1) / std::max(1, n - 1);
        report("Frame " + std::to_string(k + 1) + ": align", base);
        const Image& comp = burst[k];
        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);

        FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, tile_size);
        Image rob = compute_robustness(comp, ref_stats, flow, tile_size, work);
        if (accumulate_r) {
            if (!have_acc_rob) {
                acc_rob = Image(rob.h, rob.w, 1);
                acc_rob.data = rob.data;
                have_acc_rob = true;
            } else {
                for (size_t i = 0; i < rob.data.size(); ++i)
                    acc_rob.data[i] += rob.data[i];
            }
        }
        CovField covs = estimate_kernels(comp, work);
        merge_comp(comp, flow, covs, rob, tile_size, num, den, work);
    }

    report("Reference: merge", 0.92f);
    merge_ref(ref, ref_covs, num, den, work, have_acc_rob ? &acc_rob : nullptr);

    report("Normalizing", 0.96f);
    Image out(Hs, Ws, nch);
    for (int y = 0; y < Hs; ++y) {
        for (int x = 0; x < Ws; ++x) {
            for (int ch = 0; ch < nch; ++ch) {
                size_t i = ((size_t)y * Ws + x) * nch + ch;
                // Python utils.divide: bare num/den (no den==0 guard, no WB bake)
                out.data[i] = num.data[i] / den.data[i];
            }
        }
    }
    return out;
}

Image process_burst_to_dng(const std::vector<Image>& burst, const Config& cfg,
                           const std::string& dng_path, const ProgressFn& progress,
                           int maxPreviewDim) {
    if (burst.empty()) return Image();
    Config work = cfg;
    tune_config_snr(burst[0], work);
    const Image& ref = burst[0];
    int n = (int)burst.size();
    int tile_size = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    int nch = work.bayer_mode ? 3 : 1;
    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };

    report("Reference: grey + pyramid", 0.02f);
    // Python init_alignment: circular-pad REF only, then pyramid. Moving stays unpadded.
    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    ref_grey = pad_image_circular(ref_grey, tile_size);
    Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
    RefStats ref_stats = init_robustness(ref, work);
    CovField ref_covs = estimate_kernels(ref, work);

    const bool accumulate_r =
        work.accumulated_robustness_denoiser_enabled || work.robustness_save_mask;

    struct FrameData { FlowField flow; Image robustness; CovField covs; };
    std::vector<FrameData> frames(n - 1);
    Image acc_rob;
    bool have_acc_rob = false;

    for (int k = 1; k < n; ++k) {
        report("Frame " + std::to_string(k + 1) + ": analyze",
               0.03f + 0.50f * (float)(k - 1) / std::max(1, n - 1));
        Image comp_grey = compute_grey(burst[k], work.bayer_mode, work.grey_method);
        FrameData& fd = frames[k - 1];
        fd.flow = align(ref_pyr, ref_grey, comp_grey, work, tile_size);
        fd.robustness = compute_robustness(burst[k], ref_stats, fd.flow, tile_size, work);
        fd.covs = estimate_kernels(burst[k], work);
        if (accumulate_r) {
            if (!have_acc_rob) {
                acc_rob = Image(fd.robustness.h, fd.robustness.w, 1);
                acc_rob.data = fd.robustness.data;
                have_acc_rob = true;
            } else {
                for (size_t i = 0; i < fd.robustness.data.size(); ++i)
                    acc_rob.data[i] += fd.robustness.data[i];
            }
        }
    }

    const Image* acc_rob_ptr = have_acc_rob ? &acc_rob : nullptr;

    int Hs = (int)std::lround(work.scale * ref.h);
    int Ws = (int)std::lround(work.scale * ref.w);

    DngStreamWriter writer;
    if (!writer.open(dng_path, Ws, Hs, "HandheldSR-x2", work.orientation,
                     work.has_color_matrix ? work.color_matrix : nullptr,
                     work.bayer_mode ? work.white_balance : nullptr,
                     work.bake_srgb)) {
        report("Error: cannot open output DNG", 1.0f);
        return Image();
    }

    float pscale = std::min(1.f, (float)maxPreviewDim / std::max(Hs, Ws));
    int ph = std::max(1, (int)(Hs * pscale));
    int pw = std::max(1, (int)(Ws * pscale));
    Image preview(ph, pw, 3);

    size_t bytes_per_row = (size_t)Ws * nch * 4 * 2;
    int band_rows = (int)std::max<size_t>(8, (48u * 1024u * 1024u) / std::max<size_t>(1, bytes_per_row));
    band_rows = std::min(band_rows, Hs);
    std::vector<uint16_t> row16((size_t)band_rows * Ws * 3);

    for (int y0 = 0; y0 < Hs; y0 += band_rows) {
        int bh = std::min(band_rows, Hs - y0);
        Image num_band(bh, Ws, nch), den_band(bh, Ws, nch);

        for (int k = 1; k < n; ++k) {
            const FrameData& fd = frames[k - 1];
            merge_comp_band(burst[k], fd.flow, fd.covs, fd.robustness, tile_size,
                            num_band, den_band, y0, work);
        }
        merge_ref_band(ref, ref_covs, num_band, den_band, y0, work, acc_rob_ptr);

        auto to_srgb = [](f32 v) {
            v = clampf(v, 0.f, 1.f);
            return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
        };
        for (int i = 0; i < bh; ++i) {
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
                    if (k == 0) preview.at(py, std::min(pw - 1, (int)(x * pscale)), 0) = v;
                    else if (k == 1) preview.at(py, std::min(pw - 1, (int)(x * pscale)), 1) = v;
                    else preview.at(py, std::min(pw - 1, (int)(x * pscale)), 2) = v;
                }
            }
        }
        writer.write_rows(row16.data(), bh);
        report("Merging output", 0.55f + 0.42f * (float)(y0 + bh) / Hs);
    }

    writer.close();
    report("Done", 1.0f);
    return preview;
}

} // namespace hhsr
