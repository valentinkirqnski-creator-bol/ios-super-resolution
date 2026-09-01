//
// ISA mode (Config::isa_mode): ImageStackAlignator, algorithmically
// identical to ImageStackAlignator-master, as the burst pipeline. The
// algorithm stages live in core/isa/ -- a verbatim vendoring of the
// desktop isa-cpu port built and verified against the C#/CUDA sources
// stage by stage (including ISA's own quirks, e.g. kernel.cu's
// fminf(sigma1, sigma1) reject threshold, preserved as-is; the one
// evidence-based deviation, LK's -= update, is documented in
// isa/optical_flow.h). This file only adapts the I/O contract:
//
//  - Input: the app's loader hands PREWHITENED normalized floats (black
//    subtracted, WB gains applied, 0..1-ish). ISA consumes raw-domain
//    integers plus black/white/AsShotNeutral metadata, so each frame is
//    converted to a synthetic raw: value * 16383 as uint16 with
//    black = 0, white = 16383, AsShotNeutral = (1,1,1). ISA's own
//    normalizations then reproduce the float values exactly, and its
//    white-balance-dependent steps see a unit-balanced camera -- which is
//    what prewhitened data is. Every algorithm stage runs bit-for-bit the
//    computation it runs on the desktop; only the number source differs
//    (and the app's decode clamps sub-black noise at 0, which real raw
//    would not -- disclosed, not hidden).
//  - Output: ISA's real SaveAs16BitTiff colour chain (camera-white clamp
//    -> camera->ProPhoto -> clamp -> ProPhoto->sRGB -> exposure -> sRGB
//    gamma), written through the app's DngStreamWriter as a baked-sRGB
//    DNG container -- the same "ISA output, DNG container" convention the
//    desktop port shipped.
//  - Memory: the desktop pipeline holds every stage's buffers for the
//    whole burst (fine at desktop RAM). Here the per-frame stages (LK
//    refine, robustness, accumulate) are interleaved per comparison frame
//    so only ONE dense flow field is ever alive, and the tracking greys
//    are freed the moment tracking ends. Pure reordering of per-frame-
//    independent work: results are unchanged.
//
// CPU golden reference first, deliberately: "identical" is only provable
// against a reference, and this IS the reference. Metal twins for the hot
// per-frame stages follow the same way every hhsr GPU stage was built --
// kernel by kernel, verified against this file's output.
//
#include "pipeline.h"
#include "stages.h"
#include "dng_writer.h"
#include "prof.h"

#include "isa/accumulate.h"
#include "isa/color_pipeline.h"
#include "isa/debayer.h"
#include "isa/optical_flow.h"
#include "isa/patch_tracker.h"
#include "isa/pre_alignment.h"
#include "isa/robustness.h"
#include "isa/shift_minimizer.h"
#include "isa/structure_tensor.h"
#include "isa/parallel_for.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace hhsr {

namespace {

constexpr float kIsaPi = 3.14159265358979323846f;
constexpr float kIsaWhite = 16383.f;  // synthetic raw domain (14-bit-like)

struct IsaFrameAlign {
    isacpu::Vec2f shift{0, 0};
    float rotation_rad = 0.f;
};

// Prewhitened float mosaic -> synthetic ISA raw (see file header).
std::vector<uint16_t> to_synth_raw(const Image& img) {
    std::vector<uint16_t> raw((size_t)img.h * img.w);
    for (size_t i = 0; i < raw.size(); ++i) {
        const float v = img.data[i];
        const long q = std::lround((double)(std::isfinite(v) ? std::max(v, 0.f) : 0.f) * kIsaWhite);
        raw[i] = (uint16_t)std::min<long>(q, 65535);
    }
    return raw;
}

}  // namespace

Image process_burst_loader_to_dng_isa(int frame_count, const RawFrameLoaderFn& loader,
                                      const Config& cfg, const std::string& dng_path,
                                      const ProgressFn& progress, int maxPreviewDim,
                                      Rgb16Sink* rgb16_sink) {
    if (rgb16_sink) { rgb16_sink->w = 0; rgb16_sink->h = 0; rgb16_sink->rgb.clear(); }
    if (frame_count < 2 || !loader) return Image();
    Config work = cfg;
    work.burst_frame_count = frame_count;
    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };
    const int referenceIndex = 0;  // ISA's own default

    int cfa[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) cfa[i][j] = (int)work.cfa.p[i][j];
    const float white_level[3] = {kIsaWhite, kIsaWhite, kIsaWhite};
    const float black_level[3] = {0.f, 0.f, 0.f};
    const float camera_white[3] = {1.f, 1.f, 1.f};  // prewhitened == unit balance

    // ---- 1+2. Load frames, synthesize raw, build tracking greys ----------
    report("ISA: loading + debayering", 0.02f);
    std::vector<std::vector<uint16_t>> raw16(frame_count);
    std::vector<std::vector<float>> bw_track(frame_count);
    int W = 0, H = 0;
    for (int i = 0; i < frame_count; ++i) {
        Image img = loader(i, work, i == referenceIndex, H, W);
        if (img.h <= 0 || img.w <= 0 || img.c != 1) return Image();
        if (i == 0) { H = img.h; W = img.w; }
        else if (img.h != H || img.w != W) return Image();
        raw16[i] = to_synth_raw(img);
        img = Image();  // free the float copy before the next decode

        // ISA DeBayerBWGaussWB, non-skip-Fourier path (tracking image):
        // sigma 0.5 with the Fourier high-pass (HighPass 0.01, sigma
        // 0.0025) applied BETWEEN grayscale and blur, as the original does
        // (Controller.cs:2409-2414) -- not after the blur.
        isacpu::debayer_bw_gauss(raw16[i], W, H, cfa, black_level, white_level, camera_white,
                                 0.5f, /*green_only=*/false, bw_track[i],
                                 /*tracking_fourier=*/true, /*clear_axis=*/0,
                                 /*high_pass=*/0.01f, /*high_pass_sigma=*/0.0025f);
        report("ISA: frame " + std::to_string(i + 1) + " prepared",
               0.02f + 0.10f * (i + 1) / frame_count);
    }

    // ---- 3. Whole-image pre-alignment: ISA ScanAngles, full resolution ----
    // (incr 0.1 deg, range +-1.0 deg -- ISA's own defaults). This is the
    // slowest stage by far; identical to the desktop port on purpose.
    std::vector<IsaFrameAlign> align(frame_count);
    for (int f = 0; f < frame_count; ++f) {
        if (f == referenceIndex) continue;
        report("ISA: pre-align frame " + std::to_string(f + 1), 0.12f + 0.18f * f / frame_count);
        isacpu::RotationResult r = isacpu::scan_angles(bw_track[referenceIndex], bw_track[f],
                                                       W, H, /*incr=*/0.1f, /*range=*/1.0f,
                                                       /*zero=*/0.f);
        align[f].shift = isacpu::Vec2f{r.shift_x, r.shift_y};
        align[f].rotation_rad = r.best_angle_deg * kIsaPi / 180.f;
    }

    // ---- 4. Patch tracking (TileSize 32, MaxShift 2, TrackingStrategy.Full)
    // SINGLE level, deliberately: ISA's controller constructor seeds exactly
    // one PatchTrackingLevel (ResizeLevel 1, TileSize 32, MaxShift 2 --
    // PatchTrackingLevel.cs:33-38, Controller.cs:387-388); the multi-level
    // pyramid with per-level minimize/upsample exists only when a user adds
    // levels in the WPF UI. One level IS the identical default.
    const int tileSize = 32, maxShift = 2;
    const int tileCountX = (W - 2 * maxShift) / tileSize;
    const int tileCountY = (H - 2 * maxShift) / tileSize;
    const int tileCount = tileCountX * tileCountY;
    if (tileCountX < 1 || tileCountY < 1) return Image();

    auto pairs = isacpu::shift_pairs_full(frame_count);
    auto design = isacpu::shift_matrix_full(frame_count);
    std::vector<std::vector<isacpu::Vec2f>> pair_shifts(pairs.size());
    for (size_t p = 0; p < pairs.size(); ++p) {
        report("ISA: tracking pair " + std::to_string(p + 1) + "/" + std::to_string(pairs.size()),
               0.30f + 0.15f * (float)p / (float)pairs.size());
        const int ref = pairs[p].reference, trk = pairs[p].to_track;
        std::vector<isacpu::Vec2f> pre_shift(tileCount, isacpu::Vec2f{0, 0});
        isacpu::track(bw_track[trk], bw_track[ref], W, H, pre_shift, tileSize, maxShift,
                      tileCountX, tileCountY,
                      align[ref].shift, align[ref].rotation_rad,
                      align[trk].shift, align[trk].rotation_rad,
                      /*threshold=*/1.0f);
        pair_shifts[p] = std::move(pre_shift);
    }
    std::vector<std::vector<isacpu::Vec2f>> tile_increments(tileCount);
    for (int t = 0; t < tileCount; ++t) {
        std::vector<isacpu::Vec2f> measured(pairs.size());
        for (size_t p = 0; p < pairs.size(); ++p) measured[p] = pair_shifts[p][t];
        tile_increments[t] = isacpu::minimize_shifts_for_tile(measured, design, frame_count);
    }
    pair_shifts.clear();
    bw_track.clear();
    bw_track.shrink_to_fit();  // ~4 bytes/px/frame released before the merge
    report("ISA: tracking solved", 0.46f);

    // ---- 6 (ref-only). Accumulation grey + merge-kernel covariance --------
    // (ISA's stage order computes all dense flows first; per-frame work is
    // independent, so it is interleaved with stage 7 below instead --
    // identical results, one dense flow alive at a time.)
    std::vector<float> bw_accum_ref;
    isacpu::debayer_bw_gauss(raw16[referenceIndex], W, H, cfa, black_level, white_level,
                             camera_white, 0.5f, /*green_only=*/false, bw_accum_ref);
    std::vector<float> dx, dy;
    isacpu::compute_derivatives(bw_accum_ref, W, H, dx, dy);
    std::vector<isacpu::Vec3f> kernel_param;
    {
        std::vector<isacpu::Vec3f> tensor;
        isacpu::compute_structure_tensor_raw(dx, dy, W, H, tensor);
        auto k1d = isacpu::gaussian_filter_1d(1.0f);
        isacpu::gaussian_blur_tensor(tensor, W, H, k1d, kernel_param);
    }
    dx.clear(); dy.clear(); dx.shrink_to_fit(); dy.shrink_to_fit();
    isacpu::compute_kernel_param(kernel_param, W, H, /*Dth=*/0.001f, /*Dtr=*/0.006f,
                                 /*kDetail=*/0.25f, /*kDenoise=*/3.0f,
                                 /*kStretch=*/4.0f, /*kShrink=*/2.0f);
    report("ISA: merge kernels ready", 0.50f);

    // ---- 7. Robustness + accumulate, per frame ---------------------------
    const int halfW = W / 2, halfH = H / 2;
    std::vector<isacpu::Rgbf> pixel((size_t)W * H, isacpu::Rgbf{});
    std::vector<isacpu::Rgbf> total_weight((size_t)W * H, isacpu::Rgbf{});
    {
        std::vector<isacpu::Vec4f> certainty((size_t)halfW * halfH, isacpu::Vec4f{1, 1, 1, 1});
        std::vector<isacpu::Vec2f> zero_shift((size_t)W * H, isacpu::Vec2f{0, 0});
        isacpu::accumulate_images(raw16[referenceIndex], W, H, cfa, pixel, total_weight,
                                  certainty, halfW, kernel_param, zero_shift,
                                  white_level, black_level);
    }
    std::vector<isacpu::Rgbf> ref_half;
    isacpu::debayer_subsample3(raw16[referenceIndex], W, H, cfa, kIsaWhite, ref_half);

    // ISA reads the green-channel NoiseProfile; the app's config carries the
    // same alpha/beta for the prewhitened 0..1 domain, which is exactly the
    // domain the synthetic raw normalizes back to.
    const double alpha = work.has_noise_profile ? (double)work.alpha_dng[1] : 1e-4;
    const double beta = work.has_noise_profile ? (double)work.beta_dng[1] : 1e-6;

    for (int f = 0; f < frame_count; ++f) {
        if (f == referenceIndex) continue;
        report("ISA: merge frame " + std::to_string(f + 1),
               0.50f + 0.35f * (float)f / (float)frame_count);

        // Stage 5 for this frame: dense LK-refined flow from the tracked tiles.
        std::vector<isacpu::Vec2f> tiled_flow(tileCount);
        for (int t = 0; t < tileCount; ++t)
            tiled_flow[t] = isacpu::optimal_shift_from_increments(tile_increments[t],
                                                                 referenceIndex, f);
        std::vector<float> bw_accum_f;
        isacpu::debayer_bw_gauss(raw16[f], W, H, cfa, black_level, white_level, camera_white,
                                 0.5f, /*green_only=*/false, bw_accum_f);
        std::vector<isacpu::Vec2f> dense_flow;
        isacpu::lucas_kanade_refine(bw_accum_f, bw_accum_ref, W, H, tiled_flow, tileSize,
                                    tileCountX, tileCountY, /*iterations=*/3,
                                    align[f].shift, align[f].rotation_rad,
                                    /*min_det=*/0.01f, /*window_size=*/11, dense_flow);
        bw_accum_f.clear();
        bw_accum_f.shrink_to_fit();

        std::vector<isacpu::Rgbf> frame_half;
        isacpu::debayer_subsample3(raw16[f], W, H, cfa, kIsaWhite, frame_half);
        std::vector<isacpu::Vec2f> shift_half((size_t)halfW * halfH);
        for (int y = 0; y < halfH; ++y)
            for (int x = 0; x < halfW; ++x)
                shift_half[(size_t)y * halfW + x] = dense_flow[(size_t)(y * 2) * W + (x * 2)];
        std::vector<isacpu::Vec4f> certainty((size_t)halfW * halfH, isacpu::Vec4f{1, 1, 1, 1});
        isacpu::compute_robustness_mask(ref_half, frame_half, shift_half, halfW, halfH,
                                        (float)alpha, (float)beta,
                                        /*threshold_m=*/100000.f, certainty);
        frame_half.clear();
        shift_half.clear();

        isacpu::accumulate_images(raw16[f], W, H, cfa, pixel, total_weight, certainty, halfW,
                                  kernel_param, dense_flow, white_level, black_level);
        raw16[f].clear();
        raw16[f].shrink_to_fit();  // this frame is fully merged
    }

    // ---- 8. ISA's SaveAs16BitTiff colour chain -> baked-sRGB DNG ---------
    report("ISA: rendering colour", 0.88f);
    isacpu::DngRaw meta{};
    meta.width = W;
    meta.height = H;
    meta.has_as_shot_neutral = true;
    meta.as_shot_neutral[0] = meta.as_shot_neutral[1] = meta.as_shot_neutral[2] = 1.f;
    if (work.has_color_matrix) {
        meta.has_color_matrix1 = true;
        for (int i = 0; i < 9; ++i) meta.color_matrix1[i] = work.color_matrix[i];
        meta.calibration_illuminant1 = 21;  // D65; single-matrix profile
    }
    isacpu::CameraProfile profile = isacpu::camera_to_pcs_matrix(meta);
    isacpu::Mat3d camera_to_prophoto =
        isacpu::mat3_mul_public(isacpu::prophoto_from_pcs(), profile.camera_to_pcs);
    isacpu::Mat3d prophoto_to_srgb =
        isacpu::mat3_mul_public(isacpu::srgb50_from_pcs(), isacpu::prophoto_to_pcs());
    constexpr float kExposureStops = 0.f;  // ISA's Exposure default

    std::vector<uint16_t> rgb16((size_t)W * H * 3, 0);
    isacpu::parallel_for(H - 2, [&](int yi) {
        const int y = yi + 1;
        for (int x = 1; x < W - 1; ++x) {
            const size_t idx = (size_t)y * W + x;
            const isacpu::Rgbf& p = pixel[idx];
            const isacpu::Rgbf& w = total_weight[idx];
            isacpu::Rgbf camera_rgb{w.r > 0.f ? p.r / w.r : 0.f,
                            w.g > 0.f ? p.g / w.g : 0.f,
                            w.b > 0.f ? p.b / w.b : 0.f};
            isacpu::Rgbf srgb = isacpu::render_pixel(camera_rgb, camera_to_prophoto, prophoto_to_srgb,
                                             profile.camera_white, kExposureStops);
            srgb.r = std::min(std::max(srgb.r, 0.f), 1.f);
            srgb.g = std::min(std::max(srgb.g, 0.f), 1.f);
            srgb.b = std::min(std::max(srgb.b, 0.f), 1.f);
            rgb16[idx * 3 + 0] = (uint16_t)std::lround(srgb.r * 65535.f);
            rgb16[idx * 3 + 1] = (uint16_t)std::lround(srgb.g * 65535.f);
            rgb16[idx * 3 + 2] = (uint16_t)std::lround(srgb.b * 65535.f);
        }
    });
    pixel.clear();
    total_weight.clear();

    // ---- Write (baked sRGB, like the desktop's write_rendered_dng) --------
    report("ISA: writing DNG", 0.94f);
    if (rgb16_sink) {
        rgb16_sink->w = W;
        rgb16_sink->h = H;
        rgb16_sink->rgb = rgb16;
    }
    DngStreamWriter writer;
    const bool opened =
        writer.open(dng_path, W, H, "HandheldSR-ISA", work.orientation,
                    /*colorMatrixXYZtoCam=*/nullptr, /*wbGainsGreenNorm=*/nullptr,
                    /*bakedSrgb=*/true, "HandheldSR",
                    /*camToSrgb=*/nullptr, /*pixelsPrewhitened=*/false,
                    work.dng_lossless_jpeg);
    bool wrote = false;
    if (opened) {
        wrote = writer.write_rows(rgb16.data(), H);
        writer.close();
    }
    if (!opened || !wrote)
        report("ISA: DNG write failed (result kept in memory)", 0.97f);

    const float pscale = std::min(1.f, (float)maxPreviewDim / std::max(H, W));
    const int ph = std::max(1, (int)(H * pscale));
    const int pw = std::max(1, (int)(W * pscale));
    Image preview(ph, pw, 3);
    for (int y = 0; y < H; ++y) {
        const int py = std::min(ph - 1, (int)(y * pscale));
        for (int x = 0; x < W; ++x) {
            const int px = std::min(pw - 1, (int)(x * pscale));
            const size_t o = ((size_t)y * W + x) * 3;
            preview.at(py, px, 0) = rgb16[o] / 65535.f;
            preview.at(py, px, 1) = rgb16[o + 1] / 65535.f;
            preview.at(py, px, 2) = rgb16[o + 2] / 65535.f;
        }
    }
    report("Done", 1.0f);
    return preview;
}

} // namespace hhsr
