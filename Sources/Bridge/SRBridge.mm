#import "SRBridge.h"
#import <UIKit/UIKit.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>

#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>

#include "core/types.h"
#include "core/pipeline.h"
#include "core/metal_gpu.h"
#include "core/mps_fft.h"
#include "core/preset_lut.h"
#include "core/render_isp.h"
#include "core/dng_writer.h"
#include "core/parallel.h"

using namespace hhsr;

static UIImage* UIImageFromPreview(const Image& preview) {
    if (preview.h <= 0 || preview.w <= 0 || preview.c < 3) return nil;

    const int w = preview.w, h = preview.h;
    std::vector<uint8_t> rgba((size_t)w * h * 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t o = ((size_t)y * w + x) * 4;
            // Preview is already sRGB-encoded by the pipeline.
            rgba[o + 0] = (uint8_t)std::lround(clampf(preview.at(y, x, 0), 0.f, 1.f) * 255.f);
            rgba[o + 1] = (uint8_t)std::lround(clampf(preview.at(y, x, 1), 0.f, 1.f) * 255.f);
            rgba[o + 2] = (uint8_t)std::lround(clampf(preview.at(y, x, 2), 0.f, 1.f) * 255.f);
            rgba[o + 3] = 255;
        }
    }

    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (!cs) cs = CGColorSpaceCreateDeviceRGB();
    NSData* nsData = [NSData dataWithBytes:rgba.data() length:rgba.size()];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)nsData);
    CGImageRef cg = CGImageCreate(
        w, h, 8, 32, w * 4, cs,
        kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast,
        provider, NULL, false, kCGRenderingIntentDefault);
    UIImage* img = cg ? [UIImage imageWithCGImage:cg] : nil;
    if (cg) CGImageRelease(cg);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(cs);
    return img;
}

static inline float to_srgb_gamma(float v) {
    v = clampf(v, 0.f, 1.f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
}

static inline float render_luminance(float r, float g, float b) {
    return std::max(0.f, 0.2126f * r + 0.7152f * g + 0.0722f * b);
}

static inline float render_aces_filmic(float x) {
    x = std::max(0.f, x);
    constexpr float a = 2.51f;
    constexpr float b = 0.03f;
    constexpr float c = 2.43f;
    constexpr float d = 0.59f;
    constexpr float e = 0.14f;
    return clampf((x * (a * x + b)) / (x * (c * x + d) + e), 0.f, 1.f);
}

static inline void render_desaturate_to_luma(float& r, float& g, float& b, float amount) {
    amount = clampf(amount, 0.f, 1.f);
    const float y = render_luminance(r, g, b);
    r = y + (r - y) * (1.f - amount);
    g = y + (g - y) * (1.f - amount);
    b = y + (b - y) * (1.f - amount);
}

static inline float render_s_curve_luma(float y) {
    y = clampf(y, 0.f, 1.f);
    const float smooth = y * y * (3.f - 2.f * y);
    y = y * 0.72f + smooth * 0.28f;
    return clampf((y - 0.5f) * 1.08f + 0.5f, 0.f, 1.f);
}

// Soft over-range rolloff (linear > 1) before display gamma.
static inline float highlight_rolloff_overrange(float v) {
    if (v <= 1.f) return clampf(v, 0.f, 1.f);
    return 1.f - 1.f / (1.f + (v - 1.f) * 2.5f);
}

// Approximate Lightroom Highlights −100: compress upper tones aggressively.
static inline float apply_highlights_neg(float v, float amount) {
    v = clampf(v, 0.f, 1.f);
    amount = clampf(amount, 0.f, 1.f);
    if (amount <= 0.f) return v;
    const float knee = 0.35f; // start compressing earlier than before
    if (v <= knee) return v;
    float t = (v - knee) / (1.f - knee); // 0..1 in highlight zone
    // Stronger curve: cube for aggressive rolloff on the brightest tones.
    float t3 = t * t * t;
    float compressed = knee + (v - knee) * (1.f - amount * 0.78f * t3);
    return clampf(compressed, 0.f, 1.f);
}

// Approximate Lightroom Shadows +60: lift dark tones without washing out mids.
static inline float apply_shadows_lift(float v, float amount) {
    v = clampf(v, 0.f, 1.f);
    if (amount <= 0.f) return v;
    // Only boost below the upper shoulder (keep highlights untouched).
    const float shoulder = 0.55f;
    if (v >= shoulder) return v;
    float t = 1.f - v / shoulder; // 1 at black, 0 at shoulder
    float lift = amount * t * t * 0.28f; // quadratic fade-in from darks
    return clampf(v + lift, 0.f, shoulder);
}

// Display S-curve + mild midtone contrast (LR Contrast-ish, no CI filters).
static inline float tone_s_curve(float v) {
    v = clampf(v, 0.f, 1.f);
    const float s = v * v * (3.f - 2.f * v);
    v = clampf(v * 0.30f + s * 0.70f, 0.f, 1.f);
    // Gentler pivot contrast so shadows don't get re-crushed.
    v = clampf((v - 0.5f) * 1.06f + 0.5f, 0.f, 1.f);
    return v;
}

// Vibrance-like chroma lift in display space (keeps neutrals).
static inline void apply_vibrance_rgb(float& r, float& g, float& b, float amount) {
    const float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    const float mx = std::max(r, std::max(g, b));
    const float mn = std::min(r, std::min(g, b));
    const float sat = (mx > 1e-6f) ? (mx - mn) / mx : 0.f;
    const float boost = 1.f + amount * (1.f - sat);
    r = clampf(y + (r - y) * boost, 0.f, 1.f);
    g = clampf(y + (g - y) * boost, 0.f, 1.f);
    b = clampf(y + (b - y) * boost, 0.f, 1.f);
}

// Legacy fallback for non-app DNGs with non-neutral WB metadata.
static inline void tone_map_legacy_camera_rgb(float& sr, float& sg, float& sb) {
    sr = std::max(0.f, sr);
    sg = std::max(0.f, sg);
    sb = std::max(0.f, sb);

    constexpr float kExposure = 1.18f;
    sr *= kExposure;
    sg *= kExposure;
    sb *= kExposure;

    const float hi = std::max(sr, std::max(sg, sb));
    if (hi > 0.82f)
        render_desaturate_to_luma(sr, sg, sb, 0.80f * smoothstepf(0.82f, 1.02f, hi));

    const float old_y = render_luminance(sr, sg, sb);
    float new_y = render_aces_filmic(old_y);
    const float shadow = 1.f - smoothstepf(0.10f, 0.48f, new_y);
    new_y = clampf(new_y * (1.f + 0.38f * shadow), 0.f, 1.f);
    new_y = render_s_curve_luma(new_y);

    const float lum_scale = new_y / (old_y + 1e-6f);
    sr *= lum_scale;
    sg *= lum_scale;
    sb *= lum_scale;

    const float y = render_luminance(sr, sg, sb);
    const float mx0 = std::max(sr, std::max(sg, sb));
    const float mn0 = std::min(sr, std::min(sg, sb));
    const float sat = (mx0 > 1e-6f) ? (mx0 - mn0) / (mx0 + 1e-6f) : 0.f;
    const float vibrance = 1.f + 0.18f * (1.f - sat);
    sr = y + (sr - y) * vibrance;
    sg = y + (sg - y) * vibrance;
    sb = y + (sb - y) * vibrance;

    sr = std::max(0.f, sr);
    sg = std::max(0.f, sg);
    sb = std::max(0.f, sb);
    float mx = std::max(sr, std::max(sg, sb));
    if (mx > 1.f) {
        render_desaturate_to_luma(sr, sg, sb, smoothstepf(1.f, 1.35f, mx));
        mx = std::max(sr, std::max(sg, sb));
        if (mx > 1.f) {
            const float s = 1.f / mx;
            sr *= s;
            sg *= s;
            sb *= s;
        }
    }

    sr = to_srgb_gamma(sr);
    sg = to_srgb_gamma(sg);
    sb = to_srgb_gamma(sb);
    return;
#if 0
    // Clamp negative channels BEFORE rolloff — after the cam→sRGB matrix,
    // R or B can go negative while G stays positive, producing green speckles.
    sr = std::max(0.f, sr);
    sg = std::max(0.f, sg);
    sb = std::max(0.f, sb);
    sr = highlight_rolloff_overrange(sr);
    sg = highlight_rolloff_overrange(sg);
    sb = highlight_rolloff_overrange(sb);
    sr = to_srgb_gamma(sr);
    sg = to_srgb_gamma(sg);
    sb = to_srgb_gamma(sb);
    constexpr float kHighlightsNeg = 1.00f; // Lightroom Highlights −100
    sr = apply_highlights_neg(sr, kHighlightsNeg);
    sg = apply_highlights_neg(sg, kHighlightsNeg);
    sb = apply_highlights_neg(sb, kHighlightsNeg);
    constexpr float kShadowsLift = 1.00f; // Lightroom Shadows +60
    sr = apply_shadows_lift(sr, kShadowsLift);
    sg = apply_shadows_lift(sg, kShadowsLift);
    sb = apply_shadows_lift(sb, kShadowsLift);
    sr = tone_s_curve(sr);
    sg = tone_s_curve(sg);
    sb = tone_s_curve(sb);
    apply_vibrance_rgb(sr, sg, sb, 0.48f);
#endif
}

// Render settings for exportJPEGFromLinearDNG / embedJPEGPreviewInDNG. Those
// take only a path -- there is no Config to thread through -- so the values are
// parked here when the tuning dictionary is parsed.
static hhsr::IspParams g_isp;

static inline bool render_wb_is_neutral(const float wb[3]) {
    return std::fabs(wb[0] - 1.f) < 1e-4f &&
           std::fabs(wb[1] - 1.f) < 1e-4f &&
           std::fabs(wb[2] - 1.f) < 1e-4f;
}

static inline void tone_map_calibrated_display_rgb(float& sr, float& sg, float& sb) {
    sr = std::max(0.f, sr);
    sg = std::max(0.f, sg);
    sb = std::max(0.f, sb);

    const float mx = std::max(sr, std::max(sg, sb));
    if (mx > 1.f) {
        render_desaturate_to_luma(sr, sg, sb, smoothstepf(1.f, 1.25f, mx));
        const float mx2 = std::max(sr, std::max(sg, sb));
        if (mx2 > 1.f) {
            const float inv = 1.f / mx2;
            sr *= inv;
            sg *= inv;
            sb *= inv;
        }
    }

    sr = to_srgb_gamma(sr);
    sg = to_srgb_gamma(sg);
    sb = to_srgb_gamma(sb);
}

static inline void render_linear_dng_pixel(float r, float g, float b,
                                           const float wb[3], const float m[9],
                                           bool has_color,
                                           float& sr, float& sg, float& sb) {
    if (has_color && render_wb_is_neutral(wb) && preset_lut_enabled()) {
        // The fitted table subsumes the matrix and tone curve below: it was
        // fitted from this exact input -- pre-white-balanced linear DNG RGB --
        // straight to the target JPEG, so nothing else may be applied on top.
        //
        // Gated on the same neutral-white-balance condition as the calibrated
        // branch, because that is the case it was fitted for. A DNG carrying
        // real WB gains falls through to the legacy path below.
        const float lin[3] = {r, g, b};
        float out[3];
        preset_lut_apply(lin, out);
        sr = out[0];
        sg = out[1];
        sb = out[2];
        return;
    }

    if (has_color && render_wb_is_neutral(wb)) {
        // Calibrated from the supplied HandheldSR linear DNG -> Lightroom iOS JPEG pair.
        // The SR output is already pre-white-balanced, so using the DNG camera matrix
        // directly here double-pushes color and causes magenta/green highlight casts.
        constexpr float kDisplayMatrix[9] = {
             1.2466443f, -0.4477117f, -0.1773365f,
            -0.1616100f,  0.8074801f, -0.0321825f,
            -0.1166101f, -0.1502432f,  0.8686564f
        };
        sr = kDisplayMatrix[0] * r + kDisplayMatrix[1] * g + kDisplayMatrix[2] * b;
        sg = kDisplayMatrix[3] * r + kDisplayMatrix[4] * g + kDisplayMatrix[5] * b;
        sb = kDisplayMatrix[6] * r + kDisplayMatrix[7] * g + kDisplayMatrix[8] * b;
        tone_map_calibrated_display_rgb(sr, sg, sb);
        return;
    }

    const float wr = r * wb[0];
    const float wg = g * wb[1];
    const float wb_ = b * wb[2];
    if (has_color) {
        sr = m[0] * wr + m[1] * wg + m[2] * wb_;
        sg = m[3] * wr + m[4] * wg + m[5] * wb_;
        sb = m[6] * wr + m[7] * wg + m[8] * wb_;
    } else {
        sr = wr;
        sg = wg;
        sb = wb_;
    }
    tone_map_legacy_camera_rgb(sr, sg, sb);
}

static void ApplyTuningParams(NSDictionary<NSString *, NSNumber *> *tuning, Config& cfg) {
    if (!tuning) return;
    if (tuning[@"r_t"]) cfg.r_t = tuning[@"r_t"].floatValue;
    if (tuning[@"r_s1"]) cfg.r_s1 = tuning[@"r_s1"].floatValue;
    if (tuning[@"r_s2"]) cfg.r_s2 = tuning[@"r_s2"].floatValue;
    if (tuning[@"r_Mt"]) cfg.r_Mt = tuning[@"r_Mt"].floatValue;
    // Alignment grey: FFT low-pass at full resolution, or the 2x2 Bayer quad
    // average at half resolution that Wronski et al. describe. The quad average
    // also constrains displacements to multiples of 2 Bayer pixels, so shifted
    // samples keep coincident colors.
    if (tuning[@"alignment_grey_fft"])
        cfg.grey_method = tuning[@"alignment_grey_fft"].boolValue
                              ? GreyMethod::FFT : GreyMethod::Decimate;
    if (tuning[@"hf_artifact_removal_enabled"])
        cfg.hf_artifact_removal_enabled = tuning[@"hf_artifact_removal_enabled"].boolValue;
    if (tuning[@"hf_variance_loss_threshold"])
        cfg.hf_variance_loss_threshold = tuning[@"hf_variance_loss_threshold"].floatValue;
    if (tuning[@"hf_min_texture_snr"])
        cfg.hf_min_texture_snr = tuning[@"hf_min_texture_snr"].floatValue;
    if (tuning[@"flow_reject_1d_enabled"])
        cfg.flow_reject_1d_enabled = tuning[@"flow_reject_1d_enabled"].boolValue;
    if (tuning[@"flow_regularize_aperture_ratio"])
        cfg.flow_regularize_aperture_ratio =
            std::max(0.f, std::min(1.f, tuning[@"flow_regularize_aperture_ratio"].floatValue));
    if (tuning[@"flow_reject_1d_ambiguity_ratio"])
        cfg.flow_reject_1d_ambiguity_ratio =
            std::max(1.f, tuning[@"flow_reject_1d_ambiguity_ratio"].floatValue);
    if (tuning[@"flow_reject_1d_residual_threshold"])
        cfg.flow_reject_1d_residual_threshold =
            std::max(0.f, tuning[@"flow_reject_1d_residual_threshold"].floatValue);
    if (tuning[@"motion_edge_rejection_enabled"])
        cfg.motion_edge_rejection_enabled = tuning[@"motion_edge_rejection_enabled"].boolValue;
    if (tuning[@"motion_edge_threshold"])
        cfg.motion_edge_threshold = tuning[@"motion_edge_threshold"].floatValue;
    if (tuning[@"motion_edge_residual_threshold"])
        cfg.motion_edge_residual_threshold = tuning[@"motion_edge_residual_threshold"].floatValue;
    if (tuning[@"motion_edge_noise_floor_multiplier"])
        cfg.motion_edge_noise_floor_multiplier =
            tuning[@"motion_edge_noise_floor_multiplier"].floatValue;
    if (tuning[@"motion_edge_neighborhood_radius"])
        cfg.motion_edge_neighborhood_radius =
            std::max(0, std::min(2, tuning[@"motion_edge_neighborhood_radius"].intValue));
    if (tuning[@"k_detail"]) cfg.k_detail = tuning[@"k_detail"].floatValue;
    if (tuning[@"k_denoise"]) cfg.k_denoise = tuning[@"k_denoise"].floatValue;
    if (tuning[@"k_stretch"]) cfg.k_stretch = tuning[@"k_stretch"].floatValue;
    if (tuning[@"k_shrink"]) cfg.k_shrink = tuning[@"k_shrink"].floatValue;
    if (tuning[@"snr_auto_tune"]) cfg.snr_auto_tune = tuning[@"snr_auto_tune"].boolValue;
    if (tuning[@"debug_pixel4a_noise_profile"])
        cfg.debug_pixel4a_noise_profile = tuning[@"debug_pixel4a_noise_profile"].boolValue;
    if (tuning[@"alignment_tile_size"]) {
        const int ts = tuning[@"alignment_tile_size"].intValue;
        cfg.alignment_tile_size =
            (ts == 8 || ts == 16 || ts == 32 || ts == 64) ? ts : 0;
    }
    if (tuning[@"global_prealignment_enabled"])
        cfg.global_prealignment_enabled = tuning[@"global_prealignment_enabled"].boolValue;
    if (tuning[@"global_prealignment_choose_reference"])
        cfg.global_prealignment_choose_reference =
            tuning[@"global_prealignment_choose_reference"].boolValue;
    if (tuning[@"global_prealignment_rotation_range_deg"])
        cfg.global_prealignment_rotation_range_deg =
            std::max(0.f, std::min(2.f,
                tuning[@"global_prealignment_rotation_range_deg"].floatValue));
    if (tuning[@"global_prealignment_rotation_step_deg"])
        cfg.global_prealignment_rotation_step_deg =
            std::max(0.05f, std::min(1.f,
                tuning[@"global_prealignment_rotation_step_deg"].floatValue));
    if (tuning[@"global_prealignment_max_shift"])
        cfg.global_prealignment_max_shift =
            std::max(0, std::min(64, tuning[@"global_prealignment_max_shift"].intValue));
    if (tuning[@"robustness_save_mask"])
        cfg.robustness_save_mask = tuning[@"robustness_save_mask"].boolValue;
    if (tuning[@"accumulated_robustness_denoiser_enabled"]) {
        cfg.accumulated_robustness_denoiser_enabled =
            tuning[@"accumulated_robustness_denoiser_enabled"].boolValue;
    }
    if (tuning[@"merge_arch"]) cfg.merge_arch = tuning[@"merge_arch"].intValue;
    if (tuning[@"acc_rob_adaptive"])
        cfg.acc_rob_adaptive = tuning[@"acc_rob_adaptive"].boolValue;
    if (tuning[@"isp_enabled"])        cfg.isp.enabled = tuning[@"isp_enabled"].boolValue;
    if (tuning[@"isp_exposure_ev"])    cfg.isp.exposure_ev = tuning[@"isp_exposure_ev"].floatValue;
    if (tuning[@"isp_highlight_knee"]) cfg.isp.highlight_knee = tuning[@"isp_highlight_knee"].floatValue;
    if (tuning[@"isp_local_strength"]) cfg.isp.local_strength = tuning[@"isp_local_strength"].floatValue;
    if (tuning[@"isp_highlight"])      cfg.isp.highlight_rolloff = tuning[@"isp_highlight"].floatValue;
    if (tuning[@"isp_shadow"])         cfg.isp.shadow_lift = tuning[@"isp_shadow"].floatValue;
    if (tuning[@"isp_black_point"])    cfg.isp.black_point = tuning[@"isp_black_point"].floatValue;
    if (tuning[@"isp_warmth"])         cfg.isp.warmth = tuning[@"isp_warmth"].floatValue;
    if (tuning[@"isp_colour_strength"]) cfg.isp.colour_strength = tuning[@"isp_colour_strength"].floatValue;
    if (tuning[@"isp_contrast"])       cfg.isp.contrast = tuning[@"isp_contrast"].floatValue;
    if (tuning[@"isp_vibrance"])       cfg.isp.vibrance = tuning[@"isp_vibrance"].floatValue;
    if (tuning[@"isp_chroma_denoise"]) cfg.isp.chroma_denoise = tuning[@"isp_chroma_denoise"].floatValue;
    if (tuning[@"isp_chroma_radius"])  cfg.isp.chroma_denoise_radius = tuning[@"isp_chroma_radius"].floatValue;
    if (tuning[@"isp_saturation"])     cfg.isp.saturation = tuning[@"isp_saturation"].floatValue;
    if (tuning[@"isp_local_contrast"]) cfg.isp.local_contrast = tuning[@"isp_local_contrast"].floatValue;
    if (tuning[@"isp_skin_protect"])   cfg.isp.skin_protect = tuning[@"isp_skin_protect"].boolValue;
    g_isp = cfg.isp;
    if (tuning[@"align_ica_per_level"])
        cfg.align_ica_per_level = tuning[@"align_ica_per_level"].boolValue;
    if (tuning[@"align_ica_per_level_fft"])
        cfg.align_ica_per_level_fft = tuning[@"align_ica_per_level_fft"].boolValue;
    if (tuning[@"acc_rob_max_frame_count"])
        cfg.acc_rob_max_frame_count = tuning[@"acc_rob_max_frame_count"].floatValue;
    if (tuning[@"acc_rob_rad_max"]) cfg.acc_rob_rad_max = tuning[@"acc_rob_rad_max"].floatValue;
    if (tuning[@"acc_rob_max_multiplier"])
        cfg.acc_rob_max_multiplier = tuning[@"acc_rob_max_multiplier"].floatValue;
}

static std::string NSStringToStd(id value) {
    if (![value isKindOfClass:NSString.class]) return std::string();
    return std::string([(NSString *)value UTF8String]);
}

static NSDictionary *DictValue(NSDictionary *dict, NSString *key) {
    id v = dict[key];
    return [v isKindOfClass:NSDictionary.class] ? (NSDictionary *)v : nil;
}

static id FirstValueForKeys(NSDictionary *dict, NSArray<NSString *> *keys) {
    if (!dict) return nil;
    for (NSString *key in keys) {
        id v = dict[key];
        if (v) return v;
    }
    return nil;
}

static void CollectNumbers(id obj, std::vector<double>& out) {
    if (!obj || obj == (id)kCFNull) return;
    if ([obj isKindOfClass:NSNumber.class]) {
        out.push_back([(NSNumber *)obj doubleValue]);
    } else if ([obj isKindOfClass:NSArray.class]) {
        for (id item in (NSArray *)obj) CollectNumbers(item, out);
    }
}

static float FirstNumber(id obj, float fallback) {
    std::vector<double> vals;
    CollectNumbers(obj, vals);
    if (vals.empty() || !std::isfinite(vals[0])) return fallback;
    return (float)vals[0];
}

static NSDictionary *DNGMetadata(NSDictionary *metadata) {
    if (!metadata) return nil;
    NSDictionary *dng = DictValue(metadata, (__bridge NSString *)kCGImagePropertyDNGDictionary);
    if (dng) return dng;
    return DictValue(metadata, @"{DNG}");
}

static NSDictionary *TIFFMetadata(NSDictionary *metadata) {
    if (!metadata) return nil;
    NSDictionary *tiff = DictValue(metadata, (__bridge NSString *)kCGImagePropertyTIFFDictionary);
    if (tiff) return tiff;
    return DictValue(metadata, @"{TIFF}");
}

static NSDictionary *EXIFMetadata(NSDictionary *metadata) {
    if (!metadata) return nil;
    NSDictionary *exif = DictValue(metadata, (__bridge NSString *)kCGImagePropertyExifDictionary);
    if (exif) return exif;
    return DictValue(metadata, @"{Exif}");
}

static void FillReferenceMetadataFromRawFrame(NSDictionary *frame, Config& cfg) {
    NSDictionary *metadata = DictValue(frame, @"metadata");
    NSDictionary *dng = DNGMetadata(metadata);
    NSDictionary *tiff = TIFFMetadata(metadata);
    NSDictionary *exif = EXIFMetadata(metadata);

    cfg.camera_make = NSStringToStd(FirstValueForKeys(tiff, @[
        (__bridge NSString *)kCGImagePropertyTIFFMake, @"Make"
    ]));
    cfg.camera_model = NSStringToStd(FirstValueForKeys(tiff, @[
        (__bridge NSString *)kCGImagePropertyTIFFModel, @"Model"
    ]));

    id orientation = FirstValueForKeys(metadata, @[
        (__bridge NSString *)kCGImagePropertyOrientation, @"Orientation"
    ]);
    if (!orientation) orientation = FirstValueForKeys(tiff, @[@"Orientation"]);
    if ([orientation isKindOfClass:NSNumber.class])
        cfg.orientation = [(NSNumber *)orientation intValue];

    NSArray *cfa = frame[@"cfa"];
    if ([cfa isKindOfClass:NSArray.class] && cfa.count >= 4) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int c = [cfa[(NSUInteger)(i * 2 + j)] intValue];
                cfg.cfa.p[i][j] = (uint8_t)std::max(0, std::min(2, c));
            }
        }
    }

    std::vector<double> neutral;
    CollectNumbers(FirstValueForKeys(dng, @[@"AsShotNeutral"]), neutral);
    if (neutral.size() >= 3 && neutral[0] > 0.0 && neutral[1] > 0.0 && neutral[2] > 0.0) {
        cfg.white_balance[0] = (float)(1.0 / neutral[0]);
        cfg.white_balance[1] = (float)(1.0 / neutral[1]);
        cfg.white_balance[2] = (float)(1.0 / neutral[2]);
    } else {
        cfg.white_balance[0] = 1.f;
        cfg.white_balance[1] = 1.f;
        cfg.white_balance[2] = 1.f;
    }

    bool wb_ok = std::isfinite(cfg.white_balance[1]) && cfg.white_balance[1] > 0.f;
    for (int i = 0; i < 3; ++i)
        wb_ok = wb_ok && std::isfinite(cfg.white_balance[i]) && cfg.white_balance[i] > 0.f;
    if (!wb_ok) {
        cfg.white_balance[0] = 1.f;
        cfg.white_balance[1] = 1.f;
        cfg.white_balance[2] = 1.f;
    }

    std::vector<double> black;
    CollectNumbers(FirstValueForKeys(dng, @[@"BlackLevel"]), black);
    if (black.size() >= 4) {
        double sum[3] = {0.0, 0.0, 0.0};
        int count[3] = {0, 0, 0};
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int c = (int)cfg.cfa.p[i][j];
                c = std::max(0, std::min(2, c));
                sum[c] += black[(size_t)(i * 2 + j)];
                count[c] += 1;
            }
        }
        for (int c = 0; c < 3; ++c)
            cfg.black_levels[c] = count[c] > 0 ? (float)(sum[c] / (double)count[c]) : 0.f;
        cfg.has_black_levels = true;
    } else if (black.size() >= 3) {
        cfg.black_levels[0] = (float)black[0];
        cfg.black_levels[1] = (float)black[1];
        cfg.black_levels[2] = (float)black[2];
        cfg.has_black_levels = true;
    } else if (black.size() == 1) {
        cfg.black_levels[0] = cfg.black_levels[1] = cfg.black_levels[2] = (float)black[0];
        cfg.has_black_levels = true;
    } else {
        cfg.black_levels[0] = cfg.black_levels[1] = cfg.black_levels[2] = 0.f;
        cfg.has_black_levels = true;
    }
    cfg.white_level = FirstNumber(FirstValueForKeys(dng, @[@"WhiteLevel"]), 65535.f);
    if (!(cfg.white_level > 0.f) || !std::isfinite(cfg.white_level))
        cfg.white_level = 65535.f;

    std::vector<double> noise;
    CollectNumbers(FirstValueForKeys(dng, @[@"NoiseProfile"]), noise);
    if (noise.size() >= 2) {
        const size_t nplanes = noise.size() / 2u;
        // Read per-channel [R, G, B] noise without averaging, preserving differences.
        // If fewer than 3 planes, replicate the last one.
        bool ok = true;
        for (int c = 0; c < 3; ++c) {
            size_t src_plane = (c < (int)nplanes) ? c : (nplanes - 1);
            cfg.alpha_dng[c] = (float)noise[src_plane * 2u + 0u];
            cfg.beta_dng[c] = (float)noise[src_plane * 2u + 1u];
            if (!(cfg.alpha_dng[c] > 0.f && std::isfinite(cfg.alpha_dng[c]) &&
                  std::isfinite(cfg.beta_dng[c])))
                ok = false;
        }
        cfg.has_noise_profile = ok;
    } else {
        cfg.has_noise_profile = false;
    }
    if (cfg.debug_pixel4a_noise_profile) {
        const float iso = FirstNumber(FirstValueForKeys(exif, @[
            (__bridge NSString *)kCGImagePropertyExifISOSpeedRatings,
            @"ISOSpeedRatings", @"PhotographicSensitivity", @"ISO"
        ]), 100.f);
        apply_pixel4a_noise_profile(cfg, iso);
    }

    std::vector<double> color;
    CollectNumbers(FirstValueForKeys(dng, @[@"ColorMatrix1", @"ColorMatrix2"]), color);
    if (color.size() >= 9) {
        cfg.has_color_matrix = true;
        for (int i = 0; i < 9; ++i) cfg.color_matrix[i] = (float)color[(size_t)i];
    }
}

static Image DecodeRawFrameDictionary(NSDictionary *frame, Config& cfg,
                                      bool is_reference, int crop_h, int crop_w) {
    Image img;
    NSData *data = frame[@"data"];
    if (![data isKindOfClass:NSData.class]) {
        NSString *path = frame[@"path"];
        if ([path isKindOfClass:NSString.class]) {
            data = [NSData dataWithContentsOfFile:path
                                          options:NSDataReadingMappedIfSafe
                                            error:nil];
        }
    }
    NSNumber *wn = frame[@"width"];
    NSNumber *hn = frame[@"height"];
    NSNumber *bn = frame[@"bytesPerRow"];
    if (![data isKindOfClass:NSData.class] ||
        ![wn isKindOfClass:NSNumber.class] ||
        ![hn isKindOfClass:NSNumber.class] ||
        ![bn isKindOfClass:NSNumber.class])
        return img;

    const int w = wn.intValue;
    const int h = hn.intValue;
    const int bytes_per_row = bn.intValue;
    if (w <= 0 || h <= 0 || bytes_per_row < w * 2 ||
        data.length < (NSUInteger)((size_t)bytes_per_row * (size_t)h))
        return img;

    if (is_reference)
        FillReferenceMetadataFromRawFrame(frame, cfg);

    float maxv = cfg.has_black_levels ? cfg.white_level : 65535.f;
    float site_black[2][2];
    float site_denom[2][2];
    float site_wb[2][2];
    float flat_black[4];
    float flat_denom[4];
    float flat_wb[4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int c = (int)cfg.cfa.p[i][j];
            c = std::max(0, std::min(2, c));
            const float bl = cfg.has_black_levels ? cfg.black_levels[c] : 0.f;
            float denom = maxv - bl;
            if (!(denom > 0.f) || !std::isfinite(denom) || !std::isfinite(bl)) {
                site_black[i][j] = 0.f;
                site_denom[i][j] = (std::isfinite(maxv) && maxv > 0.f) ? maxv : 65535.f;
            } else {
                site_black[i][j] = bl;
                site_denom[i][j] = denom;
            }
            site_wb[i][j] = cfg.white_balance[c] / cfg.white_balance[1];
            const int site = i * 2 + j;
            flat_black[site] = site_black[i][j];
            flat_denom[site] = site_denom[i][j];
            flat_wb[site] = site_wb[i][j];
        }
    }

    const uint8_t *base = (const uint8_t *)data.bytes;
    if (!metal_decode_raw16_to_float(base, (size_t)data.length, h, w, bytes_per_row,
                                     flat_black, flat_denom, flat_wb, img)) {
        img = Image(h, w, 1);
        parallel_rows(h, 0, [&](int y) {
            const uint8_t *row = base + (size_t)y * (size_t)bytes_per_row;
            const int fi = y & 1;
            for (int x = 0; x < w; ++x) {
                uint16_t rawv = 0;
                std::memcpy(&rawv, row + (size_t)x * 2u, sizeof(rawv));
                const int fj = x & 1;
                float v = ((float)rawv - site_black[fi][fj]) / site_denom[fi][fj];
                v *= site_wb[fi][fj];
                if (!std::isfinite(v)) v = 0.f;
                img.at(y, x) = clampf(v, 0.f, 1.f);
            }
        });
    }
    cfg.raw_prewhitened = true;

    if (cfg.input_crop_zoom > 1.f) {
        const float z = cfg.input_crop_zoom;
        // Every extent and offset is forced even. Not tidiness: an odd origin
        // shifts the Bayer phase, so the CFA under the crop would no longer be
        // the CFA the pipeline is told it has. The epsilon stops an exact ratio
        // landing just under the integer and losing two rows.
        int ch = (int)((float)img.h / z + 1e-4f) & ~1;
        int cw = (int)((float)img.w / z + 1e-4f) & ~1;
        int y0 = ((img.h - ch) / 2) & ~1;
        int x0 = ((img.w - cw) / 2) & ~1;
        if (ch > 0 && cw > 0) {
            Image cropped(ch, cw, 1);
            for (int y = 0; y < ch; ++y)
                for (int x = 0; x < cw; ++x)
                    cropped.at(y, x) = img.at(y0 + y, x0 + x);
            img = std::move(cropped);
        }
    }

    if (crop_h > 0 && crop_w > 0 && (img.h > crop_h || img.w > crop_w)) {
        int mh = std::min(img.h, crop_h);
        int mw = std::min(img.w, crop_w);
        Image c(mh, mw, 1);
        for (int y = 0; y < mh; ++y)
            for (int x = 0; x < mw; ++x)
                c.at(y, x) = img.at(y, x);
        img = std::move(c);
    }
    return img;
}

@implementation SRBridge

+ (BOOL)processDNGs:(NSArray<NSString *> *)paths
             toPath:(NSString *)outPath
              scale:(float)scale
           cropZoom:(float)cropZoom
       tuningParams:(NSDictionary<NSString *, NSNumber *> *)tuning
           progress:(void (^)(NSString *, float))progress
        previewImage:(UIImage * _Nullable * _Nullable)previewOut {
    if (paths.count < 2) return NO;
    if (previewOut) *previewOut = nil;

    // Grey-FFT + L2 BM + merge accumulate require Metal (no CPU fallback).
    if (!metal_gpu_init()) return NO;

    std::vector<std::string> vpaths;
    vpaths.reserve(paths.count);
    for (NSString *p in paths) vpaths.emplace_back(p.UTF8String);

    Config cfg;
    cfg.scale = scale;
    cfg.input_crop_zoom = std::max(1.f, cropZoom);
    cfg.bayer_mode = true;
    cfg.bake_srgb = false;   // linear camera RGB in DNG; WB applied only for in-app preview / JPEG
    cfg.use_gpu = false;
    cfg.num_threads = 0;     // all CPU cores during active processing
    // Same as the camera path. The loader can re-read the DNG, so there is no
    // need to spill a normalized copy of every comparison frame to disk.
    //
    // This used to be left false because re-decoding a DNG (~190ms) cost more
    // than reading back a spilled .raw (~85ms). That trade is gone: comparison
    // frames are uploaded to the GPU as they are analyzed, so neither the spill
    // nor the reload happens in the normal case, and the reload is only a
    // fallback for a failed upload.
    cfg.stream_comp_raw_from_loader = true;

    ApplyTuningParams(tuning, cfg);

    ProgressFn cb = nullptr;
    if (progress) {
        cb = [progress](const std::string &stage, float f) {
            @autoreleasepool {
                progress([NSString stringWithUTF8String:stage.c_str()], f);
            }
        };
    }

    Image preview;
    @autoreleasepool {
        preview = process_burst_paths_to_dng(
            vpaths, cfg, std::string(outPath.UTF8String), cb, 256);
    }

    if (preview.w <= 0) return NO;

    if (previewOut) *previewOut = UIImageFromPreview(preview);
    return YES;
}

+ (BOOL)processRawFrames:(NSArray<NSDictionary<NSString *, id> *> *)frames
                  toPath:(NSString *)outPath
                   scale:(float)scale
                cropZoom:(float)cropZoom
            tuningParams:(NSDictionary<NSString *, NSNumber *> *)tuning
                progress:(void (^)(NSString *, float))progress
             previewImage:(UIImage * _Nullable * _Nullable)previewOut {
    if (frames.count < 2) return NO;
    if (previewOut) *previewOut = nil;

    if (!metal_gpu_init()) return NO;

    Config cfg;
    cfg.scale = scale;
    cfg.input_crop_zoom = std::max(1.f, cropZoom);
    cfg.bayer_mode = true;
    cfg.bake_srgb = false;
    cfg.use_gpu = false;
    cfg.num_threads = 0;
    cfg.stream_comp_raw_from_loader = true;
    ApplyTuningParams(tuning, cfg);

    ProgressFn cb = nullptr;
    if (progress) {
        cb = [progress](const std::string &stage, float f) {
            @autoreleasepool {
                progress([NSString stringWithUTF8String:stage.c_str()], f);
            }
        };
    }

    NSArray<NSDictionary<NSString *, id> *> *heldFrames = frames;
    RawFrameLoaderFn loader =
        [heldFrames](int index, Config& work, bool is_reference, int crop_h, int crop_w) {
            if (index < 0 || index >= (int)heldFrames.count) return Image();
            NSDictionary *frame = heldFrames[(NSUInteger)index];
            if (![frame isKindOfClass:NSDictionary.class]) return Image();
            return DecodeRawFrameDictionary(frame, work, is_reference, crop_h, crop_w);
        };

    Image preview;
    @autoreleasepool {
        preview = process_burst_loader_to_dng(
            (int)frames.count, loader, cfg, std::string(outPath.UTF8String), cb, 256);
    }

    if (preview.w <= 0) return NO;

    if (previewOut) *previewOut = UIImageFromPreview(preview);
    return YES;
}

+ (void)prewarmFFTWidth:(NSInteger)width height:(NSInteger)height {
    hhsr::mps_fft_prewarm((int)height, (int)width);
}

+ (BOOL)exportJPEGFromLinearDNG:(NSString *)dngPath toPath:(NSString *)jpgPath {
    if (dngPath.length == 0 || jpgPath.length == 0) return NO;

    std::vector<uint16_t> rgb;
    int W = 0, H = 0;
    float wb[3] = {1.f, 1.f, 1.f};
    float m[9] = {1,0,0, 0,1,0, 0,0,1};
    bool has_color = false;
    if (!load_linear_dng_rgb16_color(std::string(dngPath.UTF8String), rgb, W, H, wb, m, has_color) ||
        W <= 0 || H <= 0)
        return NO;

    // One analysis pass over the whole image before any pixel is rendered: the
    // ISP needs a global view for automatic exposure and the local gain map.
    // Before isp_analyse, so the automatic exposure and the local gain map are
    // derived from the cleaned image rather than from the noise.
    if (g_isp.enabled)
        hhsr::isp_denoise_chroma(rgb.data(), W, H, g_isp);
    hhsr::IspState isp;
    const bool use_isp = g_isp.enabled &&
                         hhsr::isp_analyse(rgb.data(), W, H, has_color ? m : nullptr, g_isp, isp);

    // Row-parallel: 48MP through a scalar loop on one core was the difference
    // between a JPEG appearing at once and taking seconds. Rows are independent
    // -- the gain map is read-only by now -- so this needs no synchronisation.
    // RGB, not RGBA. The alpha byte was 255 everywhere and JPEG has no use for
    // it, so at 48MP it cost 48MB of allocation and a quarter of the store
    // traffic for nothing.
    std::vector<uint8_t> srgb((size_t)W * (size_t)H * 3);
    hhsr::parallel_rows(H, 0, [&](int y) {
        const size_t row = (size_t)y * (size_t)W;
        for (int x = 0; x < W; ++x) {
            const size_t i = row + (size_t)x;
            float r = rgb[i * 3 + 0] * (1.f / 65535.f);
            float g = rgb[i * 3 + 1] * (1.f / 65535.f);
            float b = rgb[i * 3 + 2] * (1.f / 65535.f);
            float sr, sg, sb;
            if (use_isp)
                hhsr::isp_render(isp, r, g, b, x, y, sr, sg, sb);
            else
                render_linear_dng_pixel(r, g, b, wb, m, has_color, sr, sg, sb);
            srgb[i * 3 + 0] = (uint8_t)std::lround(sr * 255.f);
            srgb[i * 3 + 1] = (uint8_t)std::lround(sg * 255.f);
            srgb[i * 3 + 2] = (uint8_t)std::lround(sb * 255.f);
        }
    });
    rgb.clear();
    rgb.shrink_to_fit();

    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (!cs) cs = CGColorSpaceCreateDeviceRGB();
    if (!cs) return NO;
    NSData* data = [NSData dataWithBytes:srgb.data() length:srgb.size()];
    srgb.clear();
    srgb.shrink_to_fit();

    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef cgOut = CGImageCreate(
        W, H, 8, 24, W * 3, cs,
        kCGBitmapByteOrderDefault | kCGImageAlphaNone,
        provider, NULL, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(cs);
    if (!cgOut) return NO;

    NSURL* url = [NSURL fileURLWithPath:jpgPath];
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL(
        (__bridge CFURLRef)url, CFSTR("public.jpeg"), 1, NULL);
    if (!dest) {
        CGImageRelease(cgOut);
        return NO;
    }
    // 0.92 keeps 4:4:4 chroma and produced ~15MB at 48MP. Measured against that
    // output, 0.82 lands at 46% of the size for 2.8 LSB RMS -- ImageIO drops to
    // 4:2:0 below ~0.90, which is where most of the saving comes from, and
    // chroma subsampling is not visible on a photograph at this resolution.
    NSDictionary* opts = @{(__bridge NSString*)kCGImageDestinationLossyCompressionQuality: @0.82};
    CGImageDestinationAddImage(dest, cgOut, (__bridge CFDictionaryRef)opts);
    BOOL ok = CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(cgOut);
    return ok;
}

+ (BOOL)embedJPEGPreviewInDNG:(NSString *)dngPath maxSide:(NSInteger)maxSide {
    if (dngPath.length == 0) return NO;
    if (maxSide < 256) maxSide = 256;

    std::vector<uint16_t> rgb;
    int W = 0, H = 0;
    float wb[3] = {1.f, 1.f, 1.f};
    float m[9] = {1,0,0, 0,1,0, 0,0,1};
    bool has_color = false;
    if (!load_linear_dng_rgb16_color(std::string(dngPath.UTF8String), rgb, W, H, wb, m, has_color) ||
        W <= 0 || H <= 0)
        return NO;

    const int long_side = std::max(W, H);
    const float scale = (long_side > (int)maxSide)
        ? ((float)maxSide / (float)long_side) : 1.f;
    const int ow = std::max(1, (int)std::lround(W * scale));
    const int oh = std::max(1, (int)std::lround(H * scale));

    // Analysed at full resolution even though the preview is downscaled, so the
    // thumbnail and the exported JPEG get the same exposure and gain map and
    // cannot disagree about how the shot looks.
    // Before isp_analyse, so the automatic exposure and the local gain map are
    // derived from the cleaned image rather than from the noise.
    if (g_isp.enabled)
        hhsr::isp_denoise_chroma(rgb.data(), W, H, g_isp);
    hhsr::IspState isp;
    const bool use_isp = g_isp.enabled &&
                         hhsr::isp_analyse(rgb.data(), W, H, has_color ? m : nullptr, g_isp, isp);

    std::vector<uint8_t> srgb((size_t)ow * (size_t)oh * 4);
    auto sample_tonemap = [&](int sx, int sy, float& sr, float& sg, float& sb) {
        sx = std::max(0, std::min(W - 1, sx));
        sy = std::max(0, std::min(H - 1, sy));
        size_t i = (size_t)sy * (size_t)W + (size_t)sx;
        float r = rgb[i * 3 + 0] * (1.f / 65535.f);
        float g = rgb[i * 3 + 1] * (1.f / 65535.f);
        float b = rgb[i * 3 + 2] * (1.f / 65535.f);
        if (use_isp) hhsr::isp_render(isp, r, g, b, sx, sy, sr, sg, sb);
        else         render_linear_dng_pixel(r, g, b, wb, m, has_color, sr, sg, sb);
    };

    hhsr::parallel_rows(oh, 0, [&](int y) {
        int sy = (scale < 1.f) ? (int)((y + 0.5f) / scale) : y;
        for (int x = 0; x < ow; ++x) {
            int sx = (scale < 1.f) ? (int)((x + 0.5f) / scale) : x;
            float sr, sg, sb;
            sample_tonemap(sx, sy, sr, sg, sb);
            size_t o = ((size_t)y * (size_t)ow + (size_t)x) * 4;
            srgb[o + 0] = (uint8_t)std::lround(sr * 255.f);
            srgb[o + 1] = (uint8_t)std::lround(sg * 255.f);
            srgb[o + 2] = (uint8_t)std::lround(sb * 255.f);
            srgb[o + 3] = 255;
        }
    });
    rgb.clear();
    rgb.shrink_to_fit();

    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (!cs) cs = CGColorSpaceCreateDeviceRGB();
    if (!cs) return NO;
    NSData* data = [NSData dataWithBytes:srgb.data() length:srgb.size()];
    srgb.clear();
    srgb.shrink_to_fit();

    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef cgOut = CGImageCreate(
        ow, oh, 8, 32, ow * 4, cs,
        kCGBitmapByteOrder32Big | kCGImageAlphaNoneSkipLast,
        provider, NULL, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(cs);
    if (!cgOut) return NO;

    NSMutableData* jpegData = [NSMutableData data];
    CGImageDestinationRef dest = CGImageDestinationCreateWithData(
        (__bridge CFMutableDataRef)jpegData, CFSTR("public.jpeg"), 1, NULL);
    if (!dest) {
        CGImageRelease(cgOut);
        return NO;
    }
    // Embedded DNG preview: a thumbnail source, so it can be leaner still.
    NSDictionary* opts = @{(__bridge NSString*)kCGImageDestinationLossyCompressionQuality: @0.80};
    CGImageDestinationAddImage(dest, cgOut, (__bridge CFDictionaryRef)opts);
    BOOL enc_ok = CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(cgOut);
    if (!enc_ok || jpegData.length < 4) return NO;

    return embed_dng_jpeg_preview(std::string(dngPath.UTF8String),
                                  (const uint8_t*)jpegData.bytes,
                                  (size_t)jpegData.length,
                                  ow, oh) ? YES : NO;
}

@end
