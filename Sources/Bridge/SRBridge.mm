#import "SRBridge.h"
#import <UIKit/UIKit.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>

#include <string>
#include <vector>
#include <cmath>

#include "core/types.h"
#include "core/pipeline.h"
#include "core/metal_gpu.h"
#include "core/dng_writer.h"

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

// Soft display S-curve: midtone contrast without the crunchy/halo look of
// luma-preserving smoothstep + CIHighlightShadowAdjust.
static inline float tone_s_curve(float v) {
    v = clampf(v, 0.f, 1.f);
    const float s = v * v * (3.f - 2.f * v);
    return clampf(v * 0.42f + s * 0.58f, 0.f, 1.f);
}

// Mild chroma lift in display space (keeps neutrals, less dull than flat gamma).
static inline void apply_vibrance_rgb(float& r, float& g, float& b, float amount) {
    const float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    const float mx = std::max(r, std::max(g, b));
    const float mn = std::min(r, std::min(g, b));
    const float sat = (mx > 1e-6f) ? (mx - mn) / mx : 0.f;
    // Vibrance-like: boost low-sat more than already-vivid colors.
    const float boost = 1.f + amount * (1.f - sat);
    r = clampf(y + (r - y) * boost, 0.f, 1.f);
    g = clampf(y + (g - y) * boost, 0.f, 1.f);
    b = clampf(y + (b - y) * boost, 0.f, 1.f);
}

@implementation SRBridge

+ (BOOL)processDNGs:(NSArray<NSString *> *)paths
             toPath:(NSString *)outPath
              scale:(float)scale
         cropFactor:(int)cropFactor
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
    cfg.input_crop_factor = std::max(1, cropFactor);
    cfg.bayer_mode = true;
    cfg.bake_srgb = false;   // linear camera RGB in DNG; WB applied only for in-app preview / JPEG
    cfg.use_gpu = false;
    cfg.num_threads = 0;     // all CPU cores during active processing

    if (tuning) {
        if (tuning[@"r_t"]) cfg.r_t = tuning[@"r_t"].floatValue;
        if (tuning[@"r_s1"]) cfg.r_s1 = tuning[@"r_s1"].floatValue;
        if (tuning[@"r_s2"]) cfg.r_s2 = tuning[@"r_s2"].floatValue;
        if (tuning[@"r_Mt"]) cfg.r_Mt = tuning[@"r_Mt"].floatValue;
        if (tuning[@"k_detail"]) cfg.k_detail = tuning[@"k_detail"].floatValue;
        if (tuning[@"k_denoise"]) cfg.k_denoise = tuning[@"k_denoise"].floatValue;
        if (tuning[@"k_stretch"]) cfg.k_stretch = tuning[@"k_stretch"].floatValue;
        if (tuning[@"snr_auto_tune"]) cfg.snr_auto_tune = tuning[@"snr_auto_tune"].boolValue;
        if (tuning[@"accumulated_robustness_denoiser_enabled"]) {
            cfg.accumulated_robustness_denoiser_enabled = tuning[@"accumulated_robustness_denoiser_enabled"].boolValue;
        }
        if (tuning[@"acc_rob_rad_max"]) cfg.acc_rob_rad_max = tuning[@"acc_rob_rad_max"].floatValue;
        if (tuning[@"acc_rob_max_multiplier"]) cfg.acc_rob_max_multiplier = tuning[@"acc_rob_max_multiplier"].floatValue;
        if (tuning[@"acc_rob_max_frame_count"]) cfg.acc_rob_max_frame_count = tuning[@"acc_rob_max_frame_count"].floatValue;
    }

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

    // Linear camera RGB → WB → cam→sRGB → sRGB gamma → soft S-curve → vibrance.
    // No CIHighlightShadowAdjust / ColorControls (those washed midtones and
    // added a crunchy, almost-sharpened look).
    std::vector<uint8_t> srgb((size_t)W * (size_t)H * 4);
    const size_t n = (size_t)W * (size_t)H;
    for (size_t i = 0; i < n; ++i) {
        float r = rgb[i * 3 + 0] * (1.f / 65535.f);
        float g = rgb[i * 3 + 1] * (1.f / 65535.f);
        float b = rgb[i * 3 + 2] * (1.f / 65535.f);
        float wr = r * wb[0], wg = g * wb[1], wb_ = b * wb[2];
        float sr, sg, sb;
        if (has_color) {
            sr = m[0] * wr + m[1] * wg + m[2] * wb_;
            sg = m[3] * wr + m[4] * wg + m[5] * wb_;
            sb = m[6] * wr + m[7] * wg + m[8] * wb_;
        } else {
            sr = wr; sg = wg; sb = wb_;
        }
        // Soft highlight rolloff before gamma (keeps sparkle without clipping harsh).
        auto rolloff = [](float v) {
            if (v <= 1.f) return clampf(v, 0.f, 1.f);
            return 1.f - 1.f / (1.f + (v - 1.f) * 2.5f);
        };
        sr = to_srgb_gamma(rolloff(sr));
        sg = to_srgb_gamma(rolloff(sg));
        sb = to_srgb_gamma(rolloff(sb));
        sr = tone_s_curve(sr);
        sg = tone_s_curve(sg);
        sb = tone_s_curve(sb);
        apply_vibrance_rgb(sr, sg, sb, 0.28f);
        srgb[i * 4 + 0] = (uint8_t)std::lround(sr * 255.f);
        srgb[i * 4 + 1] = (uint8_t)std::lround(sg * 255.f);
        srgb[i * 4 + 2] = (uint8_t)std::lround(sb * 255.f);
        srgb[i * 4 + 3] = 255;
    }
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
        W, H, 8, 32, W * 4, cs,
        kCGBitmapByteOrder32Big | kCGImageAlphaNoneSkipLast,
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
    NSDictionary* opts = @{(__bridge NSString*)kCGImageDestinationLossyCompressionQuality: @0.92};
    CGImageDestinationAddImage(dest, cgOut, (__bridge CFDictionaryRef)opts);
    BOOL ok = CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(cgOut);
    return ok;
}

@end
