#import "SRBridge.h"
#import <UIKit/UIKit.h>

#include <string>
#include <vector>

#include "core/types.h"
#include "core/pipeline.h"

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

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
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

    std::vector<std::string> vpaths;
    vpaths.reserve(paths.count);
    for (NSString *p in paths) vpaths.emplace_back(p.UTF8String);

    Config cfg;
    cfg.scale = scale;
    cfg.input_crop_factor = std::max(1, cropFactor);
    cfg.bayer_mode = true;
    cfg.bake_srgb = false;   // linear camera RGB in DNG; WB applied only for in-app preview
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

@end
