#import "SRBridge.h"

#include <string>
#include <vector>

#include "core/types.h"
#include "core/raw_io.h"
#include "core/pipeline.h"

using namespace hhsr;

@implementation SRBridge

+ (BOOL)processDNGs:(NSArray<NSString *> *)paths
             toPath:(NSString *)outPath
              scale:(float)scale
           progress:(void (^)(NSString *, float))progress {
    if (paths.count < 2) return NO;

    std::vector<std::string> vpaths;
    vpaths.reserve(paths.count);
    for (NSString *p in paths) vpaths.emplace_back(p.UTF8String);

    Config cfg;
    cfg.scale = scale;          // 2.0 -> 12 MP burst yields a 48 MP result
    cfg.bayer_mode = true;
    cfg.bake_srgb = true;       // produce a display-ready image
    cfg.use_gpu = false;        // CPU path (Metal/Vulkan backend is a later step)
    cfg.num_threads = 0;        // use all cores

    std::vector<Image> burst = load_raw_burst(vpaths, cfg);
    if (burst.empty()) return NO;

    ProgressFn cb = nullptr;
    if (progress) {
        cb = [progress](const std::string &stage, float f) {
            @autoreleasepool {
                progress([NSString stringWithUTF8String:stage.c_str()], f);
            }
        };
    }

    Image preview = process_burst_to_dng(burst, cfg, std::string(outPath.UTF8String), cb);
    return preview.w > 0;
}

@end
