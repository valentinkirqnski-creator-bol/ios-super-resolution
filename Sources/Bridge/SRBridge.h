#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Thin Objective-C facade over the portable C++ Handheld MFSR core.
// Swift talks to this; this talks to hhsr::process_burst_to_dng.
@interface SRBridge : NSObject

// Decodes the given DNG burst (first == reference), runs the algorithm and
// writes a fully-developed 2x (e.g. 48 MP) DNG/image to `outPath`.
// `progress` is invoked on an arbitrary thread with (stageName, 0..1).
// Returns YES on success.
+ (BOOL)processDNGs:(NSArray<NSString *> *)paths
             toPath:(NSString *)outPath
              scale:(float)scale
           progress:(nullable void (^)(NSString *stage, float fraction))progress;

@end

NS_ASSUME_NONNULL_END
