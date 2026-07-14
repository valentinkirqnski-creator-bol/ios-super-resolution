#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

// Thin Objective-C facade over the portable C++ Handheld MFSR core.
@interface SRBridge : NSObject

// Decodes the given DNG burst paths from disk (first == reference), runs the
// low-memory pipeline and writes a 2x (e.g. 48 MP) DNG to `outPath`.
// `progress` is invoked on an arbitrary thread with (stageName, 0..1).
// On success, optionally returns a small sRGB preview UIImage (not the full DNG).
+ (BOOL)processDNGs:(NSArray<NSString *> *)paths
             toPath:(NSString *)outPath
              scale:(float)scale
         cropFactor:(int)cropFactor
       tuningParams:(NSDictionary<NSString *, NSNumber *> *)tuning
           progress:(nullable void (^)(NSString *stage, float fraction))progress
        previewImage:(UIImage * _Nullable * _Nullable)previewOut;

@end

NS_ASSUME_NONNULL_END
