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
           cropZoom:(float)cropZoom
       tuningParams:(NSDictionary<NSString *, NSNumber *> *)tuning
           progress:(nullable void (^)(NSString *stage, float fraction))progress
        previewImage:(UIImage * _Nullable * _Nullable)previewOut;

// Same pipeline/output as processDNGs, but frames are already captured RAW Bayer
// buffers. Each frame dictionary contains: data or path, width, height,
// bytesPerRow, cfa ([R,G,G,B] color indices), and metadata.
+ (BOOL)processRawFrames:(NSArray<NSDictionary<NSString *, id> *> *)frames
                  toPath:(NSString *)outPath
                   scale:(float)scale
                cropZoom:(float)cropZoom
            tuningParams:(NSDictionary<NSString *, NSNumber *> *)tuning
                progress:(nullable void (^)(NSString *stage, float fraction))progress
             previewImage:(UIImage * _Nullable * _Nullable)previewOut;

/// Decode our LinearRaw Deflate DNG and write a calibrated sRGB JPEG.
/// Uses the same renderer as the embedded DNG preview. No CI sharpen or NR.
+ (BOOL)exportJPEGFromLinearDNG:(NSString *)dngPath
                         toPath:(NSString *)jpgPath;

/// Render the LinearRaw DNG (same look as export JPEG), encode a JPEG preview
/// (longest side ≤ maxSide), and embed it as a DNG SubIFD so Photos can thumbnail
/// a DNG-only asset. Lightroom still reads the LinearRaw IFD0.
// Build the MPSGraph FFT plan for this sensor size ahead of the shutter.
// MPSGraph compiles on first use (~1100ms at 12MP) and that would otherwise land
// on the reference frame of the first burst. Safe to call repeatedly.
+ (void)prewarmFFTWidth:(NSInteger)width height:(NSInteger)height;

+ (BOOL)embedJPEGPreviewInDNG:(NSString *)dngPath
                      maxSide:(NSInteger)maxSide;

@end

NS_ASSUME_NONNULL_END
