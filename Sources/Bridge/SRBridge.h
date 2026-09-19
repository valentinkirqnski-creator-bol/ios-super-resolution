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

/// HDR finish (core/finish_hdr): decode the LinearRaw DNG in its own camera
/// space and write a tone-mapped sRGB JPEG -- shadows lifted, highlights
/// compressed, sensor-clipped highlights rendered neutral rather than pink, and
/// no sharpening of any kind. Independent of exportJPEGFromLinearDNG, which
/// keeps the older calibrated look. The JPEG is tagged with the DNG's own
/// orientation.
+ (BOOL)exportHDRJPEGFromLinearDNG:(NSString *)dngPath
                            toPath:(NSString *)jpgPath;

/// Render the LinearRaw DNG (same look as export JPEG), encode a JPEG preview
/// (longest side ≤ maxSide), and embed it as a DNG SubIFD so Photos can thumbnail
/// a DNG-only asset. Lightroom still reads the LinearRaw IFD0.
// Build the MPSGraph FFT plan for this sensor size ahead of the shutter.
// MPSGraph compiles on first use (~1100ms at 12MP) and that would otherwise land
// on the reference frame of the first burst. Safe to call repeatedly.
+ (void)prewarmFFTWidth:(NSInteger)width height:(NSInteger)height;

/// Create the Metal device, library and every compute pipeline state ahead of
/// the first capture. Idempotent; safe from any thread.
+ (void)prewarmGPU;

/// Start building the robustness mask's noise curves for a captured RAW frame,
/// on a background queue. They are a pure function of the frame's NoiseProfile,
/// white balance and CFA, and at high ISO they are the single largest cost in a
/// burst, so the earlier this runs the more of it the capture hides. Idempotent:
/// the pipeline asks for the same curves later and finds them built.
+ (void)prewarmNoiseCurvesForFrame:(NSDictionary *)frame
                      tuningParams:(NSDictionary<NSString *, NSNumber *> *)tuning;

// Renders the tone-mapped (ISP) preview and embeds it as the DNG's JPEG SubIFD
// so Apple Photos can thumbnail. Returns the rendered preview as a UIImage (nil
// on failure) so the app can show the EXACT same tone-mapped image in-app that
// Photos and the exported JPEG use -- the three then always match.
+ (UIImage *)embedJPEGPreviewInDNG:(NSString *)dngPath
                           maxSide:(NSInteger)maxSide;

@end

NS_ASSUME_NONNULL_END
