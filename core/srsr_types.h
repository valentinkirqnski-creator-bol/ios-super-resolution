//===----------------------------------------------------------------------===//
// srsr_types.h -- structs shared by SRSRKernels.metal and the C++ host.
//
// SRSR = the GCam 6.1 SABRE pipeline (Wronski et al. 2019, as Google ships it)
// ported to Metal: Halide-reference alignment, then GUIDE, REJECTION, DILATE,
// COPY_MASK, the fused anisotropic RBF MERGE, and the 3-level bilateral
// denoiser. Origin: Documents/srsr (SRSRShaderTypes.h), adapted for this app:
//   * PREPARE reads the loader's normalized, pre-white-balanced float raw.
//   * MERGE writes horizontal output bands, at any output scale (1x / 2x).
//   * DENOISE works on bands with an explicit halo and encodes the final band
//     straight into DNG uint16 rows (no 14-bit GCam STAGE 5 container).
// Every field is a 4-byte scalar, so the layout is identical in Metal and C++.
//===----------------------------------------------------------------------===//
#ifndef HHSR_SRSR_TYPES_H
#define HHSR_SRSR_TYPES_H

#ifndef __METAL_VERSION__
#include <stdint.h>
typedef int32_t  SRSRInt;
typedef uint32_t SRSRUInt;
#else
typedef int  SRSRInt;
typedef uint SRSRUInt;
#endif

// Merge texture slots: 4 arrays of kSRSRMaxFrames + 8 singles = 72 of the 128
// texture arguments Apple family 4+ allows.
#define kSRSRMaxFrames        16
#define kSRSRNoiseLUTColumns  10
// Quad texture encoding: value = round(x * kSRSRQuadScale) in RGBA16Uint, with x
// the loader's pre-whitened, exposure-matched raw (>= 0, saturation at
// wb[c]/wb[G] * gain < 4). One code is 6.1e-5 of full scale -- a quarter of a
// 12-bit raw step.
#define kSRSRQuadScale        16383.75f

// STAGE 0 normalization, canonical R G1 G2 B order: s = q * gain + blackTimesGain.
typedef struct {
    float gain0, gain1, gain2, gain3;
    float blackTimesGain0, blackTimesGain1, blackTimesGain2, blackTimesGain3;
} SRSRFrameNormalization;

typedef struct {
    float flowScaleX, flowScaleY;       // SABRE flow_scale_offset.xy
    float flowOffsetX, flowOffsetY;     // SABRE flow_scale_offset.zw
    SRSRInt isBase;
    SRSRInt pad0, pad1, pad2;
} SRSRFrameGeometry;

// PREPARE: float raw buffer -> quad texture (canonical RGGB, kSRSRQuadScale codes) + S2 grey.
typedef struct {
    SRSRInt quadWidth, quadHeight;
    SRSRInt cropOriginX, cropOriginY;   // phase-aligned so the first quad is R G1 / G2 B
    SRSRInt rawWidth, rawHeight;        // extent of the float raw buffer (row-major, tight)
    SRSRInt perm0, perm1, perm2, perm3; // quad read order (0,0)(1,0)(0,1)(1,1) -> R G1 G2 B
    float   quadScale;                  // kSRSRQuadScale
    SRSRInt pad0;
    SRSRFrameNormalization norm;
} SRSRPrepareUniforms;

typedef struct { SRSRInt inWidth, inHeight, outWidth, outHeight; } SRSRDownsampleUniforms;
typedef struct { SRSRInt width, height, tileSize, tilesX, tilesY; } SRSRGradientUniforms;
typedef struct {
    SRSRInt width, height, tileSize, tilesX, tilesY, searchRadius, candidateCount, useL2;
} SRSRBlockMatchingUniforms;
typedef struct { SRSRInt width, height, tileSize, tilesX, tilesY; } SRSRICAUniforms;
typedef struct {
    SRSRInt width, height;
    SRSRInt sourceTilesX, sourceTilesY;
    SRSRInt targetTileSize, targetStride;
    SRSRInt targetTilesX, targetTilesY;
    float gridScaleRatio, flowScale, offsetXPx, offsetYPx;
} SRSRFlowTransferUniforms;
typedef struct { SRSRInt tilesX, tilesY; } SRSRFlowMedianUniforms;
typedef struct { SRSRInt tilesX, tilesY; float invGreyWidth, invGreyHeight; } SRSRFlowPackUniforms;

// STAGE 1 -- GUIDE (SABRE 1a..1d)
typedef struct {
    SRSRInt quadWidth, quadHeight;
    float greenDifferenceImpact;        // 10.0
    float shrinkedSigma;                //  8.0
    float stretchedSigma;               //  1.0
    float gradientClip;                 //  0.001
    float baseSigma;                    //  4.0
    float blurredSigma;                 //  1.0
    float gradientTransition;           // 142.857
    SRSRInt pad0;
    SRSRFrameNormalization norm;
} SRSRGuideUniforms;

// STAGE 2f -- UNBLOCKER (reconstructed; off by default)
typedef struct {
    SRSRInt quadWidth, quadHeight, outWidth, outHeight;
    float varianceScale, bias, maxWhite, pad0;
    SRSRFrameNormalization norm;
} SRSRUnblockerUniforms;

// STAGE 2 -- REJECTION (SABRE 2a..2c)
typedef struct {
    SRSRInt quadWidth, quadHeight;
    SRSRInt flowTilesX, flowTilesY;
    SRSRInt unblockerWidth, unblockerHeight;
    float flowScaleX, flowScaleY, flowOffsetX, flowOffsetY;
    float unblockerScaleX, unblockerScaleY;
    float distanceMultiplier;                       // 1.0
    float rejectionThreshold;                       // 0.0
    float extraMotionRobustnessBoost;               // 6.0
    float motionRobustnessBoostStdDevThreshold;     // 5.0
    float extraMotionRobustnessMotionThreshold;     // (2016/W) * 1e-4
    float unblockerReductionThreshold;              // (2016/W) * 9.999e-5
    SRSRInt unblockerEnabled;
    SRSRInt pad0;
} SRSRRejectionUniforms;

typedef struct {
    SRSRInt quadWidth, quadHeight, frameCount;
    float accumulatedWeightScale;       // 25.0
} SRSRAccumulateUniforms;

// STAGE 3 + 3.5 -- MERGE, all frames in one dispatch, one output band.
typedef struct {
    SRSRInt outWidth, outHeight;        // full output grid (scale x Bayer)
    SRSRInt bayerWidth, bayerHeight;    // Bayer extent: the kernel's distance unit
    SRSRInt quadWidth, quadHeight;
    SRSRInt frameCount, baseIndex;
    SRSRInt bandY0, bandRows;           // output rows [bandY0, bandY0 + bandRows) -> band rows 0..
    SRSRInt applyRejectionWeight;
    SRSRInt pad0;
    float coverageThreshold;            // 4.0
    float covarianceMultiplier;         // 0.3
    float accumulatedWeightScale;       // 25.0
    float kernelEpsilon;                // 0.00005
    float normalization;                // 1.0 (SABRE 3c note)
    float footprintLimit;               // 1.5 Bayer px
    float borderMinX, borderMinY;       // frame_border_padded, normalized
    float borderMaxX, borderMaxY;
    float dehomogenizeEpsilon;          // 1e-7
    float pad1;
} SRSRMergeUniforms;

// STAGE 4 -- DENOISE (one level, one band) and the DNG encode of the last level.
// Band textures hold global rows [texY0, texY0 + texRows). The pass computes
// rows [outY0, outY0 + outRows); every row it reads (outY0 - dilation ..
// outY0 + outRows - 1 + dilation, mirrored at the image border) is inside the
// band by construction.
typedef struct {
    SRSRInt width, height;              // full output image
    SRSRInt level;                      // 0, 1, 2
    SRSRInt isLastLevel;                // converts S3 -> S2 and encodes the DNG rows
    SRSRInt dilation;                   // output pixels: (1 << level) * dilationBase
    SRSRInt texY0, texRows;
    SRSRInt outY0, outRows;
    SRSRInt passthrough;                // 1 = no filtering, only the S2 -> DNG encode
    float weightTextureScale;           // 25.0
    float weightTextureBias;            // 1.0
    float denoisingFactor;              // 0.0625
    float outlierSigmaMultiplier;       // 0.125
    float rejectLumaRevertFactor;       // 0.4
    float globalLumaRevertFactor;       // 1.0
    float vignetteStrengthBoost;        // 0.0
    float vignetteSpatialScaleFactor;   // 1.0
    float yuvNoiseModelX, yuvNoiseModelY, yuvNoiseModelZ, yuvNoiseModelW;
    float outGainR, outGainG, outGainB; // DNG un-whitening gains (1 = keep white-balanced)
    float pad0;
} SRSRDenoiseUniforms;

// SABRE 10x2 noise LUT as a constant buffer (D4). Built per frame by Monte Carlo
// (srsr_pipeline.cpp build_noise_lut) from the DNG NoiseProfile through the
// frame's real chain (WB, clip, exposure-match gain, sqrt), per-sample, GCam style:
//   row0[i] = (varY, varCo, varCg, minStdDev)  rejection, at reference S1 luma i/9:
//             one quad's YCoCg variance (mean of base and current frame; the shader
//             doubles it) and E[guide std_dev] under noise (the std floor).
//   row1[i] = (greenDiff, tensorNoise, 0, 0)  guide, at the frame's own S1 luma i/9:
//             E[green_differences] and the structure tensor's noise power.
typedef struct {
    float row0[kSRSRNoiseLUTColumns][4];
    float row1[kSRSRNoiseLUTColumns][4];
} SRSRNoiseLUT;

#endif // HHSR_SRSR_TYPES_H
