// RobustNNMetal — encodes the three robustness-refinement passes for one
// comparison frame into an existing command buffer. Buffers are grey-res,
// float32, row-major (see robust_nn_shared.h for the input contract).
#pragma once
#import <Metal/Metal.h>

#include "robust_nn_shared.h"

@interface RobustNNMetal : NSObject
// library must contain robust_nn.metal (compiled with fast math OFF).
- (instancetype)initWithDevice:(id<MTLDevice>)device library:(id<MTLLibrary>)library;

// Scratch: J >= W*H floats, feat >= RNN_NF*W*H floats. Reuse across frames.
- (void)encodeInto:(id<MTLCommandBuffer>)cb
               ref:(id<MTLBuffer>)ref
               cmp:(id<MTLBuffer>)cmp
              flow:(id<MTLBuffer>)flowXY
                r0:(id<MTLBuffer>)r0
              rOut:(id<MTLBuffer>)rOut
          scratchJ:(id<MTLBuffer>)scratchJ
       scratchFeat:(id<MTLBuffer>)scratchFeat
            params:(RNNParams)params;
@end
