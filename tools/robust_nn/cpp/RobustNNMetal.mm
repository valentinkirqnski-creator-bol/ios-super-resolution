#import "RobustNNMetal.h"

@implementation RobustNNMetal {
    id<MTLComputePipelineState> _warp, _features, _infer;
}

static id<MTLComputePipelineState> MakePSO(id<MTLDevice> d, id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    NSCAssert(fn, @"robust_nn: missing kernel %@", name);
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:fn error:&err];
    NSCAssert(p, @"robust_nn: %@", err);
    return p;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device library:(id<MTLLibrary>)library {
    if ((self = [super init])) {
        _warp = MakePSO(device, library, @"rnn_warp");
        _features = MakePSO(device, library, @"rnn_features");
        _infer = MakePSO(device, library, @"rnn_infer");
    }
    return self;
}

static void Dispatch(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, int W, int H) {
    NSUInteger tw = p.threadExecutionWidth;
    NSUInteger th = MAX((NSUInteger)1, p.maxTotalThreadsPerThreadgroup / tw);
    [e setComputePipelineState:p];
    [e dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

- (void)encodeInto:(id<MTLCommandBuffer>)cb
               ref:(id<MTLBuffer>)ref
               cmp:(id<MTLBuffer>)cmp
              flow:(id<MTLBuffer>)flowXY
                r0:(id<MTLBuffer>)r0
              rOut:(id<MTLBuffer>)rOut
          scratchJ:(id<MTLBuffer>)scratchJ
       scratchFeat:(id<MTLBuffer>)scratchFeat
            params:(RNNParams)P {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    e.label = @"robust_nn";

    [e setBuffer:cmp offset:0 atIndex:0];
    [e setBuffer:flowXY offset:0 atIndex:1];
    [e setBuffer:scratchJ offset:0 atIndex:2];
    [e setBytes:&P length:sizeof(P) atIndex:3];
    Dispatch(e, _warp, P.width, P.height);
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [e setBuffer:ref offset:0 atIndex:0];
    [e setBuffer:scratchJ offset:0 atIndex:1];
    [e setBuffer:flowXY offset:0 atIndex:2];
    [e setBuffer:r0 offset:0 atIndex:3];
    [e setBuffer:scratchFeat offset:0 atIndex:4];
    [e setBytes:&P length:sizeof(P) atIndex:5];
    Dispatch(e, _features, P.width, P.height);
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [e setBuffer:scratchFeat offset:0 atIndex:0];
    [e setBuffer:r0 offset:0 atIndex:1];
    [e setBuffer:rOut offset:0 atIndex:2];
    [e setBytes:&P length:sizeof(P) atIndex:3];
    Dispatch(e, _infer, P.width, P.height);
    [e endEncoding];
}

@end
