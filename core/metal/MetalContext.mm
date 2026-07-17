#include "MetalContext.h"
#include <iostream>

namespace hhsr {

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext() : _device(nullptr), _command_queue(nullptr), _initialized(false), _available(false) {
}

bool MetalContext::is_available() const {
    return _available;
}

void MetalContext::init() {
    if (_initialized) return;
    _initialized = true;

#ifdef __OBJC__
    _device = MTLCreateSystemDefaultDevice();
    if (_device) {
        _command_queue = [_device commandQueue];
        _library = [_device newDefaultLibrary];
        if (!_library) {
            std::cerr << "Warning: Could not load default Metal library. Make sure .metal files are compiled into the app." << std::endl;
        }
        _available = true;
        std::cout << "Metal backend initialized. Device: " << [[_device name] UTF8String] << std::endl;
    } else {
        std::cerr << "Failed to initialize Metal backend." << std::endl;
    }
#endif
}

#ifdef __OBJC__
id<MTLComputePipelineState> MetalContext::get_pipeline_state(const std::string& name) {
    if (!_available || !_library) return nil;
    if (_pipelines.count(name)) return _pipelines[name];

    NSString* ns_name = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> func = [_library newFunctionWithName:ns_name];
    if (!func) {
        std::cerr << "Metal Error: Could not find function " << name << std::endl;
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pso = [_device newComputePipelineStateWithFunction:func error:&error];
    if (!pso) {
        std::cerr << "Metal Error: Failed to create pipeline state for " << name << ": " 
                  << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }

    _pipelines[name] = pso;
    return pso;
}
#endif

MetalBuffer MetalContext::create_buffer(const Image& img) {
    MetalBuffer buf;
    if (!_available) return buf;

    buf.size_bytes = img.data.size() * sizeof(f32);
#ifdef __OBJC__
    buf.buffer = [_device newBufferWithBytes:img.data.data()
                                      length:buf.size_bytes
                                     options:MTLResourceStorageModeShared];
#endif
    return buf;
}

MetalBuffer MetalContext::create_buffer(int size_bytes) {
    MetalBuffer buf;
    if (!_available) return buf;

    buf.size_bytes = size_bytes;
#ifdef __OBJC__
    buf.buffer = [_device newBufferWithLength:size_bytes
                                      options:MTLResourceStorageModeShared];
#endif
    return buf;
}

void MetalContext::read_buffer(const MetalBuffer& mtl_buf, Image& img) {
    if (!_available || !mtl_buf.buffer) return;

#ifdef __OBJC__
    if (img.data.size() * sizeof(f32) != mtl_buf.size_bytes) {
        std::cerr << "Size mismatch in read_buffer" << std::endl;
        return;
    }
    memcpy(img.data.data(), [mtl_buf.buffer contents], mtl_buf.size_bytes);
#endif
}

#ifdef __OBJC__
id<MTLTexture> MetalContext::create_texture(const Image& img) {
    if (!_available) return nil;
    MTLPixelFormat format = MTLPixelFormatR32Float;
    int bytes_per_pixel = sizeof(float);
    if (img.c == 2) {
        format = MTLPixelFormatRG32Float;
        bytes_per_pixel = 2 * sizeof(float);
    } else if (img.c == 3 || img.c == 4) {
        format = MTLPixelFormatRGBA32Float;
        bytes_per_pixel = 4 * sizeof(float);
    }

    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                    width:img.w
                                                                                   height:img.h
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModeShared;
    
    id<MTLTexture> tex = [_device newTextureWithDescriptor:desc];
    
    std::vector<float> padded_data;
    const float* src_data = img.data.data();
    if (img.c == 3) {
        padded_data.resize(img.w * img.h * 4, 0.0f);
        for (int i = 0; i < img.w * img.h; ++i) {
            padded_data[i*4 + 0] = src_data[i*3 + 0];
            padded_data[i*4 + 1] = src_data[i*3 + 1];
            padded_data[i*4 + 2] = src_data[i*3 + 2];
            padded_data[i*4 + 3] = 1.0f;
        }
        src_data = padded_data.data();
    }
    
    MTLRegion region = MTLRegionMake2D(0, 0, img.w, img.h);
    [tex replaceRegion:region mipmapLevel:0 withBytes:src_data bytesPerRow:img.w * bytes_per_pixel];
    
    return tex;
}

void MetalContext::read_texture(id<MTLTexture> tex, Image& img) {
    if (!_available || !tex) return;
    
    int channels = 1;
    int bytes_per_pixel = sizeof(float);
    if (tex.pixelFormat == MTLPixelFormatRG32Float) {
        channels = 2;
        bytes_per_pixel = 2 * sizeof(float);
    } else if (tex.pixelFormat == MTLPixelFormatRGBA32Float) {
        channels = 4; // We might want 3, but we'll read 4 and discard alpha
        bytes_per_pixel = 4 * sizeof(float);
    }
    
    if (img.w != tex.width || img.h != tex.height || (img.c != channels && !(img.c == 3 && channels == 4))) {
        img = Image(tex.height, tex.width, channels);
    }
    
    std::vector<float> buf(tex.width * tex.height * (channels == 4 ? 4 : channels));
    MTLRegion region = MTLRegionMake2D(0, 0, tex.width, tex.height);
    [tex getBytes:buf.data() bytesPerRow:tex.width * bytes_per_pixel fromRegion:region mipmapLevel:0];
    
    if (img.c == 3 && channels == 4) {
        for (int i = 0; i < tex.width * tex.height; ++i) {
            img.data[i*3 + 0] = buf[i*4 + 0];
            img.data[i*3 + 1] = buf[i*4 + 1];
            img.data[i*3 + 2] = buf[i*4 + 2];
        }
    } else {
        memcpy(img.data.data(), buf.data(), buf.size() * sizeof(float));
    }
}

id<MTLTexture> MetalContext::create_empty_texture(int w, int h, int channels) {
    if (!_available) return nil;
    MTLPixelFormat format = MTLPixelFormatR32Float;
    if (channels == 2) format = MTLPixelFormatRG32Float;
    else if (channels >= 3) format = MTLPixelFormatRGBA32Float;
    
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModeShared;
    return [_device newTextureWithDescriptor:desc];
}
#endif

} // namespace hhsr
