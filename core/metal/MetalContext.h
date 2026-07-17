#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "../types.h"
#include "../pipeline.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
// Forward declare for C++ to hold without knowing about Objective-C
struct id;
typedef id MTLDevice_t;
typedef id MTLCommandQueue_t;
typedef id MTLBuffer_t;
#endif

namespace hhsr {

// Wrapper around a Metal buffer to map C++ Image memory to GPU memory
struct MetalBuffer {
    int size_bytes;
#ifdef __OBJC__
    id<MTLBuffer> buffer;
#else
    void* buffer;
#endif

    MetalBuffer() : size_bytes(0), buffer(nullptr) {}
};

// Orchestrator for the Metal backend
class MetalContext {
public:
    static MetalContext& instance();

    bool is_available() const;
    void init();

    // Create a Metal buffer from an existing Image (copies data to GPU)
    MetalBuffer create_buffer(const Image& img);
    
    // Create an empty Metal buffer of a specific size
    MetalBuffer create_buffer(int size_bytes);
    
    // Read back a Metal buffer into a C++ Image
    void read_buffer(const MetalBuffer& mtl_buf, Image& img);

    // Synchronize CPU with GPU (waits for all queued work)
    void sync();

#ifdef __OBJC__
    id<MTLDevice> device() const { return _device; }
    id<MTLCommandQueue> command_queue() const { return _command_queue; }
    id<MTLComputePipelineState> get_pipeline_state(const std::string& name);
    
    // Texture helpers
    id<MTLTexture> create_texture(const Image& img);
    id<MTLTexture> create_empty_texture(int w, int h, int channels = 1, bool is_private = false);
    id<MTLTexture> create_texture_from_flow(const FlowField& f);
    void read_texture(id<MTLTexture> tex, Image& img);
    void read_flow_texture(id<MTLTexture> tex, FlowField& f);
    bool validate_pipelines(const std::vector<std::string>& names);
#endif

private:
    MetalContext();
    ~MetalContext() = default;

#ifdef __OBJC__
    id<MTLDevice> _device;
    id<MTLCommandQueue> _command_queue;
    id<MTLLibrary> _library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> _pipelines;
#else
    void* _device;
    void* _command_queue;
#endif
    bool _initialized;
    bool _available;
};

// Disk-backed Metal path (streaming frames from disk). Returns true on success.
bool try_process_burst_paths_metal(const std::vector<std::string>& paths, const Config& cfg,
                                   const std::string& dng_path, const ProgressFn& progress,
                                   int maxPreviewDim, Image& out_preview);

// In-memory Metal path (called from pipeline.cpp).
Image process_burst_metal(const std::vector<Image>& burst, const Config& cfg,
                          const std::string& dng_path, const ProgressFn& progress);

} // namespace hhsr
