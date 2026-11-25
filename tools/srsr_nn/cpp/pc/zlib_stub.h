// Minimal zlib stub for systems where zlib.h is not available
// This provides just enough declarations so code can compile.
// Deflate/Inflate compression won't work, but lossless JPEG will.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define Z_OK 0
#define Z_STREAM_END 1
#define Z_NEED_DICT 2
#define Z_DATA_ERROR -3
#define Z_MEM_ERROR -4
#define Z_BUF_ERROR -5
#define Z_VERSION_ERROR -6

#define Z_NO_FLUSH 0
#define Z_PARTIAL_FLUSH 1
#define Z_SYNC_FLUSH 2
#define Z_FULL_FLUSH 3
#define Z_FINISH 4

#define Z_DEFLATED 8
#define Z_DEFAULT_COMPRESSION -1
#define Z_DEFAULT_STRATEGY 0

#define ZLIB_VERSION "1.2.11"

// Type definitions
typedef uint8_t Bytef;
typedef uint32_t uInt;
typedef uint32_t uLong;
typedef void* voidpf;

typedef struct z_stream_s {
    const Bytef* next_in;
    uInt avail_in;
    uLong total_in;
    Bytef* next_out;
    uInt avail_out;
    uLong total_out;
    const char* msg;
    struct z_stream_s* state;
    void* (*zalloc)(voidpf opaque, unsigned items, unsigned size);
    void (*zfree)(voidpf opaque, voidpf address);
    voidpf opaque;
    int data_type;
    uLong adler;
    uLong reserved;
} z_stream;

typedef z_stream* z_streamp;

// Stub functions that return error
inline int deflateInit2_(z_streamp strm, int level, int method,
                        int wbits, int memLevel, int strategy,
                        const char *version, int stream_size) {
    if (strm) strm->msg = (char*)"deflate not available (stub zlib)";
    return Z_VERSION_ERROR;
}

inline int deflate(z_streamp strm, int flush) {
    if (strm) strm->msg = (char*)"deflate not available (stub zlib)";
    return Z_BUF_ERROR;
}

inline int deflateEnd(z_streamp strm) {
    return Z_OK;
}

inline int inflateInit_(z_streamp strm, const char* version, int stream_size) {
    if (strm) strm->msg = (char*)"inflate not available (stub zlib)";
    return Z_VERSION_ERROR;
}

inline int inflate(z_streamp strm, int flush) {
    if (strm) strm->msg = (char*)"inflate not available (stub zlib)";
    return Z_DATA_ERROR;
}

inline int inflateEnd(z_streamp strm) {
    return Z_OK;
}

inline uLong crc32(uLong crc, const Bytef* buf, uInt len) {
    return 0;
}

// Inline deflateInit that calls deflateInit2_ with default parameters
inline int deflateInit(z_streamp strm, int level) {
    return deflateInit2_(strm, level, Z_DEFLATED, 15, 8, Z_DEFAULT_STRATEGY,
                        ZLIB_VERSION, (int)sizeof(z_stream));
}

#define deflateInit2(strm, level, method, wbits, memLevel, strategy) \
    deflateInit2_((strm), (level), (method), (wbits), (memLevel), (strategy), \
                  ZLIB_VERSION, (int)sizeof(z_stream))

#define inflateInit(strm) \
    inflateInit_((strm), ZLIB_VERSION, (int)sizeof(z_stream))

#ifdef __cplusplus
}
#endif
