// srsr_cpu_shim.h -- just enough of the Metal Shading Language, as working C++,
// to RUN SRSRKernels.metal on the CPU (PC test harness only; not in the iOS
// target). Vectors, matrices, textures with the device's storage quantisation
// (RGBA16Float -> half, R8Unorm -> 1/255) and clamp-to-edge bilinear sampling.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// Global, as in MSL (a same-type typedef may repeat one from sys/types.h).
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

namespace metal {

struct int2; struct uint2; struct float3; struct float4; struct uint4; struct uint3;

struct float2 {
    float x = 0, y = 0;
    float2() = default;
    explicit float2(float s) : x(s), y(s) {}
    float2(float a, float b) : x(a), y(b) {}
    explicit float2(const int2& v);
    explicit float2(const uint2& v);
};
struct float3 {
    float x = 0, y = 0, z = 0;
    float3() = default;
    explicit float3(float s) : x(s), y(s), z(s) {}
    float3(float a, float b, float c) : x(a), y(b), z(c) {}
};
struct float4 {
    float x = 0, y = 0, z = 0, w = 0;
    float4() = default;
    explicit float4(float s) : x(s), y(s), z(s), w(s) {}
    float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    float4(const float3& v, float d) : x(v.x), y(v.y), z(v.z), w(d) {}
    explicit float4(const uint4& v);
};
struct int2 {
    int x = 0, y = 0;
    int2() = default;
    int2(int a, int b) : x(a), y(b) {}
};
struct uint2 {
    uint x = 0, y = 0;
    uint2() = default;
    uint2(uint a, uint b) : x(a), y(b) {}
    explicit uint2(const int2& v) : x((uint)v.x), y((uint)v.y) {}
};
struct uint4 {
    uint x = 0, y = 0, z = 0, w = 0;
    uint4() = default;
    uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {}
};
inline float2::float2(const int2& v) : x((float)v.x), y((float)v.y) {}
inline float2::float2(const uint2& v) : x((float)v.x), y((float)v.y) {}
inline float4::float4(const uint4& v) : x((float)v.x), y((float)v.y), z((float)v.z), w((float)v.w) {}

inline int2 operator+(int2 a, int2 b) { return int2(a.x + b.x, a.y + b.y); }

#define SRSR_V2(OP) \
    inline float2 operator OP(float2 a, float2 b) { return float2(a.x OP b.x, a.y OP b.y); } \
    inline float2 operator OP(float2 a, float s) { return float2(a.x OP s, a.y OP s); } \
    inline float2 operator OP(float s, float2 a) { return float2(s OP a.x, s OP a.y); } \
    inline float2& operator OP##=(float2& a, float2 b) { a = a OP b; return a; } \
    inline float2& operator OP##=(float2& a, float s) { a = a OP s; return a; }
#define SRSR_V3(OP) \
    inline float3 operator OP(float3 a, float3 b) { return float3(a.x OP b.x, a.y OP b.y, a.z OP b.z); } \
    inline float3 operator OP(float3 a, float s) { return float3(a.x OP s, a.y OP s, a.z OP s); } \
    inline float3 operator OP(float s, float3 a) { return float3(s OP a.x, s OP a.y, s OP a.z); } \
    inline float3& operator OP##=(float3& a, float3 b) { a = a OP b; return a; } \
    inline float3& operator OP##=(float3& a, float s) { a = a OP s; return a; }
#define SRSR_V4(OP) \
    inline float4 operator OP(float4 a, float4 b) { return float4(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w); } \
    inline float4 operator OP(float4 a, float s) { return float4(a.x OP s, a.y OP s, a.z OP s, a.w OP s); } \
    inline float4 operator OP(float s, float4 a) { return float4(s OP a.x, s OP a.y, s OP a.z, s OP a.w); } \
    inline float4& operator OP##=(float4& a, float4 b) { a = a OP b; return a; } \
    inline float4& operator OP##=(float4& a, float s) { a = a OP s; return a; }
SRSR_V2(+) SRSR_V2(-) SRSR_V2(*) SRSR_V2(/)
SRSR_V3(+) SRSR_V3(-) SRSR_V3(*) SRSR_V3(/)
SRSR_V4(+) SRSR_V4(-) SRSR_V4(*) SRSR_V4(/)
#undef SRSR_V2
#undef SRSR_V3
#undef SRSR_V4
inline float2 operator-(float2 a) { return float2(-a.x, -a.y); }
inline float3 operator-(float3 a) { return float3(-a.x, -a.y, -a.z); }
inline float4 operator-(float4 a) { return float4(-a.x, -a.y, -a.z, -a.w); }

// ---- scalar functions (Metal names)
inline float abs(float v) { return std::fabs(v); }
// abs(int) is the C library's ::abs (a second overload here would be ambiguous).
inline float min(float a, float b) { return std::fmin(a, b); }
inline float max(float a, float b) { return std::fmax(a, b); }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline float clamp(float v, float lo, float hi) { return std::fmin(std::fmax(v, lo), hi); }
inline int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float saturate(float v) { return clamp(v, 0.0f, 1.0f); }
inline float mix(float a, float b, float t) { return a + (b - a) * t; }
inline float floor(float v) { return std::floor(v); }
inline float fract(float v) { return v - std::floor(v); }
inline float sqrt(float v) { return std::sqrt(v); }
inline float exp2(float v) { return std::exp2(v); }
inline float log(float v) { return std::log(v); }
inline float rint(float v) { return std::rint(v); }   // ties to even, like Halide round()
inline float sign(float v) { return v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); }
inline float smoothstep(float e0, float e1, float x) {
    const float t = clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ---- vector functions
inline float2 floor(float2 v) { return float2(std::floor(v.x), std::floor(v.y)); }
inline float2 fract(float2 v) { return float2(fract(v.x), fract(v.y)); }
inline float2 min(float2 a, float2 b) { return float2(min(a.x, b.x), min(a.y, b.y)); }
inline float2 max(float2 a, float2 b) { return float2(max(a.x, b.x), max(a.y, b.y)); }
inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
inline float length(float2 v) { return std::sqrt(dot(v, v)); }
inline float2 normalize(float2 v) { const float l = 1.0f / std::sqrt(dot(v, v)); return v * l; }
inline float3 abs(float3 v) { return float3(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)); }
inline float3 max(float3 a, float3 b) { return float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
inline float3 min(float3 a, float3 b) { return float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
inline float3 clamp(float3 v, float3 lo, float3 hi) {
    return float3(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z));
}
inline float3 mix(float3 a, float3 b, float t) { return a + (b - a) * t; }
inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float4 max(float4 a, float4 b) { return float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); }
inline float4 min(float4 a, float4 b) { return float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)); }
inline float4 sqrt(float4 v) { return float4(std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z), std::sqrt(v.w)); }
inline float4 mix(float4 a, float4 b, float t) { return a + (b - a) * t; }
inline float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

inline float2 xy(float4 v) { return float2(v.x, v.y); }
inline float3 xyz(float4 v) { return float3(v.x, v.y, v.z); }

// ---- matrices (column-major, like Metal / GLSL)
struct float2x2 {
    float2 c[2];
    float2x2() = default;
    float2x2(float2 a, float2 b) { c[0] = a; c[1] = b; }
    float2& operator[](int i) { return c[i]; }
    const float2& operator[](int i) const { return c[i]; }
};
inline float2x2 transpose(const float2x2& m) {
    return float2x2(float2(m.c[0].x, m.c[1].x), float2(m.c[0].y, m.c[1].y));
}
inline float2 operator*(const float2x2& m, float2 v) { return m.c[0] * v.x + m.c[1] * v.y; }
inline float2x2 operator*(const float2x2& a, const float2x2& b) { return float2x2(a * b.c[0], a * b.c[1]); }

// ---- textures
namespace access { enum Access { sample, read, write, read_write }; }
namespace coord { enum Coord { normalized, pixel }; }
namespace address { enum Address { clamp_to_edge, repeat }; }
namespace filter { enum Filter { nearest, linear }; }

struct sampler {
    filter::Filter flt;
    constexpr sampler(coord::Coord, address::Address, filter::Filter f) : flt(f) {}
};

enum class CpuFmt { R8Unorm, R32Float, RGBA16Float, RGBA16Uint, RGBA32Float };

inline float half_round(float v) {
    // float -> IEEE half (round to nearest even) -> float, saturating like the GPU store.
    if (!(v == v)) return v;
    const float a = std::fabs(v);
    if (a >= 65520.0f) return std::copysign(65504.0f, v);
    if (a < 6.103515625e-05f) {                       // subnormal half: step 2^-24
        const float q = std::rint(a * 16777216.0f) / 16777216.0f;
        return std::copysign(q, v);
    }
    int e;
    std::frexp(a, &e);                                // a = m * 2^e, m in [0.5, 1)
    const float step = std::ldexp(1.0f, e - 11);      // 10 mantissa bits after the leading 1
    return std::copysign(std::rint(a / step) * step, v);
}

struct CpuTex {
    CpuFmt fmt = CpuFmt::R32Float;
    int w = 0, h = 0, nc = 1;
    std::vector<float> f;
    std::vector<uint16_t> u;
    CpuTex(CpuFmt fm, int w_, int h_) : fmt(fm), w(w_), h(h_) {
        nc = (fm == CpuFmt::R8Unorm || fm == CpuFmt::R32Float) ? 1 : 4;
        if (fm == CpuFmt::RGBA16Uint) u.assign((size_t)w * h * 4, 0);
        else f.assign((size_t)w * h * nc, 0.f);
    }
    float4 readf(int x, int y) const {
        const size_t i = ((size_t)y * w + x) * nc;
        if (nc == 1) return float4(f[i], 0.f, 0.f, 1.f);
        return float4(f[i], f[i + 1], f[i + 2], f[i + 3]);
    }
    float q(float v) const {
        if (fmt == CpuFmt::R8Unorm) return std::rint(clamp(v, 0.f, 1.f) * 255.f) / 255.f;
        if (fmt == CpuFmt::RGBA16Float) return half_round(v);
        return v;
    }
    void writef(float4 v, int x, int y) {
        const size_t i = ((size_t)y * w + x) * nc;
        if (nc == 1) { f[i] = q(v.x); return; }
        f[i] = q(v.x); f[i + 1] = q(v.y); f[i + 2] = q(v.z); f[i + 3] = q(v.w);
    }
};

template <class T, access::Access A = access::sample> struct texture2d;

template <access::Access A> struct texture2d<float, A> {
    CpuTex* t = nullptr;
    texture2d() = default;
    explicit texture2d(CpuTex* p) : t(p) {}
    uint get_width() const { return (uint)t->w; }
    uint get_height() const { return (uint)t->h; }
    float4 read(uint2 p) const { return t->readf((int)p.x, (int)p.y); }
    void write(float4 v, uint2 p) const { t->writef(v, (int)p.x, (int)p.y); }
    float4 sample(const sampler& s, float2 uv) const {
        const float fx = uv.x * (float)t->w - 0.5f, fy = uv.y * (float)t->h - 0.5f;
        if (s.flt == filter::nearest) {
            const int x = clamp((int)std::floor(uv.x * (float)t->w), 0, t->w - 1);
            const int y = clamp((int)std::floor(uv.y * (float)t->h), 0, t->h - 1);
            return t->readf(x, y);
        }
        const float bx = std::floor(fx), by = std::floor(fy);
        const float ax = fx - bx, ay = fy - by;
        const int x0 = clamp((int)bx, 0, t->w - 1), x1 = clamp((int)bx + 1, 0, t->w - 1);
        const int y0 = clamp((int)by, 0, t->h - 1), y1 = clamp((int)by + 1, 0, t->h - 1);
        const float4 a = t->readf(x0, y0), b = t->readf(x1, y0);
        const float4 c = t->readf(x0, y1), d = t->readf(x1, y1);
        return (a * (1.f - ax) + b * ax) * (1.f - ay) + (c * (1.f - ax) + d * ax) * ay;
    }
};

template <access::Access A> struct texture2d<uint, A> {
    CpuTex* t = nullptr;
    texture2d() = default;
    explicit texture2d(CpuTex* p) : t(p) {}
    uint get_width() const { return (uint)t->w; }
    uint get_height() const { return (uint)t->h; }
    uint4 read(uint2 p) const {
        const size_t i = ((size_t)p.y * t->w + p.x) * 4;
        return uint4(t->u[i], t->u[i + 1], t->u[i + 2], t->u[i + 3]);
    }
    void write(uint4 v, uint2 p) const {
        const size_t i = ((size_t)p.y * t->w + p.x) * 4;
        t->u[i] = (uint16_t)std::min<uint>(v.x, 65535u);
        t->u[i + 1] = (uint16_t)std::min<uint>(v.y, 65535u);
        t->u[i + 2] = (uint16_t)std::min<uint>(v.z, 65535u);
        t->u[i + 3] = (uint16_t)std::min<uint>(v.w, 65535u);
    }
};

template <class T, int N> struct array {
    T a[N];
    T& operator[](int i) { return a[i]; }
    const T& operator[](int i) const { return a[i]; }
};

} // namespace metal

using namespace metal;

// MSL keywords that have no meaning on the CPU.
#define kernel
#define thread
#define device
#define constant const
