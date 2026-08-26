// Metal kernels for HHSR grey-FFT + L2 BM + merge accumulate.
// 1D DFT matches grey_pyramid.cpp fft1d_pow2_inplace_ref + fft1d_bluestein
// (same bit-reversal, iterative twiddles w*=wlen, Bluestein scaling).
// Merge matches merge.cpp accumulate_comp / accumulate_ref (Alg. 4 / 11).

#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;

struct RawDecodeParams {
    uint h, w, stride_shorts, _pad0;
    float4 black;
    float4 denom;
    float4 wb;
};

kernel void raw16_to_float_bayer(device const ushort* raw [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 constant RawDecodeParams& p [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    uint site = ((gid.y & 1u) << 1u) | (gid.x & 1u);
    ushort rawv = raw[gid.y * p.stride_shorts + gid.x];
    float v = (float(rawv) - p.black[site]) / p.denom[site];
    v *= p.wb[site];
    if (!isfinite(v)) v = 0.f;
    // No upper clip -- see load_raw_dng in raw_io.cpp.
    out[gid.y * p.w + gid.x] = max(v, 0.f);
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cconj(float2 a) { return float2(a.x, -a.y); }

// Exact port of fft1d_pow2_inplace_ref — one thread per batch vector.
// Does NOT divide by n (C++ ref / vDSP zip also leave scaling to the caller).
kernel void fft1d_pow2_cpp(device float2* data [[buffer(0)]],
                           constant uint& n [[buffer(1)]],
                           constant uint& stride [[buffer(2)]],
                           constant uint& batch_count [[buffer(3)]],
                           constant int& inverse [[buffer(4)]],
                           uint batch [[thread_position_in_grid]]) {
    if (batch >= batch_count || n <= 1u) return;
    device float2* a = data + batch * stride;

    // Numerical Recipes bit-reversal (same as C++)
    uint j = 0;
    for (uint i = 1; i < n; ++i) {
        uint bit = n >> 1;
        for (; (j & bit) != 0u; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float2 tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    for (uint len = 2; len <= n; len <<= 1) {
        float ang = (inverse != 0 ? 2.f : -2.f) * PI / float(len);
        float2 wlen = float2(cos(ang), sin(ang));
        for (uint i = 0; i < n; i += len) {
            float2 w = float2(1.f, 0.f);
            uint half_n = len >> 1;
            for (uint k = 0; k < half_n; ++k) {
                float2 u = a[i + k];
                float2 v = cmul(a[i + k + half_n], w);
                a[i + k] = u + v;
                a[i + k + half_n] = u - v;
                w = cmul(w, wlen);
            }
        }
    }
}

// Parallel pow2 FFT (same math as fft1d_pow2_cpp): bit-reverse + butterfly stages.
// ---------------------------------------------------------------------------
// Mixed-radix Stockham FFT.
//
// The pipeline's transform lengths factor into small primes (3024 = 7*4*4*3*3*3,
// 4032 = 7*4*4*4*3*3), so Bluestein is unnecessary for them. Bluestein inflated a
// 4032-point transform into an 8192-point one and ran a forward *and* inverse
// inside it: 26 butterfly passes over twice the data, where this needs 6 stages
// over the real length. That is roughly 8x less memory traffic, and these
// kernels are bandwidth-bound.
//
// Stockham is autosort: it ping-pongs between two buffers and needs no bit or
// digit reversal pass. Index math here is a direct transcription of a host
// prototype checked against a naive DFT at every length the pipeline uses.
//
// One dispatch per stage; thread t handles one radix-R butterfly, gid.y selects
// the batch row.
// ---------------------------------------------------------------------------
struct StockhamParams {
    uint N;            // transform length
    uint R;            // radix of this stage
    uint Ns;           // product of radices already applied
    uint NR;           // N / R  == number of threads per batch
    uint stride;       // distance between batch vectors
    uint batch_count;
    uint inverse;
    uint _pad0;        // 32 bytes for setBytes
};

kernel void fft_stockham_stage(device const float2* src [[buffer(0)]],
                               device float2* dst [[buffer(1)]],
                               constant StockhamParams& p [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.x;
    const uint batch = gid.y;
    if (t >= p.NR || batch >= p.batch_count) return;
    const uint R = p.R;
    if (R < 2u || R > 7u) return;

    device const float2* x = src + batch * p.stride;
    device float2* y = dst + batch * p.stride;

    const float sgn = (p.inverse != 0u) ? 2.f : -2.f;
    const uint j = t % p.Ns;

    // Load R inputs, each pre-multiplied by its stage twiddle.
    const float ang = sgn * PI * float(j) / float(p.Ns * R);
    float2 v[7];
    for (uint r = 0u; r < R; ++r) {
        const float a = ang * float(r);
        v[r] = cmul(x[t + r * p.NR], float2(cos(a), sin(a)));
    }

    // Radix-R DFT. W holds the R distinct roots, so the (k*m)%R lookup gives the
    // same values the host prototype computed, in the same summation order.
    float2 W[7];
    for (uint i = 0u; i < R; ++i) {
        const float a = sgn * PI * float(i) / float(R);
        W[i] = float2(cos(a), sin(a));
    }
    float2 tv[7];
    for (uint i = 0u; i < R; ++i) tv[i] = v[i];
    for (uint k = 0u; k < R; ++k) {
        float2 s = float2(0.f, 0.f);
        for (uint m = 0u; m < R; ++m) s += cmul(tv[m], W[(k * m) % R]);
        v[k] = s;
    }

    const uint base = (t / p.Ns) * p.Ns * R + j;
    for (uint r = 0u; r < R; ++r) y[base + r * p.Ns] = v[r];
}

// Copy a batch of vectors; used when an odd stage count leaves the Stockham
// result in the ping-pong buffer.
kernel void fft_copy_batch(device float2* dst [[buffer(0)]],
                           device const float2* src [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           constant uint& stride [[buffer(3)]],
                           constant uint& batch_count [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= n || gid.y >= batch_count) return;
    const uint off = gid.y * stride + gid.x;
    dst[off] = src[off];
}

kernel void fft1d_bitrev(device float2* data [[buffer(0)]],
                         constant uint& n [[buffer(1)]],
                         constant uint& stride [[buffer(2)]],
                         constant uint& batch_count [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint i = gid.x;
    if (batch >= batch_count || i >= n || n <= 1u) return;
    uint j = 0;
    uint x = i;
    for (uint bit = n >> 1; bit != 0u; bit >>= 1) {
        j = (j << 1) | (x & 1u);
        x >>= 1;
    }
    if (i < j) {
        device float2* a = data + batch * stride;
        float2 tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}

kernel void fft1d_butterfly(device float2* data [[buffer(0)]],
                            constant uint& n [[buffer(1)]],
                            constant uint& stride [[buffer(2)]],
                            constant uint& batch_count [[buffer(3)]],
                            constant uint& len [[buffer(4)]],
                            constant int& inverse [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint id = gid.x; // 0 .. n/2 - 1
    uint half_n = len >> 1;
    if (batch >= batch_count || id >= (n >> 1) || half_n == 0u) return;
    uint group = id / half_n;
    uint k = id % half_n;
    uint i = group * len;

    // Twiddle w = exp(sign * 2*pi*i*k/len), evaluated directly.
    //
    // This previously built wlen = exp(sign*2*pi*i/len) and raised it to the
    // k-th power by binary exponentiation: up to ~13 complex multiplies plus
    // ~13 squarings per thread. That is roughly 26 complex multiplies to set up
    // a butterfly that performs exactly one, so the kernel spent most of its
    // arithmetic generating twiddles rather than transforming data, and this is
    // the innermost kernel of every FFT stage in the pipeline.
    //
    // wlen^k is exp(i*ang*k) by definition, so evaluating the angle directly is
    // mathematically the same value. It is also more accurate: repeated squaring
    // compounds rounding at every step, while this is a single trig evaluation.
    // Results move in the last few ULPs, well under one 16-bit LSB.
    //
    // k < len/2, so the angle stays within [-pi, pi] and needs no reduction.
    float ang = (inverse != 0 ? 2.f : -2.f) * PI * (float(k) / float(len));
    float2 w = float2(cos(ang), sin(ang));

    device float2* a = data + batch * stride;
    float2 u = a[i + k];
    float2 v = cmul(a[i + k + half_n], w);
    a[i + k] = u + v;
    a[i + k + half_n] = u - v;
}

kernel void fft_scale_inv(device float2* data [[buffer(0)]],
                          constant uint& n [[buffer(1)]],
                          constant uint& stride [[buffer(2)]],
                          constant uint& batch_count [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint i = gid.x;
    if (batch >= batch_count || i >= n) return;
    data[batch * stride + i] /= float(n);
}

kernel void cbuf_mul_broadcast_B(device float2* A [[buffer(0)]],
                                 device const float2* B [[buffer(1)]],
                                 constant uint& m [[buffer(2)]],
                                 constant uint& batch [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint b = gid.y;
    uint i = gid.x;
    if (b >= batch || i >= m) return;
    A[b * m + i] = cmul(A[b * m + i], B[i]);
}

kernel void bluestein_pack_A(device float2* A [[buffer(0)]],
                             device const float2* in [[buffer(1)]],
                             device const float2* chirp [[buffer(2)]],
                             constant uint& n [[buffer(3)]],
                             constant uint& m [[buffer(4)]],
                             constant uint& in_stride [[buffer(5)]],
                             constant uint& batch_count [[buffer(6)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint i = gid.x;
    if (batch >= batch_count || i >= m) return;
    uint out_i = batch * m + i;
    if (i < n)
        A[out_i] = cmul(in[batch * in_stride + i], chirp[i]);
    else
        A[out_i] = float2(0.f, 0.f);
}

// Clear B then fill — avoid race that zeroed mirrored taps (old bug).
kernel void bluestein_clear_B(device float2* B [[buffer(0)]],
                              constant uint& m [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= m) return;
    B[i] = float2(0.f, 0.f);
}

kernel void bluestein_fill_B(device float2* B [[buffer(0)]],
                             device const float2* chirp [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             constant uint& m [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    // Only i in [0,n) write; mirrored index m-i is written here too.
    if (i >= n) return;
    float2 v = cconj(chirp[i]);
    B[i] = v;
    if (i > 0u) B[m - i] = v;
}

kernel void bluestein_extract(device float2* out [[buffer(0)]],
                              device const float2* A [[buffer(1)]],
                              device const float2* chirp [[buffer(2)]],
                              constant uint& n [[buffer(3)]],
                              constant uint& m [[buffer(4)]],
                              constant uint& out_stride [[buffer(5)]],
                              constant uint& batch_count [[buffer(6)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint i = gid.x;
    if (batch >= batch_count || i >= n) return;
    out[batch * out_stride + i] = cmul(A[batch * m + i], chirp[i]);
}

kernel void make_chirp(device float2* chirp [[buffer(0)]],
                       constant uint& n [[buffer(1)]],
                       constant int& inverse [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float dir = inverse != 0 ? 1.f : -1.f;
    float ang = dir * PI * float(i) * float(i) / float(n);
    chirp[i] = float2(cos(ang), sin(ang));
}

kernel void pack_rows_real(device float2* out [[buffer(0)]],
                           device const float* in [[buffer(1)]],
                           constant uint& h [[buffer(2)]],
                           constant uint& w [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint y = gid.y, x = gid.x;
    if (y >= h || x >= w) return;
    out[y * w + x] = float2(in[y * w + x], 0.f);
}

kernel void transpose_c(device float2* out [[buffer(0)]],
                        device const float2* in [[buffer(1)]],
                        constant uint& h [[buffer(2)]],
                        constant uint& w [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint y = gid.y, x = gid.x;
    if (y >= h || x >= w) return;
    out[x * h + y] = in[y * w + x];
}

// Pack/scatter column strips so 2D FFT needs only one full-frame complex buffer.
kernel void gather_cols(device float2* out [[buffer(0)]],
                        device const float2* in [[buffer(1)]],
                        constant uint& h [[buffer(2)]],
                        constant uint& w [[buffer(3)]],
                        constant uint& col0 [[buffer(4)]],
                        constant uint& ncol [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.y, y = gid.x;
    if (c >= ncol || y >= h) return;
    out[c * h + y] = in[y * w + (col0 + c)];
}

kernel void scatter_cols(device float2* out [[buffer(0)]],
                         device const float2* in [[buffer(1)]],
                         constant uint& h [[buffer(2)]],
                         constant uint& w [[buffer(3)]],
                         constant uint& col0 [[buffer(4)]],
                         constant uint& ncol [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.y, y = gid.x;
    if (c >= ncol || y >= h) return;
    out[y * w + (col0 + c)] = in[c * h + y];
}

// In-place fftshift for even h,w (swap halves). Involutory — inv unused.
kernel void fftshift_swap_x(device float2* data [[buffer(0)]],
                            constant uint& h [[buffer(1)]],
                            constant uint& w [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint y = gid.y, x = gid.x;
    uint half_w = w / 2u;
    if (y >= h || x >= half_w) return;
    uint i0 = y * w + x;
    uint i1 = y * w + x + half_w;
    float2 t = data[i0];
    data[i0] = data[i1];
    data[i1] = t;
}

kernel void fftshift_swap_y(device float2* data [[buffer(0)]],
                            constant uint& h [[buffer(1)]],
                            constant uint& w [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint y = gid.y, x = gid.x;
    uint half_h = h / 2u;
    if (y >= half_h || x >= w) return;
    uint i0 = y * w + x;
    uint i1 = (y + half_h) * w + x;
    float2 t = data[i0];
    data[i0] = data[i1];
    data[i1] = t;
}

kernel void fftshift2d_c(device float2* out [[buffer(0)]],
                         device const float2* in [[buffer(1)]],
                         constant uint& h [[buffer(2)]],
                         constant uint& w [[buffer(3)]],
                         constant int& inv [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint y = gid.y, x = gid.x;
    if (y >= h || x >= w) return;
    uint shy = h / 2u, shx = w / 2u;
    if (inv == 0) {
        out[y * w + x] = in[((y + shy) % h) * w + ((x + shx) % w)];
    } else {
        out[y * w + x] = in[((y + h - shy) % h) * w + ((x + w - shx) % w)];
    }
}

// Natural-order equivalent of: fftshift_swap_x -> fftshift_swap_y ->
// zero_fft_borders -> fftshift_swap_x -> fftshift_swap_y.
//
// The two swaps are pairwise element exchanges, so each is its own inverse and
// the pair is a permutation of the buffer. Zeroing a set of positions in
// permuted space and then undoing the permutation is exactly zeroing the
// preimages of those positions in natural space -- the surviving values are
// carried through untouched either way. So this kernel reproduces the same
// buffer while touching it once instead of five times.
//
// X()/Y() below replicate fftshift_swap_x/y verbatim rather than simplifying to
// (x + w/2) % w. For odd w that swap leaves the final element in place, which
// is not what the modular form gives, and the grey FFT pads to whatever size
// the mixed-radix factorization wants -- not necessarily even.
kernel void zero_fft_borders_natural(device float2* data [[buffer(0)]],
                                     constant uint& h [[buffer(1)]],
                                     constant uint& w [[buffer(2)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    uint y = gid.y, x = gid.x;
    if (y >= h || x >= w) return;
    const uint half_w = w / 2u, half_h = h / 2u;
    // Where this element ends up after the two swaps.
    const uint xs = (x < half_w) ? (x + half_w)
                                 : ((x < 2u * half_w) ? (x - half_w) : x);
    const uint ys = (y < half_h) ? (y + half_h)
                                 : ((y < 2u * half_h) ? (y - half_h) : y);
    const uint y0 = h / 4u, x0 = w / 4u;
    if (ys < y0 || ys >= h - y0 || xs < x0 || xs >= w - x0)
        data[y * w + x] = float2(0.f, 0.f);
}

kernel void zero_fft_borders(device float2* data [[buffer(0)]],
                             constant uint& h [[buffer(1)]],
                             constant uint& w [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint y = gid.y, x = gid.x;
    if (y >= h || x >= w) return;
    uint y0 = h / 4u, x0 = w / 4u;
    if (y < y0 || y >= h - y0 || x < x0 || x >= w - x0)
        data[y * w + x] = float2(0.f, 0.f);
}

kernel void extract_real(device float* out [[buffer(0)]],
                         device const float2* in [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = in[id].x;
}

struct L2Params {
    uint ny, nx;
    int ts, R, N;
    int ref_h, ref_w, mov_h, mov_w;
    uint tile_base, tile_count;
};

struct AlignLocalSearch460Params {
    uint ny, nx;
    int ts, R;
    int ref_h, ref_w, mov_h, mov_w;
    uint l1;
    float ambiguity_ratio;
    uint write_ambiguity;
    uint fallback_on_ambiguous;  // kunzmi: ambiguous -> keep seed, no shift
    uint subpixel;               // quadratic sub-cell fit at the winner
};

// Least-squares bivariate quadratic over the 3x3 cost neighbourhood of the
// winning offset; minimum at mu = -H^-1 g. Twin of quadratic_subpixel_3x3 in
// align.cpp -- same closed forms, same guards (finite costs, positive-definite
// Hessian, |mu| <= 0.5 per axis), so CPU and GPU block matching stay in step.
static inline bool bm_quadratic_subpixel_3x3(thread const float* D,
                                             thread float& mu_x,
                                             thread float& mu_y) {
    for (int i = 0; i < 9; ++i)
        if (!isfinite(D[i])) return false;
    const float P   = D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7] + D[8];
    const float Sx  = (D[2] + D[5] + D[8]) - (D[0] + D[3] + D[6]);
    const float Sy  = (D[6] + D[7] + D[8]) - (D[0] + D[1] + D[2]);
    const float Sxx = D[0] + D[2] + D[3] + D[5] + D[6] + D[8];
    const float Syy = D[0] + D[1] + D[2] + D[6] + D[7] + D[8];
    const float Sxy = (D[0] + D[8]) - (D[2] + D[6]);
    const float d = Sx / 6.f;
    const float e = Sy / 6.f;
    const float c = Sxy / 4.f;
    const float a = 0.5f * Sxx - P / 3.f;
    const float b = 0.5f * Syy - P / 3.f;
    const float h11 = 2.f * a, h22 = 2.f * b;
    const float det = h11 * h22 - c * c;
    if (!(h11 > 0.f) || !(h22 > 0.f) || !(det > 1e-12f)) return false;
    const float mx = -(h22 * d - c * e) / det;
    const float my = -(h11 * e - c * d) / det;
    if (!(fabs(mx) <= 0.5f) || !(fabs(my) <= 0.5f)) return false;
    mu_x = mx;
    mu_y = my;
    return true;
}

// One threadgroup per tile; threads split the candidate shifts.
//
// The previous form ran one thread per tile and looped every candidate serially,
// re-reading the whole reference tile from device memory (2R+1)^2 times. Here the
// reference tile is staged once in threadgroup memory and each candidate's SSD/SAD
// is still summed by a single thread in the original i-then-j order, so every
// `dist` is bit-identical. Candidates are visited in ascending index and accepted
// only on strict `<`, and the tree reduction breaks ties toward the lower index,
// which reproduces the original sdy-outer / sdx-inner first-minimum-wins scan.
//
// Threadgroup memory is supplied by the host: [0] = ts*ts floats for the tile,
// [1]/[2] = best float/int per lane, [3]/[4] = second-best float/int per lane,
// [5] = (2R+1)^2 floats holding every candidate's cost, kept for the sub-pixel
// quadratic fit at the winner. `lanes` must be a power of two (the host
// enforces this) for the tree reduction to cover every element.
kernel void align_local_search_460(device const float* ref [[buffer(0)]],
                                   device const float* mov [[buffer(1)]],
                                   device float* flow [[buffer(2)]],
                                   constant AlignLocalSearch460Params& p [[buffer(3)]],
                                   device uint* ambiguity [[buffer(4)]],
                                   threadgroup float* ref_tile [[threadgroup(0)]],
                                   threadgroup float* red_dist [[threadgroup(1)]],
                                   threadgroup int* red_cand [[threadgroup(2)]],
                                   threadgroup float* red_second_dist [[threadgroup(3)]],
                                   threadgroup int* red_second_cand [[threadgroup(4)]],
                                   threadgroup float* cand_cost [[threadgroup(5)]],
                                   uint tile_id [[threadgroup_position_in_grid]],
                                   uint lane [[thread_position_in_threadgroup]],
                                   uint lanes [[threads_per_threadgroup]]) {
    // Uniform across the threadgroup, so every thread leaves together and the
    // barriers below stay well-formed.
    if (tile_id >= p.ny * p.nx) return;

    const int ts = p.ts;
    const int R = p.R;
    const int span = 2 * R + 1;
    const int ncand = span * span;
    const int kNone = 0x7FFFFFFF;

    const uint ty = tile_id / p.nx;
    const uint tx = tile_id % p.nx;
    const int ox = int(tx) * ts;
    const int oy = int(ty) * ts;
    const float local_fx = flow[tile_id * 2u + 0u];
    const float local_fy = flow[tile_id * 2u + 1u];

    // The original invalidated a candidate on its first out-of-bounds sample.
    // The sampled region is a full rectangle, so that is exactly rectangle
    // containment — testable up front without touching memory.
    const bool ref_in = (ox >= 0 && oy >= 0 &&
                         ox + ts <= p.ref_w && oy + ts <= p.ref_h);

    if (ref_in) {
        for (int k = int(lane); k < ts * ts; k += int(lanes)) {
            const int i = k / ts;
            const int j = k - i * ts;
            ref_tile[k] = ref[uint(oy + i) * uint(p.ref_w) + uint(ox + j)];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best_dist = INFINITY;
    float second_dist = INFINITY;
    int best_cand = kNone;
    int second_cand = kNone;

    if (ref_in) {
        // Round, do not truncate -- see block_match_level_direct_460 in
        // align.cpp. round() here matches std::lround: ties away from zero.
        const int ifx = int(round(local_fx));
        const int ify = int(round(local_fy));
        for (int c = int(lane); c < ncand; c += int(lanes)) {
            const int row = c / span;
            const int sdy = row - R;
            const int sdx = (c - row * span) - R;
            const int mx0 = ox + ifx + sdx;
            const int my0 = oy + ify + sdy;
            // Every candidate slot is written exactly once (each c belongs to
            // one lane), so the surface is complete before the barrier below.
            cand_cost[c] = INFINITY;
            // Out-of-bounds candidates scored INFINITY and could never win a
            // strict `<`, so skipping them is equivalent.
            if (!(mx0 >= 0 && my0 >= 0 &&
                  mx0 + ts <= p.mov_w && my0 + ts <= p.mov_h))
                continue;

            float dist = 0.f;
            for (int i = 0; i < ts; ++i) {
                const uint mrow = uint(my0 + i) * uint(p.mov_w);
                const int trow = i * ts;
                for (int j = 0; j < ts; ++j) {
                    const float diff = ref_tile[trow + j] - mov[mrow + uint(mx0 + j)];
                    dist += (p.l1 != 0u) ? fabs(diff) : diff * diff;
                }
            }
            cand_cost[c] = dist;
            if (dist < best_dist) {
                second_dist = best_dist;
                second_cand = best_cand;
                best_dist = dist;
                best_cand = c;
            } else if (dist < second_dist) {
                second_dist = dist;
                second_cand = c;
            }
        }
    }

    red_dist[lane] = best_dist;
    red_cand[lane] = best_cand;
    red_second_dist[lane] = second_dist;
    red_second_cand[lane] = second_cand;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = lanes >> 1u; s > 0u; s >>= 1u) {
        if (lane < s) {
            float bd = red_dist[lane];
            int bc = red_cand[lane];
            float sd = red_second_dist[lane];
            int sc = red_second_cand[lane];

            const float od0 = red_dist[lane + s];
            const int oc0 = red_cand[lane + s];
            if (oc0 != kNone) {
                if (bc == kNone || od0 < bd || (od0 == bd && oc0 < bc)) {
                    sd = bd;
                    sc = bc;
                    bd = od0;
                    bc = oc0;
                } else if (oc0 != bc &&
                           (sc == kNone || od0 < sd || (od0 == sd && oc0 < sc))) {
                    sd = od0;
                    sc = oc0;
                }
            }

            const float od1 = red_second_dist[lane + s];
            const int oc1 = red_second_cand[lane + s];
            if (oc1 != kNone) {
                if (bc == kNone || od1 < bd || (od1 == bd && oc1 < bc)) {
                    sd = bd;
                    sc = bc;
                    bd = od1;
                    bc = oc1;
                } else if (oc1 != bc &&
                           (sc == kNone || od1 < sd || (od1 == sd && oc1 < sc))) {
                    sd = od1;
                    sc = oc1;
                }
            }
            red_dist[lane] = bd;
            red_cand[lane] = bc;
            red_second_dist[lane] = sd;
            red_second_cand[lane] = sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        const int c = red_cand[0];
        int sdx = 0, sdy = 0;   // no valid candidate -> unchanged flow, as before
        if (c != kNone) {
            const int row = c / span;
            sdy = row - R;
            sdx = (c - row * span) - R;
        }
        const float b = max(red_dist[0], 0.f);
        const float sec = max(red_second_dist[0], 0.f);
        const float denom = max(b, 1.0e-12f);
        const bool ambiguous =
            isfinite(b) && isfinite(sec) && (sec / denom) < p.ambiguity_ratio;
        float sub_x = 0.f, sub_y = 0.f;
        if (p.fallback_on_ambiguous != 0u && ambiguous) {
            // ImageStackAlignator's rule: no precise shift determinable ->
            // apply NO shift; keep the upsampled previous-level seed.
            sdx = 0;
            sdy = 0;
        } else if (p.subpixel != 0u && c != kNone &&
                   sdx > -R && sdx < R && sdy > -R && sdy < R) {
            // Winner interior to the window: the full 3x3 cost neighbourhood
            // exists in cand_cost (every slot written before the reduction's
            // barriers, which also order it for this lane). The fit's own
            // guards reject non-finite neighbours, ridges and out-of-cell
            // minima, in which case the integer result stands -- identical to
            // the CPU path in align.cpp.
            float D[9];
            for (int yy = -1; yy <= 1; ++yy)
                for (int xx = -1; xx <= 1; ++xx)
                    D[(yy + 1) * 3 + (xx + 1)] =
                        cand_cost[(sdy + yy + R) * span + (sdx + xx + R)];
            (void)bm_quadratic_subpixel_3x3(D, sub_x, sub_y);
        }
        flow[tile_id * 2u + 0u] = local_fx + float(sdx) + sub_x;
        flow[tile_id * 2u + 1u] = local_fy + float(sdy) + sub_y;
        if (p.write_ambiguity != 0u)
            ambiguity[tile_id] |= ambiguous ? 1u : 0u;
    }
}

kernel void l2_pack_tiles(device float* ref_pad [[buffer(0)]],
                          device float* mov_patch [[buffer(1)]],
                          device const float* ref [[buffer(2)]],
                          device const float* mov [[buffer(3)]],
                          device const float* flow [[buffer(4)]],
                          constant L2Params& P [[buffer(5)]],
                          uint local [[thread_position_in_grid]]) {
    if (local >= P.tile_count) return;
    uint tid = P.tile_base + local;
    uint ntiles = P.ny * P.nx;
    if (tid >= ntiles) return;
    uint ty = tid / P.nx;
    uint tx = tid % P.nx;
    int ts = P.ts, R = P.R, N = P.N;
    int oy = int(ty) * ts;
    int ox = int(tx) * ts;
    float fdx = flow[tid * 2u + 0u];
    float fdy = flow[tid * 2u + 1u];
    int flow_dx = int(rint(fdx));
    int flow_dy = int(rint(fdy));

    uint base = local * uint(N * N);
    for (int i = 0; i < N * N; ++i)
        ref_pad[base + uint(i)] = 0.f;

    for (int i = 0; i < ts; ++i) {
        for (int j = 0; j < ts; ++j) {
            int ry = oy + i, rx = ox + j;
            if (ry >= 0 && ry < P.ref_h && rx >= 0 && rx < P.ref_w)
                ref_pad[base + uint((i + R) * N + (j + R))] =
                    ref[uint(ry) * uint(P.ref_w) + uint(rx)];
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int my = clamp(oy + flow_dy + i - R, 0, P.mov_h - 1);
            int mx = clamp(ox + flow_dx + j - R, 0, P.mov_w - 1);
            mov_patch[base + uint(i * N + j)] =
                mov[uint(my) * uint(P.mov_w) + uint(mx)];
        }
    }
}

kernel void l2_conj_mul(device float2* F [[buffer(0)]],
                        device const float2* Fmov [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    F[id] = cmul(cconj(F[id]), Fmov[id]);
}

kernel void l2_argmin(device float* flow [[buffer(0)]],
                      device const float* corr [[buffer(1)]],
                      device const float* mov_patch [[buffer(2)]],
                      constant L2Params& P [[buffer(3)]],
                      uint local [[thread_position_in_grid]]) {
    if (local >= P.tile_count) return;
    uint tid = P.tile_base + local;
    uint ntiles = P.ny * P.nx;
    if (tid >= ntiles) return;
    int N = P.N, ts = P.ts, R = P.R;
    int corr_size = 2 * R + 1;
    int crop = (N - 1 - corr_size) / 2;
    int crop0 = crop + 1;
    uint base = local * uint(N * N);

    float best = 1e30f;
    int best_dy = 0, best_dx = 0;
    for (int i = 0; i < corr_size; ++i) {
        for (int j = 0; j < corr_size; ++j) {
            float sum_sq = 0.f;
            for (int ki = 0; ki < ts; ++ki)
                for (int kj = 0; kj < ts; ++kj) {
                    float v = mov_patch[base + uint((i + ki) * N + (j + kj))];
                    sum_sq += v * v;
                }
            float c = corr[base + uint((crop0 + i) * N + (crop0 + j))];
            float err = sum_sq - 2.f * c;
            if (err < best) {
                best = err;
                best_dy = i - corr_size / 2;
                best_dx = j - corr_size / 2;
            }
        }
    }
    flow[tid * 2u + 0u] += float(best_dx);
    flow[tid * 2u + 1u] += float(best_dy);
}

kernel void fftshift2d_real(device float* out [[buffer(0)]],
                            device const float* in [[buffer(1)]],
                            constant uint& h [[buffer(2)]],
                            constant uint& w [[buffer(3)]],
                            constant uint& batch_count [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint idx = gid.x;
    if (batch >= batch_count || idx >= h * w) return;
    uint y = idx / w, x = idx % w;
    uint shy = h / 2u, shx = w / 2u;
    out[batch * h * w + y * w + x] =
        in[batch * h * w + ((y + shy) % h) * w + ((x + shx) % w)];
}

kernel void pack_tile_rows(device float2* out [[buffer(0)]],
                           device const float* in [[buffer(1)]],
                           constant uint& N [[buffer(2)]],
                           constant uint& ntiles [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint x = gid.x;
    if (row >= ntiles * N || x >= N) return;
    out[row * N + x] = float2(in[row * N + x], 0.f);
}

kernel void take_rfft_half(device float2* out [[buffer(0)]],
                           device const float2* in [[buffer(1)]],
                           constant uint& N [[buffer(2)]],
                           constant uint& wh [[buffer(3)]],
                           constant uint& nrows [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint x = gid.x;
    if (row >= nrows || x >= wh) return;
    out[row * wh + x] = in[row * N + x];
}

kernel void write_rfft_cols_from_half(device float2* cols [[buffer(0)]],
                                      device const float2* rfft_pack [[buffer(1)]],
                                      constant uint& N [[buffer(2)]],
                                      constant uint& wh [[buffer(3)]],
                                      constant uint& ntiles [[buffer(4)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    uint tile = gid.y;
    uint idx = gid.x;
    if (tile >= ntiles || idx >= N * wh) return;
    uint y = idx / wh;
    uint xf = idx % wh;
    cols[(tile * wh + xf) * N + y] = rfft_pack[(tile * N + y) * wh + xf];
}

kernel void write_half_from_cols(device float2* rfft_pack [[buffer(0)]],
                                 device const float2* cols [[buffer(1)]],
                                 constant uint& N [[buffer(2)]],
                                 constant uint& wh [[buffer(3)]],
                                 constant uint& ntiles [[buffer(4)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint tile = gid.y;
    uint idx = gid.x;
    if (tile >= ntiles || idx >= N * wh) return;
    uint y = idx / wh;
    uint xf = idx % wh;
    rfft_pack[(tile * N + y) * wh + xf] = cols[(tile * wh + xf) * N + y];
}

kernel void expand_half_to_full_rows(device float2* full [[buffer(0)]],
                                     device const float2* rfft_pack [[buffer(1)]],
                                     constant uint& N [[buffer(2)]],
                                     constant uint& wh [[buffer(3)]],
                                     constant uint& ntiles [[buffer(4)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    uint tile = gid.y;
    uint idx = gid.x;
    if (tile >= ntiles || idx >= N * N) return;
    uint y = idx / N;
    uint x = idx % N;
    uint base_h = (tile * N + y) * wh;
    uint base_f = (tile * N + y) * N;
    if (x < wh)
        full[base_f + x] = rfft_pack[base_h + x];
    else {
        uint k = N - x;
        full[base_f + x] = cconj(rfft_pack[base_h + k]);
    }
}

kernel void extract_real_tiles(device float* out [[buffer(0)]],
                               device const float2* in [[buffer(1)]],
                               constant uint& N [[buffer(2)]],
                               constant uint& ntiles [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    uint tile = gid.y;
    uint i = gid.x;
    if (tile >= ntiles || i >= N * N) return;
    out[tile * N * N + i] = in[tile * N * N + i].x;
}

// ---- Merge (Alg. 4 / Alg. 11) — faithful port of merge.cpp -----------------

// Layout must match MergeCompParamsCPU / MergeRefParamsCPU in metal_gpu.mm.
// Padded to a multiple of 16 bytes for constant-buffer setBytes alignment.
struct MergeCompParams {
    uint band_h;
    uint Ws;
    uint y0;
    uint lr_h;
    uint lr_w;
    uint rob_h;
    uint rob_w;
    uint flow_ny;
    uint flow_nx;
    uint cov_h;
    uint cov_w;
    uint nch;
    uint bayer;
    uint iso;
    uint tile_size;
    float scale;
    uint cfa00;
    uint cfa01;
    uint cfa10;
    uint cfa11;
    // 1 = robustness is raw resolution this run (Config::
    // robustness_raw_resolution_active), not guide -- skip the guide-scale
    // conversion below (was _pad0).
    uint raw_res_robustness;
    uint flow_bilinear;  // 1 = interpolate the tile flow (was _pad1)
    uint fast_weights;   // skip negligible taps/hypotheses (was _pad2)
    float soften_max_inv; // inverse-covariance eigenvalue ceiling (was _pad3)
};

struct MergeRefParams {
    uint band_h;
    uint Ws;
    uint y0;
    uint lr_h;
    uint lr_w;
    uint cov_h;
    uint cov_w;
    uint acc_h;
    uint acc_w;
    uint nch;
    uint bayer;
    uint iso;
    uint robustness_denoise;
    uint rad_max;
    float scale;
    float max_multiplier;
    float burst_frames;
    uint cfa00;
    uint cfa01;
    uint cfa10;
    uint cfa11;
    uint adaptive;
    float max_frame_count;
    // 1 = acc_rob is raw resolution this run (Config::
    // robustness_raw_resolution_active) -- skip the guide-scale conversion
    // below (was _pad0).
    uint raw_res_robustness;
    float soften_max_inv; // inverse-covariance eigenvalue ceiling (100 bytes)
};

// std::lround half-away-from-zero.
inline int lround_away(float x) {
    return (x >= 0.f) ? int(floor(x + 0.5f)) : int(ceil(x - 0.5f));
}

inline float cov_at(device const float* covs, uint cov_w, int y, int x, int idx) {
    return covs[(uint(y) * cov_w + uint(x)) * 4u + uint(idx)];
}

// Twin of soften_inv_cov in core/merge.cpp -- keep them in step. See there for
// why an unclamped inverse covariance shows up as green or black speckles
// rather than as general softness.
inline void soften_inv_cov(thread float& ixx, thread float& ixy, thread float& iyy,
                           float k_max_abs) {
    if (!isfinite(ixx) || !isfinite(ixy) || !isfinite(iyy)) {
        ixx = 2.f;
        ixy = 0.f;
        iyy = 2.f;
        return;
    }
    // Eigenvalue clamp, same op order as the CPU twin: bound only the sharp
    // axis to the coverage floor; the wide axis is left alone, so edges are
    // not blurred along themselves the way the whole-matrix rescale did.
    const float mean = 0.5f * (ixx + iyy);
    const float half_diff = 0.5f * (ixx - iyy);
    const float disc = sqrt(half_diff * half_diff + ixy * ixy);
    const float l1 = mean + disc;
    if (!(l1 > k_max_abs)) return;
    const float l2 = mean - disc;
    const float c1 = k_max_abs;
    const float c2 = min(l2, k_max_abs);
    float vx = ixy;
    float vy = l1 - ixx;
    float n2 = vx * vx + vy * vy;
    if (!(n2 > 0.f)) {
        vx = l1 - iyy;
        vy = ixy;
        n2 = vx * vx + vy * vy;
    }
    if (!(n2 > 0.f)) {
        ixx = min(ixx, k_max_abs);
        iyy = min(iyy, k_max_abs);
        return;
    }
    const float inv_n2 = 1.f / n2;
    const float d = c1 - c2;
    ixx = c2 + d * (vx * vx * inv_n2);
    ixy = d * (vx * vy * inv_n2);
    iyy = c2 + d * (vy * vy * inv_n2);
}

inline float cov_lerp2(device const float* covs, uint cov_w,
                       int fy, int fx, int cy, int cx,
                       float frac_x, float frac_y, int idx) {
    float tl = cov_at(covs, cov_w, fy, fx, idx);
    float tr = cov_at(covs, cov_w, fy, cx, idx);
    float bl = cov_at(covs, cov_w, cy, fx, idx);
    float br = cov_at(covs, cov_w, cy, cx, idx);
    float top = tl + frac_x * (tr - tl);
    float bot = bl + frac_x * (br - bl);
    return top + frac_y * (bot - top);
}

// raw_det=true -> accumulate (comp); false -> accumulate_ref
inline void interp_inv_cov(device const float* covs, uint cov_h, uint cov_w,
                           float kmap_i, float kmap_j,
                           thread float& ixx, thread float& ixy, thread float& iyy,
                           bool raw_det, float soften_max) {
    float frac_x = kmap_j - trunc(kmap_j);
    float frac_y = kmap_i - trunc(kmap_i);
    int fx, fy;
    if (raw_det) {
        fx = max(int(kmap_j), 0);
        fy = max(int(kmap_i), 0);
    } else {
        fx = max(int(floor(kmap_j)), 0);
        fy = max(int(floor(kmap_i)), 0);
    }
    int cx = min(fx + 1, int(cov_w) - 1);
    int cy = min(fy + 1, int(cov_h) - 1);

    float xx = cov_lerp2(covs, cov_w, fy, fx, cy, cx, frac_x, frac_y, 0);
    float xy = cov_lerp2(covs, cov_w, fy, fx, cy, cx, frac_x, frac_y, 1);
    float yy = cov_lerp2(covs, cov_w, fy, fx, cy, cx, frac_x, frac_y, 3);
    if (raw_det) {
        float det = xx * yy - xy * xy;
        if (fabs(det) > 1e-10f) {
            float inv_det = 1.f / det;
            ixx =  inv_det * yy;
            ixy = -inv_det * xy;
            iyy =  inv_det * xx;
        } else {
            ixx = 1.f; ixy = 0.f; iyy = 1.f;
        }
    } else {
        // invert_sym_2x2 / invert_2x2 with EPSILON_DIV
        float det = xx * yy - xy * xy;
        if (fabs(det) > 1e-10f) {
            float det_i = 1.f / det;
            ixx =  yy * det_i;
            ixy = -xy * det_i;
            iyy =  xx * det_i;
        } else {
            ixx = 1.f; ixy = 0.f; iyy = 1.f;
        }
    }
    soften_inv_cov(ixx, ixy, iyy, soften_max);
}

inline int cfa_channel(constant MergeCompParams& p, int i, int j) {
    if (!p.bayer) return 0;
    int ii = i & 1, jj = j & 1;
    if (ii == 0 && jj == 0) return int(p.cfa00);
    if (ii == 0 && jj == 1) return int(p.cfa01);
    if (ii == 1 && jj == 0) return int(p.cfa10);
    return int(p.cfa11);
}

inline int cfa_channel_ref(constant MergeRefParams& p, int i, int j) {
    if (!p.bayer) return 0;
    int ii = i & 1, jj = j & 1;
    if (ii == 0 && jj == 0) return int(p.cfa00);
    if (ii == 0 && jj == 1) return int(p.cfa01);
    if (ii == 1 && jj == 0) return int(p.cfa10);
    return int(p.cfa11);
}

// Bilinear sample of the per-tile flow at a RAW position. Twin of
// FlowField::sample_bilinear in core/types.h -- keep them in step. Block
// matching gives one vector per tile; consuming it nearest makes the warp
// piecewise constant, which rotation turns into a visible tile grid because
// the true field varies continuously with position. Tile t spans raw
// [t*ts,(t+1)*ts) so its centre is (t+0.5)*ts, hence tile coord = p/ts - 0.5.
inline void flow_sample_bilinear(device const float* flow, uint fny, uint fnx,
                                 float raw_y, float raw_x, float ts,
                                 thread float& odx, thread float& ody) {
    float tcy = raw_y / ts - 0.5f, tcx = raw_x / ts - 0.5f;
    int y0 = int(floor(tcy)), x0 = int(floor(tcx));
    float ay = tcy - float(y0), ax = tcx - float(x0);
    int iy0 = clamp(y0, 0, int(fny) - 1), iy1 = clamp(y0 + 1, 0, int(fny) - 1);
    int ix0 = clamp(x0, 0, int(fnx) - 1), ix1 = clamp(x0 + 1, 0, int(fnx) - 1);
    uint i00 = (uint(iy0) * fnx + uint(ix0)) * 2u, i01 = (uint(iy0) * fnx + uint(ix1)) * 2u;
    uint i10 = (uint(iy1) * fnx + uint(ix0)) * 2u, i11 = (uint(iy1) * fnx + uint(ix1)) * 2u;
    float tx = flow[i00 + 0u] + (flow[i01 + 0u] - flow[i00 + 0u]) * ax;
    float bx = flow[i10 + 0u] + (flow[i11 + 0u] - flow[i10 + 0u]) * ax;
    float ty = flow[i00 + 1u] + (flow[i01 + 1u] - flow[i00 + 1u]) * ax;
    float by = flow[i10 + 1u] + (flow[i11 + 1u] - flow[i10 + 1u]) * ax;
    odx = tx + (bx - tx) * ay;
    ody = ty + (by - ty) * ay;
}

inline float flow_catmull1(float p0, float p1, float p2, float p3, float t) {
    return 0.5f * (2.f * p1 + (-p0 + p2) * t +
                   (2.f * p0 - 5.f * p1 + 4.f * p2 - p3) * t * t +
                   (-p0 + 3.f * p1 - 3.f * p2 + p3) * t * t * t);
}

// Catmull-Rom twin of flow_sample_bilinear -- and of
// FlowField::sample_grid_bicubic in types.h; keep the three in step.
inline void flow_sample_bicubic(device const float* flow, uint fny, uint fnx,
                                float raw_y, float raw_x, float ts,
                                thread float& odx, thread float& ody) {
    float tcy = raw_y / ts - 0.5f, tcx = raw_x / ts - 0.5f;
    int y0 = int(floor(tcy)), x0 = int(floor(tcx));
    float ay = tcy - float(y0), ax = tcx - float(x0);
    float rx[4], ry[4];
    for (int i = -1; i <= 2; ++i) {
        int iy = clamp(y0 + i, 0, int(fny) - 1);
        float px[4], py[4];
        for (int j = -1; j <= 2; ++j) {
            int ix = clamp(x0 + j, 0, int(fnx) - 1);
            uint o = (uint(iy) * fnx + uint(ix)) * 2u;
            px[j + 1] = flow[o + 0u];
            py[j + 1] = flow[o + 1u];
        }
        rx[i + 1] = flow_catmull1(px[0], px[1], px[2], px[3], ax);
        ry[i + 1] = flow_catmull1(py[0], py[1], py[2], py[3], ax);
    }
    odx = flow_catmull1(rx[0], rx[1], rx[2], rx[3], ay);
    ody = flow_catmull1(ry[0], ry[1], ry[2], ry[3], ay);
}

// mode: 1 = bilinear, 2 = bicubic (0/nearest stays in the callers' else).
inline void flow_sample(device const float* flow, uint fny, uint fnx,
                        float raw_y, float raw_x, float ts, uint mode,
                        thread float& odx, thread float& ody) {
    if (mode == 2u)
        flow_sample_bicubic(flow, fny, fnx, raw_y, raw_x, ts, odx, ody);
    else
        flow_sample_bilinear(flow, fny, fnx, raw_y, raw_x, ts, odx, ody);
}

inline float sample_robustness_bilinear(device const float* robustness,
                                        uint h, uint w,
                                        float y, float x) {
    if (h == 0u || w == 0u) return 0.f;
    y = clamp(y, 0.f, float(h - 1u));
    x = clamp(x, 0.f, float(w - 1u));
    int y0 = int(floor(y));
    int x0 = int(floor(x));
    int y1 = min(y0 + 1, int(h) - 1);
    int x1 = min(x0 + 1, int(w) - 1);
    float fy = y - float(y0);
    float fx = x - float(x0);
    float top = robustness[uint(y0) * w + uint(x0)] +
                (robustness[uint(y0) * w + uint(x1)] -
                 robustness[uint(y0) * w + uint(x0)]) * fx;
    float bot = robustness[uint(y1) * w + uint(x0)] +
                (robustness[uint(y1) * w + uint(x1)] -
                 robustness[uint(y1) * w + uint(x0)]) * fx;
    return top + (bot - top) * fy;
}

// Alg. 4 — merge.cpp accumulate_comp
// One comparison frame's contribution at one output pixel.
//
// Extracted verbatim from merge_accumulate_comp so the single-frame and fused
// kernels below cannot drift apart -- there is exactly one copy of the math.
//
// The nine taps accumulate into locals and are added to the caller's running
// totals once, at the end. That ordering is what makes fusion exact: the
// caller sees the same per-frame val/acc it would have added had this frame
// been its own dispatch.
// The post-flow half of merge_comp_contrib: one hypothesis' contribution,
// scaled by wscale (1 outside overlapped-tile mode; the raised-cosine window
// weight inside it). Split so the hypothesis loop below shares this single
// copy of the math with the ordinary single-flow path.
static inline void merge_comp_contrib_flowed(device const float* img,
                                      device const float* covs,
                                      device const float* robustness,
                                      constant MergeCompParams& p,
                                      float lr_x, float lr_y,
                                      float flowx, float flowy, float wscale,
                                      thread float& n0, thread float& n1, thread float& n2,
                                      thread float& d0, thread float& d1, thread float& d2) {

    // p.raw_res_robustness: robustness is raw resolution this run (Config::
    // robustness_raw_resolution_active), same coordinate space as lr_y/lr_x
    // already -- skip the guide-scale conversion. See CPU accumulate_comp
    // (merge.cpp) for the mirrored fix.
    float rob_y = lr_y, rob_x = lr_x;
    if (p.raw_res_robustness == 0u && p.bayer != 0u) {
        rob_y = (lr_y - 0.5f) / 2.f;
        rob_x = (lr_x - 0.5f) / 2.f;
    }
    float local_r = sample_robustness_bilinear(robustness, p.rob_h, p.rob_w,
                                               rob_y, rob_x);
    // Nothing to accumulate where the frame is fully rejected. Every
    // contribution is w * local_r * c or w * local_r, so all nine taps produce
    // exactly zero and the caller's running totals are unchanged.
    //
    // Exact, including in floating point: num and den start at +0 (blit-filled)
    // and every term added is non-negative, since w = exp(...) > 0, local_r >= 0
    // and the normalized Bayer samples are >= 0. So no -0 can arise, and x + 0
    // is bit-identical to x for every value these accumulators can hold.
    if (local_r <= 0.f) return;

    float lr_mov_x = lr_x + flowx;
    float lr_mov_y = lr_y + flowy;
    if (!(lr_mov_x >= 0.f && lr_mov_x < float(p.lr_w) &&
          lr_mov_y >= 0.f && lr_mov_y < float(p.lr_h)))
        return;

    float ixx = 0.f, ixy = 0.f, iyy = 0.f;
    if (!p.iso) {
        float kmap_j, kmap_i;
        if (p.bayer) {
            kmap_j = lr_mov_x / 2.f - 0.5f;
            kmap_i = lr_mov_y / 2.f - 0.5f;
        } else {
            kmap_j = lr_mov_x - 0.5f;
            kmap_i = lr_mov_y - 0.5f;
        }
        interp_inv_cov(covs, p.cov_h, p.cov_w, kmap_i, kmap_j, ixx, ixy, iyy, true,
                       p.soften_max_inv);
    }

    int center_j = lround_away(lr_mov_x);
    int center_i = lround_away(lr_mov_y);

    float val0 = 0.f, val1 = 0.f, val2 = 0.f;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            int j = center_j + dj;
            int i = center_i + di;
            if (!(j >= 0 && j < int(p.lr_w) && i >= 0 && i < int(p.lr_h))) continue;

            int channel = cfa_channel(p, i, j);
            float c = img[uint(i) * p.lr_w + uint(j)];
            float dist_x = float(j) - lr_mov_x;
            float dist_y = float(i) - lr_mov_y;
            float z;
            if (p.iso) z = 2.f * (dist_x * dist_x + dist_y * dist_y);
            else       z = ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
            z = max(0.f, z);
            // Negligible-tap cutoff: z > 16 means w < 3.4e-4 of the centre
            // tap -- skip the exp and the accumulate entirely.
            if (p.fast_weights != 0u && z > 16.f) continue;
            // fast::exp maps to the hardware exp2 unit (a few ULP) instead of
            // precise range reduction. Called 9x per output pixel per frame, so
            // it is a measurable share of this kernel. The relative error is
            // ~1e-7, far below one 16-bit LSB; tools/compare_dng.py gates it.
            float w = fast::exp(-0.5f * z);

            float contrib_v = w * local_r * c;
            float contrib_a = w * local_r;
            if (channel == 0)      { val0 += contrib_v; acc0 += contrib_a; }
            else if (channel == 1) { val1 += contrib_v; acc1 += contrib_a; }
            else                   { val2 += contrib_v; acc2 += contrib_a; }
        }
    }

    n0 += wscale * val0; n1 += wscale * val1; n2 += wscale * val2;
    d0 += wscale * acc0; d1 += wscale * acc1; d2 += wscale * acc2;
}

// Alg. 4 dispatcher: computes lr coords and the flow hypothesis set, then
// hands each hypothesis to merge_comp_contrib_flowed.
// p.flow_bilinear: 0 nearest / 1 bilinear / 2 bicubic -- one hypothesis;
// 3 = overlapped-tile merge (Config::flow_overlap_merge): the flow buffer is
// the half-pitch measured lattice (pitch p.tile_size, tile span 2x that),
// and each output pixel crossfades its <=4 covering tiles' OWN vectors with
// the 50%-overlap raised cosine == Hann on the lattice fraction (cos^2 /
// sin^2 per axis, sums to one). Identical vectors are deduplicated, so
// smooth regions cost one gather exactly as before.
static inline void merge_comp_contrib(device const float* img,
                                      device const float* flow,
                                      device const float* covs,
                                      device const float* robustness,
                                      constant MergeCompParams& p,
                                      uint hr_j, uint local_i,
                                      thread float& n0, thread float& n1, thread float& n2,
                                      thread float& d0, thread float& d1, thread float& d2) {
    int hr_i = int(p.y0 + local_i);
    float lr_x = (float(hr_j) + 0.5f) / p.scale;
    float lr_y = (float(hr_i) + 0.5f) / p.scale;

    if (p.flow_bilinear == 3u) {
        float P = float(p.tile_size);
        float tcy = lr_y / P - 0.5f, tcx = lr_x / P - 0.5f;
        int cy0i = int(floor(tcy)), cx0i = int(floor(tcx));
        float ay = tcy - float(cy0i), ax = tcx - float(cx0i);
        int iy0 = clamp(cy0i, 0, int(p.flow_ny) - 1);
        int iy1 = clamp(cy0i + 1, 0, int(p.flow_ny) - 1);
        int ix0 = clamp(cx0i, 0, int(p.flow_nx) - 1);
        int ix1 = clamp(cx0i + 1, 0, int(p.flow_nx) - 1);
        float sy = sin(1.57079632679f * ay);
        float sx = sin(1.57079632679f * ax);
        float wy1 = sy * sy, wy0 = 1.f - wy1;
        float wx1 = sx * sx, wx0 = 1.f - wx1;
        float hx[4], hy[4], hw[4];
        const int iy[4] = {iy0, iy0, iy1, iy1};
        const int ix[4] = {ix0, ix1, ix0, ix1};
        const float w4[4] = {wy0 * wx0, wy0 * wx1, wy1 * wx0, wy1 * wx1};
        for (int a = 0; a < 4; ++a) {
            uint o = (uint(iy[a]) * p.flow_nx + uint(ix[a])) * 2u;
            hx[a] = flow[o + 0u];
            hy[a] = flow[o + 1u];
            hw[a] = w4[a];
        }
        // Cluster within a quarter pixel (weighted mean), matching the CPU
        // path: exact dedup never fires on measured lattices and cost 4x.
        for (int a = 1; a < 4; ++a)
            for (int b = 0; b < a; ++b)
                if (hw[a] > 0.f && hw[b] > 0.f &&
                    fabs(hx[a] - hx[b]) < 0.25f && fabs(hy[a] - hy[b]) < 0.25f) {
                    float wsum = hw[b] + hw[a];
                    hx[b] = (hx[b] * hw[b] + hx[a] * hw[a]) / wsum;
                    hy[b] = (hy[b] * hw[b] + hy[a] * hw[a]) / wsum;
                    hw[b] = wsum;
                    hw[a] = 0.f;
                    break;
                }
        for (int a = 0; a < 4; ++a) {
            if (hw[a] <= 0.f) continue;
            // Sub-5% hypotheses cost a full gather for a contribution the
            // den normalisation renders invisible.
            if (p.fast_weights != 0u && hw[a] < 0.05f) continue;
            merge_comp_contrib_flowed(img, covs, robustness, p, lr_x, lr_y,
                                      hx[a], hy[a], hw[a],
                                      n0, n1, n2, d0, d1, d2);
        }
        return;
    }

    // Match CPU merge.cpp: no clamp on flow tile index (pipeline pads so in-range).
    int px = int(lr_x / float(p.tile_size));
    int py = int(lr_y / float(p.tile_size));
    float flowx, flowy;
    if (p.flow_bilinear != 0u) {
        flow_sample(flow, p.flow_ny, p.flow_nx, lr_y, lr_x,
                    float(p.tile_size), p.flow_bilinear, flowx, flowy);
    } else {
        flowx = flow[(uint(py) * p.flow_nx + uint(px)) * 2u + 0u];
        flowy = flow[(uint(py) * p.flow_nx + uint(px)) * 2u + 1u];
    }
    merge_comp_contrib_flowed(img, covs, robustness, p, lr_x, lr_y,
                              flowx, flowy, 1.f, n0, n1, n2, d0, d1, d2);
}

kernel void merge_accumulate_comp(device float* num [[buffer(0)]],
                                  device float* den [[buffer(1)]],
                                  device const float* img [[buffer(2)]],
                                  device const float* flow [[buffer(3)]],
                                  device const float* covs [[buffer(4)]],
                                  device const float* robustness [[buffer(5)]],
                                  constant MergeCompParams& p [[buffer(6)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    uint hr_j = gid.x;
    uint local_i = gid.y;
    if (hr_j >= p.Ws || local_i >= p.band_h) return;

    float v0 = 0.f, v1 = 0.f, v2 = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f;
    merge_comp_contrib(img, flow, covs, robustness, p, hr_j, local_i, v0, v1, v2, a0, a1, a2);

    uint base = (local_i * p.Ws + hr_j) * p.nch;
    if (p.nch >= 1) { num[base + 0] += v0; den[base + 0] += a0; }
    if (p.nch >= 2) { num[base + 1] += v1; den[base + 1] += a1; }
    if (p.nch >= 3) { num[base + 2] += v2; den[base + 2] += a2; }
}

// Up to MERGE_FUSE_MAX frames per dispatch.
//
// The accumulators are read once into registers, every frame adds into them in
// the same order the separate dispatches used, and they are written back once.
// A float32 stored to memory and reloaded is bit-exact, so dropping the
// intermediate round trips cannot change the result -- the sequence of
// additions is identical. The band is far larger than cache, so those round
// trips were real DRAM traffic: 48 bytes per output pixel per frame.
//
// Frames beyond a group boundary simply store and reload, which is why any
// group size stays exact and the tail group can be short.
// Templated over the ACCUMULATOR STORAGE type only. All arithmetic stays
// float32 -- values are widened on load and narrowed on store -- so the float
// instantiation is bit-identical to the untemplated kernel, and the half one
// differs only by the storage quantisation of num/den. half's dynamic range
// is ample here (den <= taps * frames, num <= den * ~4 after WB, both far
// under 65504); what fp16 costs is the 10-bit mantissa, ~0.05% relative per
// store, in exchange for HALF the accumulator's footprint and half the
// merge's dominant memory traffic.
template <typename AccT>
inline void merge_accumulate_comp_x4_body(device AccT* num,
                                     device AccT* den,
                                     constant MergeCompParams* ps,
                                     constant uint& nframes,
                                     device const float* img0,
                                     device const float* img1,
                                     device const float* img2,
                                     device const float* img3,
                                     device const float* flow0,
                                     device const float* flow1,
                                     device const float* flow2,
                                     device const float* flow3,
                                     device const float* cov0,
                                     device const float* cov1,
                                     device const float* cov2,
                                     device const float* cov3,
                                     device const float* rob0,
                                     device const float* rob1,
                                     device const float* rob2,
                                     device const float* rob3,
                                     uint2 gid) {
    uint hr_j = gid.x;
    uint local_i = gid.y;
    if (hr_j >= ps[0].Ws || local_i >= ps[0].band_h) return;

    device const float* imgs[4] = {img0, img1, img2, img3};
    device const float* flows[4] = {flow0, flow1, flow2, flow3};
    device const float* covss[4] = {cov0, cov1, cov2, cov3};
    device const float* robs[4] = {rob0, rob1, rob2, rob3};

    const uint nch = ps[0].nch;
    uint base = (local_i * ps[0].Ws + hr_j) * nch;

    float n0 = 0.f, n1 = 0.f, n2 = 0.f, e0 = 0.f, e1 = 0.f, e2 = 0.f;
    if (nch >= 1) { n0 = float(num[base + 0]); e0 = float(den[base + 0]); }
    if (nch >= 2) { n1 = float(num[base + 1]); e1 = float(den[base + 1]); }
    if (nch >= 3) { n2 = float(num[base + 2]); e2 = float(den[base + 2]); }

    for (uint g = 0; g < nframes && g < 4u; ++g) {
        merge_comp_contrib(imgs[g], flows[g], covss[g], robs[g], ps[g],
                           hr_j, local_i, n0, n1, n2, e0, e1, e2);
    }

    if (nch >= 1) { num[base + 0] = AccT(n0); den[base + 0] = AccT(e0); }
    if (nch >= 2) { num[base + 1] = AccT(n1); den[base + 1] = AccT(e1); }
    if (nch >= 3) { num[base + 2] = AccT(n2); den[base + 2] = AccT(e2); }
}

#define MERGE_X4_KERNEL(NAME, ACC_T) \
kernel void NAME(device ACC_T* num [[buffer(0)]], \
                 device ACC_T* den [[buffer(1)]], \
                 constant MergeCompParams* ps [[buffer(2)]], \
                 constant uint& nframes [[buffer(3)]], \
                 device const float* img0 [[buffer(4)]], \
                 device const float* img1 [[buffer(5)]], \
                 device const float* img2 [[buffer(6)]], \
                 device const float* img3 [[buffer(7)]], \
                 device const float* flow0 [[buffer(8)]], \
                 device const float* flow1 [[buffer(9)]], \
                 device const float* flow2 [[buffer(10)]], \
                 device const float* flow3 [[buffer(11)]], \
                 device const float* cov0 [[buffer(12)]], \
                 device const float* cov1 [[buffer(13)]], \
                 device const float* cov2 [[buffer(14)]], \
                 device const float* cov3 [[buffer(15)]], \
                 device const float* rob0 [[buffer(16)]], \
                 device const float* rob1 [[buffer(17)]], \
                 device const float* rob2 [[buffer(18)]], \
                 device const float* rob3 [[buffer(19)]], \
                 uint2 gid [[thread_position_in_grid]]) { \
    merge_accumulate_comp_x4_body<ACC_T>(num, den, ps, nframes, \
        img0, img1, img2, img3, flow0, flow1, flow2, flow3, \
        cov0, cov1, cov2, cov3, rob0, rob1, rob2, rob3, gid); \
}

MERGE_X4_KERNEL(merge_accumulate_comp_x4, float)
MERGE_X4_KERNEL(merge_accumulate_comp_x4_h, half)
#undef MERGE_X4_KERNEL

template <typename AccT>
inline void merge_accumulate_ref_body(device AccT* num,
                                 device AccT* den,
                                 device const float* img,
                                 device const float* covs,
                                 device const float* acc_rob,
                                 constant MergeRefParams& p,
                                 uint2 gid) {
    uint hr_j = gid.x;
    uint local_i = gid.y;
    if (hr_j >= p.Ws || local_i >= p.band_h) return;

    int hr_i = int(p.y0 + local_i);
    float coarse_x = float(hr_j) / p.scale;
    float coarse_y = float(hr_i) / p.scale;

    float local_acc_r = 0.f;
    float additional_denoise_power = 1.f;
    int rad = 1;
    if (p.robustness_denoise) {
        // C++ std::lround — Metal round() is half-away-from-zero (same for >=0)
        float acc_y = coarse_y, acc_x = coarse_x;
        if (p.raw_res_robustness == 0u && p.bayer != 0u) {
            acc_y = (coarse_y - 0.5f) / 2.f;
            acc_x = (coarse_x - 0.5f) / 2.f;
        }
        int ay = min(max(lround_away(acc_y), 0), int(p.acc_h) - 1);
        int ax = min(max(lround_away(acc_x), 0), int(p.acc_w) - 1);
        local_acc_r = acc_rob[uint(ay) * p.acc_w + uint(ax)];
        if (p.adaptive != 0u) {
            // Must match denoise_power_merge / denoise_range_merge in merge.cpp.
            // m = N / (r_acc + 1): kernel area scales as m, merging k frames cuts
            // noise variance by 1/k, and the reference always contributes so the
            // effective count here is r_acc + 1. No threshold, and it normalises
            // itself to burst length.
            if (p.burst_frames > 1.f) {
                float effective = max(1.f, local_acc_r + 1.f);
                float cap = max(1.f, p.max_multiplier);
                additional_denoise_power = min(max(p.burst_frames / effective, 1.f), cap);
            } else {
                additional_denoise_power = 1.f;
            }
            // sigma grows as sqrt(m), so the window must too or the widened
            // kernel is truncated. Floored under two contributing frames:
            // chroma is a sampling problem, not a noise one, and a 3x3 Bayer
            // window can hold a single red sample however short the burst was.
            int r = int(round(sqrt(max(1.f, additional_denoise_power))));
            if (local_acc_r + 1.f < 2.f) r = max(r, int(p.rad_max));
            rad = min(max(r, 1), max(1, int(p.rad_max)));
        } else {
            // Reference behaviour: step at the frame-count threshold.
            additional_denoise_power =
                (local_acc_r <= p.max_frame_count) ? p.max_multiplier : 1.f;
            rad = (local_acc_r <= p.max_frame_count) ? int(p.rad_max) : 1;
        }
    }

    float ixx = 0.f, ixy = 0.f, iyy = 0.f;
    if (!p.iso) {
        float kmap_j, kmap_i;
        if (p.bayer) {
            kmap_j = (coarse_x - 0.5f) / 2.f;
            kmap_i = (coarse_y - 0.5f) / 2.f;
        } else {
            kmap_j = coarse_x;
            kmap_i = coarse_y;
        }
        interp_inv_cov(covs, p.cov_h, p.cov_w, kmap_i, kmap_j, ixx, ixy, iyy, false,
                       p.soften_max_inv);
    }

    int center_j = int(round(coarse_x));
    int center_i = int(round(coarse_y));

    float val0 = 0.f, val1 = 0.f, val2 = 0.f;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
    for (int di = -rad; di <= rad; ++di) {
        for (int dj = -rad; dj <= rad; ++dj) {
            int j = center_j + dj;
            int i = center_i + di;
            if (!(j >= 0 && j < int(p.lr_w) && i >= 0 && i < int(p.lr_h))) continue;

            int channel = cfa_channel_ref(p, i, j);
            float c = img[uint(i) * p.lr_w + uint(j)];
            float dist_x = float(j) - coarse_x;
            float dist_y = float(i) - coarse_y;
            float y;
            if (p.iso) y = max(0.f, 2.f * (dist_x * dist_x + dist_y * dist_y));
            else       y = max(0.f, ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y +
                                    iyy * dist_y * dist_y);
            y /= additional_denoise_power;
            float w = fast::exp(-0.5f * y); // see merge_accumulate_comp
            // Coverage floor -- twin of accumulate_ref in merge.cpp. Keyed
            // on the ceiling in the params (>64), always active at 128.
            if (p.soften_max_inv > 64.f)
                w += 5e-4f * fast::exp(-0.5f * (dist_x * dist_x +
                                                dist_y * dist_y));

            if (channel == 0)      { val0 += c * w; acc0 += w; }
            else if (channel == 1) { val1 += c * w; acc1 += w; }
            else                   { val2 += c * w; acc2 += w; }
        }
    }

    // See the matching note in accumulate_ref: the overwrite belongs to the
    // reference's step, and the adaptive path always accumulates.
    bool overwrite = p.adaptive == 0u && p.robustness_denoise &&
                     (local_acc_r < p.max_frame_count);
    uint base = (local_i * p.Ws + hr_j) * p.nch;
    if (overwrite) {
        if (p.nch >= 1) { num[base + 0] = AccT(val0); den[base + 0] = AccT(acc0); }
        if (p.nch >= 2) { num[base + 1] = AccT(val1); den[base + 1] = AccT(acc1); }
        if (p.nch >= 3) { num[base + 2] = AccT(val2); den[base + 2] = AccT(acc2); }
    } else {
        if (p.nch >= 1) { num[base + 0] = AccT(float(num[base + 0]) + val0); den[base + 0] = AccT(float(den[base + 0]) + acc0); }
        if (p.nch >= 2) { num[base + 1] = AccT(float(num[base + 1]) + val1); den[base + 1] = AccT(float(den[base + 1]) + acc1); }
        if (p.nch >= 3) { num[base + 2] = AccT(float(num[base + 2]) + val2); den[base + 2] = AccT(float(den[base + 2]) + acc2); }
    }
}

#define MERGE_REF_KERNEL(NAME, ACC_T) \
kernel void NAME(device ACC_T* num [[buffer(0)]], \
                 device ACC_T* den [[buffer(1)]], \
                 device const float* img [[buffer(2)]], \
                 device const float* covs [[buffer(3)]], \
                 device const float* acc_rob [[buffer(4)]], \
                 constant MergeRefParams& p [[buffer(5)]], \
                 uint2 gid [[thread_position_in_grid]]) { \
    merge_accumulate_ref_body<ACC_T>(num, den, img, covs, acc_rob, p, gid); \
}

MERGE_REF_KERNEL(merge_accumulate_ref, float)
MERGE_REF_KERNEL(merge_accumulate_ref_h, half)
#undef MERGE_REF_KERNEL

// Stage one band of the half accumulator out to float for the host encoder,
// which stays byte-for-byte unchanged. cb.x = element count, cb.y = element
// offset of the band's first entry -- an offset PARAMETER rather than a
// buffer-binding offset, so no alignment constraint applies to odd bands.
kernel void merge_acc_half_to_float(device const half* num_h [[buffer(0)]],
                                    device const half* den_h [[buffer(1)]],
                                    device float* num_f [[buffer(2)]],
                                    device float* den_f [[buffer(3)]],
                                    constant uint2& cb [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= cb.x) return;
    num_f[gid] = float(num_h[cb.y + gid]);
    den_f[gid] = float(den_h[cb.y + gid]);
}

// =============================================================================
// Alg. 5 — estimate_kernels (matches kernels.cpp exactly)
// =============================================================================

struct KernelEstParams {
    uint raw_h, raw_w, grey_h, grey_w;
    uint bayer;     // 1 = decimate 2x2 raw to grey before GAT
    uint selection; // retained for CPU layout; 460-main always hard-thresholds
    float alpha, beta;
    float k_detail, k_denoise, D_th, D_tr, k_stretch, k_shrink;
    uint aniso_continuous;  // 1 = drive Eq. 4's shape continuously (was _pad0)
    uint _pad1;
};

inline float gat_sample(float v, float alpha, float beta) {
    // apply_gat: c = 0.375*alpha^2 + beta; out = (2/alpha)*sqrt(max(0, alpha*v+c))
    float c = 0.375f * alpha * alpha + beta;
    float t = alpha * v + c;
    return (2.f / alpha) * sqrt(max(0.f, t));
}

// linalg.h real_polyroots_2 / eigen_val / eigen_vect / eigen_elmts
inline void real_polyroots_2(float a, float b, float c, thread float roots[2]) {
    float delta = b * b - 4.f * a * c;
    float r1 = (-b + sqrt(delta)) / (2.f * a);
    float r2 = (-b - sqrt(delta)) / (2.f * a);
    if (fabs(r1) >= fabs(r2)) { roots[0] = r1; roots[1] = r2; }
    else                      { roots[0] = r2; roots[1] = r1; }
}

inline void eigen_elmts_2x2(float m00, float m01, float m10, float m11,
                            thread float l[2], thread float e1[2], thread float e2[2]) {
    float b = -(m00 + m11);
    float c = m00 * m11 - m01 * m10;
    real_polyroots_2(1.f, b, c, l);

    if (m01 == 0.f && m00 == m11) {
        e1[0] = 1.f; e1[1] = 0.f;
        e2[0] = 0.f; e2[1] = 1.f;
        return;
    }
    e1[0] = m00 + m01 - l[1];
    e1[1] = m10 + m11 - l[1];
    if (e1[0] == 0.f) {
        e1[1] = 1.f;
        e2[0] = 1.f;
        e2[1] = 0.f;
    } else if (e1[1] == 0.f) {
        e1[0] = 1.f;
        e2[0] = 0.f;
        e2[1] = 1.f;
    } else {
        float norm_ = sqrt(e1[0] * e1[0] + e1[1] * e1[1]);
        e1[0] /= norm_;
        e1[1] /= norm_;
        float sign = copysign(1.f, e1[0]);
        e2[1] = fabs(e1[0]);
        e2[0] = -e1[1] * sign;
    }
}

// Eq. 4's k1/k2. Twin of compute_k in core/kernels.cpp -- keep them in step.
// The paper drives anisotropy continuously from (l1-l2)/(l1+l2); 460-main
// switched at A > 1.95 (anisotropy > 0.9025) and was round below it, which
// denied the misalignment tolerance of Section 5.1.1 to every moderately
// anisotropic feature. The flat-patch guard is deliberate: the old 0/0 gave
// NaN, and NaN > 1.95 being false fell to isotropic by accident, whereas a
// continuous blend would propagate the NaN into k1/k2.
inline void compute_k_cpu(float l1, float l2, thread float& k1, thread float& k2,
                          constant KernelEstParams& p) {
    float tr = l1 + l2;
    float ratio = (tr > 1e-12f) ? max(0.f, (l1 - l2) / tr) : 0.f;
    if (!isfinite(ratio)) ratio = 0.f;
    float a = sqrt(ratio);   // only the legacy threshold uses this
    float D = min(1.f, max(0.f, 1.f - sqrt(max(0.f, l1)) / p.D_tr + p.D_th));
    float kk1, kk2;
    if (p.aniso_continuous != 0u) {
        // Blend on the RATIO, not sqrt(ratio): at low anisotropy the dominant
        // eigenvector is noise-dominated, and stretching along it smears in an
        // arbitrary direction. Matches compute_k in kernels.cpp.
        kk1 = 1.f + ratio * (1.f / p.k_shrink - 1.f);
        kk2 = 1.f + ratio * (p.k_stretch - 1.f);
    } else if (1.f + a > 1.95f) {
        kk1 = 1.f / p.k_shrink; kk2 = p.k_stretch;
    } else {
        kk1 = 1.f; kk2 = 1.f;
    }
    k1 = p.k_detail * ((1.f - D) * kk1 + D * p.k_denoise);
    k2 = p.k_detail * ((1.f - D) * kk2 + D * p.k_denoise);
}

kernel void kernel_gat(device float* out [[buffer(0)]],
                       device const float* grey [[buffer(1)]],
                       constant KernelEstParams& p [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.grey_w || gid.y >= p.grey_h) return;
    uint i = gid.y * p.grey_w + gid.x;
    out[i] = gat_sample(grey[i], p.alpha, p.beta);
}

kernel void kernel_decimate_grey(device float* grey [[buffer(0)]],
                                 device const float* raw [[buffer(1)]],
                                 constant KernelEstParams& p [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.grey_w || gid.y >= p.grey_h) return;
    uint y = gid.y, x = gid.x;
    if (p.bayer) {
        uint y0 = y * 2u, x0 = x * 2u;
        float s = raw[y0 * p.raw_w + x0] + raw[y0 * p.raw_w + x0 + 1u] +
                  raw[(y0 + 1u) * p.raw_w + x0] + raw[(y0 + 1u) * p.raw_w + x0 + 1u];
        grey[y * p.grey_w + x] = 0.25f * s;
    } else {
        grey[y * p.grey_w + x] = raw[y * p.raw_w + x];
    }
}

kernel void kernel_gradients(device float* grad [[buffer(0)]],
                             device const float* grey [[buffer(1)]],
                             constant KernelEstParams& p [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint gh = p.grey_h - 1u, gw = p.grey_w - 1u;
    if (gid.x >= gw || gid.y >= gh) return;
    uint y = gid.y, x = gid.x;
    float tl = grey[y * p.grey_w + x];
    float tr = grey[y * p.grey_w + x + 1u];
    float bl = grey[(y + 1u) * p.grey_w + x];
    float br = grey[(y + 1u) * p.grey_w + x + 1u];
    uint o = (y * gw + x) * 2u;
    grad[o + 0u] = 0.25f * ((tr - tl) + (br - bl));
    grad[o + 1u] = 0.25f * ((bl - tl) + (br - tr));
}

kernel void kernel_estimate_cov(device float* covs [[buffer(0)]],
                                device const float* grad [[buffer(1)]],
                                constant KernelEstParams& p [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.grey_w || gid.y >= p.grey_h) return;
    int y = int(gid.y), x = int(gid.x);
    int grad_h = int(p.grey_h) - 1;
    int grad_w = int(p.grey_w) - 1;

    float s00 = 0.f, s01 = 0.f, s11 = 0.f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int gy = y - 1 + i, gx = x - 1 + j;
            if (gy < 0 || gy >= grad_h || gx < 0 || gx >= grad_w) continue;
            uint gi = (uint(gy) * uint(grad_w) + uint(gx)) * 2u;
            float gxv = grad[gi + 0u];
            float gyv = grad[gi + 1u];
            s00 += gxv * gxv;
            s01 += gxv * gyv;
            s11 += gyv * gyv;
        }
    }

    float l[2], e1[2], e2[2];
    eigen_elmts_2x2(s00, s01, s01, s11, l, e1, e2);

    float k1, k2;
    compute_k_cpu(l[0], l[1], k1, k2, p);

    float k1s = k1 * k1, k2s = k2 * k2;
    uint base = (gid.y * p.grey_w + gid.x) * 4u;
    covs[base + 0u] = k1s * e1[0] * e1[0] + k2s * e2[0] * e2[0];
    covs[base + 1u] = k1s * e1[0] * e1[1] + k2s * e2[0] * e2[1];
    covs[base + 2u] = covs[base + 1u];
    covs[base + 3u] = k1s * e1[1] * e1[1] + k2s * e2[1] * e2[1];
}

// =============================================================================
// Robustness — 1:1 with robustness.cpp (noise curves stay on CPU / uploaded LUT)
// =============================================================================

struct RobGuideParams {
    uint raw_h, raw_w, guide_h, guide_w;
    uint bayer; // 1 = Bayer → RGB guide
    uint cfa00, cfa01, cfa10, cfa11;
    float wb0, wb1, wb2;
    uint _pad0;
};

struct RobStatsParams {
    uint h, w, nch;
    uint _pad0;
};

struct RobDogsonParams {
    uint in_h, in_w, out_h, out_w, nch;
    uint is_ref;      // 1 = no flow
    uint tile_size;
    uint flow_ny, flow_nx;
    float s;          // always 2.f (Python CUDA hardcode)
    uint flow_bilinear;  // 1 = interpolate the tile flow (was _pad0)
    uint _pad1;
};

struct RobMaskParams {
    uint h, w, nch;
    uint tile_size;
    uint flow_ny;
    uint flow_nx;
    uint curve_n;     // 1001
    uint bayer;
    float r_t;
    float r_s1;   // motion prior applied to ambiguous tiles
    // Field order and size must stay in lockstep with RobMaskParamsCPU in
    // metal_gpu.mm, which static_asserts the size.
    uint ambiguous_enabled;  // 1 = demote tiles whose BM match was ambiguous
    uint flow_bilinear;  // 1 = interpolate the tile flow
};

inline float dogson_quadratic(float x) {
    float ax = fabs(x);
    if (ax <= 0.5f) return -2.f * ax * ax + 1.f;
    if (ax <= 1.5f) return ax * ax - 2.5f * ax + 1.5f;
    return 0.f;
}

inline int clamp_edge(int v, int hi) {
    float f = clamp(float(v), 0.f, float(hi));
    return int(f);
}

inline float rob_sample_bilinear_or_inf(device const float* img,
                                        uint h, uint w, uint nch,
                                        float y, float x, uint ch) {
    if (!(y >= 0.f && y < float(h) && x >= 0.f && x < float(w))) return INFINITY;
    int y0 = int(floor(y));
    int x0 = int(floor(x));
    int y1 = min(y0 + 1, int(h) - 1);
    int x1 = min(x0 + 1, int(w) - 1);
    float fy = y - float(y0);
    float fx = x - float(x0);
    uint o00 = (uint(y0) * w + uint(x0)) * nch + ch;
    uint o01 = (uint(y0) * w + uint(x1)) * nch + ch;
    uint o10 = (uint(y1) * w + uint(x0)) * nch + ch;
    uint o11 = (uint(y1) * w + uint(x1)) * nch + ch;
    float top = img[o00] + (img[o01] - img[o00]) * fx;
    float bot = img[o10] + (img[o11] - img[o10]) * fx;
    return top + (bot - top) * fy;
}

kernel void rob_guide_bayer(device float* guide [[buffer(0)]],
                            device const float* raw [[buffer(1)]],
                            constant RobGuideParams& p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.guide_w || gid.y >= p.guide_h) return;
    uint cfa[2][2] = {{p.cfa00, p.cfa01}, {p.cfa10, p.cfa11}};
    uint o = (gid.y * p.guide_w + gid.x) * 3u;
    // Divisor per colour taken from the CFA rather than assumed -- see
    // compute_guide in robustness.cpp, which this mirrors, and
    // Config::noise_guide_weight, which must agree with it. Bit-identical to
    // the previous 0.5*gsum for any Bayer pattern.
    float sum[3] = {0.f, 0.f, 0.f};
    uint cnt[3] = {0u, 0u, 0u};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            uint c = cfa[i][j];
            if (c > 2u) continue;
            sum[c] += raw[(gid.y * 2u + uint(i)) * p.raw_w + (gid.x * 2u + uint(j))];
            cnt[c] += 1u;
        }
    }
    // p.wb0..2 carry the WB UNDO factor (1/gain per channel, 1.0 when the
    // raw was not pre-whitebalanced): the guide's statistics live in sensor
    // space -- see Config::guide_wb_undo and compute_guide in robustness.cpp,
    // which this mirrors.
    float undo[3] = {p.wb0, p.wb1, p.wb2};
    for (uint c = 0u; c < 3u; ++c)
        guide[o + c] = (cnt[c] > 0u) ? (sum[c] / float(cnt[c])) * undo[c] : 0.f;
}

kernel void rob_local_stats_3x3(device float* means [[buffer(0)]],
                                device float* vars [[buffer(1)]],
                                device const float* guide [[buffer(2)]],
                                constant RobStatsParams& p [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    int y = int(gid.y), x = int(gid.x);
    int H = int(p.h), W = int(p.w);
    for (uint ch = 0u; ch < p.nch; ++ch) {
        float s = 0.f, s2 = 0.f;
        for (int i = -1; i <= 1; ++i) {
            int yy = clamp_edge(y + i, H - 1);
            for (int j = -1; j <= 1; ++j) {
                int xx = clamp_edge(x + j, W - 1);
                float v = guide[(uint(yy) * p.w + uint(xx)) * p.nch + ch];
                s += v;
                s2 += v * v;
            }
        }
        float m = s / 9.f;
        uint o = (gid.y * p.w + gid.x) * p.nch + ch;
        means[o] = m;
        vars[o] = s2 / 9.f - m * m;
    }
}

kernel void rob_upscale_dogson(device float* out [[buffer(0)]],
                               device const float* stats [[buffer(1)]],
                               device const float* flow [[buffer(2)]],
                               constant RobDogsonParams& p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.out_w || gid.y >= p.out_h) return;
    int y = int(gid.y), x = int(gid.x);
    float flow_x = 0.f, flow_y = 0.f;
    if (p.is_ref == 0u && p.tile_size > 0u && p.flow_ny > 0u && p.flow_nx > 0u) {
        if (p.flow_bilinear != 0u) {
            flow_sample(flow, p.flow_ny, p.flow_nx, float(y), float(x),
                        float(p.tile_size), p.flow_bilinear, flow_x, flow_y);
        } else {
            int patch_idy = y / int(p.tile_size);
            int patch_idx = x / int(p.tile_size);
            if (patch_idy >= 0 && patch_idy < int(p.flow_ny) &&
                patch_idx >= 0 && patch_idx < int(p.flow_nx)) {
                uint fi = (uint(patch_idy) * p.flow_nx + uint(patch_idx)) * 2u;
                flow_x = flow[fi + 0u];
                flow_y = flow[fi + 1u];
            }
        }
    }
    float LR_y = (float(y) + flow_y + 0.5f) / p.s - 0.5f;
    float LR_x = (float(x) + flow_x + 0.5f) / p.s - 0.5f;

    for (uint ch = 0u; ch < p.nch; ++ch) {
        float val;
        if (!(LR_y >= 0.f && LR_y < float(p.in_h) && LR_x >= 0.f && LR_x < float(p.in_w))) {
            val = INFINITY;
        } else {
            int center_y = lround_away(LR_y);
            int center_x = lround_away(LR_x);
            float w_acc = 0.f, buf = 0.f;
            for (int i = -1; i <= 1; ++i) {
                int y_ = clamp_edge(center_y + i, int(p.in_h) - 1);
                float dy = float(y_) - LR_y;
                float wy = dogson_quadratic(dy);
                for (int j = -1; j <= 1; ++j) {
                    int x_ = clamp_edge(center_x + j, int(p.in_w) - 1);
                    float dx = float(x_) - LR_x;
                    float w = wy * dogson_quadratic(dx);
                    buf += stats[(uint(y_) * p.in_w + uint(x_)) * p.nch + ch] * w;
                    w_acc += w;
                }
            }
            val = buf / w_acc;
        }
        out[(gid.y * p.out_w + gid.x) * p.nch + ch] = val;
    }
}

kernel void rob_make_mask(device float* R [[buffer(0)]],
                          device const float* comp_means [[buffer(1)]],
                          device const float* ref_means [[buffer(2)]],
                          device const float* ref_vars [[buffer(3)]],
                          device const float* std_curve [[buffer(4)]],
                          device const float* diff_curve [[buffer(5)]],
                          device const float* S [[buffer(6)]],
                          device const float* flow [[buffer(7)]],
                          constant RobMaskParams& p [[buffer(8)]],
                          device const uint* match_ambiguous [[buffer(9)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    float d_sq_ = 0.f, sigma_sq_ = 0.f;
    int patch_idy;
    int patch_idx;
    float flow_x;
    float flow_y;
    // Same sampling as the merge kernel: the mask must score the
    // correspondence the merge will actually fetch.
    if (p.nch == 1u) {
        patch_idy = int(gid.y) / int(p.tile_size);
        patch_idx = int(gid.x) / int(p.tile_size);
        if (p.flow_bilinear != 0u) {
            flow_sample(flow, p.flow_ny, p.flow_nx, float(gid.y),
                        float(gid.x), float(p.tile_size), p.flow_bilinear,
                        flow_x, flow_y);
        } else {
            uint fi = (uint(patch_idy) * p.flow_nx + uint(patch_idx)) * 2u;
            flow_x = flow[fi + 0u];
            flow_y = flow[fi + 1u];
        }
    } else {
        patch_idy = int((2.f * float(gid.y) + 0.5f) / float(p.tile_size));
        patch_idx = int((2.f * float(gid.x) + 0.5f) / float(p.tile_size));
        if (p.flow_bilinear != 0u) {
            float rdx, rdy;
            flow_sample(flow, p.flow_ny, p.flow_nx,
                        2.f * float(gid.y) + 0.5f,
                        2.f * float(gid.x) + 0.5f,
                        float(p.tile_size), p.flow_bilinear, rdx, rdy);
            flow_x = 0.5f * rdx; flow_y = 0.5f * rdy;
        } else {
            uint fi = (uint(patch_idy) * p.flow_nx + uint(patch_idx)) * 2u;
            flow_x = 0.5f * flow[fi + 0u];
            flow_y = 0.5f * flow[fi + 1u];
        }
    }
    float sample_x = float(gid.x) + flow_x;
    float sample_y = float(gid.y) + flow_y;
    // Eq. 6 aggregates each term into ONE scalar across channels first
    // (sigma = sqrt(sum of per-channel variances); d/d_ms/d_md are bare
    // per-pixel scalars, not per-channel), and only then applies max()/
    // shrinkage once -- see the mirrored comment in apply_noise_model
    // (robustness.cpp). Summing per-channel max(measured, noise-floor)
    // instead of max-of-sums is not the same computation and is
    // systematically more forgiving whenever which term dominates differs
    // across channels (e.g. a colored edge).
    float sigma_ms_sq = 0.f, sigma_md_sq = 0.f;
    float d_ms_sq = 0.f, d_md_sq = 0.f;
    for (uint ch = 0u; ch < p.nch; ++ch) {
        uint o = (gid.y * p.w + gid.x) * p.nch + ch;
        float brightness = ref_means[o];
        // Python: id_noise = round(1000 * brightness) — no clamp.
        int id_noise = lround_away(1000.f * brightness);
        // GPU-only: Python OOBs on non-finite / out-of-range; avoid Metal faults.
        // Finite brightness in [0,1] -> id in [0,1000] unchanged (same as Python).
        if (!isfinite(brightness))
            id_noise = 0;
        else if (id_noise < 0)
            id_noise = 0;
        else if (id_noise >= int(p.curve_n))
            id_noise = int(p.curve_n) - 1;
        uint id = uint(id_noise);
        // std_curve/diff_curve hold up to 3 concatenated per-channel curves
        // (metal_gpu.mm), each ch its own -- not one shared by every guide
        // channel. See Config::noise_alpha_ch/noise_beta_ch (types.h).
        uint curve_id = ch * p.curve_n + id;
        float sigma_t = std_curve[curve_id];
        float d_t = diff_curve[curve_id];
        sigma_ms_sq += ref_vars[o];
        sigma_md_sq += sigma_t * sigma_t;
        float comp = rob_sample_bilinear_or_inf(comp_means, p.h, p.w, p.nch,
                                                sample_y, sample_x, ch);
        float d_p_ = isfinite(comp) ? fabs(ref_means[o] - comp) : INFINITY;
        d_ms_sq += d_p_ * d_p_;
        d_md_sq += d_t * d_t;
    }
    sigma_sq_ = max(sigma_ms_sq, sigma_md_sq);
    float shrink = d_ms_sq / (d_ms_sq + d_md_sq);
    d_sq_ = d_ms_sq * shrink * shrink;
    float s = S[uint(patch_idy) * p.flow_nx + uint(patch_idx)];
    float sig = sigma_sq_;
    uint pidx = uint(patch_idy) * p.flow_nx + uint(patch_idx);
    // Two near-equal minima in the block-matching cost surface: the offset that
    // was picked is not distinguishable from another. See compute_robustness in
    // robustness.cpp -- this is the only mask input that is not derived from the
    // image residual, and the residual cannot see this failure because the wrong
    // offset was chosen for producing a small difference.
    if (p.ambiguous_enabled != 0u && match_ambiguous[pidx] != 0u)
        s = min(s, p.r_s1);
    float r_val = clamp(s * exp(-d_sq_ / sig) - p.r_t, 0.f, 1.f);
    R[gid.y * p.w + gid.x] = r_val;
}

struct RobMaskRawParams {
    uint h, w, nch;
    uint tile_size;
    uint flow_ny, flow_nx;
    uint curve_n;
    float r_t;
    float r_s1;
    uint ambiguous_enabled;
    uint chain_reject_enabled;
    float r_s_chain;
    uint motion_magnitude_veto_enabled;
    uint _pad0;
};

// Algorithm 6, read literally: ref_means/ref_vars/comp_means are already at
// RAW resolution here (Dodgson-quadratic upscaled from guide resolution by
// rob_upscale_dogson, comp_means additionally warped by the flow in that
// same pass -- see rob_dogson in metal_gpu.mm), so this kernel needs no
// guide/raw branching and no per-pixel flow-shift/bilinear-sample the way
// rob_make_mask does: every raw pixel of comp_means already sits in the
// reference's coordinate frame. See Config::robustness_raw_resolution_
// enabled and the mirrored, more heavily-commented CPU implementation,
// compute_robustness_raw_res in robustness.cpp.
kernel void rob_make_mask_raw(device float* R [[buffer(0)]],
                              device const float* comp_means [[buffer(1)]],
                              device const float* ref_means [[buffer(2)]],
                              device const float* ref_vars [[buffer(3)]],
                              device const float* std_curve [[buffer(4)]],
                              device const float* diff_curve [[buffer(5)]],
                              device const float* S [[buffer(6)]],
                              constant RobMaskRawParams& p [[buffer(7)]],
                              device const uint* match_ambiguous [[buffer(8)]],
                              device const uint* chain_inconsistent [[buffer(9)]],
                              device const uint* motion_magnitude_reject [[buffer(10)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    uint patch_idy = gid.y / p.tile_size;
    uint patch_idx = gid.x / p.tile_size;
    uint out_o = gid.y * p.w + gid.x;
    if (patch_idy >= p.flow_ny || patch_idx >= p.flow_nx) {
        R[out_o] = 0.f;
        return;
    }
    uint pidx = patch_idy * p.flow_nx + patch_idx;

    // Same aggregate-then-max / aggregate-then-shrink as rob_make_mask's
    // fixed form -- see the comment there and in apply_noise_model
    // (robustness.cpp).
    float sigma_ms_sq = 0.f, sigma_md_sq = 0.f;
    float d_ms_sq = 0.f, d_md_sq = 0.f;
    for (uint ch = 0u; ch < p.nch; ++ch) {
        uint o = out_o * p.nch + ch;
        float brightness = ref_means[o];
        int id_noise = lround_away(1000.f * brightness);
        if (!isfinite(brightness))
            id_noise = 0;
        else if (id_noise < 0)
            id_noise = 0;
        else if (id_noise >= int(p.curve_n))
            id_noise = int(p.curve_n) - 1;
        uint id = uint(id_noise);
        uint curve_id = ch * p.curve_n + id;
        float sigma_t = std_curve[curve_id];
        float d_t = diff_curve[curve_id];
        sigma_ms_sq += ref_vars[o];
        sigma_md_sq += sigma_t * sigma_t;
        float comp = comp_means[o];
        float d_p_ = isfinite(comp) ? fabs(ref_means[o] - comp) : INFINITY;
        d_ms_sq += d_p_ * d_p_;
        d_md_sq += d_t * d_t;
    }
    float sigma_sq_ = max(sigma_ms_sq, sigma_md_sq);
    float shrink = d_ms_sq / (d_ms_sq + d_md_sq);
    float d_sq_ = d_ms_sq * shrink * shrink;

    float s = S[pidx];
    if (p.ambiguous_enabled != 0u && match_ambiguous[pidx] != 0u)
        s = min(s, p.r_s1);
    if (p.chain_reject_enabled != 0u && chain_inconsistent[pidx] != 0u)
        s = min(s, p.r_s_chain);
    bool motion_magnitude_reject_tile =
        p.motion_magnitude_veto_enabled != 0u &&
        motion_magnitude_reject[pidx] != 0u;

    float r_val = motion_magnitude_reject_tile
        ? 0.f
        : clamp(s * exp(-d_sq_ / sigma_sq_) - p.r_t, 0.f, 1.f);
    // An OOB Dodgson sample arrives as +inf in comp_means -> d_sq_ = +inf ->
    // shrink inf/inf = NaN -> r_val NaN. Metal's clamp() propagates NaN; the
    // Python reference's min/max clamp yields the intended 0. Mirror the CPU
    // guard in compute_robustness_raw_res.
    if (!isfinite(r_val)) r_val = 0.f;
    R[out_o] = r_val;
}

kernel void rob_local_min_5x5(device float* out [[buffer(0)]],
                              device const float* R [[buffer(1)]],
                              constant RobStatsParams& p [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    int y = int(gid.y), x = int(gid.x);
    int H = int(p.h), W = int(p.w);
    float mn = INFINITY;
    for (int i = -2; i <= 2; ++i) {
        int yy = clamp_edge(y + i, H - 1);
        for (int j = -2; j <= 2; ++j) {
            int xx = clamp_edge(x + j, W - 1);
            mn = min(mn, R[uint(yy) * p.w + uint(xx)]);
        }
    }
    out[gid.y * p.w + gid.x] = mn;
}

// L1 BM for ts==16: one thread per tile. Per-shift costs use the same
// warp-then-block reduce order as align.cpp; argmin matches the Python
// CUDA bug (err fixed at s_err[0], update when err < min_v).
struct L1BmParams {
    uint ref_h, ref_w, mov_h, mov_w;
    uint ny, nx, ts, R;
};

inline float cuda_shfl_down_warp_sum_lane0_local(thread float* lane /*32*/) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float next[32];
        for (int i = 0; i < 32; ++i) {
            int src = i + offset;
            float shfl = (src < 32) ? lane[src] : lane[i];
            next[i] = lane[i] + shfl;
        }
        for (int i = 0; i < 32; ++i) lane[i] = next[i];
    }
    return lane[0];
}

inline float warp_then_block_reduce_256(thread float* vals /*256*/) {
    float warp_sums[8];
    for (int w = 0; w < 8; ++w)
        warp_sums[w] = cuda_shfl_down_warp_sum_lane0_local(vals + w * 32);
    float total = warp_sums[0];
    for (int w = 1; w < 8; ++w) total += warp_sums[w];
    return total;
}

inline float warp_then_block_reduce_1024(thread float* vals /*1024*/) {
    float warp_sums[32];
    for (int w = 0; w < 32; ++w)
        warp_sums[w] = cuda_shfl_down_warp_sum_lane0_local(vals + w * 32);
    float total = warp_sums[0];
    for (int w = 1; w < 32; ++w) total += warp_sums[w];
    return total;
}

kernel void l1_bm_ts16(device const float* ref [[buffer(0)]],
                       device const float* mov [[buffer(1)]],
                       device float* flow [[buffer(2)]],
                       constant L1BmParams& p [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= p.ny * p.nx) return;
    uint ty = tid / p.nx;
    uint tx = tid % p.nx;
    int ts = int(p.ts);
    int R = int(p.R);
    int corr = 2 * R + 1;
    int oy = int(ty) * ts;
    int ox = int(tx) * ts;

    // CUDA round(): half away from zero (Metal round matches).
    float fdx = flow[tid * 2u + 0u];
    float fdy = flow[tid * 2u + 1u];
    int flow_dx = int(round(fdx));
    int flow_dy = int(round(fdy));

    float s_err[289]; // Python ts==16 supports 2*R+16 <= 32 -> corr <= 17.
    if (corr > 17) return;

    float per[256];
    for (int sdy = -R; sdy <= R; ++sdy) {
        for (int sdx = -R; sdx <= R; ++sdx) {
            for (int i = 0; i < ts; ++i) {
                int ry = oy + i;
                for (int j = 0; j < ts; ++j) {
                    int rx = ox + j;
                    int tidp = i * ts + j;
                    float rv = (ry < int(p.ref_h) && rx < int(p.ref_w))
                                   ? ref[uint(ry) * p.ref_w + uint(rx)] : 0.f;
                    // cuda_L1_local_search16: shared load omits -R, index adds +R
                    // → sample at (ry+flow+sdy+R, rx+flow+sdx+R).
                    int my = ry + flow_dy + sdy + R;
                    int mx = rx + flow_dx + sdx + R;
                    float mv = (my >= 0 && my < int(p.mov_h) && mx >= 0 && mx < int(p.mov_w))
                                   ? mov[uint(my) * p.mov_w + uint(mx)] : 0.f;
                    per[tidp] = fabs(rv - mv);
                }
            }
            s_err[(sdy + R) * corr + (sdx + R)] = warp_then_block_reduce_256(per);
        }
    }

    // Python CUDA argmin bug for ts==16
    float err = s_err[0];
    int min_shift_x = 0, min_shift_y = 0;
    for (int i = 0; i < corr; ++i) {
        for (int j = 0; j < corr; ++j) {
            float min_v = s_err[i * corr + j];
            if (err < min_v) {
                min_shift_y = i - R;
                min_shift_x = j - R;
            }
        }
    }
    flow[tid * 2u + 0u] = float(flow_dx + min_shift_x);
    flow[tid * 2u + 1u] = float(flow_dy + min_shift_y);
}

// ---------------------------------------------------------------------------
// ICA refine (ICA.py ica_kernel_8 / ica_kernel_16) — one thread per tile.
// Butterfly reduce order matches align.cpp butterfly_reduce_sum (same as CUDA
// shared-mem while N>0 tree). Bilinear: ts==8 clamp-to-edge; else OOB→0.
// ---------------------------------------------------------------------------
kernel void l1_bm_ts32(device const float* ref [[buffer(0)]],
                       device const float* mov [[buffer(1)]],
                       device float* flow [[buffer(2)]],
                       constant L1BmParams& p [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= p.ny * p.nx) return;
    uint ty = tid / p.nx;
    uint tx = tid % p.nx;
    int ts = 32;
    int R = int(p.R);
    int corr = 2 * R + 1;
    if (corr > 17) return;
    int oy = int(ty) * ts;
    int ox = int(tx) * ts;

    float fdx = flow[tid * 2u + 0u];
    float fdy = flow[tid * 2u + 1u];
    int flow_dx = int(round(fdx));
    int flow_dy = int(round(fdy));

    float s_err[289];
    float per[1024];
    for (int sdy = -R; sdy <= R; ++sdy) {
        for (int sdx = -R; sdx <= R; ++sdx) {
            for (int i = 0; i < 32; ++i) {
                int ry = oy + i;
                for (int j = 0; j < 32; ++j) {
                    int rx = ox + j;
                    int tidp = i * 32 + j;
                    float rv = (ry < int(p.ref_h) && rx < int(p.ref_w))
                                   ? ref[uint(ry) * p.ref_w + uint(rx)] : 0.f;
                    int my = ry + flow_dy + sdy;
                    int mx = rx + flow_dx + sdx;
                    float mv = (my >= 0 && my < int(p.mov_h) && mx >= 0 && mx < int(p.mov_w))
                                   ? mov[uint(my) * p.mov_w + uint(mx)] : 0.f;
                    per[tidp] = fabs(rv - mv);
                }
            }
            s_err[(sdy + R) * corr + (sdx + R)] = warp_then_block_reduce_1024(per);
        }
    }

    float err = INFINITY;
    int min_shift_x = 0, min_shift_y = 0;
    for (int i = 0; i < corr; ++i) {
        for (int j = 0; j < corr; ++j) {
            float min_v = s_err[i * corr + j];
            if (err < min_v) {
                min_shift_y = i - R;
                min_shift_x = j - R;
            }
        }
    }
    flow[tid * 2u + 0u] = float(flow_dx + min_shift_x);
    flow[tid * 2u + 1u] = float(flow_dy + min_shift_y);
}

kernel void l1_bm_ts64(device const float* ref [[buffer(0)]],
                       device const float* mov [[buffer(1)]],
                       device float* flow [[buffer(2)]],
                       constant L1BmParams& p [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= p.ny * p.nx) return;
    uint ty = tid / p.nx;
    uint tx = tid % p.nx;
    int ts = 64;
    int R = int(p.R);
    int corr = 2 * R + 1;
    if (corr > 17) return;
    int oy = int(ty) * ts;
    int ox = int(tx) * ts;

    float fdx = flow[tid * 2u + 0u];
    float fdy = flow[tid * 2u + 1u];
    int flow_dx = int(round(fdx));
    int flow_dy = int(round(fdy));

    float s_err[289];
    float per[1024];
    for (int sdy = -R; sdy <= R; ++sdy) {
        for (int sdx = -R; sdx <= R; ++sdx) {
            (void)sdy;
            (void)sdx;
            for (int tyy = 0; tyy < 16; ++tyy) {
                for (int txx = 0; txx < 64; ++txx) {
                    int ti = tyy * 64 + txx;
                    int px = ox + txx;
                    int py0 = oy + tyy * 4;
                    float acc = 0.f;
                    for (int k = 0; k < 4; ++k) {
                        int py = py0 + k;
                        float rv = (py < int(p.ref_h) && px < int(p.ref_w))
                                       ? ref[uint(py) * p.ref_w + uint(px)] : 0.f;
                        int my = py + flow_dy;
                        int mx = px + flow_dx;
                        float mv = (my >= 0 && my < int(p.mov_h) &&
                                    mx >= 0 && mx < int(p.mov_w))
                                       ? mov[uint(my) * p.mov_w + uint(mx)] : 0.f;
                        acc += fabs(rv - mv);
                    }
                    per[ti] = acc;
                }
            }
            s_err[(sdy + R) * corr + (sdx + R)] = warp_then_block_reduce_1024(per);
        }
    }

    float err = INFINITY;
    int min_shift_x = 0, min_shift_y = 0;
    for (int i = 0; i < corr; ++i) {
        for (int j = 0; j < corr; ++j) {
            float min_v = s_err[i * corr + j];
            if (err < min_v) {
                min_shift_y = i - R;
                min_shift_x = j - R;
            }
        }
    }
    flow[tid * 2u + 0u] = float(flow_dx + min_shift_x);
    flow[tid * 2u + 1u] = float(flow_dy + min_shift_y);
}

struct IcaParams {
    uint ref_h, ref_w, mov_h, mov_w;
    uint ny, nx, ts, n_iter;
    uint clamp_edge; // 1 → ica_kernel_8 clamp; 0 → ica_kernel_16 zero-OOB
    float damp_ratio;  // LM damping toward this eigenvalue ratio; 0 disables
    float max_step;    // per-iteration displacement bound in px; 0 disables
    uint _pad0;        // 48 bytes for setBytes
};

inline float sample_mov(device const float* mov, int y, int x,
                        int H, int W, bool clamp_edge) {
    if (clamp_edge) {
        int yy = clamp(y, 0, H - 1);
        int xx = clamp(x, 0, W - 1);
        return mov[uint(yy) * uint(W) + uint(xx)];
    }
    if (y < 0 || y >= H || x < 0 || x >= W) return 0.f;
    return mov[uint(y) * uint(W) + uint(x)];
}

inline float bilinear_ica_metal(device const float* mov, int py, int px,
                                int floor_off_y, int floor_off_x,
                                float frac_x, float frac_y,
                                int H, int W, bool clamp_edge) {
    int floor_y = py + floor_off_y;
    int floor_x = px + floor_off_x;
    float m00, m01, m10, m11;
    if (clamp_edge) {
        int fy = clamp(floor_y, 0, H - 1);
        int fx = clamp(floor_x, 0, W - 1);
        int cy = clamp(fy + 1, 0, H - 1);
        int cx = clamp(fx + 1, 0, W - 1);
        m00 = mov[uint(fy) * uint(W) + uint(fx)];
        m01 = mov[uint(fy) * uint(W) + uint(cx)];
        m10 = mov[uint(cy) * uint(W) + uint(fx)];
        m11 = mov[uint(cy) * uint(W) + uint(cx)];
    } else {
        m00 = sample_mov(mov, floor_y + 0, floor_x + 0, H, W, false);
        m01 = sample_mov(mov, floor_y + 0, floor_x + 1, H, W, false);
        m10 = sample_mov(mov, floor_y + 1, floor_x + 0, H, W, false);
        m11 = sample_mov(mov, floor_y + 1, floor_x + 1, H, W, false);
    }
    float lerpx_top = m00 + (m01 - m00) * frac_x;
    float lerpx_bot = m10 + (m11 - m10) * frac_x;
    return lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
}

// Same addition order as align.cpp butterfly_reduce_sum / CUDA shared tree.
inline float butterfly_reduce_sum_metal(thread float* s, int n) {
    for (int N = n / 2; N > 0; N /= 2) {
        for (int tid = 0; tid < N; ++tid)
            s[tid] += s[tid + N];
    }
    return s[0];
}

// One threadgroup per tile.
//
// The ts<=16 path previously declared two 256-float thread-private arrays and
// reduced them inside a single thread. That is 2KB of per-thread storage, well
// past what fits in registers, so it spilled to device memory and every element
// access became a round trip. Staging them in threadgroup memory with one lane
// per pixel removes the spill and parallelizes the fill.
//
// butterfly_reduce_sum_metal pairs s[t] += s[t + N] for t < N with N halving
// from n_pix/2, which is exactly a threadgroup tree reduction, so the addition
// order below is unchanged and the sums stay bit-identical. n_pix is 64 (ts=8)
// or 256 (ts=16) -- both powers of two, as the pairing requires.
//
// Host supplies threadgroup memory: [0] and [1] are n_pix floats each.
kernel void ica_refine_tile(device const float* ref [[buffer(0)]],
                            device const float* gradx [[buffer(1)]],
                            device const float* grady [[buffer(2)]],
                            device const float* hess [[buffer(3)]],
                            device const float* mov [[buffer(4)]],
                            device float* flow [[buffer(5)]],
                            constant IcaParams& p [[buffer(6)]],
                            threadgroup float* s_B0 [[threadgroup(0)]],
                            threadgroup float* s_B1 [[threadgroup(1)]],
                            uint tile_id [[threadgroup_position_in_grid]],
                            uint lane [[thread_position_in_threadgroup]],
                            uint lanes [[threads_per_threadgroup]]) {
    // Every exit below this point is uniform across the threadgroup, so the
    // barriers further down stay well-formed.
    if (tile_id >= p.ny * p.nx) return;
    const uint tid = tile_id;
    uint ty = tid / p.nx;
    uint tx = tid % p.nx;
    int ts = int(p.ts);
    if (ts != 8 && ts != 16 && ts != 32 && ts != 64) return;
    int n_pix = ts * ts;
    int oy = int(ty) * ts;
    int ox = int(tx) * ts;
    bool clamp_edge = (p.clamp_edge != 0u);
    int RH = int(p.ref_h), RW = int(p.ref_w);
    int MH = int(p.mov_h), MW = int(p.mov_w);

    uint ho = tid * 4u;
    float h00 = hess[ho + 0u], h01 = hess[ho + 1u];
    float h10 = hess[ho + 2u], h11 = hess[ho + 3u];
    // Levenberg-Marquardt damping toward the aperture ratio -- see
    // ica_refine_level in align.cpp. The Hessian is fixed across the iterations
    // below, so this runs once per tile.
    if (p.damp_ratio > 0.f) {
        float tr = h00 + h11;
        float d0 = h00 * h11 - h01 * h10;
        float disc = sqrt(max(0.f, tr * tr * 0.25f - d0));
        float l1 = tr * 0.5f + disc;
        float l2 = tr * 0.5f - disc;
        float lam = p.damp_ratio * l1 - l2;   // > 0 only when l2/l1 < ratio
        if (lam > 0.f) { h00 += lam; h11 += lam; }
    }
    float det = h00 * h11 - h01 * h10;
    if (fabs(det) < 1e-10f) return; // leave flow unchanged; pure divide-by-zero preflight
    float det_inv = 1.f / det;

    float fx = flow[tid * 2u + 0u];
    float fy = flow[tid * 2u + 1u];

    if (ts == 32 || ts == 64) {
        // Unreachable with the shipped bm_tile_sizes ({16,16,16,8}); kept
        // verbatim and confined to one lane so behaviour is preserved if the
        // configuration ever selects these sizes.
        if (lane != 0u) return;
        for (uint it = 0u; it < p.n_iter; ++it) {
            // floor, not trunc -- see ica_refine_level in align.cpp.
            float floor_fx = floor(fx);
            float floor_fy = floor(fy);
            float frac_x = fx - floor_fx;
            float frac_y = fy - floor_fy;
            int floor_off_x = int(floor_fx);
            int floor_off_y = int(floor_fy);

            float B0 = 0.f;
            float B1 = 0.f;
            if (ts == 32) {
                for (int i = 0; i < 32; ++i) {
                    int py = oy + i;
                    for (int j = 0; j < 32; ++j) {
                        int px = ox + j;
                        if (py >= RH || px >= RW) continue;
                        float mov_interp = bilinear_ica_metal(
                            mov, py, px, floor_off_y, floor_off_x, frac_x, frac_y,
                            MH, MW, false);
                        uint ro = uint(py) * p.ref_w + uint(px);
                        float gradt = mov_interp - ref[ro];
                        B0 += -gradx[ro] * gradt;
                        B1 += -grady[ro] * gradt;
                    }
                }
            } else {
                for (int tyy = 0; tyy < 16; ++tyy) {
                    for (int txx = 0; txx < 64; ++txx) {
                        int px = ox + txx;
                        int py0 = oy + tyy * 4;
                        int floor_x = px + floor_off_x;
                        int floor_y = py0 + floor_off_y;
                        float m10 = sample_mov(mov, floor_y, floor_x, MH, MW, false);
                        float m11 = sample_mov(mov, floor_y, floor_x + 1, MH, MW, false);
                        float lerpx_bot = m10 + (m11 - m10) * frac_x;
                        for (int k = 0; k < 4; ++k) {
                            int py = py0 + k;
                            floor_y += 1;
                            m10 = sample_mov(mov, floor_y + 1, floor_x, MH, MW, false);
                            m11 = sample_mov(mov, floor_y + 1, floor_x + 1, MH, MW, false);
                            float lerpx_top = lerpx_bot;
                            lerpx_bot = m10 + (m11 - m10) * frac_x;
                            float mov_interp = lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
                            if (py < RH && px < RW) {
                                uint ro = uint(py) * p.ref_w + uint(px);
                                float gradt = mov_interp - ref[ro];
                                B0 += -gradx[ro] * gradt;
                                B1 += -grady[ro] * gradt;
                            }
                        }
                    }
                }
            }
            float dfx = det_inv * (h11 * B0 - h01 * B1);
            float dfy = det_inv * (-h10 * B0 + h00 * B1);
            if (p.max_step > 0.f) {
                float st = sqrt(dfx * dfx + dfy * dfy);
                if (st > p.max_step) { float k = p.max_step / st; dfx *= k; dfy *= k; }
            }
            fx += dfx;
            fy += dfy;
        }
        flow[tid * 2u + 0u] = fx;
        flow[tid * 2u + 1u] = fy;
        return;
    }

    // ts<=16: n_pix is 64 or 256, staged in threadgroup memory (see header note).
    for (uint it = 0u; it < p.n_iter; ++it) {
        // floor, not trunc. ICA.py truncates toward zero, which turns the
        // bilinear sample into an extrapolation for negative displacements and
        // makes the converged flow direction-dependent. See ica_refine_level in
        // align.cpp for the measurement.
        float floor_fx = floor(fx);
        float floor_fy = floor(fy);
        float frac_x = fx - floor_fx;
        float frac_y = fy - floor_fy;
        int floor_off_x = int(floor_fx);
        int floor_off_y = int(floor_fy);

        // Same per-pixel values as the serial i/j fill; only who writes changes.
        for (int tpix = int(lane); tpix < n_pix; tpix += int(lanes)) {
            int i = tpix / ts;
            int j = tpix - i * ts;
            int py = oy + i;
            int px = ox + j;
            if (py >= RH || px >= RW) {
                s_B0[tpix] = 0.f;
                s_B1[tpix] = 0.f;
                continue;
            }
            float mov_interp = bilinear_ica_metal(
                mov, py, px, floor_off_y, floor_off_x, frac_x, frac_y,
                MH, MW, clamp_edge);
            uint ro = uint(py) * p.ref_w + uint(px);
            float gradt = mov_interp - ref[ro];
            s_B0[tpix] = -gradx[ro] * gradt;
            s_B1[tpix] = -grady[ro] * gradt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // butterfly_reduce_sum_metal, unrolled across lanes: identical pairing
        // and identical addition order, therefore identical sums.
        for (int N = n_pix / 2; N > 0; N /= 2) {
            for (int t = int(lane); t < N; t += int(lanes)) {
                s_B0[t] += s_B0[t + N];
                s_B1[t] += s_B1[t + N];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Every lane derives the same fx/fy from the same reduced sums, so the
        // next iteration's fill is consistent without broadcasting them.
        float B0 = s_B0[0];
        float B1 = s_B1[0];
        float dfx = det_inv * (h11 * B0 - h01 * B1);
        float dfy = det_inv * (-h10 * B0 + h00 * B1);
        if (p.max_step > 0.f) {
            float st = sqrt(dfx * dfx + dfy * dfy);
            if (st > p.max_step) { float k = p.max_step / st; dfx *= k; dfy *= k; }
        }
        fx += dfx;
        fy += dfy;
        threadgroup_barrier(mem_flags::mem_threadgroup); // before the next refill
    }

    if (lane == 0u) {
        flow[tid * 2u + 0u] = fx;
        flow[tid * 2u + 1u] = fy;
    }
}

// ---------------------------------------------------------------------------
// Pyramid downsample — exact match of grey_pyramid.cpp downsample_by /
// Python cuda_downsample: scipy gaussian_kernel1d, valid separable conv, stride.
// ---------------------------------------------------------------------------
struct PyrDownParams {
    uint in_h, in_w;
    uint out_h, out_w;
    uint klen;
    uint factor;
    uint _pad0, _pad1; // 32 bytes for setBytes
};

// Valid vertical conv: out[y,x] = sum_i ker[i] * in[y+i, x]
kernel void pyr_conv_y(device const float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       device const float* ker [[buffer(2)]],
                       constant PyrDownParams& p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.out_w || gid.y >= p.out_h) return;
    float acc = 0.f;
    for (uint i = 0u; i < p.klen; ++i)
        acc += ker[i] * in[(gid.y + i) * p.in_w + gid.x];
    out[gid.y * p.out_w + gid.x] = acc;
}

// Valid horizontal conv: out[y,x] = sum_j ker[j] * in[y, x+j]
// in_w is the temp width (same as vertical out_w); out size is filt.
kernel void pyr_conv_x(device const float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       device const float* ker [[buffer(2)]],
                       constant PyrDownParams& p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.out_w || gid.y >= p.out_h) return;
    // p.in_w = temp width, p.in_h unused for addressing rows
    float acc = 0.f;
    const uint row = gid.y * p.in_w;
    for (uint j = 0u; j < p.klen; ++j)
        acc += ker[j] * in[row + gid.x + j];
    out[gid.y * p.out_w + gid.x] = acc;
}

// filtered[y*factor, x*factor] → out[y,x]
kernel void pyr_subsample(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant PyrDownParams& p [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.out_w || gid.y >= p.out_h) return;
    out[gid.y * p.out_w + gid.x] =
        in[(gid.y * p.factor) * p.in_w + (gid.x * p.factor)];
}

// ---------------------------------------------------------------------------
// Align extras — 1:1 with align.cpp Sobel / Hessian / upscale_flow
// ---------------------------------------------------------------------------
struct AlignImgParams {
    uint h, w;
    uint _pad0, _pad1;
};

// compute_sobel_gradx: out = -I[y,x-1] + I[y,x+1], OOB → 0
kernel void align_sobel_x(device const float* img [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant AlignImgParams& p [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    float vm = (gid.x >= 1u) ? img[gid.y * p.w + (gid.x - 1u)] : 0.f;
    float vp = (gid.x + 1u < p.w) ? img[gid.y * p.w + (gid.x + 1u)] : 0.f;
    out[gid.y * p.w + gid.x] = -vm + vp;
}

// compute_sobel_grady: out = -I[y-1,x] + I[y+1,x], OOB → 0
kernel void align_sobel_y(device const float* img [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant AlignImgParams& p [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    float vm = (gid.y >= 1u) ? img[(gid.y - 1u) * p.w + gid.x] : 0.f;
    float vp = (gid.y + 1u < p.h) ? img[(gid.y + 1u) * p.w + gid.x] : 0.f;
    out[gid.y * p.w + gid.x] = -vm + vp;
}

struct AlignHessParams {
    uint h, w, ny, nx, ts;
    uint _pad0, _pad1, _pad2;
};

// compute_hessian: one thread per tile → hess[ty,tx] = Σ [gx², gx·gy, gx·gy, gy²]
kernel void align_hessian(device const float* gx [[buffer(0)]],
                          device const float* gy [[buffer(1)]],
                          device float* hess [[buffer(2)]],
                          constant AlignHessParams& p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= p.ny * p.nx) return;
    uint ty = tid / p.nx;
    uint tx = tid - ty * p.nx;
    uint oy = ty * p.ts;
    uint ox = tx * p.ts;
    float h00 = 0.f, h01 = 0.f, h11 = 0.f;
    for (uint i = 0u; i < p.ts; ++i) {
        uint py = oy + i;
        if (py >= p.h) break;
        for (uint j = 0u; j < p.ts; ++j) {
            uint px = ox + j;
            if (px >= p.w) break;
            uint idx = py * p.w + px;
            float gxv = gx[idx];
            float gyv = gy[idx];
            h00 += gxv * gxv;
            h01 += gxv * gyv;
            h11 += gyv * gyv;
        }
    }
    uint o = tid * 4u;
    hess[o + 0u] = h00;
    hess[o + 1u] = h01;
    hess[o + 2u] = h01;
    hess[o + 3u] = h11;
}

struct AlignUpscaleParams {
    uint in_ny, in_nx;
    uint target_ny, target_nx;
    uint upsample_factor, repeat_factor;
    uint up_ny, up_nx;
    int ts;
    int ref_h, ref_w, mov_h, mov_w;
    uint carry_ambiguity;   // 1 = propagate the block-matching ambiguity flag
    uint fallback_on_ambiguous; // 1 = keep parent candidate on a near-tie
    float ambiguity_ratio;
    uint _pad0;
};

// upscale_flow nearest (default.yaml): repeat then scale by upsample_factor; pad 0
kernel void align_upscale_flow(device const float* in_flow [[buffer(0)]],
                               device float* out_flow [[buffer(1)]],
                               constant AlignUpscaleParams& p [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.target_nx || gid.y >= p.target_ny) return;
    uint o = (gid.y * p.target_nx + gid.x) * 2u;
    if (gid.y < p.up_ny && gid.x < p.up_nx) {
        uint sy = min(p.in_ny - 1u, gid.y / p.repeat_factor);
        uint sx = min(p.in_nx - 1u, gid.x / p.repeat_factor);
        uint s = (sy * p.in_nx + sx) * 2u;
        float sfac = float(p.upsample_factor);
        out_flow[o + 0u] = in_flow[s + 0u] * sfac;
        out_flow[o + 1u] = in_flow[s + 1u] * sfac;
    } else {
        out_flow[o + 0u] = 0.f;
        out_flow[o + 1u] = 0.f;
    }
}

// ---------------------------------------------------------------------------
// merge normalize — 1:1 encode_band_rows num/den → RGB16 (DNG band)
// ---------------------------------------------------------------------------
kernel void align_upscale_flow_460(device const float* in_flow [[buffer(0)]],
                                   device const float* ref [[buffer(1)]],
                                   device const float* mov [[buffer(2)]],
                                   device float* out_flow [[buffer(3)]],
                                   constant AlignUpscaleParams& p [[buffer(4)]],
                                   device const uint* in_amb [[buffer(5)]],
                                   device uint* out_amb [[buffer(6)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.target_nx || gid.y >= p.target_ny) return;
    uint o = (gid.y * p.target_nx + gid.x) * 2u;

    // target_n* always exceeds repeat_factor * in_n* (see upscale_flow_460 in
    // align.cpp for why), leaving a strip of tiles along the bottom and right
    // edges with no coarse tile above them. Clamp to the nearest covered tile;
    // resetting those to zero motion left the strip unrecoverable at the finest
    // level, whose search radius is 1.
    uint prev_x = min(gid.x / p.repeat_factor, p.in_nx - 1u);
    uint prev_y = min(gid.y / p.repeat_factor, p.in_ny - 1u);
    uint ups_x = gid.x % p.repeat_factor;
    uint ups_y = gid.y % p.repeat_factor;
    int x_shift = (2u * ups_x + 1u > p.repeat_factor) ? 1 : -1;
    int y_shift = (2u * ups_y + 1u > p.repeat_factor) ? 1 : -1;
    uint cand_y = uint(clamp(int(prev_y) + y_shift, 0, int(p.in_ny) - 1));
    uint cand_x = uint(clamp(int(prev_x) + x_shift, 0, int(p.in_nx) - 1));

    float sfac = float(p.upsample_factor);
    float2 cand0 = float2(in_flow[(prev_y * p.in_nx + prev_x) * 2u + 0u] * sfac,
                          in_flow[(prev_y * p.in_nx + prev_x) * 2u + 1u] * sfac);
    float2 cand1 = float2(in_flow[(cand_y * p.in_nx + prev_x) * 2u + 0u] * sfac,
                          in_flow[(cand_y * p.in_nx + prev_x) * 2u + 1u] * sfac);
    float2 cand2 = float2(in_flow[(prev_y * p.in_nx + cand_x) * 2u + 0u] * sfac,
                          in_flow[(prev_y * p.in_nx + cand_x) * 2u + 1u] * sfac);

    int ox = int(gid.x) * p.ts;
    int oy = int(gid.y) * p.ts;
    float best = INFINITY;
    float second = INFINITY;
    float2 best_flow = cand0;
    uint best_ci = 0u;
    for (uint ci = 0u; ci < 3u; ++ci) {
        float2 c = (ci == 0u) ? cand0 : ((ci == 1u) ? cand1 : cand2);
        // The sampled region is a full ts x ts rectangle in both images, so
        // "invalidate on the first out-of-bounds sample" is exactly rectangle
        // containment. Hoisting it removes eight comparisons per pixel from the
        // inner loop and drops the loop entirely for candidates that cannot
        // win: those scored INFINITY, and INFINITY < INFINITY is false, so they
        // could never take `best` anyway. best_flow still defaults to cand0
        // when every candidate is out of bounds, as before.
        const int cx = int(c.x);
        const int cy = int(c.y);
        const bool inside =
            (ox >= 0 && oy >= 0 &&
             ox + p.ts <= p.ref_w && oy + p.ts <= p.ref_h &&
             ox + cx >= 0 && oy + cy >= 0 &&
             ox + cx + p.ts <= p.mov_w && oy + cy + p.ts <= p.mov_h);
        if (!inside) continue;

        float dist = 0.f;
        for (int i = 0; i < p.ts; ++i) {
            const uint rrow = uint(oy + i) * uint(p.ref_w) + uint(ox);
            const uint mrow = uint(oy + cy + i) * uint(p.mov_w) + uint(ox + cx);
            for (int j = 0; j < p.ts; ++j)
                dist += fabs(ref[rrow + uint(j)] - mov[mrow + uint(j)]);
        }
        if (dist < best) {
            second = best;
            best = dist;
            best_flow = c;
            best_ci = ci;
        } else if (dist < second) {
            second = dist;
        }
    }
    // ImageStackAlignator's rule extended to the candidate test: when a
    // NEIGHBOUR tile's flow wins over the parent's by less than the ambiguity
    // ratio, the cost surface cannot tell them apart -- keep the parent (the
    // previous level's own estimate) instead of inheriting a neighbour's
    // unrelated motion.
    if (p.fallback_on_ambiguous != 0u && best_ci != 0u) {
        const float b = max(best, 0.f);
        const float sec = max(second, 0.f);
        const float denom = max(b, 1.0e-12f);
        if (isfinite(b) && isfinite(sec) && (sec / denom) < p.ambiguity_ratio) {
            best_flow = cand0;
            best_ci = 0u;
        }
    }
    out_flow[o + 0u] = best_flow.x;
    out_flow[o + 1u] = best_flow.y;
    // Carry the ambiguity flag from whichever coarse tile supplied the winning
    // candidate, exactly as upscale_flow_460 does on the CPU. A wrong match at
    // the coarsest level is 8px x 32 abs factor = 256 raw px, so the flag has to
    // survive from there down to where the mask is applied.
    if (p.carry_ambiguity != 0u) {
        uint sy = (best_ci == 1u) ? cand_y : prev_y;
        uint sx = (best_ci == 2u) ? cand_x : prev_x;
        out_amb[gid.y * p.target_nx + gid.x] = in_amb[sy * p.in_nx + sx];
    }
}

struct MergeNormParams {
    uint bh, Ws, nch, bake;
    float wb0, wb1, wb2;
    float m00, m01, m02, m10, m11, m12, m20, m21, m22;
    float sg0, sg1, sg2;   // un-white-balance store gains (1 = off)
};

inline float norm_to_srgb(float v) {
    if (!isfinite(v)) return 0.f;
    v = clamp(v, 0.f, 1.f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * pow(v, 1.f / 2.4f) - 0.055f;
}

inline float safe_norm_div(float n, float d) {
    if (!(d > 0.f) || !isfinite(n) || !isfinite(d)) return 0.f;
    float v = n / d;
    return isfinite(v) ? v : 0.f;
}

kernel void merge_normalize_rgb16(device const float* num [[buffer(0)]],
                                  device const float* den [[buffer(1)]],
                                  device ushort* out [[buffer(2)]],
                                  constant MergeNormParams& p [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.Ws || gid.y >= p.bh) return;
    uint pi = (gid.y * p.Ws + gid.x) * p.nch;
    float cn0 = safe_norm_div(num[pi], den[pi]);
    float cn1 = 0.f, cn2 = 0.f;
    if (p.nch >= 2u) {
        cn1 = safe_norm_div(num[pi + 1u], den[pi + 1u]);
    }
    if (p.nch >= 3u) {
        cn2 = safe_norm_div(num[pi + 2u], den[pi + 2u]);
    }
    float lin0, lin1, lin2;
    if (p.bake != 0u && p.nch >= 3u) {
        float wr = cn0 * p.wb0, wg = cn1 * p.wb1, wb = cn2 * p.wb2;
        lin0 = p.m00 * wr + p.m01 * wg + p.m02 * wb;
        lin1 = p.m10 * wr + p.m11 * wg + p.m12 * wb;
        lin2 = p.m20 * wr + p.m21 * wg + p.m22 * wb;
    } else if (p.nch >= 3u) {
        lin0 = cn0; lin1 = cn1; lin2 = cn2;
    } else {
        lin0 = lin1 = lin2 = cn0;
    }
    float v0 = (p.bake != 0u) ? norm_to_srgb(lin0) : clamp(lin0 * p.sg0, 0.f, 1.f);
    float v1 = (p.bake != 0u) ? norm_to_srgb(lin1) : clamp(lin1 * p.sg1, 0.f, 1.f);
    float v2 = (p.bake != 0u) ? norm_to_srgb(lin2) : clamp(lin2 * p.sg2, 0.f, 1.f);
    uint o = (gid.y * p.Ws + gid.x) * 3u;
    out[o + 0u] = ushort(v0 * 65535.f + 0.5f);
    out[o + 1u] = ushort(v1 * 65535.f + 0.5f);
    out[o + 2u] = ushort(v2 * 65535.f + 0.5f);
}

// ---------------------------------------------------------------------------
// flow_densify_boundary_select twin (align.cpp): one thread per half-pitch
// fine cell. Covers both measurement branches -- the overlapped-tile
// full-window measurement (Config::flow_overlap_merge) and the cell-footprint
// selection -- plus the bilinear blend for agreeing cells, so the host does
// no per-cell work at all. All rounding matches the CPU (round = lround).
struct DensifyParams {
    uint fny, fnx;          // fine grid
    uint flow_ny, flow_nx;  // tile grid
    uint ref_h, ref_w;
    uint mov_h, mov_w;
    uint tile_size;
    uint overlap_all;       // 1 = overlap measurement branch
    float gsy, gsx;         // raw -> grey scale
    float thr;              // flow_select_threshold
    uint _pad0;             // 56 bytes total for setBytes
};

static inline float densify_cost_win(device const float* ref,
                                     device const float* mov,
                                     constant DensifyParams& p,
                                     int gy0, int gx0, int win,
                                     int idy, int idx) {
    float dist = 0.f;
    for (int i = 0; i < win; ++i) {
        for (int j = 0; j < win; ++j) {
            const int ry = gy0 + i, rx = gx0 + j;
            const int my = ry + idy, mx = rx + idx;
            if (!(ry >= 0 && ry < (int)p.ref_h && rx >= 0 && rx < (int)p.ref_w &&
                  my >= 0 && my < (int)p.mov_h && mx >= 0 && mx < (int)p.mov_w))
                return INFINITY;
            dist += fabs(ref[ry * (int)p.ref_w + rx] - mov[my * (int)p.mov_w + mx]);
        }
    }
    return dist;
}

kernel void flow_densify_select(device float* fine [[buffer(0)]],
                                device const float* flowv [[buffer(1)]],
                                device const float* ref [[buffer(2)]],
                                device const float* mov [[buffer(3)]],
                                constant DensifyParams& p [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.fnx || gid.y >= p.fny) return;
    const int fy = (int)gid.y, fx = (int)gid.x;
    const float ts2 = 0.5f * (float)p.tile_size;
    // Same lattice mapping as FlowField::sample_grid at the cell centre.
    const float cy = ((float)fy + 0.5f) * ts2;
    const float cx = ((float)fx + 0.5f) * ts2;
    const float tcy = cy / (float)p.tile_size - 0.5f;
    const float tcx = cx / (float)p.tile_size - 0.5f;
    const int y0 = (int)floor(tcy), x0 = (int)floor(tcx);
    const float ay = tcy - (float)y0, ax = tcx - (float)x0;
    const int ny = (int)p.flow_ny, nx = (int)p.flow_nx;
    const int iy0 = clamp(y0, 0, ny - 1), iy1 = clamp(y0 + 1, 0, ny - 1);
    const int ix0 = clamp(x0, 0, nx - 1), ix1 = clamp(x0 + 1, 0, nx - 1);
    float vx[4], vy[4];
    vx[0] = flowv[(iy0 * nx + ix0) * 2 + 0];
    vy[0] = flowv[(iy0 * nx + ix0) * 2 + 1];
    vx[1] = flowv[(iy0 * nx + ix1) * 2 + 0];
    vy[1] = flowv[(iy0 * nx + ix1) * 2 + 1];
    vx[2] = flowv[(iy1 * nx + ix0) * 2 + 0];
    vy[2] = flowv[(iy1 * nx + ix0) * 2 + 1];
    vx[3] = flowv[(iy1 * nx + ix1) * 2 + 0];
    vy[3] = flowv[(iy1 * nx + ix1) * 2 + 1];
    float spread = 0.f;
    for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b)
            spread = max(spread, max(fabs(vx[a] - vx[b]), fabs(vy[a] - vy[b])));
    const float tx0 = vx[0] + (vx[1] - vx[0]) * ax;
    const float bx0 = vx[2] + (vx[3] - vx[2]) * ax;
    const float ty0 = vy[0] + (vy[1] - vy[0]) * ax;
    const float by0 = vy[2] + (vy[3] - vy[2]) * ax;
    float out_x = tx0 + (bx0 - tx0) * ay;
    float out_y = ty0 + (by0 - ty0) * ay;
    if (p.overlap_all != 0u && spread > p.thr) {
        // Full tile-sized window at the fine cell centre (HDR+ layout).
        const int win = max(2, (int)round((float)p.tile_size * p.gsy));
        const int gy0 = (int)round(((float)fy + 0.5f) * ts2 * p.gsy) - win / 2;
        const int gx0 = (int)round(((float)fx + 0.5f) * ts2 * p.gsx) - win / 2;
        float best_cost = INFINITY;
        int seed_y = 0, seed_x = 0;
        bool have_seed = false;
        for (int c4 = 0; c4 < 4; ++c4) {
            const int idx = (int)round(vx[c4] * p.gsx);
            const int idy = (int)round(vy[c4] * p.gsy);
            const float d = densify_cost_win(ref, mov, p, gy0, gx0, win, idy, idx);
            if (d < best_cost) {
                best_cost = d;
                seed_y = idy;
                seed_x = idx;
                have_seed = true;
                out_x = vx[c4];
                out_y = vy[c4];
            }
        }
        // 3x3 integer refinement + quadratic sub-cell fit; the measurement
        // replaces the seed only when it MOVES (see the CPU twin's comment).
        if (have_seed) {
            float surf[9];
            int bo = 4;
            float bo_cost = best_cost;
            for (int oy = -1; oy <= 1; ++oy)
                for (int ox = -1; ox <= 1; ++ox) {
                    const int sidx = (oy + 1) * 3 + (ox + 1);
                    surf[sidx] = (ox == 0 && oy == 0)
                                     ? best_cost
                                     : densify_cost_win(ref, mov, p, gy0, gx0,
                                                        win, seed_y + oy,
                                                        seed_x + ox);
                    if (surf[sidx] < bo_cost) { bo_cost = surf[sidx]; bo = sidx; }
                }
            if (bo != 4) {
                seed_y += bo / 3 - 1;
                seed_x += bo % 3 - 1;
                for (int oy = -1; oy <= 1; ++oy)
                    for (int ox = -1; ox <= 1; ++ox)
                        surf[(oy + 1) * 3 + (ox + 1)] =
                            densify_cost_win(ref, mov, p, gy0, gx0, win,
                                             seed_y + oy, seed_x + ox);
                if (isfinite(surf[4])) {
                    float sub_x = 0.f, sub_y = 0.f;
                    (void)bm_quadratic_subpixel_3x3(surf, sub_x, sub_y);
                    out_x = ((float)seed_x + sub_x) / p.gsx;
                    out_y = ((float)seed_y + sub_y) / p.gsy;
                }
            }
        }
    } else if (spread > p.thr) {
        // Selection over the cell footprint.
        const int gy0 = (int)floor((float)fy * ts2 * p.gsy);
        const int gx0 = (int)floor((float)fx * ts2 * p.gsx);
        const int gh = max(1, (int)round(ts2 * p.gsy));
        const int gw = max(1, (int)round(ts2 * p.gsx));
        float best_cost = INFINITY;
        for (int c4 = 0; c4 < 4; ++c4) {
            const int idx = (int)round(vx[c4] * p.gsx);
            const int idy = (int)round(vy[c4] * p.gsy);
            float dist = 0.f;
            bool valid = true;
            for (int i = 0; i < gh && valid; ++i) {
                for (int j = 0; j < gw; ++j) {
                    const int ry = gy0 + i, rx = gx0 + j;
                    const int my = ry + idy, mx = rx + idx;
                    if (!(ry >= 0 && ry < (int)p.ref_h &&
                          rx >= 0 && rx < (int)p.ref_w &&
                          my >= 0 && my < (int)p.mov_h &&
                          mx >= 0 && mx < (int)p.mov_w)) {
                        valid = false;
                        break;
                    }
                    dist += fabs(ref[ry * (int)p.ref_w + rx] -
                                 mov[my * (int)p.mov_w + mx]);
                }
            }
            if (valid && dist < best_cost) {
                best_cost = dist;
                out_x = vx[c4];
                out_y = vy[c4];
            }
        }
    }
    const int o = (fy * (int)p.fnx + fx) * 2;
    fine[o + 0] = out_x;
    fine[o + 1] = out_y;
}
