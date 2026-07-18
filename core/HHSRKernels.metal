// Metal kernels for HHSR grey-FFT + L2 BM.
// 1D DFT matches grey_pyramid.cpp fft1d_pow2_inplace_ref + fft1d_bluestein
// (same bit-reversal, iterative twiddles w*=wlen, Bluestein scaling).

#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;

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
    float ang = (inverse != 0 ? 2.f : -2.f) * PI / float(len);
    float2 wlen = float2(cos(ang), sin(ang));
    // w = wlen^k
    float2 w = float2(1.f, 0.f);
    {
        uint kk = k;
        float2 b = wlen;
        while (kk != 0u) {
            if ((kk & 1u) != 0u) w = cmul(w, b);
            b = cmul(b, b);
            kk >>= 1;
        }
    }
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
