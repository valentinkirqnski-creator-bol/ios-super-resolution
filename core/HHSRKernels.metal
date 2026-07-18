// Metal kernels for HHSR grey-FFT + L2 BM.
// 1D DFT: Cooley–Tukey radix-2 (pow2) and Bluestein (arbitrary n), matching
// the CPU reference in grey_pyramid.cpp (not vDSP).

#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cconj(float2 a) { return float2(a.x, -a.y); }

// Bit-reverse permute for one length-n (pow2) vector. One thread per element.
kernel void fft_bitrev(device float2* data [[buffer(0)]],
                       constant uint& n [[buffer(1)]],
                       constant uint& stride [[buffer(2)]],
                       constant uint& batch_count [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint i = gid.x;
    if (batch >= batch_count || i >= n) return;
    uint j = 0;
    uint x = i;
    // log2(n) bits
    for (uint n2 = n; n2 > 1u; n2 >>= 1) {
        j = (j << 1) | (x & 1u);
        x >>= 1;
    }
    if (i < j) {
        uint base = batch * stride;
        float2 tmp = data[base + i];
        data[base + i] = data[base + j];
        data[base + j] = tmp;
    }
}

// One radix-2 butterfly stage. len = current DFT length (2,4,...,n).
kernel void fft_radix2_stage(device float2* data [[buffer(0)]],
                             constant uint& n [[buffer(1)]],
                             constant uint& stride [[buffer(2)]],
                             constant uint& batch_count [[buffer(3)]],
                             constant uint& len [[buffer(4)]],
                             constant int& inverse [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint i = gid.x;
    if (batch >= batch_count || i >= n) return;
    uint half_n = len >> 1;
    if ((i % len) >= half_n) return;

    float ang = (inverse ? 2.f : -2.f) * PI / float(len);
    uint k = i % half_n;
    float2 w = float2(cos(ang * float(k)), sin(ang * float(k)));

    uint base = batch * stride;
    uint i0 = (i / len) * len + k;
    uint i1 = i0 + half_n;
    float2 u = data[base + i0];
    float2 v = cmul(data[base + i1], w);
    data[base + i0] = u + v;
    data[base + i1] = u - v;
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

kernel void cbuf_mul(device float2* a [[buffer(0)]],
                     device const float2* b [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    a[id] = cmul(a[id], b[id]);
}

// A[b*m+i] *= B[i]  (broadcast B across batches)
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

kernel void cbuf_mul_chirp(device float2* a [[buffer(0)]],
                           device const float2* chirp [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           constant uint& stride [[buffer(3)]],
                           constant uint& batch_count [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint i = gid.x;
    if (batch >= batch_count || i >= n) return;
    a[batch * stride + i] = cmul(a[batch * stride + i], chirp[i]);
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

kernel void bluestein_pack_B(device float2* B [[buffer(0)]],
                             device const float2* chirp [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             constant uint& m [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= m) return;
    B[i] = float2(0.f, 0.f);
    if (i < n) {
        B[i] = cconj(chirp[i]);
        if (i > 0) B[m - i] = B[i];
    }
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
    float dir = inverse ? 1.f : -1.f;
    float ang = dir * PI * float(i) * float(i) / float(n);
    chirp[i] = float2(cos(ang), sin(ang));
}

// Pack real image rows into complex [batch=h][w]
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

kernel void fftshift_zero_lowpass(device float2* data [[buffer(0)]],
                                  constant uint& h [[buffer(1)]],
                                  constant uint& w [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    // In-place fftshift then zero outer quarters (matches CPU grey path).
    uint y = gid.y, x = gid.x;
    if (y >= h || x >= w) return;
    // Cooperative shift needs temp — done on host via two buffers; this kernel
    // only zeros after shift. See metal_gpu.mm.
    (void)data;
}

kernel void copy_c(device float2* out [[buffer(0)]],
                   device const float2* in [[buffer(1)]],
                   constant uint& count [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = in[id];
}

kernel void fftshift2d_c(device float2* out [[buffer(0)]],
                         device const float2* in [[buffer(1)]],
                         constant uint& h [[buffer(2)]],
                         constant uint& w [[buffer(3)]],
                         constant int& inv [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint y = gid.y, x = gid.x;
    if (y >= h || x >= w) return;
    int shy = int(h) / 2;
    int shx = int(w) / 2;
    int sy, sx;
    if (inv == 0) {
        // fftshift: out[y,x] = in[(y+h/2)%h, (x+w/2)%w]
        sy = int((y + uint(shy)) % h);
        sx = int((x + uint(shx)) % w);
        out[y * w + x] = in[uint(sy) * w + uint(sx)];
    } else {
        // ifftshift: out[y,x] = in[(y - h/2 + h)%h, ...]
        sy = int((y + h - uint(shy)) % h);
        sx = int((x + w - uint(shx)) % w);
        out[y * w + x] = in[uint(sy) * w + uint(sx)];
    }
}

kernel void zero_fft_borders(device float2* data [[buffer(0)]],
                             constant uint& h [[buffer(1)]],
                             constant uint& w [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint y = gid.y, x = gid.x;
    if (y >= h || x >= w) return;
    uint y0 = h / 4, x0 = w / 4;
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

// ---- L2 BM: pack tiles -------------------------------------------------
struct L2Params {
    uint ny, nx;
    int ts, R, N;
    int ref_h, ref_w, mov_h, mov_w;
};

kernel void l2_pack_tiles(device float* ref_pad [[buffer(0)]],
                          device float* mov_patch [[buffer(1)]],
                          device const float* ref [[buffer(2)]],
                          device const float* mov [[buffer(3)]],
                          device const float* flow [[buffer(4)]],
                          constant L2Params& P [[buffer(5)]],
                          uint tid [[thread_position_in_grid]]) {
    uint ntiles = P.ny * P.nx;
    if (tid >= ntiles) return;
    uint ty = tid / P.nx;
    uint tx = tid % P.nx;
    int ts = P.ts, R = P.R, N = P.N;
    int oy = int(ty) * ts;
    int ox = int(tx) * ts;
    float fdx = flow[tid * 2 + 0];
    float fdy = flow[tid * 2 + 1];
    // torch round half to even
    int flow_dx = int(rint(fdx));
    int flow_dy = int(rint(fdy));

    uint base = tid * uint(N * N);
    for (int i = 0; i < N * N; ++i) {
        ref_pad[base + uint(i)] = 0.f;
    }
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

kernel void fftshift2d_real_batch(device float* data [[buffer(0)]],
                                  constant uint& N [[buffer(1)]],
                                  constant uint& batch_count [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint i = gid.x;
    if (batch >= batch_count || i >= N * N) return;
    // needs temp — host uses ping-pong buffers
    (void)data;
}

kernel void l2_argmin(device float* flow [[buffer(0)]],
                      device const float* corr [[buffer(1)]],
                      device const float* mov_patch [[buffer(2)]],
                      constant L2Params& P [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {
    uint ntiles = P.ny * P.nx;
    if (tid >= ntiles) return;
    int N = P.N, ts = P.ts, R = P.R;
    int corr_size = 2 * R + 1;
    int crop = (N - 1 - corr_size) / 2;
    int crop0 = crop + 1;
    uint base = tid * uint(N * N);

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
    flow[tid * 2 + 0] += float(best_dx);
    flow[tid * 2 + 1] += float(best_dy);
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
    uint shy = h / 2, shx = w / 2;
    uint sy = (y + shy) % h;
    uint sx = (x + shx) % w;
    out[batch * h * w + y * w + x] = in[batch * h * w + sy * w + sx];
}

// Pack real NxN tiles → complex for row FFT batch (batch = ntiles * N rows)
kernel void pack_tile_rows(device float2* out [[buffer(0)]],
                           device const float* in [[buffer(1)]],
                           constant uint& N [[buffer(2)]],
                           constant uint& ntiles [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y; // 0 .. ntiles*N
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

kernel void hermite_complete_row(device float2* row [[buffer(0)]],
                                 constant uint& N [[buffer(1)]],
                                 constant uint& wh [[buffer(2)]],
                                 constant uint& nrows [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint r = gid.y;
    uint x = gid.x;
    if (r >= nrows || x >= N) return;
    if (x < wh) return;
    uint k = N - x;
    row[r * N + x] = cconj(row[r * N + k]);
}

kernel void write_rfft_cols_from_half(device float2* cols [[buffer(0)]],
                                      device const float2* rfft_pack [[buffer(1)]],
                                      constant uint& N [[buffer(2)]],
                                      constant uint& wh [[buffer(3)]],
                                      constant uint& ntiles [[buffer(4)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    // rfft_pack layout: [ntiles][N][wh]; cols layout for col-FFT: [ntiles*wh][N]
    uint tile = gid.y;
    uint idx = gid.x; // y * wh + xfreq
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
    uint idx = gid.x; // y*N+x
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
