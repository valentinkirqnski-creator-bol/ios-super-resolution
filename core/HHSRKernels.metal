// Metal kernels for HHSR grey-FFT + L2 BM + merge accumulate.
// 1D DFT matches grey_pyramid.cpp fft1d_pow2_inplace_ref + fft1d_bluestein
// (same bit-reversal, iterative twiddles w*=wlen, Bluestein scaling).
// Merge matches merge.cpp accumulate_comp / accumulate_ref (Alg. 4 / 11).

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

    float fdx = flow[tid * 2u + 0u];
    float fdy = flow[tid * 2u + 1u];
    int flow_dx = int(rint(fdx));
    int flow_dy = int(rint(fdy));

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
    flow[tid * 2u + 0u] = float(flow_dx + best_dx);
    flow[tid * 2u + 1u] = float(flow_dy + best_dy);
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
    uint _pad0;
    uint _pad1;
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
    float max_frame_count;
    uint cfa00;
    uint cfa01;
    uint cfa10;
    uint cfa11;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

inline void soften_inv_cov(thread float& ixx, thread float& ixy, thread float& iyy) {
    const float k_max_abs = 32.f;
    float m = max(fabs(ixx), max(fabs(iyy), fabs(ixy)));
    if (!(m > k_max_abs) || !isfinite(m)) {
        if (!isfinite(ixx) || !isfinite(ixy) || !isfinite(iyy)) {
            ixx = 2.f; ixy = 0.f; iyy = 2.f;
        }
        return;
    }
    float s = k_max_abs / m;
    ixx *= s;
    ixy *= s;
    iyy *= s;
}

inline float cov_at(device const float* covs, uint cov_w, int y, int x, int idx) {
    return covs[(uint(y) * cov_w + uint(x)) * 4u + uint(idx)];
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
                           bool raw_det) {
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
    soften_inv_cov(ixx, ixy, iyy);
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

// Alg. 4 — merge.cpp accumulate_comp
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

    int hr_i = int(p.y0 + local_i);
    float lr_x = (float(hr_j) + 0.5f) / p.scale;
    float lr_y = (float(hr_i) + 0.5f) / p.scale;

    // Match CPU merge.cpp: no clamp on flow tile index (pipeline pads so in-range).
    int px = int(lr_x / float(p.tile_size));
    int py = int(lr_y / float(p.tile_size));
    float flowx = flow[(uint(py) * p.flow_nx + uint(px)) * 2u + 0u];
    float flowy = flow[(uint(py) * p.flow_nx + uint(px)) * 2u + 1u];

    int i_r = min(int(lr_y), int(p.lr_h) - 1);
    int j_r = min(int(lr_x), int(p.lr_w) - 1);
    float local_r = robustness[uint(i_r) * p.lr_w + uint(j_r)];

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
        interp_inv_cov(covs, p.cov_h, p.cov_w, kmap_i, kmap_j, ixx, ixy, iyy, true);
    }

    int center_j = int(lr_mov_x);
    int center_i = int(lr_mov_y);
    float lr_mov_j = lr_mov_x - 0.5f;
    float lr_mov_i = lr_mov_y - 0.5f;

    float val0 = 0.f, val1 = 0.f, val2 = 0.f;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            int j = center_j + dj;
            int i = center_i + di;
            if (!(j >= 0 && j < int(p.lr_w) && i >= 0 && i < int(p.lr_h))) continue;

            int channel = cfa_channel(p, i, j);
            float c = img[uint(i) * p.lr_w + uint(j)];
            float dist_x = float(j) - lr_mov_j;
            float dist_y = float(i) - lr_mov_i;
            float z;
            if (p.iso) z = 2.f * (dist_x * dist_x + dist_y * dist_y);
            else       z = ixx * dist_x * dist_x + 2.f * ixy * dist_x * dist_y + iyy * dist_y * dist_y;
            z = max(0.f, z);
            // Same formula as CPU std::exp (fast path; was precise::exp).
            float w = exp(-0.5f * z);

            float contrib_v = w * local_r * c;
            float contrib_a = w * local_r;
            if (channel == 0)      { val0 += contrib_v; acc0 += contrib_a; }
            else if (channel == 1) { val1 += contrib_v; acc1 += contrib_a; }
            else                   { val2 += contrib_v; acc2 += contrib_a; }
        }
    }

    uint base = (local_i * p.Ws + hr_j) * p.nch;
    if (p.nch >= 1) { num[base + 0] += val0; den[base + 0] += acc0; }
    if (p.nch >= 2) { num[base + 1] += val1; den[base + 1] += acc1; }
    if (p.nch >= 3) { num[base + 2] += val2; den[base + 2] += acc2; }
}

// Alg. 11 — merge.cpp accumulate_ref (incl. accumulated-robustness denoise)
kernel void merge_accumulate_ref(device float* num [[buffer(0)]],
                                 device float* den [[buffer(1)]],
                                 device const float* img [[buffer(2)]],
                                 device const float* covs [[buffer(3)]],
                                 device const float* acc_rob [[buffer(4)]],
                                 constant MergeRefParams& p [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]]) {
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
        int ay = min(int(round(coarse_y)), int(p.acc_h) - 1);
        int ax = min(int(round(coarse_x)), int(p.acc_w) - 1);
        local_acc_r = acc_rob[uint(ay) * p.acc_w + uint(ax)];
        additional_denoise_power =
            (local_acc_r <= p.max_frame_count) ? p.max_multiplier : 1.f;
        rad = (local_acc_r <= p.max_frame_count) ? int(p.rad_max) : 1;
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
        interp_inv_cov(covs, p.cov_h, p.cov_w, kmap_i, kmap_j, ixx, ixy, iyy, false);
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
            float w = exp(-0.5f * y);

            if (channel == 0)      { val0 += c * w; acc0 += w; }
            else if (channel == 1) { val1 += c * w; acc1 += w; }
            else                   { val2 += c * w; acc2 += w; }
        }
    }

    bool overwrite = p.robustness_denoise && (local_acc_r < p.max_frame_count);
    uint base = (local_i * p.Ws + hr_j) * p.nch;
    if (overwrite) {
        if (p.nch >= 1) { num[base + 0] = val0; den[base + 0] = acc0; }
        if (p.nch >= 2) { num[base + 1] = val1; den[base + 1] = acc1; }
        if (p.nch >= 3) { num[base + 2] = val2; den[base + 2] = acc2; }
    } else {
        if (p.nch >= 1) { num[base + 0] += val0; den[base + 0] += acc0; }
        if (p.nch >= 2) { num[base + 1] += val1; den[base + 1] += acc1; }
        if (p.nch >= 3) { num[base + 2] += val2; den[base + 2] += acc2; }
    }
}

// =============================================================================
// Alg. 5 — estimate_kernels (matches kernels.cpp exactly, incl. float guards)
// =============================================================================

struct KernelEstParams {
    uint raw_h, raw_w, grey_h, grey_w;
    uint bayer;     // 1 = decimate 2x2 after GAT
    uint selection; // 0 = HardThreshold, 1 = Linear (C++ enum order)
    float alpha, beta;
    float k_detail, k_denoise, D_th, D_tr, k_stretch, k_shrink;
    uint _pad0, _pad1; // 64 bytes total for setBytes
};

inline float gat_sample(float v, float alpha, float beta) {
    // apply_gat: c = 0.375*alpha^2 + beta; out = (2/alpha)*sqrt(max(0, alpha*v+c))
    float c = 0.375f * alpha * alpha + beta;
    float t = alpha * v + c;
    return (2.f / alpha) * sqrt(max(0.f, t));
}

// linalg.h real_polyroots_2 / eigen_val / eigen_vect / eigen_elmts
inline void real_polyroots_2(float a, float b, float c, thread float roots[2]) {
    float delta = max(b * b - 4.f * a * c, 0.f);
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

// kernels.cpp compute_k (incl. flat-tensor + k_min guards)
inline void compute_k_cpu(float l1, float l2, thread float& k1, thread float& k2,
                          constant KernelEstParams& p) {
    l1 = max(0.f, l1);
    l2 = max(0.f, l2);
    float sum = l1 + l2;
    if (sum < 1e-12f) {
        k1 = k2 = p.k_detail * p.k_denoise;
        return;
    }
    float A = 1.f + sqrt(max(0.f, (l1 - l2) / sum));
    float D = clamp(1.f - sqrt(l1) / p.D_tr + p.D_th, 0.f, 1.f);
    float kk1, kk2;
    if (p.selection == 0u) {
        if (A > 1.95f) { kk1 = 1.f / p.k_shrink; kk2 = p.k_stretch; }
        else           { kk1 = 1.f; kk2 = 1.f; }
    } else {
        kk1 = 1.f + A / 2.f * (1.f / p.k_shrink - 1.f);
        kk2 = 1.f + A / 2.f * (p.k_stretch - 1.f);
    }
    k1 = p.k_detail * ((1.f - D) * kk1 + D * p.k_denoise);
    k2 = p.k_detail * ((1.f - D) * kk2 + D * p.k_denoise);
    constexpr float k_min = 0.15f;
    k1 = max(k1, k_min);
    k2 = max(k2, k_min);
}

kernel void kernel_gat(device float* out [[buffer(0)]],
                       device const float* raw [[buffer(1)]],
                       constant KernelEstParams& p [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.raw_w || gid.y >= p.raw_h) return;
    uint i = gid.y * p.raw_w + gid.x;
    out[i] = gat_sample(raw[i], p.alpha, p.beta);
}

kernel void kernel_decimate_grey(device float* grey [[buffer(0)]],
                                 device const float* vst [[buffer(1)]],
                                 constant KernelEstParams& p [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.grey_w || gid.y >= p.grey_h) return;
    uint y = gid.y, x = gid.x;
    if (p.bayer) {
        uint y0 = y * 2u, x0 = x * 2u;
        float s = vst[y0 * p.raw_w + x0] + vst[y0 * p.raw_w + x0 + 1u] +
                  vst[(y0 + 1u) * p.raw_w + x0] + vst[(y0 + 1u) * p.raw_w + x0 + 1u];
        grey[y * p.grey_w + x] = 0.25f * s;
    } else {
        grey[y * p.grey_w + x] = vst[y * p.raw_w + x];
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
    uint _pad0, _pad1;
};

struct RobMaskParams {
    uint h, w, nch;
    uint tile_size;
    uint flow_nx;
    uint curve_n;     // 1001
    float r_t;
    uint _pad0;
};

inline float dogson_quadratic(float x) {
    float ax = fabs(x);
    if (ax <= 0.5f) return -2.f * ax * ax + 1.f;
    if (ax <= 1.5f) return ax * ax - 2.5f * ax + 1.5f;
    return 0.f;
}

// std::lround half-away-from-zero (not banker's round)
inline int lround_away(float x) {
    return (x >= 0.f) ? int(floor(x + 0.5f)) : int(ceil(x - 0.5f));
}

inline int clamp_edge(int v, int hi) {
    float f = clamp(float(v), 0.f, float(hi));
    return int(f);
}

kernel void rob_guide_bayer(device float* guide [[buffer(0)]],
                            device const float* raw [[buffer(1)]],
                            constant RobGuideParams& p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.guide_w || gid.y >= p.guide_h) return;
    uint cfa[2][2] = {{p.cfa00, p.cfa01}, {p.cfa10, p.cfa11}};
    float wb[3] = {p.wb0, p.wb1, p.wb2};
    uint o = (gid.y * p.guide_w + gid.x) * 3u;
    guide[o + 0u] = 0.f;
    guide[o + 1u] = 0.f;
    guide[o + 2u] = 0.f;
    float gsum = 0.f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            uint c = cfa[i][j];
            float v = raw[(gid.y * 2u + uint(i)) * p.raw_w + (gid.x * 2u + uint(j))] / wb[c];
            if (c == 1u) gsum += v;
            else guide[o + c] = v;
        }
    }
    guide[o + 1u] = 0.5f * gsum;
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
        int patch_idy = y / int(p.tile_size);
        int patch_idx = x / int(p.tile_size);
        if (patch_idy >= 0 && patch_idy < int(p.flow_ny) &&
            patch_idx >= 0 && patch_idx < int(p.flow_nx)) {
            uint fi = (uint(patch_idy) * p.flow_nx + uint(patch_idx)) * 2u;
            flow_x = flow[fi + 0u];
            flow_y = flow[fi + 1u];
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
                          constant RobMaskParams& p [[buffer(7)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    float d_sq_ = 0.f, sigma_sq_ = 0.f;
    for (uint ch = 0u; ch < p.nch; ++ch) {
        uint o = (gid.y * p.w + gid.x) * p.nch + ch;
        float brightness = ref_means[o];
        int id_noise = lround_away(1000.f * brightness);
        // Valid brightness ∈[0,1] → id 0..1000; clamp only avoids GPU OOB on +inf tiles.
        uint id = uint(clamp(id_noise, 0, int(p.curve_n) - 1));
        float sigma_t = std_curve[id];
        float d_t = diff_curve[id];
        float sigma_p_sq = ref_vars[o];
        sigma_sq_ += max(sigma_p_sq, sigma_t * sigma_t);
        float d_p_ = fabs(ref_means[o] - comp_means[o]);
        float d_p_sq = d_p_ * d_p_;
        float shrink = d_p_sq / (d_p_sq + d_t * d_t);
        d_sq_ += d_p_sq * shrink * shrink;
    }
    int patch_idy = int(gid.y) / int(p.tile_size);
    int patch_idx = int(gid.x) / int(p.tile_size);
    float s = S[uint(patch_idy) * p.flow_nx + uint(patch_idx)];
    float sig = sigma_sq_;
    R[gid.y * p.w + gid.x] = clamp(s * exp(-d_sq_ / sig) - p.r_t, 0.f, 1.f);
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

    float s_err[9]; // R<=1 → corr<=3; default R=1. Cap at 9 for stack.
    // General R: use max corr 9 (R<=4 would need more — only dispatch for R<=1).
    if (corr > 9) return;

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
                    int my = ry + flow_dy + sdy;
                    int mx = rx + flow_dx + sdx;
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
struct IcaParams {
    uint ref_h, ref_w, mov_h, mov_w;
    uint ny, nx, ts, n_iter;
    uint clamp_edge; // 1 → ica_kernel_8 clamp; 0 → ica_kernel_16 zero-OOB
    uint _pad0, _pad1, _pad2; // 48 bytes for setBytes
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

kernel void ica_refine_tile(device const float* ref [[buffer(0)]],
                            device const float* gradx [[buffer(1)]],
                            device const float* grady [[buffer(2)]],
                            device const float* hess [[buffer(3)]],
                            device const float* mov [[buffer(4)]],
                            device float* flow [[buffer(5)]],
                            constant IcaParams& p [[buffer(6)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= p.ny * p.nx) return;
    uint ty = tid / p.nx;
    uint tx = tid % p.nx;
    int ts = int(p.ts);
    if (ts != 8 && ts != 16) return; // default pyramid sizes only
    int n_pix = ts * ts;
    int oy = int(ty) * ts;
    int ox = int(tx) * ts;
    bool clamp_edge = (p.clamp_edge != 0u);
    int RH = int(p.ref_h), RW = int(p.ref_w);
    int MH = int(p.mov_h), MW = int(p.mov_w);

    uint ho = tid * 4u;
    float h00 = hess[ho + 0u], h01 = hess[ho + 1u];
    float h10 = hess[ho + 2u], h11 = hess[ho + 3u];
    float det = h00 * h11 - h01 * h10;
    if (fabs(det) < 1e-10f) return; // leave flow unchanged (Python early exit)
    float det_inv = 1.f / det;

    float fx = flow[tid * 2u + 0u];
    float fy = flow[tid * 2u + 1u];

    // Max ts=16 → 256; ts=8 uses first 64.
    float s_B0[256];
    float s_B1[256];

    for (uint it = 0u; it < p.n_iter; ++it) {
        // math.modf + int() truncation toward zero (ICA.py)
        float floor_fx = trunc(fx);
        float floor_fy = trunc(fy);
        float frac_x = fx - floor_fx;
        float frac_y = fy - floor_fy;
        int floor_off_x = int(floor_fx);
        int floor_off_y = int(floor_fy);

        for (int i = 0; i < ts; ++i) {
            int py = oy + i;
            for (int j = 0; j < ts; ++j) {
                int px = ox + j;
                int tpix = i * ts + j;
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
        }
        float B0 = butterfly_reduce_sum_metal(s_B0, n_pix);
        float B1 = butterfly_reduce_sum_metal(s_B1, n_pix);
        fx += det_inv * (h11 * B0 - h01 * B1);
        fy += det_inv * (-h10 * B0 + h00 * B1);
    }

    flow[tid * 2u + 0u] = fx;
    flow[tid * 2u + 1u] = fy;
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
struct MergeNormParams {
    uint bh, Ws, nch, bake;
    float wb0, wb1, wb2;
    float m00, m01, m02, m10, m11, m12, m20, m21, m22;
};

inline float norm_to_srgb(float v) {
    v = clamp(v, 0.f, 1.f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * pow(v, 1.f / 2.4f) - 0.055f;
}

kernel void merge_normalize_rgb16(device const float* num [[buffer(0)]],
                                  device const float* den [[buffer(1)]],
                                  device ushort* out [[buffer(2)]],
                                  constant MergeNormParams& p [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.Ws || gid.y >= p.bh) return;
    uint pi = (gid.y * p.Ws + gid.x) * p.nch;
    float d0 = den[pi];
    float cn0 = (d0 > 0.f) ? num[pi] / d0 : 0.f;
    float cn1 = 0.f, cn2 = 0.f;
    if (p.nch >= 2u) {
        float d1 = den[pi + 1u];
        cn1 = (d1 > 0.f) ? num[pi + 1u] / d1 : 0.f;
    }
    if (p.nch >= 3u) {
        float d2 = den[pi + 2u];
        cn2 = (d2 > 0.f) ? num[pi + 2u] / d2 : 0.f;
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
    float v0 = (p.bake != 0u) ? norm_to_srgb(lin0) : clamp(lin0, 0.f, 1.f);
    float v1 = (p.bake != 0u) ? norm_to_srgb(lin1) : clamp(lin1, 0.f, 1.f);
    float v2 = (p.bake != 0u) ? norm_to_srgb(lin2) : clamp(lin2, 0.f, 1.f);
    uint o = (gid.y * p.Ws + gid.x) * 3u;
    out[o + 0u] = ushort(v0 * 65535.f + 0.5f);
    out[o + 1u] = ushort(v1 * 65535.f + 0.5f);
    out[o + 2u] = ushort(v2 * 65535.f + 0.5f);
}
