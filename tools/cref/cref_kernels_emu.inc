struct CrefParams {
    uint w, h, rows, cin;      // image width/height; band feature rows; conv input channels
    uint dil, y0, halo, slo;   // dilation; band's first output row; halo; first staged image row
    uint srows, out_rows, vlo, vhi;   // vlo..vhi: band rows that are real image rows (per-layer replicate)
    float na, nb, nfr, _f0;    // noise alpha, beta (>= 1e-12); comparison frames in the consensus
};

inline float cref_clip(float x) { return clamp(x, -8.f, 8.f) * 0.125f; }

inline float cref_stg(device const float* s, constant CrefParams& p, uint plane, int Y, int x) {
    Y = clamp(Y, 0, int(p.h) - 1);
    x = clamp(x, 0, int(p.w) - 1);
    return s[plane * p.srows * p.w + uint(Y - int(p.slo)) * p.w + uint(x)];
}

inline float cref_hp(device const float* s, constant CrefParams& p, uint plane, int Y, int x) {
    float acc = 0.f;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j) acc += cref_stg(s, p, plane, Y + i, x + j);
    return cref_stg(s, p, plane, Y, x) - acc * (1.f / 9.f);
}

kernel void cref_feat(device const float* stg [[buffer(0)]],
                      device float* F [[buffer(1)]],
                      constant CrefParams& p [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.rows) return;
    const int x = int(gid.x);
    const int Y = clamp(int(p.y0) - int(p.halo) + int(gid.y), 0, int(p.h) - 1);
    const float rf = cref_stg(stg, p, 0u, Y, x), cp = cref_stg(stg, p, 1u, Y, x), cs = cref_stg(stg, p, 2u, Y, x);
    const float n = p.nfr;
    const float loo = (n >= 2.f) ? (n * cs - cp) / (n - 1.f) : rf;
    const float m = max(rf, 0.f);
    const float v = p.na * m + p.nb;
    const float isd = 1.f / sqrt(2.f * v), is1 = 1.f / sqrt(v);
    const uint plane = p.rows * p.w, o = gid.y * p.w + gid.x;
    F[o] = clamp(cref_stg(stg, p, 3u, Y, x), 0.f, 1.f);
    F[plane + o] = cref_clip((cp - rf) * isd);
    F[2u * plane + o] = cref_clip((cp - loo) * isd);
    F[3u * plane + o] = cref_clip((rf - loo) * isd);
    F[4u * plane + o] = cref_clip(cref_hp(stg, p, 0u, Y, x) * is1);
    F[5u * plane + o] = cref_clip(cref_hp(stg, p, 1u, Y, x) * is1);
    // 6, 7: flow disagreement and HF variance loss, computed on the host
    // (cref_aux_rows) for the staged rows.
    F[6u * plane + o] = cref_stg(stg, p, 4u, Y, x);
    F[7u * plane + o] = cref_stg(stg, p, 5u, Y, x);
}

// One 3x3 dilated conv (cin -> 8) + ReLU over the band; weights [8][cin][3][3].
kernel void cref_conv(device const float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant float* Wt [[buffer(2)]],
                      constant float* Bt [[buffer(3)]],
                      constant CrefParams& p [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.rows) return;
    const uint plane = p.rows * p.w;
    float acc[8];
    for (uint co = 0u; co < 8u; ++co) acc[co] = Bt[co];
    const int d = int(p.dil);
    for (int ky = 0; ky < 3; ++ky) {
        const int yy = clamp(int(gid.y) + (ky - 1) * d, int(p.vlo), int(p.vhi));
        for (int kx = 0; kx < 3; ++kx) {
            const int xx = clamp(int(gid.x) + (kx - 1) * d, 0, int(p.w) - 1);
            const uint q = uint(yy) * p.w + uint(xx);
            for (uint ci = 0u; ci < p.cin; ++ci) {
                const float v = in[ci * plane + q];
                const uint wi = ci * 9u + uint(ky * 3 + kx);
                for (uint co = 0u; co < 8u; ++co) acc[co] += Wt[co * p.cin * 9u + wi] * v;
            }
        }
    }
    const uint o = gid.y * p.w + gid.x;
    for (uint co = 0u; co < 8u; ++co) out[co * plane + o] = max(0.f, acc[co]);
}

// 1x1 head on the last activations + logit refine of R, for the band's output rows.
kernel void cref_head(device const float* act [[buffer(0)]],
                      device const float* F [[buffer(1)]],
                      device float* Rout [[buffer(2)]],
                      constant float* WO [[buffer(3)]],
                      constant CrefParams& p [[buffer(4)]],
                      constant float& bo [[buffer(5)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.out_rows) return;
    const uint plane = p.rows * p.w, q = (gid.y + p.halo) * p.w + gid.x;
    float d = bo;
    for (uint c = 0u; c < 8u; ++c) d += WO[c] * act[c * plane + q];
    const float r0 = clamp(F[q], 0.02f, 0.98f);
    const float z = log(r0 / (1.f - r0)) + d;
    Rout[gid.y * p.w + gid.x] = 1.f / (1.f + exp(-z));
}
