// GCam 6.1.021 SABRE stages (CPU drivers around core/sabre61_shared.h). See sabre61.h.
#include "sabre61.h"
#include "parallel.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#if defined(__APPLE__)
namespace hhsr { namespace sabre61 {
// core/sabre61_metal.mm. Same contract as the CPU band below; false = not run (caller falls back).
bool merge_denoise_band_metal(const std::vector<const MergeFrame*>& alts, const MergeFrame& base,
                              const BandContext& ctx, int y0, int y1, std::vector<float>& out_rgb);
}}
#endif

namespace hhsr {
namespace sabre61 {

static const float kNsb[4] = {0.9f, 0.5f, 0.05f, 0.25f};   // ((10-1)/10, (2-1)/2, 0.5/10, 0.5/2)

S61Cfa cfa_from(const CFA& cfa) {
    S61Cfa s{0, 0, 0, 1, 1, 0, 1, 1};   // RGGB
    bool g1 = false;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            const int c = cfa.p[i][j];
            if (c == 0) { s.ry = i; s.rx = j; }
            else if (c == 2) { s.by = i; s.bx = j; }
            else if (!g1) { s.g1y = i; s.g1x = j; g1 = true; }
            else { s.g2y = i; s.g2x = j; }
        }
    return s;
}

void noise_model_s2(const Config& cfg, float slope[3], float offset[3]) {
    for (int c = 0; c < 3; ++c) {
        float g = 1.f;
        if (cfg.raw_prewhitened) {
            const float r = cfg.white_balance[c] / cfg.white_balance[1];
            if (std::isfinite(r) && r > 0.f) g = r;
        }
        slope[c] = std::max(0.f, g * cfg.alpha_dng[c]);
        offset[c] = std::max(0.f, g * g * cfg.beta_dng[c]);
    }
}

void build_noise_lut(const float slope[3], const float offset[3], const S61Cfa& cfa, float lut[80]) {
    std::memset(lut, 0, sizeof(float) * 80);
    const float zero_lut[80] = {0.f};
    std::mt19937 rng(12345u);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int kN = 20000;
    int color[2][2];
    color[cfa.ry][cfa.rx] = 0; color[cfa.g1y][cfa.g1x] = 1; color[cfa.g2y][cfa.g2x] = 1; color[cfa.by][cfa.bx] = 2;
    S61GuideParams P{};
    P.Wr = 6; P.Hr = 6; P.row0 = 0; P.nrows = 6; P.cfa = cfa;
    for (int k = 0; k < 4; ++k) P.nsb[k] = kNsb[k];
    float patch[36];
    for (int i = 0; i < 10; ++i) {
        const double L = (double)i / 9.0;
        const float s = (float)(L * L);
        float sd[3];
        for (int c = 0; c < 3; ++c) sd[c] = std::sqrt(std::max(0.f, slope[c] * s + offset[c]));
        double m[3] = {0, 0, 0}, m2[3] = {0, 0, 0}, std_sum = 0, gd = 0, gd2 = 0, grad = 0;
        for (int n = 0; n < kN; ++n) {
            for (int y = 0; y < 6; ++y)
                for (int x = 0; x < 6; ++x) {
                    const int c = color[y & 1][x & 1];
                    patch[y * 6 + x] = std::max(0.f, s + sd[c] * nd(rng));   // loader clips at 0
                }
            float ref4[4], cov3[3], dbg[5];
            s61_guide_pixel(patch, P, zero_lut, 1, 1, ref4, cov3, dbg);
            for (int c = 0; c < 3; ++c) { m[c] += ref4[c]; m2[c] += (double)ref4[c] * ref4[c]; }
            std_sum += ref4[3];
            gd += dbg[4]; gd2 += (double)dbg[4] * dbg[4];
            grad += 0.5 * ((double)dbg[1] + dbg[2]);
        }
        float* r0 = lut + (0 * 10 + i) * 4;
        float* r1 = lut + (1 * 10 + i) * 4;
        for (int c = 0; c < 3; ++c) {
            const double mu = m[c] / kN;
            r0[c] = (float)std::max(0.0, m2[c] / kN - mu * mu);
        }
        r0[3] = (float)(std_sum / kN);
        const double gmu = gd / kN;
        r1[0] = (float)std::max(0.0, gd2 / kN - gmu * gmu);
        r1[1] = (float)(grad / kN);
        r1[2] = 0.f; r1[3] = 0.f;
    }
}

void yuv_noise_model(const float slope[3], const float offset[3], float out[4]) {
    const float c2[3] = {0.2126390f * 0.2126390f, 0.7151688f * 0.7151688f, 0.0721923f * 0.0721923f};
    out[0] = slope[0] * c2[0] + slope[1] * c2[1] + slope[2] * c2[2];
    out[1] = -slope[0] * c2[0] + slope[1] * c2[1] * 0.297327f;
    out[2] = slope[1] * c2[1] * 0.100944f - slope[2] * c2[2];
    out[3] = offset[0] * c2[0] + offset[1] * c2[1] + offset[2] * c2[2];
}

static void finish_flow_tex(FlowTex& t) {
    // .z = length of the min-max extent of the flow (uv) over the 3x3 tile neighbourhood
    std::vector<float> z((size_t)t.nx * t.ny, 0.f);
    for (int ty = 0; ty < t.ny; ++ty)
        for (int tx = 0; tx < t.nx; ++tx) {
            float mnx = 1e30f, mxx = -1e30f, mny = 1e30f, mxy = -1e30f;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int yy = std::min(t.ny - 1, std::max(0, ty + dy));
                    const int xx = std::min(t.nx - 1, std::max(0, tx + dx));
                    const float* f = &t.data[((size_t)yy * t.nx + xx) * 4];
                    mnx = std::min(mnx, f[0]); mxx = std::max(mxx, f[0]);
                    mny = std::min(mny, f[1]); mxy = std::max(mxy, f[1]);
                }
            z[(size_t)ty * t.nx + tx] = std::sqrt((mxx - mnx) * (mxx - mnx) + (mxy - mny) * (mxy - mny));
        }
    for (size_t i = 0; i < z.size(); ++i) t.data[i * 4 + 2] = s61_q16(z[i]);   // RGBA16F upload
}

FlowTex make_flow_tex(const FlowField& flow, int ts, int Wr, int Hr, const Config& cfg) {
    FlowTex t;
    t.nx = flow.nx; t.ny = flow.ny;
    t.fso[0] = (float)Wr / (float)(t.nx * ts);
    t.fso[1] = (float)Hr / (float)(t.ny * ts);
    t.data.assign((size_t)t.nx * t.ny * 4, 0.f);
    // one vector per tile = the full alignment displacement at the tile centre (raw px)
    std::vector<float> px((size_t)t.nx * t.ny), py((size_t)t.nx * t.ny);
    for (int ty = 0; ty < t.ny; ++ty)
        for (int tx = 0; tx < t.nx; ++tx) {
            const float cx = (float)tx * ts + 0.5f * ts - 0.5f, cy = (float)ty * ts + 0.5f * ts - 0.5f;
            float dx, dy;
            if (flow.has_dense()) flow.sample_dense(cy, cx, dx, dy);
            else if (cfg.affine_flow_enabled && flow.has_affine()) flow.sample_affine(cy, cx, ts, dx, dy);
            else { dx = flow.dx(ty, tx); dy = flow.dy(ty, tx); }
            if (flow.has_global_h) {
                float hx, hy;
                apply_homography(flow.global_h, cx + dx, cy + dy, hx, hy);
                dx = hx - cx; dy = hy - cy;
            }
            px[(size_t)ty * t.nx + tx] = std::isfinite(dx) ? dx : 0.f;
            py[(size_t)ty * t.nx + tx] = std::isfinite(dy) ? dy : 0.f;
        }
    // ConvertAlignmentHalide (use_median_filter): 3x3 per-component median of the tile flow, edge-clamped,
    // before it is scaled to uv and packed. .z (local extent) is measured on the filtered field.
    float mn = 0.f, mx = 0.f;
    for (int ty = 0; ty < t.ny; ++ty)
        for (int tx = 0; tx < t.nx; ++tx) {
            float vx[9], vy[9];
            int k = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int qy = std::min(t.ny - 1, std::max(0, ty + dy));
                    const int qx = std::min(t.nx - 1, std::max(0, tx + dx));
                    vx[k] = px[(size_t)qy * t.nx + qx];
                    vy[k] = py[(size_t)qy * t.nx + qx];
                    ++k;
                }
            std::nth_element(vx, vx + 4, vx + 9);
            std::nth_element(vy, vy + 4, vy + 9);
            const float mdx = vx[4], mdy = vy[4];
            mn = std::min(mn, mdy); mx = std::max(mx, mdy);
            float* f = &t.data[((size_t)ty * t.nx + tx) * 4];
            f[0] = s61_q16(mdx / (float)Wr);
            f[1] = s61_q16(mdy / (float)Hr);
        }
    t.min_dy_px = mn; t.max_dy_px = mx;
    finish_flow_tex(t);
    return t;
}

FlowTex zero_flow_tex(int nx, int ny, int ts, int Wr, int Hr) {
    FlowTex t;
    t.nx = std::max(1, nx); t.ny = std::max(1, ny);
    t.fso[0] = (float)Wr / (float)(t.nx * ts);
    t.fso[1] = (float)Hr / (float)(t.ny * ts);
    t.data.assign((size_t)t.nx * t.ny * 4, 0.f);
    return t;
}

Guide compute_guide(const Image& raw, const S61Cfa& cfa, const float lut[80], int threads) {
    Guide g;
    g.w = raw.w / 2; g.h = raw.h / 2;
    g.ref4.assign((size_t)g.w * g.h * 4, 0.f);
    g.cov3.assign((size_t)g.w * g.h * 3, 0.f);
    S61GuideParams P{};
    P.Wr = raw.w; P.Hr = raw.h; P.row0 = 0; P.nrows = raw.h; P.cfa = cfa;
    for (int k = 0; k < 4; ++k) P.nsb[k] = kNsb[k];
    const float* rp = raw.data.data();
    parallel_rows(g.h, threads, [&](int gy) {
        for (int gx = 0; gx < g.w; ++gx) {
            float r4[4], c3[3];
            s61_guide_pixel(rp, P, lut, gx, gy, r4, c3, nullptr);
            float* o = &g.ref4[((size_t)gy * g.w + gx) * 4];
            for (int k = 0; k < 4; ++k) o[k] = s61_q16(r4[k]);
            float* oc = &g.cov3[((size_t)gy * g.w + gx) * 3];
            for (int k = 0; k < 3; ++k) oc[k] = s61_q16(c3[k]);
        }
    });
    return g;
}

void rejection(const Guide& base, const Guide& alt, const FlowTex& flow, const float lut[80],
               std::vector<uint8_t>& rej, int threads) {
    S61RejParams P{};
    P.Wg = base.w; P.Hg = base.h; P.fnx = flow.nx; P.fny = flow.ny;
    for (int k = 0; k < 4; ++k) { P.fso[k] = flow.fso[k]; P.nsb[k] = kNsb[k]; }
    P.mot_thr = S61_MOTION_THR_PX / (float)base.w;
    P.ub_thr = S61_UNBLOCK_THR_PX / (float)base.w;
    rej.assign((size_t)base.w * base.h, 0);
    parallel_rows(base.h, threads, [&](int gy) {
        for (int gx = 0; gx < base.w; ++gx) {
            const float r = s61_rejection_pixel(base.ref4.data(), alt.ref4.data(), flow.data.data(), lut,
                                                nullptr, 0, 0, P, gx, gy);
            rej[(size_t)gy * base.w + gx] = s61_to_u8(r);
        }
    });
}

void dilate(const std::vector<uint8_t>& rej, int w, int h, std::vector<uint8_t>& keep, int threads) {
    keep.assign((size_t)w * h, 0);
    parallel_rows(h, threads, [&](int gy) {
        for (int gx = 0; gx < w; ++gx) keep[(size_t)gy * w + gx] = s61_to_u8(s61_dilate_pixel(rej.data(), w, h, gx, gy));
    });
}

void copy_mask_accumulate(const std::vector<uint8_t>& keep, std::vector<uint8_t>& acc) {
    if (acc.size() != keep.size()) acc.assign(keep.size(), 0);
    for (size_t i = 0; i < keep.size(); ++i) acc[i] = s61_copy_mask_accumulate(acc[i], keep[i]);
}

// ------------------------------------------------------------------------------------------- merge band
// Raw rows (and guide rows of the kernel covariance) a band of output rows [m0, m1) can touch through
// this frame's flow, including the mirroring at frame_border_padded.
void band_rows(const MergeFrame& f, const BandContext& ctx, int m0, int m1, int& r0, int& r1, int& g0, int& g1) {
    const float sy = (float)ctx.Hr / (float)ctx.Ho;
    r0 = (int)std::floor(m0 * sy + f.flow.min_dy_px) - 8;
    r1 = (int)std::ceil(m1 * sy + f.flow.max_dy_px) + 8;
    if (r0 < 6) { r1 = std::max(r1, 12 - r0); r0 = 0; }
    if (r1 > ctx.Hr - 6) { r0 = std::min(r0, 2 * ctx.Hr - r1 - 12); r1 = ctx.Hr; }
    r0 = std::max(0, r0); r1 = std::min(ctx.Hr, std::max(r1, r0 + 1));
    g0 = std::max(0, r0 / 2 - 2);
    g1 = std::min(ctx.Hg, (r1 + 1) / 2 + 2);
}

static bool merge_denoise_band_cpu(const std::vector<const MergeFrame*>& alts, const MergeFrame& base,
                                   const BandContext& ctx, int y0, int y1, std::vector<float>& out_rgb) {
    const int m0 = std::max(0, y0 - kDenoiseHalo), m1 = std::min(ctx.Ho, y1 + kDenoiseHalo);
    const int mh = m1 - m0, Wo = ctx.Wo;
    std::vector<float> accC((size_t)mh * Wo * 3, 0.f), accW((size_t)mh * Wo * 3, 0.f);
    std::vector<float> cov;
    auto run_frame = [&](const MergeFrame& f, bool is_base) {
        int r0, r1, g0, g1;
        band_rows(f, ctx, m0, m1, r0, r1, g0, g1);
        cov.resize((size_t)(g1 - g0) * ctx.Wg * 3);
        const uint16_t* src = f.cov_h.data() + (size_t)g0 * ctx.Wg * 3;
        for (size_t i = 0; i < cov.size(); ++i) cov[i] = s61_h2f(src[i]);
        S61MergeParams P{};
        P.Wo = ctx.Wo; P.Ho = ctx.Ho; P.Wr = ctx.Wr; P.Hr = ctx.Hr; P.Wg = ctx.Wg; P.Hg = ctx.Hg;
        P.fnx = f.flow.nx; P.fny = f.flow.ny;
        P.raw_row0 = 0; P.raw_nrows = ctx.Hr;
        P.cov_row0 = g0; P.cov_nrows = g1 - g0;
        P.out_row0 = m0; P.is_base = is_base ? 1 : 0; P.use_rejection = ctx.use_rejection ? 1 : 0;
        for (int k = 0; k < 4; ++k) P.fso[k] = f.flow.fso[k];
        P.fbp[0] = 1.5f / ctx.Wr; P.fbp[1] = 1.5f / ctx.Hr; P.fbp[2] = 1.f - 1.5f / ctx.Wr; P.fbp[3] = 1.f - 1.5f / ctx.Hr;
        P.cfa = ctx.cfa;
        const float* rp = f.raw->data.data();
        const uint8_t* mk = is_base ? nullptr : f.keep.data();
        const uint8_t* aw = ctx.acc ? ctx.acc->data() : nullptr;
        parallel_rows(mh, ctx.threads, [&](int ly) {
            const int oy = m0 + ly;
            for (int ox = 0; ox < Wo; ++ox) {
                float c3[3], w3[3];
                s61_merge_pixel(rp, cov.data(), f.flow.data.data(), mk, aw, P, ox, oy, c3, w3);
                float* ac = &accC[((size_t)ly * Wo + ox) * 3];
                float* awp = &accW[((size_t)ly * Wo + ox) * 3];
                if (!is_base) {
                    for (int k = 0; k < 3; ++k) { ac[k] += c3[k]; awp[k] += w3[k]; }
                } else {
                    // base: alpha blend (1/total_weight, ..., color) -> (acc + color) / (acc_w + w), R16F
                    for (int k = 0; k < 3; ++k) {
                        const float inv = 1.0f / (awp[k] + w3[k]);
                        ac[k] = s61_q16(c3[k] * inv + ac[k] * inv);
                    }
                }
            }
        });
    };
    for (const MergeFrame* f : alts) run_frame(*f, false);
    run_frame(base, true);
    std::vector<float>& merged = accC;   // final merged S2 RGB, rows [m0, m1)
    accW.clear(); accW.shrink_to_fit();

    // ---- denoiser: 3 levels, each fp16 between passes ----
    S61DenoiseParams D{};
    D.Wo = ctx.Wo; D.Ho = ctx.Ho; D.Wg = ctx.Wg; D.Hg = ctx.Hg;
    for (int k = 0; k < 4; ++k) D.yuv_noise[k] = ctx.yuv_noise[k];
    D.fb[0] = 1.5f / ctx.Wo; D.fb[1] = 1.5f / ctx.Ho; D.fb[2] = (ctx.Wo - 1.5f) / ctx.Wo; D.fb[3] = (ctx.Ho - 1.5f) / ctx.Ho;
    const uint8_t* aw = ctx.acc ? ctx.acc->data() : nullptr;
    std::vector<float> cur = std::move(merged), nxt;
    int cur0 = m0, cur1 = m1;
    for (int level = 0; level < 3; ++level) {
        const int reach = (level == 0) ? 9 : (level == 1 ? 6 : 0);
        const int o0 = std::max(0, y0 - reach), o1 = std::min(ctx.Ho, y1 + reach);
        nxt.assign((size_t)(o1 - o0) * Wo * 3, 0.f);
        D.level = level; D.in_row0 = cur0; D.in_nrows = cur1 - cur0; D.out_row0 = o0;
        parallel_rows(o1 - o0, ctx.threads, [&](int ly) {
            const int oy = o0 + ly;
            for (int ox = 0; ox < Wo; ++ox) {
                const S61F3 r = s61_denoise_pixel(cur.data(), aw, D, ox, oy);
                float* o = &nxt[((size_t)ly * Wo + ox) * 3];
                if (level < 2) { o[0] = s61_q16(r.x); o[1] = s61_q16(r.y); o[2] = s61_q16(r.z); }
                else { o[0] = r.x; o[1] = r.y; o[2] = r.z; }
            }
        });
        cur.swap(nxt); cur0 = o0; cur1 = o1;
    }
    out_rgb.swap(cur);
    return true;
}

bool merge_denoise_band(const std::vector<const MergeFrame*>& alts, const MergeFrame& base,
                        const BandContext& ctx, int y0, int y1, bool allow_gpu,
                        std::vector<float>& out_rgb) {
#if defined(__APPLE__)
    if (allow_gpu && merge_denoise_band_metal(alts, base, ctx, y0, y1, out_rgb)) return true;
#else
    (void)allow_gpu;
#endif
    return merge_denoise_band_cpu(alts, base, ctx, y0, y1, out_rgb);
}

}  // namespace sabre61
}  // namespace hhsr
