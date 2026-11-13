// GCam 8.2.300 SABRE stages (CPU drivers around core/sabre82_shared.h). See sabre82.h.
#include "sabre82.h"
#include "parallel.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <random>

namespace hhsr {
namespace sabre82 {

// ---------------------------------------------------------------------------------------- per-shot uniforms
// Not fixed by the 8.2 GLSL; values captured on device from the same shader family (2x daylight shot,
// captures/fresh/dbgpull/dbg_meta.txt). Positions/sizes are expressed in guide pixels so they follow our
// frame size the way the captured ones follow GCam's.
namespace {
const float kNsb[4] = {0.9f, 0.5f, 0.05f, 0.25f};                      // 10x2 LUT: ((n-1)/n, 1/2, 0.5/n, 1/4)
const float kCov1[4] = {4.545454025268555f, 1.0101009607315063f, 0.0011417345376685262f, 3.0303030014038086f};
const float kCov2[2] = {0.7265790104866028f, 48.421653747558594f};
const float kPackRG[4] = {-0.015019307844340801f, 0.040903590619564056f, -0.014296099543571472f, 0.03893400356173515f};
const float kPackB[2] = {0.4981805980205536f, 0.071417897939682f};
const float kUnpackRG[4] = {0.3671880066394806f, 24.447731018066406f, 0.3671880066394806f, 25.684490203857422f};
const float kUnpackB[2] = {-6.9755706787109375f, 14.002092361450195f};
const float kCdm[2] = {0.07000000029802322f, 0.3499999940395355f};
const float kBoost = 6.0f;
const float kBoostVarThr = 25.0f;
const float kMotionThrGuidePx = 0.2016f;    // 1.9236641e-4 uv at a 1048-px guide
const float kGuideBorderGuidePx = 1.25f;    // guide pass frame_border_padded
const float kMergeBorderGuidePx = 0.75f;    // rejection + merge frame_border_padded (1.5 raw px)
}  // namespace

S82Cfa cfa_from(const CFA& cfa) {
    S82Cfa s{0, 0, 0, 1, 1, 0, 1, 1, 0};   // RGGB
    bool g1 = false;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            const int c = cfa.p[i][j];
            if (c == 0) { s.ry = i; s.rx = j; }
            else if (c == 2) { s.by = i; s.bx = j; }
            else if (!g1) { s.g1y = i; s.g1x = j; g1 = true; }
            else { s.g2y = i; s.g2x = j; }
        }
    // GBRG = G B / R G
    s.st_gbrg = (cfa.p[0][0] == 1 && cfa.p[0][1] == 2 && cfa.p[1][0] == 0) ? 1 : 0;
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

void build_noise_lut(const float slope[3], const float offset[3], float lut[80]) {
    std::memset(lut, 0, sizeof(float) * 80);
    std::mt19937 rng(12345u);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int kN = 40000;
    for (int i = 0; i < 10; ++i) {
        const double L = (double)i / 9.0;
        const float s = (float)(L * L);
        float sd[3];
        for (int c = 0; c < 3; ++c) sd[c] = std::sqrt(std::max(0.f, slope[c] * s + offset[c]));
        double m[3] = {0, 0, 0}, m2[3] = {0, 0, 0};
        for (int n = 0; n < kN; ++n) {
            const float r = std::sqrt(std::max(0.f, s + sd[0] * nd(rng)));       // loader clips at 0
            const float g1 = std::sqrt(std::max(0.f, s + sd[1] * nd(rng)));
            const float g2 = std::sqrt(std::max(0.f, s + sd[1] * nd(rng)));
            const float b = std::sqrt(std::max(0.f, s + sd[2] * nd(rng)));
            const float v[3] = {r, 0.5f * (g1 + g2), b};
            for (int c = 0; c < 3; ++c) { m[c] += v[c]; m2[c] += (double)v[c] * v[c]; }
        }
        for (int row = 0; row < 2; ++row) {
            float* o = lut + (row * 10 + i) * 4;
            for (int c = 0; c < 3; ++c) {
                const double mu = m[c] / kN;
                o[c] = (float)std::max(0.0, m2[c] / kN - mu * mu);
            }
            o[3] = 0.f;
        }
    }
}

S82GuideParams guide_params(int Wr, int Hr, const S82Cfa& cfa) {
    S82GuideParams P{};
    P.Wr = Wr; P.Hr = Hr; P.Wg = Wr / 2; P.Hg = Hr / 2; P.row0 = 0; P.nrows = Hr; P.cfa = cfa;
    P.force_rgb = 0;
    P.green_clip = FLT_MAX;
    P.fbp[0] = kGuideBorderGuidePx / P.Wg; P.fbp[1] = kGuideBorderGuidePx / P.Hg;
    P.fbp[2] = 1.f - kGuideBorderGuidePx / P.Wg; P.fbp[3] = 1.f - kGuideBorderGuidePx / P.Hg;
    for (int k = 0; k < 4; ++k) { P.nsb[k] = kNsb[k]; P.cov1[k] = kCov1[k]; P.pack_rg[k] = kPackRG[k]; }
    for (int k = 0; k < 2; ++k) { P.cov2[k] = kCov2[k]; P.pack_b[k] = kPackB[k]; }
    return P;
}

S82RejParams rej_params(int Wg, int Hg, const FlowTex& flow, float noise_scale) {
    S82RejParams P{};
    P.Wg = Wg; P.Hg = Hg; P.fnx = flow.nx; P.fny = flow.ny;
    for (int k = 0; k < 4; ++k) { P.fso[k] = flow.fso[k]; P.nsb[k] = kNsb[k]; }
    P.fbp[0] = kMergeBorderGuidePx / Wg; P.fbp[1] = kMergeBorderGuidePx / Hg;
    P.fbp[2] = 1.f - kMergeBorderGuidePx / Wg; P.fbp[3] = 1.f - kMergeBorderGuidePx / Hg;
    P.cdm[0] = kCdm[0]; P.cdm[1] = kCdm[1];
    P.boost = kBoost; P.boost_var_thr = kBoostVarThr;
    P.mot_thr = kMotionThrGuidePx / (float)Wg;
    P.ub_thr = kMotionThrGuidePx / (float)Wg;
    P.noise_scale = noise_scale;
    return P;
}

// ------------------------------------------------------------------------------------------- flow
static void finish_flow_tex(FlowTex& t) {
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
    for (size_t i = 0; i < z.size(); ++i) t.data[i * 4 + 2] = s82_q16(z[i]);
}

FlowTex make_flow_tex(const FlowField& flow, int ts, int Wr, int Hr, const Config& cfg) {
    FlowTex t;
    t.nx = flow.nx; t.ny = flow.ny;
    t.fso[0] = (float)Wr / (float)(t.nx * ts);
    t.fso[1] = (float)Hr / (float)(t.ny * ts);
    t.data.assign((size_t)t.nx * t.ny * 4, 0.f);
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
            mn = std::min(mn, vy[4]); mx = std::max(mx, vy[4]);
            float* f = &t.data[((size_t)ty * t.nx + tx) * 4];
            f[0] = s82_q16(vx[4] / (float)Wr);
            f[1] = s82_q16(vy[4] / (float)Hr);
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

// ------------------------------------------------------------------------------------------- per frame
Guide compute_guide(const Image& raw, const S82Cfa& cfa, const float lut[80], int threads) {
    Guide g;
    g.w = raw.w / 2; g.h = raw.h / 2;
    g.ref4.assign((size_t)g.w * g.h * 4, 0.f);
    g.cov.assign((size_t)g.w * g.h * 3, 0);
    const S82GuideParams P = guide_params(raw.w, raw.h, cfa);
    const float* rp = raw.data.data();
    parallel_rows(g.h, threads, [&](int gy) {
        for (int gx = 0; gx < g.w; ++gx) {
            float r4[4], c3[3];
            s82_guide_pixel(rp, P, lut, gx, gy, r4, c3);
            float* o = &g.ref4[((size_t)gy * g.w + gx) * 4];
            for (int k = 0; k < 4; ++k) o[k] = s82_q16(r4[k]);
            uint16_t* oc = &g.cov[((size_t)gy * g.w + gx) * 3];
            for (int k = 0; k < 3; ++k) oc[k] = (uint16_t)s82_unorm10_code(c3[k]);
        }
    });
    return g;
}

std::vector<float> base_reference(const Guide& base, int threads) {
    std::vector<float> out((size_t)base.w * base.h * 4);
    parallel_rows(base.h, threads, [&](int gy) {
        for (int gx = 0; gx < base.w; ++gx) {
            const S82F4 r = s82_base_ref_pixel(base.ref4.data(), base.w, base.h, gx, gy);
            float* o = &out[((size_t)gy * base.w + gx) * 4];
            o[0] = s82_q16(r.x); o[1] = s82_q16(r.y); o[2] = s82_q16(r.z); o[3] = s82_q16(r.w);
        }
    });
    return out;
}

void rejection(const std::vector<float>& base_ref, const Guide& alt, const FlowTex& flow, const float lut[80],
               float noise_scale, std::vector<uint8_t>& rej, int threads) {
    const S82RejParams P = rej_params(alt.w, alt.h, flow, noise_scale);
    rej.assign((size_t)alt.w * alt.h, 0);
    parallel_rows(alt.h, threads, [&](int gy) {
        for (int gx = 0; gx < alt.w; ++gx)
            rej[(size_t)gy * alt.w + gx] =
                s82_to_u8(s82_rejection_pixel(base_ref.data(), alt.ref4.data(), flow.data.data(), lut, P, gx, gy));
    });
}

void dilate(const std::vector<uint8_t>& rej, int w, int h, std::vector<uint8_t>& keep, int threads) {
    const int ow = std::max(1, w / 2), oh = std::max(1, h / 2);
    keep.assign((size_t)ow * oh, 0);
    parallel_rows(oh, threads, [&](int oy) {
        for (int ox = 0; ox < ow; ++ox)
            keep[(size_t)oy * ow + ox] = s82_to_u8(s82_dilate_pixel(rej.data(), w, h, ow, oh, ox, oy));
    });
}

// ------------------------------------------------------------------------------------------- merge band
void band_rows(const MergeFrame& f, const BandContext& ctx, int y0, int y1, int& r0, int& r1, int& g0, int& g1) {
    const float sy = (float)ctx.Hr / (float)ctx.Ho;
    r0 = (int)std::floor(y0 * sy + f.flow.min_dy_px) - 6;
    r1 = (int)std::ceil(y1 * sy + f.flow.max_dy_px) + 6;
    if (r0 < 6) { r1 = std::max(r1, 12 - r0); r0 = 0; }
    if (r1 > ctx.Hr - 6) { r0 = std::min(r0, 2 * ctx.Hr - r1 - 12); r1 = ctx.Hr; }
    r0 = std::max(0, r0); r1 = std::min(ctx.Hr, std::max(r1, r0 + 1));
    g0 = std::max(0, r0 / 2 - 2);
    g1 = std::min(ctx.Hg, (r1 + 1) / 2 + 2);
}

S82MergeParams merge_params(const MergeFrame& f, const BandContext& ctx, bool is_base, int r0, int r1,
                            int g0, int g1, int out_row0) {
    S82MergeParams P{};
    P.Wo = ctx.Wo; P.Ho = ctx.Ho; P.Wr = ctx.Wr; P.Hr = ctx.Hr; P.Wg = ctx.Wg; P.Hg = ctx.Hg;
    P.Wm = ctx.Wm; P.Hm = ctx.Hm; P.fnx = f.flow.nx; P.fny = f.flow.ny;
    P.raw_row0 = r0; P.raw_nrows = r1 - r0; P.cov_row0 = g0; P.cov_nrows = g1 - g0;
    P.out_row0 = out_row0; P.use_mask = is_base ? 0 : 1;
    for (int k = 0; k < 4; ++k) { P.fso[k] = f.flow.fso[k]; P.unpack_rg[k] = kUnpackRG[k]; }
    P.unpack_b[0] = kUnpackB[0]; P.unpack_b[1] = kUnpackB[1];
    P.fbp[0] = kMergeBorderGuidePx / ctx.Wg; P.fbp[1] = kMergeBorderGuidePx / ctx.Hg;
    P.fbp[2] = 1.f - kMergeBorderGuidePx / ctx.Wg; P.fbp[3] = 1.f - kMergeBorderGuidePx / ctx.Hg;
    P.cfa = ctx.cfa;
    return P;
}

static bool merge_band_cpu(const std::vector<const MergeFrame*>& frames, const BandContext& ctx, int y0, int y1,
                           std::vector<float>& out_rgb) {
    const int bh = y1 - y0, Wo = ctx.Wo;
    std::vector<float> accC((size_t)bh * Wo * 3, 0.f), accW((size_t)bh * Wo * 3, 0.f), cov;
    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const MergeFrame& f = *frames[fi];
        const bool is_base = (fi + 1 == frames.size());
        int r0, r1, g0, g1;
        band_rows(f, ctx, y0, y1, r0, r1, g0, g1);
        cov.resize((size_t)(g1 - g0) * ctx.Wg * 3);
        const uint16_t* src = f.cov.data() + (size_t)g0 * ctx.Wg * 3;
        for (size_t i = 0; i < cov.size(); ++i) cov[i] = (float)src[i] * (1.0f / 1023.0f);
        const S82MergeParams P = merge_params(f, ctx, is_base, 0, ctx.Hr, g0, g1, y0);   // full raw plane on CPU
        const float* rp = f.raw->data.data();
        const uint8_t* mk = is_base ? nullptr : f.keep.data();
        parallel_rows(bh, ctx.threads, [&](int ly) {
            for (int ox = 0; ox < Wo; ++ox) {
                float c3[3], w3[3];
                s82_merge_pixel(rp, cov.data(), f.flow.data.data(), mk, P, ox, y0 + ly, c3, w3);
                float* ac = &accC[((size_t)ly * Wo + ox) * 3];
                float* aw = &accW[((size_t)ly * Wo + ox) * 3];
                for (int k = 0; k < 3; ++k) { ac[k] += c3[k]; aw[k] += w3[k]; }
            }
        });
    }
    // SabreDehomogenize: colour * 1 / max(weight, 1e-7)
    for (size_t i = 0; i < accC.size(); ++i) accC[i] = accC[i] * (1.0f / std::max(accW[i], S82_DEHOMOG_EPS));
    out_rgb.swap(accC);
    return true;
}

bool merge_band(const std::vector<const MergeFrame*>& frames, const BandContext& ctx, int y0, int y1,
                bool allow_gpu, std::vector<float>& out_rgb) {
#if defined(__APPLE__)
    if (allow_gpu && merge_band_metal(frames, ctx, y0, y1, out_rgb)) return true;
#else
    (void)allow_gpu;
#endif
    return merge_band_cpu(frames, ctx, y0, y1, out_rgb);
}

}  // namespace sabre82
}  // namespace hhsr
