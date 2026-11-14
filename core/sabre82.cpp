// GCam 8.2.300 SABRE stages (CPU drivers around core/sabre82_shared.h). See sabre82.h.
#include "sabre82.h"
#include "parallel.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace hhsr {
namespace sabre82 {

// ---------------------------------------------------------------------------------------- per-shot uniforms
// Not fixed by the 8.2 GLSL; values captured on device from the same shader family (2x daylight shot,
// captures/fresh/dbgpull/dbg_meta.txt). Positions/sizes are expressed in guide pixels so they follow our
// frame size the way the captured ones follow GCam's.
namespace {
const float kNsb[4] = {0.9f, 0.5f, 0.05f, 0.25f};                      // 10x2 LUT: ((n-1)/n, 1/2, 0.5/n, 1/4)
// The captured device uniforms (= cov_tuning at K ~ 6.22); default_guide_tuning only. Bursts use cov_tuning.
const float kCov1[4] = {4.545454025268555f, 1.0101009607315063f, 0.0011417345376685262f, 3.0303030014038086f};
const float kCov2[2] = {0.7265790104866028f, 48.421653747558594f};
// MergeOptions covariance curves, (K, value) knots, built by FUN_0141068c from .rodata
// (Documents/super/SABRE_8.2_merge_covariance_SPEC.md section 5).
const float kCurveBase[] = {7.2f, 0.33f, 14.4f, 0.25f};                                   // p0
const float kCurveBlurred[] = {5.4f, 4.2f, 11.0f, 4.0f, 27.0f, 3.0f};                     // p1
const float kCurveGradClip[] = {0.9f, 0.01f, 1.0f, 0.0012f, 3.6f, 0.0012f, 12.6f, 0.001f, 27.0f, 0.001f};  // p2
const float kCurveGradTrans[] = {0.9f, 0.028f, 2.7f, 0.025f, 10.8f, 0.015f, 27.0f, 0.009f};  // p3 (1/transition)
const float kCurveStretched[] = {16.0f, 3.0f};                                             // p4
const float kCurveShrinked[] = {16.0f, 1.5f};                                              // p5
const float kPackRG[4] = {-0.015019307844340801f, 0.040903590619564056f, -0.014296099543571472f, 0.03893400356173515f};
const float kPackB[2] = {0.4981805980205536f, 0.071417897939682f};
const float kUnpackRG[4] = {0.3671880066394806f, 24.447731018066406f, 0.3671880066394806f, 25.684490203857422f};
const float kUnpackB[2] = {-6.9755706787109375f, 14.002092361450195f};
const float kCdm[2] = {0.07000000029802322f, 0.3499999940395355f};
const float kBoost = 6.0f;
const float kBoostVarThr = 25.0f;
const float kMotionThrGuidePx = 0.2016f;    // 1.9236641e-4 uv at a 1048-px guide
// Residual-shift test (hhsr extension): raw px per 2|s| of the 7x7 across-edge LK (it under-reads a thin edge;
// calibrated by s82_resid_cal: alt = base reference shifted by a known amount).
const float kResidCal = 1.7f;
const float kForceRgbMaxSnr = 10.0f;   // FUN_012bcea8: force_reference_color_rgb = (snr <= 10)   // s82_rcal on burst12: 1.98 across vertical edges, 1.46 across horizontal (geo mean)
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

// var(sqrt(max(0, s + sd*Z))), Z ~ N(0,1), by quadrature on a fine grid over [-9, 9] (exact to ~1e-9 relative
// for this integrand; deterministic, unlike the Monte Carlo it replaces).
static double var_sqrt_clipped(double s, double sd) {
    constexpr int kN = 20001;
    constexpr double kZ = 9.0;
    double wsum = 0, m1 = 0, m2 = 0;
    for (int k = 0; k < kN; ++k) {
        const double z = -kZ + 2.0 * kZ * k / (kN - 1);
        const double w = std::exp(-0.5 * z * z);
        const double f = std::sqrt(std::max(0.0, s + sd * z));
        wsum += w; m1 += w * f; m2 += w * f * f;
    }
    m1 /= wsum; m2 /= wsum;
    return std::max(0.0, m2 - m1 * m1);
}

void build_noise_lut(const float slope[3], const float offset[3], float lut[80]) {
    std::memset(lut, 0, sizeof(float) * 80);
    for (int i = 0; i < 10; ++i) {
        const double L = (double)i / 9.0;
        const double s = L * L;
        double v[3];
        for (int c = 0; c < 3; ++c) {
            const double sd = std::sqrt(std::max(0.0, (double)slope[c] * s + (double)offset[c]));
            v[c] = var_sqrt_clipped(s, sd);                   // loader clips at 0
        }
        v[1] *= 0.5;                                          // .y = 0.5*(sqrt G1 + sqrt G2), independent sites
        for (int row = 0; row < 2; ++row) {
            float* o = lut + (row * 10 + i) * 4;
            for (int c = 0; c < 3; ++c) o[c] = (float)v[c];
            o[3] = 0.f;
        }
    }
}

// ------------------------------------------------------------------------------------------- tuning
float eval_curve(const float* xy, int n, float K) {
    if (n <= 0) return 0.f;
    if (K <= xy[0]) return xy[1];
    if (xy[2 * (n - 1)] <= K) return xy[2 * (n - 1) + 1];
    int i = 1;                                   // first knot with x > K (upper bound)
    while (i < n - 1 && xy[2 * i] <= K) ++i;
    const float* hi = xy + 2 * i;
    const float t = (K - hi[-2]) / (hi[0] - hi[-2]);
    return hi[1] * t + hi[-1] * (1.0f - t);
}

float estimate_snr(float signal, float shot_noise_factor, float read_noise_var) {
    if (!(signal > 0.f) || !std::isfinite(signal)) signal = 0.18f;
    const float v = read_noise_var + shot_noise_factor * signal;
    return v > 0.f ? signal / std::sqrt(v) : 0.f;
}

CovTuning cov_tuning(float snr, int num_frames) {
    CovTuning t;
    t.snr = snr;
    t.K = std::sqrt((float)num_frames / 12.0f) * snr;
    t.p[0] = eval_curve(kCurveBase, 2, t.K);
    t.p[1] = eval_curve(kCurveBlurred, 3, t.K);
    t.p[2] = eval_curve(kCurveGradClip, 5, t.K);
    t.p[3] = eval_curve(kCurveGradTrans, 4, t.K);
    t.p[4] = eval_curve(kCurveStretched, 1, t.K);
    t.p[5] = eval_curve(kCurveShrinked, 1, t.K);
    const float p0 = t.p[0];
    t.cov1[0] = 1.0f / (p0 / t.p[5]);
    t.cov1[1] = 1.0f / (p0 * t.p[4]);
    t.cov1[2] = t.p[2];
    t.cov1[3] = 1.0f / p0;
    t.cov2[0] = 1.0f / (p0 * t.p[1]);
    t.cov2[1] = 1.0f / t.p[3];
    return t;
}

float green_clip_point(float K, float white_level) {
    const int white = (int)white_level;
    if (!(K > 0.f) || K * 1.0f >= 1.0f || white <= 3) return FLT_MAX;
    return (K * 1.0f + (K * 1.0f)) * 0.5f * ((float)(white - 3) / (float)white);
}

void extract_eps(const Config& cfg, const CFA& cfa, float K, float eps[4]) {
    for (int s = 0; s < 4; ++s) {
        eps[s] = 0.f;
        int c = cfa.p[s / 2][s % 2];
        if (c < 0 || c > 2) c = 1;
        const float black = cfg.has_black_levels ? cfg.black_levels[c] : 0.f;
        const float range = cfg.white_level - black;
        if (!(range > 0.f) || !(K > 0.f)) continue;
        float g = 1.f;
        if (cfg.raw_prewhitened) {
            const float r = cfg.white_balance[c] / cfg.white_balance[1];
            if (std::isfinite(r) && r > 0.f) g = r;
        }
        eps[s] = 1e-4f * K * g / range;
    }
}

GuideTuning guide_tuning(const CovTuning& cov, float green_clip) {
    GuideTuning t;
    for (int k = 0; k < 4; ++k) t.cov1[k] = cov.cov1[k];
    for (int k = 0; k < 2; ++k) t.cov2[k] = cov.cov2[k];
    t.green_clip = green_clip;
    // force_reference_color_rgb (FUN_012bcea8): the per-shot merge option at +0x54 is
    // `EstimateSnr(...) <= 10.0f` -- the same SNR that sets K. Low-SNR shots reject on filtered RGB only.
    // Checked on every capture with known uniforms: car SNR 20.9 -> 0, house 6.5 -> 1, c2 7.3 -> 1.
    t.force_rgb = cov.snr <= kForceRgbMaxSnr ? 1 : 0;
    return t;
}

Image gcam_bayer_input(const Image& im, const Config& cfg, float exposure_gain) {
    // The loader delivered v = max(((DN + 1e-4) - black) / (white - black) * wbr, 0) * g (g = exposure gain,
    // brightened sites capped at wbr). Recover the integer DN (the loader's rounding is far below half a DN),
    // then form GCam's guide input: the guide reads the extracted Bayer as uvec4 (DN, the +1e-4 truncated)
    // and applies one MADD, fma(DN, gains, black_levels_times_gains), with
    //   gains = ((exposure ratio * wb[c]) / (white - black[c])),  black_levels_times_gains = -(gains' * black).
    Image out(im.h, im.w, 1);
    const float g = exposure_gain > 0.f ? exposure_gain : 1.f;
    float gain[2][2], blt[2][2], wbr[2][2], blk[2][2], den[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            int c = cfg.cfa.p[i][j];
            if (c < 0 || c > 2) c = 1;
            float r = 1.f;
            if (cfg.raw_prewhitened) {
                const float q = cfg.white_balance[c] / cfg.white_balance[1];
                if (std::isfinite(q) && q > 0.f) r = q;
            }
            const float black = cfg.has_black_levels ? cfg.black_levels[c] : 0.f;
            const float white = (float)(int)cfg.white_level;
            const float d = white - black > 0.f ? white - black : 1.f;
            const float gw = g * r;
            wbr[i][j] = r; blk[i][j] = black; den[i][j] = d;
            gain[i][j] = gw / d;
            blt[i][j] = -(gw * black) / d;
        }
    const float dn_max = cfg.white_level > 0.f ? cfg.white_level : 65535.f;
    parallel_rows(im.h, 0, [&](int y) {
        for (int x = 0; x < im.w; ++x) {
            const int i = y & 1, j = x & 1;
            const float v = im.data[(size_t)y * im.w + x];
            float dn;
            if (!(v > 0.f)) dn = 0.f;                                          // at/below black
            else if (g != 1.f && v >= wbr[i][j]) dn = dn_max;                  // capped by the exposure match
            else dn = std::nearbyint((double)v / g / wbr[i][j] * den[i][j] + blk[i][j] - 1e-4);
            dn = std::min(std::max(dn, 0.f), 65535.f);
            out.data[(size_t)y * im.w + x] = s82m_fma(dn, gain[i][j], blt[i][j]);
        }
    });
    out.exposure = im.exposure;
    return out;
}

GuideTuning default_guide_tuning() {
    GuideTuning t;
    for (int k = 0; k < 4; ++k) t.cov1[k] = kCov1[k];
    for (int k = 0; k < 2; ++k) t.cov2[k] = kCov2[k];
    t.green_clip = FLT_MAX;
    return t;
}

S82GuideParams guide_params(int Wr, int Hr, const S82Cfa& cfa, const GuideTuning& t) {
    S82GuideParams P{};
    P.Wr = Wr; P.Hr = Hr; P.Wg = Wr / 2; P.Hg = Hr / 2; P.row0 = 0; P.nrows = Hr; P.cfa = cfa;
    P.force_rgb = t.force_rgb;
    P.green_clip = t.green_clip;
    P.fbp[0] = kGuideBorderGuidePx / P.Wg; P.fbp[1] = kGuideBorderGuidePx / P.Hg;
    P.fbp[2] = 1.f - kGuideBorderGuidePx / P.Wg; P.fbp[3] = 1.f - kGuideBorderGuidePx / P.Hg;
    for (int k = 0; k < 4; ++k) { P.nsb[k] = kNsb[k]; P.cov1[k] = t.cov1[k]; P.pack_rg[k] = kPackRG[k]; }
    for (int k = 0; k < 2; ++k) { P.cov2[k] = t.cov2[k]; P.pack_b[k] = kPackB[k]; }
    return P;
}

Unblocker zero_unblocker(int Wr, int Hr) {
    Unblocker u;
    // Output size per axis = ((dim >> 1) + 7) >> 3 (FUN_012c32f0), i.e. ceil(guide / 8).
    u.w = std::max(1, ((Wr >> 1) + 7) >> 3);
    u.h = std::max(1, ((Hr >> 1) + 7) >> 3);
    // unblocker_scale: the rejection's tex_coord spans the guide (Wr raw px), the texture 16*w raw px. The
    // captured burst (2096 = 131*16) had (1, 1); not recovered for other sizes, this keeps the tiles aligned.
    u.scale[0] = (float)Wr / (float)(16 * u.w);
    u.scale[1] = (float)Hr / (float)(16 * u.h);
    u.data.assign((size_t)u.w * u.h, 0);
    return u;
}

namespace {
inline int64_t uhadd(int64_t a, int64_t b) { return (a + b) >> 1; }
inline int64_t urhadd(int64_t a, int64_t b) { return (a + b + 1) >> 1; }
// Halide BoundaryConditions::mirror_image over [0, n): ..1 0 | 0 1.. (edge sample repeated).
inline int mirror_idx(int p, int n) {
    int m = p % (2 * n);
    if (m < 0) m += 2 * n;
    return m >= n ? 2 * n - 1 - m : m;
}
// 3x3 halving-add blur of p (ph x pw), output (ph-2) x (pw-2):
//   col = H2(H1(p[y-1], p[y+1]), p[y]);  out = H4(H3(col[x-1], col[x+1]), col[x])
void blur3(const std::vector<int64_t>& p, int ph, int pw, bool all_round, std::vector<int64_t>& out) {
    const int oh = ph - 2, ow = pw - 2;
    std::vector<int64_t> col((size_t)oh * pw);
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < pw; ++x) {
            const int64_t t = p[(size_t)y * pw + x], m = p[(size_t)(y + 1) * pw + x], b = p[(size_t)(y + 2) * pw + x];
            col[(size_t)y * pw + x] = urhadd(all_round ? urhadd(t, b) : uhadd(t, b), m);
        }
    out.assign((size_t)oh * ow, 0);
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x) {
            const int64_t l = col[(size_t)y * pw + x], m = col[(size_t)y * pw + x + 1], r = col[(size_t)y * pw + x + 2];
            out[(size_t)y * ow + x] = all_round ? urhadd(urhadd(l, r), m) : uhadd(urhadd(l, r), m);
        }
}
inline float f32_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
}  // namespace

Unblocker unblocker_dn(const uint16_t* raw, int W, int H, bool green_on_even, float shot, float read, float black,
                       bool gcam82_rounding) {
    Unblocker u = zero_unblocker(W, H);
    const int T = 8, N = T * T, Wh = W >> 1, Hh = H >> 1, ow = u.w, oh = u.h;
    if (Wh <= 0 || Hh <= 0) return u;
    // Half-res coordinates [-T-1, T*(o+1)+1): weight tiles -1..o (apron) plus the blur's 1-px apron.
    const int x0 = -T - 1, y0 = -T - 1, pw = T * (ow + 2) + 2, ph = T * (oh + 2) + 2;
    const int g0 = green_on_even ? 0 : 1, g1 = green_on_even ? 3 : 2;
    std::vector<int> mx(pw), my(ph);
    for (int i = 0; i < pw; ++i) mx[i] = mirror_idx(x0 + i, Wh);
    for (int i = 0; i < ph; ++i) my[i] = mirror_idx(y0 + i, Hh);
    auto plane = [&](int c, int iy, int ix) -> int64_t {
        return raw[(size_t)(2 * my[iy] + (c >> 1)) * W + 2 * mx[ix] + (c & 1)];
    };
    std::vector<int64_t> avg((size_t)ph * pw);
    for (int iy = 0; iy < ph; ++iy)
        for (int ix = 0; ix < pw; ++ix) avg[(size_t)iy * pw + ix] = urhadd(plane(g0, iy, ix), plane(g1, iy, ix));
    std::vector<int64_t> G;
    blur3(avg, ph, pw, gcam82_rounding, G);   // (ph-2) x (pw-2), covers half-res [-T, T*(o+1))
    const int gw = pw - 2;
    const int ntx = ow + 2, nty = oh + 2;
    std::vector<int64_t> weight((size_t)nty * ntx);
    const float kDen = f32_bits(0x41638E39u), kScale = f32_bits(0x3E4CCCCDu), kOffset = f32_bits(0x3EE66666u);
    for (int ty = 0; ty < nty; ++ty)
        for (int tx = 0; tx < ntx; ++tx) {
            int64_t S12 = 0, Q12 = 0, S3 = 0, Q3 = 0;
            for (int j = 0; j < T; ++j)
                for (int i = 0; i < T; ++i) {
                    const int gy = ty * T + j, gx = tx * T + i;          // index into G
                    const int iy = gy + 1, ix = gx + 1;                   // same position in the plane grid
                    const int64_t a = plane(g0, iy, ix), b = plane(g1, iy, ix), g = G[(size_t)gy * gw + gx];
                    S12 += a + b; Q12 += a * a + b * b; S3 += g; Q3 += g * g;
                }
            const int64_t mean_g = (S12 + N) / (2 * N);
            const int64_t V_raw = (2 * N * Q12 - S12 * S12 + 2 * N * N) / (4 * N * N);
            const int64_t V_blur = (N * Q3 - S3 * S3 + (N * N) / 2) / (N * N);
            const float sig = std::max((float)mean_g - black, 0.0f);
            const float noise = gcam82_rounding ? read + shot * sig : std::fma(shot, sig, read);
            const float num = (float)V_raw + noise * -4.0f;
            float uu;
            if (V_blur == 0) uu = num > 0.f ? kScale : 0.f;
            else {
                const float den = (float)V_blur * kDen;
                const float m = std::sqrt(std::max(num, 0.0f) / den);
                uu = kScale * (m - kOffset);
            }
            uu = std::max(std::min(uu, 1.0f), 0.0f);
            weight[(size_t)ty * ntx + tx] = (int64_t)(uu * 255.0f);   // fcvtzs: truncation
        }
    std::vector<int64_t> out;
    blur3(weight, nty, ntx, gcam82_rounding, out);   // blur$1: oh x ow
    for (size_t i = 0; i < out.size(); ++i) u.data[i] = (uint8_t)std::min<int64_t>(255, std::max<int64_t>(0, out[i]));
    return u;
}

Unblocker compute_unblocker(const Image& base, const Config& cfg, float slope_g, float offset_g) {
    const int W = base.w, H = base.h;
    const float black = cfg.has_black_levels ? cfg.black_levels[1] : 0.f;
    const float range = cfg.white_level - black;
    if (base.c != 1 || W < 2 || H < 2 || !(range > 0.f) || base.data.size() < (size_t)W * H) return zero_unblocker(W, H);
    // Green sites back to DN (the loader delivered (DN - black)/(white - black), gain 1 on green, K = 1 on the
    // base). The non-green sites are never read.
    std::vector<uint16_t> dn((size_t)W * H, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            if (cfg.cfa.p[y & 1][x & 1] != 1) continue;
            const float v = base.data[(size_t)y * W + x] * range + black;
            dn[(size_t)y * W + x] = (uint16_t)std::min(65535.f, std::max(0.f, std::floor(v + 0.5f)));
        }
    const bool goe = cfg.cfa.p[0][0] == 1 && cfg.cfa.p[1][1] == 1;
    return unblocker_dn(dn.data(), W, H, goe, slope_g * range, offset_g * range * range, black);
}

S82RejParams rej_params(int Wg, int Hg, const FlowTex& flow, float noise_scale, const Unblocker& unb) {
    S82RejParams P{};
    P.unx = unb.w; P.uny = unb.h; P.uscale[0] = unb.scale[0]; P.uscale[1] = unb.scale[1];
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
Guide compute_guide(const Image& raw, const S82Cfa& cfa, const float lut[80], const GuideTuning& t, int threads,
                    bool mali) {
    Guide g;
    g.w = raw.w / 2; g.h = raw.h / 2;
    g.ref4.assign((size_t)g.w * g.h * 4, 0.f);
    g.cov.assign((size_t)g.w * g.h * 3, 0);
    const S82GuideParams P = guide_params(raw.w, raw.h, cfa, t);
    const float* rp = raw.data.data();
    parallel_rows(g.h, threads, [&](int gy) {
        for (int gx = 0; gx < g.w; ++gx) {
            float r4[4], c3[3];
            if (mali) s82m_guide_pixel(rp, P, lut, gx, gy, r4, c3);
            else s82_guide_pixel(rp, P, lut, gx, gy, r4, c3);
            float* o = &g.ref4[((size_t)gy * g.w + gx) * 4];
            for (int k = 0; k < 4; ++k) o[k] = s82_q16(r4[k]);
            uint16_t* oc = &g.cov[((size_t)gy * g.w + gx) * 3];
            for (int k = 0; k < 3; ++k) oc[k] = (uint16_t)s82_unorm10_code(c3[k]);
        }
    });
    return g;
}

std::vector<float> base_reference(const Guide& base, int threads, bool mali) {
    std::vector<float> out((size_t)base.w * base.h * 4);
    parallel_rows(base.h, threads, [&](int gy) {
        for (int gx = 0; gx < base.w; ++gx) {
            const S82F4 r = mali ? s82m_base_ref_pixel(base.ref4.data(), base.w, base.h, gx, gy)
                                 : s82_base_ref_pixel(base.ref4.data(), base.w, base.h, gx, gy);
            float* o = &out[((size_t)gy * base.w + gx) * 4];
            o[0] = s82_q16(r.x); o[1] = s82_q16(r.y); o[2] = s82_q16(r.z); o[3] = s82_q16(r.w);
        }
    });
    return out;
}

void rejection(const std::vector<float>& base_ref, const Guide& alt, const FlowTex& flow, const float lut[80],
               float noise_scale, const Unblocker& unb, const ResidShift& rs, std::vector<uint8_t>& rej,
               std::vector<float>& itg, int threads, bool mali) {
    const S82RejParams P = rej_params(alt.w, alt.h, flow, noise_scale, unb);
    rej.assign((size_t)alt.w * alt.h, 0);
    const bool want_it = rs.enabled && rs.gm.size() == rej.size();
    if (want_it) itg.assign(rej.size(), 0.f); else itg.clear();
    parallel_rows(alt.h, threads, [&](int gy) {
        for (int gx = 0; gx < alt.w; ++gx) {
            float it;
            const size_t i = (size_t)gy * alt.w + gx;
            rej[i] = mali ? s82m_to_u8(s82m_rejection_pixel(base_ref.data(), alt.ref4.data(), flow.data.data(),
                                                            unb.data.data(), P, gx, gy, &it))
                          : s82_to_u8(s82_rejection_pixel(base_ref.data(), alt.ref4.data(), flow.data.data(), lut,
                                                          unb.data.data(), P, gx, gy, &it));
            if (want_it) itg[i] = it * rs.gm[i];
        }
    });
}

ResidShift resid_shift_setup(const std::vector<float>& base_ref, int Wg, int Hg, const Config& cfg, int threads) {
    ResidShift rs;
    rs.gm.assign((size_t)Wg * Hg, 0.f);
    parallel_rows(Hg, threads, [&](int y) {
        for (int x = 0; x < Wg; ++x) rs.gm[(size_t)y * Wg + x] = s82_base_grad_pixel(base_ref.data(), Wg, Hg, x, y);
    });
    rs.enabled = cfg.sabre_resid_shift_enabled;
    rs.P.Wg = Wg; rs.P.Hg = Hg; rs.P.Wm = std::max(1, Wg / 2); rs.P.Hm = std::max(1, Hg / 2);
    rs.P.cal = kResidCal;
    rs.P.t0 = cfg.sabre_resid_shift_tol_px;
    rs.P.width = std::max(1e-3f, cfg.sabre_resid_shift_width_px);
    std::vector<float> tmp(rs.gm);
    const double pct = std::min(99.9, std::max(0.0, (double)cfg.sabre_resid_shift_edge_pct));
    const size_t k = std::min(tmp.size() - 1, (size_t)(pct / 100.0 * (double)(tmp.size() - 1)));
    std::nth_element(tmp.begin(), tmp.begin() + (long)k, tmp.end());
    rs.P.thr = tmp[k];
    return rs;
}

void resid_shift_apply(const ResidShift& rs, const std::vector<float>& itg, std::vector<uint8_t>& keep, int threads) {
    if (!rs.enabled || itg.size() != rs.gm.size() || keep.size() != (size_t)rs.P.Wm * rs.P.Hm) return;
    std::vector<float> f(itg.size());
    parallel_rows(rs.P.Hg, threads, [&](int y) {
        for (int x = 0; x < rs.P.Wg; ++x)
            f[(size_t)y * rs.P.Wg + x] = s82_resid_factor_pixel(itg.data(), rs.gm.data(), rs.P, x, y);
    });
    std::vector<uint8_t> out(keep.size());
    parallel_rows(rs.P.Hm, threads, [&](int oy) {
        for (int ox = 0; ox < rs.P.Wm; ++ox)
            out[(size_t)oy * rs.P.Wm + ox] = s82_to_u8(s82_resid_apply_pixel(keep.data(), f.data(), rs.P, ox, oy));
    });
    keep.swap(out);
}

void dilate(const std::vector<uint8_t>& rej, int w, int h, std::vector<uint8_t>& keep, int threads, bool mali) {
    const int ow = std::max(1, w / 2), oh = std::max(1, h / 2);
    keep.assign((size_t)ow * oh, 0);
    parallel_rows(oh, threads, [&](int oy) {
        for (int ox = 0; ox < ow; ++ox)
            keep[(size_t)oy * ow + ox] = mali ? s82m_to_u8(s82m_dilate_pixel(rej.data(), w, h, ox, oy))
                                              : s82_to_u8(s82_dilate_pixel(rej.data(), w, h, ow, oh, ox, oy));
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
    for (int k = 0; k < 4; ++k) P.eps[k] = f.eps[k];
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
