#include "global_homography.h"
#include "parallel.h"
#include <array>
#include <cmath>
#include <algorithm>
#include <vector>

namespace hhsr {
namespace {

// Bilinear sample of a single-channel image, clamped to edge. Returns the value
// and whether the (unclamped) sample was inside the image.
inline f32 sample_clamp(const Image& im, f32 x, f32 y) {
    if (im.w <= 0 || im.h <= 0) return 0.f;
    f32 cx = std::min(std::max(x, 0.f), (f32)(im.w - 1));
    f32 cy = std::min(std::max(y, 0.f), (f32)(im.h - 1));
    int x0 = (int)std::floor(cx), y0 = (int)std::floor(cy);
    int x1 = std::min(x0 + 1, im.w - 1), y1 = std::min(y0 + 1, im.h - 1);
    f32 ax = cx - (f32)x0, ay = cy - (f32)y0;
    f32 top = im.at(y0, x0) + (im.at(y0, x1) - im.at(y0, x0)) * ax;
    f32 bot = im.at(y1, x0) + (im.at(y1, x1) - im.at(y1, x0)) * ax;
    return top + (bot - top) * ay;
}

// 2x box-average downsample (single channel).
Image downsample2x(const Image& in) {
    int oh = std::max(1, in.h / 2), ow = std::max(1, in.w / 2);
    Image out(oh, ow, 1);
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            int y0 = 2 * y, x0 = 2 * x;
            int y1 = std::min(y0 + 1, in.h - 1), x1 = std::min(x0 + 1, in.w - 1);
            out.at(y, x) = 0.25f * (in.at(y0, x0) + in.at(y0, x1) +
                                    in.at(y1, x0) + in.at(y1, x1));
        }
    }
    return out;
}

// Downsample (by halving) until max(h,w) <= max_dim. Returns the image and the
// linear scale factor small/full applied.
Image downsample_to(const Image& in, int max_dim, f32& scale_out) {
    Image cur = in;
    f32 scale = 1.f;
    while (std::max(cur.h, cur.w) > max_dim && cur.h > 8 && cur.w > 8) {
        cur = downsample2x(cur);
        scale *= 0.5f;
    }
    scale_out = scale;
    return cur;
}

// Exact pixel-centre map between a box-downsampled level and its source image:
// the 2x box average sits BETWEEN the pixel pair it averages, so after any
// number of halvings  x_level = s*x_src - (1-s)/2,  not pure scaling. M maps
// SRC -> LEVEL, Minv the reverse. Conjugating a homography with the pure-scale
// matrix instead makes a lifted rotation pivot (1/s-1)/2 px off the true
// centre, injecting an (I-R)*offset translation at full resolution -- ~0.4 px
// for a 5-deg roll at the device's lk scale (s ~ 1/16), which is the same
// order as the rigid refine's whole accuracy budget. (The RIGID path is exempt:
// rotation about the level centre plus t/s is exactly the true lift, because
// the level centre lands exactly on the source centre under this map.)
inline void level_map(f32 s, f32 M[9], f32 Minv[9]) {
    const f32 d = 0.5f * (1.f - s);
    M[0] = s;   M[1] = 0.f; M[2] = -d;
    M[3] = 0.f; M[4] = s;   M[5] = -d;
    M[6] = 0.f; M[7] = 0.f; M[8] = 1.f;
    const f32 is = 1.f / s;
    Minv[0] = is;  Minv[1] = 0.f; Minv[2] = d * is;
    Minv[3] = 0.f; Minv[4] = is;  Minv[5] = d * is;
    Minv[6] = 0.f; Minv[7] = 0.f; Minv[8] = 1.f;
}

inline void mat3_mul(const f32 A[9], const f32 B[9], f32 C[9]) {
    f32 t[9];
    for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col)
            t[r * 3 + col] = A[r * 3 + 0] * B[0 * 3 + col] +
                             A[r * 3 + 1] * B[1 * 3 + col] +
                             A[r * 3 + 2] * B[2 * 3 + col];
    for (int i = 0; i < 9; ++i) C[i] = t[i];
}

// Alignment error of comp warped by H against ref: mean absolute residual
// after removing a global photometric gain/bias (comp ~= a*ref + b, least
// squares) so exposure/gain drift between burst frames does not read as
// misalignment. Candidates are scored on a COMMON central region (each side
// inset by 1/6) so that, for the motion range the pre-align searches, every
// candidate evaluates (and fits its photometric model on) essentially the
// same pixel population -- a candidate can no longer look better by cropping
// the difficult periphery out of its own score. Residual out-of-bounds pixels
// (rare within the inset) are still skipped and charged for: the mean is
// divided by the valid fraction, and under 50% validity the candidate is not
// comparable at all and scores 1e30. Bounds tests use the require-form
// (!(x >= lo && x <= hi)) so a non-finite coordinate from a diverged iterate
// is REJECTED rather than reaching floor()/int conversion in the sampler.
f32 warp_error(const Image& ref, const Image& comp, const f32 H[9]) {
    const int y0 = ref.h / 6, y1 = ref.h - ref.h / 6;
    const int x0 = ref.w / 6, x1 = ref.w - ref.w / 6;
    // Pass 1: valid count + sums for the photometric fit.
    double sr = 0.0, sc = 0.0, srr = 0.0, src = 0.0;
    long cnt = 0, tot = 0;
    for (int y = y0; y < y1; y += 2) {
        for (int x = x0; x < x1; x += 2) {
            ++tot;
            f32 ox, oy;
            apply_homography(H, (f32)x, (f32)y, ox, oy);
            if (!(ox >= 0.f && ox <= (f32)(comp.w - 1) &&
                  oy >= 0.f && oy <= (f32)(comp.h - 1)))
                continue;
            const double cv = sample_clamp(comp, ox, oy), rv = ref.at(y, x);
            sr += rv; sc += cv; srr += rv * rv; src += rv * cv;
            ++cnt;
        }
    }
    if (tot <= 0 || cnt * 2 < tot) return 1e30f;  // <50% valid
    const double n = (double)cnt;
    const double var_r = srr / n - (sr / n) * (sr / n);
    f32 pa = 1.f, pb = 0.f;
    if (var_r > 1e-12) {
        const f32 a = (f32)((src / n - (sr / n) * (sc / n)) / var_r);
        pa = std::min(std::max(a, 0.5f), 2.f);  // a real gain change is near 1
        pb = (f32)(sc / n - (double)pa * (sr / n));
    }
    // Pass 2: mean |comp(Hp) - (a*ref(p) + b)| over the same valid set.
    double acc = 0.0;
    for (int y = y0; y < y1; y += 2) {
        for (int x = x0; x < x1; x += 2) {
            f32 ox, oy;
            apply_homography(H, (f32)x, (f32)y, ox, oy);
            if (!(ox >= 0.f && ox <= (f32)(comp.w - 1) &&
                  oy >= 0.f && oy <= (f32)(comp.h - 1)))
                continue;
            acc += std::fabs(sample_clamp(comp, ox, oy) - (pa * ref.at(y, x) + pb));
        }
    }
    const f32 frac = (f32)((double)cnt / (double)tot);
    return (f32)(acc / n) / frac;
}

// Robust noise floor of a grey level, as a gradient-energy threshold. The
// second difference a - 2b + c of i.i.d. noise has variance 6*sigma^2 and the
// MEDIAN of its magnitude ignores the sparse structure/edges, so
// sigma = 1.4826 * median|d| / sqrt(6). Spacing 2 dodges the pixel-to-pixel
// correlation left by the box downsamples. A squared central-diff gradient
// (gx^2 + gy^2) of pure noise has expectation sigma^2, so g2 ~ sigma^2 means
// "this gradient is indistinguishable from sensor noise" -- the absolute,
// noise-model-grounded floor the purely relative 0.15*mean rule lacked.
f32 noise_floor_g2(const Image& im) {
    std::vector<f32> v;
    if (im.h < 2 || im.w < 8) return 0.f;
    v.reserve(((size_t)im.h / 3 + 1) * ((size_t)im.w / 3 + 1));
    for (int y = 0; y < im.h; y += 3)
        for (int x = 2; x + 2 < im.w; x += 3)
            v.push_back(std::fabs(im.at(y, x - 2) - 2.f * im.at(y, x) + im.at(y, x + 2)));
    if (v.size() < 64) return 0.f;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    const f32 sigma = 1.4826f * v[v.size() / 2] / 2.4494897f;  // sqrt(6)
    return sigma * sigma;
}

// Build a rigid (rotation about centre + translation) homography in `g`'s
// coordinates. Shared by the rigid refine and its accept guard.
static inline void rigid_to_H(const Image& g, f32 dx, f32 dy, f32 rot, f32 H[9]) {
    const f32 cx = 0.5f * (f32)(g.w - 1), cy = 0.5f * (f32)(g.h - 1);
    const f32 cs = std::cos(rot), sn = std::sin(rot);
    H[0] = cs; H[1] = -sn; H[2] = cx + dx - cs * cx + sn * cy;
    H[3] = sn; H[4] =  cs; H[5] = cy + dy - sn * cx - cs * cy;
    H[6] = 0.f; H[7] = 0.f; H[8] = 1.f;
}

// One level of the 3-DOF rigid (translation + rotation about the level centre)
// Gauss-Newton Lucas-Kanade refine. ltx/lty in LEVEL pixels, th in radians,
// refined in place. A lag-1 photometric gain/bias fit (comp ~= a*ref + b,
// updated after each iteration) keeps exposure/gain drift between frames out
// of the geometry; it does not touch the Jacobian (the residual's derivative
// w.r.t. the warp is unchanged). Shared by refine_global_rigid_seed (per
// pyramid level) and estimate_global_homography's model ladder (rigid stage).
void rigid_lk_level(const Image& R, const Image& C, f32& ltx, f32& lty, f32& th,
                    int iters) {
    if (R.h <= 8 || R.w <= 8) return;
    const f32 cx = 0.5f * (f32)(R.w - 1), cy = 0.5f * (f32)(R.h - 1);
    const int y0 = R.h / 8, y1 = R.h - R.h / 8, x0 = R.w / 8, x1 = R.w - R.w / 8;
    f32 pa = 1.f, pb = 0.f;  // lag-1 photometric gain/bias (for the GN residual)
    // Metric-only scoring pass (no normal equations): the profiled photometric
    // score of a candidate (tx, ty, theta) over the same region and validity
    // policy as the main loop. Used to validate the FINAL applied step, which
    // the in-loop lag check never sees.
    auto metric_at = [&](f32 ttx, f32 tty, f32 tth) -> f32 {
        double spr = 0.0, spc = 0.0, sprr = 0.0, sprc = 0.0, spcc = 0.0; long pcnt = 0;
        const f32 cs = std::cos(tth), sn = std::sin(tth);
        for (int y = y0; y < y1; y += 2)
            for (int x = x0; x < x1; x += 2) {
                const f32 X = (f32)x - cx, Y = (f32)y - cy;
                const f32 wx = cs * X - sn * Y + cx + ttx;
                const f32 wy = sn * X + cs * Y + cy + tty;
                if (!(wx >= 1.f && wx < R.w - 1 && wy >= 1.f && wy < R.h - 1)) continue;
                const f32 cv = sample_clamp(C, wx, wy), rv = R.at(y, x);
                if (!(std::isfinite(cv) && std::isfinite(rv))) continue;
                spr += rv; spc += cv; sprr += (double)rv * rv; sprc += (double)rv * cv;
                spcc += (double)cv * cv; ++pcnt;
            }
        if (pcnt < 16) return 1e30f;
        const double n = (double)pcnt;
        const double var_r = sprr / n - (spr / n) * (spr / n);
        const double var_c = spcc / n - (spc / n) * (spc / n);
        const double cov   = sprc / n - (spr / n) * (spc / n);
        const f32 m = (f32)(var_c - (var_r > 1e-12 ? cov * cov / var_r : 0.0));
        return std::isfinite(m) ? m : 1e30f;
    };
    // Step acceptance: the sums accumulated in iteration k measure the
    // parameters AFTER step k-1, at zero extra cost. The acceptance metric is
    // the residual VARIANCE WITH GAIN/BIAS PROFILED OUT in closed form,
    // var(comp) - cov(ref,comp)^2 / var(ref): a pure function of the GEOMETRY
    // (the photometric nuisance is optimally removed for each iteration's own
    // warp), so successive iterations compare like-for-like -- comparing raw
    // residuals under the drifting lag-1 (pa, pb) would conflate a photometric
    // re-fit with a geometric regression. If the metric worsened, the last
    // step is reverted and the refine stops. `pending` tracks whether the
    // current parameters contain a step that has not been scored yet; EVERY
    // exit path resolves it (in-loop on the next iteration, otherwise by the
    // metric_at tail below), so no step is ever committed unevaluated.
    f32 best_metric = 1e30f, last_dtx = 0.f, last_dty = 0.f, last_dth = 0.f;
    bool pending = false;
    for (int it = 0; it < iters; ++it) {
        double Hm[3][3] = {{0,0,0},{0,0,0},{0,0,0}}, b[3] = {0,0,0};
        double spr = 0.0, spc = 0.0, sprr = 0.0, sprc = 0.0, spcc = 0.0; long pcnt = 0;
        const f32 cs = std::cos(th), sn = std::sin(th);
        for (int y = y0; y < y1; y += 2)
            for (int x = x0; x < x1; x += 2) {
                const f32 X = (f32)x - cx, Y = (f32)y - cy;
                const f32 wx = cs * X - sn * Y + cx + ltx;
                const f32 wy = sn * X + cs * Y + cy + lty;
                if (!(wx >= 1.f && wx < R.w - 1 && wy >= 1.f && wy < R.h - 1)) continue;
                const f32 cv = sample_clamp(C, wx, wy), rv = R.at(y, x);
                if (!(std::isfinite(cv) && std::isfinite(rv))) continue;  // bad data
                const f32 gx = 0.5f * (sample_clamp(C, wx + 1, wy) - sample_clamp(C, wx - 1, wy));
                const f32 gy = 0.5f * (sample_clamp(C, wx, wy + 1) - sample_clamp(C, wx, wy - 1));
                const f32 dwx = -sn * X - cs * Y, dwy = cs * X - sn * Y;  // d(wx,wy)/dtheta
                const f32 J[3] = { gx, gy, gx * dwx + gy * dwy };
                const f32 r = cv - (pa * rv + pb);
                spr += rv; spc += cv; sprr += (double)rv * rv; sprc += (double)rv * cv;
                spcc += (double)cv * cv; ++pcnt;
                for (int a = 0; a < 3; ++a) { b[a] -= J[a] * r;
                    for (int c2 = 0; c2 < 3; ++c2) Hm[a][c2] += J[a] * J[c2]; }
            }
        if (pcnt < 16) {  // warped off the frame: nothing to fit on
            if (pending) { ltx -= last_dtx; lty -= last_dty; th -= last_dth; }
            pending = false;
            break;
        }
        const double n = (double)pcnt;
        const double var_r = sprr / n - (spr / n) * (spr / n);
        const double var_c = spcc / n - (spc / n) * (spc / n);
        const double cov   = sprc / n - (spr / n) * (spc / n);
        const f32 metric = (f32)(var_c - (var_r > 1e-12 ? cov * cov / var_r : 0.0));
        if (!(metric <= best_metric)) {           // worse (or NaN): revert last step
            ltx -= last_dtx; lty -= last_dty; th -= last_dth;
            pending = false;
            break;
        }
        best_metric = metric;
        pending = false;
        // Photometric fit for the NEXT iteration (lag-1, like the IRLS scales).
        if (var_r > 1e-12) {
            pa = std::min(std::max((f32)(cov / var_r), 0.5f), 2.f);
            pb = (f32)(spc / n - (double)pa * (spr / n));
        }
        for (int a = 0; a < 3; ++a) Hm[a][a] += 1e-3 * Hm[a][a] + 1e-9;  // Levenberg
        double A[3][4];
        for (int a = 0; a < 3; ++a) { for (int c2 = 0; c2 < 3; ++c2) A[a][c2] = Hm[a][c2]; A[a][3] = b[a]; }
        bool ok = true;
        for (int c2 = 0; c2 < 3 && ok; ++c2) {
            int p = c2; for (int r2 = c2 + 1; r2 < 3; ++r2) if (std::fabs(A[r2][c2]) > std::fabs(A[p][c2])) p = r2;
            for (int k = 0; k < 4; ++k) std::swap(A[c2][k], A[p][k]);
            if (std::fabs(A[c2][c2]) < 1e-20) { ok = false; break; }
            const double iv = 1.0 / A[c2][c2];
            for (int k = 0; k < 4; ++k) A[c2][k] *= iv;
            for (int r2 = 0; r2 < 3; ++r2) if (r2 != c2) { const double f = A[r2][c2];
                for (int k = 0; k < 4; ++k) A[r2][k] -= f * A[c2][k]; }
        }
        if (!ok) break;
        const f32 d_tx = (f32)A[0][3], d_ty = (f32)A[1][3], d_th = (f32)A[2][3];
        if (!(std::isfinite(d_tx) && std::isfinite(d_ty) && std::isfinite(d_th))) break;
        ltx += d_tx; lty += d_ty; th += d_th;
        // Parameter sanity: a rigid pre-align never legitimately leaves this
        // range; stepping outside it means divergence, so undo and stop.
        if (!(std::fabs(ltx) < (f32)R.w && std::fabs(lty) < (f32)R.h &&
              std::fabs(th) < 0.7f)) {
            ltx -= d_tx; lty -= d_ty; th -= d_th;
            break;
        }
        last_dtx = d_tx; last_dty = d_ty; last_dth = d_th;
        pending = true;
        if (std::fabs(d_tx) + std::fabs(d_ty) < 1e-3f && std::fabs(d_th) < 1e-6f) break;
    }
    // Validate the FINAL applied step (convergence break or iteration budget
    // exhausted both leave `pending` true): score it with the same profiled
    // metric and revert if it did not improve on the best evaluated state.
    if (pending) {
        const f32 m = metric_at(ltx, lty, th);
        if (!(m <= best_metric)) { ltx -= last_dtx; lty -= last_dty; th -= last_dth; }
    }
}

}  // anonymous namespace -- reopened after the (externally-linked) rigid refine

// Externally visible (declared in global_homography.h); its helpers above keep
// internal linkage but stay visible to it within this translation unit.
void refine_global_rigid_seed(const Image& ref_grey, const Image& comp_grey,
                              const Config& cfg, f32& dx, f32& dy, f32& rot) {
    if (ref_grey.h <= 8 || ref_grey.w <= 8 ||
        ref_grey.h != comp_grey.h || ref_grey.w != comp_grey.w) return;
    if (ref_grey.data.size() != (size_t)ref_grey.h * ref_grey.w * ref_grey.c ||
        comp_grey.data.size() != (size_t)comp_grey.h * comp_grey.w * comp_grey.c) return;

    const int fine_dim = std::max(128, std::min(2048, cfg.isa_prealign_refine_dim));
    f32 scf = 1.f;
    const Image fine_r = downsample_to(ref_grey, fine_dim, scf);
    const Image fine_c = downsample_to(comp_grey, fine_dim, scf);
    if (fine_r.h <= 8 || fine_r.w <= 8) return;
    const Image mid_r = downsample2x(fine_r), mid_c = downsample2x(fine_c);
    const Image cor_r = downsample2x(mid_r), cor_c = downsample2x(mid_c);
    struct Lvl { const Image* r; const Image* c; f32 scale; };  // scale = level_px / full_px
    const Lvl levels[3] = { { &cor_r, &cor_c, scf * 0.25f },
                            { &mid_r, &mid_c, scf * 0.5f },
                            { &fine_r, &fine_c, scf } };

    // (tx, ty) carried in FULL-res pixels; rot in radians (scale-invariant).
    f32 tx = dx, ty = dy, th = rot;
    for (const Lvl& L : levels) {
        const Image& R = *L.r; const Image& C = *L.c;
        if (R.h <= 8 || R.w <= 8) continue;
        const f32 s = L.scale;                    // full -> level
        f32 ltx = tx * s, lty = ty * s;           // translation at this level
        rigid_lk_level(R, C, ltx, lty, th, /*iters=*/20);
        tx = ltx / s; ty = lty / s;               // back to full-res translation
    }

    // Accept only if the refine lowers the warp error vs the seed (finest level).
    if (!(std::isfinite(tx) && std::isfinite(ty) && std::isfinite(th))) return;
    f32 Href[9], Hseed[9];
    rigid_to_H(fine_r, tx * scf, ty * scf, th, Href);
    rigid_to_H(fine_r, dx * scf, dy * scf, rot, Hseed);
    const f32 e_ref = warp_error(fine_r, fine_c, Href);
    const f32 e_seed = warp_error(fine_r, fine_c, Hseed);
    // Sentinel-aware STRICT improvement: with the old `e_ref <= e_seed`, BOTH
    // scores being the 1e30 invalid sentinel satisfied the gate and accepted an
    // unvalidated refine. The refine must now itself be a valid score and
    // strictly better; a valid refine still rescues an invalid seed, since any
    // real score is strictly below the sentinel.
    if (!std::isfinite(e_ref) || e_ref >= 1e29f || !(e_ref < e_seed)) return;

    dx = tx; dy = ty; rot = th;
}

namespace {  // reopened: the remaining helpers below stay internal

// Coarse rotation (about the image centre) + integer shift search, minimising
// the warp error. Two stages: the 1-deg / 2-px grid keeps the best THREE
// candidates (under noise the quantised error surface can put the true optimum
// second or third), then a fine pass around each (0.25-deg steps over +/-0.75
// deg, 1-px shifts) picks the winner -- so the seed handed to Lucas-Kanade is
// within ~0.25 deg / 1 px instead of ~1 deg / 2 px. Builds an affine
// homography seed. Handles large roll, which LK alone cannot converge to from
// identity.
void coarse_seed(const Image& ref, const Image& comp, f32 H[9]) {
    const f32 cx = 0.5f * (f32)(ref.w - 1), cy = 0.5f * (f32)(ref.h - 1);
    const f32 range_deg = 12.f, step_deg = 1.f;
    const int shift_max = 6, shift_step = 2;
    auto make_H = [&](f32 deg, f32 sx, f32 sy, f32 Hc[9]) {
        const f32 th = deg * 3.14159265358979323846f / 180.f;
        const f32 ct = std::cos(th), st = std::sin(th);
        // rotation about centre: p -> R(p-c)+c
        Hc[0] = ct; Hc[1] = -st; Hc[2] = cx - (ct * cx - st * cy) + sx;
        Hc[3] = st; Hc[4] =  ct; Hc[5] = cy - (st * cx + ct * cy) + sy;
        Hc[6] = 0.f; Hc[7] = 0.f; Hc[8] = 1.f;
    };
    struct Cand { f32 err, deg, sx, sy; };
    Cand top[3] = { {1e30f,0,0,0}, {1e30f,0,0,0}, {1e30f,0,0,0} };
    auto consider = [&](f32 deg, f32 sx, f32 sy) {
        f32 Hc[9]; make_H(deg, sx, sy, Hc);
        const f32 e = warp_error(ref, comp, Hc);
        // Same-basin suppression: a candidate within 1.5 deg / 2.5 px of an
        // already-kept one refines THAT slot instead of occupying another, so
        // the three slots hold genuinely different motion hypotheses (adjacent
        // grid cells of one minimum would otherwise fill all three).
        for (int i = 0; i < 3; ++i) {
            if (top[i].err < 1e29f &&
                std::fabs(deg - top[i].deg) <= 1.5f &&
                std::fabs(sx - top[i].sx) <= 2.5f && std::fabs(sy - top[i].sy) <= 2.5f) {
                if (e < top[i].err) {
                    top[i] = {e, deg, sx, sy};
                    while (i > 0 && top[i].err < top[i - 1].err) { std::swap(top[i], top[i - 1]); --i; }
                }
                return;
            }
        }
        if (e < top[0].err)      { top[2] = top[1]; top[1] = top[0]; top[0] = {e, deg, sx, sy}; }
        else if (e < top[1].err) { top[2] = top[1]; top[1] = {e, deg, sx, sy}; }
        else if (e < top[2].err) { top[2] = {e, deg, sx, sy}; }
    };
    for (f32 deg = -range_deg; deg <= range_deg + 1e-3f; deg += step_deg)
        for (int sy = -shift_max; sy <= shift_max; sy += shift_step)
            for (int sx = -shift_max; sx <= shift_max; sx += shift_step)
                consider(deg, (f32)sx, (f32)sy);
    // Fine pass around each kept candidate; best overall wins.
    Cand best = top[0];
    for (int c = 0; c < 3; ++c) {
        if (!(top[c].err < 1e29f)) continue;
        for (f32 dd = -0.75f; dd <= 0.75f + 1e-3f; dd += 0.25f)
            for (int oy = -1; oy <= 1; ++oy)
                for (int ox = -1; ox <= 1; ++ox) {
                    const f32 deg = top[c].deg + dd;
                    const f32 sx = top[c].sx + (f32)ox, sy = top[c].sy + (f32)oy;
                    f32 Hc[9]; make_H(deg, sx, sy, Hc);
                    const f32 e = warp_error(ref, comp, Hc);
                    if (e < best.err) best = {e, deg, sx, sy};
                }
    }
    make_H(best.deg, best.sx, best.sy, H);
}

// Solve an 8x8 linear system A x = b in place (Gaussian elimination, partial
// pivot). Returns false if singular.
bool solve8(f32 A[8][8], f32 b[8], f32 x[8]) {
    const int n = 8;
    for (int col = 0; col < n; ++col) {
        int piv = col; f32 best = std::fabs(A[col][col]);
        for (int r = col + 1; r < n; ++r) {
            f32 v = std::fabs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-12f) return false;
        if (piv != col) {
            for (int k = 0; k < n; ++k) std::swap(A[col][k], A[piv][k]);
            std::swap(b[col], b[piv]);
        }
        const f32 inv = 1.f / A[col][col];
        for (int r = col + 1; r < n; ++r) {
            f32 f = A[r][col] * inv;
            if (f == 0.f) continue;
            for (int k = col; k < n; ++k) A[r][k] -= f * A[col][k];
            b[r] -= f * b[col];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        f32 s = b[r];
        for (int k = r + 1; k < n; ++k) s -= A[r][k] * x[k];
        x[r] = s / A[r][r];
    }
    return true;
}

// Forward-additive Lucas-Kanade homography refinement: minimise
// sum [comp(H*p) - ref(p)]^2 over the 8 homography parameters (h8 fixed = 1).
// Levenberg damping; reverts an iteration that increases the error.
void lk_refine(const Image& ref, const Image& comp, f32 H[9], int iters,
               int num_threads) {
    // Normalize to the H[8] = 1 parameterization this optimizer assumes. A
    // level_map conjugation leaves H[8] = 1 + (h6+h7)*d/s != 1 exactly, and
    // the additive update forces Hn[8] = 1 WITHOUT rescaling the other eight
    // entries -- an uncompensated jump in the first step. Rescaling the whole
    // matrix first is the same projective transform in the right chart.
    if (std::isfinite(H[8]) && std::fabs(H[8]) > 1e-3f && H[8] != 1.f) {
        const f32 inv8 = 1.f / H[8];
        for (int i = 0; i < 9; ++i) H[i] *= inv8;
    }
    f32 cur_err = warp_error(ref, comp, H);
    const int y_lo = 1, y_hi = ref.h - 1;
    const int rows = std::max(0, y_hi - y_lo);
    int nt = std::max(1, std::min(resolve_threads(num_threads), std::max(1, rows)));
    const int chunk = (rows + nt - 1) / std::max(1, nt);
    // Cauchy IRLS: weight w = 1 / (1 + (r/c)^2) down-weights large residuals, so
    // moving objects and near-depth (parallax) regions stop pulling the GLOBAL
    // fit away from the dominant static scene. The scale c is set from the
    // MEDIAN |residual| of the previous iteration (lag-1, via a histogram; the
    // mean it used before is itself inflated by the very outliers the Cauchy is
    // meant to ignore, so a moving object loosened its own down-weighting);
    // cinv2 = 1/c^2, starting 0 so the first iteration is unweighted.
    //
    // Gradient (signal) weight w_sig = g2 / (g2 + g2floor) additionally
    // down-weights low-gradient, noise-dominated flats -- the key low-light
    // accuracy fix: in dim scenes the flats are mostly sensor noise, and letting
    // them into the normal equations makes the rotation/perspective params noisy.
    // g2floor combines the relative rule (a fraction of the mean gradient energy
    // of the previous iteration) with an ABSOLUTE noise floor measured from the
    // image (noise_floor_g2): in a dim scene the mean gradient energy is itself
    // mostly noise, so the relative rule alone stops discriminating exactly
    // where it matters. 4x the floor weights a pure-noise gradient at <= 0.2.
    //
    // Photometric gain/bias (comp ~= pa*ref + pb, lag-1 least squares) keeps
    // exposure/gain drift between frames from reading as geometry.
    const f32 g2noise = noise_floor_g2(comp);
    f32 cinv2 = 0.f, g2floor = 4.f * g2noise;
    f32 pa = 1.f, pb = 0.f;
    f32 hist_max = 0.f;                   // |r| histogram range (lag-1 from mean)
    constexpr int kHistBins = 64;
    for (int it = 0; it < iters; ++it) {
        // Per-chunk partial normal equations (double for stable accumulation),
        // combined serially after. The pixel loop is the whole cost of LK, so
        // chunking it across threads is what keeps the homography refine fast.
        std::vector<std::array<double, 64>> Ap((size_t)nt);
        std::vector<std::array<double, 8>> bp((size_t)nt);
        std::vector<double> srp((size_t)nt, 0.0), sg2p((size_t)nt, 0.0);
        std::vector<long> cntp((size_t)nt, 0);
        // Per-chunk |residual| histograms (for the lag-1 MEDIAN scale) and
        // photometric-fit sums (ref, comp, ref^2, ref*comp).
        std::vector<std::array<long, kHistBins>> histp((size_t)nt);
        std::vector<std::array<double, 4>> php((size_t)nt);
        for (int t = 0; t < nt; ++t) {
            Ap[(size_t)t].fill(0.0); bp[(size_t)t].fill(0.0);
            histp[(size_t)t].fill(0); php[(size_t)t].fill(0.0);
        }
        const f32 hist_scale = (hist_max > 0.f) ? (f32)kHistBins / hist_max : 0.f;
        parallel_rows(nt, num_threads, [&](int t) {
            const int ys = y_lo + t * chunk;
            const int ye = std::min(ys + chunk, y_hi);
            double* Aa = Ap[(size_t)t].data();
            double* ba = bp[(size_t)t].data();
            long* hist = histp[(size_t)t].data();
            double* ph = php[(size_t)t].data();
            double sr = 0.0, sg2 = 0.0; long cn = 0;
            for (int y = ys; y < ye; ++y) {
                for (int x = 1; x < ref.w - 1; ++x) {
                    const f32 fx = (f32)x, fy = (f32)y;
                    const f32 D = H[6] * fx + H[7] * fy + H[8];
                    // Require-form tests reject NaN/inf (comparisons with NaN
                    // are false), so a diverged iterate cannot reach the
                    // sampler's floor()/int conversion.
                    if (!(std::fabs(D) > 1e-6f)) continue;
                    const f32 iD = 1.f / D;
                    const f32 u = (H[0] * fx + H[1] * fy + H[2]) * iD;
                    const f32 v = (H[3] * fx + H[4] * fy + H[5]) * iD;
                    if (!(u >= 1.f && u <= (f32)(comp.w - 2) &&
                          v >= 1.f && v <= (f32)(comp.h - 2)))
                        continue;
                    const f32 cv = sample_clamp(comp, u, v), rv = ref.at(y, x);
                    const f32 err = cv - (pa * rv + pb);
                    // Bad image data must not reach the histogram index
                    // conversion or poison the normal equations.
                    if (!std::isfinite(err)) continue;
                    const f32 gx = 0.5f * (sample_clamp(comp, u + 1.f, v) - sample_clamp(comp, u - 1.f, v));
                    const f32 gy = 0.5f * (sample_clamp(comp, u, v + 1.f) - sample_clamp(comp, u, v - 1.f));
                    const f32 g2 = gx * gx + gy * gy;
                    // The gradient samples different pixels than err, so the
                    // err check alone does not keep NaN out of the Jacobian.
                    if (!(g2 >= 0.f && g2 < 1e30f)) continue;
                    const f32 wsig = (g2floor > 0.f) ? g2 / (g2 + g2floor) : 1.f;
                    const f32 w = wsig / (1.f + err * err * cinv2);  // signal x Cauchy
                    f32 J[8];
                    J[0] = gx * (fx * iD);           J[1] = gx * (fy * iD);           J[2] = gx * iD;
                    J[3] = gy * (fx * iD);           J[4] = gy * (fy * iD);           J[5] = gy * iD;
                    J[6] = -(gx * u + gy * v) * (fx * iD);
                    J[7] = -(gx * u + gy * v) * (fy * iD);
                    const f32 we = w * err;
                    for (int i = 0; i < 8; ++i) {
                        ba[i] -= (double)(J[i] * we);
                        const f32 wJi = w * J[i];
                        for (int j = 0; j < 8; ++j) Aa[i * 8 + j] += (double)(wJi * J[j]);
                    }
                    const f32 ae = std::fabs(err);
                    sr += (double)ae; sg2 += (double)g2; ++cn;
                    if (hist_scale > 0.f) {
                        // Clamp in FLOAT before the int conversion: casting a
                        // float beyond int range is UB, so the old post-cast
                        // clamp did not protect against a huge residual.
                        const f32 scaled = ae * hist_scale;
                        if (!(scaled < (f32)(kHistBins - 1))) ++hist[kHistBins - 1];
                        else if (scaled > 0.f)                ++hist[(int)scaled];
                        else                                  ++hist[0];
                    }
                    ph[0] += (double)rv; ph[1] += (double)cv;
                    ph[2] += (double)rv * rv; ph[3] += (double)rv * cv;
                }
            }
            srp[(size_t)t] = sr; sg2p[(size_t)t] = sg2; cntp[(size_t)t] = cn;
        });
        f32 A[8][8]; f32 b[8];
        for (int i = 0; i < 8; ++i) {
            double bi = 0.0; for (int t = 0; t < nt; ++t) bi += bp[(size_t)t][i];
            b[i] = (f32)bi;
            for (int j = 0; j < 8; ++j) {
                double a = 0.0; for (int t = 0; t < nt; ++t) a += Ap[(size_t)t][i * 8 + j];
                A[i][j] = (f32)a;
            }
        }
        // Robust residual scale, gradient floor and photometric fit for the
        // NEXT iteration (all lag-1).
        double sum_abs = 0.0, sum_g2 = 0.0; long cnt = 0;
        double phs[4] = {0, 0, 0, 0};
        for (int t = 0; t < nt; ++t) {
            sum_abs += srp[(size_t)t]; sum_g2 += sg2p[(size_t)t]; cnt += cntp[(size_t)t];
            for (int k = 0; k < 4; ++k) phs[k] += php[(size_t)t][k];
        }
        if (cnt > 0) {
            const f32 mean_abs = (f32)(sum_abs / (double)cnt);
            // Median |r| from the merged histogram (range set by the PREVIOUS
            // iteration's mean). For a Gaussian core, median = 0.674*sigma, so
            // c = 3*median ~ 2*sigma -- the same effective scale the old
            // 2.5*mean rule gave on clean data, but immune to outlier mass.
            f32 c = 2.5f * mean_abs;              // fallback: no histogram yet
            if (hist_max > 0.f) {
                long cum = 0; int bi = 0;
                std::array<long, kHistBins> hist; hist.fill(0);
                for (int t = 0; t < nt; ++t)
                    for (int k = 0; k < kHistBins; ++k) hist[(size_t)k] += histp[(size_t)t][(size_t)k];
                for (; bi < kHistBins; ++bi) { cum += hist[(size_t)bi]; if (2 * cum >= cnt) break; }
                const f32 med = ((f32)bi + 0.5f) * hist_max / (f32)kHistBins;
                if (med > 0.f) c = 3.0f * med;
            }
            cinv2 = 1.f / (c * c + 1e-12f);
            hist_max = 8.f * mean_abs;            // histogram range for next iter
            g2floor = std::max(0.15f * (f32)(sum_g2 / (double)cnt), 4.f * g2noise);
            // Photometric gain/bias for the next iteration's residuals. The
            // Jacobian is untouched (d err / d warp params is unchanged).
            const double n = (double)cnt;
            const double var_r = phs[2] / n - (phs[0] / n) * (phs[0] / n);
            if (var_r > 1e-12) {
                const f32 a = (f32)((phs[3] / n - (phs[0] / n) * (phs[1] / n)) / var_r);
                pa = std::min(std::max(a, 0.5f), 2.f);
                pb = (f32)(phs[1] / n - (double)pa * (phs[0] / n));
            }
        }
        // Diagonal (Jacobi) preconditioning. The homography Jacobian mixes terms
        // in 1, x and x^2 (x up to the image size), so A is wildly ill-conditioned
        // and gets worse with resolution -- which made the refine take vanishing
        // steps and converge WORSE as the refine resolution rose (so higher
        // precision did nothing or hurt). Scale each parameter by 1/sqrt(diag) so
        // the system is well-conditioned at any resolution, damp the unit-diagonal
        // system, solve, then unscale the step.
        f32 s[8];
        for (int i = 0; i < 8; ++i) s[i] = std::sqrt(std::max(A[i][i], 1e-20f));
        f32 An[8][8]; f32 bn[8];
        for (int i = 0; i < 8; ++i) {
            bn[i] = b[i] / s[i];
            for (int j = 0; j < 8; ++j) An[i][j] = A[i][j] / (s[i] * s[j]);
        }
        for (int i = 0; i < 8; ++i) An[i][i] += 1e-3f;  // Levenberg on the scaled system
        f32 y[8];
        if (!solve8(An, bn, y)) break;
        f32 d[8];
        for (int i = 0; i < 8; ++i) d[i] = y[i] / s[i];
        f32 Hn[9];
        for (int i = 0; i < 8; ++i) Hn[i] = H[i] + d[i];
        Hn[8] = 1.f;
        f32 e = warp_error(ref, comp, Hn);
        if (!(e < cur_err)) break;     // no improvement (or NaN) -> stop
        for (int i = 0; i < 9; ++i) H[i] = Hn[i];
        cur_err = e;
    }
}

} // namespace

Image warp_grey_by_homography(const Image& comp_grey, const f32 H[9]) {
    Image out(comp_grey.h, comp_grey.w, 1);
    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            f32 ox, oy;
            apply_homography(H, (f32)x, (f32)y, ox, oy);
            // Require-form bounds reject NaN/inf coordinates too (-> 0 fill).
            if (!(ox >= 0.f && ox <= (f32)(comp_grey.w - 1) &&
                  oy >= 0.f && oy <= (f32)(comp_grey.h - 1))) {
                out.at(y, x) = 0.f;
            } else {
                out.at(y, x) = sample_clamp(comp_grey, ox, oy);
            }
        }
    }
    return out;
}

void refine_global_homography_seed(const Image& ref_grey, const Image& comp_grey,
                                   const Config& cfg, f32 H_inout[9]) {
    if (ref_grey.h <= 8 || ref_grey.w <= 8 ||
        ref_grey.h != comp_grey.h || ref_grey.w != comp_grey.w) return;
    if (ref_grey.data.size() != (size_t)ref_grey.h * ref_grey.w * ref_grey.c ||
        comp_grey.data.size() != (size_t)comp_grey.h * comp_grey.w * comp_grey.c) return;

    // Coarse-to-fine Lucas-Kanade. A single-level LK under-converges on larger
    // scale/shear/perspective; a 3-level pyramid (each level seeded by the
    // previous, H carried in FULL grey pixels) converges reliably. The finest
    // level is built ONCE from the full grey (one read per image); the coarser
    // levels are cheap 2x halvings of it -- no repeated full-grey reads.
    // lk_refine reverts non-improving iterations, so no level worsens the fit.
    // Reuse the already-computed full-res grey at the requested finest size.
    const int fine_dim = std::max(128, std::min(2048, cfg.isa_prealign_refine_dim));
    f32 scf = 1.f;
    const Image fine_r = downsample_to(ref_grey, fine_dim, scf);
    const Image fine_c = downsample_to(comp_grey, fine_dim, scf);
    if (fine_r.h <= 8 || fine_r.w <= 8) return;
    const Image mid_r = downsample2x(fine_r), mid_c = downsample2x(fine_c);
    const Image cor_r = downsample2x(mid_r), cor_c = downsample2x(mid_c);
    struct Lvl { const Image* r; const Image* c; f32 scale; };
    const Lvl levels[3] = { { &cor_r, &cor_c, scf * 0.25f },
                            { &mid_r, &mid_c, scf * 0.5f },
                            { &fine_r, &fine_c, scf } };

    f32 H[9]; for (int i = 0; i < 9; ++i) H[i] = H_inout[i];
    for (const Lvl& L : levels) {
        if (L.r->h <= 8 || L.r->w <= 8) continue;
        f32 S[9], Sinv[9]; level_map(L.scale, S, Sinv);  // exact pixel-centre map
        f32 Hl[9]; { f32 t[9]; mat3_mul(H, Sinv, t); mat3_mul(S, t, Hl); }
        lk_refine(*L.r, *L.c, Hl, /*iters=*/15, cfg.num_threads);
        f32 t[9]; mat3_mul(Hl, S, t); mat3_mul(Sinv, t, H);
    }

    // Accept only if the refined homography beats the seed at the finest level
    // and is non-degenerate; otherwise leave H_inout as the (robust) rigid seed.
    f32 Sf[9], Sfi[9]; level_map(scf, Sf, Sfi);
    f32 Href[9], Hseed[9];
    { f32 t[9]; mat3_mul(H, Sfi, t); mat3_mul(Sf, t, Href); }
    { f32 t[9]; mat3_mul(H_inout, Sfi, t); mat3_mul(Sf, t, Hseed); }
    const f32 e_ref = warp_error(fine_r, fine_c, Href);
    const f32 e_seed = warp_error(fine_r, fine_c, Hseed);
    const f32 det2 = H[0] * H[4] - H[1] * H[3];
    bool h_finite = true;
    for (int i = 0; i < 9; ++i) h_finite = h_finite && std::isfinite(H[i]);
    const bool degenerate = !h_finite ||
                            !(std::fabs(det2) > 0.25f && std::fabs(det2) < 4.f) ||
                            !std::isfinite(e_ref) || !std::isfinite(e_seed);
    // Require a MARGIN, not just any improvement. On a near-still burst the true
    // motion is dominated by sensor noise, and the 8-DOF refine will fit that
    // noise into spurious shear/perspective that lowers the average grey error
    // by a sliver while swinging the FRAME CORNERS by ~1px -- it warps an
    // already-aligned frame into misalignment, worst at the periphery (measured:
    // a 0.3px-motion input went from 0.36px corner error to 0.90px). The rigid
    // seed is the safe fallback; only take the homography when it earns it.
    if (degenerate || !(e_ref < 0.98f * e_seed)) return;

    // Second gate: even with a real average-error win, reject a refine whose
    // corner predictions DISAGREE with the seed's by more than the seed's own
    // motion supports -- the signature of noise-driven perspective. Evaluated
    // with H and H_inout directly, which are BOTH in full grey pixels, at the
    // full grey corners (the previous version applied the fine-level-scaled
    // Href/Hseed to full-res corners, a mixed-coordinate comparison whose
    // numbers were meaningless), and the guarded quantity is max||H*p -
    // Hseed*p||, not each model's distance from identity (two models moving a
    // corner equally far in OPPOSITE directions must not pass).
    {
        const f32 cxs[4] = {0.f, (f32)(ref_grey.w - 1), 0.f, (f32)(ref_grey.w - 1)};
        const f32 cys[4] = {0.f, 0.f, (f32)(ref_grey.h - 1), (f32)(ref_grey.h - 1)};
        f32 extra = 0.f, seed_move = 0.f;
        for (int i = 0; i < 4; ++i) {
            // Projective validity at the corners (linear in x,y, so corner-
            // positivity covers the whole frame).
            const f32 den = H[6] * cxs[i] + H[7] * cys[i] + H[8];
            if (!(den > 0.05f && den < 20.f)) return;  // keep the rigid seed
            f32 rx, ry, sx, sy;
            apply_homography(H,       cxs[i], cys[i], rx, ry);
            apply_homography(H_inout, cxs[i], cys[i], sx, sy);
            if (!(std::isfinite(rx) && std::isfinite(ry))) return;
            extra     = std::max(extra,     std::hypot(rx - sx, ry - sy));
            seed_move = std::max(seed_move, std::hypot(sx - cxs[i], sy - cys[i]));
        }
        // When the seed is ~identity (still burst), allow at most a small
        // absolute disagreement; otherwise scale with the seed's own motion,
        // hard-capped so large seed motion does not authorise unboundedly
        // large projective deviation. Both bounds are RESOLUTION-RELATIVE
        // (0.25% / 1% of the long side, ~10 / ~40 px on a 4000 px grey),
        // matching the live path's 1 / 4 lk-px policy.
        const f32 longside = (f32)std::max(ref_grey.w, ref_grey.h);
        const f32 allowed = std::min(0.01f * longside,
                                     std::max(0.0025f * longside, 0.5f * seed_move));
        if (extra > allowed) return;  // keep the rigid seed
    }

    for (int i = 0; i < 9; ++i) H_inout[i] = H[i];
}

void estimate_global_homography(const Image& ref_grey, const Image& comp_grey,
                                const Config& cfg, f32 H_out[9]) {
    (void)cfg;
    const f32 identity[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i = 0; i < 9; ++i) H_out[i] = identity[i];
    if (ref_grey.h <= 0 || ref_grey.w <= 0 ||
        ref_grey.h != comp_grey.h || ref_grey.w != comp_grey.w) return;
    // Require host pixels: on the device the grey can be GPU-resident with an
    // empty host buffer (header only). Reading it would segfault; the caller
    // forces host materialisation, but guard here regardless.
    if (ref_grey.data.size() != (size_t)ref_grey.h * ref_grey.w * ref_grey.c ||
        comp_grey.data.size() != (size_t)comp_grey.h * comp_grey.w * comp_grey.c) return;

    // Estimate at a reduced resolution (warp-then-refine tolerates an
    // imperfect H -- the per-tile align cleans up the residual), then scale the
    // result back to the full grey resolution.
    f32 lk_scale = 1.f, search_scale = 1.f;
    Image lk_ref = downsample_to(ref_grey, 384, lk_scale);
    Image lk_comp = downsample_to(comp_grey, 384, lk_scale);
    Image s_ref = downsample_to(lk_ref, 128, search_scale);   // relative to lk_*
    Image s_comp = downsample_to(lk_comp, 128, search_scale);

    // Coarse rotation+shift seed at the smallest level, in lk-level pixels.
    f32 Hc[9];
    coarse_seed(s_ref, s_comp, Hc);
    // Lift the seed from search-level pixels to lk-level pixels with the exact
    // pixel-centre map (see level_map): D maps lk->search.
    {
        f32 D[9], Dinv[9]; level_map(search_scale, D, Dinv);
        f32 tmp[9]; mat3_mul(Hc, D, tmp); mat3_mul(Dinv, tmp, Hc);
    }

    // MODEL LADDER, low-DOF first: identity -> rigid (3-DOF LK polish of the
    // coarse seed) -> full 8-DOF homography, each stage kept only on merit.
    // An 8-DOF fit on a near-still burst happily turns sensor noise into
    // shear/perspective that lowers the mean error by a sliver while swinging
    // the frame corners (measured: a 0.3px-motion input went 0.36px -> 0.90px
    // corner error) -- so the homography must now BEAT the best low-DOF model,
    // not merely beat identity, and additionally pass a corner-motion guard.
    const f32 e_id = warp_error(lk_ref, lk_comp, identity);
    // Noise-normalised acceptance floor. warp_error samples comp BILINEARLY,
    // and on noise-dominated content ANY fractional warp smooths the comp
    // noise and lowers the mean |residual| by up to ~13% of its noise part
    // without being better aligned -- identity (integer sampling) gets no such
    // discount, so on a dim still burst every candidate gets a free win
    // (measured here: a 2.1px spurious corner swing). A model is only accepted
    // if it beats identity by a margin tied to the MEASURED noise floor: the
    // expected mean |residual| of pure aligned noise is sqrt(2/pi) * sqrt(
    // var_ref + var_comp), and 20% of it safely covers the interpolation
    // discount while staying far below any genuine alignment win.
    const f32 noise_mae =
        0.7979f * std::sqrt(noise_floor_g2(lk_ref) + noise_floor_g2(lk_comp));
    const f32 margin = 0.20f * noise_mae;
    const f32 e_c  = warp_error(lk_ref, lk_comp, Hc);
    f32 Hr[9]; f32 e_r = 1e30f;
    {
        // Recover (shift, rot) from the rigid coarse seed and polish with the
        // 3-DOF LK (it has no shear/perspective freedom, so it cannot fit
        // noise into a warp that injects misalignment).
        const f32 ccx = 0.5f * (f32)(lk_ref.w - 1), ccy = 0.5f * (f32)(lk_ref.h - 1);
        f32 th = std::atan2(Hc[3], Hc[0]);
        const f32 cs = std::cos(th), sn = std::sin(th);
        f32 ltx = Hc[2] - ccx + cs * ccx - sn * ccy;
        f32 lty = Hc[5] - ccy + sn * ccx + cs * ccy;
        rigid_lk_level(lk_ref, lk_comp, ltx, lty, th, /*iters=*/30);
        if (std::isfinite(ltx) && std::isfinite(lty) && std::isfinite(th)) {
            rigid_to_H(lk_ref, ltx, lty, th, Hr);
            e_r = warp_error(lk_ref, lk_comp, Hr);
        }
    }
    // Base = best supported low-DOF model: the winner among coarse/rigid, but
    // only if it beats identity by the noise margin; identity otherwise.
    f32 Hbase[9]; for (int i = 0; i < 9; ++i) Hbase[i] = identity[i];
    f32 e_base = e_id;
    // Sanitise BEFORE choosing the winner: comparing raw values would let a
    // NaN score select the invalid candidate (NaN comparisons are false).
    const f32 e_c_s = std::isfinite(e_c) ? e_c : 1e30f;
    const f32 e_r_s = std::isfinite(e_r) ? e_r : 1e30f;
    const f32 e_low = std::min(e_c_s, e_r_s);
    if (e_low < e_id - margin) {
        const f32* Hw = (e_r_s <= e_c_s) ? Hr : Hc;
        for (int i = 0; i < 9; ++i) Hbase[i] = Hw[i];
        e_base = e_low;
    }

    f32 H[9]; for (int i = 0; i < 9; ++i) H[i] = Hbase[i];
    lk_refine(lk_ref, lk_comp, H, /*iters=*/40, cfg.num_threads);

    // Take the homography only if it beats the base by a relative margin, beats
    // identity by the noise margin, and is non-degenerate -- warp-then-refine
    // must never make alignment worse.
    const f32 e_h  = warp_error(lk_ref, lk_comp, H);
    const f32 det2 = H[0] * H[4] - H[1] * H[3];
    bool h_finite = true;
    for (int i = 0; i < 9; ++i) h_finite = h_finite && std::isfinite(H[i]);
    bool take_h = h_finite &&
                  (std::fabs(det2) > 0.25f && std::fabs(det2) < 4.f) &&
                  std::isfinite(e_h) && (e_h < 0.98f * e_base) &&
                  (e_h < e_id - margin);
    if (take_h) {
        // Projective validity: the denominator h6*x + h7*y + 1 must be finite,
        // well away from zero and POSITIVE at all four corners. It is linear
        // in (x, y), so corner-positivity implies positivity over the whole
        // frame -- no pixel maps through (or near) the plane at infinity. The
        // 2x2 determinant alone cannot guarantee this.
        //
        // Corner-motion guard (previously only in the uncalled
        // refine_global_homography_seed; this is the live path). The guarded
        // quantity is the DISAGREEMENT between the homography and the base
        // model at the corners, max ||H*p - Hbase*p|| -- not each one's
        // distance from identity, which would pass two models that move a
        // corner the same distance in opposite directions. A genuine global
        // perspective only ADDS a modest corner excursion on top of the rigid
        // base; noise-perspective buys a sliver of error with a large
        // excursion. 1 lk-level px (level <= 384 wide) is ~10 full-res grey px.
        const f32 cxs[4] = {0.f, (f32)(lk_ref.w - 1), 0.f, (f32)(lk_ref.w - 1)};
        const f32 cys[4] = {0.f, 0.f, (f32)(lk_ref.h - 1), (f32)(lk_ref.h - 1)};
        f32 extra = 0.f, b_move = 0.f;
        for (int i = 0; i < 4; ++i) {
            const f32 den = H[6] * cxs[i] + H[7] * cys[i] + H[8];
            if (!(den > 0.05f && den < 20.f)) { take_h = false; break; }
            f32 hx, hy, bx, by;
            apply_homography(H,     cxs[i], cys[i], hx, hy);
            apply_homography(Hbase, cxs[i], cys[i], bx, by);
            if (!(std::isfinite(hx) && std::isfinite(hy))) { take_h = false; break; }
            extra  = std::max(extra,  std::hypot(hx - bx, hy - by));
            b_move = std::max(b_move, std::hypot(bx - cxs[i], by - cys[i]));
        }
        // Floor for still bursts, relative growth for moderate motion, and a
        // hard CAP so large rigid motion does not authorise unboundedly large
        // projective deviation (genuine handheld perspective adds a few lk px
        // at most on top of the rigid component).
        const f32 allowed = std::min(4.0f, std::max(1.0f, 0.5f * b_move));
        if (extra > allowed) take_h = false;     // fall back to the base model
    }
    const f32* Hsel = take_h ? H : Hbase;
    if (!take_h && !(e_base < e_id)) return;     // identity: H_out already set

    // Scale from lk-level pixels to full grey pixels with the exact
    // pixel-centre map: D maps full->lk.
    f32 D[9], Dinv[9]; level_map(lk_scale, D, Dinv);
    f32 tmp[9]; mat3_mul(Hsel, D, tmp); mat3_mul(Dinv, tmp, H_out);
}

f32 prealign_fit_ncc(const Image& ref_grey, const Image& warped_comp_grey,
                     const f32* H) {
    const Image& a = ref_grey;
    const Image& b = warped_comp_grey;
    if (a.h <= 0 || a.w <= 0 || b.h != a.h || b.w != a.w) return 1.f; // can't assess
    if (a.data.size() != (size_t)a.h * a.w * a.c ||
        b.data.size() != (size_t)b.h * b.w * b.c) return 1.f;
    // Pearson correlation over the inner 80% (the warp zeroes a border strip for
    // any real shift; sampling the centre avoids counting that border as
    // mismatch), subsampled for speed. With H (the warp that produced
    // warped_comp_grey) validity is EXPLICIT: a pixel counts iff H*p landed in
    // bounds. Without H the old value==0 heuristic remains, which cannot tell a
    // genuinely black pixel from border fill and so under-samples (or falsely
    // non-fits) dark scenes -- pass H whenever it is available.
    const int stride = 4;
    const int y0 = a.h / 10, y1 = a.h - a.h / 10;
    const int x0 = a.w / 10, x1 = a.w - a.w / 10;
    const f32 bw = (f32)(b.w - 1), bh = (f32)(b.h - 1);
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    long nn = 0, tot = 0;
    for (int y = y0; y < y1; y += stride)
        for (int x = x0; x < x1; x += stride) {
            ++tot;
            const float vb = b.at(y, x);
            if (H) {
                f32 ox, oy;
                apply_homography(H, (f32)x, (f32)y, ox, oy);
                if (!(ox >= 0.f && ox <= bw && oy >= 0.f && oy <= bh))
                    continue;                     // warp out-of-bounds: not data
            } else if (vb == 0.f) {
                continue;                         // heuristic: border fill is 0
            }
            const float va = a.at(y, x);
            sa += va; sb += vb; saa += (double)va * va; sbb += (double)vb * vb;
            sab += (double)va * vb; ++nn;
        }
    if (tot <= 0) return 1.f;
    // Barely any valid overlap means the frame hardly maps onto the reference --
    // treat that as a definite non-fit.
    if ((double)nn < 0.25 * (double)tot) return -1.f;
    const double n = (double)nn;
    const double cov = sab / n - (sa / n) * (sb / n);
    const double va2 = saa / n - (sa / n) * (sa / n);
    const double vb2 = sbb / n - (sb / n) * (sb / n);
    const double den = std::sqrt(va2 * vb2);
    if (!(den > 1e-12)) return 0.f;              // flat region: no evidence either way
    return (f32)(cov / den);
}

} // namespace hhsr
