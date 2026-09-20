#include "finish_hdr.h"

#include "dng_writer.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hhsr {
namespace {

constexpr f32 kEps = 1e-6f;
// Photographic middle grey: the pivot the base layer is compressed around.
// Content above it comes down, content below is lifted toward it.
constexpr f32 kMidGrey = 0.18f;

inline f32 luma_of(f32 r, f32 g, f32 b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// XYZ(D65) -> linear sRGB. dng_writer always emits CalibrationIlluminant1 =
// D65, and sRGB's white point is D65, so this is ColorMatrix1's exact companion
// with no chromatic adaptation needed in between. (Only ever applied to DNGs
// this app wrote; a foreign profile with a different illuminant would need the
// adaptation this deliberately omits.)
constexpr f32 kXyzToSrgb[9] = {
     3.2404542f, -1.5371385f, -0.4985314f,
    -0.9692660f,  1.8760108f,  0.0415560f,
     0.0556434f, -0.2040259f,  1.0572252f
};

// --------------------------------------------------------------- 3x3 helpers
bool invert_3x3(const f32* m, f32* inv) {
    const f32 a = m[0], b = m[1], c = m[2];
    const f32 d = m[3], e = m[4], f = m[5];
    const f32 g = m[6], h = m[7], i = m[8];
    const f32 A =  (e * i - f * h), B = -(d * i - f * g), C =  (d * h - e * g);
    const f32 det = a * A + b * B + c * C;
    if (!(std::fabs(det) > 1e-9f) || !std::isfinite(det)) return false;
    const f32 s = 1.f / det;
    inv[0] =  (e * i - f * h) * s; inv[1] = -(b * i - c * h) * s; inv[2] =  (b * f - c * e) * s;
    inv[3] = -(d * i - f * g) * s; inv[4] =  (a * i - c * g) * s; inv[5] = -(a * f - c * d) * s;
    inv[6] =  (d * h - e * g) * s; inv[7] = -(a * h - b * g) * s; inv[8] =  (a * e - b * d) * s;
    for (int k = 0; k < 9; ++k)
        if (!std::isfinite(inv[k])) return false;
    return true;
}

void mul_3x3(const f32* a, const f32* b, f32* out) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c] +
                             a[r * 3 + 1] * b[1 * 3 + c] +
                             a[r * 3 + 2] * b[2 * 3 + c];
}

// Scale each row to sum to exactly 1. That is the algebraic statement of "a
// neutral input renders neutral", which is what keeps a global tint out of the
// result, and it fixes the overall gain at unity so the automatic exposure is
// the only thing setting brightness.
bool normalise_rows(const f32* m, f32* out) {
    for (int r = 0; r < 3; ++r) {
        const f32 sum = m[r * 3 + 0] + m[r * 3 + 1] + m[r * 3 + 2];
        if (!(std::fabs(sum) > 1e-6f) || !std::isfinite(sum)) return false;
        const f32 k = 1.f / sum;
        for (int c = 0; c < 3; ++c) out[r * 3 + c] = m[r * 3 + c] * k;
    }
    return true;
}

void set_identity(f32* m) {
    for (int k = 0; k < 9; ++k) m[k] = (k % 4 == 0) ? 1.f : 0.f;
}

bool is_identity(const f32* m) {
    for (int k = 0; k < 9; ++k)
        if (std::fabs(m[k] - ((k % 4 == 0) ? 1.f : 0.f)) > 1e-5f) return false;
    return true;
}

// The camera->sRGB matrix for the space this renderer actually applies it in:
// after the white-balance gains, never before.
//
// Two matrices can be in the file and they live in DIFFERENT spaces, which is
// the trap this function exists to avoid:
//
//   ColorMatrix1 (50721) is XYZ -> camera-native, so inverting it gives a
//   transform for camera values with NO gains in them. Applying it to
//   white-balanced pixels is wrong by exactly the gains, and the error is per
//   INPUT channel -- a column scale. No row normalisation can absorb it, which
//   is why forcing the rows to sum equally makes neutrals come out right while
//   everything saturated stays cast.
//
//   Tag 65000 caches LibRaw's rgb_cam, which dcraw-derived pipelines apply
//   AFTER white balance. That one is already in this renderer's space and must
//   NOT be column-scaled.
//
// So: derive from ColorMatrix1 when the file carries a real one, and fall back
// to the cached matrix only when it does not. dng_writer emits an IDENTITY
// ColorMatrix1 as a placeholder when the capture had no profile, so identity
// has to be read as "absent" rather than as "no colour transform" -- taking it
// literally treats camera RGB as XYZ, which renders a blue sky pink.
bool build_render_matrix(const LinearDngColorInfo& info, const f32 total_gain[3], f32 out[9]) {
    if (info.has_color_matrix && !is_identity(info.color_matrix)) {
        f32 cam_to_xyz[9], m[9];
        if (invert_3x3(info.color_matrix, cam_to_xyz)) {
            mul_3x3(kXyzToSrgb, cam_to_xyz, m);
            // Divide out everything the samples carry by the time the matrix
            // runs, per input channel: that is what puts inv(ColorMatrix) back
            // on the camera-native values it is defined for.
            bool ok = true;
            for (int c = 0; c < 3; ++c) {
                if (!(std::fabs(total_gain[c]) > 1e-6f) || !std::isfinite(total_gain[c])) {
                    ok = false;
                    break;
                }
                for (int r = 0; r < 3; ++r) m[r * 3 + c] /= total_gain[c];
            }
            if (ok && normalise_rows(m, out)) return true;
        }
    }
    if (info.has_cam_to_srgb && !is_identity(info.cam_to_srgb) &&
        normalise_rows(info.cam_to_srgb, out))
        return true;
    set_identity(out);
    return false;
}

// ---------------------------------------------------------------- box blur
// Separable running-sum box filter: O(pixels) whatever the radius, which is
// what makes the guided filter below affordable at any smoothing scale.
//
// Moving the window from x to x+1 drops s[x-r] -- the element that was at the
// window's left edge -- and takes in s[x+r+1]. Dropping s[x-r-1] instead, which
// is the easy slip here, leaves one extra sample inside the accumulator while
// the divisor counts the window's true width, so every output past the first
// row/column of the ramp comes back scaled by (2r+2)/(2r+1). At the small radii
// a low-resolution grid uses that is not a rounding error: r=2 inflates the
// result by 20%, which reads as amplified chroma, and on the guided filter's
// base layer as a tone map pulling harder than it was asked to.
void box_filter(const std::vector<f32>& src, std::vector<f32>& dst,
                int w, int h, int r) {
    dst.assign((size_t)w * h, 0.f);
    std::vector<f32> tmp((size_t)w * h, 0.f);
    for (int y = 0; y < h; ++y) {
        const f32* s = &src[(size_t)y * w];
        f32* t = &tmp[(size_t)y * w];
        f32 acc = 0.f;
        for (int x = 0; x <= std::min(r, w - 1); ++x) acc += s[x];
        for (int x = 0; x < w; ++x) {
            const int lo = std::max(0, x - r), hi = std::min(w - 1, x + r);
            t[x] = acc / (f32)(hi - lo + 1);
            if (x + r + 1 < w) acc += s[x + r + 1];
            if (x - r >= 0) acc -= s[x - r];
        }
    }
    for (int x = 0; x < w; ++x) {
        f32 acc = 0.f;
        for (int y = 0; y <= std::min(r, h - 1); ++y) acc += tmp[(size_t)y * w + x];
        for (int y = 0; y < h; ++y) {
            const int lo = std::max(0, y - r), hi = std::min(h - 1, y + r);
            dst[(size_t)y * w + x] = acc / (f32)(hi - lo + 1);
            if (y + r + 1 < h) acc += tmp[(size_t)(y + r + 1) * w + x];
            if (y - r >= 0) acc -= tmp[(size_t)(y - r) * w + x];
        }
    }
}

// ------------------------------------------------------------ guided filter
// He et al., self-guided. Smooths inside regions but holds edges: a Gaussian
// base layer bleeds a bright sky across a roofline and the compression then
// carves a halo along it.
void guided_self(const std::vector<f32>& I, std::vector<f32>& q,
                 int w, int h, int r, f32 eps) {
    const size_t n = (size_t)w * h;
    std::vector<f32> II(n), mean_I, mean_II;
    for (size_t i = 0; i < n; ++i) II[i] = I[i] * I[i];
    box_filter(I, mean_I, w, h, r);
    box_filter(II, mean_II, w, h, r);

    std::vector<f32> a(n), b(n);
    for (size_t i = 0; i < n; ++i) {
        const f32 var = std::max(mean_II[i] - mean_I[i] * mean_I[i], 0.f);
        a[i] = var / (var + eps);      // ~1 across an edge, ~0 inside flat areas
        b[i] = mean_I[i] * (1.f - a[i]);
    }
    std::vector<f32> mean_a, mean_b;
    box_filter(a, mean_a, w, h, r);
    box_filter(b, mean_b, w, h, r);
    q.resize(n);
    for (size_t i = 0; i < n; ++i) q[i] = mean_a[i] * I[i] + mean_b[i];
}

// Bilinear sample of a low-res map at a full-res pixel, with the half-pixel
// offset the box downsample implies. Without it the map sits a fraction of a
// cell off and the gain drifts against the image it was computed from.
inline f32 sample_map(const std::vector<f32>& map, int gw, int gh,
                      f32 fx_full, f32 fy_full, int shift) {
    const f32 fx = (fx_full + 0.5f) / (f32)(1 << shift) - 0.5f;
    const f32 fy = (fy_full + 0.5f) / (f32)(1 << shift) - 0.5f;
    const int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
    const f32 tx = fx - (f32)x0, ty = fy - (f32)y0;
    const int xa = std::min(std::max(x0, 0), gw - 1);
    const int xb = std::min(std::max(x0 + 1, 0), gw - 1);
    const int ya = std::min(std::max(y0, 0), gh - 1);
    const int yb = std::min(std::max(y0 + 1, 0), gh - 1);
    const f32 v00 = map[(size_t)ya * gw + xa], v10 = map[(size_t)ya * gw + xb];
    const f32 v01 = map[(size_t)yb * gw + xa], v11 = map[(size_t)yb * gw + xb];
    return (v00 * (1.f - tx) + v10 * tx) * (1.f - ty) +
           (v01 * (1.f - tx) + v11 * tx) * ty;
}

// ------------------------------------------------------------- output curve
// Extended Reinhard: maps [0,inf) onto [0,1] and sends `white` exactly to 1, so
// the brightest content in the frame lands at display white instead of being
// clipped there. Monotone everywhere, which the gamut step relies on.
inline f32 tone_curve(f32 x, f32 white) {
    const f32 w2 = std::max(white * white, kEps);
    return x * (1.f + x / w2) / (1.f + x);
}

inline f32 srgb_oetf_exact(f32 v) {
    v = clampf(v, 0.f, 1.f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
}

constexpr int kOetfN = 2048;

inline f32 lut_sample(const std::vector<f32>& t, f32 x, f32 scale) {
    const f32 p = x * scale;
    const int i = (int)p;
    if (i < 0) return t.front();
    if (i >= (int)t.size() - 1) return t.back();
    return t[i] + (t[i + 1] - t[i]) * (p - (f32)i);
}

// Contrast as an S about mid grey, applied to luminance and re-applied to RGB
// as a ratio, so it cannot rotate hue the way three independent curves would.
inline f32 s_curve(f32 v, f32 amount) {
    if (amount <= 0.f) return v;
    const f32 t = clampf(v, 0.f, 1.f);
    const f32 s = t * t * (3.f - 2.f * t);
    return t + (s - t) * amount;
}

// ------------------------------------------------------------------- state
struct State {
    int W = 0, H = 0;
    int gw = 0, gh = 0, shift = 3;
    f32 gains[3] = {1.f, 1.f, 1.f};    // white balance this renderer applies
    f32 black = 0.f, black_scale = 1.f;
    f32 m[9] = {1,0,0, 0,1,0, 0,0,1};
    f32 exposure = 1.f;
    f32 white = 4.f;
    f32 display_black = 0.f;   // measured after the analysis, see below
    std::vector<f32> gain;          // low-res tone-mapping gain
    std::vector<f32> oetf;
    FinishHdrParams p;
};

// Camera sample -> linear sRGB, everything that happens before exposure.
// Shared verbatim by the analysis pass and the render so the gain map cannot
// describe a different image from the one it is applied to.
inline void to_linear_srgb(const State& st, f32 r, f32 g, f32 b,
                           f32& lr, f32& lg, f32& lb) {
    // Read the clip BEFORE anything scales the channels. In camera space all
    // three saturate at the same value; after the gains they do not, and a test
    // there sees only the channel the gains pushed through full scale first.
    const f32 cam_max = std::max(r, std::max(g, b));

    // White balance in float with no ceiling: red and blue legitimately exceed
    // 1 from here on, and that over-range IS the highlight headroom the
    // compression below is going to spend.
    r *= st.gains[0];
    g *= st.gains[1];
    b *= st.gains[2];

    // Black AFTER the gains, because one scalar can only be neutral in a space
    // where a neutral pixel has three equal channels. In the un-white-balanced
    // container a neutral grey is stored as (g/2.06, g, g/1.84), so subtracting
    // one number from all three takes roughly twice as much of red as it does
    // of green -- and since the level being subtracted is itself a low
    // percentile of the darkest content, it is red that lands on zero and the
    // shadows go green. The pedestal this removes is veiling glare, which is
    // neutral in SCENE space, i.e. gains[c] times smaller per channel in the
    // stored samples; applying the gains first is what makes it one number.
    // Prewhitened files have gains of 1 and are unaffected either way.
    r = std::max((r - st.black) * st.black_scale, 0.f);
    g = std::max((g - st.black) * st.black_scale, 0.f);
    b = std::max((b - st.black) * st.black_scale, 0.f);

    // What the sensor blew renders white: pull the pixel to neutral at its own
    // peak, weighted by how far into the clip it went.
    //
    // Two independent pieces of evidence, because the first one is destroyed by
    // the merge. cam_max asks whether a stored channel is still at the sensor
    // ceiling; the merge's kernel-weighted average mixes a clipped green with
    // unclipped neighbours across the soft edge of any highlight and leaves it
    // just under, at which point this test silently gives up. The second asks
    // whether the DIMMEST balanced channel has reached display white, which no
    // saturated colour does -- see the note in finish_hdr.h for the measured
    // separation. Whichever says "blown" more strongly wins.
    const f32 bal_mx = std::max(r, std::max(g, b));
    const f32 bal_mn = std::min(r, std::min(g, b));
    const f32 t_ceiling = smoothstepf(st.p.clip_soft, st.p.clip_threshold, cam_max);
    const f32 t_white = smoothstepf(st.p.clip_neutral_min, st.p.clip_neutral_full, bal_mn) *
                        smoothstepf(1.0f, 1.15f, bal_mx);
    const f32 t = std::max(t_ceiling, t_white);
    if (t > 0.f) {
        r += (bal_mx - r) * t;
        g += (bal_mx - g) * t;
        b += (bal_mx - b) * t;
    }

    lr = st.m[0] * r + st.m[1] * g + st.m[2] * b;
    lg = st.m[3] * r + st.m[4] * g + st.m[5] * b;
    lb = st.m[6] * r + st.m[7] * g + st.m[8] * b;

    // A saturated camera colour can land outside sRGB on the LOW side. Clamping
    // the negative channel to zero leaves red and blue with no green, which is
    // magenta; desaturating toward the pixel's own luminance instead lands it
    // on the gamut boundary at the same brightness and nearly the same hue.
    const f32 mn = std::min(lr, std::min(lg, lb));
    if (mn < 0.f) {
        const f32 y = luma_of(lr, lg, lb);
        if (y <= 0.f) {
            lr = lg = lb = 0.f;
        } else {
            const f32 s = y / (y - mn);   // in [0,1): sends the low channel to 0
            lr = std::max(y + (lr - y) * s, 0.f);
            lg = std::max(y + (lg - y) * s, 0.f);
            lb = std::max(y + (lb - y) * s, 0.f);
        }
    }
}

// --------------------------------------------------------- chroma denoise
// Luma is preserved exactly; only the colour difference from luma is filtered,
// so nothing sharp is touched -- just the fine colour variation the shadow lift
// would otherwise amplify into blotching.
//
// Runs on a 1/8 grid because chroma is inherently low-frequency (the same
// reason JPEG subsamples it) and because two full-resolution float planes at
// 48MP would cost ~380MB on a device this pipeline has already been jetsammed
// on. The gains are applied before the split and divided out after it, so the
// quantity held constant is the luminance of the RENDERED pixel, not of the
// raw camera one.
void denoise_chroma_camera(uint16_t* rgb16, int W, int H, const f32 gains[3],
                           const FinishHdrParams& p) {
    const f32 amount = clampf(p.chroma_denoise, 0.f, 1.f);
    if (!rgb16 || W <= 0 || H <= 0 || amount <= 0.f) return;

    // The differences are formed on WHITE-BALANCED values, not on the stored
    // ones. (r - Y) is only a colour-opponent signal where a neutral pixel has
    // three equal channels; on un-white-balanced samples r is about half of Y
    // for neutral content, so (r - Y) tracks brightness instead of colour and
    // box-blurring it drags luminance across every edge into the red channel.
    // With gains of 1 -- the prewhitened container -- this is arithmetically
    // the same as filtering the stored values directly.
    constexpr f32 kWy[3] = {0.2126f, 0.7152f, 0.0722f};
    f32 ginv[3];
    for (int c = 0; c < 3; ++c) {
        if (!(gains[c] > kEps) || !std::isfinite(gains[c])) return;
        ginv[c] = 1.f / gains[c];
    }

    // Sensor-clipped pixels are left exactly as they are, and kept out of the
    // neighbourhood averages as well.
    //
    // to_linear_srgb decides a pixel was blown by testing cam_max on the STORED
    // sample -- and this function runs first and writes that same buffer back.
    // Inside a large blown region the blur returns the region's own value and
    // nothing moves, but within a filter radius of its edge it mixes in the
    // surroundings, and dropping a stored channel below clip_threshold there
    // switches the neutral pull off for that pixel alone. The result is a magenta
    // ring one radius wide around every blown highlight: the gains push red and
    // blue through full scale first, so an unprotected clip renders pink. Pixels
    // at or above clip_soft are the ones the pull acts on at all, so that is the
    // line -- below it the smoothstep is zero and nothing here can matter.
    const f32 clip_guard = std::isfinite(p.clip_soft) ? p.clip_soft : 1.f;

    constexpr int kCShift = 3;                  // 1/8 in each axis
    const int cw = std::max(1, (W + (1 << kCShift) - 1) >> kCShift);
    const int ch = std::max(1, (H + (1 << kCShift) - 1) >> kCShift);
    // Radius is specified in full-resolution pixels so the control means the
    // same thing regardless of sensor size.
    const int rad = std::max(1, (int)std::lround(
        std::max(1.f, p.chroma_denoise_radius) / (f32)(1 << kCShift)));

    std::vector<f32> dr((size_t)cw * ch, 0.f), db((size_t)cw * ch, 0.f);
    std::vector<f32> cnt((size_t)cw * ch, 0.f);

    const f32 inv = 1.f / 65535.f;
    for (int y = 0; y < H; ++y) {
        const int cy = y >> kCShift;
        const uint16_t* row = rgb16 + (size_t)y * (size_t)W * 3u;
        for (int x = 0; x < W; ++x) {
            const f32 s0 = row[x * 3 + 0] * inv;
            const f32 s1 = row[x * 3 + 1] * inv;
            const f32 s2 = row[x * 3 + 2] * inv;
            if (std::max(s0, std::max(s1, s2)) >= clip_guard) continue;
            const f32 r = s0 * gains[0];
            const f32 g = s1 * gains[1];
            const f32 b = s2 * gains[2];
            const f32 Y = kWy[0] * r + kWy[1] * g + kWy[2] * b;
            const size_t c = (size_t)cy * cw + (size_t)(x >> kCShift);
            dr[c] += r - Y;
            db[c] += b - Y;
            cnt[c] += 1.f;
        }
    }
    for (size_t i = 0; i < dr.size(); ++i) {
        if (cnt[i] > 0.f) {
            dr[i] /= cnt[i];
            db[i] /= cnt[i];
        }
    }
    // Cells that were entirely blown hold no samples. The pixels in them are
    // skipped, but a neighbouring cell's bilinear tap still reaches them, and a
    // zero there would read as "neutral chroma" and drag the colour of the last
    // unblown ring toward grey. Fill them from the nearest cell that has data.
    {
        std::vector<uint8_t> have((size_t)cw * ch);
        for (size_t i = 0; i < have.size(); ++i) have[i] = cnt[i] > 0.f ? 1u : 0u;
        for (int pass = 0; pass < 2; ++pass) {
            bool changed = false;
            for (int gy = 0; gy < ch; ++gy)
                for (int gx = 0; gx < cw; ++gx) {
                    const size_t i = (size_t)gy * cw + gx;
                    if (have[i]) continue;
                    f32 ar = 0.f, ab = 0.f, n = 0.f;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = gx + dx, ny = gy + dy;
                            if (nx < 0 || ny < 0 || nx >= cw || ny >= ch) continue;
                            const size_t j = (size_t)ny * cw + nx;
                            if (!have[j]) continue;
                            ar += dr[j];
                            ab += db[j];
                            n += 1.f;
                        }
                    if (n > 0.f) {
                        dr[i] = ar / n;
                        db[i] = ab / n;
                        have[i] = 2u;      // filled this pass, usable in the next
                        changed = true;
                    }
                }
            for (size_t i = 0; i < have.size(); ++i) if (have[i] == 2u) have[i] = 1u;
            if (!changed) break;
        }
    }

    std::vector<f32> br, bb;
    box_filter(dr, br, cw, ch, rad);
    box_filter(db, bb, cw, ch, rad);

    parallel_rows(H, 0, [&](int y) {
        uint16_t* row = rgb16 + (size_t)y * (size_t)W * 3u;
        for (int x = 0; x < W; ++x) {
            const f32 s0 = row[x * 3 + 0] * inv;
            const f32 s1 = row[x * 3 + 1] * inv;
            const f32 s2 = row[x * 3 + 2] * inv;
            if (std::max(s0, std::max(s1, s2)) >= clip_guard) continue;
            const f32 nr = sample_map(br, cw, ch, (f32)x, (f32)y, kCShift);
            const f32 nb = sample_map(bb, cw, ch, (f32)x, (f32)y, kCShift);
            const f32 r = s0 * gains[0];
            const f32 g = s1 * gains[1];
            const f32 b = s2 * gains[2];
            const f32 Y = kWy[0] * r + kWy[1] * g + kWy[2] * b;
            const f32 cr = (r - Y) + (nr - (r - Y)) * amount;
            const f32 cb = (b - Y) + (nb - (b - Y)) * amount;
            // The third difference follows from the identity rather than being
            // carried, so Y survives exactly instead of approximately.
            const f32 cg = -(kWy[0] * cr + kWy[2] * cb) / kWy[1];
            // Back to stored space: the gains go out the way they came in, so
            // the buffer this hands on is still the container it was read from.
            row[x * 3 + 0] = (uint16_t)clampf(std::lround((Y + cr) * ginv[0] * 65535.f), 0.f, 65535.f);
            row[x * 3 + 1] = (uint16_t)clampf(std::lround((Y + cg) * ginv[1] * 65535.f), 0.f, 65535.f);
            row[x * 3 + 2] = (uint16_t)clampf(std::lround((Y + cb) * ginv[2] * 65535.f), 0.f, 65535.f);
        }
    });
}

// -------------------------------------------------------------- black level
// A low percentile of the per-pixel MINIMUM channel: whatever floor every
// channel sits above is a pedestal, not content. Sampled on a stride -- a few
// million samples decide a percentile just as well as fifty million do.
//
// One scalar for all three channels. A per-channel black would be a shadow
// tint, and the DNG's own per-channel black was already removed upstream, so
// anything left here is veiling glare, which is neutral.
//
// Neutral in SCENE space, though, which is why the gains go on before the
// minimum is taken. Un-white-balanced samples put a neutral grey at
// (g/2.06, g, g/1.84), so the per-pixel minimum there is just the red channel
// and this would measure the red level of the darkest content rather than any
// pedestal -- and to_linear_srgb would then subtract it from all three.
f32 measure_black(const uint16_t* rgb16, int W, int H, const f32 gains[3],
                  const FinishHdrParams& p) {
    constexpr int kBins = 4096;
    constexpr f32 kRange = 0.25f;     // above this nothing counts as black
    std::vector<uint32_t> hist((size_t)kBins, 0u);
    const int step = std::max(1, (int)std::lround(std::sqrt((double)W * (double)H / 3.0e6)));
    uint64_t total = 0;
    for (int y = 0; y < H; y += step) {
        const uint16_t* row = rgb16 + (size_t)y * (size_t)W * 3u;
        for (int x = 0; x < W; x += step) {
            const f32 r = row[x * 3 + 0] * gains[0];
            const f32 g = row[x * 3 + 1] * gains[1];
            const f32 b = row[x * 3 + 2] * gains[2];
            const f32 mn = std::min(r, std::min(g, b)) * (1.f / 65535.f);
            int bin = (int)(mn / kRange * (f32)(kBins - 1));
            if (bin < 0) bin = 0;
            if (bin >= kBins) bin = kBins - 1;
            ++hist[(size_t)bin];
            ++total;
        }
    }
    if (total == 0) return 0.f;
    const uint64_t want = (uint64_t)((double)total * (double)clampf(p.black_percentile, 0.f, 0.05f));
    uint64_t acc = 0;
    int bin = 0;
    for (; bin < kBins; ++bin) {
        acc += hist[(size_t)bin];
        if (acc > want) break;
    }
    const f32 black = (f32)std::min(bin, kBins - 1) / (f32)(kBins - 1) * kRange;
    return clampf(black, 0.f, clampf(p.black_max, 0.f, 0.2f));
}

// --------------------------------------------------------------- analysis
// One pass over the image producing the automatic exposure, the white point the
// shoulder rolls off to, and the low-resolution tone-mapping gain map.
void analyse(const uint16_t* rgb16, State& st) {
    const int W = st.W, H = st.H;

    st.oetf.resize(kOetfN);
    for (int i = 0; i < kOetfN; ++i)
        st.oetf[i] = srgb_oetf_exact((f32)i / (f32)(kOetfN - 1));

    // Downsample so the short side keeps ~128 samples: enough for the base
    // layer to follow real scene structure, small enough that the whole
    // analysis is a few MB and a few milliseconds at 48MP.
    int shift = 3;
    while (shift > 0 && (std::min(W, H) >> shift) < 128) --shift;
    while ((std::min(W, H) >> shift) > 384 && shift < 6) ++shift;
    st.shift = shift;
    const int f = 1 << shift;
    const int gw = std::max(1, W / f), gh = std::max(1, H / f);
    st.gw = gw;
    st.gh = gh;

    // Two grids. The mean drives the exposure and the base layer; the MAX
    // exists only to find the white point. Taking the white point from the mean
    // would hide highlights -- box-averaging an 8x8 block buries any bright
    // feature smaller than the block, so a specular or a bright window reads
    // far dimmer than it is and the shoulder gets set below it.
    std::vector<f32> Y((size_t)gw * gh, 0.f);
    std::vector<f32> Ymax((size_t)gw * gh, 0.f);
    parallel_rows(gh, 0, [&](int gy) {
        f32* out = &Y[(size_t)gy * gw];
        f32* omax = &Ymax[(size_t)gy * gw];
        std::vector<f32> cnt((size_t)gw, 0.f);
        const int y0 = gy * f;
        const int y1 = (gy == gh - 1) ? H : std::min(H, y0 + f);
        for (int y = y0; y < y1; ++y) {
            const uint16_t* row = rgb16 + (size_t)y * (size_t)W * 3u;
            for (int x = 0; x < W; ++x) {
                const int gx = std::min(gw - 1, x / f);
                f32 lr, lg, lb;
                to_linear_srgb(st, row[x * 3 + 0] * (1.f / 65535.f),
                                   row[x * 3 + 1] * (1.f / 65535.f),
                                   row[x * 3 + 2] * (1.f / 65535.f), lr, lg, lb);
                const f32 l = luma_of(lr, lg, lb);
                out[gx] += l;
                if (l > omax[gx]) omax[gx] = l;
                cnt[gx] += 1.f;
            }
        }
        for (int gx = 0; gx < gw; ++gx) out[gx] /= std::max(cnt[gx], 1.f);
    });

    // Automatic exposure from the log-average, which tracks what the scene
    // contains rather than its extremes: a small bright window does not drag
    // the whole frame dark the way a max-based normalisation would.
    double logsum = 0.0;
    for (f32 v : Y) logsum += std::log((double)std::max(v, 1e-5f));
    const f32 log_avg = (f32)std::exp(logsum / (double)std::max<size_t>(Y.size(), 1));
    f32 exposure = std::max(st.p.auto_key, 1e-3f) / std::max(log_avg, 1e-5f);
    exposure *= std::pow(2.f, st.p.exposure_ev);
    st.exposure = clampf(exposure, 0.05f, 256.f);

    for (f32& v : Y) v *= st.exposure;
    for (f32& v : Ymax) v *= st.exposure;

    // Base layer in log2: tone mapping is a ratio operation, and doing it in
    // log makes the compression uniform across stops instead of biased toward
    // the highlights. It is also what stops the guided filter producing halos.
    std::vector<f32> L(Y.size());
    for (size_t i = 0; i < Y.size(); ++i) L[i] = std::log2(std::max(Y[i], 1e-5f));

    const int radius = std::max(2, std::min(gw, gh) / 16);
    // eps in log2 units squared: half a stop of local variation is texture to
    // smooth over, more than that is an edge to preserve.
    std::vector<f32> Lb;
    guided_self(L, Lb, gw, gh, radius, 0.25f);

    // Compress the base around middle grey, asymmetrically: above the pivot the
    // range is squeezed toward it (highlights come down), below it the same
    // squeeze lifts shadows up. Detail is untouched -- only the base moves, and
    // the result is used as a multiplier, which is the multiplicative form of
    // base+detail recombination.
    const f32 pivot = std::log2(kMidGrey);
    const f32 hi = clampf(1.f - 0.5f * st.p.highlight_rolloff, 0.15f, 1.f);
    const f32 lo = clampf(1.f - 0.5f * st.p.shadow_lift, 0.15f, 1.f);
    const f32 amt = clampf(st.p.local_strength, 0.f, 1.f);

    st.gain.resize(Y.size());
    for (size_t i = 0; i < Y.size(); ++i) {
        const f32 d = Lb[i] - pivot;
        const f32 dc = (d > 0.f) ? d * hi : d * lo;
        const f32 moved = d + amt * (dc - d);
        st.gain[i] = clampf(std::exp2(moved - d), 0.05f, 20.f);
    }

    // White point AFTER the local gain, because that is the signal the shoulder
    // actually receives. Measuring the peak on the uncompressed luminance would
    // place the shoulder for a range the tone mapper has already reduced, and
    // the whole top end would then render flat.
    std::vector<f32> peaks(Ymax.size());
    for (size_t i = 0; i < Ymax.size(); ++i) peaks[i] = Ymax[i] * st.gain[i];
    std::sort(peaks.begin(), peaks.end());
    const f32 peak = peaks[(size_t)((f32)(peaks.size() - 1) * 0.9999f)];
    st.white = clampf(peak, 1.2f, 512.f);
}

// ------------------------------------------------------------ render pixel
inline void render_pixel(const State& st, f32 r, f32 g, f32 b,
                         f32 fx, f32 fy, f32& sr, f32& sg, f32& sb) {
    f32 lr, lg, lb;
    to_linear_srgb(st, r, g, b, lr, lg, lb);

    lr *= st.exposure;
    lg *= st.exposure;
    lb *= st.exposure;

    // ONE factor for all three channels. Bending R, G and B through a tone
    // curve independently compresses whichever is largest the hardest, which
    // drains saturation and rotates hue -- the plastic look of naive tone
    // mapping. A scalar gain cannot do either: every ratio survives it.
    const f32 gain = sample_map(st.gain, st.gw, st.gh, fx, fy, st.shift);
    lr *= gain;
    lg *= gain;
    lb *= gain;

    const f32 y_lin = luma_of(lr, lg, lb);
    f32 y_out = 0.f;
    if (y_lin > kEps) {
        y_out = tone_curve(y_lin, st.white);
        const f32 k = y_out / y_lin;
        lr *= k;
        lg *= k;
        lb *= k;
    } else {
        lr = lg = lb = 0.f;
    }

    // Holding luminance lets a saturated channel land above 1. Roll it back
    // toward the neutral of that same luminance rather than clipping: a bright
    // saturated colour desaturates toward white the way it does optically, and
    // a per-channel clip here would reintroduce exactly the hue shift the
    // luminance curve just avoided.
    {
        const f32 mx = std::max(lr, std::max(lg, lb));
        if (mx > 1.f) {
            const f32 t = clampf((mx - 1.f) / std::max(mx - y_out, kEps), 0.f, 1.f);
            lr += (y_out - lr) * t;
            lg += (y_out - lg) * t;
            lb += (y_out - lb) * t;
        }
    }

    const f32 osc = (f32)(kOetfN - 1);
    sr = lut_sample(st.oetf, clampf(lr, 0.f, 1.f), osc);
    sg = lut_sample(st.oetf, clampf(lg, 0.f, 1.f), osc);
    sb = lut_sample(st.oetf, clampf(lb, 0.f, 1.f), osc);

    // Display black anchor. After the OETF, because this is a levels move on
    // the rendered image -- what is being removed is the greyness the shadow
    // lift left behind, not light.
    if (st.display_black > 0.f) {
        const f32 bp = st.display_black;
        const f32 iv = 1.f / (1.f - bp);
        sr = clampf((sr - bp) * iv, 0.f, 1.f);
        sg = clampf((sg - bp) * iv, 0.f, 1.f);
        sb = clampf((sb - bp) * iv, 0.f, 1.f);
    }

    if (st.p.contrast > 0.f) {
        const f32 yl = luma_of(sr, sg, sb);
        if (yl > kEps) {
            const f32 k = s_curve(yl, st.p.contrast) / yl;
            sr = clampf(sr * k, 0.f, 1.f);
            sg = clampf(sg * k, 0.f, 1.f);
            sb = clampf(sb * k, 0.f, 1.f);
        }
    }

    const f32 yl = luma_of(sr, sg, sb);
    f32 amount = st.p.saturation - 1.f;
    if (st.p.vibrance > 0.f) {
        const f32 mx = std::max(sr, std::max(sg, sb));
        const f32 mn = std::min(sr, std::min(sg, sb));
        const f32 sat = (mx > kEps) ? (mx - mn) / mx : 0.f;
        const f32 head = (1.f - sat) * (1.f - sat);
        amount += st.p.vibrance * head;
    }
    // Never re-saturate a highlight. The steps above neutralise a near-white
    // tone toward its own luminance; vibrance would then amplify whatever
    // channel imbalance is left, hardest exactly there (head is largest for
    // low-saturation pixels), and that is what paints clipped highlights pink.
    amount *= (1.f - smoothstepf(0.78f, 0.98f, yl));
    if (std::fabs(amount) > 1e-4f) {
        const f32 k = 1.f + amount;
        sr = yl + (sr - yl) * k;
        sg = yl + (sg - yl) * k;
        sb = yl + (sb - yl) * k;
    }

    const f32 mx = std::max(sr, std::max(sg, sb));
    if (mx > 1.f) {
        const f32 t = clampf((mx - 1.f) / std::max(mx, kEps), 0.f, 1.f);
        sr += (yl - sr) * t;
        sg += (yl - sg) * t;
        sb += (yl - sb) * t;
        const f32 mx2 = std::max(sr, std::max(sg, sb));
        if (mx2 > 1.f) {
            const f32 s = 1.f / mx2;
            sr *= s;
            sg *= s;
            sb *= s;
        }
    }
    sr = clampf(sr, 0.f, 1.f);
    sg = clampf(sg, 0.f, 1.f);
    sb = clampf(sb, 0.f, 1.f);
}

// Where the render actually puts its darkest content. Runs the real per-pixel
// chain on a stride -- a few million samples decide a percentile as well as
// fifty million do -- with the anchor still at zero, so what comes back is the
// unanchored distribution it is about to correct.
void measure_display_black(const uint16_t* rgb16, State& st) {
    const f32 frac = clampf(st.p.display_black_percentile, 0.f, 0.05f);
    const f32 cap = clampf(st.p.display_black_max, 0.f, 0.5f);
    st.display_black = 0.f;
    if (frac <= 0.f || cap <= 0.f) return;

    constexpr int kBins = 1024;
    std::vector<uint32_t> hist((size_t)kBins, 0u);
    const int step = std::max(1, (int)std::lround(
        std::sqrt((double)st.W * (double)st.H / 2.0e6)));
    uint64_t total = 0;
    for (int y = 0; y < st.H; y += step) {
        const uint16_t* row = rgb16 + (size_t)y * (size_t)st.W * 3u;
        for (int x = 0; x < st.W; x += step) {
            f32 sr, sg, sb;
            render_pixel(st, row[x * 3 + 0] * (1.f / 65535.f),
                             row[x * 3 + 1] * (1.f / 65535.f),
                             row[x * 3 + 2] * (1.f / 65535.f),
                         (f32)x, (f32)y, sr, sg, sb);
            int bin = (int)(luma_of(sr, sg, sb) * (f32)(kBins - 1));
            hist[(size_t)clampf((f32)bin, 0.f, (f32)(kBins - 1))] += 1u;
            ++total;
        }
    }
    if (total == 0) return;
    const uint64_t want = (uint64_t)((double)total * (double)frac);
    uint64_t acc = 0;
    int bin = 0;
    for (; bin < kBins; ++bin) {
        acc += hist[(size_t)bin];
        if (acc > want) break;
    }
    st.display_black = clampf((f32)std::min(bin, kBins - 1) / (f32)(kBins - 1),
                              0.f, cap);
}

bool prepare(std::vector<uint16_t>& rgb, int W, int H,
             const LinearDngColorInfo& info, const FinishHdrParams& p, State& st) {
    if (rgb.empty() || W <= 0 || H <= 0) return false;
    st = State{};
    st.W = W;
    st.H = H;
    st.p = p;

    // The writer emits one of exactly two containers, and AsShotNeutral -- the
    // camera neutral expressed in the stored pixels' own space -- says which:
    //
    //   non-neutral  the pixels are camera-native and still need balancing.
    //                Gains to apply are 1/AsShotNeutral.
    //   neutral      the pixels were already multiplied by the gains before
    //                being stored, and AnalogBalance is where those gains went.
    //                Nothing to apply here.
    //
    // Read each container on its own terms rather than combining both tags.
    // One legacy build wrote AnalogBalance AND a non-neutral AsShotNeutral,
    // which asserts two different things about the same pixels -- the gains
    // were applied and they were not. Multiplying the two claims together
    // squares the column scale and casts the whole frame green; AsShotNeutral
    // is the one the pixels can be checked against, so it wins.
    f32 baked[3] = {1.f, 1.f, 1.f};
    bool neutral_asn = true;
    if (info.has_as_shot_neutral && info.as_shot_neutral[1] > kEps) {
        const f32 n0 = info.as_shot_neutral[0] / info.as_shot_neutral[1];
        const f32 n2 = info.as_shot_neutral[2] / info.as_shot_neutral[1];
        neutral_asn = std::fabs(n0 - 1.f) < 1e-3f && std::fabs(n2 - 1.f) < 1e-3f;
        if (!neutral_asn) {
            // Green-normalised, so the gains are pure colour and the overall
            // level is left to the automatic exposure.
            st.gains[0] = 1.f / n0;
            st.gains[1] = 1.f;
            st.gains[2] = 1.f / n2;
        }
    } else if (info.has_wb && info.wb[1] > kEps) {
        // No AsShotNeutral: fall back to the private tag's gains, which the
        // writer fills in for exactly the un-white-balanced container.
        st.gains[0] = info.wb[0] / info.wb[1];
        st.gains[1] = 1.f;
        st.gains[2] = info.wb[2] / info.wb[1];
        neutral_asn = false;
    }
    if (neutral_asn && info.has_analog_balance && info.analog_balance[1] > kEps) {
        baked[0] = info.analog_balance[0] / info.analog_balance[1];
        baked[1] = 1.f;
        baked[2] = info.analog_balance[2] / info.analog_balance[1];
    }
    // What the samples carry by the time the matrix runs: what was already in
    // them plus what this renderer just multiplied in.
    f32 total_gain[3];
    for (int c = 0; c < 3; ++c) {
        if (!std::isfinite(st.gains[c]) || st.gains[c] <= kEps) st.gains[c] = 1.f;
        if (!std::isfinite(baked[c]) || baked[c] <= kEps) baked[c] = 1.f;
        total_gain[c] = st.gains[c] * baked[c];
    }

    build_render_matrix(info, total_gain, st.m);

    denoise_chroma_camera(rgb.data(), W, H, st.gains, p);

    st.black = measure_black(rgb.data(), W, H, st.gains, p);
    st.black_scale = 1.f / std::max(1.f - st.black, 1e-3f);

    analyse(rgb.data(), st);
    measure_display_black(rgb.data(), st);
    return true;
}

}  // namespace

bool finish_hdr_from_dng(const std::string& dng_path, const FinishHdrParams& p,
                         std::vector<uint8_t>& out_rgb8, int& W, int& H,
                         int& orientation_out) {
    out_rgb8.clear();
    W = H = 0;
    orientation_out = 1;

    std::vector<uint16_t> rgb;
    LinearDngColorInfo info;
    if (!load_linear_dng_rgb16_info(dng_path, rgb, W, H, info) || W <= 0 || H <= 0)
        return false;
    orientation_out = info.orientation;

    State st;
    if (!prepare(rgb, W, H, info, p, st)) return false;

    // RGB, not RGBA: the alpha byte was 255 everywhere and JPEG has no use for
    // it, so at 48MP it would cost 48MB and a quarter of the store traffic for
    // nothing.
    out_rgb8.resize((size_t)W * (size_t)H * 3u);
    parallel_rows(H, 0, [&](int y) {
        const uint16_t* src = rgb.data() + (size_t)y * (size_t)W * 3u;
        uint8_t* dst = out_rgb8.data() + (size_t)y * (size_t)W * 3u;
        for (int x = 0; x < W; ++x) {
            f32 sr, sg, sb;
            render_pixel(st, src[x * 3 + 0] * (1.f / 65535.f),
                             src[x * 3 + 1] * (1.f / 65535.f),
                             src[x * 3 + 2] * (1.f / 65535.f),
                         (f32)x, (f32)y, sr, sg, sb);
            dst[x * 3 + 0] = (uint8_t)std::lround(sr * 255.f);
            dst[x * 3 + 1] = (uint8_t)std::lround(sg * 255.f);
            dst[x * 3 + 2] = (uint8_t)std::lround(sb * 255.f);
        }
    });
    return true;
}

}  // namespace hhsr
