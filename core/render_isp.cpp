#include "render_isp.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hhsr {
namespace {

constexpr f32 kEps = 1e-6f;
// Where the tone-mapping pivot sits: content above it is compressed toward it,
// content below is lifted toward it. 0.18 is the photographic middle grey.
constexpr f32 kMidGrey = 0.18f;
// Where automatic exposure puts the scene's log-average. Deliberately higher
// than the pivot: the log-average is a geometric mean, so large dark regions
// pull it well below the median, and normalising it to 0.18 renders about 0.65
// stops dark. Set so the rendered median lands near 100/255 on the reference
// scenes, which is what the log-average alone does not achieve.
constexpr f32 kAutoKey = 0.2824f;

// clampf comes from types.h. Defining an identical one here made every call
// site ambiguous rather than shadowing it: the helpers live in an anonymous
// namespace, but isp_analyse and isp_render sit in hhsr proper, where both the
// anonymous-namespace copy and hhsr::clampf are equally visible.

inline f32 luma_of(f32 r, f32 g, f32 b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Camera linear -> sRGB linear, measured from a HandheldSR linear DNG and its
// reference render. The merge output is already white balanced, so applying the
// DNG's own ColorMatrix here would push colour twice and cast the highlights.
constexpr f32 kDefaultCamToSrgb[9] = {
     1.2466443f, -0.4477117f, -0.1773365f,
    -0.1616100f,  0.8074801f, -0.0321825f,
    -0.1166101f, -0.1502432f,  0.8686564f
};

// ---------------------------------------------------------------- box blur
// Separable running-sum box filter: O(pixels), independent of radius, which is
// what makes the guided filter below affordable at any smoothing scale.
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
            const int add = x + r + 1, sub = x - r - 1;
            const int lo = std::max(0, x - r), hi = std::min(w - 1, x + r);
            t[x] = acc / (f32)(hi - lo + 1);
            if (add < w) acc += s[add];
            if (sub >= 0) acc -= s[sub];
        }
    }
    for (int x = 0; x < w; ++x) {
        f32 acc = 0.f;
        for (int y = 0; y <= std::min(r, h - 1); ++y) acc += tmp[(size_t)y * w + x];
        for (int y = 0; y < h; ++y) {
            const int add = y + r + 1, sub = y - r - 1;
            const int lo = std::max(0, y - r), hi = std::min(h - 1, y + r);
            dst[(size_t)y * w + x] = acc / (f32)(hi - lo + 1);
            if (add < h) acc += tmp[(size_t)add * w + x];
            if (sub >= 0) acc -= tmp[(size_t)sub * w + x];
        }
    }
}

// ------------------------------------------------------------ guided filter
// He et al., self-guided (guide == input). Smooths inside regions but holds
// edges, which is the whole reason this is not a Gaussian: a Gaussian base layer
// bleeds a bright sky across a roofline and the tone mapping then carves a halo
// along it.
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
        a[i] = var / (var + eps);          // ~1 across an edge, ~0 inside flat areas
        b[i] = mean_I[i] * (1.f - a[i]);
    }
    std::vector<f32> mean_a, mean_b;
    box_filter(a, mean_a, w, h, r);
    box_filter(b, mean_b, w, h, r);
    q.resize(n);
    for (size_t i = 0; i < n; ++i) q[i] = mean_a[i] * I[i] + mean_b[i];
}

// ------------------------------------------------------------- output curve
// Extended Reinhard: maps [0,inf) onto [0,1] and sends `white` exactly to 1, so
// the brightest content in the frame lands at display white instead of being
// clipped there. Monotone everywhere, which the gamut step below relies on.
inline f32 tone_curve(f32 x, f32 white) {
    const f32 w2 = std::max(white * white, kEps);
    return x * (1.f + x / w2) / (1.f + x);
}

inline f32 srgb_oetf(f32 v) {
    v = clampf(v, 0.f, 1.f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
}

// Contrast as an S-curve about mid display grey. Applied to luminance and then
// re-applied to RGB as a ratio, so it cannot rotate hue the way three
// independent per-channel curves would.
inline f32 s_curve(f32 v, f32 amount) {
    if (amount <= 0.f) return v;
    const f32 t = clampf(v, 0.f, 1.f);
    // smoothstep is the S; blending toward it keeps the strength controllable.
    const f32 s = t * t * (3.f - 2.f * t);
    return t + (s - t) * amount;
}

// Weight peaking on skin hues (orange-red). Cheap stand-in for the segmentation
// a phone ISP has: it cannot find a face, but it can decline to push the colours
// faces are made of.
inline f32 skin_weight(f32 r, f32 g, f32 b) {
    const f32 mx = std::max(r, std::max(g, b));
    const f32 mn = std::min(r, std::min(g, b));
    const f32 c = mx - mn;
    if (c <= kEps || mx <= kEps) return 0.f;
    f32 hue;
    if (mx == r)      hue = 60.f * std::fmod(((g - b) / c) + 6.f, 6.f);
    else if (mx == g) hue = 60.f * (((b - r) / c) + 2.f);
    else              hue = 60.f * (((r - g) / c) + 4.f);
    // Centred on ~25 deg, tapering out by ~55. Saturation-limited so a saturated
    // orange object is not mistaken for skin.
    const f32 d = std::fabs(hue - 25.f) / 30.f;
    const f32 hw = clampf(1.f - d * d, 0.f, 1.f);
    const f32 sat = c / mx;
    const f32 sw = clampf((0.65f - sat) / 0.35f, 0.f, 1.f);
    return hw * sw;
}

}  // namespace

bool isp_analyse(const uint16_t* rgb16, int W, int H,
                 const float* cam_to_srgb, const IspParams& p, IspState& st) {
    st = IspState{};
    if (!rgb16 || W <= 0 || H <= 0) return false;
    st.W = W;
    st.H = H;
    st.p = p;
    for (int i = 0; i < 9; ++i)
        st.m[i] = cam_to_srgb ? cam_to_srgb[i] : kDefaultCamToSrgb[i];

    // Downsample so the short side keeps ~128 samples: enough for the base layer
    // to follow real scene structure, small enough that the whole analysis is a
    // few MB and a few milliseconds at 48MP.
    int shift = 3;
    while (shift > 0 && (std::min(W, H) >> shift) < 128) --shift;
    while ((std::min(W, H) >> shift) > 384 && shift < 6) ++shift;
    st.shift = shift;
    const int f = 1 << shift;
    const int gw = std::max(1, W / f), gh = std::max(1, H / f);
    st.gw = gw;
    st.gh = gh;

    // Box-average luminance into the low-res grid. Averaging rather than
    // point-sampling keeps the base layer free of the input's own noise.
    std::vector<f32> Y((size_t)gw * gh, 0.f);
    std::vector<f32> cnt((size_t)gw * gh, 0.f);
    for (int y = 0; y < H; ++y) {
        const int gy = std::min(gh - 1, y / f);
        const uint16_t* row = rgb16 + (size_t)y * W * 3;
        for (int x = 0; x < W; ++x) {
            const int gx = std::min(gw - 1, x / f);
            const f32 l = luma_of(row[x * 3 + 0] * (1.f / 65535.f),
                                  row[x * 3 + 1] * (1.f / 65535.f),
                                  row[x * 3 + 2] * (1.f / 65535.f));
            Y[(size_t)gy * gw + gx] += l;
            cnt[(size_t)gy * gw + gx] += 1.f;
        }
    }
    for (size_t i = 0; i < Y.size(); ++i) Y[i] /= std::max(cnt[i], 1.f);

    // Automatic exposure from the log-average, which tracks what the scene
    // actually contains rather than its extremes: a small bright window does not
    // drag the whole frame dark the way a max-based normalisation would.
    double logsum = 0.0;
    for (f32 v : Y) logsum += std::log((double)std::max(v, 1e-5f));
    const f32 log_avg = (f32)std::exp(logsum / (double)std::max<size_t>(Y.size(), 1));
    f32 exposure = kAutoKey / std::max(log_avg, 1e-5f);
    exposure *= std::pow(2.f, p.exposure_ev);
    exposure = clampf(exposure, 0.05f, 64.f);
    st.exposure = exposure;

    // Everything from here on is in exposed linear.
    for (f32& v : Y) v *= exposure;

    // White point for the output curve: the brightest content present, so the
    // shoulder rolls off to exactly that instead of clipping it or wasting range.
    std::vector<f32> sorted(Y);
    std::sort(sorted.begin(), sorted.end());
    const f32 p999 = sorted[(size_t)((sorted.size() - 1) * 0.999)];
    st.white = clampf(p999, 1.2f, 24.f);

    // Base layer in log2: tone mapping is a ratio operation, and doing it in log
    // makes the compression uniform across stops instead of biased toward the
    // highlights. It is also what stops the guided filter producing halos.
    std::vector<f32> L(Y.size());
    for (size_t i = 0; i < Y.size(); ++i)
        L[i] = std::log2(std::max(Y[i], 1e-5f));

    const int radius = std::max(2, std::min(gw, gh) / 16);
    // eps in log2 units squared: 0.5 stops of local variation counts as texture
    // to smooth over, more than that is an edge to preserve.
    std::vector<f32> Lb;
    guided_self(L, Lb, gw, gh, radius, 0.25f);

    // Compress the base around mid grey, asymmetrically. Above the pivot the
    // range is squeezed toward it (highlights come down), below it the same
    // squeeze lifts shadows up. Detail is untouched -- only the base moves.
    const f32 pivot = std::log2(kMidGrey);
    const f32 hi = clampf(1.f - 0.5f * p.highlight_rolloff, 0.15f, 1.f);
    const f32 lo = clampf(1.f - 0.5f * p.shadow_lift, 0.15f, 1.f);
    const f32 amt = clampf(p.local_strength, 0.f, 1.f);

    st.gain.resize(Y.size());
    st.base.resize(Y.size());
    for (size_t i = 0; i < Y.size(); ++i) {
        const f32 d = Lb[i] - pivot;
        const f32 dc = (d > 0.f) ? d * hi : d * lo;
        const f32 moved = d + amt * (dc - d);
        st.gain[i] = clampf(std::exp2(moved - d), 0.05f, 20.f);
        st.base[i] = std::exp2(Lb[i]);
    }
    st.valid = true;
    return true;
}

namespace {

// Bilinear sample of a low-res map at a full-res pixel, with the half-pixel
// offset the box downsample implies. Without it the map sits a fraction of a
// low-res cell off and the gain drifts against the image it was computed from.
inline f32 sample_map(const std::vector<f32>& map, int gw, int gh,
                      int x, int y, int shift) {
    const f32 fx = ((f32)x + 0.5f) / (f32)(1 << shift) - 0.5f;
    const f32 fy = ((f32)y + 0.5f) / (f32)(1 << shift) - 0.5f;
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

}  // namespace

void isp_render(const IspState& st, f32 r, f32 g, f32 b, int x, int y,
                f32& sr, f32& sg, f32& sb) {
    r = std::max(r, 0.f) * st.exposure;
    g = std::max(g, 0.f) * st.exposure;
    b = std::max(b, 0.f) * st.exposure;

    if (st.valid && st.p.local_strength > 0.f) {
        f32 gain = sample_map(st.gain, st.gw, st.gh, x, y, st.shift);
        if (st.p.local_contrast > 0.f) {
            // Y/base is the detail layer. Raising it above unity is local
            // micro-contrast: no high-pass, so no halo and no noise gained.
            const f32 base = sample_map(st.base, st.gw, st.gh, x, y, st.shift);
            const f32 ratio = luma_of(r, g, b) / std::max(base, kEps);
            gain *= std::pow(clampf(ratio, 0.05f, 20.f), st.p.local_contrast);
        }
        // One factor for all three channels: hue survives.
        r *= gain;
        g *= gain;
        b *= gain;
    }

    // Warmth trim, before the matrix so it acts on camera RGB.
    if (st.p.warmth != 0.f) r *= (1.f + st.p.warmth);

    // Camera linear -> sRGB linear. Negatives are real here (the matrix can take
    // a saturated channel below zero) and must be clipped before the curve, or
    // they come back as green speckle in deep shadow.
    f32 lr = st.m[0] * r + st.m[1] * g + st.m[2] * b;
    f32 lg = st.m[3] * r + st.m[4] * g + st.m[5] * b;
    f32 lb = st.m[6] * r + st.m[7] * g + st.m[8] * b;
    lr = std::max(lr, 0.f);
    lg = std::max(lg, 0.f);
    lb = std::max(lb, 0.f);

    // Output curve on each channel, then gamma. The curve is monotone and shared
    // across channels, so neutrals stay neutral.
    lr = tone_curve(lr, st.white);
    lg = tone_curve(lg, st.white);
    lb = tone_curve(lb, st.white);

    sr = srgb_oetf(lr);
    sg = srgb_oetf(lg);
    sb = srgb_oetf(lb);

    // Black point. Without it the shadow lift leaves the darkest pixels grey,
    // and the image reads flat however much contrast is applied afterwards.
    if (st.p.black_point > 0.f) {
        const f32 bp = clampf(st.p.black_point, 0.f, 0.5f);
        const f32 inv = 1.f / (1.f - bp);
        sr = clampf((sr - bp) * inv, 0.f, 1.f);
        sg = clampf((sg - bp) * inv, 0.f, 1.f);
        sb = clampf((sb - bp) * inv, 0.f, 1.f);
    }

    // Contrast, in display space, via a luminance ratio so hue is preserved.
    if (st.p.contrast > 0.f) {
        const f32 yl = luma_of(sr, sg, sb);
        if (yl > kEps) {
            const f32 k = s_curve(yl, st.p.contrast) / yl;
            sr = clampf(sr * k, 0.f, 1.f);
            sg = clampf(sg * k, 0.f, 1.f);
            sb = clampf(sb * k, 0.f, 1.f);
        }
    }

    // Vibrance: the boost falls away as a colour approaches saturation, so muted
    // material lifts and already-vivid material barely moves. Skin is held back
    // further, which is what stops strong tone mapping turning faces orange.
    const f32 yl = luma_of(sr, sg, sb);
    f32 amount = st.p.saturation - 1.f;
    if (st.p.vibrance > 0.f) {
        const f32 mx = std::max(sr, std::max(sg, sb));
        const f32 mn = std::min(sr, std::min(sg, sb));
        const f32 sat = (mx > kEps) ? (mx - mn) / mx : 0.f;
        const f32 head = (1.f - sat) * (1.f - sat);
        amount += st.p.vibrance * head;
    }
    if (st.p.skin_protect) {
        const f32 w = skin_weight(sr, sg, sb);
        amount *= (1.f - 0.75f * w);
    }
    if (std::fabs(amount) > 1e-4f) {
        const f32 k = 1.f + amount;
        sr = yl + (sr - yl) * k;
        sg = yl + (sg - yl) * k;
        sb = yl + (sb - yl) * k;
    }

    // Gamut: pull an out-of-range colour toward its own luminance rather than
    // clipping each channel, which would shift hue and flatten bright saturated
    // areas into flat patches.
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

}  // namespace hhsr
