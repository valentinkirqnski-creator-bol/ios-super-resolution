#include "HdrPlusPyFinish.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hhsr {
namespace {

inline float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

// finishing.py gammasRGB (float path).
inline float gamma_compress(float x) {
    if (x <= 0.0031308f) return clamp01(12.92f * x);
    return clamp01(1.055f * std::pow(std::max(x, 0.f), 1.f / 2.4f) - 0.055f);
}
inline float gamma_decompress(float x) {
    if (x <= 0.04045f) return clamp01(x / 12.92f);
    return clamp01(std::pow((x + 0.055f) / 1.055f, 2.4f));
}

// --- small single-channel plane for the reduced-res tone map ---
struct Plane {
    int h = 0, w = 0;
    std::vector<float> d;
    Plane() = default;
    Plane(int h_, int w_) : h(h_), w(w_), d((size_t)h_ * w_, 0.f) {}
    float& at(int y, int x) { return d[(size_t)y * w + x]; }
    float  at(int y, int x) const { return d[(size_t)y * w + x]; }
};

// Separable [1,4,6,4,1]/16, reflect borders, same-size (mirrors py _sepconv).
Plane sepconv(const Plane& im) {
    static const float K[5] = {1.f/16, 4.f/16, 6.f/16, 4.f/16, 1.f/16};
    const int h = im.h, w = im.w;
    auto ry = [&](int y) { y = y < 0 ? -y : (y >= h ? 2*h - 2 - y : y); return std::min(std::max(y,0),h-1); };
    auto rx = [&](int x) { x = x < 0 ? -x : (x >= w ? 2*w - 2 - x : x); return std::min(std::max(x,0),w-1); };
    Plane t(h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0.f;
            for (int k = -2; k <= 2; ++k) a += K[k+2] * im.at(y, rx(x+k));
            t.at(y, x) = a;
        }
    Plane o(h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0.f;
            for (int k = -2; k <= 2; ++k) a += K[k+2] * t.at(ry(y+k), x);
            o.at(y, x) = a;
        }
    return o;
}
Plane pyr_down(const Plane& im) {
    Plane s = sepconv(im);
    Plane o((im.h + 1) / 2, (im.w + 1) / 2);
    for (int y = 0; y < o.h; ++y)
        for (int x = 0; x < o.w; ++x) o.at(y, x) = s.at(y * 2, x * 2);
    return o;
}
Plane pyr_up(const Plane& im, int H, int W) {
    Plane up(H, W);
    for (int y = 0; y < (H + 1) / 2 && y < im.h; ++y)
        for (int x = 0; x < (W + 1) / 2 && x < im.w; ++x) up.at(y * 2, x * 2) = im.at(y, x);
    Plane s = sepconv(up);
    for (float& v : s.d) v *= 4.f;
    return s;
}

// Mertens exposure fusion of two grayscale [0,1] images, exposure weight only
// (contrast/saturation weights are 0 in finishing.py), sigma 0.2.
Plane mertens_fuse(const Plane& a, const Plane& b) {
    const int H = a.h, W = a.w;
    Plane wa(H, W), wb(H, W);
    for (size_t i = 0; i < a.d.size(); ++i) {
        const float ta = (a.d[i] - 0.5f) / 0.2f, tb = (b.d[i] - 0.5f) / 0.2f;
        float ea = std::exp(-0.5f * ta * ta) + 1e-12f;
        float eb = std::exp(-0.5f * tb * tb) + 1e-12f;
        const float s = ea + eb;
        wa.d[i] = ea / s; wb.d[i] = eb / s;
    }
    int nlev = (int)std::floor(std::log2((double)std::min(H, W)));
    if (nlev < 1) nlev = 1;
    auto build = [&](const Plane& img, const Plane& wt, std::vector<Plane>& res) {
        std::vector<Plane> gp; gp.reserve(nlev + 1); gp.push_back(img);
        for (int l = 0; l < nlev; ++l) gp.push_back(pyr_down(gp[l]));
        std::vector<Plane> gw; gw.reserve(nlev + 1); gw.push_back(wt);
        for (int l = 0; l < nlev; ++l) gw.push_back(pyr_down(gw[l]));
        for (int l = 0; l < nlev; ++l) {
            Plane up = pyr_up(gp[l+1], gp[l].h, gp[l].w);   // laplacian = g[l]-up(g[l+1])
            Plane lap(gp[l].h, gp[l].w);
            for (size_t i = 0; i < lap.d.size(); ++i) lap.d[i] = gp[l].d[i] - up.d[i];
            if ((int)res.size() <= l) res.emplace_back(lap.h, lap.w);
            for (size_t i = 0; i < lap.d.size(); ++i) res[l].d[i] += lap.d[i] * gw[l].d[i];
        }
        if ((int)res.size() <= nlev) res.emplace_back(gp[nlev].h, gp[nlev].w);
        for (size_t i = 0; i < gp[nlev].d.size(); ++i)
            res[nlev].d[i] += gp[nlev].d[i] * gw[nlev].d[i];
    };
    std::vector<Plane> res;
    build(a, wa, res);
    build(b, wb, res);
    Plane out = res[nlev];
    for (int l = nlev - 1; l >= 0; --l) {
        Plane up = pyr_up(out, res[l].h, res[l].w);
        out = res[l];
        for (size_t i = 0; i < out.d.size(); ++i) out.d[i] += up.d[i];
    }
    return out;
}

// finishing.py auto-gain loop on a 1/25-downsampled grayscale (flattened).
int auto_gain(const Plane& shortGray) {
    const int ds = 25;
    const int hs = std::max(1, shortGray.h / ds), ws = std::max(1, shortGray.w / ds);
    std::vector<float> s((size_t)hs * ws, 0.f);
    for (int y = 0; y < hs; ++y)
        for (int x = 0; x < ws; ++x) {
            float acc = 0.f; int n = 0;
            for (int dy = 0; dy < ds; ++dy)
                for (int dx = 0; dx < ds; ++dx) {
                    const int sy = y*ds+dy, sx = x*ds+dx;
                    if (sy < shortGray.h && sx < shortGray.w) { acc += shortGray.at(sy, sx); ++n; }
                }
            s[(size_t)y*ws+x] = n ? acc / n : 0.f;
        }
    double sSMean = 0.0; for (float v : s) sSMean += gamma_compress(v); sSMean /= s.size();
    int gain = 0; double compression = 1.0, saturated = 0.0; bool bestGain = false;
    while ((compression < 1.9 && saturated < 0.95) ||
           (!bestGain && compression < 6 && gain < 30 && saturated < 0.33)) {
        gain += 2;
        double lSMean = 0.0; size_t sat = 0;
        for (float v : s) {
            const float lg = clamp01(gamma_compress((float)gain * v));
            lSMean += lg; if (lg > 0.95f) ++sat;
        }
        lSMean /= s.size();
        compression = lSMean / std::max(sSMean, 1e-9);
        bestGain = lSMean > (1.0 - sSMean) / 2.0;
        saturated = (double)sat / s.size();
    }
    return gain;
}

// Horizontal Gaussian on an interleaved RGB band (reflect columns).
void gauss_h(const std::vector<float>& src, std::vector<float>& dst, int rows, int W,
             const std::vector<float>& k, int r) {
    dst.assign(src.size(), 0.f);
    auto rx = [&](int x){ x = x < 0 ? -x : (x >= W ? 2*W-2-x : x); return std::min(std::max(x,0),W-1); };
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c) {
                float a = 0.f;
                for (int t = -r; t <= r; ++t) a += k[t+r] * src[((size_t)y*W + rx(x+t))*3 + c];
                dst[((size_t)y*W + x)*3 + c] = a;
            }
}
// Vertical Gaussian (reflect rows within the band's valid range).
void gauss_v(const std::vector<float>& src, std::vector<float>& dst, int rows, int W,
             const std::vector<float>& k, int r) {
    dst.assign(src.size(), 0.f);
    auto ry = [&](int y){ y = y < 0 ? -y : (y >= rows ? 2*rows-2-y : y); return std::min(std::max(y,0),rows-1); };
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c) {
                float a = 0.f;
                for (int t = -r; t <= r; ++t) a += k[t+r] * src[((size_t)ry(y+t)*W + x)*3 + c];
                dst[((size_t)y*W + x)*3 + c] = a;
            }
}

}  // namespace

void hdrplus_py_finish(int W, int H, const HdrPlusPyParams& p,
                       const LinearBandReader& read_band,
                       const U8BandWriter& write_band,
                       int band_rows) {
    if (W <= 0 || H <= 0) return;
    if (band_rows <= 0) band_rows = 128;
    const int ds = std::max(1, p.ltm_downsample);
    const int sw = (W + ds - 1) / ds, sh = (H + ds - 1) / ds;

    auto to_srgb = [&](const float* in, float& r, float& g, float& b) {
        const float cr = in[0] * p.wb[0], cg = in[1] * p.wb[1], cb = in[2] * p.wb[2];
        r = clamp01(p.ccm[0]*cr + p.ccm[1]*cg + p.ccm[2]*cb);
        g = clamp01(p.ccm[3]*cr + p.ccm[4]*cg + p.ccm[5]*cb);
        b = clamp01(p.ccm[6]*cr + p.ccm[7]*cg + p.ccm[8]*cb);
    };

    // --- Pass 1: build the reduced-res linear-sRGB image (box-averaged) ---
    Plane rS(sh, sw), gS(sh, sw), bS(sh, sw);
    std::vector<int> cnt((size_t)sh * sw, 0);
    {
        std::vector<float> band;
        for (int y0 = 0; y0 < H; y0 += band_rows) {
            const int bh = std::min(band_rows, H - y0);
            band.assign((size_t)bh * W * 3, 0.f);
            read_band(y0, bh, band.data());
            for (int yy = 0; yy < bh; ++yy) {
                const int sy = (y0 + yy) / ds;
                for (int x = 0; x < W; ++x) {
                    float r, g, b; to_srgb(&band[((size_t)yy*W + x)*3], r, g, b);
                    const int sx = x / ds; const size_t si = (size_t)sy*sw + sx;
                    rS.d[si] += r; gS.d[si] += g; bS.d[si] += b; cnt[si] += 1;
                }
            }
        }
        for (size_t i = 0; i < cnt.size(); ++i) {
            const float inv = cnt[i] ? 1.f / cnt[i] : 0.f;
            rS.d[i] *= inv; gS.d[i] *= inv; bS.d[i] *= inv;
        }
    }

    // --- Tone map on the reduced-res grayscale -> scaling map s_small ---
    Plane shortGray(sh, sw), longGray(sh, sw);
    for (size_t i = 0; i < shortGray.d.size(); ++i)
        shortGray.d[i] = (rS.d[i] + gS.d[i] + bS.d[i]) / 3.f;
    const int gain = auto_gain(shortGray);
    for (size_t i = 0; i < longGray.d.size(); ++i) {
        const float rk = clamp01(rS.d[i]*gain), gk = clamp01(gS.d[i]*gain), bk = clamp01(bS.d[i]*gain);
        longGray.d[i] = (rk + gk + bk) / 3.f;
    }
    Plane shortg(sh, sw), longg(sh, sw);
    for (size_t i = 0; i < shortg.d.size(); ++i) {
        shortg.d[i] = gamma_compress(shortGray.d[i]);
        longg.d[i]  = gamma_compress(longGray.d[i]);
    }
    Plane fusedg = mertens_fuse(shortg, longg);
    Plane sMap(sh, sw);   // fusedGray / shortGray
    for (size_t i = 0; i < sMap.d.size(); ++i) {
        const float fg = gamma_decompress(clamp01(fusedg.d[i]));
        sMap.d[i] = (shortGray.d[i] <= 0.f) ? 1.f : fg / std::max(shortGray.d[i], 1e-9f);
    }

    // Bilinear sample of the reduced scaling map at full-res (x,y).
    auto sample_s = [&](int x, int y) -> float {
        const float fx = ((float)x + 0.5f) / ds - 0.5f, fy = ((float)y + 0.5f) / ds - 0.5f;
        int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
        const float tx = fx - x0, ty = fy - y0;
        auto cl = [](int v, int hi){ return v < 0 ? 0 : (v > hi ? hi : v); };
        const int xa = cl(x0, sw-1), xb = cl(x0+1, sw-1), ya = cl(y0, sh-1), yb = cl(y0+1, sh-1);
        const float v00 = sMap.at(ya,xa), v10 = sMap.at(ya,xb), v01 = sMap.at(yb,xa), v11 = sMap.at(yb,xb);
        return (v00*(1-tx)+v10*tx)*(1-ty) + (v01*(1-tx)+v11*tx)*ty;
    };

    // --- Pass 2: full-res, banded. scaling -> contrast -> gamma -> sharpen -> 8bit ---
    // Gaussian kernels for the three sharpen sigmas.
    int maxr = 0;
    std::vector<std::vector<float>> kern(3); std::vector<int> rad(3);
    for (int s = 0; s < 3; ++s) {
        const float sig = p.sharpen_sigma[s];
        const int r = std::max(1, (int)std::lround(sig * 3.f));
        rad[s] = r; maxr = std::max(maxr, r);
        kern[s].resize(2*r+1); float sum = 0.f;
        for (int t = -r; t <= r; ++t) { float v = std::exp(-0.5f*(t/sig)*(t/sig)); kern[s][t+r]=v; sum+=v; }
        for (float& v : kern[s]) v /= sum;
    }
    const int halo = maxr;
    const float twopi = 6.28318530717958648f;

    std::vector<float> ext, gimg, tmp, blur, acc;
    std::vector<uint8_t> out8;
    for (int y0 = 0; y0 < H; y0 += band_rows) {
        const int bh = std::min(band_rows, H - y0);
        const int ey0 = y0 - halo, erows = bh + 2 * halo;   // extended (haloed) band
        ext.assign((size_t)erows * W * 3, 0.f);
        read_band(ey0, erows, ext.data());   // reader clamps out-of-range rows
        // per-pixel: scaling -> contrast -> gamma, on the whole extended band
        gimg.assign((size_t)erows * W * 3, 0.f);
        for (int yy = 0; yy < erows; ++yy) {
            const int gy = std::min(std::max(ey0 + yy, 0), H - 1);
            for (int x = 0; x < W; ++x) {
                float r, g, b; to_srgb(&ext[((size_t)yy*W + x)*3], r, g, b);
                const float s = sample_s(x, gy);
                r = clamp01(r*s); g = clamp01(g*s); b = clamp01(b*s);        // applyScaling
                r = clamp01(r - p.gtm_contrast*std::sin(twopi*r));            // enhanceContrast
                g = clamp01(g - p.gtm_contrast*std::sin(twopi*g));
                b = clamp01(b - p.gtm_contrast*std::sin(twopi*b));
                gimg[((size_t)yy*W + x)*3 + 0] = gamma_compress(r);          // sRGB gamma
                gimg[((size_t)yy*W + x)*3 + 1] = gamma_compress(g);
                gimg[((size_t)yy*W + x)*3 + 2] = gamma_compress(b);
            }
        }
        // sharpenTriple: average of the three thresholded unsharp masks.
        acc.assign((size_t)erows * W * 3, 0.f);
        for (int s = 0; s < 3; ++s) {
            gauss_h(gimg, tmp, erows, W, kern[s], rad[s]);
            gauss_v(tmp, blur, erows, W, kern[s], rad[s]);
            const float amt = p.sharpen_amount[s], th = p.sharpen_threshold[s];
            for (size_t i = 0; i < acc.size(); ++i) {
                const float x = gimg[i], bl = blur[i];
                const float low = std::fabs(bl - x);
                acc[i] += (low < th) ? x : (x + amt * (x - bl));
            }
        }
        // core rows [halo, halo+bh) -> 8-bit
        out8.assign((size_t)bh * W * 3, 0);
        for (int yy = 0; yy < bh; ++yy) {
            const size_t src = (size_t)(yy + halo) * W * 3, dst = (size_t)yy * W * 3;
            for (int i = 0; i < W * 3; ++i) {
                const float v = clamp01(acc[src + i] / 3.f);
                out8[dst + i] = (uint8_t)std::min(255.f, v * 255.f + 0.5f);
            }
        }
        write_band(y0, bh, out8.data());
    }
}

}  // namespace hhsr
