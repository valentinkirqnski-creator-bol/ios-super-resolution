#include "render_hdrplus.h"
#include "parallel.h"

#include <cmath>
#include <algorithm>
#include <vector>

namespace hhsr {

// ---------------------------------------------------------------- transfer

// fGammaCompress_(x, 0.0031308, 12.92, 1.055, 1/2.4). gainMax*(x^e) - gainMax
// + 1 == 1.055*x^(1/2.4) - 0.055, i.e. the standard sRGB OETF; kept in the
// reference's own form so the constants line up with finishing.py.
f32 hdrplus_gamma_compress(f32 x) {
    x = (x <= 0.0031308f) ? (12.92f * x)
                          : (1.055f * std::pow(x, 1.f / 2.4f) - 1.055f + 1.f);
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

// fGammaDecompress_(x, 0.04045, 12.92, 1.055, 2.4)
f32 hdrplus_gamma_decompress(f32 x) {
    x = (x <= 0.04045f) ? (x / 12.92f)
                        : std::pow((x + 1.055f - 1.f) / 1.055f, 2.4f);
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

// ------------------------------------------------------------------ resize

// Separable box-average downscale. Used for the gain search's dsFactor 25
// decimation (cv2.resize default INTER_LINEAR on a large downscale behaves as
// an area average) and for the fusion downscale.
static void resize_area(const std::vector<f32>& src, int sh, int sw,
                        std::vector<f32>& dst, int dh, int dw) {
    dst.assign((size_t)dh * dw, 0.f);
    if (sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0) return;
    const f32 ry = (f32)sh / (f32)dh, rx = (f32)sw / (f32)dw;
    for (int y = 0; y < dh; ++y) {
        const int y0 = std::min(sh - 1, (int)(y * ry));
        const int y1 = std::min(sh, std::max(y0 + 1, (int)((y + 1) * ry)));
        for (int x = 0; x < dw; ++x) {
            const int x0 = std::min(sw - 1, (int)(x * rx));
            const int x1 = std::min(sw, std::max(x0 + 1, (int)((x + 1) * rx)));
            double acc = 0.0; int n = 0;
            for (int i = y0; i < y1; ++i)
                for (int j = x0; j < x1; ++j) { acc += src[(size_t)i * sw + j]; ++n; }
            dst[(size_t)y * dw + x] = (n > 0) ? (f32)(acc / n) : 0.f;
        }
    }
}

// Bilinear upscale, used to lift the fusion's ratio map back to full size.
static void resize_bilinear(const std::vector<f32>& src, int sh, int sw,
                            std::vector<f32>& dst, int dh, int dw,
                            int num_threads) {
    dst.assign((size_t)dh * dw, 0.f);
    if (sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0) return;
    if (sh == dh && sw == dw) { dst = src; return; }
    const f32 sy = (f32)sh / (f32)dh, sx = (f32)sw / (f32)dw;
    parallel_rows(dh, num_threads, [&](int y) {
        const f32 fy = std::min((f32)sh - 1.f, std::max(0.f, ((f32)y + 0.5f) * sy - 0.5f));
        const int y0 = (int)fy, y1 = std::min(sh - 1, y0 + 1);
        const f32 ay = fy - (f32)y0;
        for (int x = 0; x < dw; ++x) {
            const f32 fx = std::min((f32)sw - 1.f, std::max(0.f, ((f32)x + 0.5f) * sx - 0.5f));
            const int x0 = (int)fx, x1 = std::min(sw - 1, x0 + 1);
            const f32 ax = fx - (f32)x0;
            const f32 t = src[(size_t)y0 * sw + x0] +
                          (src[(size_t)y0 * sw + x1] - src[(size_t)y0 * sw + x0]) * ax;
            const f32 b = src[(size_t)y1 * sw + x0] +
                          (src[(size_t)y1 * sw + x1] - src[(size_t)y1 * sw + x0]) * ax;
            dst[(size_t)y * dw + x] = t + (b - t) * ay;
        }
    });
}

// ----------------------------------------------------------------- pyramid

// OpenCV's pyrDown/pyrUp: 5-tap binomial [1,4,6,4,1]/16, separable, with
// BORDER_REFLECT_101 (the border pixel is not repeated). MergeMertens is built
// on buildPyramid/pyrUp, so matching these is what makes the fusion match.
static inline int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

static void pyr_down(const std::vector<f32>& src, int sh, int sw,
                     std::vector<f32>& dst, int& dh, int& dw, int num_threads) {
    dh = (sh + 1) / 2; dw = (sw + 1) / 2;
    static const f32 k[5] = {1.f / 16.f, 4.f / 16.f, 6.f / 16.f, 4.f / 16.f, 1.f / 16.f};
    std::vector<f32> tmp((size_t)sh * dw, 0.f);
    parallel_rows(sh, num_threads, [&](int y) {              // horizontal + decimate
        for (int x = 0; x < dw; ++x) {
            f32 a = 0.f;
            for (int t = -2; t <= 2; ++t)
                a += k[t + 2] * src[(size_t)y * sw + reflect101(2 * x + t, sw)];
            tmp[(size_t)y * dw + x] = a;
        }
    });
    dst.assign((size_t)dh * dw, 0.f);
    parallel_rows(dh, num_threads, [&](int y) {              // vertical + decimate
        for (int x = 0; x < dw; ++x) {
            f32 a = 0.f;
            for (int t = -2; t <= 2; ++t)
                a += k[t + 2] * tmp[(size_t)reflect101(2 * y + t, sh) * dw + x];
            dst[(size_t)y * dw + x] = a;
        }
    });
}

// Upsample to (dh,dw): insert zeros, convolve with the same kernel scaled by 4.
static void pyr_up(const std::vector<f32>& src, int sh, int sw,
                   std::vector<f32>& dst, int dh, int dw, int num_threads) {
    static const f32 k[5] = {1.f / 16.f, 4.f / 16.f, 6.f / 16.f, 4.f / 16.f, 1.f / 16.f};
    std::vector<f32> tmp((size_t)dh * sw, 0.f);
    parallel_rows(dh, num_threads, [&](int y) {              // vertical
        for (int x = 0; x < sw; ++x) {
            f32 a = 0.f;
            for (int t = -2; t <= 2; ++t) {
                const int yy = y + t;
                if ((yy & 1) != 0) continue;                 // zero-inserted row
                a += k[t + 2] * src[(size_t)reflect101(yy / 2, sh) * sw + x];
            }
            tmp[(size_t)y * sw + x] = 2.f * a;
        }
    });
    dst.assign((size_t)dh * dw, 0.f);
    parallel_rows(dh, num_threads, [&](int y) {              // horizontal
        for (int x = 0; x < dw; ++x) {
            f32 a = 0.f;
            for (int t = -2; t <= 2; ++t) {
                const int xx = x + t;
                if ((xx & 1) != 0) continue;
                a += k[t + 2] * tmp[(size_t)y * sw + reflect101(xx / 2, sw)];
            }
            dst[(size_t)y * dw + x] = 2.f * a;
        }
    });
}

// ------------------------------------------------------ Mertens fusion (2)

// cv2.createMergeMertens(contrast_weight=0, saturation_weight=0,
// exposure_weight=1) over two single-channel images. With the contrast and
// saturation exponents at zero their factors are pow(x, 0) == 1 regardless of
// value, so the weight reduces to well-exposedness alone:
//     w = exp(-(x - 0.5)^2 / (2 * 0.2^2))
// normalised across the two images, then a Laplacian-pyramid blend.
static void mertens_fuse2(const std::vector<f32>& a, const std::vector<f32>& b,
                          int h, int w, std::vector<f32>& out, int num_threads) {
    const size_t n = (size_t)h * w;
    std::vector<f32> wa(n), wb(n);
    parallel_rows(h, num_threads, [&](int y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = (size_t)y * w + x;
            const f32 da = a[i] - 0.5f, db = b[i] - 0.5f;
            const f32 ea = std::exp(-(da * da) / 0.08f);      // 2 * 0.2^2
            const f32 eb = std::exp(-(db * db) / 0.08f);
            const f32 s = ea + eb + 1e-12f;                   // OpenCV's epsilon
            wa[i] = ea / s;
            wb[i] = eb / s;
        }
    });

    int maxlevel = (int)(std::log((double)std::min(h, w)) / std::log(2.0));
    if (maxlevel < 1) maxlevel = 1;

    // Walk down one level at a time, accumulating the blended Laplacian into
    // res[lvl] as we go, so only the current level of each input is resident
    // rather than five full pyramids.
    std::vector<std::vector<f32>> res(maxlevel + 1);
    std::vector<int> rh(maxlevel + 1), rw(maxlevel + 1);

    std::vector<f32> ca = a, cb = b, cwa = wa, cwb = wb;
    int ch = h, cw = w;
    for (int lvl = 0; lvl < maxlevel; ++lvl) {
        std::vector<f32> na, nb, nwa, nwb, ua, ub;
        int nh = 0, nw = 0, t1, t2;
        pyr_down(ca, ch, cw, na, nh, nw, num_threads);
        pyr_down(cb, ch, cw, nb, t1, t2, num_threads);
        pyr_down(cwa, ch, cw, nwa, t1, t2, num_threads);
        pyr_down(cwb, ch, cw, nwb, t1, t2, num_threads);
        pyr_up(na, nh, nw, ua, ch, cw, num_threads);
        pyr_up(nb, nh, nw, ub, ch, cw, num_threads);

        res[lvl].assign((size_t)ch * cw, 0.f);
        const std::vector<f32>& la = ca; const std::vector<f32>& lb = cb;
        parallel_rows(ch, num_threads, [&](int y) {
            for (int x = 0; x < cw; ++x) {
                const size_t i = (size_t)y * cw + x;
                res[lvl][i] = (la[i] - ua[i]) * cwa[i] + (lb[i] - ub[i]) * cwb[i];
            }
        });
        rh[lvl] = ch; rw[lvl] = cw;
        ca.swap(na); cb.swap(nb); cwa.swap(nwa); cwb.swap(nwb);
        ch = nh; cw = nw;
    }
    // Coarsest level: the residual itself, weighted.
    res[maxlevel].assign((size_t)ch * cw, 0.f);
    for (size_t i = 0; i < (size_t)ch * cw; ++i)
        res[maxlevel][i] = ca[i] * cwa[i] + cb[i] * cwb[i];
    rh[maxlevel] = ch; rw[maxlevel] = cw;

    // Collapse.
    for (int lvl = maxlevel - 1; lvl >= 0; --lvl) {
        std::vector<f32> up;
        pyr_up(res[lvl + 1], rh[lvl + 1], rw[lvl + 1], up, rh[lvl], rw[lvl], num_threads);
        const size_t m = (size_t)rh[lvl] * rw[lvl];
        for (size_t i = 0; i < m; ++i) res[lvl][i] += up[i];
        res[lvl + 1].clear(); res[lvl + 1].shrink_to_fit();
    }
    out.swap(res[0]);
}

// -------------------------------------------------------------- gain search

f32 hdrplus_auto_gain(const std::vector<f32>& grey, int h, int w) {
    // localToneMap: dsFactor 25, then iterate gain by 2 until the compression
    // ratio and saturation conditions are satisfied.
    const int dh = std::max(1, h / 25), dw = std::max(1, w / 25);
    std::vector<f32> s;
    resize_area(grey, h, w, s, dh, dw);
    const size_t n = s.size();
    if (n == 0) return 1.f;

    double sSMean = 0.0;
    for (size_t i = 0; i < n; ++i) sSMean += hdrplus_gamma_compress(s[i]);
    sSMean /= (double)n;

    f32 gain = 0.f;
    double compression = 1.0, saturated = 0.0;
    bool bestGain = false;
    while ((compression < 1.9 && saturated < 0.95) ||
           (!bestGain && compression < 6.0 && gain < 30.f && saturated < 0.33)) {
        gain += 2.f;
        double lSMean = 0.0; size_t sat = 0;
        for (size_t i = 0; i < n; ++i) {
            f32 v = hdrplus_gamma_compress(gain * s[i]);
            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);          // their .clip(0,1)
            lSMean += v;
            if (v > 0.95f) ++sat;
        }
        lSMean /= (double)n;
        compression = lSMean / (sSMean + 1e-12);
        bestGain = lSMean > (1.0 - sSMean) / 2.0;
        saturated = (double)sat / (double)n;
    }
    return gain;
}

// ------------------------------------------------------------------ finish

void hdrplus_finish(Image& rgb, const HdrPlusFinishParams& p) {
    if (rgb.h <= 0 || rgb.w <= 0 || rgb.c < 3) return;
    const int h = rgb.h, w = rgb.w;
    const size_t n = (size_t)h * w;
    const int nt = 0; // parallel_rows picks the default pool

    if (p.enable_ltm) {
        // shortGray = mean_(r,g,b)
        std::vector<f32> shortGray(n);
        parallel_rows(h, nt, [&](int y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = (size_t)y * w + x;
                const f32* px = &rgb.data[i * rgb.c];
                shortGray[i] = (px[0] + px[1] + px[2]) / 3.f;
            }
        });

        const f32 gain = (p.ltm_gain > 0.f) ? p.ltm_gain
                                            : hdrplus_auto_gain(shortGray, h, w);

        // Fusion resolution -- see the header. The ratio this produces is
        // smooth because both inputs come from shortGray.
        int fh = h, fw = w;
        if (p.fusion_max_dim > 0 && std::max(h, w) > p.fusion_max_dim) {
            const f32 s = (f32)p.fusion_max_dim / (f32)std::max(h, w);
            fh = std::max(2, (int)std::lround(h * s));
            fw = std::max(2, (int)std::lround(w * s));
        }

        // longGray = meanGain_: gain applied per channel, each CLIPPED to
        // [0,1] before averaging. Not the same as gain * shortGray, and the
        // difference is exactly what stops a blown channel dragging the
        // synthetic long exposure up.
        std::vector<f32> longGray(n);
        parallel_rows(h, nt, [&](int y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = (size_t)y * w + x;
                const f32* px = &rgb.data[i * rgb.c];
                f32 rk = px[0] * gain, gk = px[1] * gain, bk = px[2] * gain;
                rk = rk < 0.f ? 0.f : (rk > 1.f ? 1.f : rk);
                gk = gk < 0.f ? 0.f : (gk > 1.f ? 1.f : gk);
                bk = bk < 0.f ? 0.f : (bk > 1.f ? 1.f : bk);
                longGray[i] = (rk + gk + bk) / 3.f;
            }
        });

        std::vector<f32> sg, lg;
        if (fh == h && fw == w) { sg = shortGray; lg = longGray; }
        else {
            resize_area(shortGray, h, w, sg, fh, fw);
            resize_area(longGray, h, w, lg, fh, fw);
        }
        // Gamma before fusion, as localToneMap does.
        for (auto& v : sg) v = hdrplus_gamma_compress(v);
        for (auto& v : lg) v = hdrplus_gamma_compress(v);

        std::vector<f32> fused;
        mertens_fuse2(sg, lg, fh, fw, fused, nt);

        // Undo gamma, then the per-pixel ratio fusedGray / shortGray.
        for (auto& v : fused) v = hdrplus_gamma_decompress(v);

        std::vector<f32> ratio((size_t)fh * fw);
        {
            std::vector<f32> slin;
            if (fh == h && fw == w) slin = shortGray;
            else resize_area(shortGray, h, w, slin, fh, fw);
            for (size_t i = 0; i < ratio.size(); ++i)
                ratio[i] = (slin[i] == 0.f) ? 1.f : (fused[i] / slin[i]);
        }
        std::vector<f32> ratio_full;
        resize_bilinear(ratio, fh, fw, ratio_full, h, w, nt);

        // applyScaling_: scale each channel, clip each independently. No
        // gamut step -- see the header.
        parallel_rows(h, nt, [&](int y) {
            for (int x = 0; x < w; ++x) {
                const size_t i = (size_t)y * w + x;
                const f32 s = ratio_full[i];
                f32* px = &rgb.data[i * rgb.c];
                for (int c = 0; c < 3; ++c) {
                    const f32 v = px[c] * s;
                    px[c] = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                }
            }
        });
    }

    // enhanceContrast, in linear, before gamma -- as the reference orders it.
    if (p.enable_gtm && p.gtm_contrast > 0.f) {
        const f32 g = p.gtm_contrast;
        parallel_rows(h, nt, [&](int y) {
            for (int x = 0; x < w; ++x) {
                f32* px = &rgb.data[((size_t)y * w + x) * rgb.c];
                for (int c = 0; c < 3; ++c) {
                    f32 v = px[c] - g * std::sin(6.28318530718f * px[c]);
                    px[c] = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                }
            }
        });
    }

    parallel_rows(h, nt, [&](int y) {
        for (int x = 0; x < w; ++x) {
            f32* px = &rgb.data[((size_t)y * w + x) * rgb.c];
            for (int c = 0; c < 3; ++c) px[c] = hdrplus_gamma_compress(px[c]);
        }
    });

    if (!p.enable_sharpen) return;

    // sharpenTriple. Three Gaussian blurs of the (already gamma-encoded)
    // image; per scale, a pixel is sharpened only where |blur - x| reaches
    // that scale's threshold; the three candidates are averaged.
    std::vector<std::vector<f32>> blur(3);
    for (int s = 0; s < 3; ++s) {
        const f32 sigma = p.sharpen_sigma[s];
        int r = (int)std::ceil(3.f * sigma);
        if (r < 1) r = 1;
        std::vector<f32> k(2 * r + 1);
        f32 ksum = 0.f;
        for (int t = -r; t <= r; ++t) {
            k[t + r] = std::exp(-(f32)(t * t) / (2.f * sigma * sigma));
            ksum += k[t + r];
        }
        for (auto& v : k) v /= ksum;

        blur[s].assign(n * 3, 0.f);
        std::vector<f32> tmp(n * 3, 0.f);
        parallel_rows(h, nt, [&](int y) {                    // horizontal
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) {
                    f32 a = 0.f;
                    for (int t = -r; t <= r; ++t)
                        a += k[t + r] * rgb.data[((size_t)y * w + reflect101(x + t, w)) * rgb.c + c];
                    tmp[((size_t)y * w + x) * 3 + c] = a;
                }
        });
        parallel_rows(h, nt, [&](int y) {                    // vertical
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) {
                    f32 a = 0.f;
                    for (int t = -r; t <= r; ++t)
                        a += k[t + r] * tmp[((size_t)reflect101(y + t, h) * w + x) * 3 + c];
                    blur[s][((size_t)y * w + x) * 3 + c] = a;
                }
        });
    }

    parallel_rows(h, nt, [&](int y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = (size_t)y * w + x;
            f32* px = &rgb.data[i * rgb.c];
            for (int c = 0; c < 3; ++c) {
                const f32 v = px[c];
                f32 acc = 0.f;
                for (int s = 0; s < 3; ++s) {
                    const f32 b = blur[s][i * 3 + c];
                    const f32 low = std::fabs(b - v);        // distL1_
                    acc += (low < p.sharpen_threshold[s])
                               ? v
                               : (v + p.sharpen_amount[s] * (v - b));
                }
                const f32 r = acc / 3.f;
                px[c] = r < 0.f ? 0.f : (r > 1.f ? 1.f : r);
            }
        }
    });
}

} // namespace hhsr
