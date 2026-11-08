// Consensus refiner (see cref.h). CPU features + banded inference.
#include "cref.h"
#include "cref_weights.h"
#include "parallel.h"
#if defined(__APPLE__)
#include "metal_gpu.h"
#endif
#include <algorithm>
#include <cmath>

namespace hhsr {

const int kCrefDil[kCrefLayers] = {1, 2, 4, 1};

bool cref_available() { return kCrefTrained; }

static inline float cref_c(float x) { return std::min(8.f, std::max(-8.f, x)) * 0.125f; }

void cref_features(const Image& ref_grey, const Image& comp_aligned, const Image& cons,
                   int n_frames, const Image& R_grey, const Config& cfg,
                   int y0, int y1, int halo, std::vector<float>& out) {
    const int h = ref_grey.h, w = ref_grey.w;
    const int rows = (y1 - y0) + 2 * halo;
    const size_t plane = (size_t)rows * w;
    out.assign((size_t)kCrefIn * plane, 0.f);
    const float na = std::max(0.f, cfg.alpha_dng[1]), nb = std::max(1e-12f, cfg.beta_dng[1]);
    const int n = std::max(1, n_frames);
    auto px = [&](const Image& im, int y, int x) {
        return im.at(std::min(h - 1, std::max(0, y)), std::min(w - 1, std::max(0, x)));
    };
    // 3x3 high-pass: centre minus the 3x3 box mean (edge-clamped).
    auto hp = [&](const Image& im, int y, int x) {
        float s = 0.f;
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j) s += px(im, y + i, x + j);
        return px(im, y, x) - s * (1.f / 9.f);
    };
    parallel_rows(rows, cfg.num_threads, [&](int r) {
        const int y = std::min(h - 1, std::max(0, y0 - halo + r));
        for (int x = 0; x < w; ++x) {
            const float rf = ref_grey.at(y, x), cp = comp_aligned.at(y, x), cs = cons.at(y, x);
            const float loo = (n >= 2) ? ((float)n * cs - cp) / (float)(n - 1) : rf;
            const float m = std::max(rf, 0.f);
            const float v = na * m + nb;
            const float isd = 1.f / std::sqrt(2.f * v), is1 = 1.f / std::sqrt(v);
            const size_t o = (size_t)r * w + x;
            out[o] = std::min(1.f, std::max(0.f, R_grey.at(y, x)));
            out[plane + o] = cref_c((cp - rf) * isd);
            out[2 * plane + o] = cref_c((cp - loo) * isd);
            out[3 * plane + o] = cref_c((rf - loo) * isd);
            out[4 * plane + o] = cref_c(hp(ref_grey, y, x) * is1);
            out[5 * plane + o] = cref_c(hp(comp_aligned, y, x) * is1);
        }
    });
}

// One 3x3 dilated conv + ReLU over a band (plane-major, rows x w), replicate padding.
static void cref_conv3(const std::vector<float>& in, int cin, std::vector<float>& out, int cout,
                       int rows, int w, const float* W, const float* B, int dil, int nt,
                       int vlo, int vhi) {
    const size_t plane = (size_t)rows * w;
    out.assign((size_t)cout * plane, 0.f);
    parallel_rows(rows, nt, [&](int y) {
        float acc[kCrefCh];
        for (int x = 0; x < w; ++x) {
            for (int co = 0; co < cout; ++co) acc[co] = B[co];
            for (int ky = 0; ky < 3; ++ky) {
                const int yy = std::min(vhi, std::max(vlo, y + (ky - 1) * dil));
                for (int kx = 0; kx < 3; ++kx) {
                    const int xx = std::min(w - 1, std::max(0, x + (kx - 1) * dil));
                    const size_t q = (size_t)yy * w + xx;
                    for (int ci = 0; ci < cin; ++ci) {
                        const float v = in[(size_t)ci * plane + q];
                        // weight layout [cout][cin][3][3] (PyTorch)
                        const float* wp = W + ((size_t)ci * 9 + ky * 3 + kx);
                        for (int co = 0; co < cout; ++co) acc[co] += wp[(size_t)co * cin * 9] * v;
                    }
                }
            }
            for (int co = 0; co < cout; ++co) out[(size_t)co * plane + (size_t)y * w + x] = std::max(0.f, acc[co]);
        }
    });
}

bool cref_refine(Image& R, const Image& ref_grey, const Image& comp_aligned,
                 const Image& cons, int n_frames, const Config& cfg) {
    if (!kCrefTrained) return false;
    const int h = ref_grey.h, w = ref_grey.w;
    if (h <= 0 || w <= 0 || R.h <= 0 || R.w <= 0 || R.data.size() != (size_t)R.h * R.w * R.c) return false;
    if (comp_aligned.h != h || comp_aligned.w != w || cons.h != h || cons.w != w) return false;
    // Mask on the grey grid (nearest; identity when R is already raw/grey-res).
    Image Rg(h, w, 1);
    parallel_rows(h, cfg.num_threads, [&](int y) {
        const int sy = std::min(R.h - 1, (int)((int64_t)y * R.h / h));
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(R.w - 1, (int)((int64_t)x * R.w / w));
            Rg.at(y, x) = R.at(sy, sx, 0);
        }
    });
    Image Rout(h, w, 1);
    bool done = false;
#if defined(__APPLE__)
    done = cref_refine_bands_metal(ref_grey, comp_aligned, cons, Rg, n_frames, cfg, Rout);
#endif
    const int band = 128, halo = kCrefHalo;
    std::vector<float> f, a, b;
    for (int y0 = 0; !done && y0 < h; y0 += band) {
        const int y1 = std::min(h, y0 + band);
        cref_features(ref_grey, comp_aligned, cons, n_frames, Rg, cfg, y0, y1, halo, f);
        const int rows = (y1 - y0) + 2 * halo;
        const int vlo = std::max(0, halo - y0), vhi = std::min(rows - 1, halo + (h - 1 - y0));
        cref_conv3(f, kCrefIn, a, kCrefCh, rows, w, kCrefW0, kCrefB0, kCrefDil[0], cfg.num_threads, vlo, vhi);
        cref_conv3(a, kCrefCh, b, kCrefCh, rows, w, kCrefW1, kCrefB1, kCrefDil[1], cfg.num_threads, vlo, vhi);
        cref_conv3(b, kCrefCh, a, kCrefCh, rows, w, kCrefW2, kCrefB2, kCrefDil[2], cfg.num_threads, vlo, vhi);
        cref_conv3(a, kCrefCh, b, kCrefCh, rows, w, kCrefW3, kCrefB3, kCrefDil[3], cfg.num_threads, vlo, vhi);
        const size_t plane = (size_t)rows * w;
        parallel_rows(y1 - y0, cfg.num_threads, [&](int r) {
            const int yb = r + halo;
            for (int x = 0; x < w; ++x) {
                const size_t q = (size_t)yb * w + x;
                float d = kCrefBO;
                for (int c = 0; c < kCrefCh; ++c) d += kCrefWO[c] * b[(size_t)c * plane + q];
                const float r0 = std::min(0.98f, std::max(0.02f, Rg.at(y0 + r, x)));
                const float z = std::log(r0 / (1.f - r0)) + d;
                Rout.at(y0 + r, x) = 1.f / (1.f + std::exp(-z));
            }
        });
    }
    // Back to R's grid: mean of the grey pixels each R pixel covers (identity at raw res).
    if (R.h == h && R.w == w) {
        for (size_t i = 0; i < (size_t)h * w; ++i)
            for (int c = 0; c < R.c; ++c) R.data[i * R.c + c] = Rout.data[i];
    } else {
        parallel_rows(R.h, cfg.num_threads, [&](int Y) {
            const int ya = (int)((int64_t)Y * h / R.h), yb = std::max(ya + 1, (int)((int64_t)(Y + 1) * h / R.h));
            for (int X = 0; X < R.w; ++X) {
                const int xa = (int)((int64_t)X * w / R.w), xb = std::max(xa + 1, (int)((int64_t)(X + 1) * w / R.w));
                float s = 0.f;
                int cnt = 0;
                for (int y = ya; y < std::min(h, yb); ++y)
                    for (int x = xa; x < std::min(w, xb); ++x) { s += Rout.at(y, x); ++cnt; }
                const float v = cnt ? s / (float)cnt : R.at(Y, X, 0);
                for (int c = 0; c < R.c; ++c) R.at(Y, X, c) = v;
            }
        });
    }
    return true;
}

}  // namespace hhsr
