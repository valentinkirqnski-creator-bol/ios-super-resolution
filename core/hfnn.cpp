// HF-artifact NN (see hfnn.h). CPU features + banded inference; Metal runs the convs.
#include "hfnn.h"
#include "cref.h"            // cref_flow_disagreement, cref_aux_rows (features 4, 5)
#include "hfnn_weights.h"
#include "parallel.h"
#if defined(__APPLE__)
#include "metal_gpu.h"
#endif
#include <algorithm>
#include <cmath>

namespace hhsr {

const int kHfnnDil[kHfnnLayers] = {1, 2, 4};

bool hfnn_available() { return kHfnnTrained; }

static inline float hf_c(float x) { return std::min(8.f, std::max(-8.f, x)) * 0.125f; }

// Features for rows [y0 - halo, y1 + halo) from precomputed aux planes covering image
// rows [alo, ahi) (dis, hf), every row edge-clamped.
static void hfnn_features_from_aux(const Image& ref_grey, const Image& comp_aligned,
                                   const Image& R_grey, const Config& cfg, int y0, int y1,
                                   int halo, const std::vector<float>& adis,
                                   const std::vector<float>& ahf, int alo,
                                   std::vector<float>& out, std::vector<float>& hf_out) {
    const int h = ref_grey.h, w = ref_grey.w;
    const int rows = (y1 - y0) + 2 * halo;
    const size_t plane = (size_t)rows * w;
    out.assign((size_t)kHfnnIn * plane, 0.f);
    hf_out.assign(plane, 0.f);
    const float na = std::max(0.f, cfg.alpha_dng[1]), nb = std::max(1e-12f, cfg.beta_dng[1]);
    auto px = [&](const Image& im, int y, int x) {
        return im.at(std::min(h - 1, std::max(0, y)), std::min(w - 1, std::max(0, x)));
    };
    auto hp = [&](const Image& im, int y, int x) {
        float s = 0.f;
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j) s += px(im, y + i, x + j);
        return px(im, y, x) - s * (1.f / 9.f);
    };
    parallel_rows(rows, cfg.num_threads, [&](int r) {
        const int y = std::min(h - 1, std::max(0, y0 - halo + r));
        for (int x = 0; x < w; ++x) {
            const float rf = ref_grey.at(y, x), cp = comp_aligned.at(y, x);
            const float v = na * std::max(rf, 0.f) + nb;
            const float isd = 1.f / std::sqrt(2.f * v), is1 = 1.f / std::sqrt(v);
            const size_t o = (size_t)r * w + x;
            const size_t ao = (size_t)(y - alo) * w + x;
            out[o] = std::min(1.f, std::max(0.f, R_grey.at(y, x)));
            out[plane + o] = hf_c((cp - rf) * isd);
            out[2 * plane + o] = hf_c(hp(ref_grey, y, x) * is1);
            out[3 * plane + o] = hf_c(hp(comp_aligned, y, x) * is1);
            out[4 * plane + o] = adis[ao];
            out[5 * plane + o] = ahf[ao];
            hf_out[o] = ahf[ao];
        }
    });
}

void hfnn_features(const Image& ref_grey, const Image& comp_aligned, const Image& R_grey,
                   const std::vector<float>& tdis, int ny, int nx, int tile_size,
                   const Config& cfg, int y0, int y1, int halo,
                   std::vector<float>& out, std::vector<float>& hf_out) {
    const int h = ref_grey.h, w = ref_grey.w;
    const int alo = std::max(0, y0 - halo), ahi = std::min(h, y1 + halo);
    std::vector<float> adis((size_t)(ahi - alo) * w), ahf((size_t)(ahi - alo) * w);
    cref_aux_rows(ref_grey, tdis, ny, nx, tile_size, cfg, alo, ahi, adis.data(), ahf.data());
    hfnn_features_from_aux(ref_grey, comp_aligned, R_grey, cfg, y0, y1, halo, adis, ahf, alo, out, hf_out);
}

// One 3x3 dilated conv + ReLU over a band (plane-major), replicate padding (rows
// clamped to [vlo, vhi], the band rows that are real image rows).
static void hfnn_conv3(const std::vector<float>& in, int cin, std::vector<float>& out,
                       int rows, int w, const float* W, const float* B, int dil, int nt,
                       int vlo, int vhi) {
    const size_t plane = (size_t)rows * w;
    out.assign((size_t)kHfnnCh * plane, 0.f);
    parallel_rows(rows, nt, [&](int y) {
        float acc[kHfnnCh];
        for (int x = 0; x < w; ++x) {
            for (int co = 0; co < kHfnnCh; ++co) acc[co] = B[co];
            for (int ky = 0; ky < 3; ++ky) {
                const int yy = std::min(vhi, std::max(vlo, y + (ky - 1) * dil));
                for (int kx = 0; kx < 3; ++kx) {
                    const int xx = std::min(w - 1, std::max(0, x + (kx - 1) * dil));
                    const size_t q = (size_t)yy * w + xx;
                    for (int ci = 0; ci < cin; ++ci) {
                        const float v = in[(size_t)ci * plane + q];
                        const float* wp = W + ((size_t)ci * 9 + ky * 3 + kx);
                        for (int co = 0; co < kHfnnCh; ++co) acc[co] += wp[(size_t)co * cin * 9] * v;
                    }
                }
            }
            for (int co = 0; co < kHfnnCh; ++co) out[(size_t)co * plane + (size_t)y * w + x] = std::max(0.f, acc[co]);
        }
    });
}

bool hfnn_refine(Image& R, const Image& ref_grey, const Image& comp_aligned,
                 const FlowField& flow, int tile_size, const Config& cfg) {
    if (!kHfnnTrained) return false;
    const int h = ref_grey.h, w = ref_grey.w;
    if (h <= 0 || w <= 0 || R.h <= 0 || R.w <= 0 || R.data.size() != (size_t)R.h * R.w * R.c) return false;
    if (comp_aligned.h != h || comp_aligned.w != w) return false;
    const std::vector<float> tdis = cref_flow_disagreement(flow);
    const float g_lo = cfg.hfnn_gate - 0.1f, g_hi = cfg.hfnn_gate + 0.1f;
    // Mask on the grey grid (nearest; identity when R is already raw/grey-res).
    Image Rg(h, w, 1);
    parallel_rows(h, cfg.num_threads, [&](int y) {
        const int sy = std::min(R.h - 1, (int)((int64_t)y * R.h / h));
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(R.w - 1, (int)((int64_t)x * R.w / w));
            Rg.at(y, x) = R.at(sy, sx, 0);
        }
    });
    Image Rout = Rg;   // untouched outside the bands that hold HF pixels
    const int band = 128, halo = kHfnnHalo;
    std::vector<float> f, hfp, a, b, adis, ahf;
    bool any = false;
    for (int y0 = 0; y0 < h; y0 += band) {
        const int y1 = std::min(h, y0 + band);
        const int alo = std::max(0, y0 - halo), ahi = std::min(h, y1 + halo);
        adis.resize((size_t)(ahi - alo) * w); ahf.resize((size_t)(ahi - alo) * w);
        cref_aux_rows(ref_grey, tdis, flow.ny, flow.nx, tile_size, cfg, alo, ahi, adis.data(), ahf.data());
        // Skip the band unless one of its OUTPUT rows is in an HF area.
        float mx = 0.f;
        for (size_t i = (size_t)(y0 - alo) * w; i < (size_t)(y1 - alo) * w; ++i) mx = std::max(mx, ahf[i]);
        if (mx <= g_lo) continue;
        any = true;
        hfnn_features_from_aux(ref_grey, comp_aligned, Rg, cfg, y0, y1, halo, adis, ahf, alo, f, hfp);
        const int rows = (y1 - y0) + 2 * halo;
        const int vlo = std::max(0, halo - y0), vhi = std::min(rows - 1, halo + (h - 1 - y0));
        std::vector<float> dlt;   // per output pixel: the network's delta
        bool gpu = false;
#if defined(__APPLE__)
        gpu = hfnn_band_metal(f, rows, w, halo, y1 - y0, vlo, vhi, dlt);
#endif
        if (!gpu) {
            const float* Ws[kHfnnLayers] = {kHfnnW0, kHfnnW1, kHfnnW2};
            const float* Bs[kHfnnLayers] = {kHfnnB0, kHfnnB1, kHfnnB2};
            hfnn_conv3(f, kHfnnIn, a, rows, w, Ws[0], Bs[0], kHfnnDil[0], cfg.num_threads, vlo, vhi);
            hfnn_conv3(a, kHfnnCh, b, rows, w, Ws[1], Bs[1], kHfnnDil[1], cfg.num_threads, vlo, vhi);
            hfnn_conv3(b, kHfnnCh, a, rows, w, Ws[2], Bs[2], kHfnnDil[2], cfg.num_threads, vlo, vhi);
            const size_t plane = (size_t)rows * w;
            dlt.assign((size_t)(y1 - y0) * w, 0.f);
            parallel_rows(y1 - y0, cfg.num_threads, [&](int r) {
                for (int x = 0; x < w; ++x) {
                    const size_t q = (size_t)(r + halo) * w + x;
                    float d = kHfnnBO;
                    for (int c = 0; c < kHfnnCh; ++c) d += kHfnnWO[c] * a[(size_t)c * plane + q];
                    dlt[(size_t)r * w + x] = d;
                }
            });
        }
        // Blend into the mask only where the reference is HF texture.
        parallel_rows(y1 - y0, cfg.num_threads, [&](int r) {
            const size_t hrow = (size_t)(r + halo) * w;
            for (int x = 0; x < w; ++x) {
                const float hv = hfp[hrow + x];
                if (hv <= g_lo) continue;
                float t = std::min(1.f, (hv - g_lo) / std::max(1e-6f, g_hi - g_lo));
                t = t * t * (3.f - 2.f * t);
                const float r0 = std::min(0.98f, std::max(0.02f, Rg.at(y0 + r, x)));
                const float z = std::log(r0 / (1.f - r0)) + dlt[(size_t)r * w + x] - cfg.hfnn_strictness;
                const float rn = 1.f / (1.f + std::exp(-z));
                const float rg = Rg.at(y0 + r, x);
                Rout.at(y0 + r, x) = rg + t * (rn - rg);
            }
        });
    }
    if (!any) return true;   // no HF area in this frame: R unchanged
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
