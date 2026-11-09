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

std::vector<float> cref_flow_disagreement(const FlowField& flow) {
    const int ny = flow.ny, nx = flow.nx;
    std::vector<float> d((size_t)std::max(0, ny) * std::max(0, nx), 0.f);
    if (ny <= 0 || nx <= 0 || flow.flow.size() < (size_t)ny * nx * 2) return d;
    for (int ty = 0; ty < ny; ++ty)
        for (int tx = 0; tx < nx; ++tx) {
            float vx[9], vy[9];
            int k = 0;
            for (int i = -1; i <= 1; ++i)
                for (int j = -1; j <= 1; ++j) {
                    const int qy = std::min(ny - 1, std::max(0, ty + i)), qx = std::min(nx - 1, std::max(0, tx + j));
                    vx[k] = flow.dx(qy, qx); vy[k] = flow.dy(qy, qx); ++k;
                }
            std::nth_element(vx, vx + 4, vx + 9);
            std::nth_element(vy, vy + 4, vy + 9);
            d[(size_t)ty * nx + tx] = std::hypot(flow.dx(ty, tx) - vx[4], flow.dy(ty, tx) - vy[4]);
        }
    return d;
}

void cref_aux_rows(const Image& ref_grey, const std::vector<float>& tdis, int ny, int nx,
                   int tile_size, const Config& cfg, int ylo, int yhi, float* dis, float* hf) {
    const int h = ref_grey.h, w = ref_grey.w, nr = yhi - ylo;
    if (nr <= 0) return;
    const int ts = std::max(1, tile_size);
    const bool have_d = ny > 0 && nx > 0 && tdis.size() == (size_t)ny * nx;
    // HF variance loss on 5x5 windows: binomial [1 4 6 4 1]/16 low-pass L of the ref,
    // then box-5 mean / mean-square of ref and L. Rows [ylo-4, yhi+4) are touched,
    // every read edge-clamped (the Metal path gets these planes from here too).
    const int pad = 4, R0 = ylo - pad, NR = nr + 2 * pad;
    auto cy = [&](int y) { return std::min(h - 1, std::max(0, y)); };
    auto cx = [&](int x) { return std::min(w - 1, std::max(0, x)); };
    std::vector<float> T((size_t)NR * w), L((size_t)NR * w);
    static const float kb[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
    parallel_rows(NR, cfg.num_threads, [&](int r) {      // horizontal binomial of rows R0+r
        const int y = cy(R0 + r);
        for (int x = 0; x < w; ++x) {
            float s = 0.f;
            for (int j = -2; j <= 2; ++j) s += kb[j + 2] * ref_grey.at(y, cx(x + j));
            T[(size_t)r * w + x] = s;
        }
    });
    parallel_rows(NR, cfg.num_threads, [&](int r) {      // vertical binomial (rows clamped inside T's span)
        for (int x = 0; x < w; ++x) {
            float s = 0.f;
            for (int i = -2; i <= 2; ++i) {
                const int rr = cy(R0 + r + i) - R0;
                s += kb[i + 2] * T[(size_t)std::min(NR - 1, std::max(0, rr)) * w + x];
            }
            L[(size_t)r * w + x] = s;
        }
    });
    const float na = std::max(0.f, cfg.alpha_dng[1]), nb = std::max(1e-12f, cfg.beta_dng[1]);
    const float kE = (70.f / 256.f) * (70.f / 256.f);    // noise energy kept by the 2-D binomial
    parallel_rows(nr, cfg.num_threads, [&](int r) {
        const int y = ylo + r;
        const int ty = std::min(ny - 1, y / ts);
        for (int x = 0; x < w; ++x) {
            float s1 = 0.f, s2 = 0.f, l1 = 0.f, l2 = 0.f;
            for (int i = -2; i <= 2; ++i) {
                const int yy = cy(y + i);
                const size_t lr = (size_t)std::min(NR - 1, std::max(0, yy - R0)) * w;
                for (int j = -2; j <= 2; ++j) {
                    const int xx = cx(x + j);
                    const float v = ref_grey.at(yy, xx), l = L[lr + xx];
                    s1 += v; s2 += v * v; l1 += l; l2 += l * l;
                }
            }
            const float m = s1 / 25.f, ml = l1 / 25.f;
            const float nv = na * std::max(m, 0.f) + nb;
            const float v = std::max(0.f, s2 / 25.f - m * m) - nv;
            const float vl = std::max(0.f, std::max(0.f, l2 / 25.f - ml * ml) - kE * nv);
            hf[(size_t)r * w + x] = (v > 4.f * nv) ? std::min(1.f, std::max(0.f, 1.f - vl / v)) : 0.f;
            float d = 0.f;
            if (have_d) d = tdis[(size_t)ty * nx + std::min(nx - 1, x / ts)];
            dis[(size_t)r * w + x] = std::min(4.f, d) * 0.25f;
        }
    });
}

void cref_features(const Image& ref_grey, const Image& comp_aligned, const Image& cons,
                   int n_frames, const Image& R_grey, const std::vector<float>& tdis,
                   int ny, int nx, int tile_size, const Config& cfg,
                   int y0, int y1, int halo, std::vector<float>& out) {
    const int h = ref_grey.h, w = ref_grey.w;
    const int rows = (y1 - y0) + 2 * halo;
    const size_t plane = (size_t)rows * w;
    out.assign((size_t)kCrefIn * plane, 0.f);
    // Aux planes for the real rows this band touches, then copied with the same
    // edge-clamped row mapping as the other features.
    const int alo = std::max(0, y0 - halo), ahi = std::min(h, y1 + halo);
    std::vector<float> adis((size_t)(ahi - alo) * w), ahf((size_t)(ahi - alo) * w);
    cref_aux_rows(ref_grey, tdis, ny, nx, tile_size, cfg, alo, ahi, adis.data(), ahf.data());
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
            const size_t ao = (size_t)(y - alo) * w + x;
            out[6 * plane + o] = adis[ao];
            out[7 * plane + o] = ahf[ao];
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
                 const Image& cons, int n_frames, const FlowField& flow, int tile_size,
                 const Config& cfg) {
    if (!kCrefTrained) return false;
    const std::vector<float> tdis = cref_flow_disagreement(flow);
    const int fny = flow.ny, fnx = flow.nx;
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
    done = cref_refine_bands_metal(ref_grey, comp_aligned, cons, Rg, n_frames, tdis, fny, fnx,
                                   tile_size, cfg, Rout);
#endif
    const int band = 128, halo = kCrefHalo;
    std::vector<float> f, a, b;
    for (int y0 = 0; !done && y0 < h; y0 += band) {
        const int y1 = std::min(h, y0 + band);
        cref_features(ref_grey, comp_aligned, cons, n_frames, Rg, tdis, fny, fnx, tile_size, cfg,
                      y0, y1, halo, f);
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
