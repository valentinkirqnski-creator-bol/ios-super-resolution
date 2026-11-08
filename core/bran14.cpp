// Handheld-Multi-Frame-Super-Resolution-1.4 ("bran") pieces -- see bran14.h.
#include "bran14.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <limits>

// No contraction: 1.4's Numba kernels are compiled without fastmath, so a*b+c is two
// roundings there and must be two roundings here.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace hhsr {

// Numba's builtin max/min (do_minmax): res = (v > acc) ? v : acc, args (acc, v).
static inline double nb_max(double acc, double v) { return (v > acc) ? v : acc; }
static inline double nb_min(double acc, double v) { return (v < acc) ? v : acc; }
static inline float nb_maxf(float acc, float v) { return (v > acc) ? v : acc; }
static inline float nb_minf(float acc, float v) { return (v < acc) ? v : acc; }
// utils.clamp(x, min_, max_) = min(max_, max(min_, x))
static inline double nb_clamp(double x, double lo, double hi) { return nb_min(hi, nb_max(lo, x)); }
static inline int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
// Python round() on a float in Numba = llvm.rint = round half to even.
static inline long long nb_round(double x) { return (long long)std::nearbyint(x); }

// ---------------------------------------------------------------------------
// cfa_to_rggb
int bran14_flip_mode(const CFA& cfa) {
    auto is = [&](int a, int b, int c, int d) {
        return cfa.p[0][0] == a && cfa.p[0][1] == b && cfa.p[1][0] == c && cfa.p[1][1] == d;
    };
    if (is(0, 1, 1, 2)) return 0;   // RGGB
    if (is(2, 1, 1, 0)) return 1;   // BGGR: np.flip(axis=(-1, -2))
    if (is(1, 0, 2, 1)) return 2;   // GBRG: np.flip(axis=-1)
    if (is(1, 2, 0, 1)) return 3;   // GRBG: np.flip(axis=-2)
    return -1;
}

void bran14_flip(Image& img, int mode) {
    if (mode <= 0 || img.h <= 0 || img.w <= 0) return;
    const int h = img.h, w = img.w, c = img.c;
    f32* d = img.data.data();
    auto px = [&](int y, int x) { return d + ((size_t)y * w + x) * c; };
    if (mode == 1) {
        // 180 deg: reversing the pixel sequence (keeping each pixel's channels).
        const size_t n = (size_t)h * w;
        for (size_t i = 0, j = n - 1; i < j; ++i, --j)
            for (int k = 0; k < c; ++k) std::swap(d[i * c + k], d[j * c + k]);
    } else if (mode == 2) {
        parallel_rows(h, 0, [&](int y) {
            for (int x = 0, xr = w - 1; x < xr; ++x, --xr)
                for (int k = 0; k < c; ++k) std::swap(px(y, x)[k], px(y, xr)[k]);
        });
    } else if (mode == 3) {
        for (int y = 0, yr = h - 1; y < yr; ++y, --yr)
            for (int x = 0; x < w; ++x)
                for (int k = 0; k < c; ++k) std::swap(px(y, x)[k], px(yr, x)[k]);
    }
}

bool bran14_set_noise_profile(Config& cfg, const double* v, int n) {
    cfg.noise14_valid = false;
    if (!v || n < 2 || (n % 2) != 0) return false;
    const int planes = n / 2;
    double a[4], b[4];
    if (planes == 1) {
        for (int i = 0; i < 4; ++i) { a[i] = v[0]; b[i] = v[1]; }
    } else if (planes == 3) {
        const int src[4] = {0, 1, 2, 1};
        for (int i = 0; i < 4; ++i) { a[i] = v[2 * src[i]]; b[i] = v[2 * src[i] + 1]; }
    } else if (planes == 4) {
        for (int i = 0; i < 4; ++i) { a[i] = v[2 * i]; b[i] = v[2 * i + 1]; }
    } else {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (!(a[i] > 0.0) || !std::isfinite(a[i]) || !std::isfinite(b[i])) return false;
        cfg.noise14_alpha[i] = a[i];
        cfg.noise14_beta[i] = b[i];
    }
    cfg.noise14_valid = true;
    return true;
}

void bran14_set_black_per_channel(Config& cfg, const float black_ch[4]) {
    const int mode = bran14_flip_mode(cfg.cfa);
    const int idx[2][2] = {{0, 1}, {3, 2}};   // get_raw_arrays, post-flip positions
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const int fi = (mode == 1 || mode == 3) ? 1 - i : i;   // flipped rows
            const int fj = (mode == 1 || mode == 2) ? 1 - j : j;   // flipped columns
            cfg.black14_site[i * 2 + j] = black_ch[idx[fi][fj]];
        }
    }
    cfg.black14_valid = true;
}

// ---------------------------------------------------------------------------
// estimate_image_snr: planes r = [::2, ::2], g1 = [::2, 1::2], g2 = [1::2, ::2],
// b = [1::2, 1::2]; var = alpha*v + beta with R -> 0, G1 -> 1, G2 -> 3, B -> 2; only
// 0 < v < 1 counts. SNR = 20 log10(sqrt(sum v^2 / sum var)).
// NumPy's float32 add.reduce over a contiguous 1-D array: FLOAT_pairwise_sum
// (loops_utils.h.src) -- float accumulators, 8-way unrolled blocks up to 128 elements,
// halves rounded down to a multiple of 8 above that.
template <class F>
static float np_pairwise_sum_f32(const F& at, size_t first, size_t n) {
    if (n < 8) {
        float res = 0.f;
        for (size_t i = 0; i < n; ++i) res += at(first + i);
        return res;
    }
    if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; ++j) r[j] = at(first + j);
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += at(first + i + j);
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += at(first + i);
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_pairwise_sum_f32(at, first, n2) + np_pairwise_sum_f32(at, first + n2, n - n2);
}

// 1.4 utils_image.estimate_image_snr in NumPy 2 semantics: alpha / beta are Python
// floats (weak scalars), so alpha*plane + beta is float32; top*top float32; np.sum is
// the float32 pairwise sum over np.concatenate((r, g1, g2, b)[valid]); the ratio, sqrt,
// log10 and * 20 are float32 too. update_snr_config then takes float(np.clip(...)).
double bran14_estimate_snr(const Image& ref, const double alpha[4], const double beta[4]) {
    // The concatenation, plane by plane (r, g1, g2, b), row-major within each.
    const int offs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};   // r, g1, g2, b
    const int prof[4] = {0, 1, 3, 2};                          // alpha index per plane
    std::vector<float> vals;
    vals.reserve(ref.data.size());
    size_t start[5] = {0, 0, 0, 0, 0};
    for (int p = 0; p < 4; ++p) {
        start[p] = vals.size();
        for (int y = offs[p][0]; y < ref.h; y += 2)
            for (int x = offs[p][1]; x < ref.w; x += 2) {
                const float v = ref.at(y, x);
                if (v > 0.f && v < 1.f) vals.push_back(v);
            }
    }
    start[4] = vals.size();
    float af[4], bf[4];
    for (int p = 0; p < 4; ++p) { af[p] = (float)alpha[prof[p]]; bf[p] = (float)beta[prof[p]]; }
    auto plane_at = [&](size_t i) { int p = 0; while (i >= start[p + 1]) ++p; return p; };
    auto top = [&](size_t i) { const float v = vals[i]; return v * v; };
    auto bot = [&](size_t i) { const int p = plane_at(i); return af[p] * vals[i] + bf[p]; };
    const float st = np_pairwise_sum_f32(top, 0, vals.size());
    const float sb = np_pairwise_sum_f32(bot, 0, vals.size());
    const float snr = std::sqrt(st / sb);
    const float db = 20.f * std::log10(snr);
    return (double)db;
}

// params.lerp
static double lerp14(double x, double x0, double x1, double y0, double y1) {
    double t = (x - x0) / (x1 - x0);
    t = std::max(0.0, std::min(1.0, t));
    return y0 + (y1 - y0) * t;
}

int bran14_update_snr_config(Config& cfg, double snr) {
    if (!(snr == snr)) snr = 6.0;        // np.clip(nan) stays nan; guard a profile-less burst
    snr = std::min(30.0, std::max(6.0, snr));
    int ts = (snr <= 14.0) ? 64 : (snr <= 22.0) ? 32 : 16;
    // A manual tile size in the app replaces SNR_based, as 1.4's --alignment.tile-size.
    if (cfg.alignment_tile_size == 8 || cfg.alignment_tile_size == 16 ||
        cfg.alignment_tile_size == 32 || cfg.alignment_tile_size == 64)
        ts = cfg.alignment_tile_size;
    cfg.bm_tile_size_factors = {1.f, 1.f, 1.f, 0.5f};
    cfg.bm_tile_sizes.clear();
    for (f32 f : cfg.bm_tile_size_factors) cfg.bm_tile_sizes.push_back((int)((double)ts * (double)f));
    cfg.bran14_k_detail = cfg.k_detail_override ? (double)cfg.k_detail_override_value
                                                : lerp14(snr, 6, 30, 0.33, 0.25);
    cfg.bran14_k_denoise = lerp14(snr, 6, 30, 5.0, 3.0);
    cfg.bran14_D_th = lerp14(snr, 6, 30, 0.81, 0.71);
    cfg.bran14_D_tr = lerp14(snr, 6, 30, 1.24, 1.0);
    cfg.k_detail = (float)cfg.bran14_k_detail;
    cfg.k_denoise = (float)cfg.bran14_k_denoise;
    cfg.D_th = (float)cfg.bran14_D_th;
    cfg.D_tr = (float)cfg.bran14_D_tr;
    return ts;
}

// ---------------------------------------------------------------------------
// robustness.py

// cuda_compute_guide_image: guide[c, ty, tx], c = R, (G1 + G2) / 2, B -- sqrt'd.
// max(float32, 0) unifies to float64, so the sqrt is a double sqrt stored to float32;
// the green sum is a float32 add, then * 0.5 (a float64 literal).
static void compute_guide14(const Image& raw, int gh, int gw, std::vector<float>& g, int nt) {
    g.assign((size_t)3 * gh * gw, 0.f);
    const size_t plane = (size_t)gh * gw;
    parallel_rows(gh, nt, [&](int ty) {
        for (int tx = 0; tx < gw; ++tx) {
            const float r = raw.at(2 * ty, 2 * tx);
            const float gsum = raw.at(2 * ty, 2 * tx + 1) + raw.at(2 * ty + 1, 2 * tx);
            const float b = raw.at(2 * ty + 1, 2 * tx + 1);
            const size_t o = (size_t)ty * gw + tx;
            g[o]             = (float)std::sqrt(nb_max((double)r, 0.0));
            g[plane + o]     = (float)std::sqrt(nb_max(0.5 * (double)gsum, 0.0));
            g[2 * plane + o] = (float)std::sqrt(nb_max((double)b, 0.0));
        }
    });
}

// cuda_compute_local_stats: 3x3 clamped window; mean_/var_ start as int 0 and become
// float64 accumulators; color*color is a float32 product.
static void local_stats14(const std::vector<float>& g, int gh, int gw,
                          std::vector<float>& mean, std::vector<float>* var, int nt) {
    const size_t plane = (size_t)gh * gw;
    mean.assign(3 * plane, 0.f);
    if (var) var->assign(3 * plane, 0.f);
    parallel_rows(gh, nt, [&](int idy) {
        for (int ch = 0; ch < 3; ++ch) {
            const float* gp = g.data() + ch * plane;
            for (int idx = 0; idx < gw; ++idx) {
                double m = 0.0, v = 0.0;
                for (int i = -1; i <= 1; ++i) {
                    const int y = clampi(idy + i, 0, gh - 1);
                    for (int j = -1; j <= 1; ++j) {
                        const int x = clampi(idx + j, 0, gw - 1);
                        const float c = gp[(size_t)y * gw + x];
                        m += (double)c;
                        v += (double)(c * c);
                    }
                }
                m /= 9;
                const size_t o = ch * plane + (size_t)idy * gw + idx;
                mean[o] = (float)m;
                if (var) (*var)[o] = (float)(v / 9 - m * m);
            }
        }
    });
}

RefStats14 bran14_init_robustness(const Image& ref, int nt) {
    RefStats14 s;
    s.gh = ref.h / 2;
    s.gw = ref.w / 2;
    std::vector<float> g;
    compute_guide14(ref, s.gh, s.gw, g, nt);
    local_stats14(g, s.gh, s.gw, s.mean, &s.var, nt);
    return s;
}

// utils_image.dogson_quadratic_kernel (float64 argument)
static inline double dogson_quadratic14(double x) {
    const double a = std::fabs(x);
    if (a <= 0.5) return -2 * a * a + 1;
    if (a <= 1.5) return a * a - 5.0 / 2 * a + 1.5;
    return 0.0;
}

Image bran14_compute_robustness(const Image& comp, const RefStats14& ref,
                                const FlowField& flow, const Robustness14Params& p) {
    return bran14_compute_robustness_debug(comp, ref, flow, p, nullptr);
}

Image bran14_compute_robustness_debug(const Image& comp, const RefStats14& ref,
                                      const FlowField& flow, const Robustness14Params& p,
                                      Robustness14Debug* dbg) {
    const int gh = ref.gh, gw = ref.gw, nt = p.num_threads;
    const size_t plane = (size_t)gh * gw;
    if (gh <= 0 || gw <= 0 || comp.h / 2 != gh || comp.w / 2 != gw || flow.ny <= 0 || flow.nx <= 0)
        return Image();
    const int ts2 = p.tile_size / 2;
    const float inf = std::numeric_limits<float>::infinity();

    // Comparison guide + local means (variances unused).
    std::vector<float> g, cm;
    compute_guide14(comp, gh, gw, g, nt);
    local_stats14(g, gh, gw, cm, nullptr, nt);
    std::vector<float>().swap(g);

    // cuda_warp_dogson: per-guide-pixel flow (nearest tile), y_mov = y + flow_y * 0.5 in
    // float64; out of frame -> inf; centre = round(y_mov); 3x3 Dodgson weights at
    // clamped taps; buffer[c] is a float32 local array (+= in float64, stored float32),
    // w_acc float64; out = buffer / w_acc.
    std::vector<float> wm(3 * plane, 0.f);
    parallel_rows(gh, nt, [&](int y) {
        const int pyi = y / ts2;
        for (int x = 0; x < gw; ++x) {
            const int pxi = x / ts2;
            const float fx = flow.dx(pyi, pxi), fy = flow.dy(pyi, pxi);
            const double y_mov = (double)y + (double)fy * 0.5;
            const double x_mov = (double)x + (double)fx * 0.5;
            const size_t o = (size_t)y * gw + x;
            if (!(0 <= y_mov && y_mov < gh && 0 <= x_mov && x_mov < gw)) {
                for (int c = 0; c < 3; ++c) wm[c * plane + o] = inf;
                continue;
            }
            const long long cy = nb_round(y_mov), cx = nb_round(x_mov);
            double w_acc = 0.0;
            float buf[3] = {0.f, 0.f, 0.f};
            for (int i = -1; i <= 1; ++i) {
                const int y_ = (int)nb_clamp((double)(cy + i), 0, gh - 1);
                const double wy = dogson_quadratic14((double)y_ - y_mov);
                for (int j = -1; j <= 1; ++j) {
                    const int x_ = (int)nb_clamp((double)(cx + j), 0, gw - 1);
                    const double w = wy * dogson_quadratic14((double)x_ - x_mov);
                    const size_t so = (size_t)y_ * gw + x_;
                    for (int c = 0; c < 3; ++c)
                        buf[c] = (float)((double)buf[c] + (double)cm[c * plane + so] * w);
                    w_acc += w;
                }
            }
            for (int c = 0; c < 3; ++c) wm[c * plane + o] = (float)((double)buf[c] / w_acc);
        }
    });
    if (dbg) { dbg->comp_means = cm; dbg->warped = wm; }
    std::vector<float>().swap(cm);

    // cuda_compute_d_sigma
    std::vector<float> dsq(plane), ssq(plane);
    const int nc = p.n_curve;
    const bool corr = p.noise_correction && nc >= 2 && p.sigma_sq_curve && p.d_sq_curve;
    parallel_rows(gh, nt, [&](int y) {
        for (int x = 0; x < gw; ++x) {
            const size_t o = (size_t)y * gw + x;
            double d = 0.0, s = 0.0;
            for (int c = 0; c < 3; ++c) {
                const float err = ref.mean[c * plane + o] - wm[c * plane + o];
                d += (double)(err * err);
                s += (double)ref.var[c * plane + o];
            }
            if (corr) {
                double b = 0.0;
                for (int c = 0; c < 3; ++c) b += (double)ref.mean[c * plane + o];
                b /= 3;
                b = nb_clamp(b, 0, 1);
                const long long id = nb_round((double)(nc - 1) * b);
                const float dn = p.d_sq_curve[id];
                const float sn = p.sigma_sq_curve[id];
                s = nb_max(s, (double)sn);
                if (d > 0) {
                    const double shrink = d / (d + (double)dn);
                    d *= shrink * shrink;
                }
            }
            if (p.sigma_mult != 1.0) s *= p.sigma_mult * p.sigma_mult;   // (k sigma)^2; skipped at 1
            dsq[o] = (float)d;
            ssq[o] = (float)s;
        }
    });
    std::vector<float>().swap(wm);

    // cuda_compute_s on the tile grid. flows / mini / maxi are float32; the threshold
    // compare is against M_th * M_th in float64.
    const int ny = flow.ny, nx = flow.nx;
    std::vector<float> S((size_t)ny * nx);
    const double mt2 = p.Mt * p.Mt;
    parallel_rows(ny, nt, [&](int py) {
        for (int px = 0; px < nx; ++px) {
            float mn0 = inf, mn1 = inf, mx0 = -inf, mx1 = -inf;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    const int yy = py + i, xx = px + j;
                    if (!(0 <= xx && xx < nx && 0 <= yy && yy < ny)) continue;
                    const float f0 = flow.dx(yy, xx), f1 = flow.dy(yy, xx);
                    mx0 = nb_maxf(mx0, f0); mx1 = nb_maxf(mx1, f1);
                    mn0 = nb_minf(mn0, f0); mn1 = nb_minf(mn1, f1);
                }
            }
            const float d0 = mx0 - mn0, d1 = mx1 - mn1;
            const float m2 = d0 * d0 + d1 * d1;
            S[(size_t)py * nx + px] = ((double)m2 > mt2) ? (float)p.s1 : (float)p.s2;
        }
    });

    // cuda_robustness_threshold: -d/sigma and the exp in float32, S * exp float32, - t
    // in float64, clamp to [0, 1] with Numba's max/min (NaN -> 0).
    std::vector<float> R(plane);
    parallel_rows(gh, nt, [&](int idy) {
        const int pyi = idy / ts2;
        for (int idx = 0; idx < gw; ++idx) {
            const int pxi = idx / ts2;
            const size_t o = (size_t)idy * gw + idx;
            const float q = -dsq[o] / ssq[o];
            const float e = S[(size_t)pyi * nx + pxi] * std::exp(q);
            R[o] = (float)nb_clamp((double)e - p.t, 0, 1);
        }
    });
    if (dbg) { dbg->d_sq = dsq; dbg->sigma_sq = ssq; dbg->S = S; dbg->R = R; }

    // cuda_compute_local_min: 5x5 clamped, mini = inf (float64).
    Image r(gh, gw, 1);
    parallel_rows(gh, nt, [&](int idy) {
        for (int idx = 0; idx < gw; ++idx) {
            double mini = std::numeric_limits<double>::infinity();
            for (int i = -2; i <= 2; ++i) {
                const int y = clampi(idy + i, 0, gh - 1);
                for (int j = -2; j <= 2; ++j) {
                    const int x = clampi(idx + j, 0, gw - 1);
                    mini = nb_min(mini, (double)R[(size_t)y * gw + x]);
                }
            }
            r.at(idy, idx) = (float)mini;
        }
    });
    return r;
}

}  // namespace hhsr
