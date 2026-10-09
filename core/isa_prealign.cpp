#include "isa_prealign.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace hhsr {
namespace {

using cf = std::complex<float>;
constexpr double kPi = 3.14159265358979323846;

int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Precomputed per-size FFT plan: bit-reversal permutation and a forward twiddle
// table. Cached across calls/frames so the cos/sin and index work is paid once.
struct FftPlan {
    int n = 0;
    std::vector<int> rev;
    std::vector<float> twr, twi;  // cos/sin(-2*pi*t/n), t in [0, n/2)
};

static const FftPlan& fft_plan(int n) {
    static std::mutex mu;
    static std::vector<std::unique_ptr<FftPlan>> cache;
    std::lock_guard<std::mutex> lk(mu);
    for (const auto& p : cache)
        if (p->n == n) return *p;
    auto p = std::make_unique<FftPlan>();
    p->n = n;
    p->rev.resize((size_t)n);
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    for (int i = 0; i < n; ++i) {
        int j = 0;
        for (int b = 0; b < bits; ++b) j |= ((i >> b) & 1) << (bits - 1 - b);
        p->rev[(size_t)i] = j;
    }
    p->twr.resize((size_t)(n / 2));
    p->twi.resize((size_t)(n / 2));
    for (int t = 0; t < n / 2; ++t) {
        const double a = -2.0 * kPi * (double)t / (double)n;
        p->twr[(size_t)t] = (float)std::cos(a);
        p->twi[(size_t)t] = (float)std::sin(a);
    }
    cache.push_back(std::move(p));
    return *cache.back();
}

// In-place radix-2 FFT on one length-n row, operating on the interleaved
// [re,im] floats of a std::complex array (layout-compatible). Manual butterflies
// with a precomputed twiddle table -- several times faster than the std::complex
// operator* form. No 1/N scaling (omitted, as ISA divides once and it does not
// change the argmax). inverse: conjugate the twiddles.
static void fft1d(cf* data, int n, const FftPlan& pl, bool inverse) {
    float* a = reinterpret_cast<float*>(data);
    const int* rev = pl.rev.data();
    for (int i = 0; i < n; ++i) {
        const int j = rev[i];
        if (i < j) {
            std::swap(a[2 * i], a[2 * j]);
            std::swap(a[2 * i + 1], a[2 * j + 1]);
        }
    }
    const float* twr = pl.twr.data();
    const float* twi = pl.twi.data();
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1, step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; ++k) {
                const int t = k * step;
                const float wr = twr[t];
                const float wi = inverse ? -twi[t] : twi[t];
                const int ia = i + k, ib = ia + half;
                const float xr = a[2 * ib], xi = a[2 * ib + 1];
                const float tr = wr * xr - wi * xi;
                const float ti = wr * xi + wi * xr;
                const float ur = a[2 * ia], ui = a[2 * ia + 1];
                a[2 * ib] = ur - tr; a[2 * ib + 1] = ui - ti;
                a[2 * ia] = ur + tr; a[2 * ia + 1] = ui + ti;
            }
        }
    }
}

static void transpose_sq(cf* d, int N) {
    for (int y = 0; y < N; ++y)
        for (int x = y + 1; x < N; ++x)
            std::swap(d[(size_t)y * N + x], d[(size_t)x * N + y]);
}

// Row-major N x N complex 2D FFT. Columns are done by transpose + row FFT +
// transpose so every 1D pass runs over contiguous memory (the strided column
// access was the other half of the old cost).
void fft2d(std::vector<cf>& d, int N, bool inverse) {
    const FftPlan& pl = fft_plan(N);
    cf* p = d.data();
    for (int y = 0; y < N; ++y) fft1d(p + (size_t)y * N, N, pl, inverse);
    transpose_sq(p, N);
    for (int y = 0; y < N; ++y) fft1d(p + (size_t)y * N, N, pl, inverse);
    transpose_sq(p, N);
}

// Bilinear sample of a th x tw scratch buffer (the resized moving grey), used
// by the rotation warp. OOB -> 0.
float sample_bilinear(const std::vector<float>& img, int h, int w, float y, float x) {
    if (!(y >= 0.f && y <= (float)(h - 1) && x >= 0.f && x <= (float)(w - 1)))
        return 0.f;
    const int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    const float fx = x - (float)x0, fy = y - (float)y0;
    const float a = img[(size_t)y0 * w + x0], b = img[(size_t)y0 * w + x1];
    const float c = img[(size_t)y1 * w + x0], e = img[(size_t)y1 * w + x1];
    const float top = a + (b - a) * fx;
    const float bot = c + (e - c) * fx;
    return top + (bot - top) * fy;
}

// Bilinear sample straight from the single-channel Image, clamp-to-edge. Avoids
// first copying the whole full-res grey into a vector (that copy was ~48MB and
// the dominant serial cost / a transient RAM spike, for the sake of sampling a
// few thousand downscaled points).
static inline float sample_img(const Image& g, float y, float x) {
    if (!(y >= 0.f && y <= (float)(g.h - 1) && x >= 0.f && x <= (float)(g.w - 1)))
        return 0.f;
    const int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    const int x1 = std::min(x0 + 1, g.w - 1), y1 = std::min(y0 + 1, g.h - 1);
    const float fx = x - (float)x0, fy = y - (float)y0;
    const float a = g.at(y0, x0), b = g.at(y0, x1);
    const float c = g.at(y1, x0), e = g.at(y1, x1);
    const float top = a + (b - a) * fx;
    const float bot = c + (e - c) * fx;
    return top + (bot - top) * fy;
}

// Isotropic bilinear downscale of a grey Image to th x tw, then a small
// separable Gaussian blur (ISA's DeBayerBWGaussWB blurs the tracking grey).
std::vector<float> resize_blur(const Image& g, int th, int tw) {
    std::vector<float> out((size_t)th * tw, 0.f);
    const float sx = (float)g.w / (float)tw;
    const float sy = (float)g.h / (float)th;
    for (int y = 0; y < th; ++y) {
        const float ry = ((float)y + 0.5f) * sy - 0.5f;
        for (int x = 0; x < tw; ++x) {
            const float rx = ((float)x + 0.5f) * sx - 0.5f;
            out[(size_t)y * tw + x] = sample_img(g, ry, rx);
        }
    }

    // 3-tap Gaussian (sigma ~0.85), separable, clamp-to-edge.
    const float k0 = 0.375f, k1 = 0.3125f;  // normalized {k1,k0,k1}
    std::vector<float> tmp((size_t)th * tw, 0.f);
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            const float l = out[(size_t)y * tw + std::max(0, x - 1)];
            const float c = out[(size_t)y * tw + x];
            const float r = out[(size_t)y * tw + std::min(tw - 1, x + 1)];
            tmp[(size_t)y * tw + x] = k1 * l + k0 * c + k1 * r;
        }
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x) {
            const float u = tmp[(size_t)std::max(0, y - 1) * tw + x];
            const float c = tmp[(size_t)y * tw + x];
            const float d = tmp[(size_t)std::min(th - 1, y + 1) * tw + x];
            out[(size_t)y * tw + x] = k1 * u + k0 * c + k1 * d;
        }

    // Remove the DC component. Raw cross-correlation of two non-zero-mean
    // images is dominated by the mean product at every shift, which swamps the
    // structural peak and makes the angle score (peak height) nearly flat. ISA
    // high-passes its tracking greys (FourierFilter) for the same reason; mean
    // subtraction is the minimal equivalent and sharpens both the shift peak and
    // the rotation discrimination.
    double mean = 0.0;
    for (float v : out) mean += v;
    mean /= (double)out.size();
    for (float& v : out) v -= (float)mean;
    return out;
}

// Rotate the th x tw content by `ang` about its centre (inverse-mapped bilinear)
// and write it into the top-left of an N x N complex buffer (imag 0, zeros
// elsewhere). Mirrors ISA's WarpAffine(RotAroundCenter(ang)) + FFT-of-ROI.
void rotate_into_padded(const std::vector<float>& src, int th, int tw, float ang,
                        int N, std::vector<cf>& dst) {
    std::fill(dst.begin(), dst.end(), cf(0.f, 0.f));
    const float cx = 0.5f * (float)(tw - 1);
    const float cy = 0.5f * (float)(th - 1);
    const float ca = std::cos(ang), sa = std::sin(ang);
    for (int oy = 0; oy < th; ++oy) {
        for (int ox = 0; ox < tw; ++ox) {
            const float rx = (float)ox - cx;
            const float ry = (float)oy - cy;
            const float sxp = cx + ca * rx + sa * ry;
            const float syp = cy - sa * rx + ca * ry;
            dst[(size_t)oy * N + ox] = cf(sample_bilinear(src, th, tw, syp, sxp), 0.f);
        }
    }
}

}  // namespace

bool estimate_isa_prealign(const Image& ref_grey, const Image& comp_grey,
                           const Config& cfg,
                           float& out_dx, float& out_dy, float& out_rot_rad) {
    out_dx = out_dy = out_rot_rad = 0.f;
    if (ref_grey.h <= 8 || ref_grey.w <= 8) return false;
    if (ref_grey.h != comp_grey.h || ref_grey.w != comp_grey.w) return false;

    const int cap = std::max(64, std::min(2048, cfg.isa_prealign_fft_max_dim));
    const double deg2rad = kPi / 180.0;
    const float incr = (float)(std::max(0.01f, cfg.isa_prealign_rot_incr_deg) * deg2rad);
    const float range = (float)(std::max(0.f, cfg.isa_prealign_rot_range_deg) * deg2rad);

    // One resolution's worth of state: the resized moving grey, the conjugated
    // reference spectrum (conj(ref) for the cross power spectrum), and the pad
    // geometry. Zero-padding to ~2x the content (N) makes the cross-correlation
    // LINEAR, so large shifts resolve without wrapping into the image -- the key
    // to pre-aligning big camera pans.
    struct Res {
        std::vector<cf> refC;
        std::vector<float> movR;
        int th = 0, tw = 0, N = 0;
        float factor = 1.f;
    };
    auto build_res = [&](int rcap) -> Res {
        Res r;
        const int longer = std::max(ref_grey.h, ref_grey.w);
        r.factor = (longer > rcap) ? (float)longer / (float)rcap : 1.f;
        r.th = std::max(8, (int)std::lround((double)ref_grey.h / r.factor));
        r.tw = std::max(8, (int)std::lround((double)ref_grey.w / r.factor));
        r.N = next_pow2(2 * std::max(r.th, r.tw));
        const std::vector<float> refR = resize_blur(ref_grey, r.th, r.tw);
        r.movR = resize_blur(comp_grey, r.th, r.tw);
        r.refC.assign((size_t)r.N * r.N, cf(0.f, 0.f));
        for (int y = 0; y < r.th; ++y)
            for (int x = 0; x < r.tw; ++x)
                r.refC[(size_t)y * r.N + x] = cf(refR[(size_t)y * r.tw + x], 0.f);
        fft2d(r.refC, r.N, /*inverse=*/false);
        for (auto& v : r.refC) v = std::conj(v);
        return r;
    };

    // Score candidate angles against a resolution. Each angle is an independent
    // rotate -> FFT -> conj-mul -> iFFT -> peak, run in parallel across
    // cfg.num_threads into per-angle slots (no contention); the winner is picked
    // serially (deterministic, tie -> lowest index). Each thread owns one N*N
    // scratch buffer, so peak scratch is num_threads * N^2 complex -- this runs
    // during analysis, below the merge memory peak, so it does not raise the
    // app's ceiling.
    auto scan = [&](const Res& r, const std::vector<float>& angs,
                    float& best_val, float& best_ang, int& best_px, int& best_py) {
        const int na = (int)angs.size();
        std::vector<float> vals((size_t)na, -std::numeric_limits<float>::infinity());
        std::vector<int> pxs((size_t)na, 0), pys((size_t)na, 0);
        parallel_rows(na, cfg.num_threads, [&](int i) {
            std::vector<cf> movC((size_t)r.N * r.N);
            rotate_into_padded(r.movR, r.th, r.tw, angs[(size_t)i], r.N, movC);
            fft2d(movC, r.N, /*inverse=*/false);
            for (size_t k = 0; k < movC.size(); ++k) movC[k] = r.refC[k] * movC[k];
            fft2d(movC, r.N, /*inverse=*/true);
            float vmax = -std::numeric_limits<float>::infinity();
            int px = 0, py = 0;
            for (int y = 0; y < r.N; ++y)
                for (int x = 0; x < r.N; ++x) {
                    const float v = movC[(size_t)y * r.N + x].real();
                    if (v > vmax) { vmax = v; px = x; py = y; }
                }
            vals[(size_t)i] = vmax; pxs[(size_t)i] = px; pys[(size_t)i] = py;
        });
        for (int i = 0; i < na; ++i)
            if (vals[(size_t)i] > best_val) {
                best_val = vals[(size_t)i]; best_ang = angs[(size_t)i];
                best_px = pxs[(size_t)i]; best_py = pys[(size_t)i];
            }
    };

    const float coarse_step = 5.f * incr;         // L2 / fine window step (1 deg @ 0.2)
    const float precoarse_step = 4.f * coarse_step;  // L1 step (4 deg)

    // Candidate angles center +/- halfwidth at `step`, clamped to [-range,range]
    // and de-duplicated (so a wide window on a narrow range collapses cleanly).
    auto gen = [&](float center, float halfwidth, float step) {
        std::vector<float> v;
        for (float a = center - halfwidth; a <= center + halfwidth + 0.5f * step; a += step) {
            const float c = std::max(-range, std::min(range, a));
            if (v.empty() || std::fabs(c - v.back()) > 1e-6f) v.push_back(c);
        }
        if (v.empty()) v.push_back(std::max(-range, std::min(range, center)));
        return v;
    };
    auto best_angle_of = [&](const Res& r, const std::vector<float>& angs) {
        float bv = -std::numeric_limits<float>::infinity(), ba = 0.f;
        int bx = 0, by = 0;
        scan(r, angs, bv, ba, bx, by);
        return std::isfinite(bv) ? ba : std::numeric_limits<float>::quiet_NaN();
    };

    // HIERARCHICAL coarse-to-fine, in both angle and RESOLUTION, so the cost is
    // ~flat as the roll range grows and most work is cheap:
    //   L1  cap/4  step 4deg  over +/-range     -> localise to ~+/-2deg (cheap, wide)
    //   L2  cap/2  step 1deg  over +/-4deg      -> localise to ~+/-0.5deg (bounded)
    //   L3  cap    step incr  over +/-1deg      -> final angle + translation (full res)
    // Only L3 runs at full resolution (the resolution the result was tuned at),
    // so adding range only grows the cheap L1 list; L2 and L3 stay fixed-size.
    float best_ang = 0.f;
    if (range > 1e-6f) {
        const Res r1 = build_res(std::max(48, cap / 4));
        best_ang = best_angle_of(r1, gen(0.f, range, precoarse_step));
        if (std::isnan(best_ang)) return false;
        const Res r2 = build_res(std::max(64, cap / 2));
        best_ang = best_angle_of(r2, gen(best_ang, precoarse_step, coarse_step));
        if (std::isnan(best_ang)) return false;
    }

    // Fine pass at full resolution around the localised angle.
    const Res rf = build_res(cap);
    const std::vector<float> fine =
        (range <= 1e-6f) ? std::vector<float>{0.f} : gen(best_ang, coarse_step, incr);
    float best_val = -std::numeric_limits<float>::infinity();
    float fine_ang = 0.f;
    int best_px = 0, best_py = 0;
    scan(rf, fine, best_val, fine_ang, best_px, best_py);
    if (!std::isfinite(best_val)) return false;

    // Wrap the circular-correlation peak into a signed shift (subtract the
    // period past the half size). The cross power spectrum conj(ref)*mov peaks
    // at the shift s with comp(p+s) ~= ref(p), i.e. comp(q) ~= ref(q-s); in the
    // make_global_initial_flow convention comp is sampled at p+(dx,dy), so
    // (dx,dy) = +(peakX,peakY) (NOT ISA's returned -peak, because that drives a
    // ref->comp warp here rather than ISA's comp->ref). Verified by the
    // synthetic-recovery test in tools/.
    int px = best_px, py = best_py;
    if (px > rf.N / 2) px -= rf.N;
    if (py > rf.N / 2) py -= rf.N;

    out_dx = (float)px * rf.factor;
    out_dy = (float)py * rf.factor;
    // scan rotates COMP to match ref, so the angle that aligns comp is the
    // negative of the port's `a` (which rotates ref-space into comp-space in
    // make_global_initial_flow). Verified by the synthetic-recovery test.
    out_rot_rad = -fine_ang;
    return true;
}

}  // namespace hhsr
