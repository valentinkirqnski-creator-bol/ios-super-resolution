#include "isa_prealign.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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

// Iterative radix-2 Cooley-Tukey, in place; a.size() must be a power of two.
// No 1/N scaling (applied once by the caller, as ISA divides by W*H).
void fft1d(std::vector<cf>& a, bool inverse) {
    const int n = (int)a.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[(size_t)i], a[(size_t)j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / len * (inverse ? 1.0 : -1.0);
        const cf wlen((float)std::cos(ang), (float)std::sin(ang));
        for (int i = 0; i < n; i += len) {
            cf w(1.f, 0.f);
            for (int k = 0; k < len / 2; ++k) {
                const cf u = a[(size_t)(i + k)];
                const cf v = a[(size_t)(i + k + len / 2)] * w;
                a[(size_t)(i + k)] = u + v;
                a[(size_t)(i + k + len / 2)] = u - v;
                w *= wlen;
            }
        }
    }
}

// Row-major N x N complex 2D FFT: transform rows, then columns.
void fft2d(std::vector<cf>& d, int N, bool inverse) {
    std::vector<cf> line((size_t)N);
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) line[(size_t)x] = d[(size_t)y * N + x];
        fft1d(line, inverse);
        for (int x = 0; x < N; ++x) d[(size_t)y * N + x] = line[(size_t)x];
    }
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) line[(size_t)y] = d[(size_t)y * N + x];
        fft1d(line, inverse);
        for (int y = 0; y < N; ++y) d[(size_t)y * N + x] = line[(size_t)y];
    }
}

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

// Isotropic bilinear downscale of a grey Image to th x tw, then a small
// separable Gaussian blur (ISA's DeBayerBWGaussWB blurs the tracking grey).
std::vector<float> resize_blur(const Image& g, int th, int tw) {
    std::vector<float> src((size_t)g.h * g.w);
    for (int y = 0; y < g.h; ++y)
        for (int x = 0; x < g.w; ++x)
            src[(size_t)y * g.w + x] = g.at(y, x);

    std::vector<float> out((size_t)th * tw, 0.f);
    const float sx = (float)g.w / (float)tw;
    const float sy = (float)g.h / (float)th;
    for (int y = 0; y < th; ++y) {
        const float ry = ((float)y + 0.5f) * sy - 0.5f;
        for (int x = 0; x < tw; ++x) {
            const float rx = ((float)x + 0.5f) * sx - 0.5f;
            out[(size_t)y * tw + x] = sample_bilinear(src, g.h, g.w, ry, rx);
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
    const int longer = std::max(ref_grey.h, ref_grey.w);
    const float factor = (longer > cap) ? (float)longer / (float)cap : 1.f;
    const int th = std::max(8, (int)std::lround((double)ref_grey.h / factor));
    const int tw = std::max(8, (int)std::lround((double)ref_grey.w / factor));
    // Zero-pad to ~2x the content so the cross-correlation is LINEAR, not
    // circular: the content sits in the top-left quadrant and the surrounding
    // zeros let a shift up to ~max(th,tw) thumbnail px (= that x factor raw px)
    // resolve without wrapping/aliasing into the image. Without this the long
    // axis has little or no padding and large translations fold back on
    // themselves, which is the main reason big camera pans were not pre-aligned.
    const int N = next_pow2(2 * std::max(th, tw));
    if (N < 8) return false;

    const std::vector<float> refR = resize_blur(ref_grey, th, tw);
    const std::vector<float> movR = resize_blur(comp_grey, th, tw);

    // Reference spectrum, conjugated once (cross power spectrum is conj(ref)*mov).
    std::vector<cf> refC((size_t)N * N, cf(0.f, 0.f));
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x)
            refC[(size_t)y * N + x] = cf(refR[(size_t)y * tw + x], 0.f);
    fft2d(refC, N, /*inverse=*/false);
    for (auto& v : refC) v = std::conj(v);

    const double deg2rad = kPi / 180.0;
    const float incr = (float)(std::max(0.01f, cfg.isa_prealign_rot_incr_deg) * deg2rad);
    const float range = (float)(std::max(0.f, cfg.isa_prealign_rot_range_deg) * deg2rad);

    std::vector<cf> movC((size_t)N * N);
    float best_val = -std::numeric_limits<float>::infinity();
    float best_ang = 0.f;
    int best_px = 0, best_py = 0;

    auto scan_angle = [&](float ang) {
        rotate_into_padded(movR, th, tw, ang, N, movC);
        fft2d(movC, N, /*inverse=*/false);
        for (size_t i = 0; i < movC.size(); ++i) movC[i] = refC[i] * movC[i];
        fft2d(movC, N, /*inverse=*/true);
        // Peak of the real part (1/N^2 scaling is a positive constant -> omit).
        float vmax = -std::numeric_limits<float>::infinity();
        int px = 0, py = 0;
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const float v = movC[(size_t)y * N + x].real();
                if (v > vmax) { vmax = v; px = x; py = y; }
            }
        if (vmax > best_val) { best_val = vmax; best_ang = ang; best_px = px; best_py = py; }
    };

    // Coarse pass: step 5*incr over +/-range, centred on 0 (no gyro seed).
    if (range <= 1e-6f) {
        scan_angle(0.f);
    } else {
        const float coarse_step = 5.f * incr;
        for (float a = -range; a <= range + 0.5f * coarse_step; a += coarse_step)
            scan_angle(a);
    }
    // Fine pass: step incr over +/-10*incr around the coarse best.
    {
        const float zero = best_ang;
        const float fine_range = 10.f * incr;
        for (float a = zero - fine_range; a <= zero + fine_range + 0.5f * incr; a += incr)
            scan_angle(a);
    }

    if (!std::isfinite(best_val)) return false;

    // Wrap the circular-correlation peak into a signed shift (subtract the
    // period past the half size). The cross power spectrum conj(ref)*mov peaks
    // at the shift s with comp(p+s) ~= ref(p), i.e. comp(q) ~= ref(q-s); in the
    // make_global_initial_flow convention comp is sampled at p+(dx,dy), so
    // (dx,dy) = +(peakX,peakY) (NOT ISA's returned -peak, because that drives a
    // ref->comp warp here rather than ISA's comp->ref). Verified by the
    // synthetic-recovery test in tools/.
    int px = best_px, py = best_py;
    if (px > N / 2) px -= N;
    if (py > N / 2) py -= N;

    out_dx = (float)px * factor;
    out_dy = (float)py * factor;
    // scan_angle rotates COMP to match ref, so the angle that aligns comp is the
    // negative of the port's `a` (which rotates ref-space into comp-space in
    // make_global_initial_flow). Verified by the synthetic-recovery test.
    out_rot_rad = -best_ang;
    return true;
}

}  // namespace hhsr
