#include "stages.h"
#include "parallel.h"

namespace hhsr {

Image compute_grey_decimate(const Image& raw, bool bayer_mode) {
    if (!bayer_mode) return raw;
    int gh = raw.h / 2, gw = raw.w / 2;
    Image grey(gh, gw, 1);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            f32 s = raw.at(2 * y, 2 * x) + raw.at(2 * y, 2 * x + 1) +
                    raw.at(2 * y + 1, 2 * x) + raw.at(2 * y + 1, 2 * x + 1);
            grey.at(y, x) = 0.25f * s;
        }
    }
    return grey;
}

// 5-tap binomial kernel (1 4 6 4 1)/16, applied separably. Matches the
// Gaussian pyramid downsampling used by the reference (cuda_downsample).
static Image downsample_by(const Image& src, int factor) {
    if (factor <= 1) return src;
    // Blur then subsample. Use repeated binomial blur proportional to factor.
    Image blurred = gaussian_blur(src, 0.5f * factor);
    int dh = src.h / factor, dw = src.w / factor;
    Image out(dh, dw, 1);
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x)
            out.at(y, x) = blurred.at(y * factor, x * factor);
    return out;
}

Pyramid build_pyramid(const Image& grey, const std::vector<int>& factors) {
    Pyramid pyr;
    // factors are defined fine-to-coarse (e.g. {1,2,4,4}); build absolute
    // downsample factors and the images at each level.
    Image cur = grey;
    int abs_factor = 1;
    for (size_t i = 0; i < factors.size(); ++i) {
        int f = factors[i];
        cur = (f == 1 && i == 0) ? grey : downsample_by(cur, f);
        abs_factor *= f;
        pyr.levels.push_back(cur);
        pyr.abs_factors.push_back(abs_factor);
    }
    return pyr;
}

Image compute_gradients(const Image& grey) {
    // Matches kernels.py: separable [-0.5,0.5] pair producing a 2-channel
    // gradient (gx, gy), reducing the image by 1 pixel in each direction.
    int gh = grey.h - 1, gw = grey.w - 1;
    if (gh < 1 || gw < 1) return Image(0, 0, 2);
    Image grad(gh, gw, 2);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            f32 tl = grey.at(y, x),     tr = grey.at(y, x + 1);
            f32 bl = grey.at(y + 1, x), br = grey.at(y + 1, x + 1);
            // gx = horizontal diff averaged over rows; gy = vertical diff.
            grad.at(y, x, 0) = 0.5f * ((tr - tl) + (br - bl));
            grad.at(y, x, 1) = 0.5f * ((bl - tl) + (br - tr));
        }
    }
    return grad;
}

Image compute_edge_strength_map(const Image& grey) {
    Image grad = compute_gradients(grey);
    Image edge(grey.h, grey.w, 1);
    for (int y = 0; y < grey.h; ++y) {
        for (int x = 0; x < grey.w; ++x) {
            f32 mag = 0.f;
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dx = 0; dx <= 1; ++dx) {
                    int gy = std::min(y + dy, grad.h - 1);
                    int gx = std::min(x + dx, grad.w - 1);
                    f32 gxv = grad.at(gy, gx, 0);
                    f32 gyv = grad.at(gy, gx, 1);
                    mag = std::max(mag, std::sqrt(gxv * gxv + gyv * gyv));
                }
            }
            edge.at(y, x) = mag;
        }
    }
    return edge;
}

Image gaussian_blur(const Image& src, float sigma) {
    if (sigma <= 0.f) return src;
    int radius = std::max(1, (int)std::ceil(3.f * sigma));
    std::vector<f32> k(2 * radius + 1);
    f32 sum = 0.f;
    for (int i = -radius; i <= radius; ++i) {
        f32 v = std::exp(-0.5f * (i * i) / (sigma * sigma));
        k[i + radius] = v; sum += v;
    }
    for (auto& v : k) v /= sum;

    Image tmp(src.h, src.w, src.c);
    // Horizontal pass (clamp-to-edge borders).
    for (int y = 0; y < src.h; ++y) {
        for (int x = 0; x < src.w; ++x) {
            for (int ch = 0; ch < src.c; ++ch) {
                f32 acc = 0.f;
                for (int i = -radius; i <= radius; ++i) {
                    int xx = clampf((f32)(x + i), 0.f, (f32)(src.w - 1));
                    acc += k[i + radius] * src.at(y, xx, ch);
                }
                tmp.at(y, x, ch) = acc;
            }
        }
    }
    Image out(src.h, src.w, src.c);
    // Vertical pass.
    for (int y = 0; y < src.h; ++y) {
        for (int x = 0; x < src.w; ++x) {
            for (int ch = 0; ch < src.c; ++ch) {
                f32 acc = 0.f;
                for (int i = -radius; i <= radius; ++i) {
                    int yy = clampf((f32)(y + i), 0.f, (f32)(src.h - 1));
                    acc += k[i + radius] * tmp.at(yy, x, ch);
                }
                out.at(y, x, ch) = acc;
            }
        }
    }
    return out;
}

} // namespace hhsr
