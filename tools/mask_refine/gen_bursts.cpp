// Synthesise RAW bursts from one DNG and dump everything training needs.
//
// Everything here runs the REAL pipeline -- align(), compute_robustness(),
// merge_comp()/merge_ref() out of core/ -- rather than a reimplementation, so
// the mask the network learns to refine is the mask that actually ships and the
// merge it is scored through is the merge that actually runs.
//
// ASSUMPTIONS, stated rather than invented (spec section 6):
//
//  * Mask semantics. R is one channel on the GUIDE lattice, range [0, 1],
//    polarity ACCEPT: merge.cpp accumulates val += w * R * c and acc += w * R,
//    so R = 1 contributes the sample fully and R = 0 removes it. "Reject" below
//    always means driving R toward 0.
//  * Flow. One vector per alignment tile, (ny, nx, 2), in RAW pixels.
//  * The merge is LINEAR in R, separately per frame. So merging frame n alone
//    with R == 1 yields A_n = sum(w*c) and B_n = sum(w), and for any mask
//
//        out = (A_ref + sum_n A_n * R_n) / (B_ref + sum_n B_n * R_n)
//
//    is the exact output the pipeline would produce with that mask. That is why
//    the per-frame factors are dumped instead of a finished image: training can
//    then vary the mask without re-running the merge, and the arithmetic is the
//    pipeline's own, not a stand-in.
//
// The burst model covers what tile-based flow cannot represent: a smoothly
// varying camera field, piecewise-constant depth planes with a hard boundary
// (parallax), an independently moving object with occlusion on its leading edge,
// and per-tile jitter. The TRUE per-pixel displacement is kept so a ground-truth
// merge can be built from the same frames.
//
//   out/burst_NNN.bin   little-endian, f32 unless stated:
//     i32 n_comp, gh, gw, oh, ow, ny, nx, tile_size
//     f32 noise_gain
//     A_ref[3*oh*ow] B_ref[3*oh*ow]
//     gt  [3*oh*ow]                      ground-truth merge (true flow, clean)
//     per comparison frame n:
//       Rw   [gh*gw]                     the Wronski mask
//       A    [3*oh*ow]  B[3*oh*ow]       merge factors, R == 1
//       flow [ny*nx*2]                   the REAL aligner's estimate
//       ferr [gh*gw]                     |F_est - F_true| at that pixel
#include "stages.h"
#include "types.h"
#include "raw_io.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace hhsr;

namespace {

struct Warp {
    // Global similarity (camera): rotation about the frame centre plus shift.
    float theta = 0.f, tx = 0.f, ty = 0.f, cx = 0.f, cy = 0.f;
    // Parallax: displacement proportional to inverse depth, along the camera's
    // own translation direction. A tile straddling the depth edge cannot be
    // described by one vector, which is the case the mask has to catch.
    float par = 0.f;
    // Independent object motion inside a rectangle.
    float ox = 0.f, oy = 0.f;
    int rx0 = 0, ry0 = 0, rx1 = -1, ry1 = -1;
};

// Inverse depth in [0,1]: a far plane, a near plane, and a soft vertical ramp,
// so there are both hard depth boundaries and a continuously varying region.
float inv_depth(int y, int x, int h, int w) {
    const float fy = (float)y / (float)h, fx = (float)x / (float)w;
    float d = 0.15f + 0.5f * fy;                 // ramp: ground plane
    if (fx > 0.55f && fy > 0.30f) d = 0.95f;     // near object, hard edge
    if (fx < 0.25f && fy < 0.45f) d = 0.05f;     // far background
    return d;
}

// True displacement at a raw pixel, in raw pixels.
void true_disp(const Warp& m, int y, int x, int h, int w, float* dx, float* dy) {
    const float rx = (float)x - m.cx, ry = (float)y - m.cy;
    const float ct = std::cos(m.theta), st = std::sin(m.theta);
    float ux = ct * rx - st * ry + m.cx + m.tx - (float)x;
    float uy = st * rx + ct * ry + m.cy + m.ty - (float)y;
    if (m.par != 0.f) {
        const float n = std::sqrt(m.tx * m.tx + m.ty * m.ty) + 1e-6f;
        const float idp = inv_depth(y, x, h, w);
        ux += m.par * idp * (m.tx / n);
        uy += m.par * idp * (m.ty / n);
    }
    if (m.rx1 > m.rx0 && x >= m.rx0 && x < m.rx1 && y >= m.ry0 && y < m.ry1) {
        ux += m.ox;
        uy += m.oy;
    }
    *dx = ux;
    *dy = uy;
}

float sample_bilinear(const Image& im, float y, float x) {
    if (!(y >= 0.f && y <= (float)(im.h - 1) && x >= 0.f && x <= (float)(im.w - 1)))
        return 0.f;
    const int y0 = (int)y, x0 = (int)x;
    const int y1 = std::min(y0 + 1, im.h - 1), x1 = std::min(x0 + 1, im.w - 1);
    const float ty = y - (float)y0, tx = x - (float)x0;
    const float a = im.at(y0, x0) * (1.f - tx) + im.at(y0, x1) * tx;
    const float b = im.at(y1, x0) * (1.f - tx) + im.at(y1, x1) * tx;
    return a * (1.f - ty) + b * ty;
}

// Resample the reference by the warp. The CFA is preserved by sampling the
// mosaic itself: a Bayer plane displaced by a non-even offset would relabel the
// colour of every site, so the sample is taken from the same-parity lattice.
Image warp_raw(const Image& ref, const Warp& m) {
    Image out(ref.h, ref.w, 1);
    for (int y = 0; y < ref.h; ++y) {
        for (int x = 0; x < ref.w; ++x) {
            float dx, dy;
            true_disp(m, y, x, ref.h, ref.w, &dx, &dy);
            // Same-parity sampling: step in 2s from the nearest even anchor so
            // the colour under (y,x) is unchanged.
            const float sy = (float)y + dy, sx = (float)x + dx;
            const int py = y & 1, px = x & 1;
            const float ay = std::round((sy - (float)py) * 0.5f) * 2.f + (float)py;
            const float ax = std::round((sx - (float)px) * 0.5f) * 2.f + (float)px;
            const float fy = sy - ay, fx = sx - ax;
            // Bilinear between same-parity neighbours two apart.
            const float v00 = sample_bilinear(ref, ay, ax);
            const float v01 = sample_bilinear(ref, ay, ax + 2.f);
            const float v10 = sample_bilinear(ref, ay + 2.f, ax);
            const float v11 = sample_bilinear(ref, ay + 2.f, ax + 2.f);
            const float wx = std::min(std::max(fx * 0.5f, 0.f), 1.f);
            const float wy = std::min(std::max(fy * 0.5f, 0.f), 1.f);
            const float a = v00 * (1.f - wx) + v01 * wx;
            const float b = v10 * (1.f - wx) + v11 * wx;
            out.at(y, x) = a * (1.f - wy) + b * wy;
        }
    }
    return out;
}

// alpha and beta here must be the SAME numbers compute_robustness will use for
// this burst -- see scale_noise_profile below. They are passed in rather than
// read from a Config so the call site has to think about which it is passing.
void add_noise(Image& im, float alpha, float beta, std::mt19937& rng) {
    std::normal_distribution<float> g(0.f, 1.f);
    for (size_t i = 0; i < im.data.size(); ++i) {
        const float mu = std::max(im.data[i], 0.f);
        const float var = std::max(alpha * mu + beta, 0.f);
        im.data[i] = std::max(0.f, mu + std::sqrt(var) * g(rng));
    }
}

// Simulate a higher ISO the way a real camera does: by scaling the NOISE
// PROFILE, so the mask's sigma_t and the injected noise are the same model.
//
// Injecting alpha*gain*mu + beta*gain^2 while leaving the profile alone -- which
// is what this did first -- means the robustness stage assumes far less noise
// than is present, d^2/sigma^2 is inflated and the mask over-rejects purely as a
// function of gain. Measured before the fix: corr(gain, mean R) = -0.843, with
// mean R falling 0.944 -> 0.784 across the gain range while the flow error
// stayed flat. That is a GLOBAL miscalibration, so no spatial feature can
// explain it, and a refinement asked to fix it can only learn to do nothing.
void scale_noise_profile(Config& cfg, float gain) {
    for (int c = 0; c < 3; ++c) {
        cfg.alpha_dng[c] *= gain;
        cfg.beta_dng[c] *= gain * gain;
    }
}

// Is this crop worth a burst? Flat sky and blown highlights teach the mask
// nothing -- no detail to preserve and no structure to ghost -- and a run
// dominated by them looks like training while measuring almost nothing.
bool crop_is_useful(const Image& c) {
    double mean = 0.0;
    for (size_t i = 0; i < c.data.size(); ++i) mean += c.data[i];
    mean /= (double)c.data.size();
    // 0.004, not 0.015: a night scene sits around 0.012 mean, and the higher
    // floor threw away every crop from one -- which is the low-light content
    // the mask struggles with most. Only true black is excluded here; whether a
    // dark crop carries usable structure is the gradient test's job, and it
    // measures that relative to brightness precisely so a dim scene can pass.
    if (mean < 0.004 || mean > 0.85) return false;
    // Gradient energy measured on the SAME-PARITY lattice (step 2), so the
    // CFA's own checkerboard is not counted as detail.
    double g = 0.0;
    int n = 0;
    for (int y = 2; y < c.h - 2; y += 3) {
        for (int x = 2; x < c.w - 2; x += 3) {
            g += std::fabs(c.at(y, x) - c.at(y, x + 2));
            g += std::fabs(c.at(y, x) - c.at(y + 2, x));
            n += 2;
        }
    }
    g /= (double)std::max(n, 1);
    // Relative to brightness: a dim but textured crop is useful, a bright flat
    // one is not.
    return (g / std::max(mean, 1e-6)) > 0.045;
}

void wr_i32(std::FILE* f, int32_t v) { std::fwrite(&v, 4, 1, f); }
void wr_f32(std::FILE* f, float v) { std::fwrite(&v, 4, 1, f); }
void wr_img(std::FILE* f, const Image& im) {
    std::fwrite(im.data.data(), sizeof(f32), im.data.size(), f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: gen_bursts a.dng[,b.dng,...] out_dir n_bursts "
                    "[n_frames] [crop]\n");
        return 2;
    }
    // Comma-separated sources, so one run spans several scenes.
    std::vector<std::string> dng_paths;
    {
        std::string cur;
        for (const char* p = argv[1]; *p; ++p) {
            if (*p == ',') { if (!cur.empty()) dng_paths.push_back(cur); cur.clear(); }
            else cur += *p;
        }
        if (!cur.empty()) dng_paths.push_back(cur);
    }
    const std::string out = argv[2];
    const int n_bursts = std::atoi(argv[3]);
    const int n_frames = (argc > 4) ? std::atoi(argv[4]) : 5;
    const int crop = (argc > 5) ? std::atoi(argv[5]) : 1024;

    Config cfg;
    std::vector<Image> sources;
    for (size_t i = 0; i < dng_paths.size(); ++i) {
        Image im = load_raw_frame(dng_paths[i], cfg, sources.empty(), 0, 0);
        if (im.h <= 0) {
            std::printf("could not decode %s -- skipping\n", dng_paths[i].c_str());
            continue;
        }
        std::printf("source %-26s %dx%d\n", dng_paths[i].c_str(), im.h, im.w);
        sources.push_back(std::move(im));
    }
    if (sources.empty()) { std::printf("no usable sources\n"); return 1; }
    std::printf("%d sources, bayer=%d, scale=%.1f, crop %d\n",
                (int)sources.size(), (int)cfg.bayer_mode, cfg.scale, crop);
    const int ts = cfg.bm_tile_sizes.empty() ? 16 : cfg.bm_tile_sizes[0];

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    int rejected = 0;

    for (int b = 0; b < n_bursts; ++b) {
        // Round-robin over the sources so every scene is represented evenly
        // whatever n_bursts is, and STRATIFIED placement within each: the frame
        // is split into a grid and successive bursts from one source take
        // different cells, so the crops span the picture instead of clustering
        // wherever the random draws happened to land.
        const Image& full = sources[(size_t)(b % (int)sources.size())];
        const int rep = b / (int)sources.size();
        const int gx = std::max(1, (full.w - 4) / crop);
        const int gy = std::max(1, (full.h - 4) / crop);
        const int cell = rep % (gx * gy);
        const int cxi = cell % gx, cyi = cell / gx;
        const int span_x = std::max(1, (full.w - crop - 4) / gx);
        const int span_y = std::max(1, (full.h - crop - 4) / gy);

        Image ref(crop, crop, 1);
        bool usable = false;
        for (int attempt = 0; attempt < 12 && !usable; ++attempt) {
            // Even origin: an odd offset relabels the CFA.
            int y0 = cyi * span_y + (int)(U(rng) * (float)span_y);
            int x0 = cxi * span_x + (int)(U(rng) * (float)span_x);
            y0 = std::min(std::max((y0 / 2) * 2, 0), full.h - crop - 2);
            x0 = std::min(std::max((x0 / 2) * 2, 0), full.w - crop - 2);
            for (int y = 0; y < crop; ++y)
                for (int x = 0; x < crop; ++x) ref.at(y, x) = full.at(y0 + y, x0 + x);
            usable = crop_is_useful(ref);
            if (!usable) ++rejected;
        }
        if (!usable) {
            std::printf("  burst %3d: no useful crop in this cell, skipped\n", b);
            continue;
        }

        // Burst regime. Every burst gets subpixel camera motion; the extra
        // modes are what tile flow cannot represent.
        const float gain = std::exp(U(rng) * std::log(12.f));
        const int regime = (int)(U(rng) * 4.f);   // 0 clean 1 parallax 2 object 3 rot
        std::vector<Warp> warps((size_t)n_frames);
        for (int n = 1; n < n_frames; ++n) {
            Warp& m = warps[(size_t)n];
            m.cx = 0.5f * (float)crop;
            m.cy = 0.5f * (float)crop;
            m.tx = (U(rng) * 2.f - 1.f) * 3.0f * (float)n;
            m.ty = (U(rng) * 2.f - 1.f) * 3.0f * (float)n;
            if (regime == 3) m.theta = (U(rng) * 2.f - 1.f) * 0.004f * (float)n;
            if (regime == 1) m.par = (U(rng) * 6.f + 2.f) * (float)n / (float)n_frames;
            if (regime == 2) {
                m.rx0 = crop / 4; m.ry0 = crop / 4;
                m.rx1 = m.rx0 + crop / 3; m.ry1 = m.ry0 + crop / 3;
                m.ox = (U(rng) * 2.f - 1.f) * 5.f * (float)n;
                m.oy = (U(rng) * 2.f - 1.f) * 5.f * (float)n;
            }
        }

        // THIS burst's noise model. The profile is scaled by the gain and every
        // stage below -- the noise injection, init_robustness, the mask, the
        // kernels, the merge -- reads it from here, so the model that generated
        // the noise and the model the mask assumes are the same object.
        Config bcfg = cfg;
        scale_noise_profile(bcfg, gain);

        // Clean and noisy realisations of the same frames: the clean ones build
        // the ground-truth merge, the noisy ones are what the pipeline sees.
        std::vector<Image> clean((size_t)n_frames), noisy((size_t)n_frames);
        for (int n = 0; n < n_frames; ++n) {
            clean[(size_t)n] = (n == 0) ? ref : warp_raw(ref, warps[(size_t)n]);
            noisy[(size_t)n] = clean[(size_t)n];
            add_noise(noisy[(size_t)n], bcfg.noise_alpha(), bcfg.noise_beta(), rng);
        }

        // ---- the real pipeline on the noisy burst --------------------------
        Image ref_grey = compute_grey(noisy[0], bcfg.bayer_mode, bcfg.grey_method);
        Image ref_pad = pad_image_circular(ref_grey, bcfg.grey_tile_size(ts));
        Pyramid ref_pyr = build_pyramid(ref_pad, bcfg.bm_factors);
        RefStats rs = init_robustness(noisy[0], bcfg);
        CovField cov_ref = estimate_kernels(noisy[0], bcfg);

        const int oh = (int)std::lround(bcfg.scale * (float)crop);
        const int ow = oh;
        Image A_ref(oh, ow, 3), B_ref(oh, ow, 3);
        merge_ref(noisy[0], cov_ref, A_ref, B_ref, bcfg);

        // Ground truth: the SAME merge over the CLEAN frames with the TRUE
        // per-pixel displacement and nothing rejected.
        Image G_num(oh, ow, 3), G_den(oh, ow, 3);
        CovField cov_rc = estimate_kernels(clean[0], bcfg);
        merge_ref(clean[0], cov_rc, G_num, G_den, bcfg);

        char path[512];
        std::snprintf(path, sizeof(path), "%s/burst_%03d.bin", out.c_str(), b);
        std::FILE* f = std::fopen(path, "wb");
        if (!f) { std::printf("cannot write %s\n", path); return 1; }

        // Header is written after the per-frame loop needs ny/nx, so run one
        // alignment first to learn the grid.
        FlowField probe = align(ref_pyr, ref_grey,
                                compute_grey(noisy[1], bcfg.bayer_mode, bcfg.grey_method),
                                bcfg, ts);
        probe = flow_to_raw_tile_grid(probe, crop, crop, ref_grey.h, ref_grey.w,
                                      ts, bcfg.r_Mt, bcfg.num_threads,
                                      bcfg.grey_tile_size(ts));
        const int ny = probe.ny, nx = probe.nx;

        wr_i32(f, n_frames - 1); wr_i32(f, rs.means.h); wr_i32(f, rs.means.w);
        wr_i32(f, oh); wr_i32(f, ow); wr_i32(f, ny); wr_i32(f, nx); wr_i32(f, ts);
        wr_f32(f, gain);
        wr_img(f, A_ref); wr_img(f, B_ref);

        // Ground-truth comparison frames, true flow, R == 1.
        for (int n = 1; n < n_frames; ++n) {
            FlowField tf;
            tf.ny = ny; tf.nx = nx;
            tf.flow.assign((size_t)ny * nx * 2, 0.f);
            for (int ty = 0; ty < ny; ++ty) {
                for (int tx = 0; tx < nx; ++tx) {
                    // One vector per tile is all the merge accepts, so the
                    // ground truth uses the TILE-CENTRE true displacement. The
                    // residual within a tile is exactly the error the mask is
                    // being asked to detect, so it must not be removed here.
                    const int cy = std::min(crop - 1, ty * ts + ts / 2);
                    const int cx = std::min(crop - 1, tx * ts + ts / 2);
                    float dx, dy;
                    true_disp(warps[(size_t)n], cy, cx, crop, crop, &dx, &dy);
                    // NEGATED. warp_raw builds comp(p) = ref(p + d), so the
                    // reference content at p sits in comp at p - d, and
                    // merge.cpp samples comp at (reference position + flow).
                    // The flow the pipeline wants is therefore -d, not d.
                    tf.flow[((size_t)ty * nx + tx) * 2 + 0] = -dx;
                    tf.flow[((size_t)ty * nx + tx) * 2 + 1] = -dy;
                }
            }
            CovField cv = estimate_kernels(clean[(size_t)n], bcfg);
            Image ones(rs.means.h, rs.means.w, 1);
            std::fill(ones.data.begin(), ones.data.end(), 1.f);
            merge_comp(clean[(size_t)n], tf, cv, ones, ts, G_num, G_den, bcfg);
        }
        Image gt(oh, ow, 3);
        for (size_t i = 0; i < gt.data.size(); ++i)
            gt.data[i] = G_num.data[i] / std::max(G_den.data[i], 1e-8f);
        wr_img(f, gt);

        // ---- per comparison frame -----------------------------------------
        for (int n = 1; n < n_frames; ++n) {
            Image cg = compute_grey(noisy[(size_t)n], bcfg.bayer_mode, bcfg.grey_method);
            FlowField fl = align(ref_pyr, ref_grey, cg, bcfg, ts);
            fl = flow_to_raw_tile_grid(fl, crop, crop, ref_grey.h, ref_grey.w, ts,
                                       bcfg.r_Mt, bcfg.num_threads,
                                       bcfg.grey_tile_size(ts));
            Image Rw = compute_robustness(noisy[(size_t)n], rs, fl, ts, bcfg, nullptr);
            if (Rw.h != rs.means.h || Rw.w != rs.means.w) {
                std::printf("burst %d frame %d: mask %dx%d, expected %dx%d\n",
                            b, n, Rw.h, Rw.w, rs.means.h, rs.means.w);
                return 1;
            }
            Image A(oh, ow, 3), B(oh, ow, 3);
            Image ones(Rw.h, Rw.w, 1);
            std::fill(ones.data.begin(), ones.data.end(), 1.f);
            CovField cv = estimate_kernels(noisy[(size_t)n], bcfg);
            merge_comp(noisy[(size_t)n], fl, cv, ones, ts, A, B, bcfg);

            // |F_est - F_true| per MASK pixel. Training only; it is a diagnostic
            // and an optional oracle, never an input.
            Image ferr(Rw.h, Rw.w, 1);
            const float gsc = (float)crop / (float)Rw.w;   // guide -> raw
            for (int y = 0; y < Rw.h; ++y) {
                for (int x = 0; x < Rw.w; ++x) {
                    const int ry = std::min(crop - 1, (int)((float)y * gsc));
                    const int rx = std::min(crop - 1, (int)((float)x * gsc));
                    const int ty = std::min(ny - 1, ry / ts), tx = std::min(nx - 1, rx / ts);
                    float dxt, dyt;
                    true_disp(warps[(size_t)n], ry, rx, crop, crop, &dxt, &dyt);
                    // Same negation as the ground-truth field above.
                    const float ex = fl.flow[((size_t)ty * nx + tx) * 2 + 0] + dxt;
                    const float ey = fl.flow[((size_t)ty * nx + tx) * 2 + 1] + dyt;
                    ferr.at(y, x) = std::sqrt(ex * ex + ey * ey);
                }
            }

            wr_img(f, Rw);
            wr_img(f, A); wr_img(f, B);
            std::fwrite(fl.flow.data(), sizeof(f32), fl.flow.size(), f);
            wr_img(f, ferr);
        }
        std::fclose(f);
        clear_align_ref_ica_cache();
        std::printf("  burst %3d/%d  regime %d  gain %5.2f  mask %dx%d  out %dx%d\n",
                    b + 1, n_bursts, regime, gain, rs.means.h, rs.means.w, oh, ow);
        std::fflush(stdout);
    }
    std::printf("wrote %d bursts to %s (%d crops rejected as flat or blown)\n",
                n_bursts, out.c_str(), rejected);
    return 0;
}
