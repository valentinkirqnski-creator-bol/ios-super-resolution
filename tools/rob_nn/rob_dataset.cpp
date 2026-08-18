// Training-set generator for a learned robustness mask.
//
// The classical mask (robustness.cpp, Wronski Eq. 5-9) decides whether to
// merge a pixel from d^2/sigma^2 -- a colour difference measured on 3x3 means
// of the half-res guide. That statistic is blind to the failure mode this
// burst actually suffers from: a tile whose flow is badly wrong but whose
// wrongly-fetched content happens to look like the right content (flat
// shadow onto flat shadow), or whose error lives in structure finer than the
// 6x6-raw support of the means (thin rods).
//
// To train something better we need a label that says what the mask cannot
// measure: did the alignment fetch the RIGHT CONTENT? That is knowable only
// with ground-truth motion, so this tool synthesises the comparison frames
// instead of using the captured ones:
//
//   1. take a real reference RAW (real sensor, real scene, real noise floor)
//   2. warp it by a KNOWN rotation+translation, resampling each CFA colour on
//      its own half-res lattice so the Bayer phase is preserved, and add
//      heteroscedastic noise from the DNG model -- a synthetic comparison
//      frame whose true flow is known analytically at every pixel
//   3. run the REAL aligner on it, producing the same estimated flow (and the
//      same errors) the app would produce
//   4. label = |comp(p + flow_est) - comp(p + flow_true)|, i.e. what we
//      fetched vs what we should have fetched, BOTH from the comparison
//      frame.
//
// That last point is what makes the label correct rather than circular.
// Comparing the fetch against the REFERENCE would punish aliasing -- the
// legitimate frame-to-frame difference super-resolution exists to exploit --
// and a net trained on it would learn to reject exactly the signal we want.
// Comparing the fetch against the correct fetch is zero whenever the flow is
// right, however aliased the content is, and grows only with genuine
// misalignment.
//
// Emits, at guide resolution: 13 input channels (what a mask could see) and
// 2 target channels (harm, and the ideal R derived from it), as flat float32
// for numpy. See tools/rob_nn/train.py.
#include "stages.h"
#include "parallel.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace hhsr;

namespace {

// Bilinear sample of one CFA colour plane. The pixels of a given Bayer phase
// (oy, ox) form a regular half-resolution lattice; sampling within that
// lattice keeps the synthetic frame a true Bayer image rather than a
// demosaic-remosaic round trip, which would erase the aliasing the algorithm
// depends on.
float sample_phase(const Image& raw, float sy, float sx, int oy, int ox) {
    const float fi = (sy - (float)oy) * 0.5f;
    const float fj = (sx - (float)ox) * 0.5f;
    const int i0 = (int)std::floor(fi), j0 = (int)std::floor(fj);
    const float ai = fi - (float)i0, aj = fj - (float)j0;
    const int imax = (raw.h - 1 - oy) / 2, jmax = (raw.w - 1 - ox) / 2;
    auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
    const int i1 = cl(i0 + 1, imax), j1 = cl(j0 + 1, jmax);
    const int ic = cl(i0, imax), jc = cl(j0, jmax);
    auto at = [&](int i, int j) {
        return raw.at(2 * i + oy, 2 * j + ox);
    };
    const float top = at(ic, jc) + (at(ic, j1) - at(ic, jc)) * aj;
    const float bot = at(i1, jc) + (at(i1, j1) - at(i1, jc)) * aj;
    return top + (bot - top) * ai;
}

struct Xform {
    float cos_t, sin_t, cy, cx, ty, tx;
    // Forward: where does reference content at p end up in the comp frame.
    // This is exactly what the merge means by "flow": it fetches comp at
    // (p + flow), so flow_true(p) = T(p) - p.
    void fwd(float y, float x, float& oy, float& ox) const {
        const float dy = y - cy, dx = x - cx;
        oy = cos_t * dy - sin_t * dx + cy + ty;
        ox = sin_t * dy + cos_t * dx + cx + tx;
    }
    // Inverse, used to synthesise: comp(q) = ref(T^-1 q).
    void inv(float y, float x, float& oy, float& ox) const {
        const float dy = y - cy - ty, dx = x - cx - tx;
        oy = cos_t * dy + sin_t * dx + cy;
        ox = -sin_t * dy + cos_t * dx + cx;
    }
};

// A rectangle of the comparison frame filled with content copied from
// elsewhere in the reference: a stand-in for the scene changing between
// frames -- an object moving in, an occlusion, a light flicker. Without
// these the training set contains only camera motion, and a net trained on
// it would learn "the flow looks plausible, therefore merge", which is
// exactly the mistake that ghosts a moving subject.
struct OccRect { int y0, y1, x0, x1, sy, sx; };

bool in_rect(const std::vector<OccRect>& rs, float y, float x, int* which = nullptr) {
    for (size_t i = 0; i < rs.size(); ++i) {
        const OccRect& r = rs[i];
        if (y >= (float)r.y0 && y < (float)r.y1 && x >= (float)r.x0 && x < (float)r.x1) {
            if (which) *which = (int)i;
            return true;
        }
    }
    return false;
}

Image synth_frame(const Image& ref, const Xform& X, const Config& cfg, uint32_t seed,
                  const std::vector<OccRect>& occ) {
    Image out(ref.h, ref.w, 1);
    const float alpha = cfg.noise_alpha(), beta = cfg.noise_beta();
    // Deterministic per row so thread count can never change the data.
    parallel_rows(ref.h, cfg.num_threads, [&](int y) {
        std::mt19937 rng(seed * 2654435761u + (uint32_t)y);
        std::normal_distribution<float> gauss(0.f, 1.f);
        for (int x = 0; x < ref.w; ++x) {
            float sy, sx;
            int oi = 0;
            if (in_rect(occ, (float)y, (float)x, &oi)) {
                // Unrelated content, kept on the same Bayer phase so the
                // patch is still a valid mosaic rather than a colour shift.
                sy = (float)(y + occ[oi].sy);
                sx = (float)(x + occ[oi].sx);
            } else {
                X.inv((float)y, (float)x, sy, sx);
            }
            float v = sample_phase(ref, sy, sx, y & 1, x & 1);
            // Heteroscedastic sensor noise: var = alpha*signal + beta. The
            // reference already carries its own noise, so this is the second
            // independent realisation a real second exposure would have.
            const float var = std::max(alpha * std::max(v, 0.f) + beta, 0.f);
            v += gauss(rng) * std::sqrt(var);
            out.at(y, x) = std::max(v, 0.f);
        }
    });
    return out;
}

// Guide-resolution 3x3 mean/variance, matching robustness.cpp's local_stats_3x3.
void local_stats(const Image& g, Image& means, Image& vars) {
    means = Image(g.h, g.w, g.c);
    vars = Image(g.h, g.w, g.c);
    for (int ch = 0; ch < g.c; ++ch)
        for (int y = 0; y < g.h; ++y)
            for (int x = 0; x < g.w; ++x) {
                float s = 0.f, s2 = 0.f;
                for (int i = -1; i <= 1; ++i) {
                    const int yy = std::min(std::max(y + i, 0), g.h - 1);
                    for (int j = -1; j <= 1; ++j) {
                        const int xx = std::min(std::max(x + j, 0), g.w - 1);
                        const float v = g.at(yy, xx, ch);
                        s += v; s2 += v * v;
                    }
                }
                const float m = s / 9.f;
                means.at(y, x, ch) = m;
                vars.at(y, x, ch) = std::max(s2 / 9.f - m * m, 0.f);
            }
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        std::printf("usage: rob_dataset out_prefix ref.dng n_frames [more_refs.dng...]\n");
        return 1;
    }
    const std::string out_prefix = argv[1];
    const int n_frames = std::atoi(argv[3]);
    std::vector<std::string> refs;
    refs.push_back(argv[2]);
    for (int i = 4; i < argc; ++i) refs.push_back(argv[i]);

    // Capture-style motion. Defaults span the range this burst actually
    // exhibits -- deliberate camera rotation, not the ~1e-3 rad hand tremor
    // the published tuning was validated on -- because a net trained only on
    // tremor would never see the failure we are trying to catch.
    float rot_max_deg = 1.2f, shift_max_px = 60.f;
    if (const char* v = std::getenv("ROB_ROT_DEG")) rot_max_deg = (float)std::atof(v);
    if (const char* v = std::getenv("ROB_SHIFT_PX")) shift_max_px = (float)std::atof(v);

    size_t total_written = 0;
    const std::string bin_path = out_prefix + ".f32";
    FILE* fout = std::fopen(bin_path.c_str(), "wb");
    if (!fout) { std::printf("cannot open %s\n", bin_path.c_str()); return 1; }
    int gh_all = 0, gw_all = 0;

    for (size_t ri = 0; ri < refs.size(); ++ri) {
        Config cfg;
        cfg.scale = 2.f;
        cfg.bayer_mode = true;
        cfg.grey_method = GreyMethod::Decimate;
        cfg.r_t = 0.12f; cfg.r_s1 = 2.0f; cfg.r_s2 = 12.0f; cfg.r_Mt = 0.8f;
        cfg.num_threads = 0;
        cfg.alignment_tile_size = 16;

        std::printf("[ref %zu] decoding %s\n", ri, refs[ri].c_str());
        Image ref = load_raw_frame(refs[ri], cfg, true, 0, 0);
        if (ref.w <= 0) { std::printf("  decode failed\n"); continue; }
        Config work = cfg;
        work.burst_frame_count = n_frames + 1;
        tune_config_snr(ref, work);
        work.r_t = cfg.r_t; work.r_s1 = cfg.r_s1; work.r_s2 = cfg.r_s2;
        const int ts = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];

        Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
        Pyramid ref_pyr = build_pyramid(ref_grey, work.bm_factors);
        Image ref_guide = compute_guide(ref, work);
        Image ref_means, ref_vars;
        local_stats(ref_guide, ref_means, ref_vars);
        const int gh = ref_guide.h, gw = ref_guide.w;
        gh_all = gh; gw_all = gw;
        std::printf("[ref %zu] %dx%d raw, guide %dx%d, ts=%d\n", ri, ref.w, ref.h, gw, gh, ts);

        for (int fi = 0; fi < n_frames; ++fi) {
            const uint32_t seed = (uint32_t)(ri * 1000 + fi + 1);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            std::uniform_real_distribution<float> u01(0.f, 1.f);

            // Mix the motion regimes rather than training on rotation alone:
            // a third of the frames are near-still hand tremor (the regime the
            // published tuning was validated on), a third moderate, a third the
            // deliberate rotation this burst actually exhibits. A net that only
            // ever saw large motion would treat small motion as suspicious.
            // Systematic coverage rather than random draws, so every
            // combination is guaranteed present in every reference's frames
            // instead of merely likely. Cycling on the frame index crosses
            // four motion levels -- including a genuinely static camera, where
            // the only frame-to-frame difference is noise and aliasing and the
            // mask must NOT reject -- with occlusion counts from none to many.
            static const float kRot[6]   = {0.f, 0.05f, 0.4f, 1.f, 1.f, 0.4f};
            static const float kShift[6] = {0.f, 0.03f, 0.4f, 1.f, 1.f, 0.4f};
            static const int   kOcc[6]   = {0,   0,     2,    0,   4,   6};
            const int cfg_i = fi % 6;
            const float rot_scale = kRot[cfg_i];
            const float sh_scale  = kShift[cfg_i];
            Xform X;
            const float theta = u(rng) * rot_max_deg * rot_scale * (float)M_PI / 180.f;
            X.cos_t = std::cos(theta); X.sin_t = std::sin(theta);
            X.cy = 0.5f * (float)ref.h; X.cx = 0.5f * (float)ref.w;
            X.ty = u(rng) * shift_max_px * sh_scale;
            X.tx = u(rng) * shift_max_px * sh_scale;

            // Scene changes: a handful of rectangles showing unrelated content.
            std::vector<OccRect> occ;
            const int n_occ = kOcc[cfg_i];
            for (int k = 0; k < n_occ; ++k) {
                const int hgt = 64 + (int)(u01(rng) * 400.f);
                const int wid = 64 + (int)(u01(rng) * 400.f);
                OccRect r;
                r.y0 = (int)(u01(rng) * (float)(ref.h - hgt - 1));
                r.x0 = (int)(u01(rng) * (float)(ref.w - wid - 1));
                r.y0 &= ~1; r.x0 &= ~1;          // keep Bayer phase
                r.y1 = r.y0 + hgt; r.x1 = r.x0 + wid;
                // Source offset stays even and in bounds for the whole rect.
                r.sy = (((int)(u(rng) * 600.f)) & ~1);
                r.sx = (((int)(u(rng) * 600.f)) & ~1);
                if (r.y0 + r.sy < 0) r.sy = -r.y0;
                if (r.y1 + r.sy >= ref.h) r.sy = ref.h - 1 - r.y1;
                if (r.x0 + r.sx < 0) r.sx = -r.x0;
                if (r.x1 + r.sx >= ref.w) r.sx = ref.w - 1 - r.x1;
                r.sy &= ~1; r.sx &= ~1;
                occ.push_back(r);
            }

            Image comp = synth_frame(ref, X, work, seed, occ);
            Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
            FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, ts, 0.f, 0.f, 0.f);
            flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w,
                                         ts, work.r_Mt, work.num_threads,
                                         work.grey_tile_size(ts));

            // Deliberate flow corruption across the full error spectrum.
            // Left to itself the aligner produces whichever errors it happens
            // to make, which under-samples both ends: near-perfect tiles and
            // catastrophic ones. Training needs the whole range, and needs it
            // balanced, or the net only learns to score the middle. The
            // magnitudes below span "indistinguishable from correct" to the
            // 100+ px failures measured on this burst.
            {
                std::mt19937 crng(seed * 7919u + 13u);
                std::uniform_real_distribution<float> c01(0.f, 1.f);
                std::uniform_real_distribution<float> ang(0.f, 6.2831853f);
                long long n_cor = 0;
                for (int ty = 0; ty < flow.ny; ++ty)
                    for (int tx = 0; tx < flow.nx; ++tx) {
                        const float k = c01(crng);
                        float mag;
                        if      (k < 0.45f) continue;              // leave as aligned
                        else if (k < 0.60f) mag = 0.3f + c01(crng) * 1.7f;    // subpixel..2px
                        else if (k < 0.75f) mag = 2.f + c01(crng) * 8.f;      // 2..10px
                        else if (k < 0.90f) mag = 10.f + c01(crng) * 40.f;    // 10..50px
                        else                mag = 50.f + c01(crng) * 200.f;   // 50..250px
                        const float a = ang(crng);
                        flow.dx(ty, tx) += mag * std::cos(a);
                        flow.dy(ty, tx) += mag * std::sin(a);
                        ++n_cor;
                    }
                (void)n_cor;
            }

            Image comp_guide = compute_guide(comp, work);
            Image comp_means, comp_vars;
            local_stats(comp_guide, comp_means, comp_vars);

            // Per-guide-pixel features and label.
            const int NCH = 16;   // 13 inputs + harm + R_ideal + flow error
            std::vector<float> rec((size_t)gh * gw * NCH, 0.f);
            const float alpha = work.noise_alpha(), beta = work.noise_beta();
            double sum_harm = 0.0; double sum_err = 0.0; size_t n_bad = 0;

            parallel_rows(gh, work.num_threads, [&](int gy) {
                for (int gx = 0; gx < gw; ++gx) {
                    // Raw-space position of this guide pixel's quad origin.
                    const int ry = 2 * gy, rx = 2 * gx;
                    const int pty = std::min(flow.ny - 1, std::max(0, ry / ts));
                    const int ptx = std::min(flow.nx - 1, std::max(0, rx / ts));
                    const float fex = flow.dx(pty, ptx), fey = flow.dy(pty, ptx);

                    // Ground-truth flow at this position, analytic.
                    float ty_, tx_;
                    X.fwd((float)ry, (float)rx, ty_, tx_);
                    const float ftx = tx_ - (float)rx, fty = ty_ - (float)ry;

                    // LABEL: what we fetch vs what we should fetch, both from
                    // the comparison frame -- zero when the flow is right no
                    // matter how aliased the content, non-zero only for real
                    // misalignment. Measured at RAW resolution over the quad,
                    // so error in structure finer than the guide's 3x3-mean
                    // support (the rod case) is preserved in the label even
                    // though it is invisible to d_ms.
                    // Content that is not in the comparison frame at all
                    // cannot be merged from it, whatever the flow says. The
                    // fetch-vs-correct-fetch comparison below cannot see this
                    // on its own: both sides would read the same covering
                    // patch and agree. Ground truth knows, so mark it.
                    const bool occluded = in_rect(occ, ty_, tx_);

                    float harm = 0.f, noise_sig = 0.f;
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j) {
                            const int py = ry + i, px = rx + j;
                            const float got = sample_phase(comp, (float)py + fey,
                                                           (float)px + fex, py & 1, px & 1);
                            const float want = sample_phase(comp, (float)py + fty,
                                                            (float)px + ftx, py & 1, px & 1);
                            harm += std::fabs(got - want);
                            noise_sig += std::sqrt(std::max(alpha * std::max(want, 0.f) + beta, 0.f));
                        }
                    harm *= 0.25f; noise_sig *= 0.25f;

                    // Ideal R: 1 while the mis-fetch is within the noise the
                    // sensor would have produced anyway, falling off past it.
                    const float z = harm / std::max(noise_sig, 1e-6f);
                    const float z0 = 2.0f;
                    const float r_ideal = occluded ? 0.f
                                                   : std::exp(-(z * z) / (z0 * z0));

                    // Local flow span M, the same 3x3 tile statistic the
                    // classical mask uses for its motion prior.
                    float mnx = 1e30f, mny = 1e30f, mxx = -1e30f, mxy = -1e30f;
                    for (int i = -1; i <= 1; ++i)
                        for (int j = -1; j <= 1; ++j) {
                            const int yy = pty + i, xx = ptx + j;
                            if (yy < 0 || yy >= flow.ny || xx < 0 || xx >= flow.nx) continue;
                            const float vx = flow.dx(yy, xx), vy = flow.dy(yy, xx);
                            mnx = std::min(mnx, vx); mxx = std::max(mxx, vx);
                            mny = std::min(mny, vy); mxy = std::max(mxy, vy);
                        }
                    const float Mspan = std::sqrt((mxx - mnx) * (mxx - mnx) +
                                                  (mxy - mny) * (mxy - mny));

                    float* o = &rec[((size_t)gy * gw + gx) * NCH];
                    for (int c = 0; c < 3; ++c) o[c] = ref_means.at(gy, gx, c);
                    for (int c = 0; c < 3; ++c) o[3 + c] = std::sqrt(ref_vars.at(gy, gx, c));
                    // Comparison guide sampled where the estimated flow points
                    // -- the same warped statistic the classical mask differences.
                    const float sgy = (float)gy + 0.5f * fey;
                    const float sgx = (float)gx + 0.5f * fex;
                    const int qy = std::min(std::max((int)std::lround(sgy), 0), gh - 1);
                    const int qx = std::min(std::max((int)std::lround(sgx), 0), gw - 1);
                    for (int c = 0; c < 3; ++c) o[6 + c] = comp_means.at(qy, qx, c);
                    o[9] = fex; o[10] = fey;
                    o[11] = Mspan;
                    o[12] = noise_sig;      // expected noise, so the net can
                                            // learn to use it without the hard
                                            // dependence Wronski warned about
                    o[13] = harm;
                    o[14] = r_ideal;
                    // Analysis only -- never an input. Lets evaluation bin
                    // detection by how wrong the flow actually was.
                    o[15] = std::sqrt((fex - ftx) * (fex - ftx) +
                                      (fey - fty) * (fey - fty));

                    if (occluded || harm > 3.f * noise_sig) { ++n_bad; }
                    sum_harm += harm; sum_err += std::fabs(fex - ftx) + std::fabs(fey - fty);
                }
            });
            std::fwrite(rec.data(), sizeof(float), rec.size(), fout);
            total_written += (size_t)gh * gw;
            const double npx = (double)gh * gw;
            std::printf("[ref %zu frame %d] rot=%+.3fdeg shift=(%+.1f,%+.1f) occ=%d "
                        "meanFlowErr=%.2fpx  unmergeable: %.2f%%\n",
                        ri, fi, theta * 180.f / (float)M_PI, X.tx, X.ty, n_occ,
                        sum_err / npx, 100.0 * (double)n_bad / npx);
        }
    }
    std::fclose(fout);
    // Sidecar so the trainer needs no arguments to interpret the blob.
    const std::string meta_path = out_prefix + ".meta";
    if (FILE* mf = std::fopen(meta_path.c_str(), "w")) {
        std::fprintf(mf, "guide_h %d\nguide_w %d\nchannels 16\npixels %zu\n",
                     gh_all, gw_all, total_written);
        std::fclose(mf);
    }
    std::printf("wrote %s (%zu guide pixels x 16 ch)\n", bin_path.c_str(), total_written);
    return 0;
}
