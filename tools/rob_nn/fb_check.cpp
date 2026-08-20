// Is the BASELINE flow actually ground truth? Measure it, do not assume it.
//
// The harm label in rob_real.cpp is
//
//     harm(p) = |comp(p + flow_corrupted) - comp(p + flow_baseline)|
//
// and it is exact *as a measure of what the corruption did*. It is only a
// measure of MERGE HARM if flow_baseline is the true correspondence. The
// generator already drops tiles whose photometric residual the noise model
// cannot explain, but that test is `d^2/sigma^2` -- the analytic mask's own
// statistic -- so it passes exactly the failure this whole model exists for: a
// tile matched onto lookalike content, where the residual is genuinely small
// and the correspondence is genuinely wrong. Using it alone to certify the
// baseline is circular in the one place it matters.
//
// This tool adds an INDEPENDENT test with a different failure mode.
//
//   FORWARD-BACKWARD CONSISTENCY. Align reference -> comparison, then align
//   comparison -> reference, and require the two to cancel:
//
//       e(t) = | f_fwd(t) + f_bwd(t + f_fwd(t)) |
//
//   A tile that locked onto lookalike content generally does NOT reproduce the
//   inverse offset when the search runs the other way, because the lookalike
//   patch has its own neighbourhood and its own best match. Occlusions break
//   it too -- correctly, since content visible in only one frame has no valid
//   correspondence and must not be labelled from.
//
// Also fits a global similarity (translation + rotation) to the forward flow
// per frame pair, and reports per tile how much displacement the ROTATION
// component alone accounts for. That is not used to validate anything -- the
// generator's own note stands, a global model is photometrically worse than
// the per-tile flow because handheld parallax is real -- it exists so that
// "this tile carries strong legitimate rotation" can be identified for
// evaluation and for hard-positive mining. Rotation is measured, never assumed
// from flow magnitude or flow spread, both of which parallax also produces.
//
// Output, one file per (burst, reference frame, comparison frame):
//   <out>_b<B>_r<R>_c<C>.fb   float32, tny*tnx*2: [fb error raw px, rotation
//                             displacement raw px], plus a text sidecar with
//                             the grid dimensions and the fitted angle.
#include "stages.h"
#include "parallel.h"
#include "raw_io.h"
#include "snr_tuning.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace hhsr;

namespace {

// Bilinear lookup of a tile-grid flow at a RAW pixel position.
void sample_flow(const FlowField& f, int ts, float ry, float rx,
                 float& ox, float& oy) {
    const float tcy = ry / (float)ts - 0.5f, tcx = rx / (float)ts - 0.5f;
    const int y0 = (int)std::floor(tcy), x0 = (int)std::floor(tcx);
    const float ay = tcy - (float)y0, ax = tcx - (float)x0;
    auto cl = [](int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); };
    const int ya = cl(y0, f.ny), yb = cl(y0 + 1, f.ny);
    const int xa = cl(x0, f.nx), xb = cl(x0 + 1, f.nx);
    auto lerp2 = [&](float v00, float v01, float v10, float v11) {
        const float t = v00 + (v01 - v00) * ax;
        const float b = v10 + (v11 - v10) * ax;
        return t + (b - t) * ay;
    };
    ox = lerp2(f.dx(ya, xa), f.dx(ya, xb), f.dx(yb, xa), f.dx(yb, xb));
    oy = lerp2(f.dy(ya, xa), f.dy(ya, xb), f.dy(yb, xa), f.dy(yb, xb));
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        std::printf("usage: fb_check out_prefix burst_id frame1.dng ...\n"
                    "  env: ROB_REFS (2) -- must match the generator's\n");
        return 1;
    }
    const std::string out = argv[1];
    const int burst_id = std::atoi(argv[2]);
    std::vector<std::string> files;
    for (int i = 3; i < argc; ++i) files.push_back(argv[i]);
    const int nf = (int)files.size();
    if (nf < 2) { std::printf("need at least 2 frames\n"); return 1; }
    int n_refs = 2;
    if (const char* v = std::getenv("ROB_REFS")) n_refs = std::atoi(v);

    Config cfg;
    cfg.scale = 2.f; cfg.bayer_mode = true; cfg.grey_method = GreyMethod::Decimate;
    cfg.r_t = 0.12f; cfg.r_s1 = 2.0f; cfg.r_s2 = 12.0f; cfg.r_Mt = 0.8f;
    cfg.num_threads = 0; cfg.alignment_tile_size = 16;

    const int refs_to_use = std::min(n_refs, nf - 1);
    for (int ri = 0; ri < refs_to_use; ++ri) {
        // Identical reference selection to rob_real.cpp, so the .fb files key
        // onto the records the generator wrote.
        const int ref_idx = (ri * (nf - 1)) /
                            std::max(1, refs_to_use - 1 ? refs_to_use - 1 : 1);
        const int refi = std::min(ref_idx, nf - 1);
        Image ref = load_raw_frame(files[refi], cfg, true, 0, 0);
        if (ref.w <= 0) { std::printf("  decode failed\n"); continue; }
        Config wn = cfg;
        wn.burst_frame_count = nf;
        tune_config_snr(ref, wn);
        wn.r_t = cfg.r_t; wn.r_s1 = cfg.r_s1; wn.r_s2 = cfg.r_s2;
        const int ts = wn.bm_tile_sizes.empty() ? 16 : wn.bm_tile_sizes[0];
        Image refg = compute_grey(ref, wn.bayer_mode, wn.grey_method);
        Pyramid refp = build_pyramid(refg, wn.bm_factors);

        for (int ci = 0; ci < nf; ++ci) {
            if (ci == refi) continue;
            Image comp = load_raw_frame(files[ci], cfg, true, 0, 0);
            if (comp.w <= 0) continue;
            Image compg = compute_grey(comp, wn.bayer_mode, wn.grey_method);

            // The aligner caches the REFERENCE's Sobel/Hessian keyed on the
            // ADDRESS of the pyramid object. This tool alternates between two
            // different references (forward: refp, backward: compp) and compp
            // is a loop local, so it can be re-created at the same address
            // with different content and score a stale cache hit. Clearing
            // before each call costs one Sobel pass and removes the class of
            // bug entirely.
            clear_align_ref_ica_cache();
            FlowField fwd = align(refp, refg, compg, wn, ts, 0.f, 0.f, 0.f);
            fwd = flow_to_raw_tile_grid(fwd, comp.h, comp.w, compg.h, compg.w, ts,
                                        wn.r_Mt, wn.num_threads, wn.grey_tile_size(ts));
            // The same search with the roles swapped. A separate pyramid: the
            // comparison frame is the reference of this second alignment, and
            // reusing the first one would just be the forward search again.
            Pyramid compp = build_pyramid(compg, wn.bm_factors);
            clear_align_ref_ica_cache();
            FlowField bwd = align(compp, compg, refg, wn, ts, 0.f, 0.f, 0.f);
            bwd = flow_to_raw_tile_grid(bwd, ref.h, ref.w, refg.h, refg.w, ts,
                                        wn.r_Mt, wn.num_threads, wn.grey_tile_size(ts));

            const int tny = fwd.ny, tnx = fwd.nx;
            // ---- global similarity fit to the forward flow -----------------
            // Least squares over
            //     dx = a - theta*(y - cy)      dy = b + theta*(x - cx)
            // which is the linearisation of a rotation by theta about the frame
            // centre plus a translation. Only used to LABEL how rotational a
            // tile is; nothing is validated against it, because handheld
            // parallax makes a global model photometrically worse than the
            // per-tile flow and its deviations are correct, not errors.
            double sy = 0, sx = 0, sdx = 0, sdy = 0, num = 0, den = 0;
            const double cy = 0.5 * tny * ts, cx = 0.5 * tnx * ts;
            for (int t = 0; t < tny; ++t)
                for (int s = 0; s < tnx; ++s) { sdx += fwd.dx(t, s); sdy += fwd.dy(t, s); }
            const double n = (double)tny * tnx;
            const double mdx = sdx / n, mdy = sdy / n;
            for (int t = 0; t < tny; ++t)
                for (int s = 0; s < tnx; ++s) {
                    const double yy = (t + 0.5) * ts - cy, xx = (s + 0.5) * ts - cx;
                    num += xx * (fwd.dy(t, s) - mdy) - yy * (fwd.dx(t, s) - mdx);
                    den += xx * xx + yy * yy;
                }
            const double theta = (den > 0) ? num / den : 0.0;
            (void)sy; (void)sx;

            std::vector<float> outbuf((size_t)tny * tnx * 2, 0.f);
            if (std::getenv("FB_DUMP_FLOW")) {
                std::vector<float> fl((size_t)tny * tnx * 4, 0.f);
                for (int t = 0; t < tny; ++t)
                    for (int s2 = 0; s2 < tnx; ++s2) {
                        const float ry = (t + 0.5f) * ts, rx = (s2 + 0.5f) * ts;
                        float bx = 0.f, by = 0.f;
                        sample_flow(bwd, ts, ry + fwd.dy(t, s2), rx + fwd.dx(t, s2), bx, by);
                        float* o = &fl[((size_t)t * tnx + s2) * 4];
                        o[0] = fwd.dx(t, s2); o[1] = fwd.dy(t, s2); o[2] = bx; o[3] = by;
                    }
                char fb2[64];
                std::snprintf(fb2, sizeof(fb2), "_b%d_r%d_c%d.flow", burst_id, refi, ci);
                if (FILE* f = std::fopen((out + fb2).c_str(), "wb")) {
                    std::fwrite(fl.data(), sizeof(float), fl.size(), f);
                    std::fclose(f);
                }
            }
            double sum_e = 0, n_e = 0, n_rot = 0, n_rot_bad = 0;
            for (int t = 0; t < tny; ++t)
                for (int s = 0; s < tnx; ++s) {
                    const float ry = (t + 0.5f) * ts, rx = (s + 0.5f) * ts;
                    const float fx = fwd.dx(t, s), fy = fwd.dy(t, s);
                    float bx = 0.f, by = 0.f;
                    sample_flow(bwd, ts, ry + fy, rx + fx, bx, by);
                    const float ex = fx + bx, ey = fy + by;
                    const float e = std::sqrt(ex * ex + ey * ey);
                    const double yy = (t + 0.5) * ts - cy, xx = (s + 0.5) * ts - cx;
                    const float rot = (float)(std::fabs(theta) *
                                              std::sqrt(xx * xx + yy * yy));
                    outbuf[((size_t)t * tnx + s) * 2 + 0] = e;
                    outbuf[((size_t)t * tnx + s) * 2 + 1] = rot;
                    sum_e += e; n_e += 1;
                    if (rot > 1.f) {
                        n_rot += 1;
                        if (e > 1.f) n_rot_bad += 1;
                    }
                }
            char nb[64];
            std::snprintf(nb, sizeof(nb), "_b%d_r%d_c%d.fb", burst_id, refi, ci);
            if (FILE* f = std::fopen((out + nb).c_str(), "wb")) {
                std::fwrite(outbuf.data(), sizeof(float), outbuf.size(), f);
                std::fclose(f);
            }
            std::snprintf(nb, sizeof(nb), "_b%d_r%d_c%d.dims", burst_id, refi, ci);
            if (FILE* f = std::fopen((out + nb).c_str(), "w")) {
                std::fprintf(f, "tny %d\ntnx %d\nts %d\ntheta %.8f\n",
                             tny, tnx, ts, theta);
                std::fclose(f);
            }
            std::printf("[b%d ref %d comp %d] theta %.4f deg  mean |fb| %.3f px  "
                        "tiles |fb|>1px %.1f%%   rotational tiles %.1f%% of which "
                        "%.1f%% fail fb\n",
                        burst_id, refi, ci, theta * 57.29578, sum_e / std::max(n_e, 1.0),
                        100.0 * [&]{ double c = 0; for (int i = 0; i < tny * tnx; ++i)
                                        if (outbuf[(size_t)i * 2] > 1.f) c += 1; return c; }()
                                / std::max(n_e, 1.0),
                        100.0 * n_rot / std::max(n_e, 1.0),
                        100.0 * n_rot_bad / std::max(n_rot, 1.0));
        }
    }
    std::printf("wrote %s_b%d_*.fb\n", out.c_str(), burst_id);
    return 0;
}
