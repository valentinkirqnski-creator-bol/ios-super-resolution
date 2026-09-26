#ifndef HHSR_ROBUSTNESS_REFINE_SHARED_H
#define HHSR_ROBUSTNESS_REFINE_SHARED_H
//
// The refinement network's arithmetic, in a form BOTH compilers accept.
//
// This header is included by core/HHSRKernels.metal (the rob_refine_mask
// kernel) and by core/robustness.cpp (build_robustness_refine_features and the
// host evaluator). That is the whole point of it: the feature definitions are
// fiddly -- a sign convention on the gradient direction, a noise-derived floor,
// a scale factor converting guide pixels to raw pixels -- and two hand-written
// copies would agree on the day they were written and not a month later. The
// CPU mask and the GPU mask are the same text.
//
// What is NOT in here is the gathering. A CPU thread walks rows and takes its
// luma from precomputed planes (measured 4.4x faster than re-reading
// interleaved channels at every tap); a GPU thread gathers its own 5x5 window
// straight from device memory, where scattered reads are cheap and a prepass
// would cost a buffer nobody wants. So each backend fills RefineInputs its own
// way and then calls the same two functions.
//
// Portability rules for anything added below, since it compiles as Metal
// Shading Language (C++14-based) and as ordinary C++17:
//   - no templates, no std:: anything, no dynamic allocation, no references
//     in signatures (Metal would need an address space on them);
//   - address spaces go through RR_THREAD / RR_DEVICE, never bare;
//   - only these maths functions: sqrt, exp, log, fabs, isfinite, and the
//     rr_min / rr_max helpers below -- NOT bare min/max, which are Metal
//     builtins but live in std:: for C++, and nothing (log1p, fmaf, rsqrt)
//     whose presence in a given Metal version cannot be checked from here;
//   - plain float/int, fixed-size arrays, POD structs.

#ifdef __METAL_VERSION__
#define RR_THREAD thread
#define RR_DEVICE device
#else
#include <math.h>
// The host has one address space, so these vanish. Spelled as macros rather
// than `#define thread` so that defining them cannot collide with an ordinary
// identifier in whatever else the including translation unit pulls in.
#define RR_THREAD
#define RR_DEVICE
#endif

// Must match kRobustnessRefineChannels and the trained width. robustness.cpp
// static_asserts the first against types.h so the two cannot drift.
#define RR_CHANNELS 24
#define RR_WIDTH 16

// Flat weight-buffer layout, shared by tools/rob_refine/export_metal_weights.py
// (writes it), core/robustness_refine_weights.h (holds it) and rr_eval below
// (reads it). One buffer rather than nine, so the GPU binds one argument and
// the offsets cannot be mismatched between host and kernel.
#define RR_OFF_MU      0                                  // [24] input mean
#define RR_OFF_SD      (RR_OFF_MU + RR_CHANNELS)          // [24] input std
#define RR_OFF_LOGMASK (RR_OFF_SD + RR_CHANNELS)          // [24] 1 = signed log
#define RR_OFF_W1      (RR_OFF_LOGMASK + RR_CHANNELS)     // [16][24]
#define RR_OFF_B1      (RR_OFF_W1 + RR_WIDTH * RR_CHANNELS)
#define RR_OFF_W2      (RR_OFF_B1 + RR_WIDTH)             // [16][16]
#define RR_OFF_B2      (RR_OFF_W2 + RR_WIDTH * RR_WIDTH)
#define RR_OFF_W3      (RR_OFF_B2 + RR_WIDTH)             // [16]
#define RR_OFF_B3      (RR_OFF_W3 + RR_WIDTH)             // [1]
#define RR_WEIGHTS_N   (RR_OFF_B3 + 1)                    // 761

// Everything one pixel's decision depends on, gathered by the caller.
//
// refl is a 5x5 window of REFERENCE luma centred on the pixel, cmpl a 3x3
// window of COMPARISON luma sampled where the flow points, both already
// reduced to one channel. 5x5 because the structure tensor sums a central
// difference over a 3x3 neighbourhood, and that difference reaches one further
// out again.
struct RefineInputs {
    float refl[5][5];      // [y+2][x+2], reference luma
    float cmpl[3][3];      // [y+1][x+1], comparison luma at the estimated flow
    float R;               // the analytic mask being refined, post Eq. 9
    float d_sq;            // Eq. 6
    float sigma_sq;        // Eq. 6
    float noise_var_sum;   // summed per-channel modelled noise variance
    float gdxdx, gdydx;    // flow-field gradient, per raw pixel
    float gdxdy, gdydy;
    float u, v;            // offset from the tile centre, raw pixels
    float tile_size;
    float Mspan;           // Eq. 7 local flow span
    float s_prior;         // the motion prior actually applied
    float sc;              // raw pixels per feature pixel: 2 guide, 1 raw-res
    int   nch;             // guide channels the luma was averaged over
};

// log(1+x) rather than log1p, deliberately.
//
// The training transform is torch.log1p, so this is a substitution, and the
// reason it is safe is that the argument is always fabs(v) -- never near -1 --
// and the result is immediately divided by a per-channel std of order 0.1 to 1.
// For the smallest inputs that reach here the two differ by ~1e-11 absolute,
// which is eleven orders below anything the network can act on.
//
// What it buys is that this compiles under any Metal version without depending
// on log1p being in that version's math library, which is not something a
// machine with no Metal toolchain can check.
inline float rr_log1p(float x) { return log(1.f + x); }

inline float rr_min(float a, float b) { return a < b ? a : b; }
inline float rr_max(float a, float b) { return a > b ? a : b; }

inline float rr_clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Canonical gradient direction. (gx, gy) and -(gx, gy) describe the same edge,
// so the sign of a projection onto it is arbitrary unless it is pinned.
// Pinning makes E_perp and res_disp comparable pixel to pixel instead of each
// carrying its own coin flip, and makes their product a real sign agreement.
inline void rr_canonical_dir(float gx, float gy, float gmag,
                             RR_THREAD float* ux, RR_THREAD float* uy) {
    if (!(gmag > 1e-8f)) { *ux = 0.f; *uy = 0.f; return; }
    float sx = gx / gmag, sy = gy / gmag;
    if (sy < 0.f || (sy == 0.f && sx < 0.f)) { sx = -sx; sy = -sy; }
    *ux = sx; *uy = sy;
}

// The 24 feature channels. Documented once, in stages.h on
// build_robustness_refine_features -- that comment is the contract with the
// dataset generator and the trainer, and this is its implementation.
inline void rr_features(const RR_THREAD RefineInputs* in, RR_THREAD float* f) {
    const float sc = in->sc;
    const float inv_sc = 1.f / sc;
    const int nch = in->nch;

    // ---- 0-3: what the analytic mask saw.
    const float sig = rr_max(in->sigma_sq, 1e-12f);
    f[0] = in->R;
    f[1] = rr_clamp(in->d_sq / sig, 0.f, 64.f);
    // How much of sigma^2 is the scene's own texture rather than the sensor's
    // noise. This is the leniency itself, handed over as a number: where it is
    // large, Eq. 5's exponent is being divided by the very edge the
    // misalignment lives on, and a high R there means much less than the same
    // R on a flat patch.
    f[2] = rr_log1p(sig / rr_max(in->noise_var_sum, 1e-12f));
    f[3] = in->s_prior;

    // ---- 4-9: the one-vector-per-tile geometry. G is the flow field's
    // gradient from neighbouring tile vectors; the true displacement at an
    // offset (u,v) from the tile centre is about f_tile + G*(u,v), while the
    // merge fetches with f_tile alone. The content therefore lands displaced
    // by -G*(u,v) from where it was wanted -- which is the sign convention E
    // carries, so it can be compared against the displacement the residual
    // implies.
    const float Ex = -(in->gdxdx * in->u + in->gdxdy * in->v);
    const float Ey = -(in->gdydx * in->u + in->gdydy * in->v);
    f[4] = Ex;
    f[5] = Ey;
    f[6] = sqrt(Ex * Ex + Ey * Ey);
    // Scaled to "pixels across one tile" so the number reads the same whatever
    // the tile size is.
    f[7] = (in->gdxdx + in->gdydy) * in->tile_size;
    f[8] = (in->gdydx - in->gdxdy) * in->tile_size;
    f[9] = in->Mspan;

    // ---- 10-16: image structure. Central differences on the luma window,
    // divided by sc so the gradient is per RAW pixel and |grad I| * |E| here
    // is exactly motion_geom_reject's metric.
    const float gx = 0.5f * (in->refl[2][3] - in->refl[2][1]) * inv_sc;
    const float gy = 0.5f * (in->refl[3][2] - in->refl[1][2]) * inv_sc;
    const float gmag = sqrt(gx * gx + gy * gy);
    f[10] = gx;
    f[11] = gy;
    f[12] = gmag;

    float ux, uy;
    rr_canonical_dir(gx, gy, gmag, &ux, &uy);
    const float E_perp = Ex * ux + Ey * uy;
    f[13] = E_perp;                      // across the edge: what doubles it
    f[14] = fabs(Ex * uy - Ey * ux);     // along the edge: harmless

    // Structure tensor over 3x3. Coherence separates an edge (one dominant
    // orientation) from text, foliage, repetitive texture and plain noise (no
    // dominant orientation), which is what stops "high gradient" from being
    // read as "probably misaligned" -- the single most important thing this
    // stage must not do.
    float Jxx = 0.f, Jyy = 0.f, Jxy = 0.f;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j) {
            const int yy = 2 + i, xx = 2 + j;
            const float a = 0.5f * (in->refl[yy][xx + 1] - in->refl[yy][xx - 1]) * inv_sc;
            const float b = 0.5f * (in->refl[yy + 1][xx] - in->refl[yy - 1][xx]) * inv_sc;
            Jxx += a * a; Jyy += b * b; Jxy += a * b;
        }
    const float tr = Jxx + Jyy;
    const float disc = sqrt(rr_max((Jxx - Jyy) * (Jxx - Jyy) + 4.f * Jxy * Jxy, 0.f));
    f[15] = (tr > 1e-12f) ? rr_clamp(disc / tr, 0.f, 1.f) : 0.f;
    f[16] = (in->refl[2][3] + in->refl[2][1] + in->refl[3][2] + in->refl[1][2]
             - 4.f * in->refl[2][2]) * inv_sc * inv_sc;

    // ---- 17-19, 22-23: the correspondence itself.
    const float res = in->refl[2][2] - in->cmpl[1][1];
    f[17] = res;

    // A displacement delta across an edge changes the sampled value by about
    // delta * |grad I|, so res / |grad I| reads back the displacement the
    // photometry implies -- in raw pixels, directly comparable with E_perp.
    // Below a floor it is reported as zero, which is the honest answer: a flat
    // patch carries no displacement evidence, and dividing by a gradient that
    // is pure noise manufactures some.
    //
    // The floor is the gradient's OWN noise, not the pixel's. gx and gy are
    // central differences of 3x3 means of an nch-channel luma, so the
    // per-pixel sigma is attenuated by 3 (the mean), sqrt(nch) (the luma
    // average) and sc (the raw-pixel scaling), and amplified by sqrt(2)/2 (the
    // difference of two samples). Using the unattenuated sigma instead --
    // which is what an eyeballed "2 * nsig" amounts to -- puts the floor about
    // 6x too high and silently zeroes this channel, and the agreement channel
    // built on it, over ~95% of the frame. Measured, not guessed: median gmag
    // on real content is 0.0014 against an unattenuated 2*nsig of 0.008.
    const float nsig_px = sqrt(rr_max(in->noise_var_sum / (float)nch, 0.f));
    const float sigma_grad = nsig_px * 0.70710678f / (3.f * sqrt((float)nch) * sc);
    const float gfloor = rr_max(3.f * sigma_grad, 1e-6f);
    const float res_disp = (gmag > gfloor) ? (-res / gmag) : 0.f;
    f[18] = res_disp;

    float res_abs = 0.f;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            res_abs += fabs(in->refl[i + 1][j + 1] - in->cmpl[i][j]);
    res_abs *= (1.f / 9.f);
    f[19] = rr_clamp(res_abs / rr_max(nsig_px, 1e-6f), 0.f, 64.f);

    f[20] = rr_clamp(in->refl[2][2], 0.f, 1.f);
    f[21] = nsig_px * inv_sc;
    f[22] = E_perp * res_disp;
    f[23] = fabs(res_disp) - fabs(E_perp);

    // Where the estimated flow points outside the comparison frame, Eq. 6
    // gives d = inf, and apply_noise_model's Wiener shrink then evaluates
    // inf/inf -- so d_sq, and R with it, are NaN on those border pixels. That
    // is pre-existing behaviour and harmless to the merge (which refuses the
    // fetch on its own bounds check), but one non-finite input poisons every
    // activation downstream of it, so the vector is sanitised here. Zero is
    // the right substitute: these are pixels the analytic mask has already
    // given up on, and the refinement MULTIPLIES R, so whatever is said about
    // them changes nothing.
    for (int c = 0; c < RR_CHANNELS; ++c)
        if (!isfinite(f[c])) f[c] = 0.f;
}

// The network: signed log on the flagged channels, per-channel normalisation,
// then 24 -> 16 -> 16 -> 1 with ReLU and a sigmoid. Returns q_keep in [0,1].
//
// Pointwise on purpose. A receptive field of one pixel is what makes the CPU's
// strip decomposition exactly equal to whole-plane inference, and what lets
// the GPU do the entire stage in one kernel with no intermediate buffer at all
// -- the 24 features and both hidden layers live in registers and never reach
// memory.
inline float rr_eval(const RR_THREAD float* f, const RR_DEVICE float* w) {
    float x[RR_CHANNELS];
    for (int c = 0; c < RR_CHANNELS; ++c) {
        float v = f[c];
        if (w[RR_OFF_LOGMASK + c] > 0.5f) {
            const float a = rr_log1p(fabs(v));
            v = (v < 0.f) ? -a : a;
        }
        x[c] = (v - w[RR_OFF_MU + c]) / w[RR_OFF_SD + c];
    }
    float h0[RR_WIDTH];
    for (int o = 0; o < RR_WIDTH; ++o) {
        float s = w[RR_OFF_B1 + o];
        for (int c = 0; c < RR_CHANNELS; ++c)
            s += w[RR_OFF_W1 + o * RR_CHANNELS + c] * x[c];
        h0[o] = s > 0.f ? s : 0.f;
    }
    float h1[RR_WIDTH];
    for (int o = 0; o < RR_WIDTH; ++o) {
        float s = w[RR_OFF_B2 + o];
        for (int c = 0; c < RR_WIDTH; ++c)
            s += w[RR_OFF_W2 + o * RR_WIDTH + c] * h0[c];
        h1[o] = s > 0.f ? s : 0.f;
    }
    float s = w[RR_OFF_B3];
    for (int c = 0; c < RR_WIDTH; ++c) s += w[RR_OFF_W3 + c] * h1[c];
    return 1.f / (1.f + exp(-s));
}

// The bounded multiply, which is where the conservatism actually lives. Kept
// here rather than in each caller so the CPU and the GPU cannot end up
// enforcing different guarantees: R == 0 stays 0 because this MULTIPLIES,
// nothing exceeds kappa because drop is clamped into [0,1], and anything
// inside the dead zone returns R unchanged bit for bit.
inline float rr_apply(float R, float q, float kappa, float deadzone) {
    const float drop = rr_clamp(1.f - q, 0.f, 1.f);
    if (drop <= deadzone) return R;
    return R * (1.f - kappa * drop);
}


// ===========================================================================
// The GPU path's gather, also in here.
//
// Everything below reads plain buffers with plain arithmetic, so it compiles
// for both targets exactly as the maths above does -- and that is the point.
// Written only in HHSRKernels.metal it could not be tested anywhere without a
// Metal toolchain; written here, tools/rob_refine/refine_bench.cpp runs it
// against the independently-implemented CPU path on real frames and the whole
// GPU code path is validated except for the Metal compile itself.
//
// The Metal kernel is consequently about fifteen lines: bounds check, copy the
// params out of constant space, call rr_refine_pixel, store.
// ===========================================================================

// Scalars the gather needs. Shared by HHSRKernels.metal (as the kernel's
// `constant` argument) and metal_gpu.mm (which fills it), so there is no host
// mirror of it to fall out of step -- the usual failure mode for a Metal
// parameter block.
struct RefineParams {
    int h, w, nch;
    int tile_size;
    int flow_ny, flow_nx;
    int curve_n;
    int sqrt_index;         // 1 = index the noise curve by mean^2 (sqrt guide)
    int ambiguous_enabled;  // 1 = demote tiles whose block match was ambiguous
    float r_s1;
    float alpha, beta;      // noise model, as the mask's gated accessors give it
    float kappa;            // Config::robustness_refine_max_reduction
    float deadzone;         // Config::robustness_refine_deadzone
    int _pad0, _pad1;       // 64 bytes
};

inline int rr_clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// std::lround's rounding (half away from zero), which is what
// apply_noise_model uses to index the noise curve. The argument is
// non-negative here, but the negative branch is written out so the two
// languages cannot disagree if that ever changes.
inline int rr_lround(float x) {
    return (int)(x >= 0.f ? floor(x + 0.5f) : -floor(-x + 0.5f));
}

// Twin of sample_bilinear_or_inf in robustness.cpp: out of bounds is infinity,
// not a clamped edge sample, because Eq. 6 must see "there is no
// correspondence here" rather than a plausible-looking one.
inline float rr_sample_bilinear_or_inf(const RR_DEVICE float* img, int h, int w,
                                       int nch, float y, float x, int ch) {
    if (!(y >= 0.f && y < (float)h && x >= 0.f && x < (float)w)) return INFINITY;
    int y0 = (int)floor(y), x0 = (int)floor(x);
    int y1 = y0 + 1 < h - 1 ? y0 + 1 : h - 1;
    int x1 = x0 + 1 < w - 1 ? x0 + 1 : w - 1;
    const float fy = y - (float)y0, fx = x - (float)x0;
    const float v00 = img[((y0 * w) + x0) * nch + ch];
    const float v01 = img[((y0 * w) + x1) * nch + ch];
    const float v10 = img[((y1 * w) + x0) * nch + ch];
    const float v11 = img[((y1 * w) + x1) * nch + ch];
    const float top = v00 + (v01 - v00) * fx;
    const float bot = v10 + (v11 - v10) * fx;
    return top + (bot - top) * fy;
}

// Unclamped reference luma at an edge-replicated coordinate.
//
// Unclamped on purpose: this luma feeds gradients, a Laplacian and a structure
// tensor, where clipping a highlight into [0,1] would fabricate an edge. Only
// feature 20 clamps, inside rr_features.
inline float rr_luma(const RR_DEVICE float* means, int h, int w, int nch,
                     int y, int x) {
    const int yy = rr_clampi(y, 0, h - 1), xx = rr_clampi(x, 0, w - 1);
    float s = 0.f;
    for (int ch = 0; ch < nch; ++ch) s += means[((yy * w) + xx) * nch + ch];
    return s / (float)nch;
}

// Comparison luma where the flow points, at one edge-replicated coordinate.
//
// The tile is looked up from THIS tap's own position rather than the centre
// pixel's, because the 3x3 residual window straddles tile boundaries and each
// side must use the vector the merge will actually fetch with there. Falls back
// to the reference luma when the flow leaves the frame, which makes the
// residual zero rather than infinite -- Eq. 6 has already given that pixel
// R = 0, so nothing downstream depends on the value.
inline float rr_cmpl(const RR_DEVICE float* comp_means,
                     const RR_DEVICE float* ref_means,
                     const RR_DEVICE float* flow,
                     const RR_THREAD RefineParams* p, float sc, int y, int x) {
    const int yy = rr_clampi(y, 0, p->h - 1), xx = rr_clampi(x, 0, p->w - 1);
    const float rawy = sc * (float)yy + 0.5f * (sc - 1.f);
    const float rawx = sc * (float)xx + 0.5f * (sc - 1.f);
    const int pty = rr_clampi((int)((rawy + 0.5f) / (float)p->tile_size), 0, p->flow_ny - 1);
    const int ptx = rr_clampi((int)((rawx + 0.5f) / (float)p->tile_size), 0, p->flow_nx - 1);
    const int fi = (pty * p->flow_nx + ptx) * 2;
    const float fyg = 0.5f * flow[fi + 1];
    const float fxg = 0.5f * flow[fi + 0];
    float s = 0.f;
    for (int ch = 0; ch < p->nch; ++ch) {
        const float v = rr_sample_bilinear_or_inf(comp_means, p->h, p->w, p->nch,
                                                  (float)yy + fyg, (float)xx + fxg, ch);
        if (!isfinite(v)) return rr_luma(ref_means, p->h, p->w, p->nch, yy, xx);
        s += v;
    }
    return s / (float)p->nch;
}

// Fills RefineInputs for one pixel straight out of the mask's own buffers.
//
// Eq. 6 is recomputed here rather than carried. Carrying d_sq and sigma_sq from
// rob_make_mask would cost two resident float planes -- 25 MB at guide
// resolution -- to save three bilinear samples and a pair of curve lookups, and
// on this device the memory is the scarce side of that trade. The block is a
// transcription of rob_make_mask's; the two must be changed together, and
// refine_bench compares both against apply_noise_model.
inline void rr_gather(const RR_DEVICE float* ref_means,
                      const RR_DEVICE float* ref_vars,
                      const RR_DEVICE float* comp_means,
                      const RR_DEVICE float* std_curve,
                      const RR_DEVICE float* diff_curve,
                      const RR_DEVICE float* S,
                      const RR_DEVICE float* flow,
                      const RR_DEVICE unsigned int* match_ambiguous,
                      const RR_THREAD RefineParams* p,
                      int y, int x, float R,
                      RR_THREAD RefineInputs* in) {
    const float sc = (p->nch == 3) ? 2.f : 1.f;
    const float rawy = sc * (float)y + 0.5f * (sc - 1.f);
    const float rawx = sc * (float)x + 0.5f * (sc - 1.f);
    // Nearest tile, deliberately: this is the vector the merge will fetch
    // with, and the quantity being judged is how badly THAT vector
    // misrepresents the motion inside its own tile.
    const int pty = rr_clampi((int)((rawy + 0.5f) / (float)p->tile_size), 0, p->flow_ny - 1);
    const int ptx = rr_clampi((int)((rawx + 0.5f) / (float)p->tile_size), 0, p->flow_nx - 1);
    const int pidx = pty * p->flow_nx + ptx;
    const int o0 = (y * p->w + x) * p->nch;

    const float fx = (p->nch == 1) ? flow[pidx * 2 + 0] : 0.5f * flow[pidx * 2 + 0];
    const float fy = (p->nch == 1) ? flow[pidx * 2 + 1] : 0.5f * flow[pidx * 2 + 1];
    float sigma_ms_sq = 0.f, sigma_md_sq = 0.f, d_ms_sq = 0.f, d_md_sq = 0.f;
    float noise_var_sum = 0.f;
    for (int ch = 0; ch < p->nch; ++ch) {
        const float brightness = ref_means[o0 + ch];
        const float bidx = (p->sqrt_index != 0) ? brightness * brightness : brightness;
        int id_noise = rr_lround(1000.f * bidx);
        if (!isfinite(brightness)) id_noise = 0;
        else if (id_noise < 0) id_noise = 0;
        else if (id_noise >= p->curve_n) id_noise = p->curve_n - 1;
        const int cid = ch * p->curve_n + id_noise;
        const float sigma_t = std_curve[cid], d_t = diff_curve[cid];
        sigma_ms_sq += ref_vars[o0 + ch];
        sigma_md_sq += sigma_t * sigma_t;
        const float comp = rr_sample_bilinear_or_inf(comp_means, p->h, p->w, p->nch,
                                                     (float)y + fy, (float)x + fx, ch);
        const float d_p = isfinite(comp) ? fabs(ref_means[o0 + ch] - comp) : INFINITY;
        d_ms_sq += d_p * d_p;
        d_md_sq += d_t * d_t;
        // Twin of guide_noise_var in robustness.cpp: the green guide channel is
        // the average of two Bayer greens, so its variance is halved.
        float nv = p->alpha * rr_clamp(brightness, 0.f, 1.f) + p->beta;
        if (nv < 0.f) nv = 0.f;
        if (p->nch == 3 && ch == 1) nv *= 0.5f;
        noise_var_sum += nv;
    }
    const float shrink = d_ms_sq / (d_ms_sq + d_md_sq);
    in->R = R;
    in->d_sq = d_ms_sq * shrink * shrink;
    in->sigma_sq = rr_max(sigma_ms_sq, sigma_md_sq);
    in->noise_var_sum = noise_var_sum;

    const int ptu = rr_clampi(pty - 1, 0, p->flow_ny - 1);
    const int ptd = rr_clampi(pty + 1, 0, p->flow_ny - 1);
    const int pxl = rr_clampi(ptx - 1, 0, p->flow_nx - 1);
    const int pxr = rr_clampi(ptx + 1, 0, p->flow_nx - 1);
    const float i2 = 1.f / (2.f * (float)p->tile_size);
    const int fr = (pty * p->flow_nx + pxr) * 2, fl = (pty * p->flow_nx + pxl) * 2;
    const int fd = (ptd * p->flow_nx + ptx) * 2, fu = (ptu * p->flow_nx + ptx) * 2;
    in->gdxdx = (flow[fr + 0] - flow[fl + 0]) * i2;
    in->gdydx = (flow[fr + 1] - flow[fl + 1]) * i2;
    in->gdxdy = (flow[fd + 0] - flow[fu + 0]) * i2;
    in->gdydy = (flow[fd + 1] - flow[fu + 1]) * i2;
    in->u = rawx - ((float)ptx + 0.5f) * (float)p->tile_size;
    in->v = rawy - ((float)pty + 0.5f) * (float)p->tile_size;
    in->tile_size = (float)p->tile_size;

    // Eq. 7's local flow span over the 3x3 tile neighbourhood.
    float mnx = INFINITY, mny = INFINITY, mxx = -INFINITY, mxy = -INFINITY;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j) {
            const int yy = pty + i, xx = ptx + j;
            if (yy < 0 || yy >= p->flow_ny || xx < 0 || xx >= p->flow_nx) continue;
            const int fi = (yy * p->flow_nx + xx) * 2;
            const float vx = flow[fi + 0], vy = flow[fi + 1];
            mnx = rr_min(mnx, vx); mxx = rr_max(mxx, vx);
            mny = rr_min(mny, vy); mxy = rr_max(mxy, vy);
        }
    const float spx = (mxx > mnx) ? (mxx - mnx) : 0.f;
    const float spy = (mxy > mny) ? (mxy - mny) : 0.f;
    in->Mspan = sqrt(spx * spx + spy * spy);

    float s_prior = S[pidx];
    if (p->ambiguous_enabled != 0 && match_ambiguous[pidx] != 0u)
        s_prior = rr_min(s_prior, p->r_s1);
    in->s_prior = s_prior;
    in->sc = sc;
    in->nch = p->nch;

    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            in->refl[i][j] = rr_luma(ref_means, p->h, p->w, p->nch,
                                     y + i - 2, x + j - 2);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            in->cmpl[i][j] = rr_cmpl(comp_means, ref_means, flow, p, sc,
                                     y + i - 1, x + j - 1);
}

// The whole stage for one pixel: gather, features, network, bounded multiply.
// Returns the refined R.
inline float rr_refine_pixel(const RR_DEVICE float* ref_means,
                             const RR_DEVICE float* ref_vars,
                             const RR_DEVICE float* comp_means,
                             const RR_DEVICE float* std_curve,
                             const RR_DEVICE float* diff_curve,
                             const RR_DEVICE float* S,
                             const RR_DEVICE float* flow,
                             const RR_DEVICE unsigned int* match_ambiguous,
                             const RR_DEVICE float* weights,
                             const RR_THREAD RefineParams* p,
                             int y, int x, float R) {
    // Nothing to refine, and the multiply would be a no-op anyway. Returning
    // early skips the ~200-read gather on every pixel Eq. 5-9 already rejected.
    if (!(R > 0.f)) return R;
    RefineInputs in;
    rr_gather(ref_means, ref_vars, comp_means, std_curve, diff_curve, S, flow,
              match_ambiguous, p, y, x, R, &in);
    float f[RR_CHANNELS];
    rr_features(&in, f);
    return rr_apply(R, rr_eval(f, weights), p->kappa, p->deadzone);
}

#endif  // HHSR_ROBUSTNESS_REFINE_SHARED_H
