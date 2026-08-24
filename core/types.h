#pragma once
//
// Core data types for the Handheld Multi-Frame Super-Resolution port.
//
// This is a faithful C++ CPU re-implementation of the algorithm from
// Wronski et al. (SIGGRAPH 2019), ported from the Python/Numba reference in
// this repository. The types are intentionally simple (flat row-major float
// buffers) so the same data can later be uploaded to a GPU (Vulkan SSBOs)
// without changing the algorithm code.
//
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace hhsr {

using f32 = float;

// Row-major image / tensor with an arbitrary number of interleaved channels.
struct Image {
    int h = 0;
    int w = 0;
    int c = 1;
    std::vector<f32> data;

    Image() = default;
    Image(int h_, int w_, int c_ = 1) : h(h_), w(w_), c(c_), data((size_t)h_ * w_ * c_, 0.f) {}

    inline f32&       at(int y, int x, int ch = 0)       { return data[((size_t)y * w + x) * c + ch]; }
    inline const f32& at(int y, int x, int ch = 0) const { return data[((size_t)y * w + x) * c + ch]; }

    inline bool inside(int y, int x) const { return y >= 0 && y < h && x >= 0 && x < w; }
    inline size_t size() const { return data.size(); }
};

// Per-tile optical flow field: shape [nTilesY, nTilesX, 2] (dx, dy).
struct FlowField {
    int ny = 0;
    int nx = 0;
    std::vector<f32> flow; // ny*nx*2
    std::vector<uint32_t> match_ambiguous;  // ny*nx, 1 = best BM match is not clearly unique

    // Wronski's motion prior M: 1 where the flow field varies sharply across the
    // 3x3 tile neighbourhood (the r_Mt test), selecting the strict s1 scale.
    //
    // Carried on the field rather than derived at use time, because it must be
    // measured on the grid alignment actually produced. flow_to_raw_tile_grid
    // duplicates each grey tile across a 2x2 block of raw tiles and multiplies
    // the displacements by 2; recomputing the span afterwards then samples only
    // 2 distinct vectors per axis instead of 3, while the doubled displacements
    // inflate what it does sample. For a linearly varying field those cancel,
    // but for alignment jitter the doubling wins and the test over-fires --
    // measured at roughly 6x the flagged area near the threshold.
    //
    // Deliberately NOT allocated by the constructor, unlike match_ambiguous: an
    // all-zero vector of the right length is indistinguishable from "measured,
    // nothing irregular". Empty means not measured, and compute_s falls back to
    // deriving it, which is what the full-resolution FFT path relies on.
    std::vector<uint32_t> motion_irregular;

    FlowField() = default;
    FlowField(int ny_, int nx_) : ny(ny_), nx(nx_),
        flow((size_t)ny_ * nx_ * 2, 0.f),
        match_ambiguous((size_t)ny_ * nx_, 0u) {}

    // True when motion_irregular carries a measurement for this grid.
    inline bool has_motion_prior() const {
        return motion_irregular.size() == (size_t)ny * (size_t)nx && ny > 0 && nx > 0;
    }
    inline uint32_t& irregular(int ty, int tx) { return motion_irregular[(size_t)ty * nx + tx]; }
    inline uint32_t irregular(int ty, int tx) const { return motion_irregular[(size_t)ty * nx + tx]; }

    inline f32& dx(int ty, int tx) { return flow[((size_t)ty * nx + tx) * 2 + 0]; }
    inline f32& dy(int ty, int tx) { return flow[((size_t)ty * nx + tx) * 2 + 1]; }
    inline f32 dx(int ty, int tx) const { return flow[((size_t)ty * nx + tx) * 2 + 0]; }
    inline f32 dy(int ty, int tx) const { return flow[((size_t)ty * nx + tx) * 2 + 1]; }

    // Bilinear sample of the displacement at a RAW pixel position.
    //
    // Block matching yields ONE vector per tile, so consuming it nearest makes
    // the warp piecewise constant: v(x,y) = v_ij across each tile, jumping at
    // every boundary. For pure translation that is exact -- every tile carries
    // the same vector, so there is nothing to jump. For ROTATION it is not:
    // the true field varies continuously with position, and a per-tile
    // constant is a staircase approximation to it, with a step of about
    // theta * tile_size at every seam.
    //
    // Those steps are sub-pixel for modest rotation (0.28 raw px at 1 degree
    // on a 16-px tile) and therefore nearly invisible to Eq. 6, whose 3x3
    // guide means average over 6x6 raw pixels. But the eye detects
    // DISCONTINUITY far more readily than magnitude, so a sub-pixel error that
    // flips at every 16-pixel boundary reads as a grid while the same error
    // spread smoothly would not be seen at all. That asymmetry is why the
    // artifact is simultaneously below the mask's threshold and above the
    // viewer's.
    //
    // Tile t spans raw [t*ts, (t+1)*ts) so its CENTRE is at (t + 0.5)*ts;
    // hence a raw position p sits at tile coordinate p/ts - 0.5.
    //
    // Every consumer must use this or none of them: the mask has to score the
    // correspondence the merge actually fetches, so an interpolated merge with
    // a nearest mask would grade a fetch nobody performs.
    // Interpolation order for sample_bilinear: false = bilinear (default),
    // true = Catmull-Rom bicubic. Set from Config::flow_bicubic_sampling by
    // the pipeline once the field is final, so every CPU consumer switches
    // together without threading a Config through each call site.
    bool sample_bicubic = false;

    inline void sample_bilinear(f32 raw_y, f32 raw_x, int tile_size,
                                f32& out_dx, f32& out_dy) const {
        const f32* g = has_fine() ? fine_flow.data() : flow.data();
        const int gny = has_fine() ? fine_ny : ny;
        const int gnx = has_fine() ? fine_nx : nx;
        const f32 ts = has_fine() ? 0.5f * (f32)tile_size : (f32)tile_size;
        if (gny <= 0 || gnx <= 0 || tile_size <= 0) { out_dx = 0.f; out_dy = 0.f; return; }
        if (sample_bicubic)
            sample_grid_bicubic(g, gny, gnx, ts, raw_y, raw_x, out_dx, out_dy);
        else
            sample_grid(g, gny, gnx, ts, raw_y, raw_x, out_dx, out_dy);
    }

    static inline f32 catmull1(f32 p0, f32 p1, f32 p2, f32 p3, f32 t) {
        return 0.5f * (2.f * p1 + (-p0 + p2) * t +
                       (2.f * p0 - 5.f * p1 + 4.f * p2 - p3) * t * t +
                       (-p0 + 3.f * p1 - 3.f * p2 + p3) * t * t * t);
    }
    // Catmull-Rom twin of sample_grid: same lattice convention, 4x4 clamped
    // taps. Interpolates the lattice values exactly (passes through them), so
    // at tile centres it agrees with bilinear to the float.
    static inline void sample_grid_bicubic(const f32* grid, int gny, int gnx,
                                           f32 ts, f32 raw_y, f32 raw_x,
                                           f32& out_dx, f32& out_dy) {
        if (gny <= 0 || gnx <= 0 || !(ts > 0.f)) { out_dx = 0.f; out_dy = 0.f; return; }
        const f32 tcy = raw_y / ts - 0.5f;
        const f32 tcx = raw_x / ts - 0.5f;
        const int y0 = (int)std::floor(tcy), x0 = (int)std::floor(tcx);
        const f32 ay = tcy - (f32)y0, ax = tcx - (f32)x0;
        auto cl = [](int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); };
        f32 rx[4], ry[4];
        for (int i = -1; i <= 2; ++i) {
            const int iy = cl(y0 + i, gny);
            f32 px[4], py[4];
            for (int j = -1; j <= 2; ++j) {
                const int ix = cl(x0 + j, gnx);
                const f32* v = grid + ((size_t)iy * gnx + ix) * 2;
                px[j + 1] = v[0];
                py[j + 1] = v[1];
            }
            rx[i + 1] = catmull1(px[0], px[1], px[2], px[3], ax);
            ry[i + 1] = catmull1(py[0], py[1], py[2], py[3], ax);
        }
        out_dx = catmull1(rx[0], rx[1], rx[2], rx[3], ay);
        out_dy = catmull1(ry[0], ry[1], ry[2], ry[3], ay);
    }
    // The shared bilinear-lattice math, parameterised over which grid it
    // reads. `ts` is the grid's tile pitch in raw pixels (fractional for the
    // fine grid so the coarse pitch stays the single configured quantity).
    static inline void sample_grid(const f32* grid, int gny, int gnx, f32 ts,
                                   f32 raw_y, f32 raw_x,
                                   f32& out_dx, f32& out_dy) {
        if (gny <= 0 || gnx <= 0 || !(ts > 0.f)) { out_dx = 0.f; out_dy = 0.f; return; }
        const f32 tcy = raw_y / ts - 0.5f;
        const f32 tcx = raw_x / ts - 0.5f;
        const int y0 = (int)std::floor(tcy), x0 = (int)std::floor(tcx);
        const f32 ay = tcy - (f32)y0, ax = tcx - (f32)x0;
        auto cl = [](int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); };
        const int iy0 = cl(y0, gny), iy1 = cl(y0 + 1, gny);
        const int ix0 = cl(x0, gnx), ix1 = cl(x0 + 1, gnx);
        auto gx = [&](int ty, int tx) { return grid[((size_t)ty * gnx + tx) * 2 + 0]; };
        auto gy = [&](int ty, int tx) { return grid[((size_t)ty * gnx + tx) * 2 + 1]; };
        const f32 tx0 = gx(iy0, ix0) + (gx(iy0, ix1) - gx(iy0, ix0)) * ax;
        const f32 bx0 = gx(iy1, ix0) + (gx(iy1, ix1) - gx(iy1, ix0)) * ax;
        const f32 ty0 = gy(iy0, ix0) + (gy(iy0, ix1) - gy(iy0, ix0)) * ax;
        const f32 by0 = gy(iy1, ix0) + (gy(iy1, ix1) - gy(iy1, ix0)) * ax;
        out_dx = tx0 + (bx0 - tx0) * ay;
        out_dy = ty0 + (by0 - ty0) * ay;
    }

    // Optional boundary-selected refinement at HALF the tile pitch, built by
    // flow_densify_boundary_select. Where the four surrounding tile vectors
    // agree the fine cell holds their bilinear blend. The fine lattice is
    // quarter-shifted from the coarse one ((j+0.5)*ts/2 never lands on
    // (t+0.5)*ts), so smooth-region reproduction is first-order, not
    // bit-exact: measured float-exact (6e-8) for a locally linear field away
    // from image borders, with error bounded by ~1/8 of the flow's per-tile
    // second difference at curvature and ~0.004 px inside the border clamp
    // margin -- orders of magnitude below the 0.28 px staircase this
    // machinery exists to remove. Where the four vectors disagree (a motion
    // boundary) the cell holds whichever single tile vector best explains the
    // guide there, instead of a blend that belongs to neither side. Consumers that warp
    // (merge, Eq. 6, the raw-res mask) automatically sample this grid via
    // sample_bilinear; per-tile metadata (match_ambiguous, motion_irregular,
    // the Eq. 8 motion prior) stays on the coarse grid it was measured on.
    // Empty means not built: sampling falls through to the coarse grid.
    int fine_ny = 0, fine_nx = 0;
    std::vector<f32> fine_flow;   // fine_ny * fine_nx * 2, raw-px displacements
    inline bool has_fine() const {
        return fine_ny > 0 && fine_nx > 0 &&
               fine_flow.size() == (size_t)fine_ny * (size_t)fine_nx * 2u;
    }
    inline f32& fdx(int ty, int tx) { return fine_flow[((size_t)ty * fine_nx + tx) * 2 + 0]; }
    inline f32& fdy(int ty, int tx) { return fine_flow[((size_t)ty * fine_nx + tx) * 2 + 1]; }

    inline uint32_t& ambiguous(int ty, int tx) { return match_ambiguous[(size_t)ty * nx + tx]; }
    inline uint32_t ambiguous(int ty, int tx) const { return match_ambiguous[(size_t)ty * nx + tx]; }
};

// Per-grey-pixel 2x2 covariance field (steerable kernels): [h, w, 4] = xx,xy,yx,yy.
struct CovField {
    int h = 0;
    int w = 0;
    std::vector<f32> cov;

    CovField() = default;
    CovField(int h_, int w_) : h(h_), w(w_), cov((size_t)h_ * w_ * 4, 0.f) {}

    inline f32* at(int y, int x) { return &cov[((size_t)y * w + x) * 4]; }
    inline const f32* at(int y, int x) const { return &cov[((size_t)y * w + x) * 4]; }
};

enum class GreyMethod { FFT, Decimate };
enum class KernelShape { Iso, Steerable };
enum class SelectionLaw { HardThreshold, Linear };

struct IspParams {
    // Manual trim on top of the automatic exposure, in stops.
    float exposure_ev = 0.0f;
    // Where highlight recovery starts, as a fraction of full scale. 1.0 off.
    //
    // The merge writes the DNG pre-white-balanced, so the WB gains are applied
    // BEFORE the 16-bit ceiling. On this sensor those are R x2.06, G x1.0,
    // B x1.84, so a highlight that clipped evenly in raw arrives with R and B
    // pinned at 1.0 and G far below -- hue 300-312, magenta -- and where only
    // green clipped, the complementary green patches. Measured on a raw clip of
    // 0.55: stored (1.000, 0.550, 1.000), saturation 0.45.
    //
    // Anything above the knee is pulled toward neutral at its own peak, so a
    // blown highlight renders white. The per-channel output curve used to hide
    // this by compressing the largest channel hardest; moving the curve onto
    // luminance was correct but removed that accidental cover.
    float highlight_knee = 0.88f;
    // How much of the local (as opposed to global) tone mapping to apply.
    // 0 disables it and leaves a purely global render.
    float local_strength = 0.75f;
    // Highlight compression and shadow lift, applied to the blurred base only.
    float highlight_rolloff = 0.65f;
    float shadow_lift = 0.28f;
    // Display-space black point, applied after the output curve. The log-space
    // shadow lift raises the whole lower range uniformly, which leaves no true
    // black -- measured p1 sat at 22/255 against a reference of 15 until this
    // was added. Renormalised so display white stays at 1.
    float black_point = 0.065f;
    // Red trim on the camera RGB before the matrix. The measured matrix runs
    // about 6% short on red against the reference renders, which reads as a
    // cool, flat image. Red only: also cutting blue overshot and left B at 92
    // against a reference 98.
    float warmth = 0.05f;
    // How much of the measured camera->sRGB matrix to apply, blended toward
    // identity with the neutral response held constant.
    //
    // This was reduced to 0.60 and then 0.75 while chasing teal blues and brown
    // reds, before the cause was found: the output curve running per channel.
    // With that curve moved onto luminance, hue is stable across this parameter
    // -- probing the full chain, sky blue lands at 214 degrees at 0.75 and 212
    // at 1.0, foliage at 122 and 125 -- and only saturation moves (sky 0.85 to
    // 0.95). So the reduction was never fixing the cast, only draining colour,
    // which is what made the render look flat. Back to the full measured matrix.
    float colour_strength = 1.0f;
    // S-curve in display space, applied to luminance so hue is preserved.
    float contrast = 0.55f;
    // Saturation-dependent boost: muted colours gain, already-saturated ones
    // barely move. This is what separates vibrance from RGB *= k.
    float vibrance = 0.50f;
    // Flat multiplier applied after vibrance.
    float saturation = 1.0f;
    // Re-adds the detail layer above unity for local micro-contrast. Distinct
    // from sharpening: no high-pass, no halos, no noise amplification.
    float local_contrast = 0.20f;
    // Chroma noise reduction, applied to the linear merge before anything else
    // in the render. 0 disables it.
    //
    // Colour noise is structurally the worst channel here and nothing upstream
    // removes it. The merge accumulates each colour from same-colour Bayer
    // samples only, so red and blue get half the samples green does. The camera
    // matrix then amplifies what is left unevenly -- the row norms of
    // kDefaultCamToSrgb are R 1.34, G 0.82, B 0.89 -- leaving red roughly 1.9x
    // noisier than green, which is why the blotching reads as red/magenta. And
    // vibrance is literally a chroma multiplier, so it amplifies it again,
    // hardest in the flat desaturated areas where it shows most.
    //
    // Filtering chroma is close to free perceptually: chroma carries very
    // little detail, which is why every JPEG subsamples it. Luminance is
    // preserved exactly here -- only the colour difference from luma is
    // smoothed.
    //
    // ON by default since the filter became detail-gated (isp_denoise_chroma
    // soft-cores by the measured noise scale: deviations within ~2 sigma of
    // the local chroma are noise and get smoothed, beyond ~6 sigma they are a
    // pink flower or a red jacket and are left alone). The ungated version
    // pulled EVERY deviation toward the local mean, which drained small
    // saturated objects toward grey at useful strengths -- that is why this
    // shipped at 0 for so long while the HDR render's shadow lift (measured
    // auto exposure up to ~x5.7) was exposing exactly the chroma blotch this
    // stage exists to remove. The gate's sigma is measured per image, so a
    // clean 8-frame merge gets a tight gate and a noisy one gets a wider one.
    float chroma_denoise = 0.0f;
    // Radius of the chroma filter in FULL-RESOLUTION pixels.
    float chroma_denoise_radius = 12.f;
    // Hold saturation and hue in the skin band. Without a face detector this is
    // the cheap substitute, and it is what stops strong tone mapping plus
    // vibrance turning skin orange.
    bool skin_protect = true;
    bool enabled = true;
};

// 2x2 Bayer CFA pattern (indices into {R=0,G=1,B=2}).
struct CFA {
    uint8_t p[2][2] = {{0, 1}, {1, 2}}; // default RGGB
    // How many of the four sites carry this colour. compute_guide averages the
    // sites of each colour into one guide pixel, so this is both that divisor
    // and the factor by which averaging reduces that channel's noise variance.
    // Two greens under any Bayer pattern, one R and one B -- but derived rather
    // than assumed, so the guide and the noise model cannot disagree.
    int count(uint8_t color) const {
        int n = 0;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                if (p[i][j] == color) ++n;
        return n;
    }
};

// Full pipeline configuration, mirroring configs/default.yaml.
struct Config {
    float scale = 1.0f;
    // Centre-crop the Bayer input by this factor before SR, which is how zoom
    // is realised: at 2.0 a 12 MP sensor frame becomes a 3 MP crop that SR takes
    // back to 12 MP. Continuous rather than integral, so any magnification works
    // on the same principle. 1.0 disables it.
    float input_crop_zoom = 1.f;
    bool  bayer_mode = true;
    GreyMethod grey_method = GreyMethod::FFT;

    // Noise model: sigma^2 = alpha * I + beta   (already scaled for ISO).
    // Raw DNG values, per-channel, before white balance. Indexed by CFA color: 0=R, 1=G, 2=B.
    // Defaults are Pixel-ish fallbacks; overwritten from DNG NoiseProfile (0xC761).
    // Set in load_raw_dng from the DNG tag; NOT averaged, preserves per-channel differences.
    float alpha_dng[3] = {1.80710882e-4f, 1.80710882e-4f, 1.80710882e-4f};
    float beta_dng[3]  = {3.1937599182128e-6f, 3.1937599182128e-6f, 3.1937599182128e-6f};
    bool  has_noise_profile = false;

    // The effective noise model, derived rather than stored.
    //
    // Both loaders multiply every Bayer site by wb[c]/wb[G] before anything
    // downstream sees the data, and scaling a signal by g scales the shot term
    // of its variance by g and the read term by g^2. The NoiseProfile in the
    // file describes the sensor before that multiplication, so consuming it
    // unscaled understates the noise on whichever channels the white balance
    // lifts -- typically R and B.
    //
    // This is derived at the point of use, from three fields every decode path
    // already sets, precisely so that no decode path can forget to update it.
    // There are two independent loaders -- load_raw_dng for LibRaw and the
    // ImageIO path in SRBridge.mm for the device -- and a stored copy filled by
    // only one of them silently pinned the device to the fallback constants.
    //
    // The mean is over the three GUIDE channels, and green is not like the
    // other two. compute_guide packs a Bayer quad into RGB by taking R and B
    // from their single sites but averaging the two green ones:
    //
    //     guide.G = 0.5 * (G1 + G2)
    //
    // Averaging two independent samples halves the variance, so the guide's
    // green channel carries half the noise its profile entry describes, while
    // R and B carry all of it:
    //
    //     Var(guide.R) = a'_R*I + b'_R
    //     Var(guide.G) = (a'_G*I + b'_G) / 2
    //     Var(guide.B) = a'_B*I + b'_B
    //
    // apply_noise_model scores all three guide channels against one scalar
    // curve, so that scalar should be the mean of those three variances -- which
    // is the /3 below with green weighted by a half, not an unweighted mean.
    //
    // This diverges from the reference, which weights all three equally. The
    // correction is small (about -10% on alpha and -6% on beta at typical white
    // balance gains) and always in the direction of trusting the data slightly
    // less. Guarded on bayer_mode: a non-Bayer guide is the raw plane itself,
    // with no quad averaging to account for.
    float noise_wb_gain(int c) const {
        if (!raw_prewhitened) return 1.f;   // nothing applied yet, profile stands
        if (c < 0 || c > 2) return 1.f;
        const float g = white_balance[c] / white_balance[1];
        return (std::isfinite(g) && g > 0.f) ? g : 1.f;
    }
    // Per-guide-channel variance weight: the reciprocal of how many Bayer sites
    // compute_guide averaged into that channel. Two greens gives 1/2, single R
    // and B give 1. Read from the CFA rather than hardcoded, so it tracks
    // whatever the guide actually did.
    float noise_guide_weight(int c) const {
        if (!bayer_mode || c < 0 || c > 2) return 1.f;
        const int n = cfa.count((uint8_t)c);
        return (n > 0) ? 1.f / (float)n : 1.f;
    }
    float noise_alpha() const {
        float s = 0.f;
        for (int c = 0; c < 3; ++c)
            s += alpha_dng[c] * noise_wb_gain(c) * noise_guide_weight(c);
        return s / 3.f;
    }
    float noise_beta() const {
        float s = 0.f;
        for (int c = 0; c < 3; ++c) {
            const float g = noise_wb_gain(c);
            s += beta_dng[c] * g * g * noise_guide_weight(c);
        }
        return s / 3.f;
    }
    // Per-channel counterparts of the above, undivided by 3 and not summed
    // across channels. compute_robustness's Monte Carlo curve (apply_noise_
    // model) scores each guide channel against its own brightness; feeding
    // that a curve built from the cross-channel mean of R/G/B mixes their
    // (generally different, especially post-white-balance) noise
    // characteristics into one shared answer. These let each channel build
    // and look up its own curve instead. noise_alpha()/noise_beta() stay as
    // they are for the callers that genuinely want one representative
    // scalar for the whole (single-channel) grey image -- kernel estimation
    // (kernels.cpp apply_gat) and SNR auto-tuning.
    float noise_alpha_ch(int c) const {
        if (c < 0 || c > 2) return 0.f;
        return alpha_dng[c] * noise_wb_gain(c) * noise_guide_weight(c);
    }
    float noise_beta_ch(int c) const {
        if (c < 0 || c > 2) return 0.f;
        const float g = noise_wb_gain(c);
        return beta_dng[c] * g * g * noise_guide_weight(c);
    }
    // Debug: zero the noise model as read by the ROBUSTNESS MASK only. With
    // it on, apply_noise_model's sigma_t and d_t collapse to 0 for every
    // brightness bin, so sigma_sq = measured variance unfloored and the
    // Wiener shrink becomes d_p^2/(d_p^2+0) = 1 -- R is scored from the raw
    // measured local variance and the raw (unshrunk) pixel difference,
    // isolating whether a tile's d^2 reads small because the noise model
    // forgave it or because the content genuinely is that flat/self-similar.
    //
    // Deliberately NOT routed through make_noise_curves(cfg): that builder is
    // shared with SNR auto-tuning (noise_std_at_brightness) and gating it
    // there silently changed the alignment tile size 16 -> 32 and the four
    // SNR-lerped merge constants, confounding the probe -- measured on the
    // ok/ burst when this toggle first existed. Kernel estimation's GAT and
    // SNR tuning keep reading the ungated accessors above; only the mask's
    // own curve builds and noise floors read these.
    bool debug_noise_model_disabled = false;
    // Robustness-mask noise accessors: WB-SCALED per channel (eca686c's
    // convention), matching a guide built from the WB-multiplied, unclipped
    // raw. Under a per-channel gain g the noise law transforms exactly as
    // alpha' = g*alpha, beta' = g^2*beta, and the guide-quad weight stays
    // (green averages two samples). Gated by debug_noise_model_disabled;
    // GAT/SNR read the ungated noise_alpha()/noise_beta().
    //
    // Known, accepted imprecision of this space: the Monte-Carlo curves
    // model sensor clipping at brightness 1.0, but WB'd red/blue clip at
    // roughly their gain (>1), so the rolloff near the white point applies
    // at the wrong brightness for those channels. The sensor-space
    // alternative was measured on the ok/ burst and reverted by explicit
    // choice.
    float noise_alpha_robustness() const {
        return debug_noise_model_disabled ? 0.f : noise_alpha();
    }
    float noise_beta_robustness() const {
        return debug_noise_model_disabled ? 0.f : noise_beta();
    }
    float noise_alpha_ch_robustness(int c) const {
        return debug_noise_model_disabled ? 0.f : noise_alpha_ch(c);
    }
    float noise_beta_ch_robustness(int c) const {
        return debug_noise_model_disabled ? 0.f : noise_beta_ch(c);
    }
    // Alignment (coarse-to-fine handled internally).
    std::vector<int> bm_factors      = {1, 2, 4, 4};
    std::vector<int> bm_tile_sizes   = {16, 16, 16, 8}; // filled by SNR when tile_size=SNR_based
    std::vector<f32> bm_tile_size_factors = {1.f, 1.f, 1.f, 0.5f};
    // Finest level back to the reference's 1 (python-z default.yaml has
    // search_radii: [1, 4, 4, 4]); the coarse levels keep 4.
    //
    // The align_ica_per_level comment below works out the budget: with factors
    // {1,2,4,4} the flow arriving at a level carries up to 0.5*factor of
    // residual, so levels 2 and 1 receive 2px against a radius of 4 and have
    // slack, while level 0 receives 1px against a radius of 1 and has none.
    // Per-level ICA is what buys that margin back, by making what arrives
    // sub-pixel instead of integer -- so this value and align_ica_per_level
    // have to be reasoned about together.
    //
    // Measured on the ok/ burst -- repetitive straight-line structure, the
    // adverse case for this parameter -- comparing 0728 against 0727:
    //
    //   finest radius                    1        2        3        4
    //   adjacent tiles differing >4px    19.8%    22.2%    25.1%    28.0%
    //   tiles with M >= 8px              30.0%    31.8%    33.8%    35.3%
    //   mean R in the 8-32 M band        0.734    0.737    0.751    0.757
    //   candidates evaluated per tile    9        25       49       81
    //
    // Monotonic in the wrong direction on that scene: a wider window does not
    // only find better matches, it finds more near-ties, and on a repeating
    // pattern the minimum among near-ties is close to arbitrary. The paper's
    // small finest radius acts as a regularizer, holding level 0 near what the
    // coarser levels agreed on. Content with genuine fine-scale motion and no
    // repetition was never tested and may prefer the wider window.
    //
    // In RAW pixels, like bm_tile_sizes; grey_search_radius converts to grey
    // pixels at the point of use, so a configured 4 reaches the same physical
    // distance on both greys. The decimate finest level is the one place the
    // two cannot be equalised -- see grey_search_radius.
    //
    // Also sets the ICA step clamp, which bounds one iteration to the level's
    // search radius, over ica_n_iter iterations.
    std::vector<int> bm_search_radii = {1, 4, 4, 4};
    std::vector<std::string> bm_metrics = {"L1", "L2", "L2", "L2"};

    int  ica_n_iter = 3;
    // Run ICA after block matching on EVERY pyramid level, not only the finest.
    //
    // The reference implementation (alignment.py, align_lvl) does block matching
    // then align_lvl_ica at each level. This port refined only once, at the
    // finest, so every coarse level emits integer-only flow with up to half a
    // pixel of residual, which upscale_flow then multiplies by the upsampling
    // factor.
    //
    // That amplification is not unchecked -- each level re-searches, so it only
    // matters if the arriving error exceeds that level's search radius. With
    // factors {1,2,4,4} and radii {1,4,4,4} the arriving error is 0.5*factor:
    // 2px into levels 2 and 1 against a radius of 4, comfortable; but 1px into
    // level 0 against a radius of 1. The finest level runs with exactly zero
    // slack, so anything else that adds error -- aliasing from the box-filter
    // decimation, motion varying inside a tile, the multi-hypothesis step
    // taking a neighbour's vector -- lands outside what it can recover.
    //
    // Refining per level makes what arrives sub-pixel and restores that margin.
    // It also hands the finest-level ICA a far better starting point, which
    // matters because ICA is a local gradient method with a finite basin of
    // attraction. It does NOT improve final sub-pixel precision in raw terms:
    // the last refinement still happens on a half-resolution grey.
    //
    // Restricted to the decimate grey. It is half raw resolution, so both the
    // accumulated error and that +/-1 budget are twice as coarse in scene terms
    // as on the full-res FFT grey -- and keeping FFT on the single finest-level
    // refinement leaves its output bit-identical to before.
    bool align_ica_per_level = true;

    // Extend per-level ICA to the full-res FFT grey as well.
    //
    // The restriction above was for bit-compatibility, not for any technical
    // reason, and it leaves the FFT path in the worse of the two positions:
    // levels 3..1 emit integer-only flow, level 1 runs at half scale so its
    // residual is up to 0.5px there, upscale_flow multiplies that by 2, and
    // exactly 1.0px arrives at a finest level whose search radius is 1. The
    // budget is spent before level 0 begins, so anything else that adds error
    // -- upscale_flow_460 taking a neighbour's vector, aliasing from the
    // decimation, motion varying inside a tile -- lands outside +/-1 and is
    // never corrected. That is what a tile-shaped displacement in the output
    // looks like.
    //
    // Costs memory rather than time: the reference gradient/Hessian cache goes
    // resident for the burst, and the FFT grey has 4x the pixels of the
    // decimate grey, so roughly +120MB at 12MP. It saves compute, since those
    // gradients are currently rebuilt per frame.
    bool align_ica_per_level_fft = false;

    // True when ICA should run on every pyramid level rather than only the
    // finest.
    bool ica_every_level() const {
        return align_ica_per_level &&
               (grey_method == GreyMethod::Decimate || align_ica_per_level_fft);
    }

    // On the FFT grey, run per-level ICA on the COARSE levels only and leave
    // the finest to the single pass that already exists.
    //
    // The per-level cache holds ref + gradx + grady as full-resolution float
    // buffers for each level. On the decimate grey that is about 180MB across
    // four levels at 48MP; on the FFT grey, which has four times the pixels,
    // level 0 alone is about 576MB and the set is about 730MB. Allocation then
    // fails, prep_level_ica_gpu returns false, align_metal returns false, and
    // pipeline_paths skips the frame -- every frame, so nothing merges at all.
    //
    // Skipping level 0 in the loop costs nothing, because the finest level is
    // refined either way: the existing single pass allocates those same
    // buffers. Levels 1..3 add only about a quarter of full resolution, and
    // they are where the benefit is -- integer-only flow originates on the
    // coarse levels, and that is what arrives at level 0 with its budget spent.
    bool ica_per_level_coarse_only() const {
        return align_ica_per_level_fft && grey_method == GreyMethod::FFT;
    }

    // How many RAW pixels one alignment-grey pixel spans. The FFT grey is
    // full resolution (1); the decimate grey averages each Bayer quad, so it
    // is half resolution (2).
    //
    // bm_tile_sizes is defined in RAW pixels -- that is what the UI shows and
    // what robustness/merge index the flow field with. Block matching runs on
    // the grey, so it must divide by this to get the tile size in grey pixels.
    // Without that division a "16" tile is 16 grey px = 32 raw px on the
    // decimate path and 16 raw px on FFT: the same setting meaning two
    // different physical extents, and flow blocks twice the requested size.
    int alignment_grey_scale() const {
        return (bayer_mode && grey_method == GreyMethod::Decimate) ? 2 : 1;
    }
    // bm_tile_sizes[lvl] converted from raw pixels into alignment-grey
    // pixels. Floored at 8: the Metal block-matching kernels are specialised
    // for 8/16/32/64 and 4 would fall off that path.
    int grey_tile_size(int raw_tile_size) const {
        const int s = alignment_grey_scale();
        return (s <= 1) ? raw_tile_size : std::max(8, raw_tile_size / s);
    }
    // bm_search_radii[lvl] converted from raw pixels into alignment-grey
    // pixels, mirroring grey_tile_size so that both quantities denote the same
    // physical distance whichever grey the alignment runs on. Before this
    // existed the radii were consumed as grey pixels while the tile sizes were
    // converted, so the same configured number reached twice as far on the
    // decimate path -- 256 raw pixels at the coarsest level against FFT's 128,
    // and a finest window of +/-2 raw against FFT's +/-1.
    //
    // Floored at 1: the search is an integer grid on the grey, so half a grey
    // pixel is not expressible, and a zero-radius level would evaluate only the
    // seed it was handed.
    //
    // That floor has a consequence worth stating plainly. On the decimate grey
    // the finest level cannot reach a single RAW pixel: 1/2 floors back to 1
    // grey pixel, which spans 2 raw. Closing that last factor of two would
    // require the search itself to run at raw resolution, not a change of
    // units here.
    int grey_search_radius(int raw_radius) const {
        const int s = alignment_grey_scale();
        return (s <= 1) ? raw_radius : std::max(1, raw_radius / s);
    }
    int  alignment_tile_size = 0; // 0 = SNR auto; otherwise force 8/16/32/64.

    // Robustness (Eq. 5: R = s·exp(-d²/σ²) - t). Match configs/default.yaml.
    bool  robustness_enabled = true;
    bool  robustness_save_mask = true;

    float r_t  = 0.12f;
    float r_s1 = 2.0f;
    float r_s2 = 12.0f;
    // M (Wronski et al. Eq. 7): max-minus-min flow displacement over a 3x3
    // tile neighbourhood, literal raw span -- no detrending. A rigid rotation
    // by theta has a flow gradient of exactly theta everywhere, so M comes out
    // uniform across the whole frame under rotation alone (measured 0.790 at 1
    // degree, 1.580 at 2, on a 12MP geometry with tile 16), which is worth
    // knowing when tuning r_Mt against a rotating handheld burst -- a burst
    // that rotates by more than ~1 degree pushes the whole frame onto the
    // strict prior s1 regardless of whether any tile is actually misaligned.
    float r_Mt = 0.8f;

    // Algorithm 6 as the IPOL article reads it: R is allocated as
    // Zeros(H,W) -- RAW resolution -- and gets there by Dodgson-quadratic
    // upscaling the guide-resolution (H/2,W/2) local statistics, warping
    // the comparison frame's stats by the flow in the process, THEN
    // computing d^2/sigma^2/R per raw pixel, and only then applying the
    // 5x5 local-min (so the min spans 5x5 RAW px, not 5x5 guide px = 10x10
    // raw). This port has always computed d/sigma/R directly at guide
    // resolution instead (matching Wronski's own text and the 460-main
    // reference; the IPOL/python-p reference implements the upscale via
    // cuda_uspcale_dogson). The upscale itself already existed here
    // (robustness.cpp's upscale_warp_stats / the Metal rob_dogson kernel)
    // but was never wired into compute_robustness -- this toggle wires it
    // in. The statistics stay guide-resolution either way; only where the
    // ratio is EVALUATED and where the local-min RUNS changes.
    //
    // Restricted to grey_method == Decimate: that path's flow field is
    // already the coarser of the two, so the guide-resolution mask on top
    // compounds two sources of lost precision; FFT's flow already carries
    // native raw-tile-grid granularity, so the case for this is weaker
    // there.
    //
    // Off by default: ~4x the pixel count for R and the new upscale
    // buffers, and merge.cpp's R-sampling coordinate math has to know
    // which resolution R actually is this run (it reads it off the mask's
    // own dimensions rather than trusting this flag, because this path
    // silently falls back to guide resolution when the hires ref stats
    // are missing).

    bool robustness_raw_resolution_enabled = false;
    // True when the raw-resolution path should actually run this call --
    // single place both conditions live, so robustness.cpp, merge.cpp and
    // the Metal dispatch code in metal_gpu.mm can't drift out of step on
    // which one gates it.
    bool robustness_raw_resolution_active() const {
        return robustness_raw_resolution_enabled && grey_method == GreyMethod::Decimate;
    }
    // Sample the per-tile flow BILINEARLY between tile centres wherever it is
    // consumed -- merge, Eq. 6's d, upscale_warp_stats, the raw-resolution
    // mask -- instead of taking the containing tile's vector. Removes the
    // piecewise-constant staircase that rotation turns into a visible tile
    // grid. See FlowField::sample_bilinear for why rotation and not
    // translation. All consumers switch together or the mask would grade a
    // correspondence the merge never fetches.
    bool  flow_bilinear_sampling = true;

    // Sample the tile flow BICUBICALLY (Catmull-Rom over the 4x4 tile
    // neighbourhood) instead of bilinearly. C1-smooth field instead of C0;
    // the reference's author expects it NOT to fix boundary artifacts (flow
    // is only piecewise smooth, and a smoother interpolant just blends the
    // wrong model more smoothly there -- boundary selection handles that
    // case), but it removes the bilinear field's derivative kinks at tile
    // centres in smooth regions. Catmull-Rom can overshoot near sharp flow
    // changes by up to ~12% of the local step. Requires flow_bilinear_sampling;
    // every consumer switches together, as always.
    bool  flow_bicubic_sampling = false;

    // Sub-pixel refinement of every block-matching result: fit a bivariate
    // quadratic to the 3x3 cost neighbourhood around the winning integer
    // offset and add its sub-cell minimum (mu = -H^-1 g), the piece of
    // Wronski's alignment (present in ImageStackAlignator's kernel.cu) this
    // port previously skipped -- block matching emitted integer flow at every
    // level and ICA alone carried the sub-pixel burden. The costs are already
    // computed by the search; the fit reuses them, so it is nearly free. The
    // correction applies only when the winner is interior to the search
    // window, its 3x3 costs are all finite, the fitted Hessian is positive
    // definite, and |mu| <= 0.5 per axis -- otherwise the integer result
    // stands, and the ambiguity fallback (keep the seed) is never refined.
    // Matters most on the decimate grey, where every residual ICA cannot
    // recover is twice as large in raw pixels.
    bool  bm_subpixel_quadratic = true;

    // Anti-aliased decimation for the ALIGNMENT grey (decimate path only).
    // The quad average is a 2x2 box -- a real but weak low-pass (transfer
    // cos(pi f)) that lets energy between the grey Nyquist and the raw
    // Nyquist fold back into the half-res image as aliasing, which is what
    // makes block-matching/ICA flow wobble with fine texture instead of
    // following motion. This swaps it for a half-phase separable binomial
    // [1 3 3 1]/8 on the raw mosaic (cos^3(pi f), effective Gaussian sigma
    // ~0.87 raw px): same lattice phase (centred at 2g+0.5, so no coordinate
    // conversion changes), same exact R:G:G:B channel balance, ~10x stronger
    // alias suppression near raw Nyquist. Alignment grey only -- the
    // robustness guide and merge are untouched.
    bool  grey_decimate_lowpass = true;

    // Full-resolution ICA polish for the decimate path's flow -- the last
    // remaining accuracy gap to the full-res FFT grey. Every decimate stage,
    // the final ICA included, measures on the half-res grey, so every
    // residual is committed in grey units and doubles in raw pixels. When
    // enabled, the finished flow (post flow_to_raw_tile_grid) gets one final
    // ICA refinement at RAW resolution on the band-limited full-res FFT grey
    // -- the very image the FFT path measures on -- seeded by the decimate
    // estimate. The seed is already sub-pixel (quadratic BM fit + per-level
    // ICA + boundary selection), so the pass operates deep inside ICA's
    // convergence basin: it can only sharpen, not wander. After it the two
    // grey methods differ only in coarse-level tile decisions on pathological
    // content, not in sub-pixel accuracy. Decimate + Bayer only; the FFT path
    // already ends with exactly this pass.
    bool  align_fullres_polish = true;

    // Store the output DNG UN-white-balanced (real AsShotNeutral) instead of
    // baking the WB gains into the pixels. The pipeline merges in
    // pre-white-balanced space (Python utils_dng order), so gains of R~2.06 /
    // B~1.84 were applied BEFORE the 16-bit ceiling: any red highlight above
    // ~0.49 of raw full scale clipped at the DNG write even though the sensor
    // never clipped it -- about 1.05 stops of red and 0.9 of blue headroom
    // lost relative to the input DNGs, and the clipped areas skewed magenta
    // (R/B pinned, G below). With this on, the encoder divides each channel
    // by its gain before the clamp and the writer emits AsShotNeutral=1/gain
    // (its existing non-prewhitened branch), so editors apply WB in float and
    // their highlight recovery sees everything the sensor saw. The merge is
    // untouched -- only the container representation changes -- and the app's
    // own JPEG/preview render re-multiplies the gains on load (self-describing
    // via the private WB tag), so it renders bit-identically.
    bool  dng_store_unwhitened = true;

    // At motion boundaries, SELECT a tile vector instead of blending.
    // Bilinear sampling between tile centres is correct where the field is
    // smooth but blends two different motions across an object boundary,
    // producing flow that belongs to neither side -- exactly where robustness
    // then rejects and detail is lost. When enabled, a post-alignment pass
    // (flow_densify_boundary_select) builds a half-tile-pitch refinement of
    // the field: cells whose four surrounding tile vectors agree within
    // flow_select_threshold keep the bilinear blend (first-order faithful to
    // the smooth behaviour -- see FlowField::fine_flow for the measured
    // bounds); cells at a disagreement get whichever single tile vector
    // best explains the alignment guide there (L1 over the cell footprint).
    // Every warping consumer samples the refined grid through the same
    // sample_bilinear entry point, so mask and merge stay in lockstep.
    bool  flow_boundary_selection = true;
    // Raw-pixel disagreement (Chebyshev, across the four corner vectors)
    // above which a cell is treated as a motion boundary. Below it, blending
    // is not just harmless but preferable -- it carries sub-tile gradients
    // (rotation) that selection would discard.
    f32   flow_select_threshold = 1.0f;

    // ImageStackAlignator's rule for unreliable matches, in the author's own
    // words: "if we cannot determine a precise shift for a given patch due to
    // missing feature (or aperture) then no shift is applied at all." When a
    // tile's block match at any pyramid level is ambiguous -- best and
    // second-best cost within flow_reject_1d_ambiguity_ratio, the same test
    // that already feeds the match_ambiguous flag -- the found offset is
    // DISCARDED and the tile keeps its seed: the upsampled previous-level
    // flow, or zero at the coarsest level.
    //
    // This acts on the flow itself, unlike flow_reject_ambiguous_enabled's
    // soft demotion to s1 downstream -- which is inert under rotation, where
    // M saturates and every tile is on s1 already. Motivating measured case
    // (ok/ burst, frame 6, flat shadow): a candidate at (82,-154) beat the
    // sane parent (88,112) at cost 0.4546 vs 0.4968 -- ratio 1.093, inside
    // the 1.10 ambiguity window. This rule keeps the parent there.
    //
    // Off by default: changes which flow gets computed for every ambiguous
    // tile at every level, on CPU and Metal -- A/B against the default
    // before adopting.
    bool  align_ambiguous_fallback_enabled = false;

    // Eigenvalue ratio (lambda2/lambda1) below which a tile counts as
    // one-dimensional. Read by the ICA damping guard below -- higher treats
    // more edge-like tiles as aperture-limited, lower restricts it to very
    // purely one-dimensional tiles.
    float flow_regularize_aperture_ratio = 0.15f;
    // Regularize the ICA solve. Two independent guards, both on by default:
    //
    //   damping -- add lambda = ratio*l1 - l2 to the Hessian diagonal when the
    //     eigenvalue ratio falls below flow_regularize_aperture_ratio, capping
    //     the effective condition number at 1/ratio. A 1D edge has one large
    //     eigenvalue and one at the noise level; inverting that unregularized
    //     hands the unconstrained direction a step of B/l2 with l2 ~ 0, driven
    //     entirely by noise. Measured on a synthetic edge at ts=16: l1 = 8.03,
    //     l2 = 0.0125, step gain 80x along the aperture axis. Damping drops
    //     that to 0.83x while costing the well-constrained axis 13%.
    //     Fires only below the ratio, so well-conditioned tiles are untouched.
    //
    //   step clamp -- bound one iteration's displacement to the level's block
    //     matching search radius. Damping does not cover the other failure:
    //     a low-gradient tile whose temporal residual is real rather than
    //     noise (occlusion, subject motion, a large incoming misalignment) is
    //     near-isotropic, so no damping applies, yet the step scales as
    //     residual/gradient. Measured at 5.1px for a 30x residual. The clamp
    //     bounds ICA to what the search that preceded it could have reached.
    bool  ica_regularize_enabled = true;
    // Cost ratio that defines an ambiguous block match: a tile is flagged when
    // its second-best match costs less than this times the best. Read by both
    // the match_ambiguous flag and align_ambiguous_fallback_enabled.
    float flow_reject_1d_ambiguity_ratio = 1.10f;
    // Produce and consume the block-matching ambiguity flag: a tile is marked
    // when its second-best match costs less than ambiguity_ratio times the best,
    // i.e. the cost surface has two near-equal minima.
    //
    // This is the only signal in the pipeline that does not come from the image
    // residual. Block matching picks the offset that minimises L1, and the
    // robustness mask then scores that offset by the difference it produces --
    // so an error created by minimising the difference is invisible to a test
    // that measures it. On repetitive structure the wrong period matches nearly
    // as well as the right one, and it is selected precisely because it looked
    // good. Measured on the ok/ burst: tiles whose neighbours disagree by 8-128
    // px still merge at R ~ 0.74, because d^2/sigma^2 there is only ~0.74.
    //
    // The cost surface breaks that circularity -- two near-equal minima are
    // visible whatever either one scores.
    //
    // Accumulated down the pyramid rather than overwritten per level: a wrong
    // match at the coarsest level is 8 px x 32 abs factor = 256 raw px of error,
    // so the flag has to survive from where the mistake is made to where the
    // mask is applied. upscale_flow_460 already propagates it.
    bool  flow_reject_ambiguous_enabled = true;

    // accumulated_robustness_denoiser.merge — on in 460-main params.py
    // Store the ONLINE merge accumulator as fp16 instead of fp32. Kernel
    // arithmetic stays float32; only the stored num/den narrow. At 48 MP this
    // halves the pipeline largest allocation (1116 -> 558 MB) and the
    // bandwidth-bound merge dominant traffic (585 -> ~292 MB/frame).
    // OUTPUT CHANGES: storage quantisation is ~0.05% relative per store,
    // about 1-2 LSB of the 16-bit result -- the accepted trade. The banded
    // path and the host reference stay fp32 regardless.
    bool merge_fp16_accumulator = true;

    bool  accumulated_robustness_denoiser_enabled = false;
    float acc_rob_rad_max = 2.0f;
    float acc_rob_max_multiplier = 8.0f;
    // How the burst is merged. 0 = pick by working-set size, 1 = force banded
    // (accumulate a band at a time, every frame resident across the merge),
    // 2 = force online (one full-size accumulator, each frame released as soon
    // as it is merged, so the working set is flat in frame count).
    //
    // Not simply "online is better": the online accumulator scales with OUTPUT
    // pixels, so at 2x it costs four times what it does at 1x and loses to
    // banding until the burst is long. 0 compares the two and picks.
    int   merge_arch = 0;
    // Adapt the reference-kernel enlargement continuously to the accumulated
    // robustness, instead of the reference implementation's step. Off gives the
    // reference behaviour exactly, including the accumulator overwrite, which is
    // what tools/compare_dng.py needs to line up with the Python.
    bool  acc_rob_adaptive = true;
    // Only consulted when acc_rob_adaptive is false. 2 is the reference's own
    // value for the merge variant; its median and gauss blocks say 8, and taking
    // 8 from those is what made the step fire on every pixel of every burst.
    float acc_rob_max_frame_count = 2.0f;
    // Total frames in the burst, filled in by the pipeline rather than tuned.
    // The reference-kernel enlargement is derived from this and the accumulated
    // robustness, so there is no threshold to set. 0 means unknown, which
    // disables the enlargement rather than guessing.
    int   burst_frame_count = 0;

    // Merge / steerable kernels.
    KernelShape  kernel = KernelShape::Steerable;
    SelectionLaw selection = SelectionLaw::Linear;
    bool  snr_auto_tune = true; // Python always runs update_snr_config
    float k_detail  = 0.17f;  // SNR lerp [0.33, 0.25] when snr_auto_tune
    float k_denoise = 0.0f;   // SNR lerp [5.0, 3.0] when snr_auto_tune
    float D_th      = 0.76f;  // overwritten by SNR lerp [0.81, 0.71]
    float D_tr      = 1.12f;  // overwritten by SNR lerp [1.24, 1.0]
    // Drive Eq. 4's kernel anisotropy continuously from (l1-l2)/(l1+l2), as
    // the paper describes, instead of switching to the full stretch only above
    // 0.9025. See compute_k in kernels.cpp for why the switch mattered: it
    // left every moderately-anisotropic feature -- poles, wires, most real
    // edges -- with a round kernel and none of the misalignment tolerance
    // Section 5.1.1 designs the anisotropic kernel to provide.
    bool kernel_anisotropy_continuous = true;

    // Zero-floor the linear anisotropy law. The reference's linear selection
    // (and the continuous mode above) interpolates the kernel shape with
    // weight 0.5*A, and A = 1 + sqrt(coherence) never goes below 1 -- so the
    // stretch weight never drops below 0.5: even NEAR-ISOTROPIC detail in
    // high-contrast areas (D ~ 0) is elongated 2.5:0.75 along whichever
    // eigenvector the 2x2-gradient tensor happened to prefer. Distant text is
    // the worst case: multi-oriented strokes of 1-2 raw px produce moderate
    // coherence (~0.3-0.6) with an orientation that is mostly aliasing noise
    // at the half-res tensor's scale, and the resulting 3-6:1 kernels smear
    // glyphs into unreadability (or double their strokes, which reads as
    // misalignment). This remaps the weight to w = 0.975 * (A - 1) / 0.95,
    // clamped: zero at A = 1 (isotropic -> round kernel), the SAME value at
    // the old hard threshold A = 1.95, full stretch only for genuinely
    // coherent single-orientation edges. Clean edges keep their elongation;
    // junk orientations stop being amplified.
    bool kernel_anisotropy_zero_floor = true;
    // Exponent on the zero-floored stretch weight: w = 0.975 * t^gamma with
    // t = clamp((A-1)/0.95). 1 = the plain zero-floor law. The default 2 was
    // fitted to a measurement, not a guess: distant text (coherence ~0.36)
    // was only readable with a manual global k_stretch of 2.0, and gamma = 2
    // reproduces exactly that stretch (2.2:1) AT text's coherence while a
    // clean single-orientation edge (coherence ~0.9) keeps 95% of the full
    // k_stretch = 4 elongation the manual override was giving up. Raising
    // gamma concentrates stretch onto ever-more-coherent structure.
    f32  kernel_stretch_gamma = 2.0f;

    float k_stretch = 4.0f;
    float k_shrink  = 2.0f;

    CFA cfa;

    // JPEG / preview rendering. Never affects the linear DNG, which is always
    // written from the unmodified merge.
    IspParams isp;
    // Raw AsShot-style gains (cam_mul), same as Python camera_whitebalance.
    // Load applies k = wb[c]/wb[G]; robustness guide undoes with wb[c] (raw).
    float white_balance[3] = {1.f, 1.f, 1.f};
    // True after load_raw applies per-channel black + WB like Python utils_dng.
    // Merge output is then already white-balanced — preview/bake/DNG AsShot must
    // not apply those gains again (use 1,1,1 for display / private tag).
    bool  raw_prewhitened = false;
    // Ref-frame blacks (per color R,G,B) + white; reused for all burst frames.
    bool  has_black_levels = false;
    float black_levels[3] = {0.f, 0.f, 0.f};
    float white_level = 0.f;

    int   orientation = 1;
    bool  has_color_matrix = false;
    float color_matrix[9] = {1,0,0, 0,1,0, 0,0,1};
    bool  has_cam_to_srgb = false;
    float cam_to_srgb[9] = {1,0,0, 0,1,0, 0,0,1};
    bool  bake_srgb = false;
    // Direct RAW app path: full-res streaming can reload the captured uint16 RAW
    // file instead of spilling/reloading an extra normalized-float cache copy.
    bool  stream_comp_raw_from_loader = false;

    std::string camera_make;
    std::string camera_model;

    int num_threads = 0;      // 0 => hardware_concurrency
    bool use_gpu = false;     // opt-in Vulkan compute merge (experimental)
    std::string debug_string_capture;  // captured metadata (e.g., NoiseProfile), written to .log file
};

inline f32 clampf(f32 v, f32 lo, f32 hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline f32 smoothstepf(f32 edge0, f32 edge1, f32 x) {
    if (edge1 <= edge0) return x >= edge1 ? 1.f : 0.f;
    f32 t = clampf((x - edge0) / (edge1 - edge0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

} // namespace hhsr
