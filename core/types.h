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
// Number of feature planes the learned robustness mask consumes
// (robustness_nn.h). Fixed by the trained weights --
// tools/rob_nn/rob_dataset.cpp writes them in this order and
// tools/rob_nn/train_rob.py trains on that layout, so changing it means
// retraining, not just editing a constant.
inline constexpr int kRobustnessNnChannels = 13;

// The learned mask runs in horizontal strips to bound peak memory (see
// build_robustness_nn_features). kRobustnessNnHalo is the network's exact
// receptive-field radius in guide pixels -- conv3x3 (1) + dilation-2 3x3 (2)
// + dilation-4 3x3 (4) + conv3x3 (1) -- so outputs inside the strip are
// bit-identical to whole-plane inference. Changing the architecture changes
// this number.
inline constexpr int kRobustnessNnHalo = 8;
inline constexpr int kRobustnessNnStripRows = 192;

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
    std::vector<uint32_t> aperture_limited; // ny*nx, 1 = Hessian says 1D/aperture-limited
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
    // Deliberately NOT allocated by the constructor, unlike the two above: an
    // all-zero vector of the right length is indistinguishable from "measured,
    // nothing irregular". Empty means not measured, and compute_s falls back to
    // deriving it, which is what the full-resolution FFT path relies on.
    std::vector<uint32_t> motion_irregular;

    FlowField() = default;
    FlowField(int ny_, int nx_) : ny(ny_), nx(nx_),
        flow((size_t)ny_ * nx_ * 2, 0.f),
        aperture_limited((size_t)ny_ * nx_, 0u),
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
    inline void sample_bilinear(f32 raw_y, f32 raw_x, int tile_size,
                                f32& out_dx, f32& out_dy) const {
        if (ny <= 0 || nx <= 0 || tile_size <= 0) { out_dx = 0.f; out_dy = 0.f; return; }
        const f32 tcy = raw_y / (f32)tile_size - 0.5f;
        const f32 tcx = raw_x / (f32)tile_size - 0.5f;
        const int y0 = (int)std::floor(tcy), x0 = (int)std::floor(tcx);
        const f32 ay = tcy - (f32)y0, ax = tcx - (f32)x0;
        auto cl = [](int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi - 1 : v); };
        const int iy0 = cl(y0, ny), iy1 = cl(y0 + 1, ny);
        const int ix0 = cl(x0, nx), ix1 = cl(x0 + 1, nx);
        const f32 tx0 = dx(iy0, ix0) + (dx(iy0, ix1) - dx(iy0, ix0)) * ax;
        const f32 bx0 = dx(iy1, ix0) + (dx(iy1, ix1) - dx(iy1, ix0)) * ax;
        const f32 ty0 = dy(iy0, ix0) + (dy(iy0, ix1) - dy(iy0, ix0)) * ax;
        const f32 by0 = dy(iy1, ix0) + (dy(iy1, ix1) - dy(iy1, ix0)) * ax;
        out_dx = tx0 + (bx0 - tx0) * ay;
        out_dy = ty0 + (by0 - ty0) * ay;
    }
    inline uint32_t& aperture(int ty, int tx) { return aperture_limited[(size_t)ty * nx + tx]; }
    inline uint32_t aperture(int ty, int tx) const { return aperture_limited[(size_t)ty * nx + tx]; }
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
    // Filtering chroma is close to free perceptually: chroma carries very little
    // detail, which is why every JPEG subsamples it. Luminance is preserved
    // exactly here -- only the colour difference from luma is smoothed -- so
    // this cannot soften the image, only desaturate fine colour detail.
    // 0 = off, and off is the shipped default: it makes the JPEG bit-identical
    // to the render before this stage existed, since isp_denoise_chroma
    // early-returns and never touches the buffer. Raise it to trade fine
    // colour detail for less chroma blotching; measured -73% chroma sigma at
    // 0.75 with luma unchanged to six decimal places.
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
    // Debug parity switch: ignore the camera/DNG NoiseProfile and use the
    // Pixel 4a model from the Python data/README, scaled by ISO. Robustness
    // curves use the bundled 460-main Pixel 4a .npy tables at the rounded ISO.
    bool  debug_pixel4a_noise_profile = false;
    int   debug_pixel4a_noise_curve_iso = 0;

    // Alignment (coarse-to-fine handled internally).
    std::vector<int> bm_factors      = {1, 2, 4, 4};
    std::vector<int> bm_tile_sizes   = {16, 16, 16, 8}; // filled by SNR when tile_size=SNR_based
    std::vector<f32> bm_tile_size_factors = {1.f, 1.f, 1.f, 0.5f};
    // Finest level raised from 1 to 2.
    //
    // The align_ica_per_level comment below works out the budget: with factors
    // {1,2,4,4} the flow arriving at a level carries up to 0.5*factor of
    // residual, so levels 2 and 1 receive 2px against a radius of 4 and have
    // slack, while level 0 received 1px against a radius of 1 and had none.
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
    // Also sets the ICA step clamp, which bounds one iteration to the level's
    // search radius -- so this is 3px at the finest level.
    std::vector<int> bm_search_radii = {3, 4, 4, 4};
    std::vector<std::string> bm_metrics = {"L1", "L2", "L2", "L2"};

    // Settings "Use Neural Flow" toggle. When true, pipeline_paths.cpp routes
    // alignment through the bundled PWCNet Core ML model (neural_flow.h)
    // instead of align()'s block-matching pyramid, then re-uses the exact
    // same flow_to_raw_tile_grid-derived path into compute_robustness/merge
    // -- only the source of the flow field changes, not anything downstream
    // of it. Falls back to the classical path per-frame if neural_flow_
    // available() is false (model missing/failed to load) or the guide
    // image isn't the fixed size the bundled model was converted for.
    bool use_neural_flow = false;

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
    int  alignment_tile_size = 0; // 0 = SNR auto; otherwise force 8/16/32/64.
    // Off: alignment matches d5215ec, which had no thumbnail pre-alignment pass.
    // With this false the plan stays empty, so every frame enters align() with a
    // zero initial transform and frame 0 stays the reference.
    bool global_prealignment_enabled = false;
    // Off by default: it is the only thing that forces the pre-alignment pass
    // to decode every frame up front, and with it off the transform is computed
    // in the analysis loop from the buffer already decoded there -- the stage
    // goes from a separate decode pass to roughly the cost of one thumbnail per
    // frame. See prealign_use_decoded_frames.
    //
    // What it gives up: the merge base is frame 0 rather than the frame nearest
    // the midpoint of the burst's travel. The global transforms, the rotation,
    // the flow initialisation and the aperture-problem benefit are all
    // unaffected -- with frame 0 as reference from_reference[k] reduces to
    // to_first[k] identically, so the prior is the same value, measured
    // directly instead of composed.
    bool global_prealignment_choose_reference = false;
    float global_prealignment_rotation_range_deg = 0.0f;
    float global_prealignment_rotation_step_deg = 0.25f;
    int   global_prealignment_max_shift = 24;       // thumbnail pixels
    int   global_prealignment_thumb_max_dim = 320;
    // How many frames the pre-alignment pass decodes concurrently.
    //
    // That pass is a decode loop and nothing else: measured at 12MP the decode
    // is 750ms per frame, the thumbnail 7ms, and the NCC search itself is below
    // the timer's resolution. Overlapping the decodes is the only lever, and it
    // cannot change the result -- the iterations share nothing and the estimate
    // is deterministic.
    //
    // A decoded 12MP frame is 48MB, so 3 in flight is ~150MB. Raise it only if
    // the device has headroom; 1 restores the serial loop exactly.
    int   prealign_decode_concurrency = 3;
    // Take the pre-alignment transform from the frame the analysis loop has
    // already decoded, instead of running a separate pass that decodes every
    // frame again purely to build a 320px thumbnail.
    //
    // Measured at 12MP: that pass is 750ms of decode per frame against 7ms of
    // thumbnail and an NCC search below the timer's resolution, so it is a
    // decode loop and nothing else. Integrated, an 8-frame burst spends ~52ms
    // instead of ~3900ms and the stage stops existing as a separate step.
    //
    // Requires global_prealignment_choose_reference to be off, because picking
    // the reference needs every frame's transform before the loop starts. With
    // it off the reference is frame 0, and from_reference[k] reduces to
    // to_first[k] exactly, so the inline result is identical to the pass it
    // replaces -- same two thumbnails, same search, only computed later.
    bool  prealign_use_decoded_frames = true;

    // Robustness (Eq. 5: R = s·exp(-d²/σ²) - t). Match configs/default.yaml.
    bool  robustness_enabled = true;
    bool  robustness_save_mask = true;
    // Debug: split the accumulated mask by which motion prior each pixel used,
    // writing _robustness_s1.pgm and _robustness_s2.pgm alongside the combined
    // one. s1 is the strict prior, applied where the flow field varies sharply
    // (the r_Mt test) or the tile is aperture-limited; s2 is the permissive
    // default. The two split masks sum exactly to the combined mask.
    //
    // Off by default: it costs one extra full-resolution float buffer per
    // comparison frame while the mask is being built.
    bool  robustness_save_s_masks = false;

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
    // Replace the analytic robustness mask (Wronski Eq. 5-9) with the learned
    // one in robustness_nn.h. Off by default: the analytic mask is the
    // reference behaviour and the network is only as good as the bursts it
    // was trained on. Falls back automatically when the model is missing or
    // fails to load, so enabling it can never leave the pipeline without a
    // mask.
    //
    // Motivation, measured (tools/rob_nn, synthetic bursts with ground-truth
    // motion built from real raws): the analytic mask separates harmful from
    // harmless pixels with AUC 0.638 and cannot exceed ~26% detection at any
    // threshold, because it pins most pixels at exactly R = 1. The network
    // reaches AUC 0.926 and 73% detection at a 10% false-reject budget.
    bool use_neural_robustness = false;

    // ImageStackAlignator (Michael Kunz) robustness, reimplemented from
    // RobustnessModell.cu -- a from-scratch alternative to the Wronski Eq. 6
    // path, NOT sharing its code. Differences from Wronski (all reproduced):
    //  - PER-CHANNEL R (RGB), each colour scored and merged independently
    //    (the merge applies R_r to red, R_g to green, R_b to blue).
    //  - analytic noise std sigma_MD = sqrt(alpha*mu + beta) (green /sqrt2),
    //    no Monte-Carlo curve.
    //  - texture-based Wiener shrink d *= stdRef^2/(stdRef^2 + sigma_MD^2),
    //    not d^2/(d^2 + d_noise^2).
    //  - motion term M = |flow-span| * 0.5 * mean_colour_diff (coupled to the
    //    residual, unlike Wronski's flow-only Mt); s = 1.5, hard 0 when
    //    M > r_isa_threshold_m.
    //  - no 5x5 local-min erosion.
    // Guide-resolution only; consumes the same guide stats as the classic
    // path. When on, robustness_raw_resolution / fine term / mean-units fix
    // are all bypassed.
    bool robustness_isa = false;
    f32  r_isa_threshold_m = 0.10f; // M above this -> s = 0 (hard reject)

    bool robustness_raw_resolution_enabled = false;
    // True when the raw-resolution path should actually run this call --
    // single place both conditions live, so robustness.cpp, merge.cpp and
    // the Metal dispatch code in metal_gpu.mm can't drift out of step on
    // which one gates it. (Current-mask port: the Decimate-only gate is
    // lifted -- the raw-res mask runs on the FFT grey too, as on the main
    // branch.)
    bool robustness_raw_resolution_active() const {
        return robustness_raw_resolution_enabled;
    }

    // d/sigma UNIT-MISMATCH FIX for the classic (guide-resolution, Wronski
    // Eq. 6) mask. d is the difference of two 3x3 box MEANS; sigma_p^2
    // (ref_vars) is the PER-SAMPLE patch variance. Under noise the variance
    // of a difference of two 9-sample means is (2/9)*sigma_sample^2, so the
    // raw sigma_p^2 denominator is ~4.5x too large for d's scale -- d^2/sigma^2
    // comes out ~4.5x too small everywhere, R stays spuriously high, and
    // edge misalignments (where sigma_p^2 = contrast dominates) leak through.
    // The fix scales ONLY the measured sigma_p^2 by 2/9 to put it in d's
    // mean-difference units; the noise floor sigma_t^2 stays unscaled, so the
    // flat-region operating point s1/s2/t were tuned against is preserved.
    // A deliberate correction to Wronski's formula, off by default so the
    // classic mask stays bit-exact until asked for.
    bool robustness_mean_units_fix = false;

    // ---- Current-mask port (from feat/shape-confidence @ 769c7b3+) ------
    // Fine-scale robustness term: a second distance measured on the raw
    // GUIDE SAMPLES (2-raw-px scale) against max(kappa*sigma_t^2,
    // (phi*sigma_p)^2), combined as z = max(z_box, z_fine). Catches the
    // sub-3-px edge misalignments the box-mean statistic averages away.
    // kappa = 12 ~ 5.3 sigma of the sample difference (tail margin for
    // non-Gaussian sensor noise); phi = 0.5 forgives ~0.5 px of subpixel
    // sampling phase at edges. Gated off when the noise model is disabled
    // (sigma_t = 0 would reject all noise).
    bool robustness_fine_term = true;
    f32  r_fine_kappa = 12.f;
    f32  r_fine_phi   = 0.5f;
    bool robustness_fine_active() const {
        return robustness_fine_term && !debug_noise_model_disabled;
    }
    // Self-calibrating noise floor: DNG NoiseProfiles routinely UNDERSTATE
    // real noise (no PRNU/pattern noise, fat tails); at multi-sigma
    // operating points a 20-30% sigma_t shortfall multiplies false
    // rejections a hundredfold. init_robustness measures the per-channel
    // 25th percentile of measured-local-sigma / model-sigma_t at matching
    // brightness on the reference frame and scales sigma_t/d_t by it,
    // clamped to [1, r_noise_scale_max].
    bool r_noise_floor_autoscale = true;
    f32  r_noise_scale_max = 4.f;
    // ImageStackAlignator's rule for unreliable matches, in the author's own
    // words: "if we cannot determine a precise shift for a given patch due to
    // missing feature (or aperture) then no shift is applied at all." When a
    // tile's block match at any pyramid level is ambiguous -- best and
    // second-best cost within flow_reject_1d_ambiguity_ratio, the same test
    // that already feeds the match_ambiguous flag -- the found offset is
    // DISCARDED and the tile keeps its seed: the upsampled previous-level
    // flow, or the global initial estimate at the coarsest level.
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
    // Sample the per-tile flow BILINEARLY between tile centres wherever it is
    // consumed -- merge, Eq. 6's d, upscale_warp_stats, the raw-resolution
    // mask -- instead of taking the containing tile's vector. Removes the
    // piecewise-constant staircase that rotation turns into a visible tile
    // grid. See FlowField::sample_bilinear for why rotation and not
    // translation. All consumers switch together or the mask would grade a
    // correspondence the merge never fetches.
    bool  flow_bilinear_sampling = true;

    bool  align_ambiguous_fallback_enabled = false;

    // Test switch from the aperture experiments: force merge robustness to zero
    // only when a tile is one-dimensional and its aligned guide residual is
    // high. This does not repair flow; it rejects unsafe 1D tiles.
    bool  flow_reject_1d_enabled = false;
    // A tile is considered one-dimensional when lambda2/lambda1 is below this
    // ratio. Higher catches more edge-like tiles; lower limits rejection to
    // very purely one-dimensional tiles.
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
    // Legacy setting kept for old saved app preferences. The current 1D reject
    // gate uses flow_reject_1d_residual_threshold instead.
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
    // A 1D tile is rejected only when enough pixels in that tile have
    // d^2/sigma^2 above this threshold after noise correction. 2.5 means the
    // aligned-frame difference is about sqrt(2.5)=1.58 expected std-devs.
    float flow_reject_1d_residual_threshold = 2.5f;
    // Scales the estimated sensor noise variance subtracted before the HF
    // loss ratio. >1 assumes more noise, so less of the local variance counts
    // as signal and fewer areas are flagged as high-frequency detail; <1 is
    // more aggressive. 1.0 leaves the estimate as measured.
    // Minimum texture SNR: signal variance must exceed this multiple of the
    // noise floor before a pixel is treated as real high-frequency detail.
    // loss landed, then a hardcoded kMinTextureSnr = 4; configurable again.
    // High-frequency artifact rejection (Wronski et al., "High Frequency
    // Artifacts Removal"). Block matching cannot resolve repetitive fine
    // texture -- the aperture problem -- so it locks onto the wrong period and
    // produces blocky artifacts. Rejected only where BOTH hold:
    //
    //   1. the patch is almost entirely high-frequency: most of its local
    //      variance disappears under a low-pass filter, and
    //   2. the alignment vector field varies a lot locally -- the same r_Mt
    //      test that drives the motion prior, as the paper specifies.
    //
    // Both are needed. Hair and fur are high-frequency but track cleanly, so
    // condition 2 spares them. A flat noisy wall has unstable flow but no real
    // high-frequency signal, so condition 1 spares it.
    //
    // Off by default: it changes which pixels merge, so enable it deliberately.
    bool  hf_artifact_removal_enabled = false;
    // loss = 1 - variance_lowpass / variance_original, both noise-corrected.
    // Above this the patch counts as high-frequency. Reference points measured
    // on typical content: flat wall ~0.04, face ~0.19, brick ~0.83,
    // checkerboard ~0.92.
    float hf_variance_loss_threshold = 0.75f;
    // The patch is skipped unless its signal variance exceeds this multiple of
    // the estimated sensor noise variance. This is what stops ISO 6400 noise
    // reading as high-frequency detail, and it scales with brightness through
    // the noise model rather than being a fixed number.
    float hf_min_texture_snr = 4.0f;
    bool  motion_edge_rejection_enabled = true;
    float motion_edge_threshold = 0.025f;
    float motion_edge_residual_threshold = 2.5f;
    float motion_edge_noise_floor_multiplier = 1.0f;
    int   motion_edge_neighborhood_radius = 1;

    // accumulated_robustness_denoiser.merge — on in 460-main params.py
    bool  accumulated_robustness_denoiser_enabled = true;
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
    SelectionLaw selection = SelectionLaw::HardThreshold;
    bool  snr_auto_tune = true; // Python always runs update_snr_config
    float k_detail  = 0.17f;  // SNR lerp [0.33, 0.25] when snr_auto_tune
    float k_denoise = 0.0f;   // SNR lerp [5.0, 3.0] when snr_auto_tune
    float D_th      = 0.76f;  // overwritten by SNR lerp [0.81, 0.71]
    float D_tr      = 1.12f;  // overwritten by SNR lerp [1.24, 1.0]
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

inline int round_pixel4a_noise_curve_iso(f32 iso) {
    if (!std::isfinite(iso) || iso <= 0.f)
        iso = 100.f;
    const double n = std::round(std::log2((double)iso / 100.0));
    int rounded = (int)std::lround(100.0 * std::pow(2.0, n));
    if (rounded < 50) rounded = 50;
    if (rounded > 3200) rounded = 3200;
    return rounded;
}

inline void apply_pixel4a_noise_profile(Config& cfg, f32 iso) {
    constexpr f32 kPixel4aAlphaIso100 = 1.80710882e-4f;
    constexpr f32 kPixel4aBetaIso100  = 3.1937599182128e-6f;
    if (!std::isfinite(iso) || iso <= 0.f)
        iso = 100.f;
    const f32 scale = iso / 100.f;
    // Pixel 4a uses the same profile for all channels (no per-channel differentiation).
    const f32 a = kPixel4aAlphaIso100 * scale;
    const f32 b = kPixel4aBetaIso100 * scale * scale;
    for (int c = 0; c < 3; ++c) {
        cfg.alpha_dng[c] = a;
        cfg.beta_dng[c] = b;
    }
    cfg.has_noise_profile = true;
    cfg.debug_pixel4a_noise_curve_iso = round_pixel4a_noise_curve_iso(iso);
}

inline f32 smoothstepf(f32 edge0, f32 edge1, f32 x) {
    if (edge1 <= edge0) return x >= edge1 ? 1.f : 0.f;
    f32 t = clampf((x - edge0) / (edge1 - edge0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

} // namespace hhsr
