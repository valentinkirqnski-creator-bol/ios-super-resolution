#pragma once
//
// Declarations for each pipeline stage, mirroring the modules of the Python
// reference (grey/pyramid, alignment, robustness, kernels, merge).
//
#include "types.h"
#include <vector>
#include <complex>
#include <string>

namespace hhsr {

// ---- grey_pyramid.cpp (FFT helpers, exposed for align.cpp) ---------------
void fft1d(std::vector<std::complex<f32>>& a, bool inverse, std::vector<std::complex<f32>>* dft_buf = nullptr);
void fft2d(std::vector<std::complex<f32>>& data, int h, int w, bool inverse,
           std::vector<std::complex<f32>>* row_buf = nullptr,
           std::vector<std::complex<f32>>* dft_buf = nullptr);
void fftshift2d(std::vector<std::complex<f32>>& data, int h, int w);

// Torch-style real 2D FFT packing: out is [h, w/2+1] complex (row-major).
// Built on the same vDSP-backed fft1d as fft2d.
void rfft2(const f32* in, int h, int w, std::vector<std::complex<f32>>& out);
void irfft2(const std::vector<std::complex<f32>>& in, int h, int w, std::vector<f32>& out);
void fftshift2d_real(std::vector<f32>& data, int h, int w);

// ---- grey_pyramid.cpp ---------------------------------------------------
Image compute_grey_decimate(const Image& raw, bool bayer_mode);
Image compute_grey_fft(const Image& raw);
Image compute_grey(const Image& raw, bool bayer_mode, GreyMethod method);

// Re-express a flow field estimated on the grey onto a raw-resolution tile
// grid, scaling displacements by the resolution ratio.
//
// Alignment runs on the grey, which is half resolution when the Bayer quad
// average is used. Robustness and merge both index the flow as
// raw_coordinate / tile_size and add the displacement in raw pixels, so the
// field has to be converted before either sees it. A no-op when the grey is
// already full resolution.
// r_Mt is the motion-prior threshold, in RAW pixels. The prior is measured
// here rather than carried, because only this function knows the grey-to-raw
// scale needed to express the span in those units. num_threads as in Config.
FlowField flow_to_raw_tile_grid(const FlowField& flow, int raw_h, int raw_w,
                                int grey_h, int grey_w, int tile_size,
                                f32 r_Mt, int num_threads,
                                int guide_tile_size = 0);

// Half-tile final refinement (Config::halftile_refine_enabled): subdivide the
// finished RAW-GRID flow 2x2 (children affine-seeded at their centres, parent
// statistics and J/2 carried), then gated-LK polish in grey coordinates.
// raw_tile_size is the PARENT raw tile; consumers must then index the flow at
// raw_tile_size/2. comp_grey must be the grey align() actually matched (the
// warped one under global pre-alignment).
// emit_affine=false drops the per-tile affine Jacobian from the OUTPUT grid (the
// children are still affine-SEEDED from the parent; they just don't carry their
// own J). Used on the FINAL subdivision level when the tile is small (<=4px): at
// that granularity the tile vectors already carry the rotation, so the Jacobian
// is redundant and skipping it saves ~4 floats/tile of RAM. Intermediate levels
// must keep it (the next level's subdivision seeds from it).
void halftile_subdivide_refine(FlowField& flow, const Image& ref_grey,
                               const Image& comp_grey, const Config& cfg,
                               int raw_tile_size, int raw_h, int raw_w,
                               bool emit_affine = true);

// Build-105 "new" NN mask refiner (Config::rob_mask_nn_enabled): darken-only
// refine of the robustness mask R (R *= g, g in [0,1]) from the tiny net in
// core/rob_mask_nn.h. Features are computed from the full-res grayscale greys
// in the net's training domain; consensus_grey is the per-pixel mean of all
// aligned comparison greys (the multi-frame outlier signal). ref_grey /
// comp_aligned_grey / consensus_grey are single channel, same resolution as R.
void rob_mask_nn_refine(Image& R, const Image& ref_grey,
                        const Image& comp_aligned_grey,
                        const Image& consensus_grey, const FlowField& flow,
                        int tile_size, const Config& cfg);

// Dense per-pixel optical flow (Config::dense_flow_enabled): windowed
// inverse-compositional Lucas-Kanade producing one displacement per grey pixel,
// seeded from the finalized tile/affine flow and eigenvalue-gated per pixel so
// flat/aperture regions keep the seed. Stored grey-resolution on the flow
// (FlowField::dense + sample_dense); raw_tile_size is the tile size of the seed
// grid, raw_h/raw_w the raw frame dims (for the raw/grey scale). No-op when the
// flag is off or a grey is header-only.
void dense_lk_refine(FlowField& flow, const Image& ref_grey,
                     const Image& comp_grey, const Config& cfg,
                     int raw_tile_size, int raw_h, int raw_w);

// Feature-metric alignment refinement (Config::feat_metric_refine_enabled): a
// tiny learned CNN (core/feat_metric_shared.h + feat_metric_weights.h) maps the
// greys to FM_NF feature planes; a per-tile affine Lucas-Kanade refines the flow
// on those features, seeded from the finalized tile/affine flow, and writes a
// per-pixel dense field (FlowField::dense + sample_dense), same contract as
// dense_lk_refine. raw_tile_size is the seed grid's tile size, raw_h/raw_w the
// raw frame dims (grey->raw scale). Host-side, memory-guarded at the call site.
// No-op when the flag is off or a grey is header-only.
void feat_metric_refine(FlowField& flow, const Image& ref_grey,
                        const Image& comp_grey, const Config& cfg,
                        int raw_tile_size, int raw_h, int raw_w);

// Fresh per-level Lucas-Kanade (Config::lk_every_level_enabled): per-tile
// inverse-compositional GN translation refinement against one pyramid level,
// Levenberg-damped, step-clamped, with a per-tile SSD accept gate.
void lk_refine_level_fresh(const Image& ref, const Image& moving,
                           int tile_size, FlowField& flow, int n_iter,
                           int num_threads);

// HDR+ sub-pixel block matching (Config::bm_subpixel_enabled): LS-quadratic fit
// on the 3x3 L2 cost neighbourhood around each tile's integer shift; PD + [-1,1]
// safeguards, L2 levels only. tile_size is the level's grey tile size.
void bm_subpixel_refine_level(const Image& ref, const Image& moving,
                              int tile_size, FlowField& flow, int num_threads);

// Measured per-tile affine (Config::affine_flow_enabled): refine each tile's
// 6-DOF warp W(p)=p+d+J*off(p) by inverse-compositional Gauss-Newton against the
// pixels, seeded by fit_affine's inferred Jacobian; a tile is accepted only when
// its per-pixel SSD improves over the seed. tile_size is the GREY tile size the
// flow grid was built at; no-op when the flag is off or a grey is header-only.
void affine_ica_refine(FlowField& flow, const Image& ref_grey,
                       const Image& moving_grey, const Config& cfg,
                       int tile_size);

// 1 where the flow field is irregular over the 3x3 tile neighbourhood -- the
// r_Mt test (Wronski et al. Eq. 7/8, literally: raw max-min span, no
// detrending). sx/sy scale the stored displacements into r_Mt's units.
std::vector<uint32_t> compute_motion_irregular(const FlowField& flow, f32 Mt,
                                               f32 sx, f32 sy, int num_threads);

struct Pyramid { std::vector<Image> levels; std::vector<int> abs_factors; };
Pyramid build_pyramid(const Image& grey, const std::vector<int>& factors);

Image compute_gradients(const Image& grey);
Image gaussian_blur(const Image& src, float sigma);

// Circular pad so height/width are multiples of tile_size (alignment.init_alignment).
Image pad_image_circular(const Image& img, int tile_size);
// 0 when no padding is needed, so a caller can skip the (full-plane) copy.
int pad_image_circular_amount(const Image& img, int tile_size);

// ---- align.cpp ----------------------------------------------------------
// Forward-backward consistency core (Config::flow_fb_consistency_enabled):
// flags round-trip-inconsistent tiles in fwd, replaces them with the median of
// consistent neighbours (escalating window up to 4x the fill radius for
// blunder clusters) and marks them in fwd.fb_corrected so later pixel-driven
// refiners do not undo the correction. Exposed for the CPU harness.
void apply_fb_consistency(FlowField& fwd, const FlowField& bwd,
                          int gts, const Config& cfg);
FlowField make_global_initial_flow(int ny, int nx, int tile_size, int abs_factor,
                                   int finest_h, int finest_w,
                                   f32 initial_dx, f32 initial_dy,
                                   f32 initial_rotation_rad);
FlowField align(const Pyramid& ref_pyr, const Image& ref_grey,
                const Image& moving_grey, const Config& cfg,
                int tile_size,
                f32 initial_dx = 0.f, f32 initial_dy = 0.f,
                f32 initial_rotation_rad = 0.f);

// Free cached ref Sobel/Hessian (call when reference pyramid is released).
void clear_align_ref_ica_cache();

// ---- robustness.cpp -----------------------------------------------------
struct RefStats {
    Image means;
    Image stds;
    Image hf_loss;   // 1-channel high-frequency variance-loss map
    // Dodgson-quadratic upscale of means/stds to raw resolution (Algorithm
    // 6's literal R <- Zeros(H,W) path -- see Config::
    // robustness_raw_resolution_enabled). Computed once per burst here
    // rather than once per comparison frame, since the reference doesn't
    // change. Empty unless that toggle is on.
    Image means_hires;
    Image stds_hires;
    // GCam SABRE path only (Config::robustness_gcam_enabled): the reference's
    // dedicated 4-channel sqrt guide (rgb + signed variance), already sign-
    // preservingly pre-filtered per §2.10 so the operator reads it with a plain
    // lookup. Built once per burst in init_robustness (host, both platforms).
    // Empty on every other path.
    Image gcam_guide;
}; // guide resolution [h/2, w/2, ch] for Bayer (means_hires/stds_hires: raw [h, w, ch])
RefStats init_robustness(const Image& ref_raw, const Config& cfg);
// Eq. 9 for the raw-resolution robustness path: 2x2 min-reduce to the guide
// lattice, 5x5 min there (Wronski's 10x10-raw footprint on Wronski's grid),
// nearest-upsample back to raw. Shared by the CPU path and the Metal host
// (which runs the mask kernel on GPU and this stage on CPU so both paths
// stay bit-identical without a new shader).
Image robustness_local_min_on_guide(const Image& R);

// Bayer quad -> guide-resolution RGB (or the raw plane itself outside Bayer
// mode). Exposed so callers besides robustness.cpp (neural_flow's caller)
// can build the exact same guide image the classical path scores against.
Image compute_guide(const Image& raw, const Config& cfg);

// GCam SABRE (§3): Monte-Carlo noise LUT, 2 rows (ref/comp) x 10 luma columns x 4
// channels (R,G,B,unused), flat row-major v[(row*10+col)*4+ch]. The green column
// stores the averaged-two-green variance (half the individual); read it x2 for the
// guide gate (§6). Built on the host with a fixed seed so the CPU operator and the
// uploaded GPU copy are identical. Config::gcam_sabre_noise_model gates its use.
struct GcamNoiseLut { float v[80]; };
GcamNoiseLut build_gcam_noise_lut(const Config& cfg);

// GCam SABRE (§2): the dedicated 4-channel sqrt guide (rgb + signed variance,
// the sign carrying the green-only flag). half-res, built from the raw CFA. When
// prefilter_ref is true the §2.10 sign-preserving B-spline pre-filter is applied
// (reference frame); comparison guides pass false. Exposed for the Metal host.
Image compute_gcam_guide(const Image& raw, const Config& cfg, bool prefilter_ref);
// GCam SABRE (§7): dilate a guide-resolution KEEP mask (R in [0,1]) by the
// (sum25 - deadzone)/div rejection-contrast rule. Returns a guide-resolution keep
// mask (the 2x downsample of the shader is dropped -- strictly finer, same math).
Image robustness_gcam_dilate(const Image& R, const Config& cfg);
// GCam SABRE (§4.1): per-component 3x3 median of the per-tile flow vectors, in
// place, edge-clamped. Both robustness and merge read the result. No-op on an
// empty / dense-only field.
void median3x3_tile_flow(FlowField& flow);
// GCam SABRE (§4.2): per-tile local motion extent = length of the min-max spread
// of the flow vectors over the 3x3 tile neighbourhood (raw px). Drives the motion
// boost. Exposed so the Metal host can build the same buffer the CPU operator uses.
std::vector<f32> compute_flow_irregularity(const FlowField& flow);
// GCam SABRE (§4.2): the localMotionExtent the motion boost reads. Returns the
// pre-median extent stashed on the field (FlowField::motion_extent, captured
// before median3x3_tile_flow) when present, else recomputes from the current
// flow. Use this -- not compute_flow_irregularity -- wherever the boost is gated.
std::vector<f32> gcam_motion_extent(const FlowField& flow);
// GCam SABRE (§5+§7): the full CPU ComputeFrameWeight operator -> guide-resolution
// keep mask (dilated). The verified reference and the device default; also the
// fallback when the GPU twin is unavailable. Empty if the reference guide
// (ref_stats.gcam_guide) was not built.
Image compute_robustness_gcam_sabre(const Image& comp_raw, const RefStats& ref_stats,
                                    const FlowField& flow, int tile_size,
                                    const Config& cfg, Image* s_select_out = nullptr);

// MC noise std at brightness in [0,1]: std_curve[round(1000*b)] (fast_monte_carlo).
f32 noise_std_at_brightness(f32 brightness, f32 alpha, f32 beta);
f32 noise_std_at_brightness(f32 brightness, const Config& cfg);

// Full noise curves (1001 bins) for GPU upload — same cache as CPU robustness.
void fetch_noise_curves(f32 alpha, f32 beta,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve);
void fetch_noise_curves(const Config& cfg,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve);

// Build and cache every noise curve this Config's burst will request, in one
// batch. Depends only on the reference frame's NoiseProfile and white balance,
// so it can run on a background thread the moment those are known -- which is
// the point: built lazily, it lands on the first comparison frame and is the
// single largest cost in the burst. Idempotent, and bit-identical to the lazy
// build (same keys, same per-bin RNG seeds).
void prewarm_noise_curves(const Config& cfg);

// Per-guide-channel counterpart: ch's curve is built from Config::
// noise_alpha_ch(ch)/noise_beta_ch(ch) rather than the cross-channel mean.
// Falls back to the shared bundled table when debug_pixel4a_noise_profile
// is set (no per-channel data exists for it).
void fetch_noise_curves_channel(const Config& cfg, int ch,
                                std::vector<f32>& std_curve, std::vector<f32>& diff_curve);

// Robustness mask r at raw resolution [h, w, 1].
//
// s_select_out, when non-null, also receives a per-pixel record of which motion
// prior each pixel was scored with: 1 where the strict s1 applied (sharp local
// variation in the flow field, or an aperture-limited tile), 0 where the
// permissive s2 did. Used to split the accumulated mask for inspection.
// Feature planes for the learned robustness mask (robustness_nn.h), at guide
// resolution, interleaved, in the exact order the model was trained on:
//
//   0-2   reference 3x3 local mean, RGB
//   3-5   reference 3x3 local standard deviation, RGB
//   6-8   comparison 3x3 local mean sampled where the estimated flow points
//   9-10  estimated flow dx, dy at this pixel, in RAW pixels
//   11    local span of the flow field over the 3x3 tile neighbourhood (M)
//   12    expected noise sigma at this brightness
//
// Kept in portable C++ next to the analytic mask because this layout is a
// contract with tools/rob_nn/rob_dataset.cpp, which writes the training set;
// the two must be read side by side to stay in step. Channels 9-12 are the
// ones the analytic mask cannot use, and are why the network can reject a
// tile whose photometry looks innocent.
// Built one horizontal strip at a time. A 32-channel intermediate tensor at
// guide resolution is ~390 MB, and Core ML holds several at once, so running
// the network on the whole plane costs over a gigabyte on top of a burst
// pipeline already holding several 12 MP frames -- which is a jetsam kill,
// not a slowdown. The weights are tiny; the activations are not.
//
// Emits exactly kRobustnessNnStripRows + 2 * kRobustnessNnHalo rows starting
// at source row y0, so every window has the same shape and Core ML never
// reshapes mid-burst. The caller must keep the whole window inside the image
// (clamping y0 near the bottom rather than padding past it): the window's
// edges then coincide with the image's, making the convolutions' zero-padding
// identical to whole-plane inference. Padding past the edge does not, because
// those rows would feed bias-driven activations into the next layer where
// whole-plane inference has true zeros -- measured as visible strip seams.
// raw_res selects the resolution the decision is made at:
//   false - guide resolution. ref_stats.means/.stds and a guide-resolution
//           comp_means, sampled at the flow offset. 3 MP mask.
//   true  - raw resolution. ref_stats.means_hires/.stds_hires and a comp_means
//           that upscale_warp_stats has already Dodgson-upscaled AND warped
//           into the reference frame, so it is sampled at the same (y,x). 12 MP
//           mask, 4x the pixels and 4x the cost, but the statistics keep detail
//           that the 3x3 guide means destroy -- which is the only way a 2-4 px
//           feature can reach the decision at all.
Image build_robustness_nn_features(const RefStats& ref_stats, const Image& comp_means,
                                   const FlowField& flow, int tile_size,
                                   const Config& cfg, int y0,
                                   bool raw_res = false);

Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg,
                         Image* s_select_out = nullptr);

// ---- learned refinement of the analytic mask (Config::
// robustness_refine_nn_enabled) -------------------------------------------
//
// Feature planes for the refinement network, at the resolution `R` already
// is, interleaved, kRobustnessRefineChannels of them. This layout is a
// contract with tools/rob_refine/refine_dataset.cpp (writes the training
// set) and tools/rob_refine/train_refine.py (consumes it); all three must be
// changed together or the weights are being fed something they never saw.
//
//  idx  name        meaning (guide-resolution pixels; flow in RAW pixels)
//   0   R           the analytic mask at this pixel, post Eq. 9 minimum --
//                   the quantity being refined, so the net can learn "leave
//                   an already-doubtful pixel alone"
//   1   z_d         d^2 / sigma^2, Eq. 5's exponent argument
//   2   tex         log1p(sigma^2 / channel-summed noise variance): how much
//                   of sigma is the scene's own texture rather than noise.
//                   This is the leniency that lets a doubled edge through
//   3   s           the motion prior actually applied (s1 or s2)
//   4   Ex          within-tile flow error, x. The flow field's gradient G
//   5   Ey          (central difference of neighbouring tile vectors) times
//                   the pixel's offset from its tile centre, NEGATED so it
//                   reads as "the fetched content is displaced by E relative
//                   to what we wanted" -- directly comparable with 18 below
//   6   Emag        |E|
//   7   div         divergence of the flow field, px across one tile
//   8   curl        curl of the flow field, px across one tile
//   9   Mspan       Eq. 7's local flow span over the 3x3 tile neighbourhood
//  10   gx          reference gradient, x (guide units)
//  11   gy          reference gradient, y
//  12   gmag        |grad I|
//  13   E_perp      E . g_hat -- the component of the within-tile error
//                   ACROSS the edge. This is the one that doubles an edge;
//                   an equal displacement along the edge is invisible
//  14   E_par       |E x g_hat| -- the harmless component, given so the net
//                   can tell the two apart rather than seeing only |E|
//  15   coh         structure-tensor coherence (l1-l2)/(l1+l2). ~1 on a
//                   clean edge, ~0 on text, foliage, repetitive texture and
//                   noise -- the protection for the content that must NOT be
//                   rejected for being high-frequency
//  16   lap         Laplacian of the reference luma (thin lines, ringing)
//  17   res         signed residual, reference luma - comparison luma fetched
//                   at the estimated flow
//  18   res_disp    -res / max(gmag, floor): the displacement the RESIDUAL
//                   implies, in guide pixels. A real geometric misalignment
//                   makes this agree with E_perp; a content change, a
//                   specular flicker or an exposure difference does not
//  19   res_z       3x3 mean |res| / noise sigma -- is the residual even
//                   above the noise floor
//  20   bright      local reference brightness
//  21   nsig        modelled noise sigma at that brightness
//  22   agree       E_perp * res_disp, pointwise. Kept, but measured to be
//                   nearly uninformative: the pointwise displacement estimate
//                   has an uncertainty of 0.25-0.8 raw px, larger than the
//                   error being hunted, so per-pixel agreement is buried in
//                   noise (rank correlation with the damage +0.04). Channel
//                   32 is the version that works
//  23   mismatch    |res_disp| - |E_perp|: residual the geometry does not
//                   explain (content change, occlusion, a different failure)
//  24   sig_ms_frac sigma_ms^2 / sigma^2: how much of Eq. 6's sigma the
//                   scene's own texture accounts for, rather than the modelled
//                   noise floor. 1 means texture won. Inert in bright scenes
//                   (measured ~1 everywhere); it is there for low light, where
//                   the floor wins and a high R means something different
//
//  ---- noise-aware edge detection, then a pooled measurement of its shift.
//  Both come out of the same 3x3 structure tensor already built for channel 15,
//  which is why they cost no extra gathering: a 3x3 Lucas-Kanade window needs
//  exactly the 5x5 luma the builder already holds.
//
//  25   edge_snr    log1p(lambda1 / noise energy over the window). "Is there a
//                   real edge here", answered against the noise model instead
//                   of an absolute gradient, so one threshold works in daylight
//                   and at ISO 6400. Only ~4% of pixels carry a real edge
//  26   aniso       lambda2/lambda1: edge against corner or isotropic texture
//  27   delta_lk    the displacement along the edge normal, from
//                   J delta = b (J the structure tensor, b = sum -res grad I).
//                   Inverse-variance pooling of the per-pixel estimate, which
//                   is what buys the precision: measured uncertainty 0.097 raw
//                   px against 0.25-0.8 px pointwise
//  28   sd_lk       that uncertainty, sigma_res / sqrt(lambda1), so the network
//                   knows how far to trust channel 27
//  29   t_lk        delta_lk / sd_lk -- displacement in units of its own noise.
//                   The strongest single feature measured (+0.358 against the
//                   damage, where the pointwise residual reaches +0.150). It
//                   sits at ~1.65 on correctly aligned content, not 0: that is
//                   the aliasing and resampling floor, and it is the baseline
//                   "no evidence" level the network learns to ignore
//  30   u_n         offset from the tile centre, x, in tile widths
//  31   v_n         offset from the tile centre, y. Under rotation the error
//                   grows with this and resets at each tile boundary, so the
//                   offset is the shape of the artifact, not a detail of the
//                   parameterisation
//  32   agree_lk    E.n * delta_lk / sd_lk^2 -- the flow geometry confirmed by
//                   the pooled measurement, in units of its uncertainty. E is
//                   projected onto the STRUCTURE TENSOR's normal, the same
//                   direction delta_lk uses; projecting onto the pointwise
//                   gradient instead (as channel 13 does) canonicalises the
//                   sign independently and the product loses its meaning.
//                   |agree_lk| reaches +0.352, and |t_lk| * |E_perp| +0.356 --
//                   both above |E| alone, which is the best of the old features
//
//  ---- motion_geom_reject's own test, handed over as guidance rather than
//  applied. It is a good detector and a blunt one; the network gets its metric
//  and its verdict and is free to disagree, which it cannot do with the zeros
//  the test would otherwise already have written into R.
//
//  33   gm_abs      |grad I| * |E|, the absolute criterion, in R's own units
//  34   gm_rel      the exposure-invariant form: (|grad I| noise-subtracted,
//                   divided by local brightness) * |E|
//  35   gm_fire     1 when either criterion clears its threshold
//
//  ---- a second, coarser scale, so a small edge and a large one are separable
//  rather than both just "an edge". Smooth THEN differentiate; pooling g^2 over
//  a wider window does not do this and was measured not to (detection ratio
//  1.00 between drop and keep pixels).
//
//  36   edge_snr_c  channel 25 on the 5x5 window's smoothed derivatives
//  37   coh_c       coherence at that coarser scale
//  38   thick       log ratio of the two scales' lambda1 -- positive when the
//                   structure is fine, negative when it is broad
//
//  ---- the local affine motion model. Channels 4-9 read the flow field with a
//  CENTRAL DIFFERENCE: a two-tap estimate, divided by 2*tile_size, of a field
//  whose every sample is a noisy block-match result. Under rotation the true
//  flow is exactly affine, so it is fitted by least squares over a 5x5 window
//  of tile vectors -- 25 samples instead of 4 -- and the same questions asked
//  again of the fit. See rr_fit_affine.
//
//  Measured on the two real bursts, rank correlation against the generator's
//  recorded per-pixel flow error: 0.782 for channel 41 where channel 6 gets
//  0.429, and 0.683 against 0.302 in the sub-pixel band below one pixel. Flat
//  in rotation magnitude, where the central difference degrades. It also beats
//  Mspan (0.591), which the trained model leaned on hardest -- and Mspan is per
//  TILE, constant across the very tile whose interior this is resolving.
//
//  39   aEx         the flow this tile will be fetched with, minus what the
//  40   aEy         affine model says belongs at this pixel. Same sign
//                   convention as E. Catches two failures at once: the
//                   within-tile variation one vector cannot represent, and a
//                   tile vector that is simply wrong relative to its
//                   neighbours -- the central difference sees neither cleanly
//  41   aEmag       its length
//  42   aE_perp     its component across the edge, as 13 is for E
//  43   aE_res      log1p of the RMS fit residual. Rotation is affine and
//                   parallax is not, so this is the non-affine part of the
//                   local motion. Nothing else here measures it (0.562)
//  44   aE_rot      the fit's antisymmetric part, px across one tile: channel
//  45   aE_div      8 and 7 again, from 25 samples instead of 4
//  46   aE_z        |aE_perp| * |grad I| / sigma, clamped. The predicted damage
//                   in noise units -- a displacement across a gradient shifts
//                   the fetched value by their product, and what decides
//                   visibility is its size against the pixel's own noise. A
//                   two-layer MLP over normalised inputs can neither multiply
//                   nor divide, so the three factors separately are not this
//
// Emits exactly `strip_h` rows starting at source row y0, clamped to the
// image. With kRobustnessRefineHalo == 0 (the pointwise net) any row range
// is exact; a convolutional variant needs the same fully-inside-the-image
// windowing build_robustness_nn_features documents.
//
// Lengths and directions are in RAW pixels throughout, the same units
// Config::motion_geom_reject_threshold is expressed in, so the two tests are
// directly comparable. Channels 13 and 18 are signed along a canonicalised
// gradient direction (flipped so gy > 0), which makes their signs comparable
// between pixels and makes 22 the sign-agreement it is meant to be.
//
// `R` selects the resolution: ref_stats.means/.stds when R is guide-sized,
// .means_hires/.stds_hires when it is raw-sized, with comp_means matching.
// d_sq/sigma_sq are Eq. 6's outputs for the same plane, passed in rather than
// recomputed so this cannot drift from the mask it is refining -- the Metal
// path does not return them, so its caller runs apply_noise_model itself.
Image build_robustness_refine_features(const RefStats& ref_stats,
                                       const Image& comp_means, const Image& R,
                                       const Image& d_sq, const Image& sigma_sq,
                                       const FlowField& flow, int tile_size,
                                       const Config& cfg, int y0, int strip_h);

// Eq. 6 on its own: the correspondence the mask scores (reference statistics
// against comparison statistics fetched at the estimated flow) reduced to
// d^2 and sigma^2. Exposed because two callers outside compute_robustness
// need exactly these numbers -- apply_robustness_refinement, which must feed
// the network the mask's own values rather than a lookalike, and
// tools/rob_refine/refine_dataset.cpp, which must write the training set with
// the same numbers inference will see. raw_res tells it whether comp_means
// has already been warped into the reference frame (the raw-resolution path
// warps during upscaling, so applying the flow again would double it).
void robustness_correspondence(const Image& ref_means, const Image& ref_vars,
                               const Image& comp_means, const FlowField& flow,
                               int tile_size, bool raw_res, const Config& cfg,
                               Image& d_sq, Image& sigma_sq);

// Eq. 7/8's per-tile motion prior s (r_s1 where the flow field is irregular,
// r_s2 where it is not), one entry per tile. Exposed for the same reason as
// robustness_correspondence: the refinement's feature builder and its GPU
// parity harness both need the mask's own values rather than a lookalike.
std::vector<f32> robustness_motion_prior(const FlowField& flow, const Config& cfg);

// Applies the refinement network to `R` in place. comp_raw is needed only to
// build the comparison guide statistics. A no-op (returning false, R
// untouched) when the toggle is off, the model is unavailable, the feature
// planes cannot be built, or inference fails -- so every failure mode is
// "current behaviour", never "no mask".
//
// changed_frac, when non-null, receives the fraction of pixels the refinement
// actually moved (outside Config::robustness_refine_deadzone). That number is
// the headline the stage is judged on: this is supposed to be a sparse
// correction, and a run that rewrites half the frame is broken however good
// its loss looked.
bool apply_robustness_refinement(Image& R, const Image& comp_raw,
                                 const RefStats& ref_stats, const FlowField& flow,
                                 int tile_size, const Config& cfg,
                                 float* changed_frac = nullptr);

// ---- kernels.cpp --------------------------------------------------------
CovField estimate_kernels(const Image& raw, const Config& cfg);

// ---- merge.cpp ----------------------------------------------------------
// frame_id: optional stable id for GPU buffer reuse when raw is streamed via scratch.
// Returns false when the frame did not reach the accumulator. The banded
// callers ignore it -- a band a frame cannot contribute to is normal -- but the
// online path merges each frame exactly once, so false there means the frame is
// simply absent from the output.
bool merge_comp_band(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                     const Image& robustness, int tile_size,
                     Image& num_band, Image& den_band, int y0, const Config& cfg,
                     int frame_id = -1);

void merge_ref_band(const Image& ref_raw, const CovField& covs,
                    Image& num_band, Image& den_band, int y0, const Config& cfg,
                    const Image* acc_rob = nullptr);

void merge_comp(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                const Image& robustness, int tile_size,
                Image& num, Image& den, const Config& cfg);
void merge_ref(const Image& ref_raw, const CovField& covs,
               Image& num, Image& den, const Config& cfg,
               const Image* acc_rob = nullptr);

// Accumulator health before num/den (for green/black speckle debugging).
struct AccumDiag {
    size_t pixels = 0;
    size_t den_zero[3] = {0, 0, 0};
    size_t den_tiny[3] = {0, 0, 0};      // 0 < d < 1e-12
    size_t den_nonfinite[3] = {0, 0, 0};
    size_t num_nonfinite[3] = {0, 0, 0};
    size_t only_green = 0;               // G>0, R==0, B==0
    size_t rgb_all_zero = 0;
};
void accumulate_diag(const Image& num, const Image& den, AccumDiag& d);
void accumulate_diag_ptr(const f32* nump, const f32* denp, size_t n_pixels,
                         int c, AccumDiag& d);
std::string format_accum_diag(const AccumDiag& d);

} // namespace hhsr
