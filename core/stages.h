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

// 1 where the flow field is irregular over the 3x3 tile neighbourhood -- the
// r_Mt test (Wronski et al. Eq. 7/8, literally: raw max-min span, no
// detrending). sx/sy scale the stored displacements into r_Mt's units.
std::vector<uint32_t> compute_motion_irregular(const FlowField& flow, f32 Mt,
                                               f32 sx, f32 sy, int num_threads);

// Builds a raw-pixel tile-grid FlowField from a dense per-guide-pixel flow
// field produced by an external neural flow estimator (PWCNet), as a
// drop-in alternative to align()+flow_to_raw_tile_grid for any downstream
// consumer. dense_flow: dx plane (guide_h*guide_w floats) followed by dy
// plane (guide_h*guide_w floats), values in GUIDE-pixel units -- the layout
// a Core ML (1,2,guide_h,guide_w) MLMultiArray output has. See align.cpp
// for what's intentionally left unset (aperture_limited, match_ambiguous,
// motion_irregular) and why.
FlowField flow_from_dense_guide(const f32* dense_flow, int guide_h, int guide_w,
                                int raw_h, int raw_w, int tile_size,
                                f32 r_Mt, int num_threads);

struct Pyramid { std::vector<Image> levels; std::vector<int> abs_factors; };
Pyramid build_pyramid(const Image& grey, const std::vector<int>& factors);

Image compute_gradients(const Image& grey);
Image gaussian_blur(const Image& src, float sigma);

// Circular pad so height/width are multiples of tile_size (alignment.init_alignment).
Image pad_image_circular(const Image& img, int tile_size);

// ---- align.cpp ----------------------------------------------------------
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
    // Feature channel 17 for the learned mask: reference high-frequency
    // energy, guide resolution, 1 channel. A function of the reference alone,
    // so it is built ONCE per burst by ensure_robustness_nn_ref_hf rather
    // than per pixel per comparison frame -- a 5x5 box over 3 channels inside
    // the per-pixel feature loop is 75 reads/px at 3 MP on every frame, which
    // the runtime budget does not have. Empty unless the learned mask runs.
    Image nn_hf;
    // The reference guide's LUMA, guide resolution, 1 channel, UNSMOOTHED.
    //
    // Not derivable from `means`: those are 3x3 box means (a 6x6 raw support)
    // and the whole point of the spatial channels is the structure that box
    // destroys. A shifted edge and an unshifted one have nearly the same 3x3
    // mean; they do not have the same pixel values.
    //
    // Filled by init_robustness straight from ref_raw, which is the only
    // place the reference raw is still in scope -- including on the Metal
    // path, where means/stds stay GPU-resident. 12 MB at 3 MP; empty unless
    // the learned correction is enabled.
    Image nn_luma;
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

// MC noise std at brightness in [0,1]: std_curve[round(1000*b)] (fast_monte_carlo).
f32 noise_std_at_brightness(f32 brightness, f32 alpha, f32 beta);
f32 noise_std_at_brightness(f32 brightness, const Config& cfg);

// Full noise curves (1001 bins) for GPU upload — same cache as CPU robustness.
void fetch_noise_curves(f32 alpha, f32 beta,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve);
void fetch_noise_curves(const Config& cfg,
                        std::vector<f32>& std_curve, std::vector<f32>& diff_curve);

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
//   13    log1p(d^2/sigma^2), Wronski Eq. 6 -- the analytic mask's own
//         decision statistic, compressed because it is heavy-tailed
//   14    the analytic mask's R (Eq. 5). Evidence, NOT a target: the network
//         is trained against measured merge harm, not against this value, so
//         it is free to overrule it (and must, since Eq. 5 is what fails on
//         the camouflaged-misalignment case).
//   15    |flow - component-wise median of the 3x3 tile neighbourhood|, raw
//         px. Smooth camera rotation makes this ~0 while the span (11) is
//         large; a single tile locked onto the wrong match makes it the size
//         of the error. Separating those two is why rotation stopped being
//         read as danger.
//   16    the neighbourhood max of channel 15 -- the local roughness scale to
//         judge 15 against, so a big residual amid real parallax reads
//         differently from the same residual in a smoothly-flowing region.
//   17    reference local high-frequency energy (3x3 means against their own
//         5x5 box average). Says how FINE the structure is, which is what
//         decides whether a subpixel error costs anything; neither the mean
//         nor the std carries it.
//
//   --- match quality: the shape of the block-matching cost surface --------
//   18    log(cost at the chosen offset / best cost in a +-4 grey-px window).
//         0 when the offset being judged IS the best correspondence on offer,
//         positive in proportion to how much better something nearby was.
//   19    log(best rival outside the winner basin / best cost). How ISOLATED
//         that minimum is. Aliasing changes the residual at the BOTTOM of the
//         cost surface without flattening it, so a correctly aligned but
//         aliased tile still has a sharp isolated minimum, while a tile
//         matched onto similar-looking content elsewhere has several
//         near-ties. No photometric channel separates those two; the cost
//         surface shape does. Zero when match_q is null.
//
//   --- the spatial residual: what statistics provably cannot carry --------
//
//   Channels 0-19 are all local STATISTICS, and a rotation error can leave
//   every one of them almost unchanged while shifting an edge by a pixel:
//
//       reference edge:   |          warped edge:     |
//
//   Same mean, same variance, same noise, same flow smoothness -- and a
//   duplicated edge in the merged output. Only spatial structure separates
//   them, so the network is given the residual itself and the reference
//   gradient it has to be read against. Because the network is fully
//   convolutional with a 7-guide-pixel receptive field, handing it these as
//   PLANES gives it a 15x15 guide-pixel (30 raw-pixel) patch around every
//   pixel at the cost of four extra reads -- the same information a per-pixel
//   patch gather would provide, without the gather.
//
//   Seeing the residual VARY ACROSS a tile is the point. A translation-only
//   flow that is fitting a locally rotating scene leaves a residual that
//   ramps from one side of the tile to the other; a correct match leaves a
//   residual that is noise everywhere. That difference is invisible to any
//   pointwise statistic and obvious to a dilated convolution.
//
//   20    reference guide luma, unsmoothed
//   21    comparison guide luma, unsmoothed, sampled bilinearly where the
//         estimated flow points (flow is in RAW px, so the guide offset is
//         half of it)
//   22    signed residual, channel 20 minus channel 21
//   23    d(reference luma)/dx, guide px
//   24    d(reference luma)/dy, guide px
//   25    regularised local displacement estimate: the residual projected
//         onto the reference gradient, r*|g| / (|g|^2 + eps), clamped to
//         +-4 guide px. Where the gradient is real this is directly "how far
//         off, in pixels, along the edge normal"; where it is not, it decays
//         to 0 instead of exploding. This is the channel that reads a shifted
//         edge as a shift rather than as a brightness change, and its
//         VARIATION over a tile is the rotation signature.
//   26    residual over the expected sensor noise sigma -- the same residual
//         as 22, in the units that decide whether it is signal at all.
//
// Kept in portable C++ next to the analytic mask because this layout is a
// contract with tools/rob_nn/rob_real.cpp, which writes the training set;
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
// The analytic mask's own decision, exposed so the learned mask can take it
// as an input instead of rediscovering Eq. 5-9 from the raw statistics, and
// so tools/rob_nn/rob_real.cpp can write the IDENTICAL value into the
// training set. Two implementations of this formula would drift, and the
// network would train on a hint that differs from the one it is given at
// inference -- so there is exactly one.
//   ref_mean/ref_var/comp_mean: 3 channels each, at the pixel being judged.
//   Mspan: local flow span (Eq. 7), selecting s1 vs s2.
// Returns R (Eq. 5); *ratio_out, if given, receives d^2/sigma^2 (Eq. 6).
f32 robustness_analytic_R(const f32* ref_mean, const f32* ref_var,
                          const f32* comp_mean, f32 Mspan, const Config& cfg,
                          f32* ratio_out);

// Fills ref_stats.nn_hf (feature channel 17) if it is empty and the
// reference means are host-resident. Idempotent and cheap to call; must run
// once per burst before build_robustness_nn_features, which reads the cached
// plane rather than recomputing it.
void ensure_robustness_nn_ref_hf(RefStats& ref_stats, const Config& cfg);

// rows: 0 uses the on-device strip height (kRobustnessNnStripRows + 2 *
// kRobustnessNnHalo). The training generator passes the full plane height
// instead, so features and labels are indexed in one coordinate system.
// comp_luma / match_q are the inputs for channels 20-26 and 18-19. Either may
// be null, in which case those channels are written as zero -- exactly
// equivalent to the model never having had them, which is what a caller that
// has not measured them must accept.
Image build_robustness_nn_features(const RefStats& ref_stats, const Image& comp_means,
                                   const FlowField& flow, int tile_size,
                                   const Config& cfg, int y0,
                                   bool raw_res = false, int rows = 0,
                                   const Image* comp_luma = nullptr,
                                   const std::vector<f32>* match_q = nullptr);

// Per-tile match quality for feature channels 18-19, measured on guide-
// resolution LUMA planes (the reference and the comparison, both unwarped)
// over the flow own raw-pixel tile grid. Two floats per tile.
//
// Recomputed here rather than carried out of align.cpp: the aligner tracks
// second_dist at two search sites across every pyramid level with their own
// upsample and fallback paths, and threading a continuous value through all
// of them is a large change to make before knowing the statistic earns its
// keep on the multiplicative model.
std::vector<f32> measure_match_quality(const Image& ref_luma, const Image& comp_luma,
                                       const FlowField& flow, int tile_size,
                                       const Config& cfg);

// Guide-resolution luma (the mean of the guide colour planes), unsmoothed.
// Feature channels 20-26 and measure_match_quality both read it; exposed so
// tools/rob_nn builds the identical plane rather than a second copy of the
// formula.
Image guide_luma(const Image& guide);

// The analytic mask alone, Eq. 5-9, exactly as it has always been. Split out
// so the learned correction multiplies a value it cannot alter, and so
// tools/rob_nn can write the same R_normal into the training set that the app
// multiplies at inference.
Image compute_robustness_analytic(const Image& comp_raw, const RefStats& ref_stats,
                                  const FlowField& flow, int tile_size,
                                  const Config& cfg, Image* s_select_out = nullptr);

// R_final. The analytic mask above, multiplied by the learned correction
// confidence when cfg.use_neural_robustness is on and the model loaded.
Image compute_robustness(const Image& comp_raw, const RefStats& ref_stats,
                         const FlowField& flow, int tile_size, const Config& cfg,
                         Image* s_select_out = nullptr);

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
