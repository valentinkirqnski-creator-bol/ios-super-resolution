import AVFoundation
import Photos
import UIKit
import SwiftUI
import Combine
import ImageIO
import AudioToolbox

/// Final save format after SR (DNG always produced; JPG is a tone-mapped export).
enum ExportFormat: String, CaseIterable, Identifiable, Codable {
    case dng
    case jpg

    var id: String { rawValue }

    var label: String {
        switch self {
        case .dng: return "DNG"
        case .jpg: return "JPG"
        }
    }
}

/// Back wide (1×), ultra-wide (0.5×), or front selfie camera.
enum CameraSelection: String, CaseIterable, Identifiable {
    case wide
    case ultraWide
    case telephoto
    case front

    var id: String { rawValue }

    var label: String {
        switch self {
        case .wide: return "1×"
        case .ultraWide: return "0.5×"
        case .telephoto: return "Tele"
        case .front: return "Front"
        }
    }
}

/// Lens/zoom mode for back-camera RAW capture (2× = center crop before SR).
enum LensZoomMode: Equatable {
    case ultraWide
    case wide1x
    case wide2x
    case telephoto

    var label: String {
        switch self {
        case .ultraWide: return "0.5×"
        case .wide1x: return "1×"
        case .wide2x: return "2×"
        case .telephoto: return "Tele"
        }
    }

}

/// Algorithm zoom/output size for the 1x wide camera.
enum OutputResolutionMode: String, CaseIterable, Identifiable {
    case native12mp
    case super48mp

    var id: String { rawValue }

    var label: String {
        switch self {
        case .native12mp: return "12MP"
        case .super48mp: return "48MP"
        }
    }

    var algorithmScale: Float {
        switch self {
        case .native12mp: return 1.0
        case .super48mp: return 2.0
        }
    }
}

/// Holds the C++ algorithm tuning parameters for live adjustments.
struct TuningParams: Equatable, Codable {
    // Match 460-main params.py
    var r_t: Float = 0.12
    var r_s1: Float = 2.0
    var r_s2: Float = 12.0
    var r_Mt: Float = 0.8
    // true = full-res FFT low-pass, false = 2x2 Bayer quad average at half res
    var alignment_grey_fft: Bool = true
    var flow_regularize_aperture_ratio: Float = 0.15
    var flow_reject_1d_ambiguity_ratio: Float = 1.10
    var k_detail: Float = 0.17
    /// See Config::k_denoise -- 0 collapses the kernel in flat regions rather
    /// than widening it; 3.0 is the low end of the SNR-tuned range.
    var k_denoise: Float = 3.0
    var k_stretch: Float = 4.0
    /// Drive the merge kernel's anisotropy continuously from the structure
    /// tensor rather than switching to the full stretch only above 0.9025.
    /// Zero-floor the anisotropy law: isotropic detail gets round kernels
    /// instead of a minimum 2.5:0.75 stretch along a noise orientation.
    /// OFF = python-z's `linear` law exactly (w = A/2). See core/types.h.
    /// Exponent on the stretch weight: higher concentrates elongation onto
    /// genuinely coherent edges. 2.0 was fitted to the distant-text finding.
    /// Store the online merge accumulator as fp16 (arithmetic stays fp32).
    /// Halves its RAM and the merge's memory traffic; output shifts ~1-2 LSB.
    var merge_fp16_accumulator: Bool = true
    /// Skip merge taps and overlap hypotheses whose weight is numerically
    /// negligible (< 3.4e-4 of the centre tap / < 5% window weight).
    var merge_fast_weights: Bool = true
    /// Store the output DNG un-white-balanced (real AsShotNeutral) so editors
    /// keep the sensor's full highlight headroom (~1 stop of R/B).
    var dng_store_unwhitened: Bool = true
    /// Quadratic sub-cell fit at each block-matching winner (Wronski's
    /// sub-pixel estimator). Integer flow becomes ~0.1-0.25px flow per level.
    var bm_subpixel_quadratic: Bool = true
    /// Anti-aliased decimation for the alignment grey: half-phase binomial
    /// [1 3 3 1]/8 instead of the 2x2 box. Same lattice, less alias wobble.
    var grey_decimate_lowpass: Bool = true
    /// Final ICA refinement at full raw resolution on the FFT grey, seeded by
    /// the decimate flow. Closes the decimate path's sub-pixel gap to FFT.
    var align_fullres_polish: Bool = true
    /// At motion boundaries, pick the best single tile vector instead of
    /// blending two different motions. Smooth regions stay bilinear.
    var flow_boundary_selection: Bool = true
    /// Catmull-Rom bicubic tile-flow sampling (C1) instead of bilinear (C0).
    var flow_bicubic_sampling: Bool = false
    /// HDR+-style overlapped-tile merge: Ts at stride Ts/2, per-tile measured
    /// flow (no interpolation), raised-cosine result blending. Decimate only.
    /// Standalone: does NOT require Smooth Tile Flow (HDR+ has no flow
    /// interpolation; overlap-blending is the alternative to it).
    var flow_overlap_merge: Bool = false
    var k_shrink: Float = 2.0
    /// D gate: below-threshold gradients are routed to denoising instead of
    /// super-resolution. GAT-domain units (noise sigma ~ 1). Only applied
    /// when d_thresh_manual is on; otherwise SNR auto-tune sets them per
    /// burst (0.71-0.81 / 1.0-1.24).
    var d_thresh_manual: Bool = false
    /// Burst-only: halve the auto-metered exposure duration (2x shutter
    /// speed) and raise ISO to compensate, so each frame carries half the
    /// motion blur. The merge averages the extra noise back out; blur it
    /// cannot undo. Manual exposure mode is unaffected.
    var burst_fast_shutter: Bool = false
    /// Lossless-JPEG (Compression 7) tiled DNG: bit-identical pixels, 2-3x
    /// smaller and faster to save than uncompressed. Standard DNG codec.
    var dng_lossless_jpeg: Bool = true
    var d_th: Float = 0.76
    var d_tr: Float = 1.12
    var snr_auto_tune: Bool = true
    var alignment_tile_size: Int = 0
    /// Off merges every frame at full weight everywhere. Diagnostic: it shows
    /// what the alignment actually produced, with no mask hiding the errors.
    var robustness_enabled: Bool = true
    var robustness_save_mask: Bool = true
    var accumulated_robustness_denoiser_enabled: Bool = false
    /// 0 = pick the cheaper merge architecture by working-set size, 1 = always
    /// band, 2 = always merge online. Online keeps memory flat in frame count
    /// but its accumulator scales with output pixels, so it is not always the
    /// smaller of the two.
    var merge_arch: Int32 = 0
    /// Adapt the enlargement to the merged frame count instead of the
    /// reference implementation's step. Off reproduces the reference exactly.
    /// Run ICA after block matching on every pyramid level, as the reference
    /// implementation does, instead of only on the finest. Half-res 2x2 grey
    /// only -- the full-res FFT path is unaffected either way.
    var align_ica_per_level: Bool = true
    /// Extend the above to the full-res FFT grey. Without it that path feeds
    /// integer-only flow into a finest level whose search radius is 1, so the
    /// correction budget is already spent when level 0 starts. Costs roughly
    /// +120MB at 12MP, because the reference gradient cache goes resident.
    var align_ica_per_level_fft: Bool = false
    /// ImageStackAlignator's rule for unreliable block matches: when a
    /// tile's best and second-best costs are near-tied (flat patch, aperture,
    /// repetition -- no precise shift determinable), apply NO shift and keep
    /// the seed from the coarser level, instead of
    /// trusting a match indistinguishable from noise. Acts on the flow
    /// itself, unlike the s1 demotion, which is inert under rotation where
    /// every tile is on s1 already. Off by default -- A/B before adopting.
    var align_ambiguous_fallback_enabled: Bool = false
    /// Debug: zero the noise model as read by the robustness mask ONLY.
    /// R is then scored from the raw measured local variance and the raw
    /// (unshrunk) pixel difference. SNR auto-tune, the alignment tile size
    /// and kernel estimation are unaffected -- the gate lives on the mask's
    /// own curve builds, not the shared noise accessors.
    var debug_noise_model_disabled: Bool = false
    /// Sample the per-tile flow bilinearly between tile centres everywhere it
    /// is consumed, instead of taking the containing tile's vector.
    var flow_bilinear_sampling: Bool = true
    /// Estimate one global rotation+translation per comparison frame (FFT
    /// correlation sweep over candidate angles) and seed the coarsest
    /// block-matching level with it, so the search only has to find the
    /// residual. ImageStackAlignator's pre-alignment stage.
    var prealign_enabled: Bool = false
    /// Accumulate R-G and B-G instead of R and B, adding the reconstructed
    /// green back at the end. Removes the coloured fringing along edges.
    var merge_chroma_difference: Bool = false
    /// Replace the steerable (structure-tensor-shaped) merge kernels with a
    /// fixed round Gaussian (python-z's merging.kernel=iso): every pixel of
    /// every frame merges with the same isotropic weight and the whole
    /// kernel-estimation output is bypassed. Diagnostic mode: if colour
    /// fringing survives with this on, kernel estimation is exonerated and
    /// the cause is upstream (alignment/robustness).
    var merge_kernel_iso: Bool = false
    /// Where the comparison frames contributed exactly zero weight (every
    /// frame rejected, R = 0), reconstruct from the reference with a FIXED
    /// non-adaptive kernel (iso Gaussian, sigma 0.45 raw px) instead of the
    /// shaped/denoising kernels: the plain reference as an ordinary viewer
    /// would resample it -- noise essentially intact, no k_denoise, no
    /// adaptivity, no nearest-sample edge aliasing. Hard switch on zero
    /// coverage.
    var merge_uncovered_passthrough: Bool = false
    /// Eq. 4 selection law. On = the hard A > 1.95 threshold (commit
    /// 832f7b8's behaviour): kernels stay ROUND everywhere except at
    /// near-perfectly coherent edges, giving the smooth homogeneous
    /// rendition. Off = python-z's 12ce005 linear law (the default):
    /// anisotropy lerps continuously with A, so ordinary textured content
    /// gets thin oriented kernels -- crisper, can read as oversharpened.
    var kernel_selection_hard: Bool = false
    /// 832f7b8's soften_inv_cov: cap merge kernel sharpness at sigma ~ 0.18
    /// raw px on every fetch. The other half of the 832f7b8 rendition.
    var merge_soften_inv_cov: Bool = false
    /// HDR+ mode: replace the whole super-resolution pipeline with the HDR+
    /// burst align + merge (hdr-plus-master's Halide implementation, ported
    /// to the core with a Metal path). A denoiser, not an upscaler: the
    /// output DNG is at sensor resolution, demosaicked bilinearly. All the
    /// super-resolution tuning (kernels, robustness, flow) is bypassed
    /// while this is on.
    var hdrplus_mode: Bool = false
    /// Robustness statistics computed in colour-transformed linear sRGB
    /// instead of the camera-native space (the IPOL author's cross-camera
    /// direction). Guide and noise model transform together. CPU-only mask
    /// path while on (no Metal twin yet) -- slower per frame.
    var robustness_color_space: Bool = false
    /// ISA mode: run ImageStackAlignator -- algorithmically identical to
    /// ImageStackAlignator-master (vendored 1:1 port) -- instead of the
    /// super-resolution pipeline. Rotation-scan pre-alignment, all-pairs
    /// patch tracking, dense Lucas-Kanade flow, ISA's robustness and
    /// kernel-regression merge, ISA's own colour rendering baked into the
    /// output DNG (input resolution). CPU reference implementation for
    /// now: slow (the rotation scan especially) but exact; Metal ports of
    /// the stages follow, verified against it. Overrides HDR+ mode.
    var isa_mode: Bool = false
    /// Re-estimate the flow densely, one Lucas-Kanade solve per lattice cell
    /// at the finest pitch that fits the buffer budget, instead of the
    /// half-pitch blend-or-select densify. ImageStackAlignator's
    /// lucasKanadeOptim. This is the layer that removes the tile lattice.
    var flow_dense_lk_enabled: Bool = false
    /// Computes d^2/sigma^2/R at RAW resolution (Dodgson-quadratic upscale +
    /// flow-warp of the guide-resolution local stats) instead of directly at
    /// guide resolution, which this port otherwise does. The statistics stay
    /// half-resolution either way -- what changes is where the ratio is
    /// evaluated and where the local-min runs: at raw resolution the min is
    /// a double 5x5 (= 9x9 raw), keeping the paper's ~10x10-raw physical
    /// margin while the rejection boundary lands at raw-pixel precision. Only takes effect when "Alignment Grey:
    /// FFT" below is OFF (Decimate) -- that path's flow is already coarser
    /// than FFT's, so the guide-resolution mask on top compounds two sources
    /// of lost precision. ~4x the pixel count for the mask.
    var robustness_raw_resolution_enabled: Bool = false
    // JPEG/preview rendering (core/render_isp.cpp). Defaults mirror the C++
    // exactly; they were tuned against real DNG/reference pairs, so changing one
    // here without changing the other silently splits the two.
    var isp_enabled: Bool = true
    var isp_exposure_ev: Float = 0.0
    var isp_highlight_knee: Float = 0.88
    var isp_local_strength: Float = 0.75
    var isp_highlight: Float = 0.65
    var isp_shadow: Float = 0.28
    var isp_black_point: Float = 0.065
    var isp_warmth: Float = 0.05
    var isp_colour_strength: Float = 1.0
    var isp_contrast: Float = 0.62
    var isp_vibrance: Float = 0.50
    /// Chroma noise reduction, detail-gated (noise-sized deviations smoothed,
    /// saturated small objects preserved). Luma is preserved exactly.
    var isp_chroma_denoise: Float = 0.0
    var isp_chroma_radius: Float = 12.0
    var isp_saturation: Float = 1.0
    var isp_local_contrast: Float = 0.30
    var isp_skin_protect: Bool = true

    var acc_rob_adaptive: Bool = true
    /// Only used when acc_rob_adaptive is off.
    var acc_rob_max_frame_count: Float = 2.0
    var acc_rob_rad_max: Float = 2.0
    var acc_rob_max_multiplier: Float = 8.0

    /// App defaults — also applied by the settings Reset button.
    static let appDefaults = TuningParams()

    /// Legacy name used by the Reset button.
    static let ghostReductionPreset = TuningParams.appDefaults

    enum CodingKeys: String, CodingKey {
        case r_t, r_s1, r_s2, r_Mt
        case alignment_grey_fft
        case flow_regularize_aperture_ratio
        case flow_reject_1d_ambiguity_ratio
        case k_detail, k_denoise, k_stretch, k_shrink
        case d_thresh_manual, d_th, d_tr
        case burst_fast_shutter
        case dng_lossless_jpeg
        case merge_fp16_accumulator
        case merge_fast_weights
        case dng_store_unwhitened
        case bm_subpixel_quadratic
        case grey_decimate_lowpass
        case align_fullres_polish
        case flow_boundary_selection
        case flow_bicubic_sampling
        case flow_overlap_merge
        case snr_auto_tune, alignment_tile_size
        case robustness_enabled, robustness_save_mask
        case accumulated_robustness_denoiser_enabled
        case merge_arch
        case acc_rob_adaptive, acc_rob_max_frame_count, align_ica_per_level
        case align_ica_per_level_fft
        case align_ambiguous_fallback_enabled
        case debug_noise_model_disabled, robustness_raw_resolution_enabled
        case flow_bilinear_sampling
        case prealign_enabled
        case merge_chroma_difference
        case merge_kernel_iso
        case merge_uncovered_passthrough
        case kernel_selection_hard, merge_soften_inv_cov
        case hdrplus_mode
        case isa_mode
        case robustness_color_space
        case flow_dense_lk_enabled
        case isp_enabled, isp_exposure_ev, isp_local_strength, isp_highlight
        case isp_shadow, isp_black_point, isp_warmth, isp_contrast
        case isp_vibrance, isp_saturation, isp_local_contrast, isp_skin_protect
        case isp_chroma_denoise, isp_chroma_radius
        case isp_colour_strength, isp_highlight_knee
        case acc_rob_rad_max, acc_rob_max_multiplier
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        r_t = try c.decodeIfPresent(Float.self, forKey: .r_t) ?? r_t
        r_s1 = try c.decodeIfPresent(Float.self, forKey: .r_s1) ?? r_s1
        r_s2 = try c.decodeIfPresent(Float.self, forKey: .r_s2) ?? r_s2
        r_Mt = try c.decodeIfPresent(Float.self, forKey: .r_Mt) ?? r_Mt
        alignment_grey_fft = try c.decodeIfPresent(Bool.self, forKey: .alignment_grey_fft) ?? alignment_grey_fft
        flow_regularize_aperture_ratio = try c.decodeIfPresent(Float.self, forKey: .flow_regularize_aperture_ratio) ?? flow_regularize_aperture_ratio
        flow_reject_1d_ambiguity_ratio = try c.decodeIfPresent(Float.self, forKey: .flow_reject_1d_ambiguity_ratio) ?? flow_reject_1d_ambiguity_ratio
        k_detail = try c.decodeIfPresent(Float.self, forKey: .k_detail) ?? k_detail
        k_denoise = try c.decodeIfPresent(Float.self, forKey: .k_denoise) ?? k_denoise
        k_stretch = try c.decodeIfPresent(Float.self, forKey: .k_stretch) ?? k_stretch
        k_shrink = try c.decodeIfPresent(Float.self, forKey: .k_shrink) ?? k_shrink
        d_thresh_manual = try c.decodeIfPresent(Bool.self, forKey: .d_thresh_manual) ?? d_thresh_manual
        burst_fast_shutter = try c.decodeIfPresent(Bool.self, forKey: .burst_fast_shutter) ?? burst_fast_shutter
        dng_lossless_jpeg = try c.decodeIfPresent(Bool.self, forKey: .dng_lossless_jpeg) ?? dng_lossless_jpeg
        d_th = try c.decodeIfPresent(Float.self, forKey: .d_th) ?? d_th
        d_tr = try c.decodeIfPresent(Float.self, forKey: .d_tr) ?? d_tr
        merge_fp16_accumulator = try c.decodeIfPresent(Bool.self, forKey: .merge_fp16_accumulator) ?? merge_fp16_accumulator
        merge_fast_weights = try c.decodeIfPresent(Bool.self, forKey: .merge_fast_weights) ?? merge_fast_weights
        dng_store_unwhitened = try c.decodeIfPresent(Bool.self, forKey: .dng_store_unwhitened) ?? dng_store_unwhitened
        bm_subpixel_quadratic = try c.decodeIfPresent(Bool.self, forKey: .bm_subpixel_quadratic) ?? bm_subpixel_quadratic
        grey_decimate_lowpass = try c.decodeIfPresent(Bool.self, forKey: .grey_decimate_lowpass) ?? grey_decimate_lowpass
        align_fullres_polish = try c.decodeIfPresent(Bool.self, forKey: .align_fullres_polish) ?? align_fullres_polish
        flow_boundary_selection = try c.decodeIfPresent(Bool.self, forKey: .flow_boundary_selection) ?? flow_boundary_selection
        flow_bicubic_sampling = try c.decodeIfPresent(Bool.self, forKey: .flow_bicubic_sampling) ?? flow_bicubic_sampling
        flow_overlap_merge = try c.decodeIfPresent(Bool.self, forKey: .flow_overlap_merge) ?? flow_overlap_merge
        snr_auto_tune = try c.decodeIfPresent(Bool.self, forKey: .snr_auto_tune) ?? snr_auto_tune
        alignment_tile_size = try c.decodeIfPresent(Int.self, forKey: .alignment_tile_size) ?? alignment_tile_size
        robustness_enabled = try c.decodeIfPresent(Bool.self, forKey: .robustness_enabled) ?? robustness_enabled
        robustness_save_mask = try c.decodeIfPresent(Bool.self, forKey: .robustness_save_mask) ?? robustness_save_mask
        accumulated_robustness_denoiser_enabled = try c.decodeIfPresent(Bool.self, forKey: .accumulated_robustness_denoiser_enabled) ?? accumulated_robustness_denoiser_enabled
        merge_arch = try c.decodeIfPresent(Int32.self, forKey: .merge_arch) ?? merge_arch
        acc_rob_adaptive = try c.decodeIfPresent(Bool.self, forKey: .acc_rob_adaptive) ?? acc_rob_adaptive
        isp_enabled = try c.decodeIfPresent(Bool.self, forKey: .isp_enabled) ?? isp_enabled
        isp_exposure_ev = try c.decodeIfPresent(Float.self, forKey: .isp_exposure_ev) ?? isp_exposure_ev
        isp_highlight_knee = try c.decodeIfPresent(Float.self, forKey: .isp_highlight_knee) ?? isp_highlight_knee
        isp_local_strength = try c.decodeIfPresent(Float.self, forKey: .isp_local_strength) ?? isp_local_strength
        isp_highlight = try c.decodeIfPresent(Float.self, forKey: .isp_highlight) ?? isp_highlight
        isp_shadow = try c.decodeIfPresent(Float.self, forKey: .isp_shadow) ?? isp_shadow
        isp_black_point = try c.decodeIfPresent(Float.self, forKey: .isp_black_point) ?? isp_black_point
        isp_warmth = try c.decodeIfPresent(Float.self, forKey: .isp_warmth) ?? isp_warmth
        isp_colour_strength = try c.decodeIfPresent(Float.self, forKey: .isp_colour_strength) ?? isp_colour_strength
        isp_contrast = try c.decodeIfPresent(Float.self, forKey: .isp_contrast) ?? isp_contrast
        isp_vibrance = try c.decodeIfPresent(Float.self, forKey: .isp_vibrance) ?? isp_vibrance
        isp_chroma_denoise = try c.decodeIfPresent(
            Float.self, forKey: .isp_chroma_denoise) ?? isp_chroma_denoise
        isp_chroma_radius = try c.decodeIfPresent(
            Float.self, forKey: .isp_chroma_radius) ?? isp_chroma_radius
        isp_saturation = try c.decodeIfPresent(Float.self, forKey: .isp_saturation) ?? isp_saturation
        isp_local_contrast = try c.decodeIfPresent(Float.self, forKey: .isp_local_contrast) ?? isp_local_contrast
        isp_skin_protect = try c.decodeIfPresent(Bool.self, forKey: .isp_skin_protect) ?? isp_skin_protect
        align_ica_per_level = try c.decodeIfPresent(Bool.self, forKey: .align_ica_per_level) ?? align_ica_per_level
        align_ica_per_level_fft = try c.decodeIfPresent(Bool.self, forKey: .align_ica_per_level_fft) ?? align_ica_per_level_fft
        align_ambiguous_fallback_enabled = try c.decodeIfPresent(Bool.self, forKey: .align_ambiguous_fallback_enabled) ?? align_ambiguous_fallback_enabled
        debug_noise_model_disabled = try c.decodeIfPresent(Bool.self, forKey: .debug_noise_model_disabled) ?? debug_noise_model_disabled
        flow_bilinear_sampling = try c.decodeIfPresent(Bool.self, forKey: .flow_bilinear_sampling) ?? flow_bilinear_sampling
        prealign_enabled = try c.decodeIfPresent(Bool.self, forKey: .prealign_enabled) ?? prealign_enabled
        merge_chroma_difference = try c.decodeIfPresent(Bool.self, forKey: .merge_chroma_difference) ?? merge_chroma_difference
        merge_kernel_iso = try c.decodeIfPresent(Bool.self, forKey: .merge_kernel_iso) ?? merge_kernel_iso
        merge_uncovered_passthrough = try c.decodeIfPresent(Bool.self, forKey: .merge_uncovered_passthrough) ?? merge_uncovered_passthrough
        kernel_selection_hard = try c.decodeIfPresent(Bool.self, forKey: .kernel_selection_hard) ?? kernel_selection_hard
        merge_soften_inv_cov = try c.decodeIfPresent(Bool.self, forKey: .merge_soften_inv_cov) ?? merge_soften_inv_cov
        hdrplus_mode = try c.decodeIfPresent(Bool.self, forKey: .hdrplus_mode) ?? hdrplus_mode
        isa_mode = try c.decodeIfPresent(Bool.self, forKey: .isa_mode) ?? isa_mode
        robustness_color_space = try c.decodeIfPresent(Bool.self, forKey: .robustness_color_space) ?? robustness_color_space
        flow_dense_lk_enabled = try c.decodeIfPresent(Bool.self, forKey: .flow_dense_lk_enabled) ?? flow_dense_lk_enabled
        robustness_raw_resolution_enabled = try c.decodeIfPresent(Bool.self, forKey: .robustness_raw_resolution_enabled) ?? robustness_raw_resolution_enabled
        acc_rob_max_frame_count = try c.decodeIfPresent(Float.self, forKey: .acc_rob_max_frame_count) ?? acc_rob_max_frame_count
        acc_rob_rad_max = try c.decodeIfPresent(Float.self, forKey: .acc_rob_rad_max) ?? acc_rob_rad_max
        acc_rob_max_multiplier = try c.decodeIfPresent(Float.self, forKey: .acc_rob_max_multiplier) ?? acc_rob_max_multiplier
    }
}

/// Owns the capture session, performs a Bayer RAW (DNG) burst, then runs
/// the multi-frame super-resolution pipeline on a background queue.
final class CameraModel: NSObject, ObservableObject {

    // Published UI state.
    @Published var isSessionRunning = false
    @Published var isBusy = false
    @Published var isCapturing = false
    @Published var isProcessing = false
    @Published var statusText = ""
    /// Sticky NoiseProfile OK/FALLBACK line (not overwritten by Frame N: analyze).
    @Published var noiseDiagText = ""
    @Published var progress: Float = 0
    @Published var lastThumbnail: UIImage?
    @Published var permissionDenied = false
    @Published var cameraSelection: CameraSelection = .wide
    @Published var lensZoomMode: LensZoomMode = .wide1x
    @Published var outputResolutionMode: OutputResolutionMode = {
        if let raw = UserDefaults.standard.string(forKey: "OutputResolutionMode"),
           let mode = OutputResolutionMode(rawValue: raw) {
            return mode
        }
        return .super48mp
    }() {
        didSet { UserDefaults.standard.set(outputResolutionMode.rawValue, forKey: "OutputResolutionMode") }
    }
    /// JPEG export quality (JPG format only). 0.92 default keeps 4:4:4 chroma.
    @Published var jpegExportQuality: Double = {
        let v = UserDefaults.standard.double(forKey: "JPEGExportQuality")
        return v > 0 ? min(1.0, max(0.5, v)) : 0.92
    }() {
        didSet { UserDefaults.standard.set(jpegExportQuality, forKey: "JPEGExportQuality") }
    }
    @Published var exportFormat: ExportFormat = {
        if let raw = UserDefaults.standard.string(forKey: "ExportFormat"),
           let fmt = ExportFormat(rawValue: raw) {
            return fmt
        }
        return .dng
    }() {
        didSet { UserDefaults.standard.set(exportFormat.rawValue, forKey: "ExportFormat") }
    }
    /// Continuous RAW ring buffer: shutter grabs recent frames (no hold-still after tap).
    @Published var zslEnabled: Bool = UserDefaults.standard.bool(forKey: "ZSLEnabled") {
        didSet {
            UserDefaults.standard.set(zslEnabled, forKey: "ZSLEnabled")
            sessionQueue.async { self.applyZSLMode() }
        }
    }
    /// Play a short click when the shutter fires. Defaults on; the key is
    /// stored inverted so an untouched install gets sound without needing a
    /// migration (UserDefaults.bool returns false for a missing key).
    @Published var shutterSoundEnabled: Bool = !UserDefaults.standard.bool(forKey: "ShutterSoundOff") {
        didSet { UserDefaults.standard.set(!shutterSoundEnabled, forKey: "ShutterSoundOff") }
    }

    @Published var zslBufferReady = 0
    @Published var tuningParams: TuningParams = {
        // Bump when app defaults change so existing installs pick up the new preset once.
        let defaultsVersion = 12
        let verKey = "TuningParamsDefaultsVersion"
        if UserDefaults.standard.integer(forKey: verKey) < defaultsVersion {
            UserDefaults.standard.set(defaultsVersion, forKey: verKey)
            let params = TuningParams.appDefaults
            if let data = try? JSONEncoder().encode(params) {
                UserDefaults.standard.set(data, forKey: "TuningParams")
            }
            return params
        }
        if let data = UserDefaults.standard.data(forKey: "TuningParams"),
           let params = try? JSONDecoder().decode(TuningParams.self, from: data) {
            return params
        }
        return TuningParams.appDefaults
    }() {
        didSet {
            if let data = try? JSONEncoder().encode(tuningParams) {
                UserDefaults.standard.set(data, forKey: "TuningParams")
            }
        }
    }
    @Published var availableCameras: [CameraSelection] = [.wide]
    @Published var telephotoLensLabel = "Tele"
    @Published var frameCount: Int = CameraModel.persistedFrameCount() {
        didSet {
            let clamped = min(Self.maxFrameCount, max(Self.minFrameCount, frameCount))
            if frameCount != clamped {
                frameCount = clamped
                return
            }
            UserDefaults.standard.set(clamped, forKey: Self.frameCountDefaultsKey)
            sessionQueue.async { self.activeFrameCount = clamped }
        }
    }

    // Shutter: Auto (A), or manual via log-scaled slider (0…1).
    //
    // Always starts on. Manual exposure is a deliberate per-shot override, and
    // restoring it from the previous launch meant the app could open metering a
    // scene with a shutter speed chosen for a different one, which reads as the
    // camera being broken rather than as a setting still being in effect. The
    // slider positions are still persisted, so turning manual back on restores
    // the last values.
    @Published var shutterIsAuto = true
    /// Both sliders serve double duty: under manual they are the input, and
    /// under Auto the poll keeps them mirroring the metering, so the controls
    /// always show where the camera actually is and switching to manual starts
    /// from there rather than jumping.
    ///
    /// The manual guard in each didSet is what makes that safe -- without it,
    /// the poll's own writes would be echoed straight back at the device every
    /// 200ms. Its absence here was also the reason dragging the shutter did
    /// nothing once the control was already manual: applyShutter was only ever
    /// called from the view's "leaving Auto" branch, which by definition does
    /// not fire when the control is already off Auto, so the value moved, the
    /// label moved, and the device was never told.
    @Published var shutterSlider: Double = CameraModel.persistedShutterSlider() {
        didSet {
            guard !exposureBatch, !shutterIsAuto else { return }
            UserDefaults.standard.set(min(1.0, max(0.0, shutterSlider)),
                                      forKey: Self.shutterSliderDefaultsKey)
            applyShutter()
        }
    }
    /// Manual ISO. Auto by default; when off, isoSlider maps linearly onto the
    /// active format's supported range, which varies per lens and per device.
    /// Starts on auto every launch, for the same reason as the shutter above.
    @Published var isoIsAuto: Bool = true {
        didSet { if !exposureBatch { applyShutter() } }
    }
    @Published var isoSlider: Double = CameraModel.persistedIsoSlider() {
        didSet {
            guard !exposureBatch, !isoIsAuto else { return }
            UserDefaults.standard.set(isoSlider, forKey: Self.isoSliderDefaultsKey)
            applyShutter()
        }
    }
    /// True while setExposureAuto is moving several of the properties above at
    /// once. Their didSets each call applyShutter, so without this a single
    /// mode change would reconfigure the device three times, twice of them
    /// through a half-updated state.
    private var exposureBatch = false
    /// Last ISO the metering settled on, sampled by the auto-exposure poll.
    /// Reported directly rather than derived from isoSlider, which is only a
    /// position on a range that changes with the lens and format.
    @Published private(set) var meteredIso: Float = 0
    @Published var isoMin: Float = 30
    @Published var isoMax: Float = 3000

    private static let isoSliderDefaultsKey = "IsoSlider"

    private static func persistedIsoSlider() -> Double {
        guard UserDefaults.standard.object(forKey: isoSliderDefaultsKey) != nil else { return 0.0 }
        return min(1.0, max(0.0, UserDefaults.standard.double(forKey: isoSliderDefaultsKey)))
    }

    /// Numeric in both modes: under Auto it reports what the metering picked,
    /// which is more use than the word "Auto" next to a control whose own
    /// button already says whether it is on.
    var isoLabel: String {
        if isoIsAuto { return meteredIso > 0 ? "\(Int(meteredIso.rounded()))" : "--" }
        return "\(Int(isoValue.rounded()))"
    }

    var isoValue: Float {
        isoMin + Float(min(1.0, max(0.0, isoSlider))) * (isoMax - isoMin)
    }

    @Published var exposureMinSec: Double = 1.0 / 8000.0
    @Published var exposureMaxSec: Double = 1.0 / 15.0

    static let minFrameCount = 2
    /// Long bursts trade memory for noise reduction. The pipeline keeps each
    /// analyzed frame GPU-resident, so residency grows roughly 110MB per
    /// comparison frame at 12MP; past a point the early upload backs off to
    /// spilling rather than risking jetsam (see pipeline_paths.cpp).
    static let maxFrameCount = 15
    /// Import-from-storage cap when the merge architecture is forced Online:
    /// the online path decodes, merges, and releases one frame at a time, so
    /// its peak memory does not grow with the burst (pipeline_paths.cpp's
    /// online_peak carries a single in-flight frame). 25 keeps the run time
    /// bounded; there is no memory reason for the number.
    static let maxImportFrameCountOnline = 25
    private static let frameCountDefaultsKey = "FrameCount"
    private static let shutterSliderDefaultsKey = "ShutterSlider"

    private static func persistedFrameCount() -> Int {
        guard UserDefaults.standard.object(forKey: frameCountDefaultsKey) != nil else { return 4 }
        let saved = UserDefaults.standard.integer(forKey: frameCountDefaultsKey)
        return min(maxFrameCount, max(minFrameCount, saved))
    }

    private static func persistedShutterSlider() -> Double {
        guard UserDefaults.standard.object(forKey: shutterSliderDefaultsKey) != nil else { return 0.5 }
        let saved = UserDefaults.standard.double(forKey: shutterSliderDefaultsKey)
        return min(1.0, max(0.0, saved))
    }

    private static func fourCCString(_ code: OSType) -> String {
        let chars: [UInt8] = [
            UInt8((code >> 24) & 0xff),
            UInt8((code >> 16) & 0xff),
            UInt8((code >> 8) & 0xff),
            UInt8(code & 0xff)
        ]
        return String(bytes: chars.map { (32...126).contains($0) ? $0 : 46 },
                      encoding: .ascii) ?? ""
    }

    private static func collectInts(_ value: Any?, into out: inout [Int]) {
        if let n = value as? NSNumber {
            out.append(n.intValue)
        } else if let arr = value as? [Any] {
            for v in arr { collectInts(v, into: &out) }
        }
    }

    private static func dngMetadata(from metadata: [String: Any]) -> [String: Any]? {
        (metadata["{DNG}"] as? [String: Any])
            ?? (metadata[kCGImagePropertyDNGDictionary as String] as? [String: Any])
    }

    private static func hasEssentialDirectRawMetadata(_ metadata: [String: Any]) -> Bool {
        guard let dng = dngMetadata(from: metadata) else { return false }
        guard dng["AsShotNeutral"] != nil,
              dng["BlackLevel"] != nil,
              dng["WhiteLevel"] != nil,
              dng["NoiseProfile"] != nil else {
            return false
        }
        return dng["ColorMatrix1"] != nil || dng["ColorMatrix2"] != nil
    }

    private static func cfaPattern(for pixelFormat: OSType,
                                   metadata: [String: Any]) -> [NSNumber]? {
        let fourCC = fourCCString(pixelFormat).lowercased()
        if fourCC.hasPrefix("rgg") { return [0, 1, 1, 2].map { NSNumber(value: $0) } }
        if fourCC.hasPrefix("bgg") { return [2, 1, 1, 0].map { NSNumber(value: $0) } }
        if fourCC.hasPrefix("grb") { return [1, 0, 2, 1].map { NSNumber(value: $0) } }
        if fourCC.hasPrefix("gbr") { return [1, 2, 0, 1].map { NSNumber(value: $0) } }

        let dng = dngMetadata(from: metadata)
        var nums: [Int] = []
        collectInts(dng?["CFAPattern"], into: &nums)
        guard nums.count >= 4 else { return nil }
        let first = nums.prefix(4).map { min(2, max(0, $0)) }
        return first.map { NSNumber(value: $0) }
    }

    private func rawFrameDictionary(from photo: AVCapturePhoto, writingTo url: URL? = nil) -> [String: Any]? {
        guard photo.isRawPhoto, let pixelBuffer = photo.pixelBuffer else { return nil }
        let metadata = photo.metadata
        guard Self.hasEssentialDirectRawMetadata(metadata) else { return nil }
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard let cfa = Self.cfaPattern(for: format, metadata: metadata) else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let planeCount = CVPixelBufferGetPlaneCount(pixelBuffer)
        let width: Int
        let height: Int
        let bytesPerRow: Int
        let baseAddress: UnsafeMutableRawPointer?
        if planeCount > 0 {
            width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
            height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
            bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
            baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
        } else {
            width = CVPixelBufferGetWidth(pixelBuffer)
            height = CVPixelBufferGetHeight(pixelBuffer)
            bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        }
        guard let baseAddress,
              width > 0, height > 0,
              bytesPerRow >= width * MemoryLayout<UInt16>.stride else {
            return nil
        }

        let byteCount = bytesPerRow * height
        let data = Data(bytes: baseAddress, count: byteCount)
        var frame: [String: Any] = [
            "width": NSNumber(value: width),
            "height": NSNumber(value: height),
            "bytesPerRow": NSNumber(value: bytesPerRow),
            "pixelFormat": NSNumber(value: format),
            "cfa": cfa,
            "metadata": metadata as NSDictionary
        ]
        if let url {
            do {
                try data.write(to: url, options: [])
                frame["path"] = url.path
            } catch {
                return nil
            }
        } else {
            frame["data"] = data
        }
        return frame
    }

    override init() {
        super.init()
        NotificationCenter.default.addObserver(
            self, selector: #selector(subjectAreaDidChange),
            name: .AVCaptureDeviceSubjectAreaDidChange, object: nil)
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session", qos: .userInteractive)
    private let processingQueue = DispatchQueue(label: "handheldsr.processing", qos: .userInitiated)
    private let photoOutput = AVCapturePhotoOutput()
    private var device: AVCaptureDevice?
    private var videoInput: AVCaptureDeviceInput?
    private var activeCameraSelection: CameraSelection = .wide
    private var lastBackSelection: CameraSelection = .wide

    private var activeFrameCount = CameraModel.persistedFrameCount()
    /// Session-queue copy of cropZoom, so capture never reads published state
    /// from the wrong thread.
    private var activeCropZoom: CGFloat = 1
    private var currentBurstTotal = CameraModel.persistedFrameCount()
    private var capturesRequested = 0
    private var capturesProcessed = 0
    private var capturedRawFrames: [[String: Any]] = []
    private var capturedDNGs: [URL] = []
    private var burstDir: URL?
    /// Pre-created empty folder so mkdir is off the shutter critical path.
    private var readyBurstDir: URL?
    private var cachedRawPixelFormat: OSType?
    private var cachedMaxPhotoDimensions: CMVideoDimensions?
    private var isAppActive = true
    private var previewSuspended = false
    private var exposureSyncTimer: Timer?

    private enum CaptureKind { case none, burst, zsl }
    private enum BurstInputMode { case undecided, directRaw, dngFallback }
    private var captureKind: CaptureKind = .none
    private var burstInputMode: BurstInputMode = .undecided
    private var zslWanted = false
    private var zslCapturing = false
    /// True while a burst is capturing/processing — ZSL ring + system ZSL stay off.
    private var zslPausedForPipeline = false
    private var zslRawRing: [[String: Any]] = []
    private var zslRing: [URL] = []
    private var zslDir: URL?
    private var zslSeq = 0
    /// Session-queue flag: true while a shutter→process cycle owns the camera.
    private var pipelineBusy = false

    /// Numeric in both modes, for the same reason as isoLabel: under Auto the
    /// poll keeps shutterSlider tracking the metered duration, so formatting it
    /// reports what the camera is doing.
    var shutterLabel: String {
        let sec = durationFromSlider(shutterSlider)
        if sec >= 1.0 { return "\(Int(sec.rounded()))s" }
        let denom = max(1, Int((1.0 / sec).rounded()))
        return "1/\(denom)"
    }

    // MARK: - Setup

    func setAppActive(_ active: Bool) {
        sessionQueue.async {
            self.isAppActive = active
            guard !self.pipelineBusy else { return }
            if active && !self.previewSuspended {
                self.startPreviewIfNeeded()
                if self.zslWanted { self.scheduleNextZSL() }
            } else {
                DispatchQueue.main.async {
                    self.exposureSyncTimer?.invalidate()
                    self.exposureSyncTimer = nil
                }
                if self.session.isRunning {
                    self.session.stopRunning()
                    DispatchQueue.main.async { self.isSessionRunning = false }
                }
            }
        }
    }

    func setPreviewSuspended(_ suspended: Bool) {
        sessionQueue.async {
            self.previewSuspended = suspended
            guard !self.pipelineBusy else { return }
            if suspended {
                DispatchQueue.main.async {
                    self.exposureSyncTimer?.invalidate()
                    self.exposureSyncTimer = nil
                }
                if self.session.isRunning {
                    self.session.stopRunning()
                    DispatchQueue.main.async { self.isSessionRunning = false }
                }
            } else if self.isAppActive {
                self.startPreviewIfNeeded()
                if self.zslWanted { self.scheduleNextZSL() }
            }
        }
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self = self else { return }
            if !granted {
                DispatchQueue.main.async { self.permissionDenied = true }
                return
            }
            self.sessionQueue.async {
                self.configureSession()
                self.discoverCamerasAfterSetup()
                self.purgeStaleCaptureFiles()
            }
        }
    }

    func stop() {
        DispatchQueue.main.async {
            self.exposureSyncTimer?.invalidate()
            self.exposureSyncTimer = nil
        }
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    /// True when the camera is metering for itself. Shutter and ISO share one
    /// state deliberately -- see setExposureAuto.
    var exposureIsAuto: Bool { shutterIsAuto && isoIsAuto }

    /// Hand exposure to the metering, or take it away.
    ///
    /// Shutter and ISO move together because AVFoundation gives us exactly two
    /// usable modes: continuousAutoExposure meters both, and custom sets both.
    /// There is no ISO-priority mode. The previous code kept two independent
    /// flags and, for "manual ISO with auto shutter", took the custom branch
    /// with the duration pinned to whatever the metering had last produced --
    /// so auto-exposure silently stopped adapting while the UI went on
    /// reporting "Auto", and the poll below kept reading that frozen value back
    /// into the slider.
    func setExposureAuto(_ auto: Bool) {
        guard !isBusy else { return }
        // Both flags in one batch: their didSets each call applyShutter, and in
        // between the two assignments the state is half manual.
        exposureBatch = true
        shutterIsAuto = auto
        isoIsAuto = auto
        exposureBatch = false

        persistShutterState()
        applyShutter()
        if auto {
            startAutoExposureSyncIfNeeded()
        } else {
            exposureSyncTimer?.invalidate()
            exposureSyncTimer = nil
        }
    }

    /// Apply the persisted shutter UI state when the camera screen appears.
    func ensureShutterAutoOnLaunch() {
        applyShutter()
        if exposureIsAuto { startAutoExposureSyncIfNeeded() }
    }

    /// Called as a drag on the exposure track begins, before any value is
    /// written. Stopping the poll here rather than after the first value change
    /// is what keeps the metering from overwriting the position mid-drag; the
    /// old ordering wrote the value first and only then asked to leave Auto,
    /// and its isBusy guard could reject that request outright, leaving the
    /// control in Auto with the timer still running -- the "works sometimes".
    func beginManualExposureDrag() {
        guard exposureIsAuto else { return }
        setExposureAuto(false)
    }

    private func persistShutterState() {
        // Only the slider position: whether exposure is manual deliberately does
        // not survive a launch.
        UserDefaults.standard.set(min(1.0, max(0.0, shutterSlider)), forKey: Self.shutterSliderDefaultsKey)
    }

    private func applyShutterSoundSuppression(to settings: AVCapturePhotoSettings) {
        if #available(iOS 18.0, *) {
            if photoOutput.isShutterSoundSuppressionSupported {
                settings.isShutterSoundSuppressionEnabled = true
            }
        }
    }

    /// Keep the photo pipeline warm for immediate RAW bursts.
    private func applyFastCapturePipelineSettings() {
        photoOutput.maxPhotoQualityPrioritization = .speed
        if #available(iOS 17.0, *) {
            if photoOutput.isResponsiveCaptureSupported {
                photoOutput.isResponsiveCaptureEnabled = true
            }
            if photoOutput.isFastCapturePrioritizationSupported {
                photoOutput.isFastCapturePrioritizationEnabled = true
            }
        }
        applyZSLMode()
    }

    private func applyZSLMode() {
        // System ZSL cuts first-frame latency; our disk ring holds the multi-frame burst.
        // Never run the ring while a burst is in flight.
        if zslPausedForPipeline {
            zslWanted = zslEnabled
            if #available(iOS 17.0, *) {
                photoOutput.isZeroShutterLagEnabled = false
            }
            if !zslEnabled {
                clearZSLRing()
                DispatchQueue.main.async { self.zslBufferReady = 0 }
            }
            return
        }
        if #available(iOS 17.0, *) {
            photoOutput.isZeroShutterLagEnabled = zslEnabled
        }
        if zslEnabled {
            zslWanted = true
            ensureZSLDir()
            pumpZSL()
        } else {
            zslWanted = false
            clearZSLRing()
            DispatchQueue.main.async { self.zslBufferReady = 0 }
        }
    }

    /// Stop continuous ZSL capture (app ring + system flag) for the whole process window.
    private func pauseZSLForProcessing() {
        zslPausedForPipeline = true
        zslCapturing = false
        if captureKind == .zsl { captureKind = .none }
        if #available(iOS 17.0, *) {
            photoOutput.isZeroShutterLagEnabled = false
        }
    }

    /// Re-enable ZSL only after capture + merge (+ Photos save) fully finish.
    private func resumeZSLAfterProcessing() {
        zslPausedForPipeline = false
        pipelineBusy = false
        captureKind = .none
        guard zslEnabled else {
            zslWanted = false
            return
        }
        zslWanted = true
        if #available(iOS 17.0, *) {
            photoOutput.isZeroShutterLagEnabled = true
        }
        ensureZSLDir()
        scheduleNextZSL()
    }

    private func ensureZSLDir() {
        if let dir = zslDir, FileManager.default.fileExists(atPath: dir.path) { return }
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("zsl_\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        zslDir = dir
    }

    private func clearZSLRing() {
        let urls = zslRing
        let rawPaths = zslRawRing.compactMap { $0["path"] as? String }
        zslRawRing.removeAll(keepingCapacity: true)
        zslRing.removeAll(keepingCapacity: true)
        zslCapturing = false
        captureKind = .none
        let dir = zslDir
        zslDir = nil
        processingQueue.async {
            for u in urls { try? FileManager.default.removeItem(at: u) }
            for p in rawPaths { try? FileManager.default.removeItem(atPath: p) }
            if let dir { try? FileManager.default.removeItem(at: dir) }
        }
    }

    /// Keep at most `activeFrameCount` Apple RAW DNGs on disk; oldest dropped.
    private func pumpZSL() {
        guard zslWanted, isAppActive, !previewSuspended,
              !pipelineBusy, !zslPausedForPipeline, !zslCapturing else { return }
        guard cachedRawPixelFormat != nil
                || photoOutput.availableRawPhotoPixelFormatTypes.first != nil else { return }
        ensureZSLDir()
        guard zslDir != nil else { return }
        zslCapturing = true
        captureKind = .zsl
        if !captureNextRaw(isZSL: true) {
            // captureNextRaw cleared flags; retry ASAP.
            scheduleNextZSL()
        }
    }

    /// Steady-state ZSL capture pacing. Unpaced, the ring recaptured the
    /// moment the sensor was free: full 12MP RAW readouts back to back, ~24MB
    /// written and deleted per frame, sensor + ISP + NAND at 100% duty cycle
    /// for as long as the toggle was on — which is what heated the SoC until
    /// iOS throttled the whole device (the reported lag). At 5 captures/s the
    /// newest ring frame is at most ~0.2s old (well inside what zero shutter
    /// lag needs; the sub-pixel diversity the merge wants actually improves
    /// with a gap) while capture, ISP, memory, and disk load drop ~5-10x.
    /// The initial fill after enabling stays unpaced so the buffer reads
    /// ready quickly.
    private static let zslPaceInterval: TimeInterval = 0.20

    private func scheduleNextZSL() {
        sessionQueue.async { [weak self] in
            guard let self, self.zslWanted, !self.pipelineBusy, !self.zslPausedForPipeline else { return }
            let filled = max(self.zslRawRing.count, self.zslRing.count) >= self.activeFrameCount
            if filled {
                self.sessionQueue.asyncAfter(deadline: .now() + Self.zslPaceInterval) { [weak self] in
                    guard let self, self.zslWanted, !self.pipelineBusy, !self.zslPausedForPipeline else { return }
                    self.pumpZSL()
                }
            } else {
                self.pumpZSL()
            }
        }
    }

    private func ensureReadyBurstDir() {
        if let dir = readyBurstDir,
           FileManager.default.fileExists(atPath: dir.path) {
            return
        }
        readyBurstDir = nil
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("burst_\(UUID().uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            readyBurstDir = dir
        } catch {
            readyBurstDir = nil
        }
    }

    func setCamera(_ selection: CameraSelection) {
        guard !isBusy, availableCameras.contains(selection) else { return }
        sessionQueue.async {
            guard selection != self.activeCameraSelection else { return }
            if selection != .front { self.lastBackSelection = selection }
            self.activeCameraSelection = selection
            DispatchQueue.main.async {
                self.cameraSelection = selection
                self.syncLensModeForCameraSelection(selection)
            }
            self.switchCameraDevice(to: selection)
        }
    }

    func setLensZoom(_ mode: LensZoomMode) {
        guard !isBusy else { return }
        let target = lensCameraSelection(for: mode)
        guard availableCameras.contains(target) else { return }
        lensZoomMode = mode
        setZoom(virtualZoom(mode), animated: true)
    }

    func toggleFrontCamera() {
        guard !isBusy else { return }
        sessionQueue.async {
            if self.activeCameraSelection == .front {
                self.setCameraOnSessionQueue(self.lastBackSelection)
            } else {
                self.lastBackSelection = self.activeCameraSelection
                self.setCameraOnSessionQueue(.front)
            }
        }
    }

    private func setCameraOnSessionQueue(_ selection: CameraSelection) {
        guard availableCameras.contains(selection) else { return }
        guard selection != activeCameraSelection else { return }
        if selection != .front { lastBackSelection = selection }
        activeCameraSelection = selection
        DispatchQueue.main.async {
            self.cameraSelection = selection
            self.syncLensModeForCameraSelection(selection)
        }
        switchCameraDevice(to: selection)
    }

    private func syncLensModeForCameraSelection(_ selection: CameraSelection) {
        if selection == .front {
            zoomFactor = 1
        } else if lensFor(zoomFactor) != selection {
            // A switch this did not drive -- the front-camera toggle, or
            // discovery picking a different default -- so adopt the new lens's
            // own framing instead of keeping a zoom it cannot serve.
            zoomFactor = nativeZoom(selection)
        }
        syncLensModeForZoom()
        publishCropZoom()
        settlePreviewZoom(for: selection)
    }

    /// Reconcile the preview scale with the lens that just went live.
    ///
    /// Runs on the main thread from syncLensModeForCameraSelection, i.e. after
    /// the session has actually swapped inputs.
    private func settlePreviewZoom(for selection: CameraSelection) {
        if selection == .front {
            virtualZoomBeforeSwitch = nil
            previewZoom = 1
            return
        }
        let native = nativeZoom(selection)
        let target = max(1, zoomFactor / native)

        guard let before = virtualZoomBeforeSwitch else {
            // Zoom-in ramp already reached this framing, or a switch we did not
            // initiate: adopt the scale without animating.
            previewZoom = target
            return
        }
        virtualZoomBeforeSwitch = nil
        // Start where the old lens left off so the framing is continuous across
        // the swap, then animate outward. Clamped at 1 because the sensor cannot
        // supply a wider field than the lens itself sees.
        previewZoom = max(1, before / native)
        // Commit the snap before the animated change, or SwiftUI coalesces both
        // into one update and the animation has nothing to travel.
        DispatchQueue.main.async {
            withAnimation(.easeInOut(duration: Self.lensZoomDuration)) {
                self.previewZoom = target
            }
        }
    }

    /// Apply a deferred lens switch immediately.
    ///
    /// Both this and setCamera's work land on sessionQueue, and captureBurst
    /// enqueues its own block afterwards, so the device is swapped before any
    /// frame is requested.
    private func flushPendingLensSwitch() {
        guard let pending = pendingLensSwitch else { return }
        pendingLensSwitch = nil
        setCamera(pending.selection)
    }

    private func discoverCamerasAfterSetup() {
        var found: [CameraSelection] = []
        if device(for: .wide) != nil { found.append(.wide) }
        if device(for: .ultraWide) != nil { found.append(.ultraWide) }
        if device(for: .telephoto) != nil { found.append(.telephoto) }
        if device(for: .front) != nil { found.append(.front) }
        if found.isEmpty { found = [.wide] }
        let teleLabel = inferredTelephotoLensLabel()
        let teleZoom = measuredZoomRelativeToWide(.telephoto) ?? 2
        let uwZoom = measuredZoomRelativeToWide(.ultraWide) ?? 0.5
        DispatchQueue.main.async {
            self.availableCameras = found
            self.telephotoLensLabel = teleLabel
            self.telephotoNativeZoom = teleZoom
            self.ultraWideNativeZoom = uwZoom
            if !found.contains(self.cameraSelection) {
                self.cameraSelection = found.first ?? .wide
                self.syncLensModeForCameraSelection(self.cameraSelection)
            }
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        session.automaticallyConfiguresApplicationAudioSession = false
        // .photo is required for Bayer RAW capture on iOS.
        session.sessionPreset = .photo

        if let input = videoInput {
            session.removeInput(input)
            videoInput = nil
        }

        let selection = activeCameraSelection

        guard let dev = device(for: selection),
              let input = try? AVCaptureDeviceInput(device: dev),
              session.canAddInput(input) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.statusText = "No camera available" }
            return
        }
        session.addInput(input)
        videoInput = input
        device = dev

        if !session.outputs.contains(photoOutput) {
            guard session.canAddOutput(photoOutput) else {
                session.commitConfiguration()
                return
            }
            session.addOutput(photoOutput)
        }
        applyFastCapturePipelineSettings()
        configureRawCaptureLimits()
        refreshExposureRange()
        applyDefaultDeviceModes()

        session.commitConfiguration()
        startPreviewIfNeeded()
        ensureReadyBurstDir()
        DispatchQueue.main.async {
            self.isSessionRunning = self.session.isRunning
            self.applyShutter()
            if self.exposureIsAuto {
                self.startAutoExposureSyncIfNeeded()
            } else {
                self.exposureSyncTimer?.invalidate()
                self.exposureSyncTimer = nil
            }
            if self.cachedRawPixelFormat == nil {
                self.statusText = "RAW capture not supported on this camera"
            }
        }
    }

    private func switchCameraDevice(to selection: CameraSelection) {
        session.beginConfiguration()

        if let input = videoInput {
            session.removeInput(input)
            videoInput = nil
        }

        guard let dev = device(for: selection),
              let input = try? AVCaptureDeviceInput(device: dev),
              session.canAddInput(input) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.statusText = "Camera unavailable" }
            return
        }
        session.addInput(input)
        videoInput = input
        device = dev

        applyFastCapturePipelineSettings()
        configureRawCaptureLimits()
        refreshExposureRange()
        applyDefaultDeviceModes()

        session.commitConfiguration()
        ensureReadyBurstDir()
        DispatchQueue.main.async {
            self.applyShutter()
            if self.exposureIsAuto {
                self.startAutoExposureSyncIfNeeded()
            } else {
                self.exposureSyncTimer?.invalidate()
                self.exposureSyncTimer = nil
            }
            if self.cachedRawPixelFormat == nil {
                self.statusText = "RAW not supported on \(selection.label)"
            } else {
                self.statusText = ""
            }
        }
    }

    private func refreshExposureRange() {
        guard let d = device else { return }
        let minS = max(CMTimeGetSeconds(d.activeFormat.minExposureDuration), 1e-6)
        let maxS = max(CMTimeGetSeconds(d.activeFormat.maxExposureDuration), minS * 1.01)
        DispatchQueue.main.async {
            self.exposureMinSec = minS
            self.exposureMaxSec = maxS
            self.isoMin = d.activeFormat.minISO
            self.isoMax = d.activeFormat.maxISO
        }
    }

    private func durationFromSlider(_ t: Double, minSec: Double? = nil, maxSec: Double? = nil) -> Double {
        let clamped = min(1.0, max(0.0, t))
        let logMin = log(minSec ?? exposureMinSec)
        let logMax = log(maxSec ?? exposureMaxSec)
        return exp(logMin + clamped * (logMax - logMin))
    }

    private func sliderFromDuration(_ sec: Double) -> Double {
        let clamped = min(exposureMaxSec, max(exposureMinSec, sec))
        let logMin = log(exposureMinSec)
        let logMax = log(exposureMaxSec)
        guard logMax > logMin else { return 0.5 }
        return min(1.0, max(0.0, (log(clamped) - logMin) / (logMax - logMin)))
    }

    /// Everything applyShutterOnSessionQueue needs, snapshotted where the
    /// properties live. It used to read isoIsAuto and isoValue straight off the
    /// session queue while the UI was writing them from main.
    private struct ExposureRequest {
        var isAuto: Bool
        var slider: Double
        var minSec: Double
        var maxSec: Double
        var isoAuto: Bool
        var isoValue: Float
    }

    private func currentExposureRequest() -> ExposureRequest {
        ExposureRequest(isAuto: shutterIsAuto,
                        slider: shutterSlider,
                        minSec: exposureMinSec,
                        maxSec: exposureMaxSec,
                        isoAuto: isoIsAuto,
                        isoValue: isoValue)
    }

    /// Latest requested exposure, and whether a session-queue block is already
    /// on its way to consume it.
    private let exposureLock = NSLock()
    private var pendingExposure: ExposureRequest?
    private var exposureApplyQueued = false

    /// A drag produces a value change per touch sample, and each one lands here.
    /// Enqueuing a separate setExposureModeCustom for every sample backs the
    /// session queue up behind settings that are already stale, which shows up
    /// as the control lagging the finger and settling somewhere it was not
    /// released. Only the newest request survives to reach the device.
    private func applyShutter() {
        let req = currentExposureRequest()
        exposureLock.lock()
        pendingExposure = req
        let alreadyQueued = exposureApplyQueued
        exposureApplyQueued = true
        exposureLock.unlock()
        guard !alreadyQueued else { return }

        sessionQueue.async {
            self.exposureLock.lock()
            let latest = self.pendingExposure
            self.pendingExposure = nil
            self.exposureApplyQueued = false
            self.exposureLock.unlock()
            if let latest = latest { self.applyShutterOnSessionQueue(latest) }
        }
    }

    /// `alreadyLocked` for the two burst-teardown callers, which hold the
    /// configuration lock across a group of changes. They used to call in
    /// unconditionally, so the device was locked once and unlocked twice.
    private func applyShutterOnSessionQueue(_ req: ExposureRequest,
                                            alreadyLocked: Bool = false) {
        guard let d = device else { return }
        if !alreadyLocked, (try? d.lockForConfiguration()) == nil { return }
        defer { if !alreadyLocked { d.unlockForConfiguration() } }

        if req.isAuto && req.isoAuto {
            if d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
            }
        } else if d.isExposureModeSupported(.custom) {
            let minD = d.activeFormat.minExposureDuration
            let maxD = d.activeFormat.maxExposureDuration
            var t = CMTimeMakeWithSeconds(
                durationFromSlider(req.slider, minSec: req.minSec, maxSec: req.maxSec),
                preferredTimescale: 1_000_000_000)
            if CMTimeCompare(t, minD) < 0 { t = minD }
            if CMTimeCompare(t, maxD) > 0 { t = maxD }
            let iso = min(max(d.activeFormat.minISO, req.isoValue), d.activeFormat.maxISO)
            d.setExposureModeCustom(duration: t, iso: iso, completionHandler: nil)
        } else {
            DispatchQueue.main.async {
                self.statusText = "Manual exposure not supported on this camera"
            }
        }
    }

    private func device(for selection: CameraSelection) -> AVCaptureDevice? {
        switch selection {
        case .wide:
            return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        case .ultraWide:
            return AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back)
        case .telephoto:
            return AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: .back)
        case .front:
            return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
        }
    }

    /// Framing of `selection` in 1x-wide units, from the field of view the
    /// device reports. nil when either lens is missing or the numbers are not
    /// usable, in which case the caller keeps its default.
    private func measuredZoomRelativeToWide(_ selection: CameraSelection) -> CGFloat? {
        guard let wide = device(for: .wide),
              let other = device(for: selection) else { return nil }
        let wideFOV = Double(wide.activeFormat.videoFieldOfView) * .pi / 180.0
        let otherFOV = Double(other.activeFormat.videoFieldOfView) * .pi / 180.0
        guard wideFOV > 0, otherFOV > 0 else { return nil }
        let zoom = tan(wideFOV * 0.5) / tan(otherFOV * 0.5)
        guard zoom.isFinite, zoom > 0.05, zoom < 100 else { return nil }
        return CGFloat(zoom)
    }

    private func inferredTelephotoLensLabel() -> String {
        guard let wide = device(for: .wide),
              let tele = device(for: .telephoto) else { return "Tele" }
        let wideFOV = Double(wide.activeFormat.videoFieldOfView) * .pi / 180.0
        let teleFOV = Double(tele.activeFormat.videoFieldOfView) * .pi / 180.0
        guard wideFOV > 0, teleFOV > 0, teleFOV < wideFOV else { return "Tele" }
        let zoom = tan(wideFOV * 0.5) / tan(teleFOV * 0.5)
        guard zoom.isFinite, zoom >= 2.0 else { return "Tele" }
        return "\(Int(zoom.rounded()))×"
    }

    private func applyDefaultDeviceModes() {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
        if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            d.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        // Persisted Auto path: continuous AE only when the UI state says Auto.
        if exposureIsAuto, d.isExposureModeSupported(.continuousAutoExposure) {
            d.exposureMode = .continuousAutoExposure
        }
        d.isSubjectAreaChangeMonitoringEnabled = false
        d.unlockForConfiguration()
        applyShutter()
    }

    private func startAutoExposureSyncIfNeeded() {
        exposureSyncTimer?.invalidate()
        exposureSyncTimer = nil
        guard exposureIsAuto else { return }
        guard isAppActive, !previewSuspended else { return }
        exposureSyncTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.pollAutoExposureForSlider()
        }
    }

    /// Mirrors the metering into both sliders while it owns exposure, so the
    /// readouts show what the camera is actually doing and taking over hands
    /// control across at the exposure already on screen instead of jumping to
    /// wherever the sliders were last left.
    private func pollAutoExposureForSlider() {
        guard exposureIsAuto, !isBusy else { return }
        sessionQueue.async {
            guard let d = self.device else { return }
            let sec = CMTimeGetSeconds(d.exposureDuration)
            let iso = d.iso
            DispatchQueue.main.async {
                // Re-checked on the main thread: the hop off and back gives the
                // user time to have started a drag, and writing the metered
                // value then would yank the control out from under them.
                guard self.exposureIsAuto else { return }
                if sec.isFinite, sec > 0 { self.shutterSlider = self.sliderFromDuration(sec) }
                if iso.isFinite, iso > 0, self.isoMax > self.isoMin {
                    self.meteredIso = iso
                    self.isoSlider = Double(min(1, max(0, (iso - self.isoMin)
                                                        / (self.isoMax - self.isoMin))))
                }
            }
        }
    }

    private func startPreviewIfNeeded() {
        guard isAppActive, !previewSuspended, !session.isRunning else { return }
        session.startRunning()
        DispatchQueue.main.async {
            self.isSessionRunning = self.session.isRunning
            self.startAutoExposureSyncIfNeeded()
        }
    }

    // MARK: - Focus / exposure controls

    func focus(at devicePoint: CGPoint) {
        guard !isBusy else { return }
        // A point outside the sensor rectangle is silently ignored by
        // AVFoundation, so clamp rather than letting an edge tap do nothing.
        let p = CGPoint(x: min(1, max(0, devicePoint.x)),
                        y: min(1, max(0, devicePoint.y)))

        // Manual exposure deliberately survives a tap now. It used to be reset
        // to Auto here on the reading that a tap means "work it out for me",
        // but the preview fills the screen and doubles as the focus target, so
        // any stray tap silently discarded a manually set exposure -- and from
        // the outside that is indistinguishable from the controls not working.
        // A tap sets focus, and exposure point when the metering owns it.
        let meteringOwnsExposure = exposureIsAuto

        sessionQueue.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }

            if d.isFocusPointOfInterestSupported {
                d.focusPointOfInterest = p
                // One-shot at the tap, with subject-area monitoring below to
                // hand focus back to continuous once the scene moves on.
                if d.isFocusModeSupported(.autoFocus) {
                    d.focusMode = .autoFocus
                } else if d.isFocusModeSupported(.continuousAutoFocus) {
                    d.focusMode = .continuousAutoFocus
                }
            }

            if d.isExposurePointOfInterestSupported {
                d.exposurePointOfInterest = p
            }
            // .continuousAutoExposure, NOT .autoExpose. autoExpose is one-shot:
            // it meters once and leaves the device in .locked, so a single tap
            // froze exposure for the rest of the session -- the camera stopped
            // responding to light entirely. The point of interest persists under
            // continuous metering anyway, and the metering stays weighted to it,
            // so nothing is lost by not locking.
            //
            // Skipped entirely under manual exposure: the point of interest is
            // still worth setting for when metering resumes, but switching the
            // mode here would undo the custom duration and ISO.
            if meteringOwnsExposure, d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
            }

            d.isSubjectAreaChangeMonitoringEnabled = true
            d.unlockForConfiguration()
        }

        if meteringOwnsExposure { startAutoExposureSyncIfNeeded() }
    }

    /// The scene changed enough that the tapped subject is probably gone.
    ///
    /// Without this observer, isSubjectAreaChangeMonitoringEnabled does nothing
    /// at all -- it only asks AVFoundation to post the notification. Focus would
    /// stay fixed wherever it was last tapped.
    @objc private func subjectAreaDidChange(_ note: Notification) {
        guard !isBusy else { return }
        guard let changed = note.object as? AVCaptureDevice else { return }
        sessionQueue.async {
            guard let d = self.device, d === changed,
                  (try? d.lockForConfiguration()) != nil else { return }
            let centre = CGPoint(x: 0.5, y: 0.5)
            if d.isFocusPointOfInterestSupported { d.focusPointOfInterest = centre }
            if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
            if d.isExposurePointOfInterestSupported { d.exposurePointOfInterest = centre }
            // Only reclaim exposure if the user has not since gone manual.
            if self.exposureIsAuto,
               d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
            }
            d.isSubjectAreaChangeMonitoringEnabled = false
            d.unlockForConfiguration()
        }
    }

    // MARK: - Preview zoom

    /// Scale applied to the preview layer, in units of the active device's own
    /// framing. 1 means "show what this lens sees".
    ///
    /// 2x is produced by centre-cropping the RAW during processing, not by the
    /// device, so without this the preview showed the full wide frame while the
    /// output was a 2x crop -- the framing on screen was not the framing saved.
    /// Deliberately not videoZoomFactor: that would risk the device cropping the
    /// RAW too and compounding with the crop the pipeline already applies.
    ///
    /// Animated changes go through withAnimation in the mutating methods; plain
    /// assignments snap. The view attaches no .animation modifier, so which
    /// transitions animate is decided here rather than by SwiftUI heuristics.
    @Published private(set) var previewZoom: CGFloat = 1

    /// Framing shown just before a pending device switch, for the zoom-out case
    /// where the new lens must start at the old framing and animate outward.
    private var virtualZoomBeforeSwitch: CGFloat?
    /// Lens whose switch is deferred until the zoom-in ramp finishes. The UI
    /// already shows this mode as selected, so a shutter press inside the ramp
    /// has to apply it first or the burst comes off the previous lens.
    private var pendingLensSwitch: (mode: LensZoomMode, selection: CameraSelection)?

    static let lensZoomDuration: Double = 0.28

    // MARK: - Continuous zoom

    /// Magnification in 1x-wide-lens units. The single source of truth for both
    /// the preview scale and the crop the pipeline applies, so the framing on
    /// screen is the framing that gets saved at every magnification, not just at
    /// the 1x and 2x stops.
    @Published private(set) var zoomFactor: CGFloat = 1
    /// The zoom slider is transient: a pinch summons it and it fades again.
    @Published private(set) var zoomUIVisible = false
    private var zoomHideWork: DispatchWorkItem?

    /// Cropping harder than this off the active lens leaves too little sensor
    /// for the merge to work with -- at 6x a 12 MP frame is a 0.34 MP crop, and
    /// even doubled that is under 1.4 MP out.
    static let maxCropZoom: CGFloat = 6

    var minZoom: CGFloat {
        availableCameras.contains(.ultraWide) ? ultraWideNativeZoom : 1
    }

    var maxZoom: CGFloat {
        let widest: CGFloat = availableCameras.contains(.telephoto) ? telephotoNativeZoom : 1
        return widest * Self.maxCropZoom
    }

    /// Whether the zoom is sitting on one of the lens buttons, within a
    /// tolerance, so the chip can highlight without needing an exact float.
    func isAtZoom(_ z: CGFloat) -> Bool { abs(zoomFactor - z) < 0.02 }

    /// Longest lens whose own field of view still contains this magnification.
    ///
    /// Always the largest that fits, so the crop is inward from a lens that
    /// really sees the framing rather than an upscale of a narrower one.
    private func lensFor(_ z: CGFloat) -> CameraSelection {
        if availableCameras.contains(.telephoto), z >= telephotoNativeZoom - 1e-4 {
            return .telephoto
        }
        if availableCameras.contains(.wide), z >= 1 - 1e-4 { return .wide }
        if availableCameras.contains(.ultraWide) { return .ultraWide }
        return .wide
    }

    /// Crop the pipeline must apply on the active lens to realise `zoomFactor`.
    /// 1 at each lens's native framing, rising to maxCropZoom before the next
    /// lens takes over.
    var cropZoom: CGFloat {
        guard cameraSelection != .front else { return 1 }
        return max(1, zoomFactor / nativeZoom(cameraSelection))
    }

    func setZoom(_ requested: CGFloat, animated: Bool = false) {
        guard !isBusy else { return }
        let z = min(maxZoom, max(minZoom, requested))
        let from = zoomFactor
        guard abs(z - from) > 1e-5 || lensFor(z) != cameraSelection else { return }
        let target = lensFor(z)
        zoomFactor = z
        syncLensModeForZoom()
        publishCropZoom()

        if target == cameraSelection {
            pendingLensSwitch = nil
            let scale = max(1, z / nativeZoom(target))
            if animated {
                withAnimation(.easeInOut(duration: Self.lensZoomDuration)) { previewZoom = scale }
            } else {
                previewZoom = scale
            }
            return
        }

        if !animated {
            // Pinching across a lens boundary. Keep tracking the fingers on the
            // lens that is still live -- previewZoom is in units of the current
            // lens, so the framing stays continuous -- and let settlePreviewZoom
            // re-base it when the new one arrives.
            previewZoom = max(1, z / nativeZoom(cameraSelection))
            pendingLensSwitch = nil
            virtualZoomBeforeSwitch = nil
            setCamera(target)
            return
        }

        if z > from {
            // Zooming in past this lens: ramp the current feed up to the new
            // framing first, then swap the device underneath at matching scale.
            withAnimation(.easeInOut(duration: Self.lensZoomDuration)) {
                previewZoom = max(1, z / nativeZoom(cameraSelection))
            }
            pendingLensSwitch = (lensZoomMode, target)
            DispatchQueue.main.asyncAfter(deadline: .now() + Self.lensZoomDuration) {
                guard let pending = self.pendingLensSwitch, pending.selection == target else { return }
                self.pendingLensSwitch = nil
                self.setCamera(target)
            }
        } else {
            virtualZoomBeforeSwitch = from
            pendingLensSwitch = nil
            setCamera(target)
        }
    }

    /// Reveal the zoom slider and restart its fade-out.
    func showZoomUI() {
        zoomHideWork?.cancel()
        if !zoomUIVisible {
            withAnimation(.easeOut(duration: 0.18)) { zoomUIVisible = true }
        }
        let work = DispatchWorkItem { [weak self] in
            withAnimation(.easeInOut(duration: 0.35)) { self?.zoomUIVisible = false }
        }
        zoomHideWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.2, execute: work)
    }

    private func syncLensModeForZoom() {
        if abs(zoomFactor - ultraWideNativeZoom) < 0.02 { lensZoomMode = .ultraWide }
        else if abs(zoomFactor - 1) < 0.02 { lensZoomMode = .wide1x }
        else if abs(zoomFactor - 2) < 0.02 { lensZoomMode = .wide2x }
        else if abs(zoomFactor - telephotoNativeZoom) < 0.02 { lensZoomMode = .telephoto }
    }

    /// Hand the crop to the session queue, which is where capture reads it.
    /// Reading the published value from that queue instead would be a race.
    private func publishCropZoom() {
        let c = cropZoom
        sessionQueue.async { self.activeCropZoom = c }
    }

    /// Native framing of each lens in 1x-wide units, measured from field of view
    /// at discovery. Defaults are the common iPhone values, replaced with real
    /// numbers once the devices are known.
    @Published private(set) var ultraWideNativeZoom: CGFloat = 0.5
    @Published private(set) var telephotoNativeZoom: CGFloat = 2

    private func nativeZoom(_ selection: CameraSelection) -> CGFloat {
        switch selection {
        case .ultraWide: return ultraWideNativeZoom
        case .telephoto: return telephotoNativeZoom
        case .wide, .front: return 1
        }
    }

    private func virtualZoom(_ mode: LensZoomMode) -> CGFloat {
        switch mode {
        case .ultraWide: return ultraWideNativeZoom
        case .wide1x: return 1
        case .wide2x: return 2
        case .telephoto: return telephotoNativeZoom
        }
    }

    private func lensCameraSelection(for mode: LensZoomMode) -> CameraSelection {
        switch mode {
        case .ultraWide: return .ultraWide
        case .wide1x, .wide2x: return .wide
        case .telephoto: return .telephoto
        }
    }

    // MARK: - Capture burst

    func captureBurst() {
        guard !isBusy else { return }
        flushPendingLensSwitch()
        playShutterSound()

        let total = frameCount
        let lens = zoomFactor < 1 ? String(format: "%.1fx", Double(zoomFactor))
                                  : String(format: "%gx", Double((zoomFactor * 10).rounded() / 10))
        isBusy = true
        isCapturing = true
        isProcessing = false
        progress = 0

        sessionQueue.async(qos: .userInteractive) {
            self.pipelineBusy = true
            self.pauseZSLForProcessing()
            if self.cachedRawPixelFormat == nil {
                self.cachedRawPixelFormat = self.photoOutput.availableRawPhotoPixelFormatTypes.first
            }
            guard self.cachedRawPixelFormat != nil else {
                self.resumeZSLAfterProcessing()
                DispatchQueue.main.async {
                    self.isBusy = false
                    self.isCapturing = false
                    self.statusText = "RAW capture not supported on this camera"
                }
                return
            }

            // ZSL path: process ring DNGs in place (no copyItem — that was the slow path).
            let zslRawReady = self.zslRawRing.count >= self.activeFrameCount
            let zslDNGReady = self.zslRing.count >= self.activeFrameCount
            if self.zslEnabled && (zslRawReady || zslDNGReady) {
                self.zslCapturing = false
                self.captureKind = .none
                let n = self.activeFrameCount
                let takeRaw = zslRawReady ? Array(self.zslRawRing.suffix(n)) : []
                let takeDNG = zslRawReady ? [] : Array(self.zslRing.suffix(n))
                if zslRawReady {
                    self.zslRawRing.removeLast(n)
                } else {
                    self.zslRing.removeLast(n)
                }
                let readyLeft = max(self.zslRawRing.count, self.zslRing.count)
                // Output-only folder (inputs stay in the ZSL ring dir until processBurst cleans up).
                self.ensureReadyBurstDir()
                guard let dir = self.readyBurstDir else {
                    // Put frames back so the buffer is not lost on folder failure.
                    if zslRawReady {
                        self.zslRawRing.append(contentsOf: takeRaw)
                    } else {
                        self.zslRing.append(contentsOf: takeDNG)
                    }
                    self.resumeZSLAfterProcessing()
                    DispatchQueue.main.async {
                        self.isBusy = false
                        self.isCapturing = false
                        self.zslBufferReady = max(self.zslRawRing.count, self.zslRing.count)
                        self.statusText = "Could not create capture folder"
                    }
                    return
                }
                self.burstDir = dir
                self.readyBurstDir = nil
                self.capturedRawFrames = takeRaw
                self.capturedDNGs = takeDNG
                DispatchQueue.main.async {
                    self.zslBufferReady = readyLeft
                    self.statusText = "ZSL \(n) frames · \(lens)"
                    self.progress = 0.12
                    self.isCapturing = false
                    self.isProcessing = true
                }
                self.processingQueue.async { self.processBurst() }
                self.ensureReadyBurstDir()
                return
            }

            DispatchQueue.main.async {
                self.statusText = "Capturing \(total) frames · \(lens)"
            }

            self.ensureReadyBurstDir()
            guard let dir = self.readyBurstDir else {
                self.resumeZSLAfterProcessing()
                DispatchQueue.main.async {
                    self.isBusy = false
                    self.isCapturing = false
                    self.statusText = "Could not create capture folder"
                }
                return
            }
            self.burstDir = dir
            self.readyBurstDir = nil
            self.capturedRawFrames.removeAll(keepingCapacity: true)
            self.capturedDNGs.removeAll(keepingCapacity: true)
            self.burstInputMode = .undecided
            self.currentBurstTotal = self.activeFrameCount
            self.capturesRequested = 0
            self.capturesProcessed = 0
            self.captureKind = .burst
            self.zslCapturing = false

            self.lockForBurst {
                self.captureNextRaw(isZSL: false)
            }
            self.ensureReadyBurstDir()
        }
    }

    private func abortBurst(_ message: String) {
        sessionQueue.async {
            let dir = self.burstDir
            self.burstDir = nil
            self.capturesRequested = self.currentBurstTotal
            self.capturesProcessed = self.currentBurstTotal
            self.capturedRawFrames.removeAll(keepingCapacity: true)
            self.capturedDNGs.removeAll(keepingCapacity: true)
            self.burstInputMode = .undecided
            self.captureKind = .none
            self.removeBurstDir(dir)
            self.ensureReadyBurstDir()
            self.resumeZSLAfterProcessing()
            if let d = self.device, (try? d.lockForConfiguration()) != nil {
                if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
                if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                    d.whiteBalanceMode = .continuousAutoWhiteBalance
                }
                self.applyShutterOnSessionQueue(self.currentExposureRequest(),
                                                alreadyLocked: true)
                d.unlockForConfiguration()
            }
            if self.isAppActive && !self.previewSuspended {
                self.startPreviewIfNeeded()
            }
            DispatchQueue.main.async {
                self.isBusy = false
                self.isCapturing = false
                self.isProcessing = false
                self.progress = 0
                self.statusText = message
            }
        }
    }

    private func lockForBurst(_ then: @escaping () -> Void) {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { then(); return }
        if d.isFocusModeSupported(.locked) { d.focusMode = .locked }
        if d.isWhiteBalanceModeSupported(.locked) { d.whiteBalanceMode = .locked }
        // Only worth locking when the metering owns exposure; under manual the
        // device is already on a fixed custom duration and ISO.
        if exposureIsAuto {
            if tuningParams.burst_fast_shutter {
                // Fast-shutter burst: half the metered duration, ISO raised by
                // the same ratio (clamped to the sensor limit) so overall
                // exposure holds. Waits for the custom exposure to actually
                // take before the first frame, or frame 1 would still carry
                // the metered duration.
                let f = d.activeFormat
                let dur = CMTimeGetSeconds(d.exposureDuration)
                let minDur = CMTimeGetSeconds(f.minExposureDuration)
                let newDur = max(minDur, dur * 0.5)
                let ratio = Float(dur / max(newDur, 1e-9))
                let newIso = min(f.maxISO, max(f.minISO, d.iso * ratio))
                d.setExposureModeCustom(
                    duration: CMTimeMakeWithSeconds(newDur, preferredTimescale: 1_000_000_000),
                    iso: newIso) { [weak self] _ in
                    self?.sessionQueue.async(execute: then)
                }
                d.unlockForConfiguration()
                return
            }
            if d.isExposureModeSupported(.locked) { d.exposureMode = .locked }
        }
        d.unlockForConfiguration()
        then()
    }

    private func unlockAfterBurst() {
        sessionQueue.async {
            // Leave speed/responsive pipeline warm for the next shutter.
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
            if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                d.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            self.applyShutterOnSessionQueue(self.currentExposureRequest(),
                                            alreadyLocked: true)
            d.unlockForConfiguration()
        }
    }

    @discardableResult
    private func captureNextRaw(isZSL: Bool = false) -> Bool {
        if !isZSL, capturesRequested >= currentBurstTotal { return false }
        guard let rawFormat = cachedRawPixelFormat
                ?? photoOutput.availableRawPhotoPixelFormatTypes.first else {
            if isZSL {
                zslCapturing = false
                captureKind = .none
            } else {
                abortBurst("RAW capture unavailable")
            }
            return false
        }
        cachedRawPixelFormat = rawFormat
        if !isZSL { capturesRequested += 1 }
        let settings = AVCapturePhotoSettings(rawPixelFormatType: rawFormat)
        settings.flashMode = .off
        settings.photoQualityPrioritization = .speed
        // RAW must stay unprocessed — forcing SIS on Bayer RAW aborts capturePhoto.
        settings.isAutoStillImageStabilizationEnabled = false
        applyRawCaptureLimits(to: settings)
        applyShutterSoundSuppression(to: settings)
        photoOutput.capturePhoto(with: settings, delegate: self)
        return true
    }

    private func configureRawCaptureLimits() {
        cachedRawPixelFormat = photoOutput.availableRawPhotoPixelFormatTypes.first
        if #available(iOS 16.0, *) {
            photoOutput.isHighResolutionCaptureEnabled = false
            cachedMaxPhotoDimensions = preferredRawDimensions()
            if let dims = cachedMaxPhotoDimensions {
                photoOutput.maxPhotoDimensions = dims
                // Compile the MPSGraph FFT plan now, seconds before any shutter
                // press. It takes ~1100ms at 12MP and is otherwise paid by the
                // reference frame of the first burst after launch. Off the
                // session queue so it never delays configuration.
                // userInitiated, not utility: the profile showed ref:grey(gpu)
                // at 343ms against 79ms for comparison frames, meaning the
                // shutter was still waiting on this compile. Utility can be
                // scheduled onto efficiency cores and preempted, which is
                // exactly wrong for something the first capture blocks on.
                DispatchQueue.global(qos: .userInitiated).async {
                    SRBridge.prewarmFFTWidth(Int(dims.width), height: Int(dims.height))
                }
            }
        } else {
            cachedMaxPhotoDimensions = nil
        }
    }

    private func applyRawCaptureLimits(to settings: AVCapturePhotoSettings) {
        if #available(iOS 16.0, *) {
            settings.isHighResolutionPhotoEnabled = false
            if let dims = cachedMaxPhotoDimensions ?? preferredRawDimensions() {
                settings.maxPhotoDimensions = dims
            }
        }
    }

    @available(iOS 16.0, *)
    private func preferredRawDimensions() -> CMVideoDimensions? {
        let preferred = CMVideoDimensions(width: 4032, height: 3024)
        let supported = device?.activeFormat.supportedMaxPhotoDimensions ?? []
        if supported.isEmpty { return nil }
        for d in supported where d.width == preferred.width && d.height == preferred.height {
            return preferred
        }
        let maxPixels: Int64 = 15_000_000
        var best: CMVideoDimensions?
        var bestPixels: Int64 = 0
        for d in supported {
            let px = Int64(d.width) * Int64(d.height)
            if px <= maxPixels && px >= bestPixels {
                best = d
                bestPixels = px
            }
        }
        return best
    }

    // MARK: - Processing

    /// Remove leftover burst folders (~500 MB each) from tmp after crashes or older builds.
    /// Call on sessionQueue. Preserves the pre-created ready folder and any in-flight burst.
    private func purgeStaleCaptureFiles() {
        let keep = Set([readyBurstDir?.path, burstDir?.path].compactMap { $0 })
        let fm = FileManager.default
        let roots = [fm.temporaryDirectory]
            + (fm.urls(for: .cachesDirectory, in: .userDomainMask).first.map { [$0] } ?? [])
        for root in roots {
            guard let entries = try? fm.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { continue }
            for url in entries {
                let name = url.lastPathComponent
                guard name.hasPrefix("burst_") || name.hasSuffix("_cache") else { continue }
                if keep.contains(url.path) { continue }
                try? fm.removeItem(at: url)
            }
        }
        ensureReadyBurstDir()
    }

    private func removeBurstDir(_ dir: URL?) {
        guard let dir else { return }
        let path = dir.path
        processingQueue.async {
            try? FileManager.default.removeItem(atPath: path)
        }
    }

    /// DNGs chosen through the import picker, processed instead of a capture.
    /// Capped at maxFrameCount, or maxImportFrameCountOnline when the merge
    /// architecture is forced Online (constant-memory path).
    @Published var importedDNGs: [URL] = []

    /// Process a set of DNGs the user picked, at the resolution selected in
    /// Export settings. This used to force 48MP (and silently overwrote the
    /// saved resolution preference doing it), which made the 12MP setting
    /// appear broken: switch to 12MP, import a burst, get a 48MP file.
    func processImportedDNGs(_ urls: [URL]) {
        guard !isBusy else { return }
        let dngs = urls.filter { $0.pathExtension.lowercased() == "dng" }
        guard dngs.count >= 2 else {
            finish(success: false, message: "Pick at least 2 DNG files")
            return
        }
        // Online merge (forced via Merge Architecture) keeps one frame in
        // flight regardless of burst length, so imports may exceed the
        // capture cap: memory stays flat in frame count. Banded/Auto keep 15
        // because banding holds every frame resident through the merge.
        let cap = tuningParams.merge_arch == 2
            ? Self.maxImportFrameCountOnline : Self.maxFrameCount
        importedDNGs = Array(dngs.prefix(cap))
        isBusy = true
        isCapturing = false
        isProcessing = true
        progress = 0
        statusText = "Processing \(importedDNGs.count) imported DNGs"
        processingQueue.async { [weak self] in self?.processBurst() }
    }

    private func processBurst() {
        let rawFrames = capturedRawFrames
        var paths = capturedDNGs.map { $0.path }
        var usingDocDNGs = false

        // DNGs the user picked explicitly. This used to scan Documents and
        // silently take over whenever it found two or more, which meant a
        // stray file changed what the shutter did.
        if importedDNGs.count >= 2 {
            paths = importedDNGs.map { $0.path }.sorted()
            usingDocDNGs = true
        }

        let burstDir = self.burstDir
        guard rawFrames.count >= 2 || paths.count >= 2 else {
            removeBurstDir(burstDir)
            self.burstDir = nil
            finish(success: false, message: "Not enough frames captured")
            return
        }
        // Documents debug DNGs: no centre-crop, to match the full-frame Python run.
        let cropZoomForCapture: Float = usingDocDNGs ? 1 : Float(max(1, activeCropZoom))
        // The output-resolution choice only means anything when nothing is
        // cropped away; any zoom is realised as crop-then-2x, which is what the
        // 2x lens button always did and is now what every magnification does.
        // Imported DNGs never crop (cropZoomForCapture forced to 1 above), so
        // the selection applies to them like any uncropped capture.
        let useSelectedOutputScale = cropZoomForCapture <= 1.0001
        let algorithmScale: Float = useSelectedOutputScale
            ? outputResolutionMode.algorithmScale
            : 2.0
        let outputName = algorithmScale <= 1.01 ? "handheld_sr_x1.dng" : "handheld_sr_x2.dng"
        let outURL = (burstDir ?? FileManager.default.temporaryDirectory)
            .appendingPathComponent(outputName)

        DispatchQueue.main.async {
            self.isCapturing = false
            self.isProcessing = true
            self.statusText = "Processing…"
            self.noiseDiagText = ""
            self.progress = 0.15
        }

        let tuningDict: [String: NSNumber] = [
            "r_t": NSNumber(value: tuningParams.r_t),
            "r_s1": NSNumber(value: tuningParams.r_s1),
            "r_s2": NSNumber(value: tuningParams.r_s2),
            "r_Mt": NSNumber(value: tuningParams.r_Mt),
            "alignment_grey_fft": NSNumber(value: tuningParams.alignment_grey_fft),
            "flow_regularize_aperture_ratio": NSNumber(value: tuningParams.flow_regularize_aperture_ratio),
            "flow_reject_1d_ambiguity_ratio": NSNumber(value: tuningParams.flow_reject_1d_ambiguity_ratio),
            "k_detail": NSNumber(value: tuningParams.k_detail),
            "k_denoise": NSNumber(value: tuningParams.k_denoise),
            "k_stretch": NSNumber(value: tuningParams.k_stretch),
            "merge_fp16_accumulator": NSNumber(value: tuningParams.merge_fp16_accumulator),
            "merge_fast_weights": NSNumber(value: tuningParams.merge_fast_weights),
            "dng_store_unwhitened": NSNumber(value: tuningParams.dng_store_unwhitened),
            "bm_subpixel_quadratic": NSNumber(value: tuningParams.bm_subpixel_quadratic),
            "grey_decimate_lowpass": NSNumber(value: tuningParams.grey_decimate_lowpass),
            "align_fullres_polish": NSNumber(value: tuningParams.align_fullres_polish),
            "flow_boundary_selection": NSNumber(value: tuningParams.flow_boundary_selection),
            "flow_bicubic_sampling": NSNumber(value: tuningParams.flow_bicubic_sampling),
            "flow_overlap_merge": NSNumber(value: tuningParams.flow_overlap_merge),
            "k_shrink": NSNumber(value: tuningParams.k_shrink),
            "d_thresh_manual": NSNumber(value: tuningParams.d_thresh_manual),
            "dng_lossless_jpeg": NSNumber(value: tuningParams.dng_lossless_jpeg),
            "D_th": NSNumber(value: tuningParams.d_th),
            "D_tr": NSNumber(value: tuningParams.d_tr),
            "snr_auto_tune": NSNumber(value: tuningParams.snr_auto_tune),
            "alignment_tile_size": NSNumber(value: tuningParams.alignment_tile_size),
            "robustness_enabled": NSNumber(value: tuningParams.robustness_enabled),
            "robustness_save_mask": NSNumber(value: tuningParams.robustness_save_mask),
            "accumulated_robustness_denoiser_enabled": NSNumber(value: tuningParams.accumulated_robustness_denoiser_enabled),
            "merge_arch": NSNumber(value: tuningParams.merge_arch),
            "acc_rob_adaptive": NSNumber(value: tuningParams.acc_rob_adaptive),
            "isp_enabled": NSNumber(value: tuningParams.isp_enabled),
            "isp_exposure_ev": NSNumber(value: tuningParams.isp_exposure_ev),
            "isp_highlight_knee": NSNumber(value: tuningParams.isp_highlight_knee),
            "isp_local_strength": NSNumber(value: tuningParams.isp_local_strength),
            "isp_highlight": NSNumber(value: tuningParams.isp_highlight),
            "isp_shadow": NSNumber(value: tuningParams.isp_shadow),
            "isp_black_point": NSNumber(value: tuningParams.isp_black_point),
            "isp_warmth": NSNumber(value: tuningParams.isp_warmth),
            "isp_colour_strength": NSNumber(value: tuningParams.isp_colour_strength),
            "isp_contrast": NSNumber(value: tuningParams.isp_contrast),
            "isp_vibrance": NSNumber(value: tuningParams.isp_vibrance),
            "isp_chroma_denoise": NSNumber(value: tuningParams.isp_chroma_denoise),
            "isp_chroma_radius": NSNumber(value: tuningParams.isp_chroma_radius),
            "isp_saturation": NSNumber(value: tuningParams.isp_saturation),
            "isp_local_contrast": NSNumber(value: tuningParams.isp_local_contrast),
            "isp_skin_protect": NSNumber(value: tuningParams.isp_skin_protect),
            "align_ica_per_level": NSNumber(value: tuningParams.align_ica_per_level),
            "align_ica_per_level_fft": NSNumber(value: tuningParams.align_ica_per_level_fft),
            "align_ambiguous_fallback_enabled": NSNumber(value: tuningParams.align_ambiguous_fallback_enabled),
            "debug_noise_model_disabled": NSNumber(value: tuningParams.debug_noise_model_disabled),
            "flow_bilinear_sampling": NSNumber(value: tuningParams.flow_bilinear_sampling),
            "prealign_enabled": NSNumber(value: tuningParams.prealign_enabled),
            "merge_chroma_difference": NSNumber(value: tuningParams.merge_chroma_difference),
            "merge_kernel_iso": NSNumber(value: tuningParams.merge_kernel_iso),
            "merge_uncovered_passthrough": NSNumber(value: tuningParams.merge_uncovered_passthrough),
            "kernel_selection_hard": NSNumber(value: tuningParams.kernel_selection_hard),
            "merge_soften_inv_cov": NSNumber(value: tuningParams.merge_soften_inv_cov),
            "hdrplus_mode": NSNumber(value: tuningParams.hdrplus_mode),
            "isa_mode": NSNumber(value: tuningParams.isa_mode),
            "robustness_color_space": NSNumber(value: tuningParams.robustness_color_space),
            "flow_dense_lk_enabled": NSNumber(value: tuningParams.flow_dense_lk_enabled),
            "robustness_raw_resolution_enabled": NSNumber(value: tuningParams.robustness_raw_resolution_enabled),
            "acc_rob_max_frame_count": NSNumber(value: tuningParams.acc_rob_max_frame_count),
            "acc_rob_rad_max": NSNumber(value: tuningParams.acc_rob_rad_max),
            "acc_rob_max_multiplier": NSNumber(value: tuningParams.acc_rob_max_multiplier)
        ]

        var preview: UIImage?
        let inputURLs = capturedDNGs
        let rawInputPaths = rawFrames.compactMap { $0["path"] as? String }
        lastPipelineError = ""
        let progressBlock: (String, Float) -> Void = { [weak self] stage, frac in
            DispatchQueue.main.async {
                self?.progress = 0.15 + frac * 0.85
                if stage.hasPrefix("Noise ") {
                    self?.noiseDiagText = stage
                } else {
                    // finish() replaces statusText with its own message, so an
                    // error reported here would vanish before it could be read.
                    if stage.hasPrefix("Error") { self?.lastPipelineError = stage }
                    self?.statusText = stage
                }
            }
        }
        let ok: Bool
        if !usingDocDNGs && rawFrames.count >= 2 {
            ok = SRBridge.processRawFrames(
                rawFrames,
                toPath: outURL.path,
                scale: algorithmScale,
                cropZoom: cropZoomForCapture,
                tuningParams: tuningDict,
                progress: progressBlock,
                previewImage: &preview
            )
        } else {
            ok = SRBridge.processDNGs(
                paths,
                toPath: outURL.path,
                scale: algorithmScale,
                cropZoom: cropZoomForCapture,
                tuningParams: tuningDict,
                progress: progressBlock,
                previewImage: &preview
            )
        }

        capturedRawFrames.removeAll()
        capturedDNGs.removeAll()
        // Free ZSL ring inputs (and non-ZSL frame_*.dng) as soon as LibRaw is done.
        // Non-ZSL files also live under burstDir; removeBurstDir later is then cheaper.
        for u in inputURLs {
            try? FileManager.default.removeItem(at: u)
        }
        for p in rawInputPaths {
            try? FileManager.default.removeItem(atPath: p)
        }

        if ok {
            // Suffixes must match write_robustness_mask_pgm in pipeline.cpp.
            var maskSuffixes: [String] = []
            if tuningParams.robustness_save_mask {
                maskSuffixes.append("_robustness.pgm")
            }
            let base = outURL.deletingPathExtension().path
            let robURLs = maskSuffixes
                .map { URL(fileURLWithPath: base + $0) }
                .filter { FileManager.default.fileExists(atPath: $0.path) }
            saveToPhotos(url: outURL, robustnessMasks: robURLs, preview: preview, burstDir: burstDir)
        } else {
            removeBurstDir(burstDir)
            self.burstDir = nil
            finish(success: false,
                   message: lastPipelineError.isEmpty ? "Processing failed" : lastPipelineError)
        }
    }

    /// Load 8-bit binary PGM (P5) written by the C++ robustness export.
    private static func uiImageFromPGM(url: URL) -> UIImage? {
        guard let data = try? Data(contentsOf: url), data.count > 16 else { return nil }
        var i = 0
        func nextToken() -> String? {
            while i < data.count {
                let b = data[i]
                if b == 0x23 { // '#' comment
                    while i < data.count && data[i] != 0x0a { i += 1 }
                    continue
                }
                if b == 0x20 || b == 0x09 || b == 0x0a || b == 0x0d {
                    i += 1
                    continue
                }
                break
            }
            let start = i
            while i < data.count {
                let b = data[i]
                if b == 0x20 || b == 0x09 || b == 0x0a || b == 0x0d { break }
                i += 1
            }
            guard i > start else { return nil }
            return String(bytes: data[start..<i], encoding: .ascii)
        }
        guard nextToken() == "P5",
              let ws = nextToken(), let hs = nextToken(), let ms = nextToken(),
              let w = Int(ws), let h = Int(hs), let maxv = Int(ms), maxv == 255,
              w > 0, h > 0 else { return nil }
        // Single whitespace after maxval, then raw bytes
        while i < data.count && (data[i] == 0x20 || data[i] == 0x09 || data[i] == 0x0d) { i += 1 }
        if i < data.count && data[i] == 0x0a { i += 1 }
        let need = w * h
        guard i + need <= data.count else { return nil }
        var rgba = [UInt8](repeating: 255, count: need * 4)
        for p in 0..<need {
            let g = data[i + p]
            rgba[p * 4 + 0] = g
            rgba[p * 4 + 1] = g
            rgba[p * 4 + 2] = g
            rgba[p * 4 + 3] = 255
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let cg = ctx.makeImage() else { return nil }
        return UIImage(cgImage: cg)
    }

    /// Lightroom-like finish from the SR DNG: Highlights −70, stronger contrast
    /// + vibrance (no sharpen / NR). Uses our own Deflate LinearRaw decoder
    /// (ImageIO cannot read these DNGs).
    private static func renderExportJPEG(fromDNG dngURL: URL, quality: Double) -> URL? {
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("handheld_sr_\(UUID().uuidString).jpg")
        let ok = SRBridge.exportJPEG(fromLinearDNG: dngURL.path, toPath: outURL.path,
                                     quality: Float(quality))
        if ok { return outURL }
        try? FileManager.default.removeItem(at: outURL)
        return nil
    }

    private func saveToPhotos(url: URL, robustnessMasks: [URL], preview: UIImage?, burstDir: URL?) {
        let format = exportFormat
        let quality = jpegExportQuality
        // The JPEG render / preview embed below re-reads and re-encodes the
        // full-size DNG -- seconds of real work after the pipeline's "Done".
        // Without this the UI freezes on "Done" with no explanation.
        DispatchQueue.main.async { self.statusText = "Saving to Photos…" }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    self.lastThumbnail = preview
                    self.finish(success: false,
                                message: "Grant Photos access to save captures")
                }
                self.removeBurstDir(burstDir)
                self.burstDir = nil
                return
            }

            var saveURL = url
            var tempJPEG: URL?
            if format == .dng {
                // DNG-only asset: embed a tone-mapped JPEG SubIFD. Photos does
                // not merely thumbnail from it -- it cannot decode our Deflate
                // LinearRaw IFD0 at all, so this preview IS the image the user
                // sees at every zoom level. It therefore gets full resolution;
                // 4096 here was why a 48MP capture looked soft in Photos.
                // Same quality as the JPEG export, so the DNG's preview and a
                // JPEG of the same burst are byte-for-byte the same render.
                if !SRBridge.embedJPEGPreview(inDNG: url.path, maxSide: 32768,
                                              quality: Float(quality)) {
                    // This preview IS what Photos shows for these DNGs, so a
                    // silent failure here leaves an asset that renders as
                    // garbage with nothing explaining why.
                    DispatchQueue.main.async {
                        self.lastPipelineError = "Preview embed failed"
                    }
                }
            } else if format == .jpg {
                if let jpg = Self.renderExportJPEG(fromDNG: url, quality: quality) {
                    saveURL = jpg
                    tempJPEG = jpg
                } else {
                    DispatchQueue.main.async {
                        self.lastThumbnail = preview
                        self.finish(success: false, message: "JPG export failed")
                    }
                    self.removeBurstDir(burstDir)
                    self.burstDir = nil
                    return
                }
            }

            // Each mask becomes its own JPEG, named after the source PGM so the
            // three are distinguishable once they land in Photos.
            var maskJPEGs: [URL] = []
            for rob in robustnessMasks {
                guard FileManager.default.fileExists(atPath: rob.path),
                      let img = Self.uiImageFromPGM(url: rob),
                      let jpeg = img.jpegData(compressionQuality: 0.92) else { continue }
                let stem = rob.deletingPathExtension().lastPathComponent
                let tmp = FileManager.default.temporaryDirectory
                    .appendingPathComponent("\(stem)_\(UUID().uuidString).jpg")
                try? jpeg.write(to: tmp, options: .atomic)
                maskJPEGs.append(tmp)
            }
            let savedMask = !maskJPEGs.isEmpty
            let label = format == .jpg ? "JPG" : "DNG"
            PHPhotoLibrary.shared().performChanges({
                let req = PHAssetCreationRequest.forAsset()
                let opts = PHAssetResourceCreationOptions()
                // Move rather than copy. These sources are our own temporary
                // files -- the burst directory is deleted immediately after --
                // so copying meant writing the whole export a second time,
                // which at ~290MB for a DNG is most of the wait at the end.
                opts.shouldMoveFile = true
                if format == .dng {
                    opts.uniformTypeIdentifier = "com.adobe.raw-image"
                }
                req.addResource(with: .photo, fileURL: saveURL, options: opts)
                for maskJPEG in maskJPEGs {
                    // Its own options, one fresh request per mask. Reusing
                    // `opts` tagged the JPEG with uniformTypeIdentifier
                    // "com.adobe.raw-image" whenever the export format was DNG,
                    // so Photos was told a JPEG was a RAW file and rejected the
                    // resource -- which is why the mask never appeared even with
                    // the toggle on. Only the DNG export path was affected.
                    //
                    // A fresh PHAssetCreationRequest per mask matters too: a
                    // second addResource(with: .photo,...) on the same request
                    // is a second rendition of one asset, not a new asset.
                    let mopts = PHAssetResourceCreationOptions()
                    mopts.shouldMoveFile = false
                    let mreq = PHAssetCreationRequest.forAsset()
                    mreq.addResource(with: .photo, fileURL: maskJPEG, options: mopts)
                }
            }, completionHandler: { success, _ in
                for maskJPEG in maskJPEGs { try? FileManager.default.removeItem(at: maskJPEG) }
                if let tempJPEG { try? FileManager.default.removeItem(at: tempJPEG) }
                DispatchQueue.main.async {
                    self.lastThumbnail = preview
                    self.finish(success: success,
                                message: success
                                    ? (savedMask
                                       ? "Saved \(label) + \(maskJPEGs.count) robustness mask\(maskJPEGs.count == 1 ? "" : "s") to Photos"
                                       : "Saved super-res \(label) to Photos")
                                    : "Could not save to Photos")
                }
                self.removeBurstDir(burstDir)
                self.burstDir = nil
            })
        }
    }

    /// 1108 is the system camera-shutter sound. Using it rather than a bundled
    /// asset keeps the app consistent with the platform and respects the
    /// device's own shutter policy in regions that mandate it.
    private func playShutterSound() {
        guard shutterSoundEnabled else { return }
        AudioServicesPlaySystemSound(1108)
    }

    /// Last "Error…" line the C++ pipeline reported, kept because finish()
    /// overwrites statusText with its own summary.
    private var lastPipelineError = ""

    private func finish(success: Bool, message: String) {
        // The picker opened a security scope on each imported file; release it
        // now that the pipeline is done reading them.
        for url in importedDNGs { url.stopAccessingSecurityScopedResource() }
        importedDNGs = []
        DispatchQueue.main.async {
            self.isBusy = false
            self.isCapturing = false
            self.isProcessing = false
            self.progress = success ? 1 : 0
            self.statusText = message
            // Errors go next to the noise line, which is deliberately kept until
            // the next burst, so a failure is still readable after Done.
            if !success && !self.lastPipelineError.isEmpty {
                self.noiseDiagText = self.lastPipelineError
            }
            // Keep noiseDiagText until next burst so you can read OK/FALLBACK after Done.
        }
        sessionQueue.async {
            self.resumeZSLAfterProcessing()
        }
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraModel: AVCapturePhotoCaptureDelegate {
    private func appendDNGPhoto(_ photo: AVCapturePhoto, to dir: URL) -> Bool {
        guard let data = photo.fileDataRepresentation() else {
            abortBurst("Could not read RAW data")
            return false
        }
        let idx = capturedDNGs.count
        let url = dir.appendingPathComponent("frame_\(idx).dng")
        do {
            try data.write(to: url)
            capturedDNGs.append(url)
            return true
        } catch {
            abortBurst("Write error: \(error.localizedDescription)")
            return false
        }
    }

    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        if captureKind == .zsl {
            handleZSLPhoto(photo, error: error)
            return
        }

        if let error = error {
            abortBurst("Capture error: \(error.localizedDescription)")
            return
        }

        var storedFrame = false
        if photo.isRawPhoto, let dir = burstDir {
            autoreleasepool {
                switch burstInputMode {
                case .undecided:
                    let rawURL = dir.appendingPathComponent("frame_\(capturedRawFrames.count).raw16")
                    if let rawFrame = rawFrameDictionary(from: photo, writingTo: rawURL) {
                        burstInputMode = .directRaw
                        capturedRawFrames.append(rawFrame)
                        storedFrame = true
                    } else {
                        try? FileManager.default.removeItem(at: rawURL)
                        burstInputMode = .dngFallback
                        storedFrame = appendDNGPhoto(photo, to: dir)
                    }
                case .directRaw:
                    let rawURL = dir.appendingPathComponent("frame_\(capturedRawFrames.count).raw16")
                    guard let rawFrame = rawFrameDictionary(from: photo, writingTo: rawURL) else {
                        try? FileManager.default.removeItem(at: rawURL)
                        abortBurst("RAW pixel buffer unavailable")
                        return
                    }
                    capturedRawFrames.append(rawFrame)
                    storedFrame = true
                case .dngFallback:
                    storedFrame = appendDNGPhoto(photo, to: dir)
                }
            }
            if !storedFrame { return }
        }

        capturesProcessed += 1
        DispatchQueue.main.async {
            self.progress = Float(self.capturesProcessed) / Float(self.currentBurstTotal) * 0.15
        }

        if capturesProcessed == currentBurstTotal {
            unlockAfterBurst()
            captureKind = .none
            processingQueue.async { self.processBurst() }
        }
    }

    private func handleZSLPhoto(_ photo: AVCapturePhoto, error: Error?) {
        // Next capture is started from didFinishCaptureFor (sensor free) so we
        // do not wait on DNG encode/write before grabbing the following frame.
        if error != nil || !photo.isRawPhoto { return }
        let rawURL = zslDir?.appendingPathComponent("zsl_raw_\(UUID().uuidString).raw16")
        if let rawURL,
           let rawFrame = autoreleasepool(invoking: { rawFrameDictionary(from: photo, writingTo: rawURL) }) {
            sessionQueue.async {
                guard !self.pipelineBusy, !self.zslPausedForPipeline else {
                    if let path = rawFrame["path"] as? String {
                        try? FileManager.default.removeItem(atPath: path)
                    }
                    return
                }
                self.zslRawRing.append(rawFrame)
                let cap = max(self.activeFrameCount, 2)
                while self.zslRawRing.count > cap {
                    let old = self.zslRawRing.removeFirst()
                    if let path = old["path"] as? String {
                        try? FileManager.default.removeItem(atPath: path)
                    }
                }
                let n = max(self.zslRawRing.count, self.zslRing.count)
                DispatchQueue.main.async { self.zslBufferReady = n }
            }
            return
        }
        if let rawURL {
            try? FileManager.default.removeItem(at: rawURL)
        }

        // Fallback: pull DNG bytes off the photo callback queue, then mutate the ring.
        let data: Data? = autoreleasepool { photo.fileDataRepresentation() }
        guard let data else { return }
        sessionQueue.async {
            guard !self.pipelineBusy, !self.zslPausedForPipeline, let dir = self.zslDir else { return }
            self.zslSeq += 1
            let url = dir.appendingPathComponent("zsl_\(self.zslSeq).dng")
            do {
                try data.write(to: url)
                self.zslRing.append(url)
                let cap = max(self.activeFrameCount, 2)
                while self.zslRing.count > cap {
                    let old = self.zslRing.removeFirst()
                    try? FileManager.default.removeItem(at: old)
                }
                let n = max(self.zslRawRing.count, self.zslRing.count)
                DispatchQueue.main.async { self.zslBufferReady = n }
            } catch {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings,
                     error: Error?) {
        sessionQueue.async {
            if self.captureKind == .burst {
                if error == nil, self.capturesRequested < self.currentBurstTotal {
                    _ = self.captureNextRaw(isZSL: false)
                }
                return
            }
            // ZSL: re-arm immediately when the camera is free (before DNG write finishes).
            guard self.captureKind == .zsl || self.zslCapturing else { return }
            self.zslCapturing = false
            if self.captureKind == .zsl { self.captureKind = .none }
            guard self.zslWanted, !self.pipelineBusy, !self.zslPausedForPipeline else { return }
            if error != nil {
                self.scheduleNextZSL()
            } else {
                self.pumpZSL()
            }
        }
    }
}
