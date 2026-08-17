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
    var hf_artifact_removal_enabled: Bool = false
    var hf_variance_loss_threshold: Float = 0.75
    var hf_min_texture_snr: Float = 4.0
    var flow_reject_1d_enabled: Bool = false
    var flow_regularize_aperture_ratio: Float = 0.15
    var flow_reject_1d_ambiguity_ratio: Float = 1.10
    var flow_reject_1d_residual_threshold: Float = 2.5
    var motion_edge_rejection_enabled: Bool = true
    var motion_edge_threshold: Float = 0.025
    var motion_edge_residual_threshold: Float = 2.5
    var motion_edge_noise_floor_multiplier: Float = 1.0
    var motion_edge_neighborhood_radius: Int = 1
    var k_detail: Float = 0.17
    var k_denoise: Float = 0.0
    var k_stretch: Float = 4.0
    var k_shrink: Float = 2.0
    var snr_auto_tune: Bool = true
    var debug_pixel4a_noise_profile: Bool = false
    var alignment_tile_size: Int = 0
    var global_prealignment_enabled: Bool = true
    /// Off: keeps frame 0 as the merge base, which lets the pre-alignment run
    /// inside the analysis loop instead of as a separate decode pass.
    var global_prealignment_choose_reference: Bool = false
    var global_prealignment_rotation_range_deg: Float = 0.0
    var global_prealignment_rotation_step_deg: Float = 0.25
    var global_prealignment_max_shift: Int = 24
    /// Off merges every frame at full weight everywhere. Diagnostic: it shows
    /// what the alignment actually produced, with no mask hiding the errors.
    var robustness_enabled: Bool = true
    var robustness_save_mask: Bool = true
    /// Also write _robustness_s1.pgm and _robustness_s2.pgm, splitting the
    /// accumulated mask by which motion prior scored each pixel. Costs one extra
    /// full-resolution buffer per comparison frame while the mask is built.
    var robustness_save_s_masks: Bool = false
    var accumulated_robustness_denoiser_enabled: Bool = true
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
    /// Route alignment through the bundled PWCNet Core ML model instead of
    /// the classical block-matching pyramid, feeding the result into the
    /// same robustness/merge math either way. Falls back to the classical
    /// path per-frame if the model isn't bundled or fails to load.
    var use_neural_flow: Bool = false
    /// Computes d^2/sigma^2/R at RAW resolution (Dodgson-quadratic upscale +
    /// flow-warp of the guide-resolution local stats) instead of directly at
    /// guide resolution, which this port otherwise does. The statistics stay
    /// half-resolution either way -- what changes is where the ratio is
    /// evaluated and where the 5x5 local-min runs: at raw resolution the min
    /// spans 5x5 raw px as the IPOL article intends, instead of 5x5 guide px
    /// = an effective 10x10 raw px. Only takes effect when "Alignment Grey:
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
    var isp_contrast: Float = 0.55
    var isp_vibrance: Float = 0.50
    /// Chroma noise reduction. Luma is preserved exactly, so this cannot
    /// soften detail -- only fine colour variation.
    var isp_chroma_denoise: Float = 0.0
    var isp_chroma_radius: Float = 12.0
    var isp_saturation: Float = 1.0
    var isp_local_contrast: Float = 0.20
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
        case hf_artifact_removal_enabled, hf_variance_loss_threshold
        case hf_min_texture_snr
        case flow_reject_1d_enabled, flow_regularize_aperture_ratio
        case flow_reject_1d_ambiguity_ratio, flow_reject_1d_residual_threshold
        case motion_edge_rejection_enabled, motion_edge_threshold, motion_edge_residual_threshold
        case motion_edge_noise_floor_multiplier, motion_edge_neighborhood_radius
        case k_detail, k_denoise, k_stretch, k_shrink
        case snr_auto_tune, debug_pixel4a_noise_profile, alignment_tile_size
        case global_prealignment_enabled, global_prealignment_choose_reference
        case global_prealignment_rotation_range_deg, global_prealignment_rotation_step_deg
        case global_prealignment_max_shift
        case robustness_enabled, robustness_save_mask, robustness_save_s_masks
        case accumulated_robustness_denoiser_enabled
        case merge_arch
        case acc_rob_adaptive, acc_rob_max_frame_count, align_ica_per_level
        case align_ica_per_level_fft, use_neural_flow
        case robustness_raw_resolution_enabled
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
        hf_artifact_removal_enabled = try c.decodeIfPresent(Bool.self, forKey: .hf_artifact_removal_enabled) ?? hf_artifact_removal_enabled
        hf_variance_loss_threshold = try c.decodeIfPresent(Float.self, forKey: .hf_variance_loss_threshold) ?? hf_variance_loss_threshold
        hf_min_texture_snr = try c.decodeIfPresent(Float.self, forKey: .hf_min_texture_snr) ?? hf_min_texture_snr
        flow_reject_1d_enabled = try c.decodeIfPresent(Bool.self, forKey: .flow_reject_1d_enabled) ?? flow_reject_1d_enabled
        flow_regularize_aperture_ratio = try c.decodeIfPresent(Float.self, forKey: .flow_regularize_aperture_ratio) ?? flow_regularize_aperture_ratio
        flow_reject_1d_ambiguity_ratio = try c.decodeIfPresent(Float.self, forKey: .flow_reject_1d_ambiguity_ratio) ?? flow_reject_1d_ambiguity_ratio
        flow_reject_1d_residual_threshold = try c.decodeIfPresent(Float.self, forKey: .flow_reject_1d_residual_threshold) ?? flow_reject_1d_residual_threshold
        motion_edge_rejection_enabled = try c.decodeIfPresent(Bool.self, forKey: .motion_edge_rejection_enabled) ?? motion_edge_rejection_enabled
        motion_edge_threshold = try c.decodeIfPresent(Float.self, forKey: .motion_edge_threshold) ?? motion_edge_threshold
        motion_edge_residual_threshold = try c.decodeIfPresent(Float.self, forKey: .motion_edge_residual_threshold) ?? motion_edge_residual_threshold
        motion_edge_noise_floor_multiplier = try c.decodeIfPresent(Float.self, forKey: .motion_edge_noise_floor_multiplier) ?? motion_edge_noise_floor_multiplier
        motion_edge_neighborhood_radius = try c.decodeIfPresent(Int.self, forKey: .motion_edge_neighborhood_radius) ?? motion_edge_neighborhood_radius
        k_detail = try c.decodeIfPresent(Float.self, forKey: .k_detail) ?? k_detail
        k_denoise = try c.decodeIfPresent(Float.self, forKey: .k_denoise) ?? k_denoise
        k_stretch = try c.decodeIfPresent(Float.self, forKey: .k_stretch) ?? k_stretch
        k_shrink = try c.decodeIfPresent(Float.self, forKey: .k_shrink) ?? k_shrink
        snr_auto_tune = try c.decodeIfPresent(Bool.self, forKey: .snr_auto_tune) ?? snr_auto_tune
        debug_pixel4a_noise_profile = try c.decodeIfPresent(
            Bool.self, forKey: .debug_pixel4a_noise_profile) ?? debug_pixel4a_noise_profile
        alignment_tile_size = try c.decodeIfPresent(Int.self, forKey: .alignment_tile_size) ?? alignment_tile_size
        global_prealignment_enabled = try c.decodeIfPresent(Bool.self, forKey: .global_prealignment_enabled) ?? global_prealignment_enabled
        global_prealignment_choose_reference = try c.decodeIfPresent(Bool.self, forKey: .global_prealignment_choose_reference) ?? global_prealignment_choose_reference
        global_prealignment_rotation_range_deg = try c.decodeIfPresent(Float.self, forKey: .global_prealignment_rotation_range_deg) ?? global_prealignment_rotation_range_deg
        global_prealignment_rotation_step_deg = try c.decodeIfPresent(Float.self, forKey: .global_prealignment_rotation_step_deg) ?? global_prealignment_rotation_step_deg
        global_prealignment_max_shift = try c.decodeIfPresent(Int.self, forKey: .global_prealignment_max_shift) ?? global_prealignment_max_shift
        robustness_enabled = try c.decodeIfPresent(Bool.self, forKey: .robustness_enabled) ?? robustness_enabled
        robustness_save_mask = try c.decodeIfPresent(Bool.self, forKey: .robustness_save_mask) ?? robustness_save_mask
        robustness_save_s_masks = try c.decodeIfPresent(Bool.self, forKey: .robustness_save_s_masks) ?? robustness_save_s_masks
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
        use_neural_flow = try c.decodeIfPresent(Bool.self, forKey: .use_neural_flow) ?? use_neural_flow
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
    @Published var shutterSlider: Double = CameraModel.persistedShutterSlider()
    /// Manual ISO. Auto by default; when off, isoSlider maps linearly onto the
    /// active format's supported range, which varies per lens and per device.
    /// Manual ISO. Starts on auto every launch, for the same reason as the
    /// shutter above.
    @Published var isoIsAuto: Bool = true {
        didSet { applyShutter() }
    }
    @Published var isoSlider: Double = CameraModel.persistedIsoSlider() {
        didSet {
            UserDefaults.standard.set(isoSlider, forKey: "IsoSlider")
            if !isoIsAuto { applyShutter() }
        }
    }
    @Published var isoMin: Float = 30
    @Published var isoMax: Float = 3000

    private static func persistedIsoSlider() -> Double {
        guard UserDefaults.standard.object(forKey: "IsoSlider") != nil else { return 0.0 }
        return min(1.0, max(0.0, UserDefaults.standard.double(forKey: "IsoSlider")))
    }

    var isoLabel: String {
        if isoIsAuto { return "Auto" }
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

    var shutterLabel: String {
        if shutterIsAuto { return "Auto" }
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

    func toggleShutterAuto() {
        setShutterAuto(!shutterIsAuto)
    }

    func setShutterAuto(_ auto: Bool) {
        shutterIsAuto = auto
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
        if shutterIsAuto { startAutoExposureSyncIfNeeded() }
    }

    func applyManualShutterFromSlider() {
        guard !isBusy else { return }
        if shutterIsAuto { shutterIsAuto = false }
        persistShutterState()
        applyShutter()
        exposureSyncTimer?.invalidate()
        exposureSyncTimer = nil
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

    private func scheduleNextZSL() {
        guard zslWanted, !pipelineBusy, !zslPausedForPipeline else { return }
        // No artificial pacing — fire as soon as the session queue can run.
        sessionQueue.async { [weak self] in
            guard let self, self.zslWanted, !self.pipelineBusy, !self.zslPausedForPipeline else { return }
            self.pumpZSL()
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
            if self.shutterIsAuto {
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
            if self.shutterIsAuto {
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

    private func applyShutter() {
        let isAuto = shutterIsAuto
        let slider = shutterSlider
        let minSec = exposureMinSec
        let maxSec = exposureMaxSec
        sessionQueue.async {
            self.applyShutterOnSessionQueue(isAuto: isAuto, slider: slider, minSec: minSec, maxSec: maxSec)
        }
    }

    private func applyShutterOnSessionQueue(isAuto: Bool, slider: Double, minSec: Double, maxSec: Double) {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        if isAuto && isoIsAuto {
            if d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
            }
        } else if d.isExposureModeSupported(.custom) {
            let minD = d.activeFormat.minExposureDuration
            let maxD = d.activeFormat.maxExposureDuration
            var t = isAuto
                ? d.exposureDuration
                : CMTimeMakeWithSeconds(durationFromSlider(slider, minSec: minSec, maxSec: maxSec),
                                        preferredTimescale: 1_000_000_000)
            if CMTimeCompare(t, minD) < 0 { t = minD }
            if CMTimeCompare(t, maxD) > 0 { t = maxD }
            // Manual ISO when the user has taken it off auto; otherwise hold
            // whatever the metering had settled on, as before.
            let wanted = isoIsAuto ? d.iso : isoValue
            let iso = min(max(d.activeFormat.minISO, wanted), d.activeFormat.maxISO)
            d.setExposureModeCustom(duration: t, iso: iso, completionHandler: nil)
        } else {
            DispatchQueue.main.async {
                self.statusText = "Manual shutter not supported on this camera"
            }
        }
        d.unlockForConfiguration()
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
        if shutterIsAuto, d.isExposureModeSupported(.continuousAutoExposure) {
            d.exposureMode = .continuousAutoExposure
        }
        d.isSubjectAreaChangeMonitoringEnabled = false
        d.unlockForConfiguration()
        applyShutter()
    }

    private func startAutoExposureSyncIfNeeded() {
        exposureSyncTimer?.invalidate()
        guard shutterIsAuto else { return }
        guard isAppActive, !previewSuspended else { return }
        exposureSyncTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.pollAutoExposureForSlider()
        }
    }

    private func pollAutoExposureForSlider() {
        guard shutterIsAuto, !isBusy else { return }
        sessionQueue.async {
            guard let d = self.device else { return }
            let sec = CMTimeGetSeconds(d.exposureDuration)
            guard sec.isFinite, sec > 0 else { return }
            DispatchQueue.main.async {
                self.shutterSlider = self.sliderFromDuration(sec)
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

        // A tap says "work the exposure out for me, here", so it also takes the
        // sliders back to Auto. Setting isoIsAuto runs its didSet, which
        // re-applies the exposure mode; the block below then points it at the
        // tap.
        if !shutterIsAuto || !isoIsAuto {
            shutterIsAuto = true
            isoIsAuto = true
            persistShutterState()
        }

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
            if d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
            }

            d.isSubjectAreaChangeMonitoringEnabled = true
            d.unlockForConfiguration()
        }

        // Back on Auto, so the shutter readout has to start tracking again.
        startAutoExposureSyncIfNeeded()
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
            if self.shutterIsAuto, self.isoIsAuto,
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

            self.lockForBurst()
            self.captureNextRaw(isZSL: false)
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
                self.applyShutterOnSessionQueue(
                    isAuto: self.shutterIsAuto,
                    slider: self.shutterSlider,
                    minSec: self.exposureMinSec,
                    maxSec: self.exposureMaxSec
                )
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

    private func lockForBurst() {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        if d.isFocusModeSupported(.locked) { d.focusMode = .locked }
        if d.isWhiteBalanceModeSupported(.locked) { d.whiteBalanceMode = .locked }
        if shutterIsAuto, d.isExposureModeSupported(.locked) { d.exposureMode = .locked }
        d.unlockForConfiguration()
    }

    private func unlockAfterBurst() {
        sessionQueue.async {
            // Leave speed/responsive pipeline warm for the next shutter.
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
            if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                d.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            self.applyShutterOnSessionQueue(
                isAuto: self.shutterIsAuto,
                slider: self.shutterSlider,
                minSec: self.exposureMinSec,
                maxSec: self.exposureMaxSec
            )
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
    /// Capped at maxFrameCount; the pipeline holds every frame through the merge.
    @Published var importedDNGs: [URL] = []

    /// Process a set of DNGs the user picked. Always at 2x, since importing is
    /// a deliberate act and the extra resolution is the reason to do it.
    func processImportedDNGs(_ urls: [URL]) {
        guard !isBusy else { return }
        let dngs = urls.filter { $0.pathExtension.lowercased() == "dng" }
        guard dngs.count >= 2 else {
            finish(success: false, message: "Pick at least 2 DNG files")
            return
        }
        importedDNGs = Array(dngs.prefix(Self.maxFrameCount))
        outputResolutionMode = .super48mp
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
        let useSelectedOutputScale = !usingDocDNGs && cropZoomForCapture <= 1.0001
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
            "hf_artifact_removal_enabled": NSNumber(value: tuningParams.hf_artifact_removal_enabled),
            "hf_variance_loss_threshold": NSNumber(value: tuningParams.hf_variance_loss_threshold),
            "hf_min_texture_snr": NSNumber(value: tuningParams.hf_min_texture_snr),
            "flow_reject_1d_enabled": NSNumber(value: tuningParams.flow_reject_1d_enabled),
            "flow_regularize_aperture_ratio": NSNumber(value: tuningParams.flow_regularize_aperture_ratio),
            "flow_reject_1d_ambiguity_ratio": NSNumber(value: tuningParams.flow_reject_1d_ambiguity_ratio),
            "flow_reject_1d_residual_threshold": NSNumber(value: tuningParams.flow_reject_1d_residual_threshold),
            "motion_edge_rejection_enabled": NSNumber(value: tuningParams.motion_edge_rejection_enabled),
            "motion_edge_threshold": NSNumber(value: tuningParams.motion_edge_threshold),
            "motion_edge_residual_threshold": NSNumber(value: tuningParams.motion_edge_residual_threshold),
            "motion_edge_noise_floor_multiplier": NSNumber(value: tuningParams.motion_edge_noise_floor_multiplier),
            "motion_edge_neighborhood_radius": NSNumber(value: tuningParams.motion_edge_neighborhood_radius),
            "k_detail": NSNumber(value: tuningParams.k_detail),
            "k_denoise": NSNumber(value: tuningParams.k_denoise),
            "k_stretch": NSNumber(value: tuningParams.k_stretch),
            "k_shrink": NSNumber(value: tuningParams.k_shrink),
            "snr_auto_tune": NSNumber(value: tuningParams.snr_auto_tune),
            "debug_pixel4a_noise_profile": NSNumber(value: tuningParams.debug_pixel4a_noise_profile),
            "alignment_tile_size": NSNumber(value: tuningParams.alignment_tile_size),
            "global_prealignment_enabled": NSNumber(value: tuningParams.global_prealignment_enabled),
            "global_prealignment_choose_reference": NSNumber(value: tuningParams.global_prealignment_choose_reference),
            "global_prealignment_rotation_range_deg": NSNumber(value: tuningParams.global_prealignment_rotation_range_deg),
            "global_prealignment_rotation_step_deg": NSNumber(value: tuningParams.global_prealignment_rotation_step_deg),
            "global_prealignment_max_shift": NSNumber(value: tuningParams.global_prealignment_max_shift),
            "robustness_enabled": NSNumber(value: tuningParams.robustness_enabled),
            "robustness_save_mask": NSNumber(value: tuningParams.robustness_save_mask),
            "robustness_save_s_masks": NSNumber(value: tuningParams.robustness_save_s_masks),
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
            "use_neural_flow": NSNumber(value: tuningParams.use_neural_flow),
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
                if tuningParams.robustness_save_s_masks {
                    maskSuffixes.append("_robustness_s1.pgm")
                    maskSuffixes.append("_robustness_s2.pgm")
                }
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
    private static func renderExportJPEG(fromDNG dngURL: URL) -> URL? {
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("handheld_sr_\(UUID().uuidString).jpg")
        let ok = SRBridge.exportJPEG(fromLinearDNG: dngURL.path, toPath: outURL.path)
        if ok { return outURL }
        try? FileManager.default.removeItem(at: outURL)
        return nil
    }

    private func saveToPhotos(url: URL, robustnessMasks: [URL], preview: UIImage?, burstDir: URL?) {
        let format = exportFormat
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
                // DNG-only asset: embed tone-mapped JPEG SubIFD so Photos can
                // thumbnail (ImageIO cannot decode Deflate LinearRaw IFD0).
                _ = SRBridge.embedJPEGPreview(inDNG: url.path, maxSide: 4096)
            } else if format == .jpg {
                if let jpg = Self.renderExportJPEG(fromDNG: url) {
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
