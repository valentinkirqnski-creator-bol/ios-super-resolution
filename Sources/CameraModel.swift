import AVFoundation
import Photos
import UIKit
import Combine
import ImageIO

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

    var cropFactor: Int {
        switch self {
        case .wide2x: return 2
        default: return 1
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
    var hf_artifact_removal_enabled: Bool = true
    var hf_variance_loss_threshold: Float = 0.90
    var hf_variance_noise_multiplier: Float = 1.0
    var hf_noise_floor_multiplier: Float = 4.0
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
    var alignment_tile_size: Int = 0
    var global_prealignment_enabled: Bool = true
    var global_prealignment_choose_reference: Bool = true
    var global_prealignment_rotation_range_deg: Float = 0.0
    var global_prealignment_rotation_step_deg: Float = 0.25
    var global_prealignment_max_shift: Int = 24
    var robustness_save_mask: Bool = true
    var accumulated_robustness_denoiser_enabled: Bool = true
    var acc_rob_rad_max: Float = 2.0
    var acc_rob_max_multiplier: Float = 8.0
    var acc_rob_max_frame_count: Float = 8.0

    /// App defaults — also applied by the settings Reset button.
    static let appDefaults = TuningParams()

    /// Legacy name used by the Reset button.
    static let ghostReductionPreset = TuningParams.appDefaults

    enum CodingKeys: String, CodingKey {
        case r_t, r_s1, r_s2, r_Mt
        case alignment_grey_fft
        case hf_artifact_removal_enabled, hf_variance_loss_threshold
        case hf_variance_noise_multiplier, hf_noise_floor_multiplier
        case motion_edge_rejection_enabled, motion_edge_threshold, motion_edge_residual_threshold
        case motion_edge_noise_floor_multiplier, motion_edge_neighborhood_radius
        case k_detail, k_denoise, k_stretch, k_shrink
        case snr_auto_tune, alignment_tile_size
        case global_prealignment_enabled, global_prealignment_choose_reference
        case global_prealignment_rotation_range_deg, global_prealignment_rotation_step_deg
        case global_prealignment_max_shift
        case robustness_save_mask
        case accumulated_robustness_denoiser_enabled
        case acc_rob_rad_max, acc_rob_max_multiplier, acc_rob_max_frame_count
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
        hf_variance_noise_multiplier = try c.decodeIfPresent(Float.self, forKey: .hf_variance_noise_multiplier) ?? hf_variance_noise_multiplier
        hf_noise_floor_multiplier = try c.decodeIfPresent(Float.self, forKey: .hf_noise_floor_multiplier) ?? hf_noise_floor_multiplier
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
        alignment_tile_size = try c.decodeIfPresent(Int.self, forKey: .alignment_tile_size) ?? alignment_tile_size
        global_prealignment_enabled = try c.decodeIfPresent(Bool.self, forKey: .global_prealignment_enabled) ?? global_prealignment_enabled
        global_prealignment_choose_reference = try c.decodeIfPresent(Bool.self, forKey: .global_prealignment_choose_reference) ?? global_prealignment_choose_reference
        global_prealignment_rotation_range_deg = try c.decodeIfPresent(Float.self, forKey: .global_prealignment_rotation_range_deg) ?? global_prealignment_rotation_range_deg
        global_prealignment_rotation_step_deg = try c.decodeIfPresent(Float.self, forKey: .global_prealignment_rotation_step_deg) ?? global_prealignment_rotation_step_deg
        global_prealignment_max_shift = try c.decodeIfPresent(Int.self, forKey: .global_prealignment_max_shift) ?? global_prealignment_max_shift
        robustness_save_mask = try c.decodeIfPresent(Bool.self, forKey: .robustness_save_mask) ?? robustness_save_mask
        accumulated_robustness_denoiser_enabled = try c.decodeIfPresent(Bool.self, forKey: .accumulated_robustness_denoiser_enabled) ?? accumulated_robustness_denoiser_enabled
        acc_rob_rad_max = try c.decodeIfPresent(Float.self, forKey: .acc_rob_rad_max) ?? acc_rob_rad_max
        acc_rob_max_multiplier = try c.decodeIfPresent(Float.self, forKey: .acc_rob_max_multiplier) ?? acc_rob_max_multiplier
        acc_rob_max_frame_count = try c.decodeIfPresent(Float.self, forKey: .acc_rob_max_frame_count) ?? acc_rob_max_frame_count
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

    // Shutter: Auto (A), or manual via log-scaled slider (0…1). Default = AE on.
    @Published var shutterIsAuto = CameraModel.persistedShutterIsAuto()
    @Published var shutterSlider: Double = CameraModel.persistedShutterSlider()
    @Published var exposureMinSec: Double = 1.0 / 8000.0
    @Published var exposureMaxSec: Double = 1.0 / 15.0

    static let minFrameCount = 2
    static let maxFrameCount = 8
    private static let frameCountDefaultsKey = "FrameCount"
    private static let shutterAutoDefaultsKey = "ShutterIsAuto"
    private static let shutterSliderDefaultsKey = "ShutterSlider"

    private static func persistedFrameCount() -> Int {
        guard UserDefaults.standard.object(forKey: frameCountDefaultsKey) != nil else { return 4 }
        let saved = UserDefaults.standard.integer(forKey: frameCountDefaultsKey)
        return min(maxFrameCount, max(minFrameCount, saved))
    }

    private static func persistedShutterIsAuto() -> Bool {
        guard UserDefaults.standard.object(forKey: shutterAutoDefaultsKey) != nil else { return true }
        return UserDefaults.standard.bool(forKey: shutterAutoDefaultsKey)
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

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session", qos: .userInteractive)
    private let processingQueue = DispatchQueue(label: "handheldsr.processing", qos: .userInitiated)
    private let photoOutput = AVCapturePhotoOutput()
    private var device: AVCaptureDevice?
    private var videoInput: AVCaptureDeviceInput?
    private var activeCameraSelection: CameraSelection = .wide
    private var lastBackSelection: CameraSelection = .wide

    private var activeFrameCount = CameraModel.persistedFrameCount()
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
        UserDefaults.standard.set(shutterIsAuto, forKey: Self.shutterAutoDefaultsKey)
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
        switch mode {
        case .ultraWide:
            guard availableCameras.contains(.ultraWide) else { return }
            lensZoomMode = mode
            setCamera(.ultraWide)
        case .wide1x, .wide2x:
            guard availableCameras.contains(.wide) else { return }
            lensZoomMode = mode
            setCamera(.wide)
        case .telephoto:
            guard availableCameras.contains(.telephoto) else { return }
            lensZoomMode = mode
            setCamera(.telephoto)
        }
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
        switch selection {
        case .ultraWide:
            lensZoomMode = .ultraWide
        case .telephoto:
            lensZoomMode = .telephoto
        case .wide:
            if lensZoomMode == .ultraWide || lensZoomMode == .telephoto {
                lensZoomMode = .wide1x
            }
        case .front:
            break
        }
    }

    private func discoverCamerasAfterSetup() {
        var found: [CameraSelection] = []
        if device(for: .wide) != nil { found.append(.wide) }
        if device(for: .ultraWide) != nil { found.append(.ultraWide) }
        if device(for: .telephoto) != nil { found.append(.telephoto) }
        if device(for: .front) != nil { found.append(.front) }
        if found.isEmpty { found = [.wide] }
        let teleLabel = inferredTelephotoLensLabel()
        DispatchQueue.main.async {
            self.availableCameras = found
            self.telephotoLensLabel = teleLabel
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
        if isAuto {
            if d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
            }
        } else if d.isExposureModeSupported(.custom) {
            let minD = d.activeFormat.minExposureDuration
            let maxD = d.activeFormat.maxExposureDuration
            var t = CMTimeMakeWithSeconds(durationFromSlider(slider, minSec: minSec, maxSec: maxSec), preferredTimescale: 1_000_000_000)
            if CMTimeCompare(t, minD) < 0 { t = minD }
            if CMTimeCompare(t, maxD) > 0 { t = maxD }
            let iso = min(max(d.activeFormat.minISO, d.iso), d.activeFormat.maxISO)
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
        sessionQueue.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            if d.isFocusPointOfInterestSupported {
                d.focusPointOfInterest = devicePoint
                if d.isFocusModeSupported(.autoFocus) { d.focusMode = .autoFocus }
            }
            if d.isExposurePointOfInterestSupported {
                d.exposurePointOfInterest = devicePoint
                if self.shutterIsAuto, d.isExposureModeSupported(.continuousAutoExposure) {
                    d.exposureMode = .continuousAutoExposure
                }
            }
            d.unlockForConfiguration()
        }
    }

    // MARK: - Capture burst

    func captureBurst() {
        guard !isBusy else { return }

        let total = frameCount
        let lens = lensZoomMode.label
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

    private func processBurst() {
        let rawFrames = capturedRawFrames
        var paths = capturedDNGs.map { $0.path }
        var usingDocDNGs = false

        // DEBUG: if Documents contains ≥2 DNGs, process those (sorted) instead of the camera burst.
        let fm = FileManager.default
        if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first,
           let items = try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil) {
            let docDNGs = items.filter { $0.pathExtension.lowercased() == "dng" }
                .map { $0.path }
                .sorted()
            if docDNGs.count >= 2 {
                paths = docDNGs
                usingDocDNGs = true
                print("DEBUG OVERRIDE: Using \(paths.count) DNGs from Documents (ref=paths[0])")
            }
        }

        let burstDir = self.burstDir
        guard rawFrames.count >= 2 || paths.count >= 2 else {
            removeBurstDir(burstDir)
            self.burstDir = nil
            finish(success: false, message: "Not enough frames captured")
            return
        }
        let useSelectedOutputScale = !usingDocDNGs &&
            activeCameraSelection == .wide &&
            lensZoomMode == .wide1x
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
            "hf_variance_noise_multiplier": NSNumber(value: tuningParams.hf_variance_noise_multiplier),
            "hf_noise_floor_multiplier": NSNumber(value: tuningParams.hf_noise_floor_multiplier),
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
            "alignment_tile_size": NSNumber(value: tuningParams.alignment_tile_size),
            "global_prealignment_enabled": NSNumber(value: tuningParams.global_prealignment_enabled),
            "global_prealignment_choose_reference": NSNumber(value: tuningParams.global_prealignment_choose_reference),
            "global_prealignment_rotation_range_deg": NSNumber(value: tuningParams.global_prealignment_rotation_range_deg),
            "global_prealignment_rotation_step_deg": NSNumber(value: tuningParams.global_prealignment_rotation_step_deg),
            "global_prealignment_max_shift": NSNumber(value: tuningParams.global_prealignment_max_shift),
            "robustness_save_mask": NSNumber(value: tuningParams.robustness_save_mask),
            "accumulated_robustness_denoiser_enabled": NSNumber(value: tuningParams.accumulated_robustness_denoiser_enabled),
            "acc_rob_rad_max": NSNumber(value: tuningParams.acc_rob_rad_max),
            "acc_rob_max_multiplier": NSNumber(value: tuningParams.acc_rob_max_multiplier),
            "acc_rob_max_frame_count": NSNumber(value: tuningParams.acc_rob_max_frame_count)
        ]

        var preview: UIImage?
        // Documents debug DNGs: no center-crop (match Python full-frame run).
        let cropFactor = (usingDocDNGs || activeCameraSelection != .wide)
            ? Int32(1)
            : Int32(lensZoomMode.cropFactor)
        let inputURLs = capturedDNGs
        let rawInputPaths = rawFrames.compactMap { $0["path"] as? String }
        let progressBlock: (String, Float) -> Void = { [weak self] stage, frac in
            DispatchQueue.main.async {
                self?.progress = 0.15 + frac * 0.85
                if stage.hasPrefix("Noise ") {
                    self?.noiseDiagText = stage
                } else {
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
                cropFactor: cropFactor,
                tuningParams: tuningDict,
                progress: progressBlock,
                previewImage: &preview
            )
        } else {
            ok = SRBridge.processDNGs(
                paths,
                toPath: outURL.path,
                scale: algorithmScale,
                cropFactor: cropFactor,
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
            let robURL: URL? = tuningParams.robustness_save_mask
                ? URL(fileURLWithPath: outURL.deletingPathExtension().path + "_robustness.pgm")
                : nil
            saveToPhotos(url: outURL, robustnessMask: robURL, preview: preview, burstDir: burstDir)
        } else {
            removeBurstDir(burstDir)
            self.burstDir = nil
            finish(success: false, message: "Processing failed")
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

    private func saveToPhotos(url: URL, robustnessMask: URL?, preview: UIImage?, burstDir: URL?) {
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

            var maskJPEG: URL?
            if let rob = robustnessMask, FileManager.default.fileExists(atPath: rob.path),
               let img = Self.uiImageFromPGM(url: rob),
               let jpeg = img.jpegData(compressionQuality: 0.92) {
                let tmp = FileManager.default.temporaryDirectory
                    .appendingPathComponent("robustness_mask_\(UUID().uuidString).jpg")
                try? jpeg.write(to: tmp, options: .atomic)
                maskJPEG = tmp
            }
            let savedMask = maskJPEG != nil
            let label = format == .jpg ? "JPG" : "DNG"
            PHPhotoLibrary.shared().performChanges({
                let req = PHAssetCreationRequest.forAsset()
                let opts = PHAssetResourceCreationOptions()
                opts.shouldMoveFile = false
                if format == .dng {
                    opts.uniformTypeIdentifier = "com.adobe.raw-image"
                }
                req.addResource(with: .photo, fileURL: saveURL, options: opts)
                if let maskJPEG {
                    // Its own options. Reusing `opts` tagged this JPEG with
                    // uniformTypeIdentifier "com.adobe.raw-image" whenever the
                    // export format was DNG, so Photos was told a JPEG was a
                    // RAW file and rejected the resource -- which is why the
                    // mask never appeared even with the toggle on. Only the DNG
                    // export path was affected; JPG export sets no UTI.
                    let mopts = PHAssetResourceCreationOptions()
                    mopts.shouldMoveFile = false
                    let mreq = PHAssetCreationRequest.forAsset()
                    mreq.addResource(with: .photo, fileURL: maskJPEG, options: mopts)
                }
            }, completionHandler: { success, _ in
                if let maskJPEG { try? FileManager.default.removeItem(at: maskJPEG) }
                if let tempJPEG { try? FileManager.default.removeItem(at: tempJPEG) }
                DispatchQueue.main.async {
                    self.lastThumbnail = preview
                    self.finish(success: success,
                                message: success
                                    ? (savedMask
                                       ? "Saved \(label) + robustness mask to Photos"
                                       : "Saved super-res \(label) to Photos")
                                    : "Could not save to Photos")
                }
                self.removeBurstDir(burstDir)
                self.burstDir = nil
            })
        }
    }

    private func finish(success: Bool, message: String) {
        DispatchQueue.main.async {
            self.isBusy = false
            self.isCapturing = false
            self.isProcessing = false
            self.progress = success ? 1 : 0
            self.statusText = message
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
