import AVFoundation
import Photos
import UIKit
import Combine

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
    case front

    var id: String { rawValue }

    var label: String {
        switch self {
        case .wide: return "1×"
        case .ultraWide: return "0.5×"
        case .front: return "Front"
        }
    }
}

/// Lens/zoom mode for back-camera RAW capture (2× = center crop before SR).
enum LensZoomMode: Equatable {
    case ultraWide
    case wide1x
    case wide2x

    var label: String {
        switch self {
        case .ultraWide: return "0.5×"
        case .wide1x: return "1×"
        case .wide2x: return "2×"
        }
    }

    var cropFactor: Int {
        switch self {
        case .wide2x: return 2
        default: return 1
        }
    }
}

/// Holds the C++ algorithm tuning parameters for live adjustments.
struct TuningParams: Equatable, Codable {
    var r_t: Float = 1.0
    var r_s1: Float = 1.52
    var r_s2: Float = 2.1
    var r_Mt: Float = 1.0
    var k_detail: Float = 0.17
    var k_denoise: Float = 0.0
    var k_stretch: Float = 4.0
    var snr_auto_tune: Bool = false
    var accumulated_robustness_denoiser_enabled: Bool = true
    var acc_rob_rad_max: Float = 2.0
    var acc_rob_max_multiplier: Float = 1.8
    var acc_rob_max_frame_count: Float = 1.0

    /// App defaults — also applied by the settings Reset button.
    static let appDefaults = TuningParams()

    /// Legacy name used by the Reset button.
    static let ghostReductionPreset = TuningParams.appDefaults
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
    @Published var progress: Float = 0
    @Published var lastThumbnail: UIImage?
    @Published var permissionDenied = false
    @Published var cameraSelection: CameraSelection = .wide
    @Published var lensZoomMode: LensZoomMode = .wide1x
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
        let defaultsVersion = 3
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
    @Published var frameCount: Int = 4 {
        didSet {
            let clamped = min(Self.maxFrameCount, max(Self.minFrameCount, frameCount))
            if frameCount != clamped {
                frameCount = clamped
                return
            }
            sessionQueue.async { self.activeFrameCount = clamped }
        }
    }

    // Shutter: Auto (A), or manual via log-scaled slider (0…1). Default = AE on.
    @Published var shutterIsAuto = true
    @Published var shutterSlider: Double = 0.5
    @Published var exposureMinSec: Double = 1.0 / 8000.0
    @Published var exposureMaxSec: Double = 1.0 / 15.0

    static let minFrameCount = 2
    static let maxFrameCount = 8

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session", qos: .userInteractive)
    private let processingQueue = DispatchQueue(label: "handheldsr.processing", qos: .userInitiated)
    private let photoOutput = AVCapturePhotoOutput()
    private var device: AVCaptureDevice?
    private var videoInput: AVCaptureDeviceInput?
    private var activeCameraSelection: CameraSelection = .wide
    private var lastBackSelection: CameraSelection = .wide

    private var activeFrameCount = 4
    private var currentBurstTotal = 4
    private var capturesRequested = 0
    private var capturesProcessed = 0
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
    private var captureKind: CaptureKind = .none
    private var zslWanted = false
    private var zslCapturing = false
    /// True while a burst is capturing/processing — ZSL ring + system ZSL stay off.
    private var zslPausedForPipeline = false
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
        applyShutter()
    }

    func applyManualShutterFromSlider() {
        guard !isBusy else { return }
        if shutterIsAuto { shutterIsAuto = false }
        applyShutter()
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
        zslRing.removeAll(keepingCapacity: true)
        zslCapturing = false
        captureKind = .none
        let dir = zslDir
        zslDir = nil
        processingQueue.async {
            for u in urls { try? FileManager.default.removeItem(at: u) }
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
        _ = captureNextRaw(isZSL: true)
    }

    private func scheduleNextZSL() {
        guard zslWanted, !pipelineBusy, !zslPausedForPipeline else { return }
        // Fill quickly, then refresh slower once primed so continuous DNG writes
        // don't thermally throttle the following LibRaw + Metal merge.
        let primed = zslRing.count >= activeFrameCount
        let delay = primed ? 0.28 : 0.09
        sessionQueue.asyncAfter(deadline: .now() + delay) { [weak self] in
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
                if selection == .ultraWide { self.lensZoomMode = .ultraWide }
            }
            self.switchCameraDevice(to: selection)
        }
    }

    func setLensZoom(_ mode: LensZoomMode) {
        guard !isBusy else { return }
        lensZoomMode = mode
        switch mode {
        case .ultraWide:
            guard availableCameras.contains(.ultraWide) else { return }
            setCamera(.ultraWide)
        case .wide1x, .wide2x:
            guard availableCameras.contains(.wide) else { return }
            setCamera(.wide)
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
        DispatchQueue.main.async { self.cameraSelection = selection }
        switchCameraDevice(to: selection)
    }

    private func discoverCamerasAfterSetup() {
        var found: [CameraSelection] = []
        if device(for: .wide) != nil { found.append(.wide) }
        if device(for: .ultraWide) != nil { found.append(.ultraWide) }
        if device(for: .front) != nil { found.append(.front) }
        if found.isEmpty { found = [.wide] }
        DispatchQueue.main.async { self.availableCameras = found }
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
            self.startAutoExposureSyncIfNeeded()
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
            self.startAutoExposureSyncIfNeeded()
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
        case .front:
            return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
        }
    }

    private func applyDefaultDeviceModes() {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
        if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            d.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        d.isSubjectAreaChangeMonitoringEnabled = false
        d.unlockForConfiguration()
        applyShutter()
    }

    private func startAutoExposureSyncIfNeeded() {
        exposureSyncTimer?.invalidate()
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
            if self.zslEnabled && self.zslRing.count >= self.activeFrameCount {
                self.zslCapturing = false
                self.captureKind = .none
                let n = self.activeFrameCount
                let take = Array(self.zslRing.suffix(n))
                self.zslRing.removeLast(n)
                let readyLeft = self.zslRing.count
                // Output-only folder (inputs stay in the ZSL ring dir until processBurst cleans up).
                self.ensureReadyBurstDir()
                guard let dir = self.readyBurstDir else {
                    // Put frames back so the buffer is not lost on folder failure.
                    self.zslRing.append(contentsOf: take)
                    self.resumeZSLAfterProcessing()
                    DispatchQueue.main.async {
                        self.isBusy = false
                        self.isCapturing = false
                        self.zslBufferReady = self.zslRing.count
                        self.statusText = "Could not create capture folder"
                    }
                    return
                }
                self.burstDir = dir
                self.readyBurstDir = nil
                self.capturedDNGs = take
                DispatchQueue.main.async {
                    self.zslBufferReady = readyLeft
                    self.statusText = "ZSL \(take.count) frames · \(lens)"
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
            self.capturedDNGs.removeAll(keepingCapacity: true)
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
            self.capturedDNGs.removeAll(keepingCapacity: true)
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
        let paths = capturedDNGs.map { $0.path }
        let burstDir = self.burstDir
        guard paths.count >= 2 else {
            removeBurstDir(burstDir)
            self.burstDir = nil
            finish(success: false, message: "Not enough frames captured")
            return
        }
        let outURL = (burstDir ?? FileManager.default.temporaryDirectory)
            .appendingPathComponent("handheld_sr_x2.dng")

        DispatchQueue.main.async {
            self.isCapturing = false
            self.isProcessing = true
            self.statusText = "Processing…"
            self.progress = 0.15
        }

        let tuningDict: [String: NSNumber] = [
            "r_t": NSNumber(value: tuningParams.r_t),
            "r_s1": NSNumber(value: tuningParams.r_s1),
            "r_s2": NSNumber(value: tuningParams.r_s2),
            "r_Mt": NSNumber(value: tuningParams.r_Mt),
            "k_detail": NSNumber(value: tuningParams.k_detail),
            "k_denoise": NSNumber(value: tuningParams.k_denoise),
            "k_stretch": NSNumber(value: tuningParams.k_stretch),
            "snr_auto_tune": NSNumber(value: tuningParams.snr_auto_tune),
            "accumulated_robustness_denoiser_enabled": NSNumber(value: tuningParams.accumulated_robustness_denoiser_enabled),
            "acc_rob_rad_max": NSNumber(value: tuningParams.acc_rob_rad_max),
            "acc_rob_max_multiplier": NSNumber(value: tuningParams.acc_rob_max_multiplier),
            "acc_rob_max_frame_count": NSNumber(value: tuningParams.acc_rob_max_frame_count)
        ]

        var preview: UIImage?
        let cropFactor = Int32(lensZoomMode.cropFactor)
        let inputURLs = capturedDNGs
        let ok = SRBridge.processDNGs(
            paths,
            toPath: outURL.path,
            scale: 2.0,
            cropFactor: cropFactor,
            tuningParams: tuningDict,
            progress: { [weak self] stage, frac in
                DispatchQueue.main.async {
                    self?.progress = 0.15 + frac * 0.85
                    self?.statusText = stage
                }
            },
            previewImage: &preview
        )

        capturedDNGs.removeAll()
        // Free ZSL ring inputs (and non-ZSL frame_*.dng) as soon as LibRaw is done.
        // Non-ZSL files also live under burstDir; removeBurstDir later is then cheaper.
        for u in inputURLs {
            try? FileManager.default.removeItem(at: u)
        }

        if ok {
            let robURL = URL(fileURLWithPath:
                outURL.deletingPathExtension().path + "_robustness.pgm")
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

    /// Lightroom-like finish from the SR DNG: Highlights −100, Shadows +60,
    /// Contrast +5, Sharpening 0, Noise Reduction 0 → JPEG.
    /// Uses our own Deflate LinearRaw decoder (ImageIO cannot read these DNGs).
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
            if format == .jpg {
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
                req.addResource(with: .photo, fileURL: saveURL, options: opts)
                if let maskJPEG {
                    let mreq = PHAssetCreationRequest.forAsset()
                    mreq.addResource(with: .photo, fileURL: maskJPEG, options: opts)
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
        }
        sessionQueue.async {
            self.resumeZSLAfterProcessing()
        }
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraModel: AVCapturePhotoCaptureDelegate {
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

        if photo.isRawPhoto, let dir = burstDir {
            autoreleasepool {
                guard let data = photo.fileDataRepresentation() else {
                    abortBurst("Could not read RAW data")
                    return
                }
                let idx = capturedDNGs.count
                let url = dir.appendingPathComponent("frame_\(idx).dng")
                do {
                    try data.write(to: url)
                    capturedDNGs.append(url)
                } catch {
                    abortBurst("Write error: \(error.localizedDescription)")
                    return
                }
            }
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
        defer {
            zslCapturing = false
            if captureKind == .zsl { captureKind = .none }
            // Never keep pumping while a burst is processing.
            if !pipelineBusy && !zslPausedForPipeline {
                scheduleNextZSL()
            }
        }
        if pipelineBusy || zslPausedForPipeline { return }
        if error != nil || !photo.isRawPhoto { return }
        guard let dir = zslDir else { return }
        autoreleasepool {
            guard let data = photo.fileDataRepresentation() else { return }
            zslSeq += 1
            let url = dir.appendingPathComponent("zsl_\(zslSeq).dng")
            do {
                try data.write(to: url)
                zslRing.append(url)
                let cap = max(activeFrameCount, 2)
                while zslRing.count > cap {
                    let old = zslRing.removeFirst()
                    try? FileManager.default.removeItem(at: old)
                }
                let n = zslRing.count
                DispatchQueue.main.async { self.zslBufferReady = n }
            } catch {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings,
                     error: Error?) {
        if error != nil { return }
        if captureKind == .burst, capturesRequested < currentBurstTotal {
            _ = captureNextRaw(isZSL: false)
        }
    }
}
