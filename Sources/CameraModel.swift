import AVFoundation
import Photos
import UIKit
import Combine

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
/// Defaults match configs/default.yaml / core/types.h.
struct TuningParams: Equatable, Codable {
    var r_t: Float = 0.12
    var r_s1: Float = 2.0
    var r_s2: Float = 12.0
    var r_Mt: Float = 0.80
    var k_detail: Float = 0.25
    var k_denoise: Float = 4.0
    var k_stretch: Float = 4.0
    var snr_auto_tune: Bool = true
    var accumulated_robustness_denoiser_enabled: Bool = false
    var acc_rob_rad_max: Float = 2.0
    var acc_rob_max_multiplier: Float = 8.0
    var acc_rob_max_frame_count: Float = 2.0

    /// Aggressive motion-ghost preset used by the settings Reset button.
    static let ghostReductionPreset = TuningParams(
        r_t: 1.00,
        r_s1: 0.25,
        r_s2: 2.5,
        r_Mt: 0.80,
        k_detail: 0.20,
        k_denoise: 0.0,
        k_stretch: 4.0,
        snr_auto_tune: false,
        accumulated_robustness_denoiser_enabled: true,
        acc_rob_rad_max: 2.0,
        acc_rob_max_multiplier: 5.2,
        acc_rob_max_frame_count: 2.0
    )
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
    @Published var tuningParams: TuningParams = {
        if let data = UserDefaults.standard.data(forKey: "TuningParams"),
           let params = try? JSONDecoder().decode(TuningParams.self, from: data) {
            return params
        }
        return TuningParams()
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

    // Shutter: Auto (A), or manual via log-scaled slider (0…1).
    @Published var shutterIsAuto = false
    @Published var shutterSlider: Double = 0.5
    @Published var exposureMinSec: Double = 1.0 / 8000.0
    @Published var exposureMaxSec: Double = 1.0 / 15.0

    static let minFrameCount = 2
    static let maxFrameCount = 8

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session")
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
    private var isAppActive = true
    private var previewSuspended = false
    private var exposureSyncTimer: Timer?

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
            guard !self.isBusy else { return }
            if active && !self.previewSuspended {
                self.startPreviewIfNeeded()
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
            guard !self.isBusy else { return }
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

    private func setResponsiveCaptureEnabled(_ enabled: Bool) {
        if #available(iOS 17.0, *) {
            photoOutput.isResponsiveCaptureEnabled = enabled
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
        photoOutput.maxPhotoQualityPrioritization = .balanced
        setResponsiveCaptureEnabled(false)
        configureRawCaptureLimits()
        refreshExposureRange()
        applyDefaultDeviceModes()

        session.commitConfiguration()
        startPreviewIfNeeded()
        DispatchQueue.main.async {
            self.isSessionRunning = self.session.isRunning
            self.startAutoExposureSyncIfNeeded()
            if self.photoOutput.availableRawPhotoPixelFormatTypes.isEmpty {
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

        configureRawCaptureLimits()
        refreshExposureRange()
        applyDefaultDeviceModes()

        session.commitConfiguration()
        DispatchQueue.main.async {
            self.startAutoExposureSyncIfNeeded()
            if self.photoOutput.availableRawPhotoPixelFormatTypes.isEmpty {
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
        statusText = "Capturing \(total) frames · \(lens)"

        sessionQueue.async {
            guard self.photoOutput.availableRawPhotoPixelFormatTypes.first != nil else {
                DispatchQueue.main.async {
                    self.isBusy = false
                    self.isCapturing = false
                    self.statusText = "RAW capture not supported on this camera"
                }
                return
            }

            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("burst_\(Int(Date().timeIntervalSince1970))")
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            self.burstDir = dir
            self.capturedDNGs.removeAll()
            self.currentBurstTotal = self.activeFrameCount
            self.capturesRequested = 0
            self.capturesProcessed = 0

            self.photoOutput.maxPhotoQualityPrioritization = .speed
            self.setResponsiveCaptureEnabled(true)
            self.lockForBurst()
            self.captureNextRaw()
        }
    }

    private func abortBurst(_ message: String) {
        sessionQueue.async {
            let dir = self.burstDir
            self.burstDir = nil
            self.capturesRequested = self.currentBurstTotal
            self.capturesProcessed = self.currentBurstTotal
            self.capturedDNGs.removeAll()
            self.restoreCaptureQuality()
            self.removeBurstDir(dir)
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
            self.restoreCaptureQuality()
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
    private func captureNextRaw() -> Bool {
        if capturesRequested >= currentBurstTotal { return false }
        guard let rawFormat = photoOutput.availableRawPhotoPixelFormatTypes.first else {
            abortBurst("RAW capture unavailable")
            return false
        }
        capturesRequested += 1
        let settings = AVCapturePhotoSettings(rawPixelFormatType: rawFormat)
        settings.flashMode = .off
        settings.photoQualityPrioritization = .speed
        settings.isAutoStillImageStabilizationEnabled = false
        applyRawCaptureLimits(to: settings)
        applyShutterSoundSuppression(to: settings)
        photoOutput.capturePhoto(with: settings, delegate: self)
        return true
    }

    private func restoreCaptureQuality() {
        photoOutput.maxPhotoQualityPrioritization = .balanced
        setResponsiveCaptureEnabled(false)
    }

    private func configureRawCaptureLimits() {
        if #available(iOS 16.0, *) {
            photoOutput.isHighResolutionCaptureEnabled = false
            if let dims = preferredRawDimensions() {
                photoOutput.maxPhotoDimensions = dims
            }
        }
    }

    private func applyRawCaptureLimits(to settings: AVCapturePhotoSettings) {
        if #available(iOS 16.0, *) {
            settings.isHighResolutionPhotoEnabled = false
            if let dims = preferredRawDimensions() {
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
    private func purgeStaleCaptureFiles() {
        processingQueue.async {
            let fm = FileManager.default
            let tmp = fm.temporaryDirectory
            if let entries = try? fm.contentsOfDirectory(
                at: tmp, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) {
                for url in entries {
                    let name = url.lastPathComponent
                    if name.hasPrefix("burst_") || name.hasSuffix("_cache") {
                        try? fm.removeItem(at: url)
                    }
                }
            }
            if let caches = fm.urls(for: .cachesDirectory, in: .userDomainMask).first,
               let entries = try? fm.contentsOfDirectory(
                at: caches, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) {
                for url in entries {
                    let name = url.lastPathComponent
                    if name.hasPrefix("burst_") || name.hasSuffix("_cache") {
                        try? fm.removeItem(at: url)
                    }
                }
            }
        }
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

    private func saveToPhotos(url: URL, robustnessMask: URL?, preview: UIImage?, burstDir: URL?) {
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
            PHPhotoLibrary.shared().performChanges({
                let req = PHAssetCreationRequest.forAsset()
                let opts = PHAssetResourceCreationOptions()
                opts.shouldMoveFile = false
                req.addResource(with: .photo, fileURL: url, options: opts)
                if let maskJPEG {
                    let mreq = PHAssetCreationRequest.forAsset()
                    mreq.addResource(with: .photo, fileURL: maskJPEG, options: opts)
                }
            }, completionHandler: { success, _ in
                if let maskJPEG { try? FileManager.default.removeItem(at: maskJPEG) }
                DispatchQueue.main.async {
                    self.lastThumbnail = preview
                    self.finish(success: success,
                                message: success
                                    ? (savedMask
                                       ? "Saved DNG + robustness mask to Photos"
                                       : "Saved super-res DNG to Photos")
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
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraModel: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
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
                    try data.write(to: url, options: .atomic)
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
            processingQueue.async { self.processBurst() }
        }
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings,
                     error: Error?) {
        if error != nil { return }
        if capturesRequested < currentBurstTotal {
            sessionQueue.async {
                self.captureNextRaw()
            }
        }
    }
}
