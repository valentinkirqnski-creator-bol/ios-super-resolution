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
    @Published var lastSavedURL: URL?
    @Published var permissionDenied = false
    @Published var cameraSelection: CameraSelection = .wide
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

    // Shutter: Auto, or manual via log-scaled slider (0…1).
    @Published var shutterIsAuto = true
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
    private var pendingCaptures = 0
    private var capturedDNGs: [URL] = []
    private var burstDir: URL?
    private var isAppActive = true
    private var previewSuspended = false

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
            } else if self.session.isRunning {
                self.session.stopRunning()
                DispatchQueue.main.async { self.isSessionRunning = false }
            }
        }
    }

    func setPreviewSuspended(_ suspended: Bool) {
        sessionQueue.async {
            self.previewSuspended = suspended
            guard !self.isBusy else { return }
            if suspended {
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
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    func setShutterAuto(_ auto: Bool) {
        shutterIsAuto = auto
        applyShutter()
    }

    func applyManualShutterFromSlider() {
        shutterIsAuto = false
        applyShutter()
    }

    func setCamera(_ selection: CameraSelection) {
        guard !isBusy, availableCameras.contains(selection) else { return }
        sessionQueue.async {
            guard selection != self.activeCameraSelection else { return }
            if selection != .front { self.lastBackSelection = selection }
            self.activeCameraSelection = selection
            DispatchQueue.main.async { self.cameraSelection = selection }
            self.switchCameraDevice(to: selection)
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
        applyShutterOnSessionQueue()

        session.commitConfiguration()
        startPreviewIfNeeded()
        DispatchQueue.main.async {
            self.isSessionRunning = self.session.isRunning
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
        applyShutterOnSessionQueue()

        session.commitConfiguration()
        DispatchQueue.main.async {
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

    private func durationFromSlider(_ t: Double) -> Double {
        let clamped = min(1.0, max(0.0, t))
        let logMin = log(exposureMinSec)
        let logMax = log(exposureMaxSec)
        return exp(logMin + clamped * (logMax - logMin))
    }

    private func applyShutterOnSessionQueue() {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        if shutterIsAuto {
            if d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
            }
        } else if d.isExposureModeSupported(.custom) {
            let minD = d.activeFormat.minExposureDuration
            let maxD = d.activeFormat.maxExposureDuration
            var t = CMTimeMakeWithSeconds(durationFromSlider(shutterSlider), preferredTimescale: 1_000_000_000)
            if CMTimeCompare(t, minD) < 0 { t = minD }
            if CMTimeCompare(t, maxD) > 0 { t = maxD }
            let iso = min(max(d.activeFormat.minISO, d.iso), d.activeFormat.maxISO)
            d.setExposureModeCustom(duration: t, iso: iso, completionHandler: nil)
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
        applyShutterOnSessionQueue()
    }

    private func setResponsiveCaptureEnabled(_ enabled: Bool) {
        if #available(iOS 17.0, *) {
            photoOutput.isResponsiveCaptureEnabled = enabled
        }
    }

    private func startPreviewIfNeeded() {
        guard isAppActive, !previewSuspended, !session.isRunning else { return }
        session.startRunning()
        DispatchQueue.main.async { self.isSessionRunning = self.session.isRunning }
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

    private func applyShutter() {
        sessionQueue.async { self.applyShutterOnSessionQueue() }
    }

    // MARK: - Capture burst

    func captureBurst() {
        guard !isBusy else { return }
        sessionQueue.async {
            guard self.photoOutput.availableRawPhotoPixelFormatTypes.first != nil else {
                DispatchQueue.main.async {
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
            self.pendingCaptures = self.currentBurstTotal

            DispatchQueue.main.async {
                self.isBusy = true
                self.isCapturing = true
                self.isProcessing = false
                self.progress = 0
                let lens = self.cameraSelection.label
                self.statusText = "Capturing \(self.currentBurstTotal) frames · \(lens)"
            }

            self.photoOutput.maxPhotoQualityPrioritization = .speed
            self.setResponsiveCaptureEnabled(true)
            self.lockForBurst()
            guard self.captureNextRaw() else { return }
        }
    }

    private func abortBurst(_ message: String) {
        sessionQueue.async {
            self.pendingCaptures = 0
            self.capturedDNGs.removeAll()
            self.restoreCaptureQuality()
            if let d = self.device, (try? d.lockForConfiguration()) != nil {
                if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
                if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                    d.whiteBalanceMode = .continuousAutoWhiteBalance
                }
                self.applyShutterOnSessionQueue()
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
            self.applyShutterOnSessionQueue()
            d.unlockForConfiguration()
        }
    }

    @discardableResult
    private func captureNextRaw() -> Bool {
        guard let rawFormat = photoOutput.availableRawPhotoPixelFormatTypes.first else {
            abortBurst("RAW capture unavailable")
            return false
        }
        let settings = AVCapturePhotoSettings(rawPixelFormatType: rawFormat)
        settings.flashMode = .off
        settings.photoQualityPrioritization = .speed
        settings.isAutoStillImageStabilizationEnabled = false
        applyRawCaptureLimits(to: settings)
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

    private func processBurst() {
        let paths = capturedDNGs.map { $0.path }
        let burstDir = self.burstDir
        guard paths.count >= 2 else {
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

        var preview: UIImage?
        let ok = SRBridge.processDNGs(
            paths,
            toPath: outURL.path,
            scale: 2.0,
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
            saveToPhotos(url: outURL, preview: preview, burstDir: burstDir)
        } else {
            finish(success: false, message: "Processing failed")
        }
    }

    private func saveToPhotos(url: URL, preview: UIImage?, burstDir: URL?) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    self.lastThumbnail = preview
                    self.lastSavedURL = url
                    self.finish(success: true, message: "Saved to app storage (grant Photos access to save to library)")
                }
                self.cleanupBurstDir(burstDir, keep: url)
                return
            }
            PHPhotoLibrary.shared().performChanges({
                let req = PHAssetCreationRequest.forAsset()
                let opts = PHAssetResourceCreationOptions()
                opts.shouldMoveFile = false
                req.addResource(with: .photo, fileURL: url, options: opts)
            }, completionHandler: { success, _ in
                DispatchQueue.main.async {
                    self.lastThumbnail = preview
                    self.lastSavedURL = url
                    self.finish(success: success,
                                message: success ? "Saved super-res DNG to Photos" : "Saved to app storage")
                }
                self.cleanupBurstDir(burstDir, keep: url)
            })
        }
    }

    private func cleanupBurstDir(_ dir: URL?, keep output: URL) {
        guard let dir = dir else { return }
        processingQueue.async {
            guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return }
            for name in names {
                let url = dir.appendingPathComponent(name)
                if url.path != output.path {
                    try? FileManager.default.removeItem(at: url)
                }
            }
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
            if self.isAppActive && !self.previewSuspended {
                self.startPreviewIfNeeded()
            }
        }
    }

    private func stopSessionForProcessing(completion: @escaping () -> Void) {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
                DispatchQueue.main.async { self.isSessionRunning = false }
            }
            completion()
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

        pendingCaptures -= 1
        DispatchQueue.main.async {
            let done = self.currentBurstTotal - self.pendingCaptures
            self.progress = Float(done) / Float(self.currentBurstTotal) * 0.15
        }

        if pendingCaptures > 0 {
            sessionQueue.async {
                guard self.captureNextRaw() else { return }
            }
        } else {
            unlockAfterBurst()
            stopSessionForProcessing { [weak self] in
                guard let self = self else { return }
                self.processingQueue.async { self.processBurst() }
            }
        }
    }
}
