import AVFoundation
import Photos
import UIKit
import Combine

/// Available shutter (exposure duration) choices. `.auto` lets the camera pick
/// based on the light; the others force a manual duration.
enum ShutterSetting: Identifiable, Hashable {
    case auto
    case manual(seconds: Double)   // e.g. 1/125 -> 1.0/125.0

    var id: String { label }

    var label: String {
        switch self {
        case .auto: return "Auto"
        case .manual(let s):
            if s >= 1 { return "\(Int(s))\"" }
            return "1/\(Int((1.0 / s).rounded()))"
        }
    }

    static let choices: [ShutterSetting] = [
        .auto,
        .manual(seconds: 1.0 / 1000.0),
        .manual(seconds: 1.0 / 500.0),
        .manual(seconds: 1.0 / 250.0),
        .manual(seconds: 1.0 / 125.0),
        .manual(seconds: 1.0 / 60.0),
        .manual(seconds: 1.0 / 30.0),
        .manual(seconds: 1.0 / 15.0),
    ]
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
    @Published var shutter: ShutterSetting = .auto { didSet { applyShutter() } }
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

    static let minFrameCount = 2
    static let maxFrameCount = 8

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let processingQueue = DispatchQueue(label: "handheldsr.processing", qos: .userInitiated)
    private let photoOutput = AVCapturePhotoOutput()
    private var device: AVCaptureDevice?
    private var videoInput: AVCaptureDeviceInput?
    private var activeCameraSelection: CameraSelection = .wide

    private var activeFrameCount = 4
    private var currentBurstTotal = 4
    private var pendingCaptures = 0
    private var capturedDNGs: [URL] = []
    private var burstDir: URL?

    // MARK: - Setup

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

    func setCamera(_ selection: CameraSelection) {
        guard !isBusy, availableCameras.contains(selection) else { return }
        sessionQueue.async {
            guard selection != self.activeCameraSelection else { return }
            self.activeCameraSelection = selection
            DispatchQueue.main.async { self.cameraSelection = selection }
            self.switchCameraDevice(to: selection)
        }
    }

    /// List lenses after the live session is configured (no extra probe sessions).
    private func discoverCamerasAfterSetup() {
        var found: [CameraSelection] = []
        if device(for: .wide) != nil {
            found.append(.wide)
        }
        if device(for: .ultraWide) != nil {
            found.append(.ultraWide)
        }
        if device(for: .front) != nil {
            found.append(.front)
        }
        if found.isEmpty { found = [.wide] }
        DispatchQueue.main.async { self.availableCameras = found }
    }

    private func configureSession() {
        session.beginConfiguration()
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
        photoOutput.maxPhotoQualityPrioritization = .quality
        configureRawCaptureLimits()
        applyDefaultDeviceModes()
        applyShutterOnSessionQueue()

        session.commitConfiguration()
        session.startRunning()
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

    /// Apply shutter/exposure on the session queue (caller must already be on sessionQueue).
    private func applyShutterOnSessionQueue() {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        switch shutter {
        case .auto:
            if d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
            }
        case .manual(let seconds):
            if d.isExposureModeSupported(.custom) {
                let minD = d.activeFormat.minExposureDuration
                let maxD = d.activeFormat.maxExposureDuration
                var t = CMTimeMakeWithSeconds(seconds, preferredTimescale: 1_000_000_000)
                if CMTimeCompare(t, minD) < 0 { t = minD }
                if CMTimeCompare(t, maxD) > 0 { t = maxD }
                let iso = min(max(d.activeFormat.minISO, d.iso), d.activeFormat.maxISO)
                d.setExposureModeCustom(duration: t, iso: iso, completionHandler: nil)
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
        if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
        if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            d.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        d.unlockForConfiguration()
    }

    // MARK: - Focus / exposure controls

    /// Tap-to-focus at a normalized device point (0..1, from the preview layer).
    func focus(at devicePoint: CGPoint) {
        sessionQueue.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            if d.isFocusPointOfInterestSupported {
                d.focusPointOfInterest = devicePoint
                if d.isFocusModeSupported(.autoFocus) { d.focusMode = .autoFocus }
            }
            if d.isExposurePointOfInterestSupported {
                d.exposurePointOfInterest = devicePoint
                if self.isAutoShutter, d.isExposureModeSupported(.continuousAutoExposure) {
                    d.exposureMode = .continuousAutoExposure
                }
            }
            d.unlockForConfiguration()
        }
    }

    private var isAutoShutter: Bool {
        if case .auto = shutter { return true }
        return false
    }

    private func applyShutter() {
        sessionQueue.async { self.applyShutterOnSessionQueue() }
    }

    // MARK: - Capture burst

    func captureBurst() {
        guard !isBusy else { return }
        sessionQueue.async {
            guard self.photoOutput.availableRawPhotoPixelFormatTypes.first != nil else {
                DispatchQueue.main.async { self.statusText = "RAW capture not supported on this camera" }
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
            self.lockForBurst()
            self.captureNextRaw()
        }
    }

    private func lockForBurst() {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        if d.isFocusModeSupported(.locked) { d.focusMode = .locked }
        if d.isWhiteBalanceModeSupported(.locked) { d.whiteBalanceMode = .locked }
        if isAutoShutter, d.isExposureModeSupported(.locked) { d.exposureMode = .locked }
        d.unlockForConfiguration()
    }

    private func unlockAfterBurst() {
        sessionQueue.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
            if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                d.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            if self.isAutoShutter, d.isExposureModeSupported(.continuousAutoExposure) {
                d.exposureMode = .continuousAutoExposure
            }
            d.unlockForConfiguration()
        }
    }

    private func captureNextRaw() {
        guard let rawFormat = photoOutput.availableRawPhotoPixelFormatTypes.first else { return }
        let settings = AVCapturePhotoSettings(rawPixelFormatType: rawFormat)
        settings.flashMode = .off
        applyRawCaptureLimits(to: settings)
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    /// Target ~12 MP (4032×3024) when supported; otherwise the largest size under ~15 MP.
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
        restorePhotoPreview()
    }

    /// Drop preview to a lighter preset but keep the viewfinder alive during merge.
    private func enterLowResPreviewForProcessing(completion: @escaping () -> Void) {
        sessionQueue.async {
            if self.session.sessionPreset != .medium {
                self.session.beginConfiguration()
                self.session.sessionPreset = .medium
                self.session.commitConfiguration()
            }
            if !self.session.isRunning {
                self.session.startRunning()
            }
            DispatchQueue.main.async {
                self.isSessionRunning = self.session.isRunning
                completion()
            }
        }
    }

    private func restorePhotoPreview() {
        sessionQueue.async {
            self.session.beginConfiguration()
            self.session.sessionPreset = .photo
            self.configureRawCaptureLimits()
            self.session.commitConfiguration()
            if !self.session.isRunning {
                self.session.startRunning()
            }
            DispatchQueue.main.async { self.isSessionRunning = self.session.isRunning }
        }
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraModel: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        if let error = error {
            DispatchQueue.main.async { self.statusText = "Capture error: \(error.localizedDescription)" }
        } else if photo.isRawPhoto, let dir = burstDir {
            autoreleasepool {
                guard let data = photo.fileDataRepresentation() else { return }
                let idx = capturedDNGs.count
                let url = dir.appendingPathComponent("frame_\(idx).dng")
                do {
                    try data.write(to: url, options: .atomic)
                    capturedDNGs.append(url)
                } catch {
                    DispatchQueue.main.async { self.statusText = "Write error: \(error.localizedDescription)" }
                }
            }
        }

        pendingCaptures -= 1
        DispatchQueue.main.async {
            let done = self.currentBurstTotal - self.pendingCaptures
            self.progress = Float(done) / Float(self.currentBurstTotal) * 0.15
        }

        if pendingCaptures > 0 {
            sessionQueue.async { self.captureNextRaw() }
        } else {
            unlockAfterBurst()
            enterLowResPreviewForProcessing { [weak self] in
                guard let self = self else { return }
                self.processingQueue.async { self.processBurst() }
            }
        }
    }
}
