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

    /// Front uses more frames for hand-shake; rear cameras stay at 4 for speed/RAM.
    var burstCount: Int {
        switch self {
        case .front: return 8
        default: return 4
        }
    }
}

/// Owns the capture session, performs a Bayer RAW (DNG) burst, then runs
/// the multi-frame super-resolution pipeline on a background queue.
final class CameraModel: NSObject, ObservableObject {

    // Published UI state.
    @Published var isSessionRunning = false
    @Published var isBusy = false            // capturing or processing
    @Published var statusText = ""
    @Published var progress: Float = 0
    @Published var lastThumbnail: UIImage?
    @Published var lastSavedURL: URL?
    @Published var shutter: ShutterSetting = .auto { didSet { applyShutter() } }
    @Published var permissionDenied = false
    @Published var cameraSelection: CameraSelection = .wide
    @Published var availableCameras: [CameraSelection] = [.wide]

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let processingQueue = DispatchQueue(label: "handheldsr.processing", qos: .userInitiated)
    private let photoOutput = AVCapturePhotoOutput()
    private var device: AVCaptureDevice?
    private var videoInput: AVCaptureDeviceInput?

    // Burst bookkeeping — count depends on selected camera.
    private var burstCount = 4
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
                self.discoverCameras()
                self.configureSession()
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
            guard selection != self.cameraSelection else { return }
            DispatchQueue.main.async { self.cameraSelection = selection }
            self.burstCount = selection.burstCount
            self.switchCameraDevice(to: selection)
        }
    }

    private func discoverCameras() {
        var found: [CameraSelection] = []
        if let wide = device(for: .wide), supportsRawPhotoCapture(device: wide) {
            found.append(.wide)
        }
        if let uw = device(for: .ultraWide), supportsRawPhotoCapture(device: uw) {
            found.append(.ultraWide)
        }
        if let front = device(for: .front), supportsRawPhotoCapture(device: front) {
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

        let selection = cameraSelection
        burstCount = selection.burstCount

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
        burstCount = selection.burstCount

        configureRawCaptureLimits()
        applyDefaultDeviceModes()

        session.commitConfiguration()
        DispatchQueue.main.async {
            if self.photoOutput.availableRawPhotoPixelFormatTypes.isEmpty {
                self.statusText = "RAW not supported on \(selection.label)"
            } else {
                self.statusText = ""
            }
        }
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

    /// Probe whether a device can deliver Bayer RAW through AVCapturePhotoOutput.
    private func supportsRawPhotoCapture(device: AVCaptureDevice) -> Bool {
        let probe = AVCapturePhotoOutput()
        let probeSession = AVCaptureSession()
        probeSession.beginConfiguration()
        defer { probeSession.commitConfiguration() }
        guard let input = try? AVCaptureDeviceInput(device: device),
              probeSession.canAddInput(input),
              probeSession.canAddOutput(probe) else { return false }
        probeSession.addInput(input)
        probeSession.addOutput(probe)
        return !probe.availableRawPhotoPixelFormatTypes.isEmpty
    }

    private func applyDefaultDeviceModes() {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
        if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
        if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            d.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        d.unlockForConfiguration()
        applyShutter()
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
        sessionQueue.async {
            guard let d = self.device, (try? d.lockForConfiguration()) != nil else { return }
            switch self.shutter {
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
            self.pendingCaptures = self.burstCount

            DispatchQueue.main.async {
                self.isBusy = true
                self.progress = 0
                let lens = self.cameraSelection.label
                self.statusText = "Capturing \(self.burstCount) RAW DNG frames (\(lens))…"
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
            self.progress = success ? 1 : 0
            self.statusText = message
        }
        resumeSessionAfterProcessing()
    }

    private func pauseSessionForProcessing(completion: @escaping () -> Void) {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
                DispatchQueue.main.async { self.isSessionRunning = false }
            }
            completion()
        }
    }

    private func resumeSessionAfterProcessing() {
        sessionQueue.async {
            guard !self.session.isRunning else { return }
            self.session.startRunning()
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
            let done = self.burstCount - self.pendingCaptures
            self.progress = Float(done) / Float(self.burstCount) * 0.15
        }

        if pendingCaptures > 0 {
            sessionQueue.async { self.captureNextRaw() }
        } else {
            unlockAfterBurst()
            pauseSessionForProcessing { [weak self] in
                guard let self = self else { return }
                self.processingQueue.async { self.processBurst() }
            }
        }
    }
}
