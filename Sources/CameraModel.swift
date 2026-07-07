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

/// Owns the capture session, performs a 4-frame RAW burst, then runs the
/// multi-frame super-resolution pipeline on a background queue and saves the DNG.
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

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let processingQueue = DispatchQueue(label: "handheldsr.processing", qos: .userInitiated)
    private let photoOutput = AVCapturePhotoOutput()
    private var device: AVCaptureDevice?

    // Burst bookkeeping — 4 frames keeps peak RAM low on device.
    private let burstCount = 4
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
            self.sessionQueue.async { self.configureSession() }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .photo

        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev),
              session.canAddInput(input) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.statusText = "No camera available" }
            return
        }
        session.addInput(input)
        device = dev

        guard session.canAddOutput(photoOutput) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(photoOutput)
        photoOutput.maxPhotoQualityPrioritization = .quality

        // Light-driven defaults: continuous autofocus + auto exposure.
        if let d = device, (try? d.lockForConfiguration()) != nil {
            if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
            if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
            if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                d.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            d.unlockForConfiguration()
        }

        session.commitConfiguration()
        session.startRunning()
        DispatchQueue.main.async { self.isSessionRunning = self.session.isRunning }
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
                    // Clamp the requested duration to the device's supported range.
                    let minD = d.activeFormat.minExposureDuration
                    let maxD = d.activeFormat.maxExposureDuration
                    var t = CMTimeMakeWithSeconds(seconds, preferredTimescale: 1_000_000_000)
                    if CMTimeCompare(t, minD) < 0 { t = minD }
                    if CMTimeCompare(t, maxD) > 0 { t = maxD }
                    // Keep ISO on auto by using the current value as a starting point.
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
                DispatchQueue.main.async { self.statusText = "RAW capture not supported on this device" }
                return
            }
            // Fresh temp directory for this burst.
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("burst_\(Int(Date().timeIntervalSince1970))")
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            self.burstDir = dir
            self.capturedDNGs.removeAll()
            self.pendingCaptures = self.burstCount

            DispatchQueue.main.async {
                self.isBusy = true
                self.progress = 0
                self.statusText = "Capturing \(self.burstCount) RAW frames…"
            }
            // Lock AF/AE/WB so all frames share identical settings (clean merge)
            self.lockForBurst()
            self.captureNextRaw()
        }
    }

    /// Freezes focus, exposure and white balance for the duration of the burst.
    private func lockForBurst() {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        if d.isFocusModeSupported(.locked) { d.focusMode = .locked }
        if d.isWhiteBalanceModeSupported(.locked) { d.whiteBalanceMode = .locked }
        // Keep a user-chosen manual shutter; otherwise lock the current auto value.
        if isAutoShutter, d.isExposureModeSupported(.locked) { d.exposureMode = .locked }
        d.unlockForConfiguration()
    }

    /// Restores light-driven continuous behavior after the burst.
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
        photoOutput.capturePhoto(with: settings, delegate: self)
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
                                message: success ? "Saved 48 MP DNG to Photos" : "Saved to app storage")
                }
                self.cleanupBurstDir(burstDir, keep: url)
            })
        }
    }

    /// Removes intermediate burst RAWs and analysis cache; keeps the output DNG.
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
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraModel: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        if let error = error {
            DispatchQueue.main.async { self.statusText = "Capture error: \(error.localizedDescription)" }
        } else if photo.isRawPhoto, let data = photo.fileDataRepresentation(),
                  let dir = burstDir {
            let idx = capturedDNGs.count
            let url = dir.appendingPathComponent("frame_\(idx).dng")
            do {
                try data.write(to: url)
                capturedDNGs.append(url)
            } catch {
                DispatchQueue.main.async { self.statusText = "Write error: \(error.localizedDescription)" }
            }
        }

        pendingCaptures -= 1
        DispatchQueue.main.async {
            let done = self.burstCount - self.pendingCaptures
            self.progress = Float(done) / Float(self.burstCount) * 0.15  // capture ~ first 15%
        }

        if pendingCaptures > 0 {
            // Fire the next RAW immediately — as fast as the sensor allows. Settings
            // are locked for the burst, so there's no metering delay between frames.
            sessionQueue.async { self.captureNextRaw() }
        } else {
            unlockAfterBurst()
            // All frames are on disk — hand off to the processing queue.
            processingQueue.async { [weak self] in
                self?.processBurst()
            }
        }
    }
}
