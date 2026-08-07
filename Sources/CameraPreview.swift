import SwiftUI
import AVFoundation

/// Square-clipped live preview; reports device + view-local tap points for focus.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var mirrorFront: Bool = false
    /// Digital zoom, in units of the active lens's own framing.
    ///
    /// Applied to the preview layer rather than to this view, so the view's
    /// frame -- and therefore its hit region -- never grows. Scaling the view
    /// itself made the magnified preview swallow taps meant for the controls
    /// laid out around it: SwiftUI's .clipped() clips rendering, not hit
    /// testing, so at 2x the overflow covered the settings button.
    var zoom: CGFloat = 1
    var zoomDuration: Double = 0
    var onFocusTap: (CGPoint, CGPoint) -> Void

    func makeUIView(context: Context) -> PreviewUIView {
        let v = PreviewUIView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        v.onFocusTap = onFocusTap
        v.setZoom(zoom, duration: 0)
        applyMirroring(to: v)
        return v
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        if uiView.videoPreviewLayer.session !== session {
            uiView.videoPreviewLayer.session = session
        }
        uiView.onFocusTap = onFocusTap
        uiView.setZoom(zoom, duration: zoomDuration)
        applyMirroring(to: uiView)
    }

    private func applyMirroring(to view: PreviewUIView) {
        guard let conn = view.videoPreviewLayer.connection, conn.isVideoMirroringSupported else { return }
        if mirrorFront {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = true
        } else {
            conn.automaticallyAdjustsVideoMirroring = true
        }
    }

    final class PreviewUIView: UIView {
        var onFocusTap: ((CGPoint, CGPoint) -> Void)?

        /// Held as a sublayer rather than via layerClass so it can carry a scale
        /// transform independently of the view, and be clipped by it.
        let videoPreviewLayer = AVCaptureVideoPreviewLayer()
        private var appliedZoom: CGFloat = 1

        override init(frame: CGRect) {
            super.init(frame: frame)
            backgroundColor = .black
            layer.masksToBounds = true
            layer.addSublayer(videoPreviewLayer)
            let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            addGestureRecognizer(tap)
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

        override func layoutSubviews() {
            super.layoutSubviews()
            // bounds+position rather than frame: frame is derived from the
            // transform, so assigning it while zoomed would fight the scale.
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            videoPreviewLayer.bounds = CGRect(origin: .zero, size: bounds.size)
            videoPreviewLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
            CATransaction.commit()
        }

        func setZoom(_ z: CGFloat, duration: Double) {
            let clamped = max(1, z)
            guard abs(clamped - appliedZoom) > 0.0001 else { return }
            appliedZoom = clamped
            CATransaction.begin()
            CATransaction.setAnimationDuration(max(0, duration))
            CATransaction.setDisableActions(duration <= 0)
            videoPreviewLayer.transform = CATransform3DMakeScale(clamped, clamped, 1)
            CATransaction.commit()
        }

        @objc private func handleTap(_ g: UITapGestureRecognizer) {
            let p = g.location(in: self)
            // Undo the layer's scale so the tap lands on the sensor point the
            // finger is actually over; convert(_:from:) inverts the transform.
            let lp = videoPreviewLayer.convert(p, from: layer)
            let devicePoint = videoPreviewLayer.captureDevicePointConverted(fromLayerPoint: lp)
            onFocusTap?(devicePoint, p)
        }
    }
}
