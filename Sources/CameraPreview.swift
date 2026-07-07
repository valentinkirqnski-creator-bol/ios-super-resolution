import SwiftUI
import AVFoundation

/// Wraps an AVCaptureVideoPreviewLayer and reports tap-to-focus points
/// (already converted to normalized device coordinates 0..1).
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var mirrorFront: Bool = false
    var onFocus: (CGPoint) -> Void

    func makeUIView(context: Context) -> PreviewUIView {
        let v = PreviewUIView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        v.onFocus = onFocus
        applyMirroring(to: v)
        return v
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        uiView.onFocus = onFocus
        applyMirroring(to: uiView)
    }

    private func applyMirroring(to view: PreviewUIView) {
        guard let conn = view.videoPreviewLayer.connection, conn.isVideoMirroringSupported else { return }
        conn.automaticallyAdjustsVideoMirroring = !mirrorFront
        conn.isVideoMirrored = mirrorFront
    }

    final class PreviewUIView: UIView {
        var onFocus: ((CGPoint) -> Void)?

        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        override init(frame: CGRect) {
            super.init(frame: frame)
            let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            addGestureRecognizer(tap)
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

        @objc private func handleTap(_ g: UITapGestureRecognizer) {
            let p = g.location(in: self)
            let devicePoint = videoPreviewLayer.captureDevicePointConverted(fromLayerPoint: p)
            onFocus?(devicePoint)
        }
    }
}
