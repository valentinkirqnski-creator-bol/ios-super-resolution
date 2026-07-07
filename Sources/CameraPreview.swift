import SwiftUI
import AVFoundation

/// Square-clipped live preview; reports device + view-local tap points for focus.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var mirrorFront: Bool = false
    var onFocusTap: (CGPoint, CGPoint) -> Void

    func makeUIView(context: Context) -> PreviewUIView {
        let v = PreviewUIView()
        v.videoPreviewLayer.session = session
        v.videoPreviewLayer.videoGravity = .resizeAspectFill
        v.onFocusTap = onFocusTap
        applyMirroring(to: v)
        return v
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        uiView.onFocusTap = onFocusTap
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

        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        override init(frame: CGRect) {
            super.init(frame: frame)
            backgroundColor = .black
            let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            addGestureRecognizer(tap)
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

        @objc private func handleTap(_ g: UITapGestureRecognizer) {
            let p = g.location(in: self)
            let devicePoint = videoPreviewLayer.captureDevicePointConverted(fromLayerPoint: p)
            onFocusTap?(devicePoint, p)
        }
    }
}
