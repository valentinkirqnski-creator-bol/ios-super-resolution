import SwiftUI

struct CameraView: View {
    @StateObject private var cam = CameraModel()
    @State private var focusIndicator: CGPoint?
    @State private var showViewer = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if cam.permissionDenied {
                permissionView
            } else {
                CameraPreview(session: cam.session, mirrorFront: cam.cameraSelection == .front) { devicePoint in
                    cam.focus(at: devicePoint)
                }
                .ignoresSafeArea()
                // Focus reticle uses a screen-space overlay via a tap capture layer.
                .overlay(focusReticle, alignment: .topLeading)

                VStack {
                    shutterPicker
                        .padding(.top, 12)
                    Spacer()
                    if cam.isBusy { processingBar }
                    bottomBar
                        .padding(.bottom, 28)
                }
            }
        }
        .onAppear { cam.start() }
        .onDisappear { cam.stop() }
        .sheet(isPresented: $showViewer) { resultViewer }
    }

    // MARK: - Top: shutter speed selector

    private var shutterPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(ShutterSetting.choices) { s in
                    Button {
                        cam.shutter = s
                    } label: {
                        Text(s.label)
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .padding(.horizontal, 12).padding(.vertical, 7)
                            .background(cam.shutter == s ? Color.yellow : Color.white.opacity(0.18))
                            .foregroundColor(cam.shutter == s ? .black : .white)
                            .clipShape(Capsule())
                    }
                }
            }
            .padding(.horizontal, 16)
        }
    }

    // MARK: - Bottom: gallery + shutter + spacer

    private var bottomBar: some View {
        ZStack {
            // Center shutter button.
            Button(action: { cam.captureBurst() }) {
                ZStack {
                    Circle().stroke(Color.white, lineWidth: 5).frame(width: 78, height: 78)
                    Circle().fill(cam.isBusy ? Color.gray : Color.white).frame(width: 64, height: 64)
                }
            }
            .disabled(cam.isBusy)

            HStack {
                // Lens switcher (left).
                cameraSwitcher
                    .padding(.leading, 20)

                Spacer()

                // Gallery thumbnail (right).
                Button(action: { if cam.lastSavedURL != nil { showViewer = true } }) {
                    Group {
                        if let thumb = cam.lastThumbnail {
                            Image(uiImage: thumb).resizable().scaledToFill()
                        } else {
                            Image(systemName: "photo.on.rectangle")
                                .font(.system(size: 22)).foregroundColor(.white.opacity(0.7))
                        }
                    }
                    .frame(width: 54, height: 54)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.6), lineWidth: 1))
                }
                .padding(.trailing, 28)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var cameraSwitcher: some View {
        HStack(spacing: 6) {
            ForEach(cam.availableCameras) { lens in
                Button {
                    cam.setCamera(lens)
                } label: {
                    Text(lens.label)
                        .font(.system(size: 12, weight: .semibold))
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(cam.cameraSelection == lens ? Color.yellow : Color.white.opacity(0.18))
                        .foregroundColor(cam.cameraSelection == lens ? .black : .white)
                        .clipShape(Capsule())
                }
                .disabled(cam.isBusy)
            }
        }
    }

    // MARK: - Processing overlay

    private var processingBar: some View {
        VStack(spacing: 8) {
            ProgressView(value: Double(cam.progress))
                .progressViewStyle(.linear)
                .tint(.yellow)
                .frame(width: 260)
            Text(cam.statusText)
                .font(.caption).foregroundColor(.white)
        }
        .padding(.vertical, 14).padding(.horizontal, 20)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .padding(.bottom, 20)
    }

    // MARK: - Focus reticle

    private var focusReticle: some View {
        GeometryReader { _ in
            if let p = focusIndicator {
                Rectangle()
                    .stroke(Color.yellow, lineWidth: 1.5)
                    .frame(width: 72, height: 72)
                    .position(p)
            }
        }
        .allowsHitTesting(false)
    }

    // MARK: - Result viewer

    private var resultViewer: some View {
        VStack {
            if let thumb = cam.lastThumbnail {
                Image(uiImage: thumb).resizable().scaledToFit()
            }
            Text(cam.statusText).font(.footnote).padding()
            Button("Close") { showViewer = false }.padding()
        }
        .padding()
    }

    private var permissionView: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill").font(.system(size: 48)).foregroundColor(.white)
            Text("Camera access is required").foregroundColor(.white)
            Button("Open Settings") {
                if let u = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(u) }
            }
            .buttonStyle(.borderedProminent)
        }
    }
}
