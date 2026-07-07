import SwiftUI

struct CameraView: View {
    @StateObject private var cam = CameraModel()
    @State private var showViewer = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if cam.permissionDenied {
                permissionView
            } else {
                CameraPreview(session: cam.session, mirrorFront: cam.cameraSelection == .front) { devicePoint in
                    guard !cam.isBusy else { return }
                    cam.focus(at: devicePoint)
                }
                .ignoresSafeArea()

                if cam.isProcessing {
                    Color.black.opacity(0.12).ignoresSafeArea().allowsHitTesting(false)
                }

                vignette

                VStack(spacing: 0) {
                    topBar
                        .padding(.top, 8)
                    shutterPicker
                        .padding(.top, 10)
                    Spacer()
                    if cam.isProcessing, !cam.statusText.isEmpty {
                        Text(cam.statusText)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(.white.opacity(0.75))
                            .lineLimit(1)
                            .padding(.horizontal, 16)
                            .padding(.bottom, 10)
                    }
                    bottomBar
                        .padding(.bottom, 34)
                }
            }
        }
        .onAppear { cam.start() }
        .onDisappear { cam.stop() }
        .sheet(isPresented: $showViewer) { resultViewer }
    }

    // MARK: - Chrome

    private var vignette: some View {
        VStack {
            LinearGradient(colors: [.black.opacity(0.55), .clear], startPoint: .top, endPoint: .bottom)
                .frame(height: 120)
            Spacer()
            LinearGradient(colors: [.clear, .black.opacity(0.65)], startPoint: .top, endPoint: .bottom)
                .frame(height: 200)
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }

    private var topBar: some View {
        HStack(alignment: .center, spacing: 12) {
            frameCountControl
            Spacer(minLength: 8)
            cameraSwitcher
        }
        .padding(.horizontal, 20)
    }

    private var frameCountControl: some View {
        HStack(spacing: 0) {
            stepperButton(systemName: "minus", enabled: cam.frameCount > 2 && !cam.isBusy) {
                cam.frameCount -= 1
            }
            Text("\(cam.frameCount)")
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundColor(.white)
                .frame(minWidth: 22)
            stepperButton(systemName: "plus", enabled: cam.frameCount < 8 && !cam.isBusy) {
                cam.frameCount += 1
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial.opacity(0.85))
        .clipShape(Capsule())
        .overlay(
            Capsule().stroke(Color.white.opacity(0.15), lineWidth: 0.5)
        )
        .overlay(alignment: .bottom) {
            Text("frames")
                .font(.system(size: 9, weight: .medium))
                .foregroundColor(.white.opacity(0.45))
                .offset(y: 18)
        }
        .padding(.bottom, 10)
    }

    private func stepperButton(systemName: String, enabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(enabled ? .white : .white.opacity(0.25))
                .frame(width: 28, height: 28)
                .contentShape(Rectangle())
        }
        .disabled(!enabled)
    }

    private var shutterPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(ShutterSetting.choices) { s in
                    Button { cam.shutter = s } label: {
                        Text(s.label)
                            .font(.system(size: 13, weight: .medium, design: .monospaced))
                            .padding(.horizontal, 11).padding(.vertical, 6)
                            .background(cam.shutter == s ? Color.white : Color.white.opacity(0.12))
                            .foregroundColor(cam.shutter == s ? .black : .white.opacity(0.9))
                            .clipShape(Capsule())
                    }
                    .disabled(cam.isBusy)
                }
            }
            .padding(.horizontal, 20)
        }
    }

    private var bottomBar: some View {
        HStack(alignment: .center) {
            Color.clear.frame(width: 72, height: 72)

            Spacer()

            shutterButton

            Spacer()

            galleryButton
                .frame(width: 72, height: 72)
        }
        .padding(.horizontal, 24)
    }

    private var shutterButton: some View {
        Button(action: { cam.captureBurst() }) {
            ZStack {
                Circle()
                    .strokeBorder(Color.white.opacity(0.9), lineWidth: 3)
                    .frame(width: 76, height: 76)
                Circle()
                    .fill(cam.isBusy ? Color.white.opacity(0.35) : Color.white)
                    .frame(width: cam.isCapturing ? 58 : 64, height: cam.isCapturing ? 58 : 64)
                    .animation(.easeInOut(duration: 0.15), value: cam.isCapturing)
            }
        }
        .disabled(cam.isBusy)
        .accessibilityLabel("Capture burst")
    }

    private var galleryButton: some View {
        Button(action: { if cam.lastSavedURL != nil && !cam.isBusy { showViewer = true } }) {
            ZStack {
                Group {
                    if let thumb = cam.lastThumbnail {
                        Image(uiImage: thumb).resizable().scaledToFill()
                    } else {
                        ZStack {
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.white.opacity(0.08))
                            Image(systemName: "photo")
                                .font(.system(size: 20, weight: .light))
                                .foregroundColor(.white.opacity(0.55))
                        }
                    }
                }
                .frame(width: 50, height: 50)
                .clipShape(RoundedRectangle(cornerRadius: 12))

                if cam.isBusy {
                    Circle()
                        .stroke(Color.white.opacity(0.2), lineWidth: 2.5)
                        .frame(width: 58, height: 58)
                    Circle()
                        .trim(from: 0, to: CGFloat(max(0.02, Double(cam.progress))))
                        .stroke(Color.white, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                        .frame(width: 58, height: 58)
                        .rotationEffect(.degrees(-90))
                        .animation(.linear(duration: 0.12), value: cam.progress)
                } else {
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.white.opacity(0.35), lineWidth: 1)
                        .frame(width: 54, height: 54)
                }
            }
        }
        .disabled(cam.isBusy && cam.lastSavedURL == nil)
    }

    private var cameraSwitcher: some View {
        HStack(spacing: 4) {
            ForEach(cam.availableCameras) { lens in
                Button { cam.setCamera(lens) } label: {
                    Text(lens.label)
                        .font(.system(size: 13, weight: .semibold))
                        .padding(.horizontal, 12).padding(.vertical, 7)
                        .background(cam.cameraSelection == lens ? Color.white : Color.clear)
                        .foregroundColor(cam.cameraSelection == lens ? .black : .white.opacity(0.85))
                        .clipShape(Capsule())
                }
                .disabled(cam.isBusy)
            }
        }
        .padding(4)
        .background(.ultraThinMaterial.opacity(0.85))
        .clipShape(Capsule())
        .overlay(Capsule().stroke(Color.white.opacity(0.12), lineWidth: 0.5))
    }

    // MARK: - Sheets

    private var resultViewer: some View {
        NavigationView {
            ZStack {
                Color.black.ignoresSafeArea()
                if let thumb = cam.lastThumbnail {
                    Image(uiImage: thumb)
                        .resizable()
                        .scaledToFit()
                        .padding()
                }
            }
            .navigationTitle("Last capture")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { showViewer = false }
                }
            }
        }
        .navigationViewStyle(.stack)
        .preferredColorScheme(.dark)
    }

    private var permissionView: some View {
        VStack(spacing: 20) {
            Image(systemName: "camera.fill")
                .font(.system(size: 44, weight: .light))
                .foregroundColor(.white.opacity(0.8))
            Text("Camera access is required")
                .font(.headline)
                .foregroundColor(.white)
            Button("Open Settings") {
                if let u = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(u)
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(.white)
        }
    }
}
