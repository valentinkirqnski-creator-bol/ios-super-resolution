import SwiftUI

struct CameraView: View {
    @StateObject private var cam = CameraModel()
    @Environment(\.scenePhase) private var scenePhase
    @State private var showViewer = false
    @State private var focusPoint: CGPoint?
    @State private var focusVisible = false

    var body: some View {
        GeometryReader { geo in
            let bottomH: CGFloat = 96
            let topH: CGFloat = 72
            let vfWidth = geo.size.width
            let maxVFHeight = geo.size.height - topH - bottomH - geo.safeAreaInsets.bottom
            // Slightly taller than square (4:3) — uses more screen without ultra-wide chrome.
            let vfHeight = min(maxVFHeight, vfWidth * 4 / 3)

            ZStack {
                Color.black.ignoresSafeArea()

                if cam.permissionDenied {
                    permissionView
                } else {
                    VStack(spacing: 0) {
                        Color.clear.frame(height: topH)

                        viewfinder(width: vfWidth, height: vfHeight)

                        bottomPanel
                            .frame(height: bottomH + geo.safeAreaInsets.bottom)
                            .padding(.bottom, geo.safeAreaInsets.bottom)
                            .background(Color.black)
                    }
                }
            }
        }
        .onAppear { cam.start() }
        .onDisappear { cam.stop() }
        .onChange(of: scenePhase) { phase in
            cam.setAppActive(phase == .active)
        }
        .onChange(of: showViewer) { open in
            cam.setPreviewSuspended(open)
        }
        .sheet(isPresented: $showViewer) { resultViewer }
        .overlay(alignment: .top) {
            if !cam.permissionDenied {
                topStrip
                    .padding(.top, 6)
                    .background(
                        LinearGradient(
                            colors: [.black.opacity(0.55), .clear],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                        .ignoresSafeArea(edges: .top)
                        .allowsHitTesting(false)
                    )
            }
        }
    }

    // MARK: - Viewfinder

    private func viewfinder(width: CGFloat, height: CGFloat) -> some View {
        ZStack {
            CameraPreview(
                session: cam.session,
                mirrorFront: cam.cameraSelection == .front
            ) { devicePoint, localPoint in
                guard !cam.isBusy else { return }
                cam.focus(at: devicePoint)
                showFocusIndicator(at: localPoint)
            }
            .frame(width: width, height: height)
            .clipped()

            if cam.isProcessing {
                Color.black.opacity(0.08)
                    .frame(width: width, height: height)
                    .allowsHitTesting(false)
            }

            if focusVisible, let p = focusPoint {
                FocusIndicator()
                    .position(p)
                    .allowsHitTesting(false)
            }

            VStack {
                Spacer()
                if cam.cameraSelection != .front {
                    backLensPicker
                        .padding(.bottom, 14)
                }
            }
            .frame(width: width, height: height)
        }
        .frame(width: width, height: height)
        .background(Color.black)
    }

    private func showFocusIndicator(at point: CGPoint) {
        focusPoint = point
        withAnimation(.easeOut(duration: 0.12)) { focusVisible = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            withAnimation(.easeOut(duration: 0.25)) { focusVisible = false }
        }
    }

    // MARK: - Top strip

    private var topStrip: some View {
        VStack(spacing: 8) {
            HStack(spacing: 16) {
                frameCountControl
                Spacer()
                Text(cam.shutterLabel)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundColor(.white.opacity(0.85))
                    .frame(minWidth: 44, alignment: .trailing)
            }
            shutterSliderRow
        }
        .padding(.horizontal, 20)
    }

    private var shutterSliderRow: some View {
        HStack(spacing: 10) {
            Button {
                cam.setShutterAuto(true)
            } label: {
                Text("A")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundColor(cam.shutterIsAuto ? .black : .white)
                    .frame(width: 28, height: 28)
                    .background(cam.shutterIsAuto ? Color.yellow : Color.white.opacity(0.15))
                    .clipShape(Circle())
            }
            .disabled(cam.isBusy)

            Slider(
                value: Binding(
                    get: { cam.shutterSlider },
                    set: { cam.shutterSlider = $0 }
                ),
                in: 0...1
            ) { editing in
                if !editing { cam.applyManualShutterFromSlider() }
            }
            .tint(.yellow)
            .disabled(cam.isBusy || cam.shutterIsAuto)
            .opacity(cam.shutterIsAuto ? 0.35 : 1)
        }
    }

    private var frameCountControl: some View {
        HStack(spacing: 2) {
            miniStepper("minus", enabled: cam.frameCount > CameraModel.minFrameCount && !cam.isBusy) {
                cam.frameCount -= 1
            }
            Text("\(cam.frameCount)")
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundColor(.white)
                .frame(minWidth: 18)
            Text("frames")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.white.opacity(0.45))
            miniStepper("plus", enabled: cam.frameCount < CameraModel.maxFrameCount && !cam.isBusy) {
                cam.frameCount += 1
            }
        }
    }

    private func miniStepper(_ symbol: String, enabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(enabled ? .white : .white.opacity(0.25))
                .frame(width: 26, height: 26)
        }
        .disabled(!enabled)
    }

    // MARK: - Lens picker (over viewfinder)

    private var backLensPicker: some View {
        HStack(spacing: 18) {
            if cam.availableCameras.contains(.ultraWide) {
                lensChip(title: "0.5×", selected: cam.cameraSelection == .ultraWide) {
                    cam.setCamera(.ultraWide)
                }
            }
            if cam.availableCameras.contains(.wide) {
                lensChip(title: "1×", selected: cam.cameraSelection == .wide) {
                    cam.setCamera(.wide)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Color.black.opacity(0.45))
        .clipShape(Capsule())
    }

    private func lensChip(title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 14, weight: selected ? .bold : .medium))
                .foregroundColor(selected ? .yellow : .white.opacity(0.85))
                .frame(minWidth: 36)
        }
        .disabled(cam.isBusy)
    }

    // MARK: - Bottom panel (Apple-style)

    private var bottomPanel: some View {
        VStack(spacing: 0) {
            if cam.isProcessing, !cam.statusText.isEmpty {
                Text(cam.statusText)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white.opacity(0.5))
                    .lineLimit(1)
                    .padding(.bottom, 6)
            }

            HStack(alignment: .center) {
                flipCameraButton
                    .frame(width: 72)

                Spacer()

                shutterButton

                Spacer()

                galleryButton
                    .frame(width: 72)
            }
            .padding(.horizontal, 28)
        }
        .frame(maxHeight: .infinity, alignment: .top)
    }

    private var flipCameraButton: some View {
        Button(action: { cam.toggleFrontCamera() }) {
            Image(systemName: "arrow.triangle.2.circlepath.camera")
                .font(.system(size: 22, weight: .regular))
                .foregroundColor(cam.availableCameras.contains(.front) ? .white : .white.opacity(0.2))
        }
        .disabled(cam.isBusy || !cam.availableCameras.contains(.front))
    }

    private var shutterButton: some View {
        Button(action: { cam.captureBurst() }) {
            ZStack {
                Circle()
                    .strokeBorder(Color.white, lineWidth: 4)
                    .frame(width: 74, height: 74)
                Circle()
                    .fill(Color.white.opacity(cam.isBusy ? 0.35 : 1))
                    .frame(width: cam.isCapturing ? 56 : 62, height: cam.isCapturing ? 56 : 62)
                    .animation(.easeInOut(duration: 0.12), value: cam.isCapturing)
            }
        }
        .disabled(cam.isBusy)
    }

    private var galleryButton: some View {
        Button(action: { if cam.lastSavedURL != nil && !cam.isBusy { showViewer = true } }) {
            ZStack {
                Group {
                    if let thumb = cam.lastThumbnail {
                        Image(uiImage: thumb).resizable().scaledToFill()
                    } else {
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.white.opacity(0.1))
                            .overlay(
                                Image(systemName: "photo")
                                    .font(.system(size: 18, weight: .light))
                                    .foregroundColor(.white.opacity(0.5))
                            )
                    }
                }
                .frame(width: 46, height: 46)
                .clipShape(RoundedRectangle(cornerRadius: 8))

                if cam.isBusy {
                    Circle()
                        .stroke(Color.white.opacity(0.25), lineWidth: 2)
                        .frame(width: 54, height: 54)
                    Circle()
                        .trim(from: 0, to: CGFloat(max(0.02, Double(cam.progress))))
                        .stroke(Color.white, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                        .frame(width: 54, height: 54)
                        .rotationEffect(.degrees(-90))
                }
            }
        }
        .disabled(cam.isBusy && cam.lastSavedURL == nil)
    }

    // MARK: - Sheets

    private var resultViewer: some View {
        NavigationView {
            ZStack {
                Color.black.ignoresSafeArea()
                if let thumb = cam.lastThumbnail {
                    Image(uiImage: thumb).resizable().scaledToFit().padding()
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

// MARK: - Focus reticle

private struct FocusIndicator: View {
    @State private var scale: CGFloat = 1.35

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 2)
                .stroke(Color.yellow, lineWidth: 1.5)
                .frame(width: 72, height: 72)
            RoundedRectangle(cornerRadius: 1)
                .stroke(Color.yellow.opacity(0.5), lineWidth: 1)
                .frame(width: 6, height: 6)
        }
        .scaleEffect(scale)
        .onAppear {
            withAnimation(.spring(response: 0.28, dampingFraction: 0.62)) {
                scale = 1.0
            }
        }
    }
}
