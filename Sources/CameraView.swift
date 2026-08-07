import SwiftUI
import UniformTypeIdentifiers

struct CameraView: View {
    @StateObject private var cam = CameraModel()
    @Environment(\.scenePhase) private var scenePhase
    @State private var showViewer = false
    @State private var showSettings = false
    @State private var pinchBaseZoom: CGFloat?
    @State private var focusPoint: CGPoint?
    @State private var focusVisible = false
    /// True only while the user is actively dragging the shutter slider.
    @State private var didApplyLaunchShutter = false
    /// Manual capture controls (burst length, shutter) shown above the
    /// viewfinder. Collapsible so the preview can fill the screen.
    @State private var showImporter = false
    @State private var showGallery = false

    var body: some View {
        GeometryReader { geo in
            let topBarH: CGFloat = 88
            let bottomH: CGFloat = 96
            let vfWidth = geo.size.width
            // One value for the space below the controls, used both to reserve
            // it and to apply it. They were computed separately before, and the
            // panel ended up claiming the safe-area inset twice -- once in its
            // frame and again as padding -- while the viewfinder only reserved
            // it once, so the whole stack overflowed by an inset and the shutter
            // ran under the home indicator. The extra 10 lifts it clear rather
            // than merely flush.
            let bottomInset = geo.safeAreaInsets.bottom + 10
            let maxVFHeight = geo.size.height - topBarH - bottomH - bottomInset
            // Slightly taller than square (4:3) — uses more screen without ultra-wide chrome.
            let vfHeight = min(maxVFHeight, vfWidth * 4 / 3)

            ZStack {
                Color.black.ignoresSafeArea()

                if cam.permissionDenied {
                    permissionView
                } else {
                    VStack(spacing: 0) {
                        topStrip
                            .padding(.top, geo.safeAreaInsets.top + 4)
                            .frame(height: topBarH + geo.safeAreaInsets.top)
                            .background(Color.black)

                        viewfinder(width: vfWidth, height: vfHeight)

                        bottomPanel
                            .frame(height: bottomH)
                            .padding(.bottom, bottomInset)
                            .background(Color.black)
                    }
                }
            }
        }
        .onAppear {
            if !didApplyLaunchShutter {
                cam.ensureShutterAutoOnLaunch()
                didApplyLaunchShutter = true
            }
            cam.start()
        }
        .onDisappear { cam.stop() }
        .onChange(of: scenePhase) { phase in
            cam.setAppActive(phase == .active)
        }
        .onChange(of: showViewer) { open in
            cam.setPreviewSuspended(open)
        }
        .sheet(isPresented: $showViewer) { resultViewer }
        .sheet(isPresented: $showGallery) { GalleryView() }
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: [.image],
                      allowsMultipleSelection: true) { result in
            if case .success(let urls) = result {
                // Picked files sit outside the sandbox and the pipeline reads
                // them on a background queue, so the security scope must stay
                // open past this closure; processImportedDNGs closes it.
                cam.processImportedDNGs(urls.filter { $0.startAccessingSecurityScopedResource() })
            }
        }
        .sheet(isPresented: $showSettings) { tuningSettingsView }
    }

    // MARK: - Viewfinder

    private func viewfinder(width: CGFloat, height: CGFloat) -> some View {
        ZStack {
            // 2x is a processing-side centre crop, so the preview is zoomed to
            // match the framing that will actually be saved. The zoom is applied
            // to the preview layer inside CameraPreview, not with .scaleEffect
            // here: .clipped() clips rendering but not hit testing, so scaling
            // the view made the magnified preview swallow taps on the settings
            // button and the exposure sliders.
            CameraPreview(
                session: cam.session,
                mirrorFront: cam.cameraSelection == .front,
                zoom: cam.previewZoom,
                zoomDuration: CameraModel.lensZoomDuration
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

            thirdsGrid
                .frame(width: width, height: height)
                .allowsHitTesting(false)

            // Exposure controls sit on the frame edges rather than in a bar
            // above it, so the preview keeps the full height.
            HStack {
                edgeSlider(value: $cam.isoSlider,
                           symbol: "circle.lefthalf.fill",   // .filled variant is iOS 16+
                           active: !cam.isoIsAuto,
                           height: height * 0.42,
                           toggle: { cam.isoIsAuto.toggle() },
                           goManual: { cam.isoIsAuto = false })
                Spacer()
                edgeSlider(value: $cam.shutterSlider,
                           symbol: "sun.max",
                           active: !cam.shutterIsAuto,
                           height: height * 0.42,
                           toggle: { cam.setShutterAuto(!cam.shutterIsAuto) },
                           goManual: { cam.applyManualShutterFromSlider() })
            }
            .padding(.horizontal, 14)
            .frame(width: width, height: height)

            VStack {
                Spacer()
                if cam.cameraSelection != .front {
                    // One or the other, never both: the slider is the zoomed-in
                    // form of the same control, as on the reference UI.
                    if cam.zoomUIVisible {
                        zoomSlider(width: width)
                            .padding(.bottom, 14)
                            .transition(.opacity)
                    } else {
                        backLensPicker
                            .padding(.bottom, 14)
                            .transition(.opacity)
                    }
                }
            }
            .frame(width: width, height: height)
        }
        .frame(width: width, height: height)
        // simultaneousGesture, not gesture: the preview carries its own UIKit
        // tap recogniser for focus, and claiming the gesture outright here
        // would stop taps reaching it.
        .simultaneousGesture(
            MagnificationGesture()
                .onChanged { v in
                    guard !cam.isBusy else { return }
                    let base = pinchBaseZoom ?? cam.zoomFactor
                    if pinchBaseZoom == nil { pinchBaseZoom = base }
                    cam.setZoom(base * v)
                    cam.showZoomUI()
                }
                .onEnded { _ in
                    pinchBaseZoom = nil
                    cam.showZoomUI()
                }
        )
        .background(Color.black)
    }

    private var thirdsGrid: some View {
        GeometryReader { g in
            Path { p in
                for i in 1...2 {
                    let x = g.size.width * CGFloat(i) / 3
                    p.move(to: CGPoint(x: x, y: 0)); p.addLine(to: CGPoint(x: x, y: g.size.height))
                    let y = g.size.height * CGFloat(i) / 3
                    p.move(to: CGPoint(x: 0, y: y)); p.addLine(to: CGPoint(x: g.size.width, y: y))
                }
            }
            .stroke(Color.white.opacity(0.28), lineWidth: 0.5)
        }
    }

    /// Vertical track with a round icon handle that rides it. Tapping the
    /// handle toggles the control between auto and manual; dragging anywhere on
    /// the track sets the value, so the handle is not a small hit target.
    private func edgeSlider(value: Binding<Double>,
                            symbol: String,
                            active: Bool,
                            height: CGFloat,
                            toggle: @escaping () -> Void,
                            goManual: @escaping () -> Void) -> some View {
        GeometryReader { g in
            let h = g.size.height
            let knob: CGFloat = 34
            let travel = max(1, h - knob)
            // Top of the track is the high value, as on a physical fader.
            let y = knob / 2 + travel * CGFloat(1 - value.wrappedValue)
            ZStack(alignment: .top) {
                Capsule()
                    .fill(Color.white.opacity(active ? 0.85 : 0.35))
                    .frame(width: 2, height: h)
                    .frame(maxWidth: .infinity)
                ZStack {
                    Circle().fill(Color.black.opacity(0.55)).frame(width: knob, height: knob)
                    Circle().strokeBorder(Color.white.opacity(0.9), lineWidth: 1.5)
                        .frame(width: knob, height: knob)
                    Image(systemName: symbol)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundColor(active ? .white : .white.opacity(0.55))
                }
                .position(x: g.size.width / 2, y: y)
            }
            .contentShape(Rectangle())
            // One gesture handles both roles. The knob used to be a Button,
            // which swallowed any drag beginning on it -- and the knob is
            // exactly where a slider gets grabbed, so dragging did nothing.
            // minimumDistance 0 also means the value tracks the very first
            // touch instead of only after 4pt of travel.
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { v in
                        guard !cam.isBusy else { return }
                        // Below the tap threshold this may still turn out to be
                        // a tap, so do not move the value yet.
                        guard hypot(v.translation.width, v.translation.height) > 4 else { return }
                        let t = 1 - (v.location.y - knob / 2) / travel
                        value.wrappedValue = min(1, max(0, Double(t)))
                        // Dragging leaves Auto, as the previous slider did --
                        // otherwise the handle has to be tapped first and the
                        // drag silently does nothing.
                        if !active { goManual() }
                    }
                    .onEnded { v in
                        guard !cam.isBusy else { return }
                        if hypot(v.translation.width, v.translation.height) <= 4 {
                            toggle()
                        }
                    }
            )
        }
        .frame(width: 44, height: height)
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
            HStack(spacing: 14) {
                roundIconButton("gearshape.fill") { showSettings = true }
                // A stack of frames collapsing into one output is the closest
                // symbol to "merge several files into a larger image".
                roundIconButton("square.stack.3d.down.right.fill") { showImporter = true }
                formatButton
                Spacer()
            }

            HStack(spacing: 16) {
                frameCountControl
                Spacer()
                Text("ISO " + cam.isoLabel)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundColor(.white.opacity(0.6))
                Text(cam.shutterLabel)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundColor(.white.opacity(0.85))
                    .frame(minWidth: 44, alignment: .trailing)
            }
        }
        .padding(.horizontal, 20)
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

    // MARK: - Zoom slider (over viewfinder)

    /// Magnifications that get a labelled, emphasised tick: each physical lens,
    /// plus 2x, plus whatever the crop limit works out to.
    private var zoomStops: [CGFloat] {
        var out: [CGFloat] = []
        if cam.availableCameras.contains(.ultraWide) { out.append(cam.ultraWideNativeZoom) }
        out.append(1)
        out.append(2)
        if cam.availableCameras.contains(.telephoto) { out.append(cam.telephotoNativeZoom) }
        out.append(cam.maxZoom)
        var uniq: [CGFloat] = []
        for z in out.sorted() where z >= cam.minZoom - 1e-4 && z <= cam.maxZoom + 1e-4 {
            if uniq.last.map({ abs($0 - z) > 0.05 }) ?? true { uniq.append(z) }
        }
        return uniq
    }

    private static func zoomLabel(_ z: CGFloat) -> String {
        z < 1 ? String(format: "%.1f×", Double(z))
              : (abs(z - z.rounded()) < 0.05 ? "\(Int(z.rounded()))×"
                                             : String(format: "%.1f×", Double(z)))
    }

    /// Log scale, so each doubling takes the same distance along the track.
    /// A linear one would bunch every useful magnification into the first
    /// tenth of the bar.
    private func zoomPosition(_ z: CGFloat) -> CGFloat {
        let lo = log(max(0.01, cam.minZoom))
        let hi = log(max(cam.minZoom * 1.01, cam.maxZoom))
        return min(1, max(0, (log(max(0.01, z)) - lo) / (hi - lo)))
    }

    private func zoomAt(_ t: CGFloat) -> CGFloat {
        let lo = log(max(0.01, cam.minZoom))
        let hi = log(max(cam.minZoom * 1.01, cam.maxZoom))
        return exp(lo + min(1, max(0, t)) * (hi - lo))
    }

    private func zoomSlider(width: CGFloat) -> some View {
        let trackW = max(120, width - 96)
        let ticks = 29
        let accent = Color(red: 0.62, green: 0.85, blue: 0.88)
        return VStack(spacing: 6) {
            Text(Self.zoomLabel(cam.zoomFactor))
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundColor(.black)
                .frame(width: 52, height: 34)
                .background(Circle().fill(accent))
                .offset(x: (zoomPosition(cam.zoomFactor) - 0.5) * trackW)

            ZStack {
                Capsule().fill(Color.black.opacity(0.55))
                HStack(spacing: 0) {
                    ForEach(0..<ticks, id: \.self) { i in
                        let t = CGFloat(i) / CGFloat(ticks - 1)
                        let onStop = zoomStops.contains {
                            abs(zoomPosition($0) - t) < 0.5 / CGFloat(ticks - 1)
                        }
                        let near = abs(zoomPosition(cam.zoomFactor) - t) < 0.5 / CGFloat(ticks - 1)
                        Circle()
                            .fill(near ? accent
                                       : (onStop ? accent.opacity(0.85) : Color.white.opacity(0.45)))
                            .frame(width: near ? 7 : (onStop ? 5 : 3),
                                   height: near ? 7 : (onStop ? 5 : 3))
                            .frame(maxWidth: .infinity)
                    }
                }
                .padding(.horizontal, 10)
            }
            .frame(width: trackW, height: 34)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { v in
                        guard !cam.isBusy else { return }
                        cam.setZoom(zoomAt(v.location.x / trackW))
                        cam.showZoomUI()
                    }
                    .onEnded { _ in cam.showZoomUI() }
            )

            ZStack(alignment: .topLeading) {
                ForEach(zoomStops, id: \.self) { z in
                    Text(Self.zoomLabel(z))
                        .font(.system(size: 10, weight: .medium, design: .rounded))
                        .foregroundColor(.white.opacity(0.7))
                        .position(x: zoomPosition(z) * trackW, y: 6)
                }
            }
            .frame(width: trackW, height: 12)
        }
        .frame(width: width)
    }

    // MARK: - Lens picker (over viewfinder)

    private var backLensPicker: some View {
        VStack(spacing: 8) {
            HStack(spacing: 6) {
                if cam.availableCameras.contains(.ultraWide) {
                    lensChip(title: "0.5×", selected: cam.isAtZoom(cam.ultraWideNativeZoom)) {
                        cam.setLensZoom(.ultraWide)
                    }
                }
                if cam.availableCameras.contains(.wide) {
                    lensChip(title: "1×", selected: cam.isAtZoom(1)) {
                        cam.setLensZoom(.wide1x)
                    }
                    lensChip(title: "2×", selected: cam.isAtZoom(2)) {
                        cam.setLensZoom(.wide2x)
                    }
                }
                if cam.availableCameras.contains(.telephoto) {
                    lensChip(title: cam.telephotoLensLabel,
                             selected: cam.isAtZoom(cam.telephotoNativeZoom)) {
                        cam.setLensZoom(.telephoto)
                    }
                }
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 5)
            .background(Capsule().fill(Color.black.opacity(0.45)))
        }
    }

    private func lensChip(title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: selected ? .bold : .medium))
                .foregroundColor(selected ? .black : .white.opacity(0.92))
                .frame(width: 38, height: 38)
                .background(
                    Circle().fill(selected
                                  ? Color(red: 0.86, green: 0.78, blue: 0.60)
                                  : Color.white.opacity(0.14))
                )
        }
        .disabled(cam.isBusy)
    }

    // MARK: - Bottom panel (Apple-style)

    private var bottomPanel: some View {
        VStack(spacing: 0) {
            if !cam.noiseDiagText.isEmpty, cam.isProcessing || !cam.isBusy {
                Text(cam.noiseDiagText)
                    .font(.system(size: 10, weight: .semibold).monospacedDigit())
                    .foregroundColor(cam.noiseDiagText.contains("FALLBACK")
                        ? .orange.opacity(0.95)
                        : .white.opacity(0.7))
                    .lineLimit(2)
                    .minimumScaleFactor(0.7)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 4)
            }
            if cam.isProcessing, !cam.statusText.isEmpty {
                Text(cam.statusText)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white.opacity(0.5))
                    .lineLimit(2)
                    .minimumScaleFactor(0.75)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 6)
            } else if !cam.statusText.isEmpty, !cam.isBusy {
                Text(cam.statusText)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white.opacity(0.5))
                    .lineLimit(2)
                    .minimumScaleFactor(0.75)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 6)
            }

            HStack(alignment: .center) {
                galleryButton
                    .frame(width: 72)

                Spacer()

                shutterButton

                Spacer()

                flipCameraButton
                    .frame(width: 72)
            }
            .padding(.horizontal, 28)

        }
        .frame(maxHeight: .infinity, alignment: .top)
    }

    private var flipCameraButton: some View {
        let enabled = cam.availableCameras.contains(.front) && !cam.isBusy
        return Button(action: { cam.toggleFrontCamera() }) {
            ZStack {
                Circle()
                    .strokeBorder(enabled ? Color.white : Color.white.opacity(0.25),
                                  lineWidth: 2)
                    .frame(width: 48, height: 48)
                Image(systemName: "arrow.triangle.2.circlepath")
                    .font(.system(size: 20, weight: .medium))
                    .foregroundColor(enabled ? .white : .white.opacity(0.25))
            }
        }
        .disabled(!enabled)
    }

    private var shutterButton: some View {
        Button(action: { cam.captureBurst() }) {
            ZStack {
                Circle()
                    .strokeBorder(Color.white.opacity(cam.isBusy ? 0.35 : 1), lineWidth: 5)
                    .frame(width: 78, height: 78)
                Circle()
                    .fill(shutterFill)
                    .frame(width: cam.isCapturing ? 50 : 58, height: cam.isCapturing ? 50 : 58)
                    .animation(.spring(response: 0.22, dampingFraction: 0.6), value: cam.isCapturing)
                if cam.isProcessing {
                    // Progress reads on the control the user is waiting on,
                    // rather than only in the status line above.
                    Circle()
                        .trim(from: 0, to: max(0.02, CGFloat(cam.progress)))
                        .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                        .frame(width: 78, height: 78)
                        .animation(.easeInOut(duration: 0.2), value: cam.progress)
                }
            }
        }
        .disabled(cam.isBusy)
    }

    /// Small circular control on a translucent disc, used for the two utility
    /// buttons that flank the shutter row.
    /// DNG/JPG selector, sat alongside the other capture controls above the
    /// frame count. Two states, so it toggles rather than opening a picker.
    private var formatButton: some View {
        Button {
            cam.exportFormat = (cam.exportFormat == .dng) ? .jpg : .dng
        } label: {
            ZStack {
                Capsule()
                    .fill(Color.white.opacity(0.12))
                    .frame(width: 58, height: 42)
                Text(cam.exportFormat.label)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)
            }
        }
        .disabled(cam.isBusy)
        .accessibilityLabel("Output format")
        .accessibilityValue(cam.exportFormat.label)
    }

    private func roundIconButton(_ symbol: String,
                                 action: @escaping () -> Void) -> some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.12))
                    .frame(width: 42, height: 42)
                Image(systemName: symbol)
                    .font(.system(size: 17, weight: .medium))
                    .foregroundColor(.white)
            }
        }
        .disabled(cam.isBusy)
    }

    private var shutterFill: Color {
        if cam.isProcessing { return Color.white.opacity(0.25) }
        if cam.isBusy { return Color.white.opacity(0.35) }
        return .white
    }

    private var galleryButton: some View {
        Button(action: { if !cam.isBusy { showGallery = true } }) {
            ZStack {
                Group {
                    if let thumb = cam.lastThumbnail {
                        Image(uiImage: thumb).resizable().scaledToFill()
                    } else {
                        Circle()
                            .fill(Color.white.opacity(0.1))
                            .overlay(
                                Image(systemName: "photo")
                                    .font(.system(size: 18, weight: .light))
                                    .foregroundColor(.white.opacity(0.5))
                            )
                    }
                }
                .frame(width: 46, height: 46)
                .clipShape(Circle())
                .overlay(Circle().strokeBorder(Color.white.opacity(0.55), lineWidth: 1.5))

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
        .disabled(cam.isBusy && cam.lastThumbnail == nil)
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

    private var tuningSettingsView: some View {
        NavigationView {
            Form {
                Section(header: Text("Robustness (Motion Rejection)")) {
                    HStack {
                        Text("Threshold (r_t)")
                        Spacer()
                        Text(String(format: "%.2f", cam.tuningParams.r_t))
                    }
                    Slider(value: $cam.tuningParams.r_t, in: 0.0...1.0)
                    
                    HStack {
                        Text("Penalty (r_s1)")
                        Spacer()
                        Text(String(format: "%.2f", cam.tuningParams.r_s1))
                    }
                    Slider(value: $cam.tuningParams.r_s1, in: 0.0...8.0)
                    
                    HStack {
                        Text("Multiplier (r_s2)")
                        Spacer()
                        Text(String(format: "%.1f", cam.tuningParams.r_s2))
                    }
                    Slider(value: $cam.tuningParams.r_s2, in: 1.0...50.0)
                    
                    HStack {
                        Text("Max Robustness (r_Mt)")
                        Spacer()
                        Text(String(format: "%.2f", cam.tuningParams.r_Mt))
                    }
                    Slider(value: $cam.tuningParams.r_Mt, in: 0.0...1.0)

                    Toggle("Alignment Grey: FFT", isOn: $cam.tuningParams.alignment_grey_fft)
                    Text(cam.tuningParams.alignment_grey_fft
                         ? "Full-res FFT low-pass. Slower."
                         : "2x2 Bayer quad average at half res (Wronski et al.). Much faster.")
                        .font(.caption2).foregroundColor(.secondary)
                    Toggle("HF Artifact Rejection", isOn: $cam.tuningParams.hf_artifact_removal_enabled)
                    Text("Rejects repetitive fine texture that block matching cannot align (aperture problem). Needs high-frequency content AND unstable flow, so hair and noise are spared.")
                        .font(.caption2).foregroundColor(.secondary)

                    if cam.tuningParams.hf_artifact_removal_enabled {
                        HStack {
                            Text("Variance Loss")
                            Spacer()
                            Text(String(format: "%.2f", cam.tuningParams.hf_variance_loss_threshold))
                        }
                        Slider(value: $cam.tuningParams.hf_variance_loss_threshold, in: 0.50...0.99, step: 0.01)

                        HStack {
                            Text("Min Texture SNR")
                            Spacer()
                            Text(String(format: "%.1f", cam.tuningParams.hf_min_texture_snr))
                        }
                        Slider(value: $cam.tuningParams.hf_min_texture_snr, in: 1.0...16.0, step: 0.5)
                    }

                    Toggle("Motion Edge Guard", isOn: $cam.tuningParams.motion_edge_rejection_enabled)

                    if cam.tuningParams.motion_edge_rejection_enabled {
                        HStack {
                            Text("Edge Threshold")
                            Spacer()
                            Text(String(format: "%.3f", cam.tuningParams.motion_edge_threshold))
                        }
                        Slider(value: $cam.tuningParams.motion_edge_threshold,
                               in: 0.0...0.12,
                               step: 0.001)

                        HStack {
                            Text("Residual Threshold")
                            Spacer()
                            Text(String(format: "%.2f", cam.tuningParams.motion_edge_residual_threshold))
                        }
                        Slider(value: $cam.tuningParams.motion_edge_residual_threshold,
                               in: 0.0...8.0,
                               step: 0.05)

                        HStack {
                            Text("Edge Noise Floor")
                            Spacer()
                            Text(String(format: "%.1f", cam.tuningParams.motion_edge_noise_floor_multiplier))
                        }
                        Slider(value: $cam.tuningParams.motion_edge_noise_floor_multiplier,
                               in: 0.0...2.0,
                               step: 0.1)

                        Stepper(value: $cam.tuningParams.motion_edge_neighborhood_radius,
                                in: 0...2) {
                            HStack {
                                Text("Edge Neighborhood")
                                Spacer()
                                Text("\(cam.tuningParams.motion_edge_neighborhood_radius)")
                            }
                        }
                    }
                }
                
                Section(header: Text("Steerable Kernels (Merging)")) {
                    Toggle("SNR Auto Tune", isOn: $cam.tuningParams.snr_auto_tune)

                    Toggle("Global Pre-Alignment", isOn: $cam.tuningParams.global_prealignment_enabled)

                    if cam.tuningParams.global_prealignment_enabled {
                        Toggle("Choose Reference Frame", isOn: $cam.tuningParams.global_prealignment_choose_reference)

                        HStack {
                            Text("Rotation Search")
                            Spacer()
                            Text(String(format: "%.1f deg", cam.tuningParams.global_prealignment_rotation_range_deg))
                        }
                        Slider(value: $cam.tuningParams.global_prealignment_rotation_range_deg,
                               in: 0.0...2.0,
                               step: 0.1)

                        HStack {
                            Text("Rotation Step")
                            Spacer()
                            Text(String(format: "%.2f deg", cam.tuningParams.global_prealignment_rotation_step_deg))
                        }
                        Slider(value: $cam.tuningParams.global_prealignment_rotation_step_deg,
                               in: 0.05...1.0,
                               step: 0.05)

                        Stepper(value: $cam.tuningParams.global_prealignment_max_shift,
                                in: 0...64,
                                step: 4) {
                            HStack {
                                Text("Global Shift")
                                Spacer()
                                Text("\(cam.tuningParams.global_prealignment_max_shift)")
                            }
                        }
                    }

                    Picker("Alignment Tile Size", selection: $cam.tuningParams.alignment_tile_size) {
                        Text("Auto").tag(0)
                        Text("8").tag(8)
                        Text("16").tag(16)
                        Text("32").tag(32)
                        Text("64").tag(64)
                    }
                    .pickerStyle(.segmented)
                    Text("8 can follow smaller local motion in good light, but is slower and less stable on noise, straight edges, and repeated patterns. Auto keeps the SNR-based choice.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                    
                    HStack {
                        Text("Detail Sharpness (k_detail)")
                        Spacer()
                        Text(String(format: "%.2f", cam.tuningParams.k_detail))
                    }
                    Slider(value: $cam.tuningParams.k_detail, in: 0.1...1.0)
                    
                    HStack {
                        Text("Denoise Strength (k_denoise)")
                        Spacer()
                        Text(String(format: "%.1f", cam.tuningParams.k_denoise))
                    }
                    Slider(value: $cam.tuningParams.k_denoise, in: 0.0...10.0)
                    
                    HStack {
                        Text("Stretch (k_stretch)")
                        Spacer()
                        Text(String(format: "%.1f", cam.tuningParams.k_stretch))
                    }
                    Slider(value: $cam.tuningParams.k_stretch, in: 1.0...10.0)

                    HStack {
                        Text("Shrink (k_shrink)")
                        Spacer()
                        Text(String(format: "%.1f", cam.tuningParams.k_shrink))
                    }
                    Slider(value: $cam.tuningParams.k_shrink, in: 1.0...5.0)
                    Text("Higher shrink sharpens across edges (helps small text). Default 2.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
                
                Section(header: Text("Capture")) {
                    if cam.frameCount > 10 {
                        Text("\(cam.frameCount) frames: every frame stays resident through the merge, so long bursts are memory-heavy. If a capture is killed mid-processing, reduce the count or switch to 12MP output.")
                            .font(.footnote)
                            .foregroundColor(.orange)
                    }
                    Toggle("Shutter Sound", isOn: $cam.shutterSoundEnabled)
                    Text(cam.shutterSoundEnabled
                         ? "Plays the system camera click when a burst starts."
                         : "Silent. Some regions require a shutter sound by law; the system may override this.")
                        .font(.footnote)
                        .foregroundColor(.secondary)

                    Toggle("Zero Shutter Lag (ZSL)", isOn: $cam.zslEnabled)
                    Text(cam.zslEnabled
                         ? "Buffers \(cam.frameCount) RAW frames continuously. Tap shutter to grab them without holding still afterward. Ready: \(cam.zslBufferReady)/\(cam.frameCount)."
                         : "Off — classic burst: frames are captured after you tap the shutter.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }

                Section(header: Text("Export")) {
                    Picker("Resolution", selection: $cam.outputResolutionMode) {
                        ForEach(OutputResolutionMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(cam.cropZoom > 1.0001)
                    Text(cam.cropZoom <= 1.0001
                         ? (cam.outputResolutionMode == .super48mp
                            ? "Super-resolves to 4x the pixel count. Slower and uses more memory."
                            : "Merges at sensor resolution. Faster.")
                         : "Only applies when nothing is cropped away. Any zoom is captured as a crop that super-resolution doubles, so it already merges at 2x.")
                        .font(.footnote)
                        .foregroundColor(.secondary)

                    Picker("Format", selection: $cam.exportFormat) {
                        ForEach(ExportFormat.allCases) { fmt in
                            Text(fmt.label).tag(fmt)
                        }
                    }
                    .pickerStyle(.segmented)
                    Text(cam.exportFormat == .dng
                         ? "LinearRaw DNG with embedded tone-mapped JPEG preview (Photos thumbnail; Lightroom reads the raw)."
                         : "Tone-mapped JPEG: WB + matrix + Highlights −70 + contrast + vibrance (no sharpen).")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }

                Section(header: Text("Merge Architecture")) {
                    Picker("Merge", selection: $cam.tuningParams.merge_arch) {
                        Text("Auto").tag(Int32(0))
                        Text("Banded").tag(Int32(1))
                        Text("Online").tag(Int32(2))
                    }
                    .pickerStyle(.segmented)
                    Text(cam.tuningParams.merge_arch == 2
                         ? "Forced. Online is used whatever the burst length or the memory left, including where Auto would have refused it, so this can run the app out of memory. Use Auto unless you are measuring."
                         : (cam.tuningParams.merge_arch == 1
                            ? "Forced. One band at a time, every frame resident until the last one — so memory grows with the burst, but the accumulator stays small."
                            : "Online keeps one accumulator for the whole output and drops each frame as it merges, so memory stops growing with the burst. That accumulator scales with output pixels, so it costs four times as much at 48MP as at 12MP. Auto projects both peaks, needs at least 6 frames, and falls back to banding if the accumulator would not fit."))
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }

                Section(header: Text("Fallback Denoiser")) {
                    Toggle("Save Robustness Mask", isOn: $cam.tuningParams.robustness_save_mask)
                    Text("When on, also saves a grayscale robustness mask to Photos after processing.")
                        .font(.footnote)
                        .foregroundColor(.secondary)

                    Toggle("Enable Motion Denoiser", isOn: $cam.tuningParams.accumulated_robustness_denoiser_enabled)
                    
                    if cam.tuningParams.accumulated_robustness_denoiser_enabled {
                        HStack {
                            Text("Radius Max")
                            Spacer()
                            Text(String(format: "%.1f", cam.tuningParams.acc_rob_rad_max))
                        }
                        Slider(value: $cam.tuningParams.acc_rob_rad_max, in: 0.0...10.0)
                        
                        HStack {
                            Text("Max Multiplier")
                            Spacer()
                            Text(String(format: "%.1f", cam.tuningParams.acc_rob_max_multiplier))
                        }
                        Slider(value: $cam.tuningParams.acc_rob_max_multiplier, in: 1.0...20.0)
                        
                        Toggle("Adapt To Frame Count", isOn: $cam.tuningParams.acc_rob_adaptive)
                        Text(cam.tuningParams.acc_rob_adaptive
                             ? "Enlargement is derived from how many frames actually merged at each pixel, relative to the burst length. Nothing to set; Max Multiplier only caps it."
                             : "Reference behaviour: full enlargement below the frame count below, none above, and the reference frame replaces the merged result there.")
                            .font(.footnote)
                            .foregroundColor(.secondary)

                        if !cam.tuningParams.acc_rob_adaptive {
                            HStack {
                                Text("Max Frame Count")
                                Spacer()
                                Text(String(format: "%.1f", cam.tuningParams.acc_rob_max_frame_count))
                            }
                            Slider(value: $cam.tuningParams.acc_rob_max_frame_count, in: 1.0...10.0)
                        }
                    }
                }

                Section {
                    Button("Reset") {
                        cam.tuningParams = .appDefaults
                    }
                }
            }
            .navigationTitle("Algorithm Tuning")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { showSettings = false }
                }
            }
        }
        .preferredColorScheme(.dark)
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
