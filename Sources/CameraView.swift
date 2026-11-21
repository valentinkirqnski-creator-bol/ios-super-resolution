import SwiftUI
import UniformTypeIdentifiers

struct CameraView: View {
    @StateObject private var cam = CameraModel()
    @StateObject private var store = StoreManager.shared
    @Environment(\.scenePhase) private var scenePhase
    @State private var showViewer = false
    @State private var showPaywall = false
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
    /// GCam-style chrome: the chevron in the top pill opens the quick panel (frames,
    /// import, format); PRO shows the ISO / shutter bar over the viewfinder.
    @State private var showQuickPanel = false
    @State private var showPro = false
    /// Shutter press flash (inner disc lavender for ~100 ms, as in GCam).
    @State private var shutterFlash = false
    /// New-thumbnail animation: pop from half size, plus a fading ripple ring.
    @State private var thumbPop = true
    @State private var thumbRipple: CGFloat = 1
    @State private var thermal = ProcessInfo.processInfo.thermalState
    /// Capture ring: the viewfinder dims and shows the ring while frames are taken.
    @State private var captureOverlay = false
    /// Thumbnail shown in the library button: the burst's preview while it merges,
    /// then the finished capture; `underThumb` is the previous one while the new pops in.
    @State private var heldPreview: UIImage?
    @State private var displayedThumb: UIImage?
    @State private var underThumb: UIImage?
    /// The grey capsule (chevron + gradient disc) behind the thumbnail while merging.
    @State private var capsuleUp = false

    var body: some View {
        GeometryReader { geo in
            // GCam layout: a slim black top bar, the 4:3 viewfinder, and a black
            // bottom bar holding flip / shutter / thumbnail.
            let topBarH: CGFloat = 60
            let bottomH: CGFloat = 112
            let vfWidth = geo.size.width
            // One value for the space below the controls, used both to reserve
            // it and to apply it (see the history of the double-inset overflow).
            let bottomInset = geo.safeAreaInsets.bottom + 6
            let maxVFHeight = geo.size.height - topBarH - bottomH - bottomInset
            let vfHeight = min(maxVFHeight, vfWidth * 4 / 3)

            ZStack {
                Color.black.ignoresSafeArea()

                if cam.permissionDenied {
                    permissionView
                } else {
                    VStack(spacing: 0) {
                        topStrip
                            .padding(.top, geo.safeAreaInsets.top)
                            .frame(height: topBarH + geo.safeAreaInsets.top)
                            .background(Color.black)

                        viewfinder(width: vfWidth, height: vfHeight)

                        bottomPanel
                            .frame(minHeight: bottomH, maxHeight: .infinity)
                            .padding(.bottom, bottomInset)
                            .background(Color.black)
                    }
                }
            }
        }
        .onReceive(NotificationCenter.default.publisher(
            for: ProcessInfo.thermalStateDidChangeNotification)) { _ in
            thermal = ProcessInfo.processInfo.thermalState
        }
        .onChange(of: cam.isCapturing) { capturing in
            if capturing {
                withAnimation(.easeOut(duration: 0.18)) { captureOverlay = true }
            } else {
                // Let the completed ring show for a moment, then fade ring + dim together.
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
                    withAnimation(.easeIn(duration: 0.2)) { captureOverlay = false }
                }
            }
        }
        .onChange(of: cam.isProcessing) { processing in
            if processing {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                    if cam.isProcessing { withAnimation(.easeOut(duration: 0.2)) { capsuleUp = true } }
                }
            } else {
                DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                    if !cam.isProcessing { withAnimation(.easeIn(duration: 0.2)) { capsuleUp = false } }
                }
            }
        }
        .onChange(of: cam.burstPreviewThumb) { p in
            // The new shot pops into the library button as soon as capture ends.
            guard let p = p else { return }
            heldPreview = p
            popThumbnail(to: p)
        }
        .onChange(of: cam.lastThumbnail) { t in
            guard let t = t else { return }
            if heldPreview != nil {
                heldPreview = nil            // same shot, finished: swap without a second pop
                displayedThumb = t
            } else {
                popThumbnail(to: t)          // ZSL / import: no early preview, pop now
            }
        }
        .onChange(of: cam.isBusy) { busy in
            // A failed burst never delivers a final thumbnail: go back to the last one.
            if !busy {
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    if !cam.isBusy && heldPreview != nil {
                        heldPreview = nil
                        displayedThumb = cam.lastThumbnail
                    }
                }
            }
        }
        .onAppear {
            if !didApplyLaunchShutter {
                cam.ensureShutterAutoOnLaunch()
                cam.ensureCaptureDefaultsOnLaunch()   // 48MP · DNG · 8 frames
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
        .sheet(isPresented: $showPaywall) { PaywallView(store: store) }
        .sheet(isPresented: $showGallery) { GalleryView() }
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: [.image],
                      allowsMultipleSelection: true) { result in
            if case .success(let urls) = result {
                // Picked files sit outside the sandbox and the pipeline reads
                // them on a background queue, so the security scope must stay
                // open past this closure; processImportedDNGs closes it.
                let picked = urls.filter { $0.startAccessingSecurityScopedResource() }
                guard !picked.isEmpty else { return }
                // Importing from storage is "taking a photo" too, so it counts
                // against the free daily limit exactly like a shutter press.
                if outOfFreeCaptures {
                    picked.forEach { $0.stopAccessingSecurityScopedResource() }
                    showPaywall = true
                } else if cam.processImportedDNGs(picked) {
                    store.registerCapture()   // only count imports that actually start
                }
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

            // GCam capture progress: the viewfinder dims to ~38% and a 79 pt ring at its
            // centre fills clockwise from 12 o'clock as the frames arrive (faint track
            // underneath), then the ring and the dimming fade out together.
            ZStack {
                Color.black.opacity(0.62)
                Circle()
                    .stroke(Color.white.opacity(0.25), lineWidth: 2.3)
                    .frame(width: 79, height: 79)
                Circle()
                    .trim(from: 0, to: CGFloat(min(1, max(0, cam.captureFraction))))
                    .stroke(Color.white, style: StrokeStyle(lineWidth: 2.3, lineCap: .butt))
                    .rotationEffect(.degrees(-90))
                    .frame(width: 79, height: 79)
                    .animation(.linear(duration: 0.12), value: cam.captureFraction)
            }
            .frame(width: width, height: height)
            .opacity(captureOverlay ? 1 : 0)
            .allowsHitTesting(false)

            // Top of the viewfinder: the quick panel (chevron), the PRO exposure bar,
            // and the pipeline status line.
            VStack(spacing: 6) {
                if showQuickPanel {
                    quickPanel
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
                if showPro {
                    exposureBar
                        .background(RoundedRectangle(cornerRadius: 14, style: .continuous)
                                        .fill(Color.black.opacity(0.62)))
                        .padding(.horizontal, 8)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
                if !cam.statusText.isEmpty && (cam.isProcessing || !cam.isBusy) {
                    Text(cam.statusText)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.white.opacity(0.85))
                        .lineLimit(2)
                        .minimumScaleFactor(0.75)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(Capsule().fill(Color.black.opacity(0.45)))
                        .padding(.horizontal, 16)
                        .allowsHitTesting(false)
                }
                Spacer(minLength: 0)
            }
            .padding(.top, 8)
            .frame(width: width, height: height)

            // Right edge: the vertical lens column (tele / 1x / ultra-wide).
            if cam.cameraSelection != .front {
                HStack(spacing: 0) {
                    Spacer(minLength: 0)
                    VStack(spacing: 0) {
                        Spacer(minLength: 0)
                        backLensPicker
                        Spacer(minLength: 0).frame(height: height * 0.16)
                    }
                    .padding(.trailing, 8)
                }
                .frame(width: width, height: height)
            }

            // Bottom centre: the 1x | 2x output toggle; the zoom ruler while zooming.
            VStack {
                Spacer()
                if cam.cameraSelection != .front && cam.zoomUIVisible {
                    zoomSlider(width: width)
                        .padding(.bottom, 14)
                        .transition(.opacity)
                } else {
                    resolutionToggle
                        .padding(.bottom, 11)
                        .transition(.opacity)
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
            .stroke(Color.white.opacity(0.45), lineWidth: 0.6)
        }
    }

    // MARK: - Exposure bar (above the viewfinder)

    /// ISO + shutter, side by side, above the viewfinder. Tap a label to toggle
    /// Auto/Manual; drag a slider to set a manual value. Dragging commits through
    /// setISOFromSlider / setShutterFromSlider, which flip to manual FIRST so the
    /// auto-exposure poll can't revert the change (the old "won't change" bug).
    private var exposureBar: some View {
        HStack(spacing: 10) {
            exposureControl(
                title: "ISO",
                valueLabel: cam.isoLabel,
                isAuto: cam.isoIsAuto,
                slider: Binding(get: { cam.isoSlider },
                                set: { cam.setISOFromSlider($0) }),
                toggle: { cam.isoIsAuto.toggle() })
            Rectangle().fill(Color.white.opacity(0.12)).frame(width: 1, height: 34)
            exposureControl(
                title: "SHUTTER",
                valueLabel: cam.shutterLabel,
                isAuto: cam.shutterIsAuto,
                slider: Binding(get: { cam.shutterSlider },
                                set: { cam.setShutterFromSlider($0) }),
                toggle: { cam.toggleShutterAuto() })
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 9)
    }

    private func exposureControl(title: String, valueLabel: String, isAuto: Bool,
                                 slider: Binding<Double>,
                                 toggle: @escaping () -> Void) -> some View {
        let accentC: Color = isAuto ? Color.white.opacity(0.45) : Color.yellow
        return VStack(alignment: .leading, spacing: 2) {
            Button(action: { if !cam.isBusy { toggle() } }) {
                HStack(spacing: 5) {
                    Text(title)
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.white.opacity(0.5))
                    Text(valueLabel)
                        .font(.system(size: 13, weight: .semibold).monospacedDigit())
                        .foregroundColor(isAuto ? .white.opacity(0.75) : .yellow)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            Slider(value: slider, in: 0...1)
                .tint(accentC)
                .disabled(cam.isBusy)
        }
        .frame(maxWidth: .infinity)
        .opacity(cam.isBusy ? 0.4 : 1)
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

    /// GCam top bar: a big grey number (here the burst frame count; tap for the quick
    /// panel), the dark pill with settings + chevron, the thermometer (device thermal
    /// state), the format badge (D = DNG, J = JPG; tap to switch) and PRO (ISO / shutter).
    private var topStrip: some View {
        HStack(spacing: 0) {
            Button(action: { withAnimation(.easeOut(duration: 0.2)) { showQuickPanel.toggle() } }) {
                Text("\(cam.frameCount)")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundColor(GC.textGrey)
                    .frame(minWidth: 44, minHeight: 44)
            }
            .accessibilityLabel("Burst frames")
            .padding(.leading, 18)

            Spacer(minLength: 8)

            HStack(spacing: 20) {
                Button(action: { showSettings = true }) {
                    Image(systemName: "gearshape")
                        .font(.system(size: 19, weight: .regular))
                        .foregroundColor(GC.iconGrey)
                }
                .accessibilityLabel("Settings")
                Button(action: { withAnimation(.easeOut(duration: 0.2)) { showQuickPanel.toggle() } }) {
                    Image(systemName: "chevron.down")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(GC.iconGrey)
                        .rotationEffect(.degrees(showQuickPanel ? 180 : 0))
                }
                .accessibilityLabel("Quick settings")
            }
            .padding(.horizontal, 14)
            .frame(height: 39)
            .background(Capsule().fill(GC.pill))

            Spacer(minLength: 10)

            Image(systemName: "thermometer")
                .font(.system(size: 17, weight: .regular))
                .foregroundColor(thermalColor)
                .accessibilityLabel("Device temperature")

            Spacer(minLength: 10)

            Button(action: {
                guard !cam.isBusy else { return }
                cam.exportFormat = (cam.exportFormat == .dng) ? .jpg : .dng
            }) {
                ZStack {
                    Image(systemName: "seal")
                        .font(.system(size: 22, weight: .regular))
                    Text(cam.exportFormat == .dng ? "D" : "J")
                        .font(.system(size: 10, weight: .heavy))
                }
                .foregroundColor(GC.badge)
            }
            .disabled(cam.isBusy)
            .accessibilityLabel("Save format")
            .accessibilityValue(cam.exportFormat.label)

            Spacer(minLength: 10)

            Button(action: { withAnimation(.easeOut(duration: 0.2)) { showPro.toggle() } }) {
                Text("PRO")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(showPro ? GC.badge : GC.textGrey)
            }
            .padding(.trailing, 16)
        }
        .frame(maxHeight: .infinity)
    }

    /// The coarsest-level search radius actually used: 4 + offset (at least 1), or 4 when fixed.
    private var coarsestRadiusLabel: String {
        if cam.tuningParams.coarsest_search_radius_fixed4 { return "±4" }
        let d = cam.tuningParams.coarsest_search_radius_delta
        let sign = d >= 0 ? "+" : ""
        return "±\(max(1, 4 + d)) (\(sign)\(d))"
    }

    private var thermalColor: Color {
        switch thermal {
        case .serious: return .orange
        case .critical: return .red
        default: return GC.thermo
        }
    }

    /// Chevron drop-down: burst length, import from Files, output size and format.
    private var quickPanel: some View {
        HStack(spacing: 14) {
            frameCountControl
            Button(action: { showImporter = true; showQuickPanel = false }) {
                Image(systemName: "square.stack.3d.down.right.fill")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.white)
                    .frame(width: 34, height: 34)
                    .background(Circle().fill(Color.white.opacity(0.12)))
            }
            .disabled(cam.isBusy)
            .accessibilityLabel("Import burst")
            formatControl
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(Capsule().fill(Color.black.opacity(0.62)))
    }

    /// Bottom-centre pill: 1x (12 MP) or 2x (48 MP super-resolution).
    private var resolutionToggle: some View {
        HStack(spacing: 14) {
            ForEach(OutputResolutionMode.allCases) { mode in
                let selected = cam.outputResolutionMode == mode
                Button(action: { if !cam.isBusy { cam.outputResolutionMode = mode } }) {
                    Text(mode == .native12mp ? "1x" : "2x")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(selected ? Color(white: 0.2) : GC.toggleText)
                        .frame(width: 30, height: 30)
                        .background(Circle().fill(selected ? GC.toggleSelected : Color.clear))
                }
                .disabled(cam.isBusy)
                .accessibilityLabel(mode.label)
            }
        }
        .padding(2)
        .background(Capsule().fill(GC.toggleCapsule))
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

    /// Output upscale factor: 1× (12MP) or 2× (48MP super-resolution). Sits next
    /// to the frame-count selector; governs both live capture and imports.
    private var resolutionControl: some View {
        HStack(spacing: 0) {
            ForEach(OutputResolutionMode.allCases) { mode in
                let selected = cam.outputResolutionMode == mode
                Button(action: { if !cam.isBusy { cam.outputResolutionMode = mode } }) {
                    Text(mode.label)   // "12MP" / "48MP"
                        .font(.system(size: 11, weight: selected ? .bold : .medium, design: .rounded))
                        .foregroundColor(selected ? .black : .white.opacity(0.8))
                        .padding(.horizontal, 9)
                        .frame(height: 26)
                        .background(selected ? Color.white.opacity(0.92) : Color.clear)
                }
                .disabled(cam.isBusy)
            }
        }
        .background(Capsule().fill(Color.white.opacity(0.12)))
        .clipShape(Capsule())
        .opacity(cam.isBusy ? 0.5 : 1)
    }

    /// What lands in Photos: the linear DNG, or the HDR-finished JPG rendered
    /// from it. The label is the format that will be saved and a tap switches to
    /// the other -- one button rather than a segmented pair, because with only
    /// two states the second chip would never be the answer to anything. The
    /// merge writes the DNG either way; JPG adds the finish pass on top of it.
    private var formatControl: some View {
        Button(action: {
            guard !cam.isBusy else { return }
            cam.exportFormat = (cam.exportFormat == .dng) ? .jpg : .dng
        }) {
            Text(cam.exportFormat.label)   // "DNG" / "JPG"
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundColor(.black)
                .padding(.horizontal, 10)
                .frame(height: 26)
                .background(Capsule().fill(Color.white.opacity(0.92)))
        }
        .disabled(cam.isBusy)
        .opacity(cam.isBusy ? 0.5 : 1)
        .accessibilityLabel("Save format")
        .accessibilityValue(cam.exportFormat.label)
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
        let pos = zoomPosition(cam.zoomFactor)
        return VStack(spacing: 8) {
            // Live magnification in a small tab that rides above the thumb.
            Text(Self.zoomLabel(cam.zoomFactor))
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundColor(.black)
                .padding(.horizontal, 9).padding(.vertical, 3)
                .background(
                    RoundedRectangle(cornerRadius: 7, style: .continuous).fill(accent)
                )
                .offset(x: (pos - 0.5) * trackW)

            // Tapered "spectrum" ruler: vertical bars that ramp taller toward the
            // long end (a widening wedge that reads as increasing magnification),
            // with a lozenge fader cap riding the track. Deliberately unlike the
            // round-bubble-over-dots pattern.
            ZStack {
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(Color.white.opacity(0.16))
                    .frame(height: 3)
                HStack(spacing: 0) {
                    ForEach(0..<ticks, id: \.self) { i in
                        let t = CGFloat(i) / CGFloat(ticks - 1)
                        let onStop = zoomStops.contains {
                            abs(zoomPosition($0) - t) < 0.5 / CGFloat(ticks - 1)
                        }
                        let ramp = 5 + 9 * t            // taller toward telephoto
                        Rectangle()
                            .fill(onStop ? accent.opacity(0.9) : Color.white.opacity(0.35))
                            .frame(width: onStop ? 2.5 : 1.5,
                                   height: onStop ? ramp + 5 : ramp)
                            .frame(maxWidth: .infinity)
                    }
                }
                .padding(.horizontal, 8)

                // Fader cap.
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(accent)
                    .frame(width: 12, height: 28)
                    .overlay(
                        RoundedRectangle(cornerRadius: 5, style: .continuous)
                            .stroke(Color.black.opacity(0.25), lineWidth: 1)
                    )
                    .shadow(color: .black.opacity(0.45), radius: 3, y: 1)
                    .offset(x: (pos - 0.5) * trackW)
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

    /// GCam's vertical lens column on the right edge: longest lens on top. Only the
    /// physical lenses this device has (the 2x sensor crop stays on the zoom ruler).
    private var backLensPicker: some View {
        VStack(spacing: 5) {
            if cam.availableCameras.contains(.telephoto) {
                lensChip(title: cam.telephotoLensLabel,
                         selected: cam.isAtZoom(cam.telephotoNativeZoom)) {
                    cam.setLensZoom(.telephoto)
                }
            }
            if cam.availableCameras.contains(.wide) {
                lensChip(title: "1x", selected: cam.isAtZoom(1)) {
                    cam.setLensZoom(.wide1x)
                }
            }
            if cam.availableCameras.contains(.ultraWide) {
                lensChip(title: String(format: "%.1fx", Double(cam.ultraWideNativeZoom)),
                         selected: cam.isAtZoom(cam.ultraWideNativeZoom)) {
                    cam.setLensZoom(.ultraWide)
                }
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 2)
        .background(Capsule().fill(GC.lensCapsule))
    }

    private func lensChip(title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(selected ? Color(white: 0.15) : .white)
                .frame(width: 38, height: 38)
                .background(Circle().fill(selected ? GC.lensSelected : Color.clear))
        }
        .disabled(cam.isBusy)
    }

    // MARK: - Bottom panel (Apple-style)

    /// GCam bottom bar: flip (left), shutter (centre), last-capture thumbnail (right).
    private var bottomPanel: some View {
        VStack(spacing: 0) {
            freeTierBanner
                .padding(.top, 4)
            Spacer(minLength: 0)
            ZStack {
                flipCameraButton
                    .offset(x: -126)
                shutterButton
                galleryButton
                    .offset(x: 126)
            }
            .frame(maxWidth: .infinity)
            Spacer(minLength: 0)
        }
    }

    private var flipCameraButton: some View {
        let enabled = cam.availableCameras.contains(.front) && !cam.isBusy
        return Button(action: { cam.toggleFrontCamera() }) {
            ZStack {
                Circle()
                    .strokeBorder(GC.flipRing.opacity(enabled ? 1 : 0.3), lineWidth: 2.3)
                    .frame(width: 59, height: 59)
                Image(systemName: "arrow.triangle.2.circlepath")
                    .font(.system(size: 23, weight: .regular))
                    .foregroundColor(GC.flipRing.opacity(enabled ? 1 : 0.3))
            }
        }
        .disabled(!enabled)
    }

    /// True when a non-paying user has used the day's free captures.
    private var outOfFreeCaptures: Bool {
        !store.isUnlocked && store.photosRemaining == 0
    }

    /// GCam shutter: thin white outer ring, black gap, light-grey disc. Pressing turns
    /// the disc lavender instantly (no scale change) for as long as the finger is down,
    /// and at least ~100 ms for a quick tap, matching the reference recording.
    private var shutterButton: some View {
        Button(action: {
            if cam.isBusy { return }
            // Out of free captures: don't shoot, offer the unlock instead.
            if outOfFreeCaptures { showPaywall = true; return }
            shutterFlash = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { shutterFlash = false }
            cam.captureBurst()
            store.registerCapture()
        }) {
            EmptyView()
        }
        .buttonStyle(GCamShutterStyle(flash: shutterFlash, locked: outOfFreeCaptures,
                                      capturing: cam.isCapturing))
        // Kept tappable when out of free captures so the tap can open the paywall;
        // it will not start a capture in that state (handled in the action).
        .disabled(cam.isBusy)
    }

    /// Free-tier status: remaining count, or an unlock prompt once the daily
    /// limit is hit. Hidden entirely for unlocked users.
    @ViewBuilder private var freeTierBanner: some View {
        if !store.isUnlocked {
            if outOfFreeCaptures {
                Button(action: { showPaywall = true }) {
                    HStack(spacing: 6) {
                        Image(systemName: "lock.fill").font(.system(size: 11, weight: .bold))
                        Text("Free limit reached · Unlock Unlimited — \(store.displayPrice)")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .foregroundColor(MD3.onPrimary)
                    .padding(.horizontal, 14).padding(.vertical, 7)
                    .background(Capsule().fill(MD3.primary))
                }
                .padding(.bottom, 6)
            } else {
                Text("\(store.photosRemaining) of \(StoreManager.freeTotalLimit) free photos left")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white.opacity(0.55))
                    .padding(.bottom, 4)
            }
        }
    }

    /// Small circular control on a translucent disc, used for the two utility
    /// buttons that flank the shutter row.
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

    /// GCam library button. Thumbnail Ø63 with a 2 pt white ring. While the burst
    /// merges, a steel-blue arc (outer Ø 1.19x, stroke 0.094x the thumbnail) sweeps
    /// clockwise from 12 o'clock with the merge progress, and a grey capsule slides up
    /// from behind the thumbnail carrying a chevron and a pink-violet disc. A new shot
    /// grows from 55% over the previous one with a faint ripple expanding behind.
    private var galleryButton: some View {
        let d: CGFloat = 56.5
        let shown = displayedThumb ?? cam.lastThumbnail
        let mergeFrac = cam.isProcessing ? max(0.03, min(1, (Double(cam.progress) - 0.15) / 0.85)) : 1
        return Button(action: { if !cam.isBusy { showGallery = true } }) {
            ZStack {
                // Merge progress arc (under the capsule and the thumbnail).
                Circle()
                    .trim(from: 0, to: CGFloat(mergeFrac))
                    .stroke(GC.progress, style: StrokeStyle(lineWidth: d * 0.094, lineCap: .butt))
                    .rotationEffect(.degrees(-90))
                    .frame(width: d * 1.19 - d * 0.094, height: d * 1.19 - d * 0.094)
                    .opacity(cam.isProcessing ? 1 : 0)
                    .animation(.easeInOut(duration: 0.25), value: mergeFrac)
                    .animation(.easeOut(duration: 0.25), value: cam.isProcessing)

                // Capsule: clipped to the area above the thumbnail centre, so it slides
                // up out of the thumbnail rather than appearing below it.
                ZStack(alignment: .top) {
                    RoundedRectangle(cornerRadius: d / 2, style: .continuous)
                        .fill(GC.capsule)
                        .frame(width: d, height: d * 1.25 + d / 2)
                    Image(systemName: "chevron.up")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(GC.chevron)
                        .padding(.top, d * 1.25 - d * 0.96 - 8)
                    ZStack {
                        Circle().fill(GC.discRim).frame(width: d * 0.63, height: d * 0.63)
                        Circle().fill(GC.discGradient).frame(width: d * 0.5, height: d * 0.5)
                    }
                    .padding(.top, d * 1.25 - d * 0.52 - d * 0.315)
                }
                .frame(width: d, height: d * 1.25, alignment: .top)
                .offset(y: capsuleUp ? 0 : d * 1.25)
                .frame(width: d, height: d * 1.25, alignment: .top)
                .clipped()
                .offset(y: -d * 1.25 / 2)

                // Ripple behind the new thumbnail.
                Circle()
                    .stroke(Color.white.opacity(0.22 * Double(1 - thumbRipple)), lineWidth: 4)
                    .frame(width: d, height: d)
                    .scaleEffect(1 + 0.55 * thumbRipple)

                // Previous thumbnail stays underneath while the new one grows in.
                if let u = underThumb {
                    thumbCircle(u, d: d)
                }
                thumbCircle(shown, d: d)
                    .scaleEffect(thumbPop ? 1 : 0.55)
            }
            .frame(width: d * 1.25, height: d * 1.25)
        }
        .disabled(cam.isBusy && shown == nil)
    }

    private func thumbCircle(_ img: UIImage?, d: CGFloat) -> some View {
        Group {
            if let img = img {
                Image(uiImage: img).resizable().scaledToFill()
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
        .frame(width: d, height: d)
        .clipShape(Circle())
        .overlay(Circle().strokeBorder(GC.ring, lineWidth: 2.3))
    }

    /// New thumbnail: grows from 55% over the previous one, ripple expands and fades.
    private func popThumbnail(to img: UIImage) {
        var t = Transaction(); t.disablesAnimations = true
        withTransaction(t) {
            underThumb = displayedThumb
            displayedThumb = img
            thumbPop = false
            thumbRipple = 0
        }
        DispatchQueue.main.async {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.78)) { thumbPop = true }
            withAnimation(.easeOut(duration: 0.5)) { thumbRipple = 1 }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { underThumb = nil }
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

    /// Label, live value and slider as ONE view. Twelve parameters at two
    /// children each would blow SwiftUI's 10-child ViewBuilder limit whatever
    /// way the sections were split.
    private func ispRow(_ title: String, _ value: Binding<Float>,
                        _ range: ClosedRange<Float>, _ fmt: String = "%.2f") -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: fmt, value.wrappedValue))
                    .foregroundColor(.secondary)
                    .monospacedDigit()
            }
            Slider(value: value, in: range)
        }
    }

    @ViewBuilder
    private var fineAlignmentSection: some View {
        Toggle("Match 1.4 Alignment", isOn: $cam.tuningParams.align_match_14)
        Text("""
             Switches the three places the aligner (a 460-main derivative) diverges from \
             Handheld-Multi-Frame-Super-Resolution-1.4: finest search radius drops to 1, \
             inter-level flow upscaling becomes a plain bilinear resize (not the 460 \
             three-candidate re-match), and ICA runs on every pyramid level of the FFT grey. \
             Algorithm parity with 1.4, not bit parity (the FFT and GPU float order still \
             differ upstream). Off keeps the current 460 behaviour.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Geometry Rejection (rotation)", isOn: $cam.tuningParams.motion_geom_reject_enabled)
        Text("""
             Rejects pixels where the per-tile translation is a poor model of the \
             local motion (flow gradient × distance from tile centre, weighted by \
             edge strength) — the rotation tile-ghosts. Rejected pixels fall back to \
             the reference. Inert under one-direction motion. A hiding fix: trades \
             some burst samples for artifact-free output. Lower threshold rejects more.
             """)
            .font(.caption2).foregroundColor(.secondary)
        HStack {
            Text("Geom Threshold")
            Spacer()
            Text(String(format: "%.4f", cam.tuningParams.motion_geom_reject_threshold))
        }
        Slider(value: $cam.tuningParams.motion_geom_reject_threshold, in: 0.0...0.06)
        Text("~0.02 rejects ~15%, 0.03 ~10%, 0.06 ~3% (near-inert). Lower = cleaner, fewer samples kept.")
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Also catch low-light (exposure-invariant)", isOn: $cam.tuningParams.motion_geom_relative)
        Text(cam.tuningParams.motion_geom_relative
             ? "Adds a CONTRAST criterion (∇g/g) on top of the absolute one above: the absolute test's gradient shrinks in dim scenes so it misses low-light misalignments, while contrast is the same at any exposure. Union — keeps every good-light rejection and adds the low-light ones."
             : "Absolute edge strength only: tuned for good light; misses misalignments in low light where gradients are weaker.")
            .font(.caption2).foregroundColor(.secondary)
        if cam.tuningParams.motion_geom_relative {
            HStack {
                Text("Geom Threshold (relative)")
                Spacer()
                Text(String(format: "%.3f", cam.tuningParams.motion_geom_reject_threshold_relative))
            }
            Slider(value: $cam.tuningParams.motion_geom_reject_threshold_relative, in: 0.0...0.20)
            HStack {
                Text("Noise Floor ×")
                Spacer()
                Text(String(format: "%.2f", cam.tuningParams.motion_geom_noise_floor_mult))
            }
            Slider(value: $cam.tuningParams.motion_geom_noise_floor_mult, in: 0.0...4.0)
            Text("Noise Floor × subtracts k·σ_noise from the gradient before dividing by brightness, so dark noisy flats don't falsely reject. Lower relative threshold rejects more.")
                .font(.caption2).foregroundColor(.secondary)
        }
        Toggle("Guide: Keep White Balance", isOn: $cam.tuningParams.guide_white_balance)
        Toggle("Guide: Colour Matrix (→sRGB)", isOn: $cam.tuningParams.guide_color_matrix)
        Picker("Guide Curve", selection: $cam.tuningParams.guide_curve) {
            Text("Auto").tag(-1)
            Text("None").tag(0)
            Text("Sqrt").tag(1)
            Text("Gamma").tag(2)
            Text("sRGB").tag(3)
        }
        .pickerStyle(.segmented)
        Text("""
             Render the robustness guide as a real display RGB before the colour \
             distance is measured: keep white balance, apply the camera→sRGB matrix, \
             and a transfer curve. Separates true colour mismatches from noise. Run \
             with the noise model OFF — the noise LUT is calibrated in the sqrt-raw \
             guide domain and won't match once WB/matrix change it. All off/Auto = \
             unchanged.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Ambiguous-Match Fallback", isOn: $cam.tuningParams.align_ambiguous_fallback_enabled)
        Text("""
             ImageStackAlignator's rule: when a tile's best and second-best block-match              costs are near-tied (flat patch, aperture problem, repeating texture -- no              precise shift can be determined), apply NO shift and keep the seed from the              coarser level or global estimate, instead of trusting a match that is              indistinguishable from noise. Acts on the flow itself -- unlike the ambiguity              demotion in the robustness mask, which is inert under rotation because every              tile is already on the strict prior. Experimental -- A/B on rotating bursts.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Linear Kernel Selection (1.4)", isOn: $cam.tuningParams.kernel_selection_linear)
        Text("""
             Merge steerable-kernel selection law. ON = 'linear' (Python 1.4 default):              the kernel anisotropy ramps continuously with the local structure A.              OFF = 'hard_threshold' (460-main): round kernels until A>1.95, then snap to              full stretch. The two agree at A=1 and A=2 and differ only for moderately              anisotropic detail. ON = exact 1.4 parity.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Disable Noise Model (Robustness)", isOn: $cam.tuningParams.debug_noise_model_disabled)
        Text("""
             Debug: zeroes the noise model as read by the robustness mask ONLY. R is then              scored from the raw measured local variance and the raw (unshrunk) pixel              difference, isolating whether a tile's colour difference reads small because              the noise model forgave it, or because the content genuinely is that flat.              Unlike the earlier version of this switch, SNR auto-tune, the alignment tile              size and kernel estimation are untouched. Diagnostic only -- leave off.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Learned Robustness Mask", isOn: $cam.tuningParams.use_neural_robustness)
            .help("Replaces the analytic robustness mask (Wronski Eq. 5-9) with a small "
                + "trained network. The analytic mask decides from a colour difference "
                + "between 3x3 local means, which cannot see a misalignment that lands on "
                + "similar-looking content or one finer than that window. The network sees "
                + "the same statistics plus the estimated flow, its local spread and a wider "
                + "neighbourhood. Measured against ground truth on synthetic bursts built "
                + "from real raws: analytic AUC 0.638, learned 0.926. Falls back to the "
                + "analytic mask automatically if the model is missing.")
        Toggle("Robustness at Raw Resolution", isOn: $cam.tuningParams.robustness_raw_resolution_enabled)
        Text("""
             Algorithm 6 (Monod/IPOL): the local statistics are computed on the              half-resolution Bayer guide, then Dodgson-x2 upscaled and flow-warped to              every raw pixel, and R is computed there -- so the rejection boundary lands              with raw-pixel precision instead of in 2x2 Bayer blocks. The 5x5 local-min              is applied twice (= 9x9 raw), preserving the paper's ~10x10-raw safety margin              that s/t/Mt were tuned against. Applies under ANY alignment (FFT or decimate)              and composes the global pre-align transform, so it is correct together with              Global Pre-Alignment. Runs host-side, so it declines GPU residency (slower).
             """)
            .font(.caption2).foregroundColor(.secondary)
    }

    // Two of the eight Sections live here rather than inline. The Form body
    // was one 343-line expression and the Swift type checker gave up on it
    // ("unable to type-check this expression in reasonable time"); these are
    // the two largest, and moving them lets each be checked on its own.
    // Out of the ViewBuilder deliberately: an inline ternary in a Text is the
    // shape that has previously pushed this file past the type-checker limit.
    private var chooseReferenceHelp: String {
        if cam.tuningParams.global_prealignment_choose_reference {
            return "Picks the most central frame as the merge base. Costs a separate decode of every frame before the merge starts."
        }
        return "Frame 0 is the merge base, so pre-alignment runs inside the alignment pass at roughly the cost of one thumbnail per frame."
    }

    @ViewBuilder
    private var robustnessSection: some View {
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
                }
    }

    @ViewBuilder
    private var kernelsSection: some View {
                Section(header: Text("Steerable Kernels (Merging)")) {
                    Toggle("SNR Auto Tune", isOn: $cam.tuningParams.snr_auto_tune)

                    Toggle("Global Pre-Alignment", isOn: $cam.tuningParams.global_prealignment_enabled)

                    if cam.tuningParams.global_prealignment_enabled {
                        Toggle("Choose Reference Frame", isOn: $cam.tuningParams.global_prealignment_choose_reference)
                        Text(chooseReferenceHelp)
                            .font(.caption)
                            .foregroundColor(.secondary)

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
    }

    private var tuningSettingsView: some View {
        NavigationView {
            Form {
                Section(header: Text("Camera")) {
                    Toggle("Shutter Sound", isOn: $cam.shutterSoundEnabled)
                }
                Section(header: Text("Unlimited Shooting")) {
                    if store.isUnlocked {
                        HStack {
                            Image(systemName: "checkmark.seal.fill")
                                .foregroundColor(.green)
                            Text("Unlimited shooting unlocked")
                        }
                    } else {
                        HStack {
                            Text("Free photos left")
                            Spacer()
                            Text("\(store.photosRemaining) of \(StoreManager.freeTotalLimit)")
                                .foregroundColor(.secondary)
                        }
                        Button {
                            Task { await store.purchase() }
                        } label: {
                            HStack {
                                Text("Unlock Unlimited Shooting")
                                Spacer()
                                Text(store.displayPrice).foregroundColor(.secondary)
                            }
                        }
                        .disabled(store.purchaseInFlight)
                        Button("Restore Purchases") {
                            Task { await store.restorePurchases() }
                        }
                        .disabled(store.purchaseInFlight)
                        if let err = store.lastErrorMessage {
                            Text(err).font(.footnote).foregroundColor(.red)
                        }
                    }
                    Text("One-time \(store.displayPrice) purchase. Permanently removes the \(StoreManager.freeTotalLimit)-photo free limit.")
                        .font(.footnote).foregroundColor(.secondary)
                }

                Section(header: Text("Advanced")) {
                    Picker("Alignment Tile Size", selection: $cam.tuningParams.alignment_tile_size) {
                        Text("8").tag(8)
                        Text("16").tag(16)
                        Text("32").tag(32)
                        Text("64").tag(64)
                    }
                    Toggle("Coarsest Search Radius ±4", isOn: $cam.tuningParams.coarsest_search_radius_fixed4)
                    Text("Fixes the block-match search window at the coarsest pyramid level to ±4 px (9×9 offsets; about ±128 raw px with the default FFT grey). The offset below is ignored while this is on. Off = 4 plus the offset.")
                        .font(.footnote)
                        .foregroundColor(.secondary)

                    Stepper(value: $cam.tuningParams.coarsest_search_radius_delta,
                            in: -5...5,
                            step: 1) {
                        HStack {
                            Text("Coarsest Search Radius")
                            Spacer()
                            // The radius actually used: 4 + offset, at least 1 (or 4 when fixed).
                            Text(coarsestRadiusLabel)
                        }
                    }
                    .disabled(cam.tuningParams.coarsest_search_radius_fixed4)
                    Text("Offsets the block-match search window at the coarsest (largest-motion) pyramid level, from -5 to +5 around the default of ±4. Raise it to catch bigger inter-frame motion (handheld drift, moving subjects); lower it to tighten the window on near-still bursts and reduce false matches on repeating structure. 0 = default.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                    Toggle("Save Robustness Mask", isOn: $cam.tuningParams.robustness_save_mask)
                    Toggle("Structure Map", isOn: $cam.tuningParams.structure_map_save)
                    Text("Saves a structure-tensor map of every frame, reference included, to Photos: hue = edge orientation, saturation = how edge-like (vs corner/texture), brightness = structure strength (same scale for all frames). Black = flat.")
                        .font(.footnote).foregroundColor(.secondary)
                }

                Section(header: Text("Robustness Mask")) {
                    Toggle("Robustness", isOn: $cam.tuningParams.robustness_enabled)
                    Text("""
                         The per-pixel robustness mask (Eq. 5–9) that down-weights moving or misaligned frames during merge. Off merges every frame equally — sharper on perfectly still scenes, but ghosts anything that moved.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.robustness_enabled {
                        Toggle("Consensus Refiner (NN)", isOn: $cam.tuningParams.cref_enabled)
                        Text("A small network refines the robustness mask (Wronski or GCam) by comparing each frame's fine structure with the reference AND with the other frames at their motion-compensated positions: real detail (aliasing) agrees with the other frames, a misaligned or moving sample does not. Can lower or raise the mask. Aligns every frame twice. Inactive until a trained model is built in.")
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Demosaiced Reference Fallback", isOn: $cam.tuningParams.ref_demosaic_fallback)
                        Text("Where the comparison frames are rejected the output falls back to the reference alone, which the merge builds per colour from its raw Bayer samples — colour fringes and zipper on edges. With this on, the reference is demosaiced once per burst (edge-directed) and those areas blend toward it. Areas where other frames merged are unchanged. Works with the Wronski and GCam masks.")
                            .font(.footnote).foregroundColor(.secondary)
                        if cam.tuningParams.ref_demosaic_fallback {
                            ispRow("Blend Start", $cam.tuningParams.ref_demosaic_share_lo, 0.1...0.95, "%.2f")
                            ispRow("Blend Full", $cam.tuningParams.ref_demosaic_share_hi, 0.2...1.0, "%.2f")
                            Text("The reference's share of a pixel's merge weight (1 = every other frame rejected; about 1/N when all N frames merge). Below Blend Start the merge is untouched; at Blend Full the pixel is the demosaiced reference. Defaults 0.50 / 0.90.")
                                .font(.footnote).foregroundColor(.secondary)
                        }
                        ispRow("Threshold t", $cam.tuningParams.r_t, 0.0...0.5, "%.3f")
                        ispRow("s1 (irregular flow)", $cam.tuningParams.r_s1, 0.0...4.0, "%.2f")
                        ispRow("s2 (regular flow)", $cam.tuningParams.r_s2, 1.0...30.0, "%.1f")
                        ispRow("Mt (motion threshold)", $cam.tuningParams.r_Mt, 0.0...4.0, "%.2f")
                        Text("460-main: t 0.12, s1 2, s2 12, Mt 0.8. Higher t rejects more; higher s1/s2 trust frames more; Mt is the per-tile flow spread that switches s2→s1.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Steerable Kernels")) {
                    ispRow("k_stretch", $cam.tuningParams.k_stretch, 1.0...8.0, "%.1f")
                    ispRow("k_shrink", $cam.tuningParams.k_shrink, 0.25...4.0, "%.2f")
                    Text("Shape of the merge kernels along/across edges. 460-main: k_stretch 4, k_shrink 2. Higher stretch elongates kernels along edges; higher shrink narrows them across edges (sharper, noisier).")
                        .font(.footnote).foregroundColor(.secondary)
                }

                Section(header: Text("Motion Model")) {
                    Toggle("Bilinear Flow Upsample", isOn: Binding(
                        get: { !cam.tuningParams.flow_upsample_candidates },
                        set: { cam.tuningParams.flow_upsample_candidates = !$0 }))
                    Text("""
                         How the motion field is passed from each coarse pyramid level to the next finer one. OFF (default) = 460-main's candidate upsampling: each finer tile tests three candidate vectors (its own and two neighbours) against the actual pixels and keeps the best — robust at motion edges, but it can snap to a neighbour's vector and produce blocky flow. ON = Python 1.4's plain bilinear resize of the flow field — smooth flow that never re-tests against the image, so coarse-level mistakes are interpolated into the fine levels instead of being corrected. A/B on real bursts; affects the aligner on both the GPU and CPU paths.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Sub-Pixel Block Matching", isOn: $cam.tuningParams.bm_subpixel_enabled)
                    Text("""
                         HDR+'s quadratic sub-pixel refinement: after each pyramid level's integer search, a quadratic is fitted to the 3×3 cost surface around the best shift and the flow moves to its analytic minimum (L2 levels only, with positive-definite and ±1 px safeguards). Removes the ±0.5 px per-level quantisation before it is multiplied by the upsample factor on the way to finer levels. The final Lucas–Kanade refinement already provides sub-pixel at the finest level, so expect the effect between levels, mostly on fine texture and larger motion. Runs on both GPU and CPU aligners.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Per-Level LK Refinement", isOn: $cam.tuningParams.lk_every_level_enabled)
                    Text("""
                         Refines every pyramid level's block-match result with 3 Lucas–Kanade iterations against that level's pixels (fresh implementation: damped Gauss–Newton, step clamp, and a per-tile gate so no tile can end worse than its seed). Sub-pixel accuracy at each level stops coarse-level errors being multiplied by the upsample factor on the way to finer levels. The finest-level refinement still runs as always; this adds the between-level polish. Runs on both GPU and CPU aligners; the refined flow feeds the mask and merge as usual.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Reject Ambiguous Shifts", isOn: $cam.tuningParams.align_ambiguous_fallback_enabled)
                    Text("""
                         ImageStackAlignator's rule: when a tile's best and second-best block-match costs are nearly tied — a flat patch, repeating texture, or a STEADY handheld burst where there is almost no motion and noise picks the winner — apply NO shift and keep the seed instead of trusting a coin-flip match. Measured to halve the scattered wrong tile shifts on near-still bursts (the usual cause of rare tile-edge-shaped dark/colored lines in an otherwise clean output), at ~4% cost on rotating bursts.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Discard Unalignable Frames", isOn: $cam.tuningParams.frame_discard_enabled)
                    Text("""
                         Drops a whole comparison frame from the merge when too little of it aligned — the fraction of the robustness mask that's confidently accepted is below the threshold below. Catches a frame ruined by motion blur, a finger over the lens, a scene change, or motion beyond the search range, whose few "accepted" pixels are usually spurious. It reuses the mask pass that already runs (just a count) and routes the drop through the path that already skips a zero-mask frame, so it REMOVES merge work — it adds no RAM and no GPU. The reference and the aligned frames merge as normal. Off by default.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    HStack {
                        Text("Keep if aligned ≥ \(Int(cam.tuningParams.frame_discard_min_fraction * 100))%")
                        Slider(value: $cam.tuningParams.frame_discard_min_fraction, in: 0.05...0.6, step: 0.05)
                    }
                    Text("""
                         A frame is kept only if at least this fraction of it is confidently aligned. Lower = more permissive (drops only badly broken frames); higher = stricter. 25% is a sensible default. Only used when Discard Unalignable Frames is on.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Half-Tile Refinement", isOn: $cam.tuningParams.halftile_refine_enabled)
                    Text("""
                         After registration finishes, every motion tile is split into four half-size children: each child starts from the tile's affine model at its own centre, keeps the parent's reliability statistics (so the rejection mask doesn't over-fire on the finer grid), and is then re-measured against the pixels with a safety gate — a child with too little texture keeps its seed. The mask and merge then use the finer field. Robustness of the full tile search, spatial resolution of the half tile: helps where motion changes INSIDE a tile (subject edges, depth boundaries); smooth camera motion is already handled exactly by Affine Tile Motion. Works with the CPU and GPU pipelines.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Picker("Refine to", selection: $cam.tuningParams.halftile_refine_target_ts) {
                        Text("Half (ts/2)").tag(0)
                        Text("ts=8").tag(8)
                        Text("ts=4").tag(4)
                    }
                    .pickerStyle(.segmented)
                    Text("""
                         The target tile size the flow is refined to. "Half" = one split (the original, ts/2). "ts=8"/"ts=4" are absolute targets: the subdivision is chained from your actual Alignment Tile Size down to that size — so ts=4 is ts=4 whether the tile is 16 (2 splits), 32 (3) or 64 (4), and it works with Auto. It's a finer tile LATTICE at full resolution that follows within-tile rotation at the target size, WITHOUT the interpolated waviness of Dense Per-Pixel Flow. The pixel-accurate Lucas-Kanade polish runs while the child tile stays ≥6px, so levels ≥8px are re-measured against pixels and a 4px level carries that refined motion plus the affine gradient (4px is too small to solve reliably on its own, which is what avoids the waviness). Capped at 4 splits. Only used when Half-Tile Refinement is on.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Dense Per-Pixel Flow", isOn: $cam.tuningParams.dense_flow_enabled)
                    Text("""
                         The finest motion model: after the tile/affine/half-tile flow is finalized, it seeds a windowed Lucas-Kanade that refines the motion from the ts=16 tile grid down to a dense field (one vector per 2×2 grey block). The analysis window is ~16px (enough texture for a reliable solve), so it resolves motion that varies WITHIN a 16-tile — a rotation gradient, mild parallax, a subject/depth edge — which one vector per tile averages away. The solve runs on a half-resolution copy of the grey so its working set fits device memory (at full resolution it was ~780MB and got skipped mid-burst, so "on" did nothing); the full-resolution grey still drives block match, affine, the mask and the merge, and the merge samples the dense field at full res. Each pixel is solved only where the local structure constrains it (a straight edge keeps its seed in the slide-along direction; a flat patch keeps its seed entirely), bounded near the seed, and kept only where it beats the seed by a clear margin — so it never drifts on noise. Measured on synthetic bursts: inert on already-aligned content, cuts a motion-discontinuity error ~3x. Heaviest motion option; leave off unless a burst still ghosts with Affine + Half-Tile on.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Feature-Metric Alignment (NN)", isOn: $cam.tuningParams.feat_metric_refine_enabled)
                    Text("""
                         A tiny learned convolutional network maps the reference and each comparison frame to scene-aware feature planes; a per-tile affine Lucas-Kanade then refines the flow ON THOSE FEATURES instead of raw brightness. The learned features disambiguate handheld tremor, parallax and repetitive/aliased texture that intensity matching locks onto wrongly — measured +16–37% alignment accuracy over the classical refinement on held-out real bursts, with the affine model keeping camera rotation from regressing. Produces the same dense per-pixel field the merge and mask sample (an alternative to Dense Per-Pixel Flow; if both are on, this one wins). Runs on a half-resolution grey so its memory fits the device, and every tile is kept only where it beats the seed photometrically, so it can't make a frame worse. Off by default; try it when a handheld burst still shows fine-edge misalignment with Affine + Half-Tile on.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Half-Res Decimate + Per-Pixel Flow", isOn: $cam.tuningParams.decimate_dense_align_enabled)
                    Text("""
                         Runs alignment on the half-resolution Bayer-decimate grey (block matching + tile ICA) and then refines the plain block-match tile vectors — no affine model — to one vector per grey pixel with 3 Lucas-Kanade iterations (windowed, noise-aware conditioning, each pixel bounded to its seed and kept only where it beats the seed). The per-pixel field is stored at half resolution and sampled back to full res by the merge and the mask. This is an alternative to the default full-resolution FFT aligner: it is cheaper (half-res), but because it starts from a lower-resolution base the per-pixel refinement has to recover the precision decimation gave up — so compare it against the default on your own bursts rather than assuming it's sharper. Off by default.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Affine Tile Motion", isOn: $cam.tuningParams.affine_flow_enabled)
                    Text("""
                         Experimental. A full local affine (rotation/shear/scale) per tile instead of one translation, so camera rotation, yaw/pitch and smooth parallax align correctly within each tile rather than ghosting at tile seams. The affine is now MEASURED per tile against the pixels (6-parameter Gauss-Newton refinement seeded by the neighbour fit; a tile is kept only if it provably improves), and the robustness mask scores exactly the same affine correspondence the merge fetches — previously the mask used the plain per-tile vector while the merge warped affinely. Does not model sharp depth or moving-object edges — robustness still handles those.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("GCam SABRE Robustness (exact)", isOn: $cam.tuningParams.robustness_gcam_enabled)
                    Text("""
                         Switches the rejection mask ENTIRELY to Google Super Res Zoom's exact ComputeFrameWeight, reimplemented from the recovered shader source. It runs in square-root colour space on a dedicated 4-channel guide: a 1-2-1 blurred RGB colour plus a signed local variance whose sign flags high-frequency “green-only” regions (there it compares the single green channel so fine misalignment can't hide in the RGB average), with clipped highlights poisoned so they never merge. Per channel it subtracts the two-frame noise from the squared difference, clamps at zero, normalizes by the larger of the measured variance and that noise, and rejects as w = exp2(−distance). The comparison frame is warped and bicubic-B-spline sampled inside the pass. It rejects ×6 harder where the per-tile flow's local motion extent is large and the region is textured — the rotation / parallax / moving-object tile ghosts. The per-tile flow is also 3×3-median-cleaned first, and both the mask and the merge read that cleaned flow. By default it runs noise-blind (matching the measured device, the most aggressive setting) on the verified CPU operator — which runs on device. Structural non-matches to GCam (noted in code): the single packed guide variance, the unblocker (zero on-device), per-frame flow-origin shift, and the dilation's 2× downsample. Off by default.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.robustness_gcam_enabled {
                        Toggle("SABRE Unblocker", isOn: Binding(
                            get: { cam.tuningParams.gcam_unblocker_enabled },
                            set: { v in
                                cam.tuningParams.gcam_unblocker_enabled = v
                                if v {
                                    cam.tuningParams.hf_artifact_removal_enabled = false   // replaces HF rejection
                                    cam.tuningParams.mann_enabled = false                  // and the misalignment NN
                                }
                            }))
                        Text("GCam's variance-loss guard, replacing HF Artifact Rejection (turning one on turns the other off). Per tile of each comparison frame it compares the variance of the green plane with that of its 3x3 box-blurred copy, noise-gated at 4x the sensor noise: m = sqrt((V_blur - 4 noise) / (V_raw x 128/9)), unblocker = clamp(Scale x (m - Offset), 0, 1). Only where the local motion field is irregular (the same flow threshold as the x6 boost) it caps the keep weight at 1 - unblocker, before the dilate. m is at most 0.265 (all-low-frequency content), so GCam's own constants (Scale 0.2, Offset 0.45) never fire; defaults here are Scale 8.0, Offset 0.15 (starts at m 0.15, full reject at 0.265). Higher Offset = rejects less.")
                            .font(.footnote).foregroundColor(.secondary)
                        if cam.tuningParams.gcam_unblocker_enabled {
                            ispRow("Unblocker Scale", $cam.tuningParams.gcam_unblocker_scale, 0.0...20.0, "%.2f")
                            ispRow("Unblocker Offset", $cam.tuningParams.gcam_unblocker_offset, 0.0...0.5, "%.3f")
                        }
                        Toggle("Structure Match", isOn: $cam.tuningParams.gcam_structure_match)
                        Text("Also compares the local structure tensor of each comparison frame (at its motion-compensated position) with the reference's: edge direction (weighted by how edge-like both are) and structure strength, counted only where structure is clearly above noise. Mismatches multiply the GCam mask down. Catches moved, appeared or re-textured content and multi-pixel misalignment; a sub-pixel shift of an edge leaves the tensor unchanged, so it does not catch those.")
                            .font(.footnote).foregroundColor(.secondary)
                        if cam.tuningParams.gcam_structure_match {
                            ispRow("Strength tolerance (log ratio)", $cam.tuningParams.gcam_structure_tau_strength, 0.1...3.0, "%.2f")
                            ispRow("Direction tolerance (sin)", $cam.tuningParams.gcam_structure_tau_angle, 0.05...1.0, "%.2f")
                            ispRow("Noise gate (x noise)", $cam.tuningParams.gcam_structure_noise_k, 0.0...10.0, "%.1f")
                            Stepper("Spread window: \(cam.tuningParams.gcam_structure_min_window)", value: $cam.tuningParams.gcam_structure_min_window, in: 1...9, step: 2)
                        }
                        Toggle("Motion Field Bilinear Interpolation", isOn: $cam.tuningParams.gcam_sabre_flow_bilinear)
                        Text("""
                             To eliminate tile-boundary artifacts under rotation or sub-pixel motion, the per-tile motion vectors are bilinearly interpolated across tile centres into a smooth per-pixel motion field. ON: this one continuous field is used both for sample placement in the merge (kernel regression) and for computing the local colour difference in the robustness model, so the mask grades exactly the samples the merge fetches. Off (default) = both use one vector per tile (GCam's nearest flow texture). Dense / affine motion take precedence when they are on. Requires GCam SABRE Robustness.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Noise Model (Monte-Carlo)", isOn: $cam.tuningParams.gcam_sabre_noise_model)
                        Text("""
                             Builds the real noise LUT from the DNG NoiseProfile (shot S + read O per channel) propagated through WB gain, white clip, sqrt and the two-green average by Monte Carlo, then applies it: the guide's green-only decision uses the true green noise, and the operator subtracts (ref+comp) noise before rejecting — the paper's noise-corrected form, which lets the ×6 boost's variance test actually gate and stops calm textured areas from being over-rejected. OFF = noise-blind device match (most aggressive). When ON, the "Robustness Noise" slider below is ignored (the LUT is applied in full).
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        ispRow("Robustness Noise (0=device, 1=paper)", $cam.tuningParams.gcam_sabre_rob_noise, 0.0...1.0, "%.2f")
                        ispRow("Motion Boost (×)", $cam.tuningParams.gcam_sabre_boost, 1.0...16.0, "%.1f")
                        ispRow("Motion Boost Threshold (UV)", $cam.tuningParams.gcam_sabre_motion_thresh, 0.0...0.001, "%.5f")
                        ispRow("Green-Only Trigger", $cam.tuningParams.gcam_sabre_green_factor, 0.0...8.0, "%.2f")
                        ispRow("Strictness RGB (mult)", $cam.tuningParams.gcam_sabre_mult_rgb, 0.0...0.5, "%.3f")
                        ispRow("Strictness Green (mult)", $cam.tuningParams.gcam_sabre_mult_green, 0.0...1.0, "%.3f")
                        ispRow("Filter Var RGB (k)", $cam.tuningParams.gcam_sabre_k_rgb, 0.0...1.0, "%.4f")
                        ispRow("Filter Var Green (k)", $cam.tuningParams.gcam_sabre_k_green, 0.0...1.0, "%.4f")
                        Toggle("Biquadratic Guide Sampling (REJECTION_ONLY)", isOn: $cam.tuningParams.gcam_sabre_rejection_only)
                        Text("""
                             GCam's two BaseFrameRefColor/Rejection sampler variants, applied on EVERY path (guide-res, input-frame-res and merge-res). OFF = SampleBicubicAbsolute (the #else cubic B-spline) + the bicubic kFilterVarianceScale pair (0.0899 / 0.2117) — exactly GCam's default. ON = SampleBiquadraticAbsolute (the #ifdef REJECTION_ONLY Dodgson, "slightly sharper alternative to bicubic") + the biquadratic pair (0.0977 / 0.25). Sharper sampling grows the difference d at a given sub-pixel flow error, so slight misalignments reject more — at the cost of a little more over-rejection. Off by default (the exact non-REJECTION_ONLY match); A/B on an imported burst. Requires GCam SABRE Robustness.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        ispRow("Dilation Dead Zone", $cam.tuningParams.gcam_sabre_dilate_deadzone, 0.0...2.0, "%.2f")
                        Toggle("GCam 2× Dilation Downsample", isOn: $cam.tuningParams.gcam_sabre_dilate_2x)
                        Text("""
                             OFF (default) = byte-for-byte GCam: the recovered DilateMask shader is a plain same-resolution 5×5 dilate (gather at ±0.5/±1.5 texels, no 2× on the coordinate). ON adds a 2× downsample to a quarter-Bayer grid so each rejection decision governs a 2×2 block (coarser/blockier) — an earlier RE assumption, kept for A/B. Leave OFF to match GCam exactly; the input-frame-res path forces it off regardless.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Bilinear Upsample Mask to Output (guide-res)", isOn: $cam.tuningParams.gcam_sabre_bilinear_upscale)
                        Text("""
                             Keeps R and the 5×5 computed on the ~3 MP guide (the default operator, unchanged), but makes the merge sample that guide-res mask BILINEARLY per output pixel instead of nearest — so each of the ~48 MP output pixels gets its own interpolated rejection weight rather than one weight shared across a ~4×4 block. Zero extra memory (it's interpolated at fetch time, nothing is materialised). This is the lightweight way to get a per-output-pixel mask without the raw-res operator. Ignored when "Mask at Input-Frame Resolution" is on (that already has a per-raw-pixel mask). Off by default (nearest fetch); A/B on an imported burst. Requires GCam SABRE Robustness.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Extra N×N Minimum (darken mask)", isOn: $cam.tuningParams.gcam_sabre_min5x5)
                        Text("""
                             Applies one extra HARD N×N local-minimum pass on top of whatever the §7 refinement produced (on either the guide-res, input-frame-res, or merge-res mask): each pixel takes the LOWEST confidence weight in its N×N window. Because the minimum can never exceed the centre weight, turning this on can only DARKEN the mask — never brighten it — growing every rejection into its N×N neighbourhood for a more conservative merge. Use it when ghosts/edges are still being kept. Off by default; A/B on an imported burst. Requires GCam SABRE Robustness.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        HStack {
                            Text("Min Window Size")
                            Spacer()
                            Text("\(cam.tuningParams.gcam_sabre_min_window)×\(cam.tuningParams.gcam_sabre_min_window)")
                        }
                        Slider(value: Binding(
                                   get: { Double(cam.tuningParams.gcam_sabre_min_window) },
                                   set: { cam.tuningParams.gcam_sabre_min_window = Int($0.rounded()) }),
                               in: 3...31, step: 1)
                        Text("""
                             The size N of the hard local-minimum window used by BOTH the §7 minimum (on the input-frame / merge-res paths) and the "Extra N×N Minimum" pass above. 5 = the paper's literal 5×5. Larger (6, 10, 20…) spreads each rejection across a wider neighbourhood — darker, more conservative, better at sealing ghost edges but over-rejects more good pixels. Same logic, just a bigger window; computed separably so even 31×31 stays fast. Only affects the GCam path.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Mask at Input-Frame Resolution (Alg. 6)", isOn: $cam.tuningParams.gcam_sabre_raw_res)
                            .onChange(of: cam.tuningParams.gcam_sabre_raw_res) { on in
                                if on { cam.tuningParams.gcam_sabre_merge_res = false }
                            }
                        Text("""
                             ON BY DEFAULT. The 3×3 statistics are still computed on the ~3 MP guide, but they're then Dodgson-quadratic upsampled (reference unwarped, comparison upsampled AND warped by the flow) to the INPUT-FRAME (raw, ~12 MP) resolution, where R and the §7 refinement run — so the merge gets one robustness weight per raw pixel (≈ per 2×2 output pixels) instead of per ~4×4 block from the guide-res mask. 4× finer than guide-res, better-localized rejection at edges, fits in memory normally (~49 MB/frame, no band-streaming). §7 here is GCam's soft DilateMask at full input-frame res (the guide→quarter 2× downsample is turned off, since we're already finer). The comparison guide is prefiltered to match the reference, so aligned content stays fully kept. Turn OFF to revert to the pure guide-res operator (R + dilate on the ~3 MP guide, bilinear-upsampled by the merge). Note: forces the non-resident merge on device. Requires GCam SABRE Robustness.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Mask at Merge-Output Resolution (48 MP)", isOn: $cam.tuningParams.gcam_sabre_merge_res)
                            .onChange(of: cam.tuningParams.gcam_sabre_merge_res) { on in
                                if on { cam.tuningParams.gcam_sabre_raw_res = false }
                            }
                        Text("""
                             Gives every one of the ~48 MP output pixels its own rejection weight, but memory-safely: R and the N×N minimum are computed at the INPUT-FRAME resolution (~12 MP, the same ~49 MB/frame as "Mask at Input-Frame Resolution"), then the merge BILINEARLY upsamples that mask to each output pixel at fetch time. There is never a full 48 MP mask in memory (an earlier version materialised one ~195 MB mask per frame and crashed the app with out-of-memory). Because the statistics are guide-resolution either way, this interpolated per-output weight is numerically almost identical to literally computing R at 48 MP, without the RAM. MUTUALLY EXCLUSIVE with "Mask at Input-Frame Resolution" (turning either on turns the other off); the difference is this one adds the smooth per-output-pixel bilinear upsample. Off by default; A/B on an imported burst. Requires GCam SABRE Robustness.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("OIS During Burst", isOn: $cam.tuningParams.gcam_burst_ois)
                        Text("Keeps optical image stabilization working while the burst is taken. iOS only lets an app turn OIS on explicitly for bracketed captures, so the burst is taken as RAW brackets (as many frames per bracket as the camera allows) at exactly the locked exposure and ISO. After the burst the status line shows how many frames the camera reported OIS active for. Cameras without OIS bracket support capture as before. Off by default.")
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Save Reference Frame", isOn: $cam.tuningParams.gcam_save_reference)
                        Text("Saves a second image to Photos with the output: the reference frame on its own, run through the same merge at the same output size (48 MP at 2x) and saved the same way (DNG or JPG). Comparing it with the output shows exactly what the other frames added. Runs as a second pass after the output is written, so the burst takes longer (roughly one extra frame plus a second merge). Off by default.")
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Run operator on GPU", isOn: $cam.tuningParams.gcam_sabre_gpu)
                        if cam.tuningParams.gcam_sabre_gpu {
                            Toggle("Build Guide on GPU", isOn: $cam.tuningParams.gcam_sabre_gpu_guide)
                            Text("Builds each comparison frame's GCam guide on the GPU instead of the CPU (same math, same order; can differ from the CPU in the last bit). Much faster. Turn off to use the CPU guide.")
                                .font(.footnote).foregroundColor(.secondary)
                        }
                        Text("""
                             All defaults are GCam's measured constants, so the toggle alone gives the faithful model. Robustness Noise is kFilterVarianceScale's noise term: 0 matches the device (noise-blind, rejects hardest), 1 is the paper's full noise subtraction (more tolerant of sensor grain). Motion Boost is how much harder irregular-flow textured tiles are rejected (GCam = 6); Motion Boost Threshold is the local-motion-extent (in normalized UV) the flow must vary by before that ×6 boost arms (GCam = 0.00019). It's internally scaled by your raw tile size (× tile_size × 258) so the boost fires at the same flow GRADIENT GCam uses no matter how dense the flow grid is — important because Half-Tile / Dense flow make the tiles finer, which otherwise shrinks the measured motion and silently stops the boost from arming. LOWER it to arm the boost on smaller rotations/parallax (rejects those seams harder, but also fires on flow noise → more over-rejection); 0 = boost everywhere there's any motion variation. The two Strictness multipliers are GCam's 0.07 (RGB) / 0.35 (green-only) color_difference_multiplier — raise them to reject a detected misalignment harder across the board; the two Filter Var k's are 0.0899 / 0.2117. Green-Only Trigger is how much local green variance must exceed the green noise before a region switches to single-green comparison (GCam = 3; LOWER it to push more edges onto the stricter green-only path). Run operator on GPU moves the per-pixel loop to the Metal kernel (guides still build on the host; it falls back to the CPU operator if the kernel is unavailable) — the CPU path is the verified reference, so A/B them if in doubt. Google's constants are sensor-calibrated for the Pixel; tune on real imported bursts.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("GCam Merge Kernel (o_gauss_cov)", isOn: $cam.tuningParams.gcam_sabre_kernel)
                        Text("""
                             Builds the MERGE kernel from GCam's ComputeGuideImage structure tensor (ConstructGaussianCovariance) instead of our estimate_kernels — the second output of the guide shader, anisotropic sharpen-along-gradient / widen-along-edge. This is a MERGE change (sharpness/denoising), NOT the rejection mask. The six covariance constants are GCam's realshot2 2× capture and are PER-SHOT in GCam (derived from zoom × ISO), so they're only exact for that regime — re-tune for a different zoom/ISO. GCam's merge consumes the covariance as exp2(-0.5·offsetᵀ·rr·offset); ours uses exp, so the precision is scaled by ln2 and Kernel Scale calibrates the absolute size. Opt-in; A/B on an imported burst. Requires GCam SABRE Robustness.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        ispRow("Kernel Scale (o_gauss_cov)", $cam.tuningParams.gcam_cov_scale, 0.25...4.0, "%.3f")
                    }
                    Toggle("Paper-Exact R (guide-res → bilinear to merge grid)", isOn: $cam.tuningParams.robustness_guide_res_bilinear)
                    Text("""
                         The literal SIGGRAPH step 3: compute the robustness map R at the half-res (~3 MP) guide resolution and bilinearly upsample it to the 48 MP merge grid (the merge fetches R bilinearly). Default instead computes R at raw (12 MP) with a nearest fetch. KNOWN ISSUE: this is the mechanism that previously showed black tile-boundary lines; the cause isn't yet pinned (it is NOT geometry rejection — that's off by default). If you see tile-grid lines, please tell me exactly which other toggles are on and screenshot the mask so I can fix it. Off by default; A/B on an imported burst.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Bilinear Mask Fetch (merge)", isOn: $cam.tuningParams.merge_robustness_bilinear)
                    Text("""
                         Makes the merge sample the robustness mask R bilinearly instead of nearest/round — the paper's bilinear upsampling applied at fetch time, so it affects the 0↔1 rejection boundaries only. Softens those boundaries; on our full-resolution mask this was measured to re-admit faint ghosts at rejection edges (the outer ring of a rejected region gets a fractional weight instead of 0), which is why it's off by default. A/B against the default on the same imported burst.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Bicubic Guide Sampling (SABRE)", isOn: $cam.tuningParams.robustness_bicubic_guide)
                    Text("""
                         Samples the half-res guide statistics with a cubic B-spline (SABRE's SampleBicubic) when warping them for the robustness comparison, instead of our default Dodgson quadratic. Both are smooth sub-pixel samplers (no blocky nearest fetches); cubic B-spline is slightly higher-order and blurrier. Applied to the reference and comparison stats alike. The paper (Algorithm 6) specifies Dodgson, so this is a SABRE divergence. It's a sampling-smoothness choice — it does not reject sub-pixel misalignment, so don't expect it to remove edge ghosts; A/B it on the same imported burst. Off by default.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Global Homography (large roll)", isOn: $cam.tuningParams.global_homography_enabled)
                    Text("""
                         Experimental. Estimates one global homography per frame and warps it into the reference before alignment, so large camera roll/rotation is brought within the matcher's range before per-tile alignment. Helps heavily rolled handheld bursts; does not fix parallax or moving objects.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Global Pre-Alignment (ISA)", isOn: $cam.tuningParams.isa_prealign_enabled)
                    Text("ImageStackAlignator-style global pre-registration: an FFT phase-correlation scan estimates each frame's whole-image rotation and shift against the reference, refined to sub-pixel by a rigid Lucas-Kanade step, and warps it in before per-tile block matching. Removes global roll/shift so block matching only cleans up the residual. No gyro (search starts at 0°).")
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.isa_prealign_enabled {
                        ispRow("Roll Search (±°)", $cam.tuningParams.isa_prealign_rot_range_deg, 0.5...45.0, "%.1f")
                        Picker("Pre-Align Precision", selection: $cam.tuningParams.isa_prealign_refine_dim) {
                            Text("Balanced").tag(384)
                            Text("Precise").tag(640)
                            Text("Max").tag(1280)
                        }
                        Text("Resolution of the sub-pixel rigid refine. Higher = tighter global fit (esp. low light), slower. Balanced is the shipped speed.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                    Toggle("Gyro Pre-Alignment (IMU)", isOn: $cam.tuningParams.gyro_prealign_enabled)
                    Text("Uses the gyroscope to measure the camera's rotation between frames and seed the global pre-align directly, instead of estimating it from the image. Scene-independent, so it works where image estimation struggles — textureless walls, a moving subject, or motion blur — and handles large/fast rotation. The rigid refine still validates against the image. First pass: live (shutter) bursts only; needs on-device calibration.")
                        .font(.footnote).foregroundColor(.secondary)
                }

                Section(header: Text("Forward-Backward Consistency")) {
                    Toggle("FB Flow Check", isOn: $cam.tuningParams.flow_fb_consistency_enabled)
                    Text("""
                         Aligns each frame both ways (reference→frame and frame→reference) and, where the round trip disagrees, replaces that tile's motion vector with the median of its consistent neighbours. Catches bad block matches, occlusions and repeated-pattern lock that a one-way search can't see, improving the optical flow itself. Correctly aligned tiles — including the sub-pixel offsets super-resolution feeds on — are left untouched. Costs roughly double the alignment time.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.flow_fb_consistency_enabled {
                        ispRow("FB tolerance α", $cam.tuningParams.flow_fb_alpha, 0.0...0.5, "%.3f")
                        ispRow("FB floor β (px²)", $cam.tuningParams.flow_fb_beta, 0.0...4.0, "%.2f")
                        Text("Lower α/β flag more aggressively. Defaults keep sub-pixel offsets safe and flag only round-trip errors above ~1.5px.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Trajectory Consensus")) {
                    Toggle("Multi-Frame Consensus", isOn: $cam.tuningParams.flow_consensus_enabled)
                    Text("""
                         Cheng et al. (cryo-EM motion correction). As well as each frame's shift to the reference, it measures the shift between adjacent frames, then solves each tile's whole trajectory across the burst robustly. Because the direct shift (reference→frame) and the chained shift (reference→previous→frame) must agree, a single block-match blunder — a repeated-pattern mis-lock or an occlusion — stands out and is corrected to the consensus, while real parallax (consistent along both paths) is kept. Only tiles that are both redundantly measured and in clear disagreement are overridden; clean and unverifiable tiles keep their flow. Costs an extra coarse pass over the burst. Works alongside global pre-alignment (gyro/homography): the pre-pass applies the same pre-alignment, so it cleans up the residual field.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.flow_consensus_enabled {
                        ispRow("Reject above (grey px)", $cam.tuningParams.flow_consensus_reject_px, 1.0...8.0, "%.1f")
                        Text("A tile is corrected only when its flow and the consensus differ by more than this. Lower corrects more aggressively; too low can erode the sub-pixel offsets super-resolution feeds on.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Robustness")) {
                    Toggle("Per-Channel Mask (ISA)", isOn: $cam.tuningParams.robustness_per_channel)
                    Text("""
                         Builds a separate robustness mask for each guide colour channel (R, G, B) with per-channel noise statistics and Wiener shrink, the way ImageStackAlignator does, instead of one scalar mask summed across channels. Lets a coloured edge or a single-channel mismatch be rejected on the channel it actually affects. Needs the 3-channel RGB guide, so turning this on automatically switches the robustness guide off the single-channel full-res grey (it would otherwise have nothing to split) onto the half-res RGB guide ISA uses. Off = the 460-main single cross-channel mask.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Perceptual Mask (sqrt)", isOn: $cam.tuningParams.robustness_perceptual_sqrt)
                    Text("""
                         Computes the robustness mask in a perceptual space: the guide is mapped through √ before the local std and colour distance, with a matching √-space noise term. √ is the variance-stabilizing transform for the sensor's Poisson-Gaussian noise and also a rough perceptual curve, so one rejection threshold corresponds to one perceived / one-σ error at every brightness instead of being swamped by bright pixels and blind in the shadows. Aims to catch misalignments that are visible but numerically tiny in dark regions. Off by default. (Currently drives the CPU/harness mask; the on-device Metal mask needs the matching shader change — measure first.)
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Disable Noise Model", isOn: $cam.tuningParams.debug_noise_model_disabled)
                    Text("""
                         Diagnostic: zeroes the Poisson–Gaussian sensor-noise model as read by the robustness mask ONLY. R is then scored from the raw measured local variance and the raw (un-shrunk) pixel difference, with no noise floor subtracted — so a tile's colour difference is taken at face value instead of being forgiven as "within noise". Isolates whether a mismatch reads small because the noise model excused it or because the content genuinely is that flat. Alignment, kernels and SNR auto-tune are untouched. Leave OFF for normal use.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("NN Mask Refine (new)", isOn: $cam.tuningParams.rob_mask_nn_enabled)
                    Text("""
                         The freshly-trained darken-only mask net: a tiny 9-feature model that only attenuates the analytic mask (R·g, g in [0,1] — never trusts more), so the merge stays robust to misaligned tiles and only well-aligned samples contribute to super-resolution. Trained on synthetic bursts with deliberately corrupted alignment vectors; measured +6.4% merged-MSE on held-out synthetic data. Its decisive input is a MULTI-FRAME CONSENSUS — each frame vs the mean of all aligned frames — so enabling it runs a second alignment pass over the burst to build that consensus (opt-in, extra decode+align time). Features come from the full-res grey in the net's own training domain, so no retraining. Runs on device: the mask is refined on the host and handed to the GPU merge (no separate Metal kernel). Off by default.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Match Frame Brightness", isOn: $cam.tuningParams.exposure_match_enabled)
                    Text("""
                         Scales every frame to the reference's brightness before aligning, masking and merging, using each frame's own exposure metadata (shutter x ISO). Bursts whose exposure changes mid-burst (auto-exposure on the ZSL ring, imported bursts from other apps, e.g. 1/2000 s and 1/2200 s) otherwise look misaligned everywhere to the robustness masks. Bursts with locked exposure are left untouched. On by default.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Robust NN v21 (GPU)", isOn: $cam.tuningParams.robust_nn_enabled)
                    Text("""
                         Learned robustness mask (tools/robust_nn, v21): a per-pixel weight from 0 to 1 per frame that rejects samples whose misalignment would paint a visible ghost or doubled edge, and keeps correctly aligned samples — including aliased ones, which carry the super-resolution detail. It judges a misalignment by whether it would still be visible AFTER the burst is merged. v21 adds training on aliased fine text (perfectly aligned, and the same text misaligned), on whole-camera turns up to ~30° with the global homography, and on perfectly alignable bursts. On held-out bursts vs v17: aliased text wrongly rejected 7.6% vs 18.6% of its weight, at about the same misses on misaligned text (7.6% vs 7.4%); misalignments missed fewer on every other set (e.g. 6.6% vs 7.7% on real-scene bursts, 4.2% vs 7.4% under big rotation), at 0.6-1.5 points more good weight given up there. With Global Homography on, its motion inputs read the motion left after the homography, so large camera rotation no longer reads as misalignment. Same GPU kernel and memory as before. Off by default.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.robust_nn_enabled {
                        Toggle("Content-aware pixel net (v39)", isOn: $cam.tuningParams.robust_nn_v15_enabled)
                        Text("""
                             On top of v21, a small network that also sees the actual pixels of the reference and the aligned frame over a 17x17 neighbourhood, so a doubled, stepped or smeared edge is recognised as a shape rather than from per-pixel statistics. Trained on full 12 MP bursts of the same scene perfectly aligned and with small misalignments (a hand roll of 1-3 degrees that one translation per tile cannot follow, tile slips of 0.1-3 px, slipped small text) with the perfectly aligned twin protected. On held-out bursts vs v21: misaligned pixels kept 53% -> 23% on the roll residual, 35% -> 12% on tile slips, 13% -> 10% on slipped text; perfectly aligned 12 MP frames and aliased text given up 0.1% and 1.8% (v21 0.3% and 6.4%). On the real burst9 it keeps the same weight as v21 on texture and flat areas and rejects more at the misaligned car. Operating point +0.25 folded in; evaluated at Reject Below 0.20 / Trust Above 0.90. About 3x the mask step's GPU time and ~25 MB more memory on a 12 MP burst.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                        HStack {
                            Text("NN Reject Below")
                            Spacer()
                            Text(String(format: "%.2f", cam.tuningParams.robust_nn_reject_below))
                        }
                        Slider(value: $cam.tuningParams.robust_nn_reject_below, in: 0.0...0.6, step: 0.05)
                        HStack {
                            Text("NN Trust Above")
                            Spacer()
                            Text(String(format: "%.2f", cam.tuningParams.robust_nn_accept_above))
                        }
                        Slider(value: $cam.tuningParams.robust_nn_accept_above, in: 0.0...1.0, step: 0.05)
                        Text("""
                             Optional sharpening of the NN mask: output below "Reject Below" becomes 0 (sample dropped), above "Trust Above" becomes 1, linear in between. Defaults 0.20 / 0.90 give an exact 0-to-1 mask (v22 is tuned for them). Raise Reject Below (e.g. 0.3) to reject borderline misalignments more firmly; lower Trust Above (e.g. 0.7) to keep more samples in flat areas. Trust Above ≤ Reject Below = raw network output.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Forward-Backward Check")) {
                    Toggle("Forward-Backward Confidence", isOn: $cam.tuningParams.fb_conf_enabled)
                    Text("For every tile it measures the motion back from the comparison frame to the reference and checks that the round trip returns to where it started. The round-trip error e becomes a confidence exp(-e²/2τ²) that multiplies the rejection mask after everything else (Wronski or GCam, and the NNs): 0.05 px keeps everything, 0.3 px keeps ~92%, 1.5 px drops to ~14%, 4 px rejects. Only the error ACROSS edges counts (a slide along a straight edge is invisible), and the error sensor noise alone would cause is subtracted, so flat or dim areas are not rejected. Fast: a few sub-pixel steps per tile from the comparison side, no extra images.")
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.fb_conf_enabled {
                        ispRow("Tolerance τ (px)", $cam.tuningParams.fb_conf_tau_px, 0.0...2.0, "%.2f")
                        Text("Round-trip error at which the confidence falls to 61%. Lower = stricter. Default 0.75. At 0 every tile with any measurable round-trip error is rejected; only tiles that come back exactly (or whose error is within the noise) are kept.")
                            .font(.footnote).foregroundColor(.secondary)
                        Picker("Check Resolution", selection: $cam.tuningParams.fb_conf_subdiv) {
                            Text("Whole tile (16 px)").tag(1)
                            Text("2×2 pieces (8 px)").tag(2)
                            Text("4×4 pieces (4 px)").tag(4)
                        }
                        Text("How finely each tile is checked. Each piece is checked with its tile's motion vector, so finer pieces catch the parts of a tile the vector does not fit (a shifted edge at the end of a tile). Finer = more sensitive, slightly noisier. Default 4×4.")
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Count Slides Along Edges", isOn: $cam.tuningParams.fb_conf_along_edges)
                        Text("Also counts the part of the error ALONG an edge (applies to the Burst Consensus check too). Catches tiles that slid along a curved or textured edge, but along a straight, uniform edge that slide cannot be measured, so correctly aligned straight edges get rejected as well. Off = only the across-edge error counts, the along-edge part weighted by how much structure there is along the edge.")
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Full Backward Alignment", isOn: $cam.tuningParams.fb_conf_full)
                        Text("Measures the backward motion with a complete second alignment (comparison → reference) instead of the local check. Also catches wrong matches that look consistent locally (e.g. a slip by one period on a fence), at about twice the alignment time and extra memory.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Burst Consensus Check")) {
                    Toggle("Burst Consensus", isOn: $cam.tuningParams.fb_consensus_enabled)
                    Text("After every frame is aligned, checks each tile's shift in each frame against the OTHER frames of the burst. The camera's own motion is removed per frame; what is left (parallax, a moving object or an alignment error) must follow the smooth path the other frames and the reference trace over time. If the other frames disagree with a frame's shift there, that frame is down-weighted there: confidence exp(-d²/2τ²), multiplied into the rejection mask before the merge. Only disagreement ACROSS edges counts, noise is allowed for, and flat areas are left alone. Uses the motion vectors the alignment already produced: no extra alignment, no extra images. Works together with or without the Forward-Backward check.")
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.fb_consensus_enabled {
                        ispRow("Consensus Tolerance τ (px)", $cam.tuningParams.fb_consensus_tau_px, 0.0...2.0, "%.2f")
                        Text("Disagreement at which the confidence falls to 61%. Lower = stricter. Default 0.75.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                }

                Section(header: Text("High-Frequency Rejection")) {
                    Toggle("HF Artifact Rejection", isOn: Binding(
                        get: { cam.tuningParams.hf_artifact_removal_enabled },
                        set: { v in
                            cam.tuningParams.hf_artifact_removal_enabled = v
                            if v {
                                cam.tuningParams.gcam_unblocker_enabled = false   // exclusive with the SABRE Unblocker
                                cam.tuningParams.hfnn_enabled = false             // and with the HF-artifact NN
                                cam.tuningParams.mann_enabled = false             // and the misalignment NN
                            }
                        }))
                    Toggle("HF Artifact NN", isOn: Binding(
                        get: { cam.tuningParams.hfnn_enabled },
                        set: { v in
                            cam.tuningParams.hfnn_enabled = v
                            if v {
                                cam.tuningParams.hf_artifact_removal_enabled = false   // replaces the rule
                                cam.tuningParams.mann_enabled = false
                            }
                        }))
                    Toggle("Misalignment Artifact NN", isOn: Binding(
                        get: { cam.tuningParams.mann_enabled },
                        set: { v in
                            cam.tuningParams.mann_enabled = v
                            if v {   // replaces the HF rule, the HF NN and the SABRE Unblocker
                                cam.tuningParams.hf_artifact_removal_enabled = false
                                cam.tuningParams.hfnn_enabled = false
                                cam.tuningParams.gcam_unblocker_enabled = false
                            }
                        }))
                    Text("A tiny network (~2.6k weights) that runs only on tiles whose motion is irregular (the same flow threshold as GCam's x6 boost). There it low-pass filters the reference, the aligned frame and their difference, measures how much variance the filter removes, and from that plus the local edges decides whether the frame would leave a shifted edge, a doubled / ghosted edge or a stair of slightly offset tiles (the aperture problem). Replaces HF Artifact Rejection, the HF Artifact NN and the SABRE Unblocker. No extra alignment pass. Works with the Wronski and GCam masks. On held-out test bursts (edges, tiles with irregular motion): GCam catches 88% of visible misalignments (mask alone 65%) while wrongly rejecting 38% of good edges (39%); Wronski 88% (27%) at 37% (15%).")
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.mann_enabled {
                        ispRow("Motion Gate", $cam.tuningParams.mann_gate, 0.25...4.0, "%.2f")
                        Text("Multiplies the irregular-motion threshold that opens the network. Lower = more tiles are checked. Default 1.0.")
                            .font(.footnote).foregroundColor(.secondary)
                        ispRow("Strictness", $cam.tuningParams.mann_strictness, -2.0...2.0, "%.2f")
                        Text("Shifts the network's decision: + rejects more, - keeps more detail. 0 = trained default (edges: 88% caught / 37% good wrongly rejected); -0.5 ~ 74-80% / 21-26%; -1.0 ~ 54-72% / 10-16%; +0.5 ~ 92% / 50-55%.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                    Text("A tiny network (~1.4k weights) that replaces HF Artifact Rejection: only in fine repeated-pattern areas (fences, grids, brick, fabric, text) it decides whether each frame is misaligned there (wrong period lock, sliding along lines, tile-shaped steps) or correctly aligned fine detail, so well-aligned patterns still merge. Everywhere else the mask is untouched. Uses each frame's own alignment (no extra alignment pass). Works with the Wronski and GCam masks. Inactive until a trained model is built in.")
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.hfnn_enabled {
                        ispRow("Pattern Gate", $cam.tuningParams.hfnn_gate, 0.1...0.9, "%.2f")
                        Text("How strongly a spot must be repeated-pattern texture (variance loss) before the network decides there. Lower = more areas. Default 0.35.")
                            .font(.footnote).foregroundColor(.secondary)
                        ispRow("Strictness", $cam.tuningParams.hfnn_strictness, -2.0...2.0, "%.2f")
                        Text("Shifts the network's decision: + rejects more pattern misalignments but also more good pattern detail, - keeps more detail. 0 = the validated balance (on test patterns: GCam catches 82% of visible misalignments, rejecting 20% of good detail; +0.5 ~ 91% / 36%; -0.5 ~ 71% / 12%).")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                    Text("""
                         Wronski et al. Drops a frame's contribution only where BOTH hold: the reference patch is almost entirely high-frequency (most of its local detail vanishes under a blur, above Variance Loss) AND the per-tile motion is locally irregular (the same r_Mt test that drives the motion prior). Targets the blocky ghosts repetitive fine texture causes — roof tiles, fences, fabric weave, resolution charts — where block matching locks onto the wrong period, while leaving cleanly-tracked detail (hair, fur) and smooth noisy areas untouched. The reference frame stays the fallback. Off by default; it changes which pixels merge.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.hf_artifact_removal_enabled {
                        ispRow("Variance Loss", $cam.tuningParams.hf_variance_loss_threshold, 0.50...0.99, "%.2f")
                        Text("Reject above this variance-loss (how much local detail a blur destroys). Lower = rejects more. Reference points: flat wall ~0.04, face ~0.19, brick ~0.83, checkerboard ~0.92.")
                            .font(.footnote).foregroundColor(.secondary)
                        ispRow("Texture SNR", $cam.tuningParams.hf_min_texture_snr, 1.0...30.0, "%.1f")
                        Text("How far above the expected sensor noise a patch's detail must be before it can count as high-frequency texture (signal variance > this × noise variance, from the noise model). Lower = fires on fainter texture and, near 1, on noise itself; higher = only strong texture. Default 4.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Geometry Rejection")) {
                    Toggle("Reject Misaligned Motion", isOn: $cam.tuningParams.motion_geom_reject_enabled)
                    Text("""
                         Rejects pixels where the per-tile motion model is provably a poor fit: the within-tile motion variation (predicted from the flow field's gradient), weighted by the reference edge strength |∇I|, exceeds the threshold. Off by default. Affine-aware: with Affine Tile Motion on, only the residual beyond the measured per-tile warp counts, so the two features compose instead of double-rejecting. Runs on the shipping (raw-resolution) mask on both GPU and CPU.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.motion_geom_reject_enabled {
                        ispRow("Threshold |∇I|·|E|", $cam.tuningParams.motion_geom_reject_threshold, 0.0...0.06, "%.4f")
                        Text("Lower = rejects more (0.0045 default; ~0.02 rejects ~15%, 0.06 near-inert).")
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Exposure-Invariant (relative)", isOn: $cam.tuningParams.motion_geom_relative)
                        if cam.tuningParams.motion_geom_relative {
                            ispRow("Relative threshold", $cam.tuningParams.motion_geom_reject_threshold_relative, 0.0...0.2, "%.3f")
                            Text("Contrast-weighted variant added on top, so the same misalignment is caught at any light level.")
                                .font(.footnote).foregroundColor(.secondary)
                        }
                    }
                }

                Section(header: Text("Edge Misalignment Rejection")) {
                    Toggle("Reject Misaligned Edges", isOn: $cam.tuningParams.edge_misalign_enabled)
                    Text("""
                         Attenuates the merge weight only where the reference and the aligned comparison frame actually disagree at an edge — a measured sub-pixel shift along the edge normal, or a genuine doubled edge from motion. A correctly aligned edge, however sharp, is kept at full weight, so unlike geometry rejection it never thins or drops clean detail. Targets doubled and thickened edges from hand shake and moving subjects.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.edge_misalign_enabled {
                        ispRow("Edge SNR", $cam.tuningParams.edge_misalign_edge_snr, 1.0...12.0, "%.1f")
                        ispRow("Shift scale (px)", $cam.tuningParams.edge_misalign_shift_z, 0.3...5.0, "%.2f")
                        ispRow("Ghost scale", $cam.tuningParams.edge_misalign_ghost_z, 0.5...6.0, "%.2f")
                        ispRow("Min confidence", $cam.tuningParams.edge_misalign_min_conf, 0.0...1.0, "%.2f")
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .md3FormChrome()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(action: { showSettings = false }) {
                        Text("Done").font(.system(size: 17, weight: .semibold))
                    }
                    .foregroundColor(MD3.primary)
                }
            }
        }
        .preferredColorScheme(.dark)
    }
}

// MARK: - Focus reticle

/// Colours sampled from the reference GCam screen recording.
fileprivate enum GC {
    static let shutterDisc = Color(red: 233 / 255, green: 233 / 255, blue: 233 / 255)
    static let shutterPressed = Color(red: 231 / 255, green: 190 / 255, blue: 226 / 255)
    static let ring = Color(red: 249 / 255, green: 249 / 255, blue: 249 / 255)
    static let flipRing = Color(white: 210 / 255)
    static let pill = Color(red: 25 / 255, green: 24 / 255, blue: 26 / 255)
    static let iconGrey = Color(white: 200 / 255)
    static let textGrey = Color(white: 206 / 255)
    static let thermo = Color(red: 150 / 255, green: 205 / 255, blue: 217 / 255)
    static let badge = Color(red: 176 / 255, green: 212 / 255, blue: 220 / 255)
    static let lensSelected = Color(red: 172 / 255, green: 185 / 255, blue: 243 / 255)
    static let lensCapsule = Color.black.opacity(0.82)
    static let toggleCapsule = Color(red: 38 / 255, green: 37 / 255, blue: 35 / 255).opacity(0.92)
    static let toggleSelected = Color(red: 182 / 255, green: 186 / 255, blue: 206 / 255)
    static let toggleText = Color(red: 211 / 255, green: 211 / 255, blue: 207 / 255)
    static let progress = Color(red: 125 / 255, green: 162 / 255, blue: 178 / 255)   // merge arc
    static let capturingGrey = Color(white: 142 / 255)
    static let capsule = Color(red: 78 / 255, green: 81 / 255, blue: 87 / 255)
    static let chevron = Color(red: 214 / 255, green: 217 / 255, blue: 221 / 255)
    static let discRim = Color(red: 243 / 255, green: 246 / 255, blue: 250 / 255)
    static let discGradient = LinearGradient(
        colors: [Color(red: 236 / 255, green: 64 / 255, blue: 112 / 255),
                 Color(red: 216 / 255, green: 63 / 255, blue: 139 / 255),
                 Color(red: 150 / 255, green: 60 / 255, blue: 220 / 255)],
        startPoint: .topTrailing, endPoint: .bottomLeading)
}

/// The GCam shutter: 83 pt outer ring (3 pt), black gap, 61.5 pt disc. The disc
/// switches colour with no transition while pressed (or during the post-tap flash).
fileprivate struct GCamShutterStyle: ButtonStyle {
    var flash: Bool
    var locked: Bool
    /// While frames are being captured the ring and disc go grey (#8E8E8E), fading in
    /// over ~0.1 s and snapping back to white when capture ends.
    var capturing: Bool = false
    func makeBody(configuration: Configuration) -> some View {
        let pressed = configuration.isPressed || flash
        let disc: Color = locked ? Color.white.opacity(0.25)
            : (capturing ? GC.capturingGrey : (pressed ? GC.shutterPressed : GC.shutterDisc))
        return ZStack {
            Circle()
                .strokeBorder(capturing ? GC.capturingGrey : GC.ring.opacity(locked ? 0.35 : 1), lineWidth: 3)
                .frame(width: 83, height: 83)
                .animation(capturing ? .easeIn(duration: 0.1) : nil, value: capturing)
            Circle()
                .fill(disc)
                .frame(width: 61.5, height: 61.5)
                .animation(capturing ? .easeIn(duration: 0.1) : nil, value: capturing)
                .animation(nil, value: pressed)
            if locked {
                // A tap opens the unlock sheet rather than shooting.
                Image(systemName: "lock.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundColor(.white.opacity(0.8))
            }
        }
        .contentShape(Circle())
    }
}

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
