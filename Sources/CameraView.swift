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

    var body: some View {
        GeometryReader { geo in
            let topBarH: CGFloat = 88
            let exposureBarH: CGFloat = 58
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
            let maxVFHeight = geo.size.height - topBarH - exposureBarH - bottomH - bottomInset
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

                        exposureBar
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

            // ISO/shutter controls live in the exposure bar ABOVE the viewfinder
            // (see `exposureBar`), so nothing covers the preview here.

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

    private var topStrip: some View {
        VStack(spacing: 8) {
            HStack(spacing: 14) {
                roundIconButton("gearshape.fill") { showSettings = true }
                // A stack of frames collapsing into one output is the closest
                // symbol to "merge several files into a larger image".
                roundIconButton("square.stack.3d.down.right.fill") { showImporter = true }
                Spacer()
            }

            HStack(spacing: 14) {
                frameCountControl
                resolutionControl
                formatControl
                Spacer()
                // ISO/shutter now live in the exposure bar below the top strip,
                // so they are not duplicated here.
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
                    // No hard-coded 2× chip: the lens buttons reflect only the
                    // physical lenses this device actually has (ultra-wide 0.5×,
                    // wide 1×, and the telephoto's native factor), so a base
                    // iPhone shows 0.5/1 and a Pro shows 0.5/1/<tele>. The 2×
                    // sensor-crop remains reachable via the zoom slider.
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
            freeTierBanner
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

    /// True when a non-paying user has used the day's free captures.
    private var outOfFreeCaptures: Bool {
        !store.isUnlocked && store.photosRemaining == 0
    }

    private var shutterButton: some View {
        Button(action: {
            if cam.isBusy { return }
            // Out of free captures: don't shoot, offer the unlock instead.
            if outOfFreeCaptures { showPaywall = true; return }
            cam.captureBurst()
            store.registerCapture()
        }) {
            ZStack {
                Circle()
                    .strokeBorder(Color.white.opacity(cam.isBusy || outOfFreeCaptures ? 0.35 : 1), lineWidth: 5)
                    .frame(width: 78, height: 78)
                Circle()
                    .fill(outOfFreeCaptures ? Color.white.opacity(0.25) : shutterFill)
                    .frame(width: cam.isCapturing ? 50 : 58, height: cam.isCapturing ? 50 : 58)
                    .animation(.spring(response: 0.22, dampingFraction: 0.6), value: cam.isCapturing)
                if outOfFreeCaptures {
                    // The shutter is "disabled" for capture — a lock marks that a
                    // tap now opens the unlock sheet rather than shooting.
                    Image(systemName: "lock.fill")
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundColor(.white.opacity(0.8))
                }
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
             Evaluates the robustness mask at raw Bayer resolution instead of the              half-resolution guide grid: the guide-resolution local statistics are              Dodgson-upscaled and flow-warped to every raw pixel, and R is computed there,              so the rejection boundary lands with raw-pixel precision instead of in 2x2              Bayer blocks. The 5x5 local-min is applied twice (= 9x9 raw), preserving the              paper's ~10x10-raw physical safety margin that s/t/Mt were tuned against,              while the boundary stays raw-precision. The statistics themselves stay              half-resolution either way. Only takes effect with "Alignment Grey: FFT" below              turned OFF (Decimate) -- silently does nothing otherwise. ~4x the pixel count              for the mask itself.
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

                Section(header: Text("JPG Look")) {
                    ispRow("Vibrance", $cam.tuningParams.hdr_vibrance, 0...1)
                    Text("""
                         Saturation boost weighted toward muted colours and faded \
                         out in the brightest tones, so a highlight is never \
                         re-saturated. 0.40 by default.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    ispRow("Black level", $cam.tuningParams.hdr_black_percentile,
                           0...0.05, "%.3f")
                    Text("""
                         Fraction of the picture taken all the way to black: \
                         0.002 by default, so about one pixel in five hundred. \
                         Measured per shot rather than a fixed offset, and still \
                         capped, so a low-key scene keeps its shadows. The range \
                         runs to 0.05, where the render itself clamps it; across \
                         that span it stays monotone, taking mean luma from 126 \
                         down to 106.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Text("Both apply to the JPG export and to the preview Photos "
                         + "shows for a DNG, from the next shot on.")
                        .font(.footnote).foregroundColor(.secondary)
                }

                Section(header: Text("Alignment")) {
                    Picker("Tile size", selection: $cam.tuningParams.alignment_tile_size) {
                        Text("Auto").tag(0)
                        Text("8").tag(8)
                        Text("16").tag(16)
                        Text("32").tag(32)
                        Text("64").tag(64)
                    }
                    .pickerStyle(.segmented)
                    Text("""
                         Block-matching tile size in raw pixels, for the finest \
                         pyramid level; the coarsest is half that. Auto is the \
                         default and is what Python 1.4 does: it picks from the \
                         reference frame's SNR -- 64 at or below 14, 32 at or \
                         below 22, otherwise 16 -- so a dim scene gets bigger \
                         tiles on its own. A fixed value overrides that.

                         BIGGER tiles match more pixels at once, so the flow \
                         they return is more reliable on smooth or noisy \
                         content -- which is what makes flat sky and dim scenes \
                         misbehave -- at the cost of resolving less of the real \
                         local motion. Smaller tiles track fine motion better \
                         and are noisier about it.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Match Python 1.4 alignment",
                           isOn: $cam.tuningParams.align_match_14)
                    Text("""
                         On by default. Closes the three places this port (a \
                         460-main derivative) diverged from 1.4: the finest \
                         search radius becomes 1 rather than 3, flow is carried \
                         between pyramid levels by a plain bilinear resize \
                         rather than 460's three-candidate re-match, and ICA \
                         runs at every level rather than only the finest. It \
                         also lets Auto choose a 64px tile, which 460 capped at \
                         32. Off restores the 460 behaviour, so a regression is \
                         one tap from being ruled out.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                }

                Section(header: Text("Motion Rejection")) {
                    Toggle("Noise-aware geometry gradient",
                           isOn: $cam.tuningParams.motion_geom_denoise_gradient)
                    Text("""
                         Estimates the reference gradient used by the geometric \
                         motion-rejection test with a noise-aware 3x3 operator \
                         instead of a bare difference, so photon noise in a dark \
                         scene is less likely to read as an edge and reject a \
                         well-aligned tile. The rejection threshold is unchanged, \
                         and pixels whose gradient is already strong are left \
                         exactly as they were, so bright scenes behave as before.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    ispRow("Geometry reject threshold",
                           $cam.tuningParams.motion_geom_reject_threshold,
                           0.0005...0.02, "%.4f")
                    Text("""
                         The geometric test rejects a tile where |gradient| \
                         x |within-tile motion error| exceeds this. LOWER \
                         rejects more, so a dim or noisy scene can lose \
                         frames to it; higher is more permissive. 0.0045 by \
                         default; 0.02 was the original value.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Save robustness mask",
                           isOn: $cam.tuningParams.robustness_save_mask)
                    Text("Writes the per-frame robustness mask alongside the shot "
                         + "as extra images. Diagnostic; leave off for normal use.")
                        .font(.footnote).foregroundColor(.secondary)
                }

                Section(header: Text("Relative Rejection Criterion")) {
                    Toggle("Exposure-invariant criterion",
                           isOn: $cam.tuningParams.motion_geom_relative)
                    Text("""
                         A second geometric criterion applied ON TOP of the \
                         absolute one, using contrast -- gradient over \
                         brightness, with a noise floor subtracted -- instead \
                         of absolute gradient. It catches the low-light \
                         misalignments the absolute form misses, whose gradient \
                         shrinks with the light.

                         The two are a UNION, so this can only ADD rejections, \
                         never restore a frame. If your problem is flat sky \
                         going black in the mask, this will not help and may \
                         make it worse.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    ispRow("Relative threshold",
                           $cam.tuningParams.motion_geom_reject_threshold_relative,
                           0.005...0.2, "%.3f")
                    ispRow("Noise floor multiplier",
                           $cam.tuningParams.motion_geom_noise_floor_mult,
                           0...16, "%.1f")
                    Text("""
                         Threshold 0.04 by default; lower rejects more. The \
                         noise floor multiplier is how many sigma of guide \
                         noise are subtracted from the gradient before the \
                         contrast ratio is formed, so higher discounts noise \
                         harder and rejects less. 1.5 by default, up to 16; past \
                         about 5 sigma it subtracts more than a real dark edge carries, \
                         so the criterion goes quiet altogether. 0 disables \
                         the subtraction. Both are inert while the toggle \
                         above is off.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
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
