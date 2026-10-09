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

                Section(header: Text("Alignment")) {
                    Picker("Tile Size", selection: $cam.tuningParams.alignment_tile_size) {
                        Text("Auto (SNR)").tag(0)
                        Text("8").tag(8)
                        Text("16").tag(16)
                        Text("32").tag(32)
                        Text("64").tag(64)
                    }
                    Picker("Grey Method", selection: $cam.tuningParams.alignment_grey_fft) {
                        Text("FFT").tag(true)
                        Text("Decimate").tag(false)
                    }
                    Text("alignment.tile_size (SNR_based = auto picks 16/32/64) and alignment.grey_method. IPOL main defaults: SNR-based tiling, FFT grey.")
                        .font(.footnote).foregroundColor(.secondary)
                    Toggle("Global Pre-Alignment (ISA)", isOn: $cam.tuningParams.isa_prealign_enabled)
                    Text("ImageStackAlignator-style global pre-registration: an FFT phase-correlation scan estimates each frame's whole-image rotation and shift against the reference and warps it in before per-tile tracking. Helps large camera roll/pan. No gyro (the search starts at 0°). Not part of IPOL main; experimental.")
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.isa_prealign_enabled {
                        ispRow("Roll Search (±°)", $cam.tuningParams.isa_prealign_rot_range_deg, 0.5...15.0, "%.1f")
                        Text("How much camera ROLL the scan searches for. Raise it for large rotation; higher costs more (more angle samples). Translation range is handled automatically by the zero-padded correlation.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Robustness")) {
                    Toggle("Robustness", isOn: $cam.tuningParams.robustness_enabled)
                    Text("robustness.enabled — the per-pixel robustness mask (Alg. 6) that down-weights moving or misaligned frames during merge. Off merges every frame equally.")
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.robustness_enabled {
                        Toggle("Noise Correction", isOn: Binding(
                            get: { !cam.tuningParams.debug_noise_model_disabled },
                            set: { cam.tuningParams.debug_noise_model_disabled = !$0 }))
                        Text("robustness.noise_correction — fold the Poisson–Gaussian noise model into d/σ. Off scores R from the raw measured variance and difference.")
                            .font(.footnote).foregroundColor(.secondary)
                        Toggle("Save Robustness Mask", isOn: $cam.tuningParams.robustness_save_mask)
                        ispRow("t", $cam.tuningParams.r_t, 0.0...0.5, "%.3f")
                        ispRow("s1", $cam.tuningParams.r_s1, 0.0...4.0, "%.2f")
                        ispRow("s2", $cam.tuningParams.r_s2, 1.0...30.0, "%.1f")
                        ispRow("Mt", $cam.tuningParams.r_Mt, 0.0...4.0, "%.2f")
                        Text("robustness.t/s1/s2/Mt. IPOL main: t 0.12, s1 2, s2 12, Mt 0.8.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Merging")) {
                    Picker("Kernel Selection", selection: $cam.tuningParams.kernel_selection_linear) {
                        Text("Hard threshold").tag(false)
                        Text("Linear").tag(true)
                    }
                    ispRow("k_stretch", $cam.tuningParams.k_stretch, 1.0...8.0, "%.1f")
                    ispRow("k_shrink", $cam.tuningParams.k_shrink, 0.25...4.0, "%.2f")
                    Text("merging.selection_law (IPOL main default: linear) and merging.kernel k_stretch/k_shrink (main: 4 / 2). k_detail/k_denoise/D_th/D_tr are SNR-based (auto).")
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
