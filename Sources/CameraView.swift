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
                    Toggle("Save Robustness Mask", isOn: $cam.tuningParams.robustness_save_mask)
                }

                Section(header: Text("Robustness Mask")) {
                    Toggle("Robustness", isOn: $cam.tuningParams.robustness_enabled)
                    Text("""
                         The per-pixel robustness mask (Eq. 5–9) that down-weights moving or misaligned frames during merge. Off merges every frame equally — sharper on perfectly still scenes, but ghosts anything that moved.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.robustness_enabled {
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
                    Toggle("GCam-Style Robustness", isOn: $cam.tuningParams.robustness_gcam_enabled)
                    Text("""
                         An alternative robustness mask that reproduces the behaviour of Google's Super Res Zoom rejection (reverse-engineered design, original math): it normalizes the frame difference by local CONTRAST (not just noise) and raises its tolerance in high-frequency/aliased regions, so aliasing and real detail survive; a Wiener term shrinks noise-level differences; and it rejects much harder where the per-tile flow is IRREGULAR and the region is textured — the rotation/parallax/moving-object tile ghosts. Measured on synthetic real-content bursts to catch ~58% of visible ghosts at ~0% aliasing over-rejection (vs ~1% for the plain mask); the remaining subtle sub-pixel band still needs better flow or multi-frame consensus, not the mask. Rejected pixels fall back to the reference (robustness improves by rejecting, not by fusing — the hard regions get softer, not ghosted). Runs on the CPU raw-resolution path and is UNVERIFIED on device (heavier memory); constants tuned on synthetic bursts, will want on-device A/B. Off by default.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.robustness_gcam_enabled {
                        ispRow("Strictness (mult)", $cam.tuningParams.gcam_mult, 0.1...8.0, "%.2f")
                        ispRow("Dead Zone (thresh)", $cam.tuningParams.gcam_thresh, 0.0...4.0, "%.2f")
                        ispRow("Motion Boost", $cam.tuningParams.gcam_boost, 1.0...64.0, "%.1f")
                        ispRow("Texture Gate", $cam.tuningParams.gcam_tex_gate, 0.5...4.0, "%.2f")
                        ispRow("Irregularity Gate (px)", $cam.tuningParams.gcam_irr_gate, 0.0...3.0, "%.2f")
                        ispRow("Green-Diff Tolerance", $cam.tuningParams.gcam_green_k, 0.0...8.0, "%.2f")
                        ispRow("Soft Dilation", $cam.tuningParams.gcam_dilate_soft, 0.0...1.0, "%.2f")
                        Text("""
                             Tune on real bursts. Strictness/dead-zone set how aggressively any residual is rejected (lower mult or higher dead zone = gentler). Motion Boost + Texture/Irregularity gates aim the extra rejection at irregular-flow textured tiles (rotation/parallax ghosts) while leaving aligned pixels alone. Green-Diff Tolerance raises the allowed deviation in aliased detail so it survives (0 = off; it can only keep more, never darken). Soft Dilation controls how the rejection spreads to neighbours: 1.0 is the hard 5×5 min, lower softens it so a single rejected pixel doesn't erode clean detail around it. Google's own values are sensor-calibrated and won't match this camera — dial these in by eye.
                             """)
                            .font(.footnote).foregroundColor(.secondary)
                    }
                    Toggle("Paper-Exact R (guide-res + bilinear)", isOn: $cam.tuningParams.robustness_guide_res_bilinear)
                    Text("""
                         Computes the robustness mask at the half-resolution guide grid and bilinearly upsamples it to the merge grid — the exact SIGGRAPH 2019 scheme (step 3). The default instead computes R at full resolution with a hard (nearest) boundary, which we measured was +2.49 dB and rejects edge ghosts more completely. Turning this on matches the paper but tends to soften the mask and can re-admit faint edge ghosts (bilinear smooths the rejection boundary). A/B it against the default on the same imported burst. Off by default.
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

                Section(header: Text("High-Frequency Rejection")) {
                    Toggle("HF Artifact Rejection", isOn: $cam.tuningParams.hf_artifact_removal_enabled)
                    Text("""
                         Wronski et al. Drops a frame's contribution only where BOTH hold: the reference patch is almost entirely high-frequency (most of its local detail vanishes under a blur, above Variance Loss) AND the per-tile motion is locally irregular (the same r_Mt test that drives the motion prior). Targets the blocky ghosts repetitive fine texture causes — roof tiles, fences, fabric weave, resolution charts — where block matching locks onto the wrong period, while leaving cleanly-tracked detail (hair, fur) and smooth noisy areas untouched. The reference frame stays the fallback. Off by default; it changes which pixels merge.
                         """)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.hf_artifact_removal_enabled {
                        ispRow("Variance Loss", $cam.tuningParams.hf_variance_loss_threshold, 0.50...0.99, "%.2f")
                        Text("Reject above this variance-loss (how much local detail a blur destroys). Lower = rejects more. Reference points: flat wall ~0.04, face ~0.19, brick ~0.83, checkerboard ~0.92.")
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
