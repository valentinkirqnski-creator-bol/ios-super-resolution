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
    /// Guards the one-time exposure setup in onAppear, which runs again on
    /// every scene change otherwise.
    @State private var didApplyLaunchShutter = false
    @State private var showImporter = false
    @State private var showGallery = false
    @State private var exposureParam: ExposureParam = .shutter

    /// Warm amber, used for every active/selected state. The chrome was built
    /// around a cyan close to the one Google Camera uses; keeping the layout
    /// but moving the accent and squaring off the chips is what separates this
    /// from looking like a clone of it.
    static let accent = Color(red: 0.98, green: 0.72, blue: 0.28)
    /// Squircle rather than a circle or a capsule, for the same reason.
    static var chipShape: RoundedRectangle { RoundedRectangle(cornerRadius: 9, style: .continuous) }

    var body: some View {
        GeometryReader { geo in
            // Three rows now: utilities and readouts, the exposure control, the
            // frame count. Sized to leave the same slack around the content as
            // the two-row version did, so the viewfinder gives up only what the
            // extra row costs.
            let topBarH: CGFloat = 126
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
            // the view made the magnified preview swallow taps on the chrome
            // around it.
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

    private func showFocusIndicator(at point: CGPoint) {
        focusPoint = point
        withAnimation(.easeOut(duration: 0.12)) { focusVisible = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            withAnimation(.easeOut(duration: 0.25)) { focusVisible = false }
        }
    }

    // MARK: - Top strip

    private var topStrip: some View {
        VStack(spacing: 10) {
            HStack(spacing: 14) {
                roundIconButton("gearshape.fill") { showSettings = true }
                // A stack of frames collapsing into one output is the closest
                // symbol to "merge several files into a larger image".
                roundIconButton("square.stack.3d.down.right.fill") { showImporter = true }
                formatButton
                Spacer()
                readout("ISO", cam.isoLabel, highlighted: exposureParam == .iso)
                readout("SEC", cam.shutterLabel, highlighted: exposureParam == .shutter)
            }

            exposureBar

            HStack(spacing: 16) {
                frameCountControl
                Spacer()
            }
        }
        .padding(.horizontal, 20)
    }

    /// Which parameter the exposure track is driving. Only one is editable at a
    /// time, so the track can be full width and the readouts stay legible --
    /// the arrangement most camera apps settle on.
    private enum ExposureParam { case shutter, iso }

    private func readout(_ caption: String, _ value: String, highlighted: Bool) -> some View {
        VStack(alignment: .trailing, spacing: 1) {
            Text(caption)
                .font(.system(size: 8, weight: .semibold, design: .monospaced))
                .foregroundColor(.white.opacity(highlighted ? 0.55 : 0.3))
            Text(value)
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .foregroundColor(highlighted ? Self.accent : .white.opacity(0.7))
        }
        .frame(minWidth: 46, alignment: .trailing)
    }

    /// Parameter button, track, auto button. Tapping the parameter button
    /// swaps which of the two the track drives; Auto is deliberately its own
    /// button rather than a third position, so leaving Auto never costs you the
    /// parameter you had selected.
    private var exposureBar: some View {
        HStack(spacing: 10) {
            Button {
                exposureParam = (exposureParam == .shutter) ? .iso : .shutter
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: exposureParam == .shutter ? "stopwatch" : "speedometer")
                        .font(.system(size: 13, weight: .medium))
                    Text(exposureParam == .shutter ? "SHUT" : "ISO")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                }
                .foregroundColor(cam.exposureIsAuto ? .white.opacity(0.5) : Self.accent)
                .frame(width: 68, height: 30)
                .background(Self.chipShape.fill(Color.white.opacity(0.07)))
                .overlay(Self.chipShape.strokeBorder(
                    (cam.exposureIsAuto ? Color.white.opacity(0.14) : Self.accent.opacity(0.55)),
                    lineWidth: 1))
            }
            .buttonStyle(.plain)

            exposureTrack

            Button {
                cam.setExposureAuto(!cam.exposureIsAuto)
            } label: {
                Text("A")
                    .font(.system(size: 13, weight: .heavy, design: .rounded))
                    .foregroundColor(cam.exposureIsAuto ? .black : .white.opacity(0.55))
                    .frame(width: 34, height: 30)
                    .background(Self.chipShape.fill(cam.exposureIsAuto
                                                    ? Self.accent
                                                    : Color.white.opacity(0.07)))
                    .overlay(Self.chipShape.strokeBorder(
                        cam.exposureIsAuto ? Color.clear : Color.white.opacity(0.14),
                        lineWidth: 1))
            }
            .buttonStyle(.plain)
            .disabled(cam.isBusy)
        }
    }

    /// Horizontal fader. The whole track is the hit target, and the very first
    /// touch moves the value -- the old vertical control needed 4pt of travel
    /// before it responded, and its handle was a Button that swallowed drags
    /// starting on it, which is exactly where a slider gets grabbed.
    private var exposureTrack: some View {
        let binding = exposureParam == .shutter ? $cam.shutterSlider : $cam.isoSlider
        return GeometryReader { g in
            let w = g.size.width
            let cap: CGFloat = 14
            let travel = max(1, w - cap)
            let x = cap / 2 + travel * CGFloat(binding.wrappedValue)
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.white.opacity(0.14))
                    .frame(height: 3)
                Capsule()
                    .fill(cam.exposureIsAuto ? Color.white.opacity(0.3) : Self.accent.opacity(0.8))
                    .frame(width: max(0, x - cap / 2), height: 3)
                RoundedRectangle(cornerRadius: 3, style: .continuous)
                    .fill(cam.exposureIsAuto ? Color.white.opacity(0.6) : Self.accent)
                    .frame(width: cap, height: 22)
                    .position(x: x, y: g.size.height / 2)
            }
            .frame(height: g.size.height)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { v in
                        guard !cam.isBusy else { return }
                        // Leave Auto before writing anything: the metering poll
                        // rewrites these values every 200ms while Auto is on, so
                        // a value written first can be overwritten before the
                        // mode change lands.
                        if cam.exposureIsAuto { cam.beginManualExposureDrag() }
                        guard !cam.exposureIsAuto else { return }
                        let t = (v.location.x - cap / 2) / travel
                        binding.wrappedValue = min(1, max(0, Double(t)))
                    }
            )
        }
        .frame(height: 30)
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
        let accent = Self.accent
        return VStack(spacing: 6) {
            Text(Self.zoomLabel(cam.zoomFactor))
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundColor(.black)
                .frame(width: 52, height: 30)
                .background(Self.chipShape.fill(accent))
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
            .background(RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Color.black.opacity(0.45)))
        }
    }

    private func lensChip(title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: selected ? .bold : .medium, design: .monospaced))
                .foregroundColor(selected ? .black : .white.opacity(0.92))
                .frame(width: 38, height: 34)
                .background(Self.chipShape.fill(selected
                                                ? Self.accent
                                                : Color.white.opacity(0.12)))
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
        Toggle("ICA Per Level In FFT Mode", isOn: $cam.tuningParams.align_ica_per_level_fft)
        Text("""
             Extends the above to the full-res FFT grey. Without it that path feeds \
             integer-only flow into a finest level that can only search +/-1 pixel, so the \
             correction budget is spent before it starts and tile-shaped displacements \
             survive. Coarse levels only -- the finest is refined either way -- which keeps \
             the extra reference gradients to about a quarter of a frame.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Ambiguous-Match Fallback", isOn: $cam.tuningParams.align_ambiguous_fallback_enabled)
        Text("""
             ImageStackAlignator's rule: when a tile's best and second-best block-match              costs are near-tied (flat patch, aperture problem, repeating texture -- no              precise shift can be determined), apply NO shift and keep the seed from the              coarser level, instead of trusting a match that is              indistinguishable from noise. Acts on the flow itself -- unlike the ambiguity              demotion in the robustness mask, which is inert under rotation because every              tile is already on the strict prior. Experimental -- A/B on rotating bursts.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Full-Res Flow Polish", isOn: $cam.tuningParams.align_fullres_polish)
        Text("""
             The decimate path measures every alignment stage, the final ICA included, on              the half-resolution grey -- so every residual error doubles in raw pixels.              This adds one last ICA refinement at FULL raw resolution on the band-limited              FFT grey (the exact image the full-res FFT mode measures on), seeded by the              finished decimate flow. The seed is already sub-pixel, so the pass can only              sharpen, not wander. Closes the decimate mode's sub-pixel accuracy gap to              full-res FFT at a fraction of its cost (~40ms GPU per frame, currently              hidden behind the CPU). Decimate mode only.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Anti-Aliased Decimation", isOn: $cam.tuningParams.grey_decimate_lowpass)
        Text("""
             The half-res alignment grey was a plain 2x2 quad average -- a weak low-pass              that lets fine texture between the half-res and full-res Nyquist fold back              into the image as aliasing, which contaminates block matching and ICA: flow              that wobbles with content instead of following motion (the wavy artifact).              This decimates through a proper anti-aliasing filter instead (half-phase              binomial 1-3-3-1, effective Gaussian sigma ~0.87 raw px): same lattice, same              channel balance, ~10x stronger alias suppression. Alignment grey only; the              robustness guide and the merge are untouched. Decimate mode only.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Subpixel Block Matching", isOn: $cam.tuningParams.bm_subpixel_quadratic)
        Text("""
             Fits a bivariate quadratic to the 3x3 cost neighbourhood around each              block-matching winner and adds its sub-cell minimum -- the sub-pixel              estimator Wronski's alignment specifies, which this port previously skipped:              block matching emitted integer flow at every level and ICA alone carried the              sub-pixel burden. The costs already exist, so the fit is nearly free. Applies              on both CPU and GPU search paths; matters most on the decimate grey, where              every residual ICA cannot recover is twice as large in raw pixels. The fit is              rejected (integer result stands) at window edges, on ridge-shaped cost              surfaces, and whenever it points more than half a cell away.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Overlapped Tile Merge (HDR+)", isOn: $cam.tuningParams.flow_overlap_merge)
        Text("The HDR+ scheme, as the reference author suggests trying first: alignment tiles of Ts=16 at stride 8 (50% overlap), each half-pitch position measured on its own full-tile window -- flow is NOT interpolated -- and the merge blends each output pixel's up to four covering-tile results with the raised-cosine window (a Hann crossfade at 50% overlap, summing to one). Where the tiles agree the hypotheses deduplicate and the merge costs exactly what it does today; where they disagree the results crossfade instead of the flow blending. Decimate mode only. Works with Smooth Tile Flow either on or off: the overlap merge reads each covering tile's vector nearest and never interpolates the flow itself, so with Smooth Tile Flow OFF you get a fully piecewise-constant field at half the tile pitch -- no interpolation anywhere, and half the rotation staircase of the plain nearest mode. The robustness mask follows the same grid in both cases. Experimental; off by default.")
            .font(.footnote)
            .foregroundColor(.secondary)
        Toggle("Bicubic Flow Sampling", isOn: $cam.tuningParams.flow_bicubic_sampling)
        Text("Samples the tile flow with Catmull-Rom bicubic (C1-smooth) instead of bilinear (C0). Removes the derivative kinks at tile centres in smooth regions; NOT expected to help at motion boundaries (flow is only piecewise smooth there -- Boundary Flow Selection is the fix for that). Catmull-Rom can overshoot slightly near sharp flow changes. Every consumer (merge, mask, warped statistics) switches together. Requires Smooth Tile Flow. Experimental; off by default.")
            .font(.footnote)
            .foregroundColor(.secondary)
        Toggle("Boundary Flow Selection", isOn: $cam.tuningParams.flow_boundary_selection)
        Text("""
             Bilinear flow sampling is right where motion is smooth but wrong at object              boundaries: it blends two different motions into flow that belongs to neither              side, exactly where robustness then rejects and detail is lost. This builds a              half-tile-pitch refinement after alignment: cells whose four surrounding tile              vectors agree keep the bilinear blend (smooth regions reproduce the coarse              sampling to within the flow field's own curvature -- exact for locally linear              motion, error far below the tile staircase this fixes), while cells at              a disagreement over 1 raw px get whichever single tile vector best explains              the alignment guide there. Mask and merge consume the same refined field, so              they stay in lockstep. Requires Smooth Tile Flow.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Smooth Tile Flow (bilinear)", isOn: $cam.tuningParams.flow_bilinear_sampling)
        Text("""
             Block matching produces ONE displacement per 16-pixel tile, and consuming it              nearest makes the warp piecewise constant -- v(x,y) = v_ij across each tile,              jumping at every boundary.
             For pure translation that is exact: every tile carries the same vector, so              there is nothing to jump. For ROTATION it is not -- the true field varies              continuously with position, so a per-tile constant is a staircase, stepping              by about theta x tile_size at each seam (0.28 raw px at 1 degree, 0.84 at 3).
             Those steps are sub-pixel, so Eq. 6 barely registers them: its 3x3 guide means              average over 6x6 raw pixels. But the eye detects DISCONTINUITY far more              readily than magnitude, so a sub-pixel error that flips at every tile boundary              reads as a grid, while the same error spread smoothly would not be seen. That              is why the artifact sits below the mask's threshold and above yours.
             This interpolates between the four surrounding tile-centre vectors in ALL              consumers together -- merge, Eq. 6's d, the upscaled/warped statistics, and              the raw-resolution mask. They switch together by necessity: the mask must              score the correspondence the merge actually fetches, so an interpolated merge              with a nearest mask would grade a fetch nobody performs.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Disable Noise Model (Robustness)", isOn: $cam.tuningParams.debug_noise_model_disabled)
        Text("""
             Debug: zeroes the noise model as read by the robustness mask ONLY. R is then              scored from the raw measured local variance and the raw (unshrunk) pixel              difference, isolating whether a tile's colour difference reads small because              the noise model forgave it, or because the content genuinely is that flat.              Unlike the earlier version of this switch, SNR auto-tune, the alignment tile              size and kernel estimation are untouched. Diagnostic only -- leave off.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Fast Merge Weights", isOn: $cam.tuningParams.merge_fast_weights)
        Text("Skips merge taps whose kernel weight is below 0.03% of the centre tap and overlapped-merge hypotheses carrying under 5% window weight. The normalisation absorbs both, so the output changes far below one 16-bit step; the merge kernel -- the largest GPU cost -- drops a measurable share of its exp() work.")
            .font(.footnote)
            .foregroundColor(.secondary)
        Toggle("DNG Highlight Headroom", isOn: $cam.tuningParams.dng_store_unwhitened)
        Text("""
             The merge runs with white balance baked into the pixels (R x2.06, B x1.84 on              this sensor), so the 16-bit DNG used to clip any red highlight above ~49% of              raw full scale -- about a stop of highlight headroom the sensor captured but              the file threw away, with a magenta cast where it clipped. This stores the              DNG un-white-balanced with a real AsShotNeutral instead: Lightroom and other              editors then apply WB in floating point and their highlight recovery sees              everything the sensor saw. The in-app JPEG and preview re-apply the gains on              load and render identically. Only the file's representation changes.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("fp16 Merge Accumulator", isOn: $cam.tuningParams.merge_fp16_accumulator)
        Text("""
             Stores the online merge accumulator as 16-bit floats instead of 32-bit.              All arithmetic stays float32 in the kernels; only what lands in memory              narrows. At 2x output this halves the pipeline's single largest allocation              (1116 -> 558 MB) and the merge's dominant memory traffic, which it is              bandwidth-bound on. The cost is storage quantisation of about 0.05%              relative per store -- roughly 1-2 LSB of the 16-bit output. Turn off to              restore bit-exact fp32 accumulation at the old memory and speed.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Zero-Floor Kernel Stretch", isOn: $cam.tuningParams.kernel_anisotropy_zero_floor)
        HStack {
            Text("Stretch Selectivity")
            Slider(value: $cam.tuningParams.kernel_stretch_gamma, in: 1.0...4.0, step: 0.25)
            Text(String(format: "%.2f", cam.tuningParams.kernel_stretch_gamma))
                .font(.caption.monospacedDigit())
                .foregroundColor(.secondary)
        }
        Text("Exponent on the stretch weight. 1.0 = plain zero-floor law. 2.0 (default) reproduces the readable-text stretch (~2:1 at text-like coherence) measured by hand-tuning k_stretch to 2, while clean single-orientation edges keep ~95% of full k_stretch. Raise further to confine elongation to only the most coherent edges.")
            .font(.footnote)
            .foregroundColor(.secondary)
        Text("""
             The merge kernel's stretch weight was 0.5*A with A never below 1 -- so even              near-isotropic detail in high-contrast areas was elongated at least              2.5:0.75 along whichever direction the tiny 2x2 structure-tensor window              happened to prefer. Distant text is the worst case: 1-2px multi-oriented              strokes give moderate coherence with a noise orientation, and the resulting              3-6:1 kernels smear glyphs unreadable or double their strokes (which looks              like misalignment). This remaps the weight to reach ZERO for isotropic              content while keeping the exact same stretch at the old A=1.95 threshold --              clean single-orientation edges keep their full elongation.
             """)
            .font(.caption2).foregroundColor(.secondary)
        Toggle("Continuous Kernel Anisotropy", isOn: $cam.tuningParams.kernel_anisotropy_continuous)
        Text("""
             Merge kernels are stretched ALONG edges so that a sample slightly off the              ideal position still lands inside the kernel -- Section 5.1.1 gives this the              job of increasing "tolerance for small misalignments and uneven coverage              around edges". Wronski drives the amount of stretch continuously from the              structure tensor; the reference implementation instead switched to the full              8:1 stretch only above anisotropy 0.9025 and used a perfectly ROUND kernel              below it.
             That penalises text most. A letter puts strokes at several orientations              inside one 3x3 window, so its structure tensor measures as nearly isotropic              -- around 0.6 -- even though it is all edges. Under the switch it fell below              the threshold and got a round 0.177 px kernel, while a long straight edge              beside it got 8:1. At 2x output that round kernel only accepts samples              within about a third of a raw pixel, so coverage along the strokes is sparse              and uneven, which reads as smeared letters.
             Continuous gives anisotropy 0.6 a 4.8:1 kernel instead of 1:1: still sharp              ACROSS the stroke, but covering along it. Both endpoints are unchanged --              isotropic content stays round, a perfect edge still gets the full 8:1 --              only the middle is filled in, and the middle is where text lives.
             """)
            .font(.caption2).foregroundColor(.secondary)
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

                    Toggle("Quad-Centre Kernel Lookup", isOn: $cam.tuningParams.kernel_lookup_quad_centre)
                    Text("Where a comparison frame fetches its steerable kernel from the covariance grid. ON (default, correct): (pos-0.5)/2 -- the centre of the Bayer quad, which is where the grid is actually sampled, and the same mapping the reference frame has always used. OFF: the legacy pos/2-0.5, which fetches the kernel 0.5 raw px away from the pixel being reconstructed. Because the grid is bilinearly interpolated, that offset blends neighbouring cells and rounds every kernel slightly toward isotropic -- which accidentally buys extra tolerance for residual alignment error. So turning this OFF can look smoother even though it is wrong: it hides misalignment rather than fixing it. Reference frames are unaffected either way. A/B on real captures; if OFF looks better, the alignment residual is what needs attention.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                    Toggle("Lossless DNG (smaller, faster)", isOn: $cam.tuningParams.dng_lossless_jpeg)
                    Text("Writes the output DNG with lossless-JPEG tiles (Compression 7, the standard DNG codec -- what Apple ProRAW uses). Pixels are bit-identical to the uncompressed file; size drops to roughly a third (varies with scene content) and saving is faster because far fewer bytes hit storage. Off = uncompressed strips as before.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                    Toggle("Manual D Thresholds", isOn: $cam.tuningParams.d_thresh_manual)
                    Text("D_th/D_tr decide super-resolution vs denoising per pixel: gradients below roughly D_tr*(1+D_th) sigmas of noise are denoised with a wide kernel instead of resolved. SNR Auto-Tune normally sets them per burst (0.71-0.81 / 1.0-1.24, GAT units); this override keeps your values while auto-tune still drives k_detail and tile size. Lower both (e.g. scale by 0.45: D_th 0.34, D_tr 0.50) to push daylight texture into the super-resolution branch; raise them if flat areas turn noisy.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                    HStack {
                        Text("Denoise Threshold (D_th)")
                        Spacer()
                        Text(String(format: "%.2f", cam.tuningParams.d_th))
                    }
                    Slider(value: $cam.tuningParams.d_th, in: 0.0...1.5)
                        .disabled(!cam.tuningParams.d_thresh_manual)
                    HStack {
                        Text("Denoise Transition (D_tr)")
                        Spacer()
                        Text(String(format: "%.2f", cam.tuningParams.d_tr))
                    }
                    Slider(value: $cam.tuningParams.d_tr, in: 0.001...2.0)
                        .disabled(!cam.tuningParams.d_thresh_manual)
                    Text("Higher shrink sharpens across edges (helps small text). Default 2.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
    }

    private var tuningSettingsView: some View {
        NavigationView {
            Form {
                robustnessSection
                
                kernelsSection
                
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

                    Toggle("Fast Burst Shutter (2x)", isOn: $cam.tuningParams.burst_fast_shutter)
                    Text("Burst frames expose at half the auto-metered duration with ISO raised to compensate: half the per-frame motion blur at the same brightness. The merge averages the extra noise back out across the burst -- blur it cannot undo. No effect in manual exposure mode.")
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
                    if cam.exportFormat == .jpg {
                        HStack {
                            Text("JPEG Quality")
                            Slider(value: $cam.jpegExportQuality, in: 0.5...1.0, step: 0.01)
                            Text(String(format: "%.2f", cam.jpegExportQuality))
                                .font(.caption.monospacedDigit())
                                .foregroundColor(.secondary)
                        }
                        Text("0.92+ keeps full-resolution colour (4:4:4). Below 0.90 iOS halves the colour resolution (4:2:0), which reads as soft, smeared fine detail -- the old default of 0.82 was why exports looked low quality.")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                    Text(cam.exportFormat == .dng
                         ? "LinearRaw DNG with embedded tone-mapped JPEG preview (Photos thumbnail; Lightroom reads the raw)."
                         : "JPEG rendered by the ISP: auto exposure, local tone mapping, contrast and vibrance. No sharpening.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }

                Section(header: Text("Rendering \u{2014} Tone")) {
                    Toggle("HDR Tone Mapping", isOn: $cam.tuningParams.isp_enabled)
                    Text(cam.tuningParams.isp_enabled
                         ? "Local tone mapping, contrast and vibrance, applied to the JPEG and the DNG preview only. The DNG itself always stays the unmodified linear merge."
                         : "Off: the legacy fixed-grade render is used instead.")
                        .font(.footnote).foregroundColor(.secondary)

                    if cam.tuningParams.isp_enabled {
                        ispRow("Exposure (EV)", $cam.tuningParams.isp_exposure_ev, -2.0...2.0, "%+.2f")
                        ispRow("Highlight Recovery", $cam.tuningParams.isp_highlight_knee, 0.60...1.0, "%.2f")
                        ispRow("Colour Noise", $cam.tuningParams.isp_chroma_denoise, 0.0...1.0)
                        ispRow("Colour Noise Radius", $cam.tuningParams.isp_chroma_radius, 2.0...48.0, "%.0f")
                        ispRow("Local Strength", $cam.tuningParams.isp_local_strength, 0.0...1.0)
                        ispRow("Highlight Rolloff", $cam.tuningParams.isp_highlight, 0.0...1.0)
                        ispRow("Shadow Lift", $cam.tuningParams.isp_shadow, 0.0...1.0)
                        ispRow("Black Point", $cam.tuningParams.isp_black_point, 0.0...0.20, "%.3f")
                        ispRow("Local Contrast", $cam.tuningParams.isp_local_contrast, 0.0...0.60)
                    }
                }

                if cam.tuningParams.isp_enabled {
                    Section(header: Text("Rendering \u{2014} Colour")) {
                        ispRow("Colour Strength", $cam.tuningParams.isp_colour_strength, 0.0...1.0)
                        ispRow("Contrast", $cam.tuningParams.isp_contrast, 0.0...1.0)
                        ispRow("Vibrance", $cam.tuningParams.isp_vibrance, 0.0...1.5)
                        ispRow("Saturation", $cam.tuningParams.isp_saturation, 0.5...1.5)
                        ispRow("Warmth", $cam.tuningParams.isp_warmth, -0.15...0.15, "%+.3f")
                        Toggle("Protect Skin Tones", isOn: $cam.tuningParams.isp_skin_protect)
                        Text("Holds back saturation in the skin hue band. There is no face detector, so this is what stops strong tone mapping turning skin orange.")
                            .font(.footnote).foregroundColor(.secondary)
                    }
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

                Section(header: Text("Robustness")) {
                    Toggle("Enable Robustness", isOn: $cam.tuningParams.robustness_enabled)
                    Text(cam.tuningParams.robustness_enabled
                         ? "Rejects misaligned regions before they merge. Leave on for normal shooting."
                         : "OFF — every frame merges at full weight everywhere. Ghosting and misalignment will be visible. Diagnostic only: it shows what alignment actually produced, with no mask hiding the errors.")
                        .font(.footnote)
                        .foregroundColor(cam.tuningParams.robustness_enabled ? .secondary : .orange)

                    Toggle("Save Robustness Mask", isOn: $cam.tuningParams.robustness_save_mask)
                    Text("When on, also saves a grayscale robustness mask to Photos after processing.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }

                Section(header: Text("Fallback Denoiser")) {

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
                        
                        Toggle("ICA Every Pyramid Level", isOn: $cam.tuningParams.align_ica_per_level)
                        Text("Refines sub-pixel alignment after block matching at every "
                             + "pyramid level, as the reference does, instead of only the "
                             + "finest. 2x2 decimate grey only unless the switch below is on.")
                            .font(.caption2).foregroundColor(.secondary)

                        fineAlignmentSection

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
