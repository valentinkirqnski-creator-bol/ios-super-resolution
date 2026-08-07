import SwiftUI

struct CameraView: View {
    @StateObject private var cam = CameraModel()
    @Environment(\.scenePhase) private var scenePhase
    @State private var showViewer = false
    @State private var showSettings = false
    @State private var focusPoint: CGPoint?
    @State private var focusVisible = false
    /// True only while the user is actively dragging the shutter slider.
    @State private var shutterSliderDragging = false
    @State private var shutterSliderDragStart: Double?
    @State private var didApplyLaunchShutter = false
    /// Manual capture controls (burst length, shutter) shown above the
    /// viewfinder. Collapsible so the preview can fill the screen.
    @State private var showCaptureControls = true

    var body: some View {
        GeometryReader { geo in
            let topBarH: CGFloat = 88
            let bottomH: CGFloat = 96
            let vfWidth = geo.size.width
            let maxVFHeight = geo.size.height - topBarH - bottomH - geo.safeAreaInsets.bottom
            // Slightly taller than square (4:3) — uses more screen without ultra-wide chrome.
            let vfHeight = min(maxVFHeight, vfWidth * 4 / 3)

            ZStack {
                Color.black.ignoresSafeArea()

                if cam.permissionDenied {
                    permissionView
                } else {
                    VStack(spacing: 0) {
                        Group {
                            if showCaptureControls {
                                topStrip
                            } else {
                                Color.clear
                            }
                        }
                        .padding(.top, geo.safeAreaInsets.top + 4)
                        .frame(height: topBarH + geo.safeAreaInsets.top)
                        .background(Color.black)

                        viewfinder(width: vfWidth, height: vfHeight)

                        bottomPanel
                            .frame(height: bottomH + geo.safeAreaInsets.bottom)
                            .padding(.bottom, geo.safeAreaInsets.bottom)
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
        .sheet(isPresented: $showSettings) { tuningSettingsView }
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
                cam.toggleShutterAuto()
            } label: {
                VStack(spacing: 1) {
                    Text("A")
                        .font(.system(size: 13, weight: .bold))
                    Text("Auto")
                        .font(.system(size: 8, weight: .medium))
                }
                .foregroundColor(cam.shutterIsAuto ? .black : .white)
                .frame(width: 36, height: 36)
                .background(cam.shutterIsAuto ? Color.yellow : Color.white.opacity(0.15))
                .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .disabled(cam.isBusy)

            Slider(
                value: Binding(
                    get: { cam.shutterSlider },
                    set: { value in
                        guard !cam.isBusy else { return }
                        // Ignore programmatic AE sync updates while Auto is on.
                        guard shutterSliderDragging || !cam.shutterIsAuto else { return }
                        cam.shutterSlider = value
                    }
                ),
                in: 0...1,
                onEditingChanged: { editing in
                    guard !cam.isBusy else { return }
                    if editing {
                        // Finger down — do not leave Auto until the value actually moves.
                        shutterSliderDragging = true
                        shutterSliderDragStart = cam.shutterSlider
                    } else {
                        let start = shutterSliderDragStart
                        shutterSliderDragging = false
                        shutterSliderDragStart = nil
                        // Leave Auto only after a real user drag (not SwiftUI appear noise).
                        if let start, abs(cam.shutterSlider - start) > 0.002 {
                            cam.applyManualShutterFromSlider()
                        }
                    }
                }
            )
            .tint(.yellow)
            .opacity(cam.isBusy ? 0.4 : 1)
            .allowsHitTesting(!cam.isBusy)
        }
        .contentShape(Rectangle())
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
        VStack(spacing: 8) {
            HStack(spacing: 6) {
                if cam.availableCameras.contains(.ultraWide) {
                    lensChip(title: "0.5×", selected: cam.lensZoomMode == .ultraWide) {
                        cam.setLensZoom(.ultraWide)
                    }
                }
                if cam.availableCameras.contains(.wide) {
                    lensChip(title: "1×", selected: cam.lensZoomMode == .wide1x) {
                        cam.setLensZoom(.wide1x)
                    }
                    lensChip(title: "2×", selected: cam.lensZoomMode == .wide2x) {
                        cam.setLensZoom(.wide2x)
                    }
                }
                if cam.availableCameras.contains(.telephoto) {
                    lensChip(title: cam.telephotoLensLabel, selected: cam.lensZoomMode == .telephoto) {
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
                    Circle().fill(selected ? Color.white : Color.white.opacity(0.14))
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

            HStack {
                roundIconButton("gearshape.fill") { showSettings = true }
                Spacer()
                roundIconButton(showCaptureControls
                                ? "slider.horizontal.3"
                                : "slider.horizontal.below.rectangle") {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        showCaptureControls.toggle()
                    }
                }
            }
            .padding(.horizontal, 28)
            .padding(.top, 14)
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
        Button(action: { if cam.lastThumbnail != nil && !cam.isBusy { showViewer = true } }) {
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
                    .disabled(cam.lensZoomMode != .wide1x)
                    Text(cam.lensZoomMode == .wide1x
                         ? (cam.outputResolutionMode == .super48mp
                            ? "Super-resolves to 4x the pixel count. Slower and uses more memory."
                            : "Merges at sensor resolution. Faster.")
                         : "Only available on the 1x lens; other lenses always merge at sensor resolution.")
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
                        
                        HStack {
                            Text("Max Frame Count")
                            Spacer()
                            Text(String(format: "%.1f", cam.tuningParams.acc_rob_max_frame_count))
                        }
                        Slider(value: $cam.tuningParams.acc_rob_max_frame_count, in: 1.0...10.0)
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
