import SwiftUI
import Photos
import UIKit
import ImageIO
import UniformTypeIdentifiers

/// In-app gallery of captures saved by this app.
///
/// Reads from the photo library rather than keeping a private copy: results are
/// saved there and the local burst directory is deleted once processing
/// finishes, so the library is the only place they persist.
struct GalleryView: View {
    @Environment(\.presentationMode) private var presentationMode
    @State private var assets: [PHAsset] = []
    @State private var thumbs: [String: UIImage] = [:]
    @State private var denied = false
    /// Wrapper rather than a retroactive Identifiable conformance on PHAsset,
    /// which belongs to another module and would warn under Swift 6.
    private struct Pick: Identifiable {
        let asset: PHAsset
        var id: String { asset.localIdentifier }
    }
    @State private var selected: Pick?

    private let columns = [GridItem(.adaptive(minimum: 112), spacing: 8)]
    private let manager = PHImageManager.default()

    var body: some View {
        NavigationView {
            Group {
                if denied {
                    emptyState(icon: "lock.fill",
                               title: "Photo Access Is Off",
                               message: "Enable photo library access in Settings to browse your captures here.")
                } else if assets.isEmpty {
                    emptyState(icon: "photo.on.rectangle.angled",
                               title: "No Captures Yet",
                               message: "Photos you take with the app will appear here.")
                } else {
                    grid
                }
            }
            .navigationTitle("Gallery")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { presentationMode.wrappedValue.dismiss() } label: {
                        Text("Done").fontWeight(.semibold)
                    }
                }
            }
        }
        .sheet(item: $selected) { pick in
            GalleryDetail(asset: pick.asset, onDelete: {
                thumbs.removeValue(forKey: pick.asset.localIdentifier)
                load()
            })
        }
        .onAppear(perform: load)
    }

    private var grid: some View {
        ScrollView {
            HStack {
                Text("\(assets.count) \(assets.count == 1 ? "capture" : "captures")")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
            }
            .padding(.horizontal, 14)
            .padding(.bottom, 2)

            LazyVGrid(columns: columns, spacing: 8) {
                ForEach(assets, id: \.localIdentifier) { asset in
                    Button { selected = Pick(asset: asset) } label: {
                        GalleryCell(image: thumbs[asset.localIdentifier],
                                    megapixels: megapixels(asset))
                    }
                    .buttonStyle(PressableCard())
                    .onAppear { loadThumb(asset) }
                }
            }
            .padding(.horizontal, 10)
            .padding(.bottom, 12)
        }
    }

    private func emptyState(icon: String, title: String, message: String) -> some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(Color.secondary.opacity(0.12))
                    .frame(width: 84, height: 84)
                Image(systemName: icon)
                    .font(.system(size: 34, weight: .regular))
                    .foregroundColor(.secondary)
            }
            Text(title)
                .font(.title3.weight(.semibold))
            Text(message)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 48)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func megapixels(_ asset: PHAsset) -> Double {
        Double(asset.pixelWidth) * Double(asset.pixelHeight) / 1_000_000.0
    }

    private func load() {
        PHPhotoLibrary.requestAuthorization(for: .readWrite) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async { denied = true }
                return
            }
            let opts = PHFetchOptions()
            opts.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            opts.predicate = NSPredicate(format: "mediaType == %d", PHAssetMediaType.image.rawValue)
            let fetched = PHAsset.fetchAssets(with: opts)
            var out: [PHAsset] = []
            fetched.enumerateObjects { a, _, _ in out.append(a) }
            DispatchQueue.main.async { assets = out }
        }
    }

    private func loadThumb(_ asset: PHAsset) {
        guard thumbs[asset.localIdentifier] == nil else { return }
        let opts = PHImageRequestOptions()
        opts.deliveryMode = .opportunistic
        opts.isNetworkAccessAllowed = true
        manager.requestImage(for: asset,
                             targetSize: CGSize(width: 320, height: 320),
                             contentMode: .aspectFill,
                             options: opts) { img, _ in
            if let img {
                withAnimation(.easeOut(duration: 0.2)) {
                    thumbs[asset.localIdentifier] = img
                }
            }
        }
    }
}

/// One square thumbnail: rounded, with a gradient placeholder while loading and
/// a subtle megapixel badge that flags high-resolution super-res results.
private struct GalleryCell: View {
    let image: UIImage?
    let megapixels: Double

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color.secondary.opacity(0.18),
                                    Color.secondary.opacity(0.08)],
                           startPoint: .top, endPoint: .bottom)
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .transition(.opacity)
            } else {
                Image(systemName: "photo")
                    .font(.system(size: 22, weight: .light))
                    .foregroundColor(.secondary.opacity(0.5))
            }
        }
        .aspectRatio(1, contentMode: .fill)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(alignment: .bottomTrailing) {
            if megapixels >= 20 {
                Text("\(Int(megapixels.rounded())) MP")
                    .font(.caption2.weight(.semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(.ultraThinMaterial, in: Capsule())
                    .environment(\.colorScheme, .dark)
                    .padding(6)
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.06), lineWidth: 0.5)
        )
        .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

/// Press feedback for grid cells: a gentle scale-and-dim.
private struct PressableCard: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.95 : 1)
            .opacity(configuration.isPressed ? 0.85 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

/// Full-size view of one capture, with pinch-to-zoom so detail can actually be
/// inspected -- the point of a super-resolution result -- plus share, delete
/// and an info panel.
private struct GalleryDetail: View {
    let asset: PHAsset
    /// Called after a successful delete so the grid can refresh.
    var onDelete: () -> Void = {}
    @Environment(\.presentationMode) private var presentationMode
    @State private var image: UIImage?
    @State private var zoom: CGFloat = 1
    /// Zoom level when the current pinch began, so gestures accumulate.
    @State private var baseZoom: CGFloat = 1
    @State private var shareURL: URL?
    @State private var showShare = false
    @State private var showInfo = false
    @State private var busy = false

    var body: some View {
        NavigationView {
            ZStack {
                Color.black.ignoresSafeArea()
                if let image {
                    GeometryReader { geo in
                        ScrollView([.horizontal, .vertical]) {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                // Fit the image to the view at zoom 1, then grow
                                // the frame so the ScrollView provides panning once
                                // zoomed. Without an explicit frame the image lays
                                // out at its native pixel size (huge for a
                                // super-res capture) and opens zoomed-in.
                                .frame(width: geo.size.width * zoom,
                                       height: geo.size.height * zoom)
                        }
                        .gesture(MagnificationGesture()
                            .onChanged { zoom = max(1, min(8, baseZoom * $0)) }
                            .onEnded { _ in baseZoom = zoom })
                        // Double-tap toggles between fit and a 2.5x inspect view.
                        .onTapGesture(count: 2) {
                            withAnimation(.easeInOut(duration: 0.25)) {
                                zoom = zoom > 1 ? 1 : 2.5
                                baseZoom = zoom
                            }
                        }
                    }
                } else {
                    ProgressView()
                        .tint(.white)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(role: .destructive) { deleteAsset() } label: {
                        Image(systemName: "trash")
                    }
                    .disabled(busy)
                }
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Button { showInfo = true } label: { Image(systemName: "info.circle") }
                    Button { shareAsset() } label: { Image(systemName: "square.and.arrow.up") }
                        .disabled(busy)
                    Button { presentationMode.wrappedValue.dismiss() } label: {
                        Text("Done").fontWeight(.semibold)
                    }
                }
            }
            .immersivePhotoBar()
        }
        .onAppear(perform: loadImage)
        .sheet(isPresented: $showShare) {
            if let shareURL { ShareSheet(items: [shareURL]) }
        }
        .sheet(isPresented: $showInfo) {
            GalleryInfo(asset: asset)
                .mediumLargeDetents()
        }
    }

    private func loadImage() {
        let opts = PHImageRequestOptions()
        opts.deliveryMode = .highQualityFormat
        opts.isNetworkAccessAllowed = true
        PHImageManager.default().requestImage(
            for: asset,
            targetSize: PHImageManagerMaximumSize,
            contentMode: .aspectFit,
            options: opts) { img, _ in image = img }
    }

    /// Export the ORIGINAL file (DNG or JPG) to a temp URL and share it, so the
    /// recipient gets the real capture, not a re-encoded preview.
    private func shareAsset() {
        busy = true
        let opts = PHImageRequestOptions()
        opts.isNetworkAccessAllowed = true
        opts.version = .current
        PHImageManager.default().requestImageDataAndOrientation(for: asset, options: opts) { data, uti, _, _ in
            DispatchQueue.main.async {
                busy = false
                guard let data else { return }
                var ext = "jpg"
                if let uti, let e = UTType(uti)?.preferredFilenameExtension { ext = e }
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("FuzeFrame_\(UUID().uuidString).\(ext)")
                do {
                    try data.write(to: url)
                    shareURL = url
                    showShare = true
                } catch { }
            }
        }
    }

    /// Delete from the photo library. iOS shows its own confirmation prompt for
    /// a performChanges delete, so there is no custom alert here.
    private func deleteAsset() {
        busy = true
        PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.deleteAssets([asset] as NSArray)
        } completionHandler: { ok, _ in
            DispatchQueue.main.async {
                busy = false
                if ok {
                    onDelete()
                    presentationMode.wrappedValue.dismiss()
                }
            }
        }
    }
}

/// iOS 16-only cosmetic modifiers, no-ops on iOS 15 (the deployment target).
private extension View {
    /// Transparent nav bar with light controls, for the black photo viewer.
    @ViewBuilder func immersivePhotoBar() -> some View {
        if #available(iOS 16.0, *) {
            self.toolbarBackground(.hidden, for: .navigationBar)
                .toolbarColorScheme(.dark, for: .navigationBar)
                .tint(.white)
        } else {
            self
        }
    }
    /// Resizable info sheet; full-height by default on iOS 15.
    @ViewBuilder func mediumLargeDetents() -> some View {
        if #available(iOS 16.0, *) {
            self.presentationDetents([.medium, .large])
        } else {
            self
        }
    }
}

/// System share sheet (UIActivityViewController) for a file URL.
private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

/// Capture metadata: dimensions, date, file size, and EXIF (ISO / shutter /
/// camera / lens) read from the original file.
private struct GalleryInfo: View {
    let asset: PHAsset
    @Environment(\.presentationMode) private var presentationMode
    @State private var fileSize: Int?
    @State private var iso: Int?
    @State private var exposure: Double?
    @State private var model: String?
    @State private var lens: String?

    private var megapixels: Double {
        Double(asset.pixelWidth) * Double(asset.pixelHeight) / 1_000_000.0
    }

    var body: some View {
        NavigationView {
            List {
                Section("Image") {
                    infoRow("ruler", "Dimensions", "\(asset.pixelWidth) × \(asset.pixelHeight)")
                    infoRow("squares.below.rectangle", "Resolution",
                            String(format: "%.1f MP", megapixels))
                    if let fileSize {
                        infoRow("doc", "File Size", ByteCountFormatter.string(
                            fromByteCount: Int64(fileSize), countStyle: .file))
                    }
                    if let d = asset.creationDate {
                        infoRow("calendar", "Captured", dateString(d))
                    }
                }
                Section("Exposure") {
                    if let iso { infoRow("camera.aperture", "ISO", "\(iso)") }
                    if let exposure { infoRow("timer", "Shutter", shutterString(exposure)) }
                    if let model { infoRow("camera", "Camera", model) }
                    if let lens { infoRow("camera.metering.center.weighted", "Lens", lens) }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Info")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { presentationMode.wrappedValue.dismiss() } label: {
                        Text("Done").fontWeight(.semibold)
                    }
                }
            }
        }
        .onAppear(perform: load)
    }

    private func infoRow(_ icon: String, _ k: String, _ v: String) -> some View {
        HStack {
            Label(k, systemImage: icon)
                .labelStyle(.titleAndIcon)
                .foregroundColor(.primary)
            Spacer()
            Text(v)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }
    private func dateString(_ d: Date) -> String {
        let f = DateFormatter(); f.dateStyle = .medium; f.timeStyle = .short
        return f.string(from: d)
    }
    private func shutterString(_ s: Double) -> String {
        if s >= 1 { return String(format: "%.1fs", s) }
        return "1/\(Int((1.0 / s).rounded()))"
    }

    private func load() {
        let opts = PHImageRequestOptions()
        opts.isNetworkAccessAllowed = true
        opts.version = .current
        PHImageManager.default().requestImageDataAndOrientation(for: asset, options: opts) { data, _, _, _ in
            guard let data else { return }
            DispatchQueue.main.async { fileSize = data.count }
            guard let src = CGImageSourceCreateWithData(data as CFData, nil),
                  let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
            else { return }
            var _iso: Int?; var _exp: Double?; var _model: String?; var _lens: String?
            if let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any] {
                if let a = exif[kCGImagePropertyExifISOSpeedRatings] as? [NSNumber], let f = a.first {
                    _iso = f.intValue
                }
                if let e = exif[kCGImagePropertyExifExposureTime] as? NSNumber { _exp = e.doubleValue }
                _lens = exif[kCGImagePropertyExifLensModel] as? String
            }
            if let tiff = props[kCGImagePropertyTIFFDictionary] as? [CFString: Any] {
                _model = tiff[kCGImagePropertyTIFFModel] as? String
            }
            DispatchQueue.main.async { iso = _iso; exposure = _exp; model = _model; lens = _lens }
        }
    }
}
