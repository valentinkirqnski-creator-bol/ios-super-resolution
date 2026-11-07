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

    private let columns = [GridItem(.adaptive(minimum: 104), spacing: 3)]
    private let manager = PHImageManager.default()

    var body: some View {
        NavigationView {
            Group {
                if denied {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 40, weight: .light))
                            .foregroundColor(.secondary)
                        Text("Photo library access is off")
                            .font(.headline)
                        Text("Enable it in Settings to browse your captures here.")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 40)
                    }
                } else if assets.isEmpty {
                    Text("No captures yet")
                        .foregroundColor(.secondary)
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 3) {
                            ForEach(assets, id: \.localIdentifier) { asset in
                                Button { selected = Pick(asset: asset) } label: {
                                    ZStack {
                                        Color.secondary.opacity(0.15)
                                        if let img = thumbs[asset.localIdentifier] {
                                            Image(uiImage: img).resizable().scaledToFill()
                                        }
                                    }
                                    .frame(height: 104)
                                    .clipped()
                                }
                                .buttonStyle(.plain)
                                .onAppear { loadThumb(asset) }
                            }
                        }
                        .padding(3)
                    }
                }
            }
            .navigationTitle("Gallery")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { presentationMode.wrappedValue.dismiss() }
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
                             targetSize: CGSize(width: 312, height: 312),
                             contentMode: .aspectFill,
                             options: opts) { img, _ in
            if let img { thumbs[asset.localIdentifier] = img }
        }
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
    @State private var shareURL: URL?
    @State private var showShare = false
    @State private var showInfo = false
    @State private var busy = false

    var body: some View {
        NavigationView {
            Group {
                if let image {
                    ScrollView([.horizontal, .vertical]) {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .scaleEffect(zoom)
                            .gesture(MagnificationGesture()
                                .onChanged { zoom = max(1, min(8, $0)) })
                    }
                } else {
                    ProgressView()
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
                    Button("Done") { presentationMode.wrappedValue.dismiss() }
                }
            }
        }
        .onAppear(perform: loadImage)
        .sheet(isPresented: $showShare) {
            if let shareURL { ShareSheet(items: [shareURL]) }
        }
        .sheet(isPresented: $showInfo) { GalleryInfo(asset: asset) }
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

    var body: some View {
        NavigationView {
            List {
                infoRow("Dimensions", "\(asset.pixelWidth) × \(asset.pixelHeight)")
                infoRow("Megapixels", String(format: "%.1f MP",
                    Double(asset.pixelWidth) * Double(asset.pixelHeight) / 1_000_000.0))
                if let d = asset.creationDate { infoRow("Captured", dateString(d)) }
                if let fileSize { infoRow("File size", ByteCountFormatter.string(
                    fromByteCount: Int64(fileSize), countStyle: .file)) }
                if let iso { infoRow("ISO", "\(iso)") }
                if let exposure { infoRow("Shutter", shutterString(exposure)) }
                if let model { infoRow("Camera", model) }
                if let lens { infoRow("Lens", lens) }
            }
            .navigationTitle("Info")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { presentationMode.wrappedValue.dismiss() }
                }
            }
        }
        .onAppear(perform: load)
    }

    private func infoRow(_ k: String, _ v: String) -> some View {
        HStack { Text(k); Spacer(); Text(v).foregroundColor(.secondary) }
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
