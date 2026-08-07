import SwiftUI
import Photos

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
        .sheet(item: $selected) { pick in GalleryDetail(asset: pick.asset) }
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
/// inspected -- the point of a super-resolution result.
private struct GalleryDetail: View {
    let asset: PHAsset
    @Environment(\.presentationMode) private var presentationMode
    @State private var image: UIImage?
    @State private var zoom: CGFloat = 1

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
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { presentationMode.wrappedValue.dismiss() }
                }
            }
        }
        .onAppear {
            let opts = PHImageRequestOptions()
            opts.deliveryMode = .highQualityFormat
            opts.isNetworkAccessAllowed = true
            PHImageManager.default().requestImage(
                for: asset,
                targetSize: PHImageManagerMaximumSize,
                contentMode: .aspectFit,
                options: opts) { img, _ in image = img }
        }
    }
}
