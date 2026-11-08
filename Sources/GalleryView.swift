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
private enum GalleryFilter: String, CaseIterable, Identifiable {
    case all = "All", raw = "RAW", burst = "Burst", videos = "Videos", favorites = "Favorites"
    var id: String { rawValue }
}

private enum GalleryTab { case photos, albums }

struct GalleryView: View {
    @Environment(\.presentationMode) private var presentationMode
    @State private var assets: [PHAsset] = []
    @State private var thumbs: [String: UIImage] = [:]
    @State private var rawIDs: Set<String> = []
    @State private var denied = false
    @State private var filter: GalleryFilter = .all
    @State private var tab: GalleryTab = .photos
    @State private var selectMode = false
    @State private var selection: Set<String> = []
    @State private var sortNewestFirst = true
    /// Wrapper rather than a retroactive Identifiable conformance on PHAsset,
    /// which belongs to another module and would warn under Swift 6.
    private struct Pick: Identifiable {
        let asset: PHAsset
        var id: String { asset.localIdentifier }
    }
    /// One month's worth of captures, for the Albums tab.
    private struct MonthSection: Identifiable {
        let id: String
        let title: String
        let assets: [PHAsset]
    }
    @State private var selected: Pick?

    private let manager = PHImageManager.default()

    // MARK: - Derived data

    private var visibleAssets: [PHAsset] {
        let base: [PHAsset]
        switch filter {
        case .all:       base = assets
        case .raw:       base = assets.filter { rawIDs.contains($0.localIdentifier) }
        case .burst:     base = assets.filter { $0.representsBurst }
        case .videos:    base = assets.filter { $0.mediaType == .video }
        case .favorites: base = assets.filter { $0.isFavorite }
        }
        return sortNewestFirst ? base : base.reversed()
    }

    // MARK: - Body

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if denied {
                emptyState(icon: "lock.fill",
                           title: "Photo Access Is Off",
                           message: "Enable photo library access in Settings to browse your captures here.")
            } else if assets.isEmpty {
                emptyState(icon: "photo.on.rectangle.angled",
                           title: "No Captures Yet",
                           message: "Photos you take with the app will appear here.")
            } else {
                VStack(spacing: 0) {
                    topBar
                    if tab == .photos { photosScroll } else { albumsScroll }
                }
            }

            if !assets.isEmpty && !denied && tab == .photos {
                HStack { Spacer(); dateRail.padding(.trailing, 3) }
                    .allowsHitTesting(false)
            }

            VStack { Spacer(); bottomBar }
        }
        .preferredColorScheme(.dark)
        .sheet(item: $selected) { pick in
            GalleryDetail(asset: pick.asset, onDelete: {
                thumbs.removeValue(forKey: pick.asset.localIdentifier)
                load()
            })
        }
        .onAppear(perform: load)
    }

    // MARK: - Top bar + chips

    private var topBar: some View {
        VStack(spacing: 12) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                headerTitle
                Spacer(minLength: 8)
                if selectMode {
                    if !selection.isEmpty {
                        Button { deleteSelected() } label: {
                            Image(systemName: "trash")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundColor(.red)
                                .frame(width: 34, height: 34)
                                .background(Circle().fill(Color.white.opacity(0.12)))
                        }
                    }
                    pill("Done") { selectMode = false; selection.removeAll() }
                } else {
                    pill("Select") { selectMode = true }
                }
                Menu {
                    Picker("Sort", selection: $sortNewestFirst) {
                        Text("Newest First").tag(true)
                        Text("Oldest First").tag(false)
                    }
                } label: {
                    Image(systemName: "line.3.horizontal.decrease")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(width: 34, height: 34)
                        .background(Circle().fill(Color.white.opacity(0.12)))
                }
            }
            if tab == .photos { chipsRow }
        }
        .padding(.horizontal, 16)
        .padding(.top, 6)
        .padding(.bottom, 12)
        .background(.ultraThinMaterial)
        .environment(\.colorScheme, .dark)
    }

    private var headerTitle: some View {
        let (day, date) = headerDate()
        return HStack(spacing: 7) {
            Text(day).font(.title3.weight(.bold)).foregroundColor(.white)
            if !date.isEmpty {
                Text("·").font(.title3).foregroundColor(.white.opacity(0.5))
                Text(date).font(.title3).foregroundColor(.white.opacity(0.5))
            }
        }
    }

    private func headerDate() -> (String, String) {
        if tab == .albums { return ("Albums", "") }
        guard let d = visibleAssets.first?.creationDate ?? assets.first?.creationDate else {
            return ("Gallery", "")
        }
        let cal = Calendar.current
        let day: String
        if cal.isDateInToday(d) { day = "Today" }
        else if cal.isDateInYesterday(d) { day = "Yesterday" }
        else { let f = DateFormatter(); f.dateFormat = "EEEE"; day = f.string(from: d) }
        let f2 = DateFormatter(); f2.dateFormat = "MMM d, yyyy"
        return (day, f2.string(from: d))
    }

    private var chipsRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(GalleryFilter.allCases) { f in
                    let on = (f == filter)
                    Button {
                        withAnimation(.easeOut(duration: 0.15)) { filter = f }
                    } label: {
                        Text(f.rawValue)
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(on ? .black : .white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Capsule().fill(on ? Color.white : Color.white.opacity(0.14)))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 1)
        }
    }

    private func pill(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                .background(Capsule().fill(Color.white.opacity(0.14)))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Photos (masonry)

    private var photosScroll: some View {
        GeometryReader { geo in
            ScrollView(showsIndicators: false) {
                masonry(width: geo.size.width)
                    .padding(.top, 10)
                    .padding(.bottom, 112)
            }
        }
    }

    private func masonry(width: CGFloat) -> some View {
        let side: CGFloat = 10
        let gap: CGFloat = 8
        let tileW = max(60, (width - side * 2 - gap) / 2)
        let cols = splitColumns(visibleAssets, tileW: tileW)
        return HStack(alignment: .top, spacing: gap) {
            column(cols.0, tileW: tileW)
            column(cols.1, tileW: tileW)
        }
        .padding(.horizontal, side)
        .frame(maxWidth: .infinity, alignment: .top)
    }

    private func column(_ items: [PHAsset], tileW: CGFloat) -> some View {
        LazyVStack(spacing: 8) {
            ForEach(items, id: \.localIdentifier) { asset in
                tileButton(asset, tileW: tileW)
            }
        }
    }

    private func tileButton(_ asset: PHAsset, tileW: CGFloat) -> some View {
        let h = tileW / clampedAspect(asset)
        return Button {
            if selectMode { toggle(asset) } else { selected = Pick(asset: asset) }
        } label: {
            GalleryTile(image: thumbs[asset.localIdentifier],
                        isRAW: rawIDs.contains(asset.localIdentifier),
                        isFavorite: asset.isFavorite,
                        isVideo: asset.mediaType == .video,
                        duration: asset.duration,
                        isBurst: asset.representsBurst,
                        selectable: selectMode,
                        selected: selection.contains(asset.localIdentifier),
                        width: tileW, height: h)
        }
        .buttonStyle(.plain)
        .onAppear { loadThumb(asset, targetW: tileW) }
    }

    /// Clamp the aspect ratio (w/h) so no tile is extremely tall or wide.
    private func clampedAspect(_ a: PHAsset) -> CGFloat {
        guard a.pixelHeight > 0 else { return 1 }
        let r = CGFloat(a.pixelWidth) / CGFloat(a.pixelHeight)
        return min(1.5, max(0.62, r))
    }

    /// Greedy height-balanced split into two masonry columns. Order depends only
    /// on fixed aspect ratios, so tiles never reflow as thumbnails load in.
    private func splitColumns(_ items: [PHAsset], tileW: CGFloat) -> ([PHAsset], [PHAsset]) {
        var a: [PHAsset] = [], b: [PHAsset] = []
        var hA: CGFloat = 0, hB: CGFloat = 0
        for it in items {
            let h = tileW / clampedAspect(it) + 8
            if hA <= hB { a.append(it); hA += h } else { b.append(it); hB += h }
        }
        return (a, b)
    }

    // MARK: - Albums (month sections)

    private var albumsScroll: some View {
        let three = Array(repeating: GridItem(.flexible(), spacing: 3), count: 3)
        return ScrollView(showsIndicators: false) {
            LazyVStack(alignment: .leading, spacing: 20) {
                ForEach(monthSections) { s in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(s.title)
                            .font(.headline).foregroundColor(.white)
                            .padding(.horizontal, 12)
                        LazyVGrid(columns: three, spacing: 3) {
                            ForEach(s.assets, id: \.localIdentifier) { a in
                                Button {
                                    if selectMode { toggle(a) } else { selected = Pick(asset: a) }
                                } label: {
                                    squareCell(a)
                                }
                                .buttonStyle(.plain)
                                .onAppear { loadThumb(a, targetW: 140) }
                            }
                        }
                        .padding(.horizontal, 10)
                    }
                }
            }
            .padding(.top, 10)
            .padding(.bottom, 112)
        }
    }

    private func squareCell(_ a: PHAsset) -> some View {
        ZStack {
            Color(white: 0.14)
            if let img = thumbs[a.localIdentifier] {
                Image(uiImage: img).resizable().scaledToFill()
            }
        }
        .aspectRatio(1, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(alignment: .bottomTrailing) {
            if selectMode {
                SelectionDot(selected: selection.contains(a.localIdentifier)).padding(5)
            }
        }
    }

    private var monthSections: [MonthSection] {
        let cal = Calendar.current
        var order: [String] = []
        var map: [String: [PHAsset]] = [:]
        for a in assets {
            let date = a.creationDate ?? .distantPast
            let c = cal.dateComponents([.year, .month], from: date)
            let key = "\(c.year ?? 0)-\(c.month ?? 0)"
            if map[key] == nil { order.append(key); map[key] = [] }
            map[key]?.append(a)
        }
        let fmt = DateFormatter()
        fmt.dateFormat = "LLLL yyyy"
        return order.map { key in
            let d = map[key]?.first?.creationDate ?? .distantPast
            return MonthSection(id: key, title: fmt.string(from: d), assets: map[key] ?? [])
        }
    }

    // MARK: - Right-edge date rail (decorative)

    private var dateRail: some View {
        HStack(spacing: 5) {
            Text("DATE")
                .font(.system(size: 9, weight: .semibold))
                .foregroundColor(.white.opacity(0.4))
                .rotationEffect(.degrees(90))
                .fixedSize()
                .frame(width: 12)
            VStack(spacing: 3) {
                ForEach(0..<20, id: \.self) { i in
                    Rectangle()
                        .fill(Color.white.opacity(i % 5 == 0 ? 0.45 : 0.18))
                        .frame(width: i % 5 == 0 ? 12 : 7, height: 1.5)
                }
            }
        }
        .frame(maxHeight: 320)
    }

    // MARK: - Bottom floating bar

    private var bottomBar: some View {
        HStack(spacing: 0) {
            barTab("Photos", active: tab == .photos) { tab = .photos }
            cameraButton.frame(width: 72)
            barTab("Albums", active: tab == .albums) { tab = .albums }
        }
        .padding(.vertical, 12)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.white.opacity(0.08), lineWidth: 0.5))
        .environment(\.colorScheme, .dark)
        .padding(.horizontal, 22)
        .padding(.bottom, 4)
    }

    private func barTab(_ title: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 5) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(active ? .white : .white.opacity(0.5))
                Circle().fill(Color.white).frame(width: 5, height: 5).opacity(active ? 1 : 0)
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.plain)
    }

    private var cameraButton: some View {
        Button { presentationMode.wrappedValue.dismiss() } label: {
            ZStack {
                Circle().fill(Color.white).frame(width: 54, height: 54)
                Image(systemName: "camera.fill")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(.black)
            }
        }
        .buttonStyle(.plain)
    }

    // MARK: - Empty state

    private func emptyState(icon: String, title: String, message: String) -> some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.08))
                    .frame(width: 84, height: 84)
                Image(systemName: icon)
                    .font(.system(size: 34, weight: .regular))
                    .foregroundColor(.white.opacity(0.6))
            }
            Text(title)
                .font(.title3.weight(.semibold))
                .foregroundColor(.white)
            Text(message)
                .font(.subheadline)
                .foregroundColor(.white.opacity(0.6))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 48)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Actions

    private func toggle(_ a: PHAsset) {
        let id = a.localIdentifier
        if selection.contains(id) { selection.remove(id) } else { selection.insert(id) }
    }

    private func deleteSelected() {
        let toDelete = assets.filter { selection.contains($0.localIdentifier) }
        guard !toDelete.isEmpty else { return }
        PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.deleteAssets(toDelete as NSArray)
        } completionHandler: { ok, _ in
            DispatchQueue.main.async {
                if ok { selection.removeAll(); selectMode = false; load() }
            }
        }
    }

    // MARK: - Photo library

    private func load() {
        PHPhotoLibrary.requestAuthorization(for: .readWrite) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async { denied = true }
                return
            }
            let opts = PHFetchOptions()
            opts.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            opts.predicate = NSPredicate(format: "mediaType == %d OR mediaType == %d",
                                         PHAssetMediaType.image.rawValue,
                                         PHAssetMediaType.video.rawValue)
            let fetched = PHAsset.fetchAssets(with: opts)
            var out: [PHAsset] = []
            fetched.enumerateObjects { a, _, _ in out.append(a) }
            // Flag which images are RAW (DNG) for the RAW filter and badge.
            var raws: Set<String> = []
            for a in out where a.mediaType == .image {
                let res = PHAssetResource.assetResources(for: a)
                if res.contains(where: { Self.isRawUTI($0.uniformTypeIdentifier) }) {
                    raws.insert(a.localIdentifier)
                }
            }
            DispatchQueue.main.async { assets = out; rawIDs = raws }
        }
    }

    private static func isRawUTI(_ uti: String) -> Bool {
        let u = uti.lowercased()
        return u.contains("raw") || u.contains("dng")
    }

    private func loadThumb(_ asset: PHAsset, targetW: CGFloat) {
        let id = asset.localIdentifier
        guard thumbs[id] == nil else { return }
        let scale = UIScreen.main.scale
        let target = CGSize(width: targetW * scale, height: targetW * scale * 1.6)
        let opts = PHImageRequestOptions()
        opts.deliveryMode = .opportunistic
        opts.isNetworkAccessAllowed = true
        manager.requestImage(for: asset,
                             targetSize: target,
                             contentMode: .aspectFill,
                             options: opts) { img, _ in
            if let img {
                DispatchQueue.main.async {
                    withAnimation(.easeOut(duration: 0.2)) { thumbs[id] = img }
                }
            }
        }
    }
}

/// A selection indicator dot for select-mode cells.
private struct SelectionDot: View {
    let selected: Bool
    var body: some View {
        ZStack {
            Circle().fill(selected ? Color.blue : Color.black.opacity(0.35))
            Circle().strokeBorder(Color.white, lineWidth: 1.5)
            if selected {
                Image(systemName: "checkmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.white)
            }
        }
        .frame(width: 22, height: 22)
    }
}

/// A masonry tile: rounded image, variable height, with RAW / favorite / burst /
/// video badges and an optional selection overlay.
private struct GalleryTile: View {
    let image: UIImage?
    let isRAW: Bool
    let isFavorite: Bool
    let isVideo: Bool
    let duration: TimeInterval
    let isBurst: Bool
    let selectable: Bool
    let selected: Bool
    let width: CGFloat
    let height: CGFloat

    var body: some View {
        ZStack {
            Color(white: 0.14)
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .transition(.opacity)
            }
        }
        .frame(width: width, height: height)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(alignment: .topLeading) { if isRAW { capsuleBadge { Text("RAW") } } }
        .overlay(alignment: .topTrailing) {
            if isBurst { capsuleBadge { Image(systemName: "square.on.square") } }
        }
        .overlay(alignment: .bottomLeading) {
            if isFavorite {
                Image(systemName: "heart.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white)
                    .shadow(color: .black.opacity(0.4), radius: 2)
                    .padding(8)
            }
        }
        .overlay(alignment: .bottomTrailing) {
            if isVideo && !selectable {
                capsuleBadge {
                    HStack(spacing: 3) {
                        Image(systemName: "play.fill")
                        Text(durationString(duration))
                    }
                }
            }
        }
        .overlay {
            if selectable {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.black.opacity(selected ? 0.3 : 0.001))
            }
        }
        .overlay(alignment: .bottomTrailing) {
            if selectable { SelectionDot(selected: selected).padding(8) }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
        )
        .contentShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func capsuleBadge<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .font(.caption2.weight(.bold))
            .foregroundColor(.white)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(.ultraThinMaterial, in: Capsule())
            .environment(\.colorScheme, .dark)
            .padding(8)
    }

    private func durationString(_ t: TimeInterval) -> String {
        let s = Int(t.rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
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
