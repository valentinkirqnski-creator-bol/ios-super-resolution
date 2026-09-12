import Foundation
import StoreKit

// In-app purchase + free-tier gate.
//
// Business rules (confirmed with the product owner):
//   * One non-consumable purchase, `unlimitedProductID`, priced at $1.99/$2 tier,
//     permanently unlocks unlimited capture.
//   * Without it, the free tier allows `freeDailyLimit` completed captures per
//     calendar day (local time); a burst merge counts as one. The count resets
//     at local midnight. When it reaches the limit the shutter is disabled and
//     the paywall is offered.
//
// SETUP REQUIRED (cannot be done from code):
//   1. App Store Connect → create a Non-Consumable IAP with product id
//      `unlimitedProductID` below (change the string to match your bundle).
//   2. Add a StoreKit configuration file to the Xcode scheme for local testing
//      (Product ▸ Scheme ▸ Edit Scheme ▸ Run ▸ Options ▸ StoreKit Configuration),
//      with the same product id, so purchases work in the simulator/sandbox.
@MainActor
final class StoreManager: ObservableObject {
    // Shared so both the capture UI and the paywall observe one source of truth.
    static let shared = StoreManager()

    /// Must match the product id created in App Store Connect. Defaulted to the
    /// app's bundle-id prefix (com.handheldsr.camera) — change it there and here
    /// together if you use a different id.
    static let unlimitedProductID = "com.handheldsr.camera.unlimited"
    /// Free captures per local day before the shutter locks.
    static let freeDailyLimit = 5

    @Published private(set) var isUnlocked = false
    @Published private(set) var product: Product?
    @Published private(set) var purchaseInFlight = false
    @Published private(set) var photosUsedToday = 0
    @Published var lastErrorMessage: String?

    private let usedKey = "FreePhotosUsedCount"
    private let dayKey  = "FreePhotosDayStamp"
    private var updatesTask: Task<Void, Never>?

    private init() {
        rolloverIfNeeded()
        photosUsedToday = UserDefaults.standard.integer(forKey: usedKey)
        // Listen for transactions that arrive outside an explicit purchase()
        // (Ask to Buy approvals, purchases made on another device, refunds).
        updatesTask = Task { [weak self] in
            for await update in Transaction.updates {
                await self?.handle(verification: update)
            }
        }
        Task { await loadProduct(); await refreshEntitlement() }
    }

    deinit { updatesTask?.cancel() }

    // MARK: Free-tier accounting

    /// Captures still allowed today for a non-paying user (∞ shown as a large int
    /// is avoided; callers check `isUnlocked` first).
    var photosRemainingToday: Int { max(0, Self.freeDailyLimit - photosUsedToday) }

    /// True if a capture is allowed right now.
    var canCapture: Bool { isUnlocked || photosRemainingToday > 0 }

    /// Call exactly once per capture that actually starts. No-op when unlocked.
    func registerCapture() {
        guard !isUnlocked else { return }
        rolloverIfNeeded()
        photosUsedToday = min(Self.freeDailyLimit, photosUsedToday + 1)
        UserDefaults.standard.set(photosUsedToday, forKey: usedKey)
    }

    private func localDayStamp(_ date: Date = Date()) -> String {
        var cal = Calendar.current
        cal.timeZone = TimeZone.current
        let c = cal.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", c.year ?? 0, c.month ?? 0, c.day ?? 0)
    }

    /// Reset the free counter when the local calendar day changes.
    private func rolloverIfNeeded() {
        let today = localDayStamp()
        let saved = UserDefaults.standard.string(forKey: dayKey)
        if saved != today {
            UserDefaults.standard.set(today, forKey: dayKey)
            UserDefaults.standard.set(0, forKey: usedKey)
            photosUsedToday = 0
        }
    }

    /// Re-check the day on foreground so a midnight crossing unlocks the shutter
    /// without a relaunch.
    func refreshDayRollover() {
        let before = photosUsedToday
        rolloverIfNeeded()
        if photosUsedToday != before { objectWillChange.send() }
    }

    // MARK: StoreKit

    var displayPrice: String { product?.displayPrice ?? "$1.99" }

    func loadProduct() async {
        do {
            let products = try await Product.products(for: [Self.unlimitedProductID])
            product = products.first
        } catch {
            lastErrorMessage = "Couldn’t load the store: \(error.localizedDescription)"
        }
    }

    func purchase() async {
        guard let product else {
            lastErrorMessage = "The purchase isn’t available right now. Check your connection and try again."
            return
        }
        purchaseInFlight = true
        defer { purchaseInFlight = false }
        do {
            let result = try await product.purchase()
            switch result {
            case .success(let verification):
                await handle(verification: verification)
            case .userCancelled, .pending:
                break
            @unknown default:
                break
            }
        } catch {
            lastErrorMessage = "Purchase failed: \(error.localizedDescription)"
        }
    }

    func restorePurchases() async {
        do {
            try await AppStore.sync()
            await refreshEntitlement()
            if !isUnlocked {
                lastErrorMessage = "No previous purchase was found to restore."
            }
        } catch {
            lastErrorMessage = "Restore failed: \(error.localizedDescription)"
        }
    }

    /// Re-derive entitlement from the app's current StoreKit entitlements.
    func refreshEntitlement() async {
        var unlocked = false
        for await result in Transaction.currentEntitlements {
            if case .verified(let transaction) = result,
               transaction.productID == Self.unlimitedProductID,
               transaction.revocationDate == nil {
                unlocked = true
            }
        }
        isUnlocked = unlocked
    }

    private func handle(verification: VerificationResult<Transaction>) async {
        guard case .verified(let transaction) = verification else { return }
        if transaction.productID == Self.unlimitedProductID,
           transaction.revocationDate == nil {
            isUnlocked = true
        } else if transaction.revocationDate != nil {
            // Refunded/revoked — drop the entitlement.
            await refreshEntitlement()
        }
        await transaction.finish()
    }
}
