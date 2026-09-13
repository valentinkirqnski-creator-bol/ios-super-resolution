import Foundation
import StoreKit

// In-app purchase + free-tier gate.
//
// Business rules (confirmed with the product owner):
//   * One non-consumable purchase, `unlimitedProductID`, priced at $1.99/$2 tier,
//     permanently unlocks unlimited capture.
//   * Without it, the free tier allows `freeTotalLimit` completed captures in
//     TOTAL (lifetime, not per day); a burst merge counts as one, and an import
//     from storage counts too. The count never resets. When it reaches the limit
//     the shutter is disabled and the paywall is offered.
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
    /// Free captures TOTAL (lifetime, not per day) before the shutter locks and
    /// the paywall is offered. Never resets.
    static let freeTotalLimit = 10

    @Published private(set) var isUnlocked = false
    @Published private(set) var product: Product?
    @Published private(set) var purchaseInFlight = false
    @Published private(set) var photosUsed = 0
    @Published var lastErrorMessage: String?

    private let usedKey = "FreePhotosUsedCount"
    private var updatesTask: Task<Void, Never>?

    private init() {
        photosUsed = UserDefaults.standard.integer(forKey: usedKey)
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

    /// Captures still allowed for a non-paying user (callers check isUnlocked first).
    var photosRemaining: Int { max(0, Self.freeTotalLimit - photosUsed) }

    /// True if a capture is allowed right now.
    var canCapture: Bool { isUnlocked || photosRemaining > 0 }

    /// Call exactly once per capture that actually starts. No-op when unlocked.
    /// The count is a lifetime total and never resets.
    func registerCapture() {
        guard !isUnlocked else { return }
        photosUsed = min(Self.freeTotalLimit, photosUsed + 1)
        UserDefaults.standard.set(photosUsed, forKey: usedKey)
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
