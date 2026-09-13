import SwiftUI

// $2 one-time unlock for unlimited capture. Presented when a free user hits the
// daily limit (or taps "Unlock Unlimited"). Material Design 3 idiom.
struct PaywallView: View {
    @ObservedObject var store: StoreManager
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            MD3.surface.ignoresSafeArea()
            ScrollView {
                VStack(spacing: 20) {
                    header
                    MD3Card {
                        benefitRow("infinity", "Unlimited photos",
                                   "Shoot as many multi-frame captures as you like, every day.")
                        Divider().overlay(MD3.outline.opacity(0.4))
                        benefitRow("bolt.fill", "One-time purchase",
                                   "Pay once. No subscription, no recurring charge.")
                        Divider().overlay(MD3.outline.opacity(0.4))
                        benefitRow("lock.open.fill", "Keeps your work",
                                   "The free \(StoreManager.freeTotalLimit)-photo limit is lifted permanently.")
                    }

                    VStack(spacing: 10) {
                        Button(action: { Task { await store.purchase() } }) {
                            HStack {
                                if store.purchaseInFlight {
                                    ProgressView().tint(MD3.onPrimary)
                                } else {
                                    Text("Unlock Unlimited — \(store.displayPrice)")
                                }
                            }
                        }
                        .buttonStyle(MD3FilledButtonStyle(disabled: store.product == nil))
                        .disabled(store.purchaseInFlight || store.product == nil)

                        Button("Restore Purchase") { Task { await store.restorePurchases() } }
                            .buttonStyle(MD3TextButtonStyle())
                            .disabled(store.purchaseInFlight)
                    }

                    if let err = store.lastErrorMessage {
                        Text(err)
                            .font(.system(size: 13))
                            .foregroundColor(Color(red: 1.0, green: 0.70, blue: 0.70))
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: .infinity, alignment: .center)
                    }

                    Text("Payment is charged to your Apple ID. This is a one-time, non-consumable purchase and can be restored on your other devices.")
                        .font(.system(size: 11))
                        .foregroundColor(MD3.onSurfaceVariant)
                        .multilineTextAlignment(.center)
                        .padding(.top, 4)
                }
                .padding(20)
            }
        }
        // Close when the purchase lands.
        .onChange(of: store.isUnlocked) { unlocked in if unlocked { dismiss() } }
        .overlay(alignment: .topTrailing) {
            Button(action: { dismiss() }) {
                Image(systemName: "xmark")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(MD3.onSurfaceVariant)
                    .padding(10)
                    .background(Circle().fill(MD3.surfaceContainerHigh))
            }
            .padding(16)
        }
    }

    private var header: some View {
        VStack(spacing: 10) {
            Image(systemName: "infinity.circle.fill")
                .font(.system(size: 56))
                .foregroundColor(MD3.primary)
                .padding(.top, 24)
            Text("Unlimited Capture")
                .font(.system(size: 26, weight: .bold))
                .foregroundColor(MD3.onSurface)
            Text("You’ve used your \(StoreManager.freeTotalLimit) free photos. Unlock unlimited shooting with a single purchase.")
                .font(.system(size: 14))
                .foregroundColor(MD3.onSurfaceVariant)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 8)
        }
    }

    private func benefitRow(_ icon: String, _ title: String, _ subtitle: String) -> some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .semibold))
                .foregroundColor(MD3.primary)
                .frame(width: 28, height: 28)
                .background(Circle().fill(MD3.primaryContainer))
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(MD3.onSurface)
                Text(subtitle)
                    .font(.system(size: 13))
                    .foregroundColor(MD3.onSurfaceVariant)
            }
            Spacer()
        }
    }
}
