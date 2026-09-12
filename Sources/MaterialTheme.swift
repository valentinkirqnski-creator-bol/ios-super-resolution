import SwiftUI
import UIKit

// Material Design 3 styling primitives for SwiftUI (the app can't use the React
// "Material UI" library; this is the MD3 idiom expressed natively). The app runs
// in dark mode (HandheldSRApp sets .preferredColorScheme(.dark)), so these are a
// dark-scheme tonal palette with MD3 color roles, shape, and components.
enum MD3 {
    // Color roles (dark scheme).
    static let primary          = Color(red: 0.66, green: 0.78, blue: 1.00) // indigo-ish
    static let onPrimary        = Color(red: 0.06, green: 0.12, blue: 0.28)
    static let primaryContainer = Color(red: 0.18, green: 0.26, blue: 0.45)
    static let onPrimaryContainer = Color(red: 0.84, green: 0.89, blue: 1.00)

    static let surface          = Color(red: 0.07, green: 0.08, blue: 0.10)
    static let surfaceContainer = Color(red: 0.12, green: 0.13, blue: 0.16)
    static let surfaceContainerHigh = Color(red: 0.16, green: 0.17, blue: 0.20)
    static let surfaceVariant   = Color(red: 0.27, green: 0.28, blue: 0.32)

    static let onSurface        = Color(red: 0.90, green: 0.91, blue: 0.94)
    static let onSurfaceVariant = Color(red: 0.74, green: 0.76, blue: 0.80)
    static let outline          = Color(red: 0.55, green: 0.57, blue: 0.61)

    static let secondaryContainer = Color(red: 0.22, green: 0.24, blue: 0.30)
    static let onSecondaryContainer = Color(red: 0.86, green: 0.89, blue: 0.96)

    // Shape scale.
    static let cornerLarge: CGFloat = 24
    static let cornerMedium: CGFloat = 16
    static let cornerSmall: CGFloat = 12
}

// A filled tonal card (MD3 "surface container") used to group content.
struct MD3Card<Content: View>: View {
    var padding: CGFloat = 16
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) { content }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: MD3.cornerLarge, style: .continuous)
                    .fill(MD3.surfaceContainer)
            )
    }
}

// MD3 filled button (high-emphasis primary action).
struct MD3FilledButtonStyle: ButtonStyle {
    var disabled: Bool = false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 16, weight: .semibold))
            .foregroundColor(MD3.onPrimary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(
                RoundedRectangle(cornerRadius: 100, style: .continuous)
                    .fill(disabled ? MD3.outline.opacity(0.4) : MD3.primary)
            )
            .opacity(configuration.isPressed ? 0.85 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

// MD3 text/tonal button (low-emphasis secondary action).
struct MD3TextButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 15, weight: .medium))
            .foregroundColor(MD3.primary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 100, style: .continuous)
                    .fill(MD3.secondaryContainer.opacity(configuration.isPressed ? 0.9 : 0.0))
            )
            .contentShape(Rectangle())
    }
}

// Material Design 3 chrome for a SwiftUI Form/List settings screen: the grouped
// background is replaced with the MD3 surface, controls adopt the primary tint,
// section headers are emphasised, and the navigation bar takes MD3 colours.
// Applied at the container level so it reskins every row without rewriting them.
struct MD3FormChrome: ViewModifier {
    func body(content: Content) -> some View {
        Group {
            if #available(iOS 16.0, *) {
                content.scrollContentBackground(.hidden)
            } else {
                content
            }
        }
        .background(MD3.surface.ignoresSafeArea())
        .tint(MD3.primary)
        .headerProminence(.increased)
        .onAppear { Self.applyNavBarAppearance() }
    }

    static func applyNavBarAppearance() {
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor(MD3.surface)
        let titleColor = UIColor(MD3.onSurface)
        appearance.titleTextAttributes = [.foregroundColor: titleColor]
        appearance.largeTitleTextAttributes = [.foregroundColor: titleColor]
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
        UINavigationBar.appearance().compactAppearance = appearance
        UINavigationBar.appearance().tintColor = UIColor(MD3.primary)
    }
}

extension View {
    /// Reskin a Form/List settings screen in the Material Design 3 idiom.
    func md3FormChrome() -> some View { modifier(MD3FormChrome()) }
}

// Section header in the MD3 type scale.
struct MD3SectionHeader: View {
    let title: String
    var body: some View {
        Text(title.uppercased())
            .font(.system(size: 12, weight: .bold))
            .tracking(0.8)
            .foregroundColor(MD3.primary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}
