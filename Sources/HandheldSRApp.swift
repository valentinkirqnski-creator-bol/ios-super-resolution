import SwiftUI

@main
struct HandheldSRApp: App {
    var body: some Scene {
        WindowGroup {
            Group {
                if DeviceSupport.hasEnoughRAM {
                    CameraView()
                } else {
                    // Hard gate: on a device below the RAM floor the whole camera
                    // UI is replaced, so the capture pipeline never starts.
                    UnsupportedDeviceView()
                }
            }
            .preferredColorScheme(.dark)
            .statusBarHidden(true)
        }
    }
}

/// Device eligibility. The multi-frame merge keeps several full 48MP frames plus
/// the accumulator resident on the GPU, whose working set does not fit in 4 GB
/// devices. Every 6 GB iPhone (iPhone 12 Pro / 13 Pro and every 14/15/16-series)
/// is supported; the 4 GB models (iPhone 12 / 13 and their minis, and older) are
/// not. A 5 GB floor cleanly separates the two, because a device's reported
/// physical memory runs a little under its nominal size.
enum DeviceSupport {
    static let requiredRAMGiB = 6
    static let minimumRAMBytes: UInt64 = 5 * 1024 * 1024 * 1024

    static var physicalRAMBytes: UInt64 { ProcessInfo.processInfo.physicalMemory }
    static var hasEnoughRAM: Bool { physicalRAMBytes >= minimumRAMBytes }
    /// Approximate installed RAM in GB, for display in the unsupported screen.
    static var physicalRAMGiB: Double { Double(physicalRAMBytes) / 1_073_741_824.0 }
}

/// Shown instead of the camera when the device has too little RAM. It states the
/// cause explicitly (RAM below the requirement) rather than a generic error.
struct UnsupportedDeviceView: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 16) {
                Image(systemName: "memorychip")
                    .font(.system(size: 52, weight: .regular))
                    .foregroundColor(Color(red: 0.95, green: 0.75, blue: 0.35))

                Text("Unsupported Device")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(.white)

                Text("FuzeFrame needs at least \(DeviceSupport.requiredRAMGiB) GB of RAM to hold the multi-frame super-resolution merge in memory.")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.white.opacity(0.85))
                    .multilineTextAlignment(.center)

                Text(String(format: "This device has about %.0f GB of RAM — below the requirement — so capture is disabled.", DeviceSupport.physicalRAMGiB))
                    .font(.system(size: 14))
                    .foregroundColor(.white.opacity(0.6))
                    .multilineTextAlignment(.center)

                Text("Supported: iPhone 12 Pro, 13 Pro, and iPhone 14 or later.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.5))
                    .multilineTextAlignment(.center)
                    .padding(.top, 4)
            }
            .padding(.horizontal, 32)
        }
    }
}
