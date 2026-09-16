# App Store screenshots — no-Mac guide

App Store Connect requires screenshots for the **largest iPhone display size**.
The app is **portrait-only**, so all screenshots are portrait.

## What you need

- **Required:** at least **one** screenshot at the **6.9-inch** iPhone size
  (iPhone 16 Pro Max class): **1320 × 2868 px** (portrait).
  - App Store Connect also accepts/asks for **6.5-inch**: **1242 × 2688 px**.
    Providing the 6.9" set is the important one; App Store Connect shows you the
    exact slots it wants when you get there.
- **Count:** 1 minimum, up to 10. Aim for **3–5** good ones.
- **No transparency, no rounded corners** — a full rectangular image at the exact
  pixel size. Marketing text overlays are optional.

## The problem and the fix (you have a 6.1" iPhone, not a 6.9")

Apple checks the **pixel dimensions**, not which device produced them. So:

1. On your **iPhone 14/15**, open the app (your sideloaded build works) and take
   screenshots (press Side + Volume Up). These come out at your device size
   (e.g. iPhone 15 = 1179 × 2556).
2. On **Windows**, resize/pad each one to the required **1320 × 2868** using any
   free image tool:
   - **Scale to fit width 1320**, then **pad top/bottom** with black to reach
     height 2868 (keeps the aspect correct, no stretching), **or**
   - simply place your screenshot on a **1320 × 2868 black canvas** centered.
   - Tools: Photopea (free, in-browser, Photoshop-like), GIMP, or Paint.NET.
3. Export as **PNG or JPEG** at exactly **1320 × 2868**. Repeat for each shot.

> Tip: shooting the screenshots on the phone gives you real UI. Padding to the
> required size is fine and common — Apple only validates dimensions.

## Suggested shots (tell the app's story in 3–5 frames)

1. The **viewfinder** with the zoom slider and lens controls visible.
2. A **result** — a finished high-resolution photo (or a before/after crop).
3. The **frame-count / resolution** controls (the multi-frame idea).
4. **Settings** showing "Unlimited Shooting" (sets up the IAP expectation).
5. Optional: a caption frame explaining "multi-frame RAW super-resolution."

## Where they go

App Store Connect → your app → the **1.0 version page → Previews and
Screenshots**, under the 6.9" (and optionally 6.5") size tab. Drag the images in.

## App icon (separate, don't forget)

The store listing also needs a **1024 × 1024 px** app icon (PNG, no alpha, no
rounded corners) in the version metadata — separate from the in-app AppIcon asset.
