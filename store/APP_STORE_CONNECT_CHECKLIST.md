# App Store Connect — setup & submission checklist

Everything here is done in a browser (or the GitHub UI). No Mac needed.
Bundle id: `com.handheldsr.camera` · App name: FuzeFrame.

---

## 0. Prerequisite

- [ ] Apple Developer Program membership approved (developer.apple.com, $99/yr).
      Enrollment approval takes ~1–2 days — start it first.

---

## 1. Host the Privacy Policy (required)

Easiest no-Mac option — GitHub Pages:
1. [ ] In your GitHub repo: **Settings → Pages →** set Source to your default
       branch, folder `/ (root)` or `/docs`.
2. [ ] Put the privacy policy where Pages serves it. Two options:
       - Rename `store/PRIVACY_POLICY.md` content into a `docs/privacy.md`, or
       - Ask me to output it as `docs/index.html`.
3. [ ] Fill in the contact email and remove the HTML comment block first.
4. [ ] Note the public URL (e.g. `https://<user>.github.io/<repo>/privacy`).
       You'll paste it into App Store Connect below.

---

## 2. App Store Connect API key (for the GitHub Actions upload)

appstoreconnect.apple.com → **Users and Access → Integrations → App Store Connect API**:
1. [ ] Generate a **Team key** with role **App Manager**.
2. [ ] Record the **Key ID** and the **Issuer ID** (shown on that page).
3. [ ] Download the **`AuthKey_XXXXX.p8`** — you can only download it once. Keep it safe.

Then add GitHub repo secrets (**Settings → Secrets and variables → Actions**):
- [ ] `APPLE_TEAM_ID` — your 10-char Team ID (developer.apple.com → Membership).
- [ ] `ASC_KEY_ID` — the Key ID.
- [ ] `ASC_ISSUER_ID` — the Issuer ID.
- [ ] `ASC_KEY_P8_BASE64` — the `.p8` file, base64-encoded. On Windows PowerShell:
      ```
      [Convert]::ToBase64String([IO.File]::ReadAllBytes("AuthKey_XXXXX.p8")) > key.txt
      ```
      Paste the contents of `key.txt` as the secret value.

---

## 3. Create the app record

App Store Connect → **Apps → +  → New App**:
- [ ] Platform: **iOS**
- [ ] Name: **FuzeFrame** (must be unique across the store; have a backup name ready)
- [ ] Primary language, Bundle ID: **com.handheldsr.camera**, SKU: any string (e.g. `fuzeframe01`)
- [ ] User access: Full

---

## 4. Create the in-app purchase (MUST match the code exactly)

App Store Connect → your app → **Monetization → In-App Purchases → +**:
- [ ] Type: **Non-Consumable**
- [ ] Reference Name: `Unlimited Shooting` (internal only)
- [ ] **Product ID: `com.handheldsr.camera.unlimited`**  ← must be exactly this
- [ ] Price: choose the tier for **$1.99** (the code shows `store.displayPrice`)
- [ ] Add a localized display name + description (e.g. "Unlimited Shooting" /
      "Remove the free-capture limit forever.")
- [ ] Add a review screenshot of the paywall/Settings unlock (any 1 screenshot)
- [ ] Submit the IAP **with the app version** (it must be "Ready to Submit" /
      attached to the build, or the purchase fails in review).

> If the IAP is missing/not submitted, review fails under Guideline 2.1(b)
> ("in-app purchase not functional"). This is the #1 avoidable rejection.

---

## 5. App Privacy ("nutrition label")

App Store Connect → your app → **App Privacy**:
- [ ] Data collection: **"Data Not Collected"** (the app collects/sends nothing).
- [ ] Privacy Policy URL: paste the URL from step 1.

---

## 6. Version metadata (the "1.0 Prepare for Submission" page)

- [ ] **Screenshots** — see SCREENSHOTS.md.
- [ ] **Description** — what the app does (multi-frame RAW super-resolution camera;
      outputs a high-res DNG; free captures then a one-time unlock).
- [ ] **Keywords**, **Support URL** (can be the same GitHub Pages site or a repo),
      **Marketing URL** (optional).
- [ ] **Age rating** questionnaire — answer all "None"; result 4+.
- [ ] **Category** — Photo & Video.
- [ ] Attach the **build** (appears after the GitHub Actions upload finishes
      processing — can take 15–60 min after upload).
- [ ] Attach the **in-app purchase** to this version.

---

## 7. Reviewer notes (App Review Information → Notes)

Paste this so the reviewer isn't confused (no demo account is needed — the app
requires no login):

```
No account or login is required to use this app.

Free tier: the app allows a limited number of free captures, after which a
one-time non-consumable in-app purchase ("com.handheldsr.camera.unlimited",
Unlimited Shooting) removes the limit. A "Restore Purchases" button is in
Settings.

Device requirement: the multi-frame merge needs a device with at least 6 GB of
RAM (iPhone 12 Pro / 13 Pro and iPhone 14 or later). On a device with less RAM
the app shows an "unsupported device" screen by design. Please review on a
6 GB+ iPhone (e.g. iPhone 14 or later), where all features are available.

To test the purchase: press the shutter until the free limit is reached (or open
Settings) to reach the unlock, or use the sandbox tester.
```

---

## 8. Build → TestFlight → Submit (order matters)

1. [ ] GitHub → **Actions → "iOS App Store upload" → Run workflow**.
2. [ ] If it errors on signing, send me the log; when green the build lands in
       **TestFlight** (proves the whole no-Mac pipeline).
3. [ ] Install via **TestFlight** on a 6 GB+ iPhone. Verify capture works and the
       **$1.99 unlock + Restore** work (sandbox purchases work in TestFlight).
4. [ ] Back in App Store Connect, finish metadata, attach build + IAP.
5. [ ] **Submit for Review.** Reviews usually take ~1–2 days.

---

## Likely-rejection watch-list (all fixable, free to resubmit)

- IAP not attached/submitted → 2.1(b). (Step 4 above.)
- Missing Privacy Policy URL → blocked before review. (Step 1/5.)
- App deemed too thin → Guideline 4.2. Mitigate with a clear description +
  screenshots showing the RAW/super-res result and controls.
- Crash on the reviewer's device → 2.1. You've run it on iPhone 14/15; confirm
  again on TestFlight before submitting.
