# mfsr_conf: neural confidence for the robustness mask

A compact network that predicts, for every comparison frame and every raw pixel, the robustness weight the app's
merge multiplies that frame's samples by: R = 1 merges the sample fully, R = 0 discards it, and values in between
give partial weight. A second output map, alignment confidence, says whether the sample comes from the right scene
point at all. It is trained against what each weight does to the app's own merged 2× image, measured on synthetic
bursts with exact ground truth. The classical Wronski mask stays available as baseline and fallback.

Everything that touches the image runs either inside the app's own C++ stages or through a parity-tested replica
of them. Nothing in the MFSR pipeline was rewritten.

## 1. What the existing pipeline does (inspection)

| Stage | Where | What it does |
|---|---|---|
| Loader | `core/raw_io.cpp` | DNG through LibRaw, black-subtracted, white level 1, **pre-white-balanced** per CFA site; exposure = shutter × ISO; frames of a burst gain-matched to the reference (`apply_raw_gain`, capped at site saturation) |
| Alignment | `core/align.cpp`, `align_metal` | full-resolution FFT grey, pyramid block matching plus ICA refinement, **one vector per tile** (shipped: 16 px tiles, SNR auto-tune off), optional global homography (`global_homography_enabled`, off by default) composed as `H(p + flow)` |
| Classical robustness | `core/robustness.cpp` | Wronski Eq. 5–9 on the 460 3-channel guide (half resolution), 3×3 statistics, DNG noise model `var = a·x + b`, motion prior from the flow spread, 5×5 min, upscaled and warped to **raw resolution**. Measured: the result is **constant per RGGB quad**. Optional refiners: v21 NN plus v39 pixel net (`robust_nn_*`), geometry reject, HF reject. |
| Merge | `core/merge.cpp` (CPU), `merge_band_fused` (Metal) | kernel regression onto the 2× grid: output pixel `o` sits at raw `o/s`, nearest tile vector (+ homography), 3×3 raw taps around the moved position, anisotropic Gaussian weights from the frame's structure tensor (`k_detail 0.17, k_denoise 1.0, k_stretch 3, k_shrink 1` shipped), accumulated per CFA colour. **R enters as one factor per output pixel and frame**, fetched at raw pixel `floor(o/s + 0.5)`. The reference always has weight 1. |

Integration point: the per-frame raw-resolution robustness map, exactly where the analytic mask (and today's v21
refiner) is consumed, before the merge.

Limitations of the existing implementation, all measured here:

* **Aligner frame borders.** The aligner returns wrong vectors in a fixed band of about 3 tiles at the bottom and
  right frame borders and about 1 tile at the left (`tests/test_geometry.py`, interior tiles correct). At 12 MP that
  is a 1–2% border; in small synthetic bursts it would dominate, so the labelled patch keeps clear of it.
* **Integer-locked frames.** The aligner sometimes leaves a whole frame at the integer position: 0.26 px uniform
  error on one frame of a test burst.
* **Tile flow.** One translation per tile cannot represent rotation, parallax or object boundaries inside a tile.
  These are the in-tile misalignments the mask has to catch.
* **Classical mask resolution.** Constant per quad (half resolution). It sees a 3×3 mean colour difference
  normalised by local contrast, which hides sub-pixel shifts on strong edges.
* **No 48 MP ground truth.** The iPhone 15 cannot record 48 MP RAW and no HR captures exist on this machine. See
  the data section.

Hardware and framework: PyTorch 2.13 (CPU only), Ryzen 5 5625U (6 cores, 12 threads), 13.8 GB RAM, no GPU. Metal
cannot be compiled here.

## 2. Data: synthetic bursts with exact ground truth

**Scenes** (`mfsr/scenes.py`). Real DNGs are decoded by the app's own loader (`build/frame_dump.exe`). Each RGGB
quad becomes one camera-native RGB texel: no demosaicing and no resampling, so every texel is a real colour
measurement. The simulated sensor samples the scene at 2 texels per sensor pixel. At the 2× merge scale one output
pixel then covers one texel, so the HR target is resolved by genuine scene data and never by upscaling.

What this does *not* reproduce is the phone's absolute scale: a simulated sensor pixel spans 4 source pixels.
Procedural fine text and bar charts, rendered directly at 4 texels per pixel, cover aliasing and repetitive
patterns.

**Splits are by source image:**

| Split | Scenes |
|---|---|
| Train | APC_1340, APC_1342, HDRP_N000, APC_1258 |
| Validation | APC_1211 |
| Test (synthetic) | APC_1278 |
| Excluded entirely | APC_1186, APC_1294 (the references of the real test bursts burst9 and burst12) |

**Image formation** (`mfsr/formation.py`; world model reused from `tools/robust_nn/rnn/burst.py`). Planar layers
carry exact per-frame homographies: camera roll, zoom, perspective and translation about an optical centre,
depth-scaled parallax, independently moving objects, and small slipping content patches. Each frame is formed as:

1. Optical blur, a Gaussian PSF.
2. Sub-exposure averaging for motion blur.
3. **Pixel-aperture integration**: an n×n grid over the pixel footprint (new; the old renderer point-sampled).
4. Exposure with per-frame changes.
5. CFA sampling.
6. Poisson–Gaussian noise from the DNG noise profile × ISO.
7. Clip at white and quantise.
8. Loader domain: the per-site pre-white-balance and the app's exposure-matching gain, with a residual error and
   the saturation cap.

Recorded per frame: every layer homography, camera and object parameters, exposure and gain.

**Ground truth.** The exact ref→frame correspondence and visibility come for every reference raw pixel (occlusion,
disocclusion, out of frame). The **HR target** T is sampled on the merge's own output grid (`o/s`, verified in
`tests/test_geometry.py` against a target a quarter pixel off). It is the aperture-integrated reference view, i.e.
the band-limited sensor signal at twice the density, which is what a perfect merge converges to.

Documented simplifications: global shutter, a Gaussian PSF with no chromatic aberration, a box aperture, no
vignetting, the scene's own camera RGB, and moving objects cut from the same scene.

**The app's stages** (`cpp/conf_harness.cpp`). The harness runs on every burst with the app's shipped configuration
(16 px tiles, shipped kernel parameters): the real aligner, optionally with the homography; the analytic mask for
the correspondence actually used; and the kernel covariances. Deliberate errors are injected after the aligner and
before the mask.

**Two alignment variants per burst** over the same rendered frames. v0 is the estimate: the app's aligner, or the
exact tile vectors where a category is about correctly aligned data. v1 is v0 plus deliberate errors
(`mfsr/corrupt.py`), with continuous log-uniform magnitudes:

* sub-pixel and large tile offsets;
* smooth wrong fields;
* rotation or scale errors (vary inside a tile, so the tile centre is right and the edge wrong);
* wrong local-motion blocks;
* slips across strong edges;
* burst-consistent edge slips.

Occlusions and motion boundaries come from the moving-object scenes.

**Categories** (`mfsr/scenarios.py`):

* aligned texture
* aliasing (fine text and charts)
* subtle edge slips
* repetitive patterns
* flat noisy low light
* rotation and spatially varying motion
* moving objects and occlusion
* exposure differences
* saturation

## 3. Labels

**Alignment confidence.** `t = vis · exp(−e²/(2σ²))`, where `e = |q_used − q_true|` is the exact error of the
correspondence the merge uses for that raw pixel (σ = 0.25 raw px, set at training time).

**Merge confidence.** The merged image is an exact, differentiable function of every R map, because R factors out of
the gather (verified against the C++ merge with a random R map: `tests/test_geometry.py`). The label R\* is the
optimum of a reconstruction cost on that merge, in single-frame noise-variance units:

```
gamma · (M − I)² / σ²   +   Σ_k (1 − r_k)(g_k + n_k + prior)
```

* **M** is the merge of the noise-free frames with the used correspondences.
* **I** is the same merge, same weights, at the true correspondences with occluded samples dropped.
* **g_k** is how much frame k, perfectly aligned, brings the merge closer to the HR target. These are its new
  sub-pixel samples: aliasing that is signal, situation A.
* **n_k** is how much frame k lowers the merge's noise variance (situation C).
* **gamma = 4** prices a misalignment error above the same amount of noise: a misalignment is worse than falling back
  to fewer frames. The prior is 0.02.

No displacement threshold appears anywhere.

R\* is optimised jointly over frames from two starts, all-in and all-out, because the cost is not convex. The
include/exclude utility of every frame is stored for validation and test.

**Why this form.** Two simpler forms were measured and rejected. Pricing the whole error against T made the optimum
drop perfectly aligned samples wherever a subset of them suits the merge kernel better: 18.7% of aligned samples
on real-scene edges, 42% on fine text, and 46% in flat areas under the shipped narrow kernels. None of that is
misalignment.

With the final cost, perfectly aligned samples are kept at 99.9–100% in every category, including flat areas and
aliased text. Misaligned samples on visible edges are rejected at 55–99% while flat-area ones are kept. See
`tests/test_labels.py` and the numbers below.

## 4. Network (`mfsr/model.py`)

Inputs per frame (`mfsr/inputs.py`), all noise-normalised so one weight set spans ISO:

* At guide resolution, 19 channels: the 15 evidence features of `tools/robust_nn` (already mirrored in
  `core/robust_nn_shared.h` for Metal) and 4 pixel channels (reference, warped comparison and residual over the
  local noise σ). The classical mask enters as feature 0, or is zeroed for the pure neural model.
* At raw resolution, 3 channels from the aligner's full-resolution greys: the warped residual, reference structure
  and edge strength.

Architecture:

1. A shared per-frame encoder of 4 dilated 3×3 convolutions (receptive field 31 quads = 62 raw px).
2. Cross-frame fusion: mean and max over the burst's frames, so a frame can be judged as the odd one out.
3. Two heads, then a 2× raw-resolution refinement.

This gives ≈14k parameters and ≈16.7k MAC per quad per frame (≈50 GMAC per 12 MP frame). Replicate padding is
used throughout. Variants are switched in the config: no fusion, small receptive field, and classical refinement
(`refine_r0`: R = σ(logit R0 + δ)).

## 5. Training (`train.py`)

```
L = L_align + λ_merge · L_merge + λ_recon · L_recon
```

* `L_align` and `L_merge` are weighted BCE against the two labels.
* `L_recon` is the label's own cost of the app's merge for the predicted R. Its minimum is R\*, so it cannot push R
  towards rejecting valid samples.

Validation runs on a different scene and tracks:

* excess reconstruction cost over R\* for the network, the classical mask and R = 1;
* valid-sample retention;
* harmful-sample rejection.

The best checkpoint is chosen by validation reconstruction cost. Configs, git revision and logs are saved with
every run.

## 6. Tests (`python -m unittest discover tests`)

* `test_merge_parity`: the PyTorch merge against the app's C++ merge (tile flow, homography, rotation errors, large
  offsets). Worst relative error 1.3e-7 to 1.8e-7.
* `test_geometry`:
  * correspondence direction and exactness (translation; rotation with point sampling);
  * app flow direction and units;
  * the HR-target pixel convention;
  * the mask fetch rule;
  * recombination with a random R map equals the C++ merge run with that map.
* `test_labels`:
  * aligned samples kept (real texture and aliased text);
  * misaligned edges rejected, flat areas kept;
  * include/exclude utility;
  * R\* is the optimum of its own cost.
* `test_model`: confidence-map dimensions, frame-permutation equivariance, padded frames do not leak.

## 7. Reproduce

```
sh build.sh                                      # harness + frame dumper against the current core/
python gen_data.py configs/data_d1.yaml          # val, test, train (resumable)
python train.py configs/train_m1.yaml            # and the ablations configs/train_m*.yaml
python evaluate.py runs/m1/best.pt --more runs/m2_nofusion/best.pt ...
python real_eval.py runs/m1/best.pt --burst <dir> --box x0 y0 w h
python -m unittest discover tests
```

## 8. Results (measured, 2026-10-25)

**Test split:** scene APC_1278, never used for training or validation; 108 bursts × 2 alignment variants = 216
crops, 12 bursts per category. Every mask goes through the exact replica of the app's merge. v21 and v39 are run
exactly as the app ships them (v39 with its +0.25 shift, both with the default 0.20 / 0.90 sharpening).

| Mask | PSNR vs HR target | Expected error | Ghost energy | Ghost (edges) | False detail | Flat-area noise | Harmful rejected | Valid kept | Weight on aligned samples |
|---|---|---|---|---|---|---|---|---|---|
| R = 1 (no robustness) | 43.01 dB | 1.486 | 0.795 | 1.701 | 0.051 | 0.776 | 0% | 100% | 1.00 |
| Classical (shipped Eq. 5–9) | 43.54 dB | 1.386 | 0.566 | 1.315 | 0.035 | 0.780 | 12% | 97.5% | 0.97 |
| v21 (app) | 43.99 dB | 1.203 | 0.140 | 0.431 | 0.017 | 0.783 | 40% | 91.1% | 0.87 |
| v39 (app) | 43.95 dB | 1.223 | 0.171 | 0.467 | 0.018 | 0.784 | 43% | 90.4% | 0.89 |
| m1: fusion + raw-resolution head | **44.03 dB** | **1.173** | **0.090** | **0.316** | 0.016 | 0.782 | 39% | 92.1% | 0.89 |
| m4: refines the classical mask | 43.87 dB | 1.205 | 0.108 | 0.394 | 0.016 | 0.784 | 34% | 91.8% | 0.92 |
| **m9: per-frame, guide-res (SHIPPED)** | 44.00 dB | 1.181 | 0.127 | 0.453 | 0.016 | 0.783 | 32% | **93.8%** | **0.92** |
| R\* (label, upper bound) | 44.12 dB | 1.151 | 0.029 | 0.091 | 0.015 | 0.780 | 100% | 100% | 0.98 |

Errors are in single-frame noise-variance units.

* **m1** has lower expected error than v21 in 16 of the 18 category/variant sets; **m9** in 14 of 18.
* On perfectly aligned bursts the learned masks keep more than the classical mask (m1 aliasing: 99.2% vs 96.6%)
  and flat noisy areas keep 100%.
* **Weak spot:** samples next to real misalignments, especially rotation (m1 weight on aligned samples 0.80 vs v21
  0.92). It is reduced in m4 and m9.
* **Output sharpening** (`evaluate.py --sharpen`) trades 0.17–0.32 dB for higher aligned-sample weight; the raw
  output is the best operating point.
* Results per category: `runs/*/eval/results.md`.

**Which one ships.** m1 is best on the synthetic test, but its cross-frame fusion needs every frame's features
before any mask is built, and its raw head needs full-resolution greys inside the robustness stage. Neither exists in
the app's streaming Metal path today. m9 is the per-frame, guide-resolution form that runs on the existing banded
robust_nn path: 12.2k parameters, 12.1k MAC per quad per frame, ≈36 GMAC per 12 MP frame. Its weights are in
`core/conf_nn_weights.h` (`python export_weights.py runs/m9_guideres/best.pt`).

**Device code is verified on CPU, not compiled.**
* `parity_cnn.py`: the shared header against PyTorch, max error 3.6e-7 on R.
* `kernel_test.py`: the exact Metal kernel source plus the host band loop, NaN-poisoned band buffers, band heights
  32 and 7; max error 6.3e-7.
* Metal and Swift could not be compiled on this machine.

**Real iPhone bursts** (burst9, burst11, burst12; never used; no ground truth; `real_eval.py`, crops in
`~/Downloads/mfsr_conf_outputs/`). Mean mask weights:

| Region | Classical | v21 | m1 | m9 |
|---|---|---|---|---|
| burst11 flat sky | 0.90 | 0.94 | 0.96 | 0.97 |
| burst9 power lines, with homography | 0.84 | 0.74 | 0.60 | 0.67 |
| burst9 car, with homography | 0.74 | 0.62 | 0.54 | 0.63 |
| burst11 building, with homography | 0.72 | 0.52 | 0.38 | 0.57 |
| burst12 roof, no homography | 0.91 | 0.53 | 0.54 | 0.61 |

* m1 rejects clearly more than v21 on real bursts, flat areas included; m9 behaves close to v21. Whether m1's extra
  rejection removes real ghosts or wastes samples cannot be decided without ground truth. It is the main open
  question before m1-style models should ship.
* On the power lines m1's mask follows the wires precisely (raw-resolution head), where v21 / m9 reject wide bands.

**Not yet done when m9 shipped:** the ablations no fusion (m2), small receptive field (m3), loss weights (m5 / m6),
patch size (m7 / m8) and the retention-focused m10 were still training (`train_queue.sh`, `train_queue2.sh`).
