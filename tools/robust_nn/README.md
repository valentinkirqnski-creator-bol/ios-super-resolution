# robust_nn — learned refinement of the robustness mask

A ~2.2k-parameter network that answers, per grey pixel and per comparison
frame:

> Given the reference, the comparison frame and the estimated correspondence,
> how trustworthy is the comparison sample for reconstructing the reference
> scene here?

It **starts from the analytic mask and refines it**:
`R = sigmoid(logit(R0) + delta)`. The output layer is zero-initialised, so an
untrained network reproduces `R0` exactly, and training only learns where and
by how much to depart from it.

Written from scratch, without reading the app's existing code. The interface
contract is below.

## Layout

```
rnn/scene.py       procedural clean HR scenes (dead leaves, near-Nyquist stripes, zone plates,
                   bricks, text, gradients, flat darks, HDR highlights; optional photos)
rnn/burst.py       RAW burst formation with EXACT ground-truth correspondence + visibility
rnn/scenarios.py   scenario sampler (tremor, rotation, zoom, parallax, moving, repetitive,
                   low_light, saturated, motion_blur)
rnn/align.py       from-scratch pyramid tile aligner + ICA, run on the noisy frames;
                   ExternalAligner hook for the app's own aligner
rnn/corrupt.py     deliberate flow corruption (sub-pixel bias, smooth bias, tile errors,
                   inconsistent vectors, outliers, edge-targeted errors)
rnn/labels.py      the two supervision sources (geometric, reconstruction utility)
rnn/merge.py       kernel-gather Bayer merge (labels + evaluation)
rnn/analytic.py    analytic R0, parameters randomised during training
rnn/features.py    14 noise-normalised evidence features   <- mirrored in cpp/
rnn/model.py       the network
gen_dataset.py / train.py / evaluate.py / export_weights.py / parity_test.py
cpp/robust_nn_shared.h   per-pixel features + net; compiles as C++ AND Metal
cpp/robust_nn_weights.h  generated weights
cpp/robust_nn.{h,cpp}    CPU reference      cpp/robust_nn_cli.cpp  parity/bench CLI
cpp/robust_nn.metal      3 kernels           cpp/RobustNNMetal.{h,mm} host encoder
```

## How the network is taught what misalignment is

### The three situations

| Situation | What the evidence looks like | Target |
|---|---|---|
| A. Correct correspondence, different sub-pixel sampling | Pixel values differ (aliasing, noise), but the residual is **not** explained by one coherent displacement of the reference gradient | keep |
| B. Wrong correspondence | The residual is a spatially coherent `g·δ` pattern: doubled edges. The flow departs from a local affine model | reject |
| C. Ambiguous (repetitive texture, low light, blur, clipping) | Both of the above are plausible | calibrated probability (BCE on soft labels) |

### Ground truth by construction (`burst.py`)
A scene is a stack of planar layers at different depths. Each one has an exact
homography per frame, built from camera roll, zoom, perspective and
depth-scaled translation (parallax), plus the layer's own motion. So for every
reference pixel we know:

* the true correspondence `f_gt`
* whether the point is visible in frame k (occlusion, disocclusion, out of frame)

Frames are rendered with optical blur, sub-exposure motion blur, white-balance
gains, exposure jitter, an RGGB mosaic, Poisson–Gaussian noise, clipping and
quantisation.

### Real, imperfect flow (`align.py`, `corrupt.py`)
The network never sees perfect flow. The aligner runs on the noisy frames, and
45% of frames additionally get deliberate corruption. That way the same scene
appears with both good and bad correspondences, and only the evidence can tell
them apart.

### Label A: geometric reliability (`labels.py`)
```
e    = |f_est − f_gt|
D²   = mean3x3( C_k(p+f_est) − C_k(p+f_gt) )²     C_k = NOISE-FREE comparison frame
geo  = vis · exp(−D² / (2·2²·s²)) · exp(−e² / (2·2²))
```
Both samples come from the same clean frame, so aliasing and sub-pixel
sampling differences cancel. D measures only the ghost the flow error would
paint:

* a 0.2 px error on a sharp edge gives a large D, so the label says reject
* the same error in a flat region gives D ≈ 0, so the label says keep
* the `e` term still bounds gross errors (τ = 2 grey px)

### Label B: reconstruction utility (`labels.py`)
This is a controlled fusion experiment against the clean scene:

1. Merge with all other frames weighted by their oracle `geo`.
2. Toggle frame k fully in and fully out.
3. Set `util = sigmoid(8·log((E_without+ε)/(E_with+ε)))`, where E is the
   locally pooled squared error vs the ground truth.

A correctly aligned frame reduces noise and aliasing, so `util > 0.5` even
though its pixels differ from the reference. A ghosting frame drives util
towards 0. Utility depends on the other samples, so it is a training signal
used alongside A, not a universal truth.

Loss: `0.4·BCE(R, geo) + 0.6·BCE(R, util)`, weighted ×(1 + 2·|target − R0|).

### Features (`features.py` = `robust_nn_shared.h`)
All features are normalised by the local noise σ from `var = a·v + b`, so one
weight set spans ISO.

| # | Feature | Purpose |
|---|---|---|
| 0 | logit(R0)/4 | the starting point |
| 1 | 3×3 mean difference / σ_mean | the analytic evidence |
| 2, 13 | residual energy, 5×5 and 9×9 | how different |
| 3, 6 | gradient energy, structure-tensor anisotropy | edge vs flat vs texture |
| **4, 5** | least-squares fit `r ≈ g·δ` over 5×5 → \|δ\|, fraction of residual explained | **coherent displacement (B) vs aliasing/noise (A)** |
| 7, 8 | log brightness, clip fraction | low light, saturation |
| 9–12 | affine fit of the flow over a 5×5 lattice of step = tile stride: centre residual, RMS, gradient, max neighbour jump | correspondence-model failures: rotation, zoom, parallax, outliers |

### Network
`14 → conv3×3 → 16 → ReLU → 1×1 → 16 → ReLU → 1×1 → δ`, with replicate
padding. Zero padding would leak absolute position near the borders. Feature
standardisation is folded into layer 1 at export.

## Integration contract (app side)

Per comparison frame, at **grey resolution** (one value per RGGB quad):

| Input | Meaning |
|---|---|
| `ref`, `cmp` | grey = mean of the 2×2 quad, black-subtracted, **white level = 1.0** |
| `flow_xy` | interleaved (dx,dy) ref→cmp in grey px, i.e. `cmp` is sampled at `p + flow`. Use the same per-pixel vector the merge uses (e.g. the owning tile's flow) |
| `r0` | the app's analytic robustness for this frame |
| `noise_a`, `noise_b` | grey-domain `var(v) = a·v + b` (raw-domain a,b of the quad mean → a/4, b/4) |
| `clip_level` | 0.9 (grey level counted as clipped; trained with 0.9) |
| `flow_step` | the aligner's tile stride in grey px |

Output: refined R in [0,1], same grid. It replaces r0 wherever r0 was
consumed. If r0 drives a 2×2 or 4×4 output block in the merge, R does too.

Metal: add `cpp/` to the shader header search path, compile
`robust_nn.metal` with fast math OFF, and call
`-[RobustNNMetal encodeInto:...]` per frame. Scratch per frame is one float
plane (J) plus 14 feature planes; define `RNN_FEAT_T=half` to halve it. CPU:
`robust_nn::refine(...)`.

Retraining needs no device code changes, as long as the features are unchanged:
`python export_weights.py --model runs/<x>/model.pt` regenerates
`cpp/robust_nn_weights.h`. `python parity_test.py --model ...` must print
`PARITY OK`.

## Shipped: v6 (Settings → Robustness → "Robust NN v6 (GPU)")

v6 is trained with the **real-scene target protocol** (`real_protocol.py`):

* **Scenes:** bursts built from real DNG scenes with exactly known camera motion
  (APC_1186 and the HDR+ payload for training; APC_1211 for validation;
  burst2 never used).
* **Format:** rendered in the device's raw format (BGGR, pre-white-balanced,
  the DNG's noise profile × ISO).
* **The app's own stages:** run through `build/pipe/tp_real.exe`: the aligner
  (its real errors), optional tile corruption, the analytic `compute_robustness`,
  and the merge.
* **Target:** the app's merge with the exact flow and R = 1.

Held-out results:

* **Validation scene:** error in misaligned regions is 3.8% below the
  analytic mask.
* **Per-pixel check** (`harm_check.py`): harmful samples get weight 0.44
  (analytic 0.65), helpful ones 0.76 (0.91).
* **Unseen burst2:** visibly sharper fine detail than the analytic mask.
* **Known cost:** flat areas (sky) can be ~25% noisier.

Sharpening defaults to off for v6.

## In the app (implementation)

`Config::robust_nn_enabled`, default off. The toggle is wired through
`CameraModel.TuningParams` → `SRBridge` → `Config`.

How it runs (`core/metal_gpu.mm`, `core/HHSRKernels.metal`):

* **Per burst:** `init_robustness_metal_impl` pins the reference grey at guide
  resolution (`rnn_grey_from_guide`). The analytic path only keeps 3×3 means;
  the net needs the pixels.
* **Per frame:** after the raw-resolution analytic mask is final (after the
  Eq. 9 min on the guide lattice), `rnn_refine_raw_res` runs one fused kernel,
  `rnn_refine_mask`, at guide resolution:
  * The analytic R is constant per RGGB quad, so the guide grid loses nothing.
  * The kernel computes each pixel's correspondence exactly as `rob_make_mask`
    does: nearest tile or affine flow at the quad centre, ×½, plus the
    optional homography.
  * Each 16×16 threadgroup keeps its warped grey (26×26) and its 18×18
    feature halo in threadgroup memory (~21 KB), so no feature planes reach
    device memory.
  * The result is written back quad by quad.
* **Memory:** about 3 × (gh·gw·4 B), roughly 36 MB on a 12 MP burst, all freed
  with the burst's robustness state. Nothing is allocated while the toggle is
  off.
* **When it declines** (mask unchanged): non-Bayer input,
  `robustness_perceptual_sqrt`, `dense_flow_enabled`, per-channel R, or a
  missing kernel.

`core/robust_nn_shared.h` is the single source of the features and net, for
CPU and Metal alike. `core/robust_nn_weights.h` is generated. To ship
different weights: `python export_weights.py --model runs/<x>/model.pt`.

Checks (CPU only; Metal itself cannot be compiled here):

* `parity_test.py`: shared header vs PyTorch, about 1e-6.
* `app_kernel_test.py`: compiles the shipped `rnn_refine_mask` source
  (extracted from `HHSRKernels.metal`) as C++ through a thin MSL shim, runs it
  single-threaded on real data and compares to PyTorch. Result: 2.3e-6.
  This covers the tiling, halo, clamping and tile-flow lookup.

## Reproduce

```
python gen_dataset.py --out data/train --n 1400 --seed0 0
python gen_dataset.py --out data/val   --n 160  --seed0 1000000
python train.py --train data/train --val data/val --out runs/v1
python evaluate.py --model runs/v1/model.pt --out runs/v1/eval
python export_weights.py --model runs/v1/model.pt
python parity_test.py --model runs/v1/model.pt
```

### Using the app's own aligner (strongly recommended next step)
The training data should contain the failures *your* aligner makes. Build a
CLI around the production aligner:
`aligner_cli ref.f32 cmp.f32 W H tile out.f32` (out = th·tw·2 float32,
(dx,dy) grey px). Then pass `--align_cmd "path/to/aligner_cli"` to
`gen_dataset.py`.

## Results (held-out synthetic, 6 bursts/scenario, merged-image PSNR)

These are bursts from my own aligner, not real photos. dB is a coarse summary;
look at `runs/*/eval/*.png`.

| | ones | analytic r0 | v1 | **v2 (shipped)** | oracle |
|---|---|---|---|---|---|
| mean | 35.97 | 38.06 | 38.41 (+0.35) | **38.51 (+0.45)** | 38.87 |

v2 by scenario, NN minus r0:

| Scenario | v2 vs r0 |
|---|---|
| repetitive | +2.06 |
| rotation | +0.48 |
| zoom | +0.38 |
| tremor | +0.37 |
| saturated | +0.28 |
| motion_blur | +0.23 |
| parallax | +0.14 |
| moving | +0.10 |
| low_light | +0.03 |

Other measurements:

* **Different analytic formula as input** (no 5×5 min): the NN gains +1.86 dB
  over it and lands at about the same absolute quality.
* **Rotation sweep** (v1, real aligner output, no corruption): weight on
  harmful samples roughly halves at every angle (e.g. 0.57 → 0.30 at 1°). The
  gain holds outside the training range (+0.56 dB at 2.5°). Neutralising the
  flow-geometry features removes most of the gain, so the net uses the flow
  field's rotation evidence.

**Output sharpening (shipped default).** `R' = clamp((R − 0.30)/0.40)`,
via `Config::robust_nn_reject_below` / `_accept_above` (Settings sliders).
Measured on 27 held-out bursts (`sharpen_sweep.py`, v2):

| v2 mask | dPSNR vs r0 | d√-PSNR | leak | keep | worst scenario |
|---|---|---|---|---|---|
| raw | +0.43 | +0.21 | 0.29 | 0.72 | parallax −0.32 |
| **[0.30, 0.70]** | **+0.99** | **+0.87** | 0.22 | 0.88 | low_light −0.08 |
| oracle labels | +0.80 | +0.75 | 0.02 | 0.98 | low_light −0.77 |

The remaining 0.22 leak is outright misjudgement (raw R > 0.5 on harmful
pixels), not hedging. That part needs the asymmetric-loss retrain.

Known weakness: the NN under-trusts correct samples. Keep weight is 0.70–0.76
vs 0.85 for r0, which costs ~0.2 dB in the sqrt domain on parallax and moving
scenes. v2 (`--util_sharp 3`) reduced this relative to v1. `train_merge.py`
(v3) fine-tunes on the merged-image error directly to address it.
