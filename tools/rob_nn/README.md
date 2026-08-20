# Learned robustness correction

A small convolutional network that **multiplies** the analytic robustness mask
(Wronski et al. Eq. 5–9, `core/robustness.cpp`):

```
R_final = R_analytic * C_nn          C_nn in [0,1]
```

The analytic mask is untouched and remains the primary decision. The network
contributes only `C`, and the only thing done with it is a multiply, so the
learned part **can never raise `R`** — for any pixel, on any input, however
badly it was trained. That is a property of the graph (sigmoid head in the
exported model, multiply in `compute_robustness`), not something the loss is
hoped to deliver.

Enabled by the Settings toggle **Learned Robustness Correction**; off by
default, and it falls back to the plain analytic mask whenever the model is
missing or fails to load.

## Resolutions and coordinate systems — verified, not assumed

Two real bugs in this project came from getting this wrong (a guide-versus-raw
pixel confusion, and a double-applied warp), so it is written down. Numbers are
measured on `ok/burst1` (iPhone, 12 MP Bayer); the relationships hold generally.

| quantity | resolution | units and indexing |
|---|---|---|
| raw Bayer frame | 4032 × 3024 | raw px |
| guide image (`compute_guide`) | 2016 × 1512 × 3 | one guide px = one Bayer quad; guide `(y,x)` covers raw `(2y..2y+1, 2x..2x+1)` |
| `RefStats.means` / `.stds` | guide res, 3 ch | 3×3 box over the guide = 6×6 raw support. **`.stds` holds VARIANCE**, not σ — the feature builder takes the root |
| `RefStats.nn_luma` | guide res, 1 ch | unsmoothed guide luma; filled in `init_robustness` from `ref_raw` |
| `RefStats.means_hires` / `.stds_hires` | raw res, 3 ch | only when `robustness_raw_resolution_active()`; Dodgson upscale |
| `FlowField` after `flow_to_raw_tile_grid` | 252 × 189 tiles | `tile_size` = `bm_tile_sizes[0]` = **16 RAW px**; `dx`/`dy` are **RAW px** |
| `R_normal` (`compute_robustness_analytic`) | guide res × 1 | after Eq. 9's 5×5 minimum |
| `C_nn` (network output) | guide res × 1 | same grid as `R_normal` — the multiply needs no resampling, and `compute_robustness` refuses to run if the two disagree |
| NN feature planes | guide res, 27 ch interleaved | strips of `192 + 2*8` rows |

The conversions that have to be right, and where each lives:

* **flow → guide offset.** The flow is in raw px and the guide is half
  resolution, so a guide-space fetch is at `(y + 0.5*fy, x + 0.5*fx)`. Applied
  identically by the analytic path (`compute_robustness_analytic`) and by
  channels 6–8 and 20–26 of `build_robustness_nn_features`.
* **guide pixel → tile index.** Raw coordinate `2y`, divided by the raw
  `tile_size`. The analytic path uses `(2y + 0.5)/tile_size` (nearest tile);
  the feature builder uses `2y/tile_size - 0.5` and interpolates between tile
  *centres*. The half-guide-pixel difference is deliberate: the merge must
  fetch with the tile vector the search actually evaluated, while the mask is
  reasoning about whether the motion is plausible, and a piecewise-constant
  input makes the network draw the tile grid.
* **merge → mask.** `merge.cpp` samples a guide-resolution `R` at
  `(lr - 0.5)/2` where `lr` is the raw coordinate, and decides guide-versus-raw
  from `R`'s **actual dimensions**, not from the config flag — because the
  raw-res path silently returns guide resolution when the hires reference stats
  are missing.
* **the double warp.** At raw resolution `upscale_warp_stats` has *already*
  applied the flow to the comparison statistics, so the correct sample sits at
  `(y,x)` and shifting again would apply it twice. The feature builder guards
  this, and channels 20–26 are simply zero at raw resolution rather than
  guessing which frame the unwarped luma plane belongs in.

## Why a correction rather than a replacement

The previous design had the network replace Eq. 5–9 outright. That made every
weakness of the model a regression in regions the closed form already handled
correctly, and it needed a 0.989 decision gate to be safe at all. The gate is
gone with it.

What the analytic mask actually misses is narrow and specific:

* camera **rotation** leaving some tiles translation-aligned,
* **curved / non-straight** camera motion,
* local **object motion**, **occlusion** and **disocclusion**,
* wrong-alignment tiles during those motions.

All of these can leave the local mean, variance, noise estimate and flow
smoothness almost unchanged while shifting an edge by a pixel:

```
reference edge:   |
warped edge:        |
```

Same statistics, duplicated edge in the output. `d²/σ²` between 3×3 means
cannot see it, and no tuning of `s1`/`s2`/`t` recovers information the
statistic never carried.

## The spatial channels — the point of the redesign

Channels 0–19 are all local *statistics*. Channels 20–26 are the residual
itself:

| ch | contents |
|---|---|
| 20 | reference guide luma, **unsmoothed** |
| 21 | comparison guide luma, unsmoothed, bilinear at the flow offset |
| 22 | signed residual, 20 − 21 |
| 23–24 | reference luma gradient, guide px |
| 25 | regularised displacement estimate `r·\|g\| / (\|g\|² + eps)`, clamped ±4 guide px |
| 26 | residual over the expected sensor σ |

Channel 25 is the one that turns a brightness difference into a *shift*: for a
translation `e` along the edge normal the residual is `r ≈ e·g`, so `e ≈
r·|g|/|g|²` — with the noise as the Tikhonov regulariser, so it decays to 0 on
flat content instead of exploding.

These are handed over as **planes**, not as gathered patches. The network is
fully convolutional with a 7-guide-pixel receptive field, so a plane gives it a
15×15 guide-pixel (30 raw-pixel) patch around every output pixel for four extra
reads — and 30 raw px spans most of two 16-raw-pixel tiles, which is what lets
the residual's **ramp across a tile** be seen at all. That ramp is the
signature of a translation-only flow fitting a locally rotating scene, and no
pointwise statistic carries it.

Not derivable from `RefStats.means`: those are 3×3 box means over a 6×6 raw
support, and the box is exactly what erases the shifted edge. `nn_luma` is
therefore filled in `init_robustness`, the last place `ref_raw` is in scope —
including on the Metal path, where `means`/`stds` never leave the GPU.

## Match quality (channels 18–19)

Carried over from branch `matchq-mask-4da49a9`, where it was the only change in
many attempts that *lifted* the accuracy curve rather than sliding along it.

* **18** `log(cost at the chosen offset / best cost in a ±4 guide-px window)` —
  0 when the offset being judged is the best correspondence on offer.
* **19** `log(best rival outside the winner's basin / best cost)` — how
  *isolated* that minimum is.

Aliasing changes the residual at the **bottom** of the cost surface without
flattening it, so a correctly aligned but heavily aliased tile keeps a sharp
isolated minimum, while a tile matched onto similar-looking content elsewhere
has several near-ties. Photometric channels cannot separate those; the cost
surface's shape separates them directly.

Both are measured on the flow **being judged**, never on the baseline —
measuring at the true offset hands the network the answer and scores
wonderfully while being useless in the app.

Two pitfalls from that branch, both preserved here:

* an early version compared the chosen offset against a ring 2–3 px away and
  called it uniqueness. It scored uncorrupted tiles 0.09 and corrupted ones
  0.78 — backwards. It was measuring local *steepness*, which is as large at a
  wrong offset on a gradient as at a right one, and its exclusion ring hid the
  true optimum whenever the error was smaller than the ring. Both statistics
  are now anchored on the minimum over a real search window.
* what that branch **shipped** never ran: `measure_match_quality` existed and
  `compute_robustness` never called it, so the app fed zeros to a model trained
  on real values. It is wired in this design.

## Training data — real imagery, corrupted flow only

`rob_real.cpp`. Both frames are captured raws with their real noise, real
optics and real parallax; only the flow field is corrupted.

1. run the real aligner on a real (reference, comparison) pair → `flow_base`;
2. verify `flow_base` per tile photometrically and give the tiles it does not
   vouch for **weight 0** — not label 0, since that test is `d²/σ²` and
   labelling from it would train the network to copy the analytic mask;
3. corrupt `flow_base` into `flow_est` with a chosen failure pattern and
   magnitude;
4. label `harm(p) = |comp(p + flow_est) − comp(p + flow_base)|`, both samples
   from the **same** captured comparison frame.

Step 4 is exact and non-circular. Comparing against the *reference* would
punish ordinary aliasing — the frame-to-frame difference super-resolution
exists to exploit — and is the mistake that produced an over-rejecting mask
before. Comparing the fetch against the *correct fetch* is exactly 0 however
aliased the content.

Fourteen corruption patterns (one tile, neighbouring tiles, a group locked onto
one wrong motion, a smooth wrong field, an abrupt boundary, spurious rotation,
translation+rotation, wrong magnitude, wrong direction, along-edge,
across-edge, photometrically-camouflaged, global) crossed with a nine-rung
magnitude ladder from 0.05 px to 256 px, so failure *type* and failure *size*
vary independently. Each burst gets its own seed, pattern emphasis, band bias,
corruption density and exposure ladder.

**Paired examples.** Every corrupted variant is emitted twice — once with the
corrupted flow, once with the baseline flow — over the *same* crop of the same
frame pair at the same exposure with the same noise realisation. Only the
correspondence differs. Without that, a model can satisfy the data by learning
"rotation is dangerous"; these bursts carry 0.25–2.0° of real camera rotation
that merges perfectly well, and a mask that rejects it throws away the burst.

**Scene motion is a separate label component.** The harm measure is
comp-versus-comp, so it is identically zero over a walking person where the
baseline flow is already wrong. Detected instead by multi-frame agreement over
the 7–8 frames: a frame whose warped difference is an outlier against the
others' median is occluded *in that frame*; a pixel where every frame agrees
and disagrees with the reference is one the reference is the odd view of, which
is the case that ghosts a moving subject. Validate the maps visually
(`ROB_DUMP_OCC`) before training on them.

## Labels are HARM, not error magnitude

`R_ideal = exp(-z²/4)` with `z` the mis-fetch in units of the sensor's own σ,
measured with the bilinear interpolation footprints and their overlap accounted
for. Assuming a flat `2σ²` noise floor was measured to destroy the labels
outright — the 0.25–1 px and 1–4 px bins came out with exactly zero harm at the
99th percentile.

So 0.5 px in flat sky is labelled safe and 0.2 px across a thin wire is
labelled harmful, which is the point. The training target is then the ratio
against the analytic mask's own finished answer:

```
C_target = clip((R_ideal + 0.05) / (R_normal + 0.05), 0, 1)
```

Exactly 1 wherever the closed form is not too high, which is most of the frame.

## Pipeline

```bash
# 1. training set from real bursts (guide resolution, 35 float32 channels)
ROB_CROP=112 ROB_CROPS=2 ROB_REFS=2 ROB_VARIANTS=12 ROB_SEED=11 ROB_GAINS=1,4 \
  ./rob_real robset 1 ok/burst1/*.dng
ROB_APPEND=1 ... ./rob_real robset 2 ok/burst2/*.dng     # different seed/gains
ROB_APPEND=1 ... ./rob_real robset 3 ok/burst3/*.dng

# 2. train
python train_rob.py robset            # ROB_STEPS=4000 by default

# 3. the two numbers, on held-out records
python eval_rob.py

# 4. hard-case mining, then retrain
python mine_rob.py                    # writes hard.txt
ROB_HARD=hard.txt ... ./rob_real ...  # regenerate with that emphasis

# 5. look at it on a real burst
./rob_vis vis ok/burst1/*.dng && python vis_rob.py vis ok/ burst1

# 6. export for the app
python export_coreml.py robnet.pt RobustnessNet.mlpackage
```

`rob_real.cpp` builds against the core objects:

```bash
g++ -O2 -std=gnu++17 -D_USE_MATH_DEFINES -DHAVE_LIBRAW -I../../core \
    -I../../vendor/LibRaw -pthread \
    rob_real.cpp align.o grey_pyramid.o kernels.o merge.o raw_io.o \
    robustness.o snr_tuning.o pipeline.o prof.o debug_utils.o \
    robustness_nn.o libraw.a -o rob_real
```

`-D_USE_MATH_DEFINES` is required on MinGW, which hides `M_PI` without it.

## Feature layout

27 planes at guide resolution. This ordering is a contract between
`rob_real.cpp` (writes the training set) and `build_robustness_nn_features`
(`core/robustness.cpp`, builds them at inference); changing it means
retraining, re-exporting, **and** rebuilding the app.

| channels | contents |
|---|---|
| 0–2 | reference 3×3 local mean, RGB |
| 3–5 | reference 3×3 local standard deviation, RGB |
| 6–8 | comparison 3×3 local mean, sampled where the estimated flow points |
| 9–10 | estimated flow dx, dy, in **raw** pixels |
| 11 | local span of the flow field over the 3×3 tile neighbourhood (Eq. 7's M) |
| 12 | expected noise σ at this brightness |
| 13 | `log1p(d²/σ²)`, Eq. 6 — the analytic mask's decision statistic |
| 14 | the analytic mask's pointwise `R` (Eq. 5, **before** the 5×5 minimum) |
| 15 | `\|flow − median of the 3×3 tile neighbourhood\|`, raw px |
| 16 | the neighbourhood max of channel 15 |
| 17 | reference local high-frequency energy |
| 18–19 | match quality: off-by, and isolation |
| 20–26 | the spatial residual block (see above) |

Channel 14 is **evidence, not a target**: the network is trained against
measured merge harm and is free to overrule it. It is also *not* the divisor in
the training target — that is `R_normal`, the finished mask after Eq. 9's 5×5
minimum, carried in analysis channel 34. Confusing the two would train the
network to correct a mask nobody runs.

## Memory

The weights are tiny; the activations are not. At guide resolution
(2016×1512 = 3.05 MP) one 32-channel intermediate tensor is 390 MB, Core ML
keeps several live, and 27 interleaved feature planes cost 329 MB plus as much
again in the planar `MLMultiArray`. Whole-plane inference peaks over a gigabyte
on top of a burst pipeline already holding several 12 MP frames — a jetsam
kill, not a slowdown.

Inference therefore runs in horizontal strips of 192 guide rows with an 8-row
halo (the network's exact receptive-field radius), which brings the peak to
roughly 150 MB.

Each window is the same height and is kept fully inside the image, clamped near
the bottom rather than padded past it. That is what makes the strip output
bit-identical to whole-plane inference: the window edges coincide with the
image edges, so the convolutions' zero-padding matches. Padding past the edge —
by replication or by explicit zero rows — does **not** reproduce it, and both
wrong variants were measured and produce visible strip seams (max |diff| 0.95
and 0.21 respectively, against 3e-8 for the correct windowing).

## Pitfalls

* **The Core ML wrapper fails CLOSED.** If the built feature channel count
  differs from the bundled `.mlmodelc` the model is rejected, the pipeline
  silently uses the analytic mask, and a rebuild looks "unchanged". It logs
  this. If you change `kRobustnessNnChannels` you MUST re-export; verify the
  spec's input shape (`export_coreml.py` prints it) and confirm the bundled
  file is newer than the `.pt` it came from.
* **`init_robustness_metal` returns a `RefStats` with valid dimensions and
  EMPTY data vectors** — the statistics stay on the GPU. Guard on
  `data.size()`, never on `h`/`w`/`c` alone. Ignoring this crashed the app on
  frame 2.
* **`coremltools` cannot run predictions off macOS**, so export parity cannot
  be checked on a Windows or Linux host. The parity block in
  `export_coreml.py` prints that it was skipped; do not read a skipped check as
  a passed one.
* **Flat-on-flat misalignment is photometrically undetectable** — but it is
  also largely invisible, so the target is visible harm rather than all harm.
