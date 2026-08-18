# Learned robustness mask

A small convolutional network that replaces the analytic robustness mask
(Wronski et al. Eq. 5–9, `core/robustness.cpp`). Enabled by the Settings
toggle **Learned Robustness Mask**; off by default, and it falls back to the
analytic mask whenever the model is missing or fails to load.

## Why

The analytic mask decides from `d²/σ²` — a colour difference between 3×3 means
of the half-resolution guide image. Two failures are invisible to that
statistic no matter how `s` and `t` are tuned:

- a tile whose flow is badly wrong but whose wrongly-fetched content resembles
  the right content (flat shadow onto flat shadow): `d` is genuinely near zero;
- an error in structure finer than the 6×6-raw support of those means (a thin
  rod): the mean barely shifts while `σ` rises, so `d²/σ²` *falls*.

Both are the same shortage of information, so no tuning recovers them. What
distinguishes these cases is geometric, not photometric: the flow is
implausible, or disagrees with its neighbours. The network is given that
evidence — the estimated flow, its local spread, and a ~30 raw-pixel
receptive field — alongside the same statistics the analytic mask uses.

This follows a suggestion from the paper's first author, who noted that today
he would use a network for the robustness part specifically, to avoid the
manual tuning and heavy reliance on camera metadata the analytic model needs.

## Measured

Ground truth comes from synthetic bursts (below), scored per pixel as
"would merging here damage the output". Rank-based AUC — the probability that
a harmful pixel is trusted less than a harmless one — plus detection rate at a
fixed false-reject budget:

| mask | AUC | detection @2% FP | @10% FP |
|---|---|---|---|
| analytic (s1=2, s2=12, t=0.12) | 0.638 | 25.6% | 25.6% |
| learned | **0.926** | **37.6%** | **73.3%** |

The analytic mask's detection does not improve with a larger false-reject
budget because it pins most pixels at exactly `R = 1`; no threshold separates
them. That saturation is the blind zone `ln(s1/(1+t)) = 0.58` seen directly.

## How the labels avoid being circular

Labelling by "does the comparison frame differ from the reference" would
punish aliasing — the legitimate frame-to-frame difference that
super-resolution exists to exploit — and a network trained on it would learn
to reject exactly the signal we want.

Instead each comparison frame is *synthesised* by warping a real raw by a
known rotation and translation, so the true flow is known analytically. The
label is then

```
harm = | comp(p + flow_estimated) − comp(p + flow_true) |
```

what we fetched versus what we should have fetched, both from the same frame.
That is exactly zero whenever the flow is right, however aliased the content,
and grows only with genuine misalignment. It is measured at raw resolution, so
errors in structure finer than the guide's 3×3 mean survive into the label even
though they are invisible to `d`.

Synthesis resamples each CFA colour on its own half-resolution lattice, so the
result stays a true Bayer image (a demosaic/remosaic round trip would erase
the aliasing), and adds heteroscedastic noise from the DNG model.

## Pipeline

```bash
# 1. training set from your own raws (guide-resolution, 15 float32 channels)
./rob_dataset out_prefix ref.dng N_FRAMES [more_refs.dng ...]
#    ROB_ROT_DEG / ROB_SHIFT_PX set the synthetic motion range; the defaults
#    span deliberate camera rotation, not the ~1e-3 rad hand tremor the
#    published tuning was validated on.

# 2. train (also prints the analytic mask's score on the same data)
python train_rob.py out_prefix          # ROB_STEPS=1500 by default

# 3. fair head-to-head on held-out frames
python eval_rob.py

# 4. export for the app
python export_coreml.py robnet.pt RobustnessNet.mlpackage
```

`rob_dataset.cpp` builds against the core objects, e.g.

```bash
g++ -O2 -std=c++17 -DHAVE_LIBRAW -I../../core -I../../vendor/LibRaw -pthread \
    rob_dataset.cpp align.o grey_pyramid.o kernels.o merge.o raw_io.o \
    robustness.o snr_tuning.o prof.o debug_utils.o libraw.a -o rob_dataset
```

## Feature layout

13 planes at guide resolution, interleaved. This ordering is a contract
between `rob_dataset.cpp` (writes the training set) and
`build_robustness_nn_features` (`core/robustness.cpp`, builds them at
inference); changing it means retraining.

| channels | contents |
|---|---|
| 0–2 | reference 3×3 local mean, RGB |
| 3–5 | reference 3×3 local standard deviation, RGB |
| 6–8 | comparison 3×3 local mean, sampled where the estimated flow points |
| 9–10 | estimated flow dx, dy, in **raw** pixels |
| 11 | local span of the flow field over the 3×3 tile neighbourhood (Eq. 7's M) |
| 12 | expected noise σ at this brightness |

Channels 9–12 are the ones the analytic mask cannot use. Channel 12 goes
through the same debug-gated accessors as the analytic path, so it vanishes
with the "Disable Noise Model" toggle.

Input normalisation is folded into the exported graph, so the C++ side hands
over raw planes and there is no constant table to drift out of step with the
weights.

## Memory

The weights are ~110 KB, which says nothing useful about the cost of running
the model: the activations dominate. At guide resolution (2016x1512 = 3.05 M
pixels) one 32-channel intermediate tensor is 390 MB, Core ML keeps several
live, and the feature planes cost 158 MB interleaved plus another 158 MB in
the planar `MLMultiArray`. Running the whole plane in one call peaks over
1 GB on top of a burst pipeline already holding several 12 MP frames, which
is a jetsam kill on device, not a slowdown.

Inference therefore runs in horizontal strips of 192 guide rows with an
8-row halo (the network's exact receptive-field radius), which brings the
peak to roughly 150 MB -- about a 7x reduction.

Each window is the same height and is kept fully inside the image, clamped
near the bottom rather than padded past it. That is what makes the strip
output bit-identical to whole-plane inference: the window edges coincide with
the image edges, so the convolutions' zero-padding matches. Padding past the
edge -- by replication or by explicit zero rows -- does **not** reproduce it,
because those rows feed bias-driven activations into the next layer where
whole-plane inference has true zeros. Both wrong variants were measured and
produce visible strip seams (max |diff| 0.95 and 0.21 respectively, against
3e-8 for the correct windowing).

## Status

**Not yet exercised on device.** The model was trained and converted on a
machine with no Core ML runtime, so `robustness_nn.mm` has never been loaded
or run. Every entry point fails closed — `robustness_nn_available()` returns
false and `compute_robustness` uses the analytic mask — so a broken model
degrades to current behaviour rather than a wrong mask. First on-device run
should confirm the model loads and spot-check the mask against the analytic
one on the same burst.

`export_coreml.py` prefers the `mlprogram` format and falls back to
`neuralnetwork` when coremltools cannot write an mlprogram weight blob (its
BlobWriter extension ships only for macOS/Linux). The checked-in
`Resources/RobustnessNet.mlmodel` was produced by that fallback; re-running
the export on a Mac yields the smaller fp16 `.mlpackage`, which is preferable
if the deployment target allows it.

The current weights were trained on 12 synthetic frames from three raws of a
single burst — enough to demonstrate the gap, not enough to generalise. More
scenes and lighting conditions should go in before this becomes the default.
