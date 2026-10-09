# mask_refine — a learned refinement of the Wronski robustness mask

A tiny depthwise-separable CNN that may only ever *withdraw* acceptance from the
analytic mask:

    M_final = M_wronski · g,   g = sigmoid(net(features)) ∈ (0,1)

**Result: it does not work. Final `+0.00 dB` — the trained checkpoint is the
identity map.** The reason is measured, and it is *not* the loss, the optimiser,
the capacity or the training time: the input features lack the one quantity that
predicts where merging hurts. Details under *What the measurements say*.

Everything below is reproducible from the scripts in this directory.

## Layout

| file | what |
|---|---|
| `gen_bursts.cpp` | synthesises bursts from real DNGs and runs the **real** pipeline on them |
| `data.py` | reads the dumps, builds the 12 features, applies any mask through the real merge |
| `model.py` | the network, and the `M_final ≤ M_wronski` invariant |
| `losses.py` | Charbonnier + ghost + suppression (+ optional oracle distillation) |
| `train.py` | training against the final merge |
| `test_invariant.py` | the invariant, polarity and merge-linearity, tested automatically |
| `evaluate.py` | PSNR/SSIM/ghost/retained/latency/params, per regime, plus a feature ablation |
| `probe_headroom.py` | the free per-pixel upper bound (**overstates it — see below**) |
| `probe_distill.py` | expressivity vs optimisation: regression onto the smooth oracle, merge out of the loss |
| `analyse_gain.py` | what kind of burst carries the headroom |

## Pipeline facts this depends on (inspected, not assumed)

* **Mask polarity**: `merge.cpp` accumulates `val += w·M·c`, `acc += w·M`, so
  **M = 1 accepts** a sample fully and **M = 0 removes it**. Asserted in
  `test_invariant.py` by checking `M = 0` reproduces the reference-only merge.
* **Resolution**: one channel on the guide lattice (`ref_means`), half the raw
  dimensions here.
* **Flow**: `FlowField`, one vector per alignment tile, in raw pixels. The merge
  samples the comparison frame at *reference position + flow*
  (`merge.cpp:167`), which fixes the sign convention.
* **Linearity**: the merge is linear in M, separately per frame. Running it once
  per frame at `M ≡ 1` yields `A_n`, `B_n`, and then

      out = (A_ref + Σ A_n·M_n) / (B_ref + Σ B_n·M_n)

  is exactly what the pipeline produces for any mask. That is why the generator
  dumps factors instead of pictures: training varies the mask without
  re-merging, with an exact gradient, using the pipeline's own arithmetic.

## Reproducing

    # 1. bursts from three scenes (daylight street, night street, residential)
    ./gen_bursts.exe "C:/.../APC_1186.dng,C:/.../APC_1261.dng,C:/.../im_00.dng" \
                     data 54 5 1024

    # 2. the invariant must hold before anything is trained
    python test_invariant.py --data data

    # 3. is there anything to find, and can the features express it?
    python probe_headroom.py --data data        # free per-pixel bound
    python probe_distill.py  --data data        # the decisive one
    python analyse_gain.py   --data data --n 24 # what carries the headroom

    # 4. train, then evaluate
    python train.py --data data --minutes 24 --window 256 --w-suppress 0.0
    python evaluate.py --data data --ckpt mask_refine.pt --ablate

Build: compile `gen_bursts.cpp` with `core/{grey_pyramid,align,robustness,
kernels,merge,raw_io,snr_tuning}.cpp` plus `tools/rob_refine/host_stubs.cpp`,
`-DHAVE_LIBRAW -Icore -Ivendor/LibRaw`, link `vendor/LibRaw/lib/libraw.a
-lws2_32 -static`. **Rebuild every object after touching `core/types.h`** —
`Config` is passed by value, so one stale object gives two layouts, links
cleanly, and segfaults.

## The training data

54 attempted, 47 written (7 cells had no usable crop), 6.5 GB.

* **Three scenes**, round-robin, so each is evenly represented.
* **1024-px crops**, stratified over a grid so crops span the picture.
* **Quality filter** with up to 12 retries: rejects crops that are blown, black,
  or have too little structure *relative to their own brightness*. The
  brightness floor is 0.004, not 0.015 — 0.015 silently discarded every crop
  from the night scene, which sits at ~0.012 mean.
* **Four motion regimes**: subpixel camera motion throughout, plus parallax,
  an independently moving object with occlusion, and rotation. The CFA is
  preserved by sampling the mosaic on its own parity lattice.
* Spans brightness 0.005–0.20, noise gain 1.2–8.6, flow error 0.6–13.5 px.

### Known defect: the window sampler has no brightness floor

The quality filter screens a **1024-px crop** on its overall brightness, but
training and the probes sample a **256-px window** inside it, and nothing
constrains that window. **10 of 24 sampled windows are effectively black**
(`gt` mean < 0.001; five are exactly 0.0000). PSNR on those is meaningless, and
they inflated every headroom figure measured before this was found. Fix the
sampler in `data.py` before trusting any further probe output.

### Known defect: regimes are not recorded

`gen_bursts.cpp` prints its regime choice but does not store it in the dump, so
per-regime evaluation is impossible after the fact. `analyse_gain.py` works
around it by describing each burst through the spatial structure of the stored
`ferr` instead. Write the label into the header if the data is regenerated.

## What the measurements say

**1. The free per-pixel bound is not the learnable one.** `probe_headroom.py`
optimises one free value per mask pixel with the target in hand and reports
about **+1.1 dB**. Most of that is the oracle absorbing per-pixel noise, which
no convolutional gate can reproduce.

Measuring the *learnable* ceiling requires optimising **under** the smoothness
constraint — free parameters on a coarse lattice, bilinearly upsampled — not
blurring the free optimum afterwards. Those are different questions and give
different answers (+0.02 dB for the blurred optimum, +0.4 to +0.7 dB for the
constrained one). Getting this backwards produced a wrong "nothing to learn"
conclusion once already.

On the 4x-coarse lattice over 24 bursts: **+0.73 dB**, but restricted to
well-exposed windows (excluding the black ones above): **+0.49 dB**. That is
the honest ceiling.

**2. The features carry some of it, and none of it generalises.**
`probe_distill.py` takes the merge out of the loss entirely and regresses the
network directly onto the smooth oracle gate:

    ON FIT WINDOWS (expressivity): oracle +0.729 dB  net +0.280 dB  = 38%  R² 0.224
    HELD OUT (generalisation):     oracle +0.479 dB  net +0.043 dB  =  9%  R² -0.011

With 1641 parameters against ~123k targets the net cannot memorise pixel-wise;
what it memorises is *burst-level* structure — brightness, noise gain, the
scene — which is exactly what fails on unseen bursts. And the 38% it does fit
is concentrated in the near-black windows, where the correct gate follows
brightness and noise gain, two of its own inputs. On well-exposed windows it
recovers 4–29%.

This rules out the explanation that the merge-loss gradient was the obstacle.
Distillation is the fix for that, and distillation does not work either.

**3. Headroom tracks the *contiguity* of misalignment, not its size.**
`analyse_gain.py`, well-exposed windows only:

    ferr_blob   +0.517     contiguity of the high-error region
    noise gain  -0.342
    ferr_mean   -0.294     MAGNITUDE — and the wrong sign
    ferr_ramp   +0.284
    maskw       -0.203
    bright      +0.012

The magnitude of the true flow error does not predict headroom and if anything
predicts it inversely. `ferr_blob` — misalignment forming a compact region with
a hard boundary, i.e. an occluding object or a depth edge — is the strongest
signal, and it is the *structural* one that can be recovered from the
**estimated** flow. None of the 12 current features measures it.

**4. Training therefore converges to the identity, five times over.**

    w_suppress 0.1,   absolute losses                    → identity
    w_suppress 0.005, relative losses                    → identity
    w_suppress 0.0,   relative losses                    → identity
    + noise model corrected                              → identity
    + warped-image features, head_bias 1.0               → identity, +0.00 dB

Three real causes were found and fixed along the way. None changed the outcome,
and all three are worth knowing:

* **Scale.** At 43 dB the image error is ~0.001 while withdrawn mask is ~0.05,
  so the suppression gradient was **63,000×** the reconstruction gradient. Fixed
  by making the image terms *ratios to the Wronski baseline*, which is
  scale-free. That cut it to 183×.
* **Coherence.** The remaining 183× was still fatal: the suppression gradient
  points in *exactly* the same direction in every window (cosine +1.000, sd
  0.000) while the image gradient is genuine but variable (+0.600, sd 0.800). A
  perfectly coherent gradient beats a noisy larger one over thousands of steps,
  so *any* non-zero weight eventually wins. The guard now lives in checkpoint
  selection, not in the gradient.
* **Noise model.** The generator injected `var = α·gain·μ + β·gain²` while the
  mask's own noise model assumed no gain factor, giving
  `corr(gain, meanR) = −0.843` — brighter bursts systematically over-rejected.
  Fixed by scaling the noise profile per burst instead; `corr` became +0.378.

## What would have to change

Not the architecture, the capacity, the loss weights or the training time —
each was tested. Two concrete prerequisites before training again:

1. **A contiguity / neighbour-consensus feature.** `ferr_blob` is the only
   descriptor that both predicts headroom and has an observable counterpart.
   `tile_reject_domain_retrain` reached the same conclusion independently
   (neighbour consensus separating 4.8×), so two routes now point at the same
   missing input.
2. **A brightness floor in the window sampler**, so the metric means something.

Run `probe_distill.py` (~25 min) before any further training run: it decides
expressivity versus optimisation without spending a 24-minute training gamble
to find out.

## A pipeline defect found on the way

`core/robustness.cpp:2187` writes the mask with no `isfinite` guard, unlike the
raw-resolution path at line 1551 which has one. `compute_robustness_core` emits
a NaN roughly once in 1.3M pixels, and a NaN mask multiplies straight through
the merge into a NaN output pixel. `data.py` treats them as fully rejected,
which is what the guarded path does. **Not fixed here** — it is a one-line
change in `core/`, left for a separate decision.
