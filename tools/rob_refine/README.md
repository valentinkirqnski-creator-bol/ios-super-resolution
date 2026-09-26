# Learned refinement of the robustness mask

A small network that sits **on top of** the analytic mask (Wronski et al.
Eq. 5–9, `core/robustness.cpp`) and is only ever allowed to take weight away:

```
R_final = R_wronski * (1 - kappa * (1 - q_keep))
```

Enabled by `Config::robustness_refine_nn_enabled`; off by default, and it
leaves `R` exactly as the analytic mask produced it whenever the model is
missing or inference fails.

This is **not** `tools/rob_nn`. That trains a network that *replaces* Eq. 5–9
and targets gross alignment failures — a tile whose flow is badly wrong but
whose wrongly-fetched content happens to resemble the right content. The two
are independently toggled and can run together.

## The failure this targets

The flow field carries **one vector per tile**. Wherever the real motion has
rotation or parallax the true displacement varies *across* the tile, so the
merge fetches content that is a fraction of a pixel out — zero at the tile
centre, worst at its edges. That leaves thin doubled or thickened edges.

`d²/σ²` cannot see it, and not for want of tuning. On a textured edge `σ`
picks up the edge's own structure faster than `d` picks up a sub-pixel shift,
so the statistic becomes *more* confident exactly where the doubling is.
Measured on a rotation-smeared roofline: `d ≈ 0.0195` (genuinely larger than
the 0.0105 of well-aligned ground) against `σ ≈ 0.059`, giving `d²/σ² ≈ 0.11`
and `R ≈ 1`.

`Config::motion_geom_reject_enabled` is the hand-designed test for the same
thing: reject where `|∇I| · |E| > threshold`, with `E` the within-tile flow
error. It works, and it stays enabled — the network is measured as an addition
to it, and the four-way comparison below reports both.

What the comparison found, though, is that the hand-designed test pays heavily
for its bluntness: rejecting on `|∇I| · |E|` trips on a strong edge with a
modest `E` whether or not the misalignment is real, and it falsely rejects
2.2% of correctly-aligned pixels (4.6% of thin lines, 5.6% of the strongest
gradients) where the network rejects 0.4% (0.8%, 0.5%). See **What these say**.

## Why a network rather than more thresholds

The evidence that separates "strong edge, correctly aligned" from "strong
edge, half a pixel out" is not one number. It is the agreement between two
independent estimates of the same displacement:

- **geometric** — `E·ĝ`, the within-tile flow error projected onto the edge
  normal, from the flow field's gradient and the pixel's offset from its tile
  centre;
- **photometric** — `-res/|∇I|`, the displacement the residual implies, since
  a shift `δ` across an edge changes the sampled value by about `δ|∇I|`.

When those agree, a geometric misalignment is real. When the residual is
large but the geometry predicts nothing, it is a content change. When the
geometry predicts a shift *along* the edge rather than across it, nothing
visible happens. And structure-tensor coherence separates a clean edge from
text, foliage and repetitive texture, which is what stops "high gradient"
being read as "probably misaligned".

Both estimates, their product and their disagreement are handed over as
explicit features (channels 13, 18, 22, 23), so the network is combining
evidence rather than discovering it.

## Conservatism, enforced rather than trained

The one-sidedness is structural, not a property of the weights:

| guarantee | where it lives |
|---|---|
| `R == 0` stays `0` | the multiply in `apply_robustness_refinement` |
| `R_final ≤ R` always | same, since `q ∈ [0,1]` and `kappa ≤ 1` |
| no pixel loses more than `kappa` | `robustness_refine_max_reduction`, default 0.75 |
| small outputs change nothing at all | `robustness_refine_deadzone`, default 0.20 — inside it the pixel is passed through bit for bit |
| a missing or broken model is a no-op | every failure path returns `false` and leaves `R` untouched |

`refine_bench` checks all of these against the real code rather than
asserting them.

## Labels

Ground truth cannot come from "does the comparison frame differ from the
reference" — that punishes aliasing, which is the frame-to-frame difference
super-resolution exists to exploit, and would train the network to reject
exactly the signal we want. So comparison frames are **synthesised** from a
known warp and the label is

```
Delta = | comp(p + flow_estimated) - comp(p + flow_true) |
```

what we fetched against what we should have fetched, both from the same
frame. Zero whenever the flow is right, however aliased the content.
Measured at raw resolution, so error in structure finer than the guide's 3×3
mean survives into the label even though it is invisible to `d`.

The target weight is then derived, not tuned. A sample with noise variance
`σ²` whose misalignment contributes `Delta²` has MSE `σ² + Delta²`;
inverse-MSE weighting against a perfectly aligned sample gives it relative
weight

```
R* = 1 / (1 + Delta²/σ²)
```

1 when the flow is right, ½ when the misalignment equals one noise sigma, no
constant to pick. The network's target is `q_keep* = clamp(R*/R_wronski, 0, 1)`
— the multiplier that would turn the analytic mask into the ideal one,
clamped so it can never ask for `R` to go up.

One gate on top of that. `R*` is a little below 1 almost everywhere, because
almost every pixel carries *some* residual misalignment, so followed literally
the target asks the network to move ~96% of the frame. That is technically
optimal and the opposite of a sparse correction — and largely pointless, since
a uniform small reduction applied to every comparison frame is nearly inert
once the merge normalises `num/den`. So the target is pinned to **exactly 1**
wherever leaving `R` alone costs less than `ROB_REFINE_GATE` (default 0.02) of
excess merged-pixel MSE — about 1% in amplitude, below anything visible. On a
stratified sample that puts 47% of the target at "change nothing" and leaves
*nothing* in the 0.90–0.9999 band: what survives the gate is decisive rather
than mushy, and the network's output means something where it is not 1.

**Sanity check that matters most**: with a static camera and no flow jitter,
the label marks down 0.02% of correctly-fetched pixels; under pure sub-pixel
translation, where aliasing is maximal, 0.001%. If those numbers were not
near zero the label would be punishing aliasing and everything downstream of
it would be worthless.

## Motion

Parameterised directly by **within-tile variation** — how many pixels the
true displacement changes across one tile — swept log-uniformly, because that
is the quantity the stage judges. At `tile_size` 16 a variation of `w` px/tile
is a rotation of `w/16` radians, so the 0.8 px/tile default is 2.9°, already
past what a handheld burst does between frames. (The first draft used 3.0
px/tile, which is 10.7°; measured, that gave a 41 px mean flow error and put
the whole frame in the replacement network's territory.)

Eight regimes cycle on the frame index:

| | motion |
|---|---|
| 0 | static camera — only noise and aliasing differ; **nothing here may be rejected** |
| 1 | pure sub-pixel translation — maximal aliasing, zero within-tile variation |
| 2 | pure large translation |
| 3 | small rotation only — the target regime |
| 4 | rotation + translation |
| 5 | smooth parallax — within-tile variation not predictable from the global motion |
| 6 | parallax with a depth slab — a genuine flow discontinuity |
| 7 | everything, plus scene-change rectangles |

Parallax matters because a rigid rotation produces the *same* within-tile
variation everywhere in the frame, which a network could read off the global
flow field alone. Parallax makes it depend on the scene.

## Architecture

**Pointwise by default.** Every feature needing a neighbourhood — the
structure tensor, the flow-field gradient, the residual's local mean — is
computed in `build_robustness_refine_features`, so the network itself sees
one pixel at a time: `24 → 16 → 16 → 1`, ReLU, sigmoid, **689 parameters**.

That is a design choice, not a compromise. A receptive field of one pixel
means the on-device strip decomposition is *exactly* equivalent to whole-plane
inference — no halo, no possibility of the seams that the replacement
network's README documents at length — and the activations are 16 floats per
pixel instead of a 32-channel tensor.

`ROB_REFINE_ARCH=cnn` trains a 5×5 variant (two depthwise-separable 3×3s) for
comparison. **Shipping it requires raising `kRobustnessRefineHalo` to 2**, or
the strips seam.

## Loss

```
L = lam_fp * relu(target - pred)^2      # over-rejection, the expensive error
  + lam_fn * relu(pred - target)^2      # missed artifact
  + lam_merge * excess_merge_mse        # the physics
  + lam_id * (1 - pred)                 # a standing cost on changing anything
```

with `lam_fp = 8 lam_fn` by default. Classes are balanced by *stratified
sampling* rather than by loss weights, so the asymmetry means what it says.

`excess_merge_mse` is the term that makes this "will the merged pixel be
worse" rather than a distance between two numbers. For a reference of weight
1 and a comparison sample of weight `w`,

```
MSE(w)/sigma^2 = [1 + w^2/R*] / (1 + w)^2
```

minimised at exactly `w = R*`. It is computed **through** the cap and the
multiply the pipeline will apply, not on `q` — train on `q` alone and the
network optimises a mask nothing downstream ever uses.

Sampling is stratified by true flow error. On the natural distribution about
a third of all the mask's correctable error sits at pixels whose flow is
wrong by more than 5 px — gross search failures over flat content, the
replacement network's problem — and that mass would otherwise set the
weights.

## Pipeline

```bash
# 1. training set from your own raws (29 float32 channels, guide resolution)
./refine_dataset refineset 8 ref0.dng ref1.dng ...
#    ROB_REFINE_STRIDE (3)     spatial decimation
#    ROB_REFINE_WTV_MAX (0.8)  max within-tile variation, px/tile
#    ROB_REFINE_SHIFT (12)     max global translation, raw px
#    ROB_REFINE_JITTER (0.15)  fraction of tiles given a small extra offset;
#                              0 isolates the aligner's own error
#    ROB_REFINE_REGIME (-1)    force one motion regime, for diagnosis

# 2. train
python train_refine.py refineset            # ROB_REFINE_ARCH=mlp|cnn

# 3. the four-way comparison on held-out frames
python eval_refine.py refineset

# 4. exercise and measure the real C++ path (identity, one-sidedness, cost)
./refine_bench ref0.dng refinenet_mlp_geom.bin 1.0

# 5a. compile the weights in, for the Metal path (which uses only these)
python export_metal_weights.py refinenet_mlp_geom.pt

# 5b. export for the CPU fallback, which goes through Core ML
python export_coreml.py refinenet_mlp_geom.pt RobustnessRefineNet.mlpackage

# 6. what can be checked about the kernel without a Metal toolchain
python metal_lint.py          # run from the repo root
```

Build, from this directory (LibRaw only for the `.dng` readers):

```bash
CXXFLAGS="-O2 -std=gnu++17 -DHAVE_LIBRAW -I../../core -I. -I../../vendor/LibRaw -pthread"
for f in grey_pyramid align robustness kernels merge raw_io snr_tuning; do
    g++ $CXXFLAGS -c ../../core/$f.cpp -o $f.o
done
g++ $CXXFLAGS -c host_stubs.cpp -o host_stubs.o
g++ $CXXFLAGS -c refine_dataset.cpp -o refine_dataset.o
g++ -O2 -std=gnu++17 -static -pthread *.o ../../vendor/LibRaw/lib/libraw.a \
    -lws2_32 -o refine_dataset      # -lws2_32 is Windows only
```

Build every translation unit with the **same** `-std`. Mixing `-std=c++17`
and `-std=gnu++17` flips `__USE_MINGW_ANSI_STDIO` on MinGW, which changes
inline definitions between objects and corrupts the heap inside LibRaw's
decode — it manifests as a silent crash with a backtrace pointing at an
unrelated `std::string` assignment.

`host_stubs.cpp` stands in for the Apple-only translation units (`prof.mm`,
`debug_utils.mm`, `robustness_nn.mm`) plus the two symbols `align.cpp` needs
from `pipeline.cpp`. It also provides the `refine_host.h` hook that lets
`refine_bench` drive the real refinement stage without Core ML.

## Feature layout

24 planes, interleaved, at the resolution `R` already is. This ordering is a
contract between `build_robustness_refine_features` (`core/robustness.cpp`),
`refine_dataset.cpp` and `train_refine.py`; changing it means retraining.
Lengths are in **raw pixels** throughout, the same units
`motion_geom_reject_threshold` uses.

| ch | name | |
|---|---|---|
| 0 | `R` | the analytic mask being refined, post Eq. 9 minimum |
| 1 | `z_d` | `d²/σ²`, Eq. 5's exponent argument |
| 2 | `tex` | `log1p(σ²/noise σ²)` — how much of σ is the scene's own texture. This *is* the leniency that lets a doubled edge through |
| 3 | `s` | the motion prior actually applied |
| 4–6 | `Ex`, `Ey`, `Emag` | within-tile flow error, signed so it reads as "the fetched content is displaced by E" |
| 7–8 | `div`, `curl` | flow divergence and curl, px across one tile |
| 9 | `Mspan` | Eq. 7's local flow span |
| 10–12 | `gx`, `gy`, `gmag` | reference gradient |
| 13 | `E_perp` | `E·ĝ` — the component **across** the edge, the one that doubles it |
| 14 | `E_par` | the component along the edge, which is harmless |
| 15 | `coh` | structure-tensor coherence: ~1 on a clean edge, ~0 on text, foliage, repetitive texture and noise |
| 16 | `lap` | Laplacian — thin lines, ringing |
| 17 | `res` | signed residual, reference minus comparison at the estimated flow |
| 18 | `res_disp` | `-res/|∇I|` — the displacement the **residual** implies |
| 19 | `res_z` | 3×3 mean `|res|` over noise σ |
| 20–21 | `bright`, `nsig` | local brightness and modelled noise σ |
| 22 | `agree` | `E_perp * res_disp` — positive and large is the signature this stage exists to find |
| 23 | `mismatch` | residual the geometry does not explain |

Channels 13 and 18 are signed along a canonicalised gradient direction
(flipped so `gy > 0`), which is what makes channel 22 a sign agreement rather
than a coin flip.

The floor on channel 18 is the **gradient's own** noise, not the pixel's:
`gx`/`gy` are central differences of 3×3 means of an `nch`-channel luma, so
the per-pixel σ is attenuated by 3, `sqrt(nch)` and `sc`. Using the
unattenuated σ instead — which an eyeballed `2*nsig` amounts to — puts the
floor about 6× too high and silently zeroes this channel, and channel 22 with
it, over ~95% of the frame. Measured: median `gmag` on real content is 0.0014
against an unattenuated `2*nsig` of 0.008.

## Two defects found while building this

Both are in existing code and are noted here because they affect anyone
reading the numbers.

**`R` can be NaN.** Where the estimated flow points outside the comparison
frame, Eq. 6 gives `d = inf`, and `apply_noise_model`'s Wiener shrink then
evaluates `inf/inf`, so `d_sq` and `R` are NaN on those border pixels.
`clampf` propagates NaN rather than clamping it. Harmless to the merge, which
refuses the fetch on its own bounds check, but the feature builder sanitises
the plane because one non-finite input poisons every activation downstream of
it. Not otherwise fixed here — it is pre-existing behaviour and out of scope.

**The aligner's reference cache was never cleared between references** in
either dataset generator. `align()` caches the reference's Sobel gradients and
ICA Hessian keyed on the *address* of the `Pyramid` it is given; that pyramid
is a loop local in both tools, so from the second reference onward every
alignment ran against the **first** reference's derivatives. Measured before
the fix: a static camera on reference 0 gave a mean flow error of 0.207 px,
while references 1…N all converged to 0.589 px with 42% of the frame labelled
unmergeable instead of 6%. Fixed in both `refine_dataset.cpp` and
`tools/rob_nn/rob_dataset.cpp` by calling `clear_align_ref_ica_cache()` per
reference.

The second one means **the published `tools/rob_nn` numbers (AUC 0.638 vs
0.926) were measured on a set where every reference after the first was
mis-aligned**, and that generator should be re-run before those numbers are
quoted again.

## Measured

Eight held-out frames of the 80-frame set (ten HDR+ test raws, eight synthetic
motion regimes each), `kappa = 0.75`, `deadzone = 0.20`. All four
configurations scored against the same ground truth.

- **falseR%** — correctly-aligned pixels (`R* >= 0.9`) pushed below half the
  weight the unrefined mask gave them. The failure that costs unrecoverable
  detail.
- **cutBad%** — should-drop pixels (`R* < 0.5`) the stage reduced at all.
- **wBad** — mean weight left on should-drop pixels. Lower is better. This is
  the suppression measure to read, *not* a "missed artifact rate": with
  `kappa = 0.75` a pixel keeps at least a quarter of its weight by
  construction, so any threshold-crossing rate is bounded by the cap rather
  than by the network.
- **chg%** — pixels moved at all. Sparsity.
- **mergeMSE** — excess merged-pixel MSE against using the optimal weight
  everywhere. The bottom line, and the one number no threshold can game.

| configuration | falseR% | cutBad% | wBad | chg% | mergeMSE |
|---|---|---|---|---|---|
| 1 Wronski only | 0.072 | 16.14 | 0.9311 | 2.07 | 0.30919 |
| 2 Wronski + geometry rejection | 2.225 | 19.01 | 0.7825 | 4.23 | 0.10128 |
| 3 Wronski + NN | **0.393** | **34.14** | 0.7674 | 5.59 | 0.09106 |
| 4 Wronski + geometry rejection + NN | 2.323 | 19.08 | **0.7098** | 3.80 | **0.07806** |

False rejection on the content that must be protected — every one of these
pixels is correctly aligned, so any rejection is pure loss:

| class | px% | 1 Wronski | 2 + geom | 3 + NN | 4 both |
|---|---|---|---|---|---|
| flat / sky | 15.44 | 0.01% | 0.01% | **0.03%** | 0.01% |
| coherent edge (coh > 0.8) | 1.80 | 0.00% | 2.45% | **0.27%** | 2.54% |
| thin line (high Laplacian) | 0.76 | 0.00% | 4.64% | **0.79%** | 4.92% |
| strongest gradients (top 1%) | 0.78 | 0.00% | 5.62% | **0.53%** | 5.73% |
| darkest quarter (noisiest) | 15.01 | 0.00% | 0.00% | **0.03%** | 0.02% |

Excess merge MSE by how wrong the flow actually was at the pixel:

| true flow error | px% | 1 Wronski | 2 + geom | 3 + NN | 4 both |
|---|---|---|---|---|---|
| < 0.1 px (aligned) | 46.8 | 0.00117 | 0.00234 | 0.00133 | 0.00251 |
| **0.1–0.5 px (sub-pixel)** | 47.5 | 0.03075 | 0.03075 | 0.03028 | 0.03021 |
| 0.5–1 px | 2.9 | 0.94197 | 0.94138 | 0.88260 | 0.88323 |
| 1–3 px | 2.8 | 3.34982 | 3.28908 | 2.28887 | 2.34508 |

### What these say

**The network beats the hand-designed geometry test on its own terms, and is
five to nine times more conservative.** Configuration 3 reaches a lower
`mergeMSE` than configuration 2 (0.09106 against 0.10128) while rejecting
0.393% of correctly-aligned pixels instead of 2.225% — and on the classes that
matter the gap is wider: coherent edges 0.27% against 2.45%, thin lines 0.79%
against 4.64%, the strongest gradients 0.53% against 5.62%. On pixels the flow
got *right*, geometry rejection doubles the excess merge error (0.00117 →
0.00234) where the network adds 14% (→ 0.00133).

That is the failure `motion_geom_reject` was always going to have: it rejects
on `|grad I| * |E|`, so a strong edge with a modest `E` trips it whether or not
the misalignment is real. The network has `E_perp` against `res_disp` and the
structure-tensor coherence, so it can tell a doubled edge from a sharp one.

**Configuration 4 is still the best on raw merge error** (0.07806), and adding
the network to geometry rejection costs almost nothing in false rejection
(2.225% → 2.323%). So if the geometry test is kept for the bright-scene
behaviour it was validated on, the network is a clean addition to it.

**But if false rejection is weighted the way the brief weights it** — a wrongly
rejected pixel is unrecoverable detail, a missed artifact is one subtly doubled
edge — then **configuration 3 is the better trade**: 90% of configuration 4's
merge-error improvement for a sixth of the false rejections. That is a result
about `motion_geom_reject_enabled`, not about the network, and it is worth
A/B-ing on real bursts before deciding.

Both models are trained and checked in (`refinenet_mlp_geom.pt`,
`refinenet_mlp_plain.pt`); which one ships is a matter of which baseline the
toggle is left in.

### Sparsity

The dead zone is the control, swept at `kappa = 0.75` on a held-out frame:

| dead | pixels moved | falseR% | mergeMSE | share of available gain |
|---|---|---|---|---|
| 0.00 | 99.9% | 0.10 | 0.09858 | 100% |
| 0.05 | 71.1% | 0.10 | 0.09926 | 98% |
| 0.10 | 16.3% | 0.10 | 0.10272 | 89% |
| **0.20** | **3.1%** | **0.10** | **0.10722** | **77%** |
| 0.35 | 1.2% | 0.10 | 0.11242 | 63% |

0.20 is the default: three quarters of the benefit for three percent of the
frame. Going to 0 moves the entire frame for 30% more benefit, and a reduction
applied uniformly to every comparison frame is close to inert anyway, because
the merge normalises `num/den` and only the differences between frames survive.

## On the GPU

The Metal path runs the whole stage in **one kernel, with no intermediate
buffers and nothing read back**, and does not involve Core ML at all.

That is only possible because the network is pointwise. The 24 features and both
16-wide hidden layers live in each thread's registers and never reach memory, so
`rob_refine_mask` is a single pass over the finished mask, dispatched after
`rob_local_min_5x5` and in place. Everything it needs — the comparison guide,
the reference statistics, the noise curves, `S`, the flow — is already bound in
that command buffer because `rob_make_mask` just used it.

The CPU path cannot do that, and the reason is worth stating plainly:
`init_robustness_metal` deliberately returns a `RefStats` carrying only
dimensions, with the pixels resident on the GPU, because every consumer on that
path is itself a kernel. So `apply_robustness_refinement` has to rebuild the
guide, the 3×3 local statistics and Eq. 6 on the host, materialise a 24-channel
feature plane in strips, hand each strip to Core ML and read the result back.
That is where its ~430 ms and 124 MB go — not on the arithmetic.

```
CPU   guide -> local stats -> Eq. 6 -> feature plane (49 MB/strip)
      -> Core ML (NCHW copy in, copy out) -> multiply         ~430 ms, 124 MB

GPU   rob_make_mask -> rob_local_min_5x5 -> rob_refine_mask   one extra pass
                                            (registers only)   no new buffers
```

Run host-side on this desktop, the fused single-pass version is 590 ms against
1167 ms for the strip-and-Core-ML shape — 2× faster before any parallelism. On
device it is ~700 multiply-accumulates per pixel over a 3 MP plane, about
2.3 GMAC, on hardware that does the surrounding mask in a few milliseconds.

`compute_robustness` is told which happened: `compute_robustness_metal` reports
through `refined_out`, and the CPU refinement is skipped when the kernel ran, so
the reduction is never applied twice.

### Weights

Compiled in, as `core/robustness_refine_weights.h` — 761 floats, 16 KB of
source, generated by `tools/rob_refine/export_metal_weights.py`. At 689
parameters the model is smaller than the code that would load a file, and a
compiled-in table has no load-failure path to handle. `RobustnessRefineNet.mlmodel`
is still built and bundled, but only the CPU fallback reads it.

The generator refuses a `cnn` checkpoint: the kernel gathers a 5×5 window for
the structure tensor and nothing wider, so a convolutional variant would need a
different gather and `kRobustnessRefineHalo` raised to match.

### How this is trusted without a Metal toolchain

The machine this was written on has no Metal compiler, so the kernel has never
been built. What was done instead is to make the kernel almost nothing:
`core/robustness_refine_shared.h` holds the parameter struct, the gather, Eq. 6,
the 24 features, the network and the bounded multiply, written to compile as both
Metal Shading Language and C++17, and `rob_refine_mask` is fifteen lines of
buffer plumbing around one call to `rr_refine_pixel`.

So the GPU path's own code can be run here. `refine_bench` executes
`rr_refine_pixel` over a real 12 MP frame and compares against the CPU path,
which reaches the same answer by a different route — planes instead of gathers,
`apply_noise_model` instead of the transcribed noise block:

```
max |R_gpu - R_cpu|                      7.87e-06
pixels differing by more than 1e-5       0 of 3,282,240
dead-zone boundary flips                 1
```

The single flip is a pixel whose predicted reduction lands exactly on the dead
zone, so the two roundings take opposite branches and differ by the whole
`kappa * deadzone` step. That is inherent to a hard threshold, not a
disagreement about the arithmetic.

Getting there found one real difference worth recording: the CPU builder
multiplied by `1/nch` where the shared gather divides by `nch`. One ULP, and it
was the entire disagreement on 28 pixels. Both divide now.

`tools/rob_refine/metal_lint.py` covers what is left that a compiler would
catch first — brace balance, the kernel's `[[buffer(n)]]` indices against what
`metal_gpu.mm` binds and in what order, `sizeof(RefineParams) == 64` on both
sides, every identifier the kernel body calls being defined ahead of it, and the
shared header's own portability rules (no `std::`, no templates, no bare
`log1p`).

**What none of this covers**: that the Metal source compiles, and Metal's own
floating-point behaviour — fast-math reassociation, a 1-ULP divide, whatever the
Metal `exp` costs in accuracy. Those need a device. A syntax error would be a
*build* failure rather than a silent one, since `HHSRKernels.metal` is compiled
at build time, and `rob_refine_mask` is deliberately kept out of the pipeline
list `metal_gpu_init()` requires, so a missing function leaves the rest of the
Metal backend working with this one stage off rather than dropping the whole
pipeline to the CPU.

`rob_refine_mask` is guide-resolution only. With
`robustness_raw_resolution_enabled` the Metal path reports no refinement and the
CPU implementation is asked instead, which declines unless the hires statistics
exist.

## Measured on the real C++ path

`refine_bench`, one 12 MP frame (4208×3120 raw, 2104×1560 guide, tile 16)
against a synthetic 1° rotation, at the shipped Config defaults:

| guarantee | result |
|---|---|
| toggle on, model absent → R unchanged | **0 of 3,282,240 pixels differ** |
| `R_final > R` | **0 pixels** |
| `R == 0` became positive | **0 pixels** |
| largest single reduction | 0.7499, i.e. the `kappa` cap is binding |
| strip vs whole-plane features | **max abs difference exactly 0** |
| pixels moved | **2.00%** |
| mean R | 0.9577 → 0.9501 |

Cost per comparison frame, 7 strips of 256 rows, desktop CPU:

| stage | ms |
|---|---|
| analytic mask on this same CPU path, for scale | 2973 |
| Eq. 6 correspondence | 123 |
| feature builder, all strips | 272 |
| whole stage including a scalar-C++ stand-in for the network | 1098 |

Peak extra allocation **124 MB**: one 24-channel feature strip 49 MB, `d_sq`
plus `sigma_sq` 25 MB, comparison means 38 MB, refined mask 12 MB.

The feature builder was 1801 ms before the luma planes were precomputed; every
spatial tap was re-reading `nch` interleaved channels, about 200 scattered
loads per output pixel.

### On the 200 ms budget

These are the CPU fallback's numbers. Memory is comfortably inside the budget;
time is not, and the reason is not the network — 689 parameters evaluated
pointwise is ~2.3 GMAC over a 3 MP plane, which the ANE does in single-digit
milliseconds. The ~1 s above is scalar C++ standing in for Core ML.

The cost is the round trip: rebuilding the guide, the local statistics and
`d²/σ²` on the host, materialising a feature plane, and copying in and out of
Core ML. That is what **On the GPU** above removes — one kernel, no buffers, no
readback — and it is the path that runs on device whenever Metal is available.
This section stays because the CPU path is still what runs when it is not.

## Status

Neither on-device path has been run. `robustness_nn.mm` (the CPU fallback's
Core ML route) was written on a machine with no Core ML runtime, the same caveat
as the replacement network; `rob_refine_mask` has never been compiled, because
there is no Metal toolchain here either. Both fail closed: a missing or broken
model, or a missing kernel, leaves the analytic mask exactly as it is.

What stands behind the Metal path instead is that its arithmetic is shared
verbatim with the CPU implementation and agrees with it to 8e-06 over a real
12 MP frame, plus the structural checks in `metal_lint.py`. See **How this is
trusted without a Metal toolchain**. First on-device run should confirm the
kernel builds and that the mask matches the CPU fallback's on the same burst.

Training data is synthetic motion over one real burst (the ten HDR+ test
raws). That is enough to demonstrate the mechanism and to measure the
conservatism guarantees; it is not enough to generalise across scenes and
lighting. More bursts — especially low light, distant text and repetitive
texture — should go in before this becomes a default.
