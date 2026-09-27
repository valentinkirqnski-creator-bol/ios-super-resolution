# sr_gate — a learned per-frame robustness mask

A 1761-parameter convolutional network that replaces Wronski Eq. 5–9 and emits
the final per-pixel robustness mask. Trained in 22 minutes on this machine's CPU.

    conv 3x3 dilation 1   8 -> 8   ReLU        584 weights
    conv 3x3 dilation 2   8 -> 8   ReLU        584
    conv 3x3 dilation 3   8 -> 8   ReLU        584
    conv 1x1              8 -> 1   sigmoid       9

Receptive field 13x13 on the guide lattice, ~1.8k multiply-adds per pixel,
replicate padding at the border. `core/sr_gate.h` is the entry point,
`core/sr_gate_shared.h` holds the feature definitions shared verbatim with
`core/HHSRKernels.metal`, and `core/sr_gate_weights.h` is generated.

Enable with `Config::sr_gate_enabled`. Off by default. It declines and falls back
to the analytic mask whenever the guide is not the one its weights were fitted
on, so enabling it can never leave the pipeline without a mask.

## The idea

The mask is not trained to reproduce a better `R`. It is trained against the
**merged image**.

That is the whole design, and it follows from something this project already
measured: the Wronski merge is a *super-resolution* merge, so a sub-pixel offset
between frames is the signal it feeds on, not damage. `R = s·exp(-d²/σ²)` scores
every offset as damage, because `d²` is "what we fetched versus what we should
have fetched" and cannot represent the merge kernel's use of the offset as
information. Below about 1.6 px of per-tile flow error, rejection measured as
*net harmful*. Any label of that form — including a network trained faithfully to
a better `R*` — inherits the error rather than fixing it.

So the target is the merged image the pipeline would have produced with perfect
alignment and nothing rejected:

    loss = || merge(A, B, gate(features)) − merge(true per-pixel flow, noise-free frames, R = 1) ||

Both sides go through the same merge, the same steerable kernels, the same CFA
and the same bounds handling. Aliased, well-aligned regions therefore *pay* for
being rejected — the target contains the detail their offsets contribute — and
misaligned regions pay for being merged, because the target does not contain
their ghost. Nothing has to be hand-weighted to balance the two; the merge
prices both on one scale.

This is cheap to optimise because of one algebraic fact about
`accumulate_comp`: `local_r` is sampled once per output pixel and multiplied
into every tap of the 3x3 gather, so

    out = (A_ref + Σ_n A_n·R_n) / (B_ref + Σ_n B_n·R_n)

with `A_n`, `B_n` independent of `R`. Precompute those once per burst and the
merged image is an exact, differentiable function of the masks — no
approximation of the kernel, the CFA or the bounds handling anywhere.
`probe_oracle.py` checks the torch merge against the NumPy one: 4.5e-8.

## Features

Eight per-pixel channels, all in [0, 1] so the network sees one scale
independent of exposure, ISO and sensor. Defined once, in
`core/sr_gate_shared.h`.

| # | name | what it is |
|---|------|------------|
| 0 | `exp_a` | `exp(-d²/σ²)` — Wronski's own exponential, unscaled |
| 1 | `log_a` | `log1p(d²/σ²)/8` — the tail `exp()` has already saturated |
| 2 | `snr` | `log1p(signal var / noise var)/8` — is there detail worth merging |
| 3 | `subpix` | `|frac(raw flow)| / √½` — **the new sample phase** |
| 4 | `span` | 3x3 tile flow span / ts × 2 — local motion irregularity |
| 5 | `emag` | `|E|/2` — within-tile translation error, raw px |
| 6 | `grad` | `log1p(|∇g|/σ_n)/6` — is there an edge to smear |
| 7 | `dir_e` | `log1p(|∇g·E|/σ_n)/6` — predicted error *across* the edge |

Channel 3 is what separates this from any mask derived from the residual alone.
A half-pixel offset maximises `d²` *and* is exactly the sample placement
super-resolution needs; 0 and 1 cannot tell those two apart and 3 can. Channel 7
is the converse: a coherent per-tile misalignment produces a residual the noise
model explains away, so 0 and 1 miss it while the flow field predicts it
outright.

`d²` and `σ²` are consumed from Eq. 6, not recomputed, so the gate scores the
same correspondence through the same noise model the analytic mask does and
differs only in what it does with the result.

## Results

Held-out test split: 40 synthesised bursts, 4 motion regimes × 5 per-tile flow
errors × 2 reps, from real Bayer DNGs that **no training burst was built from**,
with different crops and seeds from the validation split the checkpoint was
selected on. Configuration is the shipping one: full-resolution FFT guide,
scale 2, steerable kernel, `ts = 16`. PSNR against the oracle-merge ground truth.

| mask | mean PSNR | gate gain | worst cell | wins |
|------|-----------|-----------|------------|------|
| shipping (Eq. 6–9 + geometry rejection) | 38.90 dB | **+5.22 dB** | +2.65 | 20/20 |
| Eq. 6–9, geometry rejection off | 40.97 dB | +3.15 dB | +1.88 | 20/20 |
| Eq. 6–8, no 5x5 minimum either | 42.97 dB | +1.16 dB | −0.04 | 19/20 |
| `R = 1` (merge everything) | 41.65 dB | +2.47 dB | −0.27 | 18/20 |
| **sr_gate** | **44.08 dB** | — | | |

Edge MSE −61%, flat MSE −80%, both against the shipping mask.

Read that table honestly: **4.07 of the 5.22 dB is from not doing two things the
shipping configuration does**, and the network adds +1.16 dB on top of the best
analytic variant. Both parts are real and they are separate findings:

* **Eq. 9's 5x5 minimum costs 2.00 dB on the full-resolution FFT guide.** It was
  designed for the half-resolution guide, where each sample already averaged a
  2x2 Bayer quad. On the full-resolution guide it takes the worst of 25 much
  noisier per-pixel tests. Measured at *zero* flow error it drops mean R from
  0.95 to 0.61 — it discards most of the burst before there is anything to
  reject. (`diag.py`)
* **Geometry rejection costs 2.07 dB there too**, consistent with the note in
  the commit that pinned it: its 0.0045 threshold is in the *decimated* guide's
  gradient units.
* The network's own contribution is the +1.16 dB, and more importantly the
  *shape* of it: at 0.12 px of flow error it tracks `R = 1` to within 0.3 dB
  (nothing should be rejected, and it rejects almost nothing), while at 0.9–5 px
  it beats `R = 1` by 2–6 dB. That is the behaviour the analytic mask cannot
  express with one exponential.

An oracle `R` — optimised per burst directly against the ground truth, so it has
seen the answer — reaches 4–8 dB above `R = 1`, which bounds what any mask of
this shape could do. The gate captures roughly a third of that.

## Two things that were tried and did NOT work

Recorded because both are reasonable things to expect, and because the cost of
re-trying them is another hour.

**The statistics behind the ratio (channels 8-11).** `d^2/sigma^2` is one number
built from four, and Eq. 6 reduces them with a `max()` and a shrinkage that throw
away which term won. Adding those back -- `shrink`, `sigdom`, a contrast-relative
`d_rel`, and `varmatch` from the comparison frame's local variance that both
backends compute and discard -- changes nothing:

| in_ch | scenes | steps | test PSNR |
|-------|--------|-------|-----------|
| 8     | 6      | 4200  | 44.07 dB  |
| 12    | 6      | 4200  | 44.07 dB  |

A ridge probe against the oracle mask agrees from a different direction: the
incremental R^2 of 8-11 over 0-7 is **-0.0027**, i.e. no linear information the
first eight do not already span (`probe_features.py`). Most likely because
`sigma_ms^2` is `ref_vars`, which channel 2 carries; `shrink` moves with
`d^2/sigma^2`; and `d_rel` is a rescaling of the same residual. The channels are
kept in `srsim.py` behind a default-off argument so the ablation can be re-run,
but the shipped net is 8-channel and `core/` is unchanged.

A trap worth naming: the first attempt at this comparison used a WALL-CLOCK budget
and looked like a 0.07 dB loss. A wider input is slower per step, so the
12-channel net had silently had 1900 fewer steps. `train.py --steps` exists to
match step counts; do not compare feature sets by minutes.

**Three more capture sessions.** `Downloads/ours`, `ours2`, `ours3` -- 8-frame
handheld bursts, one scene each, same sensor -- take the training material from 3
capture sessions to 6. At matched steps that is also neutral: 44.08 dB against
44.13. And on those very scenes, the model that never saw them scores 42.61
against the one trained on them at 42.58, winning 15 of 20 cells.

That is a better result than it looks. It says the eight features are
scene-agnostic enough that the mapping from them to a merge weight does not need
to be learned per scene -- the 3-session model already reached +5.0 dB over
Wronski on three sessions it had never seen. The extra captures therefore
CONFIRM the generalisation that the original dataset's narrowness left open,
rather than improving on it. The shipped model is trained on all six anyway,
since nothing argues against the broader data and it is the more defensible
default for content none of this has seen.

Run-to-run spread across five 8-channel runs at different data and step counts is
about 0.07 dB, so treat every difference in this section as noise.

## Parity

`parity.py` runs `core/sr_gate.cpp` on a real frame's Eq. 6 outputs and compares
its features and its mask against the NumPy and torch versions:

    worst feature error   6.0e-08
    mask, net only        4.2e-07 max
    alpha_rob C++ vs py   2.9e-08 rel

And `integration_dump.py` goes one further: it hands the C++ only the raw planes
and the flow, lets the pipeline's own `init_robustness` / `compute_robustness`
build the FFT guide, the 3x3 statistics, the real Monte-Carlo noise curves and
Eq. 6, and compares the mask that comes out against torch.

    guide: fft_active=1  guide 160x160 x1
    analytic mask mean 0.154   sr_gate mean 0.713
    max |gate - analytic| = 1.0000   -> hook FIRED
    pipeline gate mask vs torch: max 5.2e-04, mean 3.2e-05

The 5e-4 is the noise curves: the pipeline runs the real Monte Carlo, the port
uses the closed form that MC converges to. It is not a code disagreement.

This is the check that makes the training worth anything. `train.py` fits a
function of eight features computed in NumPy; the app computes those features in
C++ and Metal. A disagreement in the feature builder would still produce a
plausible-looking mask, just not the one the weights were fitted to, and nothing
would report an error.

## Reproducing

```
python make_data.py --out data --train 132 --val 20 --test 40 --jobs 6   # ~4 min
python pack.py --data data
python train.py --data data --minutes 28 --threads 8                     # CPU only
python evaluate.py --data data --split test
python baselines.py --data data --split test
python export_weights.py --ckpt sr_gate.pt      # -> core/sr_gate_weights.h
```

Parity, after building the harness (WinLibs g++ / MinGW-w64 UCRT; every TU needs
the **same** `-std`, see `tools/rob_refine` for why mixing `c++17` and `gnu++17`
corrupts the heap):

```
g++ -O2 -std=gnu++17 -Icore -pthread -c core/sr_gate.cpp -o sr_gate.o
g++ -O2 -std=gnu++17 -Icore -pthread -c tools/sr_gate/parity_main.cpp -o pm.o
g++ -O2 -std=gnu++17 -static -pthread sr_gate.o pm.o -o tools/sr_gate/sr_gate_parity.exe
python parity.py --ckpt sr_gate.pt
```

## Files

| | |
|---|---|
| `srsim.py` | NumPy port of the shipping robustness math: FFT guide, 3x3 stats, noise curves, Eq. 6–9, and the eight features |
| `srmerge.py` | port of `estimate_kernels` and `accumulate_comp`/`accumulate_ref`, factored into `A`/`B` |
| `srburst.py` | burst synthesis from real DNG, with the four alignment-failure regimes |
| `build.py` | one burst -> features + `A`/`B` + oracle ground truth |
| `make_data.py`, `pack.py` | dataset build and memory-mapping |
| `gate.py` | the network and the merge as a torch op |
| `train.py` | the merge-outcome training loop |
| `evaluate.py`, `baselines.py`, `diag.py`, `probe_oracle.py` | the measurements above |
| `export_weights.py` | -> `core/sr_gate_weights.h` |
| `parity_main.cpp`, `parity.py` | the C++ / Python parity check |
| `integration_main.cpp`, `integration_dump.py` | end-to-end check through `compute_robustness` itself |

## Limits, stated

* **The Metal path cannot reach it as things stand.** `rob_run_guide_stats` in
  `core/metal_gpu.mm` always builds the half-resolution 3-channel Bayer guide;
  it does not consult `robustness_fft_guide_active()`. So on device the shipping
  config's full-resolution guide does not happen, and the gate — whose weights
  were fitted on that guide — declines (`nch != 1`) and falls back. The Metal
  kernels and their dispatch are written and are inert while
  `sr_gate_enabled` is false, but they will only fire once either the Metal
  guide honours the FFT path or a second weight set is trained for the
  3-channel decimated guide. The feature builder already handles both lattices
  on both backends; only the weights are missing.
* **The Metal kernels have not been run.** There is no Mac or device here. They
  share the feature arithmetic with the CPU path via
  `core/sr_gate_shared.h`, and the CPU path is verified against torch, but the
  band bookkeeping and the dispatch are structurally reviewed only.
* **The ground truth is synthetic motion on real raw.** Real block-matching error
  is modelled — per-tile jitter over 0.08–6 px, a translation-only estimate of a
  rotating field, an object whose tiles lock onto the background, gross per-tile
  outliers — not measured from `align.cpp`. A burst whose alignment fails in a
  way none of those four regimes covers is out of domain.
* The scene is a demosaiced DNG treated as the latent radiance, so it carries the
  capture's own noise and its demosaic's own artifacts near Nyquist. Synthetic
  noise gains of 1–24x sit on top of that.
* `snr_auto_tune`'s `k_detail`/`k_denoise`/`D_th`/`D_tr` are computed as
  `tune_config_snr` computes them, but the alignment tile size it also chooses is
  pinned to the synthesiser's grid rather than allowed to move under the mask.
