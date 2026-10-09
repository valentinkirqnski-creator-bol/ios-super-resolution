# prealign — global rotation + translation before the tile search

Estimates one `(theta, tx, ty)` per comparison frame and hands it to the
per-tile alignment as a **starting displacement**. Written from scratch; no
existing alignment code in this repo was consulted.

**Result: it works.** The estimator recovers the transform to ~0.01 px, and
seeding the tile search with it cuts mean per-tile endpoint error ~2x and
collapses the worst-case tile from 40 px to 5.7 px. Every number below is
reproducible from the two test scripts.

## Layout

| file | what |
|---|---|
| `rigid.py` | the estimator: coarse-to-fine Gauss-Newton on 3 parameters, Huber norm |
| `test_rigid.py` | does it recover a known transform, under noise / an object / low light |
| `test_seed.py` | does seeding a pyramid tile search with it actually help |

    python test_rigid.py --scenes 4 --trials 2
    python test_seed.py  --scenes 4 --trials 2 --cond noisy
    python test_seed.py  --scenes 4 --trials 2 --cond object

Test images are the clean merged frames from `tools/mask_refine/data`, i.e.
real photographs — a 3-parameter fit is easy on broadband synthetic texture and
hard on the flat regions and repeating structure real scenes contain.

## Three design decisions, and why

**No network.** A global rigid transform is three parameters with an analytic
Jacobian. Gauss-Newton converges in under ten iterations per pyramid level and
is exact. There is no dataset, no training run and nothing that can collapse to
the identity. The top pyramid level is where a learned initialiser would go if
the angle sweep ever proved inadequate; it has not.

**No resampling of the comparison frames.** The obvious reading of
"pre-align the frames" is to warp them, and that is wrong here: the comparison
frames are CFA mosaics, and a rotation maps a pixel to a non-lattice position
whose colour belongs to a different Bayer phase. Warping the mosaic interpolates
across colour phases and destroys the sub-pixel content the merge feeds on. The
three parameters go in as a per-tile seed instead, and no pixel is resampled.

**Rotation and translation only, no scale.** Hand-held bursts over a few hundred
milliseconds have negligible zoom, and a free scale parameter is the one that
most readily absorbs a brightness mismatch into a spurious zoom.

## Does the estimator work

Corner displacement error — the worst of the four corners, in pixels. Reported
at the corner rather than as an angle because an angular error is harmless mid
frame and largest exactly where rotation already hurts most, so the corner is
what the tile search actually sees.

| condition | mean | median | p90 | worst |
|---|---|---|---|---|
| clean | 0.017 | 0.009 | 0.047 | 0.059 |
| noisy | 0.024 | 0.018 | 0.048 | 0.068 |
| object (18% of frame moving independently) | 0.026 | 0.015 | 0.057 | 0.101 |
| low light (4% exposure) | 0.469 | 0.140 | 1.227 | 2.204 |

Against a 13–14 px uncorrected displacement that is a 500–800x reduction in
normal light. The Huber norm does its job: an independently moving rectangle
over 18% of the frame changes the error by 0.002 px. **Low light is the weak
case** — 0.47 px mean, 2.2 px worst — and is the condition to watch.

## Does seeding help

Per-tile endpoint error against the true displacement field. 1 deg rotation,
translation up to 8 px, tile 16, stride 16, search radius 4, 4 levels, 8 trials
over 4 scenes.

| | rigid: mean | >1px | worst | object: mean | >1px | worst |
|---|---|---|---|---|---|---|
| unseeded | 1.491 | 21.8% | 40.5 | 2.270 | 13.9% | 44.6 |
| **seed all** | **0.697** | **18.1%** | **5.7** | 0.836 | 6.6% | **26.2** |
| seed top | 1.530 | 22.3% | 40.4 | 2.668 | 14.7% | 62.3 |
| seed both, free | 0.746 | 17.3% | 33.1 | 0.698 | 5.0% | 62.3 |
| seed margin 2x | 0.726 | 18.3% | 15.8 | **0.697** | **5.4%** | 30.1 |
| seed only, no search | 0.012 | 0.0% | 0.04 | 0.464 | 3.1% | 21.5 |

**Take `seed all`**: the model's prediction as the starting point at every
level, no hypothesis switching. Its mean and >1px rate tie with margin gating
on rigid motion, and it holds the best worst case in both conditions. A single
badly-aligned tile is a visible ghost, so the worst case decides.

## Two wrong turns, both caught by measurement

These are the useful part of this directory. Both were plausible, both were
argued from mechanism, and both were wrong.

**1. "Seed the coarsest level only."** The argument was that a coarse tile
covers a large area with varying motion, so its single translation is a
compromise that then propagates down. It sounds right and it is false: at the
coarsest level displacements are divided by `2^levels`, so a 14 px corner
displacement is under 2 px there and the search handles it comfortably.
Measured, `seed top` is a **no-op** on rigid motion (1.530 vs 1.491 unseeded)
and **worse than no seed at all** with an object (2.668 vs 2.270).

The real failure is at the **fine** levels: one parent tile seeds four children,
and across a rotation field adjacent tiles want measurably different vectors, so
the propagated parent is worst exactly where the field varies fastest. Hence
the seed must be present at *every* level.

**2. "Let each tile pick whichever hypothesis matches better."** Seeding every
level costs the pyramid's ability to follow a region moving away from the global
model — the object's worst tile went to 26 px — so offering both the seed and
the propagated parent and keeping the lower SSD looks free. It improves the mean
(0.836 to 0.698) and **wrecks the worst case**: 5.7 to 33.1 px on rigid motion,
26.2 to 62.3 px with an object.

Lower SSD is not the same as correct motion. In flat or repeating texture a
wrong vector sometimes matches better than the right one, and free choice lets a
tile commit to it. Requiring the parent to *halve* the seed's cost recovers most
of it (62.3 to 30.1 px) and keeps the mean, but still loses to plain `seed all`
on the worst case, which is the number that corresponds to something visible.

## A convention trap worth knowing

`estimate_rigid` returns the transform `W` with `cmp(W(p)) ~ ref(p)`: it maps a
**reference** position to where that content sits in the **comparison** frame.
That is the direction the tile search wants, since it looks up the comparison
frame at *reference position + displacement*.

So to synthesise a test pair with a known `W`, resample by `W` **inverse**
(`invert_rigid`). Warping by `W` directly makes the truth its inverse, and
scoring against the forward parameters then reports roughly **twice the true
displacement** as error — which is exactly what the first run of
`test_rigid.py` did: 28 px of "error" against a 9.8 px truth, with the
estimator working perfectly the whole time.

## Not done

* **Nothing is wired into `core/`.** The search in `test_seed.py` is a plain SSD
  pyramid written for this measurement, with no regularisation and no
  multi-hypothesis handling. It establishes the mechanism and the direction on a
  search whose every parameter is visible. It does **not** predict the app's
  numbers: the app's aligner is better, so it starts from a smaller deficit and
  has less to gain.
* **Real parallax is untested.** All motion here is globally rigid, or rigid
  plus one independently moving block. A scene with continuous depth variation
  has no single correct `(theta, tx, ty)`, and the Huber norm will lock onto the
  dominant depth plane and treat the rest as outliers. This is the most likely
  way for the whole idea to disappoint, and the `gen_bursts` generator already
  has a parallax regime to test it with.
* **No visual check.** Every number here is endpoint error on synthetic motion.
  Per `never-bound-quality-with-mse`, that says little about what the image
  looks like.
