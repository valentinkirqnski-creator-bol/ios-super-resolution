# Bilinear tile-flow fetch in the merge, and how to verify it

## What changed

`core/merge.cpp` now fetches the per-tile flow with `FlowField::sample_bilinear`
instead of nearest (`flow.dx(py,px)`), gated behind `Config::merge_flow_bilinear`
(default **true**). `core/types.h` carries the flag.

The merge was the only flow consumer still sampling nearest; the robustness/mask
path (`robustness.cpp`) already samples bilinearly, and
`FlowField::sample_bilinear`'s own comment requires every consumer to agree.
One vector per tile consumed nearest makes the warp piecewise constant, so a
sharp edge is reconstructed at a slightly different position in each tile and
reads as a staircase at the 16-px tile seams — visible on edges at 1x and
magnified at 2x. The step is sub-pixel and flips at every boundary, so it is
below the mask's d/sigma threshold yet above the viewer's. Bilinear removes the
discontinuity.

Validated on the PC reference (`Downloads/wronskietal_pc`): switching the Slang
merge's flow sampler point→linear changed ~4–8% of pixels, all on edges, and
removed the window/ledge stair-steps with no other change. Set
`merge_flow_bilinear=false` to A/B against the old nearest behaviour.

## Verifying on device (the part that must run on the Mac)

The app already profiles itself — `core/prof.mm` + `pipeline_paths.cpp` reset the
profiler before a run and emit `prof_report()` (per-stage `total ms` / per-call
ms / %) plus `prof_peak_footprint_bytes()` at the end, saved via
`prof_save_report()`. So the perf and RAM targets are read directly from a run;
no separate harness is needed.

1. **Build** (Mac + Xcode): `xcodegen` then build the app target for a real
   device (the profiler is on in release unless `HHSR_PROF=0`).
2. **Run** a 12 MP burst through `SRBridge processRawFrames:` (live capture) or
   `processDNGs:` (bundled DNGs) at `scale = 2`.
3. **Read the saved prof report:**
   - **Per-frame time:** the report's `(sum)` row is the whole burst; divide by
     frame count, or read the per-stage per-call ms. Target: **≤ 200 ms / 12 MP
     frame**. align / robustness / precision / merge are the stages to watch.
   - **Peak RAM:** `prof_peak_footprint_bytes()` (printed near
     `pipeline_paths.cpp:2451`). Target: **≤ 2.5 GB**. The 48 MP num/den must
     stay banded (row bands) for this to hold.
4. **Output parity vs the PC reference:** run the same burst through
   `wronskietal_pc` (`wronskietal.exe <burst> out.dng --scale 2`) and diff the
   two 2x DNGs and the `.rob.png` masks. Expect agreement within GPU rounding
   (hardware bilinear uses 8-bit sampler weights; the CPU twin emulates this —
   compare with a small tolerance, not bit-exact).

## What is NOT verified here

This change and the harness are correct by construction and were validated on the
PC, but the **200 ms / 2.5 GB / on-device output** numbers are device-measured
facts that can only be established by building and running on an actual iPhone
with Xcode — they are not claimed here. Read them from the prof report after a
device run.
