#!/usr/bin/env python3
"""
Run the Handheld SR entrypoint while capturing the exact unseeded noise curves
that run_fast_MC produced for that process — without modifying package files.

Monkeypatches handheld_super_resolution.fast_monte_carlo.run_fast_MC before
the pipeline imports it, dumps float32 bins, then forwards to the real CLI.

Example:
  set HHSR_NOISE_CURVES_DIR=C:\\tmp\\noise_curves
  python tools/run_sr_dump_noise_curves.py --package-root ..\\Handheld-Multi-Frame-Super-Resolution-main ^
      -- -d path\\to\\dngs -o out.png

Then run the iOS/C++ pipeline with the same HHSR_NOISE_CURVES_DIR (or copy the
folder into the app Documents/noise_curves).
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-root", required=True,
                   help="Directory that contains handheld_super_resolution/")
    p.add_argument("--curves-out", default=None,
                   help="Dump dir (default: $HHSR_NOISE_CURVES_DIR or ./noise_curves)")
    p.add_argument("--entrypoint", default=None,
                   help="Python file to run (default: package_root/run_handheld.py)")
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="Args after -- forwarded to the entrypoint")
    args = p.parse_args()

    root = os.path.abspath(args.package_root)
    sys.path.insert(0, root)

    out = args.curves_out or os.environ.get("HHSR_NOISE_CURVES_DIR") or "noise_curves"
    os.makedirs(out, exist_ok=True)
    os.environ["HHSR_NOISE_CURVES_DIR"] = os.path.abspath(out)

    import handheld_super_resolution.fast_monte_carlo as fmc

    _orig = fmc.run_fast_MC

    def _wrap(alpha, beta):
        sigmas, diffs = _orig(alpha, beta)
        sigmas.astype("float32").tofile(os.path.join(out, "std_curve.bin"))
        diffs.astype("float32").tofile(os.path.join(out, "diff_curve.bin"))
        with open(os.path.join(out, "meta.txt"), "w", encoding="utf-8") as f:
            f.write(f"alpha={float(alpha):.17g}\n")
            f.write(f"beta={float(beta):.17g}\n")
            f.write(f"n_bins={len(sigmas)}\n")
        print(f"[noise dump] Wrote curves to {out} (alpha={alpha}, beta={beta})")
        return sigmas, diffs

    fmc.run_fast_MC = _wrap

    fwd = list(args.rest)
    if fwd and fwd[0] == "--":
        fwd = fwd[1:]

    if args.entrypoint:
        sys.argv = [args.entrypoint] + fwd
        runpy.run_path(args.entrypoint, run_name="__main__")
        return 0

    candidate = os.path.join(root, "run_handheld.py")
    if os.path.isfile(candidate):
        sys.argv = [candidate] + fwd
        runpy.run_path(candidate, run_name="__main__")
        return 0

    # Fallback: module -m style if present
    sys.argv = ["handheld_super_resolution"] + fwd
    runpy.run_module("handheld_super_resolution", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
