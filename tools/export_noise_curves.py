#!/usr/bin/env python3
"""
Dump noise curves from the Handheld package's run_fast_MC (unseeded np.random)
without modifying that package.

C++ loads these via HHSR_NOISE_CURVES_DIR (or Documents/noise_curves on iOS):
  std_curve.bin, diff_curve.bin  — 1001 float32 each
  meta.txt                       — alpha=… / beta=…

Usage:
  python tools/export_noise_curves.py --alpha 1.8e-4 --beta 3.2e-6 -o ./noise_curves

Note: unseeded → each invocation produces different curves. To match a specific
Python SR run, use tools/run_sr_dump_noise_curves.py instead (monkeypatches
run_fast_MC at import time; still does not edit package files).
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("-o", "--out", default="noise_curves",
                   help="Output directory for bin + meta")
    p.add_argument("--package-root", default=None,
                   help="Path containing handheld_super_resolution/ (optional)")
    args = p.parse_args()

    if args.package_root:
        sys.path.insert(0, os.path.abspath(args.package_root))

    # Import package as-is (no edits).
    from handheld_super_resolution.fast_monte_carlo import run_fast_MC

    sigmas, diffs = run_fast_MC(args.alpha, args.beta)
    os.makedirs(args.out, exist_ok=True)

    sigmas.astype("float32").tofile(os.path.join(args.out, "std_curve.bin"))
    diffs.astype("float32").tofile(os.path.join(args.out, "diff_curve.bin"))
    with open(os.path.join(args.out, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"alpha={args.alpha:.17g}\n")
        f.write(f"beta={args.beta:.17g}\n")
        f.write(f"n_bins={len(sigmas)}\n")

    print(f"Wrote {args.out}/std_curve.bin, diff_curve.bin, meta.txt")
    print("Point C++ at this dir: set HHSR_NOISE_CURVES_DIR (or copy to Documents/noise_curves).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
