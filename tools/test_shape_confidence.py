#!/usr/bin/env python3
"""Synthetic check of the C_shape soft logic (mirrors compute_shape_confidence).

Does not need DNGs or a C++ build. Verifies the intended behaviour:
  - aligned edge  -> C ≈ 1
  - shifted edge  -> C significantly lower when R is high
  - flat noise    -> C ≈ 1 (no structure to verify)
  - never raises R
"""
from __future__ import annotations

import math
import sys


def soft01(x: float, lo: float, hi: float) -> float:
    if not (hi > lo):
        return 1.0 if x >= hi else 0.0
    t = (x - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def shape_c(
    resid_snr: float,
    edge_snr: float,
    cos_sim: float,
    Rn: float,
    *,
    edge_c: float | None = None,
    r_gate=0.30,
    resid_lo=2.5,
    resid_hi=7.0,
    cos_lo=0.0,
    cos_hi=0.50,
    min_edge=3.0,
    strength=0.90,
    bad_flow=0.0,
    use_flow=False,
) -> float:
    if not (Rn > r_gate):
        return 1.0
    if not (edge_snr > min_edge):
        return 1.0
    bad_photo = soft01(resid_snr, resid_lo, resid_hi)
    if edge_c is None:
        edge_c = edge_snr  # assume warped edge present
    if edge_c > 0.5 * min_edge:
        bad_grad = soft01(cos_hi - cos_sim, 0.0, cos_hi - cos_lo)
        mag_ratio = edge_c / max(edge_snr, 1e-6)
        bad_mag = soft01(1.0 - mag_ratio, 0.35, 0.85)
        bad_grad = max(bad_grad, bad_mag)
    else:
        # ref edge present, warped flat after flow
        bad_grad = soft01(edge_snr - edge_c, 1.0, min_edge)
    struct = max(bad_grad, bad_flow) if use_flow else bad_grad
    evidence = bad_photo * struct
    trust = soft01(Rn, r_gate, min(1.0, r_gate + 0.35))
    return max(0.0, min(1.0, 1.0 - strength * evidence * trust))


def main() -> int:
    # Correctly aligned textured edge: residual ~ noise, gradients agree.
    c_ok = shape_c(resid_snr=1.2, edge_snr=8.0, cos_sim=0.95, Rn=0.95, edge_c=7.5)
    # Shifted edge: high residual, warped gradient collapsed (missed the edge).
    c_miss = shape_c(resid_snr=6.5, edge_snr=10.0, cos_sim=0.0, Rn=0.95, edge_c=0.5)
    # Shifted edge with both grads present but opposing.
    c_bad = shape_c(resid_snr=6.5, edge_snr=10.0, cos_sim=-0.2, Rn=0.95, edge_c=9.0)
    # Flat noisy region: no edge -> must stay at 1 even with large residual.
    c_flat = shape_c(resid_snr=10.0, edge_snr=1.0, cos_sim=0.0, Rn=0.95, edge_c=0.2)
    # Already-distrusted pixel: leave alone.
    c_low_r = shape_c(resid_snr=8.0, edge_snr=10.0, cos_sim=-1.0, Rn=0.10, edge_c=9.0)
    # Mild suspicion only from photo, grads still agree -> little/no penalty.
    c_photo_only = shape_c(resid_snr=5.0, edge_snr=8.0, cos_sim=0.9, Rn=0.95, edge_c=7.5)

    print(f"aligned_edge   C={c_ok:.3f}")
    print(f"missing_edge   C={c_miss:.3f}")
    print(f"shifted_edge   C={c_bad:.3f}")
    print(f"flat_noise     C={c_flat:.3f}")
    print(f"low_R_gate     C={c_low_r:.3f}")
    print(f"photo_only     C={c_photo_only:.3f}")

    ok = True
    if c_ok < 0.95:
        print("FAIL: aligned edge should keep C≈1", file=sys.stderr)
        ok = False
    if c_miss > 0.35:
        print("FAIL: missing warped edge should drive C low", file=sys.stderr)
        ok = False
    if c_bad > 0.35:
        print("FAIL: shifted edge should drive C low", file=sys.stderr)
        ok = False
    if c_flat < 0.99:
        print("FAIL: flat region must not be penalised", file=sys.stderr)
        ok = False
    if c_low_r < 0.99:
        print("FAIL: low-R pixels must stay C=1", file=sys.stderr)
        ok = False
    if c_photo_only < 0.85:
        print("FAIL: photo alone must not strongly reject", file=sys.stderr)
        ok = False

    if ok:
        print("PASS")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
