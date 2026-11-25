"""Checks on generated bursts: Python merge vs the app's merged image (parity), and the
app's flow vs the true correspondence.

    python check.py <burst.npz> [...]
"""
import sys

import numpy as np
import torch

import fusion as fu


def main():
    for p in sys.argv[1:]:
        bu = fu.Burst(p)
        print("==", p, "tile", bu.tile, "flow grid", tuple(bu.flow.shape), "R", tuple(bu.R.shape))
        if bu.merged is not None:
            A, B = fu.merge_terms(bu, 0, 0, bu.H, bu.W, R_analytic=True)
            out = fu.fuse(A, B, torch.ones(bu.N, 1, 1, 1)).permute(1, 2, 0)
            d = (out - bu.merged).abs()
            rel = d / bu.merged.abs().clamp(min=1e-3)
            print("parity: max abs %.3g  mean abs %.3g  rel>1e-3: %.4f%%  rel>1e-2: %.4f%%" % (
                d.max(), d.mean(), 100 * (rel > 1e-3).float().mean(), 100 * (rel > 1e-2).float().mean()))
        G = bu.H // 2
        for f in range(1, bu.N):
            v = fu.flow_at_guide(bu.flow[f - 1], G, G, bu.tile)
            e = bu.ghost[f - 1]
            print("  frame %d  flow mean (%.2f, %.2f)  err median %.2f px  >0.5px %.1f%%  occl/16px %.1f%%" % (
                f, v[..., 0].mean(), v[..., 1].mean(), e.median(), 100 * (e > 0.5).float().mean(),
                100 * (e >= 15.9).float().mean()))


if __name__ == "__main__":
    main()
