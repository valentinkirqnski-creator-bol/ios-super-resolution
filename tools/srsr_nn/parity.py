"""Parity: the PyTorch SRSR merge (srsr_merge.py) against the merge the app's
kernel produced (the `merged` array of a sample, dumped from SRSRKernels.metal).

    python parity.py data/val/s09000000.npz [...]
"""
import sys

import numpy as np
import torch

import srsr_merge as sm


def check(path):
    z = np.load(path)
    terms = sm.precompute(z)
    masks = torch.from_numpy(z["masks"].astype(np.float32) / 255.0)
    out = sm.merge(terms, masks).numpy()
    ref = z["merged"].astype(np.float32)
    # The kernel writes an RGBA16Float band texture: compare after the same half rounding.
    out_h = out.astype(np.float16).astype(np.float32)
    d = np.abs(out_h - ref)
    rel = d / np.maximum(np.abs(ref), 1e-3)
    tgt = z["target"]
    m = sm.meta(z)
    print("%s  %dx%d %d frames %-11s  max|d| %.2e  p99.9 rel %.2e  exact(half) %.4f%%  | merge vs target MAE %.4f" % (
        path.split("/")[-1].split("\\")[-1], ref.shape[1], ref.shape[0], m["n_frames"], m["scenario"],
        d.max(), np.quantile(rel, 0.999), 100.0 * (d == 0).mean(), np.abs(ref - tgt).mean()))
    return d.max()


if __name__ == "__main__":
    torch.set_num_threads(4)
    worst = max(check(p) for p in sys.argv[1:])
    print("worst max|d| %.3e" % worst)
