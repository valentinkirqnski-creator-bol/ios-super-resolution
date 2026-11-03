"""Full-frame burst_compare output -> viewable files: merged image (PNG) and every comparison frame's masks.

    python export_full.py RUN_DIR OUT_DIR
RUN_DIR holds nn.f32 (merged with the robust-NN mask) and masks.f32 (per comparison frame: analytic R, NN R at
guide resolution = one value per RGGB quad). Masks: white = sample fully used, black = rejected.
"""
import os
import sys

import numpy as np
from PIL import Image


def main():
    run, out = sys.argv[1], sys.argv[2]
    tag = os.environ.get("NN_TAG", "v17")   # the mask net's version, for the file names
    os.makedirs(out, exist_ok=True)
    m = open(os.path.join(run, "meta.txt")).read().split()
    Hs, Ws, gh, gw, n = (int(x) for x in m[:5])
    wb = np.array([float(x) for x in m[5:8]], np.float32)
    for name in ("nn", "r0", "ones"):
        f = os.path.join(run, name + ".f32")
        if not os.path.exists(f):
            continue
        img = np.fromfile(f, np.float32).reshape(Hs, Ws, 3) * wb          # camera-native -> white-balanced
        s = float(np.percentile(img, 99.5)) + 1e-9
        rgb = (np.clip(img / s, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
        label = {"nn": f"merged_robust_nn_{tag}", "r0": "merged_analytic_mask", "ones": "merged_no_rejection"}[name]
        Image.fromarray(rgb).save(os.path.join(out, label + ".png"))
        print("wrote", label, rgb.shape, flush=True)
    M = np.fromfile(os.path.join(run, "masks.f32"), np.float32).reshape(n - 1, 2, gh, gw)
    for which, lab in ((1, f"robust_nn_{tag}"), (0, "analytic")):
        for k in range(n - 1):
            Image.fromarray((np.clip(M[k, which], 0, 1) * 255 + 0.5).astype(np.uint8)).save(
                os.path.join(out, f"mask_{lab}_frame{k + 1}.png"))
        Image.fromarray((np.clip(M[:, which].mean(0), 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            os.path.join(out, f"mask_{lab}_average_all_frames.png"))
        print(f"{lab}: mean kept weight per frame " + " ".join(f"{M[k, which].mean():.2f}" for k in range(n - 1))
              + f"  | all frames {M[:, which].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
