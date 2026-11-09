"""Reader for mann_dataset.exe crops: per crop int32 {variant, frame, y, x} + f16 [13][S][S]
(11 features in core/mann.h order, z = ghost size in one-frame noise std, g = the app's gate)."""
import glob
import os
import numpy as np

NF = 11
IZ, IG = NF, NF + 1
NAMES = ["R", "comp-ref", "hp ref", "hp comp", "VL ref", "VL comp", "VL diff", "pair HF dis", "M/T", "seam", "D"]


def read_crops(path, S=80):
    rec = np.dtype([("hd", "<i4", (4,)), ("x", "<f2", (NF + 2, S, S))])
    return np.fromfile(path, dtype=rec)


def load_dir(d, S=80):
    parts = [read_crops(p, S) for p in sorted(glob.glob(os.path.join(d, "*.crops")))]
    return np.concatenate(parts) if parts else None


if __name__ == "__main__":
    import sys
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    a = load_dir(sys.argv[1], S) if os.path.isdir(sys.argv[1]) else read_crops(sys.argv[1], S)
    x = a["x"].astype(np.float32)
    var = a["hd"][:, 0]
    R, z, g = x[:, 0], x[:, IZ], x[:, IG]
    det = np.abs(x[:, 2]) > 0.1
    gated = g > 0.5
    print("crops", len(a), "variant counts", np.bincount(var), f"gated px {gated.mean():.3f}")
    for v, name in ((0, "Wronski"), (1, "GCam")):
        m = (var == v)[:, None, None] & gated
        bad, ok = m & (z > 4), m & (z < 1) & det
        print(f"{name:8s} gated: harmful {bad.sum() / max(1, m.sum()):.3f} of px, R<0.5 on them {(R[bad] < 0.5).mean():.3f}"
              f" | harmless detailed: R<0.5 on them {(R[ok] < 0.5).mean():.3f}  (n bad {bad.sum()}, ok {ok.sum()})")
    for i, n in enumerate(NAMES):
        f = x[:, i][gated]
        fb, fo = x[:, i][gated & (z > 4)], x[:, i][gated & (z < 1) & det]
        print(f"  {i:2d} {n:12s} gated p50 {np.median(f):+.3f} p95 {np.percentile(f, 95):+.3f} | harmful mean {fb.mean():+.3f}"
              f" harmless-detail mean {fo.mean():+.3f}")
