"""Reader for cref_dataset.exe crop files: per crop int32 {variant, frame, y, x} + f16 [7][S][S]
(6 features in core/cref.h order, then z = ghost size in one-frame noise std)."""
import glob
import os
import numpy as np

NF = 6


def read_crops(path, S=80):
    rec = np.dtype([("hd", "<i4", (4,)), ("x", "<f2", (NF + 1, S, S))])
    return np.fromfile(path, dtype=rec)


def load_dir(d, S=80):
    parts = [read_crops(p, S) for p in sorted(glob.glob(os.path.join(d, "*.crops")))]
    return np.concatenate(parts) if parts else None


if __name__ == "__main__":
    import sys
    a = load_dir(sys.argv[1]) if os.path.isdir(sys.argv[1]) else read_crops(sys.argv[1])
    x = a["x"].astype(np.float32)
    var = a["hd"][:, 0]
    R, hp, z = x[:, 0], x[:, 4], x[:, 6]
    det = np.abs(hp) > 0.1
    print("crops", len(a), "variant counts", np.bincount(var))
    for v, name in ((0, "Alg.6/Wronski"), (1, "GCam")):
        m = var == v
        Rv, zv, dv = R[m], z[m], det[m]
        bad = zv > 4
        ok = (zv < 1) & dv
        print(f"{name:14s} R mean {Rv.mean():.3f} | harmful z>4: {bad.mean():.3f} of px, R<0.5 on them {(Rv[bad] < 0.5).mean():.3f}"
              f" | harmless detailed (z<1): R<0.5 on them {(Rv[ok] < 0.5).mean():.3f}")
    print("z percentiles (detailed px)", np.percentile(z[det], [50, 75, 90, 95, 99]).round(2))
