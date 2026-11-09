"""Sub-pixel offsets between coord_test outputs (output px; /scale = raw px), doubling and mask stats."""
import re
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, shift as ndshift


def load(p):
    a = tifffile.imread(p).astype(np.float32)
    return a[..., 1] / 65535.0


def lk_shift(a, b, iters=6):
    """Displacement d such that b(x) ~ a(x - d)... measured by iterative LK on smoothed images (sub-pixel)."""
    a = gaussian_filter(a, 1.0); b = gaussian_filter(b, 1.0)
    gy, gx = np.gradient(a)
    A = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
    if np.linalg.cond(A) > 1e4 or np.trace(A) < 1e-6:
        return None
    d = np.zeros(2)
    for _ in range(iters):
        bw = ndshift(b, (d[1], d[0]), order=3, mode="nearest")   # undo the current estimate
        r = (bw - a)[8:-8, 8:-8]
        g = np.array([np.sum(gx[8:-8, 8:-8] * r), np.sum(gy[8:-8, 8:-8] * r)])
        A2 = np.array([[np.sum(gx[8:-8, 8:-8] ** 2), np.sum(gx[8:-8, 8:-8] * gy[8:-8, 8:-8])],
                       [np.sum(gx[8:-8, 8:-8] * gy[8:-8, 8:-8]), np.sum(gy[8:-8, 8:-8] ** 2)]])
        d += np.linalg.solve(A2, g)
    return d   # (dx, dy): b is a shifted by +d


def tiles(a, b, T=128, margin=256):
    H, W = a.shape
    out = []
    for y in range(margin, H - margin - T + 1, T):
        for x in range(margin, W - margin - T + 1, T):
            ta, tb = a[y:y + T, x:x + T], b[y:y + T, x:x + T]
            if ta.std() < 0.01:
                continue
            d = lk_shift(ta, tb)
            if d is not None:
                out.append(d)
    return np.array(out)


def gradE(a, margin=256):
    gy, gx = np.gradient(a[margin:-margin, margin:-margin])
    return float(np.mean(gx * gx + gy * gy))


for g in ("w", "gcam"):
    A, B, C = (load(f"ct/{m}_{g}.dng") for m in "ABC")
    print(f"=== {'GCam' if g == 'gcam' else 'Wronski'} (output px = 0.5 raw px)")
    for n1, a, n2, b in (("C(ref only)", C, "A(identical)", A), ("C(ref only)", C, "B(shifted)", B), ("A", A, "B", B)):
        d = tiles(a, b)
        print(f"  {n2} vs {n1}: tiles {len(d)}  median offset dx {np.median(d[:, 0]):+.4f} dy {np.median(d[:, 1]):+.4f}"
              f"  p10/p90 dx {np.percentile(d[:, 0], 10):+.3f}/{np.percentile(d[:, 0], 90):+.3f}"
              f" dy {np.percentile(d[:, 1], 10):+.3f}/{np.percentile(d[:, 1], 90):+.3f}  max |d| {np.abs(d).max():.3f}")
    print(f"  sharpness (grad energy) C {gradE(C):.3e}  A {gradE(A):.3e}  B {gradE(B):.3e}")
    print(f"  |B-A| mean {np.abs(B - A)[256:-256, 256:-256].mean():.2e}  |A-C| mean {np.abs(A - C)[256:-256, 256:-256].mean():.2e}")
    for m in "ABC":
        txt = open(f"ct/{m}_{g}_robustness.pgm", "rb").read()
        hdr = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", txt)
        w, h = int(hdr.group(1)), int(hdr.group(2))
        im = np.frombuffer(txt[hdr.end():hdr.end() + w * h], np.uint8).reshape(h, w) / 255.0
        mg = h // 8
        inner = im[mg:-mg, mg:-mg]
        print(f"  mask {m}: {w}x{h} mean {inner.mean():.3f}  frac < 0.5 {np.mean(inner < 0.5):.3f}  frac < 0.9 {np.mean(inner < 0.9):.3f}")
