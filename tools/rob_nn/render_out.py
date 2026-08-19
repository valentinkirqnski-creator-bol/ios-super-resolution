"""Render rob_merge output: the merged photo and the mask that produced it.

Mask rendering is deliberately LINEAR -- R = 0 is black, R = 1 is white, no
gamma, no contrast stretch, no auto-levels. The whole question being asked of
these images is whether the mask uses its dynamic range, and any normalisation
would manufacture the answer. A stretched copy is written alongside under a
different name for legibility, never in place of the linear one.

The merged photo does get a tone curve, because linear sensor data is
unviewable and the question there is about ghosting and grain, not about
absolute levels. The same curve is applied to both members of an A/B pair so
the only difference between them is the mask.
"""
import os, struct, sys, zlib
import numpy as np


def save_png(path, arr):
    a = np.clip(arr, 0, 1)
    a = (a * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    open(path, "wb").write(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b""))


def read_dims(p):
    d = {}
    for line in open(p):
        k, v = line.split(None, 1)
        v = v.strip()
        d[k] = int(v) if v.lstrip("-").isdigit() else v
    return d


def tone(rgb, ref_scale=None):
    """Grey-world balance then a gamma curve. Preview only -- the merge output
    carries no white balance (that is the ISP's job), so without this it looks
    green. Returns the scale used so an A/B pair can share it exactly."""
    if ref_scale is None:
        m = np.array([max(np.percentile(rgb[..., c], 99.0), 1e-6) for c in range(3)])
        ref_scale = m / m.max() * max(m.max(), 1e-6)
        ref_scale = m
    out = rgb / ref_scale.reshape(1, 1, 3)
    return np.clip(out, 0, 1) ** (1 / 2.2), ref_scale


def main():
    prefix = sys.argv[1]           # e.g. .../B4_learned
    outdir = sys.argv[2]
    label = sys.argv[3]            # "learned" or "analytic"
    tag = sys.argv[4]              # e.g. "burst4"
    shared = sys.argv[5] if len(sys.argv) > 5 else None   # .npy of tone scale

    d = read_dims(prefix + "_merge.dims")
    h, w, c = d["h"], d["w"], d["c"]
    mh, mw = d["mask_h"], d["mask_w"]

    merged = np.fromfile(prefix + "_merged.f32", dtype=np.float32).reshape(h, w, c)
    mask = np.fromfile(prefix + "_accmask.f32", dtype=np.float32).reshape(mh, mw)

    scale = np.load(shared) if (shared and os.path.exists(shared)) else None
    img, scale = tone(merged, scale)
    if shared and not os.path.exists(shared):
        np.save(shared, scale)

    save_png(os.path.join(outdir, f"{tag}_merged_{label}.png"), img)
    # Linear, unstretched -- this is the evidence.
    save_png(os.path.join(outdir, f"{tag}_mask_{label}.png"), mask)
    # Stretched copy, clearly named, for reading detail only.
    lo, hi = np.percentile(mask, 1), np.percentile(mask, 99)
    st = (mask - lo) / max(hi - lo, 1e-6)
    save_png(os.path.join(outdir, f"{tag}_mask_{label}_STRETCHED.png"), st)

    rej = float((mask < 0.5).mean())
    print(f"{tag:8s} {label:9s} accumulated mask: mean {mask.mean():.3f}  "
          f">0.9 {(mask > 0.9).mean() * 100:5.1f}%  <0.1 {(mask < 0.1).mean() * 100:5.1f}%  "
          f"| rejected (R<0.5) {rej * 100:5.1f}%  frames {d['frames']}")


if __name__ == "__main__":
    main()
