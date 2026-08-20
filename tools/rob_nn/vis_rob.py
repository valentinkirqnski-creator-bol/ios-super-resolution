"""Render R_normal against R_final on a real burst, as PNGs you can look at.

Takes the planes rob_vis.cpp dumped, runs the trained correction over them, and
writes:

    <burst>_f<NN>_R_normal.png    the analytic mask, unchanged
    <burst>_f<NN>_C.png           the correction the network emits
    <burst>_f<NN>_R_final.png     R_normal * C, what the merge consumes
    <burst>_f<NN>_compare.png     reference | R_normal | C | R_final

All four use a PLAIN LINEAR grey mapping of [0,1] to [0,255] with no contrast
stretching, because the question these images answer is "how dark is the mask",
and a stretch answers a different one. A stretched copy of the correction is
written separately as _C_stretched.png, clearly named, because C should be so
close to 1 that a linear render of it is a white rectangle -- which is the
result, and is also why it needs the second version to show its structure.

Inference runs in row strips with the network's exact receptive-field halo,
the same windowing core/robustness.cpp uses on device. A whole 1512x2016 plane
through a 16-channel stack does not fit in the memory this host has, and the
strip result is identical because every window is kept fully inside the image
so the convolutions' zero-padding matches.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import RobNet, IN_CH

SC = os.path.dirname(os.path.abspath(__file__))
HALO = 8          # kRobustnessNnHalo -- the network's receptive-field radius
STRIP = 192       # kRobustnessNnStripRows


def read_dims(prefix):
    d = {}
    with open(prefix + ".dims") as f:
        for line in f:
            k, v = line.split()
            d[k] = int(v)
    return d


def save_png(path, arr):
    """8-bit PNG via zlib, no image library needed. Linear, never stretched."""
    import struct, zlib
    a = np.clip(arr, 0, 1)
    a = (a * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    open(path, "wb").write(png)


def tonemap(rgb):
    """Enough of a curve to see the scene; not the pipeline's ISP."""
    x = np.clip(rgb / max(np.percentile(rgb, 99.5), 1e-6), 0, 1)
    return x ** (1 / 2.2)


def run_strips(model, mu, sd, drop_ch, feat_mm, h, w, c):
    """Strip inference, matching core/robustness.cpp's windowing exactly."""
    out = np.zeros((h, w), np.float32)
    win = STRIP + 2 * HALO
    if h < win:
        raise SystemExit(f"image {h} rows is shorter than one {win}-row window")
    for y0 in range(0, h, STRIP):
        top = min(max(y0 - HALO, 0), h - win)
        band = np.asarray(feat_mm[top:top + win], dtype=np.float32)
        x = torch.from_numpy((band - mu) / sd).permute(2, 0, 1)[None]
        for ch in drop_ch:
            x[:, ch] = 0.0
        with torch.no_grad():
            r = model(x)[0, 0].numpy()
        rows = min(STRIP, h - y0)
        out[y0:y0 + rows] = r[y0 - top:y0 - top + rows]
    return out


def main():
    prefix = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(prefix)
    tag_base = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(prefix)
    ckpt = os.environ.get("ROB_CKPT", os.path.join(SC, "robnet.pt"))
    d = read_dims(prefix)
    h, w, c, nf = d["h"], d["w"], d["feat_c"], d["frames"]
    assert c == IN_CH, f"dump has {c} channels, model expects {IN_CH}"

    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    mu, sd = ck["mu"], ck["sd"]
    drop_ch = ck.get("drop_ch", [])
    model = RobNet(cin=ck.get("in_ch", IN_CH))
    model.load_state_dict(ck["state"]); model.eval()

    ref = np.fromfile(prefix + "_ref.rgb", dtype=np.float32).reshape(h, w, 3)
    ref_png = tonemap(ref)
    save_png(os.path.join(outdir, f"{tag_base}_ref.png"), ref_png)
    del ref

    print(f"{'frame':>5} {'R_normal':>9} {'C mean':>9} {'C p01':>7} {'C<0.9':>8} "
          f"{'R_final':>9} {'darkened':>9}")
    for k in range(nf):
        tag = f"_f{k:02d}"
        feat = np.memmap(prefix + tag + ".feat", dtype=np.float32, mode="r",
                         shape=(h, w, c))
        an = np.fromfile(prefix + tag + ".an", dtype=np.float32).reshape(h, w)
        cnn = run_strips(model, mu, sd, drop_ch, feat, h, w, c)
        del feat
        fin = an * cnn
        # "darkened" is the honest headline number for the failure criterion:
        # how much of the mask the correction removed, over the whole frame.
        dark = 1.0 - (fin.sum() / max(an.sum(), 1e-9))
        print(f"{k:>5} {an.mean():>9.4f} {cnn.mean():>9.5f} "
              f"{np.percentile(cnn, 1):>7.4f} {(cnn < 0.9).mean()*100:>7.3f}% "
              f"{fin.mean():>9.4f} {dark*100:>8.3f}%")
        cnn.astype(np.float32).tofile(prefix + f"_c{k:02d}.f32")
        fin.astype(np.float32).tofile(prefix + f"_mask{k:02d}.f32")
        save_png(os.path.join(outdir, f"{tag_base}{tag}_R_normal.png"), an)
        save_png(os.path.join(outdir, f"{tag_base}{tag}_C.png"), cnn)
        save_png(os.path.join(outdir, f"{tag_base}{tag}_R_final.png"), fin)
        # Separate, clearly named, and NOT the image to judge darkness from.
        lo = float(np.percentile(cnn, 0.2))
        save_png(os.path.join(outdir, f"{tag_base}{tag}_C_stretched.png"),
                 (cnn - lo) / max(1.0 - lo, 1e-6))
        strip = np.concatenate([ref_png,
                                np.stack([an] * 3, -1),
                                np.stack([cnn] * 3, -1),
                                np.stack([fin] * 3, -1)], axis=1)
        save_png(os.path.join(outdir, f"{tag_base}{tag}_compare.png"), strip)
    print(f"\nwrote PNGs to {outdir}")
    print("_compare.png is reference | R_normal | C | R_final, all linear grey")


if __name__ == "__main__":
    main()
