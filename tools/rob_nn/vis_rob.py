"""Render the analytic and learned masks side by side on a real burst.

Takes the planes rob_vis.cpp dumped, runs the trained checkpoint over them, and
writes PNGs plus the calibration histogram for the frame. The point is to see
whether the learned mask uses its full range -- white where the merge should
trust the frame, black where it should not, with a smooth rolloff between --
rather than sitting at a uniform mid-grey, which is what the previous model did
while still scoring well on AUC.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import RobNet, IN_CH

SC = os.path.dirname(os.path.abspath(__file__))


def read_dims(prefix):
    d = {}
    with open(prefix + ".dims") as f:
        for line in f:
            k, v = line.split()
            d[k] = int(v)
    return d


def save_png(path, arr):
    """8-bit PNG via zlib, no image library needed."""
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


def main():
    prefix = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(prefix)
    ckpt = os.environ.get("ROB_CKPT", os.path.join(SC, "robnet.pt"))
    d = read_dims(prefix)
    h, w, c, nf = d["h"], d["w"], d["feat_c"], d["frames"]
    assert c == IN_CH, f"dump has {c} channels, model expects {IN_CH}"

    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    mu, sd = ck["mu"], ck["sd"]
    model = RobNet(); model.load_state_dict(ck["state"]); model.eval()

    ref = np.fromfile(prefix + "_ref.rgb", dtype=np.float32).reshape(h, w, 3)
    save_png(os.path.join(outdir, "ref.png"), tonemap(ref))

    print(f"{'frame':>6} {'mask':>10} {'mean':>7} {'>0.9':>8} {'<0.1':>8} {'min':>7} {'max':>7}")
    for k in range(nf):
        tag = f"_f{k:02d}"
        feat = np.fromfile(prefix + tag + ".feat", dtype=np.float32).reshape(h, w, c)
        an = np.fromfile(prefix + tag + ".an", dtype=np.float32).reshape(h, w)
        with torch.no_grad():
            x = torch.from_numpy(((feat - mu) / sd)).permute(2, 0, 1)[None]
            nn = model(x)[0, 0].numpy()
        for nm, m in (("analytic", an), ("learned", nn)):
            print(f"{k:>6} {nm:>10} {m.mean():>7.3f} {(m > 0.9).mean() * 100:>7.1f}% "
                  f"{(m < 0.1).mean() * 100:>7.1f}% {m.min():>7.3f} {m.max():>7.3f}")
        save_png(os.path.join(outdir, f"mask_analytic{tag}.png"), an)
        save_png(os.path.join(outdir, f"mask_learned{tag}.png"), nn)
        # Side by side, plus the reference for orientation.
        strip = np.concatenate([tonemap(ref),
                                np.stack([an] * 3, -1),
                                np.stack([nn] * 3, -1)], axis=1)
        save_png(os.path.join(outdir, f"compare{tag}.png"), strip)
    print(f"\nwrote PNGs to {outdir}")
    print("compare_fNN.png is reference | analytic mask | learned mask")


if __name__ == "__main__":
    main()
