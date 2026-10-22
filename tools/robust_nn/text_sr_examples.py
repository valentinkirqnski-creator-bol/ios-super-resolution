"""Full-size super-resolution examples on held-out aliased-text bursts (v21), through the app's stages.

    python text_sr_examples.py OUT_DIR runs/v17/model.pt:0:v17 runs/v21/model_s10000.pt:0.25:v21 [--n 6]
Per burst (perfect alignment): the reference frame alone (what one aliased frame shows), the 2x merge with every
sample kept (the ground-truth mask for a perfectly aligned burst), with the analytic mask and with each net's mask
(cutoffs 0.20/0.90), the true scene, and each mask's average over frames. Plus misaligned twins (deliberate tile
errors on the same text) merged with all samples kept vs each mask, to show the ghosts each mask removes.
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("RNN_TP_REAL", os.path.join(HERE, "build", "pipe2", "tp_real20.exe"))
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
from PIL import Image, ImageDraw, ImageFont                              # noqa: E402
import real_protocol as rp                                               # noqa: E402
from real_protocol import load_net, stretch, CLIP                        # noqa: E402
from target_protocol import combine, r_index                             # noqa: E402
from rnn.features import compute_features                                # noqa: E402


def to8(img, s):
    return (np.clip(img / s, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def labelled(img8, text):
    im = Image.fromarray(img8)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "arial.ttf"), 22)
    d.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
    d.text((6, 3), text, fill=(255, 255, 0), font=f)
    return np.asarray(im)


def masks_for(nets, d, z):
    N = d["grey"].shape[0]
    Ik = d["ref_grey"][None].expand(N, -1, -1).contiguous()
    a, b = float(z["alpha"]) / 4, float(z["beta"]) / 4
    feats, _ = compute_features(Ik, d["grey"], d["flow"], d["r0"], a, b, CLIP, max(1, d["tile"] // 2))
    out = {"analytic mask": d["r0"].float()}
    with torch.no_grad():
        for name, (net, sh) in nets.items():
            R = net(feats, d["r0"])[0].clamp(1e-6, 1 - 1e-6)
            out[name] = stretch(torch.sigmoid(torch.log(R) - torch.log1p(-R) + sh), 0.2, 0.9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("models", nargs="+", help="path:logit_shift:label")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--src", default="rp21tx_val", help="v22: held-out source set (e.g. rp22scr_val, rp22tx_val)")
    ap.add_argument("--aligned_only", action="store_true", help="v22: skip the tile-corrupted twin")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nets = {}
    for m in a.models:
        p, s, lab = m.split(":")
        nets[lab + " mask"] = (load_net(os.path.join(HERE, p)), float(s))
    files = sorted(glob.glob(os.path.join(HERE, "data", a.src, "*.npz")))[:a.n]
    for f in files:
        z = np.load(f)
        stem = os.path.basename(f)[:-4]
        raw, gt = z["raw"], z["gt_flow"]
        N, H, W = raw.shape
        gy, gx = r_index(H, W, H // 2, W // 2)
        truth = z["gt_hr"].astype(np.float32)
        s = float(np.percentile(truth, 99.5)) + 1e-9
        for twin, corrupt in (("aligned", False),) + (() if a.aligned_only else (("misaligned", True),)):
            d = rp.run_tp_real(raw, gt, float(z["alpha"]), float(z["beta"]), 11, corrupt, z["cfa"], z["wb"],
                               oracle=True, edge_step=corrupt, small_only=a.src.startswith("rp22"))
            M = masks_for(nets, d, z)
            ones = torch.ones_like(M["analytic mask"])
            panels = [("one low-res frame (reference only)", (d["Aref"] / d["Bref"].clamp_min(1e-12)).numpy()),
                      ("merged, every sample kept" + (" = ground truth mask" if not corrupt else " (ghosts)"),
                       combine(d["Aref"], d["Bref"], d["Au"], d["Bu"], ones, gy, gx).numpy())]
            for name, R in M.items():
                panels.append((f"merged, {name} (keeps {float(R.mean()):.2f})",
                               combine(d["Aref"], d["Bref"], d["Au"], d["Bu"], R, gy, gx).numpy()))
            panels.append(("true scene", truth))
            sub = os.path.join(a.out, f"{stem}_{twin}")
            os.makedirs(sub, exist_ok=True)
            for i, (name, img) in enumerate(panels):
                Image.fromarray(to8(img, s)).save(os.path.join(sub, f"{i}_{name.split(' (')[0].replace(' ', '_').replace(',', '')}.png"))
            for name, R in M.items():
                Image.fromarray((R.mean(0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)).save(
                    os.path.join(sub, f"mask_{name.replace(' ', '_')}_average.png"))
            # one comparison sheet: a 360x360 centre crop of every panel, 2x enlarged, labelled
            c0 = (truth.shape[0] - 360) // 2
            tiles = [labelled(np.asarray(Image.fromarray(to8(img, s)[c0:c0 + 360, c0:c0 + 360]).resize((720, 720), Image.NEAREST)), name)
                     for name, img in panels]
            rows = [np.concatenate(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3])), 1)
                    for i in range(0, len(tiles), 3)]
            Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(a.out, f"{stem}_{twin}_comparison.png"))
            print(stem, twin, " ".join(f"{k}: keeps {float(v.mean()):.2f}" for k, v in M.items()), flush=True)


if __name__ == "__main__":
    main()
