"""Side-by-side held-out examples: GT mask vs analytic mask vs a trained net, with per-pixel agreement maps.

    python mask_examples.py runs/v13/r03/model.pt C:/Users/valen/Downloads/robust_nn_outputs/v13_r03_examples
    python mask_examples.py runs/v14/ship.pt:0.2:0.9 OUT --vs runs/v13/r03/model.pt     # net vs net instead of analytic
(net[:lo:hi] = exact-0/1 cutoffs, default 0.05:0.95; RNN_OCC_LOOK=1 uses the corrected ground truth)
Layout per image (2 x 3):  image | analytic mask | net mask
                           GT mask | analytic vs GT | net vs GT
Agreement colours: white = aligned & kept, black = misaligned & rejected, RED = misaligned but kept (ghost),
BLUE = aligned but rejected (over-rejection), grey = in between (GT 0.2-0.8, not scored).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["RNN_GEO"] = "1"
os.environ.setdefault("RNN_TP_SERVER", "1")
os.environ.setdefault("RNN_TP_REAL", os.path.join(HERE, "build", "pipe", "tp_real_new.exe"))
sys.path.insert(0, HERE)
import glob                                   # noqa: E402
import numpy as np                            # noqa: E402
import torch                                  # noqa: E402
from PIL import Image, ImageDraw              # noqa: E402
from real_protocol import run_tp_real, geo_target, compute_features, load_net, stretch, CLIP   # noqa: E402

PICK = [30, 31, 32, 33, 36, 41, 44, 47, 2, 3]       # val sample ids (>= 30: with deliberate corruption)


def agree_map(t, R):
    b, g = t < 0.2, t > 0.8
    a = np.full(t.shape + (3,), 128, np.uint8)
    a[g & (R > 0.5)] = (255, 255, 255)
    a[b & (R <= 0.5)] = (0, 0, 0)
    a[b & (R > 0.5)] = (230, 30, 30)
    a[g & (R <= 0.5)] = (40, 90, 230)
    return a


def stats(t, R):
    b, g = t < 0.2, t > 0.8
    miss, over = int((b & (R > 0.5)).sum()), int((g & (R <= 0.5)).sum())
    nb, ng = int(b.sum()), int(g.sum())
    acc = 1 - (miss + over) / max(1, nb + ng)
    bal = 0.5 * ((1 - miss / max(1, nb)) + (1 - over / max(1, ng)))
    return acc, bal, miss, nb, over, ng


def main():
    torch.set_num_threads(2)
    def spec(sp):
        path, *c = sp.split(":") if not sp[1:3] == ":/" else [sp]
        lo, hi = (float(c[0]), float(c[1])) if c else (0.05, 0.95)
        nm = path.replace("\\", "/").split("runs/")[1].replace("/model.pt", "").replace(".pt", "").replace("/", " ")
        return load_net(path), lo, hi, nm
    net, lo, hi, name = spec(sys.argv[1])
    out = sys.argv[2]
    other = spec(sys.argv[sys.argv.index("--vs") + 1]) if "--vs" in sys.argv else None
    occ = os.environ.get("RNN_OCC_LOOK", "0") == "1"
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(HERE, "data", "rp13_val", "*.npz")))
    n = len(files)
    for e, idx in enumerate(PICK):
        z = np.load(files[idx % n])
        raw, gt = z["raw"], z["gt_flow"]
        N = raw.shape[0]
        rng = np.random.default_rng(99 + idx)                  # same seeds as the training's validation
        seed, corrupt = int(rng.integers(1, 1 << 30)), idx >= n
        d = run_tp_real(raw, gt, float(z["alpha"]), float(z["beta"]), seed, corrupt, z["cfa"], z["wb"], nomerge=True)
        Ir = d["ref_grey"][None].expand(N - 1, -1, -1).contiguous()
        with torch.no_grad():
            feats, _ = compute_features(Ir, d["grey"], d["flow"], d["r0"], float(z["alpha"]) / 4, float(z["beta"]) / 4,
                                        CLIP, max(1, d["tile"] // 2))
            R = stretch(net(feats[:, :net.c1.in_channels], d["r0"])[0], lo, hi).numpy()
            A = (stretch(other[0](feats[:, :other[0].c1.in_channels], d["r0"])[0], other[1], other[2]).numpy()
                 if other else d["r0"].numpy())
        t = geo_target(torch.from_numpy(z["clean_grey"]), d["flow"], torch.from_numpy(gt[1:]),
                       torch.from_numpy(z["vis"][1:].astype(np.float32)), float(z["alpha"]), float(z["beta"]),
                       occ_look=occ).numpy()
        k = int((t < 0.2).sum((1, 2)).argmax())                # the comparison frame with the most misalignment
        tk, Ak, Rk = t[k], A[k], R[k]
        aname = other[3] if other else "analytic"
        img = d["grey"][k].numpy()
        s = np.percentile(img, 99.5) + 1e-9
        im = np.repeat((np.clip(img / s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)[..., None], 3, 2)
        gray = lambda m: np.repeat((np.clip(m, 0, 1) * 255).astype(np.uint8)[..., None], 3, 2)
        up = lambda a: np.kron(a, np.ones((2, 2, 1), np.uint8))
        H, W = tk.shape
        cap = 18
        cells = [[(im, "comparison frame"), (gray(Ak), f"{aname} mask" if other else "analytic mask (Wronski-style)"), (gray(Rk), f"{name} mask")],
                 [(gray(tk), "GT mask (perfect)"), (agree_map(tk, Ak), f"{aname} vs GT"),
                  (agree_map(tk, Rk), f"{name} vs GT")]]
        gap = 10
        canvas = np.full((60 + 2 * (2 * H + cap) + gap, 3 * 2 * W + 2 * gap, 3), 255, np.uint8)
        pil = Image.fromarray(canvas)
        dr = ImageDraw.Draw(pil)
        for r, row in enumerate(cells):
            for c, (a, label) in enumerate(row):
                y0, x0 = 60 + r * (2 * H + cap + gap), c * (2 * W + gap)
                pil.paste(Image.fromarray(up(a)), (x0, y0 + cap))
                dr.text((x0 + 2, y0 + 3), label, fill=(0, 0, 0))
        sa, sn = stats(tk, Ak), stats(tk, Rk)
        line = lambda nm, s_: (f"{nm}: accuracy {s_[0]:.1%}  balanced {s_[1]:.1%}  ghosts kept (red) {s_[2]}/{s_[3]} "
                               f"misaligned px  |  good px dropped (blue) {s_[4]}/{s_[5]}")
        dr.text((4, 4), f"held-out burst {idx} ({z['scenario']}, {'deliberate corruption' if corrupt else 'aligner only'}),"
                        f" comparison frame {k + 1}.  RED = misalignment kept, BLUE = good area rejected", fill=(0, 0, 0))
        dr.text((4, 22), line(aname, sa), fill=(0, 0, 0))
        dr.text((4, 40), line(name, sn), fill=(0, 0, 0))
        fn = os.path.join(out, f"example_{e + 1:02d}_burst{idx}_frame{k + 1}.png")
        pil.save(fn)
        print(f"example {e + 1}: burst {idx} frame {k + 1}  {aname} acc {sa[0]:.3f} bal {sa[1]:.3f} miss {sa[2]}/{sa[3]} "
              f"over {sa[4]}/{sa[5]}  |  {name} acc {sn[0]:.3f} bal {sn[1]:.3f} miss {sn[2]}/{sn[3]} over {sn[4]}/{sn[5]}",
              flush=True)


if __name__ == "__main__":
    main()
