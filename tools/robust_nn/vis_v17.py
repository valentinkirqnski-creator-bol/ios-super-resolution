"""Merged-image proof for v17: held-out bursts of the user's scenes (APC_1294 / APC_1340 low light / APC_1342) through
the app's aligner + analytic mask + merge, merged five ways: no rejection | v14 | v17 | GT mask (8-frame visibility)
| perfect alignment (exact motion, the image it should look like). Crops are picked where v14's output deviates
most from the perfect-alignment image, i.e. where v14 lets misalignments through - not where v17 looks good.

    python vis_v17.py runs/v17/model_s5000.pt OUT_DIR
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
import torch.nn.functional as F                                          # noqa: E402
from PIL import Image, ImageDraw                                         # noqa: E402
from real_protocol import run_tp_real, geo_target, compute_features, load_net, stretch, CLIP   # noqa: E402
from target_protocol import combine, r_index                             # noqa: E402

SCENES = ["APC_1294", "APC_1340 (low light)", "APC_1342"]
V17_SHIFT, LO, HI = 0.25, 0.20, 0.90
CROP, ZOOM = 72, 4


def main():
    torch.set_num_threads(4)
    ck, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    v14, v17 = load_net(os.path.join(HERE, "runs/v14/model.pt")), load_net(os.path.join(HERE, ck))
    files = sorted(glob.glob(os.path.join(HERE, "data", "rp17_val", "*.npz")))
    n = len(files)
    picks = [int(x) for x in os.environ.get("PICKS", "30,31,32,34,37,40,43,46,49").split(",")]
    tot = {}
    for idx in picks:
        z = np.load(files[idx % n])
        raw, gt = z["raw"], z["gt_flow"]
        N, H, W = raw.shape
        al, be = float(z["alpha"]), float(z["beta"])
        seed, corrupt = int(np.random.default_rng(99 + idx).integers(1, 1 << 30)), idx >= n
        o = run_tp_real(raw, gt, al, be, seed, corrupt, z["cfa"], z["wb"])
        gy, gx = r_index(H, W, H // 2, W // 2)
        Ir = o["ref_grey"][None].expand(N - 1, -1, -1).contiguous()
        with torch.no_grad():
            feats, _ = compute_features(Ir, o["grey"], o["flow"], o["r0"], al / 4, be / 4, CLIP, max(1, o["tile"] // 2))
            R14 = stretch(v14(feats, o["r0"])[0], LO, HI)
            R = v17(feats, o["r0"])[0].clamp(1e-6, 1 - 1e-6)
            R17 = stretch(torch.sigmoid(torch.log(R) - torch.log1p(-R) + V17_SHIFT), LO, HI)
        clean, gtf = torch.from_numpy(z["clean_grey"]), torch.from_numpy(gt[1:])
        vis = torch.from_numpy(z["vis"][1:].astype(np.float32))
        Rgt = geo_target(clean, o["flow"], gtf, vis, al, be, occ_look=True, n_merge=8)
        jump = torch.zeros_like(vis)
        for sh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            jump = torch.maximum(jump, (gtf - torch.roll(gtf, sh, (2, 3))).norm(dim=1))
        T = combine(o["Aref"], o["Bref"], o["Ag"], o["Bg"], vis * (jump < 0.25).float(), gy, gx)   # perfect alignment
        M = {"no rejection": combine(o["Aref"], o["Bref"], o["Au"], o["Bu"], torch.ones_like(R14), gy, gx),
             "v14": combine(o["Aref"], o["Bref"], o["Au"], o["Bu"], R14, gy, gx),
             "v17": combine(o["Aref"], o["Bref"], o["Au"], o["Bu"], R17, gy, gx),
             "GT mask": combine(o["Aref"], o["Bref"], o["Au"], o["Bu"], Rgt, gy, gx),
             "perfect alignment": T}
        # where v14's merge differs most from the GT-mask merge (what v14 lets through that perfect rejection
        # would not), sqrt domain, 9x9 smoothed; two separate crops
        e = ((M["v14"].clamp_min(0) + 1e-6).sqrt() - (M["GT mask"].clamp_min(0) + 1e-6).sqrt()).abs().mean(-1)
        e = F.avg_pool2d(e[None, None], 9, 1, 4)[0, 0]
        Ho, Wo = e.shape
        m = CROP // 2
        e[:m], e[-m:], e[:, :m], e[:, -m:] = 0, 0, 0, 0
        crops = []
        for _ in range(2):
            k = int(e.argmax())
            cy, cx = k // Wo, k % Wo
            crops.append((cy - m, cx - m))
            e[max(0, cy - CROP):cy + CROP, max(0, cx - CROP):cx + CROP] = 0
        gain = np.median(T.numpy().reshape(-1, 3), 0)
        gain = gain[1] / np.maximum(gain, 1e-9)
        scene = SCENES[idx % n % 3]
        for ci, (y0, x0) in enumerate(crops):
            sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
            s = float(np.percentile(T[sl].numpy() * gain, 99.5)) + 1e-9
            tone = lambda x: (np.clip(x[sl].numpy() * gain / s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
            tiles, errs = [], {}
            for name, img in M.items():
                tiles.append(np.kron(tone(img), np.ones((ZOOM, ZOOM, 1), np.uint8)))
                if name not in ("perfect alignment", "GT mask"):
                    G = M["GT mask"][sl]
                    d = ((img[sl].clamp_min(0) + 1e-6).sqrt() - (G.clamp_min(0) + 1e-6).sqrt()).abs().mean()
                    errs[name] = float(d)
                    tot.setdefault(name, []).append(float(d))
            gap = np.full((CROP * ZOOM, 8, 3), 255, np.uint8)
            row = np.concatenate(sum([[t, gap] for t in tiles], [])[:-1], 1)
            head = np.full((44, row.shape[1], 3), 255, np.uint8)
            img = Image.fromarray(np.concatenate([head, row], 0))
            dr = ImageDraw.Draw(img)
            dr.text((4, 4), f"{scene}, held-out burst {idx} ({'deliberate corruption' if corrupt else 'aligner only'}), "
                            f"crop {ci + 1} (picked where v14 differs most from the GT mask)", fill=(0, 0, 0))
            for i, name in enumerate(M):
                lab = name + (f"  diff vs GT mask {errs[name]:.4f}" if name in errs else "")
                dr.text((i * (CROP * ZOOM + 8) + 4, 24), lab, fill=(0, 0, 0))
            fn = os.path.join(out, f"{scene.split()[0]}_burst{idx}_crop{ci + 1}.png")
            img.save(fn)
            print(f"{os.path.basename(fn)}: " + "  ".join(f"{k} {v:.4f}" for k, v in errs.items()), flush=True)
    print("\nmean difference from the GT-mask merge over all crops (lower = closer to perfect rejection):")
    for k, v in tot.items():
        print(f"  {k:14s} {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
