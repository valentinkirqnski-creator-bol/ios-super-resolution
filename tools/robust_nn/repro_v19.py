"""Reproduce the user's 48 MP edge artifacts synthetically and show what each mask does with them.

Held-out bursts with the v19 corruptions: straight-edge steps (tiles crossed by a straight edge shifted 0.08-0.6 raw px
across it, tp_real flag 8) and tile-sized patches on straight edges that tilt 0.3-4 deg and shift 0.05-0.6 px. Each
burst goes through the app's aligner + analytic mask + merge at 2x (the 48 MP output) and is merged five ways:
no rejection | v17 (shipped) | candidate | GT mask (full-res, 8-frame) | perfect alignment. Crops are picked where
v17's merge differs most from the GT-mask merge, i.e. the edge errors v17 lets through.

    python repro_v19.py OUT_DIR [candidate.pt[:shift]]
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("RNN_TP_REAL", os.path.join(HERE, "build", "pipe", "tp_real19.exe"))
os.environ.setdefault("RNN_TP_SERVER", "1")
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
import torch.nn.functional as F                                          # noqa: E402
from PIL import Image, ImageDraw                                         # noqa: E402
from real_protocol import run_tp_real, compute_features, load_net, stretch, CLIP   # noqa: E402
from target_protocol import combine, r_index                             # noqa: E402
from hr_target import hr_target, burst_of                                # noqa: E402

SCENES = ["APC_1186", "HDRP_N000", "APC_1294", "APC_1340", "APC_1342"]
LO, HI = 0.20, 0.90
CROP, ZOOM = 64, 4


def main():
    torch.set_num_threads(4)
    out = sys.argv[1]
    cand = sys.argv[2] if len(sys.argv) > 2 else "runs/v18b/model_s4000.pt:0.25"
    cpath, cshift = (cand.split(":") + ["0"])[:2]
    cname = os.path.basename(os.path.dirname(cpath)) + " " + os.path.basename(cpath)[:-3]
    os.makedirs(out, exist_ok=True)
    v17, vc = load_net(os.path.join(HERE, "runs/v17/model.pt")), load_net(os.path.join(HERE, cpath))
    files = sorted(glob.glob(os.path.join(HERE, "data", "rp19_val", "*.npz")))
    picks = [int(x) for x in os.environ.get("PICKS", "0,1,2,3,4,5,7,8,9").split(",")]
    tot = {}
    for idx in picks:
        f = files[idx]
        z = np.load(f)
        raw, gt = z["raw"], z["gt_flow"]
        N, H, W = raw.shape
        al, be = float(z["alpha"]), float(z["beta"])
        seed = int(np.random.default_rng(777 + idx).integers(1, 1 << 30))
        o = run_tp_real(raw, gt, al, be, seed, True, z["cfa"], z["wb"], edge_step=True)
        gy, gx = r_index(H, W, H // 2, W // 2)
        Ir = o["ref_grey"][None].expand(N - 1, -1, -1).contiguous()
        with torch.no_grad():
            feats, _ = compute_features(Ir, o["grey"], o["flow"], o["r0"], al / 4, be / 4, CLIP, max(1, o["tile"] // 2))
            R17 = stretch(v17(feats, o["r0"])[0], LO, HI)
            R = vc(feats, o["r0"])[0].clamp(1e-6, 1 - 1e-6)
            Rc = stretch(torch.sigmoid(torch.log(R) - torch.log1p(-R) + float(cshift)), LO, HI)
        gtf = torch.from_numpy(gt[1:])
        Rgt = hr_target(burst_of(f), o["flow"], gtf, al, be)
        vis = torch.from_numpy(z["vis"][1:].astype(np.float32))
        jump = torch.zeros_like(vis)
        for sh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            jump = torch.maximum(jump, (gtf - torch.roll(gtf, sh, (2, 3))).norm(dim=1))
        M = {"no rejection": combine(o["Aref"], o["Bref"], o["Au"], o["Bu"], torch.ones_like(R17), gy, gx),
             "v17 (shipped)": combine(o["Aref"], o["Bref"], o["Au"], o["Bu"], R17, gy, gx),
             cname: combine(o["Aref"], o["Bref"], o["Au"], o["Bu"], Rc, gy, gx),
             "GT mask": combine(o["Aref"], o["Bref"], o["Au"], o["Bu"], Rgt, gy, gx),
             "perfect alignment": combine(o["Aref"], o["Bref"], o["Ag"], o["Bg"], vis * (jump < 0.25).float(), gy, gx)}
        sq = lambda x: (x.clamp_min(0) + 1e-6).sqrt()
        e = (sq(M["v17 (shipped)"]) - sq(M["GT mask"])).abs().mean(-1)
        e = F.avg_pool2d(e[None, None], 9, 1, 4)[0, 0]
        Ho, Wo = e.shape
        m = CROP // 2
        e[:m], e[-m:], e[:, :m], e[:, -m:] = 0, 0, 0, 0
        gain = np.median(M["perfect alignment"].numpy().reshape(-1, 3), 0)
        gain = gain[1] / np.maximum(gain, 1e-9)
        scene = SCENES[int(os.path.basename(f)[2:10]) % len(SCENES)]
        for ci in range(2):
            k = int(e.argmax())
            cy, cx = k // Wo, k % Wo
            e[max(0, cy - CROP):cy + CROP, max(0, cx - CROP):cx + CROP] = 0
            y0, x0 = cy - m, cx - m
            sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
            s = float(np.percentile(M["perfect alignment"][sl].numpy() * gain, 99.5)) + 1e-9
            tone = lambda x: (np.clip(x[sl].numpy() * gain / s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
            tiles, errs = [], {}
            for name, img in M.items():
                tiles.append(np.kron(tone(img), np.ones((ZOOM, ZOOM, 1), np.uint8)))
                if name not in ("GT mask", "perfect alignment"):
                    errs[name] = float((sq(img[sl]) - sq(M["GT mask"][sl])).abs().mean())
                    tot.setdefault(name, []).append(errs[name])
            gap = np.full((CROP * ZOOM, 8, 3), 255, np.uint8)
            row = np.concatenate(sum([[t, gap] for t in tiles], [])[:-1], 1)
            img = Image.fromarray(np.concatenate([np.full((44, row.shape[1], 3), 255, np.uint8), row], 0))
            dr = ImageDraw.Draw(img)
            dr.text((4, 4), f"{scene}, held-out burst {idx} (edge steps + small tilt/shift patches), 2x output, crop "
                            f"{ci + 1}: where v17 differs most from perfect rejection", fill=(0, 0, 0))
            for i, name in enumerate(M):
                lab = name + (f"  diff vs GT mask {errs[name]:.4f}" if name in errs else "")
                dr.text((i * (CROP * ZOOM + 8) + 4, 24), lab, fill=(0, 0, 0))
            fn = os.path.join(out, f"{scene}_burst{idx}_crop{ci + 1}.png")
            img.save(fn)
            print(f"{os.path.basename(fn)}: " + "  ".join(f"{k} {v:.4f}" for k, v in errs.items()), flush=True)
    print("\nmean difference from the GT-mask merge over all crops (lower = closer to perfect rejection):")
    for k, v in tot.items():
        print(f"  {k:28s} {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
