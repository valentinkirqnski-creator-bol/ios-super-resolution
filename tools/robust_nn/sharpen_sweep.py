"""Sweep output sharpening R' = clamp((R - lo)/(hi - lo)) of the NN mask on held-out bursts.
    python sharpen_sweep.py --model runs/v2/model.pt"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn import merge, scenarios
from rnn.model import RobustNet
from rnn.pipeline import make_example
from evaluate import masks_for

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/v2/model.pt")
ap.add_argument("--per_scenario", type=int, default=3)
ap.add_argument("--seed0", type=int, default=3_000_000)
a = ap.parse_args()
torch.set_num_threads(4)
net = RobustNet(); net.load_state_dict(torch.load(a.model, map_location="cpu")); net.eval()
SETTINGS = [("r0", None), ("nn", (0.0, 1.0)), ("nn", (0.2, 0.8)), ("nn", (0.3, 0.7)), ("nn", (0.35, 0.65)),
            ("nn", (0.4, 0.6)), ("nn", (0.3, 0.5)), ("nn", (0.4, 0.5)), ("oracle", None)]
name = lambda s: s[0] if s[1] is None else f"nn[{s[1][0]:.2f},{s[1][1]:.2f}]"
res = {name(s): {"psnr": [], "sqrt": [], "leak": [], "keep": []} for s in SETTINGS}
per = {n: {name(s): [] for s in SETTINGS} for n in scenarios.NAMES}
with torch.no_grad():
    for si, sc in enumerate(scenarios.NAMES):
        for j in range(a.per_scenario):
            ex = make_example(a.seed0 + si * 1000 + j, sc)
            m = masks_for(ex, net)
            num, den = merge.contributions(ex["raw"], ex["flows"], ex["out_scale"], ex["sigma_k"])
            Ho, Wo = num.shape[-2:]
            v, e, g = ex["vis"][1:], ex["err"][1:], ex["geo"][1:]
            keep = (v > 0.9) & (e < 0.3) & (g > 0.9); bad = (v > 0.5) & (g < 0.1)
            for s in SETTINGS:
                R = m[s[0]].clone()
                if s[1] is not None:
                    lo, hi = s[1]
                    R[1:] = ((R[1:] - lo) / (hi - lo)).clamp(0, 1)
                M = merge.combine(num, den, merge.upsample_weights(R, Ho, Wo, ex["out_scale"]))
                k = name(s)
                p = merge.psnr(M, ex["gt"]); res[k]["psnr"].append(p); per[sc][k].append(p)
                res[k]["sqrt"].append(merge.psnr(M.clamp_min(0).sqrt(), ex["gt"].sqrt()))
                if bad.any(): res[k]["leak"].append(float(R[1:][bad].mean()))
                if keep.any(): res[k]["keep"].append(float(R[1:][keep].mean()))
        print("done", sc, flush=True)
base = np.mean(res["r0"]["psnr"]); bsq = np.mean(res["r0"]["sqrt"])
print(f"{'setting':16s} {'PSNR':>7s} {'dPSNR':>6s} {'dSqrt':>6s} {'leak':>5s} {'keep':>5s} | worst scenario d")
for s in SETTINGS:
    k = name(s); r = res[k]
    worst = min(scenarios.NAMES, key=lambda n: np.mean(per[n][k]) - np.mean(per[n]["r0"]))
    wd = np.mean(per[worst][k]) - np.mean(per[worst]["r0"])
    print(f"{k:16s} {np.mean(r['psnr']):7.2f} {np.mean(r['psnr']) - base:+6.2f} {np.mean(r['sqrt']) - bsq:+6.2f} "
          f"{np.mean(r['leak']):5.2f} {np.mean(r['keep']):5.2f} | {worst} {wd:+.2f}")
