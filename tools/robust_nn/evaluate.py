"""Held-out evaluation on the MERGED image (what the mask is for), per scenario.

    python evaluate.py --model runs/v1/model.pt --out runs/v1/eval

Masks compared inside the same merge:
    ones    accept everything
    r0      analytic mask (nominal params)            <- what ships today (approx.)
    nn      network refinement of r0
    r0b/nnb same with a different analytic variant (no 5x5 min) — robustness to the app's formula
    oracle  geometric ground-truth reliability        (upper reference, not reachable)

dB tables are a coarse summary; look at the PNGs (crops of GT / r0 / nn merges
and the masks) before trusting a number.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn import merge, scenarios                         # noqa: E402
from rnn.analytic import analytic_r0, NOMINAL            # noqa: E402
from rnn.common import warp_by_flow                      # noqa: E402
from rnn.features import compute_features                # noqa: E402
from rnn.model import RobustNet                          # noqa: E402
from rnn.pipeline import make_example                    # noqa: E402

CLIP = 0.9
ALT = dict(NOMINAL, min5=False, t=0.0)


def masks_for(ex, net):
    grey, flows = ex["grey"], ex["flows"]
    a, b = ex["noise_a"] / 4.0, ex["noise_b"] / 4.0
    N = grey.shape[0]
    out = {k: torch.ones(N, *grey.shape[-2:]) for k in ("ones", "r0", "nn", "r0b", "nnb", "oracle")}
    out["oracle"] = ex["geo"].clone()
    out["oracle"][0] = 1
    for k in range(1, N):
        Ir, Ic, fl = grey[0][None], grey[k][None], flows[k][None]
        J = warp_by_flow(Ic, fl)
        for tag, p in (("", NOMINAL), ("b", ALT)):
            r0 = analytic_r0(Ir, J, a, b, p)
            feats, _ = compute_features(Ir, Ic, fl, r0, a, b, CLIP, ex["tile"])
            R, _ = net(feats, r0)
            out["r0" + tag][k] = r0[0]
            out["nn" + tag][k] = R[0]
    return out


def merged(ex, masks):
    num, den = merge.contributions(ex["raw"], ex["flows"], ex["out_scale"], ex["sigma_k"])
    Ho, Wo = num.shape[-2:]
    return {k: merge.combine(num, den, merge.upsample_weights(m, Ho, Wo, ex["out_scale"])) for k, m in masks.items()}


def disp(rgb, wb, scale):
    x = rgb / torch.from_numpy(wb)[:, None, None]
    x = (x / scale).clamp(0, 1) ** (1 / 2.2)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def gray(m):
    return np.repeat((m.clamp(0, 1).numpy() * 255).astype(np.uint8)[..., None], 3, 2)


def save_panel(path, ex, M, masks, k=1):
    gt = ex["gt"]
    s = float((gt / torch.from_numpy(ex["wb"])[:, None, None]).quantile(0.99)) + 1e-6
    S = ex["out_scale"]
    up = lambda m: np.kron(m, np.ones((2 * S, 2 * S, 1), np.uint8))
    top = np.concatenate([disp(gt, ex["wb"], s), disp(M["r0"], ex["wb"], s), disp(M["nn"], ex["wb"], s),
                          disp(M["oracle"], ex["wb"], s)], 1)
    bot = np.concatenate([up(gray(masks["r0"][k])), up(gray(masks["nn"][k])), up(gray(masks["oracle"][k])),
                          up(gray((ex["err"][k] / 2)))], 1)
    Image.fromarray(np.concatenate([top, bot], 0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v1/model.pt")
    ap.add_argument("--out", default="runs/v1/eval")
    ap.add_argument("--per_scenario", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=3_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(8)
    net = RobustNet()
    if a.model != "identity":
        net.load_state_dict(torch.load(a.model, map_location="cpu"))
    net.eval()
    keys = ["ones", "r0", "nn", "r0b", "nnb", "oracle"]
    res = {}
    with torch.no_grad():
        for si, name in enumerate(scenarios.NAMES):
            rows = []
            for j in range(a.per_scenario):
                seed = a.seed0 + si * 1000 + j
                ex = make_example(seed, name)
                masks = masks_for(ex, net)
                M = merged(ex, masks)
                gt = ex["gt"]
                row = {k: merge.psnr(M[k], gt) for k in keys}
                row.update({k + "_sqrt": merge.psnr(M[k].clamp_min(0).sqrt(), gt.sqrt()) for k in keys})
                v, e, g = ex["vis"][1:], ex["err"][1:], ex["geo"][1:]
                keep = (v > 0.9) & (e < 0.3) & (g > 0.9)
                bad = (v > 0.5) & (g < 0.1)
                for k in ("r0", "nn"):
                    row[f"keep_{k}"] = float(masks[k][1:][keep].mean()) if keep.any() else float("nan")
                    row[f"leak_{k}"] = float(masks[k][1:][bad].mean()) if bad.any() else float("nan")
                rows.append(row)
                if j < 2:
                    save_panel(os.path.join(a.out, f"{name}_{j}.png"), ex, M, masks)
            res[name] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
            r = res[name]
            print(f"{name:12s} PSNR ones {r['ones']:.2f} r0 {r['r0']:.2f} nn {r['nn']:.2f} (d {r['nn'] - r['r0']:+.2f})"
                  f" | alt r0 {r['r0b']:.2f} nn {r['nnb']:.2f} (d {r['nnb'] - r['r0b']:+.2f}) | oracle {r['oracle']:.2f}"
                  f" | sqrt-dom d {r['nn_sqrt'] - r['r0_sqrt']:+.2f} | keep r0 {r['keep_r0']:.2f} nn {r['keep_nn']:.2f}"
                  f" leak r0 {r['leak_r0']:.2f} nn {r['leak_nn']:.2f}", flush=True)
    allm = {k: float(np.mean([res[n][k] for n in res])) for k in res[scenarios.NAMES[0]]}
    res["mean"] = allm
    print(f"{'MEAN':12s} PSNR ones {allm['ones']:.2f} r0 {allm['r0']:.2f} nn {allm['nn']:.2f} "
          f"(d {allm['nn'] - allm['r0']:+.2f}) | alt d {allm['nnb'] - allm['r0b']:+.2f} | oracle {allm['oracle']:.2f}"
          f" | sqrt-dom d {allm['nn_sqrt'] - allm['r0_sqrt']:+.2f}")
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
