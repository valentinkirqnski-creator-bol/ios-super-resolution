"""Reconstruction-based evaluation on the TEST split (scene APC_1278, never seen in training or validation).

    python evaluate.py runs/m1/best.pt [--more runs/m2/best.pt ...] [--out runs/m1/eval]

Masks compared on every test burst and both alignment variants (estimated / deliberately corrupted):
  ones         R = 1 everywhere (no robustness)
  ref_only     R = 0 (reference frame alone)
  classical    the app's analytic mask (Wronski Eq. 5-9 as shipped, raw resolution)
  neural       the network's merge-confidence map
  neural*cls   product with the classical mask
  min          min(neural, classical)
  oracle       the label R* (reconstruction-optimal; an upper bound, not achievable)
Metrics: mfsr/metrics.py. Results per category and overall -> <out>/results.json and <out>/results.md; runtime and
size of the network -> <out>/runtime.json.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np          # noqa: E402
import torch                # noqa: E402

from mfsr import model as M, metrics as MT       # noqa: E402
from mfsr.data import Bursts, collate, parts_of   # noqa: E402

KEYS = ["psnr", "exp_err", "ghost", "ghost_edge", "detail", "false_detail", "noise", "harm_rejected", "valid_kept",
        "w_misaligned", "w_aligned"]


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = M.build(ck["cfg"]["model"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["cfg"]


def load_shipped():
    """The app's shipped learned refiners (tools/robust_nn): v21 (Settings "Robust NN v21") and v39 on top of it
    ("Content-aware pixel net"), exactly as exported: v39 with its +0.25 logit shift folded in, both followed by the
    app's default sharpening R' = clamp((R - 0.20) / (0.90 - 0.20))."""
    from mfsr import ROBUST_NN
    from rnn.model import RobustNet
    from rnn.pixnet import PixNet, NPIX
    sd = torch.load(os.path.join(ROBUST_NN, "runs", "v21", "model.pt"), map_location="cpu")
    v21 = RobustNet(nf=sd["c1.weight"].shape[1], hid=sd["c1.weight"].shape[0])
    v21.load_state_dict(sd)
    v21.eval()
    base = RobustNet(nf=sd["c1.weight"].shape[1], hid=sd["c1.weight"].shape[0])
    base.load_state_dict(sd)
    pix = PixNet(base, ch=16, npix=NPIX)
    pix.load_state_dict(torch.load(os.path.join(ROBUST_NN, "runs", "v39", "pix_s4790.pt"), map_location="cpu"))
    pix.eval()
    return v21, pix


def shipped_maps(v21, pix, b, K, halo, Pc1):
    """v21 / v39 on the same crop: their 14 features are the network's guide inputs with feature 0 = logit(R0)/4
    (ours zero it), their pixel channels are the same 4 channels."""
    r0g = b["r0"][0, :K, ::2, ::2]
    f = b["xg"][0, :K, :14].clone()
    rc = r0g.clamp(1e-3, 1 - 1e-3)
    f[:, 0] = 0.25 * torch.log(rc / (1 - rc))
    with torch.no_grad():
        R21, _ = v21(f, r0g)
        R39, _ = pix(f, r0g, b["xg"][0, :K, 14:18])
        R39 = torch.sigmoid(torch.logit(R39.clamp(1e-6, 1 - 1e-6)) + 0.25)
    sharp = lambda R: ((R - 0.20) / 0.70).clamp(0, 1)
    up = lambda R: R.repeat_interleave(2, -2).repeat_interleave(2, -1)[:, halo:halo + Pc1, halo:halo + Pc1]
    return {"v21 (app)": up(sharp(R21)), "v39 (app)": up(sharp(R39)), "v21 raw": up(R21)}


def single(batch, i):
    """Crop i of a collated batch as single-sample tensors (frames trimmed to the burst's own K)."""
    K = batch["meta"][i]["K"]
    p = batch["parts"]
    from mfsr import merge as mg
    P = mg.Parts(p["Aref"][i], p["Acref"][i], p["Bref"][i], p["Vref"][i], p["A"][i, :K], p["Ac"][i, :K], p["B"][i, :K],
                 p["V"][i, :K], batch["fy"], batch["fx"], p["Ag"][i, :K], p["Bg"][i, :K], p["vis"][i, :K])
    P.g = p["g"][i, :K]
    return P, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--more", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shipped", action="store_true", help="also evaluate the app's v21 / v39 learned masks")
    ap.add_argument("--sharpen", default="", help="extra masks R' = clamp((R - lo) / (hi - lo)), e.g. 0.2:0.8,0.3:0.9")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    nets = [(os.path.basename(os.path.dirname(a.model)) or "neural", *load(a.model))]
    for p in a.more:
        nets.append((os.path.basename(os.path.dirname(p)), *load(p)))
    cfg0 = nets[0][2]
    out = a.out or os.path.join(os.path.dirname(a.model), "eval")
    os.makedirs(out, exist_ok=True)
    root = os.path.join(HERE, cfg0["data"]) if not os.path.isabs(cfg0["data"]) else cfg0["data"]
    halo = cfg0["halo"]
    use_r0 = {n: bool(c["model"].get("use_r0") or c["model"].get("refine_r0")) for n, _, c in nets}
    ds = {u: Bursts(root, a.split, crop=128, halo=halo, random_crop=False, use_r0=u, limit=a.limit, with_noisy=True)
          for u in set(use_r0.values())}
    n_items = len(next(iter(ds.values())))
    res = defaultdict(lambda: defaultdict(list))        # (category, variant) -> mask -> list of metric dicts
    shipped = load_shipped() if a.shipped else None
    t_net, n_frames_net = 0.0, 0
    for i in range(n_items):
        batches = {u: collate([d[i]]) for u, d in ds.items()}
        b = batches[False] if False in batches else next(iter(batches.values()))
        parts, K = single(b, 0)
        Rs, e, vis = b["R"][0, :K], b["e"][0, :K], b["vis"][0, :K]
        T, s2 = b["T"][0], b["sig2"][0]
        Pc1 = Rs.shape[-1]
        r0 = b["r0"][0, :K, halo:halo + Pc1, halo:halo + Pc1]
        maps = {"ones": torch.ones_like(Rs), "ref_only": torch.zeros_like(Rs), "classical": r0, "oracle": Rs}
        if shipped is not None:
            maps.update(shipped_maps(*shipped, batches[False] if False in batches else b, K, halo, Pc1))
        for name, net, c in nets:
            bb = batches[use_r0[name]]
            t0 = time.time()
            with torch.no_grad():
                lg = net(bb["xg"], bb["xr"], bb["mask"], bb["r0"] if use_r0[name] else None)
            t_net += time.time() - t0
            n_frames_net += K
            Rn = torch.sigmoid(lg[0, :K, 1, halo:halo + Pc1, halo:halo + Pc1])
            tag = "neural" if len(nets) == 1 else name
            maps[tag] = Rn
            maps[f"{tag}*cls"] = Rn * r0
            maps[f"min({tag},cls)"] = torch.minimum(Rn, r0)
            for sp in [x for x in a.sharpen.split(",") if x]:
                lo, hi = (float(v) for v in sp.split(":"))
                maps[f"{tag}[{lo:g},{hi:g}]"] = ((Rn - lo) / (hi - lo)).clamp(0, 1)
        meta = b["meta"][0]
        key = (meta["category"], "estimated" if meta["variant"] == 0 else "corrupted")
        for mname, Rm in maps.items():
            res[key][mname].append(MT.evaluate_map(parts, Rm, Rs, e, vis, T, s2))
        if i % 20 == 0:
            print(f"{i + 1}/{n_items}", flush=True)
    # aggregate
    def agg(lst):
        o = {}
        for k in KEYS:
            v = [d[k] for d in lst if not (isinstance(d[k], float) and np.isnan(d[k]))]
            o[k] = float(np.mean(v)) if v else float("nan")
        o["n"] = len(lst)
        return o
    table = {}
    for (cat, var), dm in sorted(res.items()):
        table[f"{cat}/{var}"] = {m: agg(l) for m, l in dm.items()}
    overall = defaultdict(list)
    for (cat, var), dm in res.items():
        for m, l in dm.items():
            overall[m] += l
    table["ALL"] = {m: agg(l) for m, l in overall.items()}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(table, f, indent=1)
    masks = list(table["ALL"].keys())
    lines = ["# Test split (scene APC_1278) - reconstruction-based evaluation", "",
             "Errors in single-frame noise-variance units unless noted; PSNR on the noisy merged 2x image vs the HR "
             "target. Lower is better for exp_err / ghost / false_detail / noise; higher for psnr / detail / "
             "harm_rejected / valid_kept.", ""]
    for key in ["ALL"] + [k for k in table if k != "ALL"]:
        lines += [f"## {key}  (n = {table[key][masks[0]]['n']} crops)", "",
                  "| mask | " + " | ".join(KEYS) + " |", "|" + "---|" * (len(KEYS) + 1)]
        for m in masks:
            if m not in table[key]:
                continue
            r = table[key][m]
            lines.append(f"| {m} | " + " | ".join(f"{r[k]:.4g}" for k in KEYS) + " |")
        lines.append("")
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write("\n".join(lines))
    rt = {}
    for name, net, c in nets:
        rt[name] = dict(params=net.n_params(), macs_per_quad_per_frame=net.macs_per_quad(),
                        est_gmac_12mp_per_frame=net.macs_per_quad() * 3.0e6 / 1e9)
    rt["cpu_ms_per_frame_128px_patch"] = 1000.0 * t_net / max(1, n_frames_net)
    with open(os.path.join(out, "runtime.json"), "w") as f:
        json.dump(rt, f, indent=1)
    print("\n".join(lines[:12 + len(masks)]))
    print(json.dumps(rt))


if __name__ == "__main__":
    main()
