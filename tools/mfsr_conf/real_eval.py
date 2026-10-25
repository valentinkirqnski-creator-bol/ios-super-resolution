"""Real iPhone bursts NOT used for training (burst9, burst11, burst12): the app's own loader, exposure matching,
aligner, analytic mask and (exact replica of the) merge, with the classical mask vs the network.

    python real_eval.py runs/m1/best.pt --burst C:/Users/valen/Downloads/burst11 --box 1800 1200 384 384 \
        [--ref APC_1278] [--homog] [--out <dir>]

There is NO ground truth for real bursts. What this reports is what can be measured without it: the mean weight the
masks give in flat and in high-gradient areas (retention), and side-by-side renders of the merged 2x crops and of the
masks for visual inspection. Synthetic validation does not prove real-camera generalisation; these crops are the check.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                     # noqa: E402
import torch                           # noqa: E402
from PIL import Image                  # noqa: E402

from mfsr import harness, merge as mg, model as M, sample               # noqa: E402
from mfsr.config import tool_path                                       # noqa: E402
from mfsr.inputs import frame_inputs, quad_mean                          # noqa: E402

DUMP = tool_path("build", "frame_dump.exe")
CACHE = tool_path("data", "real_cache")
APP_TUNE = [0.17, 1.0, 0.76, 1.12, 3.0, 1.0]       # the shipped k_detail k_denoise D_th D_tr k_stretch k_shrink


def dump(path):
    name = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(CACHE, os.path.basename(os.path.dirname(path)))
    os.makedirs(d, exist_ok=True)
    f32, meta = os.path.join(d, name + ".f32"), os.path.join(d, name + ".json")
    if not os.path.exists(meta):
        out = subprocess.run([DUMP, path, f32], capture_output=True, text=True, check=True).stdout.split()
        keys = ["h", "w", "c00", "c01", "c10", "c11", "alpha", "beta", "wb0", "wb1", "wb2", "prewhite", "exposure"]
        json.dump(dict(zip(keys, [float(v) for v in out])), open(meta, "w"))
    m = json.load(open(meta))
    img = np.fromfile(f32, np.float32).reshape(int(m["h"]), int(m["w"]))
    return img, m


def gain_match(img, g, prewhite, cfa):
    """The app's exposure matching (pipeline_paths.cpp + raw_io.cpp apply_raw_gain, as fixed in fa1fb8b): one gain
    from shutter x ISO, brightened values capped at each CFA site's saturation (prewhite[c] in the loader domain)."""
    if abs(g - 1.0) <= 0.002:
        return img
    cap = np.empty_like(img)
    for i in range(2):
        for j in range(2):
            cap[i::2, j::2] = prewhite[cfa[i][j]]
    q = img * g
    if g > 1.0:
        q = np.where(q > cap, np.maximum(img, cap), q)
    return q


def render(rgb, ref_lum_p99):
    """Display only: the loader domain is white-balanced, so green saturates at 1.0 while red / blue can reach their
    white-balance gain; clip every channel at 1.0 first so clipped highlights stay neutral."""
    x = np.clip(np.minimum(rgb, 1.0) / max(min(ref_lum_p99, 1.0), 1e-6), 0, 1) ** (1 / 2.2)
    return (x * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--burst", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--box", nargs=4, type=int, required=True, help="x0 y0 w h of the evaluated region (raw px)")
    ap.add_argument("--homog", action="store_true")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/mfsr_conf_outputs"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    files = sorted(glob.glob(os.path.join(a.burst, "*.dng")) + glob.glob(os.path.join(a.burst, "*.DNG")))
    if a.ref:
        files = [f for f in files if a.ref in f] + [f for f in files if a.ref not in f]
    files = files[:a.frames]
    frames, metas = zip(*[dump(f) for f in files])
    m0 = metas[0]
    cfa = ((int(m0["c00"]), int(m0["c01"])), (int(m0["c10"]), int(m0["c11"])))
    wb = np.array([m0["wb0"], m0["wb1"], m0["wb2"]], np.float32)
    x0, y0, w, h = (v // 2 * 2 for v in a.box)
    lo, hi = 64, 128                                   # network halo + clear of the aligner's border band
    H, W = frames[0].shape
    cy0, cx0 = max(0, y0 - lo), max(0, x0 - lo)
    cy1, cx1 = min(H, y0 + h + hi), min(W, x0 + w + hi)
    cy0, cx0 = cy0 // 2 * 2, cx0 // 2 * 2
    S_h, S_w = (cy1 - cy0) // 2 * 2, (cx1 - cx0) // 2 * 2
    raw = []
    for img, mt in zip(frames, metas):
        g = 1.0
        if m0["exposure"] > 0 and mt["exposure"] > 0:
            g = float(np.clip(m0["exposure"] / mt["exposure"], 0.67, 1.5))
        raw.append(gain_match(img[cy0:cy0 + S_h, cx0:cx0 + S_w], g, wb, cfa))
    raw = np.stack(raw).astype(np.float32)
    N = raw.shape[0]
    a_n, b_n = m0["alpha"], m0["beta"]
    z2 = np.zeros((N, S_h, S_w, 2), np.float32)
    hres = harness.run(raw, z2, z2, np.tile(np.eye(3), (N, 1, 1)), cfa, 2.0, a_n, b_n, wb,
                       flags=harness.FLAG_HOMOG if a.homog else 0, tile=16, tune=APP_TUNE)
    tile = hres["tile"]
    rawt = torch.from_numpy(raw)
    # the network inputs expect square full frames only through their own index math; crop is rectangular -> pad
    S = max(S_h, S_w)
    pad = lambda x: torch.nn.functional.pad(x, (0, S - x.shape[-1], 0, S - x.shape[-2]), mode="replicate")
    rawp = pad(rawt[:, None])[:, 0]
    greyr = pad(torch.from_numpy(hres["grey_ref"])[None, None])[0, 0]
    grey = pad(torch.from_numpy(hres["grey"])[:, None])[:, 0]
    r0 = torch.from_numpy(hres["r0"])
    r0g = pad(r0[:, None, ::2, ::2])[:, 0, :S // 2, :S // 2]
    flows, Hs, hh = torch.from_numpy(hres["flow"]), torch.from_numpy(hres["H"]), hres["has_h"]
    ry, rx = y0 - cy0, x0 - cx0                         # region origin inside the crop
    maps = {"ones": torch.ones(N - 1, h + 1, w + 1), "classical": r0[:, ry:ry + h + 1, rx:rx + w + 1]}
    # the app's shipped learned refiners on the same inputs (evaluate.py load_shipped / shipped_maps)
    import evaluate as EV
    v21, pix = EV.load_shipped()
    xg0, _ = frame_inputs(rawp, greyr, grey, r0g, flows, Hs, hh, tile, a_n, b_n, wb, False, cfa)
    fake = {"r0": r0g.repeat_interleave(2, -2).repeat_interleave(2, -1)[None], "xg": xg0[None]}
    sm = EV.shipped_maps(v21, pix, fake, N - 1, 0, S + 1)
    for k_ in ("v21 (app)", "v39 (app)"):
        maps[k_] = sm[k_][:, ry:ry + h + 1, rx:rx + w + 1]
    for mp in a.models:
        ck = torch.load(mp, map_location="cpu", weights_only=False)
        net = M.build(ck["cfg"]["model"])
        net.load_state_dict(ck["state"])
        net.eval()
        use_r0 = bool(ck["cfg"]["model"].get("use_r0") or ck["cfg"]["model"].get("refine_r0"))
        xg, xr = frame_inputs(rawp, greyr, grey, r0g, flows, Hs, hh, tile, a_n, b_n, wb, use_r0, cfa)
        r0raw = r0g.repeat_interleave(2, -2).repeat_interleave(2, -1)
        with torch.no_grad():
            lg = net(xg[None], xr[None], torch.ones(1, N - 1), r0raw[None] if use_r0 else None)[0]
        Rn = torch.sigmoid(lg[:, 1, ry:ry + h + 1, rx:rx + w + 1])
        tag = os.path.basename(os.path.dirname(mp))
        maps[tag] = Rn
        maps[f"{tag}*cls"] = Rn * maps["classical"]
    # merge (exact replica of the app's merge) of the region with every map
    var = torch.zeros_like(rawt)
    hv = dict(flow=hres["flow"], H=hres["H"], has_h=hh)
    cov = np.concatenate([hres["cov_ref"][None], hres["cov"]], 0)
    parts = sample.merge_parts_xy(rawt, rawt, var, cov, hv, 2.0, cfa, ry, rx, max(h, w), tile, with_noisy=True) \
        if h == w else None
    if parts is None:
        raise SystemExit("use a square --box")
    lum_ref = None
    tag = os.path.basename(a.burst.rstrip("/\\")) + f"_x{x0}_y{y0}_{w}" + ("_H" if a.homog else "")
    od = os.path.join(a.out, tag)
    os.makedirs(od, exist_ok=True)
    gy, gx = np.gradient(quad_mean(rawt[0]).numpy())
    gmag = np.hypot(gx, gy)[ry // 2:(ry + h) // 2 + 1, rx // 2:(rx + w) // 2 + 1]
    gmag = np.repeat(np.repeat(gmag, 2, 0), 2, 1)[:h + 1, :w + 1]
    s_noise = np.sqrt(a_n * quad_mean(rawt[0]).numpy().mean() + b_n)
    flat, edge = gmag < 0.5 * s_noise, gmag > 4 * s_noise
    stats, tiles = {}, []
    for name, R in maps.items():
        Mrgb = parts.merged(parts.fetch(R)).permute(1, 2, 0).numpy()
        if lum_ref is None:
            lum_ref = float(np.percentile(Mrgb.mean(-1), 99.5))
        Image.fromarray(render(Mrgb, lum_ref)).save(os.path.join(od, f"merge_{name.replace('*', 'x')}.png"))
        mR = R.mean(0).numpy()
        Image.fromarray((np.clip(mR, 0, 1) * 255).astype(np.uint8)).resize((2 * (w + 1), 2 * (h + 1)),
                                                                          Image.NEAREST).save(
            os.path.join(od, f"mask_{name.replace('*', 'x')}.png"))
        Rn_ = R.numpy()
        stats[name] = dict(mean=float(Rn_.mean()), flat=float(Rn_[:, flat].mean()) if flat.any() else None,
                           edge=float(Rn_[:, edge].mean()) if edge.any() else None,
                           flat_rejected=float((Rn_[:, flat] < 0.5).mean()) if flat.any() else None,
                           edge_rejected=float((Rn_[:, edge] < 0.5).mean()) if edge.any() else None)
    json.dump(dict(burst=a.burst, box=[x0, y0, w, h], frames=[os.path.basename(f) for f in files], homog=a.homog,
                   tile=tile, exposure=[mt["exposure"] for mt in metas], stats=stats),
              open(os.path.join(od, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    print("saved", od)


if __name__ == "__main__":
    main()
