"""Train v15 (PixNet: pixel-seeing correction over the frozen shipped v14) on the cached app-stage outputs.

    python train_pix.py --base runs/v14/model.pt --out runs/v15 --steps 30000
Scored against the corrected ground-truth mask on the 60 held-out samples, always next to v14 as shipped
(cutoffs 0.20/0.90): similarity (1 - mean|R - t|, all pixels), boundary error (mean|R - t| where the GT changes
from reject to keep within 5x5 - how FINE the mask is), balanced accuracy, misses, over-rejection, small-shift /
rotation misses, flat over-rejection, partial ghost weight. v15 is also shown at a few logit shifts (weights-only).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from torch.utils.data import DataLoader, Dataset            # noqa: E402
from real_protocol import load_net, stretch, geo_loss, geo_target, gt_edge_mask, CLIP   # noqa: E402
from rnn.features import compute_features, consensus_features, edge_line_features   # noqa: E402
from rnn.common import warp_by_flow                        # noqa: E402
from rnn.pixnet import PixNet, pixel_inputs, consensus_inputs, expand_inputs, NPIX, NCONS   # noqa: E402

LO, HI = 0.20, 0.90                                         # the shipped exact-0/1 cutoffs


CLEAN = os.environ.get("PIX_CLEAN", "0") == "1"   # ceiling test: pixel inputs from the NOISE-FREE frames
CONS = os.environ.get("PIX_CONS", "0") == "1"     # v16: + multi-frame consensus pixel channels


SRC = {"c15_val": ["rp13_val"], "c15_train": ["rp13_train", "rp13_train_hdrp"], "c17_val": ["rp17_val"],
       "c17_train": ["rp17_train"], "c19_val": ["rp19_val"], "c19_train": ["rp19_train"],
       "c19s1_val": ["rp19s1_val"], "c19s1_train": ["rp19s1_train"], "c19s05_val": ["rp19s05_val"],
       "c19s05_train": ["rp19s05_train"],
       # v20: whole-camera rotation + homography, normal bursts + homography, perfectly alignable bursts
       "c20rot_train": ["rp20rot_train"], "c20rot_val": ["rp20rot_val"], "c20h_train": ["rp20h_train"],
       "c20h_val": ["rp20h_val"], "c20pH_train": ["rp20pH_train"], "c20pH_val": ["rp20pH_val"],
       "c20p_train": ["rp20p_train"], "c20p_val": ["rp20p_val"],
       # v21: aliased fine-text bursts, perfectly aligned (p) and with deliberate misalignment on the same text (m)
       "c21txp_train": ["rp21tx_train"], "c21txm_train": ["rp21tx_train"], "c21txp_val": ["rp21tx_val"],
       "c21txm_val": ["rp21tx_val"],
       # v22: text 1.8-9 px perfectly aligned (p) / tiny edge slips (m); text + tiny tilt/shift patches (r);
       # real scenes, translation-only camera + tiny tilt/shift patches (s)
       "c22txp_train": ["rp22tx_train"], "c22txm_train": ["rp22tx_train"], "c22txp_val": ["rp22tx_val"],
       "c22txm_val": ["rp22tx_val"], "c22txr_train": ["rp22txr_train"], "c22txr_val": ["rp22txr_val"],
       "c22scr_train": ["rp22scr_train"], "c22scr_val": ["rp22scr_val"],
       # v23: edge segments slightly rotated the same way in most frames (the "same edge slightly rotated" artifact)
       "c23er_train": ["rp23er_train"], "c23er_val": ["rp23er_val"], "c23ert_train": ["rp23ert_train"],
       "c23ert_val": ["rp23ert_val"],
       # v24: 768 px bursts at camera pixel scale - perfectly aligned (p), ghosted / thicker edges from small vector
       # noise on edge tiles (g), zippery edges from a slight camera roll under one translation per tile (z)
       "c24p_train": ["rp24ph_train", "rp24tx_train"], "c24g_train": ["rp24ph_train", "rp24tx_train"],
       "c24z_train": ["rp24z_train"], "c24p_val": ["rp24ph_val"], "c24pt_val": ["rp24tx_val"],
       "c24g_val": ["rp24ph_val"], "c24gt_val": ["rp24tx_val"], "c24z_val": ["rp24z_val"],
       # v26: 3 MP (1728 px) bursts, real photos at camera scale - ghosted / thicker (g), zippery (z), perfect (p)
       "c26g_train": ["rp26ph_train"], "c26p_train": ["rp26ph_train"], "c26z_train": ["rp26z_train"],
       "c26g_val": ["rp26ph_val"], "c26p_val": ["rp26ph_val"], "c26z_val": ["rp26z_val"],
       **{f"cchart{i}_val": [f"rpchart{i}"] for i in range(4)}, "c8mp_val": ["rp8mp"],
       # v32: pairs (P perfect by the app's aligner + homography / C corrupted) of rotated bursts at camera scale
       **{f"c32{k}{v}_{sp}": [f"rp32{k}_{sp}"] for k in ("rot", "roll") for v in ("P", "C") for sp in ("train", "val")},
       # v33: burst12-like slow roll + zoom, app aligner WITHOUT the homography (A: stepped edges) / WITH it (H: clean)
       **{f"c33{v}_{sp}": [f"rp33_{sp}"] for v in ("A", "H", "O") for sp in ("train", "val")},
       # v27: edge-chart training bursts (thin lines .. big blocks) - slipped edge tiles (T) and the same perfect (P)
       **{f"cchartT{i}_train": [f"rpchartT{i}"] for i in range(4)}, **{f"cchartP{i}_train": [f"rpchartT{i}"] for i in range(4)},
       # v28: uniform 0.2-2 px slips (tile vectors anywhere + translated / rotated content patches), photos / text / charts
       **{f"c28{k}_{s}": [f"rp28{k}_{s}"] for k in ("p", "t", "c") for s in ("train", "val")}}
PRE_M = 24   # v24: guide px of context kept around a training crop before features / target are computed
REGT = os.environ.get("RNN_GT_NMERGE") is not None   # v17: recompute the target (merged-noise visibility)
GT_HR = os.environ.get("RNN_GT_HR", "0") == "1"       # v18: full-raw-resolution target from <cache>_hr/


def _source(cache_file):
    """The synthetic burst a cache sample was made from (for its noise-free frames)."""
    stem = os.path.basename(cache_file)[:-4]
    roots = SRC[os.path.basename(os.path.dirname(cache_file))]
    if stem.startswith("val"):
        src = sorted(glob.glob(os.path.join(HERE, "data", roots[0], "*.npz")))
        return src[int(stem[3:]) % len(src)]
    name = stem.rsplit("_v", 1)[0] + ".npz"
    for r in roots:
        if os.path.exists(os.path.join(HERE, "data", r, name)):
            return os.path.join(HERE, "data", r, name)
    raise FileNotFoundError(name)


class Cache(Dataset):
    def __init__(self, root, crop=None, aug=False):
        self.files = sorted(f for f in glob.glob(os.path.join(HERE, root, "*.npz")) if ".tmp." not in f)
        self.crop, self.aug = crop, aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        rng = np.random.default_rng()
        # v24: for big samples, cut the training crop (plus PRE_M context) out first, so features and target are only
        # computed where they are used. Small samples keep the old full-image path.
        Hf, Wf = z["ref"].shape
        win = None
        if self.crop and min(Hf, Wf) > self.crop + 2 * PRE_M:
            y0, x0 = int(rng.integers(0, Hf - self.crop + 1)), int(rng.integers(0, Wf - self.crop + 1))
            if os.environ.get("RNN_TILE_CROP") == "1":   # v30: crops on the tile grid (tile-aligned model)
                tg = max(1, int(z["tile"]) // 2); y0, x0 = y0 // tg * tg, x0 // tg * tg
            ya, xa = max(0, y0 - PRE_M), max(0, x0 - PRE_M)
            win = (ya, min(Hf, y0 + self.crop + PRE_M), xa, min(Wf, x0 + self.crop + PRE_M), y0 - ya, x0 - xa)
        S = (lambda v: np.ascontiguousarray(v[..., win[0]:win[1], win[2]:win[3]])) if win else (lambda v: v)
        Ir, G = torch.from_numpy(S(z["ref"])), torch.from_numpy(S(z["grey"]))
        fl, r0 = torch.from_numpy(S(z["flow"])), torch.from_numpy(S(z["r0"]).astype(np.float32))
        K, (H, W) = G.shape[0], G.shape[1:]
        a, b = float(z["a"]), float(z["b"])
        Ik = Ir[None].expand(K, -1, -1).contiguous()
        ffl = torch.from_numpy(S(z["rflow"])) if "rflow" in z.files else None   # v20: residual after the homography
        with torch.no_grad():
            feats, J = compute_features(Ik, G, fl, r0, a, b, CLIP, max(1, int(z["tile"]) // 2), ffl)
            pix = pixel_inputs(Ik, J, a, b)
            if os.environ.get("EDGEF") == "1":   # v26: edge-line shift of this frame's edges (no extra buffers)
                feats = torch.cat([feats, edge_line_features(Ik, J, a, b)], 1)
            if os.environ.get("CONSF") == "1" and K > 1:   # v25: burst consistency - each frame vs the leave-one-out
                feats = torch.cat([feats, consensus_features(Ir, J, r0, a, b)], 1)   # consensus of the others
            if CONS:
                pix = torch.cat([pix, consensus_inputs(Ir, J, a, b)], 1)
            if CLEAN:
                cl = torch.from_numpy(np.load(_source(self.files[i]))["clean_grey"])
                pix = pixel_inputs(cl[0][None].expand(K, -1, -1).contiguous(), warp_by_flow(cl[1:], fl), a, b)
        geo = torch.from_numpy(S(z["geo"]).astype(np.float32))
        if GT_HR:   # v18: full-raw-resolution target (precompute_hr.py)
            hp = os.path.join(os.path.dirname(self.files[i]) + "_hr", os.path.basename(self.files[i]))
            geo = torch.from_numpy(np.load(hp)["geo_hr"].astype(np.float32))
        elif REGT:   # target recomputed from the burst's noise-free frames with the current visibility rule
            sz = np.load(_source(self.files[i]))
            cg_ = torch.from_numpy(S(sz["clean_grey"]))
            vis_px = torch.from_numpy(S(sz["vis"][1:]).astype(np.float32))   # v22: lets counts skip out-of-view pixels
            gtedge = (gt_edge_mask(cg_).float()[None].expand(K, -1, -1) if os.environ.get("RNN_GT_EDGE") else None)
            # v24: the geometry rule on outlines only for the sets named in RNN_GT_GEO_SETS (prefixes)
            gsets = [x for x in os.environ.get("RNN_GT_GEO_SETS", "").split(",") if x]
            gtau = None
            if gsets:
                setname = os.path.basename(os.path.dirname(self.files[i]))
                gtau = os.environ.get("RNN_GT_GEO_TAU", "0.15") if any(setname.startswith(p) for p in gsets) else "0"
            geo = geo_target(cg_, fl, torch.from_numpy(S(sz["gt_flow"][1:])), vis_px, 4 * a, 4 * b, occ_look=True,
                             geo_tau=gtau)
        d = dict(feats=feats, pix=pix, r0=r0, geo=geo, ts=torch.full((K,), max(1, int(z["tile"]) // 2)),
                 perfect=torch.full((K,), 0.0 if bool(z["corrupt"]) else 1.0),   # v32: no deliberate slip in this sample
                 **({"vis": vis_px} if REGT and not GT_HR else {}),
                 **({"gtedge": gtedge} if REGT and not GT_HR and gtedge is not None else {}),
                 edge=torch.from_numpy(S(z["edge"]).astype(np.float32))[None].expand(K, -1, -1),
                 rot=torch.from_numpy(S(z["rot"])), ferr=torch.from_numpy(S(z["ferr"]).astype(np.float32)))
        if win:
            oy, ox = win[4], win[5]
            d = {k: (v[..., oy:oy + self.crop, ox:ox + self.crop] if v.dim() >= 2 else v) for k, v in d.items()}
        elif self.crop:
            y, x = int(rng.integers(0, H - self.crop + 1)), int(rng.integers(0, W - self.crop + 1))
            if os.environ.get("RNN_TILE_CROP") == "1":
                tg = max(1, int(z["tile"]) // 2); y, x = y // tg * tg, x // tg * tg
            d = {k: (v[..., y:y + self.crop, x:x + self.crop] if v.dim() >= 2 else v) for k, v in d.items()}
        if self.aug:   # every input is a rotation-invariant scalar field, so flips / 90-degree turns are exact
            k90, fl_ = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
            d = {k: (torch.rot90(v, k90, (-2, -1)) if v.dim() >= 2 else v) for k, v in d.items()}
            if fl_:
                d = {k: (torch.flip(v, (-1,)) if v.dim() >= 2 else v) for k, v in d.items()}
        return {k: v.contiguous() for k, v in d.items()}


def cat(batch):
    return {k: torch.cat([b[k] for b in batch], 0) for k in batch[0]}


def _wi(_):
    torch.set_num_threads(1)


def counts(R, d, lo=LO, hi=HI):
    t, Rs = d["geo"], stretch(R, lo, hi)
    bad, good = t < 0.2, t > 0.8
    if os.environ.get("RNN_MISS_VISIBLE") == "1" and "vis" in d:   # v22: misses only where the frame has content
        bad = bad & (d["vis"] > 0.5)
    miss, over = bad & (Rs >= 0.5), good & (Rs <= 0.5)
    se, ro, fl = bad & (d["ferr"] < 1), bad & d["rot"], good & (d["edge"] < 0.5)
    band = (F.max_pool2d(t[:, None], 5, 1, 2) + F.max_pool2d(-t[:, None], 5, 1, 2))[:, 0] > 0.5
    err = (Rs - t).abs()
    return np.array([float(v) for v in (miss.sum(), bad.sum(), over.sum(), good.sum(), (miss & se).sum(), se.sum(),
                                        (miss & ro).sum(), ro.sum(), (over & fl).sum(), fl.sum(), Rs[bad].sum(),
                                        Rs[t < 0.05].sum(), (t < 0.05).sum(), err.sum(), t.numel(), err[band].sum(),
                                        band.sum(), (1 - Rs)[good].sum(), good.sum())])


def summary(c):
    q = lambda j: 100 * c[j] / max(1.0, c[j + 1])
    return dict(sim=1 - c[13] / c[14], bnd=c[15] / max(1.0, c[16]), bal=100 - 0.5 * (q(0) + q(2)), missed=q(0),
                over=q(2), small=q(4), rot=q(6), flat=q(8), leak=c[10] / max(1.0, c[1]), leak05=c[11] / max(1.0, c[12]),
                lost=100 * c[17] / max(1.0, c[18]) if len(c) > 18 else float("nan"))


def fmt(name, s):
    return (f"{name:16s} sim {s['sim']:.4f}  boundary err {s['bnd']:.3f}  bal {s['bal']:.2f}  missed {s['missed']:.2f}  "
            f"over {s['over']:.2f}  small {s['small']:.2f}  rot {s['rot']:.2f}  flat {s['flat']:.2f}  "
            f"leak {s['leak']:.3f}  leak<.05 {s['leak05']:.3f}  good-weight lost {s['lost']:.2f}%")


SHIFTS = [-1.5, -1.25, -1.125, -1.0, -0.875, -0.75, -0.5, 0.0]


@torch.no_grad()
def evaluate(net, dv, base_too=False):
    C = {}
    for d in dv:
        if base_too:
            C["v14"] = C.get("v14", 0) + counts(net.base(d["feats"], d["r0"])[0], d)
        npix = getattr(net, "npix", None) or net.p1.in_channels - net.base.c1.weight.shape[1] - 1
        if hasattr(net, "set_ts"):
            net.set_ts(d.get("ts"))
        R = net(d["feats"], d["r0"], d["pix"][:, :npix])[0].clamp(1e-6, 1 - 1e-6)
        lg = torch.log(R) - torch.log1p(-R)
        for s in SHIFTS:
            C[s] = C.get(s, 0) + counts(torch.sigmoid(lg + s), d)
    return {k: summary(v) for k, v in C.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/v14/model.pt")
    ap.add_argument("--train", default="data/c15_train")
    ap.add_argument("--val", default="data/c15_val")
    ap.add_argument("--out", default="runs/v15")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    npix = NPIX + (NCONS if CONS else 0)
    net = PixNet(load_net(os.path.join(HERE, a.base)), ch=a.ch, npix=npix)
    if a.init:
        sd = torch.load(os.path.join(HERE, a.init), map_location="cpu")
        nf = net.base.c1.weight.shape[1]
        old = sd["p1.weight"].shape[1] - nf - 1
        if old != npix:   # start exactly at the smaller net (new input channels zero)
            sd = expand_inputs(sd, nf, old, npix)
        net.load_state_dict(sd)
    mp = dict(persistent_workers=True, prefetch_factor=2) if a.workers else {}
    dl = DataLoader(Cache(a.train, a.crop, True), a.batch, shuffle=True, num_workers=a.workers, collate_fn=cat,
                    worker_init_fn=_wi, drop_last=True, **mp)
    dv = DataLoader(Cache(a.val), 1, shuffle=False, num_workers=a.workers, collate_fn=cat, worker_init_fn=_wi)
    params = [p for p in net.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params)}; train {len(dl.dataset)} val {len(dv.dataset)}", flush=True)
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    net.eval()
    ev = evaluate(net, dv, base_too=True)
    print(fmt("v14 (shipped)", ev["v14"]), flush=True)
    print(fmt("v15 step 0", ev[0.0]), flush=True)
    ref = ev["v14"]
    hist = [dict(step=0, ev={str(k): v for k, v in ev.items()})]
    step, t0, run = 0, time.time(), None
    net.train()
    net.base.eval()
    while step < a.steps:
        for d in dl:
            R, _ = net(d["feats"], d["r0"], d["pix"])
            loss = geo_loss(R, d["geo"], d=d)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run = float(loss) if run is None else 0.99 * run + 0.01 * float(loss)
            if step % 250 == 0:
                print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                net.eval()
                ev = evaluate(net, dv)
                net.train()
                net.base.eval()
                for s in SHIFTS:
                    print(fmt(f"v15 s{step} {s:+.2f}", ev[s]), flush=True)
                ok = [s for s in SHIFTS if ev[s]["missed"] <= ref["missed"] and ev[s]["leak05"] <= ref["leak05"]]
                if ok:   # at v14's miss rate and partial-ghost weight or better: how close to the GT, how fine
                    s = max(ok, key=lambda k: ev[k]["sim"])
                    print(">> MATCHED " + fmt(f"v15 s{step} {s:+.2f}", ev[s]), flush=True)
                torch.save(net.state_dict(), os.path.join(HERE, a.out, f"pix_s{step}.pt"))
                hist.append(dict(step=step, ev={str(k): v for k, v in ev.items()}))
                json.dump(dict(args=vars(a), hist=hist), open(os.path.join(HERE, a.out, "metrics.json"), "w"), indent=1)
            if step >= a.steps:
                break
    print("done", flush=True)


if __name__ == "__main__":
    main()
