"""Perfect-alignment target protocol, on the APP's own merge.

1. A synthetic burst where perfect alignment exists (one planar scene, camera
   motion only: tremor / rotation / zoom / perspective; no occlusion, no blur,
   no exposure jitter), with its exact per-pixel flow.
2. TARGET T = app merge (core/merge.cpp via build/merge_dump.exe) of those
   frames with the PERFECT flow and R = 1: the best output the pipeline can make.
3. Deliberately corrupt random tiles of the flow (known mask C), run the mask,
   merge with the corrupted flow and the mask's R -> output O.
4. Score O against T separately in
     clean region     (no corrupted tile nearby in any frame): must equal T.
                      R = 1 reproduces T exactly, so any loss here is the mask
                      throwing away good samples.
     corrupted region: must be as close to T as possible. The reference point is
                      the ORACLE mask R_n = 1 - C_n (drop exactly the corrupted
                      tiles).
   "Reaching the target" = clean error ~ 0 and corrupted error ~ oracle.

    python target_protocol.py gen   --out data/tp_train --n 400 --seed0 0
    python target_protocol.py gen   --out data/tp_val   --n 60  --seed0 5000000
    python target_protocol.py eval  --model runs/v2/model.pt
    python target_protocol.py train --init runs/v3/model.pt --out runs/v4
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rnn import scenarios                                         # noqa: E402
from rnn.analytic import analytic_r0, random_params, NOMINAL      # noqa: E402
from rnn.burst import Burst                                       # noqa: E402
from rnn.common import warp_by_flow, bayer_to_grey                # noqa: E402
from rnn.features import compute_features                         # noqa: E402
from rnn.model import RobustNet                                   # noqa: E402

MERGE_DUMP = os.path.join(HERE, "build", "merge_dump.exe")
SCALE = 2.0
CLIP = 0.9
SCEN = ["tremor", "rotation", "zoom", "repetitive", "low_light", "saturated"]
EPS = 1e-4


# ---------------------------------------------------------------- 1. clean bursts
def _gen_one(args):
    seed, out, size = args
    path = os.path.join(out, f"tp{seed:08d}.npz")
    if os.path.exists(path):
        return
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    name = SCEN[seed % len(SCEN)]
    sc = scenarios.sample(rng, name)
    sc.update(n_objects=0, obj_speed=0.0, blur_prob=0.0, exposure_jitter=0.0, n_frames=int(rng.integers(4, 8)))
    if rng.random() < 0.3:
        sc.update(persp=float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-5)))))
    b = Burst(sc, rng, Hr=size, Wr=size)
    d = b.render(out_scale=1, seed=int(rng.integers(1 << 31)))
    np.savez(path, raw=d["raw"].numpy().astype(np.float32), gt_flow=d["gt_flow"].numpy().astype(np.float32),
             vis=d["vis"].numpy().astype(np.float16), noise_a=np.float32(d["noise_a"] / 4.0),
             noise_b=np.float32(d["noise_b"] / 4.0), scenario=name)


def cmd_gen(a):
    os.makedirs(a.out, exist_ok=True)
    seeds = [(s, a.out, a.size) for s in range(a.seed0, a.seed0 + a.n)]
    t0 = time.time()
    with Pool(a.workers) as p:
        for i, _ in enumerate(p.imap_unordered(_gen_one, seeds)):
            if i % 50 == 0:
                print(f"{i}/{a.n} {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 2/3. corruption + app merge
def corrupt(gt_flow, rng, tile):
    """gt_flow [N,2,Hg,Wg]. Returns corrupted flow and mask C [N,Hg,Wg] (1 = corrupted tile)."""
    N, _, Hg, Wg = gt_flow.shape
    th, tw = (Hg + tile - 1) // tile, (Wg + tile - 1) // tile
    off = np.zeros((N, 2, th, tw), np.float32)
    C = np.zeros((N, th, tw), np.float32)
    for n in range(1, N):
        if rng.random() < 0.25:
            continue                                              # some frames entirely clean
        if rng.random() < 0.3:                                    # a coherent block (moving-object-like)
            h, w = rng.integers(1, max(2, th // 2) + 1), rng.integers(1, max(2, tw // 2) + 1)
            y, x = rng.integers(0, th - h + 1), rng.integers(0, tw - w + 1)
            m = np.zeros((th, tw), bool)
            m[y:y + h, x:x + w] = True
            mag = np.exp(rng.uniform(np.log(0.15), np.log(6.0)))
            ang = rng.uniform(0, 2 * np.pi)
            off[n, 0][m], off[n, 1][m] = mag * np.cos(ang), mag * np.sin(ang)
        else:                                                     # scattered tiles, each its own error
            m = rng.random((th, tw)) < rng.uniform(0.05, 0.4)
            k = int(m.sum())
            mag = np.exp(rng.uniform(np.log(0.15), np.log(6.0), k))
            ang = rng.uniform(0, 2 * np.pi, k)
            off[n, 0][m], off[n, 1][m] = mag * np.cos(ang), mag * np.sin(ang)
        C[n][m] = 1.0
    up = lambda t: torch.from_numpy(t).repeat_interleave(tile, -2).repeat_interleave(tile, -1)[..., :Hg, :Wg]
    return gt_flow + up(off), up(C)


def app_merge(raw, flow_variants):
    """raw [N,H,W]; flow_variants list of [N,2,Hg,Wg]. Returns A_ref,B_ref [Ho,Wo,3] and A,B [V,N-1,Ho,Wo,3]."""
    N, H, W = raw.shape
    V = len(flow_variants)
    Ho, Wo = int(H * SCALE), int(W * SCALE)
    with tempfile.TemporaryDirectory() as d:
        pi, po = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(pi, "wb") as fh:
            np.array([N, H, W, V], np.int32).tofile(fh)
            np.array([SCALE], np.float32).tofile(fh)
            raw.numpy().astype(np.float32).tofile(fh)
            for fl in flow_variants:
                fl.permute(0, 2, 3, 1).numpy().astype(np.float32).tofile(fh)
        subprocess.run([MERGE_DUMP, pi, po], check=True)
        o = torch.from_numpy(np.fromfile(po, np.float32))
    plane = Ho * Wo * 3
    Aref, Bref = o[:plane].view(Ho, Wo, 3), o[plane:2 * plane].view(Ho, Wo, 3)
    rest = o[2 * plane:].view(V, N - 1, 2, Ho, Wo, 3)
    return Aref, Bref, rest[:, :, 0], rest[:, :, 1]


def r_index(H, W, Hg, Wg):
    """Output pixel -> grey pixel whose R the app merge reads (nearest, floor(x+0.5) like the C++)."""
    Ho, Wo = int(H * SCALE), int(W * SCALE)
    iy = torch.floor(torch.arange(Ho, dtype=torch.float32) / SCALE + 0.5).long().clamp(0, H - 1)
    ix = torch.floor(torch.arange(Wo, dtype=torch.float32) / SCALE + 0.5).long().clamp(0, W - 1)
    return (iy // 2).clamp(max=Hg - 1), (ix // 2).clamp(max=Wg - 1)


def combine(Aref, Bref, A, B, R, gy, gx):
    """R [N-1,Hg,Wg] -> merged [Ho,Wo,3]."""
    Ro = R[:, gy][:, :, gx][..., None]
    return (Aref + (Ro * A).sum(0)) / (Bref + (Ro * B).sum(0)).clamp_min(1e-12)


def err_map(O, T):
    return ((O.clamp_min(0) + EPS).sqrt() - (T.clamp_min(0) + EPS).sqrt()).abs().mean(-1)


class TP(Dataset):
    def __init__(self, root, train):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx])
        rng = np.random.default_rng(None if self.train else 12345 + idx)
        raw = torch.from_numpy(z["raw"])
        gt = torch.from_numpy(z["gt_flow"])
        N, H, W = raw.shape
        Hg, Wg = H // 2, W // 2
        tile = int(rng.choice([8, 16]))
        fl, C = corrupt(gt, rng, tile)
        Aref, Bref, A, B = app_merge(raw, [gt, fl])
        gy, gx = r_index(H, W, Hg, Wg)
        T = combine(Aref, Bref, A[0], B[0], torch.ones(N - 1, Hg, Wg), gy, gx)
        grey = bayer_to_grey(raw)
        a, b = float(z["noise_a"]), float(z["noise_b"])
        p = random_params(rng) if self.train else NOMINAL
        Ir = grey[0:1].expand(N - 1, -1, -1).contiguous()
        with torch.no_grad():
            r0 = analytic_r0(Ir, warp_by_flow(grey[1:], fl[1:]), a, b, p)
            feats, _ = compute_features(Ir, grey[1:], fl[1:], r0, a, b, CLIP, tile)
        anyc = F.max_pool2d(C[1:].amax(0)[None, None], 3, 1, 1)[0, 0]       # dilate by 1 grey px
        corr = anyc[gy][:, gx] > 0
        return dict(feats=feats, r0=r0, A=A[1], B=B[1], Aref=Aref, Bref=Bref, T=T, C=C[1:], corr=corr,
                    gy=gy, gx=gx, scen=str(z["scenario"]))


def _first(x):
    return x[0]


def _wi(_):
    torch.set_num_threads(1)


def masks(d, net, sharp=(0.3, 0.7)):
    with torch.no_grad():
        R, _ = net(d["feats"], d["r0"])
    lo, hi = sharp
    return {"ones": torch.ones_like(R), "r0": d["r0"], "nn": R,
            "nn_sharp": ((R - lo) / (hi - lo)).clamp(0, 1), "oracle": 1.0 - d["C"],
            "ref_only": torch.zeros_like(R)}


def region_psnr(O, T, m):
    if m.sum() == 0:
        return float("nan")
    mse = ((O - T) ** 2).mean(-1)[m].mean().clamp_min(1e-14)
    return float(10 * torch.log10(1.0 / mse))


# ---------------------------------------------------------------- 4. evaluation
def cmd_eval(a):
    torch.set_num_threads(4)
    net = RobustNet()
    net.load_state_dict(torch.load(a.model, map_location="cpu"))
    net.eval()
    dl = DataLoader(TP(a.data, False), 1, shuffle=False, num_workers=a.workers, collate_fn=_first,
                    worker_init_fn=_wi)
    keys = ["ones", "r0", "nn", "nn_sharp", "oracle", "ref_only"]
    acc = {k: {"clean": [], "corr": [], "leak": [], "keep": []} for k in keys}
    per = {}
    for d in dl:
        ms = masks(d, net, (a.lo, a.hi))
        clean = ~d["corr"]
        Cm = d["C"] > 0.5
        for k in keys:
            O = combine(d["Aref"], d["Bref"], d["A"], d["B"], ms[k], d["gy"], d["gx"])
            pc, pr = region_psnr(O, d["T"], clean), region_psnr(O, d["T"], d["corr"])
            acc[k]["clean"].append(pc)
            acc[k]["corr"].append(pr)
            if Cm.any():
                acc[k]["leak"].append(float(ms[k][Cm].mean()))
            acc[k]["keep"].append(float(ms[k][~Cm].mean()))
            per.setdefault(d["scen"], {}).setdefault(k, []).append((pc, pr))
    print(f"vs the perfect-alignment target T (PSNR dB; higher = closer). {len(acc['nn']['clean'])} bursts")
    print(f"{'mask':10s} {'clean':>7s} {'corrupt':>8s} {'w on corrupted':>15s} {'w on clean':>11s}")
    for k in keys:
        x = acc[k]
        print(f"{k:10s} {np.nanmean(x['clean']):7.2f} {np.nanmean(x['corr']):8.2f} {np.nanmean(x['leak']):15.3f} "
              f"{np.nanmean(x['keep']):11.3f}")
    print("\nper scenario  (clean / corrupt):")
    for s in SCEN:
        if s in per:
            print(f"  {s:11s} " + "  ".join(f"{k} {np.nanmean([v[0] for v in per[s][k]]):.1f}/"
                                             f"{np.nanmean([v[1] for v in per[s][k]]):.1f}"
                                             for k in ("r0", "nn_sharp", "oracle")))
    json.dump({k: {m: float(np.nanmean(v)) for m, v in x.items()} for k, x in acc.items()},
              open(a.json, "w"), indent=1) if a.json else None


# ---------------------------------------------------------------- training against T
def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = RobustNet()
    net.load_state_dict(torch.load(a.init, map_location="cpu"))
    dl = DataLoader(TP(a.train, True), 1, shuffle=True, num_workers=a.workers, collate_fn=_first,
                    persistent_workers=True, worker_init_fn=_wi)
    dv = DataLoader(TP(a.val, False), 1, shuffle=False, num_workers=a.workers, collate_fn=_first,
                    persistent_workers=True, worker_init_fn=_wi)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    def loss_of(d, R):
        O = combine(d["Aref"], d["Bref"], d["A"], d["B"], R, d["gy"], d["gx"])
        E = err_map(O, d["T"])
        with torch.no_grad():
            z = torch.zeros_like(R)
            e_ref = err_map(combine(d["Aref"], d["Bref"], d["A"], d["B"], z, d["gy"], d["gx"]), d["T"])
            e_one = err_map(combine(d["Aref"], d["Bref"], d["A"], d["B"], 1 + z, d["gy"], d["gx"]), d["T"])
        clean, corr = ~d["corr"], d["corr"]
        lc = E[clean].mean() / e_ref[clean].mean().clamp_min(1e-6)        # 0 == target reached
        lr_ = E[corr].mean() / e_one[corr].mean().clamp_min(1e-6) if corr.any() else torch.zeros(())
        return lc + a.w_corr * lr_, float(lc), float(lr_)

    @torch.no_grad()
    def val():
        c, r = [], []
        for d in dv:
            R, _ = net(d["feats"], d["r0"])
            _, lc, lr_ = loss_of(d, R)
            c.append(lc)
            r.append(lr_)
        return float(np.mean(c)), float(np.mean(r))

    vc, vr = val()
    print(f"init val: clean {vc:.4f} corrupt {vr:.4f} (0 = target; corrupt is relative to merging everything)",
          flush=True)
    best = vc + a.w_corr * vr
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    hist = [dict(step=0, clean=vc, corr=vr)]
    step, acc, t0, run = 0, 0, time.time(), None
    opt.zero_grad()
    while step < a.steps:
        for d in dl:
            R, _ = net(d["feats"], d["r0"])
            loss, lc, lr_ = loss_of(d, R)
            (loss / a.accum).backward()
            run = (lc, lr_) if run is None else (0.98 * run[0] + 0.02 * lc, 0.98 * run[1] + 0.02 * lr_)
            acc += 1
            if acc % a.accum:
                continue
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step} clean {run[0]:.4f} corrupt {run[1]:.4f} {time.time() - t0:.0f}s", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                vc, vr = val()
                hist.append(dict(step=step, clean=vc, corr=vr))
                print(f"[val step {step}] clean {vc:.4f} corrupt {vr:.4f}", flush=True)
                torch.save(net.state_dict(), os.path.join(a.out, f"model_s{step}.pt"))
                if vc + a.w_corr * vr < best:
                    best = vc + a.w_corr * vr
                    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
                json.dump(dict(args=vars(a), hist=hist), open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
            if step >= a.steps:
                break
    print("best", best)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=400)
    g.add_argument("--seed0", type=int, default=0)
    g.add_argument("--size", type=int, default=256)
    g.add_argument("--workers", type=int, default=10)
    e = sp.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--data", default=os.path.join(HERE, "data", "tp_val"))
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--lo", type=float, default=0.3)
    e.add_argument("--hi", type=float, default=0.7)
    e.add_argument("--json", default=None)
    t = sp.add_parser("train")
    t.add_argument("--init", required=True)
    t.add_argument("--train", default=os.path.join(HERE, "data", "tp_train"))
    t.add_argument("--val", default=os.path.join(HERE, "data", "tp_val"))
    t.add_argument("--out", default=os.path.join(HERE, "runs", "v4"))
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--w_corr", type=float, default=1.0)
    t.add_argument("--workers", type=int, default=10)
    t.add_argument("--eval_every", type=int, default=250)
    a = ap.parse_args()
    {"gen": cmd_gen, "eval": cmd_eval, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
