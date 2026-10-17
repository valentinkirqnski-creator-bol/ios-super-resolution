"""Target protocol on a REAL scene with the APP's own aligner and analytic mask.

Scene: one real DNG (APC_1186), decoded by the app's loader, each RGGB/BGGR quad
-> one camera-native RGB texel (no demosaic, noise averaged). Bursts: exact
camera motion over random crops of it (incl. large random shake), rendered in
the device's raw format (BGGR, pre-white-balanced, the DNG's own noise profile
scaled over ISO). Per sample, build/pipe/tp_real.exe runs the app's aligner
(real errors), optional deliberate tile corruption, the app's analytic mask and
the app's merge, with the used flow and with the exact flow.

Target T = app merge, exact flow, R = 1.  Output O = app merge, used flow, R.
Regions: BAD = the used flow is > 0.1 grey px from the truth (aligner error or
corruption, dilated 1 px); CLEAN = the rest. Loss as target_protocol.py.

    python real_protocol.py gen   --out data/rp_train --n 300 --seed0 0
    python real_protocol.py gen   --out data/rp_val   --n 40  --seed0 7000000 --val
    python real_protocol.py eval  --model runs/v4/model.pt
    python real_protocol.py train --init runs/v4/model.pt --out runs/v5
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
from rnn.burst import Burst                                       # noqa: E402
from rnn.features import compute_features                         # noqa: E402
from rnn.model import RobustNet                                   # noqa: E402
from target_protocol import combine, err_map, r_index, region_psnr, SCALE, CLIP   # noqa: E402

TP_REAL = os.path.join(HERE, "build", "pipe", "tp_real.exe")
REAL = os.path.join(HERE, "data", "real")
TRAIN_SCENES = ["APC_1186"]                             # --scenes overrides; val scene: a different capture
VAL_SCENE = "APC_1211"
SIZE = 384                                              # raw crop per burst: big enough for the app's pyramid


def load_scene(name):
    """The app loader's output for one DNG -> camera-native RGB per quad, plus its CFA/WB/noise."""
    h, w, c00, c01, c10, c11, a, b, w0, w1, w2, _pw = open(os.path.join(REAL, name + ".meta")).read().split()[:12]
    cfa = ((int(c00), int(c01)), (int(c10), int(c11)))
    wb = np.array([float(w0), float(w1), float(w2)], np.float32)
    r = np.fromfile(os.path.join(REAL, name + ".f32"), np.float32).reshape(int(h), int(w))
    r = r[:r.shape[0] // 2 * 2, :r.shape[1] // 2 * 2]
    q = {}
    for i in range(2):
        for j in range(2):
            q.setdefault(cfa[i][j], []).append(r[i::2, j::2])
    rgb = np.stack([np.mean(q[c], 0) / wb[c] for c in range(3)], -1)
    return np.clip(rgb, 0, None).astype(np.float32), cfa, wb, float(a), float(b)


_SCENE = {}


def _gen_one(args):
    seed, out, val, scenes = args
    global _SCENE
    path = os.path.join(out, f"rp{seed:08d}.npz")
    if os.path.exists(path):
        return
    torch.set_num_threads(1)
    name_s = VAL_SCENE if val else scenes[seed % len(scenes)]
    if name_s not in _SCENE:
        _SCENE[name_s] = load_scene(name_s)
    sc_rgb, cfa, wb, alpha0, beta0 = _SCENE[name_s]
    rng = np.random.default_rng(seed)
    name = ["tremor", "tremor", "rotation", "zoom"][seed % 4]
    sc = scenarios.sample(rng, name)
    sc.update(n_objects=0, obj_speed=0.0, blur_prob=0.0, exposure_jitter=0.0, n_frames=int(rng.integers(4, 6)),
              exposure=float(rng.uniform(0.5, 1.6)))
    if rng.random() < 0.35:                                  # big random hand shake
        sc.update(tremor=float(rng.uniform(3.0, 8.0)))
    if name == "rotation":
        sc.update(rot_deg=float(rng.uniform(0.2, 2.0)))
    if rng.random() < 0.3:
        sc.update(persp=float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-5)))))
    k = float(np.exp(rng.uniform(0, np.log(30))))            # ISO multiplier on the DNG's own profile
    sc.update(noise_a=alpha0 * k, noise_b=beta0 * k * k)
    b = Burst(sc, rng, Hr=SIZE, Wr=SIZE, bg_rgb=sc_rgb, cfa=cfa, prewhite=wb)
    d = b.render(out_scale=1, seed=int(rng.integers(1 << 31)))
    np.savez(path, raw=d["raw"].numpy().astype(np.float32), gt_flow=d["gt_flow"].numpy().astype(np.float32),
             alpha=np.float32(alpha0 * k), beta=np.float32(beta0 * k * k), scenario=name,
             cfa=np.array(cfa, np.int32), wb=wb)


def cmd_gen(a):
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers) as p:
        jobs = [(s, a.out, a.val, a.scenes) for s in range(a.seed0, a.seed0 + a.n)]
        for i, _ in enumerate(p.imap_unordered(_gen_one, jobs)):
            if i % 50 == 0:
                print(f"{i}/{a.n} {time.time() - t0:.0f}s", flush=True)


def run_tp_real(raw, gt, alpha, beta, seed, corrupt, cfa, wb):
    N, H, W = raw.shape
    Hg, Wg = H // 2, W // 2
    Ho, Wo = int(H * SCALE), int(W * SCALE)
    with tempfile.TemporaryDirectory() as d:
        pi, po = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(pi, "wb") as fh:
            np.array([N, H, W, seed, int(corrupt), cfa[0][0], cfa[0][1], cfa[1][0], cfa[1][1]], np.int32).tofile(fh)
            np.array([SCALE, alpha, beta, wb[0], wb[1], wb[2], 1.0], np.float32).tofile(fh)
            raw.astype(np.float32).tofile(fh)
            gt.transpose(0, 2, 3, 1).astype(np.float32).tofile(fh)
        subprocess.run([TP_REAL, pi, po], check=True)
        o = np.fromfile(po, np.float32)
    tile = int(o[-1:].view(np.int32)[0])
    o = torch.from_numpy(o[:-1].copy())
    P, G = Ho * Wo * 3, Hg * Wg
    Aref, Bref = o[:P].view(Ho, Wo, 3), o[P:2 * P].view(Ho, Wo, 3)
    m = o[2 * P:2 * P + 4 * (N - 1) * P].view(N - 1, 4, Ho, Wo, 3)
    g = o[2 * P + 4 * (N - 1) * P:]
    ref_grey = g[:G].view(Hg, Wg)
    per = g[G:].view(N - 1, 5, Hg, Wg)
    return dict(Aref=Aref, Bref=Bref, Au=m[:, 0], Bu=m[:, 1], Ag=m[:, 2], Bg=m[:, 3], ref_grey=ref_grey,
                grey=per[:, 0], r0=per[:, 1], flow=per[:, 2:4], corr=per[:, 4], tile=tile)


class RP(Dataset):
    def __init__(self, root, train):
        # root may list several folders separated by commas (several training scenes)
        self.files = sorted(f for r in root.split(",") for f in glob.glob(os.path.join(r, "*.npz")))
        self.train = train

    def __len__(self):
        return len(self.files) * (1 if self.train else 2)

    def __getitem__(self, idx):
        # The app pipeline runs in a subprocess; if one call fails (e.g. memory pressure), retry with a
        # different seed instead of letting the DataLoader wait forever.
        last = None
        for attempt in range(4):
            try:
                return self._item(idx, attempt)
            except Exception as e:   # noqa: BLE001
                last = e
                time.sleep(2.0)
        raise last

    def _item(self, idx, attempt):
        fi = idx % len(self.files)
        z = np.load(self.files[fi])
        raw, gt = z["raw"], z["gt_flow"]
        N, H, W = raw.shape
        rng = np.random.default_rng(None if self.train else 99 + idx)
        corrupt = bool(rng.random() < 0.5) if self.train else bool(idx >= len(self.files))
        seed = int(rng.integers(1, 1 << 30)) + attempt
        d = run_tp_real(raw, gt, float(z["alpha"]), float(z["beta"]), seed, corrupt, z["cfa"], z["wb"])
        Hg, Wg = H // 2, W // 2
        gy, gx = r_index(H, W, Hg, Wg)
        T = combine(d["Aref"], d["Bref"], d["Ag"], d["Bg"], torch.ones(N - 1, Hg, Wg), gy, gx)
        a_g, b_g = float(z["alpha"]) / 4.0, float(z["beta"]) / 4.0           # quad-mean grey noise
        Ir = d["ref_grey"][None].expand(N - 1, -1, -1).contiguous()
        fs = max(1, d["tile"] // 2)
        with torch.no_grad():
            feats, _ = compute_features(Ir, d["grey"], d["flow"], d["r0"], a_g, b_g, CLIP, fs)
        ferr = (d["flow"] - torch.from_numpy(gt[1:])).norm(dim=1)             # used vs exact, grey px
        badg = ((ferr > 0.1) | (d["corr"] > 0.5)).float().amax(0)
        badg = F.max_pool2d(badg[None, None], 3, 1, 1)[0, 0]
        bad = badg[gy][:, gx] > 0
        h = lambda t: t.half()   # halve inter-process transfer; maths stays fp32 (loss_of upcasts)
        return dict(feats=feats, r0=d["r0"], A=h(d["Au"]), B=h(d["Bu"]), Aref=h(d["Aref"]), Bref=h(d["Bref"]), T=h(T),
                    corr=bad, gy=gy, gx=gx, ferr=ferr, scen=str(z["scenario"]), corrupt=corrupt)


def _first(x):
    return x[0]


def _wi(_):
    torch.set_num_threads(1)


def loss_of(d, R, w_corr=1.0):
    for k in ("A", "B", "Aref", "Bref", "T"):
        d[k] = d[k].float()
    O = combine(d["Aref"], d["Bref"], d["A"], d["B"], R, d["gy"], d["gx"])
    E = err_map(O, d["T"])
    with torch.no_grad():
        z = torch.zeros_like(R)
        e_ref = err_map(combine(d["Aref"], d["Bref"], d["A"], d["B"], z, d["gy"], d["gx"]), d["T"])
        e_one = err_map(combine(d["Aref"], d["Bref"], d["A"], d["B"], 1 + z, d["gy"], d["gx"]), d["T"])
    clean, corr = ~d["corr"], d["corr"]
    lc = E[clean].mean() / e_ref[clean].mean().clamp_min(1e-6) if clean.any() else torch.zeros(())
    lr_ = E[corr].mean() / e_one[corr].mean().clamp_min(1e-6) if corr.any() else torch.zeros(())
    return lc + w_corr * lr_, float(lc.detach()), float(lr_.detach())


def cmd_eval(a):
    torch.set_num_threads(4)
    nets = {}
    for spec in a.model:
        net = RobustNet()
        net.load_state_dict(torch.load(spec, map_location="cpu"))
        net.eval()
        nets[os.path.basename(os.path.dirname(spec)) or spec] = net
    dl = DataLoader(RP(a.data, False), 1, shuffle=False, num_workers=a.workers, collate_fn=_first,
                    worker_init_fn=_wi)
    rows = {}
    for d in dl:
        ms = {"ones": torch.ones_like(d["r0"]), "app_r0": d["r0"],
              "oracle": (d["ferr"] <= 0.1).float()}
        with torch.no_grad():
            for k, net in nets.items():
                ms[k] = net(d["feats"], d["r0"])[0]
        tag = "corrupted" if d["corrupt"] else "aligner-only"
        for k, R in ms.items():
            _, lc, lr_ = loss_of(d, R)        # upcasts d's fp16 tensors in place
            O = combine(d["Aref"], d["Bref"], d["A"], d["B"], R, d["gy"], d["gx"])
            r = rows.setdefault((tag, k), {"clean": [], "bad": [], "lc": [], "lr": [], "w_bad": [], "w_good": []})
            r["clean"].append(region_psnr(O, d["T"], ~d["corr"]))
            r["bad"].append(region_psnr(O, d["T"], d["corr"]))
            r["lc"].append(lc)
            r["lr"].append(lr_)
            bad_g = d["ferr"] > 0.1
            if bad_g.any():
                r["w_bad"].append(float(R[bad_g].mean()))
            r["w_good"].append(float(R[~bad_g].mean()))
    for tag in ("aligner-only", "corrupted"):
        print(f"\n[{tag}]  PSNR vs perfect-alignment target (dB)   loss terms (clean: 0 = target; bad: 1 = merge all)")
        print(f"{'mask':10s} {'clean':>7s} {'bad':>7s} | {'L_clean':>8s} {'L_bad':>7s} | {'w bad px':>8s} {'w good px':>9s}")
        for (t, k), r in rows.items():
            if t != tag:
                continue
            print(f"{k:10s} {np.nanmean(r['clean']):7.2f} {np.nanmean(r['bad']):7.2f} | {np.mean(r['lc']):8.4f} "
                  f"{np.mean(r['lr']):7.4f} | {np.nanmean(r['w_bad']):8.3f} {np.mean(r['w_good']):9.3f}")


def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = RobustNet()
    net.load_state_dict(torch.load(a.init, map_location="cpu"))
    if a.zero_head:   # keep the learned feature layers, start exactly at the app's analytic mask
        torch.nn.init.zeros_(net.c3.weight)
        torch.nn.init.zeros_(net.c3.bias)
    dl = DataLoader(RP(a.train, True), 1, shuffle=True, num_workers=a.workers, collate_fn=_first,
                    persistent_workers=True, worker_init_fn=_wi, prefetch_factor=1, timeout=900)
    dv = DataLoader(RP(a.val, False), 1, shuffle=False, num_workers=a.workers, collate_fn=_first,
                    persistent_workers=False, worker_init_fn=_wi, prefetch_factor=1, timeout=900)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    @torch.no_grad()
    def val():
        c, r = [], []
        for d in dv:
            _, lc, lr_ = loss_of(d, net(d["feats"], d["r0"])[0])
            c.append(lc)
            r.append(lr_)
        return float(np.mean(c)), float(np.mean(r))

    vc, vr = val()
    print(f"init val: clean {vc:.4f} bad {vr:.4f}", flush=True)
    best = vc + vr
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    hist = [dict(step=0, clean=vc, bad=vr)]
    step, acc, t0, run = 0, 0, time.time(), None
    opt.zero_grad()
    while step < a.steps:
        for d in dl:
            loss, lc, lr_ = loss_of(d, net(d["feats"], d["r0"])[0], a.w_corr)
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
                print(f"step {step} clean {run[0]:.4f} bad {run[1]:.4f} {time.time() - t0:.0f}s", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                vc, vr = val()
                hist.append(dict(step=step, clean=vc, bad=vr))
                print(f"[val step {step}] clean {vc:.4f} bad {vr:.4f}", flush=True)
                torch.save(net.state_dict(), os.path.join(a.out, f"model_s{step}.pt"))
                if vc + vr < best:
                    best = vc + vr
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
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--seed0", type=int, default=0)
    g.add_argument("--val", action="store_true")
    g.add_argument("--scenes", nargs="+", default=TRAIN_SCENES)
    g.add_argument("--workers", type=int, default=10)
    e = sp.add_parser("eval")
    e.add_argument("--model", nargs="+", required=True)
    e.add_argument("--data", default=os.path.join(HERE, "data", "rp_val"))
    e.add_argument("--workers", type=int, default=8)
    t = sp.add_parser("train")
    t.add_argument("--init", required=True)
    t.add_argument("--train", default=os.path.join(HERE, "data", "rp_train"))
    t.add_argument("--val", default=os.path.join(HERE, "data", "rp_val"))
    t.add_argument("--out", default=os.path.join(HERE, "runs", "v5"))
    t.add_argument("--steps", type=int, default=1200)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=10)
    t.add_argument("--eval_every", type=int, default=200)
    t.add_argument("--zero_head", action="store_true")
    t.add_argument("--w_corr", type=float, default=1.0)
    a = ap.parse_args()
    {"gen": cmd_gen, "eval": cmd_eval, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
