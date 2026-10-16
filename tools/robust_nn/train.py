"""Train the robustness refinement net.

    python train.py --train data/train --val data/val --out runs/v1

Loss = w_geo * BCE(R, geo) + w_util * BCE(R, util), per pixel, weighted up where
the analytic mask disagrees with the targets (that is where refinement matters).
The analytic mask's parameters are re-drawn for every sample.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn.analytic import analytic_r0, random_params, NOMINAL      # noqa: E402
from rnn.common import warp_by_flow, logit                         # noqa: E402
from rnn.features import compute_features, NF, NAMES               # noqa: E402
from rnn.model import RobustNet                                    # noqa: E402
from rnn import scenarios                                          # noqa: E402

CLIP = 0.9


class Pairs(Dataset):
    def __init__(self, root, train=True, max_bursts=None):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))[:max_bursts]
        self.items = []
        for i, f in enumerate(self.files):
            with np.load(f) as z:
                n = z["corrupted"].shape[0]
            self.items += [(i, k) for k in range(1, n)]
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, k = self.items[idx]
        with np.load(self.files[i]) as z:
            b = {key: z[key] for key in z.files}
        rng = np.random.default_rng(None if self.train else idx)
        Ir = torch.from_numpy(b["grey"][0].astype(np.float32))[None]
        Ic = torch.from_numpy(b["grey"][k].astype(np.float32))[None]
        fl = torch.from_numpy(b["flows"][k].astype(np.float32))[None]
        lab = torch.from_numpy(np.stack([b["geo"][k], b["util"][k], b["vis"][k], b["err"][k]]).astype(np.float32))
        if self.train:                                  # dihedral augmentation (flow signs follow)
            if rng.random() < 0.5:
                Ir, Ic, lab = Ir.flip(-1), Ic.flip(-1), lab.flip(-1)
                fl = fl.flip(-1) * torch.tensor([-1.0, 1.0]).view(1, 2, 1, 1)
            if rng.random() < 0.5:
                Ir, Ic, lab = Ir.flip(-2), Ic.flip(-2), lab.flip(-2)
                fl = fl.flip(-2) * torch.tensor([1.0, -1.0]).view(1, 2, 1, 1)
            if rng.random() < 0.5:
                Ir, Ic, lab = Ir.transpose(-1, -2), Ic.transpose(-1, -2), lab.transpose(-1, -2)
                fl = fl.transpose(-1, -2).flip(1)
        a, bb, tile = float(b["noise_a"]), float(b["noise_b"]), int(b["tile"])
        p = random_params(rng) if self.train else NOMINAL
        with torch.no_grad():
            J = warp_by_flow(Ic, fl)
            r0 = analytic_r0(Ir, J, a, bb, p)
            feats, _ = compute_features(Ir, Ic, fl, r0, a, bb, CLIP, tile)
        sid = scenarios.NAMES.index(str(b["scenario"]))
        return feats[0], r0[0], lab, sid


def _worker_init(_):
    torch.set_num_threads(1)


def loss_fn(z, r0, lab, w_geo, w_util, emph, util_sharp=1.0):
    geo, util = lab[:, 0], lab[:, 1]
    # util is P(sample improves the merge). As a merge WEIGHT, "clearly beneficial" should be ~1,
    # not the 0.6-0.8 a calibrated probability of a small noise-reduction gain sits at.
    util = ((util - 0.5) * util_sharp + 0.5).clamp(0.0, 1.0)
    t_mix = w_geo * geo + w_util * util
    w = 1.0 + emph * (t_mix - r0).abs()
    l = w_geo * F.binary_cross_entropy_with_logits(z, geo, reduction="none") \
        + w_util * F.binary_cross_entropy_with_logits(z, util, reduction="none")
    return (w * l).mean()


@torch.no_grad()
def evaluate(net, dl, args):
    acc = {}
    keys = ["loss_nn", "loss_r0", "keep_nn", "keep_r0", "leak_nn", "leak_r0", "occl_nn", "occl_r0"]
    per = {n: {k: [0.0, 0.0] for k in keys} for n in scenarios.NAMES + ["all"]}

    def add(sname, key, s, c):
        for n in (sname, "all"):
            per[n][key][0] += float(s)
            per[n][key][1] += float(c)

    for feats, r0, lab, sid in dl:
        R, d = net(feats, r0)
        z = logit(r0) + d
        z0 = logit(r0)
        geo, util, vis, err = lab[:, 0], lab[:, 1], lab[:, 2], lab[:, 3]
        keep = (vis > 0.9) & (err < 0.3) & (geo > 0.9)       # A: correct correspondence
        leak = (vis > 0.5) & (geo < 0.1)                      # B: wrong, visible
        occl = vis < 0.1                                      # occluded / out of frame
        for j in range(feats.shape[0]):
            n = scenarios.NAMES[int(sid[j])]
            add(n, "loss_nn", loss_fn(z[j:j + 1], r0[j:j + 1], lab[j:j + 1], args.w_geo, args.w_util, 0.0, args.util_sharp), 1)
            add(n, "loss_r0", loss_fn(z0[j:j + 1], r0[j:j + 1], lab[j:j + 1], args.w_geo, args.w_util, 0.0, args.util_sharp), 1)
            for key, m in (("keep", keep[j]), ("leak", leak[j]), ("occl", occl[j])):
                add(n, key + "_nn", R[j][m].sum(), m.sum())
                add(n, key + "_r0", r0[j][m].sum(), m.sum())
    return {n: {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in d.items()} for n, d in per.items()}


def fmt(m):
    a = m["all"]
    return (f"loss nn {a['loss_nn']:.4f} r0 {a['loss_r0']:.4f} | keep(A) nn {a['keep_nn']:.3f} r0 {a['keep_r0']:.3f}"
            f" | leak(B) nn {a['leak_nn']:.3f} r0 {a['leak_r0']:.3f} | occl nn {a['occl_nn']:.3f} r0 {a['occl_r0']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--val", default="data/val")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w_geo", type=float, default=0.4)
    ap.add_argument("--w_util", type=float, default=0.6)
    ap.add_argument("--emph", type=float, default=2.0)
    ap.add_argument("--util_sharp", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)

    tr = Pairs(args.train, True)
    va = Pairs(args.val, False)
    print(f"train pairs {len(tr)}  val pairs {len(va)}", flush=True)
    dl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=True, worker_init_fn=_worker_init)
    dv = DataLoader(va, 16, shuffle=False, num_workers=args.workers, persistent_workers=True,
                    worker_init_fn=_worker_init)

    net = RobustNet()
    # feature standardisation from a sample of training data (folded into layer 1 at export)
    fs = []
    for i, (f, *_rest) in enumerate(dl):
        fs.append(f.permute(1, 0, 2, 3).reshape(NF, -1))
        if i >= 10:
            break
    fs = torch.cat(fs, 1)
    net.mu.copy_(fs.mean(1))
    net.sd.copy_(fs.std(1).clamp_min(1e-3))
    print("feature mean", [f"{n}={m:.2f}" for n, m in zip(NAMES, net.mu.tolist())])

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    m0 = evaluate(net, dv, args)
    print("step 0 (== analytic mask):", fmt(m0), flush=True)
    step, t0, run = 0, time.time(), 0.0
    hist = [dict(step=0, **m0["all"])]
    while step < args.steps:
        for feats, r0, lab, sid in dl:
            R, d = net(feats, r0)
            loss = loss_fn(logit(r0) + d, r0, lab, args.w_geo, args.w_util, args.emph, args.util_sharp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
            if step % 100 == 0:
                print(f"step {step} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(net, dv, args)
                print(f"[val step {step}]", fmt(m), flush=True)
                hist.append(dict(step=step, **m["all"]))
                torch.save(net.state_dict(), os.path.join(args.out, "model.pt"))
                json.dump(dict(args=vars(args), hist=hist, final=m), open(os.path.join(args.out, "metrics.json"), "w"),
                          indent=1)
            if step >= args.steps:
                break
    m = evaluate(net, dv, args)
    print("\nper-scenario (val, nominal analytic mask):")
    for n in scenarios.NAMES:
        x = m[n]
        print(f"  {n:12s} keep nn {x['keep_nn']:.3f}/r0 {x['keep_r0']:.3f}  leak nn {x['leak_nn']:.3f}/r0 {x['leak_r0']:.3f}"
              f"  occl nn {x['occl_nn']:.3f}/r0 {x['occl_r0']:.3f}  loss nn {x['loss_nn']:.4f}/r0 {x['loss_r0']:.4f}")


if __name__ == "__main__":
    main()
