"""Fine-tune on the MERGED image (needs datasets generated with gen_dataset.py --full).

    python train_merge.py --init runs/v2/model.pt --train data/train_full --val data/val_full --out runs/v3

The per-pixel labels say what a sample *is*; this loss says what it *does*.
merged = sum_k R_k num_k / sum_k R_k den_k is differentiable in R, so the net
is trained directly on reconstruction error vs the clean scene:

    L_merge = E|sqrt(M_nn) - sqrt(GT)| / E|sqrt(M_r0) - sqrt(GT)|      (1.0 == analytic mask)
    L       = w_merge * L_merge + w_lab * label BCE (regulariser, as in train.py)

sqrt domain so shadows count; normalising by the analytic merge makes every
burst weigh the same regardless of noise level or content.
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
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn import merge, scenarios                                   # noqa: E402
from rnn.analytic import analytic_r0, random_params, NOMINAL       # noqa: E402
from rnn.common import warp_by_flow, logit                         # noqa: E402
from rnn.features import compute_features                          # noqa: E402
from rnn.model import RobustNet                                    # noqa: E402
from train import loss_fn, CLIP                                    # noqa: E402

EPS = 1e-4


class Bursts(Dataset):
    def __init__(self, root, train=True):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as z:
            b = {k: z[k] for k in z.files}
        rng = np.random.default_rng(None if self.train else idx)
        grey = torch.from_numpy(b["grey"].astype(np.float32))
        flows = torch.from_numpy(b["flows"].astype(np.float32))
        N = grey.shape[0]
        a, bb, tile = float(b["noise_a"]), float(b["noise_b"]), int(b["tile"])
        p = random_params(rng) if self.train else NOMINAL
        Ir = grey[0:1].expand(N - 1, -1, -1).contiguous()
        Ic, fl = grey[1:], flows[1:]
        with torch.no_grad():
            r0 = analytic_r0(Ir, warp_by_flow(Ic, fl), a, bb, p)
            feats, _ = compute_features(Ir, Ic, fl, r0, a, bb, CLIP, tile)
            raw = torch.from_numpy(b["raw"].astype(np.float32))
            S = int(b["out_scale"])
            num, den = merge.contributions(raw, flows, S, float(b["sigma_k"]))
        lab = torch.from_numpy(np.stack([b["geo"][1:], b["util"][1:], b["vis"][1:], b["err"][1:]], 1).astype(np.float32))
        gt = torch.from_numpy(b["gt"].astype(np.float32))
        return dict(feats=feats, r0=r0, lab=lab, num=num, den=den, gt=gt, S=S,
                    sid=scenarios.NAMES.index(str(b["scenario"])))


def merge_err(R, d):
    """R [N-1,Hg,Wg] for comparison frames -> mean |sqrt M - sqrt GT| (8 px border cropped)."""
    Rall = torch.cat([torch.ones_like(R[:1]), R], 0)
    Ho, Wo = d["num"].shape[-2:]
    M = merge.combine(d["num"], d["den"], merge.upsample_weights(Rall, Ho, Wo, d["S"]))
    e = ((M.clamp_min(0) + EPS).sqrt() - (d["gt"] + EPS).sqrt()).abs()
    return e[:, 8:-8, 8:-8].mean()


def burst_loss(net, d, args):
    R, delta = net(d["feats"], d["r0"])
    with torch.no_grad():
        e0 = merge_err(d["r0"], d)
    lm = merge_err(R, d) / e0.clamp_min(1e-8)
    ll = loss_fn(logit(d["r0"]) + delta, d["r0"], d["lab"], args.w_geo, args.w_util, args.emph, args.util_sharp)
    return args.w_merge * lm + args.w_lab * ll, lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_full")
    ap.add_argument("--val", default="data/val_full")
    ap.add_argument("--out", default="runs/v3")
    ap.add_argument("--init", default=None)
    ap.add_argument("--steps", type=int, default=2500, help="optimizer steps (each = --accum bursts)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_merge", type=float, default=1.0)
    ap.add_argument("--w_lab", type=float, default=0.1)
    ap.add_argument("--w_geo", type=float, default=0.4)
    ap.add_argument("--w_util", type=float, default=0.6)
    ap.add_argument("--emph", type=float, default=0.0)
    ap.add_argument("--util_sharp", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = RobustNet()
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu"))
    else:
        raise SystemExit("--init required (feature standardisation comes from train.py)")
    from train import _worker_init
    dl = DataLoader(Bursts(args.train, True), 1, shuffle=True, num_workers=args.workers, collate_fn=_first,
                    persistent_workers=True, worker_init_fn=_worker_init)
    dv = DataLoader(Bursts(args.val, False), 1, shuffle=False, num_workers=args.workers, collate_fn=_first,
                    persistent_workers=True, worker_init_fn=_worker_init)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    @torch.no_grad()
    def val():
        per = {n: [] for n in scenarios.NAMES}
        for d in dv:
            R, _ = net(d["feats"], d["r0"])
            per[scenarios.NAMES[d["sid"]]].append(float(merge_err(R, d) / merge_err(d["r0"], d)))
        s = {n: float(np.mean(v)) for n, v in per.items() if v}
        s["mean"] = float(np.mean(list(s.values())))
        return s

    v = val()
    print("val merge-error ratio (1.0 == analytic) at init:", json.dumps({k: round(x, 4) for k, x in v.items()}),
          flush=True)
    hist, best = [dict(step=0, **v)], v["mean"]
    torch.save(net.state_dict(), os.path.join(args.out, "model.pt"))
    step, t0, run, acc = 0, time.time(), None, 0
    opt.zero_grad()
    while step < args.steps:
        for d in dl:
            loss, lm = burst_loss(net, d, args)
            (loss / args.accum).backward()
            lmf = float(lm.detach())
            run = lmf if run is None else 0.98 * run + 0.02 * lmf
            acc += 1
            if acc % args.accum:
                continue
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step} merge-ratio(train,ema) {run:.4f} {time.time() - t0:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                v = val()
                hist.append(dict(step=step, **v))
                print(f"[val step {step}]", json.dumps({k: round(x, 4) for k, x in v.items()}), flush=True)
                torch.save(net.state_dict(), os.path.join(args.out, f"model_s{step}.pt"))
                if v["mean"] < best:
                    best = v["mean"]
                    torch.save(net.state_dict(), os.path.join(args.out, "model.pt"))
                json.dump(dict(args=vars(args), hist=hist), open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
            if step >= args.steps:
                break
    print("best val mean merge-ratio", best)


def _first(x):
    return x[0]


if __name__ == "__main__":
    main()
