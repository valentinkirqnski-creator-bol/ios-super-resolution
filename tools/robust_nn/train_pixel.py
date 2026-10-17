"""v11 prototype: a small CNN on the ACTUAL full-resolution pixels (no hand-made summary features), trained
with sharp per-pixel labels "does including this sample make the app's merge worse?".

    python train_pixel.py --minutes 30 --out runs/v11

Inputs per comparison frame, at sensor resolution (7 channels):
    reference and comparison (gathered through its flow), each contrast-normalised by the local mean;
    their difference over the noise sigma; the flow's deviation from its 5x5-tile neighbourhood (raw px, x/y);
    its sub-pixel phase; logit of the app's analytic mask.
Net: pixel_unshuffle(2) (each 2x2 Bayer block -> channels, nothing averaged away) -> 3x3 conv x3 (one dilated)
     -> 1x1 -> delta;  R = sigmoid(logit(R0) + delta), head zero-initialised (starts as the analytic mask).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_protocol import RP, loss_of, _first, _wi                      # noqa: E402
from rnn.common import logit, win_mean                                   # noqa: E402
from rnn.features import gather_raw                                      # noqa: E402
from target_protocol import combine                                      # noqa: E402

CIN = 7


class PixelNet(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.c1 = nn.Conv2d(CIN * 4, hid, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(hid, hid, 3, padding=1, padding_mode="replicate")
        self.c3 = nn.Conv2d(hid, hid, 3, padding=2, dilation=2, padding_mode="replicate")
        self.head = nn.Conv2d(hid, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, r0):
        h = F.pixel_unshuffle(x, 2)
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h)) + h
        h = F.relu(self.c3(h)) + h
        d = self.head(h)[:, 0]
        return torch.sigmoid(logit(r0) + d), d


def pixel_inputs(raw, flow_g, r0, alpha, beta):
    """raw [N,H,W] pre-WB'd Bayer, flow_g [K,2,H/2,W/2] grey px, r0 [K,H/2,W/2] -> [K,7,H,W]."""
    N, H, W = raw.shape
    K = N - 1
    up = lambda t: t.repeat_interleave(2, -2).repeat_interleave(2, -1)[..., :H, :W]
    L0 = gather_raw(raw[0], torch.zeros(2, H, W))
    fr = up(2.0 * flow_g)                                                  # raw px
    Ls = torch.stack([gather_raw(raw[k + 1], fr[k]) for k in range(K)])
    m0 = win_mean(L0[None], 3)[0].clamp_min(1e-4)
    sig = (alpha * L0.clamp_min(0) + beta).clamp_min(1e-10).sqrt()
    a = (L0 / m0 - 1.0)[None].expand(K, -1, -1)
    b = Ls / m0[None] - 1.0
    dlt = (Ls - L0[None]) / sig[None]
    # flow deviation from its 5x5-tile neighbourhood (tile ~ 16 grey px): local mean over a wide window
    fmean = torch.stack([F.avg_pool2d(F.pad(flow_g[:, c:c + 1], (16, 16, 16, 16), mode="replicate"), 33, 1)[:, 0]
                         for c in range(2)], 1)
    dev = up(2.0 * (flow_g - fmean))
    phase = up((2.0 * flow_g - torch.floor(2.0 * flow_g + 0.5)).norm(dim=1))
    lr0 = up(0.25 * logit(r0))
    x = torch.stack([a, b, dlt.clamp(-20, 20) / 5.0, dev[:, 0], dev[:, 1], phase, lr0], 1)
    return x.float()


def sharp_labels(d, T):
    """Per guide pixel and frame: 0 = including this sample makes the merge worse (misaligned-harmful),
    1 = it helps, -1 = unclear (ignored). Other frames held at the analytic mask."""
    r0 = d["r0"]
    lab = torch.full_like(r0, -1.0)
    for f in range(r0.shape[0]):
        Rw, Ro = r0.clone(), r0.clone()
        Rw[f], Ro[f] = 1.0, 0.0
        ew = ((combine(d["Aref"], d["Bref"], d["A"], d["B"], Rw, d["gy"], d["gx"]) - T) ** 2).mean(-1)
        eo = ((combine(d["Aref"], d["Bref"], d["A"], d["B"], Ro, d["gy"], d["gx"]) - T) ** 2).mean(-1)
        rel = F.avg_pool2d((ew - eo)[None, None], 4)[0, 0] / (F.avg_pool2d(((ew + eo) / 2)[None, None], 4)[0, 0] + 1e-9)
        lab[f][rel > 0.10] = 0.0
        lab[f][rel < -0.02] = 1.0
    return lab


class PixDS(Dataset):
    def __init__(self, root, train):
        self.rp = RP(root, train)

    def __len__(self):
        return len(self.rp)

    def __getitem__(self, i):
        d = self.rp[i]
        z = np.load(self.rp.files[i % len(self.rp.files)])
        for k in ("A", "B", "Aref", "Bref", "T"):
            d[k] = d[k].float()
        with torch.no_grad():
            d["pix"] = pixel_inputs(torch.from_numpy(z["raw"]), d["flowg"], d["r0"], float(z["alpha"]), float(z["beta"]))
            d["lab"] = sharp_labels(d, d["T"])
        for k in ("A", "B", "Aref", "Bref", "T"):
            d[k] = d[k].half()
        return d


def bce_balanced(R, lab):
    pos, neg = lab == 1.0, lab == 0.0
    l = torch.zeros(())
    eps = 1e-6
    if pos.any():
        l = l - torch.log(R[pos].clamp_min(eps)).mean()
    if neg.any():
        l = l - torch.log((1 - R[neg]).clamp_min(eps)).mean()
    return l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/rp7_train,data/rp7_train_hdrp")
    ap.add_argument("--val", default="data/rp7_val")
    ap.add_argument("--out", default="runs/v11")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--w_merge", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = PixelNet()
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    dl = DataLoader(PixDS(a.train, True), 1, shuffle=True, num_workers=a.workers, collate_fn=_first,
                    persistent_workers=True, worker_init_fn=_wi, prefetch_factor=1, timeout=900)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    t0, step, acc, run = time.time(), 0, 0, None
    budget = a.minutes * 60
    opt.zero_grad()
    while time.time() - t0 < budget:
        for d in dl:
            R, _ = net(d["pix"], d["r0"])
            lb = bce_balanced(R, d["lab"])
            lm, lc, lr_ = loss_of(d, R, 2.0)
            loss = lb + a.w_merge * lm
            (loss / a.accum).backward()
            run = (float(lb), lc, lr_) if run is None else tuple(0.98 * r + 0.02 * v for r, v in zip(run, (float(lb), lc, lr_)))
            acc += 1
            if acc % a.accum:
                continue
            frac = min(1.0, (time.time() - t0) / budget)                 # cosine LR over the wall-clock budget
            for g in opt.param_groups:
                g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * frac))
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0:
                print(f"step {step} bce {run[0]:.4f} clean {run[1]:.4f} bad {run[2]:.4f} {time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 >= budget:
                break
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    print("trained", step, "steps", flush=True)


if __name__ == "__main__":
    main()
