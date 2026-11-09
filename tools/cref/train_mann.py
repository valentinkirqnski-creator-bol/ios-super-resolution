"""Train the misalignment-artifact NN (core/mann.h) on the mann crops. CPU, time-budgeted.
DO NOT RUN without the user's go.

usage: python train_mann.py --train data/mtrain --val data/mval --out runs/mann1 [--minutes 60]

Inputs: the 11 mann features (mcrops.NAMES). Target T = exp(-0.5 (z / tau)^2), weighted BCE (harmful z > 4
x w_bad), ONLY where the app applies the network: per-pixel weight = the app's gate g (exported per pixel by
mann_dataset.exe: smoothstep of the tile-flow irregularity M around the GCam motion threshold), on the crop interior
(8 px halo). Validation on gated pixels (g > 0.5), per mask variant (0 Wronski, 1 GCam), for the BLENDED output
R' = R + g (R_nn - R): catch = harmful (z > 4) with R' < 0.5, false = harmless (z < 1) detailed (|hp ref| > 0.1)
with R' < 0.5, on all gated px and on edge px (|hp ref| > 0.25); score = mean over variants of (catch - false) on
edge px + 0.5 x on all detail."""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mcrops import load_dir, NF, IZ, IG

HALO = 8
DIL = (1, 2, 4, 1)
CH = 8
S = 64


class MannNet(nn.Module):
    def __init__(self):
        super().__init__()
        cin = NF
        self.convs = nn.ModuleList()
        for d in DIL:
            self.convs.append(nn.Conv2d(cin, CH, 3, padding=d, dilation=d, padding_mode="replicate"))
            cin = CH
        self.head = nn.Conv2d(CH, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, strict=0.0):            # x: [B, 11, H, W] -> R_nn (before the gate blend)
        r = x[:, :1].clamp(0.02, 0.98)
        h = x
        for c in self.convs:
            h = F.relu(c(h))
        return torch.sigmoid(torch.logit(r) + self.head(h) - strict)


def to_tensors(b, augment, rng):
    x = torch.from_numpy(b["x"].astype(np.float32))
    if augment:
        k = int(rng.integers(4))
        x = torch.rot90(x, k, (2, 3))
        if rng.random() < 0.5:
            x = torch.flip(x, (3,))
    return x[:, :NF], x[:, IZ:IZ + 1], x[:, IG:IG + 1], torch.from_numpy(b["hd"][:, 0].astype(np.int64))


def evaluate(model, val, strict=0.0):
    model.eval()
    st = {(v, r): np.zeros(6) for v in (0, 1) for r in ("edge", "detail")}
    with torch.no_grad():
        for i in range(0, len(val), 128):
            feats, z, g, var = to_tensors(val[i:i + 128], False, None)
            s = slice(HALO, -HALO)
            rn = model(feats, strict)
            R0 = feats[:, :1]
            R1 = R0 + g * (rn - R0)
            R1, R0, zz, gg = R1[:, 0, s, s], R0[:, 0, s, s], z[:, 0, s, s], g[:, 0, s, s]
            hp = feats[:, 2, s, s].abs()
            for rname, reg in (("edge", hp > 0.25), ("detail", hp > 0.1)):
                for v in (0, 1):
                    m = (var == v)[:, None, None] & reg & (gg > 0.5)
                    bad, ok = m & (zz > 4), m & (zz < 1)
                    a = st[(v, rname)]
                    a[0] += (bad & (R1 < 0.5)).sum().item(); a[1] += bad.sum().item()
                    a[2] += (ok & (R1 < 0.5)).sum().item(); a[3] += ok.sum().item()
                    a[4] += (bad & (R0 < 0.5)).sum().item(); a[5] += (ok & (R0 < 0.5)).sum().item()
    model.train()
    res = {k: dict(catch=a[0] / max(1, a[1]), false=a[2] / max(1, a[3]),
                   catch0=a[4] / max(1, a[1]), false0=a[5] / max(1, a[3])) for k, a in st.items()}
    score = float(np.mean([res[(v, "edge")]["catch"] - res[(v, "edge")]["false"]
                           + 0.5 * (res[(v, "detail")]["catch"] - res[(v, "detail")]["false"]) for v in (0, 1)]))
    return score, res


def fmt(res):
    out = []
    for v, n in ((0, "wronski"), (1, "gcam")):
        for r in ("edge", "detail"):
            q = res[(v, r)]
            out.append(f"{n}/{r} {100*q['catch']:.0f}/{100*q['false']:.0f} (was {100*q['catch0']:.0f}/{100*q['false0']:.0f})")
    return " | ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--w_bad", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_every", type=int, default=400)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    os.makedirs(a.out, exist_ok=True)
    tr, va = load_dir(a.train, S), load_dir(a.val, S)
    keep = (tr["x"][:, IG].astype(np.float32) > 0.05).reshape(len(tr), -1).mean(1) > 0.02
    tr = tr[keep]
    print(f"train crops with gated area {len(tr)}  val crops {len(va)}", flush=True)
    model = MannNet()
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    t0, step, best = time.time(), 0, -1e9
    budget = a.minutes * 60.0
    score, res = evaluate(model, va)
    print(f"step 0 (identity) score {score:.4f} catch/false % on gated px: {fmt(res)}", flush=True)
    while time.time() - t0 < budget:
        idx = rng.permutation(len(tr))
        for i in range(0, len(idx) - a.bs + 1, a.bs):
            frac = (time.time() - t0) / budget
            if frac >= 1.0:
                break
            for gr in opt.param_groups:
                gr["lr"] = a.lr * 0.5 * (1.0 + np.cos(np.pi * frac))
            feats, z, g, _ = to_tensors(tr[np.sort(idx[i:i + a.bs])], True, rng)
            out = model(feats)
            T = torch.exp(-0.5 * (z / a.tau) ** 2)
            w = (1.0 + a.w_bad * (z > 4).float()) * g
            s = slice(HALO, -HALO)
            bce = F.binary_cross_entropy(out[:, :, s, s].clamp(1e-5, 1 - 1e-5), T[:, :, s, s], reduction="none")
            loss = (bce * w[:, :, s, s]).sum() / w[:, :, s, s].sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % a.eval_every == 0:
                score, res = evaluate(model, va)
                tag = ""
                if score > best:
                    best = score
                    torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
                    tag = " *"
                print(f"step {step} {time.time() - t0:.0f}s loss {loss.item():.4f} score {score:.4f}{tag} {fmt(res)}", flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
    print("done", step, "steps, best score", best, flush=True)


if __name__ == "__main__":
    main()
