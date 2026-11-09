"""Train the consensus refiner (core/cref.h) on cref_dataset crops. CPU, time-budgeted.
DO NOT RUN without the user's go.

usage: python train_cref.py --train data/train --val data/val --out runs/cref1 [--minutes 60] [--tau 2.0]

Target per pixel: T = exp(-0.5 (z / tau)^2), z = the ghost a sample would paint (|clean_k(p + used) -
clean_k(p + true)|) in one-frame noise std; z = 50 where the pixel is not visible in frame k. Weighted BCE
on the crop interior (8 px halo = receptive-field radius, so training sees exactly what banded inference
sees), harmful pixels (z > 4) weighted --w_bad. The network starts as the identity refine (zero head).
Selection metric on validation, per mask variant (0 = Alg.6/Wronski, 1 = GCam): catch = share of harmful
detailed pixels (z > 4) with R' < 0.5, false = share of harmless detailed pixels (z < 1) with R' < 0.5;
score = mean over variants of (catch - false)."""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from crops import load_dir, NF

HALO = 8
DIL = (1, 2, 4, 1)
CH = 8


class CrefNet(nn.Module):
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

    def forward(self, x):
        r = x[:, :1].clamp(0.02, 0.98)
        h = x
        for c in self.convs:
            h = F.relu(c(h))
        return torch.sigmoid(torch.logit(r) + self.head(h))


def batches(arr, bs, rng):
    idx = rng.permutation(len(arr))
    for i in range(0, len(idx) - bs + 1, bs):
        yield arr[np.sort(idx[i:i + bs])]


def to_tensors(b, tau, augment, rng):
    x = torch.from_numpy(b["x"].astype(np.float32))
    if augment:
        k = int(rng.integers(4))
        x = torch.rot90(x, k, (2, 3))
        if rng.random() < 0.5:
            x = torch.flip(x, (3,))
    feats, z = x[:, :NF], x[:, NF:NF + 1]
    var = torch.from_numpy(b["hd"][:, 0].astype(np.int64))
    return feats, z, var


def evaluate(model, val, tau):
    """Per mask variant v (0 Alg.6, 1 GCam) and region (all detailed px / HF-pattern px, feature 7 > 0.5):
    catch / false of the refined mask and of the input mask. res[v] = all, res[v + 2] = HF region.
    score = mean over variants of 0.5 (catch - false)_all + 0.5 (catch - false)_hf."""
    model.eval()
    stats = {k: np.zeros(6) for k in range(4)}   # caught, harmful, false, harmless (refined); caught0, false0
    with torch.no_grad():
        for i in range(0, len(val), 128):
            b = val[i:i + 128]
            feats, z, var = to_tensors(b, tau, False, None)
            out = model(feats)
            s = slice(HALO, -HALO)
            R1, R0, zz = out[:, 0, s, s], feats[:, 0, s, s], z[:, 0, s, s]
            det = feats[:, 4, s, s].abs() > 0.1
            hfm = feats[:, 7, s, s] > 0.5
            for v in (0, 1):
                m = (var == v)[:, None, None]
                for reg, rm in ((v, det), (v + 2, det & hfm)):
                    bad = m & (zz > 4) & rm
                    ok = m & (zz < 1) & rm
                    st = stats[reg]
                    st[0] += (bad & (R1 < 0.5)).sum().item(); st[1] += bad.sum().item()
                    st[2] += (ok & (R1 < 0.5)).sum().item(); st[3] += ok.sum().item()
                    st[4] += (bad & (R0 < 0.5)).sum().item(); st[5] += (ok & (R0 < 0.5)).sum().item()
    model.train()
    res = {}
    for v, st in stats.items():
        res[v] = dict(catch=st[0] / max(1, st[1]), false=st[2] / max(1, st[3]),
                      catch0=st[4] / max(1, st[1]), false0=st[5] / max(1, st[3]), n_bad=int(st[1]), n_ok=int(st[3]))
    score = np.mean([0.5 * (res[v]["catch"] - res[v]["false"]) + 0.5 * (res[v + 2]["catch"] - res[v + 2]["false"])
                     for v in (0, 1)])
    return score, res


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
    tr, va = load_dir(a.train), load_dir(a.val)
    print(f"train crops {len(tr)}  val crops {len(va)}", flush=True)
    model = CrefNet()
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    t0, step, best = time.time(), 0, -1e9
    budget = a.minutes * 60.0
    score, res = evaluate(model, va, a.tau)
    print(f"step 0 (identity) score {score:.4f} {json.dumps(res)}", flush=True)
    while time.time() - t0 < budget:
        for b in batches(tr, a.bs, rng):
            frac = (time.time() - t0) / budget
            if frac >= 1.0:
                break
            for g in opt.param_groups:
                g["lr"] = a.lr * 0.5 * (1.0 + np.cos(np.pi * frac))
            feats, z, _ = to_tensors(b, a.tau, True, rng)
            out = model(feats)
            T = torch.exp(-0.5 * (z / a.tau) ** 2)
            w = 1.0 + a.w_bad * (z > 4).float()
            s = slice(HALO, -HALO)
            loss = (F.binary_cross_entropy(out[:, :, s, s].clamp(1e-5, 1 - 1e-5), T[:, :, s, s], reduction="none")
                    * w[:, :, s, s]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % a.eval_every == 0:
                score, res = evaluate(model, va, a.tau)
                tag = ""
                if score > best:
                    best = score
                    torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
                    tag = " *"
                print(f"step {step} {time.time() - t0:.0f}s loss {loss.item():.4f} score {score:.4f}{tag} "
                      f"alg6 catch {res[0]['catch']:.3f} (was {res[0]['catch0']:.3f}) false {res[0]['false']:.3f} "
                      f"(was {res[0]['false0']:.3f}) | gcam catch {res[1]['catch']:.3f} (was {res[1]['catch0']:.3f}) "
                      f"false {res[1]['false']:.3f} (was {res[1]['false0']:.3f}) || HF: "
                      f"alg6 catch {res[2]['catch']:.3f} (was {res[2]['catch0']:.3f}) false {res[2]['false']:.3f} "
                      f"(was {res[2]['false0']:.3f}) | gcam catch {res[3]['catch']:.3f} (was {res[3]['catch0']:.3f}) "
                      f"false {res[3]['false']:.3f} (was {res[3]['false0']:.3f})", flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
    print("done", step, "steps, best score", best, flush=True)


if __name__ == "__main__":
    main()
