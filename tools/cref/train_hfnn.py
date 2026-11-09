"""Train the HF-artifact NN (core/hfnn.h) on the cref v2 crops. CPU, time-budgeted.
DO NOT RUN without the user's go.

usage: python train_hfnn.py --train data/train --val data/val --out runs/hfnn1 [--minutes 30]

The crops hold the 8 cref inputs; the HF NN uses 6 of them (no multi-frame consensus):
  cref 0 R, 1 comp-ref, 4 hp(ref), 5 hp(comp), 6 flow disagreement, 7 HF variance loss.
Target as cref: T = exp(-0.5 (z / tau)^2), weighted BCE (harmful z > 4 x w_bad), but ONLY
where the app applies the network: per-pixel weight t = smoothstep(gate - 0.1, gate + 0.1,
HF loss) (the inference blend), on the crop interior (7 px halo). Validation reports, on
HF-pattern pixels (HF loss > 0.5) with detail, per mask variant (0 Alg.6, 1 GCam): catch
(harmful with R' < 0.5) and false (harmless z < 1 with R' < 0.5) of the BLENDED output vs
the input mask; score = mean over variants of (catch - false)."""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from crops import load_dir, NF

SEL = [0, 1, 4, 5, 6, 7]       # cref feature indices -> hfnn inputs 0..5
HALO = 7
DIL = (1, 2, 4)
CH = 8


class HfnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        cin = len(SEL)
        self.convs = nn.ModuleList()
        for d in DIL:
            self.convs.append(nn.Conv2d(cin, CH, 3, padding=d, dilation=d, padding_mode="replicate"))
            cin = CH
        self.head = nn.Conv2d(CH, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):            # x: [B, 6, H, W] -> network mask R_nn (before the HF blend)
        r = x[:, :1].clamp(0.02, 0.98)
        h = x
        for c in self.convs:
            h = F.relu(c(h))
        return torch.sigmoid(torch.logit(r) + self.head(h))


def blend_w(hf, gate):
    t = ((hf - (gate - 0.1)) / 0.2).clamp(0, 1)
    return t * t * (3 - 2 * t)


def to_tensors(b, augment, rng):
    x = torch.from_numpy(b["x"].astype(np.float32))
    if augment:
        x = torch.rot90(x, int(rng.integers(4)), (2, 3))
        if rng.random() < 0.5:
            x = torch.flip(x, (3,))
    return x[:, SEL], x[:, NF:NF + 1], torch.from_numpy(b["hd"][:, 0].astype(np.int64))


def evaluate(model, val, gate):
    model.eval()
    st = {v: np.zeros(6) for v in (0, 1)}
    with torch.no_grad():
        for i in range(0, len(val), 128):
            feats, z, var = to_tensors(val[i:i + 128], False, None)
            s = slice(HALO, -HALO)
            rn = model(feats)
            R0, hf = feats[:, :1], feats[:, 5:6]
            R1 = R0 + blend_w(hf, gate) * (rn - R0)
            R1, R0, zz, hf = R1[:, 0, s, s], R0[:, 0, s, s], z[:, 0, s, s], hf[:, 0, s, s]
            det = feats[:, 2, s, s].abs() > 0.1
            reg = det & (hf > 0.5)
            for v in (0, 1):
                m = (var == v)[:, None, None] & reg
                bad, ok = m & (zz > 4), m & (zz < 1)
                a = st[v]
                a[0] += (bad & (R1 < 0.5)).sum().item(); a[1] += bad.sum().item()
                a[2] += (ok & (R1 < 0.5)).sum().item(); a[3] += ok.sum().item()
                a[4] += (bad & (R0 < 0.5)).sum().item(); a[5] += (ok & (R0 < 0.5)).sum().item()
    model.train()
    res = {v: dict(catch=a[0] / max(1, a[1]), false=a[2] / max(1, a[3]),
                   catch0=a[4] / max(1, a[1]), false0=a[5] / max(1, a[3])) for v, a in st.items()}
    return float(np.mean([res[v]["catch"] - res[v]["false"] for v in res])), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--w_bad", type=float, default=4.0)
    ap.add_argument("--gate", type=float, default=0.35)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_every", type=int, default=400)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    os.makedirs(a.out, exist_ok=True)
    tr, va = load_dir(a.train), load_dir(a.val)
    # keep crops that contain HF-pattern pixels (the only place the network acts)
    keep = (tr["x"][:, 7].astype(np.float32) > a.gate - 0.1).reshape(len(tr), -1).mean(1) > 0.02
    tr = tr[keep]
    print(f"train crops with HF area {len(tr)}  val crops {len(va)}", flush=True)
    model = HfnnNet()
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    t0, step, best = time.time(), 0, -1e9
    budget = a.minutes * 60.0
    score, res = evaluate(model, va, a.gate)
    print(f"step 0 (identity) score {score:.4f} alg6 catch {res[0]['catch']:.3f} false {res[0]['false']:.3f} | "
          f"gcam catch {res[1]['catch']:.3f} false {res[1]['false']:.3f}", flush=True)
    while time.time() - t0 < budget:
        idx = rng.permutation(len(tr))
        for i in range(0, len(idx) - a.bs + 1, a.bs):
            frac = (time.time() - t0) / budget
            if frac >= 1.0:
                break
            for g in opt.param_groups:
                g["lr"] = a.lr * 0.5 * (1.0 + np.cos(np.pi * frac))
            feats, z, _ = to_tensors(tr[np.sort(idx[i:i + a.bs])], True, rng)
            out = model(feats)
            T = torch.exp(-0.5 * (z / a.tau) ** 2)
            w = (1.0 + a.w_bad * (z > 4).float()) * blend_w(feats[:, 5:6], a.gate)
            s = slice(HALO, -HALO)
            bce = F.binary_cross_entropy(out[:, :, s, s].clamp(1e-5, 1 - 1e-5), T[:, :, s, s], reduction="none")
            loss = (bce * w[:, :, s, s]).sum() / w[:, :, s, s].sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % a.eval_every == 0:
                score, res = evaluate(model, va, a.gate)
                tag = ""
                if score > best:
                    best = score
                    torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
                    tag = " *"
                print(f"step {step} {time.time() - t0:.0f}s loss {loss.item():.4f} score {score:.4f}{tag} HF-pattern px: "
                      f"alg6 catch {res[0]['catch']:.3f} (was {res[0]['catch0']:.3f}) false {res[0]['false']:.3f} "
                      f"(was {res[0]['false0']:.3f}) | gcam catch {res[1]['catch']:.3f} (was {res[1]['catch0']:.3f}) "
                      f"false {res[1]['false']:.3f} (was {res[1]['false0']:.3f})", flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
    print("done", step, "steps, best score", best, flush=True)


if __name__ == "__main__":
    main()
