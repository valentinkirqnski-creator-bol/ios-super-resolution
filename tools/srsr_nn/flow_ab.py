"""A/B of SRSR's alignment on generated samples with exact ground truth.

    python flow_ab.py dirA dirB      (the same seeds generated with SRSR_OV=0 / 1)

Per sample and setting: the flow the merge reads (nearest flow tile at each quad
centre) against the exact correspondence at visible points, in grey (quad) px,
and the merge (analytic mask) against the noise-free target in noise std.
"""
import os
import sys

import numpy as np
import torch

import data as D
import srsr_merge as sm


def flow_err(z):
    S = sm.Sample(z)
    m = sm.meta(z)
    u = S.u
    qW, qH = u["quadW"], u["quadH"]
    cx, cy = m["crop"]
    gt = torch.from_numpy(z["gt_flow"].astype(np.float32))        # [N, 2, Hg, Wg]
    vis = torch.from_numpy(z["vis"].astype(np.float32) / 255)     # [N, Hg, Wg]
    qy, qx = torch.meshgrid(torch.arange(qH, dtype=torch.float32), torch.arange(qW, dtype=torch.float32), indexing="ij")
    uv = torch.stack([(qx.reshape(-1) + 0.5) / qW, (qy.reshape(-1) + 0.5) / qH], 1)
    # quad (qx, qy) = raw (crop + 2q .. +1), centre in the burst's grey grid at q + crop/2
    gx = (qx.reshape(-1) + 0.5 * cx).round().long().clamp(0, gt.shape[3] - 1)
    gy = (qy.reshape(-1) + 0.5 * cy).round().long().clamp(0, gt.shape[2] - 1)
    errs = []
    for f in range(1, S.n):
        fl = S.flow_at(f, uv) * torch.tensor([float(qW), float(qH)])
        g = gt[f][:, gy, gx].T
        v = vis[f][gy, gx] > 0.5
        errs.append((fl - g).norm(dim=1)[v])
    return torch.cat(errs).numpy()


def merge_err(z):
    t = sm.precompute(z)
    masks = torch.from_numpy(z["masks"].astype(np.float32) / 255)
    out = sm.merge(t, masks)
    tgt = torch.from_numpy(z["target"].astype(np.float32))
    m = sm.meta(z)
    a, b = D.frame_noise(m, 1)
    sig = D.noise_sigma(tgt.view(-1, 3), a[0], b[0]).view(tgt.shape)
    e = ((out - tgt) / sig).abs().mean(-1)
    c = slice(8, -8)
    return float(e[c, c].mean()), float(np.quantile(e[c, c].numpy(), 0.99)), float(masks.mean())


def main():
    da, db = sys.argv[1], sys.argv[2]
    names = sorted(f for f in os.listdir(da) if f.endswith(".npz") and os.path.exists(os.path.join(db, f)))
    tot = {k: [] for k in ("A", "B")}
    print("%-14s %-11s | %-44s | %-44s" % ("sample", "scenario", "A: flow err med/p90/>0.5px  merge e/p99/keep",
                                            "B: same"))
    for n in names:
        row = []
        for k, d in (("A", da), ("B", db)):
            z = np.load(os.path.join(d, n))
            fe = flow_err(z)
            me = merge_err(z)
            tot[k].append((fe, me))
            row.append("%.3f/%.3f/%5.1f%%  %.3f/%.2f/%.2f" % (np.median(fe), np.quantile(fe, 0.9), 100 * (fe > 0.5).mean(), *me))
        print("%-14s %-11s | %-44s | %-44s" % (n, sm.meta(z)["scenario"], *row))
    for k in ("A", "B"):
        fe = np.concatenate([x[0] for x in tot[k]])
        me = np.array([x[1] for x in tot[k]])
        print("%s all: flow err median %.3f  mean %.3f  p90 %.3f  >0.25px %.1f%%  >0.5px %.1f%%  >1px %.1f%% | merge err %.4f  p99 %.3f  keep %.3f" % (
            k, np.median(fe), fe.mean(), np.quantile(fe, 0.9), 100 * (fe > 0.25).mean(), 100 * (fe > 0.5).mean(),
            100 * (fe > 1).mean(), me[:, 0].mean(), me[:, 1].mean(), me[:, 2].mean()))


if __name__ == "__main__":
    torch.set_num_threads(6)
    main()
