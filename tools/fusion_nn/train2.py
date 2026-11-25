"""Train the DBSR-style fusion network at 2x through the app's exact merge, on gen2.py data.

    python train2.py --data DIR/train --val DIR/val --out DIR/runs/r3 [--steps N --minutes M]

Per step: --batch windows (two per burst), all 8 frames. Window = 128x128 raw (3 MP grid);
the network runs on its 64x64 guide pixels plus an 8-pixel halo (as banded inference on the
phone does); the merge output is 256x256 on the HR grid; loss on the centre 224x224 against
the sharp HR target: Charbonnier in noise std + gradient (edge) term, both weighted up to
1 + gamma where a frame's flow is > 0.15..0.5 px off (true misalignment from the generator;
it weights the loss, the network never sees it). No mask labels. The schedule (cosine) is
fitted to --minutes: the step count is re-estimated after 30 steps.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import fusion as fu

RAW = 128                  # raw window
HALO = 8                   # guide px each side
G = RAW // 2 + 2 * HALO    # 80 guide px for the network
CROP = 16                  # HR px dropped at each window edge for the loss
_CACHE = {}


def window_item(path, oy, ox):
    if path not in _CACHE:
        _CACHE.clear()
        bu = fu.Burst(path)
        _CACHE[path] = (bu, fu.inputs_dbsr(bu))
    bu, (gin, grid, xd) = _CACHE[path]
    s = int(bu.scale)
    A, B = fu.merge_terms(bu, s * oy, s * ox, s * RAW, s * RAW)
    gy0, gx0 = oy // 2 - HALO, ox // 2 - HALO
    g = slice(gy0, gy0 + G), slice(gx0, gx0 + G)
    tgt = bu.target[s * oy:s * (oy + RAW), s * ox:s * (ox + RAW)].permute(2, 0, 1).contiguous()
    gh = bu.ghost[:, oy // 2:oy // 2 + RAW // 2, ox // 2:ox // 2 + RAW // 2]
    return dict(gin=gin, grid=grid[:, g[0], g[1]].contiguous(), xd=xd[:, :, g[0], g[1]].contiguous(), A=A, B=B,
                tgt=tgt, a=torch.tensor(bu.a), b=torch.tensor(bu.b), ghost=gh.contiguous())


class Windows(Dataset):
    def __init__(self, files, n):
        self.files, self.n = files, n
        rng = np.random.default_rng(0)
        self.perm = np.concatenate([rng.permutation(len(files)) for _ in range(n // (2 * len(files)) + 1)])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        p = self.files[self.perm[(i // 2) % len(self.perm)]]
        rng = np.random.default_rng(i * 7919 + 17)
        oy, ox = (int(v) & ~1 for v in rng.integers(2 * HALO, 256 - RAW - 2 * HALO + 1, 2))
        return window_item(p, oy, ox)


def worker_init(_):
    torch.set_num_threads(1)


def out_coords(scale):
    """guide coordinate (relative to the network window) of each output pixel of the window."""
    j = torch.arange(int(scale) * RAW, dtype=torch.float32)
    return (j / scale - 0.5) / 2 + HALO


def mis_map(ghost, gamma, scale):
    m = ((ghost.amax(1, keepdim=True) - 0.15) / 0.35).clamp(0, 1)
    return 1 + gamma * torch.nn.functional.interpolate(m, scale_factor=2 * scale, mode="nearest")


def forward(net, it, ref_cap, scale):
    lg = net(it["gin"], it["grid"], it["xd"])
    c = out_coords(scale)
    w = fu.weights_at(lg, c, c, ref_cap)
    return fu.fuse(it["A"], it["B"], w), w


def stats(out, it, scale):
    c = slice(CROP, -CROP)
    s = fu.noise_sigma(it["tgt"], it["a"][:, None, None, None], it["b"][:, None, None, None])
    e = ((out - it["tgt"]) / s).abs().mean(1, keepdim=True)[..., c, c]
    gm = torch.nn.functional.interpolate(it["ghost"].amax(1, keepdim=True), scale_factor=2 * scale, mode="nearest")[..., c, c]
    slip = ((gm > 0.15) & (gm <= 1.5)).float()
    return np.array([e.sum().item(), (e * slip).sum().item(), e.numel(), slip.sum().item()])


def build_val(files):
    items = []
    t0 = time.time()
    for p in files:
        o = (256 - RAW) // 2
        it = window_item(p, o, o)
        bu = _CACHE[p][0]
        s = int(bu.scale)
        if bu.R is not None:
            Ar, Br = fu.merge_terms(bu, s * o, s * o, s * RAW, s * RAW, R_analytic=True)
            it["out_analytic"] = fu.fuse(Ar, Br, torch.ones(bu.N, 1, 1, 1))
        it["out_all"] = fu.fuse(it["A"], it["B"], torch.ones(bu.N, 1, 1, 1))
        it["out_ref"] = it["A"][0] / it["B"][0].clamp(min=1e-12)
        items.append(it)
    print("val: %d windows cached in %.0fs" % (len(items), time.time() - t0), flush=True)
    return items


def evaluate(net, val, cfg, baselines=False):
    acc = {}
    net.eval()
    c = slice(CROP, -CROP)
    with torch.no_grad():
        for i in range(0, len(val), 6):
            ch = val[i:i + 6]
            it = {k: torch.stack([x[k] for x in ch]) for k in ch[0] if all(k in x for x in ch)}
            Bt = it["tgt"].shape[0]
            outs = {"nn": forward(net, it, cfg["ref_cap"], cfg["scale"])[0]}
            if baselines:
                outs.update({k[4:]: it[k] for k in ("out_analytic", "out_all", "out_ref") if k in it})
            wmap = mis_map(it["ghost"], cfg["gamma"], cfg["scale"])[..., c, c]
            for k, o in outs.items():
                l = fu.loss_fn(o[..., c, c], it["tgt"][..., c, c], it["a"], it["b"], cfg["w_grad"], wmap).item() * Bt
                a = acc.setdefault(k, np.zeros(5))
                a += np.array([l, *stats(o, it, cfg["scale"])])
    net.train()
    return {k: dict(loss=a[0] / len(val), err=a[1] / a[3], err_slip=a[2] / max(a[4], 1)) for k, a in acc.items()}


def fmt(r):
    return "  ".join("%s: loss %.4f err %.3f slip %.3f" % (k, v["loss"], v["err"], v["err_slip"]) for k, v in r.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=110, help="training time budget (schedule fitted to it)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ce", type=int, default=12)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--w_grad", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=400)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # no sleep while training
    files = sorted(f for d in a.data for f in glob.glob(os.path.join(d, "*.npz")))
    vfiles = sorted(f for d in a.val for f in glob.glob(os.path.join(d, "*.npz")))
    cfg = dict(gamma=a.gamma, w_grad=a.w_grad, ref_cap=True, scale=2.0)
    net = fu.FusionNetD(ce=a.ce, c=a.c)
    enc, dec = net.macs_per_px()
    print("train bursts %d, val %d | params %d | MAC/guide px: encoder %d + decoder %d | %s" % (
        len(files), len(vfiles), sum(p.numel() for p in net.parameters()), enc, dec, cfg), flush=True)
    val = build_val(vfiles)
    r0 = evaluate(net, val, cfg, baselines=True)
    print("step 0 | " + fmt(r0), flush=True)
    log = [dict(step=0, **r0)]

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    total = [100000]
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(s, total[0]) / total[0]))))
    dl = DataLoader(Windows(files, 100000 * a.batch), batch_size=a.batch, num_workers=a.workers,
                    worker_init_fn=worker_init, persistent_workers=True, prefetch_factor=4)
    c = slice(CROP, -CROP)
    t0 = time.time()
    tval = 0.0
    run, best, step = 0.0, 1e9, 0
    for step, it in enumerate(dl, 1):
        out, _ = forward(net, it, cfg["ref_cap"], cfg["scale"])
        wmap = mis_map(it["ghost"], cfg["gamma"], cfg["scale"])[..., c, c]
        loss = fu.loss_fn(out[..., c, c], it["tgt"][..., c, c], it["a"], it["b"], cfg["w_grad"], wmap)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        el = (time.time() - t0) / 60
        if step == 30:                       # fit the schedule to the time budget
            per = (time.time() - t0) / 30
            total[0] = max(200, int((a.minutes * 60 * 0.97) / (per * (1 + 0.02))))
            print("%.2fs/step -> %d steps in %.0f min" % (per, total[0], a.minutes), flush=True)
        if step % 50 == 0:
            print("step %d/%d  loss %.4f  lr %.2e  %.1f min" % (step, total[0], run, sched.get_last_lr()[0], el), flush=True)
        last = step >= total[0] or el >= a.minutes
        if step % a.val_every == 0 or last:
            tv = time.time()
            r = evaluate(net, val, cfg)
            tval += time.time() - tv
            print("step %d | %s" % (step, fmt(r)), flush=True)
            log.append(dict(step=step, minutes=el, **r))
            torch.save(net.state_dict(), os.path.join(a.out, "last.pt"))
            if r["nn"]["loss"] < best:
                best = r["nn"]["loss"]
                torch.save(net.state_dict(), os.path.join(a.out, "best.pt"))
            json.dump(dict(log=log, args=vars(a), cfg=cfg), open(os.path.join(a.out, "log.json"), "w"), indent=1)
        if last:
            break
    print("done: %d steps in %.1f min (validation %.1f min), best val loss %.4f" % (
        step, (time.time() - t0) / 60, tval / 60, best), flush=True)


if __name__ == "__main__":
    main()
