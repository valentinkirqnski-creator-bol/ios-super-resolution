"""Train the fusion-weight network through the app's exact merge.

    python train.py --data DIR/train --val DIR/val --out runs/r1 [--steps 6000 --batch 8]

Per step: 8 bursts, one 160x160 output window each (80x80 guide pixels per frame for the
network), all 8 frames. Loss on the centre 128x128: Charbonnier in noise std + gradient term.
Validation (fixed centre windows of every validation burst) every --val_every steps, against
three baselines through the same merge: the app's analytic R, all frames equal, reference only.
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

WIN = 160
CFG = dict(gamma=4.0, w_grad=1.0, ref_cap=True, n_in=fu.N_IN)
CROP = 16          # loss on [CROP, WIN-CROP)


_CACHE = {}


def window_item(path, oy, ox):
    if path not in _CACHE:                       # one burst per worker: consecutive items share it
        _CACHE.clear()
        bu = fu.Burst(path)
        _CACHE[path] = (bu, fu.net_inputs(bu, CFG["n_in"]))
    bu, xin = _CACHE[path]
    A, B = fu.merge_terms(bu, oy, ox, WIN, WIN)
    x = xin[:, :, oy // 2:oy // 2 + WIN // 2, ox // 2:ox // 2 + WIN // 2]
    tgt = bu.target[oy:oy + WIN, ox:ox + WIN].permute(2, 0, 1).contiguous()
    gh = bu.ghost[:, oy // 2:oy // 2 + WIN // 2, ox // 2:ox // 2 + WIN // 2]
    return dict(A=A, B=B, x=x.contiguous(), tgt=tgt, a=torch.tensor(bu.a), b=torch.tensor(bu.b), ghost=gh.contiguous())


class Windows(Dataset):
    def __init__(self, files, n):
        self.files, self.n = files, n
        rng = np.random.default_rng(0)
        self.perm = np.concatenate([rng.permutation(len(files)) for _ in range(n // (2 * len(files)) + 1)])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # items 2k and 2k+1 (same batch, same worker) use the same burst, different windows
        p = self.files[self.perm[(i // 2) % len(self.perm)]]
        rng = np.random.default_rng(i * 7919 + 17)
        oy, ox = (int(v) & ~1 for v in rng.integers(8, 256 - WIN - 8 + 1, 2))
        return window_item(p, oy, ox)


def worker_init(_):
    torch.set_num_threads(1)


def ghost_px(gh):
    """[.., N-1, 80, 80] px -> output-res masks [.., 1, 160, 160]: any frame > 0.5 px off."""
    m = (gh.amax(-3, keepdim=True) > 0.5).float()
    return torch.nn.functional.interpolate(m, scale_factor=2, mode="nearest")


def stats(out, it):
    """mean |err| in noise std on all / ghost / clean pixels of the loss crop."""
    c = slice(CROP, WIN - CROP)
    s = fu.noise_sigma(it["tgt"], it["a"][:, None, None, None], it["b"][:, None, None, None])
    e = ((out - it["tgt"]) / s).abs().mean(1, keepdim=True)[..., c, c]
    g = ghost_px(it["ghost"])[..., c, c]
    return e.sum().item(), (e * g).sum().item(), (e * (1 - g)).sum().item(), e.numel(), g.sum().item()


def build_val(files):
    items = []
    t0 = time.time()
    for p in files:
        o = (256 - WIN) // 2
        it = window_item(p, o, o)
        bu = fu.Burst(p)
        Ar, Br = fu.merge_terms(bu, o, o, WIN, WIN, R_analytic=True)
        one = torch.ones(bu.N, 1, 1, 1)
        it["out_analytic"] = fu.fuse(Ar, Br, one)
        it["out_all"] = fu.fuse(it["A"], it["B"], one)
        it["out_ref"] = it["A"][0] / it["B"][0].clamp(min=1e-12)
        items.append(it)
    print("val: %d windows cached in %.0fs" % (len(items), time.time() - t0), flush=True)
    return items


def batched(items, k=10):
    for i in range(0, len(items), k):
        ch = items[i:i + k]
        yield {key: torch.stack([c[key] for c in ch]) for key in ch[0]}


def evaluate(net, val, baselines=False):
    acc = {}
    net.eval()
    with torch.no_grad():
        for it in batched(val):
            Bt, N = it["x"].shape[:2]
            lg = net(it["x"].flatten(0, 1)).view(Bt, N, 1, WIN // 2, WIN // 2)
            outs = {"nn": fu.fuse(it["A"], it["B"], fu.weights_from_logits(lg, CFG["ref_cap"]))}
            if baselines:
                outs.update(analytic=it["out_analytic"], all=it["out_all"], ref=it["out_ref"])
            for k, o in outs.items():
                c = slice(CROP, WIN - CROP)
                wmap = fu.mis_weight(it["ghost"], CFG["gamma"])[..., c, c]
                l = fu.loss_fn(o[..., c, c], it["tgt"][..., c, c], it["a"], it["b"], CFG["w_grad"], wmap).item() * Bt
                s = stats(o, it)
                a = acc.setdefault(k, np.zeros(6))
                a += np.array([l, *s[:3], s[3], s[4]])
    net.train()
    res = {}
    for k, a in acc.items():
        n = len(val)
        res[k] = dict(loss=a[0] / n, err=a[1] / a[4], err_ghost=a[2] / max(a[5], 1), err_clean=a[3] / max(a[4] - a[5], 1),
                      ghost_frac=a[5] / a[4])
    return res


def fmt(r):
    return "  ".join("%s: loss %.4f err %.3f ghost %.3f clean %.3f" % (k, v["loss"], v["err"], v["err_ghost"], v["err_clean"])
                     for k, v in r.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=4.0, help="extra loss weight where a frame is >0.5 px off")
    ap.add_argument("--w_grad", type=float, default=1.0)
    ap.add_argument("--no_ref_cap", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--max_minutes", type=float, default=240)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    CFG.update(gamma=a.gamma, w_grad=a.w_grad, ref_cap=not a.no_ref_cap)
    files = sorted(f for d in a.data for f in glob.glob(os.path.join(d, "*.npz")))
    vfiles = sorted(f for d in a.val for f in glob.glob(os.path.join(d, "*.npz")))
    if os.name == "nt":   # keep the PC awake while training (process-scoped, released on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    print("config", CFG, "c", a.c, flush=True)
    print("train bursts %d, val bursts %d" % (len(files), len(vfiles)), flush=True)

    net = fu.FusionNet(c=a.c)
    print("params %d" % sum(p.numel() for p in net.parameters()), flush=True)
    val = build_val(vfiles)
    r0 = evaluate(net, val, baselines=True)
    print("step 0 | " + fmt(r0), flush=True)
    log = [dict(step=0, **{k: v for k, v in r0.items()})]

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    dl = DataLoader(Windows(files, a.steps * a.batch), batch_size=a.batch, num_workers=a.workers,
                    worker_init_fn=worker_init, persistent_workers=True, prefetch_factor=4)
    t0 = time.time()
    run, tdata, tlast = 0.0, 0.0, time.time()
    c = slice(CROP, WIN - CROP)
    best = 1e9
    step = 0
    for step, it in enumerate(dl, 1):
        tdata += time.time() - tlast
        Bt, N = it["x"].shape[:2]
        logits = net(it["x"].flatten(0, 1)).view(Bt, N, 1, WIN // 2, WIN // 2)
        out = fu.fuse(it["A"], it["B"], fu.weights_from_logits(logits, CFG["ref_cap"]))
        wmap = fu.mis_weight(it["ghost"], CFG["gamma"])[..., c, c]
        loss = fu.loss_fn(out[..., c, c], it["tgt"][..., c, c], it["a"], it["b"], CFG["w_grad"], wmap)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        tlast = time.time()
        el = (time.time() - t0) / 60
        if step % 50 == 0:
            print("step %d  loss %.4f  lr %.2e  %.1f min  (%.2fs/step, data wait %.2fs)" % (
                step, run, sched.get_last_lr()[0], el, (time.time() - t0) / step, tdata / step), flush=True)
        last = step >= a.steps or el >= a.max_minutes
        if step % a.val_every == 0 or last:
            r = evaluate(net, val)
            print("step %d | %s" % (step, fmt(r)), flush=True)
            log.append(dict(step=step, minutes=el, **r))
            torch.save(net.state_dict(), os.path.join(a.out, "last.pt"))
            if r["nn"]["loss"] < best:
                best = r["nn"]["loss"]
                torch.save(net.state_dict(), os.path.join(a.out, "best.pt"))
            json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
        if last:
            break
    print("done: %d steps in %.1f min, best val loss %.4f" % (step, (time.time() - t0) / 60, best), flush=True)


if __name__ == "__main__":
    main()
