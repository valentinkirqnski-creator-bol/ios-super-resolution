"""Compact v15: distil the v15b teacher (v14 + PixNet, at its chosen logit shift) into ONE small depthwise-separable
net (rnn/compact.py) that fits v14's GPU memory with less compute, keeping a ground-truth term so it stays anchored
to the real target. Scored like train_pix (matched against v14: misses <= v14's and partial-ghost weight <= v14's).

    python train_compact.py --out runs/v15c --steps 20000
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                                # noqa: E402
from torch.utils.data import DataLoader                     # noqa: E402
import train_pix as tp                                      # noqa: E402
from real_protocol import load_net, geo_loss                # noqa: E402
from rnn.pixnet import PixNet                               # noqa: E402
from rnn.compact import CompactNet                          # noqa: E402

# measured on the held-out cache (eval_pix.py), cutoffs 0.20/0.90
V14 = dict(sim=0.8903, bnd=0.283, bal=94.47, missed=5.00, over=6.05, small=8.50, rot=1.68, flat=6.63, leak=0.066,
           leak05=0.075)
TEACHER = dict(sim=0.8952, bnd=0.267, bal=94.65, missed=4.80, over=5.90, small=7.34, rot=1.36, flat=6.74, leak=0.061,
               leak05=0.075)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/v15b/pix_s15000.pt")
    ap.add_argument("--tshift", type=float, default=-0.75)
    ap.add_argument("--out", default="runs/v15c")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--dil", default="1,2,4,1")
    ap.add_argument("--w_gt", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--first3", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    base = load_net(os.path.join(HERE, "runs", "v14", "model.pt"))
    teacher = PixNet(base)
    teacher.load_state_dict(torch.load(os.path.join(HERE, a.teacher), map_location="cpu"))
    teacher.eval()
    net = CompactNet(base.mu, base.sd, ch=a.ch, dil=tuple(int(x) for x in a.dil.split(",")), first3=a.first3)
    print(f"student params {sum(p.numel() for p in net.parameters())}, ~{net.macs_per_pixel()} MAC/pixel "
          f"(v14 ~5.1k, v14+PixNet ~15k)", flush=True)
    print(tp.fmt("v14 (shipped)", V14), flush=True)
    print(tp.fmt("teacher v15b", TEACHER), flush=True)
    dl = DataLoader(tp.Cache("data/c15_train", 96, True), 2, shuffle=True, num_workers=a.workers, collate_fn=tp.cat,
                    worker_init_fn=tp._wi, drop_last=True, persistent_workers=True, prefetch_factor=2)
    dv = DataLoader(tp.Cache("data/c15_val"), 1, shuffle=False, num_workers=a.workers, collate_fn=tp.cat,
                    worker_init_fn=tp._wi)
    tp.SHIFTS = [-1.0, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, 0.0]
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    hist, step, t0, run = [], 0, time.time(), None
    net.train()
    while step < a.steps:
        for d in dl:
            with torch.no_grad():
                Rt = teacher(d["feats"], d["r0"], d["pix"][:, :4])[0].clamp(1e-6, 1 - 1e-6)
                Rt = torch.sigmoid(torch.log(Rt) - torch.log1p(-Rt) + a.tshift)
            Rs, _ = net(d["feats"], d["r0"], d["pix"])
            Rc = Rs.clamp(1e-6, 1 - 1e-6)
            kd = -(Rt * torch.log(Rc) + (1 - Rt) * torch.log(1 - Rc)).mean()
            loss = kd + a.w_gt * geo_loss(Rs, d["geo"], d=d)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run = float(loss.detach()) if run is None else 0.99 * run + 0.01 * float(loss.detach())
            if step % 500 == 0:
                print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                net.eval()
                ev = tp.evaluate(net, dv)
                net.train()
                for s in tp.SHIFTS:
                    print(tp.fmt(f"v15c s{step} {s:+.3f}", ev[s]), flush=True)
                ok = [s for s in tp.SHIFTS if ev[s]["missed"] <= V14["missed"] and ev[s]["leak05"] <= V14["leak05"]]
                if ok:
                    s = max(ok, key=lambda k: ev[k]["sim"])
                    print(">> MATCHED " + tp.fmt(f"v15c s{step} {s:+.3f}", ev[s]), flush=True)
                torch.save(net.state_dict(), os.path.join(HERE, a.out, f"compact_s{step}.pt"))
                hist.append(dict(step=step, ev={str(k): v for k, v in ev.items()}))
                json.dump(dict(args=vars(a), hist=hist), open(os.path.join(HERE, a.out, "metrics.json"), "w"), indent=1)
            if step >= a.steps:
                break
    print("done", flush=True)


if __name__ == "__main__":
    main()
