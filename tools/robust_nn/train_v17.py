"""v17: v14's exact network (weights-only swap: same GPU kernel and memory) retrained from v14 on the old scenes plus
the user's new ones (APC_1294 / APC_1340 low light / APC_1342), against the merged-noise ground truth
(RNN_GT_NMERGE=8: a ghost counts if it would be visible after ~8 frames are merged). Scored on the old held-out
scene and on held-out bursts of the new scenes, always next to v14, at several logit shifts (weights-only).

    RNN_GT_NMERGE=8 python train_v17.py --out runs/v17 --steps 15000
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                                  # noqa: E402
import torch.nn as nn                                         # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader        # noqa: E402
import train_pix as tp                                        # noqa: E402
from real_protocol import load_net, geo_loss                  # noqa: E402


class Wrap(nn.Module):
    """RobustNet behind train_pix's (feats, r0, pix) interface; pixel channels unused."""
    def __init__(self, net):
        super().__init__()
        self.net, self.npix = net, 4

    def forward(self, f, r0, pix):
        return self.net(f, r0)


def report(tag, base, net, loaders):
    for name, dv in loaders.items():
        ev = tp.evaluate(net, dv)
        evb = tp.evaluate(base, dv)
        print(tp.fmt(f"[{name}] v14", evb[0.0]), flush=True)
        for s in tp.SHIFTS:
            print(tp.fmt(f"[{name}] {tag} {s:+.3f}", ev[s]), flush=True)
        ref = evb[0.0]
        ok = [s for s in tp.SHIFTS if ev[s]["missed"] <= ref["missed"] and ev[s]["leak05"] <= ref["leak05"]]
        if ok:
            s = min(ok, key=lambda k: ev[k]["over"])
            print(f">> MATCHED [{name}] " + tp.fmt(f"{tag} {s:+.3f}", ev[s]), flush=True)
        else:
            print(f">> [{name}] no shift reaches v14's misses and partial-ghost weight", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/v14/model.pt")
    ap.add_argument("--out", default="runs/v17")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--new_weight", type=int, default=2, help="repeat the new-scene cache this many times")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    base = Wrap(load_net(os.path.join(HERE, "runs", "v14", "model.pt"))).eval()
    net = Wrap(load_net(os.path.join(HERE, a.init)))
    tr = ConcatDataset([tp.Cache("data/c15_train", 96, True)] + [tp.Cache("data/c17_train", 96, True)] * a.new_weight)
    dl = DataLoader(tr, 2, shuffle=True, num_workers=a.workers, collate_fn=tp.cat, worker_init_fn=tp._wi,
                    drop_last=True, persistent_workers=True, prefetch_factor=2)
    vals = {k: DataLoader(tp.Cache(f"data/{k}"), 1, num_workers=2, collate_fn=tp.cat, worker_init_fn=tp._wi)
            for k in ("c15_val", "c17_val")}
    tp.SHIFTS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25]
    print(f"train samples {len(tr)} (new scenes x{a.new_weight}); GT n_merge {os.environ.get('RNN_GT_NMERGE')}",
          flush=True)
    net.eval()
    report("v17 s0", base, net, vals)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step, t0, run, hist = 0, time.time(), None, []
    net.train()
    while step < a.steps:
        for d in dl:
            R, _ = net(d["feats"], d["r0"], d["pix"])
            loss = geo_loss(R, d["geo"], d=d)
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
                report(f"v17 s{step}", base, net, vals)
                net.train()
                torch.save(net.net.state_dict(), os.path.join(HERE, a.out, f"model_s{step}.pt"))
            if step >= a.steps:
                break
    print("done", flush=True)


if __name__ == "__main__":
    main()
