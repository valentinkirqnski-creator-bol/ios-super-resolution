"""v39: the pixel-seeing PixNet (the app's "v15 pixel net" kernels: 14 features + 4 noise-normalised pixel channels +
the base logit through four dilated 3x3 convs, a 17x17 guide-px view) trained on the twin sets with the consistent
slip labels, over a frozen base (v21). Multi-set training like train_v17 (EXTRA_TRAIN / EXTRA_VAL), time budget.

    EXTRA_TRAIN=... EXTRA_VAL=... python train_pixnet.py --base runs/v21/model.pt --out runs/v39 --minutes 40
"""
import argparse
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
from rnn.pixnet import PixNet, NPIX                           # noqa: E402


class Wrap(nn.Module):
    """PixNet behind train_pix's (feats, r0, pix) interface; feats sliced to the base's inputs."""
    def __init__(self, net):
        super().__init__()
        self.net, self.npix = net, NPIX
        self.nf = net.base.c1.weight.shape[1]

    def forward(self, f, r0, pix):
        return self.net(f[:, :self.nf], r0, pix[:, :NPIX])


class BaseWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net, self.npix = net, NPIX
        self.nf = net.c1.weight.shape[1]

    def forward(self, f, r0, pix):
        return self.net(f[:, :self.nf], r0)


def report(tag, base, net, loaders, base_name):
    for name, dv in loaders.items():
        ev, evb = tp.evaluate(net, dv), tp.evaluate(base, dv)
        print(tp.fmt(f"[{name}] {base_name}", evb[0.0]), flush=True)
        for s in tp.SHIFTS:
            print(tp.fmt(f"[{name}] {tag} {s:+.3f}", ev[s]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/v21/model.pt")
    ap.add_argument("--out", default="runs/v39")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="v39")
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    base_net = load_net(os.path.join(HERE, a.base))
    net = Wrap(PixNet(load_net(os.path.join(HERE, a.base)), ch=a.ch, npix=NPIX))
    if a.init:
        net.net.load_state_dict(torch.load(os.path.join(HERE, a.init), map_location="cpu"))
    base = BaseWrap(base_net).eval()
    extra = [x for x in os.environ.get("EXTRA_TRAIN", "").split(",") if x]
    tr = ConcatDataset([tp.Cache("data/c15_train", 96, True)] + [tp.Cache("data/c17_train", 96, True)] * 2
                       + [tp.Cache(f"data/{x.split(':')[0]}", 96, True) for x in extra
                          for _ in range(int(x.split(":")[1]) if ":" in x else 1)])
    dl = DataLoader(tr, 2, shuffle=True, num_workers=a.workers, collate_fn=tp.cat, worker_init_fn=tp._wi,
                    drop_last=True, persistent_workers=True, prefetch_factor=2)
    vals = {k: DataLoader(tp.Cache(f"data/{k}"), 1, num_workers=2, collate_fn=tp.cat, worker_init_fn=tp._wi)
            for k in [x for x in os.environ.get("EXTRA_VAL", "").split(",") if x]}
    tp.SHIFTS = [-0.5, -0.25, 0.0, 0.25, 0.5]
    params = [p for p in net.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params)}; train samples {len(tr)}; GT n_merge {os.environ.get('RNN_GT_NMERGE')}",
          flush=True)
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step, t0, run = 0, time.time(), None
    net.train()
    net.net.base.eval()
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
            if a.minutes and step == 100:
                a.steps = min(a.steps, max(200, int(a.minutes * 60 / ((time.time() - t0) / 100))))
                print(f"time budget {a.minutes:.0f} min -> {a.steps} steps", flush=True)
            if step % 500 == 0:
                print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
            if step >= a.steps:
                break
    net.eval()
    torch.save(net.net.state_dict(), os.path.join(HERE, a.out, f"pix_s{step}.pt"))
    report(f"{a.tag} s{step}", base, net, vals, os.path.basename(os.path.dirname(a.base)))
    print("done", flush=True)


if __name__ == "__main__":
    main()
