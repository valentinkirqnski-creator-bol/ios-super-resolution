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

    def set_ts(self, ts):
        self._ts = ts

    def forward(self, f, r0, pix):
        f = f[:, :self.net.c1.weight.shape[1]]                        # a 14-input base ignores extra channels
        ts = getattr(self, "_ts", None)
        if ts is None or not hasattr(self.net, "tile_model"):
            return self.net(f, r0)
        vals = sorted(set(int(v) for v in ts.tolist()))
        if len(vals) == 1:
            return self.net(f, r0, vals[0])
        R, D = torch.zeros_like(r0), torch.zeros_like(r0)             # a batch mixing tile sizes: per size
        for v in vals:
            m = ts == v
            R[m], D[m] = self.net(f[m], r0[m], v)
        return R, D


def report(tag, base, net, loaders, base_name="v14"):
    for name, dv in loaders.items():
        ev = tp.evaluate(net, dv)
        evb = tp.evaluate(base, dv)
        print(tp.fmt(f"[{name}] {base_name}", evb[0.0]), flush=True)
        for s in tp.SHIFTS:
            print(tp.fmt(f"[{name}] {tag} {s:+.3f}", ev[s]), flush=True)
        ref = evb[0.0]
        ok = [s for s in tp.SHIFTS if ev[s]["missed"] <= ref["missed"] and ev[s]["leak05"] <= ref["leak05"]]
        if ok:
            s = min(ok, key=lambda k: ev[k]["over"])
            print(f">> MATCHED [{name}] " + tp.fmt(f"{tag} {s:+.3f}", ev[s]), flush=True)
        else:
            print(f">> [{name}] no shift reaches {base_name}'s misses and partial-ghost weight", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/v14/model.pt")
    ap.add_argument("--out", default="runs/v17")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--new_weight", type=int, default=2, help="repeat the new-scene cache this many times")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--base", default="runs/v14/model.pt", help="shipped model to compare against (cutoffs 0.20/0.90)")
    ap.add_argument("--tag", default="v17")
    ap.add_argument("--widen", action="store_true", help="v25: start from --init with extra (zero-weight) inputs")
    ap.add_argument("--ctx", type=int, default=0, help="v28: add the wide-view branch (this many channels) to --init")
    ap.add_argument("--minutes", type=float, default=0, help="v28: train for this long (steps set from the speed)")
    ap.add_argument("--tile", type=int, default=0, help="v30: tile-aligned wide view (this many channels) on --init's base")
    ap.add_argument("--skip_s0", action="store_true", help="v28: no step-0 report (the net starts as --base)")
    a = ap.parse_args()
    hold = os.path.join(HERE, "HOLD_TRAINING")   # the user starts training explicitly: remove this file to train
    if os.path.exists(hold):
        print(f"training held ({hold} exists) - not started", flush=True)
        return
    os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    base = Wrap(load_net(os.path.join(HERE, a.base))).eval()
    bname = os.path.basename(os.path.dirname(a.base))
    net = None if a.widen else Wrap(load_net(os.path.join(HERE, a.init)))
    if a.ctx:   # v28: --init's weights + a zero-output wide-view branch: starts exactly as --init
        from rnn.model import RobustNetCtx
        sd = torch.load(os.path.join(HERE, a.init), map_location="cpu")
        cn = RobustNetCtx(nf=sd["c1.weight"].shape[1], hid=sd["c1.weight"].shape[0], ctx=a.ctx)
        missing = cn.load_state_dict(sd, strict=False).missing_keys
        assert all(k.startswith("k") for k in missing), missing
        net = Wrap(cn)
    if a.tile:   # v30: --init's base layers + a zero-output tile-aligned wide view: starts as --init's base net
        from rnn.model import RobustNetTile
        sd = torch.load(os.path.join(HERE, a.init), map_location="cpu")
        sd = {k: v for k, v in sd.items() if k.split(".")[0] in ("mu", "sd", "c1", "c2", "c3")}
        tn = RobustNetTile(nf=sd["c1.weight"].shape[1], hid=sd["c1.weight"].shape[0], ctx=a.tile)
        missing = tn.load_state_dict(sd, strict=False).missing_keys
        assert all(k.startswith("k") or k == "tile_model" for k in missing), missing
        net = Wrap(tn)
    extra = [x for x in os.environ.get("EXTRA_TRAIN", "").split(",") if x]   # v19: e.g. "c19_train:2"
    tr = ConcatDataset([tp.Cache("data/c15_train", 96, True)] + [tp.Cache("data/c17_train", 96, True)] * a.new_weight
                       + [tp.Cache(f"data/{x.split(':')[0]}", 96, True) for x in extra
                          for _ in range(int(x.split(":")[1]) if ":" in x else 1)])
    dl = DataLoader(tr, 2, shuffle=True, num_workers=a.workers, collate_fn=tp.cat, worker_init_fn=tp._wi,
                    drop_last=True, persistent_workers=True, prefetch_factor=2)
    if a.widen:   # new input channels inferred from the data, zero weights: starts exactly as --init
        from real_protocol import widen
        net = Wrap(widen(os.path.join(HERE, a.init), 0, dl))
    vals = {k: DataLoader(tp.Cache(f"data/{k}"), 1, num_workers=2, collate_fn=tp.cat, worker_init_fn=tp._wi)
            for k in ["c15_val", "c17_val"] + [x for x in os.environ.get("EXTRA_VAL", "").split(",") if x]}
    tp.SHIFTS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25]
    print(f"train samples {len(tr)} (new scenes x{a.new_weight}); GT n_merge {os.environ.get('RNN_GT_NMERGE')}",
          flush=True)
    net.eval()
    if not a.skip_s0:
        report(f"{a.tag} s0", base, net, vals, bname)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step, t0, run, hist = 0, time.time(), None, []
    net.train()
    while step < a.steps:
        for d in dl:
            net.set_ts(d.get("ts"))
            R, _ = net(d["feats"], d["r0"], d["pix"])
            loss = geo_loss(R, d["geo"], d=d)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run = float(loss.detach()) if run is None else 0.99 * run + 0.01 * float(loss.detach())
            if a.minutes and step == 100:   # the cosine schedule reads a.steps, so it anneals over the real length
                a.steps = min(a.steps, max(200, int(a.minutes * 60 / ((time.time() - t0) / 100))))
                print(f"time budget {a.minutes:.0f} min -> {a.steps} steps", flush=True)
            if step % 500 == 0:
                print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                net.eval()
                report(f"{a.tag} s{step}", base, net, vals, bname)
                net.train()
                torch.save(net.net.state_dict(), os.path.join(HERE, a.out, f"model_s{step}.pt"))
            if step >= a.steps:
                break
    print("done", flush=True)


if __name__ == "__main__":
    main()
