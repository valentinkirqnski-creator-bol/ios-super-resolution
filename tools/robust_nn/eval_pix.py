"""Matched-operating-point check of v15 checkpoints against shipped v14 on the held-out cache: sweeps the logit
shift (weights-only) and reports, for each checkpoint, the setting with v14's miss rate or fewer and no more weight
on clearly misaligned pixels - then how close to the GT and how fine the mask is there.

    python eval_pix.py runs/v15/pix_s2500.pt [more checkpoints]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                         # noqa: E402
from torch.utils.data import DataLoader              # noqa: E402
import train_pix as tp                               # noqa: E402
from real_protocol import load_net                   # noqa: E402
from rnn.pixnet import PixNet                        # noqa: E402

tp.SHIFTS = [float(x) for x in os.environ.get("SHIFTS", "-2.5,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0").split(",")]


def main():
    torch.set_num_threads(2)
    dv = DataLoader(tp.Cache("data/c15_val"), 1, shuffle=False, num_workers=1, collate_fn=tp.cat, worker_init_fn=tp._wi)
    base = load_net(os.path.join(HERE, "runs/v14/model.pt"))
    first = True
    for ck in sys.argv[1:]:
        sd = torch.load(os.path.join(HERE, ck), map_location="cpu")
        npix = sd["p1.weight"].shape[1] - base.c1.weight.shape[1] - 1     # 4 pixel channels, 7 with consensus
        assert npix == 4 or tp.CONS, "consensus checkpoint: run with PIX_CONS=1"
        net = PixNet(base, ch=sd["p1.weight"].shape[0], npix=npix)
        net.load_state_dict(sd)
        net.eval()
        ev = tp.evaluate(net, dv, base_too=first)
        if first:
            ref = ev["v14"]
            print(tp.fmt("v14 (shipped)", ref), flush=True)
            first = False
        for s in tp.SHIFTS:
            print(tp.fmt(f"{os.path.basename(ck)[4:-3]} {s:+.2f}", ev[s]), flush=True)
        ok = [s for s in tp.SHIFTS if ev[s]["missed"] <= ref["missed"] + 1e-9 and ev[s]["leak05"] <= ref["leak05"] + 1e-9]
        if ok:
            s = max(ok, key=lambda k: ev[k]["sim"])
            print(">> matched  " + tp.fmt(f"{os.path.basename(ck)[4:-3]} {s:+.2f}", ev[s]), flush=True)
        else:
            print(">> no shift reaches v14's misses and partial-ghost weight", flush=True)


if __name__ == "__main__":
    main()
