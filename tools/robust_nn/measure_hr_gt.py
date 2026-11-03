"""Does the guide-resolution ground truth miss visible misalignments? On the held-out caches, compare the target
v17 was trained on (guide res, 8-frame noise) with the full-raw-resolution target (hr_target.py), and score v17
(shipped: weights + cutoffs 0.20/0.90) against both.

    python measure_hr_gt.py [n_per_set]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["RNN_GT_NMERGE"] = "8"
import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402
import train_pix as tp                                 # noqa: E402
from hr_target import hr_target, burst_of              # noqa: E402
from real_protocol import load_net                     # noqa: E402


def main():
    torch.set_num_threads(4)
    nper = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    v14, v17 = load_net("runs/v14/model.pt"), load_net("runs/v17/model.pt")
    for vs in ("c17_val", "c15_val"):
        ds = tp.Cache(f"data/{vs}")
        idx = list(range(len(ds) // 2, len(ds)))[:nper // 2] + list(range(0, len(ds) // 2))[:nper - nper // 2]
        C = {}
        agree = np.zeros(4)
        for i in idx:
            d = ds[i]
            z = np.load(ds.files[i])
            src = tp._source(ds.files[i])
            sz = np.load(src)
            b = burst_of(src)
            t_hr = hr_target(b, torch.from_numpy(z["flow"]), torch.from_numpy(sz["gt_flow"][1:]),
                             4 * float(z["a"]), 4 * float(z["b"]))
            t_g = d["geo"]
            # how the two targets classify clear pixels
            agree += np.array([float(((t_g > .8) & (t_hr < .2)).sum()), float((t_g > .8).sum()),
                               float(((t_g < .2) | (t_hr < .2)).sum()), float((t_hr < .2).sum())])
            with torch.no_grad():
                R = {"v14": v14(d["feats"], d["r0"])[0], "v17": v17(d["feats"], d["r0"])[0]}
            for name, r in R.items():
                for gname, t in (("guide GT", t_g), ("full-res GT", t_hr)):
                    dd = dict(d, geo=t)
                    C[(name, gname)] = C.get((name, gname), 0) + tp.counts(r, dd, 0.2, 0.9)
            print(f"{vs} {i}: misaligned px guide GT {float((t_g < .2).float().mean()):.3f}  full-res GT "
                  f"{float((t_hr < .2).float().mean()):.3f}", flush=True)
        print(f"\n[{vs}] pixels the guide GT calls ALIGNED (>0.8) but full-res GT calls MISALIGNED (<0.2): "
              f"{100 * agree[0] / agree[1]:.2f}% of guide-aligned; full-res misaligned px are "
              f"{agree[3] / max(1, agree[2] - 0) * 100:.0f}% of the union")
        for k, v in C.items():
            print(tp.fmt(f"[{vs}] {k[0]} vs {k[1]}", tp.summary(v)), flush=True)
        print()


if __name__ == "__main__":
    main()
