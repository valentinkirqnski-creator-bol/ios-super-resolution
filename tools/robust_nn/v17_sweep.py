"""Pick v17's shipping setting: logit shift (folded into the last bias, weights-only) x exact-0/1 cutoffs, over
BOTH held-out sets (old scene c15_val, user's scenes c17_val) under the 8-frame ground truth. Lists the settings
that beat shipped v14 (0.20/0.90) on every measure on both sets, ranked by total misses + over-rejection.

    RNN_GT_NMERGE=8 python v17_sweep.py runs/v17/model_s10000.pt runs/v17/model_s12500.pt runs/v17/model_s15000.pt
"""
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                            # noqa: E402
import train_pix as tp                                  # noqa: E402
from real_protocol import load_net                      # noqa: E402

SHIFTS = [float(x) for x in os.environ.get("SHIFTS", "-0.25,0,0.125,0.25,0.375,0.5,0.75,1.0").split(",")]
CUTS = [tuple(float(y) for y in x.split(":")) for x in
        os.environ.get("CUTS", "0.05:0.95,0.1:0.95,0.15:0.95,0.2:0.9,0.25:0.9,0.3:0.9").split(",")]
KEYS = ["missed", "over", "small", "rot", "flat", "leak05"]       # lower is better


def main():
    assert os.environ.get("RNN_GT_NMERGE") == "8" or os.environ.get("RNN_GT_HR") == "1", (
        "run with RNN_GT_NMERGE=8 (guide-res target) or RNN_GT_HR=1 (full-res target)")
    torch.set_num_threads(4)
    v14 = load_net(os.path.join(HERE, os.environ.get("BASE", "runs/v14/model.pt")))   # the shipped model to beat
    nets = {os.path.basename(ck)[:-3]: load_net(os.path.join(HERE, ck)) for ck in sys.argv[1:]}
    S = {}
    for vs in os.environ.get("VALS", "c15_val,c17_val").split(","):
        ds = tp.Cache(f"data/{vs}")
        C = {}
        with torch.no_grad():
            for i in range(len(ds)):
                d = ds[i]
                C["v14"] = C.get("v14", 0) + tp.counts(v14(d["feats"], d["r0"])[0], d)
                for k, n in nets.items():
                    R = n(d["feats"], d["r0"])[0].clamp(1e-6, 1 - 1e-6)
                    lg = torch.log(R) - torch.log1p(-R)
                    for s, (lo, hi) in itertools.product(SHIFTS, CUTS):
                        C[(k, s, lo, hi)] = C.get((k, s, lo, hi), 0) + tp.counts(torch.sigmoid(lg + s), d, lo, hi)
        S[vs] = {k: tp.summary(v) for k, v in C.items()}
        print(tp.fmt(f"[{vs}] v14", S[vs]["v14"]), flush=True)
    cand = [k for k in S["c15_val"] if k != "v14"]
    dom = [k for k in cand if all(S[vs][k][m] <= S[vs]["v14"][m] + 1e-9 for vs in S for m in KEYS)]
    score = lambda k: sum(S[vs][k]["missed"] + S[vs][k]["over"] for vs in S)
    print(f"\nsettings beating v14 on every measure on both sets: {len(dom)}")
    for k in sorted(dom, key=score)[:10]:
        for vs in S:
            print(tp.fmt(f"[{vs}] {k[0]} {k[1]:+.3f} {k[2]:.2f}/{k[3]:.2f}", S[vs][k]))
        print()
    if not dom:
        print("closest (fewest measures worse than v14 over both sets):")
        worse = lambda k: sum(S[vs][k][m] > S[vs]["v14"][m] + 1e-9 for vs in S for m in KEYS)
        for k in sorted(cand, key=lambda k: (worse(k), score(k)))[:5]:
            for vs in S:
                print(tp.fmt(f"[{vs}] {k[0]} {k[1]:+.3f} {k[2]:.2f}/{k[3]:.2f}", S[vs][k]))
            print()


if __name__ == "__main__":
    main()
