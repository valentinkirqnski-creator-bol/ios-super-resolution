"""Pick the shipping setting of a compact-v15 checkpoint: sweep the logit shift (folded into the head bias, weights
only) x the app's exact-0/1 cutoffs (Reject Below / Trust Above) on the held-out cache, and list the settings that
beat shipped v14 (shift folded, 0.20/0.90) on EVERY measure at once: misses, over-rejection, small-shift and
rotation misses, flat over-rejection, partial ghost weight (all and clearly misaligned), similarity, boundary error.

    python compact_sweep.py runs/v15d/compact_s20000.pt [more]
"""
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                            # noqa: E402
import train_pix as tp                                  # noqa: E402
from real_protocol import load_net                      # noqa: E402
from rnn.compact import CompactNet                      # noqa: E402

SHIFTS = [float(x) for x in os.environ.get("SHIFTS", "-0.75,-0.625,-0.5,-0.375,-0.25,-0.125,0").split(",")]
CUTS = [tuple(float(y) for y in x.split(":")) for x in
        os.environ.get("CUTS", "0.05:0.95,0.1:0.95,0.15:0.95,0.2:0.9,0.25:0.9,0.3:0.9,0.25:0.85").split(",")]
KEYS = ["missed", "over", "small", "rot", "flat", "leak", "leak05"]          # lower is better
UP = ["sim"]                                                                  # higher is better


def main():
    torch.set_num_threads(4)
    base = load_net(os.path.join(HERE, "runs", "v14", "model.pt"))
    nets = {}
    for ck in sys.argv[1:]:
        sd = torch.load(os.path.join(HERE, ck), map_location="cpu")
        first3 = sd["pin.weight"].shape[-1] == 3
        n = CompactNet(base.mu, base.sd, ch=sd["pin.weight"].shape[0], first3=first3)
        n.load_state_dict(sd)
        n.eval()
        nets[os.path.basename(ck)[:-3]] = n
    ds = tp.Cache("data/c15_val")
    C = {}
    with torch.no_grad():
        for i in range(len(ds)):
            d = ds[i]
            C["v14"] = C.get("v14", 0) + tp.counts(base(d["feats"], d["r0"])[0], d)
            for k, n in nets.items():
                R = n(d["feats"], d["r0"], d["pix"])[0].clamp(1e-6, 1 - 1e-6)
                lg = torch.log(R) - torch.log1p(-R)
                for s, (lo, hi) in itertools.product(SHIFTS, CUTS):
                    C[(k, s, lo, hi)] = C.get((k, s, lo, hi), 0) + tp.counts(torch.sigmoid(lg + s), d, lo, hi)
            if i % 10 == 9:
                print(f"{i + 1}/{len(ds)}", flush=True)
    S = {k: tp.summary(v) for k, v in C.items()}
    ref = S.pop("v14")
    print(tp.fmt("v14 (shipped)", ref))
    dom = [k for k, s in S.items() if all(s[m] <= ref[m] + 1e-9 for m in KEYS) and all(s[m] >= ref[m] for m in UP)]
    print(f"\nsettings beating v14 on every measure: {len(dom)}")
    for k in sorted(dom, key=lambda k: S[k]["bnd"])[:12]:
        print(tp.fmt(f"{k[0]} {k[1]:+.3f} {k[2]:.2f}/{k[3]:.2f}", S[k]))
    print("\nbest balanced per checkpoint (for reference):")
    for name in nets:
        k = max((k for k in S if k[0] == name), key=lambda k: S[k]["bal"])
        print(tp.fmt(f"{k[0]} {k[1]:+.3f} {k[2]:.2f}/{k[3]:.2f}", S[k]))
    # closest to dominating: fewest measures worse than v14, then smallest total shortfall
    def short(k):
        s = S[k]
        bad = [m for m in KEYS if s[m] > ref[m] + 1e-9] + [m for m in UP if s[m] < ref[m]] + (["bnd"] if s["bnd"] > ref["bnd"] else [])
        return len(bad), bad
    print("\nnearest misses (measures where still worse than v14):")
    for k in sorted(S, key=lambda k: (short(k)[0], S[k]["bnd"]))[:8]:
        print(tp.fmt(f"{k[0]} {k[1]:+.3f} {k[2]:.2f}/{k[3]:.2f}", S[k]), " worse:", ",".join(short(k)[1]) or "-")


if __name__ == "__main__":
    main()
