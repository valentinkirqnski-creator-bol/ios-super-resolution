"""Pick v20's shipping setting and compare it with v17 on every held-out set (8-frame merged-noise target).

    RNN_GT_NMERGE=8 python v20_sweep.py runs/v20/model_s8000.pt runs/v20/model_s10000.pt runs/v20/model_s12000.pt
v17 is scored twice: as the app runs it today (flow features read the flow WITH the global homography) and with
the kernel fix (flow features read the residual tile flow). Every v20 setting (checkpoint x logit shift x cutoffs)
runs with the fix. Listed: the settings that miss no more than v17+fix on every miss measure of every set, ranked
by the good-pixel weight they lose on the perfectly-aligned sets.
"""
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                            # noqa: E402
import train_pix as tp                                  # noqa: E402
from real_protocol import load_net                      # noqa: E402

VALS = os.environ.get("VALS", "c15_val,c17_val,c19s05_val,c20rot_val,c20h_val,c20pH_val,c20p_val").split(",")
PERFECT = os.environ.get("PERFECT", "c20pH_val,c20p_val").split(",")   # sets where every visible pixel should be kept
SHIFTS = [float(x) for x in os.environ.get("SHIFTS", "-0.25,0,0.25,0.5,0.75").split(",")]
CUTS = [tuple(float(y) for y in x.split(":")) for x in os.environ.get("CUTS", "0.2:0.9,0.15:0.9,0.1:0.9").split(",")]
MISS = ["missed", "small", "rot", "leak05"]             # must not get worse than v17+fix


class _NoResidual:
    """compute_features without the residual flow: the flow features read the homography-composed flow."""
    def __init__(self, f):
        self.f = f

    def __call__(self, *a):
        return self.f(*a[:8])


def run(nets, vs, as_app=False):
    orig = tp.compute_features
    if as_app:
        tp.compute_features = _NoResidual(orig)
    try:
        ds = tp.Cache(f"data/{vs}")
        C = {}
        with torch.no_grad():
            for i in range(len(ds)):
                d = ds[i]
                for k, n in nets.items():
                    R = n(d["feats"], d["r0"])[0].clamp(1e-6, 1 - 1e-6)
                    lg = torch.log(R) - torch.log1p(-R)
                    for s, (lo, hi) in (itertools.product(SHIFTS, CUTS) if k != "v17" else [(0.0, (0.2, 0.9))]):
                        C[(k, s, lo, hi)] = C.get((k, s, lo, hi), 0) + tp.counts(torch.sigmoid(lg + s), d, lo, hi)
        return {k: tp.summary(v) for k, v in C.items()}
    finally:
        tp.compute_features = orig


def main():
    assert os.environ.get("RNN_GT_NMERGE") == "8", "run with RNN_GT_NMERGE=8 (the target v17 was trained on)"
    torch.set_num_threads(8)
    v17 = load_net(os.path.join(HERE, "runs/v17/model.pt"))
    nets = {"v17": v17}
    nets.update({os.path.basename(ck)[:-3]: load_net(os.path.join(HERE, ck)) for ck in sys.argv[1:]})
    S, APP = {}, {}
    for vs in VALS:
        S[vs] = run(nets, vs)
        if os.environ.get("AS_APP", "1") == "1" and (vs.startswith("c20rot") or vs.startswith("c20h") or vs.startswith("c20pH")):
            APP[vs] = run({"v17": v17}, vs, as_app=True)[("v17", 0.0, 0.2, 0.9)]
        print(tp.fmt(f"[{vs}] v17+fix", S[vs][("v17", 0.0, 0.2, 0.9)]), flush=True)
        if vs in APP:
            print(tp.fmt(f"[{vs}] v17 app", APP[vs]), flush=True)
    base = {vs: S[vs][("v17", 0.0, 0.2, 0.9)] for vs in VALS}
    cand = [k for k in S[VALS[0]] if k[0] != "v17"]
    ok = [k for k in cand if all(S[vs][k][m] <= base[vs][m] + 1e-9 for vs in VALS for m in MISS)]
    lost = lambda k: sum(S[vs][k]["lost"] for vs in PERFECT)
    print(f"\nsettings missing no more than v17+fix on every miss measure of every set: {len(ok)}", flush=True)
    for k in sorted(ok, key=lost)[:6]:
        for vs in VALS:
            print(tp.fmt(f"[{vs}] {k[0]} {k[1]:+.2f} {k[2]:.2f}/{k[3]:.2f}", S[vs][k]), flush=True)
        print(flush=True)
    if not ok:
        worse = lambda k: sum(S[vs][k][m] > base[vs][m] + 1e-9 for vs in VALS for m in MISS)
        print("closest (fewest miss measures worse than v17+fix):", flush=True)
        for k in sorted(cand, key=lambda k: (worse(k), lost(k)))[:4]:
            for vs in VALS:
                print(tp.fmt(f"[{vs}] {k[0]} {k[1]:+.2f} {k[2]:.2f}/{k[3]:.2f}", S[vs][k]), flush=True)
            print(flush=True)


if __name__ == "__main__":
    main()
