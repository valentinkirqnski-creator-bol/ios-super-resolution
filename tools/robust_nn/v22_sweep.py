"""Pick v22's setting: catch the most edge slips WITHOUT changing anything else, against the shipped v21.

    RNN_GT_NMERGE=8 RNN_GT_EDGE=1.0,0.01,0.08,0.5 python v22_sweep.py runs/v22b/model_s8000.pt runs/v22b/model_s10000.pt
Constraints vs v21 (same target, cutoffs 0.20/0.90): good weight lost no higher (+TOL points) on every KEEP set, and
misses no higher (+TOL points) on every OLD set. Objective: fewest missed slips (mean over the SLIP sets).
"""
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                            # noqa: E402
import v20_sweep as vs                                  # noqa: E402
from real_protocol import load_net                      # noqa: E402

KEEP = os.environ.get("KEEP", "c15_val,c17_val,c20pH_val,c20p_val,c22txp_val,c21txp_val").split(",")
OLD = os.environ.get("OLD", "c15_val,c17_val,c20pH_val,c20p_val").split(",")
SLIP = os.environ.get("SLIP", "c22txm_val,c22txr_val,c22scr_val").split(",")
TOL = float(os.environ.get("TOL", "0.3"))


def main():
    torch.set_num_threads(10)
    vs.SHIFTS = [float(x) for x in os.environ.get("SHIFTS", "0,0.25,0.5,0.75,1.0,1.25,1.5").split(",")]
    vs.CUTS = [tuple(float(y) for y in x.split(":")) for x in os.environ.get("CUTS", "0.2:0.9").split(",")]
    nets = {"v17": load_net(os.path.join(HERE, "runs/v21/model.pt"))}       # "v17" = the baseline slot: shipped v21
    nets.update({os.path.basename(os.path.dirname(c)) + "/" + os.path.basename(c)[:-3]: load_net(os.path.join(HERE, c))
                 for c in sys.argv[1:]})
    sets = list(dict.fromkeys(KEEP + OLD + SLIP))
    S = {v: vs.run(nets, v) for v in sets}
    base = {v: S[v][("v17", 0.0, 0.2, 0.9)] for v in sets}
    for v in sets:
        b = base[v]
        print(f"[{v}] v21: missed {b['missed']:.2f}  small {b['small']:.2f}  lost {b['lost']:.2f}%", flush=True)
    cand = [k for k in S[sets[0]] if k[0] != "v17"]
    ok = [k for k in cand if all(S[v][k]["lost"] <= base[v]["lost"] + TOL for v in KEEP)
          and all(S[v][k]["missed"] <= base[v]["missed"] + TOL for v in OLD)]
    slip = lambda k: sum(S[v][k]["missed"] for v in SLIP) / len(SLIP)
    print(f"\nsettings within v21's kept weight and misses (tolerance {TOL} pt): {len(ok)}", flush=True)
    for k in sorted(ok, key=slip)[:5] + sorted(cand, key=slip)[:2]:
        tag = "OK " if k in ok else "ref"
        print(f"{tag} {k[0]} {k[1]:+.2f}: slips missed {slip(k):.2f} (v21 {sum(base[v]['missed'] for v in SLIP) / len(SLIP):.2f})", flush=True)
        for v in sets:
            s = S[v][k]
            print(f"      [{v}] missed {s['missed']:.2f} (v21 {base[v]['missed']:.2f})  lost {s['lost']:.2f}% "
                  f"(v21 {base[v]['lost']:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
