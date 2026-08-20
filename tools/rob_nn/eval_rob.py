"""Score the multiplicative robustness correction, on held-out records.

Two numbers, always reported together, because either alone is trivial to win:

  1. RESCUED. Of the pixels R_normal gets WRONG -- it still trusts them and
     merging them does visible damage -- how many does R_final now reject.
  2. COST. Of the pixels that are correctly aligned, how many does the
     correction needlessly hold back. This must stay extremely small.

A model that rejects everything wins (1) outright. A model that is the
identity wins (2) outright. Neither is worth having, so neither number is
quoted on its own anywhere in this file.

Plus the failure criterion that overrides both: the distribution of C on
correctly-aligned pixels. C multiplies, so a model sitting at 0.97 on the 95%
of the frame that is perfectly aligned has darkened the whole mask by 3% and
merely rescaled it -- and that shows up in no average, in no AUC, and very
clearly in the photograph. If mean C on clean pixels is not essentially 1, the
model has failed whatever else it scores.
"""
import os, sys
import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
from train_rob import (RobNet, load_dataset, split_records, target_and_weight,
                       IN_CH, CH_IDEAL_R, CH_W, CH_RNORM, CH_REP, CH_OCC,
                       CH_DIS, CH_HARM, PATTERNS)

PREFIX = os.environ.get("ROB_DATA") or os.path.join(SC, "robset")
CKPT = os.environ.get("ROB_CKPT", os.path.join(SC, "robnet.pt"))

# "Visible damage": the mis-fetch differs from the correct fetch by enough that
# a viewer would see the duplicated edge. R_ideal = exp(-z^2/4) with z the
# mis-fetch in units of the sensor's own sigma, so R_ideal < 0.5 is z > 1.67 --
# a difference well clear of the grain. Below that the "damage" is inside the
# noise the merge is there to average away.
HARM_CUT = 0.5
# "R_normal still trusts it": the analytic mask is passing at least half the
# weight through, so the artefact reaches the output.
TRUST_CUT = 0.5
# "R_final now rejects it": the final mask has been brought down to where the
# pixel contributes little enough not to show.
REJECT_CUT = 0.5


def infer(model, mu, sd, drop_ch, px):
    x = (px[..., :IN_CH] - mu) / sd
    t = torch.from_numpy(x[None].astype(np.float32)).permute(0, 3, 1, 2)
    for c in drop_ch:
        t[:, c] = 0.0
    with torch.no_grad():
        return model(t)[0, 0].numpy()


def main():
    data, meta, recs = load_dataset(PREFIX)
    if data is None:
        raise FileNotFoundError(f"no dataset at {PREFIX}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = RobNet(cin=ck.get("in_ch", IN_CH))
    model.load_state_dict(ck["state"]); model.eval()
    mu, sd = ck["mu"], ck["sd"]
    drop_ch = ck.get("drop_ch", [])
    holdout = ck.get("holdout_burst")
    _, ev_region, ev_burst = split_records(recs, holdout_burst=holdout)
    ev = ev_region + ev_burst
    cap = int(os.environ.get("ROB_EVAL_N", 400))
    if len(ev) > cap:
        step = len(ev) / cap
        ev = [ev[int(i * step)] for i in range(cap)]
    print(f"checkpoint {os.path.basename(CKPT)}   drop_ch={drop_ch}   "
          f"holdout_burst={holdout}")
    print(f"evaluating {len(ev)} held-out records "
          f"({len(ev_region)} region + {len(ev_burst)} burst available)\n")

    # Accumulators. Every category is a boolean mask over pixels, and each
    # collects the same two counts so the table columns mean the same thing
    # everywhere.
    cats = {}

    def acc(name, sel, rescued, missed_all, cost, clean_all, csum, cmin_hist):
        d = cats.setdefault(name, dict(res=0, res_n=0, cost=0, clean_n=0,
                                       csum=0.0, chist=np.zeros(5)))
        d["res"] += rescued; d["res_n"] += missed_all
        d["cost"] += cost; d["clean_n"] += clean_all
        d["csum"] += csum; d["chist"] += cmin_hist

    for r in ev:
        px = np.asarray(data[r["rec"]], dtype=np.float32)
        c = infer(model, mu, sd, drop_ch, px)
        r_ideal = px[..., CH_IDEAL_R]
        r_norm = px[..., CH_RNORM]
        valid = px[..., CH_W] > 0
        r_final = r_norm * c

        # (1) harmful AND missed by the analytic mask
        missed = valid & (r_ideal < HARM_CUT) & (r_norm >= TRUST_CUT)
        rescued = missed & (r_final < REJECT_CUT)
        # (2) provably safe, and how much the correction takes off it
        clean = valid & (r_ideal > 0.999)
        cost = clean & (c < 0.9)

        csum = float(c[clean].sum()) if clean.any() else 0.0
        # Where does C sit on clean pixels: <0.5, <0.9, <0.99, <0.999, >=0.999
        cc = c[clean]
        hist = np.array([(cc < 0.5).sum(), ((cc >= 0.5) & (cc < 0.9)).sum(),
                         ((cc >= 0.9) & (cc < 0.99)).sum(),
                         ((cc >= 0.99) & (cc < 0.999)).sum(),
                         (cc >= 0.999).sum()], dtype=np.float64)

        def add(name, m):
            mm_missed = missed & m
            mm_clean = clean & m
            ccm = c[mm_clean]
            h = np.array([(ccm < 0.5).sum(), ((ccm >= 0.5) & (ccm < 0.9)).sum(),
                          ((ccm >= 0.9) & (ccm < 0.99)).sum(),
                          ((ccm >= 0.99) & (ccm < 0.999)).sum(),
                          (ccm >= 0.999).sum()], dtype=np.float64)
            acc(name, m, int((rescued & m).sum()), int(mm_missed.sum()),
                int((cost & m).sum()), int(mm_clean.sum()),
                float(ccm.sum()), h)

        acc("ALL", None, int(rescued.sum()), int(missed.sum()),
            int(cost.sum()), int(clean.sum()), csum, hist)

        pat = PATTERNS[r["pattern"]] if r["pattern"] < len(PATTERNS) else "?"
        phase = r["phase"]
        rot = pat in ("rotation", "trans_rot")
        curved = pat in ("smooth", "abrupt")
        allm = np.ones_like(valid)
        if rot and phase == 0:
            add("wrong tiles during rotation", allm)
        if rot and phase == 1:
            add("CORRECT camera rotation", allm)
        if curved and phase == 0:
            add("curved / non-rigid motion", allm)
        if curved and phase == 1:
            add("CORRECT curved motion", allm)
        if phase == 1 or pat == "none":
            add("correctly aligned (all)", allm)
        # Content categories, per pixel rather than per record.
        add("thin edges", px[..., 17] > np.quantile(px[..., 17], 0.9))
        add("repetitive texture", px[..., CH_REP] > 0.7)
        add("moving objects (disoccl.)", px[..., CH_DIS] > 4.0)
        add("occlusion", px[..., CH_OCC] > 4.0)
        add("noisy (low light)", np.full(valid.shape, r["gain"] >= 2.0))
        add(f"burst {r['burst']}", allm)

    order = ["ALL", "correctly aligned (all)", "CORRECT camera rotation",
             "wrong tiles during rotation", "CORRECT curved motion",
             "curved / non-rigid motion", "thin edges", "repetitive texture",
             "moving objects (disoccl.)", "occlusion", "noisy (low light)",
             "burst 1", "burst 2", "burst 3"]
    print(f"{'category':<28} {'rescued':>18}   {'needlessly held back':>22}   "
          f"{'mean C':>8}")
    print(f"{'':<28} {'(harmful, R_normal':>18}   {'(correct px with':>22}")
    print(f"{'':<28} {'still trusts it)':>18}   {'C < 0.9)':>22}")
    print("-" * 84)
    for k in order:
        d = cats.get(k)
        if not d:
            continue
        res = f"{100.0*d['res']/d['res_n']:6.1f}%  n={d['res_n']:<7d}" if d["res_n"] else \
              f"{'--':>6}   n=0      "
        cost = f"{100.0*d['cost']/d['clean_n']:6.3f}%  n={d['clean_n']:<9d}" if d["clean_n"] else \
               f"{'--':>6}   n=0        "
        mc = f"{d['csum']/d['clean_n']:8.5f}" if d["clean_n"] else "      --"
        print(f"{k:<28} {res:>18}   {cost:>22}   {mc:>8}")

    d = cats["ALL"]
    h = cats.get("correctly aligned (all)", d)["chist"]
    n = h.sum()
    print("\nC on correctly-aligned pixels -- the failure criterion.")
    print("A model whose mass here is not piled against 1.0 has rescaled the")
    print("mask rather than corrected it, whatever it scores above.")
    if n:
        lab = ["< 0.5", "0.5-0.9", "0.9-0.99", "0.99-0.999", ">= 0.999"]
        for i, l in enumerate(lab):
            print(f"  C {l:<11} {100.0*h[i]/n:7.3f}%")
        cm = cats.get("correctly aligned (all)", d)
        print(f"  mean C      {cm['csum']/max(cm['clean_n'],1):9.5f}")


if __name__ == "__main__":
    main()
