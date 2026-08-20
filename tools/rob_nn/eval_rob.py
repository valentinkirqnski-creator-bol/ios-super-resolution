"""Score the multiplicative robustness correction on held-out records.

THE HEADLINE IS A CONSTRAINED NUMBER, NOT AUC.

AUC measures ranking, and ranking is not the problem: a correction that ranks
perfectly and takes 3% off every correctly-aligned pixel has darkened the whole
photograph. The objective is

    maximise   harmful misalignments corrected
    subject to P(C < 0.9 | correctly aligned) < 1%

so every model is reported at that same false-rejection budget. Comparing two
models at their own natural operating points compares two different trades and
tells you nothing; the budget sweep below puts them on the same one.

The sweep is a post-hoc LOGIT OFFSET, C_b = sigmoid(logit(C) + b). It is
monotone, so it cannot manufacture ranking the model does not have -- it only
moves where the model sits on its own curve, which is exactly what "the same
budget" requires. b = 0 is reported alongside, because that is the model as it
would actually ship.

SECOND HEADLINE: the four-group separation. On held-out records, split pixels
into correct translation / correct rotation / correct complex motion / harmful
misalignment that R_normal missed, and look at where C lands for each. The
first three must sit at 1. The last must sit at 0. The two that must SEPARATE
are correct rotation against missed rotational misalignment, because "reject
rotation" is the cheap wrong solution to this problem and it would score well
on everything else. The distributions are written as a PNG.
"""
import os, sys
import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
from train_rob import (RobNet, load_dataset, split_records, target_and_weight,
                       quadrant, IN_CH, CH_IDEAL_R, CH_W, CH_RNORM, CH_REP,
                       CH_OCC, CH_DIS, PATTERNS)
import fb_valid
from vis_rob import save_png

PREFIX = os.environ.get("ROB_DATA") or os.path.join(SC, "robset")
CKPT = os.environ.get("ROB_CKPT", os.path.join(SC, "robnet.pt"))
OUTDIR = os.environ.get("ROB_PLOT_DIR", SC)
TAG = os.environ.get("ROB_TAG", "robnet")

# "Visible damage": R_ideal = exp(-z^2/4) with z the mis-fetch in units of the
# sensor's own sigma, so R_ideal < 0.5 is z > 1.67 -- clear of the grain.
HARM_CUT = 0.5
TRUST_CUT = 0.5      # "R_normal still passes at least half the weight"
REJECT_CUT = 0.5     # "R_final no longer does"
SAFE_CUT = 0.9       # the budget is on P(C < 0.9 | correctly aligned)
BUDGET = float(os.environ.get("ROB_BUDGET", 0.01))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def infer(model, mu, sd, drop_ch, px):
    x = (px[..., :IN_CH] - mu) / sd
    t = torch.from_numpy(x[None].astype(np.float32)).permute(0, 3, 1, 2)
    for c in drop_ch:
        t[:, c] = 0.0
    with torch.no_grad():
        return model(t)[0, 0].numpy()


def hist_png(path, groups, bins=60):
    """Two-panel linear-grey histogram plot, drawn without a plotting library.

    Top panel: the classes that must sit at 1. Bottom: correct rotation against
    the missed rotational misalignment it must separate from.
    """
    H, Wd = 420, 720
    img = np.ones((H, Wd, 3), np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    colours = {"correct translation": (0.10, 0.35, 0.75),
               "correct rotation": (0.05, 0.55, 0.25),
               "correct complex motion": (0.55, 0.35, 0.05),
               "missed harmful misalignment": (0.80, 0.10, 0.10),
               "missed harmful, rotational": (0.80, 0.10, 0.10)}

    def panel(y0, y1, names):
        h = y1 - y0 - 24
        img[y0:y1, 40:Wd - 10] = 0.97
        for i in range(5):
            xg = 40 + int((Wd - 50 - 40) * i / 4)
            img[y0:y0 + h, xg:xg + 1] = 0.85
        for nm in names:
            v = groups.get(nm)
            if v is None or len(v) == 0:
                continue
            cnt, _ = np.histogram(np.clip(v, 0, 1), bins=edges)
            cnt = cnt / max(cnt.max(), 1)
            col = np.array(colours.get(nm, (0.3, 0.3, 0.3)), np.float32)
            for b in range(bins):
                xa = 40 + int((Wd - 50 - 40) * b / bins)
                xb = 40 + int((Wd - 50 - 40) * (b + 1) / bins)
                hb = int(cnt[b] * (h - 2))
                if hb <= 0:
                    continue
                reg = img[y0 + h - hb:y0 + h, xa:max(xb, xa + 1)]
                img[y0 + h - hb:y0 + h, xa:max(xb, xa + 1)] = reg * 0.45 + col * 0.55
    panel(20, 200, ["correct translation", "correct rotation",
                    "correct complex motion"])
    panel(230, 410, ["correct rotation", "missed harmful, rotational"])
    save_png(path, img)


def main():
    data, meta, recs = load_dataset(PREFIX)
    if data is None:
        raise FileNotFoundError(f"no dataset at {PREFIX}")
    gh, gw = meta["guide_h"], meta["guide_w"]
    fb = fb_valid.load_fb(os.environ.get("ROB_FB", os.path.join(SC, "fb")))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = RobNet(cin=ck.get("in_ch", IN_CH))
    model.load_state_dict(ck["state"]); model.eval()
    mu, sd = ck["mu"], ck["sd"]
    drop_ch = ck.get("drop_ch", [])
    holdout = ck.get("holdout_burst")
    _, ev_region, ev_burst = split_records(recs, holdout_burst=holdout)
    ev = ev_region + ev_burst
    cap = int(os.environ.get("ROB_EVAL_N", 300))
    if len(ev) > cap:
        step = len(ev) / cap
        ev = [ev[int(i * step)] for i in range(cap)]
    print(f"checkpoint {os.path.basename(CKPT)}   drop_ch={drop_ch}   "
          f"holdout_burst={holdout}")
    print(f"{len(ev)} held-out records "
          f"({len(ev_region)} region + {len(ev_burst)} burst available)\n")

    cats, groups = {}, {}
    # Pooled logits, for the budget sweep. Only two populations are needed: the
    # correctly-aligned pixels that set the budget, and the harmful ones that
    # R_normal missed, which are what the budget buys.
    L_safe, L_miss, RN_miss = [], [], []

    def acc(name, rescued, missed_n, cost, clean_n, csum, chist):
        d = cats.setdefault(name, dict(res=0, res_n=0, cost=0, clean_n=0,
                                       csum=0.0, chist=np.zeros(5)))
        d["res"] += rescued; d["res_n"] += missed_n
        d["cost"] += cost; d["clean_n"] += clean_n
        d["csum"] += csum; d["chist"] += chist

    def chist(c):
        return np.array([(c < 0.5).sum(), ((c >= 0.5) & (c < 0.9)).sum(),
                         ((c >= 0.9) & (c < 0.99)).sum(),
                         ((c >= 0.99) & (c < 0.999)).sum(),
                         (c >= 0.999).sum()], np.float64)

    for r in ev:
        px = np.asarray(data[r["rec"]], dtype=np.float32)
        c = infer(model, mu, sd, drop_ch, px)
        mag = np.hypot(px[..., 9], px[..., 10])
        fb_ok, rot = fb_valid.record_tile_maps(fb, r, gh, gw, mag)
        _, w = target_and_weight(px, fb_ok)
        valid = w > 0
        r_ideal = px[..., CH_IDEAL_R]
        r_norm = px[..., CH_RNORM]
        r_final = r_norm * c

        missed = valid & (r_ideal < HARM_CUT) & (r_norm >= TRUST_CUT)
        rescued = missed & (r_final < REJECT_CUT)
        clean = valid & (r_ideal > 0.999)
        cost = clean & (c < SAFE_CUT)
        L_safe.append(logit(c[clean])); L_miss.append(logit(c[missed]))
        RN_miss.append(r_norm[missed])

        def add(name, m):
            cc = c[clean & m]
            acc(name, int((rescued & m).sum()), int((missed & m).sum()),
                int((cost & m).sum()), int(cc.size), float(cc.sum()), chist(cc))

        acc("ALL", int(rescued.sum()), int(missed.sum()), int(cost.sum()),
            int(clean.sum()), float(c[clean].sum()), chist(c[clean]))

        pat = PATTERNS[r["pattern"]] if r["pattern"] < len(PATTERNS) else "?"
        phase, allm = r["phase"], np.ones_like(valid)
        rotational = (rot > fb_valid.ROT_STRONG) if rot is not None else allm
        translation = (rot < fb_valid.ROT_NONE) if rot is not None else ~allm
        # Complex / non-rigid motion: the baseline flow disagrees with its own
        # 3x3 tile neighbourhood by more than 2 raw px. On a CLEAN phase that is
        # real parallax or subject motion, not an error. Channel 15 is a scene
        # descriptor here, not a label.
        complexm = px[..., 15] > 2.0
        clean_phase = (phase == 1) or (pat == "none")
        if clean_phase:
            add("correct camera rotation", rotational)
            add("correct pure translation", translation)
            add("correct complex motion", complexm)
            add("correctly aligned (all)", allm)
            groups.setdefault("correct rotation", []).append(c[clean & rotational])
            groups.setdefault("correct translation", []).append(c[clean & translation])
            groups.setdefault("correct complex motion", []).append(c[clean & complexm])
        else:
            if pat in ("rotation", "trans_rot"):
                add("wrong tiles during rotation", allm)
                groups.setdefault("missed harmful, rotational", []).append(c[missed])
            if pat in ("smooth", "abrupt"):
                add("wrong during curved motion", allm)
            groups.setdefault("missed harmful misalignment", []).append(c[missed])
        add("thin edges", px[..., 17] > 0.004)
        add("repetitive texture", px[..., CH_REP] > 0.7)
        add("moving objects (disoccl.)", px[..., CH_DIS] > 4.0)
        add("occlusion", px[..., CH_OCC] > 4.0)
        add("noisy (low light)", np.full(valid.shape, r["gain"] >= 2.0))
        add(f"burst {r['burst']}", allm)

    # ---------------------------------------------------------- the headline
    ls = np.concatenate(L_safe) if L_safe else np.zeros(0)
    lm = np.concatenate(L_miss) if L_miss else np.zeros(0)
    rnm = np.concatenate(RN_miss) if RN_miss else np.zeros(0)
    lcut = logit(SAFE_CUT)
    print("CONSTRAINED OBJECTIVE -- maximise corrections of harmful "
          "misalignment that")
    print(f"R_normal missed, subject to P(C < {SAFE_CUT} | correctly aligned) "
          f"< {BUDGET*100:.1f}%")
    print(f"  correctly-aligned px {ls.size}, missed-harmful px {lm.size}\n")
    print(f"  {'offset b':>9} {'P(C<0.9|safe)':>14} {'harmful corrected':>18}")
    rows = []
    if ls.size and lm.size:
        # The b that exactly meets the budget: the BUDGET-quantile of the safe
        # logits has to land at logit(0.9).
        b_need = lcut - np.quantile(ls, BUDGET)
        for b in sorted({0.0, float(b_need)}):
            cs = 1.0 / (1.0 + np.exp(-(ls + b)))
            cm = 1.0 / (1.0 + np.exp(-(lm + b)))
            fr = float((cs < SAFE_CUT).mean())
            det = float((rnm * cm < REJECT_CUT).mean())
            tag = "  <- as it would ship" if b == 0.0 else "  <- at the budget"
            rows.append((b, fr, det))
            print(f"  {b:9.3f} {fr*100:13.3f}% {det*100:17.1f}%{tag}")
    print()

    order = ["ALL", "correctly aligned (all)", "correct pure translation",
             "correct camera rotation", "correct complex motion",
             "wrong tiles during rotation", "wrong during curved motion",
             "thin edges", "repetitive texture", "moving objects (disoccl.)",
             "occlusion", "noisy (low light)", "burst 1", "burst 2", "burst 3"]
    print(f"{'category':<28} {'harmful corrected':>19}   "
          f"{'needlessly held back':>24}   {'mean C':>8}")
    print(f"{'':<28} {'(R_normal missed it)':>19}   "
          f"{'(correct px, C < 0.9)':>24}")
    print("-" * 88)
    for k in order:
        d = cats.get(k)
        if not d:
            continue
        res = (f"{100.0*d['res']/d['res_n']:6.1f}%  n={d['res_n']:<8d}"
               if d["res_n"] else f"{'--':>6}   n=0       ")
        cost = (f"{100.0*d['cost']/d['clean_n']:7.4f}%  n={d['clean_n']:<10d}"
                if d["clean_n"] else f"{'--':>7}   n=0         ")
        mc = f"{d['csum']/d['clean_n']:8.5f}" if d["clean_n"] else "      --"
        print(f"{k:<28} {res:>19}   {cost:>24}   {mc:>8}")

    cm = cats.get("correctly aligned (all)", cats["ALL"])
    h = cm["chist"]; n = h.sum()
    print("\nC on correctly-aligned pixels -- the criterion that overrides the rest.")
    print("C multiplies, so a model sitting at 0.97 here has darkened the whole")
    print("frame by 3% and merely rescaled the mask, whatever it detects.")
    if n:
        for i, l in enumerate(["< 0.5", "0.5-0.9", "0.9-0.99", "0.99-0.999",
                               ">= 0.999"]):
            print(f"  C {l:<11} {100.0*h[i]/n:8.4f}%")
        print(f"  mean C      {cm['csum']/max(cm['clean_n'],1):10.6f}")

    g = {k: np.concatenate(v) for k, v in groups.items() if v}
    print("\nfour-group separation (the two that MUST separate are the last two)")
    print(f"  {'group':<32} {'n':>10} {'mean C':>9} {'p05':>8} {'p50':>8} {'p95':>8}")
    for k in ["correct translation", "correct rotation", "correct complex motion",
              "missed harmful misalignment", "missed harmful, rotational"]:
        v = g.get(k)
        if v is None or not len(v):
            continue
        print(f"  {k:<32} {len(v):>10} {v.mean():>9.5f} "
              f"{np.percentile(v,5):>8.4f} {np.percentile(v,50):>8.4f} "
              f"{np.percentile(v,95):>8.4f}")
    a, b = g.get("correct rotation"), g.get("missed harmful, rotational")
    if a is not None and b is not None and len(a) and len(b):
        # Overlap of the two distributions: the fraction of correct-rotation
        # pixels below the median of the harmful ones, and vice versa.
        print(f"  correct rotation below harmful median: "
              f"{100.0*(a < np.median(b)).mean():.3f}%")
        print(f"  harmful above correct-rotation p05:    "
              f"{100.0*(b > np.percentile(a,5)).mean():.3f}%")
    if g:
        out = os.path.join(OUTDIR, f"{TAG}_C_separation.png")
        hist_png(out, g)
        print(f"\nwrote {out}")
        print("  top: the three correctly-aligned classes (mass must be at 1)")
        print("  bottom: correct rotation vs missed rotational misalignment")


if __name__ == "__main__":
    main()
