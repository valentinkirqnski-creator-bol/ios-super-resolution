"""Head-to-head scoring of the learned mask against the analytic one.

Three things are reported, and all three are needed. Any one on its own has
already misled this project at least once.

  RANKING (AUC, PR AUC). Does the mask order harmful pixels below harmless
  ones. Computed rank-based (Mann-Whitney), never by sweeping a threshold: the
  analytic mask pins most pixels at exactly R = 1, so its swept curve is
  truncated and integrating it measures the truncation rather than the mask.

  OPERATING POINT (detection, false rejection). Ranking is free to be good at
  a threshold nothing in the pipeline can actually place. The merge multiplies
  by R as emitted, so what matters is what happens at the mask's own values --
  and false rejection is a real cost, not a free win. Every good pixel rejected
  is one fewer frame averaged there, which comes back as noise and grain. A
  previous model caught 91.5% of harmful pixels while discarding 17.7% of good
  ones against the analytic mask's 1.9%; that is not obviously an improvement,
  which is why the trade-off curve is printed rather than one chosen point.

  CALIBRATION (mean, >0.9, <0.1). The point of the whole exercise. A mask that
  ranks perfectly while never leaving mid-grey trusts nothing fully and rejects
  nothing fully, and the merge behaves as though it were a constant. That is
  exactly how the previous model passed its benchmarks and failed in the real
  pipeline: AUC 0.95, and 0.0% of pixels above 0.9.

  The yardstick for calibration is the IDEAL R on the same pixels, not the
  analytic mask. A model that regresses the label perfectly reproduces the
  label's histogram; that is the target.

Everything is broken out by failure type, by injected error size, by scene
content and by simulated light level, because a single pooled number hides
precisely the cases the user reported.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import (RobNet, classical_R, load_dataset, split_records, PATTERNS,
                       IN_CH, CH_IDEAL_R, CH_FERR, CH_W, CH_REP, CH_OCC, CH_DIS,
                       PREFIX)

SC = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("ROB_CKPT", os.path.join(SC, "robnet.pt"))


# --------------------------------------------------------------------- metrics
def auc_rank(R, bad, cap=300000, seed=0):
    """Mann-Whitney AUC: the probability that a randomly chosen harmful pixel
    is trusted less than a randomly chosen harmless one, ties counted half."""
    rng = np.random.RandomState(seed)
    b, g = R[bad].ravel(), R[~bad].ravel()
    if len(b) == 0 or len(g) == 0:
        return float("nan")
    if len(b) > cap: b = b[rng.choice(len(b), cap, replace=False)]
    if len(g) > cap: g = g[rng.choice(len(g), cap, replace=False)]
    s = np.concatenate([b, g])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(ss):                       # average ranks within ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    nb, ng = len(b), len(g)
    # U for "harmful ranked LOW" -> subtract from 1
    U = ranks[:nb].sum() - nb * (nb + 1) / 2.0
    return 1.0 - U / (nb * ng)


def pr_auc(R, bad, cap=300000, seed=0):
    """Average precision for detecting harmful pixels by LOW R."""
    rng = np.random.RandomState(seed)
    idx = np.arange(R.size)
    if R.size > 2 * cap:
        idx = rng.choice(R.size, 2 * cap, replace=False)
    sc, y = -R.ravel()[idx], bad.ravel()[idx]
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-sc, kind="mergesort")
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def det_at_fp(R, bad, target):
    """Detection when the accept threshold is set to spend exactly `target` of
    the harmless pixels as false rejections."""
    g = R[~bad].ravel()
    if g.size == 0 or bad.sum() == 0:
        return float("nan")
    th = np.quantile(g, target)
    return float((R[bad] < th).mean())


def operating_point(R, bad, th=0.5):
    if bad.sum() == 0 or (~bad).sum() == 0:
        return float("nan"), float("nan")
    return float((R[bad] < th).mean()), float((R[~bad] < th).mean())


def row(name, R, bad, extra=""):
    det, fp = operating_point(R, bad)
    print(f"  {name:22s} {auc_rank(R, bad):6.3f} {pr_auc(R, bad):7.3f} "
          f"{det * 100:8.1f}% {fp * 100:8.1f}% "
          f"{det_at_fp(R, bad, 0.02) * 100:8.1f}% {det_at_fp(R, bad, 0.05) * 100:8.1f}%{extra}")


def header(title):
    print(f"\n{title}")
    print(f"  {'':22s} {'AUC':>6} {'PR AUC':>7} {'det@0.5':>9} {'falsrej':>9} "
          f"{'@2%FP':>9} {'@5%FP':>9}")


def main():
    data, meta, recs = load_dataset(PREFIX)
    if data is None:
        raise SystemExit(f"no dataset at {PREFIX}")
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    mu, sd = ck["mu"], ck["sd"]
    model = RobNet(); model.load_state_dict(ck["state"]); model.eval()
    hb = ck.get("holdout_burst")
    _, ev_region, ev_burst = split_records(recs, holdout_burst=hb)

    which = os.environ.get("ROB_EVAL", "region")
    ev = ev_burst if which == "burst" else ev_region
    if not ev:
        raise SystemExit(f"no records in the '{which}' eval split")
    cap = int(os.environ.get("ROB_EVAL_N", 400))
    if len(ev) > cap:
        ev = ev[:: max(1, len(ev) // cap)]
    print(f"evaluating on the '{which}' split: {len(ev)} records"
          + (f", burst {hb} withheld from training" if which == "burst" else ""))

    L, C, B, I = [], [], [], []
    FE, REP, PAT, GAIN, NS, STD, HF, OCC, DIS, BURST = ([] for _ in range(10))
    by_rec = {}
    for r in ev:
        a = np.asarray(data[r["rec"]], dtype=np.float32)
        w = a[..., CH_W] > 0
        if not w.any():
            continue
        with torch.no_grad():
            xin = torch.from_numpy(((a[..., :IN_CH] - mu) / sd)).permute(2, 0, 1)[None]
            pred = model(xin)[0, 0].numpy()
        ideal = a[..., CH_IDEAL_R]
        L.append(pred[w]); C.append(classical_R(a)[w]); I.append(ideal[w])
        B.append(ideal[w] < 0.5)
        FE.append(a[..., CH_FERR][w]); REP.append(a[..., CH_REP][w])
        PAT.append(np.full(int(w.sum()), r["pattern"]))
        GAIN.append(np.full(int(w.sum()), r.get("gain", 1.0)))
        NS.append(a[..., 12][w]); STD.append(a[..., 3:6].mean(-1)[w]); HF.append(a[..., 17][w])
        OCC.append(a[..., CH_OCC][w]); DIS.append(a[..., CH_DIS][w])
        BURST.append(np.full(int(w.sum()), r["burst"]))
        by_rec.setdefault(r["pattern"], []).append(r["rec"])

    cat = lambda v: np.concatenate(v)
    L, C, B, I = cat(L), cat(C), cat(B), cat(I)
    FE, REP, PAT, GAIN = cat(FE), cat(REP), cat(PAT), cat(GAIN)
    NS, STD, HF = cat(NS), cat(STD), cat(HF)
    OCC, DIS, BURST = cat(OCC), cat(DIS), cat(BURST)
    print(f"{B.size} scored pixels, {B.mean() * 100:.2f}% harmful "
          f"(ideal R < 0.5)\n")

    header("POOLED")
    row("analytic", C, B)
    row("learned", L, B)

    # ---------------------------------------------------------- calibration
    print("\nCALIBRATION -- the merge consumes the value, not the rank.")
    print("The target is the IDEAL R column: a model that regresses the label")
    print("reproduces the label's own histogram.")
    print(f"  {'mask':22s} {'mean':>7} {'>0.9':>8} {'<0.1':>8} {'min':>7} {'max':>7}")
    for nm, R in (("ideal R (target)", I), ("analytic", C), ("learned", L)):
        print(f"  {nm:22s} {R.mean():>7.3f} {(R > 0.9).mean() * 100:>7.1f}% "
              f"{(R < 0.1).mean() * 100:>7.1f}% {R.min():>7.3f} {R.max():>7.3f}")
    if (L > 0.9).mean() < 0.20:
        print("\n  FAIL: the learned mask still will not commit. Under 20% of")
        print("  pixels above 0.9 means the merge sees a near-constant mask,")
        print("  whatever the AUC says.")

    # ------------------------------------------------------- trade-off curve
    print("\nTRADE-OFF -- detection at each false-rejection budget.")
    print("  False rejection is not free: each one is a frame not averaged.")
    budgets = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
    print(f"  {'mask':22s}" + "".join(f"{b * 100:>8.1f}%FP" for b in budgets))
    for nm, R in (("analytic", C), ("learned", L)):
        print(f"  {nm:22s}" + "".join(f"{det_at_fp(R, B, b) * 100:>9.1f}" for b in budgets))

    # ---------------------------------------------------------- by failure type
    groups = [
        ("rotation", np.isin(PAT, [6, 7])),
        ("smooth wrong flow", np.isin(PAT, [4, 6, 7])),
        ("single wrong tiles", PAT == 1),
        ("groups of tiles", np.isin(PAT, [2, 3])),
        ("abrupt boundary", PAT == 5),
        ("wrong magnitude", PAT == 8),
        ("wrong direction", PAT == 9),
        ("along edge", PAT == 10),
        ("across edge", PAT == 11),
        ("similar content", PAT == 12),
        ("global offset", PAT == 13),
        ("translation (all)", np.isin(PAT, [1, 2, 3, 8, 9, 10, 11, 12, 13])),
    ]
    header("BY FAILURE TYPE  (learned / analytic on the same pixels)")
    for nm, m in groups:
        if m.sum() < 5000 or B[m].sum() < 50:
            continue
        row(nm + " [L]", L[m], B[m])
        row(nm + " [A]", C[m], B[m])

    # -------------------------------------------------------- by error size
    header("BY INJECTED ERROR SIZE  (raw px)")
    for lo, hi in [(0.0, 1e-6), (1e-6, 0.25), (0.25, 1), (1, 4), (4, 16), (16, 1e9)]:
        m = (FE >= lo) & (FE < hi)
        if m.sum() < 5000 or B[m].sum() < 50:
            continue
        row(f"{lo:g}-{hi:g} px [L]", L[m], B[m])
        row(f"{lo:g}-{hi:g} px [A]", C[m], B[m])

    # ------------------------------------------------------- by scene content
    hf_hi = np.quantile(HF, 0.85)
    rep_hi = np.quantile(REP, 0.85)
    header("BY SCENE CONTENT")
    for nm, m in (("thin/fine structure", HF > hf_hi),
                  ("repetitive texture", REP > rep_hi),
                  ("flat (std < noise)", STD < NS)):
        if m.sum() < 5000 or B[m].sum() < 50:
            continue
        row(nm + " [L]", L[m], B[m])
        row(nm + " [A]", C[m], B[m])

    # ------------------------------------------------ occlusion / disocclusion
    # Reported separately because they are different failures and the user
    # listed both. Occlusion: content hidden behind a moving object in THIS
    # frame. Disocclusion: the reference is the odd view, i.e. the object was
    # there and has moved away -- the case that ghosts a moving subject,
    # because the aligner matched the background underneath it.
    header("SCENE MOTION  (multi-frame consensus, structure-gated)")
    for nm, m in (("occluded in this frame", OCC > 4),
                  ("reference-anomalous", DIS > 4),
                  ("static scene only", (OCC <= 4) & (DIS <= 4))):
        if m.sum() < 5000 or B[m].sum() < 50:
            continue
        row(nm + " [L]", L[m], B[m])
        row(nm + " [A]", C[m], B[m])

    # ------------------------------------------------------------- per burst
    header("PER BURST")
    for b in sorted(set(BURST.tolist())):
        m = BURST == b
        if m.sum() < 5000 or B[m].sum() < 50:
            continue
        row(f"burst {b} [L]", L[m], B[m])
        row(f"burst {b} [A]", C[m], B[m])

    # -------------------------------------------------------- by light level
    header("BY SIMULATED LIGHT LEVEL  (1/k exposure; k>1 is SYNTHESISED)")
    for g in sorted(set(GAIN.tolist())):
        m = GAIN == g
        if m.sum() < 5000 or B[m].sum() < 50:
            continue
        row(f"1/{g:g} exposure [L]", L[m], B[m])
        row(f"1/{g:g} exposure [A]", C[m], B[m])
    print("\n  Light levels above 1 are emulated from the daylight captures by")
    print("  scaling the linear signal and re-noising to the sensor's own")
    print("  alpha*v+beta law. No genuinely low-light burst was available.")


if __name__ == "__main__":
    main()
