"""Four-way comparison of the robustness mask on held-out frames.

    1  Wronski only                     (Eq. 5-9, motion_geom_reject off)
    2  Wronski + geometry rejection     (the current shipping baseline)
    3  Wronski + NN refinement
    4  Wronski + geometry rejection + NN refinement

Every configuration is scored against the same ground truth -- R*, the
inverse-MSE optimal merge weight computed from known motion in
refine_dataset.cpp -- and the numbers reported are the ones that decide
whether this stage is worth shipping, not the ones that are easiest to make
look good:

  false rejection rate   how often a pixel whose flow was right (R* >= 0.9)
                         is pushed below half the weight it had. The headline
                         failure. Reported overall and, separately, on the
                         content classes that must be protected -- fine
                         detail, coherent edges, flat sky, high noise -- since
                         an average over a frame that is mostly sky can hide a
                         catastrophe in the text.
  missed artifact rate   how often a pixel the ground truth says to drop
                         (R* < 0.5) still carries most of its weight.
  changed                fraction of pixels the refinement moved at all. This
                         is supposed to be a sparse correction; a run that
                         rewrites half the frame has failed even if every
                         other number improved.
  merge MSE              the physical bottom line: mean excess mean-squared
                         error of the merged pixel relative to using the
                         optimal weight everywhere. Lower is better, and
                         unlike the rates above it cannot be gamed by moving
                         a threshold.

A per-regime breakdown follows, because the whole claim is that the gain
appears specifically where one flow vector per tile cannot represent the
motion, and a single frame-wide average cannot show that.
"""
import os, sys
import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
from train_refine import (NAMES, IN_CH, CH_RGEOM, CH_RSTAR, CH_FLOWERR,
                          CH_DELTA, CH_SIGMA, LOG_CH, RefineMLP, RefineCNN,
                          load, features, signed_log_np, finite, visible_rstar)

PREFIX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SC, "refineset")
CKPT = sys.argv[2] if len(sys.argv) > 2 else None
KAPPA = float(os.environ.get("ROB_REFINE_KAPPA", 0.75))
DEAD = float(os.environ.get("ROB_REFINE_DEADZONE", 0.05))
HOLDOUT = int(os.environ.get("ROB_REFINE_HOLDOUT", 8))

# Ground-truth classes.
GOOD = 0.9      # R* at or above this: the flow was right, keep it
BAD = 0.5       # R* below this: merging here damages the output
HALF = 0.5      # "rejected": lost more than half the weight it had


def predict(model, mu, sd, px, baseline):
    f = signed_log_np(features(px, baseline))
    x = torch.from_numpy((f - mu) / sd)
    if x.ndim == 3:
        x = x.permute(2, 0, 1)[None]
    else:
        x = x.view(-1, IN_CH, 1, 1)
    with torch.no_grad():
        q = model(x)
    return q.numpy().reshape(px.shape[:-1]) if px.ndim > 1 else q.numpy().ravel()


def apply_refine(R, q):
    """Exactly what apply_robustness_refinement does, dead zone included."""
    drop = np.clip(1.0 - q, 0.0, 1.0)
    out = np.where(drop <= DEAD, R, R * (1.0 - KAPPA * drop))
    return out.astype(np.float32)


def merge_excess(w, rstar):
    rs = np.clip(rstar, 1e-3, 1.0)
    return ((1.0 + w * w / rs) / (1.0 + w) ** 2) * (1.0 + rs) - 1.0


def stats(R, base, rstar, ok):
    good, bad = ok & (rstar >= GOOD), ok & (rstar < BAD)
    out = {}
    # False rejection: a pixel whose flow was right, pushed below half the
    # weight the unrefined analytic mask gave it.
    out["fr"] = ((R[good] < HALF * np.maximum(base[good], 1e-6)).mean() * 100
                 if good.any() else float("nan"))
    # Missed artifact, strict. Bounded below by the cap: with kappa = 0.75 a
    # pixel keeps at least a quarter of its weight, so this can only fire
    # where the network asks for a reduction of more than two thirds. Kept for
    # continuity, but read w_bad and cut_bad instead.
    out["ms"] = ((R[bad] >= HALF * np.maximum(base[bad], 1e-6)).mean() * 100
                 if bad.any() else float("nan"))
    # Mean weight actually left on the pixels ground truth says to drop. This
    # is the suppression number the cap does not preclude: lower is better.
    out["w_bad"] = R[bad].mean() if bad.any() else float("nan")
    # And how many of them were touched at all.
    out["cut_bad"] = ((R[bad] < base[bad] - 1e-6).mean() * 100
                      if bad.any() else float("nan"))
    out["ch"] = (np.abs(R[ok] - base[ok]) > 1e-6).mean() * 100
    out["mse"] = merge_excess(R[ok], rstar[ok]).mean()
    return out


def main():
    data, H, W, C, NF = load(PREFIX)
    n_train = max(1, NF - HOLDOUT)
    held = list(range(n_train, NF))
    if not held:
        held = [NF - 1]
    print(f"{PREFIX}: {NF} frames, evaluating on held-out {held}")
    print(f"kappa={KAPPA} deadzone={DEAD}\n")

    ckpts = []
    if CKPT:
        ckpts = [CKPT]
    else:
        for a in ("mlp", "cnn"):
            for b in ("geom", "plain"):
                p = os.path.join(SC, f"refinenet_{a}_{b}.pt")
                if os.path.exists(p):
                    ckpts.append(p)
    if not ckpts:
        raise SystemExit("no refinenet_*.pt found; run train_refine.py first")

    models = {}
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = RefineMLP(w=ck["width"]) if ck["arch"] == "mlp" else RefineCNN(w=ck["width"])
        m.load_state_dict(ck["state"]); m.eval()
        models[(ck["arch"], ck["baseline"])] = (m, ck["mu"], ck["sd"])
        print(f"loaded {os.path.basename(p)}: {ck['arch']}, "
              f"trained to refine the {ck['baseline']} baseline, "
              f"{sum(x.numel() for x in m.parameters())} parameters")

    # ---- accumulate over the held-out frames ---------------------------
    acc = {}
    cls_acc = {}
    for fi in held:
        px = np.asarray(data[fi], dtype=np.float32)
        ok = finite(px)
        rstar = np.nan_to_num(visible_rstar(px))
        R_plain = np.nan_to_num(px[..., 0])
        R_geom = np.nan_to_num(px[..., CH_RGEOM])
        full = np.ones_like(R_plain)

        rows = [("1  Wronski only", R_plain, full),
                ("2  Wronski + geometry reject", R_geom, R_plain)]
        for (arch, base), (m, mu, sd) in sorted(models.items()):
            q = predict(m, mu, sd, px, base)
            src = R_geom if base == "geom" else R_plain
            lab = ("4" if base == "geom" else "3") + f"  + NN ({arch}, on {base})"
            rows.append((lab, apply_refine(src, q), src))

        for name, R, base in rows:
            s = stats(R, base, rstar, ok)
            a = acc.setdefault(name, {k: [] for k in s})
            for k, v in s.items():
                if v == v:            # skip NaN
                    a[k].append(v)

    print("\n" + "=" * 90)
    print(f"  mean over {len(held)} held-out frames")
    print("    falseR%   of correctly-aligned pixels pushed below half weight")
    print("    cutBad%   of should-drop pixels the stage reduced at all")
    print("    wBad      mean weight left on should-drop pixels (lower better)")
    print("    chg%      of all pixels moved at all (sparsity)")
    print("    mergeMSE  excess merged-pixel MSE vs the optimal weight")
    print("=" * 90)
    print(f"  {'configuration':<30} {'falseR%':>8} {'cutBad%':>8} {'wBad':>7} "
          f"{'chg%':>7} {'mergeMSE':>9}")
    for name, a in acc.items():
        print(f"  {name:<30} {np.mean(a['fr']):8.3f} {np.mean(a['cut_bad']):8.2f} "
              f"{np.mean(a['w_bad']):7.4f} {np.mean(a['ch']):7.2f} "
              f"{np.mean(a['mse']):9.5f}")

    # ---- protected content ---------------------------------------------
    # The averages above are dominated by whatever the scene is mostly made
    # of. These are the classes where a false rejection actually costs
    # something, isolated so they cannot be averaged away.
    print("\n" + "=" * 78)
    print("  FALSE REJECTION on correctly-aligned content, by class")
    print("  (all of these are pixels the flow got right: rejecting any of")
    print("   them is pure loss)")
    print("=" * 78)
    px = np.asarray(data[held[0]], dtype=np.float32)
    ok = finite(px)
    rstar = np.nan_to_num(visible_rstar(px))
    R_plain, R_geom = np.nan_to_num(px[..., 0]), np.nan_to_num(px[..., CH_RGEOM])
    gmag, coh, lap = px[..., 12], px[..., 15], px[..., 16]
    nsig, bright = px[..., 21], px[..., 20]
    g99 = np.percentile(gmag[ok], 99) if ok.any() else 1.0
    classes = {
        "flat / sky (low gradient)":      gmag < 0.3 * nsig,
        "coherent edge (coh > 0.8)":      (coh > 0.8) & (gmag > 2 * nsig),
        "fine detail (coh < 0.4, strong)": (coh < 0.4) & (gmag > 4 * nsig),
        "thin line (|laplacian| high)":   np.abs(lap) > np.percentile(np.abs(lap[ok]), 99),
        "strongest gradients (top 1%)":   gmag > g99,
        "darkest quarter (noisiest)":     bright < np.percentile(bright[ok], 25),
    }
    hdr = f"  {'class':<32} {'px%':>6}"
    rows = [("1  Wronski only", R_plain, np.ones_like(R_plain)),
            ("2  + geometry reject", R_geom, R_plain)]
    for (arch, base), (m, mu, sd) in sorted(models.items()):
        q = predict(m, mu, sd, px, base)
        src = R_geom if base == "geom" else R_plain
        rows.append((("4" if base == "geom" else "3") + f"  + NN ({arch})",
                     apply_refine(src, q), src))
    for nm, _, _ in rows:
        hdr += f" {nm[:14]:>15}"
    print(hdr)
    for cname, sel in classes.items():
        m = ok & sel & (rstar >= GOOD)
        if m.sum() < 50:
            print(f"  {cname:<32} {'--':>6}   (too few pixels)")
            continue
        line = f"  {cname:<32} {m.mean()*100:6.3f}"
        for _nm, R, base in rows:
            fr = (R[m] < HALF * np.maximum(base[m], 1e-6)).mean() * 100
            line += f" {fr:14.2f}%"
        print(line)

    # ---- by motion regime ------------------------------------------------
    print("\n" + "=" * 78)
    print("  merge MSE by true flow error at the pixel")
    print("  (the claim is that the gain is specific to the sub-pixel band,")
    print("   where one vector per tile cannot represent the motion)")
    print("=" * 78)
    fe = px[..., CH_FLOWERR]
    hdr = f"  {'flow error':<32} {'px%':>6}"
    for nm, _, _ in rows:
        hdr += f" {nm[:14]:>15}"
    print(hdr)
    for lo, hi, lab in [(0, 0.1, "<0.1px  (aligned)"),
                        (0.1, 0.5, "0.1-0.5px  (sub-pixel)"),
                        (0.5, 1.0, "0.5-1px"),
                        (1.0, 3.0, "1-3px"),
                        (3.0, 1e9, ">3px  (gross)")]:
        m = ok & (fe >= lo) & (fe < hi)
        if m.sum() < 50:
            continue
        line = f"  {lab:<32} {m.mean()*100:6.2f}"
        for _nm, R, base in rows:
            line += f" {merge_excess(R[m], rstar[m]).mean():15.5f}"
        print(line)


    # ---- the dead zone / cap trade-off ----------------------------------
    # The target is the inverse-MSE optimal weight, and that is slightly below
    # 1 almost everywhere, because almost every pixel carries SOME residual
    # misalignment. Followed literally the stage would therefore touch most of
    # the frame -- technically optimal, and not what was asked for. The dead
    # zone is the knob that turns it back into a sparse correction, so the
    # honest way to pick it is to print what each value costs and buys rather
    # than to assert a default.
    #
    # Note when reading the change column: a reduction applied uniformly to
    # every comparison frame is close to inert, because the merge normalises
    # num/den and only the DIFFERENCES between frames (and against the
    # reference's own weight) survive. The rows that matter are the ones where
    # the merge MSE moves.
    print()
    print("=" * 78)
    print("  dead zone sweep, best model, held-out frame")
    print("=" * 78)
    print(f"  {'dead':>5} {'kappa':>6} {'chg%':>7} {'falseR%':>8} {'missed%':>8} "
          f"{'mergeMSE':>10}")
    base_mse = merge_excess(R_geom[ok], rstar[ok]).mean()
    print(f"  {'--':>5} {'--':>6} {0.0:7.2f} "
          f"{(R_geom[ok & (rstar >= GOOD)] < HALF * np.maximum(R_plain[ok & (rstar >= GOOD)], 1e-6)).mean()*100:8.2f} "
          f"{(R_geom[ok & (rstar < BAD)] >= HALF * np.maximum(R_plain[ok & (rstar < BAD)], 1e-6)).mean()*100:8.2f} "
          f"{base_mse:10.5f}   (config 2, unrefined)")
    if models:
        (arch, base), (m, mu, sd) = sorted(models.items())[0]
        q = predict(m, mu, sd, px, base)
        src = R_geom if base == "geom" else R_plain
        good, bad = ok & (rstar >= GOOD), ok & (rstar < BAD)
        for kap in (0.5, 0.75, 1.0):
            for dead in (0.0, 0.05, 0.10, 0.20, 0.35):
                drop = np.clip(1.0 - q, 0.0, 1.0)
                R = np.where(drop <= dead, src, src * (1.0 - kap * drop))
                print(f"  {dead:5.2f} {kap:6.2f} "
                      f"{(np.abs(R[ok] - src[ok]) > 1e-6).mean()*100:7.2f} "
                      f"{(R[good] < HALF * np.maximum(src[good], 1e-6)).mean()*100:8.2f} "
                      f"{(R[bad] >= HALF * np.maximum(src[bad], 1e-6)).mean()*100:8.2f} "
                      f"{merge_excess(R[ok], rstar[ok]).mean():10.5f}")


if __name__ == "__main__":
    main()
