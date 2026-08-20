"""Find what still fools the correction, and write a list the generator acts on.

The loop this belongs to is: train, find what fools it, generate more of that,
retrain. Two kinds of failure are mined, because they cost different things and
a model can trade one for the other without improving at all:

  MISSED CORRECTION -- a pixel the ground truth says merging would damage, that
  R_normal still trusts, and that C leaves alone. These are the ghosts and the
  duplicated edges. Mined only where R_normal is high: where the analytic mask
  has already rejected the pixel there is nothing for C to fix, and counting
  those would reward the model for piling on.

  NEEDLESS CORRECTION -- a pixel that is correctly aligned, that C pulls down
  anyway. Not free: each one is a frame not averaged, which comes back as
  noise and grain. Mining only the first kind produces a correction that
  darkens everything and scores wonderfully on detection.

Writes hard.txt -- corruption pattern ids repeated in proportion to how badly
each is failing -- which rob_real.cpp reads through ROB_HARD and uses to
oversample those failure types on the next generation pass. Hard GOOD examples
are fed back the same way: a pattern whose PAIRED CLEAN phase is being pulled
down is listed too, and every corrupted variant of it brings its clean twin
with it, so oversampling the pattern oversamples both halves.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import (RobNet, load_dataset, split_records, PATTERNS, IN_CH,
                       CH_IDEAL_R, CH_W, CH_RNORM, PREFIX)

SC = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("ROB_CKPT", os.path.join(SC, "robnet.pt"))
OUT = os.environ.get("ROB_HARD_OUT", os.path.join(SC, "hard.txt"))
# "Still merged": R_final is above half, so the artefact reaches the output.
KEEP = float(os.environ.get("ROB_MINE_KEEP", 0.5))
# "Needlessly held back": C took more than a tenth off a provably safe pixel.
CUT = float(os.environ.get("ROB_MINE_CUT", 0.9))


def main():
    data, meta, recs = load_dataset(PREFIX)
    if data is None:
        raise SystemExit(f"no dataset at {PREFIX}")
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    mu, sd = ck["mu"], ck["sd"]
    drop_ch = ck.get("drop_ch", [])
    model = RobNet(cin=ck.get("in_ch", IN_CH))
    model.load_state_dict(ck["state"]); model.eval()

    train, _, _ = split_records(recs, holdout_burst=ck.get("holdout_burst"))
    n = int(os.environ.get("ROB_MINE_N", 400))
    sample = train[:: max(1, len(train) // n)]

    miss = {}    # pattern -> [harmful & trusted by R_normal, of which still merged]
    over = {}    # pattern -> [provably safe px, of which C pulled down]
    for r in sample:
        a = np.asarray(data[r["rec"]], dtype=np.float32)
        w = a[..., CH_W] > 0
        if not w.any():
            continue
        with torch.no_grad():
            x = torch.from_numpy(((a[..., :IN_CH] - mu) / sd)).permute(2, 0, 1)[None]
            for c in drop_ch:
                x[:, c] = 0.0
            p = model(x)[0, 0].numpy()
        ideal = a[..., CH_IDEAL_R]
        rnorm = a[..., CH_RNORM]
        final = rnorm * p
        missed = w & (ideal < 0.5) & (rnorm >= 0.5)
        clean = w & (ideal > 0.999)
        k = r["pattern"]
        d = miss.setdefault(k, [0, 0])
        d[0] += int(missed.sum()); d[1] += int((missed & (final > KEEP)).sum())
        d = over.setdefault(k, [0, 0])
        d[0] += int(clean.sum()); d[1] += int((clean & (p < CUT)).sum())

    print(f"mined {len(sample)} training records\n")
    print(f"{'pattern':>14} {'harmful & trusted':>18} {'still merged':>13} "
          f"{'correct px':>11} {'needlessly cut':>15}")
    score = {}
    for k in sorted(set(list(miss) + list(over))):
        h, hk = miss.get(k, [0, 0])
        g, gc = over.get(k, [0, 0])
        ra = hk / h if h else 0.0
        rr = gc / g if g else 0.0
        nm = PATTERNS[k] if k < len(PATTERNS) else str(k)
        print(f"{nm:>14} {h:>18} {ra * 100:>12.1f}% {g:>11} {rr * 100:>14.3f}%")
        # A pattern counts as hard only if it has enough of the relevant pixels
        # to get wrong. The false-rejection term is scaled up hard because it
        # is meant to stay two orders of magnitude smaller than the other, so
        # at equal weight it would never influence the mix.
        score[k] = (ra if h > 2000 else 0.0) + 20.0 * (rr if g > 20000 else 0.0)

    tot = sum(score.values())
    if tot <= 0:
        print("\nnothing is failing at these thresholds; not writing a hard list")
        return
    with open(OUT, "w") as f:
        for k, v in sorted(score.items(), key=lambda kv: -kv[1]):
            for _ in range(int(round(24 * v / tot))):
                f.write(f"{k}\n")
    print(f"\nwrote {OUT} -- pass it to rob_real.cpp as ROB_HARD on the next")
    print("generation pass, then retrain from scratch.")


if __name__ == "__main__":
    main()
