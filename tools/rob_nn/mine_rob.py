"""Find what still fools the mask, and write a list the generator can act on.

The loop this belongs to is: train, find what fools it, generate more of that,
retrain. Two kinds of failure are mined, because they cost different things and
a model can trade one for the other without improving:

  FALSE ACCEPTANCE -- a pixel the ground truth says is harmful that the mask
  still trusts. These are the ghosts and the merged-in misalignments.

  FALSE REJECTION -- a pixel that is genuinely safe that the mask throws away.
  These are not free: each one is a frame not averaged, which returns as noise
  and grain. Mining only the first kind produces a mask that rejects
  everything and scores wonderfully on detection.

Writes hard.txt -- corruption pattern ids, repeated in proportion to how badly
each is failing -- which rob_real.cpp reads through ROB_HARD and uses to
oversample those failure types on the next generation pass.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import (RobNet, load_dataset, split_records, PATTERNS, IN_CH,
                       CH_IDEAL_R, CH_FERR, CH_W, PREFIX)

SC = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("ROB_CKPT", os.path.join(SC, "robnet.pt"))
OUT = os.environ.get("ROB_HARD_OUT", os.path.join(SC, "hard.txt"))
ACCEPT = float(os.environ.get("ROB_MINE_ACCEPT", 0.7))   # trusted despite harm
REJECT = float(os.environ.get("ROB_MINE_REJECT", 0.3))   # discarded despite safety


def main():
    data, meta, recs = load_dataset(PREFIX)
    if data is None:
        raise SystemExit(f"no dataset at {PREFIX}")
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    mu, sd = ck["mu"], ck["sd"]
    model = RobNet(); model.load_state_dict(ck["state"]); model.eval()

    train, _, _ = split_records(recs, holdout_burst=ck.get("holdout_burst"))
    n = int(os.environ.get("ROB_MINE_N", 500))
    sample = train[:: max(1, len(train) // n)]

    fa = {}      # pattern -> [harmful pixels, of which wrongly accepted]
    fr = {}      # pattern -> [safe pixels, of which wrongly rejected]
    for r in sample:
        a = np.asarray(data[r["rec"]], dtype=np.float32)
        w = a[..., CH_W] > 0
        if not w.any():
            continue
        with torch.no_grad():
            x = torch.from_numpy(((a[..., :IN_CH] - mu) / sd)).permute(2, 0, 1)[None]
            p = model(x)[0, 0].numpy()
        ideal = a[..., CH_IDEAL_R]
        bad = w & (ideal < 0.5)
        good = w & (ideal > 0.9)
        k = r["pattern"]
        d = fa.setdefault(k, [0, 0]); d[0] += int(bad.sum()); d[1] += int((bad & (p > ACCEPT)).sum())
        d = fr.setdefault(k, [0, 0]); d[0] += int(good.sum()); d[1] += int((good & (p < REJECT)).sum())

    print(f"mined {len(sample)} training records\n")
    print(f"{'pattern':>14} {'harmful px':>11} {'wrongly kept':>13} "
          f"{'safe px':>10} {'wrongly cut':>12}")
    score = {}
    for k in sorted(set(list(fa) + list(fr))):
        h, hk = fa.get(k, [0, 0])
        g, gc = fr.get(k, [0, 0])
        ra = hk / h if h else 0.0
        rr = gc / g if g else 0.0
        nm = PATTERNS[k] if k < len(PATTERNS) else str(k)
        print(f"{nm:>14} {h:>11} {ra * 100:>12.1f}% {g:>10} {rr * 100:>11.1f}%")
        # Weight by both failure modes; a pattern only counts as "hard" if it
        # actually has harmful pixels to get wrong, or safe ones to lose.
        score[k] = ra * (1.0 if h > 2000 else 0.0) + rr * (1.0 if g > 2000 else 0.0)

    tot = sum(score.values())
    if tot <= 0:
        print("\nnothing is failing at these thresholds; not writing a hard list")
        return
    with open(OUT, "w") as f:
        for k, v in sorted(score.items(), key=lambda kv: -kv[1]):
            reps = int(round(24 * v / tot))
            for _ in range(reps):
                f.write(f"{k}\n")
    print(f"\nwrote {OUT} -- pass it to rob_real.cpp as ROB_HARD on the next")
    print("generation pass, then retrain from scratch.")


if __name__ == "__main__":
    main()
