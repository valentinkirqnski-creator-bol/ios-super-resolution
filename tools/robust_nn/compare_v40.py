"""Compare v40-family runs against v21 from their training logs (train_v17.py report lines).

    python compare_v40.py runs/v40_train.log [runs/v40b_train.log ...] [--step 10000]
For each run: ONE operating point for all sets (a logit shift = weights-only bias fold), chosen as the shift whose
summed misses over every held-out set do not exceed v21's while losing the least good weight (over-rejection summed).
Then per set: v21 vs the run at that shift - missed (misaligned px kept), small (misses with true error < 1 raw px,
i.e. visible only on an edge), rot, over (aligned px rejected), flat (aligned flat px rejected), lost (good weight lost),
leak<.05 (weight left on clearly misaligned px)."""
import argparse
import re

PAT = re.compile(r"^\[(\S+)\] (.+?)\s+sim ([\d.]+)\s+boundary err ([\d.]+)\s+bal ([\d.]+)\s+missed ([\d.]+)\s+"
                 r"over ([\d.]+)\s+small ([\d.]+)\s+rot ([\d.]+)\s+flat ([\d.]+)\s+leak ([\d.]+)\s+leak<\.05 ([\d.]+)"
                 r"\s+good-weight lost ([\d.]+)%")
KEYS = ["sim", "bnd", "bal", "missed", "over", "small", "rot", "flat", "leak", "leak05", "lost"]


def parse(path):
    base, runs = {}, {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = PAT.match(line.strip())
        if not m:
            continue
        vset, who = m.group(1), m.group(2).strip()
        vals = dict(zip(KEYS, (float(x) for x in m.groups()[2:])))
        parts = who.split()
        if len(parts) == 3 and parts[1].startswith("s"):
            runs.setdefault(int(parts[1][1:]), {}).setdefault(float(parts[2]), {})[vset] = vals
        elif len(parts) == 1:
            base[vset] = vals
    return base, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--step", type=int, default=0, help="checkpoint step (default: the last one logged)")
    a = ap.parse_args()
    for path in a.logs:
        base, runs = parse(path)
        if not runs:
            print(path, ": no results yet")
            continue
        step = a.step or max(runs)
        rs = runs[step]
        sets = [s for s in base if all(s in rs[sh] for sh in rs)]
        bm = sum(base[s]["missed"] for s in sets)
        ok = [sh for sh in rs if sum(rs[sh][s]["missed"] for s in sets) <= bm]
        pick = min(ok, key=lambda sh: sum(rs[sh][s]["over"] for s in sets)) if ok else min(
            rs, key=lambda sh: sum(rs[sh][s]["missed"] for s in sets))
        print(f"\n== {path}  step {step}  shift {pick:+.3f}" + ("" if ok else "  (NO shift reaches v21's total misses)"))
        print(f"{'set':12s} | {'missed':>13s} | {'small<1px':>13s} | {'rot':>13s} | {'over':>13s} | {'flat':>13s} | "
              f"{'lost%':>13s} | {'leak<.05':>13s}")
        tot = {k: [0.0, 0.0] for k in ("missed", "small", "rot", "over", "flat", "lost")}
        for s in sets:
            b, r = base[s], rs[pick][s]
            cells = []
            for k in ("missed", "small", "rot", "over", "flat", "lost", "leak05"):
                cells.append(f"{b[k]:6.2f} {r[k]:6.2f}" if k != "leak05" else f"{b[k]:6.3f} {r[k]:6.3f}")
                if k in tot:
                    tot[k][0] += b[k]; tot[k][1] += r[k]
            print(f"{s:12s} | " + " | ".join(cells))
        n = max(1, len(sets))
        print(f"{'MEAN':12s} | " + " | ".join(f"{tot[k][0] / n:6.2f} {tot[k][1] / n:6.2f}" for k in
                                           ("missed", "small", "rot", "over", "flat", "lost")) + " |   (v21 | run)")


if __name__ == "__main__":
    main()
