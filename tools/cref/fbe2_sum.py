"""Sum fb_eval2 stats (one-token config names)."""
import glob
from collections import defaultdict

acc = defaultdict(lambda: [0.0] * 8)
for p in glob.glob("fbe2run/*.txt"):
    for ln in open(p):
        t = ln.split()
        if len(t) != 11:
            continue
        key = (int(t[0]), t[1], t[2])
        for i, v in enumerate(t[3:11]):
            acc[key][i] += float(v)
order = ["FB4x4", "FB4x4+along", "CONS", "CONS+along", "FB+CONS"]
print("rejected = confidence < 0.5. caught = misaligned px (ghost > 4 sigma); wrong = good detail px (ghost < 1 sigma)")
for v, vn in ((0, "Wronski"), (1, "GCam")):
    print(f"\n{vn}  (all px | only where the mask KEPT the px)")
    for tau in ("0.75", "0.20"):
        for name in order:
            a = acc.get((v, name, tau))
            if not a:
                continue
            nb, cb, nok, cf, nbk, cbk, nokk, cfk = a
            print(f"  tau {tau} {name:12s} | caught {100*cb/max(1,nb):5.1f}%  wrong {100*cf/max(1,nok):5.1f}% | "
                  f"kept: caught {100*cbk/max(1,nbk):5.1f}%  wrong {100*cfk/max(1,nokk):5.1f}%")
