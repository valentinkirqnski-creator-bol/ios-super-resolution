"""Both models swept to the SAME visible-false-acceptance budgets, so the
comparison is never at an operating point that happens to favour one."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import (RobNet, load_dataset, split_records, IN_CH,
                       CH_IDEAL_R, CH_W, PREFIX)
data, meta, recs = load_dataset(PREFIX)
_, ev, _ = split_records(recs)
ev = ev[:: max(1, len(ev) // 300)]
mis = [r for r in ev if r["burst"] != 7]
sta = [r for r in ev if r["burst"] == 7]
def gather(rs, m, mu, sd, drop):
    L, I, D = [], [], []
    for r in rs:
        a = np.asarray(data[r["rec"]], dtype=np.float32)
        w = a[..., CH_W] > 0
        if not w.any(): continue
        xn = ((a[..., :IN_CH] - mu) / sd)
        for c in drop: xn[..., c] = 0.0
        with torch.no_grad():
            p = m(torch.from_numpy(xn).permute(2, 0, 1)[None])[0, 0].numpy()
        L.append(p[w]); I.append(a[..., CH_IDEAL_R][w])
        D.append((a[..., 3:6].mean(-1) / np.maximum(a[..., 12], 1e-9))[w])
    return map(np.concatenate, (L, I, D))
def load(p):
    ck = torch.load(p, weights_only=False, map_location="cpu")
    m = RobNet(); m.load_state_dict(ck["state"]); m.eval()
    return m, ck["mu"], ck["sd"], ck.get("drop_ch", [])
models = [("no match-quality", os.environ["ROB_BASE"]),
          ("with ch18-19", os.environ["ROB_NEW"])]
print(f"{'budget':>8} {'model':>18} {'gate':>8} {'safe kept':>10} "
      f"{'b7 flat':>8} {'b7 mod':>8} {'b7 detail':>10} {'b7 ALL':>8}")
for budget in (1.0, 0.5, 0.25, 0.1):
    for nm, path in models:
        m, mu, sd, drop = load(path)
        L, I, _ = gather(mis, m, mu, sd, drop)
        vis = I < 0.1; safe = I > 0.999
        g = None
        for t in np.linspace(0.0, 0.999999, 12000):
            if (L[vis] >= t).mean() * 100 <= budget: g = t; break
        if g is None: continue
        SL, _, SD = gather(sta, m, mu, sd, drop)
        f = lambda lo, hi: (SL[(SD >= lo) & (SD < hi)] >= g).mean() * 100
        print(f"{budget:>7.2f}% {nm:>18} {g:>8.5f} {(L[safe]>=g).mean()*100:>9.1f}% "
              f"{f(0,1.0):>7.1f}% {f(1.0,2.5):>7.1f}% {f(2.5,1e9):>9.1f}% "
              f"{(SL>=g).mean()*100:>7.1f}%")
    print()
