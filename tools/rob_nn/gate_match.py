"""If the new model is gated to the OLD model's misalignment rate, does the
static win survive? That is the only way to satisfy both requirements at once:
no regression on misalignment, and no over-rejection on static content."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import (RobNet, load_dataset, split_records, IN_CH,
                       CH_IDEAL_R, CH_W, PREFIX)
data, meta, recs = load_dataset(PREFIX)
_, ev, _ = split_records(recs)
ev = ev[:: max(1, len(ev) // 260)]
mis = [r for r in ev if r["burst"] != 7]
sta = [r for r in ev if r["burst"] == 7]
def gather(rs, m, mu, sd, drop=()):
    L, I, D = [], [], []
    for r in rs:
        a = np.asarray(data[r["rec"]], dtype=np.float32)
        w = a[..., CH_W] > 0
        if not w.any(): continue
        with torch.no_grad():
            xn = ((a[..., :IN_CH] - mu) / sd)
            for c in drop:
                xn[..., c] = 0.0
            x = torch.from_numpy(xn).permute(2, 0, 1)[None]
            p = m(x)[0, 0].numpy()
        L.append(p[w]); I.append(a[..., CH_IDEAL_R][w])
        D.append((a[..., 3:6].mean(-1) / np.maximum(a[..., 12], 1e-9))[w])
    return map(np.concatenate, (L, I, D))
def load(p):
    ck = torch.load(p, weights_only=False, map_location="cpu")
    m = RobNet(); m.load_state_dict(ck["state"]); m.eval()
    return m, ck["mu"], ck["sd"], ck.get("drop_ch", [])
here = os.path.dirname(os.path.abspath(__file__))
sym = load(os.environ.get("ROB_BASE", os.path.join(here, "robnet_symmetric.pt")))
new = load(os.environ["ROB_NEW"])
Ls, Is, _ = gather(mis, *sym)
TARGET = (Ls[Is < 0.1] >= 0.989).mean() * 100
print(f"BASELINE visible false-acceptance at gate 0.989: {TARGET:.3f}%")
Ln, In, _ = gather(mis, *new)
vis = In < 0.1; safe = In > 0.999
g = None
for t in np.linspace(0.0, 0.999999, 20000):
    if (Ln[vis] >= t).mean() * 100 <= TARGET:
        g = t; break
print(f"gate on the new model that matches it:               {g:.5f}")
print(f"  new model visible FA there:  {(Ln[vis] >= g).mean()*100:.3f}%")
print(f"  safe pixels kept (bursts1-6): {(Ln[safe] >= g).mean()*100:.1f}%  "
      f"(baseline at 0.989: {(Ls[Is>0.999] >= 0.989).mean()*100:.1f}%)")
SLn, _, SDn = gather(sta, *new)
SLs, _, SDs = gather(sta, *sym)
print(f"\nburst7 STATIC retention (all pixels safe), fraction clearing the gate:")
print(f"  {'region':>10} {'baseline@0.989':>16} {'new@%.5f' % g:>16}")
for nm, lo, hi in (("flat", 0, 1.0), ("moderate", 1.0, 2.5), ("detailed", 2.5, 1e9)):
    ms, mn = (SDs >= lo) & (SDs < hi), (SDn >= lo) & (SDn < hi)
    if ms.sum() < 500: continue
    print(f"  {nm:>10} {(SLs[ms] >= 0.989).mean()*100:15.1f}% {(SLn[mn] >= g).mean()*100:15.1f}%")
print(f"  {'ALL':>10} {(SLs >= 0.989).mean()*100:15.1f}% {(SLn >= g).mean()*100:15.1f}%")
