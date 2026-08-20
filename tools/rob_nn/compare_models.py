"""Both halves of the user's requirement, for every candidate, side by side.

Improving one at the other's expense is the same curve re-parameterised and is
not progress, so the two are always reported together:

  MISALIGNMENT (bursts 1-6, which contain corrupted flow)
    visible false-acceptance at the 0.989 gate, and the safe-pixel retention
    available at a matched <=0.1% visible false-acceptance budget
  OVER-REJECTION (burst7, static: every pixel ground-truth safe)
    what fraction is kept, split by local detail, since aliasing confusion
    shows up as detail-dependent rejection and under-confidence does not
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import (RobNet, load_dataset, split_records, IN_CH,
                       CH_IDEAL_R, CH_FERR, CH_W, PREFIX)

data, meta, recs = load_dataset(PREFIX)
_, ev, _ = split_records(recs)
ev = ev[:: max(1, len(ev) // int(os.environ.get("ROB_EVAL_N", 260)))]
mis = [r for r in ev if r["burst"] != 7]
sta = [r for r in ev if r["burst"] == 7]
print(f"eval: {len(mis)} records from bursts 1-6, {len(sta)} from burst7 (static)\n")

def gather(rs, model, mu, sd):
    L, I, D = [], [], []
    for r in rs:
        a = np.asarray(data[r["rec"]], dtype=np.float32)
        w = a[..., CH_W] > 0
        if not w.any():
            continue
        with torch.no_grad():
            x = torch.from_numpy(((a[..., :IN_CH] - mu) / sd)).permute(2, 0, 1)[None]
            p = model(x)[0, 0].numpy()
        L.append(p[w]); I.append(a[..., CH_IDEAL_R][w])
        D.append((a[..., 3:6].mean(-1) / np.maximum(a[..., 12], 1e-9))[w])
    return map(np.concatenate, (L, I, D))

rows = []
for name, path in (("symmetric", "robnet_symmetric.pt"),
                   ("asymmetric", "robnet.pt"),
                   ("asym + burst7", os.environ.get("ROB_NEW", "robnet.pt"))):
    p = path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(p):
        continue
    ck = torch.load(p, weights_only=False, map_location="cpu")
    m = RobNet(); m.load_state_dict(ck["state"]); m.eval()
    mu, sd = ck["mu"], ck["sd"]
    L, I, _ = gather(mis, m, mu, sd)
    vis = I < 0.1
    safe = I > 0.999
    fa_at_gate = (L[vis] >= 0.989).mean() * 100
    # threshold achieving <= 0.1% visible false-acceptance
    t_hit, keep = None, 0.0
    for t in np.linspace(0.0, 0.99999, 5000):
        if (L[vis] >= t).mean() * 100 <= 0.1:
            t_hit = t; keep = (L[safe] >= t).mean() * 100; break
    SL, SI, SD = gather(sta, m, mu, sd)
    b = {}
    for nm, msk in (("flat", SD < 1.0), ("moderate", (SD >= 1.0) & (SD < 2.5)),
                    ("detailed", SD >= 2.5)):
        b[nm] = ((SL[msk] >= 0.989).mean() * 100, SL[msk].mean()) if msk.sum() > 500 else (float("nan"),) * 2
    rows.append((name, fa_at_gate, t_hit, keep, b, SL.mean()))

print("MISALIGNMENT (bursts 1-6)                    OVER-REJECTION (burst7 static, kept at gate 0.989)")
print(f"{'model':>15} {'visFA@.989':>11} {'safe kept':>10} {'':4} "
      f"{'b7 mean R':>10} {'flat':>8} {'moderate':>9} {'detailed':>9}")
print(f"{'':15} {'':11} {'@<=0.1%FA':>10}")
for name, fa, t, keep, b, sm in rows:
    print(f"{name:>15} {fa:10.2f}% {keep:9.1f}% {'':4} {sm:10.4f} "
          f"{b['flat'][0]:7.1f}% {b['moderate'][0]:8.1f}% {b['detailed'][0]:8.1f}%")
print("\nvisFA@.989  = visible misalignment still merged at the shipped gate (lower better)")
print("safe kept   = provably-safe pixels retained at a matched <=0.1% visible-FA budget")
print("b7 columns  = fraction of STATIC pixels clearing the gate (higher better; all are safe)")
