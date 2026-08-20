"""Gate -> (visible false-acceptance, safe retention, burst7 static) for ONE
checkpoint, so the shipped default is derived from THIS model's calibration
rather than inherited from a model whose output distribution was different."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import (RobNet, load_dataset, split_records, IN_CH,
                       CH_IDEAL_R, CH_W, PREFIX)
data, meta, recs = load_dataset(PREFIX)
_, ev, _ = split_records(recs)
ev = ev[:: max(1, len(ev) // 300)]
mis = [r for r in ev if r["burst"] != 7]
sta = [r for r in ev if r["burst"] == 7]
ck = torch.load(os.environ["ROB_CKPT"], weights_only=False, map_location="cpu")
m = RobNet(); m.load_state_dict(ck["state"]); m.eval()
mu, sd, drop = ck["mu"], ck["sd"], ck.get("drop_ch", [])
def gather(rs):
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
L, I, _ = gather(mis)
SL, _, SD = gather(sta)
vis = I < 0.1; safe = I > 0.999
det = SD >= 2.5
print(f"checkpoint: {os.path.basename(os.environ['ROB_CKPT'])}")
print(f"{L.size} misalignment px ({vis.mean()*100:.2f}% visibly harmful), "
      f"{SL.size} static px\n")
print(f"{'gate':>7} {'visible FA':>11} {'safe kept':>10} {'b7 detailed':>12} {'b7 ALL':>8}")
for g in (0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
          0.97, 0.98, 0.99, 0.995, 0.999):
    print(f"{g:>7.3f} {(L[vis]>=g).mean()*100:>10.3f}% {(L[safe]>=g).mean()*100:>9.1f}% "
          f"{(SL[det]>=g).mean()*100:>11.1f}% {(SL>=g).mean()*100:>7.1f}%")
print()
for budget in (0.5, 0.25, 0.1, 0.05):
    g = None
    for t in np.linspace(0.0, 0.999999, 20000):
        if (L[vis] >= t).mean() * 100 <= budget: g = t; break
    if g is None: continue
    print(f"  visible FA <= {budget:>5.2f}%  ->  gate {g:.4f}   safe kept {(L[safe]>=g).mean()*100:5.1f}%   "
          f"b7 detailed {(SL[det]>=g).mean()*100:5.1f}%   b7 ALL {(SL>=g).mean()*100:5.1f}%")
