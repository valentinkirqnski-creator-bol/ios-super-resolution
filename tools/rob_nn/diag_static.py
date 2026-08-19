"""Why does the mask reject in static regions? Three candidate causes, measured.

  (1) LABEL BUG   -- the target never says 1.0 for a genuinely safe pixel, so
                     no loss could make the model say it.
  (2) UNDER-CONFIDENCE -- the label does say 1.0 and the model still stops short.
  (3) GATE        -- both are fine and the threshold is simply too high.

These are distinguishable: look at the label's own histogram on zero-harm
pixels, then at the model's histogram on those same pixels.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import (RobNet, load_dataset, split_records, IN_CH,
                       CH_IDEAL_R, CH_HARM, CH_FERR, CH_W, PREFIX)

SC = os.path.dirname(os.path.abspath(__file__))
ck = torch.load(os.environ.get("ROB_CKPT", os.path.join(SC, "robnet.pt")),
                weights_only=False, map_location="cpu")
mu, sd = ck["mu"], ck["sd"]
model = RobNet(); model.load_state_dict(ck["state"]); model.eval()
data, meta, recs = load_dataset(PREFIX)
_, ev, _ = split_records(recs, holdout_burst=ck.get("holdout_burst"))
ev = ev[:: max(1, len(ev) // 200)]

L, I, FE, H = [], [], [], []
for r in ev:
    a = np.asarray(data[r["rec"]], dtype=np.float32)
    w = a[..., CH_W] > 0
    if not w.any():
        continue
    with torch.no_grad():
        x = torch.from_numpy(((a[..., :IN_CH] - mu) / sd)).permute(2, 0, 1)[None]
        p = model(x)[0, 0].numpy()
    L.append(p[w]); I.append(a[..., CH_IDEAL_R][w])
    FE.append(a[..., CH_FERR][w]); H.append(a[..., CH_HARM][w])
L, I, FE, H = map(np.concatenate, (L, I, FE, H))

def hist(name, v):
    print(f"  {name:26s} mean {v.mean():.4f} | >0.99 {(v>0.99).mean()*100:5.1f}% "
          f">0.989 {(v>0.989).mean()*100:5.1f}% >0.98 {(v>0.98).mean()*100:5.1f}% "
          f">0.95 {(v>0.95).mean()*100:5.1f}% >0.90 {(v>0.90).mean()*100:5.1f}%")

print(f"{L.size} held-out pixels\n")
print("(1) THE LABEL on pixels with EXACTLY zero injected flow error")
static = FE == 0.0
print(f"    {static.mean()*100:.1f}% of held-out pixels have zero injected error")
hist("ideal R (label)", I[static])
hist("  ... and zero harm", I[static & (H == 0.0)])
print(f"    label == 1.0 exactly: {(I[static]>=1.0).mean()*100:.1f}% of zero-error pixels")
print()
print("(2) THE MODEL on those same unambiguous pixels")
hist("learned R", L[static])
hist("  ... where label==1.0", L[static & (I >= 1.0)])
print()
print("(3) WHAT THE GATE COSTS on those pixels")
for t in (0.989, 0.98, 0.95, 0.90):
    print(f"    gate {t}: rejects {(L[static] < t).mean()*100:5.1f}% of "
          f"zero-error pixels ({(L[static & (I>=1.0)] < t).mean()*100:5.1f}% of "
          f"provably-safe ones)")
print()
print("(4) WOULD A POST-HOC CALIBRATION FIX IT? (temperature on the logit,")
print("    fitted on held-out data -- preserves ranking, moves absolute values)")
eps = 1e-6
lg = np.log(np.clip(L, eps, 1 - eps) / (1 - np.clip(L, eps, 1 - eps)))
best = None
for T in np.arange(0.30, 1.05, 0.05):
    p = 1.0 / (1.0 + np.exp(-lg / T))
    keep_safe = (p[static & (I >= 1.0)] >= 0.989).mean()
    vis = I < 0.1
    fa = (p[vis] >= 0.989).mean() if vis.any() else 0.0
    if best is None or keep_safe > best[1]:
        best = (T, keep_safe, fa)
    print(f"    T={T:.2f}: provably-safe kept at gate 0.989 {keep_safe*100:5.1f}%, "
          f"visible false-accept {fa*100:5.2f}%")
