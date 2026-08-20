"""burst7 is a STATIC scene: real hand tremor, real aliasing, zero misalignment.
Every pixel is ground-truth safe, so any rejection is a false rejection.
Split by local detail to separate the two candidate causes:
  aliasing confusion -> rejection concentrated in DETAILED regions
  under-confidence   -> rejection everywhere, flat regions included
"""
import os, sys, numpy as np, torch
sys.path.insert(0, sys.argv[3])
from train_rob import RobNet, IN_CH
pref, ck_path = sys.argv[1], sys.argv[2]
d = {}
for line in open(pref + ".dims"):
    k, v = line.split(); d[k] = int(v)
h, w, c, nf = d["h"], d["w"], d["feat_c"], d["frames"]
ck = torch.load(ck_path, weights_only=False, map_location="cpu")
m = RobNet(); m.load_state_dict(ck["state"]); m.eval()
mu, sd = ck["mu"], ck["sd"]
print(f"{os.path.basename(ck_path)}  burst7 static scene, {nf} comparison frames\n")
print(f"{'frame':>5} {'region':>12} {'px%':>6} {'mean R':>8} {'>0.989':>8} {'>0.95':>7} {'>0.90':>7} {'<0.5':>7}")
for k in range(nf):
    f = np.fromfile(f"{pref}_f{k:02d}.feat", dtype=np.float32).reshape(h, w, c)
    with torch.no_grad():
        R = m(torch.from_numpy(((f - mu) / sd)).permute(2, 0, 1)[None])[0, 0].numpy()
    hf = f[..., 17]                      # reference high-frequency energy
    ns = f[..., 12]                      # expected sensor sigma
    std = f[..., 3:6].mean(-1)
    detail = std / np.maximum(ns, 1e-9)  # local contrast in units of noise
    bins = [("flat", detail < 1.0), ("moderate", (detail >= 1.0) & (detail < 2.5)),
            ("detailed", detail >= 2.5), ("ALL", np.ones_like(detail, bool))]
    for nm, msk in bins:
        if msk.sum() < 1000: continue
        r = R[msk]
        print(f"{k:>5} {nm:>12} {msk.mean()*100:5.1f}% {r.mean():8.4f} "
              f"{(r>0.989).mean()*100:7.1f}% {(r>0.95).mean()*100:6.1f}% "
              f"{(r>0.90).mean()*100:6.1f}% {(r<0.5).mean()*100:6.1f}%")
    print()
