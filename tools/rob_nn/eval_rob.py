"""Fair head-to-head: ROC over the decision threshold, so neither mask is
judged at an operating point that happens to favour it.

Both masks emit R in [0,1] and the merge weights by it, so sweeping a
threshold on R traces each mask's full detection/false-positive trade-off.
Comparing areas under those curves answers "which statistic carries more
information about merge safety", independent of tuning.
"""
import os, sys
import numpy as np
import torch
from train_rob import (RobNet, classical_R, data, n_frames, n_train, IN_CH, GH, GW)

SC = os.path.dirname(os.path.abspath(__file__))
ck = torch.load(os.path.join(SC, "robnet.pt"), weights_only=False)
mu, sd = ck["mu"], ck["sd"]
model = RobNet(); model.load_state_dict(ck["state"]); model.eval()

def auc_rank(R, bad, cap=400000, seed=0):
    """Mann-Whitney AUC: the probability that a randomly chosen harmful pixel
    is scored lower (less trusted) than a randomly chosen good one, with ties
    counted as half.

    Rank-based rather than area-under-a-swept-threshold, because a mask that
    pins most pixels at exactly R=1 cannot reach high false-positive rates by
    thresholding at all -- its swept curve is truncated, and integrating it
    measures the truncation rather than the mask.
    """
    rng = np.random.RandomState(seed)
    b = R[bad].ravel(); g = R[~bad].ravel()
    if len(b) > cap: b = b[rng.choice(len(b), cap, replace=False)]
    if len(g) > cap: g = g[rng.choice(len(g), cap, replace=False)]
    s = np.concatenate([-b, -g])            # higher score = less trusted
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tie groups (R saturated at 1 makes ties the norm)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    nb, ng = len(b), len(g)
    U = ranks[:nb].sum() - nb * (nb + 1) / 2.0
    return U / (nb * ng)

def det_at_fp(R, bad, target):
    """Detection rate when the accept threshold is set to spend exactly
     of the good pixels as false rejections."""
    g = R[~bad].ravel()
    th = np.quantile(g, target)             # reject the lowest-R good fraction
    return float((R[bad] < th).mean())

frames = list(range(n_train, n_frames))[:3]   # held-out reference only
rows = []
for fi in frames:
    ev = np.asarray(data[fi, ::2, ::2, :], dtype=np.float32)
    bad = ev[..., 14] < 0.5
    if not bad.any():
        continue
    with torch.no_grad():
        xin = torch.from_numpy(((ev[..., :IN_CH] - mu) / sd)).permute(2, 0, 1)[None]
        learned = model(xin)[0, 0].numpy()
    out = {"frame": fi, "bad%": bad.mean() * 100}
    for name, R in (("classical", classical_R(ev)), ("learned", learned)):
        out[name] = (auc_rank(R, bad),
                     det_at_fp(R, bad, 0.02), det_at_fp(R, bad, 0.10))
    rows.append(out)

print(f"{'frame':>6} {'harmful':>8} | {'classical AUC':>13} {'@2%FP':>7} {'@10%FP':>7}"
      f" | {'learned AUC':>11} {'@2%FP':>7} {'@10%FP':>7}")
for r in rows:
    c, l = r["classical"], r["learned"]
    print(f"{r['frame']:>6} {r['bad%']:>7.2f}% | {c[0]:>13.3f} {c[1]*100:>6.1f}% "
          f"{c[2]*100:>6.1f}% | {l[0]:>11.3f} {l[1]*100:>6.1f}% {l[2]*100:>6.1f}%")

ca = np.mean([r["classical"][0] for r in rows])
la = np.mean([r["learned"][0] for r in rows])
print(f"\nmean AUC  classical {ca:.3f}   learned {la:.3f}")
print("(AUC 0.5 = no information; 1.0 = perfect separation of harmful pixels)")
