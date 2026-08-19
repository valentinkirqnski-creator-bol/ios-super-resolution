"""Operating-point curve: what does driving harmful acceptance to zero cost?

The user's requirement is asymmetric -- a merged misalignment is unacceptable,
a forgone merge is acceptable -- so the headline number is the harmful
false-ACCEPTANCE rate, not AUC and not balanced detection.

Thresholding is a hard gate: R >= t merges with its own weight, R < t merges
nothing. The floor is well defined. At t > 1 nothing merges and the output is
exactly the reference frame: no misalignment, no multi-frame benefit. So "no
misalignments" is always reachable; the only question this table answers is how
much merging survives on the way there.

"Visible" harm is broken out separately because it is the thing the user can
actually see. Two photometrically indistinguishable regions merged together
produce no visible artefact, so the undetectable cases are largely the
invisible ones; what must be driven to zero is the harm big enough to show.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import (RobNet, classical_R, load_dataset, split_records,
                       IN_CH, CH_IDEAL_R, CH_W, PREFIX)

SC = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("ROB_CKPT", os.path.join(SC, "robnet.pt"))
OUTF = os.environ.get("ROB_SWEEP_OUT", os.path.join(SC, "sweep.txt"))

data, meta, recs = load_dataset(PREFIX)
ck = torch.load(CKPT, weights_only=False, map_location="cpu")
mu, sd = ck["mu"], ck["sd"]
model = RobNet(); model.load_state_dict(ck["state"]); model.eval()
_, ev, _ = split_records(recs, holdout_burst=ck.get("holdout_burst"))
cap = int(os.environ.get("ROB_EVAL_N", 200))
if len(ev) > cap:
    ev = ev[:: max(1, len(ev) // cap)]

L, I, A = [], [], []
for r in ev:
    a = np.asarray(data[r["rec"]], dtype=np.float32)
    w = a[..., CH_W] > 0
    if not w.any():
        continue
    with torch.no_grad():
        x = torch.from_numpy(((a[..., :IN_CH] - mu) / sd)).permute(2, 0, 1)[None]
        p = model(x)[0, 0].numpy()
    L.append(p[w]); I.append(a[..., CH_IDEAL_R][w]); A.append(classical_R(a)[w])
L = np.concatenate(L); I = np.concatenate(I); A = np.concatenate(A)

harmful = I < 0.5          # merging here damages the output
visible = I < 0.1          # ... and damages it enough to see
safe = I > 0.9
lines = []
def emit(s):
    print(s); lines.append(s)

emit(f"held-out records {len(ev)}, {L.size} pixels, "
     f"{harmful.mean()*100:.2f}% harmful, {visible.mean()*100:.2f}% visibly harmful")
emit("")
emit("  thresh | harmful FALSE-ACCEPT | visible FALSE-ACCEPT | pixels merged | mean R")
emit("  -------+----------------------+----------------------+---------------+-------")
for t in (0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
          0.95, 0.98, 0.99, 0.995, 0.999):
    keep = L >= t
    fa = keep[harmful].mean() * 100 if harmful.any() else float("nan")
    fv = keep[visible].mean() * 100 if visible.any() else float("nan")
    merged = keep.mean() * 100
    meanr = np.where(keep, L, 0.0).mean()
    emit(f"  {t:6.3f} | {fa:19.2f}% | {fv:19.2f}% | {merged:12.1f}% | {meanr:6.3f}")

emit("")
for tgt, nm in ((1.0, "1%"), (0.5, "0.5%"), (0.1, "0.1%"), (0.0, "0%")):
    ts = np.linspace(0.0, 0.9999, 4000)
    hit = None
    for t in ts:
        keep = L >= t
        if keep[visible].mean() * 100 <= tgt:
            hit = (t, keep[visible].mean() * 100, keep[harmful].mean() * 100,
                   keep.mean() * 100, keep[safe].mean() * 100)
            break
    if hit:
        emit(f"  visible false-acceptance <= {nm:5s}: threshold {hit[0]:.4f} -> "
             f"visible {hit[1]:.3f}%, all-harmful {hit[2]:.2f}%, "
             f"{hit[3]:.1f}% of pixels merged ({hit[4]:.1f}% of SAFE pixels kept)")
    else:
        emit(f"  visible false-acceptance <= {nm:5s}: NOT REACHABLE by thresholding")

# analytic baseline at its own natural cut, for scale
keep = A >= 0.5
emit("")
emit(f"  analytic mask at R>=0.5: visible false-accept {keep[visible].mean()*100:.2f}%, "
     f"{keep.mean()*100:.1f}% of pixels merged")
open(OUTF, "w").write("\n".join(lines) + "\n")
print(f"\nsaved {OUTF}")
