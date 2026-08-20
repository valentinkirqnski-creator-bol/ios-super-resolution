"""How much does the mask lean on channels 13-14 -- the analytic mask's own
answer? Those are built from |ref_mean - comp_mean|, which under perfect
alignment is ALIASING. The analytic mask cannot tell aliasing from
misalignment, so whatever the network borrows from 13-14 it inherits that
blindness with.

Replace a channel by its training mean (its normalised value 0) and see how
far R moves. A large move on a STATIC scene means the decision is riding on a
statistic that is firing on aliasing."""
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[3])
from train_rob import RobNet, IN_CH
pref, ck_path = sys.argv[1], sys.argv[2]
d = {}
for line in open(pref + ".dims"):
    k, v = line.split(); d[k] = int(v)
h, w, c = d["h"], d["w"], d["feat_c"]
ck = torch.load(ck_path, weights_only=False, map_location="cpu")
m = RobNet(); m.load_state_dict(ck["state"]); m.eval()
mu, sd = ck["mu"], ck["sd"]
f = np.fromfile(f"{pref}_f00.feat", dtype=np.float32).reshape(h, w, c)
x0 = ((f - mu) / sd).astype(np.float32)
std = f[..., 3:6].mean(-1); ns = f[..., 12]
detail = std / np.maximum(ns, 1e-9)
det = detail >= 2.5
def run(x):
    with torch.no_grad():
        return m(torch.from_numpy(x).permute(2, 0, 1)[None])[0, 0].numpy()
base = run(x0)
print(f"{ck_path.split(chr(92))[-1].split('/')[-1]}   (burst7, static)")
print(f"  {'ablated':>22} {'mean R all':>11} {'mean R detailed':>16} {'<0.5 detailed':>14}")
print(f"  {'none (baseline)':>22} {base.mean():11.4f} {base[det].mean():16.4f} "
      f"{(base[det]<0.5).mean()*100:13.1f}%")
for name, chans in (("ch13 log(d2/s2)", [13]), ("ch14 analytic R", [14]),
                    ("ch13+14 both", [13, 14]), ("ch6-8 comp mean", [6, 7, 8]),
                    ("ch15 flow residual", [15]), ("ch17 ref HF", [17])):
    x = x0.copy()
    for ci in chans:
        x[..., ci] = 0.0            # normalised mean
    r = run(x)
    print(f"  {name:>22} {r.mean():11.4f} {r[det].mean():16.4f} "
          f"{(r[det]<0.5).mean()*100:13.1f}%   (delta detailed {r[det].mean()-base[det].mean():+.4f})")
