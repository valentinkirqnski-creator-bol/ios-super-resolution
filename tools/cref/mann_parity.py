"""Compare mann_parity.exe's windowed mann_refine to torch on the full image. usage: python mann_parity.py ckpt bin"""
import sys
import numpy as np
import torch
from train_mann import MannNet

m = MannNet()
m.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))
m.eval()
raw = np.fromfile(sys.argv[2], dtype=np.int32, count=3)
H, W, ok = (int(v) for v in raw)
v = np.fromfile(sys.argv[2], dtype=np.float32)[3:]
n = H * W
F = v[:11 * n].reshape(11, H, W); G = v[11 * n:12 * n].reshape(H, W)
R0 = v[12 * n:13 * n].reshape(H, W); R1 = v[13 * n:14 * n].reshape(H, W)
with torch.no_grad():
    rn = m(torch.from_numpy(F)[None])[0, 0].numpy()
ref = R0 + G * (rn - R0)
d = np.abs(ref - R1)
print(f"refine ok {ok}  gated px {(G > 0).sum()}  max |torch - app| {d.max():.3g}  mean {d.mean():.3g}  "
      f"changed px {(np.abs(R1 - R0) > 1e-6).sum()}  untouched outside gate: {np.abs(R1 - R0)[G <= 0].max():.3g}")
