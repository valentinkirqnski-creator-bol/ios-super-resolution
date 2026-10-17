"""Render burst_compare outputs: full previews, masks, and zoomed crops where ones/r0/nn differ most."""
import os, sys
import numpy as np
from PIL import Image

d = sys.argv[1]
Hs, Ws, gh, gw, n, wr, wg, wb = open(os.path.join(d, "meta.txt")).read().split()
Hs, Ws, gh, gw, n = int(Hs), int(Ws), int(gh), int(gw), int(n)
wbv = np.array([float(wr), float(wg), float(wb)], np.float32)
wbv = wbv / wbv[1]
im = {k: np.fromfile(os.path.join(d, k + ".f32"), np.float32).reshape(Hs, Ws, 3) for k in ("ones", "r0", "nn")}
scale = np.percentile(im["r0"] * wbv, 99.5)
tone = lambda x: (np.clip(x * wbv / scale, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
for k, v in im.items():
    Image.fromarray(tone(v)).resize((Ws // 2, Hs // 2), Image.LANCZOS).save(os.path.join(d, f"full_{k}.jpg"), quality=92)
m = np.fromfile(os.path.join(d, "masks.f32"), np.float32).reshape(n - 1, 2, gh, gw)
mr0, mnn = m[:, 0].mean(0), m[:, 1].mean(0)
row = np.concatenate([mr0, mnn, np.clip(0.5 + (mnn - mr0), 0, 1)], 1)
Image.fromarray((row * 255).astype(np.uint8)).save(os.path.join(d, "masks_r0_nn_diff.png"))
print("mean weight per comp frame: analytic", np.round(m[:, 0].mean((1, 2)), 3), " nn", np.round(m[:, 1].mean((1, 2)), 3))
# regions: where merge-all differs most from r0 (ghost candidates), and where nn differs most from r0
def crops(score, tag, k=3, s=192):
    sc = score.copy()
    out = []
    for i in range(k):
        y, x = np.unravel_index(np.argmax(sc), sc.shape)
        y0, x0 = min(max(0, y - s // 2), Hs - s), min(max(0, x - s // 2), Ws - s)
        sc[max(0, y0 - s):y0 + 2 * s, max(0, x0 - s):x0 + 2 * s] = -1
        tiles = [tone(im[kk][y0:y0 + s, x0:x0 + s]) for kk in ("ones", "r0", "nn")]
        out.append(np.concatenate(tiles, 1))
        print(f"{tag} crop {i}: y={y0} x={x0}")
    big = np.concatenate(out, 0)
    Image.fromarray(big).resize((big.shape[1] * 2, big.shape[0] * 2), Image.NEAREST).save(os.path.join(d, f"crops_{tag}.png"))
from scipy.ndimage import uniform_filter
g = lambda a, b: uniform_filter(np.abs(a - b).mean(-1), 31)
crops(g(im["ones"], im["r0"]), "ghost_regions")
crops(g(im["nn"], im["r0"]), "nn_vs_r0")
