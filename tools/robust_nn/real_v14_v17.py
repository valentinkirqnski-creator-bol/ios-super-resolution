"""Real bursts (no ground truth): where do v14 and v17 (shipping cutoffs) disagree most, and what does the merged
image look like there? Columns: no rejection | analytic | v14 | v17, plus the v14 / v17 masks of the frame that
differs most. Inputs: data/real_cmp/<burst>/<v14s|v17s>/ from burst_compare runs with lo/hi = 0.20/0.90.

    python real_v14_v17.py burst9 OUT_DIR
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CROP, ZOOM = 128, 3


def main():
    b, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d14, d17 = os.path.join(HERE, "data", "real_cmp", b, "v14s"), os.path.join(HERE, "data", "real_cmp", b, "v17s")
    m = open(os.path.join(d14, "meta.txt")).read().split()
    H, W, gh, gw, n = int(m[0]), int(m[1]), int(m[2]), int(m[3]), int(m[4])
    wb = np.array([float(m[5]), float(m[6]), float(m[7])], np.float32)
    rd = lambda d, f: np.fromfile(os.path.join(d, f), np.float32).reshape(H, W, 3)
    imgs = {"no rejection": rd(d14, "ones.f32"), "analytic": rd(d14, "r0.f32"), "v14": rd(d14, "nn.f32"),
            "v17": rd(d17, "nn.f32")}
    M14 = np.fromfile(os.path.join(d14, "masks.f32"), np.float32).reshape(n - 1, 2, gh, gw)[:, 1]
    M17 = np.fromfile(os.path.join(d17, "masks.f32"), np.float32).reshape(n - 1, 2, gh, gw)[:, 1]
    sq = lambda x: np.sqrt(np.clip(x, 0, None) + 1e-6)
    e = torch.from_numpy(np.abs(sq(imgs["v14"]) - sq(imgs["v17"])).mean(-1))
    e = F.avg_pool2d(e[None, None], 15, 1, 7)[0, 0]
    h = CROP // 2
    e[:h], e[-h:], e[:, :h], e[:, -h:] = 0, 0, 0, 0
    for ci in range(4):
        k = int(e.argmax())
        cy, cx = k // W, k % W
        e[max(0, cy - CROP):cy + CROP, max(0, cx - CROP):cx + CROP] = 0
        y0, x0 = cy - h, cx - h
        sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        ref = imgs["analytic"][sl] * wb
        s = float(np.percentile(ref, 99.5)) + 1e-9
        tone = lambda x: (np.clip(x[sl] * wb / s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        tiles = [np.kron(tone(v), np.ones((ZOOM, ZOOM, 1), np.uint8)) for v in imgs.values()]
        gy0, gx0 = y0 // 2, x0 // 2
        gsl = (slice(gy0, gy0 + CROP // 2), slice(gx0, gx0 + CROP // 2))
        fk = int(np.abs(M14[(slice(None),) + gsl] - M17[(slice(None),) + gsl]).mean((1, 2)).argmax())
        mk = lambda M: np.repeat(np.kron((np.clip(M[fk][gsl], 0, 1) * 255).astype(np.uint8),
                                         np.ones((2 * ZOOM, 2 * ZOOM), np.uint8))[..., None], 3, 2)
        tiles += [mk(M14), mk(M17)]
        gap = np.full((CROP * ZOOM, 8, 3), 255, np.uint8)
        row = np.concatenate(sum([[t, gap] for t in tiles], [])[:-1], 1)
        img = Image.fromarray(np.concatenate([np.full((40, row.shape[1], 3), 255, np.uint8), row], 0))
        dr = ImageDraw.Draw(img)
        dr.text((4, 4), f"{b} (real burst, no ground truth), crop {ci + 1} at ({x0},{y0}): where v14 and v17 merges "
                        f"differ most; masks of comparison frame {fk + 1}", fill=(0, 0, 0))
        for i, lab in enumerate(list(imgs) + [f"v14 mask (frame {fk + 1})", f"v17 mask (frame {fk + 1})"]):
            dr.text((i * (CROP * ZOOM + 8) + 4, 24), lab, fill=(0, 0, 0))
        img.save(os.path.join(out, f"{b}_crop{ci + 1}.png"))
        print(f"{b} crop {ci + 1} at ({x0},{y0}) frame {fk + 1}: mean mask v14 {M14[fk][gsl].mean():.3f} "
              f"v17 {M17[fk][gsl].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
