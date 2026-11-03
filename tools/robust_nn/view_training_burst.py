"""Render one training burst for viewing.
    python view_training_burst.py data/rp7_train/rp00200005.npz [out_dir]"""
import os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_protocol import run_tp_real
from target_protocol import combine, r_index

f = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else 'C:/Users/valen/Downloads/robust_nn_outputs/training_burst'
os.makedirs(out, exist_ok=True)
z = np.load(f)
raw, gt = z['raw'], z['gt_flow']
cfa, wb = z['cfa'], z['wb']
N, H, W = raw.shape


def to_rgb(r):
    q = {}
    for i in range(2):
        for j in range(2):
            q.setdefault(int(cfa[i][j]), []).append(r[i::2, j::2])
    return np.stack([np.mean(q[c], 0) for c in range(3)], -1)


s = np.percentile(to_rgb(raw[0]), 99.5)
tone = lambda x: (np.clip(x / s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
up = lambda a: np.kron(a, np.ones((2, 2, 1), np.uint8)) if a.ndim == 3 else np.kron(a, np.ones((2, 2), np.uint8))
frames = [Image.fromarray(up(tone(to_rgb(raw[k])))) for k in range(N)]
frames[0].save(os.path.join(out, 'frames.gif'), save_all=True, append_images=frames[1:] + frames[:1], duration=500, loop=0)
for k, fr in enumerate(frames):
    fr.save(os.path.join(out, f'frame_{k}.png'))

d = run_tp_real(raw, gt, float(z['alpha']), float(z['beta']), 12345, True, cfa, wb)
gtt = torch.from_numpy(gt[1:])
err = (d['flow'] - gtt).norm(dim=1)
heat = lambda m, vmax: (np.clip(m / vmax, 0, 1) * 255).astype(np.uint8)
rows = []
for k in range(N - 1):
    mag = gtt[k].norm(dim=0).numpy()
    tiles = [heat(mag - mag.min(), max(1e-6, np.ptp(mag))),       # true motion (relative): rotated patches stand out
             heat(err[k].numpy(), 2.0),                             # aligner+corruption error, white = 2+ px
             (d['corr'][k].numpy() * 255).astype(np.uint8),         # deliberately corrupted tiles
             (d['r0'][k].numpy() * 255).astype(np.uint8)]           # the app's analytic mask
    sep = np.full((H // 2, 4), 128, np.uint8)
    rows.append(np.concatenate(sum([[t, sep] for t in tiles], [])[:-1], 1))
Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(out, 'maps_motion_error_corruption_mask.png'))

gy, gx = r_index(H, W, H // 2, W // 2)
ones = torch.ones(N - 1, H // 2, W // 2)
T = combine(d['Aref'], d['Bref'], d['Ag'], d['Bg'], ones, gy, gx).numpy()
O = combine(d['Aref'], d['Bref'], d['Au'], d['Bu'], d['r0'], gy, gx).numpy()
O1 = combine(d['Aref'], d['Bref'], d['Au'], d['Bu'], ones, gy, gx).numpy()
sp = np.full((T.shape[0], 8, 3), 255, np.uint8)
Image.fromarray(np.concatenate([tone(T), sp, tone(O1), sp, tone(O)], 1)).save(
    os.path.join(out, 'merges_target_vs_corrupted_all_vs_corrupted_analytic.png'))
open(os.path.join(out, 'README.txt'), 'w').write(f"""Training burst {os.path.basename(f)} (scene {z['scenario']}, {N} frames, crop {W}x{H} raw).

frames.gif / frame_k.png : the burst (frame 0 = reference), quick demosaic, 2x enlarged.
maps_motion_error_corruption_mask.png : one row per comparison frame, columns:
  1. true motion (relative; rotated-content patches show as small squares with different motion)
  2. error of the flow fed to the mask = app aligner + deliberate corruption (white = 2+ grey px off)
  3. tiles deliberately corrupted (white)
  4. the app's analytic mask for that frame (white = kept)
merges_*.png : left  = TARGET: app merge with the exact flow, all samples (what training aims for)
               middle= app merge with the corrupted flow, no rejection (the ghosting to remove)
               right = app merge with the corrupted flow + the app's analytic mask
""")
print('written to', out)
