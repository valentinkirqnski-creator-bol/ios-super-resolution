"""How alignment errors affect the reconstruction in different image regions, read off the generated labels (which
use no displacement threshold): the fraction of samples the reconstruction-optimal mask R* rejects, binned by the
correspondence error |e| and by the local structure contrast of the HR target relative to the noise.

    python analyze_labels.py [--splits val test] -> runs/label_analysis.json + .md
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np          # noqa: E402
import torch                # noqa: E402

from mfsr import merge as mg   # noqa: E402

E_BINS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 1e9]
C_BINS = [0.0, 0.25, 1.0, 3.0, 10.0, 1e9]      # local max of |grad T| / sigma_1 over a 5x5 raw window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    a = ap.parse_args()
    cnt = np.zeros((len(C_BINS) - 1, len(E_BINS) - 1))
    rej = np.zeros_like(cnt)
    wsum = np.zeros_like(cnt)
    for split in a.splits:
        for fn in sorted(glob.glob(os.path.join(HERE, "data", "d1", split, "*.npz"))):
            z = np.load(fn)
            meta = json.loads(str(z["meta"]))
            T = torch.from_numpy(z["T"].astype(np.float32))
            s = mg.single_frame_sig2(T, np.array(meta["prewhite"], np.float32), meta["noise_a"], meta["noise_b"]).sqrt()
            lum = (T * torch.tensor(mg.LUMA_W).view(3, 1, 1)).sum(0)
            g = torch.sqrt(sum(d ** 2 for d in torch.gradient(lum))) / s
            g = torch.nn.functional.max_pool2d(g[None, None], 5, 1, 2)[0, 0, ::2, ::2]
            P1 = z["R0"].shape[-1]
            g = torch.nn.functional.pad(g[None, None], (0, P1 - g.shape[1], 0, P1 - g.shape[0]), mode="replicate")[0, 0].numpy()
            vis = z["vis"].astype(np.float32)
            for v in (0, 1):
                e, R = z[f"e{v}"].astype(np.float32), z[f"R{v}"].astype(np.float32)
                ok = vis > 0.5
                ci = np.clip(np.digitize(np.broadcast_to(g, e.shape)[ok], C_BINS) - 1, 0, len(C_BINS) - 2)
                ei = np.clip(np.digitize(e[ok], E_BINS) - 1, 0, len(E_BINS) - 2)
                np.add.at(cnt, (ci, ei), 1)
                np.add.at(rej, (ci, ei), (R[ok] < 0.5).astype(np.float64))
                np.add.at(wsum, (ci, ei), R[ok].astype(np.float64))
    frac = np.where(cnt > 0, rej / np.maximum(cnt, 1), np.nan)
    meanw = np.where(cnt > 0, wsum / np.maximum(cnt, 1), np.nan)
    out = dict(e_bins=E_BINS, contrast_bins=C_BINS, count=cnt.tolist(), rejected=frac.tolist(), mean_R=meanw.tolist())
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "runs", "label_analysis.json"), "w"), indent=1)
    lab = lambda b, i: f"{b[i]:g}-{b[i + 1]:g}" if b[i + 1] < 1e8 else f">{b[i]:g}"
    lines = ["Fraction of visible samples the reconstruction-optimal mask rejects (R* < 0.5), by correspondence error "
             "|e| (raw px, columns) and local structure contrast |grad T| / sigma_1 (rows). Count in brackets.", "",
             "| contrast \\ error | " + " | ".join(lab(E_BINS, j) for j in range(len(E_BINS) - 1)) + " |",
             "|" + "---|" * len(E_BINS)]
    for i in range(len(C_BINS) - 1):
        cells = [(f"{100 * frac[i, j]:.0f}% ({int(cnt[i, j])})" if cnt[i, j] >= 30 else "-") for j in range(len(E_BINS) - 1)]
        lines.append(f"| {lab(C_BINS, i)} | " + " | ".join(cells) + " |")
    open(os.path.join(HERE, "runs", "label_analysis.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
