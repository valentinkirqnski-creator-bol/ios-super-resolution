"""Dataset over generated samples: one item = one (burst, alignment variant), returned as `n_crops` training crops.

A crop is a Pc x Pc raw patch inside the labelled region (raw [m, m+P]), with `halo` raw px of input context on each
side. Per crop:
  xg [K,NG,(Pc+2h)/2,(Pc+2h)/2], xr [K,NR,Pc+2h,Pc+2h]   network inputs (computed on the full frame, then cropped)
  r0 [K,Pc+2h,Pc+2h]                                     the app's analytic mask (raw res), for refine_r0 / baselines
  R* / e / vis [K,Pc+1,Pc+1]                              labels on the raw pixels the merge reads for this crop's
                                                          output block [2 y0, 2 y0 + 2 Pc)
  parts                                                   the app merge's contributions for that output block (exact
                                                          replica), the ideal counterpart, frame gains, T, sigma_1^2
Frames are padded to K_MAX (mask = 0) so bursts of 4-8 frames batch together.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from . import merge as mg, sample
from .inputs import frame_inputs

K_MAX = 7


class Bursts(Dataset):
    def __init__(self, root, split, crop=96, halo=32, n_crops=2, random_crop=True, use_r0=False, variants=(0, 1),
                 limit=0, with_noisy=False):
        files = sorted(f for f in glob.glob(os.path.join(root, split, "*.npz")) if ".tmp." not in f)
        if limit:
            files = files[:limit]
        self.items = [(f, v) for f in files for v in variants]
        self.crop, self.halo, self.n_crops, self.random_crop, self.use_r0 = crop, halo, n_crops, random_crop, use_r0
        self.with_noisy = with_noisy

    def __len__(self):
        return len(self.items)

    def load(self, i):
        f, v = self.items[i]
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        t = lambda k: torch.from_numpy(z[k].astype(np.float32))
        return f, v, z, meta, t

    def __getitem__(self, i):
        f, v, z, meta, t = self.load(i)
        P, m, scale = meta["P"], meta["m"], meta["scale"]
        cfa = tuple(tuple(r) for r in meta["cfa"])
        wb = np.array(meta["prewhite"], np.float32)
        a, b, tile = meta["noise_a"], meta["noise_b"], meta["tile"]
        raw, clean = t("raw"), t("clean")
        var = sample.var_from_clean(clean, z["gain"], wb, cfa, a, b)
        K = raw.shape[0] - 1
        flows, Hs, hh = t(f"flow{v}"), t(f"H{v}"), z[f"hash{v}"]
        r0g = t(f"r0{v}")
        xg, xr = frame_inputs(raw, t("grey_ref"), t("grey"), r0g, flows, Hs, hh, tile, a, b, wb, self.use_r0, cfa)
        r0 = r0g.repeat_interleave(2, -2).repeat_interleave(2, -1)
        cov = z["cov"].astype(np.float32)
        hv = dict(flow=z[f"flow{v}"], H=z[f"H{v}"], has_h=hh)
        Rl, el, visl = t(f"R{v}"), t(f"e{v}"), t("vis")
        disp = z["disp"].astype(np.float64)
        T_all = t("T")
        Pc, h = min(self.crop, P), self.halo
        # merge contributions once for the whole labelled patch (output block [s m, s m + s P)); every crop is a
        # slice of it -- each output pixel's contributions are independent of the others
        gt_d = sample.dense_gt_xy(disp, m, m, P, scale)
        full = sample.merge_parts_xy(raw, clean, var, cov, hv, scale, cfa, m, m, P, tile, gt_dense=gt_d, vis=visl,
                                     with_noisy=self.with_noisy)
        sig2_full = mg.single_frame_sig2(T_all, wb, a, b)
        full.g = mg.frame_gains(full, T_all, sig2_full)
        so = int(round(scale * m))
        crops = []
        rng = np.random.default_rng() if self.random_crop else None
        for _ in range(self.n_crops if self.random_crop else 1):
            if self.random_crop and P > Pc:
                y0 = m + 2 * int(rng.integers(0, (P - Pc) // 2 + 1))
                x0 = m + 2 * int(rng.integers(0, (P - Pc) // 2 + 1))
            else:
                y0 = x0 = m + 2 * ((P - Pc) // 4)
            ly, lx = y0 - m, x0 - m                                    # offset inside the stored label region
            sl = (slice(None), slice(ly, ly + Pc + 1), slice(lx, lx + Pc + 1))
            Ho = int(round(scale * Pc))
            oy, ox = int(round(scale * y0)) - so, int(round(scale * x0)) - so
            osl = (Ellipsis, slice(oy, oy + Ho), slice(ox, ox + Ho))
            fy, fx = mg.fetch_index(int(round(scale * y0)), int(round(scale * x0)), Ho, Ho, scale, y0, x0, Pc + 1, Pc + 1)
            ry, rx = y0 - h, x0 - h
            crops.append(dict(
                xg=xg[:, :, ry // 2:(ry + Pc + 2 * h) // 2, rx // 2:(rx + Pc + 2 * h) // 2],
                xr=xr[:, :, ry:ry + Pc + 2 * h, rx:rx + Pc + 2 * h],
                r0=r0[:, ry:ry + Pc + 2 * h, rx:rx + Pc + 2 * h],
                R=Rl[sl], e=el[sl], vis=visl[sl], T=T_all[osl], sig2=sig2_full[osl],
                parts={k: getattr(full, k)[osl] for k in ("Aref", "Acref", "Bref", "Vref", "A", "Ac", "B", "V", "Ag",
                                                         "Bg", "vis", "g")},
                fy=fy, fx=fx, K=K, category=meta["category"], file=os.path.basename(f), variant=v,
                y0=y0, x0=x0, iso=meta["iso"]))
        return crops


def _pad_k(x, K):
    if x.shape[0] == K:
        return x
    pad = torch.zeros((K - x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype)
    return torch.cat([x, pad], 0)


def collate(batch):
    crops = [c for item in batch for c in item]
    Kb = K_MAX
    out = {}
    for key in ("xg", "xr", "r0", "R", "e", "vis"):
        out[key] = torch.stack([_pad_k(c[key], Kb) for c in crops])
    out["mask"] = torch.stack([(torch.arange(Kb) < c["K"]).float() for c in crops])
    out["T"] = torch.stack([c["T"] for c in crops])
    out["sig2"] = torch.stack([c["sig2"] for c in crops])
    P = {}
    for key in ("Aref", "Acref", "Bref", "Vref"):
        P[key] = torch.stack([c["parts"][key] for c in crops])
    for key in ("A", "Ac", "B", "V", "Ag", "Bg", "vis", "g"):
        P[key] = torch.stack([_pad_k(c["parts"][key], Kb) for c in crops])
    out["parts"] = P
    out["fy"], out["fx"] = crops[0]["fy"], crops[0]["fx"]
    out["meta"] = [dict(category=c["category"], file=c["file"], variant=c["variant"], K=c["K"]) for c in crops]
    return out


def parts_of(batch):
    p = batch["parts"]
    P = mg.Parts(p["Aref"], p["Acref"], p["Bref"], p["Vref"], p["A"], p["Ac"], p["B"], p["V"], batch["fy"],
                 batch["fx"], p["Ag"], p["Bg"], p["vis"])
    P.g = p["g"]
    return P
