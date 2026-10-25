"""Reconstruction-based evaluation of a robustness map on one crop, through the exact replica of the app's merge.

All errors are measured on the merged 2x image against the HR target T (luma, loader domain), in units of the
single-frame noise variance sigma_1^2 at that pixel unless stated otherwise:
  psnr          PSNR of the actual NOISY merge vs T (peak 1.0, 4 px border excluded)
  exp_err       expected squared error of the merge vs T, bias^2 + variance (what PSNR estimates, without the one
                noise draw)
  ghost         misalignment energy (M - I)^2: the error the used correspondences paint compared with the same merge
                at the true correspondences (ghosts, doubled / shifted edges, wrong content); ghost_edge = on the 20%
                strongest-gradient pixels of T
  detail        high-frequency fidelity on textured pixels: 1 - |HP(M) - HP(T)|^2 / |HP(T)|^2  (HP = x - blur_1px x;
                1 = all fine detail of T recovered, lower = lost or wrong)
  false_detail  |HP(M) - HP(T)|^2 on the flattest 40% of pixels (structure that is not in the scene)
  noise         merged noise standard deviation on the flattest 40% of pixels, relative to one frame
  harm_rejected fraction of harmful samples (label R* < 0.5) the map gives R < 0.5
  valid_kept    fraction of valid samples (R* >= 0.5) the map gives R >= 0.5
  w_misaligned  mean weight on geometrically misaligned samples (|e| > 0.5 px or not visible)
  w_aligned     mean weight on well-aligned visible samples (|e| < 0.1 px)
"""
import torch
import torch.nn.functional as F

from . import merge as mg

LW = torch.tensor(mg.LUMA_W).view(3, 1, 1)


def luma(x):
    return (x * LW).sum(-3)


def blur(x, s=1.0):
    r = max(1, int(3 * s + 0.5))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / s) ** 2)
    k = k / k.sum()
    y = F.pad(x[None, None], (r, r, r, r), mode="replicate")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.conv2d(y, k.view(1, 1, -1, 1))
    return y[0, 0]


def evaluate_map(parts, R, Rstar, e, vis, T, sig2, mask_k=None, border=4):
    """parts: single-crop mg.Parts (K frames); R / Rstar / e / vis [K,P+1,P+1]. -> dict of metrics."""
    Rf = parts.fetch(R)
    Mn = parts.merged(Rf, clean=False)
    Mc = parts.merged(Rf, clean=True)
    I = parts.ideal(Rf)
    Var = parts.variance(Rf)
    b = (slice(border, -border), slice(border, -border))
    Tl, Mnl, Mcl, Il = luma(T)[b], luma(Mn)[b], luma(Mc)[b], luma(I)[b]
    Vl = luma(Var)[b]
    s2 = sig2[b]
    g = torch.sqrt(sum(d ** 2 for d in torch.gradient(luma(T))))[b]
    edge = g >= torch.quantile(g.flatten(), 0.8)
    loc_sd = torch.sqrt((blur(luma(T) ** 2) - blur(luma(T)) ** 2).clamp_min(0))[b]
    flat = loc_sd <= torch.quantile(loc_sd.flatten(), 0.4)
    tex = ~flat
    hpM = (luma(Mc) - blur(luma(Mc)))[b]
    hpT = (luma(T) - blur(luma(T)))[b]
    out = {}
    mse = ((Mnl - Tl) ** 2).mean().clamp_min(1e-12)
    out["psnr"] = float(10 * torch.log10(1.0 / mse))
    out["exp_err"] = float((((Mcl - Tl) ** 2 + Vl) / s2).mean())
    gh = (luma((Mc - I) ** 2)[b]) / s2
    out["ghost"] = float(gh.mean())
    out["ghost_edge"] = float(gh[edge].mean()) if edge.any() else 0.0
    den = (hpT[tex] ** 2).sum().clamp_min(1e-12)
    out["detail"] = float(1.0 - ((hpM[tex] - hpT[tex]) ** 2).sum() / den)
    out["false_detail"] = float((((hpM - hpT) ** 2)[flat] / s2[flat]).mean()) if flat.any() else 0.0
    out["noise"] = float(torch.sqrt((Vl[flat] / s2[flat]).mean())) if flat.any() else 0.0
    harm = Rstar < 0.5
    valid = ~harm
    out["harm_rejected"] = float((R[harm] < 0.5).float().mean()) if harm.any() else float("nan")
    out["valid_kept"] = float((R[valid] >= 0.5).float().mean()) if valid.any() else float("nan")
    mis = (e > 0.5) | (vis < 0.5)
    ali = (e < 0.1) & (vis >= 0.5)
    out["w_misaligned"] = float(R[mis].mean()) if mis.any() else float("nan")
    out["w_aligned"] = float(R[ali].mean()) if ali.any() else float("nan")
    out["n_harm"], out["n_valid"] = int(harm.sum()), int(valid.sum())
    return out
