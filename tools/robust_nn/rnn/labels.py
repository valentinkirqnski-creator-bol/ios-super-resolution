"""The two supervision sources. Neither is "do the two images look different".

A. Geometric reliability (is the proposed correspondence right, and would its
   error be visible?)

     e      = |f_est - f_gt|                       (true flow error, grey px)
     D^2    = mean_3x3 ( C_k(p + f_est) - C_k(p + f_gt) )^2
              C_k = the NOISE-FREE comparison frame. Both samples come from the
              same frame, so aliasing / sub-pixel sampling differences cancel:
              D measures only the ghost that the *flow error* would paint.
              A 0.2 px error on a sharp edge gives a large D; the same error
              in a flat region gives D ~ 0.
     geo    = vis * exp(-D^2 / (2 k^2 s^2)) * exp(-e^2 / (2 tau^2))
              s^2 = noise var at this brightness + (2% contrast floor)^2,
              tau bounds gross errors even where colours happen to match,
              vis = 0 for occluded / disoccluded / out-of-frame points.

B. Reconstruction utility (does including this sample improve the merge?)
   Controlled fusion against the known clean scene. Other frames are weighted
   by their oracle geometric reliability (a realistic "good mask" context);
   frame k is toggled fully in vs fully out:

     util = sigmoid( beta * log( (E_without + eps) / (E_with + eps) ) )

   E = locally pooled squared error of the merge vs ground truth. Correctly
   aligned frames reduce noise and aliasing -> util > 0.5 even though their
   pixels differ from the reference; ghosting frames -> util -> 0. Utility is
   context dependent (it depends on the other samples), so it is used as a
   training signal alongside A, not as a sole universal truth.
"""
import math
import torch
import torch.nn.functional as F

from .common import warp_by_flow, win_mean
from . import merge


def geometric_label(clean_k, clean_ref, flow_est, flow_gt, vis, a_raw, b_raw, k_sig=2.0, tau=2.0):
    """All grey-res [H,W] / flows [2,H,W]. Returns geo [H,W], err [H,W]."""
    Ce = warp_by_flow(clean_k[None], flow_est[None])
    Cg = warp_by_flow(clean_k[None], flow_gt[None])
    D2 = win_mean((Ce - Cg) ** 2, 1)[0]
    mu = win_mean(clean_ref[None], 1)[0].clamp_min(0)
    s2 = (a_raw * mu + b_raw) / 4.0 + (0.02 * mu) ** 2 + 1e-10
    e = (flow_est - flow_gt).norm(dim=0)
    geo = vis * torch.exp(-0.5 * D2 / (k_sig * k_sig * s2)) * torch.exp(-0.5 * (e / tau) ** 2)
    return geo, e


def utility_labels(raw, flows_est, geo, gt, a_raw, b_raw, scale=1, sigma_k=0.7, beta=8.0):
    """raw [N,Hr,Wr], flows_est [N,2,Hg,Wg], geo [N,Hg,Wg] (geo[0] ignored), gt [3,Ho,Wo]."""
    N = raw.shape[0]
    Hg, Wg = geo.shape[-2:]
    num, den = merge.contributions(raw, flows_est, scale, sigma_k)
    Ho, Wo = num.shape[-2:]
    w = geo.clone()
    w[0] = 1.0
    Rout = merge.upsample_weights(w, Ho, Wo, scale)
    bn = (num * Rout[:, None]).sum(0)
    bd = (den * Rout[:, None]).sum(0)
    lum = gt.mean(0).clamp_min(0)
    eps = 3.0 * 0.1 * (a_raw * lum + b_raw) / N + 1e-9   # noise floor: tiny E ratios are not evidence
    sig = 2.0 * scale
    util = torch.ones(N, Hg, Wg)
    for k in range(1, N):
        nm = bn - Rout[k] * num[k]
        dm = bd - Rout[k] * den[k]
        Mm = nm / dm.clamp_min(1e-12)
        Mp = (nm + num[k]) / (dm + den[k]).clamp_min(1e-12)
        Em = merge.local_err(Mm, gt, sig)
        Ep = merge.local_err(Mp, gt, sig)
        u = torch.sigmoid(beta * torch.log((Em + eps) / (Ep + eps)))
        util[k] = F.avg_pool2d(u[None, None], 2 * scale)[0, 0]
    return util
