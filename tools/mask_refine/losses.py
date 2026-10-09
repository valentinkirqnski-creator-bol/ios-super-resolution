"""Losses on the FINAL MERGE, not on a surrogate for the mask.

The target is the merge built from the same frames with ground-truth alignment,
so a mask is judged by the picture it produces. Four terms:

  recon      Charbonnier on the merged image. Robust: a few badly wrong pixels
             should not dominate the way L2 lets them.
  ghost      the defect the eye actually sees, and the reason a plain
             reconstruction loss is not enough. A ghost is gradient energy the
             target does not have, beside structure the target does have. It is
             small in total error and large in appearance, so it is weighted
             separately rather than left to compete inside the mean.
  suppress   a penalty on withdrawn acceptance, sum(M_wronski - M_final). Without
             it the cheapest way to remove every ghost is to reject everything,
             which also removes the multi-frame benefit the burst exists for.
  distill    OPTIONAL. An offline oracle: reject where the known flow error is
             large. Off by default (w_distill = 0) because the flow error is a
             proxy -- a 0.3 px error in flat sky does nothing while the same
             error across an edge is a visible double -- so it is available as
             guidance, never as the objective.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps * eps).mean()


def _grad(x):
    """Forward differences of the luminance."""
    y = x.mean(dim=-3, keepdim=True)
    gx = y[..., :, 1:] - y[..., :, :-1]
    gy = y[..., 1:, :] - y[..., :-1, :]
    return gx, gy


def ghost_loss(out, gt, dilate=2):
    """Gradient the output has that the target does not, NEXT TO real structure.

    Both halves matter. Excess gradient alone also fires on noise, which is not
    a ghost; requiring it to sit beside a true edge is what makes it the
    doubling signature rather than a general sharpness penalty.
    """
    ox, oy = _grad(out)
    tx, ty = _grad(gt)
    ex = (ox.abs() - tx.abs()).clamp_min(0)
    ey = (oy.abs() - ty.abs()).clamp_min(0)
    # Where the TARGET has structure, dilated so the band beside an edge counts.
    k = 2 * dilate + 1
    nx = F.max_pool2d(tx.abs(), k, 1, dilate)
    ny = F.max_pool2d(ty.abs(), k, 1, dilate)
    sx = nx.mean().clamp_min(1e-6)
    sy = ny.mean().clamp_min(1e-6)
    return (ex * (nx / sx)).mean() + (ey * (ny / sy)).mean()


def suppression(mask_w, mask_f):
    """Mean acceptance withdrawn, relative to what Wronski offered.

    Normalised by the Wronski mass so a burst it already rejects heavily is not
    penalised for the rejection it did not choose.
    """
    return (mask_w - mask_f).clamp_min(0).sum() / mask_w.sum().clamp_min(1.0)


def distill(mask_w, mask_f, ferr, thresh=1.0, soft=0.5):
    """Optional oracle: a soft target that withdraws where flow error is large."""
    want = torch.sigmoid((thresh - ferr) / soft)
    return (mask_f - mask_w * want).abs().mean()


def total(out_f, out_w, gt, mask_w, mask_f, ferr=None, w=None):
    """out_f: merged with the refined mask. out_w: with the Wronski mask.

    The image terms are RATIOS against the Wronski baseline, not absolute
    errors. Measured at the identity init, the absolute reconstruction gradient
    is 7.8e-07 and the suppression gradient 4.9e-02 -- a factor of 63,000 -- so
    with absolute terms the suppression penalty is the only thing steering the
    weights and the network slides to "change nothing" whatever the pictures
    say. That is a scale problem, not a weighting one: at 43 dB the image error
    is ~0.001 while withdrawn mask is ~0.05.

    Dividing by the baseline fixes it at the source. Both terms are then ~1.0 at
    the identity, lower is better, and the quantity being minimised is exactly
    the question worth asking -- does this beat the mask it refines -- rather
    than an absolute error whose scale depends on how bright the scene was.

    The baselines are detached: they are a yardstick, not something to optimise.
    """
    w = w or {}
    rec_w = charbonnier(out_w, gt).detach().clamp_min(1e-8)
    gh_w = ghost_loss(out_w, gt).detach().clamp_min(1e-8)
    l_rec = charbonnier(out_f, gt) / rec_w
    l_gh = ghost_loss(out_f, gt) / gh_w
    l_sup = suppression(mask_w, mask_f)
    l_dis = (distill(mask_w, mask_f, ferr)
             if (ferr is not None and w.get('distill', 0.0) > 0) else
             torch.zeros((), device=out_f.device))
    loss = (w.get('recon', 1.0) * l_rec + w.get('ghost', 1.0) * l_gh +
            w.get('suppress', 0.1) * l_sup + w.get('distill', 0.0) * l_dis)
    return loss, dict(recon=float(l_rec.detach()), ghost=float(l_gh.detach()),
                      suppress=float(l_sup.detach()), distill=float(l_dis.detach()))
