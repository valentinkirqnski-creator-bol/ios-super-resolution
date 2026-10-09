"""Count what a viewer would SEE, instead of summing squared error.

Every model this project has selected on a merged-error loss has measured better
and looked worse, four times running, and the reason is arithmetic rather than
weighting. A ghost is a small amount of error over a few pixels. A merge that is
slightly softer everywhere is a larger amount of error. Any loss that INTEGRATES
error therefore prefers the soft merge that kept the ghost -- and no choice of
edge weight, gradient term or band split changes that, because the problem is the
summation, not the weights on it.

So count events instead. A ghost is visible when three things hold at once:

  1. there is gradient in the output that the target does not have;
  2. it sits beside real structure -- that adjacency is what makes it read as
     DOUBLING rather than as noise;
  3. its amplitude clears the local noise floor, because a ghost at noise level
     is not a ghost.

and the mirror of (1) and (3), with no adjacency requirement, is detail LOST.
Both come out in the same unit -- fraction of pixels carrying a visible defect --
so ghosting and softness trade against each other directly rather than through an
arbitrary lambda.

The decisive property is that it is a COUNT. A large area of sub-threshold error
cannot outvote a small area of visible error, which is exactly what went wrong
before. Every threshold is applied through a sigmoid so the whole thing stays
differentiable and can be trained against, not just reported.

Self-calibrating: the noise scale is estimated from the output's own gradient in
the regions where the TARGET is flat, so it needs no noise model and no extra
data, and it tracks however many frames the mask actually merged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _lum(x):
    """(B, C, H, W) -> (B, 1, H, W). Mean over channels: the merge's output is
    CFA-ish and the defect being counted is achromatic structure."""
    return x.mean(dim=-3, keepdim=True)


def grad_mag(x):
    """Central-difference gradient magnitude of the luminance, same size in."""
    y = _lum(x)
    p = F.pad(y, (1, 1, 1, 1), mode='replicate')
    gx = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
    gy = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _median(x):
    return x.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)


def noise_scale(go, gt):
    """Gradient noise sigma of an image, measured where the TARGET is flat.

    MAD rather than a mean, so a few real edges leaking into the flat set cannot
    inflate it.

    NOT to be called on the model's own output during training. An earlier
    version did, reasoning that a model could not game the metric by rejecting
    more because rejecting more raises the threshold its defects are measured
    against. That reasoning is backwards -- a higher threshold means FEWER
    defects counted, which is the exploit, not a guard against it. Training found
    it in 756 steps: meanR fell 0.311 -> 0.132 while the counted `lost` went
    DOWN, which is not physically possible (rejecting frames must lose more
    detail, not less) and is the signature of a moving goalpost. Use this only on
    the reference merge, via reference_sigma, which does not depend on any model.
    """
    flat = (gt <= _median(gt)).float()
    # Weighted median via sorting is awkward in torch; a masked quantile over the
    # flat set is equivalent here and cheaper.
    big = go.max().detach() + 1.0
    masked = torch.where(flat > 0, go, torch.full_like(go, big))
    n_flat = flat.flatten(1).sum(dim=1).clamp_min(1.0)
    srt, _ = masked.flatten(1).sort(dim=1)
    idx = (n_flat * 0.5).long().clamp(0, srt.shape[1] - 1)
    med = srt.gather(1, idx.view(-1, 1)).view(-1, 1, 1, 1)
    return (1.4826 * med).clamp_min(1e-6)


def near_structure(gt, frac=0.75, radius=2):
    """Soft indicator of 'beside real structure': within `radius` of a target
    edge. A ghost in open sky is a different defect from a doubled edge, and it
    is the doubled edge this is chasing."""
    thr = torch.quantile(gt.flatten(1), frac, dim=1).view(-1, 1, 1, 1)
    e = torch.sigmoid((gt - thr) / thr.clamp_min(1e-6) * 4.0)
    k = 2 * radius + 1
    return F.max_pool2d(e, k, stride=1, padding=radius)


def coherent(x, radius=1):
    """Mean of x over a small window.

    This is what separates a ghost from noise, and it is the half the amplitude
    test cannot do. A ghost is a contiguous edge: averaging a few neighbours
    leaves it essentially unchanged. Noise is uncorrelated: averaging n pixels
    divides it by sqrt(n). So the same threshold applied after this counts ghosts
    and stops counting noise.
    """
    k = 2 * radius + 1
    return F.avg_pool2d(F.pad(x, (radius,) * 4, mode='replicate'), k, stride=1)


def eff_frames(R, out_hw):
    """1 + sum_n R_n, on the OUTPUT lattice: how many frames each output pixel
    effectively averaged. Pass a FIXED mask, not the one being trained -- see
    visible().

    Merging N frames divides the noise sigma by sqrt(N), so this is the whole
    noise model needed here -- and the mask already carries it. Without it the
    threshold has to be one number per image, which is wrong in both directions
    at once on a partially merged frame: too high where the mask merged and too
    low where it rejected, so the leftover noise in the rejected regions gets
    counted as ghosting. Measured: that alone made the shipped mask score WORSE
    than merging everything, which is the opposite of the truth.
    """
    e = 1.0 + R.clamp(0, 1).sum(dim=1, keepdim=True)
    if e.shape[-2:] != out_hw:
        e = F.interpolate(e, size=out_hw, mode='nearest')
    return e.clamp_min(1.0)


def local_peak(gt, radius=3):
    """Strongest target gradient within `radius`. The thing a ghost is a copy
    OF."""
    kk = 2 * radius + 1
    return F.max_pool2d(gt, kk, stride=1, padding=radius)


def visible(out, tgt, R_ref=None, sigma_ref=None, k=3.0, tau=0.75, radius=2,
            coh=1, rel=0.15, tau_rel=0.05, peak_radius=3):
    """-> (ghost_frac, lost_frac), each the fraction of pixels carrying a
    visible defect of that kind. Differentiable in `out`.

    R_ref and sigma_ref together give the per-pixel noise floor: sigma_ref is the
    gradient noise of the reference frame alone (one merge with R = 0), divided
    by sqrt of the effective frame count R_ref produced.

    R_ref MUST be a fixed mask -- the shipped one -- and never the mask being
    trained or compared. The yardstick cannot depend on what it measures. Passing
    the model's own mask lets it score better by rejecting more until the picture
    is noisy enough to hide its defects, which is measured to happen (see
    noise_scale). With a fixed R_ref the threshold is identical for every model
    on a given burst, so a difference in the count is a difference in defects.

    k = 3: three sigma, the usual threshold for 'distinguishable from noise'.
    tau softens the step so a gradient exists either side of it.
    """
    go, gt = grad_mag(out), grad_mag(tgt)
    if R_ref is not None and sigma_ref is not None:
        sg = sigma_ref / torch.sqrt(eff_frames(R_ref, go.shape[-2:]))
    else:
        sg = noise_scale(go, gt)
    sg = sg.detach()                        # a scale, not a thing to optimise
    # Coherence first, threshold second: see coherent().
    ex_g = coherent((go - gt).clamp_min(0), coh)
    ex_l = coherent((gt - go).clamp_min(0), coh)
    # Measured as a FRACTION OF THE LOCAL EDGE, not as a multiple of the noise.
    #
    # Thresholding on the noise floor was measured to count noise as ghosting:
    # it made ghost% HIGHEST at R = 0, where no frame but the reference is
    # merged and ghosting is impossible by construction. Amplitude and local
    # coherence cannot separate "displaced copy of an edge" from "noise" --
    # both are gradient the target lacks, and a 3x3 average only divides noise
    # by three.
    #
    # What does separate them: a ghost is a displaced copy of nearby structure,
    # so its magnitude SCALES with the edge it came from. Noise is a fixed
    # amount set by the sensor, so against a strong edge it is a small fraction
    # and against a weak one a large one. Dividing by the local peak therefore
    # keeps the ghost and drops the noise, which no amount of absolute
    # thresholding can do.
    #
    # The floor keeps flat regions sane: where there is no structure within
    # peak_radius the denominator becomes k*sg/rel, so the test reduces to the
    # old absolute one, ex > k sigma, rather than dividing by nearly zero.
    peak = local_peak(gt, peak_radius).clamp_min(k * sg / max(rel, 1e-6))
    ghost = torch.sigmoid((ex_g / peak - rel) / tau_rel)
    lost = torch.sigmoid((ex_l / peak - rel) / tau_rel)
    near = near_structure(gt, radius=radius)
    return (ghost * near).mean(), lost.mean()


def reference_sigma(out_ref, tgt):
    """sigma_ref for a burst: the gradient noise of the reference-frame-only
    merge. One extra merge per burst, and it does not depend on the model, so it
    is computed once and reused across every mask being compared."""
    return noise_scale(grad_mag(out_ref), grad_mag(tgt)).detach()


def visible_loss(out, tgt, w_ghost=1.0, w_lost=1.0, **kw):
    g, l = visible(out, tgt, **kw)
    return w_ghost * g + w_lost * l, g, l
