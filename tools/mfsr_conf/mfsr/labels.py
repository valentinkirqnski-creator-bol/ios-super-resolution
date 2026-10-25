"""Training targets.

Alignment confidence (geometry only). For every reference raw pixel p and frame k the merge reads frame k at
q_used(p) = H(p + flow[tile(p)]) (merge.tile_correspondence). The formation model gives the exact q_gt(p) and the
visibility. Stored: e_k(p) = |q_used(p) - q_gt(p)| in raw px and vis_k(p). The target is
    t_align = vis * exp(-e^2 / (2 sigma_a^2))
i.e. "this sample comes from the right scene point, to within sigma_a" (sigma_a = loss.align_sigma, 0.25 raw px by
default, set at training time). It says nothing about whether the sample is useful: a flat region can be badly
aligned and still help.

Merge confidence (usefulness for the reconstruction). The robustness value is what the app's merge multiplies a
frame's contribution by, so the target is the set of R maps that makes the app's merge best, judged with the
reconstruction cost of merge.cost_map (mode "linear"), per output pixel, in single-frame noise-variance units:
    gamma sum_c w_c (M_c - I_c)^2 / sigma_1^2  +  sum_k (1 - r_k) (g_k + n_k + keep_prior)
M = the merge of the noise-free frames with the used correspondences (so the label is not fitted to one noise
realisation); I = the same merge, same weights, at the TRUE correspondences with occluded samples dropped;
g_k = how much frame k, perfectly aligned, brings the merge closer to the HR target T (its new sub-pixel samples of
aliased detail, situation A); n_k = how much it lowers the merge's noise variance (situation C) -- both measured on
the app's merge at the all-frames point, clamped >= 0 (merge.frame_gains); keep_prior = 0.02.
A well-aligned sample (M = I) is kept by every term (A and C); a misaligned or occluded one is rejected exactly when
the error it paints, gamma (M - I)^2 -- ghosts, doubled / shifted edges, wrong content -- outweighs the detail and
the noise reduction it brings (B). gamma = 4: a misalignment error of 0.5 sigma costs as much as a full sigma^2 of
noise -- a misalignment is worse than falling back to fewer frames.
Why not the exact reconstruction cost (merge.cost_map mode "full": gamma (M - I)^2 + (I - T)^2 + Var)? Its per-pixel
optimum re-weights PERFECTLY aligned samples to suit the merge kernel -- measured on smoke samples: 42% of aligned
fine-text samples dropped (subset selection sharpens text under the kernel), 46% of aligned samples in flat areas
under the shipped narrow kernels (inverse-variance re-balancing of very unequal per-tap weights), 18.7% on real-scene
edges when the whole error is priced against T. None of that is misalignment; a robustness mask that learned it would
reject valid aliasing and flat-area samples, which is exactly what it must not do.
No displacement threshold appears anywhere: how much an error matters is decided by what it does to the merged
image at that place.

R* is optimised jointly over all frames (Adam on logits, mild total-variation term), from two starts (all 1 and all
~0) because the cost is not convex in R; per raw pixel the start with the lower cost wins. The trivial solutions
cannot win: rejecting everything leaves the reference's noise variance, accepting everything pays gamma x the
ghost energy.

Include / exclude utility (diagnostic, stored for val / test): with the other frames at R*, frame k fully in versus
fully out, Delta_k(p) = cost(r_k = 0) - cost(r_k = 1) summed over the output pixels that read raw pixel p
(noise-variance units; > 0 = including the frame helps).
"""
import torch

from . import merge as mg


def per_raw(cmap, fy, fx, Hr, Wr):
    """Sum an output-grid map [..., Ho, Wo] onto the raw pixels the mask is read at. -> [..., Hr, Wr]."""
    idx = (fy * Wr + fx).reshape(-1)
    flat = cmap.reshape(*cmap.shape[:-2], -1)
    out = torch.zeros(*cmap.shape[:-2], Hr * Wr, dtype=cmap.dtype)
    out.index_add_(out.dim() - 1, idx, flat)
    return out.view(*cmap.shape[:-2], Hr, Wr)


def oracle_R(parts, T, sig2, K, Hr, Wr, gamma=4.0, tv=0.02, iters=160, lr=0.15, utility=False, keep_prior=0.02,
             mode="linear"):
    """-> R* [K,Hr,Wr] in [0,1], per-raw cost of R* [Hr,Wr], (utility) Delta [K,Hr,Wr]."""
    if mode == "linear" and parts.g is None:
        parts.g = mg.frame_gains(parts, T, sig2)
    res = []
    for init in (4.0, -4.0):
        z = torch.full((K, Hr, Wr), init, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(iters):
            R = torch.sigmoid(z)
            c = mg.cost_map(parts, parts.fetch(R), T, sig2, gamma, keep_prior, mode).mean()
            if tv > 0:
                c = c + tv * ((R[:, 1:] - R[:, :-1]).abs().mean() + (R[:, :, 1:] - R[:, :, :-1]).abs().mean())
            opt.zero_grad()
            c.backward()
            opt.step()
        with torch.no_grad():
            R = torch.sigmoid(z)
            pc = per_raw(mg.cost_map(parts, parts.fetch(R), T, sig2, gamma, keep_prior, mode), parts.fy, parts.fx, Hr, Wr)
        res.append((R, pc))
    (R1, c1), (R0, c0) = res
    take1 = (c1 <= c0)
    R = torch.where(take1[None], R1, R0)
    pc = torch.where(take1, c1, c0)
    if not utility:
        return R, pc, None
    D = torch.zeros(K, Hr, Wr)
    with torch.no_grad():
        for k in range(K):
            Ra, Rb = R.clone(), R.clone()
            Ra[k] = 0.0
            Rb[k] = 1.0
            ca = per_raw(mg.cost_map(parts, parts.fetch(Ra), T, sig2, gamma, keep_prior, mode), parts.fy, parts.fx, Hr, Wr)
            cb = per_raw(mg.cost_map(parts, parts.fetch(Rb), T, sig2, gamma, keep_prior, mode), parts.fy, parts.fx, Hr, Wr)
            D[k] = ca - cb
    return R, pc, D


def align_error(flow, Hm, has_h, tile, disp_gt, ry0, rx0, Hr, Wr):
    """|q_used - q_gt| on raw pixels [ry0,+Hr) x [rx0,+Wr) for one frame. disp_gt [2,Hr,Wr] float64 (ref->frame)."""
    yy, xx = torch.meshgrid(torch.arange(ry0, ry0 + Hr, dtype=torch.float32),
                            torch.arange(rx0, rx0 + Wr, dtype=torch.float32), indexing="ij")
    my, mx = mg.tile_correspondence(yy, xx, flow, Hm, has_h, tile)
    gx = xx.double() + disp_gt[0]
    gy = yy.double() + disp_gt[1]
    return torch.sqrt((mx.double() - gx) ** 2 + (my.double() - gy) ** 2).float()
