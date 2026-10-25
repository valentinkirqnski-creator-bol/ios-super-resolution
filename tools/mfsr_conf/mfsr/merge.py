"""Exact PyTorch replica of the app's merge (core/merge.cpp accumulate_comp / accumulate_ref, steerable kernels,
Bayer), plus recombination for ANY robustness map and the reconstruction cost used by the labels and the loss.

Why this is enough to train against the real merge: for a one-channel mask the app fetches ONE robustness value per
output pixel and frame (merge.cpp: local_r is constant over the 3x3 taps), so

    num(o) = A_ref(o) + sum_k R_k[f(o)] * A_k(o)        den(o) = B_ref(o) + sum_k R_k[f(o)] * B_k(o)

with A_k / B_k the frame's contributions computed with R = 1 and f(o) the raw pixel the mask is read at:
f(o) = (floor(o_y / s + 0.5), floor(o_x / s + 0.5)) for a raw-resolution mask (merge.cpp, nearest fetch). The
merged image is therefore an exact, differentiable function of every R map.

Conventions (identical to merge.cpp): raw pixel centres at integer coordinates; output pixel o at raw o / s; the
tile flow is consumed nearest (tile index trunc(lr / tile)); the moved position is H(lr + flow) when a homography is
present; samples whose moved position falls outside [0, W) x [0, H) are skipped; centre = round half away from zero;
covariance fetched at (moved - 0.5) / 2 on the half-resolution kernel grid; comparison frames invert it with a raw
1/det, the reference with invert_2x2; both then go through soften_inv_cov.

Besides the noisy contributions it accumulates the noise-free ones (Ac, from the clean frames) and the noise
variance V = sum_taps w^2 * var(tap) per channel, so the bias and the expected variance of the merged image are
known exactly for any R:  M_clean = (Ac_ref + sum r Ac) / den,  Var = (V_ref + sum r^2 V) / den^2.
"""
import torch

F32 = torch.float32
K_MAX_ABS = 32.0
EPS_DIV = 1e-10


def _soften(ixx, ixy, iyy):
    m = torch.maximum(ixx.abs(), torch.maximum(iyy.abs(), ixy.abs()))
    finite = torch.isfinite(ixx) & torch.isfinite(ixy) & torch.isfinite(iyy)
    big = (m > K_MAX_ABS) & torch.isfinite(m)
    s = torch.where(big, K_MAX_ABS / torch.where(big, m, torch.ones_like(m)), torch.ones_like(m))
    ixx, ixy, iyy = ixx * s, ixy * s, iyy * s
    two, zero = torch.full_like(ixx, 2.0), torch.zeros_like(ixx)
    return torch.where(finite, ixx, two), torch.where(finite, ixy, zero), torch.where(finite, iyy, two)


def interp_inv_cov(covs, kmap_i, kmap_j, raw_det):
    """covs [h,w,4] (xx, xy, yx, yy); kmap float32 maps. -> ixx, ixy, iyy (merge.cpp interp_inv_cov)."""
    h, w = covs.shape[:2]
    ti, tj = torch.trunc(kmap_i), torch.trunc(kmap_j)
    frac_x, frac_y = kmap_j - tj, kmap_i - ti
    if raw_det:
        fx, fy = tj.long().clamp_min(0), ti.long().clamp_min(0)
    else:
        fx, fy = torch.floor(kmap_j).long().clamp_min(0), torch.floor(kmap_i).long().clamp_min(0)
    fx, fy = fx.clamp(max=w - 1), fy.clamp(max=h - 1)          # (C++ would read out of bounds; never happens there)
    cx, cy = (fx + 1).clamp(max=w - 1), (fy + 1).clamp(max=h - 1)
    flat = covs.reshape(-1, 4)
    tl, tr, bl, br = flat[fy * w + fx], flat[fy * w + cx], flat[cy * w + fx], flat[cy * w + cx]

    def lerp2(i):
        top = tl[..., i] + frac_x * (tr[..., i] - tl[..., i])
        bot = bl[..., i] + frac_x * (br[..., i] - bl[..., i])
        return top + frac_y * (bot - top)

    xx, xy, yy = lerp2(0), lerp2(1), lerp2(3)
    det = xx * yy - xy * xy
    ok = det.abs() > EPS_DIV
    inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
    one, zero = torch.ones_like(det), torch.zeros_like(det)
    ixx = torch.where(ok, inv * yy, one)
    ixy = torch.where(ok, -inv * xy, zero)
    iyy = torch.where(ok, inv * xx, one)
    return _soften(ixx, ixy, iyy)


def _round_half_away(x):
    """lround for x >= 0 (merge positions are non-negative where used); exact, no float tie surprises."""
    f = torch.floor(x)
    return (f + ((x - f) >= 0.5).to(x.dtype)).long()


def apply_h(Hm, x, y):
    """merge.cpp apply_homography, float32, same operation order."""
    w = Hm[6] * x + Hm[7] * y + Hm[8]
    iw = torch.where(w != 0, 1.0 / torch.where(w != 0, w, torch.ones_like(w)), torch.zeros_like(w))
    return (Hm[0] * x + Hm[1] * y + Hm[2]) * iw, (Hm[3] * x + Hm[4] * y + Hm[5]) * iw


def out_grid(oy0, ox0, Ho, Wo, scale):
    """Reference raw coordinates of output pixels [oy0, oy0+Ho) x [ox0, ox0+Wo): lr = hr / scale (float32)."""
    s = torch.tensor(scale, dtype=F32)
    hy = torch.arange(oy0, oy0 + Ho, dtype=F32)[:, None].expand(Ho, Wo)
    hx = torch.arange(ox0, ox0 + Wo, dtype=F32)[None, :].expand(Ho, Wo)
    return hy / s, hx / s


def tile_correspondence(lr_y, lr_x, flow, Hm, has_h, tile):
    """Moved position of the merge for reference raw coords (lr_y, lr_x): H(lr + flow[trunc(lr/tile)]).
    flow [ny,nx,2] raw px (x, y); Hm [9] float32 (ignored unless has_h)."""
    ny, nx = flow.shape[:2]
    t = torch.tensor(float(tile), dtype=F32)
    px = torch.trunc(lr_x / t).long().clamp(0, nx - 1)
    py = torch.trunc(lr_y / t).long().clamp(0, ny - 1)
    fl = flow.reshape(-1, 2)[py * nx + px]
    mx, my = lr_x + fl[..., 0], lr_y + fl[..., 1]
    if has_h:
        mx, my = apply_h(Hm, mx, my)
    return my, mx


def _cfa_lut(cfa):
    return torch.tensor([cfa[0][0], cfa[0][1], cfa[1][0], cfa[1][1]], dtype=torch.long)


_TAPS_I = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.long).view(9, 1, 1)
_TAPS_J = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.long).view(9, 1, 1)


def accumulate(planes, my, mx, covs, cfa, raw_det, skip_outside):
    """Kernel gather of each plane in `planes` (list of ([H,W] tensor, squared_weights: bool)) at moved positions
    (my, mx) [Ho,Wo]: the 3x3 raw taps around round(moved), Gaussian weight exp(-0.5 d^T Omega d) from the covariance
    at the moved position, each tap accumulated into its CFA colour. -> list of [3,Ho,Wo], plus the weight sum B.
    All nine taps are evaluated at once (same per-tap arithmetic as merge.cpp; only the order in which the nine
    products are summed differs, a float32 rounding difference of ~1e-7 relative -- see tests/test_merge_parity)."""
    H, W = planes[0][0].shape
    Ho, Wo = my.shape
    kmap_i, kmap_j = (my - 0.5) / 2.0, (mx - 0.5) / 2.0
    ixx, ixy, iyy = interp_inv_cov(covs, kmap_i, kmap_j, raw_det)
    ci, cj = _round_half_away(my), _round_half_away(mx)
    valid = ((mx >= 0) & (mx < W) & (my >= 0) & (my < H)) if skip_outside else torch.ones_like(mx, dtype=torch.bool)
    i = ci[None] + _TAPS_I                                          # [9,Ho,Wo]
    j = cj[None] + _TAPS_J
    inb = valid[None] & (j >= 0) & (j < W) & (i >= 0) & (i < H)
    ii, jj = i.clamp(0, H - 1), j.clamp(0, W - 1)
    dx = jj.to(F32) - mx[None]
    dy = ii.to(F32) - my[None]
    z = ixx[None] * dx * dx + 2.0 * ixy[None] * dx * dy + iyy[None] * dy * dy
    wgt = torch.exp(-0.5 * z.clamp_min(0.0)) * inb.to(F32)
    lut = _cfa_lut(cfa)
    ch = lut[((ii & 1) << 1) | (jj & 1)]                            # [9,Ho,Wo] colour of each tap
    idx = (ii * W + jj).reshape(9, -1)
    onehot = [(ch == c).to(F32) for c in range(3)]
    wc = [wgt * oh for oh in onehot]                                # weight in its own colour, 0 elsewhere
    B = torch.stack([w_.sum(0) for w_ in wc])
    outs = []
    for p, sq in planes:
        v = p.reshape(-1)[idx].view(9, Ho, Wo)
        if sq:
            outs.append(torch.stack([(w_ * w_ * v).sum(0) for w_ in wc]))   # sum w^2 * var: the noise variance added
        else:
            outs.append(torch.stack([(w_ * v).sum(0) for w_ in wc]))        # sum w * c (R = 1)
    return outs, B


def comp_contrib(raw, clean, var, flow, Hm, has_h, covs, tile, scale, cfa, oy0, ox0, Ho, Wo, dense=None):
    """One comparison frame over output block [oy0,+Ho) x [ox0,+Wo). raw/clean/var [H,W] loader domain.
    dense: optional (my, mx) moved positions [Ho,Wo] replacing the tile/homography correspondence (ground truth).
    -> A (noisy), Ac (clean), B, V : [3,Ho,Wo] each (R = 1)."""
    lr_y, lr_x = out_grid(oy0, ox0, Ho, Wo, scale)
    if dense is None:
        my, mx = tile_correspondence(lr_y, lr_x, flow, Hm, has_h, tile)
    else:
        my, mx = dense
    (A, Ac, V), B = accumulate([(raw, False), (clean, False), (var, True)], my, mx, covs, cfa, True, True)
    return A, Ac, B, V


def ref_contrib(raw, clean, var, covs, scale, cfa, oy0, ox0, Ho, Wo):
    lr_y, lr_x = out_grid(oy0, ox0, Ho, Wo, scale)
    (A, Ac, V), B = accumulate([(raw, False), (clean, False), (var, True)], lr_y, lr_x, covs, cfa, False, False)
    return A, Ac, B, V


def fetch_index(oy0, ox0, Ho, Wo, scale, ry0, rx0, Hr, Wr, strict=True):
    """Raw pixel each output pixel reads a raw-resolution mask at (merge.cpp nearest fetch), as indices into an R map
    whose element [0,0] is raw pixel (ry0, rx0) and which is Hr x Wr. -> (fy, fx) long [Ho,Wo].
    strict=False clamps into the map exactly like merge.cpp clamps into the frame (use when the map IS the frame)."""
    lr_y, lr_x = out_grid(oy0, ox0, Ho, Wo, scale)
    fy = torch.floor(lr_y + 0.5).long() - ry0
    fx = torch.floor(lr_x + 0.5).long() - rx0
    if not strict:
        return fy.clamp(0, Hr - 1), fx.clamp(0, Wr - 1)
    if fy.min() < 0 or fx.min() < 0 or fy.max() >= Hr or fx.max() >= Wr:
        raise ValueError("R map does not cover the output block")
    return fy, fx


# ------------------------------------------------------------------------------------------------ recombination
class Parts:
    """Per-sample merge contributions over one output block. Tensors may carry a leading batch dim."""

    def __init__(self, Aref, Acref, Bref, Vref, A, Ac, B, V, fy, fx, Ag=None, Bg=None, vis=None):
        self.Aref, self.Acref, self.Bref, self.Vref = Aref, Acref, Bref, Vref
        self.A, self.Ac, self.B, self.V = A, Ac, B, V
        self.fy, self.fx = fy, fx
        # ideal counterpart: the same frames merged at their GROUND-TRUTH correspondences (clean), and the
        # visibility of the true point in each frame ([K,1,Ho,Wo], fetched like R): occluded samples are dropped
        self.Ag, self.Bg, self.vis = Ag, Bg, vis
        self.g = None          # per-frame worth g_k + n_k (frame_gains), set by the caller for the "linear" cost

    def fetch(self, R):
        """R [..., K, Hr, Wr] -> [..., K, 1, Ho, Wo] (the value each output pixel uses)."""
        return R[..., self.fy, self.fx].unsqueeze(-3)

    def den(self, Rf):
        return self.Bref + (Rf * self.B).sum(-4)

    def merged(self, Rf, clean=False):
        A, Aref = (self.Ac, self.Acref) if clean else (self.A, self.Aref)
        return (Aref + (Rf * A).sum(-4)) / self.den(Rf).clamp_min(1e-12)

    def variance(self, Rf):
        d = self.den(Rf).clamp_min(1e-12)
        return (self.Vref + (Rf * Rf * self.V).sum(-4)) / (d * d)

    def ideal(self, Rf):
        """The merge with the same weights, perfect correspondences and occluded samples removed."""
        rv = Rf * self.vis
        return (self.Acref + (rv * self.Ag).sum(-4)) / (self.Bref + (rv * self.Bg).sum(-4)).clamp_min(1e-12)


LUMA_W = (0.25, 0.5, 0.25)


def frame_gains(parts, T, sig2):
    """Per frame and output pixel, what the frame -- PERFECTLY aligned -- is worth to the merge at the all-frames
    operating point, in single-frame noise-variance units (luma-weighted):
        g_k = [(I_{-k} - T)^2 - (I_all - T)^2]_+      detail: its new sub-pixel samples bring the merge closer to the
                                                      HR target (aliasing that is signal, situation A)
        n_k = [Var_{-k} - Var_all]_+                  noise: how much it lowers the merge's noise variance (situation C)
    Clamped at 0, and evaluated at the all-frames point on purpose: the per-pixel optimum of the full cost also
    re-weights perfectly aligned samples to suit the merge kernel (subset selection on fine text, inverse-variance
    re-balancing of the narrow shipped kernels in flat areas -- measured, see labels.py). Neither is misalignment, and
    a robustness mask must not learn them. -> [K,1,Ho,Wo]."""
    w = torch.tensor(LUMA_W, dtype=F32).view(3, 1, 1)
    K = parts.A.shape[-4]
    ones = torch.ones(K, 1, *parts.fy.shape)
    e_all = (((parts.ideal(ones) - T) ** 2) * w).sum(-3) / sig2
    v_all = (parts.variance(ones) * w).sum(-3) / sig2
    g = []
    for k in range(K):
        r = ones.clone()
        r[k] = 0.0
        e_k = (((parts.ideal(r) - T) ** 2) * w).sum(-3) / sig2
        v_k = (parts.variance(r) * w).sum(-3) / sig2
        g.append((e_k - e_all).clamp_min(0.0) + (v_k - v_all).clamp_min(0.0))
    return torch.stack(g).unsqueeze(-3)


def cost_map(parts, Rf, T, sig2, gamma, keep_prior=0.0, mode="linear"):
    """Reconstruction cost per output pixel, in units of the single-frame noise variance sigma_1^2.

    mode "linear" (default; needs parts.Ag / Bg / vis and parts.g = frame_gains(...)):
        gamma sum_c w_c (M_c - I_c)^2 / sigma_1^2  +  sum_k (1 - r_k) (g_k + n_k + keep_prior)
      M = the merge of the noise-free frames with the USED correspondences, I = the same merge (same weights) at the
      TRUE correspondences with occluded samples dropped (Parts.ideal), g_k + n_k = what frame k, perfectly aligned,
      adds in detail and in noise reduction (frame_gains), keep_prior = the least a rejection has to buy. gamma prices
      the misalignment component (ghosts, doubled / shifted edges, wrong content) above noise: a misalignment is
      worse than falling back to fewer frames. A well-aligned sample (M = I) is kept by every term; a misaligned one
      is dropped exactly when the error it paints outweighs what it brings.
    mode "full": gamma (M - I)^2 + (I - T)^2 + Var (+ keep prior): the exact reconstruction cost. Its optimum
      re-weights perfectly aligned samples (42% of aligned fine-text samples, 46% of aligned flat-area samples with
      the shipped kernels -- measured), so it is kept for evaluation, not as the target.
    mode "hr": gamma (M - T)^2 + Var: also drops aligned samples on edges (18.7% on a real scene).
    -> [..., Ho, Wo]."""
    w = torch.tensor(LUMA_W, dtype=F32).view(3, 1, 1)
    M = parts.merged(Rf, clean=True)
    if mode == "linear" and parts.Ag is not None:
        I = parts.ideal(Rf)
        c = ((gamma * (M - I) ** 2) * w).sum(-3) / sig2
        g = parts.g if getattr(parts, "g", None) is not None else 0.0
        return c + ((1.0 - Rf) * (g + keep_prior)).sum((-4, -3))
    Var = parts.variance(Rf)
    if mode == "full" and parts.Ag is not None:
        I = parts.ideal(Rf)
        e = (gamma * (M - I) ** 2 + (I - T) ** 2 + Var) * w
    else:
        e = (gamma * (M - T) ** 2 + Var) * w
    c = e.sum(-3) / sig2
    if keep_prior > 0:
        c = c + keep_prior * (1.0 - Rf).sum((-4, -3))
    return c


def single_frame_sig2(T, prewhite, a, b):
    """Luma-weighted single-frame noise variance at the target (loader domain), floor 1e-12."""
    pw = torch.as_tensor(prewhite, dtype=F32).view(3, 1, 1)
    w = torch.tensor(LUMA_W, dtype=F32).view(3, 1, 1)
    s = (T / pw).clamp(0, 1)
    v = pw * pw * (a * s + b)
    return (v * w).sum(-3).clamp_min(1e-12)
