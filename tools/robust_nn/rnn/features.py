"""Per-pixel evidence features. MIRRORED EXACTLY in cpp/robust_nn_shared.h —
change both or neither (parity_test.py checks them against each other).

Inputs (grey = mean of each RGGB quad, black-subtracted, white level = 1):
  Ir    reference grey                [B,H,W]
  Ic    comparison grey (unwarped)     [B,H,W]
  flow  dense ref->cmp flow, grey px   [B,2,H,W]
  R0    analytic mask                  [B,H,W]
  a, b  grey-domain noise: var(v) = a*v + b
  clip  grey level treated as clipped
  fs    flow-sampling step in px (the app's tile stride)

Everything is normalised by the local noise sigma, so the same weights work
from ISO 50 to ISO 6400.

The features that separate "correct but different sub-pixel sampling" (KEEP)
from "wrong correspondence" (REJECT) are f4/f5: a local least-squares fit of
the residual r = J - Ir to the reference gradient, r ~ g . delta. A coherent
geometric displacement is explained by a single delta over a 5x5 window;
aliasing differences and noise are not. The flow features (f9-f12) carry the
correspondence-model evidence: tile flow that departs from a local affine
model (rotation / zoom / parallax / outliers) is suspect even where the images
happen to agree.
"""
import torch
import torch.nn.functional as F

from .common import pad_rep, win_mean, win_mean_offsets, warp_by_flow

NF = 14
NAMES = ["r0_logit", "mean_diff", "resid_e5", "edge_e5", "ls_shift", "ls_explained", "anisotropy",
         "log_bright", "clip_frac", "flow_affine_center", "flow_affine_rms", "flow_affine_grad",
         "flow_max_nbr_dev", "resid_e9"]


def _flow_features(flow, fs):
    B, _, H, W = flow.shape
    P = 2 * fs
    fxp = pad_rep(flow[:, 0], P)
    fyp = pad_rep(flow[:, 1], P)
    xs = torch.arange(W)
    ys = torch.arange(H)

    def tap(i, j):
        tx = fxp[:, P + j * fs:P + j * fs + H, P + i * fs:P + i * fs + W]
        ty = fyp[:, P + j * fs:P + j * fs + H, P + i * fs:P + i * fs + W]
        dxe = ((xs + i * fs).clamp(0, W - 1) - xs).float() / float(fs)
        dye = ((ys + j * fs).clamp(0, H - 1) - ys).float() / float(fs)
        return tx, ty, dxe.view(1, 1, W).expand(B, H, W), dye.view(1, H, 1).expand(B, H, W)

    z = torch.zeros(B, H, W)
    n = z.clone(); sx = z.clone(); sy = z.clone(); sxx = z.clone(); sxy = z.clone(); syy = z.clone()
    bx0 = z.clone(); bx1 = z.clone(); bx2 = z.clone(); by0 = z.clone(); by1 = z.clone(); by2 = z.clone()
    for j in range(-2, 3):
        for i in range(-2, 3):
            tx, ty, dx, dy = tap(i, j)
            n = n + 1.0
            sx = sx + dx; sy = sy + dy
            sxx = sxx + dx * dx; sxy = sxy + dx * dy; syy = syy + dy * dy
            bx0 = bx0 + tx; bx1 = bx1 + dx * tx; bx2 = bx2 + dy * tx
            by0 = by0 + ty; by1 = by1 + dx * ty; by2 = by2 + dy * ty
    c00 = sxx * syy - sxy * sxy
    c01 = sy * sxy - sx * syy
    c02 = sx * sxy - sxx * sy
    c11 = n * syy - sy * sy
    c12 = sx * sy - n * sxy
    c22 = n * sxx - sx * sx
    det = n * c00 + sx * c01 + sy * c02
    ok = det > 1e-3
    inv = torch.where(ok, 1.0 / torch.where(ok, det, torch.ones_like(det)), torch.zeros_like(det))
    ax0 = torch.where(ok, (c00 * bx0 + c01 * bx1 + c02 * bx2) * inv, bx0 / n)
    ax1 = (c01 * bx0 + c11 * bx1 + c12 * bx2) * inv
    ax2 = (c02 * bx0 + c12 * bx1 + c22 * bx2) * inv
    ay0 = torch.where(ok, (c00 * by0 + c01 * by1 + c02 * by2) * inv, by0 / n)
    ay1 = (c01 * by0 + c11 * by1 + c12 * by2) * inv
    ay2 = (c02 * by0 + c12 * by1 + c22 * by2) * inv
    e2 = z.clone()
    for j in range(-2, 3):
        for i in range(-2, 3):
            tx, ty, dx, dy = tap(i, j)
            ex = tx - (ax0 + ax1 * dx + ax2 * dy)
            ey = ty - (ay0 + ay1 * dx + ay2 * dy)
            e2 = e2 + (ex * ex + ey * ey)
    cx = flow[:, 0] - ax0
    cy = flow[:, 1] - ay0
    center = torch.sqrt(cx * cx + cy * cy)
    rms = torch.sqrt(e2 / n)
    grad = torch.sqrt(ax1 * ax1 + ax2 * ax2 + ay1 * ay1 + ay2 * ay2)
    md = z.clone()
    for (i, j) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        tx, ty, _, _ = tap(i, j)
        ex = flow[:, 0] - tx
        ey = flow[:, 1] - ty
        md = torch.maximum(md, torch.sqrt(ex * ex + ey * ey))
    return center, rms, grad, md


def compute_features(Ir, Ic, flow, R0, a, b, clip, fs, fflow=None):
    """Returns features [B,NF,H,W] and the warped comparison J [B,H,W]. fflow (v20): the flow the flow-consistency
    features read instead of flow (the residual tile flow after the app's global homography); the warp uses flow."""
    J = warp_by_flow(Ic, flow)
    mu_r = win_mean(Ir, 1)
    mu_j = win_mean(J, 1)
    s2 = (a * mu_r.clamp_min(0.0) + b).clamp_min(1e-10)
    r = J - Ir

    Rc = R0.clamp(1e-3, 1.0 - 1e-3)
    f0 = 0.25 * torch.log(Rc / (1.0 - Rc))
    dm = mu_j - mu_r
    f1 = torch.log(1.0 + dm * dm * 9.0 / s2)
    rr = win_mean(r * r, 2)
    f2 = torch.log(1.0 + rr / s2)

    Ip = pad_rep(Ir, 1)
    gx = 0.5 * (Ip[:, 1:-1, 2:] - Ip[:, 1:-1, :-2])
    gy = 0.5 * (Ip[:, 2:, 1:-1] - Ip[:, :-2, 1:-1])
    Sxx = win_mean(gx * gx, 2)
    Syy = win_mean(gy * gy, 2)
    Sxy = win_mean(gx * gy, 2)
    bx = win_mean(gx * r, 2)
    by = win_mean(gy * r, 2)
    f3 = torch.log(1.0 + (Sxx + Syy) / s2)
    lam = 0.5 * s2
    A11 = Sxx + lam
    A22 = Syy + lam
    det = A11 * A22 - Sxy * Sxy
    u = (A22 * bx - Sxy * by) / det
    v = (A11 * by - Sxy * bx) / det
    f4 = torch.sqrt(u * u + v * v).clamp(max=4.0)
    f5 = ((bx * u + by * v) / (rr + 1e-3 * s2)).clamp(0.0, 1.0)
    tr = Sxx + Syy
    dd = torch.sqrt((Sxx - Syy) * (Sxx - Syy) + 4.0 * Sxy * Sxy)
    f6 = dd / (tr + 2.0 * s2)
    f7 = 0.25 * torch.log(mu_r.clamp_min(1e-4))
    ci = (torch.maximum(Ir, J) > clip).float()
    f8 = win_mean(ci, 1)
    center, rms, grad, md = _flow_features(flow if fflow is None else fflow, fs)
    f9 = torch.log(1.0 + center)
    f10 = torch.log(1.0 + rms)
    f11 = torch.log(1.0 + grad)
    f12 = torch.log(1.0 + md)
    r9 = win_mean_offsets(r * r, [-4, -2, 0, 2, 4])
    f13 = torch.log(1.0 + r9 / s2)
    return torch.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13], 1), J


# ---- burst consistency (prototype, v8) ---------------------------------------------------------------
NF_CONS = 4
NAMES_CONS = ["cons_dis3", "cons_dis5", "cons_vs_ref3", "cons_explained"]


def consensus_features(Ir, J, r0, a, b):
    """Ir [H,W] reference grey; J [K,H,W] warped comparison greys (each by its own flow); r0 [K,H,W] analytic
    mask. For frame k the consensus is leave-one-out: (Ir + sum_{j!=k} r0_j J_j) / (1 + sum_{j!=k} r0_j),
    so a sample that disagrees with what the OTHER frames (and the reference) show stands out, while
    aliasing -- which differs between correctly aligned frames -- is averaged by the consensus."""
    K = J.shape[0]
    num = Ir[None] + (r0 * J).sum(0, keepdim=True)
    den = 1.0 + r0.sum(0, keepdim=True)
    C = (num - r0 * J) / (den - r0)                          # leave-one-out
    mu = win_mean(Ir[None], 1).expand(K, -1, -1)
    s2 = (a * mu.clamp_min(0.0) + b).clamp_min(1e-10)
    dJC = J - C
    dCR = C - Ir[None]
    dJR = J - Ir[None]
    c0 = torch.log(1.0 + win_mean(dJC * dJC, 1) / s2)
    c1 = torch.log(1.0 + win_mean(dJC * dJC, 2) / s2)
    c2 = torch.log(1.0 + win_mean(dCR * dCR, 1) / s2)
    c3 = win_mean(dJC * dJC, 2) / (win_mean(dJR * dJR, 2) + win_mean(dJC * dJC, 2) + s2)
    return torch.stack([c0, c1, c2, c3], 1)


# ---- full-resolution burst consistency (v9) ----------------------------------------------------------
NF_FULL = 5
NAMES_FULL = ["full_dis2", "full_dis4", "full_peak", "full_ls_shift", "full_ls_explained"]


def gather_raw(raw, flow_raw, sigma=0.6):
    """raw [H,W] pre-WB'd Bayer of one frame; flow_raw [2,H,W] raw-px ref->frame displacement per ref pixel.
    Returns a full-resolution luminance-like estimate of that frame at every reference pixel: Gaussian-weighted
    3x3 gather of its raw samples around p + flow (pre-white-balance makes the sites roughly comparable)."""
    H, W = raw.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    px, py = xx + flow_raw[0], yy + flow_raw[1]
    bx, by = torch.floor(px + 0.5).long(), torch.floor(py + 0.5).long()
    num = torch.zeros(H, W)
    den = torch.zeros(H, W)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx, ny = bx + dx, by + dy
            ok = ((nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)).float()
            w = torch.exp(-((nx.float() - px) ** 2 + (ny.float() - py) ** 2) / (2 * sigma * sigma)) * ok
            num += w * raw[ny.clamp(0, H - 1), nx.clamp(0, W - 1)]
            den += w
    return num / den.clamp_min(1e-8)


def full_consensus_features(raws, flows_guide, r0, alpha, beta):
    """raws [N,H,W] pre-WB'd Bayer (frame 0 = reference); flows_guide [K,2,H/2,W/2] the per-pixel flow the mask
    scores (grey px); r0 [K,H/2,W/2]. Leave-one-out consensus at SENSOR resolution, features pooled to the
    guide grid. Device cost: one fp16 plane at sensor res + one guide-res weight plane (~36 MB at 12 MP)."""
    N, H, W = raws.shape
    K = N - 1
    up = lambda t: t.repeat_interleave(2, -2).repeat_interleave(2, -1)[..., :H, :W]
    L0 = gather_raw(raws[0], torch.zeros(2, H, W))
    Ls = torch.stack([gather_raw(raws[k + 1], up(2.0 * flows_guide[k])) for k in range(K)])
    w = up(r0)
    num = L0 + (w * Ls).sum(0)
    den = 1.0 + w.sum(0)
    C = (num[None] - w * Ls) / (den[None] - w)               # leave-one-out, per frame
    s2 = (alpha * C.clamp_min(0) + beta).clamp_min(1e-10)
    e = Ls - C
    e2 = e * e / s2
    pool2 = lambda t: F.avg_pool2d(t[:, None], 2)[:, 0]
    g0 = torch.log(1.0 + pool2(e2))
    g1 = torch.log(1.0 + F.avg_pool2d(F.pad(e2[:, None], (1, 1, 1, 1), mode="replicate"), 4, 2, 0)[:, 0][:, :H // 2, :W // 2])
    g2 = torch.log(1.0 + F.max_pool2d((e.abs() / s2.sqrt())[:, None], 2)[:, 0])
    # least-squares fit of the residual to a shift of the consensus, over a 6x6 raw window per quad
    gxC = 0.5 * (F.pad(C, (1, 1, 0, 0), mode="replicate")[..., 2:] - F.pad(C, (1, 1, 0, 0), mode="replicate")[..., :-2])
    gyC = 0.5 * (F.pad(C[:, None], (0, 0, 1, 1), mode="replicate")[:, 0, 2:] - F.pad(C[:, None], (0, 0, 1, 1), mode="replicate")[:, 0, :-2])
    box = lambda t: F.avg_pool2d(F.pad(t[:, None], (2, 2, 2, 2), mode="replicate"), 6, 2, 0)[:, 0][:, :H // 2, :W // 2]
    Sxx, Syy, Sxy = box(gxC * gxC), box(gyC * gyC), box(gxC * gyC)
    bx, by, ee = box(gxC * e), box(gyC * e), box(e * e)
    lam = 0.5 * box(s2)
    det = (Sxx + lam) * (Syy + lam) - Sxy * Sxy
    u = ((Syy + lam) * bx - Sxy * by) / det
    v = ((Sxx + lam) * by - Sxy * bx) / det
    g3 = torch.sqrt(u * u + v * v).clamp(max=4.0)
    g4 = ((bx * u + by * v) / (ee + 1e-3 * lam)).clamp(0.0, 1.0)
    return torch.stack([g0, g1, g2, g3, g4], 1)
