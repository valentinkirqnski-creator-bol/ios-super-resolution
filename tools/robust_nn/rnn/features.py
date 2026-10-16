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


def compute_features(Ir, Ic, flow, R0, a, b, clip, fs):
    """Returns features [B,NF,H,W] and the warped comparison J [B,H,W]."""
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
    center, rms, grad, md = _flow_features(flow, fs)
    f9 = torch.log(1.0 + center)
    f10 = torch.log(1.0 + rms)
    f11 = torch.log(1.0 + grad)
    f12 = torch.log(1.0 + md)
    r9 = win_mean_offsets(r * r, [-4, -2, 0, 2, 4])
    f13 = torch.log(1.0 + r9 / s2)
    return torch.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13], 1), J
