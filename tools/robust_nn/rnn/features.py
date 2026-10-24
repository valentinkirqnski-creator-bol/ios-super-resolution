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

NF = 15
NAMES = ["r0_logit", "mean_diff", "resid_e5", "edge_e5", "ls_shift", "ls_explained", "anisotropy",
         "log_bright", "clip_frac", "flow_affine_center", "flow_affine_rms", "flow_affine_grad",
         "flow_max_nbr_dev", "resid_e9", "tile_pred_resid"]


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
    return center, rms, grad, md, (ax1, ax2, ay1, ay2)


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
    center, rms, grad, md, (ax1, ax2, ay1, ay2) = _flow_features(flow if fflow is None else fflow, fs)
    f9 = torch.log(1.0 + center)
    f10 = torch.log(1.0 + rms)
    f11 = torch.log(1.0 + grad)
    f12 = torch.log(1.0 + md)
    r9 = win_mean_offsets(r * r, [-4, -2, 0, 2, 4])
    f13 = torch.log(1.0 + r9 / s2)
    # v34: predicted in-tile residual of the one-translation-per-tile model (mirrors robust_nn_shared.h f[14]):
    # the affine fit's change of the tile flow per tile step, times the pixel's offset from its own tile centre
    # (tile grid origin 0, fs px) -> how far this pixel's true motion differs from its tile's vector (a roll: the
    # stepped edge at tile borders). Flow only, noise-free.
    H, W = Ir.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=Ir.dtype), torch.arange(W, dtype=Ir.dtype), indexing="ij")
    ox = (xs - torch.div(xs, fs, rounding_mode="floor") * fs) - 0.5 * (fs - 1)
    oy = (ys - torch.div(ys, fs, rounding_mode="floor") * fs) - 0.5 * (fs - 1)
    rx = (ax1 * ox + ax2 * oy) / float(fs)
    ry = (ay1 * ox + ay2 * oy) / float(fs)
    f14 = torch.log(1.0 + torch.sqrt(rx * rx + ry * ry))
    return torch.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14], 1), J


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


# ---- v26: edge-line shift (what the eye sees on a straight edge) -------------------------------------------------
NF_EDGE = 3
NAMES_EDGE = ["edge_line_shift", "edge_line_signif", "edge_line_step"]


def edge_line_features(Ir, J, a, b, L=None, band=None):
    """Ir [K,H,W] reference grey (per comparison frame), J [K,H,W] warped comparison grey. At every pixel: the edge
    normal n from the 5x5 structure tensor of the reference, then 2L+1 samples ALONG the edge (p + s t, t = n rotated
    90 deg, bilinear) of the residual r = J - Ir and of the reference gradient across the edge g = grad Ir . n. A shift
    of this frame's edge across itself is delta = sum(g r) / (sum(g^2) + lambda); averaging along the edge length is
    what lets a 0.2-0.3 px shift of a straight edge stand out of one frame's noise (the eye integrates the same way).
    Returns [K,3,H,W]: |delta| (guide px, clamped 2), its significance log(1 + z^2) with z^2 = (sum g r)^2 /
    (s2r sum g^2), and the significance of the difference between the left (s <= 0) and right (s >= 0) halves - a step
    along the edge, i.e. a zipper at a tile border."""
    import os as _os
    L = int(_os.environ.get("EDGEF_L", "8")) if L is None else L            # half length along the edge
    band = int(_os.environ.get("EDGEF_BAND", "1")) if band is None else band  # half width across it
    K, H, W = J.shape
    Ip = pad_rep(Ir, 1)
    gx = 0.5 * (Ip[:, 1:-1, 2:] - Ip[:, 1:-1, :-2])
    gy = 0.5 * (Ip[:, 2:, 1:-1] - Ip[:, :-2, 1:-1])
    Sxx, Syy, Sxy = win_mean(gx * gx, 2), win_mean(gy * gy, 2), win_mean(gx * gy, 2)
    th = 0.5 * torch.atan2(2.0 * Sxy, Sxx - Syy)
    nx, ny = torch.cos(th), torch.sin(th)
    tx, ty = -ny, nx
    mu = win_mean(Ir, 1).clamp_min(0.0)
    s2 = (a * mu + b).clamp_min(1e-10)
    s2r = 2.0 * s2                                            # variance of a residual of two noisy frames
    lam = 0.5 * s2
    src = torch.stack([gx, gy, J - Ir], 1)                    # [K,3,H,W]
    yy, xx = torch.meshgrid(torch.arange(H, dtype=Ir.dtype), torch.arange(W, dtype=Ir.dtype), indexing="ij")
    A_l = torch.zeros_like(Ir); B_l = torch.zeros_like(Ir); A_r = torch.zeros_like(Ir); B_r = torch.zeros_like(Ir)
    A = torch.zeros_like(Ir); B = torch.zeros_like(Ir)
    for s in range(-L, L + 1):
        for w in range(-band, band + 1):               # a band 2*band+1 wide across the edge
            px = (xx[None] + s * tx + w * nx) / max(W - 1, 1) * 2 - 1
            py = (yy[None] + s * ty + w * ny) / max(H - 1, 1) * 2 - 1
            smp = F.grid_sample(src, torch.stack([px, py], -1), mode="bilinear", padding_mode="border",
                                align_corners=True)
            g = smp[:, 0] * nx + smp[:, 1] * ny
            gr, gg = g * smp[:, 2], g * g
            A, B = A + gg, B + gr
            if s < 0:
                A_l, B_l = A_l + gg, B_l + gr
            elif s > 0:
                A_r, B_r = A_r + gg, B_r + gr
    delta = B / (A + lam)
    f0 = delta.abs().clamp(max=2.0)
    f1 = torch.log(1.0 + B * B / (s2r * A.clamp_min(1e-12)))
    dl, dr = B_l / (A_l + lam), B_r / (A_r + lam)
    var_d = s2r * (1.0 / A_l.clamp_min(1e-12) + 1.0 / A_r.clamp_min(1e-12))
    f2 = torch.log(1.0 + (dl - dr) ** 2 / var_d)
    return torch.stack([f0, f1, f2], 1)


def tile_pool_features(Ir, J, a, b, ts):
    """Ir/J [K,H,W] reference and warped comparison grey, ts = the flow's tile size in these pixels. A wrong tile
    vector moves the WHOLE tile, so the evidence for it is pooled over the tile (ts*ts pixels, not a 5x5 window): a
    noise-weighted least-squares fit of the residual r = J - Ir to grad(Ir) . d (translation, 2 params) and to
    grad(Ir) . (d + M x) (affine, 6 params - tile content slightly rotated). Returns [K,4,H,W], constant per tile:
    |d| (px, clamped 2), log(1 + z2_trans), log(1 + z2_aff - z2_trans) (what the rotation adds), and the fraction of
    the tile's residual energy the affine fit explains. z2 = b^T A^-1 b is chi-square (2 / 6 dof) under pure noise."""
    K, H, W = J.shape
    Ip = pad_rep(Ir, 1)
    gx = 0.5 * (Ip[:, 1:-1, 2:] - Ip[:, 1:-1, :-2])
    gy = 0.5 * (Ip[:, 2:, 1:-1] - Ip[:, :-2, 1:-1])
    w = 1.0 / (2.0 * (a * win_mean(Ir, 1).clamp_min(0.0) + b)).clamp_min(1e-10)   # 1 / var of a two-frame residual
    r = J - Ir
    yy, xx = torch.meshgrid(torch.arange(H, dtype=Ir.dtype), torch.arange(W, dtype=Ir.dtype), indexing="ij")
    xc = ((xx % ts) - 0.5 * (ts - 1)) / ts                    # position inside the tile, [-0.5, 0.5]
    yc = ((yy % ts) - 0.5 * (ts - 1)) / ts
    phi = torch.stack([gx, gy, gx * xc, gx * yc, gy * xc, gy * yc], 1)            # [K,6,H,W]
    Ht, Wt = -(-H // ts), -(-W // ts)
    pool = lambda v: F.avg_pool2d(v, ts, ts, ceil_mode=True) * (ts * ts)          # per-tile sums (approx at edges)
    A = pool((phi[:, :, None] * phi[:, None] * w[:, None, None]).reshape(K, 36, H, W)).reshape(K, 6, 6, Ht, Wt)
    bb = pool(phi * (r * w)[:, None])                                             # [K,6,Ht,Wt]
    e = pool((r * r * w)[:, None])[:, 0]
    A = A.permute(0, 3, 4, 1, 2)
    bb = bb.permute(0, 2, 3, 1)
    lam = 0.25 * ts * ts * torch.eye(6, dtype=Ir.dtype) * 1e-2
    At = A[..., :2, :2] + lam[:2, :2]
    dt = torch.linalg.solve(At, bb[..., :2, None])[..., 0]
    z2t = (bb[..., :2] * dt).sum(-1)
    da = torch.linalg.solve(A + lam, bb[..., None])[..., 0]
    z2a = (bb * da).sum(-1)
    feats_t = torch.stack([dt.norm(dim=-1).clamp(max=2.0), torch.log1p(z2t.clamp_min(0.0)),
                           torch.log1p((z2a - z2t).clamp_min(0.0)), (z2a / e.clamp_min(1e-6)).clamp(0.0, 1.0)], 1)
    up = feats_t.repeat_interleave(ts, 2).repeat_interleave(ts, 3)
    return up[..., :H, :W]
