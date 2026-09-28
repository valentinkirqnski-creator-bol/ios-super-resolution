"""The merge, ported so that R can be trained against its output.

The point of this file is one algebraic observation about core/merge.cpp
accumulate_comp: `local_r` is sampled ONCE per output pixel and then multiplied
into every tap of the 3x3 gather. So per output pixel and colour channel

    num = A_ref + sum_n A_n * R_n      A_n = sum_k w_nk c_nk [chan(k) == ch]
    den = B_ref + sum_n B_n * R_n      B_n = sum_k w_nk      [chan(k) == ch]

with A_n, B_n independent of R. Precompute those once per burst and the merged
image becomes an exact, cheap, differentiable function of the masks -- no
approximation of the kernel, the CFA or the bounds handling anywhere.

Mirrors accumulate_comp / accumulate_ref / estimate_kernels for the shipping
configuration (scale 2, Bayer RGGB, Steerable kernel, Linear selection).
"""
from __future__ import annotations

import numpy as np

import srsim
from srsim import CFA, SCALE, Cfg


# --------------------------------------------------------------------------
# kernel estimation (core/kernels.cpp estimate_kernels, Alg. 5)
# --------------------------------------------------------------------------

def compute_grey_decimate(raw):
    """One output pixel per 2x2 quad, the plain average."""
    return (0.25 * (raw[0::2, 0::2].astype(np.float32) + raw[0::2, 1::2] +
                    raw[1::2, 0::2] + raw[1::2, 1::2])).astype(np.float32)


def apply_gat(img, alpha, beta):
    """Generalized Anscombe VST, utils_image.GAT."""
    c = 0.375 * alpha * alpha + beta
    return ((2.0 / alpha) * np.sqrt(np.maximum(0.0, alpha * img + c))).astype(np.float32)


def compute_gradients(grey):
    """core/grey_pyramid.cpp compute_gradients; output is 1 px smaller each way.
       gx = 0.25*((tr-tl)+(br-bl)), gy = 0.25*((bl-tl)+(br-tr))"""
    tl = grey[:-1, :-1]
    tr = grey[:-1, 1:]
    bl = grey[1:, :-1]
    br = grey[1:, 1:]
    gx = 0.25 * ((tr - tl) + (br - bl))
    gy = 0.25 * ((bl - tl) + (br - tr))
    return gx.astype(np.float32), gy.astype(np.float32)


def _eigen_sym_2x2(m00, m01, m11):
    """linalg.h eigen_elmts_2x2, vectorised, including its exact branches.
    Returns (l_max, l_min, e1, e2) with e1/e2 stacked on the last axis."""
    b = -(m00 + m11)
    c = m00 * m11 - m01 * m01
    delta = np.maximum(b * b - 4.0 * c, 0.0)
    sq = np.sqrt(delta)
    r1 = (-b + sq) / 2.0
    r2 = (-b - sq) / 2.0
    # real_polyroots_2 orders by |root| descending; for a PSD structure tensor
    # r1 >= r2 >= 0, so this is (max, min).
    take1 = np.abs(r1) >= np.abs(r2)
    l0 = np.where(take1, r1, r2)
    l1 = np.where(take1, r2, r1)

    # e1 = (m00 + m01 - l1, m01 + m11 - l1), then the three C++ branches.
    e1x = m00 + m01 - l1
    e1y = m01 + m11 - l1
    nrm = np.sqrt(e1x * e1x + e1y * e1y)
    safe = np.where(nrm > 0.0, nrm, 1.0)
    nx_ = e1x / safe
    ny_ = e1y / safe
    sign = np.where(nx_ >= 0.0, 1.0, -1.0)
    e2x = -ny_ * sign
    e2y = np.abs(nx_)

    # branch: e1x == 0 -> e1 = (0, 1), e2 = (1, 0)
    z0 = e1x == 0.0
    nx_ = np.where(z0, 0.0, nx_)
    ny_ = np.where(z0, 1.0, ny_)
    e2x = np.where(z0, 1.0, e2x)
    e2y = np.where(z0, 0.0, e2y)
    # branch: e1y == 0 (and e1x != 0) -> e1 = (1, 0), e2 = (0, 1)
    z1 = (~z0) & (e1y == 0.0)
    nx_ = np.where(z1, 1.0, nx_)
    ny_ = np.where(z1, 0.0, ny_)
    e2x = np.where(z1, 0.0, e2x)
    e2y = np.where(z1, 1.0, e2y)
    # branch: m01 == 0 and m00 == m11 -> axis aligned (checked FIRST in C++)
    iso = (m01 == 0.0) & (m00 == m11)
    nx_ = np.where(iso, 1.0, nx_)
    ny_ = np.where(iso, 0.0, ny_)
    e2x = np.where(iso, 0.0, e2x)
    e2y = np.where(iso, 1.0, e2y)
    return l0, l1, (nx_, ny_), (e2x, e2y)


def estimate_kernels(raw, cfg: Cfg):
    """Alg. 5. Returns covs of shape (h/2, w/2, 4) laid out xx, xy, yx, yy."""
    grey = compute_grey_decimate(raw)
    vst = apply_gat(grey, cfg.alpha, cfg.beta)
    gx, gy = compute_gradients(vst)
    H, W = vst.shape
    gh, gw = gx.shape

    # Structure tensor over the 2x2 gradient neighbourhood at (y-1+i, x-1+j),
    # out-of-range taps skipped (cuda_estimate_kernel).
    s00 = np.zeros((H, W), np.float64)
    s01 = np.zeros((H, W), np.float64)
    s11 = np.zeros((H, W), np.float64)
    for i in range(2):
        for j in range(2):
            yy = np.arange(H) - 1 + i
            xx = np.arange(W) - 1 + j
            vy = (yy >= 0) & (yy < gh)
            vx = (xx >= 0) & (xx < gw)
            yi = np.clip(yy, 0, gh - 1)
            xi = np.clip(xx, 0, gw - 1)
            m = vy[:, None] & vx[None, :]
            gxv = gx[np.ix_(yi, xi)] * m
            gyv = gy[np.ix_(yi, xi)] * m
            s00 += gxv * gxv
            s01 += gxv * gyv
            s11 += gyv * gyv

    l0, l1, e1, e2 = _eigen_sym_2x2(s00, s01, s11)

    # compute_k, SelectionLaw::Linear
    tot = l0 + l1
    with np.errstate(invalid='ignore', divide='ignore'):
        A = 1.0 + np.sqrt(np.maximum((l0 - l1) / np.where(tot > 0, tot, 1.0), 0.0))
    D = np.clip(1.0 - np.sqrt(np.maximum(l0, 0.0)) / cfg.D_tr + cfg.D_th, 0.0, 1.0)
    ok = (tot > 0.0) & np.isfinite(A)
    kk1 = np.where(ok, (2.0 - A) + (A - 1.0) / srsim.K_SHRINK, 1.0)
    kk2 = np.where(ok, (2.0 - A) + (A - 1.0) * srsim.K_STRETCH, 1.0)
    k1 = cfg.k_detail * ((1.0 - D) * kk1 + D * cfg.k_denoise)
    k2 = cfg.k_detail * ((1.0 - D) * kk2 + D * cfg.k_denoise)

    k1s, k2s = k1 * k1, k2 * k2
    e1x, e1y = e1
    e2x, e2y = e2
    covs = np.empty((H, W, 4), np.float32)
    covs[..., 0] = k1s * e1x * e1x + k2s * e2x * e2x
    covs[..., 1] = k1s * e1x * e1y + k2s * e2x * e2y
    covs[..., 2] = covs[..., 1]
    covs[..., 3] = k1s * e1y * e1y + k2s * e2y * e2y
    return covs


# --------------------------------------------------------------------------
# inverse covariance sampling (merge.cpp interp_inv_cov + soften_inv_cov)
# --------------------------------------------------------------------------

def _soften(ixx, ixy, iyy):
    k = 32.0
    m = np.maximum(np.abs(ixx), np.maximum(np.abs(iyy), np.abs(ixy)))
    bad = ~(np.isfinite(ixx) & np.isfinite(ixy) & np.isfinite(iyy))
    big = np.isfinite(m) & (m > k)
    s = np.where(big, k / np.where(m > 0, m, 1.0), 1.0)
    ixx = np.where(bad, 2.0, ixx * s)
    ixy = np.where(bad, 0.0, ixy * s)
    iyy = np.where(bad, 2.0, iyy * s)
    return ixx, ixy, iyy


def interp_inv_cov(covs, kmap_i, kmap_j, raw_det):
    """Bilinear sample of covs then invert. raw_det selects the accumulate()
    variant (int() index, bare 1/det); otherwise accumulate_ref's (floor index,
    invert_2x2). Both take the fractional part with trunc, which KEEPS THE SIGN
    for negative positions -- reproduced rather than tidied up."""
    H, W = covs.shape[:2]
    frac_x = kmap_j - np.trunc(kmap_j)
    frac_y = kmap_i - np.trunc(kmap_i)
    if raw_det:
        fx = np.maximum(kmap_j.astype(np.int64), 0)
        fy = np.maximum(kmap_i.astype(np.int64), 0)
    else:
        fx = np.maximum(np.floor(kmap_j).astype(np.int64), 0)
        fy = np.maximum(np.floor(kmap_i).astype(np.int64), 0)
    cx = np.minimum(fx + 1, W - 1)
    cy = np.minimum(fy + 1, H - 1)
    fx = np.minimum(fx, W - 1)
    fy = np.minimum(fy, H - 1)

    def lerp2(idx):
        tl = covs[fy, fx, idx]
        tr = covs[fy, cx, idx]
        bl = covs[cy, fx, idx]
        br = covs[cy, cx, idx]
        top = tl + frac_x * (tr - tl)
        bot = bl + frac_x * (br - bl)
        return top + frac_y * (bot - top)

    xx = lerp2(0).astype(np.float64)
    xy = lerp2(1).astype(np.float64)
    yy = lerp2(3).astype(np.float64)
    det = xx * yy - xy * xy
    good = np.abs(det) > 1e-10
    inv_det = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
    ixx = np.where(good, inv_det * yy, 1.0)
    ixy = np.where(good, -inv_det * xy, 0.0)
    iyy = np.where(good, inv_det * xx, 1.0)
    return _soften(ixx, ixy, iyy)


# --------------------------------------------------------------------------
# A / B accumulation
# --------------------------------------------------------------------------

def _round_half_away(x):
    """std::lround / CUDA round: half away from zero. np.rint is half to EVEN,
    which differs on exact .5 -- and these positions land on exact halves
    constantly, because lr = hr/2. Getting this wrong shifts the whole gather
    by one pixel on every other output column."""
    return np.floor(np.abs(x) + 0.5).astype(np.int64) * np.where(x < 0, -1, 1)


def accumulate_comp_ab(raw, flowx, flowy, covs, cfg: Cfg, stride=1):
    """accumulate_comp with local_r factored out. flowx/flowy are given at
    OUTPUT resolution so that either a per-tile or a per-pixel (ground truth)
    field can be fed in. Returns A, B of shape (3, Hs, Ws)."""
    h, w = raw.shape
    # The loss is evaluated on every `stride`-th output pixel. The merge is
    # pointwise in output space once A and B are factored out, so a subsampled
    # grid is the same estimator on a subset -- and it is what makes a crop big
    # enough for the coarse branch to have real context affordable to store.
    Hs, Ws = (h * SCALE) // stride, (w * SCALE) // stride
    hi, hj = np.mgrid[0:Hs, 0:Ws]
    hi = hi * stride
    hj = hj * stride
    lr_x = hj / SCALE
    lr_y = hi / SCALE

    lr_mov_x = lr_x + flowx
    lr_mov_y = lr_y + flowy
    valid = (lr_mov_x >= 0) & (lr_mov_x < w) & (lr_mov_y >= 0) & (lr_mov_y < h)

    kmap_j = lr_mov_x / 2.0 - 0.5
    kmap_i = lr_mov_y / 2.0 - 0.5
    ixx, ixy, iyy = interp_inv_cov(covs, kmap_i, kmap_j, raw_det=True)

    cj = _round_half_away(lr_mov_x)
    ci = _round_half_away(lr_mov_y)

    A = np.zeros((3, Hs, Ws), np.float32)
    B = np.zeros((3, Hs, Ws), np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = ci + di
            j = cj + dj
            inb = (i >= 0) & (i < h) & (j >= 0) & (j < w) & valid
            ic = np.clip(i, 0, h - 1)
            jc = np.clip(j, 0, w - 1)
            ch = CFA[ic & 1, jc & 1]
            c = raw[ic, jc]
            dx = j - lr_mov_x
            dy = i - lr_mov_y
            z = np.maximum(ixx * dx * dx + 2.0 * ixy * dx * dy + iyy * dy * dy, 0.0)
            wt = np.where(inb, np.exp(-0.5 * z), 0.0)
            for cc in range(3):
                m = (ch == cc)
                A[cc] += np.where(m, wt * c, 0.0)
                B[cc] += np.where(m, wt, 0.0)
    return A, B


def accumulate_ref_ab(raw, covs, cfg: Cfg, stride=1):
    """accumulate_ref (Alg. 11): no flow, R == 1, fixed radius 1."""
    h, w = raw.shape
    Hs, Ws = (h * SCALE) // stride, (w * SCALE) // stride
    hi, hj = np.mgrid[0:Hs, 0:Ws]
    hi = hi * stride
    hj = hj * stride
    coarse_x = hj / SCALE
    coarse_y = hi / SCALE
    kmap_j = (coarse_x - 0.5) / 2.0
    kmap_i = (coarse_y - 0.5) / 2.0
    ixx, ixy, iyy = interp_inv_cov(covs, kmap_i, kmap_j, raw_det=False)
    cj = _round_half_away(coarse_x)
    ci = _round_half_away(coarse_y)

    A = np.zeros((3, Hs, Ws), np.float32)
    B = np.zeros((3, Hs, Ws), np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = ci + di
            j = cj + dj
            inb = (i >= 0) & (i < h) & (j >= 0) & (j < w)
            ic = np.clip(i, 0, h - 1)
            jc = np.clip(j, 0, w - 1)
            ch = CFA[ic & 1, jc & 1]
            c = raw[ic, jc]
            dx = j - coarse_x
            dy = i - coarse_y
            z = np.maximum(ixx * dx * dx + 2.0 * ixy * dx * dy + iyy * dy * dy, 0.0)
            wt = np.where(inb, np.exp(-0.5 * z), 0.0)
            for cc in range(3):
                m = (ch == cc)
                A[cc] += np.where(m, wt * c, 0.0)
                B[cc] += np.where(m, wt, 0.0)
    return A, B


def tile_flow_at_output(flow, h, w, ts, stride=1):
    """The per-tile flow the merge actually fetches: nearest tile, indexed by
    int(lr / tile_size) -- accumulate_comp does not interpolate."""
    Hs, Ws = (h * SCALE) // stride, (w * SCALE) // stride
    hi, hj = np.mgrid[0:Hs, 0:Ws]
    hi = hi * stride
    hj = hj * stride
    lr_x = hj / SCALE
    lr_y = hi / SCALE
    px = (lr_x / ts).astype(np.int64)
    py = (lr_y / ts).astype(np.int64)
    py = np.clip(py, 0, flow.shape[0] - 1)
    px = np.clip(px, 0, flow.shape[1] - 1)
    return flow[py, px, 0].astype(np.float64), flow[py, px, 1].astype(np.float64)


def sample_r_at_output(R, h, w, stride=1, guide_scale=1):
    """merge_robustness_bilinear with rob_is_raw: R is sampled bilinearly at
    (lr_y, lr_x) = (hr/2, hr/2). Returns the 4 corner indices and weights so
    the same gather can be done in torch during training."""
    Hs, Ws = (h * SCALE) // stride, (w * SCALE) // stride
    hi, hj = np.mgrid[0:Hs, 0:Ws]
    hi = hi * stride
    hj = hj * stride
    # merge.cpp: rob_is_raw -> sample at lr directly; otherwise the guide is half
    # resolution and the position is (lr - 0.5) / 2.
    gh = h // guide_scale
    gw = w // guide_scale
    if guide_scale == 1:
        y = np.clip(hi / SCALE, 0, gh - 1)
        x = np.clip(hj / SCALE, 0, gw - 1)
    else:
        y = np.clip((hi / SCALE - 0.5) / 2.0, 0, gh - 1)
        x = np.clip((hj / SCALE - 0.5) / 2.0, 0, gw - 1)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    # Clamped to the GUIDE extent, not the raw one: R lives on the guide
    # lattice, which is half the raw size when guide_scale is 2.
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    fy = (y - y0).astype(np.float32)
    fx = (x - x0).astype(np.float32)
    if R is None:
        return (y0, x0, y1, x1, fy, fx)
    top = R[y0, x0] + (R[y0, x1] - R[y0, x0]) * fx
    bot = R[y1, x0] + (R[y1, x1] - R[y1, x0]) * fx
    return top + (bot - top) * fy


def merge_from_ab(A_ref, B_ref, A, B, Rout):
    """num/den with the masks applied. Rout is (N, Hs, Ws)."""
    num = A_ref.astype(np.float64).copy()
    den = B_ref.astype(np.float64).copy()
    for n in range(A.shape[0]):
        num += A[n] * Rout[n][None, :, :]
        den += B[n] * Rout[n][None, :, :]
    return (num / np.maximum(den, 1e-8)).astype(np.float32)
