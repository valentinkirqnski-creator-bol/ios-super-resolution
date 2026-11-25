"""The SRSR merge (core/SRSRKernels.metal srsr_merge) as an exact, differentiable
function of the robustness masks.

The merge samples each comparison frame's mask ONCE per output pixel (bilinear,
at the output pixel's own position) and multiplies every tap of that frame by it:

    out = (A_ref + sum_f R_f * A_f) / max(B_ref + sum_f R_f * B_f, eps)

A_f / B_f (summed colour x weight, summed weight; 3 channels: R, G1+G2, B) do
not depend on the masks, so they are computed once per sample, without
gradients, by a line-by-line port of SampleNeighborhoodRBF / ProcessBayerQuad.
The only mask-dependent switch is the reference-kernel widening where the
accumulated weight is low (covariance x covarianceMultiplier); both reference
variants are precomputed and chosen per pixel (no gradient through the switch).
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

_INT = ["outW", "outH", "bayerW", "bayerH", "quadW", "quadH", "frameCount", "baseIndex",
        "bandY0", "bandRows", "applyRejectionWeight", "pad0"]
_FLT = ["coverageThreshold", "covarianceMultiplier", "accumulatedWeightScale", "kernelEpsilon",
        "normalization", "footprintLimit", "borderMinX", "borderMinY", "borderMaxX", "borderMaxY",
        "dehomogenizeEpsilon", "pad1"]


def consts(z):
    """Merge uniforms, per-frame normalisation and geometry from a sample."""
    ub = np.asarray(z["uniforms"], np.uint8).tobytes()
    u = dict(zip(_INT, np.frombuffer(ub[:48], np.int32).tolist()))
    u.update(zip(_FLT, np.frombuffer(ub[48:96], np.float32).tolist()))
    norms = np.frombuffer(np.asarray(z["norms"], np.uint8).tobytes(), np.float32).reshape(-1, 8)
    gb = np.asarray(z["geometry"], np.uint8).tobytes()
    geo = np.frombuffer(gb, np.float32).reshape(-1, 8)[:, :4]
    is_base = np.frombuffer(gb, np.int32).reshape(-1, 8)[:, 4]
    return u, norms, geo, is_base


def meta(z):
    return json.loads(np.asarray(z["meta"], np.uint8).tobytes().decode())


def bilinear(img, uv):
    """Metal sample(kLinearClamp, uv) on img [H, W, C] at normalised uv [P, 2] (clamp to edge)."""
    H, W = img.shape[:2]
    fx = uv[:, 0] * W - 0.5
    fy = uv[:, 1] * H - 0.5
    bx, by = torch.floor(fx), torch.floor(fy)
    ax, ay = (fx - bx)[:, None], (fy - by)[:, None]
    x0 = bx.long().clamp(0, W - 1); x1 = (bx.long() + 1).clamp(0, W - 1)
    y0 = by.long().clamp(0, H - 1); y1 = (by.long() + 1).clamp(0, H - 1)
    a, b, c, d = img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]
    return (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay


def _neighbourhood(quad, norm, uv, cov, u):
    """SampleNeighborhoodRBF: colour and weight sums [P, 3] of one frame at uv [P, 2]."""
    qH, qW = quad.shape[:2]
    bayer = torch.tensor([u["bayerW"], u["bayerH"]], dtype=torch.float32)
    coord = uv * bayer - 0.5
    gather = torch.floor((coord - 0.5) * 0.5) * 2.0
    sub = gather - coord
    base = torch.floor(gather * 0.5).long()
    gains = torch.tensor(norm[:4], dtype=torch.float32)
    blacks = torch.tensor(norm[4:], dtype=torch.float32)
    eps, lim, nrm = u["kernelEpsilon"], u["footprintLimit"], u["normalization"]
    cxx, cyy, cxy = cov[:, 0], cov[:, 1], cov[:, 2]
    col = torch.zeros(uv.shape[0], 3)
    wsum = torch.zeros(uv.shape[0], 3)
    for y in range(2):
        for x in range(2):
            qx = (base[:, 0] + x).clamp(0, qW - 1)
            qy = (base[:, 1] + y).clamp(0, qH - 1)
            rggb = quad[qy, qx].float() * gains + blacks              # [P, 4]
            off = sub + torch.tensor([2.0 * x, 2.0 * y])
            ws = []
            for dx, dy in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):   # R, G1, G2, B
                ddx, ddy = off[:, 0] + dx, off[:, 1] + dy
                dist = ddx * ddx * cxx + ddy * ddy * cyy + 2.0 * cxy * ddx * ddy
                inside = torch.maximum(ddx.abs(), ddy.abs()) <= lim
                ws.append(torch.where(inside, torch.exp2(-0.5 * dist) * nrm + eps, torch.zeros_like(dist)))
            col += torch.stack([ws[0] * rggb[:, 0], ws[1] * rggb[:, 1] + ws[2] * rggb[:, 2], ws[3] * rggb[:, 3]], 1)
            wsum += torch.stack([ws[0], ws[1] + ws[2], ws[3]], 1)
    return col, wsum


def output_uv(u):
    oH, oW = u["outH"], u["outW"]
    yy, xx = torch.meshgrid(torch.arange(oH, dtype=torch.float32), torch.arange(oW, dtype=torch.float32), indexing="ij")
    return torch.stack([(xx.reshape(-1) + 0.5) / oW, (yy.reshape(-1) + 0.5) / oH], 1)


class Sample:
    """The merge inputs of one sample as tensors (decoded once)."""

    def __init__(self, z):
        self.u, self.norms, self.geo, is_base = consts(z)
        assert is_base[0] == 1 and not is_base[1:].any(), "slot 0 must be the reference"
        self.quads = torch.from_numpy(np.asarray(z["quads"]).astype(np.int32))
        self.covs = torch.from_numpy(np.asarray(z["covs"]).astype(np.float32))
        self.flows = torch.from_numpy(np.asarray(z["flows"]).astype(np.float32))
        self.n = self.quads.shape[0]

    def rggb(self, f):
        """Frame f's normalised quads [qH, qW, 4] (R G1 G2 B, merge units)."""
        nm = self.norms[f]
        return self.quads[f].float() * torch.tensor(nm[:4]) + torch.tensor(nm[4:])

    def flow_at(self, f, uv):
        """The merge's flow fetch for slot f >= 1 at uv [P, 2]: nearest flow tile -> uv offset [P, 2]."""
        g = self.geo[f]
        ftc = uv * torch.tensor([g[0], g[1]]) + torch.tensor([g[2], g[3]])
        fl = self.flows[f - 1]
        fh, fw = fl.shape[0], fl.shape[1]
        fx = torch.floor(ftc[:, 0] * fw).long().clamp(0, fw - 1)
        fy = torch.floor(ftc[:, 1] * fh).long().clamp(0, fh - 1)
        return fl[fy, fx, :2]


@torch.no_grad()
def terms_uv(S, uv, chunk=1 << 18):
    """Merge terms at output positions uv [P, 2] (normalised):
       A, B [N-1, P, 3] (comparison slots 1.., mask = 1); A0, B0 / A0w, B0w [P, 3] (reference,
       normal / widened kernel)."""
    u = S.u
    n, P = S.n, uv.shape[0]
    A = torch.zeros(n - 1, P, 3); B = torch.zeros(n - 1, P, 3)
    A0 = torch.zeros(P, 3); B0 = torch.zeros(P, 3); A0w = torch.zeros(P, 3); B0w = torch.zeros(P, 3)
    bmin = torch.tensor([u["borderMinX"], u["borderMinY"]]); bmax = torch.tensor([u["borderMaxX"], u["borderMaxY"]])
    for s in range(0, P, chunk):
        q = uv[s:s + chunk]
        for f in range(1, n):
            su = q + S.flow_at(f, q)
            su = torch.where(su <= bmin, 2.0 * bmin - su, su)
            su = torch.where(su > bmax, 2.0 * bmax - su, su)
            c, w = _neighbourhood(S.quads[f], S.norms[f], su, bilinear(S.covs[f], su), u)
            A[f - 1, s:s + chunk] = c; B[f - 1, s:s + chunk] = w
        cov0 = bilinear(S.covs[0], q)
        c, w = _neighbourhood(S.quads[0], S.norms[0], q, cov0, u)
        A0[s:s + chunk] = c; B0[s:s + chunk] = w
        c, w = _neighbourhood(S.quads[0], S.norms[0], q, cov0 * u["covarianceMultiplier"], u)
        A0w[s:s + chunk] = c; B0w[s:s + chunk] = w
    return dict(A=A, B=B, A0=A0, B0=B0, A0w=A0w, B0w=B0w, u=u)


@torch.no_grad()
def precompute(z, chunk=1 << 18):
    """Merge terms over the whole output grid, shaped [.., oH, oW, 3]."""
    S = Sample(z)
    u = S.u
    oH, oW = u["outH"], u["outW"]
    t = terms_uv(S, output_uv(u), chunk)
    out = {k: v.view(*v.shape[:-2], oH, oW, 3) for k, v in t.items() if k != "u"}
    out["u"] = u
    return out


def sample_crop(m, x0, y0, uv, qW, qH):
    """Metal sample(kLinearClamp) of a quad-resolution map cropped out of the full [qH, qW]
    grid at (x0, y0): m [.., h, w], uv [P, 2] normalised over the full grid -> [.., P].
    The crop must contain every texel the full-grid sample would read (after edge clamping)."""
    fx = uv[:, 0] * qW - 0.5
    fy = uv[:, 1] * qH - 0.5
    bx, by = torch.floor(fx), torch.floor(fy)
    ax, ay = fx - bx, fy - by
    x0i = bx.long().clamp(0, qW - 1) - x0; x1i = (bx.long() + 1).clamp(0, qW - 1) - x0
    y0i = by.long().clamp(0, qH - 1) - y0; y1i = (by.long() + 1).clamp(0, qH - 1) - y0
    a, b = m[..., y0i, x0i], m[..., y0i, x1i]
    c, d = m[..., y1i, x0i], m[..., y1i, x1i]
    return (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay


def merge_uv(t, masks, x0, y0, uv, quantise_masks=False):
    """Differentiable merge at uv [P, 2] from terms_uv(..., uv) and the masks of slots 1..
    over a quad crop at (x0, y0): masks [F, h, w]. Returns [P, 3]."""
    u = t["u"]
    qW, qH = u["quadW"], u["quadH"]
    if quantise_masks:
        masks = masks + (torch.round(masks * 255) / 255 - masks).detach()
    acc = accumulated_r8(masks, u["accumulatedWeightScale"])
    weight_so_far = sample_crop(acc, x0, y0, uv, qW, qH) * u["accumulatedWeightScale"]
    widen = (weight_so_far < u["coverageThreshold"])[:, None]
    A0 = torch.where(widen, t["A0w"], t["A0"])
    B0 = torch.where(widen, t["B0w"], t["B0"])
    r = sample_crop(masks, x0, y0, uv, qW, qH)[..., None]            # [F, P, 1]
    num = A0 + (r * t["A"]).sum(0)
    den = B0 + (r * t["B"]).sum(0)
    return num / torch.clamp(den, min=u["dehomogenizeEpsilon"])


def masks_at_output(masks, oH, oW):
    """Bilinear sample of quad-resolution masks [.., F, qH, qW] at every output pixel centre
    (Metal sample(kLinearClamp, zoomed_tc)) -> [.., F, oH, oW]. Differentiable."""
    shp = masks.shape
    m = masks.reshape(-1, 1, shp[-2], shp[-1])
    yy = (torch.arange(oH, dtype=torch.float32) + 0.5) / oH * 2 - 1
    xx = (torch.arange(oW, dtype=torch.float32) + 0.5) / oW * 2 - 1
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None].expand(m.shape[0], oH, oW, 2)
    out = F.grid_sample(m, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.reshape(*shp[:-2], oH, oW)


def accumulated_r8(masks, scale):
    """COPY_MASK: saturate(sum of masks / 25) stored R8 (quantised, no gradient needed)."""
    with torch.no_grad():
        return torch.round((masks.sum(-3) / scale).clamp(0, 1) * 255) / 255


def merge(terms, masks, quantise_masks=False):
    """Differentiable SRSR merge. masks: [F, qH, qW] keep weights in [0, 1] for slots 1..
    (R8-quantise them with quantise_masks=True to reproduce the device bit for bit).
    Returns [oH, oW, 3]."""
    u = terms["u"]
    oH, oW = u["outH"], u["outW"]
    if quantise_masks:
        masks = masks + (torch.round(masks * 255) / 255 - masks).detach()
    acc = accumulated_r8(masks, u["accumulatedWeightScale"])                    # [qH, qW]
    weight_so_far = masks_at_output(acc[None], oH, oW)[0] * u["accumulatedWeightScale"]
    widen = (weight_so_far < u["coverageThreshold"])[..., None]
    A0 = torch.where(widen, terms["A0w"], terms["A0"])
    B0 = torch.where(widen, terms["B0w"], terms["B0"])
    r = masks_at_output(masks, oH, oW)[..., None]                               # [F, oH, oW, 1]
    num = A0 + (r * terms["A"]).sum(0)
    den = B0 + (r * terms["B"]).sum(0)
    return num / torch.clamp(den, min=u["dehomogenizeEpsilon"])
