"""Kernel-gather multi-frame Bayer merge (super-resolution style).

For output pixel o (reference raw coords x_r) and frame k: look at
x_k = x_r + 2*flow_k(grey(x_r)), gather the 3x3 raw samples around it, weight
each by a Gaussian of its distance to x_k, accumulate into its own colour
channel. merged_c = sum_k R_k * num_k,c / sum_k R_k * den_k,c.

Per-frame (num, den) are kept separate, so any weighting — including
leave-one-out for the utility label — is a cheap recombination.
"""
import torch

from .common import gauss_blur


def contributions(raw, flows, scale=1, sigma=0.7):
    """raw [N,Hr,Wr]; flows [N,2,Hg,Wg] grey px (frame 0 zeros) -> num, den [N,3,Ho,Wo]."""
    N, Hr, Wr = raw.shape
    Hg, Wg = flows.shape[-2:]
    Ho, Wo = Hr * scale, Wr * scale
    oy, ox = torch.meshgrid(torch.arange(Ho, dtype=torch.float32), torch.arange(Wo, dtype=torch.float32),
                            indexing="ij")
    xr = (ox + 0.5) / scale - 0.5
    yr = (oy + 0.5) / scale - 0.5
    gx = torch.floor(xr / 2).long().clamp(0, Wg - 1)
    gy = torch.floor(yr / 2).long().clamp(0, Hg - 1)
    num = torch.zeros(N, 3, Ho, Wo)
    den = torch.zeros(N, 3, Ho, Wo)
    chan_of = torch.tensor([0, 1, 1, 2])
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for k in range(N):
        xk = xr + 2.0 * flows[k, 0][gy, gx]
        yk = yr + 2.0 * flows[k, 1][gy, gx]
        bx = torch.floor(xk + 0.5).long()
        by = torch.floor(yk + 0.5).long()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = bx + dx, by + dy
                valid = ((nx >= 0) & (nx < Wr) & (ny >= 0) & (ny < Hr)).float()
                v = raw[k][ny.clamp(0, Hr - 1), nx.clamp(0, Wr - 1)]
                d2 = (nx.float() - xk) ** 2 + (ny.float() - yk) ** 2
                w = torch.exp(-d2 * inv2s2) * valid
                ch = chan_of[((ny & 1) << 1) | (nx & 1)]
                for c in range(3):
                    m = (ch == c).float() * w
                    num[k, c] += m * v
                    den[k, c] += m
    return num, den


def upsample_weights(R, Ho, Wo, scale):
    """Grey-res weights [N,Hg,Wg] -> output grid (nearest, same lookup as the gather)."""
    Hg, Wg = R.shape[-2:]
    oy = torch.arange(Ho, dtype=torch.float32)
    ox = torch.arange(Wo, dtype=torch.float32)
    gy = torch.floor(((oy + 0.5) / scale - 0.5) / 2).long().clamp(0, Hg - 1)
    gx = torch.floor(((ox + 0.5) / scale - 0.5) / 2).long().clamp(0, Wg - 1)
    return R[..., gy, :][..., gx]


def combine(num, den, Rout):
    """Rout [N,Ho,Wo] (frame 0 should be 1). Returns merged [3,Ho,Wo]."""
    n = (num * Rout[:, None]).sum(0)
    d = (den * Rout[:, None]).sum(0)
    return n / d.clamp_min(1e-12)


def psnr(a, b, border=8):
    if border:
        a = a[..., border:-border, border:-border]
        b = b[..., border:-border, border:-border]
    mse = ((a - b) ** 2).mean().clamp_min(1e-12)
    return float(10 * torch.log10(1.0 / mse))


def local_err(m, gt, sigma):
    return gauss_blur(((m - gt) ** 2).sum(0, keepdim=True), sigma)[0]
