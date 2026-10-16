"""Shared numeric helpers.

Every helper that the C++/Metal port mirrors is written so the arithmetic order
matches the shader loop order (window sums accumulate dy-outer / dx-inner and
divide once at the end; bilinear uses the same expression tree). That keeps the
float results within a few ULP of the device port instead of "close enough".
"""
import torch
import torch.nn.functional as F


def pad_rep(x, r):
    """Replicate-pad a [B,H,W] tensor by r on every side (== clamp-to-edge)."""
    if r == 0:
        return x
    return F.pad(x.unsqueeze(1), (r, r, r, r), mode="replicate").squeeze(1)


def win_mean(x, r):
    """Mean over a (2r+1)^2 window with clamp-to-edge, shader summation order."""
    return win_mean_offsets(x, list(range(-r, r + 1)))


def win_mean_offsets(x, offs):
    """Mean over taps (dy,dx) in offs x offs, clamp-to-edge."""
    B, H, W = x.shape
    P = max(abs(o) for o in offs)
    xp = pad_rep(x, P)
    acc = None
    for dy in offs:
        for dx in offs:
            t = xp[:, P + dy:P + dy + H, P + dx:P + dx + W]
            acc = t.clone() if acc is None else acc + t
    return acc / float(len(offs) * len(offs))


def bilinear_clamp(img, x, y):
    """Bilinear sample of img [B,H,W] at float coords x,y [B,h,w], clamp-to-edge."""
    B, H, W = img.shape
    x = x.clamp(0.0, float(W - 1))
    y = y.clamp(0.0, float(H - 1))
    x0f = torch.floor(x)
    y0f = torch.floor(y)
    fx = x - x0f
    fy = y - y0f
    x0 = x0f.long()
    y0 = y0f.long()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    flat = img.reshape(B, -1)

    def g(yy, xx):
        return torch.gather(flat, 1, (yy * W + xx).reshape(B, -1)).reshape(x.shape)

    v00, v01, v10, v11 = g(y0, x0), g(y0, x1), g(y1, x0), g(y1, x1)
    return (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01) + fy * ((1.0 - fx) * v10 + fx * v11)


def warp_by_flow(img, flow):
    """img [B,H,W], flow [B,2,H,W] (x,y) in pixels: out(p) = img(p + flow(p))."""
    B, H, W = img.shape
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, W) + flow[:, 0]
    ys = torch.arange(H, dtype=torch.float32).view(1, H, 1) + flow[:, 1]
    return bilinear_clamp(img, xs, ys)


def gauss_blur(x, sigma):
    """Separable Gaussian blur of [B,H,W] (replicate pad). Not mirrored on device."""
    if sigma <= 0:
        return x
    r = max(1, int(3.0 * sigma + 0.5))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    xp = F.pad(x.unsqueeze(1), (r, r, r, r), mode="replicate")
    xp = F.conv2d(xp, k.view(1, 1, 1, -1))
    xp = F.conv2d(xp, k.view(1, 1, -1, 1))
    return xp.squeeze(1)


def bayer_to_grey(raw):
    """raw [..., H, W] RGGB -> grey [..., H/2, W/2] = mean of each 2x2 quad."""
    return 0.25 * (raw[..., 0::2, 0::2] + raw[..., 0::2, 1::2]
                   + raw[..., 1::2, 0::2] + raw[..., 1::2, 1::2])


def logit(p, eps=1e-3):
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))
