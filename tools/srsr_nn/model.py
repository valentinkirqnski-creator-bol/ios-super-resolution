"""Learned robustness mask for SRSR (DBSR-style, Bhat et al. 2021, kept boring).

Per comparison frame, at quad resolution (one pixel per 2x2 CFA quad):
  1. a shared encoder turns the reference and the comparison frame into features
     (input: the quad's 4 CFA samples, variance-stabilised by the frame's own
     noise model, so the network sees everything in units of noise std);
  2. the comparison features are bilinearly warped onto the reference with the
     flow SRSR's aligner estimated (not the true flow: the net must learn the
     aligner's errors);
  3. a decoder reads [reference, warped, their difference, the analytic mask]
     and outputs a correction to the analytic mask's logit.
The last layer is zero-initialised, so an untrained network IS the analytic mask.

Two sizes: `big` (prove it trains) and `small` (the phone: 8-16 channels, a few
3x3 convs at quad resolution, run once per frame in the existing merge pass).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = {
    "big":   dict(c=64, enc=3, dec=4),
    "small": dict(c=16, enc=2, dec=3),
    "tiny":  dict(c=8, enc=2, dec=3),
}


def gat(rggb, a, b):
    """Generalised Anscombe transform: rggb [B, 4, H, W] with Var(x) = a x + b per CFA site
    (a, b [B, 4], in the merge's units) -> unit-variance units."""
    a = a.view(-1, 4, 1, 1).clamp_min(1e-9)
    b = b.view(-1, 4, 1, 1)
    return (2.0 / a) * torch.sqrt(torch.clamp(a * rggb + 0.375 * a * a + b, min=0.0))


EPS = 0.5 / 255


def logit(p, eps=EPS):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


def warp(feat, pos):
    """Bilinear fetch of feat [B, C, Hm, Wm] at pos [B, 2, H, W]: for each reference pixel, the
    position (x, y in feat's pixel grid) its correspondence has in the comparison frame."""
    Hm, Wm = feat.shape[-2:]
    grid = torch.stack([(pos[:, 0] + 0.5) / Wm * 2 - 1, (pos[:, 1] + 0.5) / Hm * 2 - 1], -1)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=False)


class RobustMaskNet(nn.Module):
    def __init__(self, c=64, enc=3, dec=4):
        super().__init__()
        e = [nn.Conv2d(4, c, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(enc - 1):
            e += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True)]
        self.enc = nn.Sequential(*e)
        d = [nn.Conv2d(3 * c + 2, c, 3, padding=1), nn.ReLU(inplace=True)]
        for i in range(dec - 1):
            dil = 2 ** (i + 1)
            d += [nn.Conv2d(c, c, 3, padding=dil, dilation=dil), nn.ReLU(inplace=True)]
        self.dec = nn.Sequential(*d)
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ref, mov, pos, flow, analytic, a_ref, b_ref, a_mov, b_mov):
        """ref [B, 4, H, W] reference quads (merge units, R G1 G2 B); mov [B, 4, Hm, Wm] comparison
        quads around the window; pos [B, 2, H, W] each reference pixel's position in mov's grid
        (from the aligner's flow); flow [B, 2, H, W] that flow in quad px; analytic [B, 1, H, W]
        the analytic dilated keep mask; a_*, b_* [B, 4] each frame's noise model per CFA site.
        Returns the keep mask [B, 1, H, W] in [0, 1]."""
        fr = self.enc(gat(ref, a_ref, b_ref))
        fm = warp(self.enc(gat(mov, a_mov, b_mov)), pos)
        mag = flow.norm(dim=1, keepdim=True)
        x = torch.cat([fr, fm, fr - fm, logit(analytic, EPS), torch.tanh(mag / 4.0)], 1)
        # The logit clamp maps the analytic mask into [EPS, 1 - EPS]; stretching back makes an
        # untrained net (zero head) return the analytic mask, 0 and 1 exactly included.
        p = torch.sigmoid(logit(analytic, EPS) + self.head(self.dec(x)))
        return ((p - EPS) / (1.0 - 2.0 * EPS)).clamp(0.0, 1.0)


def build(name="big"):
    return RobustMaskNet(**CONFIGS[name])


def receptive_halo(name="big"):
    cfg = CONFIGS[name]
    enc = cfg["enc"]
    dec = 1 + sum(2 ** (i + 1) for i in range(cfg["dec"] - 1))
    return enc + dec + 2       # + the warp's reach is the flow itself (handled by the crop margin)
