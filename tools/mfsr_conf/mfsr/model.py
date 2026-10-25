"""ConfNet: per-frame confidence maps for the app's merge.

    per frame k (shared weights)        guide res (one value per RGGB quad)
      x_g [18]  -> conv3x3 -> 4 x (dilated conv3x3, dilations 1,2,4,8)       receptive field 31 quads = 62 raw px
                -> features f_k [ch]                                         (about two 16 px tiles each side)
    cross-frame fusion (optional)
      [f_k, mean_j f_j, max_j f_j] -> conv1x1 -> conv3x3 -> h_k [ch]         consensus: is this frame the odd one out?
    heads at guide res
      h_k -> conv1x1 -> (alignment logit, merge logit)
    raw-resolution refinement (x2)
      [up(conv1x1(h_k)), x_r [3], up(logits)] -> conv3x3 -> conv1x1 -> residual logits
    outputs per raw pixel: A_k = sigmoid(alignment logit)  "this sample comes from the right scene point"
                           R_k = sigmoid(merge logit)      the robustness weight the merge multiplies the sample by
                           (0 = discarded, 1 = fully merged, anything between = partial weight)

Optional classical combination (model.refine_r0): merge logit = logit(R0) + delta, i.e. the network refines the
app's analytic mask instead of replacing it (R0 then also enters as feature 0).

Replicate padding everywhere (zero padding leaks absolute position near borders). Inputs are standardised by fixed
per-channel statistics measured on training data (buffers; folded into the first layer at export).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .inputs import NG, NR


def _conv(i, o, k=3, d=1):
    return nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, padding_mode="replicate")


class ConfNet(nn.Module):
    def __init__(self, ch=16, fusion=True, dils=(1, 2, 4, 8), up_ch=8, refine_r0=False, nin_g=NG, nin_r=NR):
        super().__init__()
        self.ch, self.fusion, self.refine_r0 = ch, bool(fusion), bool(refine_r0)
        self.register_buffer("mu_g", torch.zeros(nin_g))
        self.register_buffer("sd_g", torch.ones(nin_g))
        self.register_buffer("mu_r", torch.zeros(nin_r))
        self.register_buffer("sd_r", torch.ones(nin_r))
        self.enc = nn.ModuleList([_conv(nin_g, ch, 3, dils[0])] + [_conv(ch, ch, 3, d) for d in dils[1:]])
        self.f1 = nn.Conv2d(3 * ch if self.fusion else ch, ch, 1)
        self.f2 = _conv(ch, ch)
        self.head = nn.Conv2d(ch, 2, 1)
        self.up_ch = int(up_ch)
        if self.up_ch > 0:      # up_ch = 0: no raw-resolution refinement, guide-res logits upsampled nearest (v21-like)
            self.u0 = nn.Conv2d(ch, up_ch, 1)
            self.u1 = _conv(up_ch + nin_r + 2, up_ch)
            self.u2 = nn.Conv2d(up_ch, 2, 1)
            nn.init.zeros_(self.u2.weight)
            nn.init.zeros_(self.u2.bias)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.5, 0.0 if self.refine_r0 else 2.0]))

    def forward(self, xg, xr, mask, r0=None):
        """xg [B,K,NG,h,w], xr [B,K,NR,2h,2w], mask [B,K] (1 = real frame), r0 [B,K,2h,2w] (refine_r0 only).
        -> logits [B,K,2,2h,2w]: channel 0 alignment, 1 merge."""
        B, K, _, h, w = xg.shape
        x = (xg - self.mu_g.view(1, 1, -1, 1, 1)) / self.sd_g.view(1, 1, -1, 1, 1)
        f = x.reshape(B * K, -1, h, w)
        for c in self.enc:
            f = F.relu(c(f))
        f = f.view(B, K, self.ch, h, w)
        if self.fusion:
            m = mask.view(B, K, 1, 1, 1).to(f.dtype)
            n = m.sum(1).clamp_min(1.0)
            mean = (f * m).sum(1) / n
            mx = (f * m + (m - 1.0) * 1e4).max(1).values
            g = torch.cat([f, mean[:, None].expand_as(f), mx[:, None].expand_as(f)], 2)
        else:
            g = f
        g = F.relu(self.f1(g.reshape(B * K, -1, h, w)))
        g = F.relu(self.f2(g))
        lg = self.head(g)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        lg_up = up(lg)
        if self.up_ch > 0:
            hu = F.relu(up(self.u0(g)))
            xr_ = (xr - self.mu_r.view(1, 1, -1, 1, 1)) / self.sd_r.view(1, 1, -1, 1, 1)
            r = F.relu(self.u1(torch.cat([hu, xr_.reshape(B * K, -1, 2 * h, 2 * w), lg_up], 1)))
            lg_up = lg_up + self.u2(r)
        out = lg_up.view(B, K, 2, 2 * h, 2 * w)
        if self.refine_r0:
            z = torch.logit(r0.clamp(1e-3, 1 - 1e-3))
            out = torch.stack([out[:, :, 0], out[:, :, 1] + z], 2)
        return out

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def macs_per_quad(self):
        """Multiply-accumulates per guide pixel (one RGGB quad) and frame, i.e. per 4 raw pixels."""
        m = 0
        for c in self.enc:
            m += c.in_channels * c.out_channels * 9
        m += self.f1.in_channels * self.f1.out_channels + self.f2.in_channels * self.f2.out_channels * 9
        m += self.head.in_channels * 2
        if self.up_ch > 0:
            m += self.u0.in_channels * self.u0.out_channels
            m += 4 * (self.u1.in_channels * self.u1.out_channels * 9 + self.u2.in_channels * 2)
        return m


def build(cfg_model):
    return ConfNet(ch=cfg_model["ch"], fusion=cfg_model["fusion"], dils=tuple(cfg_model["dils"]),
                   up_ch=cfg_model["up_ch"], refine_r0=cfg_model.get("refine_r0", False))
