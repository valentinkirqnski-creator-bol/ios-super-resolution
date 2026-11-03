"""Compact v15: ONE small pixel-seeing net that replaces the v14 + PixNet stack, for the same GPU memory as v14
and less compute.

Inputs per guide pixel: v14's 14 features (standardised with v14's statistics) + the 4 noise-normalised pixel
channels. A 1x1 projection, then depthwise-separable blocks (3x3 per-channel filter, dilated to widen the view,
followed by a 1x1 channel mix), then a 1x1 head that adds a correction to the analytic mask's logit (as v14 does).
With ch=16 and dilations (1, 2, 4, 1): ~1.9k multiply-adds per pixel and a 17x17 view (v14: ~5.1k, v14 + PixNet:
~15k). Replicate padding (no zero-padding position code)."""
import torch
import torch.nn as nn


class CompactNet(nn.Module):
    def __init__(self, mu, sd, npix=4, ch=16, dil=(1, 2, 4, 1), first3=False):
        super().__init__()
        self.register_buffer("mu", mu.clone())
        self.register_buffer("sd", sd.clone())
        nf = mu.numel()
        self.npix = npix
        self.first3 = first3   # full 3x3 first layer (like v14's) instead of a 1x1 projection
        self.pin = (nn.Conv2d(nf + npix, ch, 3, padding=1, padding_mode="replicate") if first3
                    else nn.Conv2d(nf + npix, ch, 1))
        self.dw = nn.ModuleList(nn.Conv2d(ch, ch, 3, padding=d, dilation=d, groups=ch, padding_mode="replicate")
                                for d in dil)
        self.pw = nn.ModuleList(nn.Conv2d(ch, ch, 1) for _ in dil)
        self.head = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.act = nn.ReLU()

    def macs_per_pixel(self):
        ch, cin = self.pin.out_channels, self.pin.in_channels
        return cin * ch * (9 if self.first3 else 1) + len(self.dw) * (9 * ch + ch * ch) + ch

    def forward(self, feat, r0, pix):
        x = torch.cat([(feat - self.mu.view(1, -1, 1, 1)) / self.sd.view(1, -1, 1, 1), pix[:, :self.npix]], 1)
        h = self.act(self.pin(x))
        for dw, pw in zip(self.dw, self.pw):
            h = self.act(pw(dw(h)))
        d = self.head(h)[:, 0]
        r = r0.clamp(1e-3, 1 - 1e-3)
        return torch.sigmoid(torch.log(r) - torch.log1p(-r) + d), d
