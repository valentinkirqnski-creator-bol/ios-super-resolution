"""A tiny depthwise-separable CNN that REFINES the Wronski mask.

    M_final = M_wronski * g,      g = sigmoid(net(features)) in (0, 1)

so M_final <= M_wronski at every pixel, for every input, for any weights. That
is a property of the form, not of training: the sigmoid is strictly positive and
strictly below one, and the product with a non-negative mask cannot exceed it.
The network can preserve or withdraw acceptance and can never grant it.

Polarity, from merge.cpp and asserted in test_invariant.py: the merge
accumulates val += w*M*c and acc += w*M, so M = 1 accepts a sample fully and
M = 0 removes it. "Reject more" therefore means driving M toward 0, and the
guarantee above is that the refinement only ever moves in that direction.

Shape: depthwise 3x3 + pointwise 1x1 blocks, 16 channels, four of them, then a
1x1 head. No attention, no normalisation layers, no large feature maps -- every
op is one a mobile NPU runs natively and quantises without special handling.
Dilation widens the last two blocks to a 25-pixel receptive field, which is what
reaches across a motion boundary without the cost of downsampling and upsampling
again.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SepBlock(nn.Module):
    """Depthwise 3x3 (dilated) then pointwise 1x1, ReLU on both.

    Replicate padding, never zeros: a zero border is an absolute position code,
    and a fully convolutional mask network that learns to read one stops working
    at any size but the one it trained at.
    """

    def __init__(self, cin, cout, dil=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, groups=cin, dilation=dil, bias=True)
        self.pw = nn.Conv2d(cin, cout, 1, bias=True)
        self.dil = dil

    def forward(self, x):
        x = F.pad(x, (self.dil,) * 4, mode='replicate')
        return F.relu(self.pw(F.relu(self.dw(x))))


class MaskRefineNet(nn.Module):
    def __init__(self, in_ch=12, width=16, dilations=(1, 1, 2, 4),
                 head_bias=1.0):
        super().__init__()
        chans = [in_ch] + [width] * len(dilations)
        self.blocks = nn.ModuleList(
            SepBlock(chans[i], chans[i + 1], d) for i, d in enumerate(dilations))
        self.head = nn.Conv2d(width, 1, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        # Opens at g = sigmoid(head_bias). 1.0 gives 0.73, not the 0.95 the
        # first runs used: the smooth oracle's gate averages 0.55, so starting
        # at 0.95 meant travelling a long way in logit space before any useful
        # structure could form, and three runs slid back to the identity
        # instead. 0.73 still merges most of the burst, so a failed run is
        # conservative rather than destructive, but there is gradient room in
        # both directions from the start.
        nn.init.constant_(self.head.bias, head_bias)
        self.dilations = dilations

    def gate(self, feat):
        x = feat
        for b in self.blocks:
            x = b(x)
        return torch.sigmoid(self.head(x))

    def forward(self, feat, mask):
        """feat (B,N,C,h,w), mask (B,N,h,w) -> refined mask (B,N,h,w)."""
        b, n = feat.shape[0], feat.shape[1]
        g = self.gate(feat.flatten(0, 1))[:, 0].view(b, n, *feat.shape[-2:])
        return mask * g, g

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def receptive_field(self):
        rf = 1
        for d in self.dilations:
            rf += 2 * d
        return rf
