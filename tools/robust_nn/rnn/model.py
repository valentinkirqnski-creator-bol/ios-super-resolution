"""The refinement network (~2.2k parameters).

    features (14) --3x3 conv--> 16 --ReLU--> 1x1 --> 16 --ReLU--> 1x1 --> delta
    R = sigmoid( logit(R0) + delta )

The last layer is zero-initialised, so at step 0 the network IS the analytic
mask; training only learns where and by how much to move away from it.
Padding is replicate (== clamp-to-edge in the shader): zero padding would let
the net read absolute position near borders. Feature standardisation is folded
into the first layer at export, so the device runs exactly three matmuls.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import logit
from .features import NF

HID = 16


class RobustNet(nn.Module):
    def __init__(self, nf=NF, hid=HID):
        super().__init__()
        self.register_buffer("mu", torch.zeros(nf))
        self.register_buffer("sd", torch.ones(nf))
        self.c1 = nn.Conv2d(nf, hid, 3)
        self.c2 = nn.Conv2d(hid, hid, 1)
        self.c3 = nn.Conv2d(hid, 1, 1)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def forward(self, feat, r0):
        x = (feat - self.mu.view(1, -1, 1, 1)) / self.sd.view(1, -1, 1, 1)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        d = self.c3(h)[:, 0]
        return torch.sigmoid(logit(r0) + d), d

    def folded(self):
        """First-layer weights/bias with the standardisation folded in."""
        w = self.c1.weight.detach() / self.sd.view(1, -1, 1, 1)
        b = self.c1.bias.detach() - (w * self.mu.view(1, -1, 1, 1)).sum((1, 2, 3))
        return w, b


class RobustNetCtx(RobustNet):
    """v28: RobustNet plus a WIDE view. The 3x3 net sees ~11 guide px; a wrong tile vector moves a whole 16 px tile
    and a zipper shows across tile borders. A context branch on 4x-pooled features (average AND max per 4x4 block,
    3x3 conv, then a dilated 3x3 conv: ~32 guide px = ~2x2 tiles) is upsampled (bilinear) and added into the hidden
    layer through a zero-initialised 1x1, so at step 0 the network IS the base RobustNet. Cost on device: one
    1/16-resolution pass (ctx channels) plus ctx x hid MACs per pixel."""
    def __init__(self, nf=NF, hid=HID, ctx=8, pool=4):
        super().__init__(nf, hid)
        self.pool = pool
        self.k1 = nn.Conv2d(2 * nf, ctx, 3)
        self.k2 = nn.Conv2d(ctx, ctx, 3, dilation=2)
        self.k3 = nn.Conv2d(ctx, hid, 1)
        nn.init.zeros_(self.k3.weight)
        nn.init.zeros_(self.k3.bias)

    def forward(self, feat, r0):
        x = (feat - self.mu.view(1, -1, 1, 1)) / self.sd.view(1, -1, 1, 1)
        H, W = x.shape[-2:]
        xp = torch.cat([F.avg_pool2d(x, self.pool, ceil_mode=True), F.max_pool2d(x, self.pool, ceil_mode=True)], 1)
        c = F.relu(self.k1(F.pad(xp, (1, 1, 1, 1), mode="replicate")))
        c = F.relu(self.k2(F.pad(c, (2, 2, 2, 2), mode="replicate")))
        c = F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)
        h = F.relu(self.c1(F.pad(x, (1, 1, 1, 1), mode="replicate")))
        h = F.relu(self.c2(h) + self.k3(c))
        d = self.c3(h)[:, 0]
        return torch.sigmoid(logit(r0) + d), d


class RobustNetTile(RobustNet):
    """v30: the wide view aligned to the flow TILE grid (origin 0, ts guide px). Each feature is averaged and maxed
    over the pixel's OWN tile (a wrong tile vector moves the whole tile), a 1x1 makes a per-tile summary, a 3x3 over
    the tile grid mixes in the 8 neighbour tiles (a zipper is the border between a right and a wrong tile), and the
    result is broadcast back to that tile's pixels only - no smoothing across tile borders, so a well-aligned tile
    next to a slipped one keeps its own evidence (RobustNetCtx leaked ~1 tile). Zero-init 1x1 into the hidden layer:
    starts exactly as the base net. Device cost: one per-tile reduction of the features + ctx x hid MACs per pixel."""
    def __init__(self, nf=NF, hid=HID, ctx=8):
        super().__init__(nf, hid)
        self.register_buffer("tile_model", torch.ones(1))
        self.k1 = nn.Conv2d(2 * nf, ctx, 1)
        self.k2 = nn.Conv2d(ctx, ctx, 3)
        self.k3 = nn.Conv2d(ctx, hid, 1)
        nn.init.zeros_(self.k3.weight)
        nn.init.zeros_(self.k3.bias)

    def forward(self, feat, r0, ts=16):
        x = (feat - self.mu.view(1, -1, 1, 1)) / self.sd.view(1, -1, 1, 1)
        H, W = x.shape[-2:]
        xp = torch.cat([F.avg_pool2d(x, ts, ceil_mode=True), F.max_pool2d(x, ts, ceil_mode=True)], 1)
        c = F.relu(self.k1(xp))
        c = F.relu(self.k2(F.pad(c, (1, 1, 1, 1), mode="replicate")))
        c = c.repeat_interleave(ts, 2).repeat_interleave(ts, 3)[..., :H, :W]
        h = F.relu(self.c1(F.pad(x, (1, 1, 1, 1), mode="replicate")))
        h = F.relu(self.c2(h) + self.k3(c))
        d = self.c3(h)[:, 0]
        return torch.sigmoid(logit(r0) + d), d
