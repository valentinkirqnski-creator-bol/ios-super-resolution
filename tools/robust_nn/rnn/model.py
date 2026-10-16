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
