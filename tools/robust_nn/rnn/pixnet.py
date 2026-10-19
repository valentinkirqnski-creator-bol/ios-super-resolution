"""v15: pixel-seeing refinement of the v14 mask.

v14 sees 14 summary features through one 3x3 window, so its mask inherits the analytic mask's tile blocks. PixNet
additionally sees the PIXELS - reference and aligned comparison, noise-normalised - over a 17x17 guide-pixel
neighbourhood (dilated 3x3 convs), so it can follow where inside a misaligned tile a ghost actually appears (the
displaced edge) and keep the flat part, as the ground-truth mask does. It adds a correction to v14's logit with a
zero-initialised head: at the start it IS v14. Replicate padding everywhere (zero padding would act as a position
code, see tiny-coarsest-level memory)."""
import torch
import torch.nn as nn

from .common import win_mean

NPIX = 4
PIX_NAMES = ["diff", "ref_hp", "cmp_hp", "mean_diff"]


def pixel_inputs(Ir, J, a, b):
    """[B,H,W] reference and warped comparison grey -> [B,NPIX,H,W] noise-normalised pixel channels."""
    mr, mj = win_mean(Ir, 1), win_mean(J, 1)
    sig = (a * mr.clamp_min(0.0) + b).clamp_min(1e-10).sqrt()
    ch = [(J - Ir) / sig, (Ir - mr) / sig, (J - mj) / sig, (mj - mr) / sig]
    return torch.stack(ch, 1).clamp(-10.0, 10.0)


class PixNet(nn.Module):
    def __init__(self, base, ch=16, npix=NPIX):
        super().__init__()
        self.base = base                                   # frozen v14 RobustNet (its own standardisation)
        for p in self.base.parameters():
            p.requires_grad_(False)
        nf = base.c1.weight.shape[1]
        cin = nf + npix + 1
        conv = lambda i, o, d: nn.Conv2d(i, o, 3, padding=d, dilation=d, padding_mode="replicate")
        self.p1 = conv(cin, ch, 1)
        self.p2 = conv(ch, ch, 2)
        self.p3 = conv(ch, ch, 4)
        self.p4 = conv(ch, ch, 1)
        self.p5 = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.p5.weight)
        nn.init.zeros_(self.p5.bias)
        self.act = nn.ReLU()

    def forward(self, feat, r0, pix):
        with torch.no_grad():
            R14, _ = self.base(feat, r0)
            R14 = R14.clamp(1e-4, 1 - 1e-4)
            l14 = torch.log(R14) - torch.log1p(-R14)
            x = (feat - self.base.mu.view(1, -1, 1, 1)) / self.base.sd.view(1, -1, 1, 1)
            x = torch.cat([x, pix, (l14 / 4.0)[:, None]], 1)
        h = self.act(self.p1(x))
        h = self.act(self.p2(h))
        h = self.act(self.p3(h))
        h = self.act(self.p4(h))
        d = self.p5(h)[:, 0]
        return torch.sigmoid(l14 + d), d


NCONS = 3
CONS_NAMES = ["vs_others", "ref_vs_others", "others_spread"]


def consensus_inputs(Ir, J, a, b):
    """Multi-frame consensus per comparison frame k: the mean / spread of all OTHER aligned frames (the reference
    plus every comparison frame j != k at its own flow). Averaging frames with different sub-pixel phases cancels
    much of the aliasing that makes a single reference ambiguous, so a frame that is actually misaligned stands
    out against it. Device-friendly: only a running sum and sum of squares over the burst are needed.
    Ir [H,W] or [K,H,W] reference, J [K,H,W] warped comparisons -> [K,NCONS,H,W]."""
    I0 = Ir if Ir.dim() == 2 else Ir[0]
    K = J.shape[0]
    s1 = I0 + J.sum(0)
    s2 = I0 * I0 + (J * J).sum(0)
    mo = (s1[None] - J) / K                                  # K others per frame: reference + (K - 1) comparisons
    so = ((s2[None] - J * J) / K - mo * mo).clamp_min(0.0).sqrt()
    sig = (a * win_mean(I0[None], 1).clamp_min(0.0) + b).clamp_min(1e-10).sqrt()
    ch = [(J - mo) / sig, (I0[None] - mo) / sig, so / sig]
    return torch.stack(ch, 1).clamp(-10.0, 10.0)


def expand_inputs(sd, nf, npix_old, npix_new):
    """State dict of a PixNet with npix_old pixel channels -> one with npix_new: new input weights are zero, so the
    expanded net computes exactly the same function."""
    w = sd["p1.weight"]
    ch = w.shape[0]
    nw = torch.zeros(ch, nf + npix_new + 1, 3, 3)
    nw[:, :nf + npix_old] = w[:, :nf + npix_old]
    nw[:, nf + npix_new] = w[:, nf + npix_old]              # v14-logit channel stays last
    sd = dict(sd)
    sd["p1.weight"] = nw
    return sd
