"""The sr_gate network, and the merge as a torch op.

Architecture, and why it is this small: the mask is a per-pixel decision about a
correspondence, and every quantity that decision depends on is already computed
by the pipeline and handed in as a feature. What the network adds over the
analytic formula is (a) a nonlinear combination of eight indicators instead of
one exponential of one of them, and (b) spatial context, so a rejection can be
shaped by its neighbourhood rather than dilated by a blanket 5x5 minimum.
Neither needs depth.

    conv 3x3 dilation 1   8 -> 8   ReLU       584 weights
    conv 3x3 dilation 2   8 -> 8   ReLU       584
    conv 3x3 dilation 3   8 -> 8   ReLU       584
    conv 1x1              8 -> 1   sigmoid      9
                                              ----
                                              1761 parameters

Receptive field 13x13 on the raw lattice, against Eq. 9's 5x5 minimum. Edges are
handled by replicate padding, matching how every other window in robustness.cpp
clamps its indices.

Eq. 9's 5x5 minimum is NOT applied on top of the output. That step exists to
spread rejection outward from a pointwise test that has no spatial context; this
network has context and emits the final decision.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import srsim

# Imported rather than restated: srsim.build_features defines what the channels
# ARE, and a second hardcoded count here is how the two silently disagree.
NUM_FEATURES = srsim.NUM_FEATURES
WIDTH = 8
DILATIONS = (1, 2, 3)


# Coarse branch: pooling factor, width, and dilations. Receptive field is
# (1 + 2*sum(dilations)) coarse pixels = about 500 RAW pixels at POOL 16, against
# the fine branch's 13.
POOL = 16
COARSE_WIDTH = 10
COARSE_DILATIONS = (1, 2, 4, 8)


class SRGate(nn.Module):
    """Per-pixel mask from per-pixel features, with a coarse branch for context.

    The fine branch alone has a 13x13 receptive field, and that is what made the
    mask fail on a real burst where the subject moved ~500 raw px: on the
    TEXTURED parts of a displaced person d^2/sigma^2 is ~52, overwhelming, while
    on the FLAT parts it is ~3, indistinguishable from a good frame. Both must be
    rejected -- they are the same displaced object -- and 13x13 cannot see that.

    Applying Eq. 9's minimum on top does fix it (person R 0.146 -> 0.022) and
    costs 2.05 dB on the held-out set, because a blanket minimum spreads every
    reduction including the ones the gate makes for good reasons. The coarse
    branch is the same idea with the discrimination put back: the network decides
    when regional evidence should spread and when it should not.

    Both AVG and MAX pooling feed it. "Is there overwhelming evidence anywhere in
    this region" is a max question, and an average over 256 pixels would dilute a
    few very strong ones into nothing.
    """

    def __init__(self, in_ch=NUM_FEATURES, width=WIDTH, dilations=DILATIONS,
                 coarse=True, pool=POOL, coarse_width=COARSE_WIDTH,
                 coarse_dilations=COARSE_DILATIONS):
        super().__init__()
        self.dilations = tuple(dilations)
        self.coarse = bool(coarse)
        self.pool = int(pool)
        self.coarse_dilations = tuple(coarse_dilations)
        if self.coarse:
            cc = [2 * in_ch] + [coarse_width] * len(self.coarse_dilations)
            self.cconvs = nn.ModuleList([
                nn.Conv2d(cc[i], cc[i + 1], 3, padding=0, dilation=d)
                for i, d in enumerate(self.coarse_dilations)
            ])
            fine_in = in_ch + coarse_width
        else:
            self.cconvs = nn.ModuleList()
            fine_in = in_ch
        chans = [fine_in] + [width] * len(self.dilations)
        self.convs = nn.ModuleList([
            nn.Conv2d(chans[i], chans[i + 1], 3, padding=0, dilation=d)
            for i, d in enumerate(self.dilations)
        ])
        self.head = nn.Conv2d(chans[-1], 1, 1)
        # Start permissive. R near 1 is the do-no-harm initialisation: the mask
        # begins as "merge everything" and training has to earn every rejection,
        # rather than starting from a collapse it then has to climb out of.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 2.0)

    def _coarse(self, x):
        n, c, h, w = x.shape
        p = self.pool
        # Pad up to a whole number of pool cells so the upsample lines up.
        ph = (p - h % p) % p
        pw = (p - w % p) % p
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        a = F.avg_pool2d(x, p)
        m = F.max_pool2d(x, p)
        y = torch.cat([a, m], dim=1)
        for conv, d in zip(self.cconvs, self.coarse_dilations):
            y = F.pad(y, (d, d, d, d), mode='replicate')
            y = F.relu(conv(y))
        y = F.interpolate(y, scale_factor=p, mode='nearest')
        return y[:, :, :h, :w]

    def _trunk(self, x):
        """Everything before the 1x1 head. Split out so a correction network can
        share the shape without duplicating the forward pass."""
        if self.coarse:
            x = torch.cat([x, self._coarse(x)], dim=1)
        for conv, d in zip(self.convs, self.dilations):
            x = F.pad(x, (d, d, d, d), mode='replicate')
            x = F.relu(conv(x))
        return x

    def forward(self, x):
        return torch.sigmoid(self.head(self._trunk(x)))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# the merge, as a torch op
# --------------------------------------------------------------------------

def from_checkpoint(ck):
    """Build the net a checkpoint describes. Checkpoints written before the
    coarse branch existed have no 'coarse' key and are fine-only."""
    net = SRGate(in_ch=ck['in_ch'], width=ck['width'],
                 dilations=ck['dilations'], coarse=ck.get('coarse', False),
                 pool=ck.get('pool', POOL),
                 coarse_dilations=tuple(ck.get('coarse_dilations',
                                               COARSE_DILATIONS)))
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net


def upsample_r(R):
    """The exact gather merge.cpp performs on R when rob_is_raw and
    merge_robustness_bilinear are both true: output pixel hr samples R
    bilinearly at hr/2. For scale 2 that is
        R_out[2k]     = R[k]
        R_out[2k + 1] = (R[k] + R[k + 1]) / 2,  with R[h - 1] repeated at the end
    which is separable and exact -- not any of torch's interpolate modes, whose
    coordinate conventions differ from merge.cpp's by a quarter pixel.

    R: (..., h, w) -> (..., 2h, 2w)
    """
    nxt_r = torch.cat([R[..., 1:, :], R[..., -1:, :]], dim=-2)
    rows = torch.stack([R, 0.5 * (R + nxt_r)], dim=-2)
    rows = rows.flatten(-3, -2)                      # (..., 2h, w)
    nxt_c = torch.cat([rows[..., 1:], rows[..., -1:]], dim=-1)
    cols = torch.stack([rows, 0.5 * (rows + nxt_c)], dim=-1)
    return cols.flatten(-2, -1)                      # (..., 2h, 2w)


def merge(A_ref, B_ref, A, B, R, eps=1e-8):
    """out = (A_ref + sum_n A_n R_n) / (B_ref + sum_n B_n R_n)

    A_ref, B_ref : (batch, 3, Hs, Ws)
    A, B         : (batch, N, 3, Hs, Ws)
    R            : (batch, N, h, w)     masks on the raw lattice
    """
    Rout = upsample_r(R).unsqueeze(2)                # (batch, N, 1, Hs, Ws)
    num = A_ref + (A * Rout).sum(dim=1)
    den = B_ref + (B * Rout).sum(dim=1)
    return num / den.clamp_min(eps)


class SRGateOnlyStricter(nn.Module):
    """The shipped mask, times a learned attenuation that can only reduce it.

        R = R_base(f) * sigmoid(correction(f))

    R_base is the CURRENT shipped network, frozen. sigmoid is in (0, 1), so

        R <= R_base   at every pixel, for every input, for any weights

    which is a property of the form and not something training has to be trusted
    to discover. That matters because on this branch every model that was more
    permissive than the shipped one looked worse, in all three attempts, while
    measuring better -- mask mean 0.093 good, 0.255 artifacts, 0.482 worse, 0.681
    worse. A mean-error loss rewards merging more, because an unrejected
    misalignment is glaring but contributes little squared error, so the training
    signal points the wrong way and no amount of reweighting fixes a sign.

    Under this form the network cannot act on that pull at all. The only thing it
    can learn is where the shipped mask merges something it should not.
    """

    def __init__(self, base: 'SRGate', corr: 'SRGate'):
        super().__init__()
        self.base = base
        self.corr = corr
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()          # frozen: never in training mode
        return self

    def forward(self, x):
        with torch.no_grad():
            r0 = self.base(x)
        return r0 * torch.sigmoid(self.corr.head(self.corr._trunk(x)))

    def n_params(self):
        return sum(p.numel() for p in self.corr.parameters())

    # The training and eval code reads the architecture off the net. Report the
    # CORRECTION's, which is the only part being trained; both halves are the
    # same shape anyway, so nothing can read the wrong one.
    @property
    def convs(self):
        return self.corr.convs

    @property
    def cconvs(self):
        return self.corr.cconvs

    @property
    def dilations(self):
        return self.corr.dilations

    @property
    def coarse(self):
        return self.corr.coarse

    @property
    def pool(self):
        return self.corr.pool

    @property
    def coarse_dilations(self):
        return self.corr.coarse_dilations

    def checkpoint(self, **extra):
        """Self-contained: carries BOTH weight sets, so a checkpoint can be
        evaluated and exported without also having to track down which base it
        was trained against."""
        ck = _arch(self.corr)
        ck['state_dict'] = self.corr.state_dict()
        ck['base_state_dict'] = self.base.state_dict()
        ck['only_stricter'] = True
        ck.update(extra)
        return ck


def _arch(net):
    return {'dilations': net.dilations, 'width': net.convs[0].weight.shape[0],
            'in_ch': ((net.cconvs[0].weight.shape[1] // 2) if net.coarse
                      else net.convs[0].weight.shape[1]),
            'coarse': net.coarse, 'pool': net.pool,
            'coarse_dilations': net.coarse_dilations}


def plain_checkpoint(net, **extra):
    ck = _arch(net)
    ck['state_dict'] = net.state_dict()
    ck.update(extra)
    return ck


def any_from_checkpoint(ck):
    """from_checkpoint, but also rebuilds an only-stricter pair."""
    if not ck.get('only_stricter'):
        return from_checkpoint(ck)
    base = from_checkpoint({**ck, 'state_dict': ck['base_state_dict']})
    corr = from_checkpoint(ck)
    net = SRGateOnlyStricter(base, corr)
    net.eval()
    return net
