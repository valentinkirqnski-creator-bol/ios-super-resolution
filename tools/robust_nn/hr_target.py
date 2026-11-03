"""Ground truth at FULL raw resolution (v18 candidate).

The v12-v17 target measures a ghost on the half-resolution grey guide (2x2 quad means). That blurs sharp edges,
so a 0.1-guide-px tile offset (0.4 px in the 2x / 48 MP output) looks invisible to it - yet in the output it is
exactly the "shifted straight" edge step at a tile border. Here the ghost is measured where the merge samples: for
every RAW pixel p the noise-free scene the merge would pull from frame k, C_k(p + f_used(p)), is rendered exactly
(Burst.render_rgb, no interpolation) and compared with the scene point under the reference pixel, C_0(p) (the same
point frame k would show at the true correspondence; no blur / exposure difference in these bursts). Luma, 3x3
mean of the squared difference, against the merged noise (per raw sample / n_merge) plus the 2 % visibility floor.
A guide pixel (one RGGB quad) is as bad as its worst raw pixel (min over the quad). Out-of-frame points are 0;
occluded points are judged by what they look like (the comparison is with what the reference shows).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from rnn.common import win_mean
import real_protocol as rp


def _luma(c):
    return 0.25 * c[0] + 0.5 * c[1] + 0.25 * c[2]


def hr_target(b, flow_used, gt_flow, alpha, beta, n_merge=8, k=1.5, floor=0.02):
    """b: Burst; flow_used / gt_flow: [K,2,Hg,Wg] guide px (ref -> comparison). Returns [K,Hg,Wg] in [0,1]."""
    K, _, Hg, Wg = flow_used.shape
    H, W = 2 * Hg, 2 * Wg
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    pts = torch.stack([xs.reshape(-1), ys.reshape(-1)], 1)
    rad0 = b.render_rgb(0, pts)                                            # [3,P] scene radiance under ref pixels
    E = [float(e) for e in b.E]
    mu = None
    out = []
    up = lambda f: f.repeat_interleave(2, -2).repeat_interleave(2, -1)    # guide -> raw, nearest (the quad's vector)
    for i in range(K):
        fu = up(flow_used[i]).double() * 2.0                               # raw px
        q = pts + torch.stack([fu[0].reshape(-1), fu[1].reshape(-1)], 1)
        ck = _luma((b.render_rgb(i + 1, q) * E[i + 1]).clamp(0, 1)).view(1, H, W).float()
        c0 = _luma((rad0 * E[i + 1]).clamp(0, 1)).view(1, H, W).float()
        if mu is None:
            mu = win_mean(_luma((rad0 * E[0]).clamp(0, 1)).view(1, H, W).float(), 1).clamp_min(0)
        D2 = win_mean((ck - c0) ** 2, 1)
        s2 = (alpha * mu + beta) / n_merge + (floor * mu) ** 2 + 1e-10
        t = torch.exp(-0.5 * D2 / (k * k * s2))[0]
        g = up(gt_flow[i]).float() * 2.0                                   # out of frame at the TRUE correspondence -> 0
        inside = ((xs.float() + g[0] >= 0) & (xs.float() + g[0] <= W - 1) & (ys.float() + g[1] >= 0)
                  & (ys.float() + g[1] <= H - 1))
        t = t * inside.float()
        out.append(-F.max_pool2d(-t[None, None], 2, 2)[0, 0])             # min over each RGGB quad
    return torch.stack(out)


def burst_of(src_file):
    """Rebuild the Burst of a stored synthetic burst file (rp13_* / rp17_*) from its seed and generation env."""
    seed = int(os.path.basename(src_file)[2:10])
    d = os.path.basename(os.path.dirname(src_file))
    env = {"RNN_ROT_PATCHES": "2,10,1,5"}
    if d.startswith("rp19"):
        scenes = ["APC_1186", "HDRP_N000", "APC_1294", "APC_1340", "APC_1342"]
        val = d.endswith("val")
        env.update(RNN_VAL_FROM_SCENES="1", RNN_ROT_SMALL="3,10")
        if d.startswith("rp19s05"):
            env["RNN_TEX_SCALE"] = "0.5"
        elif d.startswith("rp19s1"):
            env["RNN_TEX_SCALE"] = "1"
    elif d.startswith("rp17"):
        scenes, val = ["APC_1294", "APC_1340", "APC_1342"], d.endswith("val")
        env["RNN_VAL_FROM_SCENES"] = "1"
    elif d == "rp13_train_hdrp":
        scenes, val = ["HDRP_N000"], False
    else:
        scenes, val = ["APC_1186"], d.endswith("val")
    old = {k: os.environ.get(k) for k in ("RNN_ROT_PATCHES", "RNN_VAL_FROM_SCENES", "RNN_ROT_SMALL", "RNN_TEX_SCALE")}
    os.environ.update(env)
    for k in ("RNN_VAL_FROM_SCENES", "RNN_ROT_SMALL", "RNN_TEX_SCALE"):
        if k not in env:
            os.environ.pop(k, None)
    try:
        b, *_ = rp.build_burst(seed, val, scenes)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return b
